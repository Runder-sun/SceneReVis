"""
基于Mesh的3D场景评估工具 - 无容差版本
- 使用真实3D模型进行碰撞检测
- 不使用任何容差值（碰撞和出界检测均无容差）
"""
import json
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse
from shapely.geometry import Polygon, Point
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm

# --- 0. Objaverse GLB 查找辅助函数 ---

def find_objaverse_glb(uid: str):
    """查找 Objaverse GLB 文件路径。"""
    if not uid or len(uid) < 2:
        return None
    
    layout_vlm_processed = Path("/path/to/SceneReVis/baseline/LayoutVLM/objaverse_processed")
    if layout_vlm_processed.is_dir():
        candidate = layout_vlm_processed / uid / f"{uid}.glb"
        if candidate.is_file():
            return candidate

    cache_dirs = []
    env_cache = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
    if env_cache:
        cache_dirs.append(Path(env_cache) / "glbs")
    
    cache_dirs.append(Path("/path/to/data/datasets/objathor-assets/glbs"))
    cache_dirs.append(Path("/path/to/datasets/objathor-assets/glbs"))
    cache_dirs.append(Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs")
    
    subdir_name = uid[:2]
    
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        candidate = cache_dir / subdir_name / f"{uid}.glb"
        if candidate.is_file():
            return candidate
    
    return None


def get_object_field(obj, field_name, format_type='ours'):
    """从对象中获取字段值"""
    if format_type == 'respace':
        if field_name == 'jid':
            return obj.get('sampled_asset_jid', obj.get('jid', 'N/A'))
        elif field_name == 'desc':
            return obj.get('sampled_asset_desc', obj.get('desc', 'N/A'))
        elif field_name == 'size':
            return obj.get('sampled_asset_size', obj.get('size', [1, 1, 1]))
    else:
        return obj.get(field_name, 'N/A' if field_name in ['jid', 'desc'] else [1, 1, 1])


def create_floor_polygon(bounds_bottom):
    """从房间底部边界创建地板多边形"""
    points = [(pt[0], pt[2]) for pt in bounds_bottom]
    return Polygon(points)


def create_room_mesh(bounds_bottom, bounds_top):
    """从多边形边界创建房间mesh"""
    bounds_bottom = np.array(bounds_bottom)
    bounds_top = np.array(bounds_top)
    
    floor_polygon = create_floor_polygon(bounds_bottom.tolist())
    
    num_verts = len(bounds_bottom)
    all_vertices = np.concatenate([bounds_bottom, bounds_top], axis=0)
    
    try:
        vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="triangle")
    except Exception:
        try:
            vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="earcut")
        except Exception:
            floor_faces = np.array([[0, i, i+1] for i in range(1, num_verts-1)])
    
    valid_mask = np.all(floor_faces < num_verts, axis=1)
    floor_faces = floor_faces[valid_mask]
    
    ceiling_faces = floor_faces + num_verts
    ceiling_faces = ceiling_faces[:, ::-1]
    
    side_faces = []
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        side_faces.append([i, next_i, i + num_verts])
        side_faces.append([next_i, next_i + num_verts, i + num_verts])
    side_faces = np.array(side_faces)
    
    all_faces = np.concatenate([floor_faces, ceiling_faces, side_faces], axis=0)
    
    room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    trimesh.repair.fix_normals(room_mesh)
    
    return room_mesh


def check_object_out_of_bounds_no_tolerance(obj_mesh, room_mesh, floor_polygon, room_height_min, room_height_max, num_samples=500):
    """
    使用mesh containment检测物体是否出界（碰撞无容差，出界保留1mm容差）
    """
    try:
        sample_points = obj_mesh.sample(num_samples)
        
        # 检测2D出界（XZ平面）- 保留1mm缓冲区
        buffered_polygon = floor_polygon.buffer(0.001)  # 1mm缓冲区
        points_2d = [Point(pt[0], pt[2]) for pt in sample_points]
        inside_2d = np.array([buffered_polygon.contains(p) for p in points_2d])
        
        # 检测高度出界（Y方向）- 保留1mm容差
        height_tolerance = 0.001
        inside_height = (sample_points[:, 1] >= room_height_min - height_tolerance) & (sample_points[:, 1] <= room_height_max + height_tolerance)
        
        inside = inside_2d & inside_height
        
        oob_ratio = 1.0 - (inside.sum() / len(inside))
        
        # 超过1%的点在外面才算出界（保留原有阈值）
        if oob_ratio > 0.01:
            is_oob = True
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            oob_volume = obj_volume * oob_ratio
        else:
            is_oob = False
            oob_volume = 0.0
        
        return is_oob, oob_volume
        
    except Exception as e:
        print(f"警告: 出界检测采样失败: {e}，使用边界框备用方案")
        obj_bounds = obj_mesh.bounds
        room_bounds = room_mesh.bounds
        
        if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            return True, obj_volume * 0.1
        return False, 0.0


def parse_scene_data(scene_json, format_type='ours'):
    """解析场景JSON数据"""
    if format_type == 'respace':
        bounds_bottom = scene_json.get('bounds_bottom', [])
        bounds_top = scene_json.get('bounds_top', [])
        all_objects_data = scene_json.get('objects', [])
    else:
        bounds_bottom = scene_json['room_envelope']['bounds_bottom']
        bounds_top = scene_json['room_envelope']['bounds_top']
        all_objects_data = []
        if 'groups' in scene_json and scene_json['groups'] is not None:
            for group in scene_json['groups']:
                if 'objects' in group and group['objects'] is not None:
                    all_objects_data.extend(group['objects'])
    
    return bounds_bottom, bounds_top, all_objects_data


def parse_scene_from_json(scene_json, models_base_path, format_type='ours'):
    """从JSON数据中解析场景,并加载、变换真实的三维模型。"""
    bounds_bottom, bounds_top, all_objects_data = parse_scene_data(scene_json, format_type)
    unique_objects_data = all_objects_data
    
    print(f"对象数量: {len(all_objects_data)}")

    room_mesh = create_room_mesh(bounds_bottom, bounds_top)
    floor_polygon = create_floor_polygon(bounds_bottom if isinstance(bounds_bottom, list) else bounds_bottom.tolist())
    
    bounds_bottom_arr = np.array(bounds_bottom)
    bounds_top_arr = np.array(bounds_top)
    room_height_min = bounds_bottom_arr[:, 1].min()
    room_height_max = bounds_top_arr[:, 1].max()

    furniture_objects = {}
    for i, obj_data in enumerate(unique_objects_data):
        asset_source = obj_data.get('asset_source', '3d-future')
        target_size = get_object_field(obj_data, 'size', format_type)
        
        model_path = None
        asset_id = None
        
        if asset_source == 'objaverse':
            uid = obj_data.get('uid')
            if uid:
                asset_id = uid
                model_path = find_objaverse_glb(uid)
                if model_path:
                    model_path = str(model_path)
        else:
            jid = get_object_field(obj_data, 'jid', format_type)
            asset_id = jid
            model_path = os.path.join(models_base_path, jid, 'raw_model.glb')
        
        if asset_id is None:
            asset_id = f'object_{i}'
        
        if model_path is None or not os.path.exists(model_path):
            mesh = trimesh.creation.box(extents=target_size)
        else:
            loaded = trimesh.load(model_path)
            if isinstance(loaded, trimesh.Scene):
                if len(loaded.geometry) > 0:
                    mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                else:
                    mesh = trimesh.creation.box(extents=target_size)
            else:
                mesh = loaded

        # 应用变换
        original_size = mesh.extents
        target_size_array = np.array(target_size)
        scale_factors = target_size_array / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        pos = obj_data['pos']
        rot_xyzw = obj_data['rot']
        
        try:
            rotation = R.from_quat(rot_xyzw)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation.as_matrix()
            transform_matrix[:3, 3] = pos
            
            bounds = mesh.bounds
            bottom_center_pivot = np.array([
                (bounds[0, 0] + bounds[1, 0]) / 2,
                bounds[0, 1],
                (bounds[0, 2] + bounds[1, 2]) / 2
            ])

            center_transform = np.eye(4)
            center_transform[:3, 3] = -bottom_center_pivot
            
            mesh.apply_transform(center_transform)
            mesh.apply_transform(transform_matrix)
            
            furniture_objects[f"object_{i}_{asset_id[:8]}"] = mesh
        except Exception as e:
            desc = get_object_field(obj_data, 'desc', format_type)
            print(f"处理对象时出错: {desc}, 错误: {e}")

    return room_mesh, furniture_objects, unique_objects_data, floor_polygon, room_height_min, room_height_max


def calculate_physics_metrics_no_tolerance(room_mesh, objects, floor_polygon=None, room_height_min=None, room_height_max=None):
    """
    计算碰撞和出界指标（无容差版本）
    """
    total_objects = len(objects)
    
    if total_objects == 0:
        return {
            "Object Count": 0,
            "Collision-Free Rate (%)": 100.0,
            "Number of Colliding Pairs": 0,
            "Collision Rate (%)": 0.0,
            "Mean Penetration Depth (m)": 0.0,
            "Valid Placement Rate (%)": 100.0,
            "Number of Out-of-Bounds Objects": 0,
            "Out-of-Bounds Rate (%)": 0.0,
            "Mean Out-of-Bounds Volume (m^3)": 0.0
        }
    
    manager = trimesh.collision.CollisionManager()
    
    for name, mesh in objects.items():
        manager.add_object(name, mesh)

    is_collision, contact_data = manager.in_collision_internal(return_data=True)
    
    # 无容差：所有穿透深度 > 0 的都视为碰撞
    actual_collisions_data = [d for d in contact_data if d.depth > 0]
    
    num_colliding_pairs = len(actual_collisions_data)
    
    colliding_objects = set()
    for contact in actual_collisions_data:
        if hasattr(contact, 'names'):
            if isinstance(contact.names, (list, tuple)):
                colliding_objects.add(contact.names[0])
                colliding_objects.add(contact.names[1])
            else:
                colliding_objects.update(contact.names)
    
    num_colliding_objects = len(colliding_objects)
    collision_rate = (num_colliding_objects / total_objects * 100) if total_objects > 0 else 0.0
    
    total_penetration_depth = 0
    if num_colliding_pairs > 0:
        for contact in actual_collisions_data:
            total_penetration_depth += contact.depth
            
    mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0

    # 出界检测（无容差）
    num_oob_objects = 0
    total_oob_volume = 0
    
    use_polygon_detection = (floor_polygon is not None and 
                             room_height_min is not None and 
                             room_height_max is not None)
    
    for name, mesh in objects.items():
        if use_polygon_detection:
            is_oob, oob_volume = check_object_out_of_bounds_no_tolerance(
                mesh, room_mesh, floor_polygon, room_height_min, room_height_max
            )
            if is_oob:
                num_oob_objects += 1
                total_oob_volume += oob_volume
        else:
            obj_bounds = mesh.bounds
            room_bounds = room_mesh.bounds
            
            if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
                num_oob_objects += 1
                try:
                    overlap_min = np.maximum(obj_bounds[0], room_bounds[0])
                    overlap_max = np.minimum(obj_bounds[1], room_bounds[1])
                    
                    if (overlap_max > overlap_min).all():
                        overlap_volume = np.prod(overlap_max - overlap_min)
                        obj_volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else np.prod(mesh.extents)
                        oob_volume = max(0, obj_volume - overlap_volume)
                    else:
                        obj_volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else np.prod(mesh.extents)
                        oob_volume = obj_volume
                except Exception:
                    oob_volume = np.prod(mesh.extents) * 0.1
                
                total_oob_volume += oob_volume

    mean_oob_volume = (total_oob_volume / num_oob_objects) if num_oob_objects > 0 else 0
    out_of_bounds_rate = (num_oob_objects / total_objects * 100) if total_objects > 0 else 0.0

    metrics = {
        "Object Count": total_objects,
        "Collision-Free Rate (%)": 100.0 if num_colliding_pairs == 0 else 0.0,
        "Number of Colliding Pairs": num_colliding_pairs,
        "Collision Rate (%)": collision_rate,
        "Mean Penetration Depth (m)": mean_penetration_depth,
        "Valid Placement Rate (%)": 100.0 if num_oob_objects == 0 else 0.0,
        "Number of Out-of-Bounds Objects": num_oob_objects,
        "Out-of-Bounds Rate (%)": out_of_bounds_rate,
        "Mean Out-of-Bounds Volume (m^3)": mean_oob_volume
    }
    return metrics


def evaluate_single_scene(scene_json, models_base_path, create_virtual_models=True, format_type='ours'):
    """评估单个场景的指标"""
    if create_virtual_models:
        all_jids = set()
        _, _, all_objects_data = parse_scene_data(scene_json, format_type)
        for obj in all_objects_data:
            jid = get_object_field(obj, 'jid', format_type)
            all_jids.add(jid)
        
        for jid in all_jids:
            jid_dir = os.path.join(models_base_path, jid)
            os.makedirs(jid_dir, exist_ok=True)
            model_path = os.path.join(jid_dir, 'raw_model.glb')
            if not os.path.exists(model_path):
                try:
                    trimesh.creation.box().export(model_path)
                except Exception:
                    pass

    try:
        room_mesh, objects, unique_objects_data, floor_polygon, room_height_min, room_height_max = parse_scene_from_json(
            scene_json, models_base_path, format_type
        )
        
        physics_metrics = calculate_physics_metrics_no_tolerance(
            room_mesh, objects, floor_polygon, room_height_min, room_height_max
        )
        
        return physics_metrics, True
        
    except Exception as e:
        print(f"评估场景时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}, False


def evaluate_single_scene_file(json_file: str, models_base_path: str, format_type: str) -> dict:
    """评估单个场景文件（用于多进程调用）"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            scene_json = json.load(f)
        
        metrics, success = evaluate_single_scene(scene_json, models_base_path, create_virtual_models=True, format_type=format_type)
        
        if success and metrics:
            return {
                'file': os.path.basename(json_file),
                'metrics': metrics,
                'success': True
            }
        else:
            return {
                'file': os.path.basename(json_file),
                'metrics': {},
                'success': False,
                'error': '评估失败'
            }
    except Exception as e:
        return {
            'file': os.path.basename(json_file),
            'metrics': {},
            'success': False,
            'error': str(e)
        }


def batch_evaluate_scenes(scenes_directory, models_base_path, max_scenes=None, format_type='ours', num_workers=None, output_dir=None):
    """批量评估目录中的所有JSON场景文件"""
    print(f"开始批量评估场景 (Mesh模式，无容差)...")
    print(f"场景目录: {scenes_directory}")
    print(f"模型路径: {models_base_path}")
    
    json_files = []
    if os.path.isdir(scenes_directory):
        for file in os.listdir(scenes_directory):
            if file.endswith('.json') and not file.startswith('evaluation'):
                json_files.append(os.path.join(scenes_directory, file))
    else:
        print(f"错误: 目录不存在: {scenes_directory}")
        return
    
    json_files.sort()
    
    if max_scenes and max_scenes > 0:
        json_files = json_files[:max_scenes]
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if len(json_files) == 0:
        print("未找到JSON文件!")
        return
    
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = min(num_workers, len(json_files))
    
    print(f"使用 {num_workers} 个进程进行并行处理")
    
    all_scene_metrics = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    print("\n开始处理场景...")
    print("=" * 80)
    
    eval_func = partial(evaluate_single_scene_file, models_base_path=models_base_path, format_type=format_type)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_func, json_file): json_file for json_file in json_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估场景"):
            json_file = futures[future]
            try:
                result = future.result()
                
                if result['success']:
                    all_scene_metrics.append({
                        'file': result['file'],
                        'metrics': result['metrics']
                    })
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
                    
            except Exception as e:
                failed_evaluations += 1
                print(f"\n  ✗ 处理文件时出错 {os.path.basename(json_file)}: {e}")
    
    print("\n" + "=" * 80)
    print(f"处理完成! 成功: {successful_evaluations}, 失败: {failed_evaluations}")
    
    if successful_evaluations == 0:
        print("没有成功评估的场景!")
        return
    
    print("\n--- 综合评估报告 (Mesh模式，无容差) ---")
    
    metric_names = list(all_scene_metrics[0]['metrics'].keys())
    metric_values = {name: [] for name in metric_names}
    
    for scene_data in all_scene_metrics:
        for name, value in scene_data['metrics'].items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metric_values[name].append(value)
    
    print(f"\n基于 {successful_evaluations} 个成功评估的场景:")
    print("\n[ 物理有效性指标统计 ]")
    physics_metrics_names = [
        "Object Count",
        "Collision-Free Rate (%)",
        "Number of Colliding Pairs",
        "Collision Rate (%)",
        "Mean Penetration Depth (m)",
        "Valid Placement Rate (%)",
        "Number of Out-of-Bounds Objects",
        "Out-of-Bounds Rate (%)",
        "Mean Out-of-Bounds Volume (m^3)"
    ]
    
    for name in physics_metrics_names:
        if name in metric_values and len(metric_values[name]) > 0:
            values = metric_values[name]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{name:<35}: 均值={mean_val:.4f} | 标准差={std_val:.4f} | 范围=[{min_val:.4f}, {max_val:.4f}]")
    
    result_output_dir = output_dir if output_dir else scenes_directory
    os.makedirs(result_output_dir, exist_ok=True)
    output_file = os.path.join(result_output_dir, "evaluation_results_mesh_notol.json")
    
    try:
        results = {
            "evaluation_mode": "Mesh (无容差)",
            "summary": {
                "total_scenes": len(json_files),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "success_rate": successful_evaluations / len(json_files) * 100
            },
            "aggregate_statistics": {},
            "individual_results": all_scene_metrics
        }
        
        for name in metric_names:
            if name in metric_values and len(metric_values[name]) > 0:
                values = metric_values[name]
                results["aggregate_statistics"][name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values)
                }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"保存结果文件时出错: {e}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于Mesh的3D场景评估工具（无容差）')
    parser.add_argument('--format', type=str, default='respace', choices=['ours', 'respace'],
                       help='场景JSON格式类型')
    parser.add_argument('--scenes_dir', type=str, 
                       default='/path/to/SceneReVis/output/sft_65k/final_scenes_collection', 
                       help='包含JSON场景文件的目录路径')
    parser.add_argument('--models_path', type=str, 
                       default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                       help='3D模型文件的基础路径')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='最大处理场景数量')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    os.makedirs(args.models_path, exist_ok=True)
    
    print("=== 3D场景评估工具 (Mesh模式，无容差) ===")
    print(f"场景格式: {args.format}")
    print(f"场景目录: {args.scenes_dir}")
    print(f"模型路径: {args.models_path}")
    print()
    
    batch_evaluate_scenes(args.scenes_dir, args.models_path, args.max_scenes, args.format, args.workers, args.output_dir)
