"""
基于边界框(Bounding Box)的3D场景评估工具
- 使用AABB边界框进行碰撞和出界检测
- 不使用任何容差值
"""
import json
import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse
from shapely.geometry import Polygon
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from tqdm import tqdm

# --- 0. Objaverse GLB 查找辅助函数 ---

def find_objaverse_glb(uid: str):
    """
    查找 Objaverse GLB 文件路径。
    """
    if not uid or len(uid) < 2:
        return None
    
    # 0. Special check for LayoutVLM processed objaverse
    layout_vlm_processed = Path("/path/to/SceneReVis/baseline/LayoutVLM/objaverse_processed")
    if layout_vlm_processed.is_dir():
        candidate = layout_vlm_processed / uid / f"{uid}.glb"
        if candidate.is_file():
            return candidate

    # GLB 缓存目录列表
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


# --- 1. 数据解析与准备 ---

def get_object_field(obj, field_name, format_type='ours'):
    """从对象中获取字段值,处理不同格式的字段名称"""
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
    """从房间底部边界创建地板多边形（使用X和Z坐标）"""
    points = [(pt[0], pt[2]) for pt in bounds_bottom]
    return Polygon(points)


def parse_scene_data(scene_json, format_type='ours'):
    """解析场景JSON数据,支持两种格式"""
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


def parse_scene_from_json_bbox(scene_json, models_base_path, format_type='ours'):
    """
    从JSON数据中解析场景,返回每个物体的AABB边界框信息
    不加载完整的3D模型，而是直接计算变换后的边界框
    """
    bounds_bottom, bounds_top, all_objects_data = parse_scene_data(scene_json, format_type)
    unique_objects_data = all_objects_data
    
    print(f"对象数量: {len(all_objects_data)}")

    # 创建地板多边形
    floor_polygon = create_floor_polygon(bounds_bottom if isinstance(bounds_bottom, list) else bounds_bottom.tolist())
    
    # 计算房间边界
    bounds_bottom_arr = np.array(bounds_bottom)
    bounds_top_arr = np.array(bounds_top)
    room_height_min = bounds_bottom_arr[:, 1].min()
    room_height_max = bounds_top_arr[:, 1].max()
    
    # 房间AABB边界框
    room_bbox = {
        'min': np.array([bounds_bottom_arr[:, 0].min(), room_height_min, bounds_bottom_arr[:, 2].min()]),
        'max': np.array([bounds_top_arr[:, 0].max(), room_height_max, bounds_top_arr[:, 2].max()])
    }

    # 计算每个物体变换后的AABB边界框
    object_bboxes = {}
    for i, obj_data in enumerate(unique_objects_data):
        asset_source = obj_data.get('asset_source', '3d-future')
        target_size = np.array(get_object_field(obj_data, 'size', format_type))
        
        # 获取asset_id
        if asset_source == 'objaverse':
            uid = obj_data.get('uid')
            asset_id = uid if uid else f'object_{i}'
        else:
            jid = get_object_field(obj_data, 'jid', format_type)
            asset_id = jid if jid != 'N/A' else f'object_{i}'
        
        pos = np.array(obj_data['pos'])
        rot_xyzw = obj_data['rot']
        
        try:
            rotation = R.from_quat(rot_xyzw)
            
            # 从底部中心创建初始边界框的8个角点
            # 物体的锚点在底部中心
            half_size = target_size / 2
            corners = np.array([
                [-half_size[0], 0, -half_size[2]],
                [-half_size[0], 0, half_size[2]],
                [half_size[0], 0, -half_size[2]],
                [half_size[0], 0, half_size[2]],
                [-half_size[0], target_size[1], -half_size[2]],
                [-half_size[0], target_size[1], half_size[2]],
                [half_size[0], target_size[1], -half_size[2]],
                [half_size[0], target_size[1], half_size[2]],
            ])
            
            # 应用旋转
            rotated_corners = rotation.apply(corners)
            
            # 应用平移
            transformed_corners = rotated_corners + pos
            
            # 计算变换后的AABB
            bbox_min = transformed_corners.min(axis=0)
            bbox_max = transformed_corners.max(axis=0)
            
            obj_name = f"object_{i}_{asset_id[:8] if len(asset_id) >= 8 else asset_id}"
            object_bboxes[obj_name] = {
                'min': bbox_min,
                'max': bbox_max,
                'size': target_size,
                'volume': np.prod(target_size)
            }
        except Exception as e:
            desc = get_object_field(obj_data, 'desc', format_type)
            print(f"处理对象时出错: {desc}, 错误: {e}")

    return room_bbox, object_bboxes, floor_polygon, room_height_min, room_height_max


def check_bbox_collision(bbox1, bbox2, tolerance=0.0):
    """
    检测两个AABB边界框是否碰撞
    
    Args:
        bbox1, bbox2: 边界框字典
        tolerance: 碰撞容差，穿透深度超过此值才算碰撞
    
    Returns:
        (is_collision: bool, penetration_depth: float)
    """
    # 计算每个轴上的重叠
    overlap_x = min(bbox1['max'][0], bbox2['max'][0]) - max(bbox1['min'][0], bbox2['min'][0])
    overlap_y = min(bbox1['max'][1], bbox2['max'][1]) - max(bbox1['min'][1], bbox2['min'][1])
    overlap_z = min(bbox1['max'][2], bbox2['max'][2]) - max(bbox1['min'][2], bbox2['min'][2])
    
    # 如果所有轴都有正的重叠，则发生碰撞
    if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
        # 穿透深度为最小重叠
        penetration_depth = min(overlap_x, overlap_y, overlap_z)
        # 只有穿透深度超过容差才算碰撞
        if penetration_depth > tolerance:
            return True, penetration_depth
    
    return False, 0.0


def check_bbox_out_of_bounds(obj_bbox, room_bbox, floor_polygon, room_height_min, room_height_max, tolerance=0.0):
    """
    检测物体边界框是否出界
    
    Args:
        tolerance: 出界容差，超过此值才算出界
    
    使用两种检测方式：
    1. 简单AABB检测（对矩形房间）
    2. 多边形检测（对异型房间）
    
    Returns:
        (is_oob: bool, oob_volume: float)
    """
    obj_min = obj_bbox['min']
    obj_max = obj_bbox['max']
    obj_volume = obj_bbox['volume']
    
    is_oob = False
    
    # 检测高度出界（Y方向）- 使用容差
    if obj_min[1] < room_height_min - tolerance or obj_max[1] > room_height_max + tolerance:
        is_oob = True
    
    # 检测2D出界（XZ平面）
    # 使用多边形检测：检查物体边界框的4个角点是否都在房间多边形内
    from shapely.geometry import Point, box as shapely_box
    
    # 创建物体的2D边界框（XZ平面）
    obj_box_2d = shapely_box(obj_min[0], obj_min[2], obj_max[0], obj_max[2])
    
    # 给多边形添加容差缓冲区
    buffered_polygon = floor_polygon.buffer(tolerance) if tolerance > 0 else floor_polygon
    
    # 检查物体边界框是否完全在房间多边形内
    if not buffered_polygon.contains(obj_box_2d):
        # 检查是否有任何交集
        if buffered_polygon.intersects(obj_box_2d):
            # 部分出界
            is_oob = True
        else:
            # 完全出界
            is_oob = True
    
    if is_oob:
        # 估算出界体积
        # 计算在房间内的部分
        try:
            intersection = floor_polygon.intersection(obj_box_2d)
            if intersection.is_empty:
                oob_volume = obj_volume
            else:
                # 计算在房间内的体积比例
                in_bounds_ratio = intersection.area / obj_box_2d.area
                
                # 考虑高度方向的出界
                height_in_bounds = min(obj_max[1], room_height_max) - max(obj_min[1], room_height_min)
                obj_height = obj_max[1] - obj_min[1]
                height_ratio = max(0, height_in_bounds / obj_height) if obj_height > 0 else 0
                
                in_bounds_volume_ratio = in_bounds_ratio * height_ratio
                oob_volume = obj_volume * (1 - in_bounds_volume_ratio)
        except Exception:
            oob_volume = obj_volume * 0.1
        
        return True, oob_volume
    
    return False, 0.0


def calculate_physics_metrics_bbox(room_bbox, object_bboxes, floor_polygon, room_height_min, room_height_max, collision_tolerance=0.0, oob_tolerance=0.0):
    """
    使用AABB边界框计算碰撞和出界指标
    
    Args:
        collision_tolerance: 碰撞容差（默认0，无容差）
        oob_tolerance: 出界容差（默认0，无容差）
    """
    total_objects = len(object_bboxes)
    
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
    
    # --- 碰撞检测 ---
    obj_names = list(object_bboxes.keys())
    num_colliding_pairs = 0
    colliding_objects = set()
    total_penetration_depth = 0
    
    for i in range(len(obj_names)):
        for j in range(i + 1, len(obj_names)):
            bbox1 = object_bboxes[obj_names[i]]
            bbox2 = object_bboxes[obj_names[j]]
            
            is_collision, penetration_depth = check_bbox_collision(bbox1, bbox2, collision_tolerance)
            
            if is_collision:
                num_colliding_pairs += 1
                colliding_objects.add(obj_names[i])
                colliding_objects.add(obj_names[j])
                total_penetration_depth += penetration_depth
    
    num_colliding_objects = len(colliding_objects)
    collision_rate = (num_colliding_objects / total_objects * 100) if total_objects > 0 else 0.0
    mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0
    
    # --- 出界检测 ---
    num_oob_objects = 0
    total_oob_volume = 0
    
    for name, bbox in object_bboxes.items():
        is_oob, oob_volume = check_bbox_out_of_bounds(
            bbox, room_bbox, floor_polygon, room_height_min, room_height_max, oob_tolerance
        )
        if is_oob:
            num_oob_objects += 1
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


def evaluate_single_scene_bbox(scene_json, models_base_path, format_type='ours', collision_tolerance=0.0, oob_tolerance=0.0):
    """评估单个场景的指标（基于AABB边界框）"""
    try:
        room_bbox, object_bboxes, floor_polygon, room_height_min, room_height_max = parse_scene_from_json_bbox(
            scene_json, models_base_path, format_type
        )
        
        physics_metrics = calculate_physics_metrics_bbox(
            room_bbox, object_bboxes, floor_polygon, room_height_min, room_height_max,
            collision_tolerance, oob_tolerance
        )
        
        return physics_metrics, True
        
    except Exception as e:
        print(f"评估场景时出错: {e}")
        import traceback
        traceback.print_exc()
        return {}, False


def evaluate_single_scene_file_bbox(json_file: str, models_base_path: str, format_type: str, 
                                     collision_tolerance: float = 0.0, oob_tolerance: float = 0.0) -> dict:
    """评估单个场景文件（用于多进程调用）"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            scene_json = json.load(f)
        
        metrics, success = evaluate_single_scene_bbox(scene_json, models_base_path, format_type,
                                                       collision_tolerance, oob_tolerance)
        
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


def batch_evaluate_scenes_bbox(scenes_directory, models_base_path, max_scenes=None, format_type='ours', 
                                num_workers=None, output_dir=None, collision_tolerance=0.0, oob_tolerance=0.0):
    """批量评估目录中的所有JSON场景文件（基于AABB边界框）"""
    tolerance_mode = "无容差" if collision_tolerance == 0 and oob_tolerance == 0 else f"碰撞容差={collision_tolerance}m, 出界容差={oob_tolerance}m"
    print(f"开始批量评估场景 (BBOX模式，{tolerance_mode})...")
    print(f"场景目录: {scenes_directory}")
    print(f"模型路径: {models_base_path}")
    
    # 获取所有JSON文件
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
    
    eval_func = partial(evaluate_single_scene_file_bbox, models_base_path=models_base_path, format_type=format_type,
                        collision_tolerance=collision_tolerance, oob_tolerance=oob_tolerance)
    
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
    
    # 计算综合统计指标
    tolerance_mode = "无容差" if collision_tolerance == 0 and oob_tolerance == 0 else f"碰撞容差={collision_tolerance}m, 出界容差={oob_tolerance}m"
    print(f"\n--- 综合评估报告 (BBOX模式，{tolerance_mode}) ---")
    
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
    
    # 保存详细结果到文件
    result_output_dir = output_dir if output_dir else scenes_directory
    os.makedirs(result_output_dir, exist_ok=True)
    
    # 根据容差设置确定输出文件名
    if collision_tolerance == 0 and oob_tolerance == 0:
        output_file = os.path.join(result_output_dir, "evaluation_results_bbox.json")
        eval_mode = "BBOX (无容差)"
    else:
        output_file = os.path.join(result_output_dir, f"evaluation_results_bbox_tol{collision_tolerance}_{oob_tolerance}.json")
        eval_mode = f"BBOX (碰撞容差={collision_tolerance}m, 出界容差={oob_tolerance}m)"
    
    try:
        results = {
            "evaluation_mode": eval_mode,
            "collision_tolerance": collision_tolerance,
            "oob_tolerance": oob_tolerance,
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
    parser = argparse.ArgumentParser(description='基于BBOX的3D场景评估工具')
    parser.add_argument('--format', type=str, default='respace', choices=['ours', 'respace'],
                       help='场景JSON格式类型: ours (groups结构) 或 respace (直接objects结构)')
    parser.add_argument('--scenes_dir', type=str, 
                       default='/path/to/SceneReVis/output/sft_65k/final_scenes_collection', 
                       help='包含JSON场景文件的目录路径')
    parser.add_argument('--models_path', type=str, 
                       default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                       help='3D模型文件的基础路径')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='最大处理场景数量 (默认: 处理所有场景)')
    parser.add_argument('--workers', type=int, default=None,
                       help='并行进程数 (默认: CPU核心数的一半)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录路径 (默认: 与scenes_dir相同)')
    parser.add_argument('--collision_tolerance', type=float, default=0.0,
                       help='碰撞检测容差(米)，穿透深度超过此值才算碰撞 (默认: 0，无容差)')
    parser.add_argument('--oob_tolerance', type=float, default=0.0,
                       help='出界检测容差(米)，超出边界此值才算出界 (默认: 0，无容差)')
    
    args = parser.parse_args()
    
    os.makedirs(args.models_path, exist_ok=True)
    
    tolerance_mode = "无容差" if args.collision_tolerance == 0 and args.oob_tolerance == 0 else f"碰撞容差={args.collision_tolerance}m, 出界容差={args.oob_tolerance}m"
    print(f"=== 3D场景评估工具 (BBOX模式，{tolerance_mode}) ===")
    print(f"场景格式: {args.format}")
    print(f"场景目录: {args.scenes_dir}")
    print(f"模型路径: {args.models_path}")
    if args.max_scenes:
        print(f"最大场景数: {args.max_scenes}")
    if args.workers:
        print(f"并行进程数: {args.workers}")
    if args.output_dir:
        print(f"输出目录: {args.output_dir}")
    print()
    
    batch_evaluate_scenes_bbox(args.scenes_dir, args.models_path, args.max_scenes, args.format, 
                                args.workers, args.output_dir, args.collision_tolerance, args.oob_tolerance)
