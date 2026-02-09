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
    
    搜索顺序（与 blender_renderer.py 保持一致）:
    1. OBJAVERSE_GLB_CACHE_DIR 环境变量指定的路径
    2. /path/to/data/datasets/objathor-assets/glbs (本地开发)
    3. /path/to/datasets/objathor-assets/glbs (云存储)
    4. ~/.objaverse/hf-objaverse-v1/glbs (本地缓存)
    
    使用 uid[:2] 直接构建子目录路径（优化：避免遍历所有子目录）
    
    Args:
        uid: Objaverse 资产 UID
        
    Returns:
        GLB 文件路径，如果未找到则返回 None
    """
    if not uid or len(uid) < 2:
        return None
    
    # 0. Special check for LayoutVLM processed objaverse (structure: {uid}/{uid}.glb)
    layout_vlm_processed = Path("/path/to/SceneReVis/baseline/LayoutVLM/objaverse_processed")
    if layout_vlm_processed.is_dir():
        candidate = layout_vlm_processed / uid / f"{uid}.glb"
        if candidate.is_file():
            return candidate

    # GLB 缓存目录列表（按优先级排序，与 blender_renderer.py 保持一致）
    cache_dirs = []
    
    # 1. 环境变量指定的路径
    env_cache = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
    if env_cache:
        cache_dirs.append(Path(env_cache) / "glbs")
    
    # 2. 本地开发路径
    cache_dirs.append(Path("/path/to/data/datasets/objathor-assets/glbs"))
    
    # 3. 云存储
    cache_dirs.append(Path("/path/to/datasets/objathor-assets/glbs"))
    
    # 4. 本地缓存
    cache_dirs.append(Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs")
    
    # 使用 uid[:2] 直接构建子目录路径（优化）
    subdir_name = uid[:2]
    
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        
        # 直接查找 uid[:2] 子目录下的 GLB 文件
        candidate = cache_dir / subdir_name / f"{uid}.glb"
        if candidate.is_file():
            return candidate
    
    return None


# --- 1. 数据解析与准备 ---

def get_object_field(obj, field_name, format_type='ours'):
    """
    从对象中获取字段值,处理不同格式的字段名称
    
    Args:
        obj: 对象字典
        field_name: 字段名称 ('jid', 'desc', 'size')
        format_type: 'ours' 或 'respace'
    
    Returns:
        字段值
    """
    if format_type == 'respace':
        # respace格式使用sampled_asset_前缀
        if field_name == 'jid':
            return obj.get('sampled_asset_jid', obj.get('jid', 'N/A'))
        elif field_name == 'desc':
            return obj.get('sampled_asset_desc', obj.get('desc', 'N/A'))
        elif field_name == 'size':
            return obj.get('sampled_asset_size', obj.get('size', [1, 1, 1]))
    else:
        # ours格式直接使用字段名
        return obj.get(field_name, 'N/A' if field_name in ['jid', 'desc'] else [1, 1, 1])


# ============================================================================
# 公共函数：房间多边形和mesh创建（支持异型房间）
# ============================================================================

def create_floor_polygon(bounds_bottom):
    """
    从房间底部边界创建地板多边形（支持异型房间如L型、T型等）
    
    Args:
        bounds_bottom: 底部边界顶点列表 [[x1,y1,z1], [x2,y2,z2], ...]
    
    Returns:
        shapely.geometry.Polygon: 地板多边形（使用X和Z坐标）
    """
    # 使用X和Z坐标创建2D多边形（Y是高度方向）
    points = [(pt[0], pt[2]) for pt in bounds_bottom]
    return Polygon(points)


def create_room_mesh(bounds_bottom, bounds_top):
    """
    从多边形边界创建真实的房间mesh（支持异型房间）
    使用三角化方法创建底面、顶面和侧面
    
    Args:
        bounds_bottom: 底部边界顶点列表
        bounds_top: 顶部边界顶点列表
    
    Returns:
        trimesh.Trimesh: 房间mesh对象
    """
    bounds_bottom = np.array(bounds_bottom)
    bounds_top = np.array(bounds_top)
    
    # 创建地板多边形
    floor_polygon = create_floor_polygon(bounds_bottom.tolist())
    
    num_verts = len(bounds_bottom)
    all_vertices = np.concatenate([bounds_bottom, bounds_top], axis=0)
    
    # 使用trimesh三角化地板多边形
    try:
        vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="triangle")
    except Exception:
        # 如果triangle引擎失败，尝试使用earcut
        try:
            vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="earcut")
        except Exception:
            # 最后的回退方案：使用简单的fan三角化
            floor_faces = np.array([[0, i, i+1] for i in range(1, num_verts-1)])
    
    # 移除无效面（索引超出范围）
    valid_mask = np.all(floor_faces < num_verts, axis=1)
    floor_faces = floor_faces[valid_mask]
    
    # 创建天花板面（偏移索引到顶部顶点）
    ceiling_faces = floor_faces + num_verts
    # 翻转天花板面的绕向以确保法线朝向内部
    ceiling_faces = ceiling_faces[:, ::-1]
    
    # 创建侧面
    side_faces = []
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        # 两个三角形组成一个四边形侧面
        side_faces.append([i, next_i, i + num_verts])
        side_faces.append([next_i, next_i + num_verts, i + num_verts])
    side_faces = np.array(side_faces)
    
    # 合并所有面
    all_faces = np.concatenate([floor_faces, ceiling_faces, side_faces], axis=0)
    
    # 创建mesh并修复法线
    room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    trimesh.repair.fix_normals(room_mesh)
    
    return room_mesh


def check_object_out_of_bounds(obj_mesh, room_mesh, floor_polygon, room_height_min, room_height_max, num_samples=500):
    """
    使用mesh containment检测物体是否出界（支持异型房间和高度检测）
    
    Args:
        obj_mesh: 物体的trimesh对象
        room_mesh: 房间的trimesh对象
        floor_polygon: 房间地板的shapely多边形
        room_height_min: 房间最低高度（地板Y坐标）
        room_height_max: 房间最高高度（天花板Y坐标）
        num_samples: 采样点数量
    
    Returns:
        (is_oob: bool, oob_volume: float)
    """
    try:
        # 在物体mesh上采样点
        sample_points = obj_mesh.sample(num_samples)
        
        # 检测2D出界（XZ平面）
        # 注意：shapely的contains()对于边界上的点返回False，这会导致贴墙放置的家具被误判为出界
        # 解决方案：给多边形添加微小缓冲区(1mm)，使边界上的点被正确判定为在内部
        from shapely.geometry import Point
        buffered_polygon = floor_polygon.buffer(0.001)  # 1mm缓冲区
        points_2d = [Point(pt[0], pt[2]) for pt in sample_points]
        inside_2d = np.array([buffered_polygon.contains(p) for p in points_2d])
        
        # 检测高度出界（Y方向）- 同样添加微小容差
        height_tolerance = 0.001  # 1mm容差
        inside_height = (sample_points[:, 1] >= room_height_min - height_tolerance) & (sample_points[:, 1] <= room_height_max + height_tolerance)
        
        # 综合判断：必须同时在2D范围内且在高度范围内
        inside = inside_2d & inside_height
        
        oob_ratio = 1.0 - (inside.sum() / len(inside))
        
        if oob_ratio > 0.01:  # 超过1%的点在外面
            is_oob = True
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            oob_volume = obj_volume * oob_ratio
        else:
            is_oob = False
            oob_volume = 0.0
        
        return is_oob, oob_volume
        
    except Exception as e:
        print(f"警告: 出界检测采样失败: {e}，使用边界框备用方案")
        # 备用方案：使用简单的边界框检测
        obj_bounds = obj_mesh.bounds
        room_bounds = room_mesh.bounds
        
        if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            return True, obj_volume * 0.1
        return False, 0.0


def parse_scene_data(scene_json, format_type='ours'):
    """
    解析场景JSON数据,支持两种格式
    
    Args:
        scene_json: 场景JSON数据
        format_type: 'ours' 或 'respace'
    
    Returns:
        bounds_bottom, bounds_top, all_objects_data
    """
    if format_type == 'respace':
        # respace格式:objects直接在根级别,bounds直接在根级别
        bounds_bottom = scene_json.get('bounds_bottom', [])
        bounds_top = scene_json.get('bounds_top', [])
        all_objects_data = scene_json.get('objects', [])
    else:
        # ours格式:使用room_envelope和groups结构
        bounds_bottom = scene_json['room_envelope']['bounds_bottom']
        bounds_top = scene_json['room_envelope']['bounds_top']
        all_objects_data = []
        if 'groups' in scene_json and scene_json['groups'] is not None:
            for group in scene_json['groups']:
                if 'objects' in group and group['objects'] is not None:
                    all_objects_data.extend(group['objects'])
    
    return bounds_bottom, bounds_top, all_objects_data


def parse_scene_from_json(scene_json, models_base_path, format_type='ours'):
    """
    从JSON数据中解析场景,并加载、变换真实的三维模型。
    支持两种JSON格式:'ours' 和 'respace'
    """
    # 使用新的解析函数获取场景数据
    bounds_bottom, bounds_top, all_objects_data = parse_scene_data(scene_json, format_type)
    
    # 直接使用所有对象，不进行去重
    unique_objects_data = all_objects_data
    
    print(f"对象数量: {len(all_objects_data)}")

    # 创建房间的trimesh对象（使用多边形挤出支持异型房间）
    room_mesh = create_room_mesh(bounds_bottom, bounds_top)
    
    # 创建地板多边形用于2D出界检测
    floor_polygon = create_floor_polygon(bounds_bottom.tolist() if isinstance(bounds_bottom, np.ndarray) else bounds_bottom)
    
    # 计算房间高度范围
    bounds_bottom_arr = np.array(bounds_bottom)
    bounds_top_arr = np.array(bounds_top)
    room_height_min = bounds_bottom_arr[:, 1].min()
    room_height_max = bounds_top_arr[:, 1].max()

    # 加载并变换家具的trimesh对象
    furniture_objects = {}
    for i, obj_data in enumerate(unique_objects_data):
        # 确定资产来源
        asset_source = obj_data.get('asset_source', '3d-future')
        target_size = get_object_field(obj_data, 'size', format_type)
        
        model_path = None
        asset_id = None
        
        if asset_source == 'objaverse':
            # Objaverse 资产：使用 uid 和 GLB 缓存
            uid = obj_data.get('uid')
            if uid:
                asset_id = uid
                model_path = find_objaverse_glb(uid)
                if model_path:
                    model_path = str(model_path)
        else:
            # 3D-FUTURE 资产：使用 jid
            jid = get_object_field(obj_data, 'jid', format_type)
            asset_id = jid
            model_path = os.path.join(models_base_path, jid, 'raw_model.glb')
        
        if asset_id is None:
            asset_id = f'object_{i}'
        
        if model_path is None or not os.path.exists(model_path):
            print(f"警告: 找不到模型文件 {asset_id} (source: {asset_source})。将使用长方体代替。")
            mesh = trimesh.creation.box(extents=target_size)
        else:
            loaded = trimesh.load(model_path)
            # 处理 trimesh.load 返回 Scene 对象的情况
            if isinstance(loaded, trimesh.Scene):
                # 将场景中所有几何体合并为一个 mesh
                if len(loaded.geometry) > 0:
                    mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                else:
                    print(f"警告: 模型文件 {asset_id} 为空场景。将使用长方体代替。")
                    mesh = trimesh.creation.box(extents=target_size)
            else:
                mesh = loaded

        # --- 应用变换：1.缩放 -> 2.旋转 -> 3.平移 ---
        
        # 1. 缩放 (Scale)
        # 计算缩放比例以匹配JSON中定义的尺寸
        original_size = mesh.extents
        # 防止除以零
        target_size_array = np.array(target_size)
        scale_factors = target_size_array / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        # 2. 旋转 (Rotate) & 3. 平移 (Translate)
        pos = obj_data['pos']
        rot_xyzw = obj_data['rot']
        
        try:
            rotation = R.from_quat(rot_xyzw)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation.as_matrix()
            transform_matrix[:3, 3] = pos
            
            # --- START: 核心修改区域 ---
            # 之前是使用 'mesh.bounds.mean(axis=0)'，这会导致物体从几何中心定位，引发OOB错误。
            # 现在我们计算物体的“底部中心”作为锚点。
            bounds = mesh.bounds
            bottom_center_pivot = np.array([
                (bounds[0, 0] + bounds[1, 0]) / 2,  # X 轴的中心
                bounds[0, 1],                      # Y 轴的底部 (最小值)
                (bounds[0, 2] + bounds[1, 2]) / 2   # Z 轴的中心
            ])

            # 创建一个变换，将物体的“底部中心”移动到坐标原点
            center_transform = np.eye(4)
            center_transform[:3, 3] = -bottom_center_pivot
            # --- END: 核心修改区域 ---
            
            # 完整变换：先将锚点移到原点，再应用旋转和平移到最终位置
            mesh.apply_transform(center_transform)
            mesh.apply_transform(transform_matrix)
            
            # 使用 asset_id 的前8个字符作为标识
            furniture_objects[f"object_{i}_{asset_id[:8]}"] = mesh
        except Exception as e:
            desc = get_object_field(obj_data, 'desc', format_type)
            print(f"处理对象时出错: {desc}, 错误: {e}")

    # 返回房间mesh、家具对象、原始数据、地板多边形和高度范围
    return room_mesh, furniture_objects, unique_objects_data, floor_polygon, room_height_min, room_height_max


# --- 2. 物理有效性指标 ---

def calculate_physics_metrics(room_mesh, objects, floor_polygon=None, room_height_min=None, room_height_max=None):
    """
    计算碰撞和出界指标。
    此版本已删除碰撞容差，并支持异型房间的精确出界检测。
    
    Args:
        room_mesh: 房间的trimesh对象
        objects: 家具对象字典 {name: mesh}
        floor_polygon: 房间地板的shapely多边形（用于异型房间检测）
        room_height_min: 房间最低高度
        room_height_max: 房间最高高度
    """
    # 获取对象总数
    total_objects = len(objects)
    
    manager = trimesh.collision.CollisionManager()
    
    # 仅将家具对象添加到碰撞管理器
    for name, mesh in objects.items():
        manager.add_object(name, mesh)

    # 获取所有潜在的接触数据
    is_collision, contact_data = manager.in_collision_internal(return_data=True)
    
    # 所有穿透深度 > 0.01 的都视为碰撞（添加1cm容差）
    collision_tolerance = 0.01
    actual_collisions_data = [d for d in contact_data if d.depth > collision_tolerance]
    
    num_colliding_pairs = len(actual_collisions_data)
    
    # 计算涉及碰撞的对象数量（新增指标1：碰撞率）
    colliding_objects = set()
    for contact in actual_collisions_data:
        # contact包含碰撞的两个对象名称
        if hasattr(contact, 'names'):
            # contact.names 可能是 set 或其他集合类型
            if isinstance(contact.names, (list, tuple)):
                colliding_objects.add(contact.names[0])
                colliding_objects.add(contact.names[1])
            else:
                # 如果是 set 或其他可迭代对象，直接添加所有元素
                colliding_objects.update(contact.names)
    
    num_colliding_objects = len(colliding_objects)
    collision_rate = (num_colliding_objects / total_objects * 100) if total_objects > 0 else 0.0
    
    total_penetration_depth = 0
    if num_colliding_pairs > 0:
        for contact in actual_collisions_data:
            # 累加完整的穿透深度（删除容差减法）
            total_penetration_depth += contact.depth
            
    mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0

    # --- 出界检测（支持异型房间）---
    num_oob_objects = 0
    total_oob_volume = 0
    
    # 如果没有提供异型房间检测所需的参数，则使用边界框备用方案
    use_polygon_detection = (floor_polygon is not None and 
                             room_height_min is not None and 
                             room_height_max is not None)
    
    for name, mesh in objects.items():
        if use_polygon_detection:
            # 使用精确的多边形+高度检测（支持异型房间）
            is_oob, oob_volume = check_object_out_of_bounds(
                mesh, room_mesh, floor_polygon, room_height_min, room_height_max
            )
            if is_oob:
                num_oob_objects += 1
                total_oob_volume += oob_volume
        else:
            # 备用方案：使用边界框检测
            obj_bounds = mesh.bounds
            room_bounds = room_mesh.bounds
            
            if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
                num_oob_objects += 1
                
                # 计算出界体积
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
                        
                except Exception as e:
                    print(f"警告: 出界体积计算失败 {name}: {e}")
                    oob_volume = np.prod(mesh.extents) * 0.1
                
                total_oob_volume += oob_volume

    mean_oob_volume = (total_oob_volume / num_oob_objects) if num_oob_objects > 0 else 0
    
    # 计算越界率（新增指标2：越界率）
    out_of_bounds_rate = (num_oob_objects / total_objects * 100) if total_objects > 0 else 0.0

    metrics = {
        "Object Count": total_objects,
        "Collision-Free Rate (%)": 100.0 if num_colliding_pairs == 0 else 0.0,
        "Number of Colliding Pairs": num_colliding_pairs,
        "Collision Rate (%)": collision_rate,  # 新增：碰撞率
        "Mean Penetration Depth (m)": mean_penetration_depth,
        "Valid Placement Rate (%)": 100.0 if num_oob_objects == 0 else 0.0,
        "Number of Out-of-Bounds Objects": num_oob_objects,
        "Out-of-Bounds Rate (%)": out_of_bounds_rate,  # 新增：越界率
        "Mean Out-of-Bounds Volume (m^3)": mean_oob_volume
    }
    return metrics
    

# --- 功能性指标已禁用 ---
# 沙发可达性评估功能已被移除


# --- 5. 主程序 ---

def evaluate_single_scene(scene_json, models_base_path, create_virtual_models=True, format_type='ours'):
    """
    评估单个场景的指标
    支持两种JSON格式:'ours' 和 'respace'
    """
    # 为JSON中提到的JID创建虚拟的模型文件(如果需要)
    if create_virtual_models:
        all_jids = set()
        # 使用parse_scene_data获取所有对象
        _, _, all_objects_data = parse_scene_data(scene_json, format_type)
        for obj in all_objects_data:
            jid = get_object_field(obj, 'jid', format_type)
            all_jids.add(jid)
        
        for jid in all_jids:
            jid_dir = os.path.join(models_base_path, jid)
            os.makedirs(jid_dir, exist_ok=True)
            model_path = os.path.join(jid_dir, 'raw_model.glb')
            if not os.path.exists(model_path):
                # 创建一个简单的立方体作为占位符
                try:
                    trimesh.creation.box().export(model_path)
                except Exception as e:
                    print(f"警告: 无法创建虚拟模型 {model_path}: {e}")

    try:
        # 解析场景（现在返回额外的异型房间检测参数）
        room_mesh, objects, unique_objects_data, floor_polygon, room_height_min, room_height_max = parse_scene_from_json(scene_json, models_base_path, format_type)
        
        # 仅计算物理指标（功能性指标已禁用）
        physics_metrics = calculate_physics_metrics(room_mesh, objects, floor_polygon, room_height_min, room_height_max)
        
        # 返回物理指标
        all_metrics = physics_metrics
        return all_metrics, True
        
    except Exception as e:
        print(f"评估场景时出错: {e}")
        return {}, False


def evaluate_single_scene_file(json_file: str, models_base_path: str, format_type: str) -> dict:
    """
    评估单个场景文件（用于多进程调用）
    
    Args:
        json_file: JSON文件路径
        models_base_path: 3D模型文件的基础路径
        format_type: 场景JSON格式类型
    
    Returns:
        包含评估结果的字典
    """
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
    """
    批量评估目录中的所有JSON场景文件（支持多进程）
    
    Args:
        scenes_directory: 包含JSON文件的目录路径
        models_base_path: 3D模型文件的基础路径
        max_scenes: 最大处理场景数量，None表示处理所有场景
        format_type: 场景JSON格式类型,'ours' 或 'respace'
        num_workers: 并行进程数，None表示使用CPU核心数的一半
        output_dir: 输出目录路径，None表示使用scenes_directory
    """
    print(f"开始批量评估场景...")
    print(f"场景目录: {scenes_directory}")
    print(f"模型路径: {models_base_path}")
    
    # 获取所有JSON文件
    json_files = []
    if os.path.isdir(scenes_directory):
        for file in os.listdir(scenes_directory):
            if file.endswith('.json'):
                json_files.append(os.path.join(scenes_directory, file))
    else:
        print(f"错误: 目录不存在: {scenes_directory}")
        return
    
    json_files.sort()  # 按文件名排序
    
    if max_scenes and max_scenes > 0:
        json_files = json_files[:max_scenes]
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if len(json_files) == 0:
        print("未找到JSON文件!")
        return
    
    # 设置进程数
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = min(num_workers, len(json_files))
    
    print(f"使用 {num_workers} 个进程进行并行处理")
    
    # 存储所有场景的指标
    all_scene_metrics = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    print("\n开始处理场景...")
    print("=" * 80)
    
    # 使用多进程并行处理
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
    
    # 计算综合统计指标
    print("\n--- 综合评估报告 ---")
    
    # 收集所有指标的数值
    metric_names = list(all_scene_metrics[0]['metrics'].keys())
    metric_values = {name: [] for name in metric_names}
    
    for scene_data in all_scene_metrics:
        for name, value in scene_data['metrics'].items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metric_values[name].append(value)
    
    # 计算统计量
    print(f"\n基于 {successful_evaluations} 个成功评估的场景:")
    print("\n[ 物理有效性指标统计 ]")
    physics_metrics_names = [
        "Object Count",
        "Collision-Free Rate (%)",
        "Number of Colliding Pairs",
        "Collision Rate (%)",  # 新增：碰撞率
        "Mean Penetration Depth (m)",
        "Valid Placement Rate (%)",
        "Number of Out-of-Bounds Objects",
        "Out-of-Bounds Rate (%)",  # 新增：越界率
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
    
    # 功能性指标已禁用
    
    # 保存详细结果到文件
    # 使用自定义输出目录，如果未指定则使用场景目录
    result_output_dir = output_dir if output_dir else scenes_directory
    os.makedirs(result_output_dir, exist_ok=True)
    output_file = os.path.join(result_output_dir, "evaluation_results.json")
    try:
        results = {
            "summary": {
                "total_scenes": len(json_files),
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": failed_evaluations,
                "success_rate": successful_evaluations / len(json_files) * 100
            },
            "aggregate_statistics": {},
            "individual_results": all_scene_metrics
        }
        
        # 添加聚合统计
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量评估3D场景的物理和功能性指标 - 支持多种数据格式')
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
    
    args = parser.parse_args()
    
    # 确保模型路径存在
    os.makedirs(args.models_path, exist_ok=True)
    
    print("=== 3D场景评估工具 ===")
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
    
    # 执行批量评估
    batch_evaluate_scenes(args.scenes_dir, args.models_path, args.max_scenes, args.format, args.workers, args.output_dir)