import json
import os
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation as R
from shapely.geometry import box, Polygon
from trimesh.voxel.encoding import DenseEncoding
import matplotlib.pyplot as plt
import argparse
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


# --- 1. 从respace项目学习的体素评估方法 ---

def get_y_angle_from_xyzw_quaternion(quaternion_xyzw):
    """从四元数中提取Y轴旋转角度"""
    # 检查是否为单位四元数（恒等旋转）
    if np.allclose(quaternion_xyzw, [0, 0, 0, 1]):
        return 0.0, 0.0
    
    rotation = R.from_quat(quaternion_xyzw)
    # 使用'yxz'顺序避免gimbal lock警告
    try:
        euler = rotation.as_euler('yxz', degrees=False)
        return euler[0], euler[0]  # 返回Y轴旋转弧度
    except:
        # 如果仍有问题，直接从旋转矩阵计算Y角
        matrix = rotation.as_matrix()
        y_angle = np.arctan2(matrix[0, 2], matrix[2, 2])
        return y_angle, y_angle

def get_xz_bbox_from_obj(obj):
    """获取对象在XZ平面的2D边界框"""
    bbox_position = obj.get("pos")
    bbox_size = obj.get("size")
    
    rotation_xyzw = np.array(obj.get("rot"))
    asset_rot_angle_euler, asset_rot_angle_radians = get_y_angle_from_xyzw_quaternion(rotation_xyzw)
    
    half_size_x = bbox_size[0] / 2
    half_size_z = bbox_size[2] / 2
    corners_2d_floor = np.array([
        [half_size_x, half_size_z],
        [-half_size_x, half_size_z],
        [-half_size_x, -half_size_z],
        [half_size_x, -half_size_z]
    ])
    
    cos_theta = np.cos(asset_rot_angle_radians)
    sin_theta = np.sin(asset_rot_angle_radians)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    rotated_corners_2d_floor = np.dot(corners_2d_floor, rotation_matrix.T)
    translated_corners_2d_floor = rotated_corners_2d_floor + np.array([bbox_position[0], bbox_position[2]])
    
    polygon_coords_2d_floor = [(corner[0], corner[1]) for corner in translated_corners_2d_floor]
    bbox_2d_obj = Polygon(polygon_coords_2d_floor)
    
    # 获取3D边界框的高度信息
    obj_height = bbox_size[1]
    obj_y_start = bbox_position[1]
    obj_y_end = bbox_position[1] + obj_height
    
    return bbox_2d_obj, obj_height, obj_y_start, obj_y_end

def create_floor_plan_polygon(bounds_bottom):
    """从房间底部边界创建地板多边形"""
    points = [(pt[0], pt[2]) for pt in bounds_bottom]  # 使用X和Z坐标
    return Polygon(points)

def create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon):
    """创建房间mesh"""
    num_verts = len(bounds_bottom)
    all_vertices = np.array(bounds_bottom + bounds_top)
    
    # 使用trimesh创建地板三角形
    vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
    # 移除无效面
    idxs = []
    for i, row in enumerate(floor_faces):
        if np.any(row == num_verts):
            idxs.append(i)
    floor_faces = np.delete(floor_faces, idxs, axis=0)
    
    floor_mesh = trimesh.Trimesh(vertices=vtx, faces=floor_faces)
    
    # 创建天花板面
    ceiling_faces = floor_faces + num_verts
    
    # 创建侧面
    side_faces = []
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        side_faces.append([i, next_i, i + num_verts])
        side_faces.append([next_i, next_i + num_verts, i + num_verts])
    side_faces = np.array(side_faces)
    
    all_faces = np.concatenate((floor_faces, ceiling_faces, side_faces), axis=0)
    room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    
    trimesh.repair.fix_normals(room_mesh)
    return room_mesh

def voxelize_mesh_and_get_matrix(asset_mesh, voxel_size):
    """体素化mesh并返回体素矩阵"""
    asset_voxels = asset_mesh.voxelized(pitch=voxel_size).fill()
    asset_voxel_matrix = asset_voxels.matrix
    return asset_voxel_matrix

def prepare_asset_voxel(obj, models_base_path, voxel_size):
    """
    准备对象的体素表示 - 修复版
    关键修复：
    1. 先将mesh居中到底部中心，再应用旋转
    2. 使用体素网格的实际边界来计算正确的偏移
    
    支持两种资产来源:
    - 3D-FUTURE: 使用 'jid' 字段，从 models_base_path/{jid}/raw_model.glb 加载
    - Objaverse: 使用 'uid' 字段，从 GLB 缓存加载
    """
    # 确定资产来源和模型路径
    asset_source = obj.get('asset_source', '3d-future')
    model_path = None
    asset_id = None
    
    if asset_source == 'objaverse':
        # Objaverse 资产：使用 uid 和 GLB 缓存
        uid = obj.get('uid')
        if uid:
            asset_id = uid
            model_path = find_objaverse_glb(uid)
            if model_path:
                model_path = str(model_path)
    else:
        # 3D-FUTURE 资产：使用 jid
        jid = obj.get('jid', 'N/A')
        asset_id = jid
        model_path = os.path.join(models_base_path, jid, 'raw_model.glb')
    
    if asset_id is None:
        asset_id = 'unknown'
    
    if model_path is None or not os.path.exists(model_path):
        # 使用边界框创建替代mesh
        mesh = trimesh.creation.box(extents=obj['size'])
    else:
        try:
            asset_scene = trimesh.load(model_path)
            if isinstance(asset_scene, trimesh.Scene):
                mesh = asset_scene.to_geometry()
            else:
                mesh = asset_scene
        except Exception as e:
            print(f"加载mesh失败 {asset_id}: {e}")
            mesh = trimesh.creation.box(extents=obj['size'])
    
    # 1. 应用缩放到目标尺寸
    original_size = mesh.extents
    target_size = np.array(obj['size'])
    scale_factors = target_size / (original_size + 1e-6)
    mesh.apply_scale(scale_factors)
    
    # 2. 关键修复：将mesh居中到底部中心 (与myeval.py保持一致)
    # 计算当前mesh的底部中心
    bounds = mesh.bounds
    bottom_center = np.array([
        (bounds[0, 0] + bounds[1, 0]) / 2,  # X轴中心
        bounds[0, 1],                        # Y轴底部
        (bounds[0, 2] + bounds[1, 2]) / 2   # Z轴中心
    ])
    # 将底部中心移动到原点
    center_transform = np.eye(4)
    center_transform[:3, 3] = -bottom_center
    mesh.apply_transform(center_transform)
    
    # 3. 应用旋转（绕原点，即底部中心）
    rot_xyzw = obj['rot']
    rotation = R.from_quat(rot_xyzw)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    mesh.apply_transform(rotation_matrix)
    
    # 4. 体素化旋转后的mesh
    try:
        asset_voxels = mesh.voxelized(pitch=voxel_size).fill()
        asset_voxel_matrix = asset_voxels.matrix
        # 获取体素网格在世界坐标系中的原点
        voxel_origin = asset_voxels.translation  # 体素(0,0,0)对应的世界坐标
    except Exception as e:
        print(f"体素化失败 {obj.get('desc', 'Unknown')}: {e}")
        return None, None, mesh
    
    # 5. 计算体素空间的位置偏移
    # 物体的世界位置（底部中心）
    pos = np.array(obj['pos'])
    
    # 关键修复：正确计算物体体素在世界坐标中的位置
    # - voxel_origin 是体素(0,0,0)在本地坐标系中的位置（即mesh.bounds[0]）
    # - 物体的底部中心（本地坐标原点）移动到世界坐标 pos
    # - 所以物体体素(0,0,0)的世界坐标 = pos + voxel_origin
    # - 在房间体素网格中的索引 = (pos + voxel_origin - room_voxel_origin) / voxel_size
    # - 在 compute_voxel_oob 中，我们计算: room_origin_shift + asset_shift_from_origin
    # - room_origin_shift = -room_voxel_origin / voxel_size
    # - 所以 asset_shift_from_origin = (pos + voxel_origin) / voxel_size
    asset_shift_from_origin = np.floor((pos + voxel_origin) / voxel_size)
    
    return asset_voxel_matrix, asset_shift_from_origin, mesh

def occupancy_overlap(voxel_matrix_a, voxel_matrix_b, offset_b):
    """计算两个体素矩阵的重叠"""
    overlap_matrix = copy.deepcopy(voxel_matrix_a).astype(int)
    
    for i in range(voxel_matrix_b.shape[0]):
        for j in range(voxel_matrix_b.shape[1]):
            for k in range(voxel_matrix_b.shape[2]):
                if voxel_matrix_b[i, j, k]:
                    shifted_pos = (i + offset_b[0], j + offset_b[1], k + offset_b[2])
                    if (0 <= shifted_pos[0] < overlap_matrix.shape[0] and 
                        0 <= shifted_pos[1] < overlap_matrix.shape[1] and 
                        0 <= shifted_pos[2] < overlap_matrix.shape[2]):
                        overlap_matrix[shifted_pos[0], shifted_pos[1], shifted_pos[2]] += 1
    
    return (overlap_matrix == 2)

def compute_voxel_oob(obj, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume, models_base_path):
    """计算基于体素的出界体积"""
    asset_voxel_matrix, asset_shift_from_origin, mesh = prepare_asset_voxel(obj, models_base_path, voxel_size)
    
    if asset_voxel_matrix is None:
        return 0.0
    
    asset_offset = np.floor(room_origin_shift + asset_shift_from_origin).astype(int)
    
    inside_voxels = occupancy_overlap(room_voxel_matrix, asset_voxel_matrix, asset_offset)
    num_asset_voxels = np.sum(asset_voxel_matrix)
    asset_volume = num_asset_voxels * voxel_volume
    
    num_inside_voxels = np.sum(inside_voxels)
    inside_volume = num_inside_voxels * voxel_volume
    
    voxel_oob = asset_volume - inside_volume
    return max(0.0, voxel_oob)

def compute_voxel_collision(obj_x, obj_y, voxel_size, voxel_volume, models_base_path):
    """计算基于体素的碰撞体积"""
    asset_voxel_matrix_x, asset_shift_x, mesh_x = prepare_asset_voxel(obj_x, models_base_path, voxel_size)
    asset_voxel_matrix_y, asset_shift_y, mesh_y = prepare_asset_voxel(obj_y, models_base_path, voxel_size)
    
    if asset_voxel_matrix_x is None or asset_voxel_matrix_y is None:
        return 0.0
    
    offset = np.floor(asset_shift_y - asset_shift_x).astype(int)
    inside_voxels = occupancy_overlap(asset_voxel_matrix_x, asset_voxel_matrix_y, offset)
    
    num_inside_voxels = np.sum(inside_voxels)
    intersection_volume = num_inside_voxels * voxel_volume
    
    return intersection_volume

# --- 2. 传统几何评估方法（用于对比）---

def parse_scene_data(scene_json, format_type='ours'):
    """
    解析场景JSON数据，支持两种格式
    
    Args:
        scene_json: 场景JSON数据
        format_type: 'ours' 或 'respace'
    
    Returns:
        bounds_bottom, bounds_top, all_objects
    """
    if format_type == 'respace':
        # respace格式：objects直接在根级别
        bounds_bottom = scene_json.get('bounds_bottom', [])
        bounds_top = scene_json.get('bounds_top', [])
        all_objects = scene_json.get('objects', [])
    else:
        # ours格式：使用room_envelope和groups结构
        if 'room_envelope' not in scene_json:
            raise ValueError(f"场景JSON缺少'room_envelope'字段，请检查数据格式是否为'ours'格式")
        
        bounds_bottom = scene_json['room_envelope']['bounds_bottom']
        bounds_top = scene_json['room_envelope']['bounds_top']
        all_objects = []
        if 'groups' in scene_json and scene_json['groups'] is not None:
            for group in scene_json['groups']:
                if 'objects' in group and group['objects'] is not None:
                    all_objects.extend(group['objects'])
    
    return bounds_bottom, bounds_top, all_objects

def get_intersection_area(obj_x, obj_y, epsilon=1e-7):
    """计算两个对象在2D平面的交集面积"""
    intersection = obj_x.intersection(obj_y)
    if intersection.is_empty:
        return 0.0
    area = intersection.area
    if area < epsilon:
        return 0.0
    return area

def compute_geometric_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top, epsilon=1e-7):
    """计算基于几何的出界体积"""
    bbox_obj, obj_height, obj_y_start, obj_y_end = get_xz_bbox_from_obj(obj)
    
    intersection_area = get_intersection_area(floor_plan_polygon, bbox_obj)
    
    room_bottom = bounds_bottom[0][1]
    room_top = bounds_top[0][1]
    
    if (obj_y_start < room_bottom and obj_y_end < room_bottom) or (obj_y_start > room_top and obj_y_end > room_top):
        obj_intersection_height = 0
    else:
        obj_intersection_height = abs(np.clip(obj_y_end, room_bottom, room_top) - np.clip(obj_y_start, room_bottom, room_top))
    
    bbox_vol_total = (bbox_obj.area) * obj_height
    bbox_vol_inside = (intersection_area * obj_intersection_height)
    
    oob = bbox_vol_total - bbox_vol_inside
    
    if oob < epsilon:
        return 0.0
    
    return oob

def compute_geometric_collision(obj_x, obj_y, epsilon=1e-7):
    """计算基于几何的碰撞体积"""
    bbox_x, height_x, y_start_x, y_end_x = get_xz_bbox_from_obj(obj_x)
    bbox_y, height_y, y_start_y, y_end_y = get_xz_bbox_from_obj(obj_y)
    
    # 2D交集面积
    intersection_area = get_intersection_area(bbox_x, bbox_y)
    
    if intersection_area < epsilon:
        return 0.0
    
    # Y轴重叠高度
    y_overlap_start = max(y_start_x, y_start_y)
    y_overlap_end = min(y_end_x, y_end_y)
    
    if y_overlap_end <= y_overlap_start:
        return 0.0
    
    y_overlap_height = y_overlap_end - y_overlap_start
    collision_volume = intersection_area * y_overlap_height
    
    if collision_volume < epsilon:
        return 0.0
    
    return collision_volume

# --- 3. 主要评估函数 ---

def evaluate_scene_voxel_based(scene_json, models_base_path, voxel_size=0.05, format_type='ours'):
    """使用体素方法评估场景"""
    
    # 提取场景数据 - 支持两种格式
    bounds_bottom, bounds_top, all_objects = parse_scene_data(scene_json, format_type)
    floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
    
    # 创建房间mesh并体素化
    room_mesh = create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
    room_voxels = room_mesh.voxelized(pitch=voxel_size).fill()
    room_voxel_matrix = room_voxels.matrix
    
    # 关键修复：获取房间体素网格的实际原点
    # room_voxels.translation 是体素(0,0,0)对应的世界坐标
    room_voxel_origin = room_voxels.translation
    
    # room_origin_shift 用于将世界坐标转换为体素索引
    # 对于世界坐标 world_pos，对应的体素索引是:
    # voxel_idx = (world_pos - room_voxel_origin) / voxel_size
    # 但在 compute_voxel_oob 中，我们计算的是:
    # asset_offset = room_origin_shift + asset_shift_from_origin
    # 其中 asset_shift_from_origin = (pos - asset_voxel_origin) / voxel_size
    # 所以 room_origin_shift = -room_voxel_origin / voxel_size
    room_origin_shift = -room_voxel_origin / voxel_size
    
    voxel_volume = voxel_size ** 3
    
    # 计算指标
    voxel_oobs = []
    voxel_collisions = []
    geometric_oobs = []
    geometric_collisions = []
    
    for i, obj in enumerate(all_objects):
        # 体素出界
        voxel_oob = compute_voxel_oob(obj, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume, models_base_path)
        voxel_oobs.append(voxel_oob)
        
        # 几何出界 (用于对比)
        geometric_oob = compute_geometric_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top)
        geometric_oobs.append(geometric_oob)
        
        # 与其他对象的碰撞
        for j, other_obj in enumerate(all_objects[i+1:], i+1):
            # 体素碰撞
            voxel_collision = compute_voxel_collision(obj, other_obj, voxel_size, voxel_volume, models_base_path)
            voxel_collisions.append(voxel_collision)
            
            # 几何碰撞 (用于对比)
            geometric_collision = compute_geometric_collision(obj, other_obj)
            geometric_collisions.append(geometric_collision)
    
    # 汇总指标
    metrics = {
        "voxel_metrics": {
            "total_oob_volume": sum(voxel_oobs),
            "total_collision_volume": sum(voxel_collisions),
            "num_oob_objects": sum(1 for oob in voxel_oobs if oob > 1e-6),
            "num_collision_pairs": sum(1 for col in voxel_collisions if col > 1e-6),
            "mean_oob_volume": np.mean(voxel_oobs) if voxel_oobs else 0.0,
            "mean_collision_volume": np.mean(voxel_collisions) if voxel_collisions else 0.0,
            "voxel_size": voxel_size,
            "room_volume": np.sum(room_voxel_matrix) * voxel_volume
        },
        "geometric_metrics": {
            "total_oob_volume": sum(geometric_oobs),
            "total_collision_volume": sum(geometric_collisions),
            "num_oob_objects": sum(1 for oob in geometric_oobs if oob > 1e-6),
            "num_collision_pairs": sum(1 for col in geometric_collisions if col > 1e-6),
            "mean_oob_volume": np.mean(geometric_oobs) if geometric_oobs else 0.0,
            "mean_collision_volume": np.mean(geometric_collisions) if geometric_collisions else 0.0
        },
        "comparison": {
            "oob_volume_ratio": sum(voxel_oobs) / (sum(geometric_oobs) + 1e-9),
            "collision_volume_ratio": sum(voxel_collisions) / (sum(geometric_collisions) + 1e-9)
        }
    }
    
    return metrics

# --- 4. 批量评估函数 ---

def load_scene_from_file(json_file_path):
    """从文件加载场景JSON数据"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            scene_json = json.load(f)
        return scene_json
    except Exception as e:
        print(f"加载场景文件失败 {json_file_path}: {e}")
        return None

def evaluate_single_scene_file(json_file_path, models_base_path, voxel_size=0.05, format_type='ours'):
    """评估单个场景文件"""
    scene_json = load_scene_from_file(json_file_path)
    if scene_json is None:
        return None
    
    try:
        metrics = evaluate_scene_voxel_based(scene_json, models_base_path, voxel_size, format_type)
        return metrics
    except Exception as e:
        print(f"评估场景失败 {json_file_path}: {e}")
        return None

def evaluate_scene_file_wrapper(json_file_path: str, models_base_path: str, voxel_size: float, format_type: str) -> dict:
    """
    评估单个场景文件的包装器（用于多进程调用）
    """
    try:
        metrics = evaluate_single_scene_file(json_file_path, models_base_path, voxel_size, format_type)
        
        if metrics is not None:
            voxel_metrics = metrics["voxel_metrics"]
            total_oob_loss = voxel_metrics['total_oob_volume']
            total_mbl_loss = voxel_metrics['total_collision_volume']
            total_pbl_loss = total_oob_loss + total_mbl_loss
            
            loss_threshold = 0.1
            is_valid_scene = bool(total_pbl_loss <= loss_threshold)
            
            return {
                'scene_name': os.path.basename(json_file_path),
                'scene_path': json_file_path,
                'metrics': metrics,
                'total_oob_loss': total_oob_loss,
                'total_mbl_loss': total_mbl_loss,
                'total_pbl_loss': total_pbl_loss,
                'is_valid_scene': is_valid_scene,
                'success': True
            }
        else:
            return {
                'scene_name': os.path.basename(json_file_path),
                'success': False,
                'error': '评估失败'
            }
    except Exception as e:
        return {
            'scene_name': os.path.basename(json_file_path),
            'success': False,
            'error': str(e)
        }


def batch_evaluate_scenes(scenes_directory, models_base_path, voxel_size=0.05, format_type='ours', num_workers=None):
    """批量评估目录下的所有场景文件（支持多进程）"""
    
    if not os.path.exists(scenes_directory):
        print(f"错误：场景目录不存在: {scenes_directory}")
        return None
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(scenes_directory) if f.endswith('.json')]
    if not json_files:
        print(f"警告：在目录 {scenes_directory} 中未找到JSON文件")
        return None
    
    print(f"找到 {len(json_files)} 个场景文件")
    
    # 存储所有场景的评估结果
    all_scene_results = []
    all_scene_metrics = {
        'total_oob_losses': [],
        'total_mbl_losses': [], 
        'total_pbl_losses': [],
        'valid_scenes': [],
        'scene_names': []
    }
    
    # 创建必需的模型目录（确保所有jid都有对应的虚拟模型）
    all_jids = set()
    
    # 预处理：收集所有jid
    for json_file in json_files:
        json_file_path = os.path.join(scenes_directory, json_file)
        scene_json = load_scene_from_file(json_file_path)
        if scene_json is not None:
            # 根据格式提取objects
            try:
                _, _, objects = parse_scene_data(scene_json, format_type)
                for obj in objects:
                    all_jids.add(obj.get('jid', 'N/A'))
            except (ValueError, KeyError) as e:
                print(f"警告: 跳过文件 {json_file} - {e}")
                continue
    
    # 创建虚拟模型文件
    print(f"为 {len(all_jids)} 个不同的模型创建虚拟模型文件...")
    for jid in all_jids:
        if jid != 'N/A':
            jid_dir = os.path.join(models_base_path, jid)
            os.makedirs(jid_dir, exist_ok=True)
            model_path = os.path.join(jid_dir, 'raw_model.glb')
            if not os.path.exists(model_path):
                trimesh.creation.box().export(model_path)
    
    # 设置进程数
    if num_workers is None:
        # 由于体素化是内存密集型，默认使用较少的进程
        num_workers = max(1, min(8, multiprocessing.cpu_count() // 2))
    num_workers = min(num_workers, len(json_files))
    
    print(f"使用 {num_workers} 个进程进行并行处理")
    
    # 批量评估（多进程）
    failed_scenes = []
    json_file_paths = [os.path.join(scenes_directory, f) for f in json_files]
    
    eval_func = partial(evaluate_scene_file_wrapper, 
                        models_base_path=models_base_path, 
                        voxel_size=voxel_size, 
                        format_type=format_type)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_func, path): path for path in json_file_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估场景"):
            json_file_path = futures[future]
            try:
                result = future.result()
                
                if result['success']:
                    all_scene_metrics['total_oob_losses'].append(result['total_oob_loss'])
                    all_scene_metrics['total_mbl_losses'].append(result['total_mbl_loss'])
                    all_scene_metrics['total_pbl_losses'].append(result['total_pbl_loss'])
                    all_scene_metrics['valid_scenes'].append(result['is_valid_scene'])
                    all_scene_metrics['scene_names'].append(result['scene_name'])
                    all_scene_results.append(result)
                else:
                    failed_scenes.append(result['scene_name'])
                    
            except Exception as e:
                failed_scenes.append(os.path.basename(json_file_path))
                print(f"\n   ✗ 处理文件时出错: {e}")
    
    if failed_scenes:
        print(f"\n警告：{len(failed_scenes)} 个场景评估失败")
    
    return all_scene_results, all_scene_metrics

# --- 5. 主程序 ---

if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='场景体素评估工具 - 支持多种数据格式')
    parser.add_argument('--format', type=str, default='respace', choices=['ours', 'respace'],
                        help='场景JSON格式类型: ours (groups结构) 或 respace (直接objects结构)')
    parser.add_argument('--scenes_dir', type=str, 
                        default='/path/to/SceneReVis/output/sft_65k/final_scenes_collection',
                        help='场景JSON文件所在目录')
    parser.add_argument('--models_path', type=str,
                        default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                        help='3D模型文件基础路径')
    parser.add_argument('--output_file', type=str,
                        default='/path/to/SceneReVis/eval/ours_voxel_evaluation_results_new.json',
                        help='评估结果输出文件路径')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='体素大小 (单位: 米)')
    parser.add_argument('--workers', type=int, default=None,
                        help='并行进程数 (默认: 自动选择, 最多4个)')
    
    args = parser.parse_args()
    
    # 定义路径
    SCENES_DIRECTORY = args.scenes_dir
    MODELS_BASE_PATH = args.models_path
    OUTPUT_FILE = args.output_file
    FORMAT_TYPE = args.format
    
    os.makedirs(MODELS_BASE_PATH, exist_ok=True)
    
    # 运行批量评估
    print("=== 批量场景评估 ===")
    print(f"场景格式: {FORMAT_TYPE}")
    print(f"场景目录: {SCENES_DIRECTORY}")
    print(f"模型路径: {MODELS_BASE_PATH}")
    print(f"输出文件: {OUTPUT_FILE}")
    if args.workers:
        print(f"并行进程数: {args.workers}")
    
    # 测试不同的体素分辨率
    voxel_sizes = [args.voxel_size]  # 使用命令行指定的体素大小
    
    for voxel_size in voxel_sizes:
        print(f"\n=== 使用体素大小: {voxel_size}m 进行批量评估 ===")
        
        # 批量评估所有场景 - 使用指定的格式
        all_scene_results, all_scene_metrics = batch_evaluate_scenes(
            SCENES_DIRECTORY, MODELS_BASE_PATH, voxel_size, FORMAT_TYPE, args.workers
        )
        
        if all_scene_results is None:
            print("批量评估失败")
            continue
        
        # 计算汇总统计
        total_scenes = len(all_scene_results)
        successful_scenes = len(all_scene_metrics['total_oob_losses'])
        
        if successful_scenes == 0:
            print("没有成功评估的场景")
            continue
        
        # 计算均值和标准差
        mean_total_oob_loss = np.mean(all_scene_metrics['total_oob_losses'])
        mean_total_mbl_loss = np.mean(all_scene_metrics['total_mbl_losses'])
        mean_total_pbl_loss = np.mean(all_scene_metrics['total_pbl_losses'])
        
        std_total_oob_loss = np.std(all_scene_metrics['total_oob_losses'])
        std_total_mbl_loss = np.std(all_scene_metrics['total_mbl_losses'])
        std_total_pbl_loss = np.std(all_scene_metrics['total_pbl_losses'])
        
        # # 缩放到与respace相同的单位 (x 1000)
        mean_total_oob_loss_scaled = mean_total_oob_loss * 1e3
        mean_total_mbl_loss_scaled = mean_total_mbl_loss * 1e3
        mean_total_pbl_loss_scaled = mean_total_pbl_loss * 1e3
        
        std_total_oob_loss_scaled = std_total_oob_loss * 1e3
        std_total_mbl_loss_scaled = std_total_mbl_loss * 1e3
        std_total_pbl_loss_scaled = std_total_pbl_loss * 1e3
        
        # 计算有效场景统计
        valid_scenes_count = sum(all_scene_metrics['valid_scenes'])
        valid_scene_ratio = valid_scenes_count / successful_scenes
        
        # 输出RESPACE风格的汇总结果
        print(f"\n=== 批量评估汇总结果 ({successful_scenes}/{total_scenes} 场景) ===")
        print(f"total_oob_loss: {mean_total_oob_loss_scaled:.2f} (+/- {std_total_oob_loss_scaled:.2f}) (x 0.001)")
        print(f"total_mbl_loss: {mean_total_mbl_loss_scaled:.2f} (+/- {std_total_mbl_loss_scaled:.2f}) (x 0.001)")
        print(f"total_pbl_loss: {mean_total_pbl_loss_scaled:.2f} (+/- {std_total_pbl_loss_scaled:.2f}) (x 0.001)")
        print("")
        print(f"valid_scene_ratio_pbl: {valid_scene_ratio:.2f} (+/- 0.00)")
        print(f"valid_scenes_count: {valid_scenes_count}/{successful_scenes}")
        print("")
        
        # 详细统计
        print("=== 详细统计信息 ===")
        print(f"总场景数: {total_scenes}")
        print(f"成功评估: {successful_scenes}")
        print(f"失败评估: {total_scenes - successful_scenes}")
        print(f"有效场景: {valid_scenes_count}")
        print(f"无效场景: {successful_scenes - valid_scenes_count}")
        print("")
        
        print("平均损失指标 (均值 ± 标准差):")
        print(f"  OOB损失: {mean_total_oob_loss:.6f} ± {std_total_oob_loss:.6f}")
        print(f"  MBL损失: {mean_total_mbl_loss:.6f} ± {std_total_mbl_loss:.6f}")
        print(f"  PBL损失: {mean_total_pbl_loss:.6f} ± {std_total_pbl_loss:.6f}")
        print("")
        
        # 单个场景详情
        print("=== 各场景评估详情 ===")
        for i, result in enumerate(all_scene_results):
            scene_name = result['scene_name']
            oob_loss = result['total_oob_loss']
            mbl_loss = result['total_mbl_loss']
            pbl_loss = result['total_pbl_loss']
            is_valid = result['is_valid_scene']
            
            status_icon = "✅" if is_valid else "❌"
            print(f"{i+1:2d}. {status_icon} {scene_name}")
            print(f"     OOB: {oob_loss:.6f} | MBL: {mbl_loss:.6f} | PBL: {pbl_loss:.6f}")
        
        # 保存批量评估结果
        output_metrics = {
            "batch_evaluation_summary": {
                "format_type": FORMAT_TYPE,
                "total_scenes": total_scenes,
                "successful_scenes": successful_scenes,
                "failed_scenes": total_scenes - successful_scenes,
                "valid_scenes_count": valid_scenes_count,
                "valid_scene_ratio_pbl": valid_scene_ratio,
                "voxel_size_used": voxel_size,
                "loss_threshold": 0.1,
                
                # 均值指标 (原始单位)
                "mean_total_oob_loss": mean_total_oob_loss,
                "mean_total_mbl_loss": mean_total_mbl_loss,
                "mean_total_pbl_loss": mean_total_pbl_loss,
                
                # 标准差指标 (原始单位)
                "std_total_oob_loss": std_total_oob_loss,
                "std_total_mbl_loss": std_total_mbl_loss,
                "std_total_pbl_loss": std_total_pbl_loss,
                
                # 均值指标 (缩放单位 x1000)
                "mean_total_oob_loss_scaled": mean_total_oob_loss_scaled,
                "mean_total_mbl_loss_scaled": mean_total_mbl_loss_scaled,
                "mean_total_pbl_loss_scaled": mean_total_pbl_loss_scaled,
                
                # 标准差指标 (缩放单位 x1000)
                "std_total_oob_loss_scaled": std_total_oob_loss_scaled,
                "std_total_mbl_loss_scaled": std_total_mbl_loss_scaled,
                "std_total_pbl_loss_scaled": std_total_pbl_loss_scaled,
                
                # 平均损失
                "mean_oob_loss": np.mean(all_scene_metrics['total_oob_losses']),
                "mean_mbl_loss": np.mean(all_scene_metrics['total_mbl_losses']),
                "mean_pbl_loss": np.mean(all_scene_metrics['total_pbl_losses'])
            },
            "individual_scene_results": all_scene_results,
            "evaluation_timestamp": str(np.datetime64('now'))
        }
        
        # 保存到文件
        output_file = OUTPUT_FILE
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n批量评估结果已保存到: {output_file}")
        
        print("\n=== 评估完成 ===")
        print("主要指标汇总 (均值 ± 标准差):")
        print(f"total_oob_loss: {mean_total_oob_loss_scaled:.2f} ± {std_total_oob_loss_scaled:.2f}")
        print(f"total_mbl_loss: {mean_total_mbl_loss_scaled:.2f} ± {std_total_mbl_loss_scaled:.2f}")
        print(f"total_pbl_loss: {mean_total_pbl_loss_scaled:.2f} ± {std_total_pbl_loss_scaled:.2f}")
        print(f"valid_scene_ratio: {valid_scene_ratio:.2f}")