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


# --- 0. Objaverse GLB Lookup Helper Functions ---

def find_objaverse_glb(uid: str):
    """
    Find the Objaverse GLB file path.
    
    Search order (consistent with blender_renderer.py):
    1. Path specified by OBJAVERSE_GLB_CACHE_DIR environment variable
    2. /path/to/data/datasets/objathor-assets/glbs (local development)
    3. /path/to/datasets/objathor-assets/glbs (cloud storage)
    4. ~/.objaverse/hf-objaverse-v1/glbs (local cache)
    
    Uses uid[:2] to directly construct subdirectory path (optimization: avoids traversing all subdirectories)
    
    Args:
        uid: Objaverse asset UID
        
    Returns:
        GLB file path, or None if not found
    """
    if not uid or len(uid) < 2:
        return None
    
    # 0. Special check for LayoutVLM processed objaverse (structure: {uid}/{uid}.glb)
    layout_vlm_processed = Path("/path/to/SceneReVis/baseline/LayoutVLM/objaverse_processed")
    if layout_vlm_processed.is_dir():
        candidate = layout_vlm_processed / uid / f"{uid}.glb"
        if candidate.is_file():
            return candidate

    # GLB cache directory list (sorted by priority, consistent with blender_renderer.py)
    cache_dirs = []
    
    # 1. Path specified by environment variable
    env_cache = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
    if env_cache:
        cache_dirs.append(Path(env_cache) / "glbs")
    
    # 2. Local development path
    cache_dirs.append(Path("/path/to/data/datasets/objathor-assets/glbs"))
    
    # 3. Cloud storage
    cache_dirs.append(Path("/path/to/datasets/objathor-assets/glbs"))
    
    # 4. Local cache
    cache_dirs.append(Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs")
    
    # Use uid[:2] to directly construct subdirectory path (optimization)
    subdir_name = uid[:2]
    
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        
        # Directly look for GLB file under uid[:2] subdirectory
        candidate = cache_dir / subdir_name / f"{uid}.glb"
        if candidate.is_file():
            return candidate
    
    return None


# --- 1. Voxel Evaluation Method (learned from respace project) ---

def get_y_angle_from_xyzw_quaternion(quaternion_xyzw):
    """Extract Y-axis rotation angle from a quaternion"""
    # Check if it's a unit quaternion (identity rotation)
    if np.allclose(quaternion_xyzw, [0, 0, 0, 1]):
        return 0.0, 0.0
    
    rotation = R.from_quat(quaternion_xyzw)
    # Use 'yxz' order to avoid gimbal lock warning
    try:
        euler = rotation.as_euler('yxz', degrees=False)
        return euler[0], euler[0]  # Return Y-axis rotation in radians
    except:
        # If still problematic, compute Y angle directly from rotation matrix
        matrix = rotation.as_matrix()
        y_angle = np.arctan2(matrix[0, 2], matrix[2, 2])
        return y_angle, y_angle

def get_xz_bbox_from_obj(obj):
    """Get the 2D bounding box of an object on the XZ plane"""
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
    
    # Get 3D bounding box height information
    obj_height = bbox_size[1]
    obj_y_start = bbox_position[1]
    obj_y_end = bbox_position[1] + obj_height
    
    return bbox_2d_obj, obj_height, obj_y_start, obj_y_end

def create_floor_plan_polygon(bounds_bottom):
    """Create floor polygon from room bottom boundaries"""
    points = [(pt[0], pt[2]) for pt in bounds_bottom]  # Use X and Z coordinates
    return Polygon(points)

def create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon):
    """Create room mesh"""
    num_verts = len(bounds_bottom)
    all_vertices = np.array(bounds_bottom + bounds_top)
    
    # Use trimesh to triangulate the floor
    vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
    # Remove invalid faces
    idxs = []
    for i, row in enumerate(floor_faces):
        if np.any(row == num_verts):
            idxs.append(i)
    floor_faces = np.delete(floor_faces, idxs, axis=0)
    
    floor_mesh = trimesh.Trimesh(vertices=vtx, faces=floor_faces)
    
    # Create ceiling faces
    ceiling_faces = floor_faces + num_verts
    
    # Create side faces
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
    """Voxelize a mesh and return the voxel matrix"""
    asset_voxels = asset_mesh.voxelized(pitch=voxel_size).fill()
    asset_voxel_matrix = asset_voxels.matrix
    return asset_voxel_matrix

def prepare_asset_voxel(obj, models_base_path, voxel_size):
    """
    Prepare voxel representation of an object - fixed version.
    Key fixes:
    1. First center the mesh to bottom center, then apply rotation
    2. Use actual bounds of the voxel grid to calculate correct offsets
    
    Supports two asset sources:
    - 3D-FUTURE: Uses 'jid' field, loads from models_base_path/{jid}/raw_model.glb
    - Objaverse: Uses 'uid' field, loads from GLB cache
    """
    # Determine asset source and model path
    asset_source = obj.get('asset_source', '3d-future')
    model_path = None
    asset_id = None
    
    if asset_source == 'objaverse':
        # Objaverse asset: use uid and GLB cache
        uid = obj.get('uid')
        if uid:
            asset_id = uid
            model_path = find_objaverse_glb(uid)
            if model_path:
                model_path = str(model_path)
    else:
        # 3D-FUTURE asset: use jid
        jid = obj.get('jid', 'N/A')
        asset_id = jid
        model_path = os.path.join(models_base_path, jid, 'raw_model.glb')
    
    if asset_id is None:
        asset_id = 'unknown'
    
    if model_path is None or not os.path.exists(model_path):
        # Use bounding box to create substitute mesh
        mesh = trimesh.creation.box(extents=obj['size'])
    else:
        try:
            asset_scene = trimesh.load(model_path)
            if isinstance(asset_scene, trimesh.Scene):
                mesh = asset_scene.to_geometry()
            else:
                mesh = asset_scene
        except Exception as e:
            print(f"Failed to load mesh {asset_id}: {e}")
            mesh = trimesh.creation.box(extents=obj['size'])
    
    # 1. Apply scaling to target size
    original_size = mesh.extents
    target_size = np.array(obj['size'])
    scale_factors = target_size / (original_size + 1e-6)
    mesh.apply_scale(scale_factors)
    
    # 2. Key fix: Center mesh to bottom center (consistent with myeval.py)
    # Calculate current mesh's bottom center
    bounds = mesh.bounds
    bottom_center = np.array([
        (bounds[0, 0] + bounds[1, 0]) / 2,  # X-axis center
        bounds[0, 1],                        # Y-axis bottom
        (bounds[0, 2] + bounds[1, 2]) / 2   # Z-axis center
    ])
    # Move bottom center to origin
    center_transform = np.eye(4)
    center_transform[:3, 3] = -bottom_center
    mesh.apply_transform(center_transform)
    
    # 3. Apply rotation (around origin, i.e., bottom center)
    rot_xyzw = obj['rot']
    rotation = R.from_quat(rot_xyzw)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = rotation.as_matrix()
    mesh.apply_transform(rotation_matrix)
    
    # 4. Voxelize the rotated mesh
    try:
        asset_voxels = mesh.voxelized(pitch=voxel_size).fill()
        asset_voxel_matrix = asset_voxels.matrix
        # Get the origin of the voxel grid in world coordinates
        voxel_origin = asset_voxels.translation  # World coordinate corresponding to voxel (0,0,0)
    except Exception as e:
        print(f"Voxelization failed for {obj.get('desc', 'Unknown')}: {e}")
        return None, None, mesh
    
    # 5. Calculate voxel space position offset
    # Object's world position (bottom center)
    pos = np.array(obj['pos'])
    
    # Key fix: Correctly calculate the object voxel position in world coordinates
    # - voxel_origin is the world coordinate of voxel(0,0,0) in local coordinate system (i.e., mesh.bounds[0])
    # - The object's bottom center (local coordinate origin) is moved to world coordinate pos
    # - So the world coordinate of object voxel(0,0,0) = pos + voxel_origin
    # - Index in room voxel grid = (pos + voxel_origin - room_voxel_origin) / voxel_size
    # - In compute_voxel_oob, we calculate: room_origin_shift + asset_shift_from_origin
    # - room_origin_shift = -room_voxel_origin / voxel_size
    # - So asset_shift_from_origin = (pos + voxel_origin) / voxel_size
    asset_shift_from_origin = np.floor((pos + voxel_origin) / voxel_size)
    
    return asset_voxel_matrix, asset_shift_from_origin, mesh

def occupancy_overlap(voxel_matrix_a, voxel_matrix_b, offset_b):
    """Calculate overlap between two voxel matrices"""
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
    """Calculate voxel-based out-of-bounds volume"""
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
    """Calculate voxel-based collision volume"""
    asset_voxel_matrix_x, asset_shift_x, mesh_x = prepare_asset_voxel(obj_x, models_base_path, voxel_size)
    asset_voxel_matrix_y, asset_shift_y, mesh_y = prepare_asset_voxel(obj_y, models_base_path, voxel_size)
    
    if asset_voxel_matrix_x is None or asset_voxel_matrix_y is None:
        return 0.0
    
    offset = np.floor(asset_shift_y - asset_shift_x).astype(int)
    inside_voxels = occupancy_overlap(asset_voxel_matrix_x, asset_voxel_matrix_y, offset)
    
    num_inside_voxels = np.sum(inside_voxels)
    intersection_volume = num_inside_voxels * voxel_volume
    
    return intersection_volume

# --- 2. Traditional Geometric Evaluation Methods (for comparison) ---

def parse_scene_data(scene_json, format_type='ours'):
    """
    Parse scene JSON data, supporting two formats.
    
    Args:
        scene_json: Scene JSON data
        format_type: 'ours' or 'respace'
    
    Returns:
        bounds_bottom, bounds_top, all_objects
    """
    if format_type == 'respace':
        # respace format: objects directly at root level
        bounds_bottom = scene_json.get('bounds_bottom', [])
        bounds_top = scene_json.get('bounds_top', [])
        all_objects = scene_json.get('objects', [])
    else:
        # ours format: uses room_envelope and groups structure
        if 'room_envelope' not in scene_json:
            raise ValueError(f"Scene JSON is missing 'room_envelope' field, please verify the data format is 'ours' format")
        
        bounds_bottom = scene_json['room_envelope']['bounds_bottom']
        bounds_top = scene_json['room_envelope']['bounds_top']
        all_objects = []
        if 'groups' in scene_json and scene_json['groups'] is not None:
            for group in scene_json['groups']:
                if 'objects' in group and group['objects'] is not None:
                    all_objects.extend(group['objects'])
    
    return bounds_bottom, bounds_top, all_objects

def get_intersection_area(obj_x, obj_y, epsilon=1e-7):
    """Calculate the intersection area of two objects on the 2D plane"""
    intersection = obj_x.intersection(obj_y)
    if intersection.is_empty:
        return 0.0
    area = intersection.area
    if area < epsilon:
        return 0.0
    return area

def compute_geometric_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top, epsilon=1e-7):
    """Calculate geometry-based out-of-bounds volume"""
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
    """Calculate geometry-based collision volume"""
    bbox_x, height_x, y_start_x, y_end_x = get_xz_bbox_from_obj(obj_x)
    bbox_y, height_y, y_start_y, y_end_y = get_xz_bbox_from_obj(obj_y)
    
    # 2D intersection area
    intersection_area = get_intersection_area(bbox_x, bbox_y)
    
    if intersection_area < epsilon:
        return 0.0
    
    # Y-axis overlap height
    y_overlap_start = max(y_start_x, y_start_y)
    y_overlap_end = min(y_end_x, y_end_y)
    
    if y_overlap_end <= y_overlap_start:
        return 0.0
    
    y_overlap_height = y_overlap_end - y_overlap_start
    collision_volume = intersection_area * y_overlap_height
    
    if collision_volume < epsilon:
        return 0.0
    
    return collision_volume

# --- 3. Main Evaluation Functions ---

def evaluate_scene_voxel_based(scene_json, models_base_path, voxel_size=0.05, format_type='ours'):
    """Evaluate scene using voxel method"""
    
    # Extract scene data - supports two formats
    bounds_bottom, bounds_top, all_objects = parse_scene_data(scene_json, format_type)
    floor_plan_polygon = create_floor_plan_polygon(bounds_bottom)
    
    # Create room mesh and voxelize
    room_mesh = create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
    room_voxels = room_mesh.voxelized(pitch=voxel_size).fill()
    room_voxel_matrix = room_voxels.matrix
    
    # Key fix: Get the actual origin of the room voxel grid
    # room_voxels.translation is the world coordinate of voxel(0,0,0)
    room_voxel_origin = room_voxels.translation
    
    # room_origin_shift is used to convert world coordinates to voxel indices
    # For world coordinate world_pos, the corresponding voxel index is:
    # voxel_idx = (world_pos - room_voxel_origin) / voxel_size
    # But in compute_voxel_oob, we calculate:
    # asset_offset = room_origin_shift + asset_shift_from_origin
    # where asset_shift_from_origin = (pos - asset_voxel_origin) / voxel_size
    # So room_origin_shift = -room_voxel_origin / voxel_size
    room_origin_shift = -room_voxel_origin / voxel_size
    
    voxel_volume = voxel_size ** 3
    
    # Calculate metrics
    voxel_oobs = []
    voxel_collisions = []
    geometric_oobs = []
    geometric_collisions = []
    
    for i, obj in enumerate(all_objects):
        # Voxel out-of-bounds
        voxel_oob = compute_voxel_oob(obj, voxel_size, room_origin_shift, room_voxel_matrix, voxel_volume, models_base_path)
        voxel_oobs.append(voxel_oob)
        
        # Geometric out-of-bounds (for comparison)
        geometric_oob = compute_geometric_oob(obj, floor_plan_polygon, bounds_bottom, bounds_top)
        geometric_oobs.append(geometric_oob)
        
        # Collisions with other objects
        for j, other_obj in enumerate(all_objects[i+1:], i+1):
            # Voxel collision
            voxel_collision = compute_voxel_collision(obj, other_obj, voxel_size, voxel_volume, models_base_path)
            voxel_collisions.append(voxel_collision)
            
            # Geometric collision (for comparison)
            geometric_collision = compute_geometric_collision(obj, other_obj)
            geometric_collisions.append(geometric_collision)
    
    # Summary metrics
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

# --- 4. Batch Evaluation Functions ---

def load_scene_from_file(json_file_path):
    """Load scene JSON data from file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            scene_json = json.load(f)
        return scene_json
    except Exception as e:
        print(f"Failed to load scene file {json_file_path}: {e}")
        return None

def evaluate_single_scene_file(json_file_path, models_base_path, voxel_size=0.05, format_type='ours'):
    """Evaluate a single scene file"""
    scene_json = load_scene_from_file(json_file_path)
    if scene_json is None:
        return None
    
    try:
        metrics = evaluate_scene_voxel_based(scene_json, models_base_path, voxel_size, format_type)
        return metrics
    except Exception as e:
        print(f"Failed to evaluate scene {json_file_path}: {e}")
        return None

def evaluate_scene_file_wrapper(json_file_path: str, models_base_path: str, voxel_size: float, format_type: str) -> dict:
    """
    Wrapper for evaluating a single scene file (for multiprocess invocation).
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
                'error': 'Evaluation failed'
            }
    except Exception as e:
        return {
            'scene_name': os.path.basename(json_file_path),
            'success': False,
            'error': str(e)
        }


def batch_evaluate_scenes(scenes_directory, models_base_path, voxel_size=0.05, format_type='ours', num_workers=None):
    """Batch evaluate all scene files in a directory (supports multiprocessing)"""
    
    if not os.path.exists(scenes_directory):
        print(f"Error: Scene directory does not exist: {scenes_directory}")
        return None
    
    # Get all JSON files
    json_files = [f for f in os.listdir(scenes_directory) if f.endswith('.json')]
    if not json_files:
        print(f"Warning: No JSON files found in directory {scenes_directory}")
        return None
    
    print(f"Found {len(json_files)} scene files")
    
    # Store evaluation results for all scenes
    all_scene_results = []
    all_scene_metrics = {
        'total_oob_losses': [],
        'total_mbl_losses': [], 
        'total_pbl_losses': [],
        'valid_scenes': [],
        'scene_names': []
    }
    
    # Create required model directories (ensure all jids have corresponding virtual models)
    all_jids = set()
    
    # Preprocessing: collect all jids
    for json_file in json_files:
        json_file_path = os.path.join(scenes_directory, json_file)
        scene_json = load_scene_from_file(json_file_path)
        if scene_json is not None:
            # Extract objects based on format
            try:
                _, _, objects = parse_scene_data(scene_json, format_type)
                for obj in objects:
                    all_jids.add(obj.get('jid', 'N/A'))
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping file {json_file} - {e}")
                continue
    
    # Create virtual model files
    print(f"Creating virtual model files for {len(all_jids)} distinct models...")
    for jid in all_jids:
        if jid != 'N/A':
            jid_dir = os.path.join(models_base_path, jid)
            os.makedirs(jid_dir, exist_ok=True)
            model_path = os.path.join(jid_dir, 'raw_model.glb')
            if not os.path.exists(model_path):
                trimesh.creation.box().export(model_path)
    
    # Set number of processes
    if num_workers is None:
        # Since voxelization is memory-intensive, use fewer processes by default
        num_workers = max(1, min(8, multiprocessing.cpu_count() // 2))
    num_workers = min(num_workers, len(json_files))
    
    print(f"Using {num_workers} processes for parallel evaluation")
    
    # Batch evaluation (multiprocess)
    failed_scenes = []
    json_file_paths = [os.path.join(scenes_directory, f) for f in json_files]
    
    eval_func = partial(evaluate_scene_file_wrapper, 
                        models_base_path=models_base_path, 
                        voxel_size=voxel_size, 
                        format_type=format_type)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_func, path): path for path in json_file_paths}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating scenes"):
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
                print(f"\n   ✗ Error processing file: {e}")
    
    if failed_scenes:
        print(f"\nWarning: {len(failed_scenes)} scenes failed evaluation")
    
    return all_scene_results, all_scene_metrics

# --- 5. Main Program ---

if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description='Scene Voxel Evaluation Tool - supports multiple data formats')
    parser.add_argument('--format', type=str, default='respace', choices=['ours', 'respace'],
                        help='Scene JSON format type: ours (groups structure) or respace (direct objects structure)')
    parser.add_argument('--scenes_dir', type=str, 
                        default='/path/to/SceneReVis/output/sft_65k/final_scenes_collection',
                        help='Directory containing scene JSON files')
    parser.add_argument('--models_path', type=str,
                        default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                        help='Base path for 3D model files')
    parser.add_argument('--output_file', type=str,
                        default='/path/to/SceneReVis/eval/ours_voxel_evaluation_results_new.json',
                        help='Output file path for evaluation results')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='Voxel size (unit: meters)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel processes (default: auto-select, max 4)')
    
    args = parser.parse_args()
    
    # Define paths
    SCENES_DIRECTORY = args.scenes_dir
    MODELS_BASE_PATH = args.models_path
    OUTPUT_FILE = args.output_file
    FORMAT_TYPE = args.format
    
    os.makedirs(MODELS_BASE_PATH, exist_ok=True)
    
    # Run batch evaluation
    print("=== Batch Scene Evaluation ===")
    print(f"Scene format: {FORMAT_TYPE}")
    print(f"Scene directory: {SCENES_DIRECTORY}")
    print(f"Model path: {MODELS_BASE_PATH}")
    print(f"Output file: {OUTPUT_FILE}")
    if args.workers:
        print(f"Parallel processes: {args.workers}")
    
    # Test different voxel resolutions
    voxel_sizes = [args.voxel_size]  # Use voxel size specified via command line
    
    for voxel_size in voxel_sizes:
        print(f"\n=== Batch evaluation with voxel size: {voxel_size}m ===")
        
        # Batch evaluate all scenes - using specified format
        all_scene_results, all_scene_metrics = batch_evaluate_scenes(
            SCENES_DIRECTORY, MODELS_BASE_PATH, voxel_size, FORMAT_TYPE, args.workers
        )
        
        if all_scene_results is None:
            print("Batch evaluation failed")
            continue
        
        # Calculate summary statistics
        total_scenes = len(all_scene_results)
        successful_scenes = len(all_scene_metrics['total_oob_losses'])
        
        if successful_scenes == 0:
            print("No scenes were successfully evaluated")
            continue
        
        # Calculate mean and standard deviation
        mean_total_oob_loss = np.mean(all_scene_metrics['total_oob_losses'])
        mean_total_mbl_loss = np.mean(all_scene_metrics['total_mbl_losses'])
        mean_total_pbl_loss = np.mean(all_scene_metrics['total_pbl_losses'])
        
        std_total_oob_loss = np.std(all_scene_metrics['total_oob_losses'])
        std_total_mbl_loss = np.std(all_scene_metrics['total_mbl_losses'])
        std_total_pbl_loss = np.std(all_scene_metrics['total_pbl_losses'])
        
        # # Scale to the same units as respace (x 1000)
        mean_total_oob_loss_scaled = mean_total_oob_loss * 1e3
        mean_total_mbl_loss_scaled = mean_total_mbl_loss * 1e3
        mean_total_pbl_loss_scaled = mean_total_pbl_loss * 1e3
        
        std_total_oob_loss_scaled = std_total_oob_loss * 1e3
        std_total_mbl_loss_scaled = std_total_mbl_loss * 1e3
        std_total_pbl_loss_scaled = std_total_pbl_loss * 1e3
        
        # Calculate valid scene statistics
        valid_scenes_count = sum(all_scene_metrics['valid_scenes'])
        valid_scene_ratio = valid_scenes_count / successful_scenes
        
        # Output RESPACE-style summary results
        print(f"\n=== Batch Evaluation Summary ({successful_scenes}/{total_scenes} scenes) ===")
        print(f"total_oob_loss: {mean_total_oob_loss_scaled:.2f} (+/- {std_total_oob_loss_scaled:.2f}) (x 0.001)")
        print(f"total_mbl_loss: {mean_total_mbl_loss_scaled:.2f} (+/- {std_total_mbl_loss_scaled:.2f}) (x 0.001)")
        print(f"total_pbl_loss: {mean_total_pbl_loss_scaled:.2f} (+/- {std_total_pbl_loss_scaled:.2f}) (x 0.001)")
        print("")
        print(f"valid_scene_ratio_pbl: {valid_scene_ratio:.2f} (+/- 0.00)")
        print(f"valid_scenes_count: {valid_scenes_count}/{successful_scenes}")
        print("")
        
        # Detailed statistics
        print("=== Detailed Statistics ===")
        print(f"Total scenes: {total_scenes}")
        print(f"Successfully evaluated: {successful_scenes}")
        print(f"Failed evaluations: {total_scenes - successful_scenes}")
        print(f"Valid scenes: {valid_scenes_count}")
        print(f"Invalid scenes: {successful_scenes - valid_scenes_count}")
        print("")
        
        print("Average loss metrics (mean ± std):")
        print(f"  OOB loss: {mean_total_oob_loss:.6f} ± {std_total_oob_loss:.6f}")
        print(f"  MBL loss: {mean_total_mbl_loss:.6f} ± {std_total_mbl_loss:.6f}")
        print(f"  PBL loss: {mean_total_pbl_loss:.6f} ± {std_total_pbl_loss:.6f}")
        print("")
        
        # Individual scene details
        print("=== Per-Scene Evaluation Details ===")
        for i, result in enumerate(all_scene_results):
            scene_name = result['scene_name']
            oob_loss = result['total_oob_loss']
            mbl_loss = result['total_mbl_loss']
            pbl_loss = result['total_pbl_loss']
            is_valid = result['is_valid_scene']
            
            status_icon = "✅" if is_valid else "❌"
            print(f"{i+1:2d}. {status_icon} {scene_name}")
            print(f"     OOB: {oob_loss:.6f} | MBL: {mbl_loss:.6f} | PBL: {pbl_loss:.6f}")
        
        # Save batch evaluation results
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
                
                # Mean metrics (original units)
                "mean_total_oob_loss": mean_total_oob_loss,
                "mean_total_mbl_loss": mean_total_mbl_loss,
                "mean_total_pbl_loss": mean_total_pbl_loss,
                
                # Standard deviation metrics (original units)
                "std_total_oob_loss": std_total_oob_loss,
                "std_total_mbl_loss": std_total_mbl_loss,
                "std_total_pbl_loss": std_total_pbl_loss,
                
                # Mean metrics (scaled units x1000)
                "mean_total_oob_loss_scaled": mean_total_oob_loss_scaled,
                "mean_total_mbl_loss_scaled": mean_total_mbl_loss_scaled,
                "mean_total_pbl_loss_scaled": mean_total_pbl_loss_scaled,
                
                # Standard deviation metrics (scaled units x1000)
                "std_total_oob_loss_scaled": std_total_oob_loss_scaled,
                "std_total_mbl_loss_scaled": std_total_mbl_loss_scaled,
                "std_total_pbl_loss_scaled": std_total_pbl_loss_scaled,
                
                # Average losses
                "mean_oob_loss": np.mean(all_scene_metrics['total_oob_losses']),
                "mean_mbl_loss": np.mean(all_scene_metrics['total_mbl_losses']),
                "mean_pbl_loss": np.mean(all_scene_metrics['total_pbl_losses'])
            },
            "individual_scene_results": all_scene_results,
            "evaluation_timestamp": str(np.datetime64('now'))
        }
        
        # Save to file
        output_file = OUTPUT_FILE
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nBatch evaluation results saved to: {output_file}")
        
        print("\n=== Evaluation Complete ===")
        print("Key metrics summary (mean ± std):")
        print(f"total_oob_loss: {mean_total_oob_loss_scaled:.2f} ± {std_total_oob_loss_scaled:.2f}")
        print(f"total_mbl_loss: {mean_total_mbl_loss_scaled:.2f} ± {std_total_mbl_loss_scaled:.2f}")
        print(f"total_pbl_loss: {mean_total_pbl_loss_scaled:.2f} ± {std_total_pbl_loss_scaled:.2f}")
        print(f"valid_scene_ratio: {valid_scene_ratio:.2f}")