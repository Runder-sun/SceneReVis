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


# --- 1. Data Parsing and Preparation ---

def get_object_field(obj, field_name, format_type='ours'):
    """
    Get field value from an object, handling different format field names.
    
    Args:
        obj: Object dictionary
        field_name: Field name ('jid', 'desc', 'size')
        format_type: 'ours' or 'respace'
    
    Returns:
        Field value
    """
    if format_type == 'respace':
        # respace format uses sampled_asset_ prefix
        if field_name == 'jid':
            return obj.get('sampled_asset_jid', obj.get('jid', 'N/A'))
        elif field_name == 'desc':
            return obj.get('sampled_asset_desc', obj.get('desc', 'N/A'))
        elif field_name == 'size':
            return obj.get('sampled_asset_size', obj.get('size', [1, 1, 1]))
    else:
        # ours format uses field names directly
        return obj.get(field_name, 'N/A' if field_name in ['jid', 'desc'] else [1, 1, 1])


# ============================================================================
# Common Functions: Room Polygon and Mesh Creation (supports irregular rooms)
# ============================================================================

def create_floor_polygon(bounds_bottom):
    """
    Create a floor polygon from room bottom boundaries (supports irregular rooms like L-shaped, T-shaped, etc.)
    
    Args:
        bounds_bottom: List of bottom boundary vertices [[x1,y1,z1], [x2,y2,z2], ...]
    
    Returns:
        shapely.geometry.Polygon: Floor polygon (using X and Z coordinates)
    """
    # Create 2D polygon using X and Z coordinates (Y is the height direction)
    points = [(pt[0], pt[2]) for pt in bounds_bottom]
    return Polygon(points)


def create_room_mesh(bounds_bottom, bounds_top):
    """
    Create a real room mesh from polygon boundaries (supports irregular rooms).
    Uses triangulation to create bottom, top, and side faces.
    
    Args:
        bounds_bottom: List of bottom boundary vertices
        bounds_top: List of top boundary vertices
    
    Returns:
        trimesh.Trimesh: Room mesh object
    """
    bounds_bottom = np.array(bounds_bottom)
    bounds_top = np.array(bounds_top)
    
    # Create floor polygon
    floor_polygon = create_floor_polygon(bounds_bottom.tolist())
    
    num_verts = len(bounds_bottom)
    all_vertices = np.concatenate([bounds_bottom, bounds_top], axis=0)
    
    # Use trimesh to triangulate the floor polygon
    try:
        vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="triangle")
    except Exception:
        # If triangle engine fails, try using earcut
        try:
            vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="earcut")
        except Exception:
            # Final fallback: use simple fan triangulation
            floor_faces = np.array([[0, i, i+1] for i in range(1, num_verts-1)])
    
    # Remove invalid faces (indices out of range)
    valid_mask = np.all(floor_faces < num_verts, axis=1)
    floor_faces = floor_faces[valid_mask]
    
    # Create ceiling faces (offset indices to top vertices)
    ceiling_faces = floor_faces + num_verts
    # Flip ceiling face winding to ensure normals point inward
    ceiling_faces = ceiling_faces[:, ::-1]
    
    # Create side faces
    side_faces = []
    for i in range(num_verts):
        next_i = (i + 1) % num_verts
        # Two triangles form one quadrilateral side face
        side_faces.append([i, next_i, i + num_verts])
        side_faces.append([next_i, next_i + num_verts, i + num_verts])
    side_faces = np.array(side_faces)
    
    # Merge all faces
    all_faces = np.concatenate([floor_faces, ceiling_faces, side_faces], axis=0)
    
    # Create mesh and fix normals
    room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
    trimesh.repair.fix_normals(room_mesh)
    
    return room_mesh


def check_object_out_of_bounds(obj_mesh, room_mesh, floor_polygon, room_height_min, room_height_max, num_samples=500):
    """
    Detect whether an object is out of bounds using mesh containment (supports irregular rooms and height detection).
    
    Args:
        obj_mesh: Object's trimesh object
        room_mesh: Room's trimesh object
        floor_polygon: Room floor's shapely polygon
        room_height_min: Minimum room height (floor Y coordinate)
        room_height_max: Maximum room height (ceiling Y coordinate)
        num_samples: Number of sample points
    
    Returns:
        (is_oob: bool, oob_volume: float)
    """
    try:
        # Sample points on the object mesh
        sample_points = obj_mesh.sample(num_samples)
        
        # Detect 2D out-of-bounds (XZ plane)
        # Note: shapely's contains() returns False for points on the boundary, which causes
        # wall-adjacent furniture to be incorrectly flagged as out of bounds.
        # Solution: Add a tiny buffer (1mm) to the polygon so boundary points are correctly identified as inside.
        from shapely.geometry import Point
        buffered_polygon = floor_polygon.buffer(0.001)  # 1mm buffer
        points_2d = [Point(pt[0], pt[2]) for pt in sample_points]
        inside_2d = np.array([buffered_polygon.contains(p) for p in points_2d])
        
        # Detect height out-of-bounds (Y direction) - also add a tiny tolerance
        height_tolerance = 0.001  # 1mm tolerance
        inside_height = (sample_points[:, 1] >= room_height_min - height_tolerance) & (sample_points[:, 1] <= room_height_max + height_tolerance)
        
        # Combined check: must be within both 2D range and height range
        inside = inside_2d & inside_height
        
        oob_ratio = 1.0 - (inside.sum() / len(inside))
        
        if oob_ratio > 0.01:  # More than 1% of points are outside
            is_oob = True
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            oob_volume = obj_volume * oob_ratio
        else:
            is_oob = False
            oob_volume = 0.0
        
        return is_oob, oob_volume
        
    except Exception as e:
        print(f"Warning: Out-of-bounds detection sampling failed: {e}, using bounding box fallback")
        # Fallback: use simple bounding box detection
        obj_bounds = obj_mesh.bounds
        room_bounds = room_mesh.bounds
        
        if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
            obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
            return True, obj_volume * 0.1
        return False, 0.0


def parse_scene_data(scene_json, format_type='ours'):
    """
    Parse scene JSON data, supporting two formats.
    
    Args:
        scene_json: Scene JSON data
        format_type: 'ours' or 'respace'
    
    Returns:
        bounds_bottom, bounds_top, all_objects_data
    """
    if format_type == 'respace':
        # respace format: objects directly at root level, bounds directly at root level
        bounds_bottom = scene_json.get('bounds_bottom', [])
        bounds_top = scene_json.get('bounds_top', [])
        all_objects_data = scene_json.get('objects', [])
    else:
        # ours format: uses room_envelope and groups structure
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
    Parse scene from JSON data, loading and transforming real 3D models.
    Supports two JSON formats: 'ours' and 'respace'
    """
    # Use the new parse function to get scene data
    bounds_bottom, bounds_top, all_objects_data = parse_scene_data(scene_json, format_type)
    
    # Use all objects directly without deduplication
    unique_objects_data = all_objects_data
    
    print(f"Object count: {len(all_objects_data)}")

    # Create room trimesh object (using polygon extrusion to support irregular rooms)
    room_mesh = create_room_mesh(bounds_bottom, bounds_top)
    
    # Create floor polygon for 2D out-of-bounds detection
    floor_polygon = create_floor_polygon(bounds_bottom.tolist() if isinstance(bounds_bottom, np.ndarray) else bounds_bottom)
    
    # Calculate room height range
    bounds_bottom_arr = np.array(bounds_bottom)
    bounds_top_arr = np.array(bounds_top)
    room_height_min = bounds_bottom_arr[:, 1].min()
    room_height_max = bounds_top_arr[:, 1].max()

    # Load and transform furniture trimesh objects
    furniture_objects = {}
    for i, obj_data in enumerate(unique_objects_data):
        # Determine asset source
        asset_source = obj_data.get('asset_source', '3d-future')
        target_size = get_object_field(obj_data, 'size', format_type)
        
        model_path = None
        asset_id = None
        
        if asset_source == 'objaverse':
            # Objaverse asset: use uid and GLB cache
            uid = obj_data.get('uid')
            if uid:
                asset_id = uid
                model_path = find_objaverse_glb(uid)
                if model_path:
                    model_path = str(model_path)
        else:
            # 3D-FUTURE asset: use jid
            jid = get_object_field(obj_data, 'jid', format_type)
            asset_id = jid
            model_path = os.path.join(models_base_path, jid, 'raw_model.glb')
        
        if asset_id is None:
            asset_id = f'object_{i}'
        
        if model_path is None or not os.path.exists(model_path):
            print(f"Warning: Model file not found for {asset_id} (source: {asset_source}). Using cuboid as substitute.")
            mesh = trimesh.creation.box(extents=target_size)
        else:
            loaded = trimesh.load(model_path)
            # Handle the case where trimesh.load returns a Scene object
            if isinstance(loaded, trimesh.Scene):
                # Merge all geometries in the scene into a single mesh
                if len(loaded.geometry) > 0:
                    mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                else:
                    print(f"Warning: Model file {asset_id} is an empty scene. Using cuboid as substitute.")
                    mesh = trimesh.creation.box(extents=target_size)
            else:
                mesh = loaded

        # --- Apply Transforms: 1.Scale -> 2.Rotate -> 3.Translate ---
        
        # 1. Scale
        # Calculate scale factors to match the size defined in JSON
        original_size = mesh.extents
        # Prevent division by zero
        target_size_array = np.array(target_size)
        scale_factors = target_size_array / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        # 2. Rotate & 3. Translate
        pos = obj_data['pos']
        rot_xyzw = obj_data['rot']
        
        try:
            rotation = R.from_quat(rot_xyzw)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation.as_matrix()
            transform_matrix[:3, 3] = pos
            
            # --- START: Core Modification Area ---
            # Previously used 'mesh.bounds.mean(axis=0)', which positions the object from its geometric center, causing OOB errors.
            # Now we calculate the object's "bottom center" as the anchor point.
            bounds = mesh.bounds
            bottom_center_pivot = np.array([
                (bounds[0, 0] + bounds[1, 0]) / 2,  # X-axis center
                bounds[0, 1],                      # Y-axis bottom (minimum)
                (bounds[0, 2] + bounds[1, 2]) / 2   # Z-axis center
            ])

            # Create a transform that moves the object's "bottom center" to the coordinate origin
            center_transform = np.eye(4)
            center_transform[:3, 3] = -bottom_center_pivot
            # --- END: Core Modification Area ---
            
            # Full transform: first move anchor to origin, then apply rotation and translation to final position
            mesh.apply_transform(center_transform)
            mesh.apply_transform(transform_matrix)
            
            # Use first 8 characters of asset_id as identifier
            furniture_objects[f"object_{i}_{asset_id[:8]}"] = mesh
        except Exception as e:
            desc = get_object_field(obj_data, 'desc', format_type)
            print(f"Error processing object: {desc}, error: {e}")

    # Return room mesh, furniture objects, raw data, floor polygon, and height range
    return room_mesh, furniture_objects, unique_objects_data, floor_polygon, room_height_min, room_height_max


# --- 2. Physical Validity Metrics ---

def calculate_physics_metrics(room_mesh, objects, floor_polygon=None, room_height_min=None, room_height_max=None):
    """
    Calculate collision and out-of-bounds metrics.
    This version has removed collision tolerance and supports precise out-of-bounds detection for irregular rooms.
    
    Args:
        room_mesh: Room's trimesh object
        objects: Furniture object dictionary {name: mesh}
        floor_polygon: Room floor's shapely polygon (for irregular room detection)
        room_height_min: Minimum room height
        room_height_max: Maximum room height
    """
    # Get total object count
    total_objects = len(objects)
    
    manager = trimesh.collision.CollisionManager()
    
    # Only add furniture objects to the collision manager
    for name, mesh in objects.items():
        manager.add_object(name, mesh)

    # Get all potential contact data
    is_collision, contact_data = manager.in_collision_internal(return_data=True)
    
    # All penetration depths > 0.01 are considered collisions (1cm tolerance added)
    collision_tolerance = 0.01
    actual_collisions_data = [d for d in contact_data if d.depth > collision_tolerance]
    
    num_colliding_pairs = len(actual_collisions_data)
    
    # Calculate number of objects involved in collisions (new metric 1: collision rate)
    colliding_objects = set()
    for contact in actual_collisions_data:
        # contact contains names of the two colliding objects
        if hasattr(contact, 'names'):
            # contact.names may be a set or other collection type
            if isinstance(contact.names, (list, tuple)):
                colliding_objects.add(contact.names[0])
                colliding_objects.add(contact.names[1])
            else:
                # If it's a set or other iterable, add all elements directly
                colliding_objects.update(contact.names)
    
    num_colliding_objects = len(colliding_objects)
    collision_rate = (num_colliding_objects / total_objects * 100) if total_objects > 0 else 0.0
    
    total_penetration_depth = 0
    if num_colliding_pairs > 0:
        for contact in actual_collisions_data:
            # Accumulate full penetration depth (removed tolerance subtraction)
            total_penetration_depth += contact.depth
            
    mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0

    # --- Out-of-Bounds Detection (supports irregular rooms) ---
    num_oob_objects = 0
    total_oob_volume = 0
    
    # If parameters needed for irregular room detection are not provided, use bounding box fallback
    use_polygon_detection = (floor_polygon is not None and 
                             room_height_min is not None and 
                             room_height_max is not None)
    
    for name, mesh in objects.items():
        if use_polygon_detection:
            # Use precise polygon + height detection (supports irregular rooms)
            is_oob, oob_volume = check_object_out_of_bounds(
                mesh, room_mesh, floor_polygon, room_height_min, room_height_max
            )
            if is_oob:
                num_oob_objects += 1
                total_oob_volume += oob_volume
        else:
            # Fallback: use bounding box detection
            obj_bounds = mesh.bounds
            room_bounds = room_mesh.bounds
            
            if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
                num_oob_objects += 1
                
                # Calculate out-of-bounds volume
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
                    print(f"Warning: Out-of-bounds volume calculation failed for {name}: {e}")
                    oob_volume = np.prod(mesh.extents) * 0.1
                
                total_oob_volume += oob_volume

    mean_oob_volume = (total_oob_volume / num_oob_objects) if num_oob_objects > 0 else 0
    
    # Calculate out-of-bounds rate (new metric 2: OOB rate)
    out_of_bounds_rate = (num_oob_objects / total_objects * 100) if total_objects > 0 else 0.0

    metrics = {
        "Object Count": total_objects,
        "Collision-Free Rate (%)": 100.0 if num_colliding_pairs == 0 else 0.0,
        "Number of Colliding Pairs": num_colliding_pairs,
        "Collision Rate (%)": collision_rate,  # New: collision rate
        "Mean Penetration Depth (m)": mean_penetration_depth,
        "Valid Placement Rate (%)": 100.0 if num_oob_objects == 0 else 0.0,
        "Number of Out-of-Bounds Objects": num_oob_objects,
        "Out-of-Bounds Rate (%)": out_of_bounds_rate,  # New: OOB rate
        "Mean Out-of-Bounds Volume (m^3)": mean_oob_volume
    }
    return metrics
    

# --- Functional Metrics Disabled ---
# Sofa accessibility evaluation has been removed


# --- 5. Main Program ---

def evaluate_single_scene(scene_json, models_base_path, create_virtual_models=True, format_type='ours'):
    """
    Evaluate metrics for a single scene.
    Supports two JSON formats: 'ours' and 'respace'
    """
    # Create virtual model files for JIDs mentioned in JSON (if needed)
    if create_virtual_models:
        all_jids = set()
        # Use parse_scene_data to get all objects
        _, _, all_objects_data = parse_scene_data(scene_json, format_type)
        for obj in all_objects_data:
            jid = get_object_field(obj, 'jid', format_type)
            all_jids.add(jid)
        
        for jid in all_jids:
            jid_dir = os.path.join(models_base_path, jid)
            os.makedirs(jid_dir, exist_ok=True)
            model_path = os.path.join(jid_dir, 'raw_model.glb')
            if not os.path.exists(model_path):
                # Create a simple cuboid as a placeholder
                try:
                    trimesh.creation.box().export(model_path)
                except Exception as e:
                    print(f"Warning: Unable to create virtual model {model_path}: {e}")

    try:
        # Parse scene (now returns additional irregular room detection parameters)
        room_mesh, objects, unique_objects_data, floor_polygon, room_height_min, room_height_max = parse_scene_from_json(scene_json, models_base_path, format_type)
        
        # Only calculate physics metrics (functional metrics disabled)
        physics_metrics = calculate_physics_metrics(room_mesh, objects, floor_polygon, room_height_min, room_height_max)
        
        # Return physics metrics
        all_metrics = physics_metrics
        return all_metrics, True
        
    except Exception as e:
        print(f"Error evaluating scene: {e}")
        return {}, False


def evaluate_single_scene_file(json_file: str, models_base_path: str, format_type: str) -> dict:
    """
    Evaluate a single scene file (for multiprocess invocation).
    
    Args:
        json_file: JSON file path
        models_base_path: Base path for 3D model files
        format_type: Scene JSON format type
    
    Returns:
        Dictionary containing evaluation results
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
                'error': 'Evaluation failed'
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
    Batch evaluate all JSON scene files in a directory (supports multiprocessing).
    
    Args:
        scenes_directory: Directory path containing JSON files
        models_base_path: Base path for 3D model files
        max_scenes: Maximum number of scenes to process, None means process all
        format_type: Scene JSON format type, 'ours' or 'respace'
        num_workers: Number of parallel processes, None means half of CPU cores
        output_dir: Output directory path, None means use scenes_directory
    """
    print(f"Starting batch scene evaluation...")
    print(f"Scene directory: {scenes_directory}")
    print(f"Model path: {models_base_path}")
    
    # Get all JSON files
    json_files = []
    if os.path.isdir(scenes_directory):
        for file in os.listdir(scenes_directory):
            if file.endswith('.json'):
                json_files.append(os.path.join(scenes_directory, file))
    else:
        print(f"Error: Directory does not exist: {scenes_directory}")
        return
    
    json_files.sort()  # Sort by filename
    
    if max_scenes and max_scenes > 0:
        json_files = json_files[:max_scenes]
    
    print(f"Found {len(json_files)} JSON files")
    
    if len(json_files) == 0:
        print("No JSON files found!")
        return
    
    # Set number of processes
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 2)
    num_workers = min(num_workers, len(json_files))
    
    print(f"Using {num_workers} processes for parallel processing")
    
    # Store metrics for all scenes
    all_scene_metrics = []
    successful_evaluations = 0
    failed_evaluations = 0
    
    print("\nStarting scene processing...")
    print("=" * 80)
    
    # Use multiprocessing for parallel processing
    eval_func = partial(evaluate_single_scene_file, models_base_path=models_base_path, format_type=format_type)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(eval_func, json_file): json_file for json_file in json_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating scenes"):
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
                print(f"\n  ✗ Error processing file {os.path.basename(json_file)}: {e}")
    
    print("\n" + "=" * 80)
    print(f"Processing complete! Success: {successful_evaluations}, Failed: {failed_evaluations}")
    
    if successful_evaluations == 0:
        print("No successfully evaluated scenes!")
        return
    
    # Calculate aggregate statistics
    print("\n--- Aggregate Evaluation Report ---")
    
    # Collect values for all metrics
    metric_names = list(all_scene_metrics[0]['metrics'].keys())
    metric_values = {name: [] for name in metric_names}
    
    for scene_data in all_scene_metrics:
        for name, value in scene_data['metrics'].items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metric_values[name].append(value)
    
    # Calculate statistics
    print(f"\nBased on {successful_evaluations} successfully evaluated scenes:")
    print("\n[ Physical Validity Metrics Statistics ]")
    physics_metrics_names = [
        "Object Count",
        "Collision-Free Rate (%)",
        "Number of Colliding Pairs",
        "Collision Rate (%)",  # New: collision rate
        "Mean Penetration Depth (m)",
        "Valid Placement Rate (%)",
        "Number of Out-of-Bounds Objects",
        "Out-of-Bounds Rate (%)",  # New: OOB rate
        "Mean Out-of-Bounds Volume (m^3)"
    ]
    
    for name in physics_metrics_names:
        if name in metric_values and len(metric_values[name]) > 0:
            values = metric_values[name]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            print(f"{name:<35}: Mean={mean_val:.4f} | Std={std_val:.4f} | Range=[{min_val:.4f}, {max_val:.4f}]")
    
    # Functional metrics disabled
    
    # Save detailed results to file
    # Use custom output directory; if not specified, use scenes directory
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
        
        # Add aggregate statistics
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
        
        print(f"\nDetailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results file: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch evaluate physics and functional metrics of 3D scenes - supports multiple data formats')
    parser.add_argument('--format', type=str, default='respace', choices=['ours', 'respace'],
                       help='Scene JSON format type: ours (groups structure) or respace (direct objects structure)')
    parser.add_argument('--scenes_dir', type=str, 
                       default='/path/to/SceneReVis/output/sft_65k/final_scenes_collection', 
                       help='Directory path containing JSON scene files')
    parser.add_argument('--models_path', type=str, 
                       default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                       help='Base path for 3D model files')
    parser.add_argument('--max_scenes', type=int, default=None,
                       help='Maximum number of scenes to process (default: process all)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel processes (default: half of CPU cores)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory path (default: same as scenes_dir)')
    
    args = parser.parse_args()
    
    # Ensure model path exists
    os.makedirs(args.models_path, exist_ok=True)
    
    print("=== 3D Scene Evaluation Tool ===")
    print(f"Scene format: {args.format}")
    print(f"Scene directory: {args.scenes_dir}")
    print(f"Model path: {args.models_path}")
    if args.max_scenes:
        print(f"Max scenes: {args.max_scenes}")
    if args.workers:
        print(f"Parallel processes: {args.workers}")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print()
    
    # Execute batch evaluation
    batch_evaluate_scenes(args.scenes_dir, args.models_path, args.max_scenes, args.format, args.workers, args.output_dir)