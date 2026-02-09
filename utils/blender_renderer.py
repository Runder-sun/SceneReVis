#!/usr/bin/env python3
"""
Blender renderer utilities for ReSpace scenes.
Provides both external (subprocess) and in-process bpy rendering.

Usage (external CLI):
  python -m blender_renderer --scene /path/scene.json --out /path/output_dir

In-process (inside Blender):
  blender --background --python - <<'PY'
  import json
  from blender_renderer import render_scene_frame_bpy_inproc
  scene = json.load(open('/path/to/scene.json'))
  render_scene_frame_bpy_inproc(scene, '/tmp/out')
  PY
"""
from __future__ import annotations
import os, json, tempfile, subprocess, argparse, sys
from pathlib import Path

_FAST_MODE = os.environ.get('BPY_FAST_MODE', '0') == '1'
_FAST_RESOLUTION = 512 if _FAST_MODE else 1024
_FAST_SAMPLES = 8 if _FAST_MODE else 32

# Global verbose control
def _should_print_debug():
    """Check if debug prints should be shown based on environment variable"""
    return os.environ.get('BPY_VERBOSE', '0') == '1'

def _debug_print(*args, **kwargs):
    """Print debug message only if verbose mode is enabled"""
    if _should_print_debug():
        print(*args, **kwargs)

# ---------------- In-process bpy helpers ----------------
try:  # noqa: SIM105
    import bpy  # type: ignore
    _HAVE_BPY = True
except Exception:
    _HAVE_BPY = False

# ---------------- Texture helpers (mirror intent of viz.py fix_textures) ----------------
def _bpy_force_texture(material, tex_path, reuse_loaded=True):  # type: ignore
    """Ensure material uses texture at tex_path as Base Color (override if forced).
    Similar目标: 将贴图作为基色, metallic=0, roughness=1, 双面。
    """
    import bpy  # type: ignore
    from pathlib import Path as _P
    if not tex_path or not _P(tex_path).is_file():
        _debug_print(f'[bpy] texture path invalid: {tex_path}')
        return False
    _debug_print(f'[bpy] Applying texture {tex_path} to material {material.name}')
    if not material.use_nodes:
        material.use_nodes = True
        _debug_print(f'[bpy] Enabled nodes for material {material.name}')
    nt = material.node_tree
    # 删除已有节点若需要强制
    force = os.environ.get('BPY_FORCE_REAPPLY_TEXTURE','0')=='1'
    if force:
        removed = 0
        for n in list(nt.nodes):
            if n.type == 'TEX_IMAGE':
                nt.nodes.remove(n)
                removed += 1
        if removed > 0:
            _debug_print(f'[bpy] Removed {removed} existing texture nodes (force mode)')
    # 若已有贴图且不强制, 跳过
    existing_tex = [n for n in nt.nodes if n.type=='TEX_IMAGE']
    if not force and existing_tex:
        _debug_print(f'[bpy] Material {material.name} already has {len(existing_tex)} texture nodes, skipping')
        return True
    img_node = nt.nodes.new('ShaderNodeTexImage')
    try:
        # 重复加载相同图像时可复用已有 datablock
        existing = next((im for im in bpy.data.images if im.filepath == str(tex_path)), None) if reuse_loaded else None
        if existing:
            _debug_print(f'[bpy] Reusing existing image datablock: {existing.name}')
            img_node.image = existing
        else:
            _debug_print(f'[bpy] Loading new image: {tex_path}')
            img_node.image = bpy.data.images.load(str(tex_path))
            _debug_print(f'[bpy] Loaded image: {img_node.image.name} ({img_node.image.size[0]}x{img_node.image.size[1]})')
    except Exception as e:  # noqa: BLE001
        _debug_print(f'[bpy] load texture failed {tex_path}: {e}')
        nt.nodes.remove(img_node)
        return False
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf:
        _debug_print(f'[bpy] Creating new Principled BSDF for {material.name}')
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        # 连接到输出
        out = next((n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if out:
            nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
            _debug_print(f'[bpy] Connected BSDF to material output')
    else:
        _debug_print(f'[bpy] Using existing Principled BSDF')
    # 链接颜色
    nt.links.new(img_node.outputs['Color'], bsdf.inputs['Base Color'])
    _debug_print(f'[bpy] Connected texture to Base Color')
    # 调整金属/粗糙度
    try:
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 1.0
        _debug_print(f'[bpy] Set Metallic=0.0, Roughness=1.0')
    except Exception:
        _debug_print(f'[bpy] Failed to set metallic/roughness values')
        pass
    # 双面: 关闭背面剔除
    material.use_backface_culling = False
    _debug_print(f'[bpy] Material {material.name}: texture application complete')
    return True

def _bpy_clean_scene():
    import bpy
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for _ in range(2):
        try:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
        except:
            pass

def _bpy_create_floor(scene_data):
    import bpy, bmesh
    
    # Support both old format (bounds_bottom) and new format (room_envelope.bounds_bottom)
    bounds = None
    if 'room_envelope' in scene_data:
        bounds = scene_data['room_envelope'].get('bounds_bottom', [])
    else:
        bounds = scene_data.get('bounds_bottom', [])
        
    if bounds and len(bounds) >= 3:
        mesh = bpy.data.meshes.new('Floor')
        bm = bmesh.new()
        
        # Create floor vertices with thickness matching PyRender
        floor_thickness = 0.15  # Match PyRender version
        
        # Position floor so its top surface is at z=0 (bottom at z=-thickness)
        # This allows objects to sit properly on the floor
        bottom_verts = [bm.verts.new((p[0], -p[2], -floor_thickness)) for p in bounds]
        top_verts = [bm.verts.new((p[0], -p[2], 0.0)) for p in bounds]
        
        # Create bottom face
        bm.faces.new(bottom_verts)
        # Create top face (reversed order for correct normal)
        bm.faces.new(reversed(top_verts))
        
        # Create side faces
        num_verts = len(bounds)
        for i in range(num_verts):
            next_i = (i + 1) % num_verts
            # Create side face connecting bottom and top edges
            side_face = [
                bottom_verts[i],
                bottom_verts[next_i], 
                top_verts[next_i],
                top_verts[i]
            ]
            bm.faces.new(side_face)
        
        # Recalculate normals
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        
        bm.to_mesh(mesh); bm.free()
        obj = bpy.data.objects.new('Floor', mesh)
        bpy.context.collection.objects.link(obj)
    else:
        # 如果没有bounds信息，不创建地板
        _debug_print('[bpy] No floor bounds available, skipping floor creation')
        return
        
    mat = bpy.data.materials.new('FloorMat'); mat.use_nodes=True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs[0].default_value=(0.8,0.8,0.8,1)
    if obj.data.materials: obj.data.materials[0]=mat
    else: obj.data.materials.append(mat)
    return obj

def _bpy_import_real_asset(obj_data, index, assets_base_env_var='PTH_3DFUTURE_ASSETS'):
    """Import real asset via jid/uid using GLB format only.
    
    Supports both asset sources:
    - 3D-FUTURE: uses 'jid' field, loads from PTH_3DFUTURE_ASSETS/{jid}/raw_model.glb
    - Objaverse: uses 'uid' field, loads GLB from ~/.objaverse cache
    
    For Objaverse assets, GLBs should be pre-downloaded by the retriever.
    Uses PathConfig for unified path configuration.
    
    Skip object if import fails or path missing - no placeholders.
    Coordinate mapping: original (x,y,z)-> blender (x,-z,y).
    Rotation quaternion in JSON is [x,y,z,w] in original axes; convert to matrix then axis-map.
    """
    import bpy  # type: ignore
    import colorsys, math  # type: ignore
    import mathutils  # type: ignore
    
    # Determine asset source and get appropriate ID
    asset_source = obj_data.get('asset_source', '3d-future')
    
    mesh_path = None
    asset_dir = None
    asset_id = None
    
    # First, check if there's an explicit path in the JSON (e.g., objaverse_path from LayoutVLM)
    explicit_path = obj_data.get('objaverse_path') or obj_data.get('glb_path') or obj_data.get('mesh_path')
    if explicit_path:
        explicit_path = Path(explicit_path)
        if explicit_path.is_file():
            mesh_path = explicit_path
            asset_dir = explicit_path.parent
            asset_id = obj_data.get('uid') or explicit_path.stem
            _debug_print(f'[bpy] Using explicit path from JSON: {mesh_path}')
    
    if asset_source == 'objaverse' and mesh_path is None:
        # Objaverse asset: use uid and load GLB from cache
        uid = obj_data.get('uid')
        if uid and len(uid) >= 2:
            asset_id = uid
            _debug_print(f'[bpy] Looking for Objaverse GLB: {uid}')
            
            # Build cache directories list - use PathConfig first, then fallbacks
            objaverse_cache_dirs = []
            
            # Try PathConfig first
            try:
                from path_config import PathConfig
                config = PathConfig.get_instance()
                if config.objaverse_glb_cache_dir:
                    objaverse_cache_dirs.append(Path(config.objaverse_glb_cache_dir))
            except ImportError:
                pass
            
            # Check multiple possible GLB cache locations
            # Priority: env var > local dev > cloud storage > home fallback
            if os.environ.get("OBJAVERSE_PROCESSED_DIR"):
                 objaverse_cache_dirs.append(Path(os.environ["OBJAVERSE_PROCESSED_DIR"]))

            if os.environ.get("OBJAVERSE_GLB_CACHE_DIR"):
                objaverse_cache_dirs.append(Path(os.environ["OBJAVERSE_GLB_CACHE_DIR"]) / "glbs")
            objaverse_cache_dirs.extend([
                Path("/path/to/data/datasets/objathor-assets/glbs"),  # Local development
                Path("/path/to/datasets/objathor-assets/glbs"),  # Cloud storage (blobfuse)
                Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs",  # Local fallback
            ])
            
            # Debug: print all cache dirs being checked
            _debug_print(f'[bpy] Checking Objaverse cache dirs: {objaverse_cache_dirs}')
            
            subdir_name = uid[:2]
            
            for objaverse_cache in objaverse_cache_dirs:
                _debug_print(f'[bpy] Checking cache dir: {objaverse_cache}, is_dir={objaverse_cache.is_dir() if objaverse_cache.exists() else "not_exists"}')
                if not objaverse_cache.is_dir():
                    continue
                
                # 1. Check flat structure: CACHE/uid/uid.glb (LayoutVLM processed style)
                flat_candidate = objaverse_cache / uid / f"{uid}.glb"
                _debug_print(f'[bpy] Checking Flat GLB candidate: {flat_candidate}')
                if flat_candidate.is_file():
                    mesh_path = flat_candidate
                    asset_dir = flat_candidate.parent
                    _debug_print(f'[bpy] Found GLB at: {mesh_path}')
                    break

                # 2. Check nested structure: CACHE/uid[:2]/uid.glb (Standard Objaverse style)
                candidate_glb = objaverse_cache / subdir_name / f"{uid}.glb"
                _debug_print(f'[bpy] Checking Nested GLB candidate: {candidate_glb}')
                if candidate_glb.is_file():
                    mesh_path = candidate_glb
                    asset_dir = candidate_glb.parent
                    _debug_print(f'[bpy] Found cached GLB: {mesh_path}')
                    break
            
            if mesh_path is None:
                _debug_print(f'[bpy] WARNING: GLB not in cache for {uid}. Please ensure retriever downloads GLBs.')
    
    else:
        # 3D-FUTURE asset: use jid (default)
        jid = obj_data.get('jid') or obj_data.get('sampled_asset_jid') or f'obj_{index}'
        asset_id = jid
        asset_root = os.getenv(assets_base_env_var)
        if asset_root:
            candidate = Path(asset_root) / jid / 'raw_model.glb'
            if candidate.is_file():
                mesh_path = candidate
                asset_dir = candidate.parent
    
    if asset_id is None:
        asset_id = f'obj_{index}'
    
    imported_objects = []
    
    # Load GLB format
    if mesh_path is not None:
        try:
            bpy.ops.import_scene.gltf(filepath=str(mesh_path))
            # Newly imported objects are selected; collect them
            imported_objects = [o for o in bpy.context.selected_objects]
            
            # ==== Objaverse 坐标系校正 ====
            if asset_source == 'objaverse' and imported_objects:
                _debug_print(f'[bpy] Asset {asset_id}: applying Objaverse GLB orientation correction (reset to zero only)')
                
                # 选中所有导入的对象
                bpy.ops.object.select_all(action='DESELECT')
                for o in imported_objects:
                    o.select_set(True)
                if imported_objects:
                    bpy.context.view_layer.objects.active = imported_objects[0]
                
                # 重置旋转到零
                for o in imported_objects:
                    o.rotation_euler = (0, 0, 0)
                
                # 应用变换
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
                
                _debug_print(f'[bpy] Asset {asset_id}: orientation correction applied')
        except Exception as e:  # noqa: BLE001
            _debug_print(f'[bpy] Failed to import {mesh_path}: {e}')
    
    if not imported_objects:
        # 如果无法导入真实资产，不创建任何占位符
        _debug_print(f'[bpy] Skipping object {index} - failed to import real asset {asset_id} (source: {asset_source})')
        return None
    
    # Helper function to identify non-furniture meshes (floors, environments, etc.)
    def is_non_furniture_mesh(obj_name):
        """Check if mesh is likely a non-furniture object (floor, environment, etc.)
        that should be excluded from bbox calculation and possibly deleted."""
        name_lower = obj_name.lower()
        # Common names for scene elements that are not the actual furniture
        exclude_patterns = [
            'floor', 'plane', 'ground', 'surface',  # ground planes
            'environment', 'ambient', 'light', 'lamp_light',  # lighting
            'sky', 'backdrop', 'background',  # backgrounds
            'camera', 'empty',  # scene setup
        ]
        return any(pattern in name_lower for pattern in exclude_patterns)
    
    # Remove non-furniture meshes (like floor planes) from imported objects
    # These can interfere with bbox calculation and appear in renders
    objects_to_remove = []
    for obj in imported_objects:
        if obj.type == 'MESH' and is_non_furniture_mesh(obj.name):
            objects_to_remove.append(obj)
    
    for obj in objects_to_remove:
        _debug_print(f'[bpy] Removing non-furniture mesh: {obj.name}')
        imported_objects.remove(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    
    if not imported_objects:
        _debug_print(f'[bpy] Skipping object {index} - no furniture meshes found after filtering')
        return None
    
    # Create parent empty to hold multi-part asset
    parent = bpy.data.objects.new(f'A_{index}_{asset_id[:8]}', None)
    bpy.context.collection.objects.link(parent)
    for o in imported_objects:
        o.parent = parent
    
    # Calculate bounding box of imported objects to determine actual size
    # Note: Non-furniture meshes have already been removed above, so no filtering needed here
    def get_combined_bbox(objects):
        """Get combined bounding box of all mesh objects."""
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        for obj in objects:
            if obj.type != 'MESH' or not obj.data:
                continue
            # Get world-space bounding box corners
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            for corner in bbox_corners:
                for i in range(3):
                    min_coords[i] = min(min_coords[i], corner[i])
                    max_coords[i] = max(max_coords[i], corner[i])
        
        if min_coords[0] == float('inf'):
            return [1.0, 1.0, 1.0]  # fallback
        
        return [max_coords[i] - min_coords[i] for i in range(3)]
    
    # Get actual model dimensions (in Blender coordinates: x, y, z where y is depth, z is height)
    model_bbox = get_combined_bbox(imported_objects)
    _debug_print(f'[bpy] Asset {asset_id}: model bbox (blender coords) = {model_bbox}')
    
    # Build transform matrix
    pos = obj_data.get('pos',[0,0,0])
    rot = obj_data.get('rot')
    scale = obj_data.get('scale')  # explicit scale if provided
    target_size = obj_data.get('size')  # target size from scene JSON
    retrieved_size = obj_data.get('retrieved_size')  # original asset size from retrieval
    
    # Calculate scale factor
    # Scene uses (x, y, z) where y is height, z is depth
    # Blender uses (x, y, z) where z is height, y is depth
    # So target_size[0,1,2] = (width, height, depth) in scene coords
    # model_bbox[0,1,2] = (width, depth, height) in blender coords
    
    if target_size and len(target_size) == 3:
        # Target size is in scene coordinates: (width, height, depth)
        # Model bbox is in blender coordinates: (width, depth, height)
        # We need to scale model to match target size
        
        # Avoid division by zero
        model_width = max(model_bbox[0], 0.001)
        model_depth = max(model_bbox[1], 0.001)  # blender Y = scene Z (depth)
        model_height = max(model_bbox[2], 0.001)  # blender Z = scene Y (height)
        
        target_width = target_size[0]
        target_height = target_size[1]  # scene Y = height
        target_depth = target_size[2]   # scene Z = depth
        
        scale_x = target_width / model_width
        scale_y = target_depth / model_depth   # blender Y = depth
        scale_z = target_height / model_height  # blender Z = height
        
        _debug_print(f'[bpy] Asset {asset_id}: target_size={target_size}, scale factors=({scale_x:.3f}, {scale_y:.3f}, {scale_z:.3f})')
        
        S = mathutils.Matrix.Diagonal((scale_x, scale_y, scale_z, 1))
    elif scale:
        # Use explicit scale if provided
        if isinstance(scale, (int,float)):
            S = mathutils.Matrix.Scale(scale,4)
        elif isinstance(scale, (list,tuple)) and len(scale)==3:
            # Apply axis mapping to scale: (x,y,z) -> (x,z,y) to match coordinate system
            S = mathutils.Matrix.Diagonal((scale[0], scale[2], scale[1], 1))
        else:
            S = mathutils.Matrix.Identity(4)
    else:
        S = mathutils.Matrix.Identity(4)
    
    if rot:
        # Quaternion given as [x,y,z,w]; need to transform to Blender coordinate system
        # Original quaternion represents rotation in (x,y,z) space
        # We need to map this to Blender's (x,-z,y) space
        
        # Convert to wxyz format for Blender: [x,y,z,w] -> (w,x,y,z)
        q_orig = mathutils.Quaternion((rot[3], rot[0], rot[1], rot[2]))
        
        # For coordinate system transformation (x,y,z) -> (x,-z,y):
        # The quaternion needs to be adjusted to account for axis remapping
        # Since we're mapping y->-z and z->y, we need to swap and negate appropriately
        # q_blender = (w, x, z, -y) to handle the axis swap
        q_blender = mathutils.Quaternion((q_orig.w, q_orig.x, -q_orig.z, q_orig.y))
        
        R = q_blender.to_matrix().to_4x4()
    else:
        R = mathutils.Matrix.Identity(4)
    
    # First, center the model and place bottom at origin
    # This ensures consistent positioning regardless of where the model's origin is
    # Note: Non-furniture meshes have already been removed above
    def get_model_bounds(objects):
        """Get min/max bounds of all mesh objects"""
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        for obj in objects:
            if obj.type != 'MESH' or not obj.data:
                continue
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            for corner in bbox_corners:
                for i in range(3):
                    min_coords[i] = min(min_coords[i], corner[i])
                    max_coords[i] = max(max_coords[i], corner[i])
        
        if min_coords[0] == float('inf'):
            return None, None
        return min_coords, max_coords
    
    min_bounds, max_bounds = get_model_bounds(imported_objects)
    
    # Calculate center offset to move model so its center-bottom is at origin
    if min_bounds and max_bounds:
        # Center XY, but place bottom at Z=0
        center_x = (min_bounds[0] + max_bounds[0]) / 2
        center_y = (min_bounds[1] + max_bounds[1]) / 2
        bottom_z = min_bounds[2]  # bottom of the model
        
        # Move all imported objects to center the model
        offset = mathutils.Vector((-center_x, -center_y, -bottom_z))
        for obj in imported_objects:
            obj.location += offset
        
        _debug_print(f'[bpy] Asset {asset_id}: centered model, offset={offset}')
    
    # Position transformation: scene (x,y,z) -> blender (x,-z,y)
    # In scene coords: x=width, y=height (vertical), z=depth
    # In blender coords: x=width, y=depth, z=height
    # pos[1] in scene is the Y coordinate (height/vertical position)
    # For objects on the floor, pos[1]=0 means bottom is at ground level
    T = mathutils.Matrix.Translation(mathutils.Vector((pos[0], -pos[2], pos[1])))
    
    # Apply transformations in the correct order: Translation * Rotation * Scale
    parent.matrix_world = T @ R @ S
    # Simple material tint for parent children if they lack materials (optional)
    tint_needed = [o for o in imported_objects if not o.data or not getattr(o.data, 'materials', [])]
    # If textures are missing but there is a texture.png file, try to apply it.
    if asset_dir:
        # 纹理候选列表 (可通过环境变量 BPY_TEXTURE_CANDIDATES 自定义, 逗号分隔)
        candidates_env = os.environ.get('BPY_TEXTURE_CANDIDATES')
        if candidates_env:
            names = [c.strip() for c in candidates_env.split(',') if c.strip()]
        else:
            names = ['texture.png', 'texture.jpg', 'image.png', 'image.jpg']
        tex_path = None
        _debug_print(f'[bpy] Asset {asset_id}: searching textures in {asset_dir}')
        for n in names:
            p = asset_dir / n
            if p.is_file():
                tex_path = p
                _debug_print(f'[bpy] Found texture: {tex_path}')
                break
            else:
                _debug_print(f'[bpy] Not found: {p}')
        
        # If no textures found and directory has scale suffix, try base directory
        if tex_path is None and "-(" in str(asset_dir.name):
            # Remove scale suffix from directory name
            base_dir_name = str(asset_dir.name).split("-(")[0]
            base_asset_dir = asset_dir.parent / base_dir_name
            
            if base_asset_dir.exists():
                _debug_print(f'[bpy] Asset {asset_id}: trying base directory {base_asset_dir}')
                for n in names:
                    p = base_asset_dir / n
                    if p.is_file():
                        tex_path = p
                        _debug_print(f'[bpy] Found texture: {tex_path}')
                        break
                    else:
                        _debug_print(f'[bpy] Not found: {p}')
        if tex_path:
            texture_applied = 0
            for oi, o in enumerate(imported_objects):
                if not getattr(o, 'data', None):
                    _debug_print(f'[bpy] Object {oi} has no mesh data, skipping')
                    continue
                mats = getattr(o.data, 'materials', None)
                if not mats or len(mats)==0:
                    # 创建一个材质
                    mat = bpy.data.materials.new(f'T_{index}_{oi}')
                    success = _bpy_force_texture(mat, tex_path)
                    o.data.materials.append(mat)
                    if success:
                        texture_applied += 1
                        _debug_print(f'[bpy] Created & applied texture to object {oi} (new material)')
                    else:
                        _debug_print(f'[bpy] Failed to apply texture to object {oi} (new material)')
                else:
                    # 为所有材质尝试应用纹理（强制/或如果未有图像节点）
                    for mi, m in enumerate(mats):
                        success = _bpy_force_texture(m, tex_path)
                        if success:
                            texture_applied += 1
                            _debug_print(f'[bpy] Applied texture to object {oi} material {mi}')
                        else:
                            _debug_print(f'[bpy] Failed to apply texture to object {oi} material {mi}')
            _debug_print(f'[bpy] Asset {asset_id}: {texture_applied} materials got textures applied')
        else:
            _debug_print(f'[bpy] Asset {asset_id}: no texture files found')
    if tint_needed:
        mat=bpy.data.materials.new(f'M_{index}') ; mat.use_nodes=True
        hue=(index*0.61803398875)%1.0; rgb=colorsys.hsv_to_rgb(hue,0.4,0.9)
        bsdf=mat.node_tree.nodes.get('Principled BSDF')
        if bsdf: bsdf.inputs[0].default_value=(*rgb,1.0)
        for o in tint_needed:
            if o.data and hasattr(o.data,'materials'):
                if o.data.materials: o.data.materials[0]=mat
                else: o.data.materials.append(mat)
    return parent

def _bpy_setup_camera(view_type, scene_data):
    import bpy, math, random, mathutils, time, os
    
    # Seed random with nanoseconds + process ID to ensure different angles each run
    random.seed(time.time_ns() + os.getpid())
    
    # Support both old format (bounds_bottom) and new format (room_envelope.bounds_bottom)
    bounds = None
    if 'room_envelope' in scene_data:
        bounds = scene_data['room_envelope'].get('bounds_bottom', [])
    else:
        bounds = scene_data.get('bounds_bottom', [])
        
    if bounds:
        xs=[p[0] for p in bounds]; zs=[p[2] for p in bounds]
        span=max(max(xs)-min(xs), max(zs)-min(zs)) or 10
        center_x=(max(xs)+min(xs))/2
        center_z_scene=(max(zs)+min(zs))/2
    else:
        span=10
        center_x=0
        center_z_scene=0
    center_y=-center_z_scene  # Coordinate mapping: scene z -> blender -Y
    dist=max(8, span*1.5)

    def aim_camera(cam_obj, target):
        target_vec=mathutils.Vector(target)
        direction=target_vec - cam_obj.location
        if direction.length == 0:
            direction = mathutils.Vector((0.0, 0.0, -1.0))
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    if view_type=='top':
        loc=(center_x, center_y, dist)
        bpy.ops.object.camera_add(location=loc); cam=bpy.context.active_object
        cam.data.type = 'ORTHO'
        cam.data.ortho_scale = span * 1.2
        aim_camera(cam, (center_x, center_y, 0.0))
    else:
        angles = [math.radians(-45), math.radians(45), math.radians(135), math.radians(-135)]
        angle = random.choice(angles)
        loc=(center_x + dist*0.9*math.cos(angle),
             center_y + dist*0.9*math.sin(angle),
             dist*0.6)
        bpy.ops.object.camera_add(location=loc); cam=bpy.context.active_object
        aim_camera(cam, (center_x, center_y, 0.5))
        cam.data.lens = 40
    
    bpy.context.scene.camera=cam; return cam

def _bpy_setup_lighting():
    import bpy
    # Global sun light for consistent illumination
    bpy.ops.object.light_add(type='SUN', location=(0,0,20))
    sun=bpy.context.active_object
    sun.data.energy=2.2
    sun.data.angle=0.4
    sun.data.use_shadow=True
    sun.data.shadow_soft_size=1.2
    
    # Soft area lights around the scene for fill light
    positions=[(0,0,12),(8,8,10),(-8,8,10),(8,-8,10),(-8,-8,10),(0,8,10),(8,0,10),(0,-8,10),(-8,0,10)]
    for pos in positions:
        bpy.ops.object.light_add(type='AREA', location=pos)
        light=bpy.context.active_object
        light.data.energy=12.0
        light.data.size=10
        light.data.use_shadow=True
        light.data.shadow_soft_size=2.0
        light.data.color=(1.0,0.96,0.9)
    
    # Ambient world light to avoid dark corners
    scene=bpy.context.scene
    world=scene.world or bpy.data.worlds.new('SceneWorld')
    scene.world=world
    world.use_nodes=True
    bg=world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value=(1.0,1.0,1.0,1.0)
        bg.inputs[1].default_value=0.32

def _bpy_render(output_path: Path, filename: str):
    import bpy
    sc=bpy.context.scene
    if _FAST_MODE:
        _debug_print('[bpy] Fast render mode enabled (512px, 8 samples)')
    else:
        _debug_print('[bpy] High-quality render mode (1024px, 32 samples)')
    sc.render.engine='BLENDER_EEVEE'
    # Use PNG format to support transparency
    sc.render.image_settings.file_format='PNG'
    sc.render.image_settings.color_mode='RGBA'
    sc.render.image_settings.compression=15  # PNG compression
    sc.render.resolution_x=_FAST_RESOLUTION; sc.render.resolution_y=_FAST_RESOLUTION; sc.render.resolution_percentage=100
    
    # Enable transparent background
    sc.render.film_transparent = True
    
    # Enable compositor for transparent background
    sc.use_nodes = True

    # Use a neutral color transform to avoid gray/washed-out look
    sc.view_settings.view_transform='Standard'
    sc.view_settings.look='None'
    sc.view_settings.exposure=0.0
    sc.view_settings.gamma=1.0
    sc.display_settings.display_device='sRGB'

    # Slightly soften EEVEE shadows for depth cues
    if hasattr(sc, 'eevee'):
        sc.eevee.use_soft_shadows=True
        sc.eevee.shadow_cube_size='1024'
        sc.eevee.shadow_cascade_size='1024'
        sc.eevee.light_threshold=0.001
        sc.eevee.taa_render_samples=_FAST_SAMPLES
        sc.eevee.taa_samples=_FAST_SAMPLES
    
    sc.render.filepath=str(output_path / f'{filename}.png')
    bpy.ops.render.render(write_still=True)
    return sc.render.filepath

def render_scene_frame_bpy_inproc(scene_data: dict, output_dir: str | Path, filename: str='frame', enable_visualization: bool = None):
    """Render scene with Blender, optionally adding 3D visualization helpers.
    
    Args:
        scene_data: Scene dict
        output_dir: Output directory
        filename: Output filename (without extension)
        enable_visualization: Whether to add 3D visualization (bbox, arrows, etc.)
                              If None, checks BPY_ENABLE_VISUALIZATION env var
    """
    if not _HAVE_BPY:
        raise RuntimeError('bpy not available in this interpreter. Run inside Blender.')
    from pathlib import Path as _P
    output_dir=_P(output_dir)
    (output_dir/'top').mkdir(parents=True, exist_ok=True)
    (output_dir/'diag').mkdir(parents=True, exist_ok=True)
    generated=[]
    
    # Check visualization setting
    if enable_visualization is None:
        enable_visualization = os.environ.get('BPY_ENABLE_VISUALIZATION', '0') == '1'
    
    # Try to import visualization module
    add_scene_visualization = None
    if enable_visualization:
        try:
            # Try relative import first, then absolute
            try:
                from .visualization_3d import add_scene_visualization
            except ImportError:
                from visualization_3d import add_scene_visualization
            _debug_print('[bpy] 3D visualization module loaded')
        except ImportError as e:
            _debug_print(f'[bpy] Warning: Could not import visualization_3d: {e}')
            enable_visualization = False
    
    # Extract objects from either old format (objects) or new format (groups)
    objects_list = []
    if 'groups' in scene_data:
        # New format: extract objects from groups
        for group in scene_data.get('groups', []):
            objects_list.extend(group.get('objects', []))
    elif 'objects' in scene_data:
        # Old format: direct objects list
        objects_list = scene_data.get('objects', [])
    
    for view in ('top','diag'):
        _bpy_clean_scene(); _bpy_create_floor(scene_data)
        for i,obj in enumerate(objects_list):
            try:
                _bpy_import_real_asset(obj,i)
            except Exception as e:  # noqa: BLE001
                _debug_print(f'[bpy] skipping object {i} due to import failure: {e}')
        
        # Add 3D visualization if enabled
        if enable_visualization and add_scene_visualization:
            try:
                _debug_print(f'[bpy] Adding 3D visualization for {view} view...')
                add_scene_visualization(
                    scene_data,
                    enable_bbox=True,
                    enable_arrows=True,
                    enable_labels=False,  # Labels can be heavy
                    enable_coord=True,    # Enable coordinate grid on floor
                    enable_axes=True
                )
                _debug_print(f'[bpy] 3D visualization added successfully')
            except Exception as e:
                _debug_print(f'[bpy] Warning: Failed to add visualization: {e}')
        
        _bpy_setup_lighting(); _bpy_setup_camera(view, scene_data)
        path=_bpy_render(output_dir/view, filename)
        generated.append(path)
    return generated

# ---------------- External (subprocess) Blender launcher ----------------

def create_blender_script(scene_data, output_path, view_type='diagonal'):
    return f"""import bpy, json, colorsys, bmesh\nscene_data=json.loads('''{json.dumps(scene_data)}''')\n{_BLENDER_INLINE}render('{output_path}','{view_type}')\n"""

_BLENDER_INLINE = """
import bpy, colorsys, bmesh, os

def clean():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def floor(sd):
    # Support both old format (bounds_bottom) and new format (room_envelope.bounds_bottom)
    b = None
    if 'room_envelope' in sd:
        b = sd['room_envelope'].get('bounds_bottom', [])
    else:
        b = sd.get('bounds_bottom', [])
    if b and len(b)>=3:
        import bmesh
        m=bpy.data.meshes.new('Floor'); bm=bmesh.new(); vs=[bm.verts.new((p[0],-p[2],0)) for p in b]; bm.faces.new(vs); bm.to_mesh(m); bm.free(); o=bpy.data.objects.new('Floor',m); bpy.context.collection.objects.link(o)
    else:
        bpy.ops.mesh.primitive_plane_add(size=20,location=(0,0,0)); o=bpy.context.active_object; o.name='Floor'
    mat=bpy.data.materials.new('FloorMat'); mat.use_nodes=True; bsdf=mat.node_tree.nodes.get('Principled BSDF');
    if bsdf: bsdf.inputs[0].default_value=(0.8,0.8,0.8,1)

def add_objs(sd):
    # Extract objects from either old format (objects) or new format (groups)
    objects_list = []
    if 'groups' in sd:
        for group in sd.get('groups', []):
            objects_list.extend(group.get('objects', []))
    elif 'objects' in sd:
        objects_list = sd.get('objects', [])
    
    import os, mathutils
    from pathlib import Path
    asset_root = os.getenv('PTH_3DFUTURE_ASSETS')
    
    # Multiple GLB cache locations (local development > cloud storage > home fallback)
    objaverse_cache_dirs = [
        os.environ.get('OBJAVERSE_GLB_CACHE_DIR', '') + '/glbs' if os.environ.get('OBJAVERSE_GLB_CACHE_DIR') else None,
        '/path/to/data/datasets/objathor-assets/glbs',  # Local development path
        '/path/to/datasets/objathor-assets/glbs',  # Cloud storage (blobfuse)
        os.path.expanduser('~/.objaverse/hf-objaverse-v1/glbs'),  # Home fallback
    ]
    objaverse_cache_dirs = [p for p in objaverse_cache_dirs if p]
    
    def get_combined_bbox(objects):
        min_c = [float('inf')] * 3
        max_c = [float('-inf')] * 3
        for obj in objects:
            if obj.type != 'MESH' or not obj.data: continue
            for corner in obj.bound_box:
                wc = obj.matrix_world @ mathutils.Vector(corner)
                for i in range(3):
                    min_c[i] = min(min_c[i], wc[i])
                    max_c[i] = max(max_c[i], wc[i])
        if min_c[0] == float('inf'): return [1.0, 1.0, 1.0], None, None
        return [max_c[i] - min_c[i] for i in range(3)], min_c, max_c
    
    for i, obj in enumerate(objects_list):
        try:
            asset_source = obj.get('asset_source', '3d-future')
            glb_path = None
            asset_id = None
            
            if asset_source == 'objaverse':
                uid = obj.get('uid')
                if uid:
                    asset_id = uid
                    # Search in all cache directories
                    for cache_dir in objaverse_cache_dirs:
                        if os.path.isdir(cache_dir):
                            for subdir in os.listdir(cache_dir):
                                candidate = Path(cache_dir) / subdir / f'{uid}.glb'
                                if candidate.is_file():
                                    glb_path = candidate
                                    break
                            if glb_path:
                                break
            else:
                jid = obj.get('jid', '')
                if jid and jid != '<NEED_RETRIEVAL>' and asset_root:
                    asset_id = jid
                    candidate = Path(asset_root) / jid / 'raw_model.glb'
                    if candidate.exists():
                        glb_path = candidate
            
            if not glb_path or not asset_id:
                continue
            
            # Import GLB
            bpy.ops.import_scene.gltf(filepath=str(glb_path))
            imported = [o for o in bpy.context.selected_objects]
            if not imported:
                continue
            
            # ==== Filter out non-furniture meshes (floors, environments, etc.) ====
            def is_non_furniture_mesh(obj_name):
                name_lower = obj_name.lower()
                exclude_patterns = ['floor', 'plane', 'ground', 'surface', 'environment', 
                                   'ambient', 'light', 'lamp_light', 'sky', 'backdrop', 
                                   'background', 'camera', 'empty']
                return any(pattern in name_lower for pattern in exclude_patterns)
            
            # Remove non-furniture meshes
            objects_to_remove = []
            for o in imported:
                if o.type == 'MESH' and is_non_furniture_mesh(o.name):
                    objects_to_remove.append(o)
            for o in objects_to_remove:
                print(f'[bpy] Asset {asset_id}: Removing non-furniture mesh: {o.name}')
                imported.remove(o)
                bpy.data.objects.remove(o, do_unlink=True)
            
            if not imported:
                print(f'[bpy] Asset {asset_id}: No furniture meshes found after filtering')
                continue
            
            # ==== Objaverse 坐标系校正 ====
            # Objaverse 模型：重置旋转到零（测试无额外旋转）
            import math
            if asset_source == 'objaverse':
                print(f'[bpy] Asset {asset_id}: applying Objaverse orientation correction (reset to zero only)')
                bpy.ops.object.select_all(action='DESELECT')
                for o in imported:
                    o.select_set(True)
                    o.rotation_euler = (0, 0, 0)
                if imported:
                    bpy.context.view_layer.objects.active = imported[0]
                bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
            
            # Create parent empty
            parent = bpy.data.objects.new(f'A_{i}_{asset_id[:8]}', None)
            bpy.context.collection.objects.link(parent)
            for o in imported:
                o.parent = parent
            
            # Force update to ensure transforms are applied
            bpy.context.view_layer.update()
            
            # Get model bbox for scaling
            bbox_orig, min_b_orig, max_b_orig = get_combined_bbox(imported)
            print(f'[bpy] Asset {asset_id}: Model bbox (X,Y,Z)={bbox_orig}')
            
            # Center model: XY center, Z bottom at 0
            bbox, min_b, max_b = bbox_orig, min_b_orig, max_b_orig
            if min_b and max_b:
                cx = (min_b[0] + max_b[0]) / 2
                cy = (min_b[1] + max_b[1]) / 2
                bz = min_b[2]
                offset = mathutils.Vector((-cx, -cy, -bz))
                for o in imported:
                    o.location += offset
                print(f'[bpy] Asset {asset_id}: centered with offset {offset}')
            
            # Calculate scale from target size
            target_size = obj.get('size')
            if target_size and len(target_size) == 3:
                # Model bounding box in Blender coordinates (after import and potential rotation fix)
                mw = max(bbox[0], 0.001)  # Blender X = width
                md = max(bbox[1], 0.001)  # Blender Y = depth  
                mh = max(bbox[2], 0.001)  # Blender Z = height
                
                # Scene coordinates: x=width, y=height, z=depth
                # Mapping:
                # Scene X (width) -> Blender X
                # Scene Y (height) -> Blender Z
                # Scene Z (depth) -> Blender Y
                
                sx = target_size[0] / mw  # scene width -> blender X
                sy = target_size[2] / md  # scene depth -> blender Y
                sz = target_size[1] / mh  # scene height -> blender Z
                
                print(f'[bpy] Asset {asset_id}: Model bbox (X,Y,Z)=({mw:.3f}, {md:.3f}, {mh:.3f})')
                print(f'[bpy] Asset {asset_id}: Target size (W,H,D)={target_size}')
                print(f'[bpy] Asset {asset_id}: Scale factors (sX,sY,sZ)=({sx:.3f}, {sy:.3f}, {sz:.3f})')
                
                S = mathutils.Matrix.Diagonal((sx, sy, sz, 1))
                
            else:
                S = mathutils.Matrix.Identity(4)
            
            # Rotation
            rot = obj.get('rot')
            if rot:
                q = mathutils.Quaternion((rot[3], rot[0], rot[1], rot[2]))
                q_b = mathutils.Quaternion((q.w, q.x, -q.z, q.y))
                R = q_b.to_matrix().to_4x4()
            else:
                R = mathutils.Matrix.Identity(4)
            
            # Position: scene (x,y,z) -> blender (x,-z,y)
            pos = obj.get('pos', [0,0,0])
            T = mathutils.Matrix.Translation(mathutils.Vector((pos[0], -pos[2], pos[1])))
            
            parent.matrix_world = T @ R @ S
            
        except Exception as e:
            print(f'Failed to add object {i}: {e}')

def cam(sd,view):
    import random, math, mathutils
    # Support both old format (bounds_bottom) and new format (room_envelope.bounds_bottom)
    b = None
    if 'room_envelope' in sd:
        b = sd['room_envelope'].get('bounds_bottom', [])
    else:
        b = sd.get('bounds_bottom', [])
    if b:
        xs=[p[0] for p in b]; zs=[p[2] for p in b]; span=max(max(xs)-min(xs), max(zs)-min(zs)) or 10
        center_x=(max(xs)+min(xs))/2
        center_z_scene=(max(zs)+min(zs))/2
    else:
        span=10
        center_x=0
        center_z_scene=0
    center_y=-center_z_scene
    d=max(8, span*1.5)

    def aim(cam_obj, target):
        direction=mathutils.Vector(target) - cam_obj.location
        if direction.length == 0:
            direction=mathutils.Vector((0.0,0.0,-1.0))
        cam_obj.rotation_euler=direction.to_track_quat('-Z','Y').to_euler()
    
    if view=='top':
        loc=(center_x, center_y, d)
        bpy.ops.object.camera_add(location=loc); c=bpy.context.active_object
        c.data.type='ORTHO'; c.data.ortho_scale=span*1.2
        aim(c,(center_x, center_y, 0.0))
    else:
        angles=[math.radians(-45), math.radians(45), math.radians(135), math.radians(-135)]
        angle=random.choice(angles)
        loc=(center_x + d*0.9*math.cos(angle), center_y + d*0.9*math.sin(angle), d*0.6)
        bpy.ops.object.camera_add(location=loc); c=bpy.context.active_object
        aim(c,(center_x, center_y, 0.5))
        c.data.lens=40
    bpy.context.scene.camera=c

def light():
    bpy.ops.object.light_add(type='SUN',location=(0,0,20)); sun=bpy.context.active_object; sun.data.energy=2.2; sun.data.angle=0.4; sun.data.use_shadow=True; sun.data.shadow_soft_size=1.2
    positions=[(0,0,12),(8,8,10),(-8,8,10),(8,-8,10),(-8,-8,10),(0,8,10),(8,0,10),(0,-8,10),(-8,0,10)]
    for pos in positions:
        bpy.ops.object.light_add(type='AREA',location=pos); light=bpy.context.active_object; light.data.energy=12.0; light.data.size=10; light.data.use_shadow=True; light.data.shadow_soft_size=2.0; light.data.color=(1.0,0.96,0.9)
    world=bpy.context.scene.world or bpy.data.worlds.new('SceneWorld'); bpy.context.scene.world=world; world.use_nodes=True
    bg=world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs[0].default_value=(1.0,1.0,1.0,1.0)
        bg.inputs[1].default_value=0.32

def do_render(path,view):
    fast_mode=os.environ.get('BPY_FAST_MODE','0')=='1'
    res=400 if fast_mode else 1024
    samples=8 if fast_mode else 32
    sc=bpy.context.scene; sc.render.engine='BLENDER_EEVEE'; sc.render.image_settings.file_format='PNG'; sc.render.image_settings.color_mode='RGBA'; sc.render.image_settings.compression=15; sc.render.resolution_x=res; sc.render.resolution_y=res; sc.render.resolution_percentage=100; sc.render.film_transparent=True; sc.use_nodes=True; sc.view_settings.view_transform='Standard'; sc.view_settings.look='None'; sc.view_settings.exposure=0.0; sc.view_settings.gamma=1.0; sc.display_settings.display_device='sRGB';
    if hasattr(sc,'eevee'):
        sc.eevee.use_soft_shadows=True; sc.eevee.shadow_cube_size='1024'; sc.eevee.shadow_cascade_size='1024'; sc.eevee.light_threshold=0.001; sc.eevee.taa_render_samples=samples; sc.eevee.taa_samples=samples
    sc.render.filepath=path; bpy.ops.render.render(write_still=True)

def render(out_dir, view_type):
    import pathlib
    for v in ('top','diag'):
        clean(); floor(scene_data); add_objs(scene_data); light(); cam(scene_data,v); target=pathlib.Path(out_dir)/v; target.mkdir(parents=True, exist_ok=True); do_render(str(target/'frame.png'), v)
"""

def render_with_blender(scene, output_path, filename='frame', view_type='diagonal', resolution=(1024,1024)):
    output_dir=Path(output_path); output_dir.mkdir(parents=True, exist_ok=True)
    script_fd=tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    script_content=create_blender_script(scene, str(output_dir), view_type)
    script_fd.write(script_content); script_fd.flush(); script_path=script_fd.name; script_fd.close()
    cmd=['blender','--background','--python',script_path]
    # Use errors='replace' to avoid UnicodeDecodeError if blender outputs non-utf8 characters
    res=subprocess.run(cmd, capture_output=True, text=True, errors='replace')
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr, file=sys.stderr)
    os.unlink(script_path)
    return res.returncode==0

def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='Path to scene JSON')
    ap.add_argument('--out', required=True, help='Output directory')
    ap.add_argument('--inproc', action='store_true', help='Assume running inside Blender and use in-process API')
    args=ap.parse_args()
    scene=json.load(open(args.scene))
    if args.inproc:
        if not _HAVE_BPY:
            print('bpy not available', file=sys.stderr); return 1
        render_scene_frame_bpy_inproc(scene, args.out)
    else:
        ok=render_with_blender(scene, args.out)
        if not ok:
            print('External Blender render failed', file=sys.stderr); return 2
    print('Done.')
    return 0

if __name__=='__main__':
    sys.exit(cli())
