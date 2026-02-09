"""
3D Visualization utilities for llmscene.

This module provides 3D auxiliary visualization (bounding boxes, arrows, coordinate grids)
for rendered scenes, adapted from SceneWeaver's draw_bbox.py.

Features:
1. Bounding boxes with wireframe visualization
2. Direction arrows showing object orientation
3. Coordinate grid for spatial reference
4. 2-pass rendering with transparent overlay

Usage:
    # In Blender context:
    from visualization_3d import add_scene_visualization
    
    add_scene_visualization(scene_data, enable_bbox=True, enable_arrows=True, enable_coord=True)
"""

import os
from typing import Dict, List, Optional, Tuple, Any

# Check if running in Blender
try:
    import bpy
    import mathutils
    from mathutils import Matrix, Vector
    _HAVE_BPY = True
except ImportError:
    _HAVE_BPY = False
    # Define placeholder types for type hints when bpy is not available
    Matrix = None
    Vector = None


# ==================== Collection Management ====================

def link_to_collection(collection_name: str = "visualization", obj=None):
    """Link an object to a specific collection for organization."""
    if not _HAVE_BPY or obj is None:
        return
    
    # Create (or get) the visualization collection
    if collection_name in bpy.data.collections:
        viz_collection = bpy.data.collections[collection_name]
    else:
        viz_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(viz_collection)
    
    # Remove obj from other collections to avoid duplication
    for coll in obj.users_collection:
        coll.objects.unlink(obj)
    
    # Link obj to the visualization collection
    viz_collection.objects.link(obj)


# ==================== Bounding Box Visualization ====================

def create_wireframe_bbox(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    rotation_matrix: Optional[Matrix] = None,
    color: Tuple[float, float, float, float] = (0.0, 0.3, 1.0, 1.0),
    thickness: float = 0.02,
    name: str = "BBox"
):
    """
    Create a wireframe bounding box.
    
    Args:
        center: (x, y, z) center position in Blender coordinates
        size: (x, y, z) dimensions
        rotation_matrix: 4x4 rotation matrix (optional)
        color: RGBA color tuple
        thickness: wireframe thickness
        name: object name
    
    Returns:
        The created bounding box object
    """
    if not _HAVE_BPY:
        return None
    
    # Create a cube with unit size at origin
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    bbox_cube = bpy.context.active_object
    bbox_cube.name = f"BBox_{name}"
    
    # Build transformation matrix
    S = Matrix.Diagonal((*size, 1))
    T = Matrix.Translation(Vector(center))
    
    if rotation_matrix is not None:
        bbox_cube.matrix_world = T @ rotation_matrix @ S
    else:
        bbox_cube.matrix_world = T @ S
    
    # Add wireframe modifier
    mod = bbox_cube.modifiers.new("Wireframe", type="WIREFRAME")
    mod.thickness = thickness
    
    # Create material
    mat = bpy.data.materials.new(name=f"WireframeMat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Specular IOR Level"].default_value = 0
    
    bbox_cube.data.materials.append(mat)
    bbox_cube.display_type = "WIRE"
    bbox_cube.hide_render = False
    
    link_to_collection(obj=bbox_cube)
    
    return bbox_cube


def create_arrow(
    start: Tuple[float, float, float] = (0, 0, 0),
    direction: Tuple[float, float, float] = (0, 0, 1),
    shaft_length: float = 0.5,
    shaft_radius: float = 0.02,
    head_length: float = 0.1,
    head_radius: float = 0.05,
    color: Tuple[float, float, float, float] = (1, 0, 0, 1),
    name: str = "Arrow"
):
    """
    Create a 3D arrow.
    
    Args:
        start: Starting point
        direction: Direction vector
        shaft_length: Length of arrow shaft
        shaft_radius: Radius of arrow shaft
        head_length: Length of arrow head
        head_radius: Radius of arrow head base
        color: RGBA color
        name: Object name
    
    Returns:
        The created arrow object
    """
    if not _HAVE_BPY:
        return None
    
    # Normalize direction
    dir_vector = Vector(direction).normalized()
    
    # Calculate shaft end point
    shaft_end = Vector(start) + dir_vector * shaft_length
    
    # Create shaft (cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius,
        depth=shaft_length,
        location=(Vector(start) + shaft_end) / 2
    )
    shaft = bpy.context.object
    
    # Align shaft to direction
    shaft.rotation_mode = "QUATERNION"
    shaft.rotation_quaternion = dir_vector.to_track_quat("Z", "Y")
    
    # Create head (cone)
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_radius,
        depth=head_length,
        location=shaft_end + dir_vector * (head_length / 2)
    )
    head = bpy.context.object
    
    # Align head
    head.rotation_mode = "QUATERNION"
    head.rotation_quaternion = dir_vector.to_track_quat("Z", "Y")
    
    # Join shaft and head
    bpy.ops.object.select_all(action="DESELECT")
    shaft.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()
    arrow = bpy.context.object
    arrow.name = f"Arrow_{name}"
    
    # Create material
    mat = bpy.data.materials.new(name=f"ArrowMat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
    
    if arrow.data.materials:
        arrow.data.materials[0] = mat
    else:
        arrow.data.materials.append(mat)
    
    link_to_collection(obj=arrow)
    
    return arrow


def create_text_label(
    text: str,
    location: Tuple[float, float, float],
    scale: float = 0.1,
    color: Tuple[float, float, float, float] = (1, 1, 1, 1),
    with_background: bool = True,
    bg_color: Tuple[float, float, float, float] = (0.0, 0.3, 1.0, 1.0)
):
    """
    Create a 3D text label.
    
    Args:
        text: Text content
        location: (x, y, z) position
        scale: Text scale
        color: Text RGBA color
        with_background: Whether to add a background plane
        bg_color: Background RGBA color
    
    Returns:
        Tuple of (text_object, background_plane) or (text_object, None)
    """
    if not _HAVE_BPY:
        return None, None
    
    # Create text object
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.active_object
    text_obj.name = f"Label_{text[:20]}"
    text_obj.data.body = text
    text_obj.scale = (scale, scale, scale)
    
    # Create text material with enhanced visibility
    text_mat = bpy.data.materials.new(name=f"TextMat_{text[:10]}")
    text_mat.use_nodes = True
    bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        # For dark colors, use high emission to make them stand out
        if color[0] < 0.3 and color[1] < 0.3 and color[2] < 0.3:
            bsdf.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            bsdf.inputs["Emission Strength"].default_value = 3.0
        else:
            bsdf.inputs["Emission Color"].default_value = color[:3] + (1,)
            bsdf.inputs["Emission Strength"].default_value = 1.0
    text_obj.data.materials.append(text_mat)
    
    link_to_collection(obj=text_obj)
    
    bg_plane = None
    if with_background:
        bpy.context.view_layer.update()
        text_size = text_obj.dimensions.xy
        
        bpy.ops.mesh.primitive_plane_add(size=1)
        bg_plane = bpy.context.active_object
        bg_plane.name = f"LabelBG_{text[:10]}"
        
        padding = 0.02
        bg_plane.scale.x = (text_size.x + padding) / scale
        bg_plane.scale.y = (text_size.y + padding) / scale
        bg_plane.location.x = location[0] + text_size.x / 2
        bg_plane.location.y = location[1] + text_size.y / 2
        bg_plane.location.z = location[2] - 0.01
        
        # Background material
        bg_mat = bpy.data.materials.new(name=f"BGMat_{text[:10]}")
        bg_mat.use_nodes = True
        bg_bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
        if bg_bsdf:
            bg_bsdf.inputs["Base Color"].default_value = bg_color
            bg_bsdf.inputs["Roughness"].default_value = 1.0
        bg_plane.data.materials.append(bg_mat)
        
        bg_plane.parent = text_obj
        link_to_collection(obj=bg_plane)
    
    return text_obj, bg_plane


# ==================== Coordinate Grid ====================

def create_coordinate_grid(
    room_size: Tuple[float, float],
    z_level: float = 0.01,
    step: float = 1.0,
    circle_radius: float = 0.05,
    color: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
    show_labels: bool = True,
    label_scale: float = 0.16
):
    """
    Create a coordinate grid on the floor with coordinate labels.
    
    Args:
        room_size: (width, depth) of the room
        z_level: Height of the grid markers
        step: Grid spacing
        circle_radius: Radius of grid point markers
        color: RGBA color for markers
        show_labels: Whether to show coordinate text labels
        label_scale: Scale of the coordinate labels
    """
    if not _HAVE_BPY:
        return
    
    # Create material once
    mat = bpy.data.materials.new(name="GridMarkerMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Emission Color"].default_value = color[:3] + (1,)
        bsdf.inputs["Emission Strength"].default_value = 1.5
    
    width, depth = room_size
    
    for x in range(int(width / step) + 1):
        for y in range(int(depth / step) + 1):
            pos_x = x * step - width / 2
            pos_y = y * step - depth / 2
            
            # Create circle marker
            bpy.ops.mesh.primitive_circle_add(
                vertices=32,
                radius=circle_radius,
                fill_type="NGON",
                location=(pos_x, pos_y, z_level)
            )
            circle = bpy.context.active_object
            circle.name = f"GridPoint_{x}_{y}"
            
            if circle.data.materials:
                circle.data.materials[0] = mat
            else:
                circle.data.materials.append(mat)
            
            link_to_collection(obj=circle)
            
            # Add coordinate text label next to the marker
            if show_labels:
                # Display in SCENE coordinates to match JSON file
                # Blender position (pos_x, pos_y) maps to Scene (X, Z) as:
                # - pos_x = scene X (width)
                # - pos_y = -scene Z (depth is negated in conversion)
                # So: scene_z = -pos_y
                scene_x = pos_x
                scene_z = -pos_y
                coord_text = f"({scene_x:.1f}, {scene_z:.1f})"
                label_pos = (
                    pos_x + circle_radius * 2,  # Offset to the right
                    pos_y + circle_radius * 2,  # Offset forward
                    z_level + 0.02  # Slightly above the marker
                )
                create_text_label(
                    text=coord_text,
                    location=label_pos,
                    scale=label_scale,
                    color=(0.0, 0.0, 0.0, 1.0),  # Black text for readability
                    with_background=False
                )


def create_axis_arrows(origin: Tuple[float, float, float] = (0, 0, 0), length: float = 1.0):
    """
    Create RGB axis arrows at origin (X=red, Y=green, Z=blue).
    
    Args:
        origin: Origin point
        length: Arrow length
    """
    if not _HAVE_BPY:
        return
    
    # X axis (red)
    create_arrow(
        start=origin,
        direction=(1, 0, 0),
        shaft_length=length,
        color=(1, 0, 0, 1),
        name="X_Axis"
    )
    
    # Y axis (green)
    create_arrow(
        start=origin,
        direction=(0, 1, 0),
        shaft_length=length,
        color=(0, 1, 0, 1),
        name="Y_Axis"
    )
    
    # Z axis (blue)
    create_arrow(
        start=origin,
        direction=(0, 0, 1),
        shaft_length=length,
        color=(0, 0, 1, 1),
        name="Z_Axis"
    )


# ==================== Scene Visualization ====================

def add_object_bbox_and_arrow(
    obj_data: Dict,
    index: int,
    show_bbox: bool = True,
    show_arrow: bool = True,
    show_label: bool = True,
    bbox_color: Tuple[float, float, float, float] = (0.0, 0.3, 1.0, 1.0),
    arrow_color: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 1.0)
):
    """
    Add bounding box and direction arrow for a single object.
    
    Args:
        obj_data: Object data dict with pos, size, rot, desc
        index: Object index for naming
        show_bbox: Whether to show bounding box
        show_arrow: Whether to show direction arrow
        show_label: Whether to show text label
        bbox_color: Bounding box color
        arrow_color: Arrow color
    """
    if not _HAVE_BPY:
        return
    
    pos = obj_data.get('pos', [0, 0, 0])
    size = obj_data.get('size', [1, 1, 1])
    rot = obj_data.get('rot', [0, 0, 0, 1])
    desc = obj_data.get('desc', f'Object_{index}')
    
    # Convert to Blender coordinates: (x, y, z) -> (x, -z, y)
    blender_pos = (pos[0], -pos[2], pos[1] + size[1] / 2)  # Center height
    blender_size = (size[0], size[2], size[1])  # Swap y and z
    
    # Build rotation matrix
    # Quaternion [x, y, z, w] -> Blender adjusted
    q_orig = mathutils.Quaternion((rot[3], rot[0], rot[1], rot[2]))
    q_blender = mathutils.Quaternion((q_orig.w, q_orig.x, -q_orig.z, q_orig.y))
    R = q_blender.to_matrix().to_4x4()
    
    # Short name for label (first 30 chars)
    short_name = desc[:30] + "..." if len(desc) > 30 else desc
    
    if show_bbox:
        bbox = create_wireframe_bbox(
            center=blender_pos,
            size=blender_size,
            rotation_matrix=R,
            color=bbox_color,
            name=f"{index}_{short_name[:10]}"
        )
    
    if show_arrow:
        # Arrow starts at top center of bounding box
        arrow_start = (
            blender_pos[0],
            blender_pos[1],
            blender_pos[2] + blender_size[2] / 2 + 0.1
        )
        
        # Direction is local X axis in world space
        local_x = Vector((1, 0, 0))
        world_direction = R.to_3x3() @ local_x
        
        # Rotate 90 degrees clockwise in XZ plane (which is XY in Blender coords)
        # Clockwise 90° rotation: (x, y, z) -> (y, -x, z) in Blender coords
        rotated_direction = Vector((world_direction.y, -world_direction.x, world_direction.z))
        
        arrow = create_arrow(
            start=arrow_start,
            direction=tuple(rotated_direction),
            shaft_length=blender_size[0] * 0.5,
            color=arrow_color,
            name=f"{index}_{short_name[:10]}"
        )
    
    if show_label:
        # Label above the bounding box
        label_pos = (
            blender_pos[0] - len(short_name) * 0.03,
            blender_pos[1],
            blender_pos[2] + blender_size[2] / 2 + 0.2
        )
        create_text_label(
            text=short_name,
            location=label_pos,
            scale=0.08
        )


def add_scene_visualization(
    scene_data: Dict,
    enable_bbox: bool = True,
    enable_arrows: bool = True,
    enable_labels: bool = True,
    enable_coord: bool = False,
    enable_axes: bool = False
):
    """
    Add 3D visualization to a scene.
    
    Args:
        scene_data: Scene dict with objects/groups and bounds
        enable_bbox: Show bounding boxes around objects
        enable_arrows: Show direction arrows
        enable_labels: Show text labels
        enable_coord: Show coordinate grid
        enable_axes: Show world axis arrows
    """
    if not _HAVE_BPY:
        print("[visualization_3d] Warning: bpy not available")
        return
    
    # Extract objects
    objects_list = []
    if 'groups' in scene_data:
        for group in scene_data.get('groups', []):
            objects_list.extend(group.get('objects', []))
    elif 'objects' in scene_data:
        objects_list = scene_data.get('objects', [])
    
    # Add visualization for each object
    for i, obj_data in enumerate(objects_list):
        add_object_bbox_and_arrow(
            obj_data=obj_data,
            index=i,
            show_bbox=enable_bbox,
            show_arrow=enable_arrows,
            show_label=enable_labels
        )
    
    # Get room bounds
    bounds = None
    if 'room_envelope' in scene_data:
        bounds = scene_data['room_envelope'].get('bounds_bottom', [])
    else:
        bounds = scene_data.get('bounds_bottom', [])
    
    if enable_coord and bounds:
        xs = [p[0] for p in bounds]
        zs = [p[2] for p in bounds]
        width = max(xs) - min(xs)
        depth = max(zs) - min(zs)
        create_coordinate_grid(
            (width, depth),
            show_labels=True,  # Enable coordinate labels
            label_scale=0.16
        )
    
    if enable_axes:
        # Place axis arrows at room corner
        if bounds:
            origin = (min(xs), -max(zs), 0.1)
        else:
            origin = (0, 0, 0.1)
        create_axis_arrows(origin, length=0.5)


def set_visualization_visibility(visible: bool = True, collection_name: str = "visualization"):
    """
    Toggle visibility of visualization elements.
    
    Args:
        visible: Whether to show visualization
        collection_name: Name of the visualization collection
    """
    if not _HAVE_BPY:
        return
    
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        collection.hide_render = not visible
        collection.hide_viewport = not visible


# ==================== Two-Pass Rendering ====================

def render_with_visualization(
    scene_data: Dict,
    output_dir: str,
    view_type: str = 'top',
    enable_viz: bool = True
):
    """
    Render scene with optional visualization overlay.
    
    This uses a 2-pass approach:
    1. Render the scene normally
    2. Render visualization with transparent background
    3. Composite the two images
    
    Args:
        scene_data: Scene data dict
        output_dir: Output directory
        view_type: 'top' or 'diag'
        enable_viz: Whether to add visualization
    
    Returns:
        Path to the final rendered image
    """
    if not _HAVE_BPY:
        return None
    
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if enable_viz:
        # Add visualization elements
        add_scene_visualization(
            scene_data,
            enable_bbox=True,
            enable_arrows=True,
            enable_labels=False,  # Labels can be heavy
            enable_coord=False,
            enable_axes=True
        )
    
    # Render (uses existing Blender render setup)
    bpy.context.scene.render.filepath = str(output_path / f'{view_type}_with_viz.png')
    bpy.ops.render.render(write_still=True)
    
    return bpy.context.scene.render.filepath


if __name__ == "__main__":
    # Test code - only runs in Blender
    if _HAVE_BPY:
        print("[visualization_3d] Running test visualization...")
        
        # Create test scene
        test_scene = {
            "bounds_bottom": [[-2, 0, 2], [2, 0, 2], [2, 0, -2], [-2, 0, -2]],
            "objects": [
                {
                    "desc": "Modern desk",
                    "pos": [0, 0, 0],
                    "size": [1.2, 0.75, 0.6],
                    "rot": [0, 0, 0, 1]
                },
                {
                    "desc": "Office chair",
                    "pos": [0.8, 0, 0],
                    "size": [0.6, 1.0, 0.6],
                    "rot": [0, 0.707, 0, 0.707]
                }
            ]
        }
        
        add_scene_visualization(
            test_scene,
            enable_bbox=True,
            enable_arrows=True,
            enable_labels=True,
            enable_axes=True
        )
        
        print("[visualization_3d] Test visualization complete!")
