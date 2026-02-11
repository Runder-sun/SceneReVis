"""
Scene Editor Module
Contains 6 core functions for automatically modifying scenes based on tool_calls.
Supported operations: add_object, remove_object, move_object, rotate_object, scale_object, replace_object
"""

import json
import uuid
import copy
from typing import List, Dict, Any, Optional


def _get_object_id(obj: Dict[str, Any]) -> Optional[str]:
    """
    Get the object identifier (jid first, then uid).
    
    Args:
        obj: Object data dictionary
        
    Returns:
        Object identifier, or None if neither exists
    """
    return obj.get('jid') or obj.get('uid')


def _match_object_id(obj: Dict[str, Any], target_id: str) -> bool:
    """
    Check if an object matches the target ID (supports both jid and uid).
    
    Args:
        obj: Object data dictionary
        target_id: Target ID to match (can be jid or uid)
        
    Returns:
        Whether it matches
    """
    if not target_id:
        return False
    return obj.get('jid') == target_id or obj.get('uid') == target_id


def add_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add a new object to the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - object_description: Object description
            - position: [x, y, z] position coordinates
            - rotation: [x, y, z, w] quaternion rotation
            - size: [width, height, depth] dimensions
            - group_name: Group name (only used in groups format)
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # Extract parameters
    object_description = arguments.get('object_description', 'New furniture piece')
    position = arguments.get('position', [0, 0, 0])
    rotation = arguments.get('rotation', [0, 0, 0, 1])
    size = arguments.get('size', [1, 1, 1])
    group_name = arguments.get('group_name', 'default_group')
    
    # Create new object - do not set jid, let the asset retrieval module handle it
    new_object = {
        "desc": object_description,
        "size": size,
        "pos": position,
        "rot": rotation,
    }
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: find or create group
        group_found = False
        for group in modified_scene.get('groups', []):
            if group.get('group_name') == group_name:
                group['objects'].append(new_object)
                group_found = True
                break
        
        # If group not found, create a new one
        if not group_found:
            new_group = {
                "group_name": group_name,
                "group_type": "functional_area",
                "description": f"Functional area containing {group_name.lower()} elements.",
                "objects": [new_object]
            }
            if 'groups' not in modified_scene:
                modified_scene['groups'] = []
            modified_scene['groups'].append(new_group)
    else:
        # Flat format: add directly to objects array
        if 'objects' not in modified_scene:
            modified_scene['objects'] = []
        modified_scene['objects'].append(new_object)
    
    return modified_scene


def remove_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove an object from the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - jid: ID of the object to remove (supports jid or uid)
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # The jid parameter accepts either jid or uid
    id_to_remove = arguments.get('jid', '')
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: iterate all groups and remove the specified object
        for group in modified_scene.get('groups', []):
            original_count = len(group.get('objects', []))
            group['objects'] = [obj for obj in group.get('objects', []) if not _match_object_id(obj, id_to_remove)]
            if len(group['objects']) < original_count:
                print(f"Removed object {id_to_remove} from group '{group.get('group_name', 'unknown')}'")
    else:
        # Flat format: remove directly from objects array
        if 'objects' in modified_scene:
            original_count = len(modified_scene['objects'])
            modified_scene['objects'] = [obj for obj in modified_scene['objects'] if not _match_object_id(obj, id_to_remove)]
            if len(modified_scene['objects']) < original_count:
                print(f"Removed object {id_to_remove}")
    
    return modified_scene


def move_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Move an object in the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - jid: ID of the object to move (supports jid or uid)
            - new_position: [x, y, z] new position coordinates
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # The jid parameter accepts either jid or uid
    target_id = arguments.get('jid', '')
    new_position = arguments.get('new_position', [0, 0, 0])
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: find and move the object
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['pos'] = new_position
                    print(f"Moved object {target_id} to position {new_position}")
                    return modified_scene
    else:
        # Flat format: search directly in the objects array
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['pos'] = new_position
                    print(f"Moved object {target_id} to position {new_position}")
                    return modified_scene
    
    print(f"Warning: object with ID {target_id} not found")
    return modified_scene


def rotate_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rotate an object in the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - jid: ID of the object to rotate (supports jid or uid)
            - new_rotation: [x, y, z, w] new quaternion rotation
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # The jid parameter accepts either jid or uid
    target_id = arguments.get('jid', '')
    new_rotation = arguments.get('new_rotation', [0, 0, 0, 1])
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: find and rotate the object
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['rot'] = new_rotation
                    print(f"Rotated object {target_id} to {new_rotation}")
                    return modified_scene
    else:
        # Flat format: search directly in the objects array
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['rot'] = new_rotation
                    print(f"Rotated object {target_id} to {new_rotation}")
                    return modified_scene
    
    print(f"Warning: object with ID {target_id} not found")
    return modified_scene


def scale_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scale an object in the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - jid: ID of the object to scale (supports jid or uid)
            - new_size: [width, height, depth] new dimensions
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # The jid parameter accepts either jid or uid
    target_id = arguments.get('jid', '')
    new_size = arguments.get('new_size', [1, 1, 1])
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: find and scale the object
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['size'] = new_size
                    print(f"Scaled object {target_id} to size {new_size}")
                    return modified_scene
    else:
        # Flat format: search directly in the objects array
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['size'] = new_size
                    print(f"Scaled object {target_id} to size {new_size}")
                    return modified_scene
    
    print(f"Warning: object with ID {target_id} not found")
    return modified_scene


def replace_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace an object in the scene.
    
    Args:
        scene: Scene data (supports both with-groups and without-groups formats)
        arguments: Tool parameters
            - jid_to_replace: ID of the object to replace (supports jid or uid)
            - new_object_description: Description of the new object
    
    Returns:
        Modified scene data
    """
    modified_scene = copy.deepcopy(scene)
    
    # The jid_to_replace parameter accepts either jid or uid
    id_to_replace = arguments.get('jid_to_replace', '')
    new_object_desc = arguments.get('new_object_description', 'Replacement object')
    
    # Check if it's groups format or flat format
    if 'groups' in modified_scene:
        # Groups format: find and replace the object
        for group in modified_scene.get('groups', []):
            for i, obj in enumerate(group.get('objects', [])):
                if _match_object_id(obj, id_to_replace):
                    # Keep position, size, and type, but change description - do not set jid/uid, let the asset retrieval module handle it
                    group['objects'][i] = {
                        "desc": new_object_desc,
                        "size": obj.get('size', [1, 1, 1]),
                        "pos": obj.get('pos', [0, 0, 0]),
                        "rot": obj.get('rot', [0, 0, 0, 1])
                    }
                    print(f"Replaced object {id_to_replace} with {new_object_desc}")
                    return modified_scene
    else:
        # Flat format: search directly in the objects array
        if 'objects' in modified_scene:
            for i, obj in enumerate(modified_scene['objects']):
                if _match_object_id(obj, id_to_replace):
                    # Keep position, size, and type, but change description - do not set jid/uid, let the asset retrieval module handle it
                    modified_scene['objects'][i] = {
                        "desc": new_object_desc,
                        "size": obj.get('size', [1, 1, 1]),
                        "pos": obj.get('pos', [0, 0, 0]),
                        "rot": obj.get('rot', [0, 0, 0, 1])
                    }
                    print(f"Replaced object {id_to_replace} with {new_object_desc}")
                    return modified_scene
    
    print(f"Warning: object with ID {id_to_replace} not found")
    return modified_scene


def apply_tool_calls(initial_scene: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Automatically match and apply tool_calls to the scene, transforming the initial scene into the final scene.
    
    Args:
        initial_scene: Initial scene data
        tool_calls: List of tool calls, each containing name and arguments
    
    Returns:
        Final scene after applying all operations
    """
    # Tool function mapping
    tool_functions = {
        'add_object': add_object,
        'remove_object': remove_object,
        'move_object': move_object,
        'rotate_object': rotate_object,
        'scale_object': scale_object,
        'replace_object': replace_object
    }
    
    current_scene = copy.deepcopy(initial_scene)
    
    print(f"Starting to apply {len(tool_calls)} tool operations...")
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get('name', '')
        arguments = tool_call.get('arguments', {})
        
        if tool_name == 'terminate':
            print(f"Tool {i+1}: {tool_name} - operation terminated")
            break
        
        if tool_name in tool_functions:
            print(f"Tool {i+1}: {tool_name}")
            current_scene = tool_functions[tool_name](current_scene, arguments)
        else:
            print(f"Warning: unknown tool name '{tool_name}'")
    
    print("All tool operations applied successfully!")
    return current_scene


def validate_scene_integrity(scene: Dict[str, Any]) -> bool:
    """
    Validate scene data integrity.
    
    Args:
        scene: Scene data
    
    Returns:
        Whether validation passed
    """
    if not isinstance(scene, dict):
        print("Error: scene data is not a dictionary")
        return False
    
    if 'groups' not in scene:
        print("Warning: scene does not contain a 'groups' field")
        return True
    
    groups = scene['groups']
    if not isinstance(groups, list):
        print("Error: 'groups' field is not a list")
        return False
    
    total_objects = 0
    for group in groups:
        if 'objects' in group and isinstance(group['objects'], list):
            total_objects += len(group['objects'])
    
    print(f"Scene validation passed: {len(groups)} groups, {total_objects} objects in total")
    return True


# Example usage function
def example_usage():
    """
    Example: how to use the scene editor
    """
    # Example initial scene - using original format
    initial_scene = {
        "room_type": "bedroom",
        "room_id": "example_room",
        "room_envelope": {
            "bounds_top": [[-2, 3, 2], [2, 3, 2], [2, 3, -2], [-2, 3, -2]],
            "bounds_bottom": [[-2, 0, 2], [2, 0, 2], [2, 0, -2], [-2, 0, -2]]
        },
        "groups": [
            {
                "group_name": "living_area",
                "group_type": "functional_area",
                "description": "Main living area with seating",
                "objects": [
                    {
                        "desc": "wooden chair",
                        "size": [0.6, 0.8, 0.6],
                        "pos": [1.0, 0.0, 1.0],
                        "rot": [0, 0, 0, 1],
                        "jid": "chair_001"
                    }
                ]
            }
        ]
    }
    
    # Example tool calls
    tool_calls = [
        {
            "id": "tool_1",
            "name": "add_object",
            "arguments": {
                "object_description": "coffee table",
                "position": [2.0, 0.0, 2.0],
                "rotation": [0, 0, 0, 1],
                "size": [1.2, 0.4, 0.8],
                "group_name": "living_area"
            }
        },
        {
            "id": "tool_2",
            "name": "move_object",
            "arguments": {
                "jid": "chair_001",
                "new_position": [0.5, 0.0, 0.5]
            }
        }
    ]
    
    print("=== Scene Editor Example ===")
    print("\nInitial scene:")
    print(json.dumps(initial_scene, indent=2, ensure_ascii=False))
    
    print(f"\nTool calls:")
    for tool_call in tool_calls:
        print(f"- {tool_call['name']}: {tool_call['arguments']}")
    
    # Apply tool calls
    final_scene = apply_tool_calls(initial_scene, tool_calls)
    
    print(f"\nFinal scene:")
    print(json.dumps(final_scene, indent=2, ensure_ascii=False))
    
    # Validate scene integrity
    validate_scene_integrity(final_scene)


if __name__ == "__main__":
    example_usage()
