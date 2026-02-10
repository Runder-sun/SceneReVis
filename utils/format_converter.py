"""
Scene format conversion tool.
Supports conversion between ungrouped format and grouped format.
"""

import json
import copy
from typing import Dict, Any


def convert_flat_to_grouped(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ungrouped scene format to grouped format.
    
    Args:
        scene: Ungrouped scene data containing a direct objects array
        
    Returns:
        Grouped scene data
    """
    # If groups field already exists, return as is
    if 'groups' in scene:
        return scene
    
    # If no objects field, return as is
    if 'objects' not in scene:
        return scene
    
    # Create new scene data, preserving original room info
    grouped_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # Handle room_envelope or bounds fields
    if 'room_envelope' in scene:
        grouped_scene['room_envelope'] = scene['room_envelope']
    elif 'bounds_top' in scene and 'bounds_bottom' in scene:
        # Convert old format bounds to room_envelope
        grouped_scene['room_envelope'] = {
            'bounds_top': scene['bounds_top'],
            'bounds_bottom': scene['bounds_bottom']
        }
    
    # Put all objects into a default group
    grouped_scene['groups'] = [
        {
            'group_name': 'main_group',
            'group_type': 'functional_area',
            'description': 'Main functional area containing all objects',
            'objects': scene['objects']
        }
    ]
    
    print(f"Converted flat format to grouped format: {len(scene['objects'])} objects → 1 group")
    return grouped_scene


def convert_grouped_to_flat(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Convert grouped scene format back to ungrouped format.
    
    Args:
        scene: Grouped scene data
        
    Returns:
        Ungrouped scene data containing a direct objects array
    """
    # If no groups field, return as is
    if 'groups' not in scene:
        return scene
    
    # Create new scene data, preserving original room info
    flat_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # Handle room_envelope or bounds fields
    if 'room_envelope' in scene:
        # Extract bounds to top level
        flat_scene['bounds_top'] = scene['room_envelope'].get('bounds_top', [])
        flat_scene['bounds_bottom'] = scene['room_envelope'].get('bounds_bottom', [])
    
    # Collect all objects from all groups into a single array
    all_objects = []
    for group in scene.get('groups', []):
        all_objects.extend(group.get('objects', []))
    
    flat_scene['objects'] = all_objects
    
    print(f"Converted grouped format to flat format: {len(scene.get('groups', []))} groups → {len(all_objects)} objects")
    return flat_scene


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Scene Format Conversion Tool Test")
    print("="*60)
    
    # Test 1: Flat → Grouped
    print("\nTest 1: Flat → Grouped")
    print("-"*60)
    
    flat_scene = {
        "bounds_top": [[-1, 2.8, 1], [1, 2.8, 1], [1, 2.8, -1], [-1, 2.8, -1]],
        "bounds_bottom": [[-1, 0.0, 1], [1, 0.0, 1], [1, 0.0, -1], [-1, 0.0, -1]],
        "room_type": "bedroom",
        "room_id": "test_room",
        "objects": [
            {"desc": "chair", "size": [0.6, 0.8, 0.6], "pos": [1.0, 0.0, 1.0], 
             "rot": [0, 0, 0, 1], "jid": "chair_001"}
        ]
    }
    
    print("Original Flat format:")
    print(json.dumps(flat_scene, indent=2))
    
    grouped = convert_flat_to_grouped(flat_scene)
    print("\nConverted to Grouped format:")
    print(json.dumps(grouped, indent=2))
    
    # Test 2: Grouped → Flat
    print("\n\nTest 2: Grouped → Flat")
    print("-"*60)
    
    flat_back = convert_grouped_to_flat(grouped)
    print("Converted back to Flat format:")
    print(json.dumps(flat_back, indent=2))
    
    # Verify
    assert 'objects' in flat_back
    assert len(flat_back['objects']) == 1
    print("\n✓ All tests passed!")
