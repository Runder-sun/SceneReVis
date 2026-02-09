"""
场景格式转换工具
支持不带groups的格式和带groups的格式之间的相互转换
"""

import json
import copy
from typing import Dict, Any


def convert_flat_to_grouped(scene: Dict[str, Any]) -> Dict[str, Any]:
    """将不带groups的场景格式转换为带groups的格式
    
    Args:
        scene: 不带groups的场景数据，包含直接的objects数组
        
    Returns:
        带groups的场景数据
    """
    # 如果已经有groups字段，直接返回
    if 'groups' in scene:
        return scene
    
    # 如果没有objects字段，也直接返回
    if 'objects' not in scene:
        return scene
    
    # 创建新的场景数据，保留原有的room信息
    grouped_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # 处理room_envelope或bounds字段
    if 'room_envelope' in scene:
        grouped_scene['room_envelope'] = scene['room_envelope']
    elif 'bounds_top' in scene and 'bounds_bottom' in scene:
        # 将旧格式的bounds转换为room_envelope
        grouped_scene['room_envelope'] = {
            'bounds_top': scene['bounds_top'],
            'bounds_bottom': scene['bounds_bottom']
        }
    
    # 将所有objects放入一个默认组
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
    """将带groups的场景格式转换回不带groups的格式
    
    Args:
        scene: 带groups的场景数据
        
    Returns:
        不带groups的场景数据，包含直接的objects数组
    """
    # 如果没有groups字段，直接返回
    if 'groups' not in scene:
        return scene
    
    # 创建新的场景数据，保留原有的room信息
    flat_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # 处理room_envelope或bounds字段
    if 'room_envelope' in scene:
        # 提取bounds到顶层
        flat_scene['bounds_top'] = scene['room_envelope'].get('bounds_top', [])
        flat_scene['bounds_bottom'] = scene['room_envelope'].get('bounds_bottom', [])
    
    # 收集所有组中的objects到一个数组
    all_objects = []
    for group in scene.get('groups', []):
        all_objects.extend(group.get('objects', []))
    
    flat_scene['objects'] = all_objects
    
    print(f"Converted grouped format to flat format: {len(scene.get('groups', []))} groups → {len(all_objects)} objects")
    return flat_scene


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("场景格式转换工具测试")
    print("="*60)
    
    # 测试1: Flat → Grouped
    print("\n测试1: Flat → Grouped")
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
    
    print("原始Flat格式:")
    print(json.dumps(flat_scene, indent=2))
    
    grouped = convert_flat_to_grouped(flat_scene)
    print("\n转换后的Grouped格式:")
    print(json.dumps(grouped, indent=2))
    
    # 测试2: Grouped → Flat
    print("\n\n测试2: Grouped → Flat")
    print("-"*60)
    
    flat_back = convert_grouped_to_flat(grouped)
    print("转换回Flat格式:")
    print(json.dumps(flat_back, indent=2))
    
    # 验证
    assert 'objects' in flat_back
    assert len(flat_back['objects']) == 1
    print("\n✓ 所有测试通过!")
