"""
场景编辑器模块
包含6个核心函数，用于根据tool_calls自动修改场景
支持的操作：add_object, remove_object, move_object, rotate_object, scale_object, replace_object
"""

import json
import uuid
import copy
from typing import List, Dict, Any, Optional


def _get_object_id(obj: Dict[str, Any]) -> Optional[str]:
    """
    获取物体的标识符（优先 jid，其次 uid）
    
    Args:
        obj: 物体数据字典
        
    Returns:
        物体标识符，如果都不存在则返回 None
    """
    return obj.get('jid') or obj.get('uid')


def _match_object_id(obj: Dict[str, Any], target_id: str) -> bool:
    """
    检查物体是否匹配目标ID（支持 jid 和 uid）
    
    Args:
        obj: 物体数据字典
        target_id: 要匹配的目标ID（可以是 jid 或 uid）
        
    Returns:
        是否匹配
    """
    if not target_id:
        return False
    return obj.get('jid') == target_id or obj.get('uid') == target_id


def add_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    向场景中添加新物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - object_description: 物体描述
            - position: [x, y, z] 位置坐标
            - rotation: [x, y, z, w] 四元数旋转
            - size: [width, height, depth] 尺寸
            - group_name: 所属组名（仅在groups格式中使用）
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 提取参数
    object_description = arguments.get('object_description', 'New furniture piece')
    position = arguments.get('position', [0, 0, 0])
    rotation = arguments.get('rotation', [0, 0, 0, 1])
    size = arguments.get('size', [1, 1, 1])
    group_name = arguments.get('group_name', 'default_group')
    
    # 创建新物体 - 不设置jid，让资产检索模块处理
    new_object = {
        "desc": object_description,
        "size": size,
        "pos": position,
        "rot": rotation,
    }
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：查找或创建组
        group_found = False
        for group in modified_scene.get('groups', []):
            if group.get('group_name') == group_name:
                group['objects'].append(new_object)
                group_found = True
                break
        
        # 如果没有找到组，创建新组
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
        # Flat格式：直接添加到objects数组
        if 'objects' not in modified_scene:
            modified_scene['objects'] = []
        modified_scene['objects'].append(new_object)
    
    return modified_scene


def remove_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    从场景中移除物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - jid: 要移除的物体ID（支持 jid 或 uid）
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 支持 jid 参数传入 jid 或 uid
    id_to_remove = arguments.get('jid', '')
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：遍历所有组，移除指定的物体
        for group in modified_scene.get('groups', []):
            original_count = len(group.get('objects', []))
            group['objects'] = [obj for obj in group.get('objects', []) if not _match_object_id(obj, id_to_remove)]
            if len(group['objects']) < original_count:
                print(f"已从组 '{group.get('group_name', 'unknown')}' 中移除物体 {id_to_remove}")
    else:
        # Flat格式：直接从objects数组中移除
        if 'objects' in modified_scene:
            original_count = len(modified_scene['objects'])
            modified_scene['objects'] = [obj for obj in modified_scene['objects'] if not _match_object_id(obj, id_to_remove)]
            if len(modified_scene['objects']) < original_count:
                print(f"已移除物体 {id_to_remove}")
    
    return modified_scene


def move_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    移动场景中的物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - jid: 要移动的物体ID（支持 jid 或 uid）
            - new_position: [x, y, z] 新位置坐标
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 支持 jid 参数传入 jid 或 uid
    target_id = arguments.get('jid', '')
    new_position = arguments.get('new_position', [0, 0, 0])
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：查找并移动物体
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['pos'] = new_position
                    print(f"已将物体 {target_id} 移动到位置 {new_position}")
                    return modified_scene
    else:
        # Flat格式：直接在objects数组中查找
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['pos'] = new_position
                    print(f"已将物体 {target_id} 移动到位置 {new_position}")
                    return modified_scene
    
    print(f"警告: 未找到ID为 {target_id} 的物体")
    return modified_scene


def rotate_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    旋转场景中的物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - jid: 要旋转的物体ID（支持 jid 或 uid）
            - new_rotation: [x, y, z, w] 新的四元数旋转
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 支持 jid 参数传入 jid 或 uid
    target_id = arguments.get('jid', '')
    new_rotation = arguments.get('new_rotation', [0, 0, 0, 1])
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：查找并旋转物体
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['rot'] = new_rotation
                    print(f"已将物体 {target_id} 旋转到 {new_rotation}")
                    return modified_scene
    else:
        # Flat格式：直接在objects数组中查找
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['rot'] = new_rotation
                    print(f"已将物体 {target_id} 旋转到 {new_rotation}")
                    return modified_scene
    
    print(f"警告: 未找到ID为 {target_id} 的物体")
    return modified_scene


def scale_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    缩放场景中的物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - jid: 要缩放的物体ID（支持 jid 或 uid）
            - new_size: [width, height, depth] 新尺寸
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 支持 jid 参数传入 jid 或 uid
    target_id = arguments.get('jid', '')
    new_size = arguments.get('new_size', [1, 1, 1])
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：查找并缩放物体
        for group in modified_scene.get('groups', []):
            for obj in group.get('objects', []):
                if _match_object_id(obj, target_id):
                    obj['size'] = new_size
                    print(f"已将物体 {target_id} 缩放到尺寸 {new_size}")
                    return modified_scene
    else:
        # Flat格式：直接在objects数组中查找
        if 'objects' in modified_scene:
            for obj in modified_scene['objects']:
                if _match_object_id(obj, target_id):
                    obj['size'] = new_size
                    print(f"已将物体 {target_id} 缩放到尺寸 {new_size}")
                    return modified_scene
    
    print(f"警告: 未找到ID为 {target_id} 的物体")
    return modified_scene


def replace_object(scene: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    替换场景中的物体
    
    Args:
        scene: 场景数据（支持带groups或不带groups的格式）
        arguments: 工具参数
            - jid_to_replace: 要替换的物体ID（支持 jid 或 uid）
            - new_object_description: 新物体的描述
    
    Returns:
        修改后的场景数据
    """
    modified_scene = copy.deepcopy(scene)
    
    # 支持 jid_to_replace 参数传入 jid 或 uid
    id_to_replace = arguments.get('jid_to_replace', '')
    new_object_desc = arguments.get('new_object_description', 'Replacement object')
    
    # 检查是groups格式还是flat格式
    if 'groups' in modified_scene:
        # Groups格式：查找并替换物体
        for group in modified_scene.get('groups', []):
            for i, obj in enumerate(group.get('objects', [])):
                if _match_object_id(obj, id_to_replace):
                    # 保持位置、大小和类型，但更改描述 - 不设置jid/uid，让资产检索模块处理
                    group['objects'][i] = {
                        "desc": new_object_desc,
                        "size": obj.get('size', [1, 1, 1]),
                        "pos": obj.get('pos', [0, 0, 0]),
                        "rot": obj.get('rot', [0, 0, 0, 1])
                    }
                    print(f"已将物体 {id_to_replace} 替换为 {new_object_desc}")
                    return modified_scene
    else:
        # Flat格式：直接在objects数组中查找
        if 'objects' in modified_scene:
            for i, obj in enumerate(modified_scene['objects']):
                if _match_object_id(obj, id_to_replace):
                    # 保持位置、大小和类型，但更改描述 - 不设置jid/uid，让资产检索模块处理
                    modified_scene['objects'][i] = {
                        "desc": new_object_desc,
                        "size": obj.get('size', [1, 1, 1]),
                        "pos": obj.get('pos', [0, 0, 0]),
                        "rot": obj.get('rot', [0, 0, 0, 1])
                    }
                    print(f"已将物体 {id_to_replace} 替换为 {new_object_desc}")
                    return modified_scene
    
    print(f"警告: 未找到ID为 {id_to_replace} 的物体")
    return modified_scene


def apply_tool_calls(initial_scene: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自动匹配并应用tool_calls到场景，将初始场景转换为最终场景
    
    Args:
        initial_scene: 初始场景数据
        tool_calls: 工具调用列表，每个调用包含name和arguments
    
    Returns:
        应用所有操作后的最终场景
    """
    # 工具函数映射
    tool_functions = {
        'add_object': add_object,
        'remove_object': remove_object,
        'move_object': move_object,
        'rotate_object': rotate_object,
        'scale_object': scale_object,
        'replace_object': replace_object
    }
    
    current_scene = copy.deepcopy(initial_scene)
    
    print(f"开始应用 {len(tool_calls)} 个工具操作...")
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get('name', '')
        arguments = tool_call.get('arguments', {})
        
        if tool_name == 'terminate':
            print(f"工具 {i+1}: {tool_name} - 操作结束")
            break
        
        if tool_name in tool_functions:
            print(f"工具 {i+1}: {tool_name}")
            current_scene = tool_functions[tool_name](current_scene, arguments)
        else:
            print(f"警告: 未知的工具名称 '{tool_name}'")
    
    print("所有工具操作应用完成!")
    return current_scene


def validate_scene_integrity(scene: Dict[str, Any]) -> bool:
    """
    验证场景数据完整性
    
    Args:
        scene: 场景数据
    
    Returns:
        验证是否通过
    """
    if not isinstance(scene, dict):
        print("错误: 场景数据不是字典类型")
        return False
    
    if 'groups' not in scene:
        print("警告: 场景中没有groups字段")
        return True
    
    groups = scene['groups']
    if not isinstance(groups, list):
        print("错误: groups字段不是列表类型")
        return False
    
    total_objects = 0
    for group in groups:
        if 'objects' in group and isinstance(group['objects'], list):
            total_objects += len(group['objects'])
    
    print(f"场景验证通过: 共有 {len(groups)} 个组，{total_objects} 个物体")
    return True


# 示例使用函数
def example_usage():
    """
    示例：如何使用场景编辑器
    """
    # 示例初始场景 - 使用原始格式
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
    
    # 示例工具调用
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
    
    print("=== 场景编辑器示例 ===")
    print("\n初始场景:")
    print(json.dumps(initial_scene, indent=2, ensure_ascii=False))
    
    print(f"\n工具调用:")
    for tool_call in tool_calls:
        print(f"- {tool_call['name']}: {tool_call['arguments']}")
    
    # 应用工具调用
    final_scene = apply_tool_calls(initial_scene, tool_calls)
    
    print(f"\n最终场景:")
    print(json.dumps(final_scene, indent=2, ensure_ascii=False))
    
    # 验证场景完整性
    validate_scene_integrity(final_scene)


if __name__ == "__main__":
    example_usage()
