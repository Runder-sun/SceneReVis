#!/usr/bin/env python3
"""
场景格式适配器：将新的分组格式转换为原始的objects列表格式
"""

import json
import uuid
from typing import Dict, List, Any
from pathlib import Path

class SceneFormatAdapter:
    """场景格式适配器"""
    
    def __init__(self):
        pass
    
    def convert_grouped_to_objects_format(self, grouped_scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        将分组格式的场景转换为objects列表格式
        
        Args:
            grouped_scene: 分组格式的场景数据
            
        Returns:
            转换后的场景数据（objects列表格式）
        """
        # 基础场景信息
        converted_scene = {
            "room_type": grouped_scene.get("room_type", "unknown"),
            "room_id": grouped_scene.get("room_id", "unknown"),
        }
        
        # 处理房间边界
        if "room_envelope" in grouped_scene:
            envelope = grouped_scene["room_envelope"]
            converted_scene["bounds_top"] = envelope.get("bounds_top", [])
            converted_scene["bounds_bottom"] = envelope.get("bounds_bottom", [])
        elif "bounds_top" in grouped_scene and "bounds_bottom" in grouped_scene:
            # 兼容原有格式
            converted_scene["bounds_top"] = grouped_scene["bounds_top"]
            converted_scene["bounds_bottom"] = grouped_scene["bounds_bottom"]
        
        # 将分组中的objects提取到统一列表
        objects = []
        if "groups" in grouped_scene:
            for group in grouped_scene["groups"]:
                if "objects" in group:
                    for obj in group["objects"]:
                        # 确保必要字段存在
                        converted_obj = {
                            "desc": obj.get("desc", "Unknown object"),
                            "size": obj.get("size", [1.0, 1.0, 1.0]),
                            "pos": obj.get("pos", [0.0, 0.0, 0.0]),
                            "rot": obj.get("rot", [0, 0, 0, 1]),
                            "jid": obj.get("jid", str(uuid.uuid4()))
                        }
                        
                        # 添加可选字段
                        if "type" in obj:
                            converted_obj["type"] = obj["type"]
                        if "sampled_asset_jid" in obj:
                            converted_obj["sampled_asset_jid"] = obj["sampled_asset_jid"]
                            
                        objects.append(converted_obj)
        elif "objects" in grouped_scene:
            # 兼容原有格式
            objects = grouped_scene["objects"]
        
        converted_scene["objects"] = objects
        return converted_scene
    
    def convert_objects_to_grouped_format(self, objects_scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        将objects列表格式转换为分组格式（用于输出新格式）
        
        Args:
            objects_scene: objects列表格式的场景数据
            
        Returns:
            转换后的场景数据（分组格式）
        """
        # 基础场景信息
        converted_scene = {
            "room_type": objects_scene.get("room_type", "unknown"),
            "room_id": objects_scene.get("room_id", "unknown"),
        }
        
        # 处理房间边界
        if "bounds_top" in objects_scene and "bounds_bottom" in objects_scene:
            converted_scene["room_envelope"] = {
                "bounds_top": objects_scene["bounds_top"],
                "bounds_bottom": objects_scene["bounds_bottom"]
            }
        
        # 将objects按功能分组（简单版本，实际可以更智能）
        groups = []
        if "objects" in objects_scene:
            # 这里简化处理，将所有objects放在一个组中
            # 实际应用中可以根据物体类型、位置等进行智能分组
            main_group = {
                "group_name": "Main Area",
                "group_type": "mixed",
                "description": "Main furniture and objects",
                "objects": []
            }
            
            for obj in objects_scene["objects"]:
                converted_obj = {
                    "desc": obj.get("desc", "Unknown object"),
                    "size": obj.get("size", [1.0, 1.0, 1.0]),
                    "pos": obj.get("pos", [0.0, 0.0, 0.0]),
                    "rot": obj.get("rot", [0, 0, 0, 1]),
                    "jid": obj.get("jid", str(uuid.uuid4())),
                    "type": obj.get("type", "major")
                }
                main_group["objects"].append(converted_obj)
            
            if main_group["objects"]:
                groups.append(main_group)
        
        converted_scene["groups"] = groups
        return converted_scene

def load_and_convert_scene(scene_path: str | Path, target_format: str = "objects") -> Dict[str, Any]:
    """
    加载场景文件并转换为指定格式
    
    Args:
        scene_path: 场景文件路径
        target_format: 目标格式，"objects" 或 "groups"
        
    Returns:
        转换后的场景数据
    """
    adapter = SceneFormatAdapter()
    
    with open(scene_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    # 检测当前格式
    if "groups" in scene_data:
        current_format = "groups"
    elif "objects" in scene_data:
        current_format = "objects"
    else:
        raise ValueError(f"Unknown scene format in {scene_path}")
    
    if target_format == "objects" and current_format == "groups":
        return adapter.convert_grouped_to_objects_format(scene_data)
    elif target_format == "groups" and current_format == "objects":
        return adapter.convert_objects_to_grouped_format(scene_data)
    else:
        # 格式相同，直接返回
        return scene_data

def main():
    """测试函数"""
    # 测试转换
    test_scene = {
        "room_type": "bedroom",
        "room_id": "test-bedroom",
        "room_envelope": {
            "bounds_top": [[-3, 3, 3], [3, 3, 3], [3, 3, -3], [-3, 3, -3]],
            "bounds_bottom": [[-3, 0, 3], [3, 0, 3], [3, 0, -3], [-3, 0, -3]]
        },
        "groups": [
            {
                "group_name": "Sleeping Area",
                "group_type": "sleeping",
                "description": "Bed and nightstands",
                "objects": [
                    {
                        "desc": "Modern bed",
                        "size": [2.0, 1.0, 1.5],
                        "pos": [0.0, 0.0, 0.0],
                        "rot": [0, 0, 0, 1],
                        "jid": "test-bed-123",
                        "type": "major"
                    }
                ]
            }
        ]
    }
    
    adapter = SceneFormatAdapter()
    converted = adapter.convert_grouped_to_objects_format(test_scene)
    print("Converted scene:")
    print(json.dumps(converted, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
