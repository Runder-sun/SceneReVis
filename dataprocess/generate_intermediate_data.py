"""
第一步：从最终场景逆推生成原始场景，记录操作工具，生成对话模板数据
"""

import json
import os
import random
import uuid
import copy
import math
from typing import List, Dict, Any, Tuple
from pathlib import Path
from openai import AzureOpenAI
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from config import *

def setup_azure_client() -> AzureOpenAI:
    """Create AzureOpenAI client using Azure CLI or managed identity tokens."""
    scope = AZURE_OPENAI_SCOPE
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=credential,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    return client

# 设置Azure OpenAI客户端
client = setup_azure_client()

def normalize_scene_format(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    将场景标准化为带groups的格式
    支持两种输入格式:
    1. 原格式: {"groups": [...], "room_envelope": {...}}
    2. SSR格式: {"objects": [...], "bounds_top": [...], "bounds_bottom": [...], "room_type": "..."}
    """
    # 如果已经有groups结构，直接返回
    if "groups" in scene:
        return scene
    
    # SSR格式转换
    if "objects" in scene and "bounds_top" in scene and "bounds_bottom" in scene:
        print("    检测到SSR格式场景,转换为标准格式...")
        normalized_scene = {
            "room_envelope": {
                "bounds_top": scene["bounds_top"],
                "bounds_bottom": scene["bounds_bottom"]
            },
            "room_type": scene.get("room_type", "room"),
            "room_id": scene.get("room_id", "unknown"),
            "groups": [
                {
                    "group_name": "Main Group",
                    "group_type": scene.get("room_type", "general"),
                    "objects": scene["objects"]
                }
            ]
        }
        return normalized_scene
    
    # 如果格式不识别，返回原样
    return scene

def denormalize_scene_format(scene: Dict[str, Any], original_format: str = "groups") -> Dict[str, Any]:
    """
    将标准化的场景转换回原始格式
    """
    if original_format == "ssr":
        # 转换回SSR格式
        all_objects = []
        
        # 检查场景格式并提取物体
        if "groups" in scene:
            # 从groups格式提取
            for group in scene["groups"]:
                all_objects.extend(group.get("objects", []))
        elif "objects" in scene:
            # 已经是SSR格式，直接使用objects
            all_objects = scene.get("objects", [])
        
        # 提取边界信息
        if "room_envelope" in scene:
            bounds_top = scene["room_envelope"].get("bounds_top", [])
            bounds_bottom = scene["room_envelope"].get("bounds_bottom", [])
        else:
            # 直接从场景中获取（可能已经是SSR格式）
            bounds_top = scene.get("bounds_top", [])
            bounds_bottom = scene.get("bounds_bottom", [])
        
        ssr_scene = {
            "bounds_top": bounds_top,
            "bounds_bottom": bounds_bottom,
            "room_type": scene.get("room_type", "room"),
            "room_id": scene.get("room_id", "unknown"),
            "objects": all_objects
        }
        return ssr_scene
    
    # 默认返回groups格式
    return scene

def detect_scene_format(scene: Dict[str, Any]) -> str:
    """
    检测场景的格式类型
    返回: "groups" 或 "ssr"
    """
    if "groups" in scene:
        return "groups"
    elif "objects" in scene and "bounds_top" in scene and "bounds_bottom" in scene:
        return "ssr"
    else:
        return "unknown"

def apply_tool_calls_to_scene(scene: Dict[str, Any], tool_calls: List[Dict[str, Any]], jid_mapping: Dict[str, str] = None) -> Dict[str, Any]:
    """
    将工具调用应用到场景中，返回更新后的场景
    支持两种场景格式
    
    Args:
        scene: 原始场景数据
        tool_calls: 工具调用列表
        jid_mapping: 可选的jid映射字典，用于在多轮对话中保持jid一致性
                    格式：{"add_object_<index>": "real_jid_value"}
    """
    import uuid
    
    # 检测并标准化场景格式
    original_format = detect_scene_format(scene)
    normalized_scene = normalize_scene_format(scene)
    updated_scene = copy.deepcopy(normalized_scene)
    add_object_count = 0
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        if tool_name == "add_object":
            # 添加物体 - 如果有jid_mapping，则使用映射的真实jid，否则生成新UUID
            add_object_key = f"add_object_{add_object_count}"
            if jid_mapping and add_object_key in jid_mapping:
                jid = jid_mapping[add_object_key]
            else:
                jid = str(uuid.uuid4())
            add_object_count += 1
            
            new_object = {
                "desc": args.get("object_description", ""),
                "size": args.get("size", [1.0, 1.0, 1.0]),
                "pos": args.get("position", [0.0, 0.0, 0.0]),
                "rot": args.get("rotation", [0, 0, 0, 1]),
                "jid": jid,
                "type": "major"
            }
            
            # 处理group_name：如果指定了group_name，添加到对应的group；否则添加到第一个group
            group_name = args.get("group_name")
            target_group = None
            
            if group_name:
                # 如果指定了group_name，查找对应的group
                for group in updated_scene["groups"]:
                    if group.get("group_name") == group_name:
                        target_group = group
                        break
            
            # 如果没有指定group_name或没找到对应的group，使用第一个group
            if target_group is None and updated_scene["groups"]:
                target_group = updated_scene["groups"][0]
            
            # 添加物体
            if target_group is not None:
                target_group["objects"].append(new_object)
                    
        elif tool_name == "remove_object":
            # 移除物体
            jid_to_remove = args.get("jid")
            for group in updated_scene["groups"]:
                group["objects"] = [obj for obj in group["objects"] if obj.get("jid") != jid_to_remove]
                
        elif tool_name == "move_object":
            # 移动物体
            jid_to_move = args.get("jid")
            new_position = args.get("new_position")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_move:
                        obj["pos"] = new_position
                        break
                        
        elif tool_name == "rotate_object":
            # 旋转物体
            jid_to_rotate = args.get("jid")
            new_rotation = args.get("new_rotation")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_rotate:
                        obj["rot"] = new_rotation
                        break
                        
        elif tool_name == "scale_object":
            # 缩放物体
            jid_to_scale = args.get("jid")
            new_size = args.get("new_size")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_scale:
                        obj["size"] = new_size
                        break
                        
        elif tool_name == "replace_object":
            # 替换物体 - 如果有jid_mapping，需要分配新的真实jid
            jid_to_replace = args.get("jid_to_replace")
            new_description = args.get("new_object_description")
            
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_replace:
                        obj["desc"] = new_description
                        
                        # 检查jid_mapping是否提供了新的jid用于替换
                        replace_jid_key = f"replace_object_{jid_to_replace}"
                        if jid_mapping and replace_jid_key in jid_mapping:
                            obj["jid"] = jid_mapping[replace_jid_key]
                        # 否则保持原有JID不变
                        break
    
    # 转换回原始格式
    return denormalize_scene_format(updated_scene, original_format)

def _generate_random_reverse_operations(scene: Dict[str, Any], tool_id_start: int) -> List[Dict[str, Any]]:
    """
    为给定场景生成随机的逆向操作
    支持两种场景格式
    """
    import uuid
    import random
    
    # 标准化场景格式
    normalized_scene = normalize_scene_format(scene)
    
    tool_calls = []
    all_objects = []
    object_group_mapping = {}
    
    # 收集所有物体
    for group in normalized_scene["groups"]:
        for obj in group["objects"]:
            all_objects.append(obj)
            object_group_mapping[obj["jid"]] = group.get("group_name", "Unknown Group")
    
    if not all_objects:
        return []
    
    # 定义操作类型和对应的概率权重（不包括add，因为add意味着从空场景开始）
    operation_probabilities = {
        "move": 0.35,      # 35% - 移动操作最常见
        "rotate": 0.25,    # 25% - 旋转操作较常见
        "scale": 0.20,     # 20% - 缩放操作中等频率
        "replace": 0.15,   # 15% - 替换操作较少
        "remove": 0.05     # 5% - 移除操作最少，保持场景稳定
    }
    
    operation_types = list(operation_probabilities.keys())
    operation_weights = list(operation_probabilities.values())
    num_operations = random.randint(1, min(3, len(all_objects)))  # 1-3个操作，不超过物体数量
    
    print(f"    使用操作概率: {operation_probabilities}")
    
    for i in range(num_operations):
        operation_type = random.choices(operation_types, weights=operation_weights, k=1)[0]
        print(f"    选择操作类型: {operation_type} (概率: {operation_probabilities[operation_type]*100:.1f}%)")
        tool_id = tool_id_start + i
        
        if operation_type == "remove" and all_objects:
            # 移除操作
            obj_to_remove = random.choice(all_objects)
            tool_calls.append({
                "id": f"tool_{tool_id}",
                "name": "remove_object",
                "arguments": {
                    "jid": obj_to_remove["jid"]
                }
            })
            # 从可用物体列表中移除，避免重复操作同一物体
            all_objects.remove(obj_to_remove)
            
        elif operation_type == "move" and all_objects:
            # 移动操作
            obj_to_move = random.choice(all_objects)
            # 生成新的随机位置（在房间边界内）
            room_bounds = normalized_scene.get("room_envelope", {}).get("bounds_bottom", [])
            if room_bounds and len(room_bounds) >= 4:
                min_x = min(point[0] for point in room_bounds)
                max_x = max(point[0] for point in room_bounds)
                min_z = min(point[2] for point in room_bounds)
                max_z = max(point[2] for point in room_bounds)
                
                new_x = random.uniform(min_x + 0.5, max_x - 0.5)
                new_z = random.uniform(min_z + 0.5, max_z - 0.5)
                new_position = [new_x, obj_to_move["pos"][1], new_z]
            else:
                # 如果没有房间边界信息，使用小幅度偏移
                current_pos = obj_to_move["pos"]
                new_position = [
                    current_pos[0] + random.uniform(-1.0, 1.0),
                    current_pos[1],
                    current_pos[2] + random.uniform(-1.0, 1.0)
                ]
            
            tool_calls.append({
                "id": f"tool_{tool_id}",
                "name": "move_object",
                "arguments": {
                    "jid": obj_to_move["jid"],
                    "new_position": new_position
                }
            })
            
        elif operation_type == "rotate" and all_objects:
            # 旋转操作
            obj_to_rotate = random.choice(all_objects)
            # 生成新的随机旋转（绕Y轴旋转）
            angle = random.uniform(0, 2 * 3.14159)  # 0 to 2π
            new_rotation = [0, math.sin(angle/2), 0, math.cos(angle/2)]
            
            tool_calls.append({
                "id": f"tool_{tool_id}",
                "name": "rotate_object",
                "arguments": {
                    "jid": obj_to_rotate["jid"],
                    "new_rotation": new_rotation
                }
            })
            
        elif operation_type == "scale" and all_objects:
            # 缩放操作
            obj_to_scale = random.choice(all_objects)
            current_size = obj_to_scale["size"]
            # 随机缩放因子 (0.8 到 1.2)
            scale_factor = random.uniform(0.8, 1.2)
            new_size = [s * scale_factor for s in current_size]
            
            tool_calls.append({
                "id": f"tool_{tool_id}",
                "name": "scale_object",
                "arguments": {
                    "jid": obj_to_scale["jid"],
                    "new_size": new_size
                }
            })
            
        elif operation_type == "replace" and all_objects:
            # 替换操作
            obj_to_replace = random.choice(all_objects)
            # 生成新的物体描述（简单变体）
            current_desc = obj_to_replace["desc"]
            
            # 简单的描述变体
            color_variants = ["white", "black", "gray", "brown", "beige", "dark", "light"]
            style_variants = ["modern", "contemporary", "minimalist", "traditional", "vintage"]
            
            new_desc = current_desc
            for color in color_variants:
                if color in current_desc.lower():
                    new_color = random.choice([c for c in color_variants if c != color])
                    new_desc = new_desc.replace(color, new_color)
                    break
            else:
                # 如果没有找到颜色，添加一个风格
                style = random.choice(style_variants)
                new_desc = f"{style} {new_desc.lower()}"
            
            tool_calls.append({
                "id": f"tool_{tool_id}",
                "name": "replace_object",
                "arguments": {
                    "jid_to_replace": obj_to_replace["jid"],
                    "new_object_description": new_desc
                }
            })
    
    return tool_calls

def _apply_reverse_operations(scene: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将逆向操作应用到场景，生成"之前"的场景状态
    支持两种场景格式
    """
    import uuid
    import random
    
    # 检测并标准化场景格式
    original_format = detect_scene_format(scene)
    normalized_scene = normalize_scene_format(scene)
    updated_scene = copy.deepcopy(normalized_scene)
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("arguments", {})
        
        if tool_name == "remove_object":
            # 逆向：添加这个物体回去（但要生成不同的变体）
            jid_to_remove = args.get("jid")
            
            # 找到要被移除的物体
            removed_obj = None
            group_name = None
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_remove:
                        removed_obj = copy.deepcopy(obj)
                        group_name = group.get("group_name", "Unknown Group")
                        break
            
            # 不实际移除物体，保持原有的jid
            # 注意：这里不需要添加变体物体，因为我们只是在逆向生成工具调用
                        
        elif tool_name == "move_object":
            # 逆向：将物体移动到不同位置
            jid_to_move = args.get("jid")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_move:
                        # 随机生成一个不同的"原始"位置
                        current_pos = obj["pos"]
                        obj["pos"] = [
                            current_pos[0] + random.uniform(-1.5, 1.5),
                            current_pos[1],
                            current_pos[2] + random.uniform(-1.5, 1.5)
                        ]
                        break
                        
        elif tool_name == "rotate_object":
            # 逆向：设置不同的旋转
            jid_to_rotate = args.get("jid")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_rotate:
                        # 生成不同的旋转
                        angle = random.uniform(0, 2 * 3.14159)
                        obj["rot"] = [0, math.sin(angle/2), 0, math.cos(angle/2)]
                        break
                        
        elif tool_name == "scale_object":
            # 逆向：设置不同的尺寸
            jid_to_scale = args.get("jid")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_scale:
                        # 生成不同的尺寸
                        current_size = obj["size"]
                        scale_factor = random.uniform(0.7, 1.3)
                        obj["size"] = [s * scale_factor for s in current_size]
                        break
                        
        elif tool_name == "replace_object":
            # 逆向：使用原始描述
            jid_to_replace = args.get("jid_to_replace")
            for group in updated_scene["groups"]:
                for obj in group["objects"]:
                    if obj.get("jid") == jid_to_replace:
                        # 生成一个"原始"描述（当前描述的变体）
                        current_desc = obj["desc"]
                        
                        # 简单的逆向变体
                        color_variants = ["white", "black", "gray", "brown", "beige"]
                        for color in color_variants:
                            if color in current_desc.lower():
                                original_color = random.choice([c for c in color_variants if c != color])
                                obj["desc"] = current_desc.replace(color, original_color)
                                break
                        
                        # 不改变jid，保持原有的真实jid
                        break
    
    # 转换回原始格式
    return denormalize_scene_format(updated_scene, original_format)

def generate_global_user_instruction(final_scene: Dict[str, Any], instruction_type: str = "mixed") -> str:
    """
    基于最终场景（子场景00）生成一个全局用户指令，
    描述用户想要构造出这个完整场景的意图
    
    Args:
        final_scene: 最终完整场景的数据
        instruction_type: 指令类型 - "detailed"(详细), "brief"(简短), 或 "mixed"(混合，随机选择)
    """
    
    # 分析最终场景内容
    def analyze_scene(scene):
        # 标准化场景格式
        normalized_scene = normalize_scene_format(scene)
        
        if not normalized_scene or 'groups' not in normalized_scene:
            return {
                "room_type": "room",
                "total_objects": 0,
                "main_categories": [],
                "object_types": {},
                "groups": [],
                "specific_items": []
            }
        
        groups = normalized_scene.get('groups', [])
        all_objects = []
        group_types = []
        specific_items = []
        
        # 收集所有物体和组信息
        for group in groups:
            group_types.append(group.get('group_type', 'unknown'))
            objects = group.get('objects', [])
            all_objects.extend(objects)
            
            # 提取具体物体描述
            for obj in objects:
                desc = obj.get('desc', '').lower()
                specific_items.append(desc)
        
        # 统计物体类型和数量
        object_types = {}
        furniture_categories = []
        
        for obj in all_objects:
            jid = obj.get('jid', 'unknown_object')
            desc = obj.get('desc', '').lower()
            
            # 从描述中提取家具类型
            if 'chair' in desc:
                category = 'chair'
            elif 'table' in desc:
                category = 'table'
            elif 'sofa' in desc:
                category = 'sofa'
            elif 'bed' in desc:
                category = 'bed'
            elif 'desk' in desc:
                category = 'desk'
            elif 'shelf' in desc or 'bookshelf' in desc or 'bookcase' in desc:
                category = 'storage'
            elif 'lamp' in desc:
                category = 'lighting'
            elif 'cabinet' in desc or 'dresser' in desc:
                category = 'storage'
            else:
                # 从jid中提取类型
                category = jid.split('_')[0] if '_' in jid else 'furniture'
            
            furniture_categories.append(category)
            object_types[category] = object_types.get(category, 0) + 1
        
        # 推断房间类型
        room_type = normalized_scene.get('room_type', 'room')
        
        return {
            "room_type": room_type,
            "total_objects": len(all_objects),
            "main_categories": list(set(furniture_categories))[:5],
            "object_types": object_types,
            "groups": [(g.get('group_name', ''), g.get('group_type', '')) for g in groups],
            "furniture_categories": furniture_categories,
            "specific_items": specific_items[:3]  # 取前3个具体物品描述
        }
    
    scene_analysis = analyze_scene(final_scene)
    
    # 如果是混合模式，随机选择指令类型
    if instruction_type == "mixed":
        import random
        # 50% 简短指令，50% 详细指令
        actual_instruction_type = "brief" if random.random() < 0.5 else "detailed"
    else:
        actual_instruction_type = instruction_type
    
    # 根据实际选择的指令类型生成不同的提示词模板
    if actual_instruction_type == "brief":
        # 简短一句话指令的提示词
        instruction_prompt = f"""You are helping to generate very brief, single-sentence user instructions for interior design. Based on the target scene description below, create a concise, natural user request (ONE sentence only) that expresses their desire to create this space.

## Target Scene Analysis
- Room Type: {scene_analysis['room_type']}
- Total Objects: {scene_analysis['total_objects']}
- Main Furniture: {', '.join(scene_analysis['main_categories'])}
- Key Groups: {[group[1] for group in scene_analysis['groups']]}

## Task Requirements

Generate a brief, single-sentence user instruction. Choose ONE style randomly:

**Style A - Simple Goal (30%)**
- "I want a comfortable bedroom"
- "Create a functional workspace"
- "I need a cozy living room"

**Style B - Basic Requirements (40%)**
- "I want a bedroom with a bed and storage"
- "Create a workspace with a desk and chair"
- "I need a living room with seating and lighting"

**Style C - Style + Function (30%)**
- "I want a modern bedroom that's both stylish and functional"
- "Create a minimalist workspace with good lighting"
- "I need a cozy living room with comfortable seating"

## Output Requirements:
- Exactly ONE sentence
- No more than 15-20 words
- Natural and conversational
- Reference the room type: {scene_analysis['room_type']}

User Instruction:"""

    else:
        # 详细指令的提示词（保持原有逻辑）
        instruction_prompt = f"""You are helping to generate highly diverse and realistic user instructions for interior design. Based on the target scene description below, create a natural user request that expresses their desire to create or design this complete space.

## Target Scene Analysis
- Room Type: {scene_analysis['room_type']}
- Total Objects: {scene_analysis['total_objects']}
- Functional Groups: {scene_analysis['groups']}
- Main Furniture Categories: {', '.join(scene_analysis['main_categories'])}
- Object Distribution: {scene_analysis['object_types']}
- Specific Items Present: {scene_analysis['specific_items'][:3]}

## Task Requirements

Generate a user instruction that expresses their vision for creating this complete space. The instruction should be HIGHLY DIVERSE in style and specificity.

## Instruction Style Variations (choose one randomly with varying probabilities):

### A. Highly Specific with Object Details (25%)
Be very specific about exact furniture pieces, quantities, colors, materials, or styles:
- "I need a workspace with a gray dressing table, a leather lounge chair, and a wooden bookcase with glass doors"
- "Create a living room with a Victorian-style three-seat sofa, two blue fabric dining chairs, and a modern coffee table"
- "I want a bedroom with a king-size bed, two matching nightstands, and a large wardrobe in dark wood"
- "Design a dining area with a rectangular dark wood table and at least four chairs with white frames"

### B. Category-Focused Requirements (20%)
Focus on specific furniture categories or functional requirements:
- "I need a complete workspace setup with proper seating, storage, and lighting solutions"
- "Create a dining area with a good table and matching chairs that can seat 4-6 people"
- "I want a living room with comfortable seating options and proper lighting"
- "Design a bedroom with essential sleeping and storage furniture"

### C. Style and Aesthetic Goals (20%)
Emphasize design styles, colors, materials, or overall aesthetic:
- "I want a modern minimalist workspace with clean lines and neutral colors"
- "Create a cozy traditional living room with warm wood tones and comfortable furniture"
- "Design a contemporary bedroom with sleek furniture and sophisticated styling"
- "I need a functional space with an industrial aesthetic and metal accents"

### D. Functional Lifestyle Needs (15%)
Focus on how the space will be used and lifestyle requirements:
- "I need a productive workspace where I can focus on writing and research"
- "Create a family-friendly living room perfect for entertaining and relaxation"
- "Design a bedroom that's both a peaceful retreat and efficient getting-ready space"
- "I want a dining area that works for both daily meals and special occasions"

### E. Spatial and Layout Focused (10%)
Emphasize spatial organization, flow, and layout considerations:
- "Help me create distinct zones in this space for different activities"
- "I want an efficient layout that maximizes the use of available space"
- "Design a well-organized room with clear circulation paths"
- "Create functional areas that flow naturally into each other"

### F. Mixed Specific and General (10%)
Combine specific requirements with general goals:
- "I need a workspace with a proper desk setup, plus whatever storage and seating would make it complete"
- "Create a living room centered around comfortable seating, with appropriate lighting and accent pieces"
- "I want a bedroom with a large bed and good storage, designed in a calming, modern style"

## Enhanced Guidelines for More Diversity:

**Specificity Levels:**
- Ultra-specific: Mention exact materials, colors, quantities, dimensions
- Moderate: Reference general furniture types and basic requirements  
- Abstract: Focus on feelings, atmospheres, and general goals

**Language Styles:**
- Professional: "I require a comprehensive workspace solution..."
- Casual: "I want to set up a cool workspace where I can..."
- Emotional: "I dream of having a space that feels..."
- Practical: "I need a functional setup that includes..."

**Requirements Complexity:**
- Simple: Single primary goal with basic requirements
- Moderate: Multiple requirements with some constraints
- Complex: Detailed specifications with multiple criteria

## Important Guidelines:
- Make each instruction sound unique and personal
- Reference the specific room type: {scene_analysis['room_type']}
- Vary sentence structure, length, and complexity
- Include both must-have and nice-to-have elements
- Use diverse vocabulary and phrasing
- Balance specificity with natural conversation flow

## Output Format
Provide only the user instruction text, nothing else. Make it sound conversational and natural.

User Instruction:"""

    try:
        instruction_length = "简短" if actual_instruction_type == "brief" else "详细"
        print(f"    正在为 {scene_analysis['room_type']} 生成{instruction_length}全局用户指令...")
        print(f"    场景包含: {scene_analysis['total_objects']} 个物体, 分组: {[group[0] for group in scene_analysis['groups']]}")
        
        # 根据实际指令类型调整参数
        max_tokens = 50 if actual_instruction_type == "brief" else 250
        temperature = 0.8 if actual_instruction_type == "brief" else 0.9
        
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "user", "content": instruction_prompt}
            ],
        )
        
        user_instruction = response.choices[0].message.content.strip()
        
        # 清理可能的引号或多余文本
        if user_instruction.startswith('"') and user_instruction.endswith('"'):
            user_instruction = user_instruction[1:-1]
        if user_instruction.startswith("User Instruction:"):
            user_instruction = user_instruction.replace("User Instruction:", "").strip()
        
        print(f"    ✓ 成功生成全局指令: {user_instruction}")
        return user_instruction
        
    except Exception as e:
        print(f"    ✗ 生成全局用户指令时出错: {e}")
        # 如果GPT调用失败，生成一个基于场景分析的多样化默认指令
        room_type = scene_analysis['room_type']
        if scene_analysis['groups']:
            group_purposes = [group[1] for group in scene_analysis['groups']]
            specific_items = scene_analysis['specific_items']
            
            # 随机选择不同风格的备用指令
            import random
            fallback_options = [
                f"Help me create a complete {room_type} with comfortable seating and dining areas.",
                f"I want to design a functional {room_type} with all the essential furniture pieces.",
                f"Create a well-organized {room_type} that includes proper storage and seating solutions.",
                f"I need a {room_type} setup that works for both daily use and entertaining guests."
            ]
            
            if 'workspace' in group_purposes:
                fallback_options.extend([
                    f"Design a productive {room_type} workspace with a desk, chair, and storage.",
                    f"I want a {room_type} that's perfect for focused work with good lighting."
                ])
            
            fallback = random.choice(fallback_options)
        else:
            fallback = f"Help me design a complete and functional {room_type} layout."
        
        print(f"    → 使用备用指令: {fallback}")
        return fallback

class SceneReverseEditor:
    """场景逆向编辑器：从最终场景推导原始场景和操作"""
    
    def __init__(self, assets_file_path: str = None):
        self.operations = []
        self.replacement_mapping = {}  # 跟踪替换操作的映射
        self.assets_data = {}
        
        # 加载资产描述数据
        if assets_file_path and os.path.exists(assets_file_path):
            try:
                with open(assets_file_path, 'r', encoding='utf-8') as f:
                    self.assets_data = json.load(f)
                print(f"已加载 {len(self.assets_data)} 个资产描述")
            except Exception as e:
                print(f"加载资产文件失败: {e}")
                self.assets_data = {}
        
    def add_random_variations(self, final_scene: Dict[str, Any], current_subscene_name: str = "", num_operations: int = None, favor_add: bool = False, step_progress: float = 0.0) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        从最终场景随机生成原始场景和操作序列
        支持两种场景格式
        Args:
            final_scene: 最终场景数据
            current_subscene_name: 当前编辑的子场景名称（如 "subscene_000"）
            num_operations: 指定操作数量（如果为None，则使用默认的1-6个）
            favor_add: 是否偏向add操作（用于多轮对话，帮助逐步到达空场景）
            step_progress: 当前步骤进度 (0.0=逆向开始/正向结束 -> 1.0=逆向结束/正向开始)
        返回：(原始场景, 工具调用列表)
        """
        # 检测原始格式
        # 仅当original_format未设置时才进行检测并设置
        if not hasattr(self, 'original_format') or self.original_format is None:
            self.original_format = detect_scene_format(final_scene)
        
        # 标准化为groups格式进行处理
        normalized_final_scene = normalize_scene_format(final_scene)
        
        # 深拷贝最终场景作为起始点
        initial_scene = copy.deepcopy(normalized_final_scene)
        tool_calls = []
        
        # 跟踪替换操作的映射关系
        self.replacement_mapping = {}  # {original_jid: final_description}
        
        # 只有当编辑的是子场景00时，才可能使用终止操作（约10%的概率）
        is_final_target_scene = current_subscene_name == "subscene_000"
        if is_final_target_scene and random.random() < 0.1:
            # 使用终止操作：场景无需修改（只在目标场景subscene_000中可能发生）
            tool_call = self._create_terminate_operation()
            tool_calls.append(tool_call)
            print(f"      → 使用终止操作（场景已达到最终目标状态）")
            return initial_scene, tool_calls
        
        # 根据是否需要偏向add操作来设置概率
        if favor_add:
            # 多轮对话模式：平衡分布，略微偏向add（35%），降低scale概率
            operation_probabilities = {
                "add": 0.35,
                "move": 0.20,
                "rotate": 0.20,
                "scale": 0.05,  # 降低scale概率
                "replace": 0.10,
                "remove": 0.10
            }
        else:
            # 单轮对话模式：平衡的操作分布，降低scale概率
            operation_probabilities = {
                "add": 0.3,       
                "move": 0.2,      
                "rotate": 0.2,    
                "scale": 0.1,     # 降低scale概率
                "replace": 0.1,   
                "remove": 0.1     
            }
        
        operation_types = list(operation_probabilities.keys())
        operation_weights = list(operation_probabilities.values())
        
        # 确定操作数量
        all_objects = []
        for group in initial_scene["groups"]:
            all_objects.extend(group["objects"])
        
        if num_operations is None:
            # 单轮对话：1-6个操作
            num_ops = random.randint(1, min(6, len(all_objects))) if all_objects else 0
        else:
            # 多轮对话：1到物体数量之间的随机数
            num_ops = random.randint(1, len(all_objects)) if all_objects else 0
        
        print(f"      → {'多轮' if favor_add else '单轮'}对话操作概率: add={operation_probabilities['add']*100:.0f}%, 操作数={num_ops}")
        
        for i in range(num_ops):
            operation_type = random.choices(operation_types, weights=operation_weights, k=1)[0]
            print(f"      → 选择操作: {operation_type} (概率: {operation_probabilities[operation_type]*100:.1f}%)")
            
            if operation_type == "add":
                # 添加操作：从原始场景中移除一个物体（基于进度分阶段选择）
                tool_call = self._reverse_add_operation(initial_scene, len(tool_calls) + 1, step_progress)
                if tool_call:
                    tool_calls.append(tool_call)
                    
            elif operation_type == "remove":
                # 移除操作：在原始场景中添加一个物体
                tool_call = self._reverse_remove_operation(initial_scene, len(tool_calls) + 1)
                if tool_call:
                    tool_calls.append(tool_call)
                    
            elif operation_type == "replace":
                # 替换操作：在原始场景中使用不同的物体
                tool_call = self._reverse_replace_operation(initial_scene, len(tool_calls) + 1)
                if tool_call:
                    tool_calls.append(tool_call)
                    
            elif operation_type == "move":
                # 移动操作：在原始场景中改变物体位置
                tool_call = self._reverse_move_operation(initial_scene, len(tool_calls) + 1)
                if tool_call:
                    tool_calls.append(tool_call)
                    
            elif operation_type == "rotate":
                # 旋转操作：在原始场景中改变物体旋转
                tool_call = self._reverse_rotate_operation(initial_scene, len(tool_calls) + 1)
                if tool_call:
                    tool_calls.append(tool_call)
                    
            elif operation_type == "scale":
                # 缩放操作：在原始场景中改变物体尺寸
                tool_call = self._reverse_scale_operation(initial_scene, len(tool_calls) + 1)
                if tool_call:
                    tool_calls.append(tool_call)
        
        # 转换回原始格式
        initial_scene = denormalize_scene_format(initial_scene, self.original_format)
        
        return initial_scene, tool_calls
    
    def _create_terminate_operation(self) -> Dict[str, Any]:
        """创建终止操作：表示场景无需修改"""
        return {
            "id": "tool_1",
            "name": "terminate",
            "arguments": {
                "reason": "The current scene layout is already optimal and no modifications are needed."
            }
        }
    
    def _reverse_add_operation(self, initial_scene: Dict[str, Any], tool_id: int, step_progress: float = 0.0) -> Dict[str, Any]:
        """逆向添加操作：从原始场景中移除一个物体，生成add_object工具调用
        
        Args:
            step_progress: 0.0(逆向初期/正向末期) -> 1.0(逆向末期/正向初期)
        """
        all_objects = []
        object_group_mapping = {}  # 跟踪物体所在的组
        
        # 收集所有物体及其体积信息（用于分阶段选择）
        for group in initial_scene["groups"]:
            for obj in group["objects"]:
                all_objects.append(obj)
                object_group_mapping[obj["jid"]] = group.get("group_name", "Unknown Group")
        
        if not all_objects:
            return None
        
        # 方案2：根据step_progress分阶段选择物体
        # step_progress < 0.3 -> 小物件（装饰品，体积 < 0.5 m³）
        # 0.3 <= step_progress < 0.7 -> 中等物件（次要家具，0.5-2.0 m³）
        # step_progress >= 0.7 -> 大物件（核心家具，>= 2.0 m³）
        candidates = []
        if step_progress < 0.3:
            # 逆向初期：优先选择小物件
            for obj in all_objects:
                size = obj.get("size", [1.0, 1.0, 1.0])
                volume = size[0] * size[1] * size[2]
                if volume < 0.5:
                    candidates.append(obj)
            if candidates:
                print(f"        → 阶段1(装饰品): 候选 {len(candidates)}/{len(all_objects)}")
        elif step_progress < 0.7:
            # 逆向中期：优先选择中等物件
            for obj in all_objects:
                size = obj.get("size", [1.0, 1.0, 1.0])
                volume = size[0] * size[1] * size[2]
                if 0.5 <= volume < 2.0:
                    candidates.append(obj)
            if candidates:
                print(f"        → 阶段2(次要家具): 候选 {len(candidates)}/{len(all_objects)}")
        else:
            # 逆向末期：优先选择大物件
            for obj in all_objects:
                size = obj.get("size", [1.0, 1.0, 1.0])
                volume = size[0] * size[1] * size[2]
                if volume >= 2.0:
                    candidates.append(obj)
            if candidates:
                print(f"        → 阶段3(核心家具): 候选 {len(candidates)}/{len(all_objects)}")
        
        # 如果当前阶段没有合适的物体，使用所有物体
        if not candidates:
            candidates = all_objects
            
        # 随机选择一个物体移除
        obj_to_remove = random.choice(candidates)
        group_name = object_group_mapping.get(obj_to_remove["jid"], "Unknown Group")
        
        # 从原始场景中移除，但保留空的group
        for group in initial_scene["groups"]:
            if obj_to_remove in group["objects"]:
                group["objects"].remove(obj_to_remove)
                break
        
        # 生成add_object工具调用
        # 对于SSR格式，不包含group_name参数
        tool_call = {
            "id": f"tool_{tool_id}",
            "name": "add_object",
            "arguments": {
                "object_description": obj_to_remove["desc"],
                "position": obj_to_remove["pos"],
                "rotation": obj_to_remove["rot"],
                "size": obj_to_remove["size"]
            }
        }
        
        # 只有在groups格式下才包含group_name
        if self.original_format == "groups":
            tool_call["arguments"]["group_name"] = group_name
        
        return tool_call
    
    def _reverse_remove_operation(self, initial_scene: Dict[str, Any], tool_id: int) -> Dict[str, Any]:
        """逆向移除操作：在原始场景中添加一个物体，生成remove_object工具调用"""
        # 从资产数据库中选择一个真实物体
        if self.assets_data:
            # 随机选择一个真实的资产
            random_jid = random.choice(list(self.assets_data.keys()))
            random_descriptions = self.assets_data[random_jid]
            object_description = random_descriptions[-1] if random_descriptions else "unknown object"
            jid_to_use = random_jid
        else:
            # 如果没有资产数据，跳过这个操作
            return None
        
        # 创建新物体添加到原始场景
        new_object = {
            "desc": object_description,
            "size": [
                round(random.uniform(0.3, 2.0), 2),
                round(random.uniform(0.3, 2.0), 2), 
                round(random.uniform(0.3, 2.0), 2)
            ],
            "pos": [
                round(random.uniform(-2, 2), 2),
                0.0,
                round(random.uniform(-2, 2), 2)
            ],
            "rot": self._generate_random_rotation(),
            "jid": jid_to_use,
            "type": "major"
        }
        
        # 找到合适的组添加物体
        if initial_scene["groups"]:
            target_group = random.choice(initial_scene["groups"])
            target_group["objects"].append(new_object)
        
        # 生成remove_object工具调用
        return {
            "id": f"tool_{tool_id}",
            "name": "remove_object", 
            "arguments": {
                "jid": jid_to_use
            }
        }
    
    def _reverse_replace_operation(self, initial_scene: Dict[str, Any], tool_id: int) -> Dict[str, Any]:
        """逆向替换操作：在原始场景中改变物体，生成replace_object工具调用"""
        all_objects = []
        for group in initial_scene["groups"]:
            all_objects.extend(group["objects"])
        
        if not all_objects:
            return None
            
        # 随机选择一个物体进行替换
        obj_to_modify = random.choice(all_objects)
        final_jid = obj_to_modify["jid"]  # 保存最终的JID（LLM无法预知）
        final_description = obj_to_modify["desc"]  # 保存最终描述
        
        # 从资产数据库中选择一个不同的物体作为原始物体
        original_jid, original_description = self._select_replacement_asset(obj_to_modify)
        
        # 如果没有可用的替换资产，跳过这个操作
        if original_jid is None or original_description is None:
            return None
        
        # 记录替换映射关系：原始JID -> 最终JID和描述
        self.replacement_mapping[original_jid] = {
            "final_jid": final_jid,
            "final_description": final_description
        }
        
        # 修改原始场景中的物体为选中的原始物体
        obj_to_modify["jid"] = original_jid
        obj_to_modify["desc"] = original_description
        
        # 生成replace_object工具调用，LLM只知道原始JID和新的描述，不知道新的JID
        return {
            "id": f"tool_{tool_id}",
            "name": "replace_object",
            "arguments": {
                "jid_to_replace": original_jid,  # 使用真实的原始JID
                "new_object_description": final_description  # 新的描述，但没有new_jid
            }
        }
    
    def _select_replacement_asset(self, current_obj):
        """从资产数据库中选择一个合适的替换物体"""
        if not self.assets_data:
            # 如果没有资产数据，跳过这个操作
            return None, None
        
        # 从所有可用资产中随机选择一个不同的
        available_jids = list(self.assets_data.keys())
        if current_obj['jid'] in available_jids:
            available_jids.remove(current_obj['jid'])  # 排除当前物体
        
        if not available_jids:
            # 如果没有其他可用的，跳过这个操作
            return None, None
        
        # 随机选择一个替换JID
        replacement_jid = random.choice(available_jids)
        replacement_descriptions = self.assets_data[replacement_jid]
        
        # 选择最后一个描述作为主要描述（通常是最详细的描述）
        replacement_description = replacement_descriptions[-1] if replacement_descriptions else "unknown object"
        
        return replacement_jid, replacement_description
    
    def _reverse_move_operation(self, initial_scene: Dict[str, Any], tool_id: int) -> Dict[str, Any]:
        """逆向移动操作：在原始场景中改变物体位置，生成move_object工具调用
        
        使用局部微扰（±0.5米）而非全局随机移动（±1.0米）
        """
        all_objects = []
        for group in initial_scene["groups"]:
            all_objects.extend(group["objects"])
        
        if not all_objects:
            return None
            
        # 随机选择一个物体移动
        obj_to_move = random.choice(all_objects)
        final_position = obj_to_move["pos"].copy()
        
        # 限制移动范围：从±1.0改为±0.5米（局部微扰）
        # 这样产生的"错误"看起来像是"没放好"，而不是"瞬移"
        obj_to_move["pos"] = [
            final_position[0] + random.uniform(-0.5, 0.5),
            final_position[1],
            final_position[2] + random.uniform(-0.5, 0.5)
        ]
        
        # 生成move_object工具调用
        return {
            "id": f"tool_{tool_id}",
            "name": "move_object",
            "arguments": {
                "jid": obj_to_move["jid"],
                "new_position": final_position
            }
        }
    
    def _reverse_rotate_operation(self, initial_scene: Dict[str, Any], tool_id: int) -> Dict[str, Any]:
        """逆向旋转操作：在原始场景中改变物体旋转，生成rotate_object工具调用
        
        方案3：使用离散化角度（曼哈顿对齐）：80%概率吸附到90度倍数
        """
        all_objects = []
        for group in initial_scene["groups"]:
            all_objects.extend(group["objects"])
        
        if not all_objects:
            return None
            
        # 随机选择一个物体旋转
        obj_to_rotate = random.choice(all_objects)
        final_rotation = obj_to_rotate["rot"].copy()
        
        # 方案3：离散化旋转
        # 将当前旋转转换为角度
        current_w = final_rotation[3]
        current_y = final_rotation[1]
        current_angle = 2 * math.atan2(current_y, current_w)  # 弧度
        current_angle_deg = math.degrees(current_angle) % 360
        
        # 80%概率：吸附到90度的倍数
        # 20%概率：随机微调（±15度）
        if random.random() < 0.8:
            # 吸附到最近的90度倍数
            snap_angle_deg = round(current_angle_deg / 90) * 90
            # 逆向生成一个"歪"的角度作为上一状态
            prev_angle_deg = snap_angle_deg + random.uniform(-15, 15)
        else:
            # 完全随机旋转
            prev_angle_deg = random.uniform(0, 360)
        
        # 转换回四元数
        prev_angle_rad = math.radians(prev_angle_deg)
        obj_to_rotate["rot"] = [0, math.sin(prev_angle_rad/2), 0, math.cos(prev_angle_rad/2)]
        
        # 生成rotate_object工具调用
        return {
            "id": f"tool_{tool_id}",
            "name": "rotate_object",
            "arguments": {
                "jid": obj_to_rotate["jid"],
                "new_rotation": final_rotation
            }
        }
    
    def _reverse_scale_operation(self, initial_scene: Dict[str, Any], tool_id: int) -> Dict[str, Any]:
        """逆向缩放操作：在原始场景中改变物体尺寸，生成scale_object工具调用"""
        all_objects = []
        for group in initial_scene["groups"]:
            all_objects.extend(group["objects"])
        
        if not all_objects:
            return None
            
        # 随机选择一个物体缩放
        obj_to_scale = random.choice(all_objects)
        final_size = obj_to_scale["size"].copy()
        
        # 修改原始场景中的尺寸
        scale_factor = random.uniform(0.7, 1.3)
        obj_to_scale["size"] = [s * scale_factor for s in final_size]
        
        # 生成scale_object工具调用
        return {
            "id": f"tool_{tool_id}",
            "name": "scale_object",
            "arguments": {
                "jid": obj_to_scale["jid"],
                "new_size": final_size
            }
        }

    
    def _generate_random_rotation(self) -> List[float]:
        """生成随机旋转四元数"""
        # 生成绕Y轴的随机旋转
        angle = random.uniform(0, 2 * math.pi)
        return [0, round(math.sin(angle/2), 5), 0, round(math.cos(angle/2), 5)]
    
    
    def process_final_scene_jids(self, final_scene: Dict[str, Any], tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理final_scene中的JID，对于LLM无法预知的JID进行特殊标记
        支持两种场景格式
        
        Args:
            final_scene: 最终场景数据
            tool_calls: 工具调用列表
            
        Returns:
            处理后的final_scene，其中新增/替换物体的JID被特殊标记
        """
        # 检测原始格式（优先使用已保存的格式）
        if hasattr(self, 'original_format') and self.original_format is not None:
            original_format = self.original_format
        else:
            original_format = detect_scene_format(final_scene)
        
        # 标准化场景（无论输入是什么格式，都标准化为groups格式）
        normalized_scene = normalize_scene_format(final_scene)
        processed_scene = copy.deepcopy(normalized_scene)
        
        # 检查是否有终止操作
        has_terminate = any(tool_call["name"] == "terminate" for tool_call in tool_calls)
        if has_terminate:
            # 如果是终止操作，直接返回原始场景，无需JID处理
            return processed_scene
        
        # 分析工具调用，找出涉及添加和替换的操作
        add_operations = []
        replace_operations = []
        
        for tool_call in tool_calls:
            if tool_call["name"] == "add_object":
                add_operations.append(tool_call)
            elif tool_call["name"] == "replace_object":
                replace_operations.append(tool_call)
        
        # 处理添加操作：新增物体的JID设为特殊标记
        for add_op in add_operations:
            # 通过描述和位置匹配找到对应的物体
            target_desc = add_op["arguments"]["object_description"]
            target_pos = add_op["arguments"]["position"]
            
            for group in processed_scene["groups"]:
                for obj in group["objects"]:
                    if (obj["desc"] == target_desc and 
                        self._positions_match(obj["pos"], target_pos)):
                        obj["jid"] = "<NEED_RETRIVEAL>"
                        break
        
        # 处理替换操作：被替换后的新物体JID设为特殊标记
        for replace_op in replace_operations:
            original_jid = replace_op["arguments"]["jid_to_replace"]
            new_desc = replace_op["arguments"]["new_object_description"]
            
            # 从替换映射中找到对应的最终JID
            if original_jid in self.replacement_mapping:
                mapping_info = self.replacement_mapping[original_jid]
                final_jid = mapping_info["final_jid"]
                final_description = mapping_info["final_description"]
                
                for group in processed_scene["groups"]:
                    for obj in group["objects"]:
                        # 找到JID和描述都匹配的物体，这就是被替换的物体
                        if obj["jid"] == final_jid and obj["desc"] == final_description:
                            obj["jid"] = "<NEED_RETRIVEAL>"
                            break
        
        # 转换回原始格式
        return denormalize_scene_format(processed_scene, original_format)
    
    def _positions_match(self, pos1: List[float], pos2: List[float], tolerance: float = 0.01) -> bool:
        """检查两个位置是否匹配（允许小的误差）"""
        if len(pos1) != len(pos2):
            return False
        return all(abs(p1 - p2) < tolerance for p1, p2 in zip(pos1, pos2))

def generate_multi_turn_conversation(final_scene_path: str, output_path: str, global_user_instruction: str, assets_file_path: str = None) -> bool:
    """
    为subscene_000生成10种不同的多轮对话编辑链
    采用正确的逆向推导逻辑：从最终场景开始，通过逆编辑器生成多轮对话
    
    注意：每条编辑链会独立调用 generate_global_user_instruction 生成不同的用户指令，
    以确保指令多样性。传入的 global_user_instruction 参数已不再使用。
    """
    
    try:
        # 加载最终场景
        with open(final_scene_path, 'r', encoding='utf-8') as f:
            final_scene = json.load(f)
        
        # 标准化场景格式
        normalized_final_scene = normalize_scene_format(final_scene)
        
        # 提取子场景名称
        scene_id = Path(final_scene_path).stem
        
        # 检查场景是否有物体
        all_final_objects = []
        for group in normalized_final_scene["groups"]:
            all_final_objects.extend(group["objects"])
        
        if not all_final_objects:
            print(f"    场景中没有物体，跳过多轮对话生成")
            return False
        
        # 创建system prompt
        system_prompt = """You are an expert interior design assistant specializing in spatial layout optimization and furniture arrangement. Your role is to help users transform their living spaces through thoughtful design decisions.

Key capabilities:
- Analyze room layouts and identify optimization opportunities
- Recommend furniture placement for improved functionality and flow
- Suggest appropriate furniture pieces for specific needs and spaces
- Apply design principles including balance, proportion, and visual harmony
- Consider practical factors like traffic patterns, lighting, and room usage
- Provide clear reasoning for all design recommendations

When working with users:
- Listen carefully to their specific needs and preferences
- Ask clarifying questions when requirements are unclear
- Explain your design reasoning step by step
- Consider both aesthetic and functional aspects
- Respect budget constraints and existing furniture when specified
- Provide multiple options when appropriate

Your goal is to create spaces that are both beautiful and highly functional for the user's lifestyle.

## OUTPUT FORMAT REQUIREMENTS

You must structure your response using the following special tokens in this exact order:

1. **Thinking Process**: Wrap your analysis and reasoning in <think></think> tags
2. **Conclusion**: Wrap a one-sentence summary of your thinking process in <conclusion></conclusion> tags
3. **Tool Calls**: Wrap your actions in <tool_calls></tool_calls> tags

Example format:
<think>
Your detailed analysis and reasoning process here...
</think>

<conclusion>
Wrap a one-sentence summary of your thinking process
</conclusion>

<tool_calls>
[
  {
    "id": "tool_1",
    "name": "action_name",
    "arguments": {...}
  }
]
</tool_calls>

## AVAILABLE TOOLS

You have access to the following tools for scene editing:

### 1. add_object
Add a new furniture piece to the scene.
**Parameters:**
- object_description (string): Detailed description of the furniture piece
- position (array): [x, y, z] coordinates in the room
- rotation (array): [x, y, z, w] quaternion rotation
- size (array): [width, height, depth] dimensions
- group_name (string): Name of the group this object belongs to

### 2. remove_object
Remove an existing object from the scene.
**Parameters:**
- jid (string): Unique identifier of the object to remove

### 3. move_object
Change the position of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to move
- new_position (array): [x, y, z] new coordinates

### 4. rotate_object
Change the rotation of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to rotate
- new_rotation (array): [x, y, z, w] new quaternion rotation

### 5. scale_object
Change the size of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to scale
- new_size (array): [width, height, depth] new dimensions

### 6. replace_object
Replace an existing object with a different object type.
**Parameters:**
- jid_to_replace (string): Unique identifier of the object to replace
- new_object_description (string): Description of the new object to place

### 7. terminate
End the editing session when the scene is complete.
**Parameters:**
- reason (string): Explanation for why editing is complete

## CRITICAL SPATIAL AWARENESS REQUIREMENTS

**MANDATORY**: You must carefully observe the rendered top-view image and analyze the <current_scene> data to ensure proper spatial arrangement.

### Overlap Prevention:
- **NO OVERLAPPING**: Objects must never occupy the same 3D space
- Check object positions and sizes to ensure clear separation
- Maintain minimum 0.1 meter clearance between objects when possible
- If you detect overlapping objects in the current scene, you MUST fix them by moving one or both objects

### Boundary Compliance:
- **NO OUT-OF-BOUNDS**: All objects must stay within the room_envelope boundaries
- Check that object position + half of object size stays within room bounds
- For position [x,y,z] and size [w,h,d], ensure:
  - x ± w/2 stays within room x-bounds
  - z ± d/2 stays within room z-bounds
- If you detect out-of-bounds objects, you MUST move them to valid positions

### Editing Validation:
- Before any move/add operation, calculate the final object boundaries
- Verify no conflicts with existing objects
- Verify the object stays within room boundaries
- If conflicts exist, choose different positions or modify other objects first
- Priority: Fix existing violations before adding new objects

### Top-View Analysis:
- Study the provided rendered image carefully
- The top-view shows the actual spatial relationships between objects
- Use this visual information to identify spatial problems
- Cross-reference with the numerical data in <current_scene>
- Trust the visual evidence when planning modifications"""

        # 为每个场景创建单独的文件夹
        scene_output_dir = Path(output_path).parent / Path(final_scene_path).stem
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 持续生成直到获得3条合格的编辑链
        success_count = 0
        attempt_count = 0
        max_attempts = 30  # 设置最大尝试次数，避免无限循环
        
        print(f"    目标：生成3条合格的编辑链，保存到 {scene_output_dir}")
        
        while success_count < 3 and attempt_count < max_attempts:
            attempt_count += 1
            print(f"    尝试生成编辑链 {attempt_count} (已成功: {success_count}/3)")
            
            try:
                # 为每条编辑链单独生成用户指令，确保多样性
                chain_user_instruction = generate_global_user_instruction(final_scene, "mixed")
                print(f"      本次编辑链用户指令: {chain_user_instruction}")
                
                # 创建场景逆编辑器
                editor = SceneReverseEditor(assets_file_path)
                
                # 检测并保存原始场景格式（在normalize之前）
                original_scene_format = detect_scene_format(final_scene)
                editor.original_format = original_scene_format
                
                # 生成多轮对话链：从最终场景开始，逐步逆推到空场景
                # 策略：生成4-8轮对话，如果第8轮还有物体，则全部删除
                conversation_steps = []
                current_target_scene = copy.deepcopy(normalized_final_scene)
                target_turns = random.randint(4, 8)  # 随机选择4-8轮
                
                print(f"      开始生成多轮对话（目标轮次: {target_turns}，确保到达空场景）")
                
                for turn_idx in range(target_turns):
                    # 检查当前目标场景是否为空场景
                    # 确保使用标准化格式进行检查
                    normalized_target_scene = normalize_scene_format(current_target_scene)
                    current_objects = []
                    for group in normalized_target_scene["groups"]:
                        current_objects.extend(group["objects"])
                    
                    if not current_objects:
                        print(f"        第 {turn_idx + 1} 轮: 场景已空，提前完成（共 {turn_idx} 轮编辑）")
                        break
                    
                    is_last_turn = (turn_idx == target_turns - 1)
                    
                    if is_last_turn:
                        # 最后一轮：删除所有剩余物体
                        print(f"        生成第 {turn_idx + 1} 轮操作（最后一轮，删除全部 {len(current_objects)} 个物体）")
                        
                        # 为所有剩余物体生成add操作（逆向就是删除）
                        tool_calls = []
                        # 使用标准化场景作为基础，方便操作
                        initial_scene_for_turn = copy.deepcopy(normalized_target_scene)
                        
                        for obj_idx, obj in enumerate(current_objects):
                            # 从initial_scene中移除这个物体
                            for group in initial_scene_for_turn["groups"]:
                                group["objects"] = [o for o in group["objects"] if o["jid"] != obj["jid"]]
                            
                            # 生成add操作
                            tool_call = {
                                "id": f"tool_{obj_idx + 1}",
                                "name": "add_object",
                                "arguments": {
                                    "object_description": obj["desc"],
                                    "position": obj["pos"],
                                    "rotation": obj["rot"],
                                    "size": obj["size"]
                                }
                            }
                            
                            # 只有在groups格式下才包含group_name（使用保存的原始格式）
                            if original_scene_format == "groups":
                                # 找到该物体所属的group
                                for group in normalized_target_scene["groups"]:
                                    if obj in group["objects"]:
                                        tool_call["arguments"]["group_name"] = group.get("group_name", "Unknown Group")
                                        break
                            
                            tool_calls.append(tool_call)
                        
                        # 转换回原始格式
                        initial_scene_for_turn = denormalize_scene_format(initial_scene_for_turn, original_scene_format)
                        
                        print(f"        最后一轮生成了 {len(tool_calls)} 个add操作（删除全部物体）")
                    else:
                        # 非最后一轮：正常生成操作
                        # 计算当前步骤进度（逆向：从0.0到1.0）
                        step_progress = turn_idx / (target_turns - 1) if target_turns > 1 else 0.0
                        
                        print(f"        生成第 {turn_idx + 1} 轮操作（目标场景有 {len(current_objects)} 个物体，进度 {step_progress:.2f}）")
                        
                        # 确保使用原始格式（防止被add_random_variations内部覆盖）
                        editor.original_format = original_scene_format
                        
                        # 使用add_random_variations方法，操作数量为1到物体数量之间
                        # 注意：传递已标准化的场景，避免格式问题
                        initial_scene_for_turn, tool_calls = editor.add_random_variations(
                            normalized_target_scene,  # 使用标准化后的场景
                            current_subscene_name="",  # 不使用terminate
                            num_operations=0,  # 传入0表示使用随机数量（1到物体数量）
                            favor_add=True,  # 多轮对话模式，35%的add操作
                            step_progress=step_progress  # 传递进度信息用于分阶段选择
                        )
                    
                    if not tool_calls:
                        print(f"        第 {turn_idx + 1} 轮: 无法生成有效操作，跳过")
                        continue
                    
                    # 后处理目标场景，标记LLM无法预知的JID
                    processed_target_scene = editor.process_final_scene_jids(current_target_scene, tool_calls)
                    
                    # 将这一轮的数据加入对话链（注意：我们是逆向构建的）
                    conversation_steps.append({
                        "initial_scene": initial_scene_for_turn,
                        "tool_calls": tool_calls,
                        "target_scene": processed_target_scene
                    })
                    
                    # 更新下一轮的目标场景：使用当前轮的初始场景
                    current_target_scene = copy.deepcopy(initial_scene_for_turn)
                    
                    print(f"        第 {turn_idx + 1} 轮完成: 生成了 {len(tool_calls)} 个操作")
                
                # 验证是否成功到达空场景
                # 先标准化场景格式以确保可以访问groups
                final_check_scene = normalize_scene_format(current_target_scene)
                final_objects = []
                for group in final_check_scene["groups"]:
                    final_objects.extend(group["objects"])
                
                if final_objects:
                    print(f"      ✗ 尝试 {attempt_count} 意外错误：未能到达空场景（剩余 {len(final_objects)} 个物体）")
                    continue
                
                if not conversation_steps:
                    print(f"      ✗ 尝试 {attempt_count} 无有效步骤，重试")
                    continue
                
                print(f"      ✓ 尝试 {attempt_count} 成功: {len(conversation_steps)} 轮编辑，成功到达空场景")
                
                # 逆转对话步骤顺序，使其从空场景开始到最终场景结束
                conversation_steps.reverse()
                
                # 构建多轮对话消息
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                
                # 构建每一轮的用户-助手对话
                for step_idx, step_data in enumerate(conversation_steps):
                    initial_scene = step_data["initial_scene"]
                    tool_calls = step_data["tool_calls"]
                    
                    # 统一转换为flat格式（SSR格式），确保所有中间场景都是flat格式
                    initial_scene_formatted = denormalize_scene_format(initial_scene, "ssr")
                    
                    # 用户消息：显示当前场景状态
                    user_content = f"<image>\n{chain_user_instruction}\n\n<current_scene>\n```json\n{json.dumps(initial_scene_formatted, indent=2, ensure_ascii=False)}\n```\n</current_scene>"
                    
                    messages.append({
                        "role": "user",
                        "content": user_content
                    })
                    
                    # 助手消息：包含工具调用
                    tool_calls_content = ""
                    if tool_calls:
                        tool_calls_json = json.dumps(tool_calls, indent=2, ensure_ascii=False)
                        tool_calls_content = f"\n\n<tool_calls>\n{tool_calls_json}\n</tool_calls>"
                    
                    messages.append({
                        "role": "assistant",
                        "content": f"<think>\n{{THINK_PROCESS}}\n</think>\n\n<conclusion>\n</conclusion>{tool_calls_content}"
                    })
                    
                    print(f"        构建对话轮 {step_idx + 1}: {len(tool_calls)} 个操作")
                
                # 添加最后一轮（显示最终场景并终止）
                # 统一使用flat格式（SSR格式）
                final_scene_flat = denormalize_scene_format(final_scene, "ssr") if "groups" in final_scene else final_scene
                final_user_content = f"<image>\n{chain_user_instruction}\n\n<current_scene>\n```json\n{json.dumps(final_scene_flat, indent=2, ensure_ascii=False)}\n```\n</current_scene>"
                
                messages.append({
                    "role": "user",
                    "content": final_user_content
                })
                
                # 终止操作
                terminate_tool_call = {
                    "id": "tool_1",
                    "name": "terminate",
                    "arguments": {
                        "reason": "The current scene layout is already optimal and matches the user's requirements perfectly. No further modifications are needed."
                    }
                }
                
                terminate_tool_calls_content = f"\n\n<tool_calls>\n[{json.dumps(terminate_tool_call, indent=2, ensure_ascii=False)}]\n</tool_calls>"
                
                messages.append({
                    "role": "assistant",
                    "content": f"<think>\n{{THINK_PROCESS}}\n</think>\n\n<conclusion>\n</conclusion>{terminate_tool_calls_content}"
                })
                
                total_turns = len(conversation_steps) + 1  # +1 for terminate turn
                
                # 创建对话数据结构
                conversation_data = {
                    "id": f"multi_turn_edit_{success_count + 1}_{uuid.uuid4().hex[:8]}",
                    "images": [f"images/scene_{scene_id}_chain_{success_count + 1}_step_{i}.png" for i in range(total_turns)],
                    "messages": messages,
                    "metadata": {
                        "scene_id": scene_id,
                        "conversation_type": "multi_turn",
                        "chain_id": success_count + 1,
                        "total_turns": total_turns,
                        "generated_at": str(uuid.uuid4()),
                        "final_scene": denormalize_scene_format(final_scene, "ssr") if "groups" in final_scene else final_scene,
                        "global_user_instruction": chain_user_instruction,
                        "generation_method": "reverse_engineering"
                    }
                }
                
                # 保存对话链文件到场景专属文件夹
                chain_filename = f"chain_{success_count + 1}.json"
                chain_output_path = scene_output_dir / chain_filename
                
                with open(chain_output_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation_data, f, indent=2, ensure_ascii=False)
                
                success_count += 1
                print(f"      ✓ 编辑链 {success_count} 保存成功: {chain_filename} ({total_turns} 轮)")
                
            except Exception as e:
                print(f"      ✗ 尝试 {attempt_count} 生成失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if success_count < 3:
            print(f"    ⚠️  警告：达到最大尝试次数 ({max_attempts})，仅成功生成 {success_count}/3 条编辑链")
        else:
            print(f"    ✓ 多轮对话生成完成: 成功生成 {success_count}/3 条编辑链")
        
        return success_count > 0
        
    except Exception as e:
        print(f"生成多轮对话时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_intermediate_json(final_scene_path: str, output_path: str, global_user_instruction: str, assets_file_path: str = None) -> bool:
    """
    从最终场景文件生成对话模板数据
    支持两种场景格式
    """
    try:
        # 加载最终场景
        with open(final_scene_path, 'r', encoding='utf-8') as f:
            final_scene = json.load(f)
        
        # 检测场景格式
        scene_format = detect_scene_format(final_scene)
        print(f"    检测到场景格式: {scene_format}")
        
        # 创建场景编辑器，传入资产文件路径
        editor = SceneReverseEditor(assets_file_path)
        
        # 提取子场景名称（从文件路径）
        scene_id = Path(final_scene_path).stem
        
        # 生成原始场景和工具调用，传入当前子场景名称
        initial_scene, tool_calls = editor.add_random_variations(final_scene, scene_id)
        
        # 后处理final_scene，标记LLM无法预知的JID
        processed_final_scene = editor.process_final_scene_jids(final_scene, tool_calls)
        
        # 生成对话格式的模板数据
        # scene_id 已经在上面定义了
        
        # 创建system prompt
        system_prompt = """You are an expert interior design assistant specializing in spatial layout optimization and furniture arrangement. Your role is to help users transform their living spaces through thoughtful design decisions.

Key capabilities:
- Analyze room layouts and identify optimization opportunities
- Recommend furniture placement for improved functionality and flow
- Suggest appropriate furniture pieces for specific needs and spaces
- Apply design principles including balance, proportion, and visual harmony
- Consider practical factors like traffic patterns, lighting, and room usage
- Provide clear reasoning for all design recommendations

When working with users:
- Listen carefully to their specific needs and preferences
- Ask clarifying questions when requirements are unclear
- Explain your design reasoning step by step
- Consider both aesthetic and functional aspects
- Respect budget constraints and existing furniture when specified
- Provide multiple options when appropriate

Your goal is to create spaces that are both beautiful and highly functional for the user's lifestyle.

## OUTPUT FORMAT REQUIREMENTS

You must structure your response using the following special tokens in this exact order:

1. **Thinking Process**: Wrap your analysis and reasoning in <think></think> tags
2. **Conclusion**: Wrap a one-sentence summary of your thinking process in <conclusion></conclusion> tags
3. **Tool Calls**: Wrap your actions in <tool_calls></tool_calls> tags

Example format:
<think>
Your detailed analysis and reasoning process here...
</think>

<conclusion>
Wrap a one-sentence summary of your thinking process
</conclusion>

<tool_calls>
[
  {
    "id": "tool_1",
    "name": "action_name",
    "arguments": {...}
  }
]
</tool_calls>

## AVAILABLE TOOLS

You have access to the following tools for scene editing:

### 1. add_object
Add a new furniture piece to the scene.
**Parameters:**
- object_description (string): Detailed description of the furniture piece
- position (array): [x, y, z] coordinates in the room
- rotation (array): [x, y, z, w] quaternion rotation
- size (array): [width, height, depth] dimensions
- group_name (string): Name of the group this object belongs to

### 2. remove_object
Remove an existing object from the scene.
**Parameters:**
- jid (string): Unique identifier of the object to remove

### 3. move_object
Change the position of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to move
- new_position (array): [x, y, z] new coordinates

### 4. rotate_object
Change the rotation of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to rotate
- new_rotation (array): [x, y, z, w] new quaternion rotation

### 5. scale_object
Change the size of an existing object.
**Parameters:**
- jid (string): Unique identifier of the object to scale
- new_size (array): [width, height, depth] new dimensions

### 6. replace_object
Replace an existing object with a different object type.
**Parameters:**
- jid_to_replace (string): Unique identifier of the object to replace
- new_object_description (string): Description of the new object to place

### 7. terminate
End the editing session when the scene is complete.
**Parameters:**
- reason (string): Explanation for why editing is complete

## CRITICAL SPATIAL AWARENESS REQUIREMENTS

**MANDATORY**: You must carefully observe the rendered top-view image and analyze the <current_scene> data to ensure proper spatial arrangement.

### Overlap Prevention:
- **NO OVERLAPPING**: Objects must never occupy the same 3D space
- Check object positions and sizes to ensure clear separation
- Maintain minimum 0.1 meter clearance between objects when possible
- If you detect overlapping objects in the current scene, you MUST fix them by moving one or both objects

### Boundary Compliance:
- **NO OUT-OF-BOUNDS**: All objects must stay within the room_envelope boundaries
- Check that object position + half of object size stays within room bounds
- For position [x,y,z] and size [w,h,d], ensure:
  - x ± w/2 stays within room x-bounds
  - z ± d/2 stays within room z-bounds
- If you detect out-of-bounds objects, you MUST move them to valid positions

### Editing Validation:
- Before any move/add operation, calculate the final object boundaries
- Verify no conflicts with existing objects
- Verify the object stays within room boundaries
- If conflicts exist, choose different positions or modify other objects first
- Priority: Fix existing violations before adding new objects

### Top-View Analysis:
- Study the provided rendered image carefully
- The top-view shows the actual spatial relationships between objects
- Use this visual information to identify spatial problems
- Cross-reference with the numerical data in <current_scene>
- Trust the visual evidence when planning modifications"""

        # 格式化tool_calls为content中的<tool_calls>部分
        tool_calls_content = ""
        if tool_calls:
            tool_calls_json = json.dumps(tool_calls, indent=2, ensure_ascii=False)
            tool_calls_content = f"\n\n<tool_calls>\n{tool_calls_json}\n</tool_calls>"

        conversation_template = {
            "id": f"scene_edit_{uuid.uuid4().hex[:8]}",
            "images": [f"images/scene_{scene_id}_initial.png"],
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"<image>\n{global_user_instruction}\n\n<current_scene>\n```json\n{json.dumps(initial_scene, indent=2, ensure_ascii=False)}\n```\n</current_scene>"
                },
                {
                    "role": "assistant",
                    "content": f"<think>\n{{THINK_PROCESS}}\n</think>\n\n<conclusion>\n</conclusion>{tool_calls_content}"
                }
            ],
            "metadata": {
                "scene_id": scene_id,
                "num_operations": len(tool_calls),
                "operation_types": [call["name"] for call in tool_calls],
                "generated_at": str(uuid.uuid4()),
                "initial_scene": initial_scene,
                "final_scene": processed_final_scene,
                "global_user_instruction": global_user_instruction
            }
        }
        
        # 保存对话模板数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_template, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"生成中间数据时出错: {e}")
        return False

def process_scenes_batch(input_dir: str, output_dir: str, assets_file_path: str = None, max_files: int = None, instruction_type: str = "mixed", generation_mode: str = "both", resume: bool = False):
    """批量处理场景文件生成对话模板数据 - 支持两种输入模式
    
    模式1: 嵌套目录结构 (以_labeled结尾的目录，包含subscene_*.json文件)
    模式2: 扁平JSON文件 (直接包含场景JSON文件的目录)
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录  
        assets_file_path: 资产文件路径
        max_files: 最大处理文件数
        instruction_type: 指令类型
        generation_mode: 生成模式 - "single"(仅单轮), "multi"(仅多轮), "both"(两种都生成)
        resume: 是否从断点恢复，跳过已经处理的文件
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = Path(input_dir)
    
    # 检测输入模式
    # 如果有_labeled目录，使用嵌套目录模式
    # 否则检查是否有JSON文件，使用扁平文件模式
    scene_directories = []
    json_files = []
    
    for item in input_path.iterdir():
        if item.is_dir() and item.name.endswith('_labeled'):
            scene_directories.append(item)
    
    if not scene_directories:
        # 没有_labeled目录，尝试扁平文件模式
        for item in input_path.iterdir():
            if item.is_file() and item.suffix == '.json':
                json_files.append(item)
    
    # 根据检测到的模式处理
    if scene_directories:
        print(f"检测到嵌套目录模式: 找到 {len(scene_directories)} 个场景目录")
        return _process_nested_directories(scene_directories, output_dir, assets_file_path, max_files, instruction_type, generation_mode, resume)
    elif json_files:
        print(f"检测到扁平文件模式: 找到 {len(json_files)} 个JSON文件")
        return _process_flat_json_files(json_files, output_dir, assets_file_path, max_files, instruction_type, generation_mode, resume)
    else:
        print("错误: 未找到有效的场景文件或目录")
        return

def _process_flat_json_files(json_files: List[Path], output_dir: str, assets_file_path: str = None, max_files: int = None, instruction_type: str = "mixed", generation_mode: str = "both", resume: bool = False):
    """处理扁平的JSON文件列表"""
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"将处理 {len(json_files)} 个JSON文件")
    if assets_file_path:
        print(f"使用资产文件: {assets_file_path}")
    
    if resume:
        print("启用断点续传模式")
    
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, json_file in enumerate(json_files):
        print(f"\n处理场景 {i+1}/{len(json_files)}: {json_file.name}")
        
        try:
            # 检查是否已处理
            output_filename = f"intermediate_{json_file.name}"
            output_path = Path(output_dir) / output_filename
            
            if resume and output_path.exists():
                print(f"  → 已存在，跳过")
                skipped_count += 1
                continue
            
            # 加载场景生成全局指令
            with open(json_file, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            print(f"  生成全局用户指令...")
            global_user_instruction = generate_global_user_instruction(scene_data, instruction_type)
            print(f"  全局指令: {global_user_instruction}")
            
            # 根据generation_mode生成对应的数据
            if generation_mode == 'single':
                # 仅生成单轮对话
                if generate_intermediate_json(str(json_file), str(output_path), global_user_instruction, assets_file_path):
                    successful_count += 1
                    print(f"  ✓ 成功生成单轮对话: {output_filename}")
                else:
                    failed_count += 1
                    print(f"  ✗ 单轮对话生成失败")
                    
            elif generation_mode == 'multi':
                # 仅生成多轮对话
                multi_turn_output_filename = f"multi_turn_{json_file.name}"
                multi_turn_output_path = Path(output_dir) / multi_turn_output_filename
                
                if resume and multi_turn_output_path.exists():
                    print(f"  → 多轮对话已存在，跳过")
                    skipped_count += 1
                    continue
                
                if generate_multi_turn_conversation(str(json_file), str(multi_turn_output_path), global_user_instruction, assets_file_path):
                    successful_count += 1
                    print(f"  ✓ 成功生成多轮对话")
                else:
                    failed_count += 1
                    print(f"  ✗ 多轮对话生成失败")
                    
            else:  # both
                # 生成单轮对话
                single_success = generate_intermediate_json(str(json_file), str(output_path), global_user_instruction, assets_file_path)
                if single_success:
                    print(f"  ✓ 成功生成单轮对话: {output_filename}")
                else:
                    print(f"  ✗ 单轮对话生成失败")
                
                # 生成多轮对话
                multi_turn_output_filename = f"multi_turn_{json_file.name}"
                multi_turn_output_path = Path(output_dir) / multi_turn_output_filename
                
                multi_success = generate_multi_turn_conversation(str(json_file), str(multi_turn_output_path), global_user_instruction, assets_file_path)
                if multi_success:
                    print(f"  ✓ 成功生成多轮对话")
                else:
                    print(f"  ✗ 多轮对话生成失败")
                
                # 只要有一个成功就算成功
                if single_success or multi_success:
                    successful_count += 1
                else:
                    failed_count += 1
                
        except Exception as e:
            failed_count += 1
            print(f"  ✗ 处理文件时出错: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n处理完成!")
    print(f"总文件数: {len(json_files)}")
    print(f"成功: {successful_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skipped_count}")
    print(f"完成率: {(successful_count + skipped_count) / len(json_files) * 100:.1f}%" if json_files else "完成率: 0%")

def _process_nested_directories(scene_directories: List[Path], output_dir: str, assets_file_path: str = None, max_files: int = None, instruction_type: str = "mixed", generation_mode: str = "both", resume: bool = False):
    """处理嵌套目录结构（原有逻辑）"""
    
    # 获取所有场景目录（以_labeled结尾的目录）
    if assets_file_path:
        print(f"使用资产文件: {assets_file_path}")
    
    if resume:
        print("\n开始批量检查已生成文件状态...")
    
    # 批量分析需要处理的任务
    tasks_to_process = []
    total_subscenes = 0
    already_completed = 0
    
    for i, scene_dir in enumerate(scene_directories):
        if i % 100 == 0:  # 每100个目录打印一次进度
            print(f"  检查进度: {i+1}/{len(scene_directories)}")
        
        # 为每个场景创建对应的输出目录
        scene_output_dir = Path(output_dir) / scene_dir.name
        
        # 获取该场景目录下的所有 subscene 文件
        subscene_files = list(scene_dir.glob("subscene_*.json"))
        subscene_files.sort()
        total_subscenes += len(subscene_files)
        
        # 检查多轮对话是否需要生成（根据generation_mode参数）
        target_scene_file = scene_dir / "subscene_000.json"
        multi_turn_needed = False
        if target_scene_file.exists() and generation_mode in ['multi', 'both']:
            # 检查多轮对话链文件是否已存在（检查第一个链文件即可）
            multi_turn_output_dir = Path(output_dir).parent / "intermediate_data_multi_turn" / scene_dir.name
            multi_turn_chain_1_path = multi_turn_output_dir / f"multi_turn_{target_scene_file.stem}_chain_1.json"
            if not resume or not multi_turn_chain_1_path.exists():
                multi_turn_needed = True
            else:
                already_completed += 1
        
        # 检查哪些子场景文件需要处理（根据generation_mode参数）
        subscenes_to_process = []
        if generation_mode in ['single', 'both']:
            for subscene_file in subscene_files:
                output_filename = f"intermediate_{subscene_file.name}"
                output_path = scene_output_dir / output_filename
                
                if not resume or not output_path.exists():
                    subscenes_to_process.append(subscene_file)
                else:
                    already_completed += 1
        
        # 如果有任务需要处理，添加到任务列表
        if multi_turn_needed or subscenes_to_process:
            tasks_to_process.append({
                'scene_dir': scene_dir,
                'scene_output_dir': scene_output_dir,
                'target_scene_file': target_scene_file,
                'multi_turn_needed': multi_turn_needed,
                'subscenes_to_process': subscenes_to_process,
                'total_subscenes': len(subscene_files)
            })
    
    print(f"\n批量检查完成!")
    print(f"生成模式: {generation_mode}")
    print(f"总场景目录: {len(scene_directories)}")
    print(f"总子场景文件: {total_subscenes}")
    print(f"已完成: {already_completed}")
    print(f"需要处理的场景目录: {len(tasks_to_process)}")
    
    # 根据生成模式统计任务数量
    single_turn_tasks = sum(len(task['subscenes_to_process']) for task in tasks_to_process)
    multi_turn_tasks = sum(1 if task['multi_turn_needed'] else 0 for task in tasks_to_process)
    
    if generation_mode == 'single':
        print(f"需要处理的单轮对话任务数: {single_turn_tasks}")
    elif generation_mode == 'multi':
        print(f"需要处理的多轮对话任务数: {multi_turn_tasks}")
    else:  # both
        print(f"需要处理的单轮对话任务数: {single_turn_tasks}")
        print(f"需要处理的多轮对话任务数: {multi_turn_tasks}")
        print(f"总任务数: {single_turn_tasks + multi_turn_tasks}")
    
    if not tasks_to_process:
        print("所有文件都已生成，无需处理!")
        return
    
    # 现在开始实际处理
    successful_count = 0
    failed_count = 0
    skipped_count = already_completed
    
    for i, task in enumerate(tasks_to_process):
        scene_dir = task['scene_dir']
        scene_output_dir = task['scene_output_dir']
        target_scene_file = task['target_scene_file']
        multi_turn_needed = task['multi_turn_needed']
        subscenes_to_process = task['subscenes_to_process']
        
        print(f"\n处理场景目录 {i+1}/{len(tasks_to_process)}: {scene_dir.name}")
        print(f"  需要处理 {len(subscenes_to_process)} 个子场景文件 + {'1个多轮对话' if multi_turn_needed else '0个多轮对话'}")
        
        # 确保输出目录存在
        scene_output_dir.mkdir(exist_ok=True)
        
        
        # 为这个场景生成全局用户指令（基于subscene_000.json，即最终目标场景）
        global_user_instruction = None
        
        if target_scene_file.exists():
            try:
                # 只有在需要生成内容时才生成全局指令
                if multi_turn_needed or subscenes_to_process:
                    print(f"  基于 {target_scene_file.name} 生成全局用户指令...")
                    with open(target_scene_file, 'r', encoding='utf-8') as f:
                        target_scene = json.load(f)
                    global_user_instruction = generate_global_user_instruction(target_scene, instruction_type)
                    print(f"  全局用户指令: {global_user_instruction}")
                
                # 处理多轮对话生成
                if multi_turn_needed:
                    multi_turn_output_filename = f"multi_turn_{target_scene_file.stem}.json"
                    multi_turn_output_path = Path(output_dir).parent / "intermediate_data_multi_turn" / scene_dir.name / multi_turn_output_filename
                    multi_turn_output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    print(f"  生成多轮对话: {multi_turn_output_filename}")
                    if generate_multi_turn_conversation(str(target_scene_file), str(multi_turn_output_path), global_user_instruction, assets_file_path):
                        print(f"    ✓ 成功生成多轮对话: {multi_turn_output_filename}")
                        successful_count += 1
                    else:
                        print(f"    ✗ 多轮对话生成失败: {multi_turn_output_filename}")
                        failed_count += 1
                    
            except Exception as e:
                print(f"  ✗ 生成全局用户指令失败: {e}")
                global_user_instruction = "Help me design a complete and functional room layout."
                failed_count += 1
        else:
            print(f"  ✗ 未找到目标场景文件 {target_scene_file.name}，使用默认指令")
            global_user_instruction = "Help me design a complete and functional room layout."
        
        # 处理该场景下需要生成的子场景文件
        for j, subscene_file in enumerate(subscenes_to_process):
            try:
                print(f"  处理子场景 {j+1}/{len(subscenes_to_process)}: {subscene_file.name}")
                
                # 生成输出文件名
                output_filename = f"intermediate_{subscene_file.name}"
                output_path = scene_output_dir / output_filename
                
                # 生成中间数据，传入全局用户指令和资产文件路径
                if generate_intermediate_json(str(subscene_file), str(output_path), global_user_instruction, assets_file_path):
                    successful_count += 1
                    print(f"    ✓ 成功生成: {output_filename}")
                else:
                    failed_count += 1
                    print(f"    ✗ 生成失败: {subscene_file.name}")
                
            except Exception as e:
                failed_count += 1
                print(f"    ✗ 处理文件时出错 {subscene_file.name}: {e}")
    
    print(f"\n处理完成!")
    print(f"生成模式: {generation_mode}")
    print(f"总场景目录: {len(scene_directories)}")
    
    if generation_mode == 'single':
        print(f"总单轮对话文件: {total_subscenes}")
        print(f"成功: {successful_count}")
        print(f"失败: {failed_count}")
        print(f"跳过: {skipped_count}")
        print(f"完成率: {(successful_count + skipped_count) / total_subscenes * 100:.1f}%" if total_subscenes > 0 else "完成率: 0%")
    elif generation_mode == 'multi':
        print(f"多轮对话任务完成")
        print(f"成功: {successful_count}")
        print(f"失败: {failed_count}")
        print(f"跳过: {skipped_count}")
    else:  # both
        print(f"总子场景文件: {total_subscenes}")
        print(f"成功: {successful_count}")
        print(f"失败: {failed_count}")
        print(f"跳过: {skipped_count}")
        print(f"完成率: {(successful_count + skipped_count) / total_subscenes * 100:.1f}%" if total_subscenes > 0 else "完成率: 0%")

def main():
    """主函数"""
    import argparse
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='生成中间数据文件')
    parser.add_argument('--instruction-type', choices=['detailed', 'brief', 'mixed'], default='mixed',
                       help='指令类型: detailed (详细指令), brief (简短指令), 或 mixed (混合，30%%简短+70%%详细)')
    parser.add_argument('--generation-mode', choices=['single', 'multi', 'both'], default='multi',
                       help='生成模式: single (仅单轮对话), multi (仅多轮对话), both (生成两种)')
    parser.add_argument('--resume', action='store_true',
                       help='从断点继续，跳过已经生成的文件')
    args = parser.parse_args()
    
    # 配置路径
    INPUT_DIRECTORY = r"/path/to/datasets/ssr/scenes_filtered"
    OUTPUT_DIRECTORY = r"/path/to/datasets/llmscene/intermediate_data_v2"
    ASSETS_FILE = r"/path/to/SceneReVis/metadata/model_info_3dfuture_assets_prompts.json"
    # 移除文件数量限制，处理所有文件
    
    print(f"使用指令类型: {args.instruction_type}")
    print(f"生成模式: {args.generation_mode}")
    if args.resume:
        print("启用断点续传模式，将跳过已存在的文件")
    
    # 处理场景文件
    process_scenes_batch(INPUT_DIRECTORY, OUTPUT_DIRECTORY, ASSETS_FILE, max_files=None, instruction_type=args.instruction_type, generation_mode=args.generation_mode, resume=args.resume)

if __name__ == "__main__":
    main()
