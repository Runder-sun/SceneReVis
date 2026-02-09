#!/usr/bin/env python3
"""
VLM Scene Evaluation - 使用 GPT-4o Vision 进行场景多维度评估
评估维度：合理性(rationality)、空间layout(spatial_layout)、可达性(accessibility)
评分范围：0-100 百分制 (Implementation uses 1-10 scale internally)
"""

import os
import sys
import json
import base64
import asyncio
import argparse
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image

# 目标分辨率
TARGET_RESOLUTION = (1080, 570)

# Azure OpenAI imports
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

# Azure OpenAI配置 (与 infer.py 保持一致)
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
AZURE_OPENAI_SCOPE = "YOUR_AZURE_OPENAI_SCOPE"


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


def image_to_base64(image_path: str, target_resolution: tuple = TARGET_RESOLUTION) -> str:
    """将图像文件转换为base64编码的data URL，并统一分辨率"""
    # 打开图片并调整分辨率
    with Image.open(image_path) as img:
        # 转换为RGB模式（如果是RGBA）
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # 调整分辨率
        if img.size != target_resolution:
            img = img.resize(target_resolution, Image.Resampling.LANCZOS)
        
        # 保存到内存buffer
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_data}"


def call_gpt4o_vision(
    client: AzureOpenAI,
    image_path: str,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.1,
    max_retries: int = 3
) -> Optional[str]:
    """
    调用 GPT-4o Vision API
    
    Args:
        client: AzureOpenAI 客户端
        image_path: 图像文件路径
        prompt: 提示文本
        max_tokens: 最大生成token数
        temperature: 温度参数
        max_retries: 最大重试次数
        
    Returns:
        模型响应文本，或 None（如果失败）
    """
    if not Path(image_path).exists():
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    try:
        img_base64 = image_to_base64(image_path)
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64,
                        "detail": "high"
                    }
                }
            ]
        }
    ]
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API call attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)  # 指数退避
    
    return None


def parse_score_from_response(response: str) -> Optional[int]:
    """
    从模型响应中解析 1-10 的分数
    
    Args:
        response: 模型响应文本
        
    Returns:
        1-10 的整数分数，或 None（如果解析失败）
    """
    if response is None:
        return None
    # 匹配 1-10 分
    patterns = [
        r'\b([1-9]|10)\s*/\s*10\b',  # "8/10"
        r'\bscore[:\s]*([1-9]|10)\b',
        r'\b([1-9]|10)\s*(?:points?|分)\b',
        r'^([1-9]|10)$',
        r'\b([1-9]|10)\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                return score
    return None


def generate_scene_description(
    client: AzureOpenAI,
    image_path: str,
    user_requirement: str
) -> Optional[str]:
    """
    第一阶段：让 GPT-4o 详细描述场景
    
    Args:
        client: AzureOpenAI 客户端
        image_path: 渲染图路径
        user_requirement: 用户需求
        
    Returns:
        详细场景描述文本，或 None（如果失败）
    """
    prompt_describe = f"""You are a highly critical interior design expert. Carefully examine this final scene rendering (left: top view, right: diagonal view).

**IMPORTANT - USE VISUAL ANNOTATIONS IN THE IMAGE:**
- **Floor coordinate grid**: The TOP VIEW (left) shows a coordinate grid on the floor with axis labels. Use this to precisely describe object positions.
- **Bounding boxes**: Each object has a colored bounding box (bbox) drawn around it. Use these to describe object sizes and spatial relationships.

User requirement: {user_requirement}

**YOUR TASK: Provide a comprehensive, detailed description of this scene.**

Describe the scene thoroughly, covering ALL of the following aspects:

## 1. OBJECT INVENTORY
List every visible object in the scene:
- Object name/type
- Approximate position in the room (use coordinate references from TOP VIEW)
- Size (small/medium/large)
- Color/material if visible

## 2. SPATIAL LAYOUT
- Room dimensions and shape (from floor grid)
- How objects are distributed across the room
- Which areas are occupied vs empty
- Overall layout pattern (clustered, spread out, L-shaped arrangement, etc.)

## 3. OBJECT RELATIONSHIPS
- Which objects are grouped together (functional groups)
- Distances between key furniture pieces
- Facing directions (what faces what)
- Supporting relationships (what's on top of what)

## 4. ORIENTATIONS
- Direction each major furniture piece faces
- Whether orientations make functional sense
- Any furniture facing walls or corners inappropriately

## 5. WALL PROXIMITY
- Which furniture is against walls (and which walls)
- Which furniture is floating in open space
- Distance estimates from walls for key pieces

## 6. IDENTIFIED PROBLEMS
Based on your detailed analysis, list any problems found:

**Physical Issues:**
- Any overlapping objects (bboxes intersecting)
- Any out-of-bounds objects
- Any floating objects

**Rationality Issues:**
- Any misplaced core furniture (bed/sofa not against wall)
- Any missing essential items
- Any wrong orientations

**Distribution Issues:**
- Any clustering problems
- Any large empty areas
- Any layout imbalance

## 7. OVERALL ASSESSMENT
- Does the scene fulfill the user's requirement?
- What works well in this design?
- What are the most critical issues to address?
- Overall quality rating: EXCELLENT / GOOD / ACCEPTABLE / POOR / VERY POOR

Be thorough and specific in your description. Reference the coordinate grid and bounding boxes for precise locations."""

    return call_gpt4o_vision(client, image_path, prompt_describe, max_tokens=1500, temperature=0.2)


def evaluate_rationality(
    client: AzureOpenAI,
    image_path: str,
    user_requirement: str,
    scene_metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    评估场景合理性（1-10分）
    只评估：碰撞、出界、悬空、房间大小、物体数量
    """
    # 构建元数据信息
    metadata_info = ""
    room_size_warning = ""
    object_count_warning = ""
    if scene_metadata:
        room_type = scene_metadata.get('room_type', 'unknown')
        room_area = scene_metadata.get('room_area', 0)
        min_area, max_area = scene_metadata.get('room_size_range', (10, 40))
        room_size_valid = scene_metadata.get('room_size_valid', True)
        object_count = scene_metadata.get('object_count', 0)
        
        metadata_info = f"""
**SCENE METADATA (from JSON):**
- Room Type: {room_type}
- Room Size: {scene_metadata.get('room_width', 0):.2f}m x {scene_metadata.get('room_depth', 0):.2f}m (Area: {room_area:.2f} m²)
- Expected Size Range for {room_type}: {min_area}-{max_area} m²
- Object Count: {object_count}
- Objects: {', '.join(scene_metadata.get('object_list', []))}
"""
        
        if not room_size_valid:
            if room_area < min_area:
                room_size_warning = f"\n**⚠️ ROOM SIZE WARNING: Room is TOO SMALL ({room_area:.1f} m² < {min_area} m²). MAX SCORE: 6**"
            else:
                room_size_warning = f"\n**⚠️ ROOM SIZE WARNING: Room is TOO LARGE ({room_area:.1f} m² > {max_area} m²). MAX SCORE: 6**"
        
        # 检查物体数量是否过少
        if object_count < 4:
            object_count_warning = f"\n**⚠️ OBJECT COUNT WARNING: Scene is TOO SIMPLE (only {object_count} objects, minimum 4 required). MAX SCORE: 6**"
    
    prompt = f"""You are a highly critical interior design expert evaluating scene RATIONALITY (score 1-10).

**CONTEXT:**
User requirement: {user_requirement}
{metadata_info}{room_size_warning}{object_count_warning}

**TASK: Evaluate RATIONALITY based ONLY on PHYSICAL ISSUES, ROOM SIZE, and OBJECT COUNT. Give a score from 1 to 10.**

**FOCUS ONLY ON THESE ASPECTS:**
1. **Collisions**: Are there any objects overlapping/intersecting each other?
2. **Out-of-Bounds (OOB)**: Are any objects extending beyond the room boundaries? Check TOP VIEW carefully.
3. **Floating Objects**: Are any objects floating in the air (not touching floor/table/etc.)?
4. **Room Size Validity**: Is the room area within the expected range for this room type?
   - Bedroom: 10-25 m²
   - Living Room: 15-35 m²
   - Dining Room: 10-25 m²
   - Study Room: 10-25 m²
5. **Object Count**: Does the scene have at least 4 objects? Scenes with fewer than 4 objects are too simple.

**DO NOT EVALUATE:** Furniture placement logic, functional layout, style, or aesthetics.

**SCORING GUIDELINES (1-10):**
- **9-10**: Zero physical issues (no collision, no OOB, no floating), room size is valid, at least 4 objects.
- **7-8**: Very minor physical issues (slight overlap), room size is acceptable, sufficient objects.
- **5-6**: Some noticeable physical issues OR room size slightly outside range OR fewer than 4 objects.
- **3-4**: Multiple physical issues OR room size significantly outside range.
- **1-2**: Severe physical issues (major collisions, many OOB objects, floating furniture).

**IMPORTANT CONSTRAINTS:**
- If room size is outside the valid range, the MAXIMUM score is 6.
- If object count is less than 4, the MAXIMUM score is 6.

**Output ONLY a single integer from 1 to 10.**"""

    response = call_gpt4o_vision(client, image_path, prompt, max_tokens=20, temperature=0.1)
    return parse_score_from_response(response)



def evaluate_spatial_layout(
    client: AzureOpenAI,
    image_path: str,
    user_requirement: str,
    scene_metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    评估空间layout（1-10分）
    """
    # 构建元数据信息
    metadata_info = ""
    if scene_metadata:
        metadata_info = f"""
**SCENE METADATA (from JSON):**
- Room Type: {scene_metadata.get('room_type', 'unknown')}
- Room Size: {scene_metadata.get('room_width', 0):.2f}m x {scene_metadata.get('room_depth', 0):.2f}m (Area: {scene_metadata.get('room_area', 0):.2f} m²)
- Object Count: {scene_metadata.get('object_count', 0)}
- Objects: {', '.join(scene_metadata.get('object_list', []))}
"""
    
    prompt = f"""You are a highly critical interior design expert evaluating SPATIAL LAYOUT (score 1-10).

**CONTEXT:**
User requirement: {user_requirement}
{metadata_info}
**TASK: Evaluate the SPATIAL LAYOUT based on the provided image and metadata. Give a score from 1 to 10.**

**CRITICAL EVALUATION CRITERIA (in order of importance):**

1. **Boundary Compliance (HIGHEST PRIORITY)**:
   - Are ALL objects fully within the room boundaries? Check TOP VIEW carefully.
   - **SEVERE PENALTY (-3 points)**: Any object extending beyond room boundaries (OOB).

2. **Functional Zoning (NOT uniform coverage)**:
   - Are furniture pieces grouped into logical functional zones?
   - Does the layout support intended activities (sleeping zone, work zone, etc.)?
   - **BONUS**: Clear, purposeful zones with intentional spacing between them.
   - **NOTE**: Empty center/corners are ACCEPTABLE if furniture is properly zoned along walls.

3. **Anti-Crowding & Collision Avoidance**:
   - Is there adequate spacing between furniture pieces (at least 60cm)?
   - **SEVERE PENALTY**: Furniture pieces touching, overlapping, or crammed together.
   - **SEVERE PENALTY**: Cluttered areas where multiple objects compete for the same space.
   - **BONUS**: Generous spacing that allows easy movement and visual clarity.

4. **Intentional Negative Space**:
   - Professional interior design often leaves open floor areas for circulation and visual rest.
   - **DO NOT penalize** open center areas if furniture is well-organized along perimeter.
   - **DO penalize** when ALL furniture is piled in one corner/side, leaving 70%+ of room unused.

**WHAT SCORES HIGH (7-10):**
- All objects within boundaries with comfortable margins
- Furniture arranged in clear functional groups with spacing between them
- Open circulation paths through the room
- Even if center is open, furniture along walls is well-distributed

**WHAT SCORES LOW (3-5):**
- Objects extending beyond room boundaries (OOB)
- Furniture crammed together with no spacing
- Severe clustering where 70%+ of room is empty and all furniture is piled in one area
- Overlapping or colliding furniture pieces

**SCORING GUIDELINES (1-10):**
- **9-10**: All objects within bounds, excellent functional zoning, generous spacing, no crowding.
- **7-8**: Objects within bounds, good distribution, adequate spacing between pieces.
- **5-6**: Objects within bounds but some crowding issues or mild imbalance.
- **3-4**: OOB issues, significant crowding, OR extreme imbalance (all furniture in one corner).
- **1-2**: Multiple OOB objects, severe collisions, or completely dysfunctional distribution.

Look at TOP VIEW (left side of image) to verify boundaries and distribution.

**Output ONLY a single integer from 1 to 10.**"""

    response = call_gpt4o_vision(client, image_path, prompt, max_tokens=20, temperature=0.1)
    return parse_score_from_response(response)


def evaluate_accessibility(
    client: AzureOpenAI,
    image_path: str,
    user_requirement: str,
    scene_metadata: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    评估场景可达性（1-10分）
    """
    # 构建元数据信息
    metadata_info = ""
    if scene_metadata:
        metadata_info = f"""
**SCENE METADATA (from JSON):**
- Room Type: {scene_metadata.get('room_type', 'unknown')}
- Room Size: {scene_metadata.get('room_width', 0):.2f}m x {scene_metadata.get('room_depth', 0):.2f}m (Area: {scene_metadata.get('room_area', 0):.2f} m²)
- Object Count: {scene_metadata.get('object_count', 0)}
- Objects: {', '.join(scene_metadata.get('object_list', []))}
"""
    
    prompt = f"""You are a highly critical interior design expert evaluating scene ACCESSIBILITY (score 1-10).

**CONTEXT:**
User requirement: {user_requirement}
{metadata_info}
**TASK: Evaluate ACCESSIBILITY based on the provided image and metadata. Give a score from 1 to 10.**

**CRITICAL EVALUATION CRITERIA (in order of importance):**

1. **Circulation Quality (HIGHEST PRIORITY)**:
   - Is there at least 80cm clearance for walking paths?
   - Are main pathways unobstructed and clearly defined?
   - **SEVERE PENALTY**: Furniture placed too close together (<60cm gaps) making movement difficult.
   - **SEVERE PENALTY**: Cluttered or cramped arrangements that impede natural flow.

2. **Spatial Breathing Room**:
   - Does the room feel open and comfortable, NOT cramped?
   - Are there adequate spaces between furniture pieces?
   - **BONUS**: Generous spacing that allows easy navigation and flexible use.

3. **Core Furniture Accessibility**:
   - Can the main furniture (bed/sofa) be accessed from multiple sides?
   - Is there clear space around key pieces for actual use?
   - **PENALTY**: Furniture pushed too tightly against walls or corners with no access.

4. **Practical Movement Patterns**:
   - Can a person naturally move through the space?
   - Are there logical traffic flow patterns?
   - **PENALTY**: Layouts requiring awkward navigation or detours.

**WHAT TO AVOID SCORING HIGH:**
- Overly dense arrangements where furniture is crammed together
- Scenes where every wall is lined with furniture leaving no breathing room
- Traditional layouts that sacrifice circulation for "completeness"

**SCORING GUIDELINES (1-10):**
- **9-10**: Excellent open flow, generous clearances, furniture easily accessible from multiple angles.
- **7-8**: Good circulation with comfortable spacing, minor tight spots acceptable.
- **5-6**: Adequate but some cramped areas or circulation issues.
- **3-4**: Cramped layout, difficult movement, furniture too close together.
- **1-2**: Severely blocked paths, unusable cramped space.

**Output ONLY a single integer from 1 to 10.**"""

    response = call_gpt4o_vision(client, image_path, prompt, max_tokens=20, temperature=0.1)
    return parse_score_from_response(response)


def clean_user_requirement(text: str) -> str:
    """
    清理用户需求文本，移除 <current_scene>...</current_scene> 标签
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    import re
    # 移除 <current_scene>...</current_scene> 标签及其内容
    cleaned = re.sub(r'<current_scene>.*?</current_scene>', '', text, flags=re.DOTALL)
    # 移除多余的空白
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def normalize_room_type(room_type_raw: str) -> str:
    """
    标准化房间类型名称
    
    Args:
        room_type_raw: 原始房间类型字符串
        
    Returns:
        标准化的房间类型 (bedroom, livingroom, diningroom, studyroom, etc.)
    """
    if not room_type_raw:
        return "unknown"
    
    room_type_lower = room_type_raw.lower()
    
    # 检查是否包含关键词
    if "bedroom" in room_type_lower or "bed" in room_type_lower:
        return "bedroom"
    elif "living" in room_type_lower:
        return "livingroom"
    elif "dining" in room_type_lower:
        return "diningroom"
    elif "study" in room_type_lower or "office" in room_type_lower:
        return "studyroom"
    elif "kitchen" in room_type_lower:
        return "kitchen"
    elif "bathroom" in room_type_lower:
        return "bathroom"
    
    # 如果是简单的房间类型名称
    simple_types = ["bedroom", "livingroom", "diningroom", "studyroom", "kitchen", "bathroom"]
    for t in simple_types:
        if t in room_type_lower.replace(" ", "").replace("_", ""):
            return t
    
    return room_type_raw


def get_room_size_range(room_type: str) -> Tuple[float, float]:
    """
    获取房间类型的合理面积范围
    
    Args:
        room_type: 标准化的房间类型
        
    Returns:
        (min_area, max_area) 元组，单位为平方米
    """
    room_size_ranges = {
        "bedroom": (10, 25),
        "livingroom": (15, 35),
        "diningroom": (10, 25),
        "studyroom": (10, 25),
        "kitchen": (6, 20),
        "bathroom": (4, 15),
    }
    return room_size_ranges.get(room_type, (10, 40))  # 默认范围


def parse_scene_json(json_path: str) -> Optional[Dict[str, Any]]:
    """
    解析场景JSON文件，提取房间元数据
    支持多种JSON命名格式：
    - bedroom_prompt_000.json (DiffuScene)
    - prompt_2.json (Holodeck)
    - 0001.json (IDesign)
    - 1.json (LayoutGPT/LayoutVLM)
    - scene_000.json (Respace)
    - prompt_3_final_scene.json (Ours)
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        包含房间元数据的字典，或 None（如果解析失败）
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 计算房间尺寸
        bounds_bottom = data.get('bounds_bottom', [])
        room_width = 0.0
        room_depth = 0.0
        room_area = 0.0
        
        if len(bounds_bottom) >= 4:
            # 从 bounds_bottom 计算房间尺寸
            # bounds_bottom 是四个角点的坐标 [x, y, z]
            xs = [p[0] for p in bounds_bottom]
            zs = [p[2] for p in bounds_bottom]
            room_width = max(xs) - min(xs)
            room_depth = max(zs) - min(zs)
            room_area = room_width * room_depth
        
        # 获取原始房间类型并标准化
        room_type_raw = data.get('room_type', 'unknown')
        # 如果 room_type 是长文本（如 Holodeck 的 prompt），尝试从文件名或 room_id 提取
        if len(room_type_raw) > 50:  # 太长说明是 prompt 不是房间类型
            room_id = data.get('room_id', '')
            # 从 JSON 文件路径提取
            json_filename = Path(json_path).stem.lower()
            parent_folder = Path(json_path).parent.name.lower()
            
            # 优先从父文件夹名提取
            room_type_normalized = normalize_room_type(parent_folder)
            if room_type_normalized == parent_folder:  # 未匹配到
                room_type_normalized = normalize_room_type(room_id)
            if room_type_normalized == room_id:  # 仍未匹配
                room_type_normalized = normalize_room_type(json_filename)
        else:
            room_type_normalized = normalize_room_type(room_type_raw)
        
        # 获取物体列表
        objects = data.get('objects', [])
        object_count = len(objects)
        
        # 统计物体类型
        object_types = {}
        object_list = []
        for obj in objects:
            desc = obj.get('desc', 'unknown')
            object_list.append(desc)
            object_types[desc] = object_types.get(desc, 0) + 1
        
        # 获取房间大小合理范围
        min_area, max_area = get_room_size_range(room_type_normalized)
        room_size_valid = min_area <= room_area <= max_area
        
        # 构建元数据
        metadata = {
            "room_type": room_type_normalized,
            "room_type_raw": room_type_raw,
            "room_id": data.get('room_id', 'unknown'),
            "room_width": round(room_width, 2),
            "room_depth": round(room_depth, 2),
            "room_area": round(room_area, 2),
            "room_size_range": (min_area, max_area),
            "room_size_valid": room_size_valid,
            "object_count": object_count,
            "object_list": object_list,
            "object_types": object_types,
            "json_path": json_path
        }
        
        return metadata
        
    except Exception as e:
        print(f"Warning: Error parsing scene JSON {json_path}: {e}")
        return None


def format_scene_metadata(metadata: Dict[str, Any]) -> str:
    """
    格式化场景元数据为可读字符串
    
    Args:
        metadata: 场景元数据字典
        
    Returns:
        格式化的字符串
    """
    if metadata is None:
        return ""
    
    lines = [
        f"Room Type: {metadata.get('room_type', 'unknown')}",
        f"Room Size: {metadata.get('room_width', 0):.2f}m x {metadata.get('room_depth', 0):.2f}m (Area: {metadata.get('room_area', 0):.2f} m²)",
        f"Object Count: {metadata.get('object_count', 0)}",
        f"Objects: {', '.join(metadata.get('object_list', []))}"
    ]
    
    return "\n".join(lines)


def evaluate_single_scene(
    client: AzureOpenAI,
    render_image_path: str,
    user_requirement: str,
    scene_metadata: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    评估单个场景
    
    Args:
        client: AzureOpenAI 客户端
        render_image_path: 渲染图像路径
        user_requirement: 用户需求文本
        scene_metadata: 场景元数据（从JSON解析）
        verbose: 是否输出详细信息
        
    Returns:
        包含评分结果的字典
    """
    # 清理用户需求文本
    user_requirement = clean_user_requirement(user_requirement)
    
    # 格式化元数据
    metadata_str = format_scene_metadata(scene_metadata) if scene_metadata else ""
    
    result = {
        "render_file": Path(render_image_path).name,
        "user_requirement": user_requirement,
        "scene_metadata": scene_metadata,
        "scores": {
            "rationality": None,
            "spatial_layout": None,
            "accessibility": None
        },
        "average_score": None,
        "scene_description": None,
        "success": False,
        "error": None
    }
    
    try:
        # 检查渲染图是否存在
        if not Path(render_image_path).exists():
            result["error"] = f"Render image not found: {render_image_path}"
            return result
        
        if verbose:
            print(f"  Evaluating rationality...")
        
        result["scores"]["rationality"] = evaluate_rationality(
            client, render_image_path, user_requirement, scene_metadata
        )
        
        if verbose:
            print(f"  Evaluating spatial layout...")
        
        result["scores"]["spatial_layout"] = evaluate_spatial_layout(
            client, render_image_path, user_requirement, scene_metadata
        )

        if verbose:
            print(f"  Evaluating accessibility...")
        
        result["scores"]["accessibility"] = evaluate_accessibility(
            client, render_image_path, user_requirement, scene_metadata
        )
        
        # 计算平均分
        valid_scores = [s for s in result["scores"].values() if s is not None]
        if valid_scores:
            result["average_score"] = sum(valid_scores) / len(valid_scores)
            result["success"] = True
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def collect_evaluation_tasks(render_dir: str, prompts_file: str, json_dir: str = None) -> List[Dict[str, str]]:
    """
    收集评估任务
    
    Args:
        render_dir: 渲染图目录
        prompts_file: prompt文件路径
        json_dir: JSON文件目录（可选，用于解析场景元数据）
        
    Returns:
        任务列表
    """
    render_path = Path(render_dir)
    prompts_path = Path(prompts_file)
    json_path_dir = Path(json_dir) if json_dir else None
    
    if not render_path.exists():
        print(f"Error: Render directory not found: {render_dir}")
        return []
    
    if not prompts_path.exists():
        print(f"Error: Prompts file not found: {prompts_file}")
        return []
    
    # 检查JSON目录
    if json_path_dir and not json_path_dir.exists():
        print(f"Warning: JSON directory not found: {json_dir}, will try to find JSON files alongside images")
        json_path_dir = None
        
    # 读取所有prompts
    with open(prompts_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # 如果有JSON目录，建立索引
    json_files_map = {}
    if json_path_dir:
        for json_file in json_path_dir.glob("*.json"):
            # 尝试从文件名提取索引
            match = re.search(r'(?:prompt|scene)?_?(\d+)', json_file.stem)
            if match:
                idx = int(match.group(1))
                json_files_map[idx] = str(json_file)
        print(f"Found {len(json_files_map)} JSON files in {json_dir}")
    
    tasks = []
    
    # 遍历渲染图
    # 支持多种图片格式
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(render_path.glob(ext))
    
    print(f"Scanning {len(image_files)} images...")

    # 自动检测索引偏移 (0-based vs 1-based)
    indices = []
    for img_path in image_files:
        match = re.search(r'(?:prompt|scene)_(\d+)', img_path.name)
        if match:
            indices.append(int(match.group(1)))
    
    offset = 0
    if indices:
        min_idx = min(indices)
        has_zero = 0 in indices
        
        if has_zero:
            print(f"Detected 0-based indexing (found index 0). Offset: 0")
            offset = 0
        elif min_idx == 1:
            print(f"Detected 1-based indexing (starts at 1, no 0). Offset: -1")
            offset = -1
        else:
            print(f"Indices start at {min_idx}. Assuming 0-based indexing. Offset: 0")
            offset = 0
        
    for img_path in sorted(image_files):
        matched_idx = -1
        match_method = "none"
        
        # 策略1: 文件名ID匹配 (prompt_123.png 或 scene_123.png)
        match = re.search(r'(?:prompt|scene)_(\d+)', img_path.name)
        if match:
            idx = int(match.group(1))
            # 应用偏移
            adj_idx = idx + offset
            if 0 <= adj_idx < len(prompts):
                matched_idx = adj_idx
                match_method = f"id_offset_{offset}" if offset != 0 else "id"
        
        # 策略2: 同名JSON文件匹配 (读取 room_type 或 prompt 字段)
        if matched_idx == -1:
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 尝试获取 prompt 文本
                        prompt_text = data.get('room_type') or data.get('prompt') or data.get('user_requirement')
                        
                        if prompt_text:
                            # 在 prompts 列表中查找
                            prompt_text = prompt_text.strip()
                            # 1. 精确匹配
                            if prompt_text in prompts:
                                matched_idx = prompts.index(prompt_text)
                                match_method = "json_exact"
                            else:
                                # 2. 模糊匹配 (如果 JSON 中的 prompt 是完整的话)
                                for i, p in enumerate(prompts):
                                    # 检查是否包含 (双向)
                                    if p in prompt_text or prompt_text in p:
                                        matched_idx = i
                                        match_method = "json_contain"
                                        break
                except Exception as e:
                    print(f"Warning: Error reading JSON {json_path}: {e}")

        # 策略3: 文件名内容匹配
        if matched_idx == -1:
            # 清理文件名
            stem = img_path.stem
            # 移除常见的时间戳后缀 (e.g., -2025-12-23-...)
            stem_clean = re.sub(r'-\d{4}-\d{2}-\d{2}.*$', '', stem)
            # 将下划线替换为空格
            stem_text = stem_clean.replace('_', ' ').lower()
            
            # 在 prompts 中查找以此开头的
            candidates = []
            for i, p in enumerate(prompts):
                p_clean = p.lower()
                # 检查文件名是否是 prompt 的开头，或者 prompt 包含文件名
                if p_clean.startswith(stem_text) or stem_text in p_clean:
                    candidates.append(i)
            
            if len(candidates) == 1:
                matched_idx = candidates[0]
                match_method = "filename_content"
            elif len(candidates) > 1:
                # 如果有多个匹配，取第一个
                matched_idx = candidates[0]
                match_method = "filename_content_ambiguous"

        if matched_idx != -1:
            # 尝试查找对应的JSON文件
            json_file_path = None
            
            # 策略1: 从文件名中提取索引，查找JSON目录中的文件
            match = re.search(r'(?:prompt|scene)_(\d+)', img_path.name)
            if match and json_files_map:
                idx = int(match.group(1))
                if idx in json_files_map:
                    json_file_path = json_files_map[idx]
            
            # 策略2: 查找同名JSON文件（与图片同目录或JSON目录）
            if not json_file_path:
                # 先在JSON目录中查找
                if json_path_dir:
                    potential_json = json_path_dir / f"{img_path.stem}.json"
                    if potential_json.exists():
                        json_file_path = str(potential_json)
                
                # 再在图片同目录查找
                if not json_file_path:
                    potential_json = img_path.with_suffix('.json')
                    if potential_json.exists():
                        json_file_path = str(potential_json)
            
            tasks.append({
                "prompt_id": str(matched_idx),
                "render_image": str(img_path),
                "user_requirement": prompts[matched_idx],
                "match_method": match_method,
                "json_file": json_file_path
            })
        else:
            print(f"Warning: Could not match prompt for image: {img_path.name}")
            
    return tasks


def process_single_task_safe(client, task, existing_result, verbose, task_id, total_tasks, print_lock):
    """Wrapper for parallel execution"""
    render_name = Path(task["render_image"]).name
    
    if existing_result:
        with print_lock:
            print(f"[{task_id}/{total_tasks}] Skipping {render_name} (already evaluated)")
        return existing_result

    with print_lock:
        print(f"[{task_id}/{total_tasks}] Evaluating: prompt_{task['prompt_id']}")
        # print(f"  Requirement: {task['user_requirement'][:80]}...")

    # 解析场景元数据
    scene_metadata = None
    if task.get("json_file"):
        scene_metadata = parse_scene_json(task["json_file"])
        
    result = evaluate_single_scene(
        client,
        task["render_image"],
        task["user_requirement"],
        scene_metadata=scene_metadata,
        verbose=verbose
    )
    return result


def run_evaluation(
    render_dir: str,
    prompts_file: str,
    output_file: str = None,
    json_dir: str = None,
    max_scenes: int = None,
    max_workers: int = 4,
    delay: float = 0.5,
    verbose: bool = False,
    resume: bool = False
) -> Dict[str, Any]:
    """
    运行批量评估
    
    Args:
        render_dir: 渲染图目录
        prompts_file: prompt文件路径
        output_file: 输出文件路径
        json_dir: JSON场景文件目录（可选）
        max_scenes: 最大评估场景数
        max_workers: 最大并行工作线程数
        delay: 请求间隔（秒）
        verbose: 是否输出详细信息
        resume: 是否从上次中断处继续
        
    Returns:
        评估结果字典
    """
    print("=" * 80)
    print("VLM Scene Evaluation - GPT-4o Vision")
    print("=" * 80)
    
    # 收集评估任务
    print("\nCollecting evaluation tasks...")
    tasks = collect_evaluation_tasks(render_dir, prompts_file, json_dir)
    
    if not tasks:
        print("No tasks found!")
        return {}
    
    print(f"Found {len(tasks)} scenes to evaluate")
    
    # 限制场景数量
    if max_scenes and max_scenes > 0:
        tasks = tasks[:max_scenes]
        print(f"Limiting to {max_scenes} scenes")
    
    # 检查是否需要恢复
    existing_results = {}
    if resume and output_file and Path(output_file).exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Handle both list format (old) and dict format with individual_results (new)
                if isinstance(existing_data, dict) and "individual_results" in existing_data:
                    for result in existing_data["individual_results"]:
                        existing_results[result["render_file"]] = result
                elif isinstance(existing_data, list):
                    for result in existing_data:
                        existing_results[result["render_file"]] = result
            print(f"Resuming: Found {len(existing_results)} existing results")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
    
    # 初始化客户端
    print("\nInitializing Azure OpenAI client...")
    client = setup_azure_client()
    
    # 评估所有场景
    print(f"\nStarting evaluation with {max_workers} workers...")
    all_results = []
    successful = 0
    failed = 0
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    # Lock for printing and saving results
    print_lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, task in enumerate(tasks):
            render_name = Path(task["render_image"]).name
            existing = existing_results.get(render_name)
            
            future = executor.submit(
                process_single_task_safe,
                client,
                task,
                existing,
                verbose,
                i + 1,
                len(tasks),
                print_lock
            )
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                
                with print_lock:
                    if result["success"]:
                        successful += 1
                        scores = result["scores"]
                        avg = result["average_score"]
                        print(f"  ✓ [{result['render_file']}] Scores: R={scores['rationality']} "
                              f"L={scores['spatial_layout']} A={scores['accessibility']} | Avg={avg:.1f}")
                    else:
                        failed += 1
                        print(f"  ✗ [{result['render_file']}] Failed: {result.get('error', 'Unknown error')}")
                    
                    # 定期保存结果
                    if output_file and len(all_results) % 10 == 0:
                        save_results(all_results, output_file)
                        print(f"  [Checkpoint saved]")
            except Exception as e:
                with print_lock:
                    print(f"Worker exception: {e}")
    
    # 计算统计信息
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print(f"Successful: {successful}, Failed: {failed}")
    
    # 生成最终结果
    final_results = generate_final_results(all_results, len(tasks))
    # 保存结果
    if output_file:
        save_results(all_results, output_file, final_results)
        print(f"\nResults saved to: {output_file}")
    # 打印统计摘要
    print_statistics(final_results)
    return final_results


def save_results(all_results: List[Dict], output_file: str, final_results: Dict = None):
    """保存结果到文件"""
    if final_results is None:
        final_results = generate_final_results(all_results, len(all_results))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

def generate_final_results(all_results: List[Dict], total_scenes: int) -> Dict[str, Any]:
    """生成最终结果统计（恢复实现）"""
    successful_results = [r for r in all_results if r.get("success")]

    # 收集各维度分数
    score_names = ["rationality", "spatial_layout", "accessibility"]
    score_values = {name: [] for name in score_names}
    average_scores = []

    for result in successful_results:
        for name in score_names:
            score = result["scores"].get(name)
            if score is not None:
                score_values[name].append(score)
        if result.get("average_score") is not None:
            average_scores.append(result["average_score"])

    # 计算统计量
    import numpy as np

    statistics = {}
    for name in score_names:
        values = score_values[name]
        if values:
            statistics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values)
            }

    if average_scores:
        statistics["average"] = {
            "mean": float(np.mean(average_scores)),
            "std": float(np.std(average_scores)),
            "min": float(np.min(average_scores)),
            "max": float(np.max(average_scores)),
            "count": len(average_scores)
        }

    return {
        "summary": {
            "total_scenes": total_scenes,
            "successful_evaluations": len(successful_results),
            "failed_evaluations": total_scenes - len(successful_results),
            "success_rate": len(successful_results) / total_scenes * 100 if total_scenes > 0 else 0
        },
        "aggregate_statistics": statistics,
        "individual_results": all_results
    }

def print_statistics(final_results: Dict):
    """打印统计摘要"""
    print("\n--- Evaluation Statistics ---")
    summary = final_results.get("summary", {})
    print(f"\nTotal Scenes: {summary.get('total_scenes', 0)}")
    print(f"Successful: {summary.get('successful_evaluations', 0)}")
    print(f"Failed: {summary.get('failed_evaluations', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    stats = final_results.get("aggregate_statistics", {})
    print("\n--- Score Statistics (1-10) ---")
    print(f"{'Dimension':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for name in ["rationality", "spatial_layout", "accessibility", "average"]:
        if name in stats:
            s = stats[name]
            display_name = name.replace("_", " ").title()
            print(f"{display_name:<25} {s['mean']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")

def main():
    parser = argparse.ArgumentParser(
        description='VLM Scene Evaluation using GPT-4o Vision (1-5 scoring)'
    )
    parser.add_argument(
        '--render-dir', type=str, required=True,
        help='Directory containing render images'
    )
    parser.add_argument(
        '--prompts-file', type=str, required=True,
        help='File containing prompts (one per line)'
    )
    parser.add_argument(
        '--json-dir', type=str,
        default=None,
        help='Directory containing scene JSON files (optional, for metadata extraction)'
    )
    parser.add_argument(
        '--output', type=str,
        default=None,
        help='Output JSON file path (default: <render-dir>/vlm_evaluation_results.json)'
    )
    parser.add_argument(
        '--max-scenes', type=int,
        default=None,
        help='Maximum number of scenes to evaluate'
    )
    parser.add_argument(
        '--max-workers', type=int,
        default=4,
        help='Maximum parallel workers (currently not used, sequential for API rate limits)'
    )
    parser.add_argument(
        '--delay', type=float,
        default=0.5,
        help='Delay between API calls in seconds'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume from previous evaluation checkpoint'
    )
    args = parser.parse_args()
    # 设置默认输出路径
    if args.output is None:
        args.output = str(Path(args.render_dir) / "vlm_evaluation_results.json")
    # 运行评估
    run_evaluation(
        render_dir=args.render_dir,
        prompts_file=args.prompts_file,
        output_file=args.output,
        json_dir=args.json_dir,
        max_scenes=args.max_scenes,
        max_workers=args.max_workers,
        delay=args.delay,
        verbose=args.verbose,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
