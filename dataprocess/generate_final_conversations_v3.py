"""
第三步：基于v2版本的改进，让GPT-4o读取图片并生成分步推理的CoT对话数据
支持多进程并行处理以加速生成
"""

import json
import os
import uuid
import time
import shutil
import base64
import random
import re
from typing import Dict, Any, List
from multiprocessing import Pool, Manager, Value
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from openai import AzureOpenAI
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from pathlib import Path
from config import *
import threading

# 全局速率限制器
class RateLimiter:
    """线程安全的速率限制器，确保API调用间隔至少为指定秒数"""
    def __init__(self, min_interval: float = 1.0):
        self.min_interval = min_interval
        self.last_call_time = 0
        self.lock = threading.Lock()
    
    def wait(self):
        """等待直到可以进行下一次调用"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time
            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)
            self.last_call_time = time.time()

# 全局速率限制器实例：每秒最多1次调用
_rate_limiter = RateLimiter(min_interval=1.0)

class RateLimitedAzureOpenAI:
    """带速率限制的Azure OpenAI客户端包装器"""
    def __init__(self, client: AzureOpenAI):
        self._client = client
        self.chat = self
        self.completions = self
    
    def create(self, **kwargs):
        """带速率限制的API调用"""
        _rate_limiter.wait()
        return self._client.chat.completions.create(**kwargs)

def setup_azure_client() -> RateLimitedAzureOpenAI:
    """Create AzureOpenAI client using Azure CLI or managed identity tokens with rate limiting."""
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
    # 返回带速率限制的包装器
    return RateLimitedAzureOpenAI(client)

def extract_current_scene_from_user_message(user_content: str) -> dict:
    """
    从用户消息中提取 <current_scene> 标签内的JSON内容
    """
    try:
        start_tag = "<current_scene>"
        end_tag = "</current_scene>"
        
        start_idx = user_content.find(start_tag)
        if start_idx == -1:
            print("警告：未找到 <current_scene> 标签")
            return {}
        
        start_idx += len(start_tag)
        end_idx = user_content.find(end_tag, start_idx)
        if end_idx == -1:
            print("警告：未找到 </current_scene> 结束标签")
            return {}
        
        scene_content = user_content[start_idx:end_idx].strip()
        
        # 移除可能的 ```json 标记
        if scene_content.startswith("```json"):
            scene_content = scene_content[7:]
        if scene_content.endswith("```"):
            scene_content = scene_content[:-3]
        
        scene_content = scene_content.strip()
        
        # 解析JSON
        import json
        current_scene = json.loads(scene_content)
        return current_scene
        
    except json.JSONDecodeError as e:
        print(f"解析current_scene JSON失败: {e}")
        return {}
    except Exception as e:
        print(f"提取current_scene时出错: {e}")
        return {}

def extract_conversation_turns(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从messages中提取所有对话轮次，每个轮次包含user和assistant的消息对
    
    Returns:
        List of turns, each containing: {
            'turn_index': int,
            'user_message': Dict,
            'assistant_message': Dict,
            'has_tool_calls': bool
        }
    """
    turns = []
    turn_index = 0
    i = 0
    
    while i < len(messages):
        # 寻找user消息
        user_message = None
        while i < len(messages) and messages[i]["role"] != "user":
            i += 1
        
        if i < len(messages):
            user_message = messages[i]
            i += 1
        else:
            break
            
        # 寻找对应的assistant消息
        assistant_message = None
        while i < len(messages) and messages[i]["role"] != "assistant":
            i += 1
            
        if i < len(messages):
            assistant_message = messages[i]
            i += 1
        else:
            break
            
        # 检查assistant消息是否包含tool_calls
        has_tool_calls = bool(assistant_message.get("tool_calls", []))
        if not has_tool_calls and "content" in assistant_message:
            # 检查内容中是否有XML格式的tool_calls
            content = assistant_message["content"]
            has_tool_calls = "<tool_calls>" in content and "</tool_calls>" in content
        
        if has_tool_calls:
            turns.append({
                'turn_index': turn_index,
                'user_message': user_message,
                'assistant_message': assistant_message,
                'has_tool_calls': has_tool_calls
            })
            turn_index += 1
    
    return turns

def extract_think_process(cot_content: str) -> str:
    """
    从生成的CoT内容中提取主要的思考过程，过滤掉格式标记和冗余信息
    """
    # 移除常见的格式标记
    cleaned_content = cot_content
    
    # 移除markdown格式标记
    import re
    cleaned_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_content)  # 移除**粗体**
    cleaned_content = re.sub(r'\*([^*]+)\*', r'\1', cleaned_content)      # 移除*斜体*
    cleaned_content = re.sub(r'#{1,6}\s*([^\n]+)', r'\1', cleaned_content)  # 移除标题#
    cleaned_content = re.sub(r'\[([^\]]+)\]', r'\1', cleaned_content)     # 移除[说明文字]
    
    # 移除多余的空行
    cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
    
    return cleaned_content.strip()


def encode_image(image_path: str) -> str:
    """将图片编码为base64格式供GPT-4o使用"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_cot_template_a(global_user_instruction, current_scene, tool_calls, metadata):
    """模板A: 详细分析型 - 4步法"""
    
    # 分析tool_calls中的具体操作
    operations_summary = []
    if tool_calls:
        for call in tool_calls:
            if isinstance(call, dict):
                operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                operations_summary.append(f"- {operation_name}: {args}")
    
    if not operations_summary:
        operations_summary = ["- terminate: No modifications needed"]
    
    operations_text = "\n".join(operations_summary)
    
    return f"""You are a professional interior design assistant analyzing a room design step. Look at the provided room image and generate a step-by-step thinking process that explains the SPECIFIC design decisions that were actually made.

## User's Design Goal
"{global_user_instruction}"

## ACTUAL OPERATIONS PERFORMED (You must explain these exact actions):
{operations_text}

## Current Scene Context
{json.dumps(current_scene, indent=2, ensure_ascii=False)}

## CRITICAL INSTRUCTION: 
Your thinking process must explain and justify the EXACT operations listed above. Do not suggest different operations - explain why these specific operations were the right choice. The problems you identify MUST directly correspond to the operations performed.

## Problem Classification Guide
When identifying problems, categorize them into these types:

**1. Physical Bugs (物理类问题)**:
   - Object Overlap/Collision: Two or more objects occupying the same space
   - Out of Bounds: Objects extending beyond room boundaries
   - Floating Objects: Objects not properly supported (e.g., lamp floating in mid-air)
   - Example: "The coffee table is overlapping with the sofa by 0.3m" → requires move_object
   - Example: "The wardrobe extends 0.5m beyond the north wall" → requires move_object or scale_object

**2. Layout Rationality Bugs (场景合理性问题)**:
   - Core Furniture Misplacement: Bed/sofa not against wall, wrong orientation
   - Missing Essential Items: Room lacks core furniture for its type
   - Improper Orientation: Furniture facing wrong direction (e.g., sofa facing wall instead of TV)
   - Example: "The bed is placed in the center of the room, not against any wall" → requires move_object
   - Example: "The sofa is facing the wall instead of the TV area" → requires rotate_object
   - Example: "This bedroom has no bed" → requires add_object

**3. Spatial Distribution Bugs (空间分布问题)**:
   - Clustering: All furniture crowded in one corner/side
   - Large Empty Areas: More than 40% of room completely empty
   - Unbalanced Layout: One half crowded, other half empty
   - Example: "All furniture is clustered in the northeast corner, leaving 70% of the room empty" → requires move_object for multiple items
   - Example: "The room has 8 objects but they're all along one wall" → requires move_object to redistribute

## Generate Your Analysis

**Step 1: Problem Identification (MUST match operations)**
[Look at the current scene image and identify the specific design problems. Categorize each problem as Physical Bug, Layout Rationality Bug, or Spatial Distribution Bug. Each identified problem MUST correspond to one of the actual operations performed.]

**Step 2: Strategic Planning & Tool Selection**
[For each operation actually performed, explain:
- WHAT specific categorized problem this operation addresses
- WHY this particular tool was chosen (vs. the 6 other available options)
- HOW this tool specifically solves the identified problem
- WHAT design principle guides this choice]

**Step 3: Parameter Justification & Execution**
[For each operation performed, explain the specific parameters chosen:
- If moving/adding objects: Why these exact coordinates solve the problem?
- If rotating objects: How does this orientation address the issue?
- If scaling objects: What size considerations fix the problem?
- If replacing/removing: Why this specific change resolves the concern?
- If terminating: Why no changes were needed?]

**Step 4: Impact Validation**
[Assess how these specific operations address the identified problems:
- Which categorized problems were solved by these exact actions?
- How do these changes move closer to the user's design goal?
- What functional and aesthetic improvements were achieved?]

REMEMBER: Every problem you identify must be addressed by one of the actual operations performed. Do not identify problems that aren't being fixed."""

def get_cot_template_b(global_user_instruction, current_scene, tool_calls, metadata):
    """模板B: 策略先行型"""
    
    # 分析实际执行的操作
    operations_summary = []
    operation_types = []
    if tool_calls:
        for call in tool_calls:
            if isinstance(call, dict):
                operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                operations_summary.append(f"- {operation_name}: {args}")
                operation_types.append(operation_name)
    
    if not operations_summary:
        operations_summary = ["- terminate: No modifications needed"]
        operation_types = ["terminate"]
    
    operations_text = "\n".join(operations_summary)
    
    return f"""You are an experienced interior designer explaining your design strategy. Analyze the provided room image and explain the strategic thinking behind the SPECIFIC operations that were actually performed.

## Design Mission
"{global_user_instruction}"

## ACTUAL OPERATIONS PERFORMED (Explain these exact actions):
{operations_text}

## Current Scene Context
{json.dumps(current_scene, indent=2, ensure_ascii=False)}

## CRITICAL INSTRUCTION: 
Your response must explain and justify these specific operations. Do not suggest alternative actions - focus on why these particular choices were made. The problems you diagnose MUST directly correspond to the operations performed.

## Problem Classification Reference
Categorize identified problems into:

**Physical Bugs**: Object collisions/overlaps, out-of-bounds placement, floating objects without support
- Severe example: "Two sofas occupy the exact same position - complete overlap"
- Severe example: "The dining table extends 1m outside the room boundary"
- Severe example: "A floor lamp is floating 0.5m above the ground"

**Layout Rationality Bugs**: Core furniture misplacement, missing essentials, wrong orientations
- Severe example: "The bed is placed diagonally in the room center, not against any wall"
- Severe example: "The sofa faces the corner instead of the living area"
- Severe example: "This is labeled as a bedroom but contains no bed"

**Spatial Distribution Bugs**: Furniture clustering, large empty zones, unbalanced layouts
- Severe example: "All 12 objects are squeezed into the southeast corner, 75% of room is empty"
- Severe example: "One side has a sofa, table, chairs, lamps; opposite side is completely bare"

Structure your response as follows:

**Problem Diagnosis (MUST match operations)**
[Identify the specific problems in the current scene that led to these exact operations. Categorize each as Physical Bug, Layout Rationality Bug, or Spatial Distribution Bug:
- What design issues required these particular interventions?
- How do these categorized problems prevent achieving the user's vision?
- Which specific areas needed the attention they received?]

**Strategic Approach**
[Explain your strategy for the operations actually performed:
- Why was this particular combination of {', '.join(set(operation_types))} operations the right approach?
- How do these specific actions work together to create a cohesive solution?
- What was the priority logic behind this exact sequence of operations?]

**Execution Breakdown**
[For each operation actually performed, explain the problem-solving logic:
{chr(10).join([f"- **Operation {i+1}**: {op.split(':')[0]} - Explain exactly why this operation was needed and how the specific parameters address the identified problem." for i, op in enumerate(operations_summary)])}]

**Problem Resolution Validation**
[Assess how these specific operations address the initial categorized problems:
- Which Physical/Layout/Distribution bugs were resolved by these exact actions?
- How do these particular modifications improve the space?
- What design principles are now better represented through these changes?
- How much closer are we to achieving the user's vision with these specific interventions?]

Focus on justifying the actual operations performed. Every diagnosed problem must be addressed by one of the operations."""

def get_cot_template_c(global_user_instruction, current_scene, tool_calls, metadata):
    """模板C: 问答型"""
    
    # 分析实际执行的操作
    operations_summary = []
    if tool_calls:
        for call in tool_calls:
            if isinstance(call, dict):
                operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                operations_summary.append(f"- {operation_name}: {args}")
    
    if not operations_summary:
        operations_summary = ["- terminate: No modifications needed"]
    
    operations_text = "\n".join(operations_summary)
    
    return f"""As an interior design consultant, answer key questions about the SPECIFIC design intervention that was actually performed based on the room image provided.

## Client's Request
"{global_user_instruction}"

## ACTUAL OPERATIONS PERFORMED (Explain these exact actions):
{operations_text}

## Design Context
{json.dumps(current_scene, indent=2, ensure_ascii=False)}

## CRITICAL INSTRUCTION: 
Answer the questions below by explaining and justifying the specific operations that were actually performed. Focus on why these exact actions were taken. The problems you identify MUST match the operations.

## Problem Categories with Severe Examples

**Physical Bugs (物理类Bug)**:
- Collision: "The nightstand and bed are intersecting by 20cm"
- Out of bounds: "Half of the wardrobe is outside the room wall"
- Floating: "The ceiling lamp is positioned at floor level, floating objects nearby"

**Layout Rationality Bugs (布局合理性Bug)**:
- Misplacement: "The king-size bed is in the room center, 2m away from all walls"
- Wrong orientation: "The sofa back faces the TV, users would have to turn around to watch"
- Missing core: "A master bedroom without any bed or sleeping furniture"

**Spatial Distribution Bugs (空间分布Bug)**:
- Clustering: "Everything piled in one corner - bed, dresser, nightstands all within 2m²"
- Empty zones: "The entire left half of the living room has zero furniture"
- Imbalance: "North wall has 6 items, south wall has nothing"

Answer these design questions based on your visual analysis:

**What problems did you identify in the current space? (Categorize each)**
[Identify the specific issues that led to these exact operations being performed. For each problem, specify if it's a Physical Bug, Layout Rationality Bug, or Spatial Distribution Bug:
- What spatial, functional, or aesthetic problems required these particular interventions?
- How do these categorized problems prevent achieving the user's goal?
- Which specific areas needed the attention they received through these operations?]

**What was your strategic plan to solve these problems?**
[Explain your approach for the operations actually performed:
- Why were these specific operations ({', '.join([op.split(':')[0].strip('- ') for op in operations_summary])}) the right solution for the categorized problems?
- How did you prioritize which problems to address with these particular actions?
- Why was this combination of operations chosen over alternative approaches?]

**How did you execute each solution?**
[For each operation actually performed, explain the problem-solving logic:
{chr(10).join([f"- **{op.split(':')[0].strip('- ')}**: What specific categorized problem did this operation solve and why were the chosen parameters the right approach?" for op in operations_summary])}]

**Did the changes successfully resolve the initial problems?**
[Evaluate the success of these specific operations:
- Which Physical/Layout/Distribution bugs were effectively addressed by these exact actions?
- How do these particular modifications improve spatial function?
- What aesthetic improvements were achieved through these operations?
- How much progress was made toward the user's overall goal with these specific interventions?
- Which design principles are now better implemented because of these changes?]

Be specific about what categorized problems you observe in the image. Every problem identified must be solved by one of the actual operations."""

def get_cot_template_flexible(global_user_instruction, current_scene, tool_calls, metadata):
    """模板D: 柔性指令型"""
    
    # 分析实际执行的操作
    operations_summary = []
    if tool_calls:
        for call in tool_calls:
            if isinstance(call, dict):
                operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                operations_summary.append(f"- {operation_name}: {args}")
    
    if not operations_summary:
        operations_summary = ["- terminate: No modifications needed"]
    
    operations_text = "\n".join(operations_summary)
    
    return f"""You are a professional interior design assistant analyzing a room design intervention. Look at the provided image and generate a thoughtful chain of thought explanation for the SPECIFIC operations that were actually performed.

## Design Context
**User's Goal:** "{global_user_instruction}"
**Current Scene:** {json.dumps(current_scene, indent=2, ensure_ascii=False)}

## ACTUAL OPERATIONS PERFORMED (Explain these exact actions):
{operations_text}

## CRITICAL INSTRUCTION: 
Your chain of thought must explain and justify these specific operations. Do not suggest different operations - focus on why these particular actions were the right choice. Problems identified MUST correspond to operations performed.

## Problem Classification Framework

**Physical Bugs** - Violations of physical reality:
- Object overlap/collision (two items sharing same space)
- Out of bounds (furniture extending beyond room walls)
- Floating objects (items not properly grounded or supported)
- Severe: "The sofa and coffee table share 50% overlap - physically impossible"
- Severe: "The bookshelf clips through the east wall by 40cm"

**Layout Rationality Bugs** - Violations of furniture logic:
- Core furniture not against walls (bed/sofa floating in room center)
- Wrong orientation (sofa facing wall, desk facing corner)
- Missing essential furniture (bedroom without bed, living room without seating)
- Severe: "The double bed is positioned in the exact center, equidistant from all walls"
- Severe: "The dining chairs face away from the dining table"

**Spatial Distribution Bugs** - Violations of spatial balance:
- Furniture clustering (everything in one corner/side)
- Large empty zones (>40% of usable space empty)
- Severe imbalance (one area overcrowded, another barren)
- Severe: "All furniture occupies only the northwest quadrant, rest is empty"
- Severe: "10 objects line the north wall, south/east/west walls have nothing"

## Instructions
Your response should be a chain of thought that follows this problem-solving approach:

1. **Problem Analysis (with Classification)**: Examine the current scene image and identify the specific design problems that necessitated these exact operations. Classify each problem as Physical Bug, Layout Rationality Bug, or Spatial Distribution Bug. Each problem you identify MUST be addressed by one of the actual operations.

2. **Operation Logic**: For each operation actually performed, explain:
   - Which specific categorized problem this operation addresses
   - Why this particular operation was the best choice to solve that type of problem
   - How the chosen parameters (positions, rotations, scales, object descriptions) specifically address the identified issue

3. **Execution Justification**: Detail the reasoning behind the specific parameters used in each operation from a design perspective - explain how these exact values solve the identified categorized problems and support the user's vision.

4. **Problem Resolution**: Conclude by assessing how these specific operations address the initial categorized problems (Physical/Layout/Distribution bugs) and move closer to the user's vision. Focus on what was actually accomplished through these particular actions.

Remember to:
- Classify every problem you identify (Physical/Layout/Distribution)
- Every problem must be addressed by one of the actual operations performed
- Reference visual elements you can observe in the image
- Use spatial design terminology
- Justify the actual parameter choices in terms of solving identified issues

Let's think step by step about what categorized problems existed and how each of these specific operations addresses them."""

def get_cot_template_zero_shot(global_user_instruction, current_scene, tool_calls, metadata):
    """模板E: 零样本CoT型"""
    
    # 分析实际执行的操作
    operations_summary = []
    if tool_calls:
        for call in tool_calls:
            if isinstance(call, dict):
                operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                operations_summary.append(f"- {operation_name}: {args}")
    
    if not operations_summary:
        operations_summary = ["- terminate: No modifications needed"]
    
    operations_text = "\n".join(operations_summary)
    
    return f"""You are a professional interior designer analyzing a room design step. 

## Context
**User's Vision:** "{global_user_instruction}"
**Current Scene:** {json.dumps(current_scene, indent=2, ensure_ascii=False)}

## ACTUAL OPERATIONS PERFORMED (Explain these exact actions):
{operations_text}

## CRITICAL INSTRUCTION: 
Explain the design thinking behind these specific operations that were actually performed. Focus on why these exact actions were taken rather than suggesting alternatives. Problems you identify MUST match the operations.

## Problem Categories (Classify all identified issues)

**Physical Bugs**: 
- Collisions/overlaps between objects
- Objects extending outside room boundaries  
- Floating objects without proper support
- Examples: "Table and chair overlap by 15cm", "Bed extends past wall", "Lamp hovers in mid-air"

**Layout Rationality Bugs**:
- Bed/sofa not against wall (in room center)
- Furniture facing wrong direction
- Missing core furniture for room type
- Examples: "Bed floats in center 1.5m from walls", "Sofa faces blank wall not TV", "Living room has no seating"

**Spatial Distribution Bugs**:
- All furniture clustered in one area
- Large portions of room completely empty
- Severe imbalance between room sections
- Examples: "Everything in northeast corner", "60% of floor space empty", "Left side crowded, right side bare"

Look at the provided room image and explain the design thinking behind these changes. 

**Your analysis must:**
1. Start by identifying the specific CATEGORIZED problems (Physical/Layout/Distribution bugs) in the current scene that led to these exact operations being performed
2. For each problem identified, it MUST correspond to one of the actual operations
3. Analyze why each operation was the right choice to address the identified categorized problem, including the specific parameters chosen
4. Demonstrate how each actual operation moves the space closer to achieving the user's vision by solving particular design challenges

Let's think step by step about what categorized problems existed and how each of these specific operations addresses them."""

def generate_conversation_with_vision(conversation_template: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用GPT-4o的视觉能力根据对话模板和图片生成分步推理的CoT
    支持多轮对话处理，为每个包含tool_calls的assistant消息生成CoT
    支持新旧两种数据格式：
    - 旧格式：conversation + image 字段（单轮）
    - 新格式：messages + images 字段（多轮）
    """
    
    # 在每个进程中创建自己的客户端
    client = setup_azure_client()
    
    # 检测数据格式并提取相应信息
    is_new_format = "messages" in conversation_template and "images" in conversation_template
    
    if is_new_format:
        # 新格式：multi_turn 数据 - 支持多轮对话
        messages = conversation_template["messages"]
        images = conversation_template.get("images", [])
        
        # 提取所有对话轮次
        turns = extract_conversation_turns(messages)
        
        if not turns:
            print("警告：在新格式数据中未找到有效的对话轮次")
            return None
        
        print(f"检测到 {len(turns)} 个对话轮次")
        
        # 获取全局上下文信息
        metadata = conversation_template.get("metadata", {})
        global_user_instruction = "Help me design a complete room layout."  # 默认值
        
        # 从第一个用户消息中提取全局指令和current_scene
        first_turn = turns[0]
        if first_turn["user_message"] and "content" in first_turn["user_message"]:
            user_content = first_turn["user_message"]["content"]
            
            # 提取 current_scene
            current_scene = extract_current_scene_from_user_message(user_content)
            
            # 提取全局指令
            if "<image>" in user_content:
                after_image = user_content.split("<image>", 1)[1].strip()
                if "\n\n" in after_image:
                    global_user_instruction = after_image.split("\n\n")[0].strip()
                elif "<current_scene>" in after_image:
                    global_user_instruction = after_image.split("<current_scene>")[0].strip()
                else:
                    global_user_instruction = after_image.strip()
        else:
            current_scene = {}
        
        # 处理每个对话轮次
        result = conversation_template.copy()
        previous_actions = []  # 记录之前的操作历史
        
        for turn_idx, turn in enumerate(turns):
            print(f"  处理第 {turn_idx + 1} 轮对话...")
            
            # 获取对应的图片
            image_path = ""
            if turn_idx < len(images):
                image_path = images[turn_idx]
            elif images:
                image_path = images[0]  # 使用第一张图片作为默认
            
            if not image_path or not os.path.exists(image_path):
                print(f"    警告：第 {turn_idx + 1} 轮的图片路径不存在: {image_path}")
                continue
            
            # 编码图片
            try:
                base64_image = encode_image(image_path)
            except Exception as e:
                print(f"    编码第 {turn_idx + 1} 轮图片时出错: {e}")
                continue
            
            # 提取当前轮次的tool_calls
            assistant_message = turn["assistant_message"]
            tool_calls = assistant_message.get("tool_calls", [])
            
            # 如果没有直接的 tool_calls 字段，尝试从内容中的 XML 标签提取
            if not tool_calls and "content" in assistant_message:
                content = assistant_message["content"]
                if "<tool_calls>" in content and "</tool_calls>" in content:
                    start_tag = "<tool_calls>"
                    end_tag = "</tool_calls>"
                    start_idx = content.find(start_tag) + len(start_tag)
                    end_idx = content.find(end_tag)
                    
                    if start_idx < end_idx:
                        tool_calls_content = content[start_idx:end_idx].strip()
                        try:
                            import json
                            tool_calls = json.loads(tool_calls_content)
                        except json.JSONDecodeError as e:
                            print(f"    解析第 {turn_idx + 1} 轮 tool_calls JSON 失败: {e}")
                            continue
            
            if not tool_calls:
                print(f"    第 {turn_idx + 1} 轮未找到 tool_calls，跳过")
                continue
            
            # 生成CoT
            cot_process = generate_single_turn_cot(
                client=client,
                base64_image=base64_image,
                global_user_instruction=global_user_instruction,
                current_scene=current_scene,
                tool_calls=tool_calls,
                metadata=metadata,
                turn_index=turn_idx,
                previous_actions=previous_actions.copy()
            )
            
            if cot_process:
                # 更新对应的assistant消息
                for i, msg in enumerate(result["messages"]):
                    if msg is assistant_message:
                        if "content" in result["messages"][i]:
                            content = result["messages"][i]["content"]
                            # 替换占位符
                            content = content.replace("{THINK_PROCESS}", cot_process)
                            # 删除 conclusion 标签及其内容
                            content = re.sub(r'<conclusion>.*?</conclusion>', '', content, flags=re.DOTALL)
                            result["messages"][i]["content"] = content
                        break
                
                # 记录当前轮次的操作到历史中
                previous_actions.append({
                    "turn": turn_idx + 1,
                    "tool_calls": tool_calls,
                    "reasoning": "Design operation performed"
                })
        
        return result
        
    else:
        # 旧格式：保持原有的单轮处理逻辑
        return generate_single_conversation_legacy(conversation_template, client)

def generate_single_turn_cot(client, base64_image: str, global_user_instruction: str, 
                           current_scene: dict, tool_calls: list, metadata: dict, 
                           turn_index: int, previous_actions: list) -> str:
    """
    为单个对话轮次生成CoT思考过程
    
    Returns:
        str: cot_process 或 None
    """
    # 随机选择CoT模板
    template_functions = [
        get_cot_template_a,      # 20% - 详细分析型
        get_cot_template_b,      # 20% - 策略先行型  
        get_cot_template_c,      # 20% - 问答型
        get_cot_template_flexible, # 20% - 柔性指令型
        get_cot_template_zero_shot # 20% - 零样本CoT型
    ]
    
    weights = [0.20, 0.20, 0.20, 0.20, 0.20]
    selected_template_func = random.choices(template_functions, weights=weights)[0]
    
    # 生成提示词，包含历史操作信息
    prompt = generate_multi_turn_prompt(
        selected_template_func,
        global_user_instruction, 
        current_scene, 
        tool_calls, 
        metadata,
        turn_index,
        previous_actions
    )

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "system", 
                    "content": """You are a professional interior design assistant with expertise in spatial planning and furniture arrangement. You analyze room images and provide thoughtful step-by-step explanations for design decisions.

**Image Format Note**: The room images you will analyze show a merged dual-view rendering:
- **Left side**: Top-down view (bird's eye view) showing the spatial layout of the scene
- **Right side**: Diagonal/perspective view showing the 3D appearance of the scene

This dual-view format allows you to see both the spatial arrangement and the visual aesthetics of the design."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
        )
        
        # 提取响应内容
        cot_process = response.choices[0].message.content.strip()
        
        # 速率限制：每秒最多1次API调用
        time.sleep(1.0)
        
        return cot_process
            
    except Exception as e:
        print(f"    调用Azure OpenAI API时出错: {e}")
        time.sleep(2.0)  # 错误时等待更长时间
        return None

def generate_multi_turn_prompt(template_func, global_user_instruction: str, 
                             current_scene: dict, tool_calls: list, metadata: dict,
                             turn_index: int, previous_actions: list) -> str:
    """
    生成包含多轮上下文的提示词
    """
    # 生成基础提示词
    base_prompt = template_func(global_user_instruction, current_scene, tool_calls, metadata)
    
    # 如果有历史操作，添加上下文信息
    if previous_actions and turn_index > 0:
        history_context = "\n\n## Previous Design Actions Context\n"
        history_context += f"This is turn {turn_index + 1} of a multi-step design process. Here's what happened in previous turns:\n\n"
        
        for action in previous_actions:
            history_context += f"**Turn {action['turn']}:** {action['reasoning']}\n"
            for call in action['tool_calls']:
                if isinstance(call, dict):
                    operation_name = call.get('name', call.get('function', {}).get('name', 'unknown'))
                    args = call.get('arguments', call.get('function', {}).get('arguments', {}))
                    history_context += f"  - {operation_name}: {args}\n"
            history_context += "\n"
        
        history_context += f"Consider this context when explaining the current turn's operations. How do the current actions build upon or complement the previous design decisions?\n"
        
        # 在基础提示词中插入历史上下文
        base_prompt = base_prompt + "\n" + history_context
    
    return base_prompt

def generate_single_conversation_legacy(conversation_template: Dict[str, Any], client) -> Dict[str, Any]:
    """
    处理旧格式的单轮对话（保持向后兼容）
    """
    # 旧格式：intermediate 数据
    tool_calls = conversation_template["conversation"][2]["tool_calls"]
    metadata = conversation_template["metadata"]
    current_scene = metadata.get("initial_scene", {})  # 旧格式用initial_scene作为当前场景
    global_user_instruction = metadata.get("global_user_instruction", "Help me design a complete room layout.")
    
    # 获取图片路径
    image_path = conversation_template.get("image", "")
    
    if not image_path or not os.path.exists(image_path):
        print(f"警告：图片路径不存在或为空: {image_path}")
        return None
    
    # 编码图片
    try:
        base64_image = encode_image(image_path)
    except Exception as e:
        print(f"编码图片时出错: {e}")
        return None
    
    # 生成CoT
    cot_process = generate_single_turn_cot(
        client=client,
        base64_image=base64_image,
        global_user_instruction=global_user_instruction,
        current_scene=current_scene,
        tool_calls=tool_calls,
        metadata=metadata,
        turn_index=0,
        previous_actions=[]
    )
    
    if not cot_process:
        return None
    
    # 创建最终对话数据
    result = conversation_template.copy()
    
    # 旧格式：替换思考过程（assistant现在是第3个消息）
    content = result["conversation"][2]["content"]
    # 先替换 THINK_PROCESS
    content = content.replace("{THINK_PROCESS}", cot_process)
    # 删除 conclusion 标签及其内容
    content = re.sub(r'<conclusion>.*?</conclusion>', '', content, flags=re.DOTALL)
    result["conversation"][2]["content"] = content
    
    return result

def copy_and_update_image_paths_multi_turn(source_images: List[str], target_dir: str) -> List[str]:
    """
    复制多张图片到目标目录，支持多轮对话
    
    Args:
        source_images: 源图片路径列表
        target_dir: 目标目录
    
    Returns:
        新的图片路径列表
    """
    try:
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        new_image_paths = []
        for i, source_image_path in enumerate(source_images):
            if source_image_path and os.path.exists(source_image_path):
                # 为多轮对话的图片添加编号后缀
                image_filename = os.path.basename(source_image_path)
                name, ext = os.path.splitext(image_filename)
                if len(source_images) > 1:
                    target_filename = f"{name}_turn_{i+1:03d}{ext}"
                else:
                    target_filename = image_filename
                
                target_image_path = os.path.join(target_dir, target_filename)
                
                # 复制图片
                shutil.copy2(source_image_path, target_image_path)
                new_image_paths.append(target_image_path)
            else:
                # 保留原路径（可能是相对路径）
                new_image_paths.append(source_image_path)
        
        return new_image_paths
        
    except Exception as e:
        print(f"复制多轮图片时出错: {e}")
        return source_images

def copy_and_update_image_path(source_image_path: str, target_dir: str, json_file_path: str) -> str:
    """
    复制图片到目标目录并更新JSON文件中的图片路径
    支持新旧格式的图片处理（单张图片）
    
    Args:
        source_image_path: 源图片路径
        target_dir: 目标目录
        json_file_path: JSON文件路径，需要更新其中的图片路径
    
    Returns:
        新的图片路径
    """
    try:
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        # 生成目标图片路径
        image_filename = os.path.basename(source_image_path)
        target_image_path = os.path.join(target_dir, image_filename)
        
        # 复制图片
        shutil.copy2(source_image_path, target_image_path)
        
        return target_image_path
        
    except Exception as e:
        print(f"复制图片时出错: {e}")
        return source_image_path

def process_single_subscene(args):
    """
    处理单个子场景的包装函数，用于多进程处理
    
    Args:
        args: (template_file_path, output_dir, scene_name, subscene_name) 的元组
    
    Returns:
        (success: bool, scene_name: str, subscene_name: str)
    """
    template_file_path, output_dir, scene_name, subscene_name = args
    
    try:
        success = process_template_file(template_file_path, output_dir, scene_name, subscene_name)
        return (success, scene_name, subscene_name)
    except Exception as e:
        print(f"    ✗ 处理子场景时出错 {subscene_name}: {e}")
        return (False, scene_name, subscene_name)

def process_template_file(template_path: str, output_dir: str, scene_name: str, subscene_name: str) -> bool:
    """
    处理单个对话模板文件生成完整对话数据，同时复制图片并更新路径
    支持新旧两种数据格式的自动检测和处理
    
    Args:
        template_path: 模板文件路径
        output_dir: 输出目录
        scene_name: 场景名称
        subscene_name: 子场景名称
    """
    try:
        # 加载对话模板
        with open(template_path, 'r', encoding='utf-8') as f:
            conversation_template = json.load(f)
        
        # 检测数据格式
        is_new_format = "messages" in conversation_template and "images" in conversation_template
        
        # 生成完整的对话数据
        final_conversation = generate_conversation_with_vision(conversation_template)
        
        if final_conversation:
            # 提取子场景编号，支持两种格式：
            # 1. 旧格式：intermediate_subscene_003 -> 003
            # 2. 新格式：multi_turn_subscene_000_chain_1 -> 000_chain_1
            if subscene_name.startswith("intermediate_subscene_"):
                subscene_id = subscene_name.split('_')[-1]  # 提取 003
            elif subscene_name.startswith("multi_turn_subscene_"):
                # 从 multi_turn_subscene_000_chain_1 提取 000_chain_1
                parts = subscene_name.split('_')
                if len(parts) >= 4:
                    subscene_id = f"{parts[3]}_chain_{parts[5]}"  # 000_chain_1
                else:
                    subscene_id = subscene_name.replace("multi_turn_subscene_", "")
            else:
                # 如果不匹配已知格式，使用完整名称
                subscene_id = subscene_name
            
            # 创建输出子目录结构：场景/子场景编号/
            subscene_output_dir = os.path.join(output_dir, scene_name, subscene_id)
            os.makedirs(subscene_output_dir, exist_ok=True)
            
            # 处理图片复制和路径更新
            if is_new_format:
                # 新格式：处理 images 数组（支持多轮对话）
                images = conversation_template.get("images", [])
                if images:
                    # 使用多轮图片复制函数
                    new_images = copy_and_update_image_paths_multi_turn(images, subscene_output_dir)
                    # 更新JSON中的图片路径
                    final_conversation["images"] = new_images
                else:
                    final_conversation["images"] = []
            else:
                # 旧格式：处理单个 image 字段
                source_image_path = conversation_template.get("image", "")
                if source_image_path and os.path.exists(source_image_path):
                    # 复制图片到新位置
                    new_image_path = copy_and_update_image_path(
                        source_image_path, 
                        subscene_output_dir,
                        ""
                    )
                    
                    # 更新JSON中的图片路径
                    final_conversation["image"] = new_image_path
            
            # 移除metadata字段（生产环境不需要）
            if "metadata" in final_conversation:
                del final_conversation["metadata"]
            
            # 生成输出文件名：conversation_xxx.json（只保留编号）
            output_filename = f"conversation_{subscene_id}.json"
            output_path = os.path.join(subscene_output_dir, output_filename)
            
            # 保存完整对话数据
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_conversation, f, indent=2, ensure_ascii=False)
            
            return True
        else:
            print(f"    ✗ 生成对话失败: {subscene_name}")
            return False
            
    except Exception as e:
        print(f"    ✗ 处理模板文件时出错 {subscene_name}: {e}")
        return False

def process_rendered_outputs_batch(input_dir: str, output_dir: str, max_scenes: int = None, num_processes: int = 4):
    """
    批量处理rendered_outputs目录中的文件，生成分步推理的对话数据
    使用多进程并行处理以提高效率
    支持新旧两种数据格式的自动检测和处理
    
    Args:
        input_dir: 输入目录（rendered_outputs）
        output_dir: 输出目录
        max_scenes: 最大处理场景数，None表示处理所有
        num_processes: 并行进程数，默认4个
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有场景目录
    # 支持两种格式：
    # 1. 带 _labeled 后缀的目录（旧格式）
    # 2. UUID 格式的目录，不带 _labeled 后缀（新格式）
    scene_directories = []
    input_path = Path(input_dir)
    
    for item in input_path.iterdir():
        if item.is_dir():
            # 接受所有目录，但排除明显的非场景目录
            if not item.name.startswith('.'):
                scene_directories.append(item)
    
    # 限制处理数量
    if max_scenes:
        scene_directories = scene_directories[:max_scenes]
    
    print(f"找到 {len(scene_directories)} 个场景目录")
    print(f"使用 {num_processes} 个进程并行处理")
    
    # 收集所有需要处理的任务
    tasks = []
    total_subscenes = 0
    
    for scene_dir in scene_directories:
        scene_name = scene_dir.name
        print(f"扫描场景目录: {scene_name}")
        
        # 获取该场景目录下的所有子场景目录
        subscene_dirs = []
        for item in scene_dir.iterdir():
            # 支持三种子场景格式：
            # 1. 旧格式：intermediate_subscene_XXX
            # 2. 带标签格式：multi_turn_subscene_XXX_chain_X
            # 3. 简洁格式：chain_X (新格式)
            if item.is_dir() and (item.name.startswith("intermediate_subscene_") or 
                                 item.name.startswith("multi_turn_subscene_") or
                                 item.name.startswith("chain_")):
                subscene_dirs.append(item)
        
        subscene_dirs.sort()  # 确保按顺序处理
        print(f"  发现 {len(subscene_dirs)} 个子场景")
        total_subscenes += len(subscene_dirs)
        
        for subscene_dir in subscene_dirs:
            subscene_name = subscene_dir.name
            
            # 查找JSON文件
            json_files = list(subscene_dir.glob("*.json"))
            if not json_files:
                print(f"    ✗ 未找到JSON文件在 {subscene_name}")
                continue
                
            template_file = json_files[0]  # 应该只有一个JSON文件
            
            # 添加到任务列表
            tasks.append((str(template_file), output_dir, scene_name, subscene_name))
    
    print(f"\n准备处理 {len(tasks)} 个子场景任务")
    
    # 使用多进程处理任务，配合tqdm进度条
    successful_count = 0
    failed_count = 0
    
    try:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(process_single_subscene, task): task for task in tasks}
            
            # 使用tqdm显示进度条
            with tqdm(total=len(tasks), desc="处理子场景", unit="scene") as pbar:
                # 处理完成的任务
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        success, scene_name, subscene_name = future.result()
                        if success:
                            successful_count += 1
                        else:
                            failed_count += 1
                            print(f"\n✗ 失败: {scene_name}/{subscene_name}")
                    except Exception as e:
                        failed_count += 1
                        _, _, scene_name, subscene_name = task
                        print(f"\n✗ 异常: {scene_name}/{subscene_name} - {e}")
                    
                    # 更新进度条
                    pbar.update(1)
                    pbar.set_postfix({
                        'Success': successful_count, 
                        'Failed': failed_count,
                        'Rate': f"{successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "0%"
                    })
    
    except KeyboardInterrupt:
        print("\n接收到中断信号，正在停止...")
    
    print(f"\n处理完成!")
    print(f"总场景目录: {len(scene_directories)}")
    print(f"总子场景: {total_subscenes}")
    print(f"总任务: {len(tasks)}")
    print(f"成功: {successful_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "N/A")

def test_multi_turn_functionality():
    """
    测试多轮对话功能的简单验证函数
    """
    print("=== 多轮对话功能测试 ===")
    
    # 创建一个模拟的多轮对话数据
    mock_conversation = {
        "messages": [
            {
                "role": "user",
                "content": "<image>\nPlease help me design a modern living room.\n\n<current_scene>\n{\"objects\": [], \"room_type\": \"living_room\"}\n</current_scene>"
            },
            {
                "role": "assistant", 
                "content": "<think>\n{THINK_PROCESS}\n</think>\n\n<tool_calls>\n[{\"name\": \"add_object\", \"arguments\": {\"object_name\": \"sofa\"}}]\n</tool_calls>\n\n<conclusion>\n</conclusion>"
            },
            {
                "role": "user",
                "content": "That looks good! Now add a coffee table."
            },
            {
                "role": "assistant",
                "content": "<think>\n{THINK_PROCESS}\n</think>\n\n<tool_calls>\n[{\"name\": \"add_object\", \"arguments\": {\"object_name\": \"coffee_table\"}}]\n</tool_calls>\n\n<conclusion>\n</conclusion>"
            }
        ],
        "images": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
        "metadata": {"scene_id": "test_001"}
    }
    
    # 测试提取对话轮次
    turns = extract_conversation_turns(mock_conversation["messages"])
    print(f"提取到 {len(turns)} 个对话轮次")
    
    for i, turn in enumerate(turns):
        print(f"  轮次 {i+1}: {turn['has_tool_calls']}")
        if turn['has_tool_calls']:
            print(f"    用户消息长度: {len(turn['user_message']['content'])}")
            print(f"    助手消息包含tool_calls: {bool(turn['assistant_message'].get('tool_calls', []))}")
    
    print("多轮对话功能测试完成\n")

def main():
    """主函数"""
    
    # 添加多轮对话功能测试
    test_multi_turn_functionality()
    
    # 配置路径 - 更新为新的数据路径
    INPUT_DIRECTORY = "/path/to/datasets/llmscene/rendered_outputs/multi_turn_v2"
    OUTPUT_DIRECTORY = "/path/to/datasets/llmscene/sft/sft_v6"
    
    # 多进程配置
    NUM_PROCESSES = 8  
    MAX_SCENES =  None    # 先只处理少量测试，设为None处理所有
    
    print("开始批量处理渲染输出数据，生成分步推理对话...")
    print(f"输入目录: {INPUT_DIRECTORY}")
    print(f"输出目录: {OUTPUT_DIRECTORY}")
    print(f"并行进程数: {NUM_PROCESSES}")
    print(f"最大场景数: {MAX_SCENES if MAX_SCENES else '所有'}")
    print(f"支持格式: 新格式 (messages + images，支持多轮) 和旧格式 (conversation + image，单轮)")
    
    # 处理渲染输出数据
    process_rendered_outputs_batch(
        INPUT_DIRECTORY, 
        OUTPUT_DIRECTORY, 
        max_scenes=MAX_SCENES,
        num_processes=NUM_PROCESSES
    )

if __name__ == "__main__":
    main()
