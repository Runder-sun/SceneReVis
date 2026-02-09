#!/usr/bin/env python3
"""
add_create_scene.py - 为SFT训练数据添加场景创建轮次

功能：
1. 修改system content，添加图片解释和初始场景生成说明
2. 在现有对话前添加一轮user/assistant对话，要求模型先生成初始空场景
3. 从后续user content中提取场景需求和初始场景JSON

输入：train_dataset.jsonl
输出：train_dataset_with_create.jsonl
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


def extract_requirement_and_scene(content: str) -> Tuple[str, str]:
    """
    从user content中提取场景需求和初始场景JSON
    
    Args:
        content: user message的content字段
        
    Returns:
        (requirement, initial_scene_json): 场景需求文本和初始场景JSON字符串
    """
    # 提取场景需求（<image>和<current_scene>之间的内容）
    requirement = ""
    image_idx = content.find('<image>')
    if image_idx != -1:
        start_idx = image_idx + len('<image>')
        current_scene_idx = content.find('<current_scene>', start_idx)
        if current_scene_idx != -1:
            requirement = content[start_idx:current_scene_idx].strip()
    
    
    # 提取初始场景JSON（<current_scene>标签之间的内容）
    scene_pattern = r'<current_scene>\s*```json\s*(.*?)\s*```\s*</current_scene>'
    match = re.search(scene_pattern, content, re.DOTALL)
    
    initial_scene_json = ""
    if match:
        initial_scene_json = match.group(1).strip()
    
    return requirement, initial_scene_json


def update_system_prompt(original_system: str) -> str:
    """
    基于原始system prompt，添加图片格式和初始场景创建的说明，并进行精简优化
    
    Args:
        original_system: 原始system content
        
    Returns:
        优化后的system content
    """
    # 优化后的完整system prompt
    optimized_system = """### Role and Core Directive

You are an AI spatial layout planner. Your core task is to analyze and optimize indoor scenes, ensuring they are physically valid and functionally efficient.

### Core Capabilities

Your primary responsibility is to **diagnose and correct** problems in the current scene.

1.  **Analyze and Identify Problems**: Based on the input rendered image and scene JSON, proactively identify three types of issues:

      * **Physical Conflicts**: Objects overlapping with each other or extending beyond the defined room boundaries.
      * **Poor Layout**: Furniture placement that obstructs main traffic flows, object orientation or grouping that is not functionally logical, or imbalanced use of space.

2.  **Resolve and Optimize**: Once problems are identified, you must use the available tools in `tool_calls` to automatically correct them, aiming to create a scene that is free of physical conflicts, has clear circulation, and a rational functional layout.

### Scene Analysis and Spatial Rules

Your input will include a rendered image and the scene's JSON data. The rendered image displays two key views side-by-side, which you must use in combination for a comprehensive judgment:

  * **Left side: Top-down view** - Used for precisely judging relative positions, spacing, overlaps, and boundary compliance. This is the primary basis for detecting physical conflicts.
  * **Right side: Diagonal perspective view** - Used for understanding the space's 3D feel, the actual appearance of furniture, and the harmony and functionality of the overall layout. This is the primary basis for judging layout quality.

**Visual Annotations in Rendered Images:**
Both views contain helpful visual annotations to assist your spatial reasoning:
  * **Floor coordinate markers**: Red dots with (x, z) text labels are placed on the floor to indicate world coordinates, helping you understand the spatial scale and positions.
  * **Object bounding boxes**: Each object is displayed with its 3D bounding box, allowing you to precisely determine object boundaries and detect potential overlaps or collisions.

**Mandatory Execution Requirements:**
You must analyze the scene by combining the visual image and JSON data, and strictly adhere to the following rules for corrections:

1.  **Fix Physical Conflicts**: No objects are ever allowed to overlap or extend beyond the room boundaries (defined by `bounds_top` and `bounds_bottom`). Upon detecting such issues, you must immediately use tools like `move_object` to correct them.
2.  **Optimize Functional Layout**: Based on your understanding of both views, adjust furniture positions to ensure clear traffic flow, functional soundness, and spatial balance.
3.  **Validate All Operations**: Before every tool call, you must mentally pre-calculate its final state during your thinking process to ensure it does not create new conflicts or layout problems.

### Output Format Requirements

You must differentiate your output format based on the task type:

**1. When Editing a Scene (using `tool_calls`):**
You must strictly follow the `<think>`, `<tool_calls>` order.

Format template:
```xml
<think>
[Your detailed analysis and reasoning process here]
- Analyze the rendered image (top-down and perspective views)
- Identify any physical conflicts or layout issues
- Calculate object boundaries and validate positions
- Determine the necessary corrections
</think>

<tool_calls>
[
  {
    "id": "tool_1",
    "name": "tool_name",
    "arguments": {
      "param1": "value1",
      "param2": [array_values]
    }
  },
  {
    "id": "tool_2",
    "name": "tool_name",
    "arguments": {
      "param": "value"
    }
  }
]
</tool_calls>
```

**2. When Creating an Initial Scene (using `create_scene`):**
Directly output the `<create_scene>` tag **without** `<think>`.

**Room Shape Guidelines:**
When creating the initial room boundaries, prefer **rectangular rooms** for simplicity and practicality. Define the room shape using:
  * `bounds_top`: An array of [x, y, z] coordinates defining the upper boundary vertices (ceiling level, y = room height).
  * `bounds_bottom`: An array of [x, y, z] coordinates defining the lower boundary vertices (floor level, y = 0).
For a rectangular room, use exactly 4 vertices in order. The x and z values define the floor plan dimensions, while y defines the height.
Example for a 5m × 5m (25㎡) square room with 2.8m ceiling height:
  * `bounds_top`: [[-2.5, 2.8, 2.5], [2.5, 2.8, 2.5], [2.5, 2.8, -2.5], [-2.5, 2.8, -2.5]]
  * `bounds_bottom`: [[-2.5, 0.0, 2.5], [2.5, 0.0, 2.5], [2.5, 0.0, -2.5], [-2.5, 0.0, -2.5]]

Format template:
```xml
<create_scene>
```json
{
  "bounds_top": [...],
  "bounds_bottom": [...],
  "room_type": "bedroom",
  "room_id": "...",
  "objects": []
}
```
</create_scene>
```

### Available Tools
**1. add_object**: Add a new furniture piece.
* `object_description` (string)
* `position` (array)
* `rotation` (array)
* `size` (array)

**2. remove_object**: Remove an existing object.
* `jid` (string)

**3. move_object**: Change an object's position.
* `jid` (string)
* `new_position` (array)

**4. rotate_object**: Change an object's rotation.
* `jid` (string)
* `new_rotation` (array)

**5. scale_object**: Change an object's size.
* `jid` (string)
* `new_size` (array)

**6. replace_object**: Replace an existing object.
* `jid_to_replace` (string)
* `new_object_description` (string)

**7. terminate**: End the editing session.
* `reason` (string)"""
    
    return optimized_system


def create_initial_turn(requirement: str, scene_json: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    创建初始的user/assistant对话轮次
    
    Args:
        requirement: 用户场景需求
        scene_json: 初始场景JSON字符串
        
    Returns:
        (user_message, assistant_message): user和assistant的message字典
    """
    # 创建user message
    user_message = {
        "role": "user",
        "content": requirement
    }
    
    # 创建assistant message（输出初始空场景）
    assistant_content = f"""<create_scene>
```json
{scene_json}
```
</create_scene>"""
    
    assistant_message = {
        "role": "assistant",
        "content": assistant_content
    }
    
    return user_message, assistant_message


def process_conversation(data: Dict[str, Any], verbose: bool = False) -> tuple[Dict[str, Any], bool]:
    """
    处理单条conversation数据，添加初始场景创建轮次
    
    Args:
        data: 原始数据字典
        verbose: 是否输出详细日志
        
    Returns:
        (处理后的数据字典, 是否成功添加)
    """
    messages = data.get("messages", [])
    
    if len(messages) < 2:
        if verbose:
            print(f"Warning: Conversation {data.get('id', 'unknown')} has less than 2 messages, skipping")
        return data, False
    
    # 1. 更新system prompt
    if messages[0].get("role") == "system":
        original_system = messages[0]["content"]
        updated_system = update_system_prompt(original_system)
        messages[0]["content"] = updated_system
        
        if verbose:
            print(f"✓ Updated system prompt for {data.get('id', 'unknown')}")
    
    # 2. 从第二条消息（第一个user消息）提取需求和场景
    first_user_idx = 1
    if messages[first_user_idx].get("role") != "user":
        if verbose:
            print(f"Warning: Second message is not user role in {data.get('id', 'unknown')}, skipping")
        return data, False
    
    first_user_content = messages[first_user_idx]["content"]
    requirement, scene_json = extract_requirement_and_scene(first_user_content)
    
    if not requirement or not scene_json:
        if verbose:
            print(f"Warning: Could not extract requirement or scene from {data.get('id', 'unknown')}, skipping")
        return data, False
    
    # 3. 创建初始轮次
    initial_user, initial_assistant = create_initial_turn(requirement, scene_json)
    
    # 4. 插入到messages的开头（在system之后）
    new_messages = [messages[0], initial_user, initial_assistant] + messages[1:]
    
    # 5. 更新数据
    data["messages"] = new_messages
    
    # 注意：images数组保持不变，仍然对应原有的user消息
    
    if verbose:
        print(f"✓ Added initial turn for {data.get('id', 'unknown')}")
        print(f"  Requirement length: {len(requirement)} chars")
        print(f"  Scene JSON length: {len(scene_json)} chars")
    
    return data, True


def main():
    parser = argparse.ArgumentParser(
        description="为SFT训练数据添加场景创建轮次"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/path/to/datasets/llmscene/sft/train_dataset_v3.jsonl",
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/path/to/datasets/llmscene/sft/train_dataset_v3_create.jsonl",
        help="输出JSONL文件路径（默认为输入文件名_with_create.jsonl）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="输出详细日志"
    )
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output is None:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_with_create{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {args.input}...")
    print(f"Output will be saved to {output_path}")
    
    # 处理数据
    processed_count = 0
    skipped_count = 0
    
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                processed_data, success = process_conversation(data, verbose=args.verbose)
                
                # 根据返回的成功标志统计
                if success:
                    processed_count += 1
                else:
                    skipped_count += 1
                
                # 写入输出文件
                fout.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                skipped_count += 1
                continue
    
    print(f"\n✓ Processing completed!")
    print(f"  Total processed: {processed_count + skipped_count}")
    print(f"  Successfully added initial turns: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    main()
