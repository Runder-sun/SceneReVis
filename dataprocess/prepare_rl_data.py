import pandas as pd  
import json  
import os  
import re  
from typing import List, Dict, Any, Tuple  
import argparse  
import random  


# System prompt for scene generation task (same as add_create_scene.py)
SCENE_GENERATION_SYSTEM_PROMPT = """### Role and Core Directive

You are an AI spatial layout planner. Your core task is to analyze and optimize indoor scenes, ensuring they are physically valid and functionally efficient.

### Core Capabilities

Your primary responsibility is to **diagnose and correct** problems in the current scene.

1.  **Analyze and Identify Problems**: Based on the input rendered image and scene JSON, proactively identify three types of issues:

      * **Physical Conflicts**: Objects overlapping with each other or extending beyond the defined room boundaries.
      * **Poor Layout**: Furniture placement that obstructs main traffic flows, object orientation or grouping that is not functionally logical, or imbalanced use of space.

2.  **Resolve and Optimize**: Once problems are identified, you must use the available tools in `tool_calls` to automatically correct them, aiming to create a scene that is free of physical conflicts, has clear circulation, and a rational functional layout.

### Common Objects Reference

Here are some common objects found in various room types to guide your scene generation and optimization:

*   **Living Room**: Sofa, Coffee Table, TV Stand, Armchair, Bookshelf, Floor Lamp, Rug, Side Table, Plant.
*   **Bedroom**: Bed, Nightstand, Wardrobe, Dresser, Desk, Chair, Mirror, Lamp, Rug.
*   **Dining Room**: Dining Table, Dining Chair, Sideboard, Chandelier, Rug, Cabinet.
*   **Office**: Desk, Office Chair, Conference Table, Filing Cabinet, Whiteboard, Sofa, Plant.
*   **Study Room**: Desk, Office Chair, Bookshelf, Filing Cabinet, Lamp, Armchair, Rug.
*   **Gym**: Treadmill, Exercise Bike, Dumbbell Rack, Yoga Mat, Bench, Mirror, Gym Ball.
*   **Entertainment Room**: Sofa, TV Stand, Pool Table, Ping Pong Table, Gaming Desk, Gaming Chair, Karaoke Machine, Speaker, Bar Counter.

### Scene Analysis and Spatial Rules

Your input will include a rendered image and the scene's JSON data. The rendered image displays two key views side-by-side, which you must use in combination for a comprehensive judgment:

  * **Left side: Top-down view** - Used for precisely judging relative positions, spacing, overlaps, and boundary compliance. This is the primary basis for detecting physical conflicts.
  * **Right side: Diagonal perspective view** - Used for understanding the space's 3D feel, the actual appearance of furniture, and the harmony and functionality of the overall layout. This is the primary basis for judging layout quality.

**Mandatory Execution Requirements:**
You must analyze the scene by combining the visual image and JSON data, and strictly adhere to the following rules for corrections:

1.  **Fix Physical Conflicts**: No objects are ever allowed to overlap or extend beyond the room boundaries (defined by `bounds_top` and `bounds_bottom`). Upon detecting such issues, you must immediately use tools like `move_object` to correct them.
2.  **Optimize Functional Layout**: Based on your understanding of both views, adjust furniture positions to ensure clear traffic flow, functional soundness, and spatial balance.
3.  **Validate All Operations**: Before every tool call, you must mentally pre-calculate its final state during your thinking process to ensure it does not create new conflicts or layout problems.
4.  **Empty Scene Strategy**: If the scene is empty or lacks essential furniture, prioritize adding all necessary objects first to establish the functional base, then refine their positions and layout in subsequent steps.

### Output Format Requirements

You must differentiate your output format based on the task type:

**1. When Editing a Scene (using `tool_calls`):**
You must strictly follow the `<think>`,  `<tool_calls>` order.

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
* `jid/uid` (string)

**3. move_object**: Change an object's position.
* `jid/uid` (string)
* `new_position` (array)

**4. rotate_object**: Change an object's rotation.
* `jid/uid` (string)
* `new_rotation` (array)

**5. scale_object**: Change an object's size.
* `jid/uid` (string)
* `new_size` (array)

**6. replace_object**: Replace an existing object.
* `jid/uid_to_replace` (string)
* `new_object_description` (string)

**7. terminate**: End the editing session.
* `reason` (string)"""


def extract_scene_from_content(content: str) -> Tuple[str, dict]:
    """  
    从content中提取场景JSON和清理后的文本  
      
    Args:  
        content: 包含<current_scene>标记的原始content  
      
    Returns:  
        (cleaned_content, scene_dict): 清理后的文本和提取的场景JSON  
    """  
    # 提取<current_scene>标记内的JSON  
    scene_pattern = r'<current_scene>\s*```json\s*(.*?)\s*```\s*</current_scene>'  
    match = re.search(scene_pattern, content, re.DOTALL)  
      
    if match:  
        scene_json_str = match.group(1)  
        try:  
            scene_dict = json.loads(scene_json_str)  
        except json.JSONDecodeError as e:  
            print(f"Warning: Failed to parse scene JSON: {e}")  
            return content, {}  
          
        return content, scene_dict  
    else:  
        return content, {}  
  

def extract_user_requirement(content: str) -> str:
    """从user content中提取<image>与<current_scene>之间的用户需求描述"""
    image_idx = content.find('<image>')
    if image_idx == -1:
        return ""

    start_idx = image_idx + len('<image>')
    current_scene_idx = content.find('<current_scene>', start_idx)
    end_idx = current_scene_idx if current_scene_idx != -1 else len(content)

    requirement = content[start_idx:end_idx].strip()
    return requirement

  
def convert_messages_to_rl_format(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:  
    """  
    将messages格式的多轮对话数据转换为VERL RL训练格式  
      
    Args:  
        raw_data: 原始数据列表,每个元素包含:  
            - messages: List[Dict] 多轮对话消息  
            - images: List[str] 图像路径列表(可选,第一轮不需要)  
            - target_scene: Dict 目标场景JSON(可选)  
            - max_turns: int 最大轮数(可选)  
      
    Returns:  
        转换后的RL格式数据列表  
    """  
    rl_data = []  
      
    for idx, item in enumerate(raw_data):  
        messages = item.get("messages", [])  
          
        # 提取初始prompt(只保留第一轮对话: system + user需求)  
        initial_messages = []  
        user_requirement = ""
          
        for msg in messages:  
            if msg["role"] == "system":  
                initial_messages.append(msg)  
            elif msg["role"] == "user":  
                # 第一轮对话:只有用户需求文本,没有<image>和<current_scene>  
                content = msg["content"]
                
                # 提取纯文本需求(移除可能存在的<image>和<current_scene>标签)
                # 如果有<image>标签,提取其后的文本
                if '<image>' in content:
                    # 兼容性:处理旧格式数据
                    user_requirement = extract_user_requirement(content)
                else:
                    # 新格式:纯文本需求
                    user_requirement = content.strip()
                
                # 第一轮只保留纯文本需求
                initial_messages.append({  
                    "role": "user",  
                    "content": user_requirement  
                })  
                  
                break  # 只保留第一个user消息  
          
        # 验证是否有有效的初始消息  
        if not initial_messages:  
            print(f"Warning: Sample {idx} has no valid initial messages, skipping")  
            continue
        
        # 验证是否有用户需求
        if not user_requirement:
            print(f"Warning: Sample {idx} has no user requirement, skipping")
            continue
          
        # 构造RL数据格式  
        rl_item = {  
            "prompt": initial_messages,  # List[Dict]格式,符合chat template要求  
            "data_source": "scene_editing",  
            "ability": "scene_editing",  
            "reward_model": {"style": "rule",
                             "ground_truth": "placeholder"},
            "agent_name": "tool_agent",  # 指定使用工具代理  
        }  
          
        # 第一轮不需要图像
        # 注释掉图像相关代码
        # if "images" in item and item["images"]:  
        #     image_path = item["images"][0]  
        #     rl_item["images"] = [{"image": image_path}]
          
        # 第一轮不需要initial_scene,模型会自己生成
        # 构建interaction_kwargs (不包含initial_scene以避免Parquet序列化错误)
        interaction_kwargs = {  
            "name": "scene_editing",
            # 不保存initial_scene,避免空struct导致Parquet错误
            # SceneEditingInteraction会检查kwargs中是否有initial_scene
        }  
        
        # 保存用户需求到interaction_kwargs
        if user_requirement:
            interaction_kwargs["user_requirement"] = user_requirement  
          
        # 只有当原始数据包含这些字段时才添加  
        if "target_scene" in item:  
            interaction_kwargs["target_scene"] = item["target_scene"]  
          
        # 不再添加max_turns到interaction_kwargs
        # if "max_turns" in item:  
        #     interaction_kwargs["max_turns"] = item["max_turns"]  
        # else:  
        #     interaction_kwargs["max_turns"] = 10  # 使用默认值  
          
        # 添加extra_info,包含interaction_kwargs  
        rl_item["extra_info"] = {  
            "index": idx,  
            "interaction_kwargs": interaction_kwargs  
        }  
          
        rl_data.append(rl_item)  
      
    return rl_data  
  
  
def split_train_val(data: List[Dict[str, Any]], val_ratio: float = 0.1, seed: int = 42, max_train_samples: int = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:  
    """  
    将数据划分为训练集和验证集  
      
    Args:  
        data: 完整数据列表  
        val_ratio: 验证集比例(默认0.1即10%)  
        seed: 随机种子,用于可复现的划分
        max_train_samples: 最大训练样本数，如果指定则从完整数据中抽取，保持val_ratio不变
      
    Returns:  
        (train_data, val_data): 训练集和验证集  
    """  
    # 设置随机种子以确保可复现性  
    random.seed(seed)  
      
    # 随机打乱数据  
    shuffled_data = data.copy()  
    random.shuffle(shuffled_data)  
    
    # 如果指定了max_train_samples，先计算需要的总样本数
    if max_train_samples is not None and max_train_samples > 0:
        # 根据val_ratio计算需要的总样本数
        # train_samples / total_samples = (1 - val_ratio)
        # 所以 total_samples = train_samples / (1 - val_ratio)
        total_needed = int(max_train_samples / (1 - val_ratio))
        
        # 如果需要的样本数超过数据集大小，使用全部数据
        if total_needed > len(shuffled_data):
            print(f"Warning: Requested {max_train_samples} train samples (total {total_needed} with val)")
            print(f"         but only {len(shuffled_data)} samples available. Using all data.")
            total_needed = len(shuffled_data)
        else:
            # 从打乱的数据中取前total_needed个
            shuffled_data = shuffled_data[:total_needed]
            print(f"Sampling {total_needed} samples from full dataset to get ~{max_train_samples} train samples")
      
    # 计算划分点  
    total_size = len(shuffled_data)  
    val_size = int(total_size * val_ratio)  
    train_size = total_size - val_size  
      
    # 划分数据  
    train_data = shuffled_data[:train_size]  
    val_data = shuffled_data[train_size:]  
      
    return train_data, val_data  
  
  
def load_messages_data(input_file: str) -> List[Dict[str, Any]]:  
    """  
    从JSON或JSONL文件加载messages格式的数据  
      
    Args:  
        input_file: 输入文件路径(.json或.jsonl)  
      
    Returns:  
        数据列表  
    """  
    if input_file.endswith('.jsonl'):  
        data = []  
        with open(input_file, 'r', encoding='utf-8') as f:  
            for line in f:  
                if line.strip():  
                    data.append(json.loads(line))  
        return data  
    elif input_file.endswith('.json'):  
        with open(input_file, 'r', encoding='utf-8') as f:  
            return json.load(f)  
    else:  
        raise ValueError(f"Unsupported file format: {input_file}. Only .json and .jsonl are supported.")  


def load_room_descriptions(input_file: str) -> List[Dict[str, Any]]:
    """
    从room_descriptions JSON文件加载房间描述数据
    
    Args:
        input_file: 输入文件路径(.json)，格式为room_descriptions_2k.json
    
    Returns:
        房间描述列表，每个元素包含 room_type, detail_level, target_object_count, description
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 检查是否是room_descriptions格式
    if "descriptions" in data and "metadata" in data:
        print(f"✓ Detected room_descriptions format")
        print(f"  Total descriptions: {data['metadata'].get('total_descriptions', len(data['descriptions']))}")
        print(f"  Room types: {data['metadata'].get('room_types', [])}")
        print(f"  Detail levels: {data['metadata'].get('detail_levels', [])}")
        return data["descriptions"]
    else:
        raise ValueError("Invalid room_descriptions format. Expected 'descriptions' and 'metadata' keys.")


def convert_room_descriptions_to_rl_format(descriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将room_descriptions格式的数据转换为VERL RL训练格式
    
    生成与scene_editing完全相同的格式，只使用description字段作为用户需求
    
    Args:
        descriptions: 房间描述列表，每个元素包含:
            - description: 房间描述文本（唯一使用的字段）
    
    Returns:
        转换后的RL格式数据列表（与convert_messages_to_rl_format输出格式一致）
    """
    rl_data = []
    
    for idx, item in enumerate(descriptions):
        # 只使用description字段
        user_requirement = item.get("description", "").strip()
        
        if not user_requirement:
            print(f"Warning: Sample {idx} has no description, skipping")
            continue
        
        # 构造初始消息（system + user），与scene_editing格式一致
        initial_messages = [
            {
                "role": "system",
                "content": SCENE_GENERATION_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_requirement
            }
        ]
        
        # 构造RL数据格式，与scene_editing完全一致
        rl_item = {
            "prompt": initial_messages,
            "data_source": "scene_editing",
            "ability": "scene_editing",
            "reward_model": {
                "style": "rule",
                "ground_truth": "placeholder"
            },
            "agent_name": "tool_agent",  # 指定使用工具代理
        }
        
        # 构建interaction_kwargs，与scene_editing格式一致
        interaction_kwargs = {
            "name": "scene_editing",
            "user_requirement": user_requirement,
        }
        
        # 添加extra_info，与scene_editing格式一致
        rl_item["extra_info"] = {
            "index": idx,
            "interaction_kwargs": interaction_kwargs
        }
        
        rl_data.append(rl_item)
    
    return rl_data


def save_to_parquet(data: List[Dict[str, Any]], output_file: str):  
    """  
    将数据保存为parquet格式  
      
    Args:  
        data: 数据列表  
        output_file: 输出文件路径(.parquet)  
    """  
    df = pd.DataFrame(data)  
      
    # 确保输出目录存在  
    output_dir = os.path.dirname(output_file)  
    if output_dir:  
        os.makedirs(output_dir, exist_ok=True)  
      
    # 保存为parquet  
    df.to_parquet(output_file, index=False)  
    print(f"✓ Saved {len(data)} samples to {output_file}")  
  
  
def main():  
    parser = argparse.ArgumentParser(  
        description="Convert messages format to VERL RL format for scene editing training with train/val split"  
    )  
    parser.add_argument(  
        "--input",   
        type=str,   
        default="/path/to/datasets/llmscene/room_descriptions_1k.json", 
        help="Input data file (.json or .jsonl)"  
    )  
    parser.add_argument(
        "--input_format",
        type=str,
        choices=["messages", "room_descriptions"],
        default="room_descriptions",
        help="Input data format: 'messages' for SFT conversation data, 'room_descriptions' for room description JSON"
    )
    parser.add_argument(  
        "--output_dir",   
        type=str,   
        default="/path/to/data/datasets/rl",  
        help="Output directory for parquet files"  
    )  
    parser.add_argument(  
        "--train_output",   
        type=str,   
        default="scene_editing_train_ood.parquet",  
        help="Output training parquet filename"  
    )  
    parser.add_argument(  
        "--val_output",   
        type=str,   
        default="scene_editing_val_ood.parquet",  
        help="Output validation parquet filename"  
    )  
    parser.add_argument(  
        "--val_ratio",  
        type=float,  
        default=0.1,  
        help="Ratio of validation set (default: 0.1 for 10%%)"  
    )  
    parser.add_argument(  
        "--seed",  
        type=int,  
        default=42,  
        help="Random seed for reproducible train/val split"  
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=900,
        help="Maximum number of training samples to use. If specified, will sample from full dataset while maintaining val_ratio. If None or 0, use all data."
    )  
      
    args = parser.parse_args()  
    
    # 根据input_format选择不同的加载和转换方式
    if args.input_format == "room_descriptions":
        # 加载房间描述数据
        print(f"Loading room descriptions from {args.input}...")
        descriptions = load_room_descriptions(args.input)
        print(f"✓ Loaded {len(descriptions)} room descriptions")
        
        # 转换为RL格式
        print("Converting room descriptions to RL format...")
        rl_data = convert_room_descriptions_to_rl_format(descriptions)
        print(f"✓ Converted {len(rl_data)} samples")
    else:
        # 原有的messages格式处理
        print(f"Loading data from {args.input}...")  
        raw_data = load_messages_data(args.input)  
        print(f"✓ Loaded {len(raw_data)} samples")  
          
        # 转换为RL格式  
        print("Converting data to RL format...")  
        rl_data = convert_messages_to_rl_format(raw_data)  
        print(f"✓ Converted {len(rl_data)} samples")  
      
    # 划分训练集和验证集  
    if args.max_train_samples:
        print(f"Splitting data with max_train_samples={args.max_train_samples}, val_ratio={args.val_ratio}, seed={args.seed}...")
    else:
        print(f"Splitting data with val_ratio={args.val_ratio}, seed={args.seed}...")
    train_data, val_data = split_train_val(rl_data, val_ratio=args.val_ratio, seed=args.seed, max_train_samples=args.max_train_samples)  
    print(f"✓ Train set: {len(train_data)} samples")  
    print(f"✓ Val set: {len(val_data)} samples")
    
    # 显示实际的train/val比例
    if len(train_data) > 0 and len(val_data) > 0:
        actual_ratio = len(val_data) / (len(train_data) + len(val_data))
        print(f"✓ Actual val ratio: {actual_ratio:.3f} (target: {args.val_ratio})")  
      
    # 保存训练集  
    train_output_path = os.path.join(args.output_dir, args.train_output)  
    save_to_parquet(train_data, train_output_path)  
      
    # 保存验证集  
    val_output_path = os.path.join(args.output_dir, args.val_output)  
    save_to_parquet(val_data, val_output_path)  
      
    print("\n✓ Data conversion and split completed successfully!")  
    print(f"  Training data: {train_output_path}")  
    print(f"  Validation data: {val_output_path}")  
    print(f"  Train/Val ratio: {len(train_data)}/{len(val_data)} ({(1-args.val_ratio)*100:.0f}%/{args.val_ratio*100:.0f}%)")  
  
  
if __name__ == "__main__":  
    main()