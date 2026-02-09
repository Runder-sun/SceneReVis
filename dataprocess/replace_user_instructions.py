#!/usr/bin/env python3
"""
根据scenes_filtered中的场景数据，重新生成用户指令并替换sft_v6中的对话

流程：
1. 遍历sft_v6中的所有场景目录
2. 根据目录名（场景ID）在scenes_filtered中找到对应的场景数据
3. 使用generate_global_user_instruction生成新的用户指令
4. 替换每个编辑链每一轮对话中的用户指令
"""

import os
import sys
import json
import re
import random
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 导入config配置
try:
    from config import (
        AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_DEPLOYMENT_NAME,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_SCOPE,
        MAX_TOKENS,
        TEMPERATURE,
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # 默认配置
    AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
    AZURE_OPENAI_DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
    AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
    AZURE_OPENAI_SCOPE = "YOUR_AZURE_OPENAI_SCOPE"
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    print("Warning: config.py not found, using default configuration")

# 检测是否使用 GPT-5.x 系列（不支持 max_tokens 和 temperature）
IS_GPT5_MODEL = "gpt-5" in AZURE_OPENAI_DEPLOYMENT_NAME.lower()

# 尝试导入Azure OpenAI客户端和认证
try:
    from openai import AzureOpenAI
    from azure.identity import (
        ChainedTokenCredential,
        AzureCliCredential,
        ManagedIdentityCredential,
        get_bearer_token_provider,
    )
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai/azure-identity package not available, will use fallback instructions")


def setup_azure_client() -> Optional['AzureOpenAI']:
    """Create AzureOpenAI client using Azure CLI or managed identity tokens."""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        credential = get_bearer_token_provider(
            ChainedTokenCredential(
                AzureCliCredential(),
                ManagedIdentityCredential(),
            ),
            AZURE_OPENAI_SCOPE,
        )
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_ad_token_provider=credential,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        return client
    except Exception as e:
        print(f"Warning: Failed to initialize Azure OpenAI client: {e}")
        return None


# 初始化客户端
client = setup_azure_client()


def normalize_scene_format(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    将SSR格式（flat objects list）转换为groups格式
    如果已经是groups格式，直接返回
    """
    if "groups" in scene:
        return scene
    
    if "objects" in scene:
        # SSR格式：将所有物体放入一个默认组
        all_objects = scene.get("objects", [])
        bounds_top = scene.get("bounds_top", [])
        bounds_bottom = scene.get("bounds_bottom", [])
        
        groups_scene = {
            "room_envelope": {
                "bounds_top": bounds_top,
                "bounds_bottom": bounds_bottom,
            },
            "room_type": scene.get("room_type", "room"),
            "room_id": scene.get("room_id", "unknown"),
            "groups": [
                {
                    "group_name": "Main Area",
                    "group_type": "living_area",
                    "objects": all_objects
                }
            ]
        }
        return groups_scene
    
    return scene


def analyze_scene(scene: Dict[str, Any]) -> Dict[str, Any]:
    """分析场景内容，提取关键信息用于生成用户指令"""
    # 标准化场景格式
    normalized_scene = normalize_scene_format(scene)
    
    if not normalized_scene or 'groups' not in normalized_scene:
        return {
            "room_type": scene.get("room_type", "room"),
            "total_objects": 0,
            "main_categories": [],
            "object_types": {},
            "groups": [],
            "specific_items": [],
            "furniture_categories": []
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
        elif 'sofa' in desc or 'loveseat' in desc or 'couch' in desc:
            category = 'sofa'
        elif 'bed' in desc:
            category = 'bed'
        elif 'desk' in desc:
            category = 'desk'
        elif 'shelf' in desc or 'bookshelf' in desc or 'bookcase' in desc:
            category = 'storage'
        elif 'lamp' in desc or 'light' in desc:
            category = 'lighting'
        elif 'cabinet' in desc or 'dresser' in desc or 'wardrobe' in desc:
            category = 'storage'
        elif 'tv' in desc or 'television' in desc:
            category = 'media'
        elif 'nightstand' in desc:
            category = 'nightstand'
        else:
            category = 'furniture'
        
        furniture_categories.append(category)
        object_types[category] = object_types.get(category, 0) + 1
    
    # 推断房间类型
    room_type = normalized_scene.get('room_type', scene.get('room_type', 'room'))
    
    return {
        "room_type": room_type,
        "total_objects": len(all_objects),
        "main_categories": list(set(furniture_categories))[:5],
        "object_types": object_types,
        "groups": [(g.get('group_name', ''), g.get('group_type', '')) for g in groups],
        "furniture_categories": furniture_categories,
        "specific_items": specific_items[:5]  # 取前5个具体物品描述
    }


def generate_global_user_instruction(final_scene: Dict[str, Any], instruction_type: str = "mixed") -> str:
    """
    基于最终场景生成全局用户指令
    
    Args:
        final_scene: 最终完整场景的数据
        instruction_type: 指令类型 - "detailed"(详细), "brief"(简短), 或 "mixed"(混合)
    """
    scene_analysis = analyze_scene(final_scene)
    
    # 如果是混合模式，随机选择指令类型
    if instruction_type == "mixed":
        actual_instruction_type = "brief" if random.random() < 0.5 else "detailed"
    else:
        actual_instruction_type = instruction_type
    
    # 如果没有可用的OpenAI客户端，抛出异常
    if client is None:
        raise RuntimeError("Azure OpenAI client is not available. Cannot generate user instructions.")
    
    # 根据实际选择的指令类型生成不同的提示词模板
    if actual_instruction_type == "brief":
        instruction_prompt = f"""You are helping to generate very brief, single-sentence user instructions for interior design. Based on the target scene description below, create a concise, natural user request (ONE sentence only) that expresses their desire to create this space.

## Target Scene Analysis
- Room Type: {scene_analysis['room_type']}
- Total Objects: {scene_analysis['total_objects']}
- Main Furniture: {', '.join(scene_analysis['main_categories'])}
- Key Items: {scene_analysis['specific_items'][:3]}

## Task Requirements

Generate a brief, single-sentence user instruction. Choose ONE style randomly:

**Style A - Simple Goal (30%)**
- "I want a comfortable {scene_analysis['room_type']}"
- "Create a functional workspace"
- "I need a cozy living room"

**Style B - Basic Requirements (40%)**
- "I want a {scene_analysis['room_type']} with seating and storage"
- "Create a workspace with a desk and chair"
- "I need a living room with comfortable seating"

**Style C - Style + Function (30%)**
- "I want a modern {scene_analysis['room_type']} that's both stylish and functional"
- "Create a minimalist workspace with good lighting"
- "I need a cozy living room with comfortable seating"

## Output Requirements:
- Exactly ONE sentence
- No more than 15-20 words
- Natural and conversational
- Reference the room type: {scene_analysis['room_type']}

User Instruction:"""

    else:
        instruction_prompt = f"""You are helping to generate highly diverse and realistic user instructions for interior design. Based on the target scene description below, create a natural user request that expresses their desire to create or design this complete space.

## Target Scene Analysis
- Room Type: {scene_analysis['room_type']}
- Total Objects: {scene_analysis['total_objects']}
- Main Furniture Categories: {', '.join(scene_analysis['main_categories'])}
- Object Distribution: {scene_analysis['object_types']}
- Specific Items Present: {scene_analysis['specific_items'][:3]}

## Task Requirements

Generate a user instruction that expresses their vision for creating this complete space. The instruction should be HIGHLY DIVERSE in style and specificity.

## Instruction Style Variations (choose one randomly):

### A. Highly Specific (25%)
Be very specific about exact furniture pieces, quantities, colors, materials, or styles.

### B. Category-Focused (20%)
Focus on furniture categories without exact descriptions.

### C. Activity-Based (20%)
Frame the request around activities or lifestyle needs.

### D. Aesthetic-Focused (20%)
Emphasize style, mood, or design philosophy.

### E. Simple and Direct (15%)
Brief, straightforward request.

## Output Requirements:
- Natural and conversational
- 1-3 sentences
- Reference the room type: {scene_analysis['room_type']}

User Instruction:"""

    try:
        # GPT-5.x 系列不支持 max_tokens 和 temperature 参数
        if IS_GPT5_MODEL:
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "user", "content": instruction_prompt}
                ],
            )
        else:
            # GPT-4o 等模型支持这些参数
            response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[
                    {"role": "user", "content": instruction_prompt}
                ],
                max_tokens=150 if actual_instruction_type == "brief" else 300,
                temperature=0.9
            )
        
        user_instruction = response.choices[0].message.content.strip()
        
        # 清理可能的引号或多余文本
        if user_instruction.startswith('"') and user_instruction.endswith('"'):
            user_instruction = user_instruction[1:-1]
        if user_instruction.startswith("User Instruction:"):
            user_instruction = user_instruction.replace("User Instruction:", "").strip()
        
        return user_instruction
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate instruction with GPT: {e}")


def replace_instruction_in_message(message_content: str, new_instruction: str) -> str:
    """
    替换消息内容中的用户指令
    
    用户消息格式：
    <image>
    {user_instruction}
    
    <current_scene>
    ...
    """
    # 匹配 <image>\n...指令...\n\n<current_scene> 的模式
    pattern = r'(<image>\n)(.+?)(\n\n<current_scene>)'
    
    def replacer(match):
        return f"{match.group(1)}{new_instruction}{match.group(3)}"
    
    # 使用 DOTALL 标志让 . 匹配换行符
    new_content = re.sub(pattern, replacer, message_content, flags=re.DOTALL)
    
    return new_content


def process_conversation_file(conversation_path: Path, scene_data: Dict[str, Any], dry_run: bool = False) -> bool:
    """
    处理单个对话文件，替换其中的用户指令
    
    Args:
        conversation_path: 对话JSON文件路径
        scene_data: 对应的场景数据
        dry_run: 如果为True，不实际写入文件
    
    Returns:
        是否成功处理
    """
    try:
        with open(conversation_path, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        
        # 为这个编辑链生成新的用户指令
        new_instruction = generate_global_user_instruction(scene_data, "mixed")
        
        # 替换所有用户消息中的指令
        messages = conversation.get('messages', [])
        modified = False
        
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if '<image>' in content and '<current_scene>' in content:
                    new_content = replace_instruction_in_message(content, new_instruction)
                    if new_content != content:
                        msg['content'] = new_content
                        modified = True
        
        # 更新metadata中的global_user_instruction
        if 'metadata' in conversation:
            conversation['metadata']['global_user_instruction'] = new_instruction
        
        if modified and not dry_run:
            with open(conversation_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"    Error processing {conversation_path}: {e}")
        return False


def process_scene_directory(scene_dir: Path, scenes_filtered_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """
    处理单个场景目录
    
    Args:
        scene_dir: sft_v6中的场景目录
        scenes_filtered_dir: scenes_filtered目录
        dry_run: 如果为True，不实际写入文件
    
    Returns:
        处理结果统计
    """
    scene_id = scene_dir.name
    result = {
        "scene_id": scene_id,
        "success": False,
        "chains_processed": 0,
        "chains_failed": 0,
        "error": None
    }
    
    # 查找对应的场景数据文件
    scene_file = scenes_filtered_dir / f"{scene_id}.json"
    
    if not scene_file.exists():
        result["error"] = f"Scene file not found: {scene_file}"
        return result
    
    try:
        with open(scene_file, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
    except Exception as e:
        result["error"] = f"Failed to load scene file: {e}"
        return result
    
    # 遍历场景目录下的所有chain子目录
    for chain_dir in scene_dir.iterdir():
        if chain_dir.is_dir() and chain_dir.name.startswith('chain_'):
            # 查找conversation文件
            conversation_files = list(chain_dir.glob('conversation_*.json'))
            
            for conv_file in conversation_files:
                if process_conversation_file(conv_file, scene_data, dry_run):
                    result["chains_processed"] += 1
                else:
                    result["chains_failed"] += 1
    
    result["success"] = result["chains_processed"] > 0 and result["chains_failed"] == 0
    return result


def main():
    import argparse
    import time
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Replace user instructions in SFT data based on scene descriptions')
    parser.add_argument('--sft_dir', type=str, 
                        default='/path/to/datasets/llmscene/sft/sft_v6',
                        help='Path to sft_v6 directory')
    parser.add_argument('--scenes_dir', type=str,
                        default='/path/to/datasets/ssr/scenes_filtered',
                        help='Path to scenes_filtered directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, do not write changes to files')
    parser.add_argument('--max_workers', type=int, default=6,
                        help='Number of parallel processes (default: 6)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of scenes to process')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint (skip already processed scenes)')
    parser.add_argument('--checkpoint_file', type=str, default='processed_scenes.txt',
                        help='File to track processed scenes')
    parser.add_argument('--log_file', type=str, default='replace_instructions.log',
                        help='Log file path')
    
    args = parser.parse_args()
    
    sft_dir = Path(args.sft_dir)
    scenes_dir = Path(args.scenes_dir)
    checkpoint_file = Path(args.checkpoint_file)
    log_file = Path(args.log_file)
    
    # 设置日志
    def log(message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    if not sft_dir.exists():
        log(f"Error: SFT directory not found: {sft_dir}")
        return
    
    if not scenes_dir.exists():
        log(f"Error: Scenes directory not found: {scenes_dir}")
        return
    
    # 加载已处理的场景（断点续传）
    processed_scenes = set()
    if args.resume and checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            processed_scenes = set(line.strip() for line in f)
        log(f"Resuming from checkpoint, {len(processed_scenes)} scenes already processed")
    
    # 获取所有场景目录
    scene_dirs = [d for d in sft_dir.iterdir() if d.is_dir()]
    
    # 过滤已处理的场景
    if args.resume:
        scene_dirs = [d for d in scene_dirs if d.name not in processed_scenes]
    
    if args.limit:
        scene_dirs = scene_dirs[:args.limit]
    
    log(f"Found {len(scene_dirs)} scene directories to process")
    log(f"Dry run: {args.dry_run}")
    log(f"Using OpenAI API: {client is not None}")
    log(f"Model: {AZURE_OPENAI_DEPLOYMENT_NAME}")
    log(f"Threads: {args.max_workers}")
    log("")
    
    # 统计
    total_success = 0
    total_failed = 0
    total_chains = 0
    start_time = time.time()
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_scene_directory, scene_dir, scenes_dir, args.dry_run): scene_dir
            for scene_dir in scene_dirs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
            scene_dir = futures[future]
            try:
                result = future.result()
                if result["success"]:
                    total_success += 1
                    # 记录已处理的场景
                    if not args.dry_run:
                        with open(checkpoint_file, 'a') as f:
                            f.write(result["scene_id"] + '\n')
                else:
                    total_failed += 1
                    if result["error"]:
                        log(f"  {result['scene_id']}: {result['error']}")
                
                total_chains += result["chains_processed"]
                
                # 每100个场景记录一次进度
                processed = total_success + total_failed
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed
                    remaining = avg_time * (len(futures) - processed)
                    log(f"Progress: {processed}/{len(futures)} scenes, "
                        f"Elapsed: {elapsed:.1f}s, ETA: {remaining:.1f}s")
                
            except Exception as e:
                total_failed += 1
                log(f"  Error processing {scene_dir.name}: {e}")
    
    elapsed = time.time() - start_time
    log("")
    log("=" * 50)
    log("Summary:")
    log(f"  Scenes processed successfully: {total_success}")
    log(f"  Scenes failed: {total_failed}")
    log(f"  Total chains updated: {total_chains}")
    log(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    log(f"  Average time per scene: {elapsed/(total_success+total_failed):.2f}s")
    log("=" * 50)


if __name__ == "__main__":
    main()
