#!/usr/bin/env python3
"""
生成房间描述数据集
使用GPT-4生成2000条不同类型房间的描述，涵盖多种房间类型和详细程度
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
from openai import AzureOpenAI
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入配置
sys.path.append(str(Path(__file__).parent))
from config import *

# 定义房间类型（10种常见室内房间）
ROOM_TYPES = [
    "living room",      # 客厅
    "bedroom",          # 卧室
    "dining room",      # 餐厅
    "office",           # 办公室
    "study room",       # 书房
    "gym",              # 健身房
    "entertainment room", # 娱乐室（桌游、KTV、台球、乒乓球、电子游戏等）
]

# Define description detail levels (max 5 sentences)
# Weights control generation ratio: detailed has higher weight, generating more detailed descriptions
DESCRIPTION_LEVELS = {
    "brief": {
        "name": "Brief Description",
        "instruction": "Generate a brief room description in 1-2 sentences, mentioning only key furniture and basic layout.",
        "weight": 3  # 10% ratio
    },
    "moderate": {
        "name": "Moderate Description",
        "instruction": "Generate a moderate room description in 3-4 sentences, including furniture, basic decorations, and spatial relationships.",
        "weight": 4  # 20% ratio
    },
    "detailed": {
        "name": "Detailed Description",
        "instruction": "Generate a detailed room description in 4-5 sentences, including furniture, decorations, materials, colors, lighting, and spatial layout.",
        "weight": 3  # 50% ratio
    }
}

# 物体数量范围（与描述详细程度独立）
OBJECT_COUNT_RANGE = {
    "min": 5,
    "max": 30
}

# 增加风格定义以提升多样性
STYLES = [
    "Modern", "Minimalist", "Industrial", "Scandinavian", 
    "Traditional", "Bohemian", "Rustic", "Contemporary",
    "Mid-century Modern", "Japanese"
]

# 娱乐室的具体类型
ENTERTAINMENT_TYPES = [
    "Board Game Room", "KTV Room", "Billiards Room", 
    "Ping Pong Room", "Video Game Room", "Home Theater",
    "Music Room", "Arcade Room"
]

# 房间大小/空间特征
ROOM_SIZES = [
    "Small and compact", "Spacious and airy", "Narrow and long", 
    "Open-plan", "Standard size", "Large with high ceiling"
]

# 氛围/色调主题
ATMOSPHERES = [
    "Cozy and warm", "Bright and energetic", "Dark and moody", 
    "Luxurious and elegant", "Cluttered and lived-in", "Organized and neat",
    "Natural and earthy", "Cool and futuristic"
]


def setup_azure_client() -> AzureOpenAI:
    """创建AzureOpenAI客户端，优先使用API Key，必要时回退到Azure AD凭据"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        return AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=api_key,
            api_version=AZURE_OPENAI_API_VERSION,
        )

    scope = AZURE_OPENAI_SCOPE
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=credential,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def generate_single_room_description(
    client: AzureOpenAI,
    room_type: str,
    level: str,
    object_count: int,
    index: int,
    style: str = None,
    sub_type: str = None,
    room_size: str = None,
    atmosphere: str = None
) -> Dict[str, Any]:
    """
    生成单个房间描述
    
    Args:
        client: Azure OpenAI客户端
        room_type: 房间类型
        level: 描述详细程度（brief/moderate/detailed）
        object_count: 房间中物体的数量
        index: 房间描述的索引号
        style: 装修风格
        sub_type: 具体的房间子类型
        room_size: 房间大小/空间特征
        atmosphere: 氛围/色调主题
        
    Returns:
        包含房间描述信息的字典
    """
    level_info = DESCRIPTION_LEVELS[level]
    
    # 确定显示的房间类型名称
    display_room_type = sub_type if sub_type else room_type
    
    # 构建额外的指令
    style_instruction = f"- Style: {style}" if style else ""
    size_instruction = f"- Room Size/Layout: {room_size}" if room_size else ""
    atmosphere_instruction = f"- Atmosphere/Mood: {atmosphere}" if atmosphere else ""
    
    # 构建提示词
    prompt = f"""Generate a realistic {display_room_type} description for 3D scene generation.

Requirements:
- Description level: {level_info['name']} ({level_info['instruction']})
{style_instruction}
{size_instruction}
{atmosphere_instruction}
- Explicitly mention that this is a "{display_room_type}" in the description
- Include approximately {object_count} furniture items
- Focus ONLY on furniture and movable objects (beds, chairs, tables, sofas, lamps, shelves, etc.)
- Do NOT describe walls, floors, ceilings, windows, doors, or any architectural elements
- Be specific about furniture types, positions, and arrangements
- Include realistic spatial relationships between furniture (e.g., "next to", "in the corner", "facing")
- Mention materials, colors, or styles of furniture when appropriate for this detail level
- Make it sound natural and practical for actual room design

Output format: Provide only the room description text focusing on furniture, no additional commentary.
IMPORTANT: Output MUST be in English only. Do not use any other language.

Room type: {display_room_type}
Style: {style if style else 'Any'}
Size: {room_size if room_size else 'Any'}
Atmosphere: {atmosphere if atmosphere else 'Any'}
Number of furniture items to include: {object_count}"""

    try:
        # 调用 GPT-5 Responses API（支持新版 response schema）
        response = client.responses.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            input=[
                {
                    "role": "system",
                    "content": "You are an expert interior designer who creates detailed, realistic room descriptions for 3D scene generation. Your descriptions should be practical and suitable for creating actual 3D scenes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_output_tokens=min(MAX_TOKENS, 1000),
        )

        # 兼容新版 Responses 返回结构（优先使用 output 字段）
        description = None
        if getattr(response, "output", None):
            try:
                description = response.output[0].content[0].text.strip()
            except Exception:
                pass

        # 兼容旧版 choices 结构（如仍返回 chat.completions 风格）
        if description is None and getattr(response, "choices", None):
            description = response.choices[0].message.content.strip()

        if description is None:
            raise ValueError("No text found in response (output/choices empty)")
        
        # 构建返回结果
        result = {
            "index": index,
            "room_type": room_type,
            "sub_type": sub_type,
            "style": style,
            "room_size": room_size,
            "atmosphere": atmosphere,
            "detail_level": level,
            "target_object_count": object_count,
            "description": description,
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            }
        }
        
        return result
        
    except Exception as e:
        print(f"\n✗ 生成失败 (index={index}): {e}")
        return None


def generate_room_descriptions(
    total_count: int = 2000,
    output_file: str = None,
    batch_size: int = 10,
    delay: float = 0.5,
    max_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    生成指定数量的房间描述
    
    Args:
        total_count: 总共生成的描述数量
        output_file: 输出文件路径
        batch_size: 每批处理数量（用于中间保存）
        delay: 每次API调用之间的延迟（秒），仅在单线程模式下生效
        max_workers: 并行生成的线程数
        
    Returns:
        生成的房间描述列表
    """
    print(f"开始生成 {total_count} 条房间描述...")
    print(f"房间类型: {len(ROOM_TYPES)} 种")
    print(f"详细程度: {len(DESCRIPTION_LEVELS)} 级")
    print(f"物体数量范围: 5-30 个")
    print(f"并行线程数: {max_workers}\n")
    
    # 创建Azure OpenAI客户端
    client = setup_azure_client()
    
    descriptions = []
    
    # 计算每种房间类型和详细程度的分配
    # 使用权重来控制不同详细程度的生成比例
    total_weight = sum(info['weight'] for info in DESCRIPTION_LEVELS.values())
    items_per_room_type = total_count // len(ROOM_TYPES)
    
    # 生成任务列表
    tasks = []
    task_index = 0
    
    for room_type in ROOM_TYPES:
        # 根据权重分配每种详细程度的数量
        for level, level_info in DESCRIPTION_LEVELS.items():
            level_count = int(items_per_room_type * level_info['weight'] / total_weight)
            for _ in range(level_count):
                # 物体数量在5-30之间随机选择，与描述详细程度独立
                object_count = random.randint(
                    OBJECT_COUNT_RANGE['min'],
                    OBJECT_COUNT_RANGE['max']
                )
                # 随机选择风格
                style = random.choice(STYLES)
                # 随机选择房间大小和氛围
                room_size = random.choice(ROOM_SIZES)
                atmosphere = random.choice(ATMOSPHERES)
                # 如果是娱乐室，随机选择子类型
                sub_type = random.choice(ENTERTAINMENT_TYPES) if room_type == "entertainment room" else None
                
                tasks.append((room_type, level, object_count, task_index, style, sub_type, room_size, atmosphere))
                task_index += 1
    
    # 补充剩余任务到总数（使用加权随机选择，偏向detailed）
    level_weights = [(level, info['weight']) for level, info in DESCRIPTION_LEVELS.items()]
    levels = [l[0] for l in level_weights]
    weights = [l[1] for l in level_weights]
    
    while len(tasks) < total_count:
        room_type = random.choice(ROOM_TYPES)
        # 使用加权随机选择详细程度
        level = random.choices(levels, weights=weights, k=1)[0]
        # 物体数量在5-30之间随机选择
        object_count = random.randint(
            OBJECT_COUNT_RANGE['min'],
            OBJECT_COUNT_RANGE['max']
        )
        # 随机选择风格
        style = random.choice(STYLES)
        # 随机选择房间大小和氛围
        room_size = random.choice(ROOM_SIZES)
        atmosphere = random.choice(ATMOSPHERES)
        # 如果是娱乐室，随机选择子类型
        sub_type = random.choice(ENTERTAINMENT_TYPES) if room_type == "entertainment room" else None
        
        tasks.append((room_type, level, object_count, task_index, style, sub_type, room_size, atmosphere))
        task_index += 1
    
    # 随机打乱任务顺序以增加多样性
    random.shuffle(tasks)
    
    # 执行生成任务
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(
                    generate_single_room_description,
                    client,
                    room_type,
                    level,
                    object_count,
                    idx,
                    style,
                    sub_type,
                    room_size,
                    atmosphere
                ): idx for room_type, level, object_count, idx, style, sub_type, room_size, atmosphere in tasks
            }
            
            for i, future in enumerate(tqdm(as_completed(future_to_task), total=len(tasks), desc="生成房间描述")):
                try:
                    result = future.result()
                    if result:
                        descriptions.append(result)
                except Exception as e:
                    print(f"任务执行异常: {e}")
                
                # 定期保存中间结果
                if (i + 1) % batch_size == 0 and output_file:
                    save_descriptions(descriptions, output_file + ".partial")
                    # print(f"\n已保存中间结果: {len(descriptions)} 条描述")
    else:
        for i, (room_type, level, object_count, idx, style, sub_type, room_size, atmosphere) in enumerate(tqdm(tasks, desc="生成房间描述")):
            result = generate_single_room_description(
                client,
                room_type,
                level,
                object_count,
                idx,
                style,
                sub_type,
                room_size,
                atmosphere
            )
            
            if result:
                descriptions.append(result)
            
            # 添加延迟避免API速率限制
            if i < len(tasks) - 1:  # 最后一个不需要延迟
                time.sleep(delay)
            
            # 定期保存中间结果
            if (i + 1) % batch_size == 0 and output_file:
                save_descriptions(descriptions, output_file + ".partial")
                print(f"\n已保存中间结果: {len(descriptions)} 条描述")
    
    # 保存最终结果
    if output_file:
        save_descriptions(descriptions, output_file)
        # 删除中间文件
        partial_file = Path(output_file + ".partial")
        if partial_file.exists():
            partial_file.unlink()
    
    return descriptions


def save_descriptions(descriptions: List[Dict[str, Any]], output_file: str):
    """保存描述到JSON文件"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 计算统计信息
    stats = {
        "total_count": len(descriptions),
        "room_type_distribution": {},
        "detail_level_distribution": {},
        "object_count_stats": {
            "min": min(d['target_object_count'] for d in descriptions) if descriptions else 0,
            "max": max(d['target_object_count'] for d in descriptions) if descriptions else 0,
            "avg": sum(d['target_object_count'] for d in descriptions) / len(descriptions) if descriptions else 0
        }
    }
    
    # 统计房间类型分布
    for desc in descriptions:
        room_type = desc['room_type']
        stats['room_type_distribution'][room_type] = stats['room_type_distribution'].get(room_type, 0) + 1
        
        level = desc['detail_level']
        stats['detail_level_distribution'][level] = stats['detail_level_distribution'].get(level, 0) + 1
    
    # 保存数据
    output_data = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_descriptions": len(descriptions),
            "room_types": ROOM_TYPES,
            "detail_levels": list(DESCRIPTION_LEVELS.keys())
        },
        "statistics": stats,
        "descriptions": descriptions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ 描述已保存到: {output_file}")
    print(f"总计: {len(descriptions)} 条")
    print(f"\n房间类型分布:")
    for room_type, count in sorted(stats['room_type_distribution'].items()):
        print(f"  {room_type}: {count}")
    print(f"\n详细程度分布:")
    for level, count in sorted(stats['detail_level_distribution'].items()):
        print(f"  {level}: {count}")
    print(f"\n物体数量统计:")
    print(f"  最小: {stats['object_count_stats']['min']}")
    print(f"  最大: {stats['object_count_stats']['max']}")
    print(f"  平均: {stats['object_count_stats']['avg']:.1f}")
    print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='生成房间描述数据集，用于3D场景生成训练'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=350,
        help='生成的描述总数（默认: 1000）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/path/to/datasets/llmscene/room_descriptions_test_350.json',
        help='输出文件路径'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='中间保存的批次大小（默认: 8）'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='API调用之间的延迟秒数（默认: 0.5）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='并行生成的线程数（默认: 1）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 生成描述
    descriptions = generate_room_descriptions(
        total_count=args.count,
        output_file=args.output,
        batch_size=args.batch_size,
        delay=args.delay,
        max_workers=args.workers
    )
    
    print(f"\n✓ 生成完成! 共生成 {len(descriptions)} 条房间描述")


if __name__ == "__main__":
    main()
