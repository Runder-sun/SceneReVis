#!/usr/bin/env python3
"""
评估和筛选编辑链脚本
使用GPT-4根据场景渲染图和场景JSON评估编辑链质量，保留最好的3条，删除其他7条
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import AzureOpenAI
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from tqdm import tqdm
import base64
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 导入配置
sys.path.append(str(Path(__file__).parent))
from config import *


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


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为base64字符串"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_chain_data(chain_dir: Path) -> Dict[str, Any]:
    """加载编辑链的JSON数据"""
    chain_json = chain_dir / f"{chain_dir.name}.json"
    if not chain_json.exists():
        return None
    
    with open(chain_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_chain_images(chain_dir: Path) -> List[str]:
    """收集编辑链中的所有渲染图片"""
    images = []
    # 查找所有merged图片，按顺序排序
    merged_images = sorted(chain_dir.glob("*_merged.png"))
    for img in merged_images:
        images.append(str(img))
    return images


def evaluate_single_chain(chain_dir: Path, scene_folder_name: str) -> Dict[str, Any]:
    """
    评估单条编辑链的质量
    
    返回评估结果字典，包含：
    - chain_name: 编辑链名称
    - overall_score: 总分 (0-100)
    - coherence_score: 连贯性分数
    - quality_score: 编辑质量分数
    - reasoning: 评分理由
    """
    print(f"  评估编辑链: {chain_dir.name}")
    
    # 在每个进程中创建自己的客户端
    client = setup_azure_client()
    
    # 加载编辑链数据
    chain_data = load_chain_data(chain_dir)
    if not chain_data:
        print(f"    ✗ 无法加载编辑链数据")
        return None
    
    # 收集图片
    images = collect_chain_images(chain_dir)
    if not images:
        print(f"    ✗ 未找到渲染图片")
        return None
    
    print(f"    找到 {len(images)} 张渲染图片")
    
    # 准备提示词
    # 提取对话内容
    messages = chain_data.get('messages', [])
    conversation_summary = []
    for i, msg in enumerate(messages):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        # 截取前200字符避免太长
        content_preview = content[:200] + "..." if len(content) > 200 else content
        conversation_summary.append(f"Turn {i+1} [{role}]: {content_preview}")
    
    conversation_text = "\n".join(conversation_summary)
    
    # 构建评估提示词
    evaluation_prompt = f"""You are an expert in interior design and 3D scene editing. Your task is to evaluate the quality of a multi-turn scene editing conversation chain.

## Scene Information
- Scene Folder: {scene_folder_name}
- Chain: {chain_dir.name}
- Number of editing turns: {len(images)}

## Conversation Summary
{conversation_text}

## Evaluation Task

You will be shown {len(images)} rendered images representing the progressive scene editing steps. 

**Image Format Explanation**: Each rendered image is a merged view with:
- **Left side**: Top-down view (bird's eye view) of the scene
- **Right side**: Diagonal/perspective view of the scene

This dual-view format allows you to see both the spatial layout (top view) and the 3D appearance (diagonal view) at each editing step.

Evaluate this editing chain based on the following criteria:

**IMPORTANT**: Since all 10 chains lead to the SAME final scene, focus primarily on evaluating the EDITING PROCESS quality, not the final result.

### 1. Editing Coherence (40 points) - HIGHEST PRIORITY
- Do the editing operations logically build upon each other?
- Is there a clear, intuitive progression through the editing steps?
- Are the edits consistent with the conversation flow?
- Does each step make sense given the previous state?

### 2. Editing Naturalness (35 points) - VERY IMPORTANT
- Do the editing steps feel natural and intuitive?
- Are the intermediate states meaningful and useful?
- Does the editing flow resemble a real design process?
- Are the edit types (add/remove/move/rotate/scale) appropriately chosen?
- Is the pacing of edits reasonable (not too rushed or too slow)?

### 3. Instruction Following (15 points) - Process Alignment
- Do the edits accurately reflect the user's step-by-step requests?
- Are the operations correctly executed at each turn?
- Is the assistant's understanding of instructions clear?

### 4. Visual Transition Quality (10 points) - Secondary
- Are the intermediate visual states reasonable?
- Do transitions between steps look smooth and logical?
- Note: Final scene quality is NOT evaluated here (all chains end at same scene)

## Output Format

Provide your evaluation in the following JSON format:

```json
{{
    "coherence_score": <0-40>,
    "naturalness_score": <0-35>,
    "instruction_following_score": <0-15>,
    "visual_transition_score": <0-10>,
    "overall_score": <sum of above, 0-100>,
    "reasoning": "<2-3 sentences explaining your evaluation, focusing on the PROCESS quality>",
    "strengths": "<1-2 key strengths of the editing process>",
    "weaknesses": "<1-2 key weaknesses of the editing process if any>"
}}
```

Provide only the JSON output, no additional text."""

    try:
        # 准备消息内容（包含图片）
        message_content = [
            {"type": "input_text", "text": evaluation_prompt}
        ]
        
        # 添加所有图片（不限制数量，发送完整的编辑序列）
        print(f"    准备发送 {len(images)} 张完整的渲染图片序列...")
        for i, img_path in enumerate(images):
            try:
                image_base64 = encode_image_to_base64(img_path)
                message_content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}"
                })
            except Exception as e:
                print(f"    ⚠ 警告: 无法编码图片 {img_path}: {e}")
        
        # 调用 GPT-5 Responses API 进行评估
        response = client.responses.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            input=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            max_output_tokens=min(MAX_TOKENS, 4000),
        )

        # 解析响应文本（Responses API 的返回结构不同于 Chat Completions）
        response_text = (getattr(response, "output_text", None) or "").strip()
        if not response_text:
            text_chunks: List[str] = []
            for output in getattr(response, "output", []) or []:
                for item in getattr(output, "content", []) or []:
                    item_type = getattr(item, "type", None)
                    if item_type is None and isinstance(item, dict):
                        item_type = item.get("type")
                    if item_type not in ("text", "output_text"):
                        continue
                    item_text = getattr(item, "text", None)
                    if item_text is None and isinstance(item, dict):
                        item_text = item.get("text")
                    if item_text:
                        text_chunks.append(item_text)
            response_text = "\n".join(text_chunks).strip()
        if not response_text:
            raise ValueError("模型未返回文本内容，请检查输入或API响应格式")
        
        # 提取JSON部分
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        
        evaluation = json.loads(response_text)
        evaluation['chain_name'] = chain_dir.name
        
        print(f"    ✓ 评估完成 - 总分: {evaluation.get('overall_score', 0)}/100")
        print(f"      理由: {evaluation.get('reasoning', 'N/A')[:100]}...")
        
        return evaluation
        
    except Exception as e:
        print(f"    ✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_all_chains(scene_dir: Path) -> List[Dict[str, Any]]:
    """评估场景目录下的所有编辑链"""
    print(f"\n评估场景: {scene_dir.name}")
    
    # 查找所有chain目录
    chain_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('chain_')])
    
    if len(chain_dirs) != 10:
        print(f"  ⚠ 警告: 预期10条编辑链，实际找到 {len(chain_dirs)} 条")
    
    evaluations = []
    for chain_dir in chain_dirs:
        evaluation = evaluate_single_chain(chain_dir, scene_dir.name)
        if evaluation:
            evaluations.append(evaluation)
    
    return evaluations


def evaluate_chain_wrapper(args: Tuple[Path, str]) -> Dict[str, Any]:
    """
    包装函数用于并行处理单条编辑链评估
    
    Args:
        args: (chain_dir, scene_folder_name) 的元组
    
    Returns:
        评估结果字典或None
    """
    chain_dir, scene_folder_name = args
    try:
        # 添加随机延迟避免API速率限制
        time.sleep(0.2)
        return evaluate_single_chain(chain_dir, scene_folder_name)
    except Exception as e:
        print(f"  ✗ 评估链 {chain_dir.name} 时出错: {e}")
        return None


def evaluate_all_chains_parallel(scene_dir: Path, max_workers: int = 4) -> List[Dict[str, Any]]:
    """使用并行处理评估场景目录下的所有编辑链"""
    print(f"\n评估场景: {scene_dir.name}")
    
    # 查找所有chain目录
    chain_dirs = sorted([d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('chain_')])
    
    if len(chain_dirs) != 10:
        print(f"  ⚠ 警告: 预期10条编辑链，实际找到 {len(chain_dirs)} 条")
    
    # 准备任务参数
    tasks = [(chain_dir, scene_dir.name) for chain_dir in chain_dirs]
    
    evaluations = []
    
    # 使用进程池并行评估
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_chain = {executor.submit(evaluate_chain_wrapper, task): task for task in tasks}
        
        # 收集结果
        for future in as_completed(future_to_chain):
            task = future_to_chain[future]
            chain_dir, _ = task
            try:
                evaluation = future.result()
                if evaluation:
                    evaluations.append(evaluation)
            except Exception as e:
                print(f"  ✗ 处理链 {chain_dir.name} 时出现异常: {e}")
    
    return evaluations


def select_top_chains(evaluations: List[Dict[str, Any]], top_k: int = 3) -> List[str]:
    """根据评分选择最好的K条编辑链"""
    # 按overall_score降序排序
    sorted_evaluations = sorted(evaluations, key=lambda x: x.get('overall_score', 0), reverse=True)
    
    # 选择前K个
    top_chains = [eval_data['chain_name'] for eval_data in sorted_evaluations[:top_k]]
    
    print(f"\n最佳编辑链排名:")
    for i, eval_data in enumerate(sorted_evaluations[:top_k], 1):
        print(f"  {i}. {eval_data['chain_name']}: {eval_data.get('overall_score', 0)}/100")
        print(f"     {eval_data.get('reasoning', 'N/A')[:100]}...")
    
    return top_chains


def filter_and_cleanup_chains(scene_dir: Path, top_chains: List[str], dry_run: bool = False):
    """保留最佳编辑链，删除其他编辑链"""
    print(f"\n清理场景目录: {scene_dir.name}")
    
    # 查找所有chain目录
    chain_dirs = [d for d in scene_dir.iterdir() if d.is_dir() and d.name.startswith('chain_')]
    
    deleted_count = 0
    kept_count = 0
    
    for chain_dir in chain_dirs:
        if chain_dir.name in top_chains:
            print(f"  ✓ 保留: {chain_dir.name}")
            kept_count += 1
        else:
            if dry_run:
                print(f"  [模拟] 删除: {chain_dir.name}")
            else:
                try:
                    shutil.rmtree(chain_dir)
                    print(f"  ✗ 删除: {chain_dir.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ⚠ 无法删除 {chain_dir.name}: {e}")
    
    print(f"\n清理完成: 保留 {kept_count} 条, 删除 {deleted_count} 条")


def save_evaluation_results(scene_dir: Path, evaluations: List[Dict[str, Any]], top_chains: List[str]):
    """保存评估结果到JSON文件"""
    result_file = scene_dir / "chain_evaluation_results.json"
    
    results = {
        "scene_folder": scene_dir.name,
        "evaluation_timestamp": str(Path(__file__).stat().st_mtime),
        "total_chains_evaluated": len(evaluations),
        "top_chains_selected": top_chains,
        "evaluations": evaluations
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估结果已保存到: {result_file}")


def process_scene_folder(scene_dir: Path, top_k: int = 3, dry_run: bool = False, 
                        save_results: bool = True, max_workers: int = 4, 
                        use_parallel: bool = True):
    """处理单个场景目录"""
    try:
        # 评估所有编辑链（使用并行或串行处理）
        if use_parallel:
            evaluations = evaluate_all_chains_parallel(scene_dir, max_workers)
        else:
            evaluations = evaluate_all_chains(scene_dir)
        
        if not evaluations:
            print(f"  ✗ 没有成功评估的编辑链")
            return False
        
        if len(evaluations) < top_k:
            print(f"  ⚠ 警告: 只评估成功 {len(evaluations)} 条链，少于预期的 {top_k} 条")
            actual_top_k = len(evaluations)
        else:
            actual_top_k = top_k
        
        # 选择最佳编辑链
        top_chains = select_top_chains(evaluations, actual_top_k)
        
        # 保存评估结果
        if save_results:
            save_evaluation_results(scene_dir, evaluations, top_chains)
        
        # 清理其他编辑链
        filter_and_cleanup_chains(scene_dir, top_chains, dry_run)
        
        return True
        
    except Exception as e:
        print(f"  ✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='评估和筛选场景编辑链，保留质量最好的3条'
    )
    parser.add_argument(
        '--input-dir',
        default='/path/to/datasets/llmscene/rendered_outputs/multi_turn_v2',
        help='包含场景文件夹的输入目录'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='保留最好的K条编辑链（默认: 3）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='模拟运行，不实际删除文件'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='限制处理的场景数量（用于测试）'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存评估结果JSON文件'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='并行处理的最大工作进程数（默认: 10）'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='禁用并行处理，使用串行模式'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_path}")
        return
    
    # 查找所有场景目录（包含chain_开头的子目录）
    scene_dirs = []
    for item in input_path.iterdir():
        if item.is_dir():
            # 检查是否包含chain_目录
            chain_dirs = [d for d in item.iterdir() if d.is_dir() and d.name.startswith('chain_')]
            # 跳过已经只有3个chain的场景（已处理成功）
            if chain_dirs and len(chain_dirs) != 3:
                scene_dirs.append(item)
    
    if not scene_dirs:
        print(f"错误: 在 {input_path} 中未找到需要处理的场景目录（已跳过只有3个chain的场景）")
        return
    
    print(f"找到 {len(scene_dirs)} 个需要处理的场景目录（已跳过已处理成功的场景）")
    
    if args.limit:
        scene_dirs = scene_dirs[:args.limit]
        print(f"限制处理前 {args.limit} 个场景")
    
    if args.dry_run:
        print("\n⚠ 模拟运行模式 - 不会实际删除文件\n")
    
    if not args.no_parallel:
        print(f"使用并行处理模式，工作进程数: {args.max_workers}\n")
    else:
        print("使用串行处理模式\n")
    
    # 处理所有场景
    successful = 0
    failed = 0
    
    for scene_dir in tqdm(scene_dirs, desc="处理场景"):
        if process_scene_folder(
            scene_dir, 
            args.top_k, 
            args.dry_run, 
            args.no_save,
            args.max_workers,
            not args.no_parallel
        ):
            successful += 1
        else:
            failed += 1
    
    # 最终统计
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"总场景数: {len(scene_dirs)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"保留最佳编辑链数: {args.top_k}")
    print(f"并行模式: {'是 (' + str(args.max_workers) + ' 进程)' if not args.no_parallel else '否（串行）'}")
    if args.dry_run:
        print(f"模式: 模拟运行（未实际删除文件）")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
