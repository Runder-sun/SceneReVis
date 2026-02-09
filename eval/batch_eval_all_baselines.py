#!/usr/bin/env python3
"""
批量评估所有baseline的所有场景类别
运行VLM评估(vlm_scene_eval)

使用示例:
  # 评估所有baseline的所有房间类型
  python batch_eval_all_baselines.py

  # 只评估setting1/2/3
  python batch_eval_all_baselines.py --settings-only

  # 只评估指定的baseline
  python batch_eval_all_baselines.py --baselines A_Ours DiffuScene LayoutVLM

  # 评估指定baseline的setting
  python batch_eval_all_baselines.py --baselines A_Ours DiffuScene LayoutVLM --settings-only

  # 评估指定的房间类型
  python batch_eval_all_baselines.py --room-types bedroom living_room

  # 评估指定的setting
  python batch_eval_all_baselines.py --room-types setting1 setting2 setting3
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pandas as pd

# 基础路径配置
RESULTS_BASE_DIR = "/path/to/SceneReVis/baseline/results"
EVAL_SCRIPT_DIR = "/path/to/SceneReVis/eval"
VLM_EVAL_SCRIPT = os.path.join(EVAL_SCRIPT_DIR, "vlm_scene_eval.py")

# 场景类别名称映射（统一命名）
ROOM_TYPE_MAPPING = {
    "bedroom": "bedroom",
    "living_room": "living_room",
    "livingroom": "living_room",
    "dining_room": "dining_room",
    "diningroom": "dining_room",
    "study_room": "study_room",
    "studyroom": "study_room",
    "entertainment_room": "entertainment_room",
    "gym": "gym",
    "office": "office"
}

# 支持的房间类别（规范化名称）
SUPPORTED_ROOM_TYPES = {"bedroom", "living_room", "dining_room", "study_room", 
                        "entertainment_room", "gym", "office"}

# 支持的setting类别
SUPPORTED_SETTINGS = {"setting1", "setting2", "setting3"}

# Prompts文件路径配置
PROMPTS_FILES = {
    "bedroom": "/path/to/SceneReVis/test/split_prompts/bedroom.txt",
    "living_room": "/path/to/SceneReVis/test/split_prompts/living_room.txt",
    "dining_room": "/path/to/SceneReVis/test/split_prompts/dining_room.txt",
    "study_room": "/path/to/SceneReVis/test/split_prompts/study_room.txt",
    "entertainment_room": "/path/to/SceneReVis/test/split_prompts/entertainment_room.txt",
    "gym": "/path/to/SceneReVis/test/split_prompts/gym.txt",
    "office": "/path/to/SceneReVis/test/split_prompts/office.txt",
    # Setting prompts (统一使用bedroom的prompts，因为setting包含多种场景)
    "setting1": "/path/to/data/eval/benchmark_generated/setting1.txt",
    "setting2": "/path/to/data/eval/benchmark_generated/setting2.txt",
    "setting3": "/path/to/data/eval/benchmark_generated/setting3.txt"
}

# 需要使用merged子目录的baseline列表
BASELINES_WITH_MERGED_DIR = {"Ours", "A_Ours"}


def discover_baselines() -> List[str]:
    """发现所有baseline文件夹"""
    baselines = []
    base_path = Path(RESULTS_BASE_DIR)
    
    if not base_path.exists():
        print(f"❌ 结果目录不存在: {RESULTS_BASE_DIR}")
        return baselines
    
    for item in base_path.iterdir():
        # 只要有 json/ 或 render/ 目录就认为是有效的 baseline
        if item.is_dir() and ((item / "json").exists() or (item / "render").exists()):
            baselines.append(item.name)
    
    return sorted(baselines)


def discover_room_types(baseline: str, settings_only: bool = False, 
                         room_types_filter: Optional[List[str]] = None) -> List[Tuple[str, str]]:
    """
    发现baseline下的所有场景类别（只返回支持的四种房间类型和setting类别）
    
    Args:
        baseline: baseline名称
        settings_only: 是否只返回setting类别
        room_types_filter: 只返回指定的房间类型/setting
    
    Returns:
        List of (original_name, normalized_name) tuples
    """
    room_types = []
    found_names = set()  # 避免重复
    
    # 检查json目录
    json_dir = Path(RESULTS_BASE_DIR) / baseline / "json"
    if json_dir.exists():
        for item in json_dir.iterdir():
            if item.is_dir():
                original_name = item.name
                normalized_name = ROOM_TYPE_MAPPING.get(original_name.lower(), original_name)
                
                # 如果指定了过滤器，检查是否匹配
                if room_types_filter:
                    if original_name not in room_types_filter and normalized_name not in room_types_filter:
                        continue
                
                # 如果只要setting，跳过房间类型
                if settings_only:
                    if original_name in SUPPORTED_SETTINGS and original_name not in found_names:
                        room_types.append((original_name, original_name))
                        found_names.add(original_name)
                else:
                    # 只添加支持的房间类型
                    if normalized_name in SUPPORTED_ROOM_TYPES and original_name not in found_names:
                        room_types.append((original_name, normalized_name))
                        found_names.add(original_name)
    
    # 检查render目录中的房间类型和setting类别
    render_dir = Path(RESULTS_BASE_DIR) / baseline / "render"
    if render_dir.exists():
        for item in render_dir.iterdir():
            if item.is_dir() and item.name not in found_names:
                original_name = item.name
                
                # 如果指定了过滤器，检查是否匹配
                if room_types_filter:
                    normalized_name = ROOM_TYPE_MAPPING.get(original_name.lower(), original_name)
                    if original_name not in room_types_filter and normalized_name not in room_types_filter:
                        continue
                
                # 检查是否是setting类别
                if original_name in SUPPORTED_SETTINGS:
                    if settings_only or not room_types_filter:  # settings_only模式或没有指定过滤器时包含setting
                        room_types.append((original_name, original_name))
                        found_names.add(original_name)
                    elif room_types_filter and original_name in room_types_filter:  # 或者在过滤器中明确指定了
                        room_types.append((original_name, original_name))
                        found_names.add(original_name)
                elif not settings_only:
                    # 检查是否是支持的房间类型
                    normalized_name = ROOM_TYPE_MAPPING.get(original_name.lower(), original_name)
                    if normalized_name in SUPPORTED_ROOM_TYPES:
                        room_types.append((original_name, normalized_name))
                        found_names.add(original_name)
    
    return sorted(room_types)


def run_vlm_eval(baseline: str, room_type_original: str, room_type_normalized: str, 
                  output_dir: str, max_workers: int = 4) -> Tuple[bool, str]:
    """
    运行VLM评估
    
    Returns:
        (success, result_file_path)
    """
    # 根据baseline类型确定渲染目录和JSON目录
    if baseline in BASELINES_WITH_MERGED_DIR:
        # Ours baseline: render/房间类型/merged
        render_dir = Path(RESULTS_BASE_DIR) / baseline / "render" / room_type_original / "merged"
        json_dir = Path(RESULTS_BASE_DIR) / baseline / "json" / room_type_original
    else:
        # 其他baseline: render/房间类型
        render_dir = Path(RESULTS_BASE_DIR) / baseline / "render" / room_type_original
        json_dir = Path(RESULTS_BASE_DIR) / baseline / "json" / room_type_original
    
    prompts_file = PROMPTS_FILES.get(room_type_normalized)
    
    if not render_dir.exists():
        print(f"  ⚠️  跳过VLM评估: 渲染目录不存在 ({render_dir})")
        return False, ""
    
    if not prompts_file or not Path(prompts_file).exists():
        print(f"  ⚠️  跳过VLM评估: prompts文件不存在 ({room_type_normalized})")
        return False, ""
    
    # 输出目录
    eval_output_dir = Path(output_dir) / "vlm_eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    result_file = eval_output_dir / "vlm_evaluation_results.json"
    
    print(f"  🎨 运行VLM评估...")
    print(f"     - 渲染目录: {render_dir}")
    print(f"     - JSON目录: {json_dir}")
    print(f"     - Prompts: {prompts_file}")
    print(f"     - Workers: {max_workers}")
    
    cmd = [
        "python", VLM_EVAL_SCRIPT,
        "--render-dir", str(render_dir),
        "--prompts-file", prompts_file,
        "--output", str(result_file),
        "--max-workers", str(max_workers),
        "--resume"
    ]
    
    # 如果JSON目录存在，添加参数
    if json_dir.exists():
        cmd.extend(["--json-dir", str(json_dir)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.returncode == 0 and result_file.exists():
            print(f"  ✅ VLM评估完成")
            return True, str(result_file)
        else:
            print(f"  ❌ VLM评估失败")
            if result.stderr:
                print(f"     错误: {result.stderr[:500]}")
            return False, ""
            
    except subprocess.TimeoutExpired:
        print(f"  ❌ VLM评估超时")
        return False, ""
    except Exception as e:
        print(f"  ❌ VLM评估异常: {e}")
        return False, ""


def load_evaluation_results(vlm_eval_file: str) -> Dict:
    """加载评估结果"""
    results = {
        "vlm": {},
        "success": False
    }
    
    # 加载VLM评估结果
    if vlm_eval_file and Path(vlm_eval_file).exists():
        try:
            with open(vlm_eval_file, 'r') as f:
                data = json.load(f)
                if "average_scores" in data:
                    results["vlm"] = data["average_scores"]
                results["success"] = True
        except Exception as e:
            print(f"    ⚠️  加载VLM结果失败: {e}")
    
    return results


def evaluate_baseline_room_vlm(baseline: str, room_type_original: str, room_type_normalized: str, 
                                output_base_dir: str, max_workers: int = 4) -> Tuple[str, bool, str]:
    """评估单个baseline的单个场景类别的VLM指标"""
    print(f"\n{'='*80}")
    print(f"🎨 VLM评估: {baseline} - {room_type_original}")
    print(f"{'='*80}")
    
    # 创建输出目录
    output_dir = Path(output_base_dir) / baseline / room_type_original
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行VLM评估
    vlm_success, vlm_file = run_vlm_eval(baseline, room_type_original, room_type_normalized, 
                                          str(output_dir), max_workers=max_workers)
    
    return str(output_dir), vlm_success, vlm_file


def combine_evaluation_results(baseline: str, room_type_original: str, room_type_normalized: str, 
                               vlm_file: str) -> Dict:
    """合并评估结果"""
    # 加载结果
    results = load_evaluation_results(vlm_file)
    
    return {
        "baseline": baseline,
        "room_type": room_type_original,
        "room_type_normalized": room_type_normalized,
        "results": results,
        "files": {
            "vlm_eval": vlm_file
        }
    }


def create_summary_report(all_results: List[Dict], output_file: str):
    """创建汇总报告"""
    print(f"\n{'='*80}")
    print(f"📝 生成汇总报告...")
    print(f"{'='*80}")
    
    # 保存完整JSON结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(all_results),
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 完整结果已保存到: {output_file}")
    
    # 创建Excel汇总表
    excel_file = output_file.replace('.json', '.xlsx')
    create_excel_summary(all_results, excel_file)
    
    # 打印控制台汇总
    print_console_summary(all_results)


def create_excel_summary(all_results: List[Dict], excel_file: str):
    """创建Excel汇总表"""
    try:
        import pandas as pd
        
        # VLM指标表
        vlm_rows = []
        
        for result in all_results:
            baseline = result["baseline"]
            room = result["room_type"]
            
            # VLM指标
            if result["results"]["success"]:
                row = {"Baseline": baseline, "Room Type": room}
                row.update(result["results"]["vlm"])
                vlm_rows.append(row)
        
        # 写入Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            if vlm_rows:
                df_vlm = pd.DataFrame(vlm_rows)
                df_vlm.to_excel(writer, sheet_name='VLM Metrics', index=False)
        
        print(f"✅ Excel汇总表已保存到: {excel_file}")
        
    except ImportError:
        print("⚠️  未安装pandas/openpyxl，跳过Excel生成")
    except Exception as e:
        print(f"⚠️  Excel生成失败: {e}")


def print_console_summary(all_results: List[Dict]):
    """打印控制台汇总"""
    print(f"\n{'='*80}")
    print(f"📊 评估汇总")
    print(f"{'='*80}\n")
    
    # 按baseline分组
    baselines = {}
    for result in all_results:
        baseline = result["baseline"]
        if baseline not in baselines:
            baselines[baseline] = []
        baselines[baseline].append(result)
    
    # 打印每个baseline的汇总
    for baseline, results in sorted(baselines.items()):
        print(f"\n【{baseline}】")
        print("-" * 60)
        
        for result in results:
            room = result["room_type"]
            print(f"\n  {room}:")
            
            # VLM指标
            if result["results"]["success"]:
                vlm = result["results"]["vlm"]
                print(f"    VLM指标:")
                for metric, value in sorted(vlm.items()):
                    print(f"      - {metric}: {value:.2f}")
            else:
                print(f"    VLM指标: ❌ 评估失败")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='批量评估所有Baseline (VLM评估)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 评估所有baseline的所有房间类型
  python batch_eval_all_baselines.py

  # 只评估setting1/2/3
  python batch_eval_all_baselines.py --settings-only

  # 只评估指定的baseline
  python batch_eval_all_baselines.py --baselines A_Ours DiffuScene LayoutVLM

  # 评估指定baseline的setting
  python batch_eval_all_baselines.py --baselines A_Ours DiffuScene LayoutVLM --settings-only

  # 评估指定的房间类型
  python batch_eval_all_baselines.py --room-types bedroom living_room

  # 评估指定的setting
  python batch_eval_all_baselines.py --room-types setting1 setting2 setting3
        """
    )
    parser.add_argument(
        '--baselines', '-b', nargs='+', type=str, default=None,
        help='指定要评估的baseline列表 (默认: 所有baseline)'
    )
    parser.add_argument(
        '--settings-only', '-s', action='store_true',
        help='只评估setting1/2/3，不评估房间类型'
    )
    parser.add_argument(
        '--room-types', '-r', nargs='+', type=str, default=None,
        help='指定要评估的房间类型或setting (例如: bedroom setting1 setting2)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default=None,
        help='指定输出目录 (默认: 自动生成带时间戳的目录)'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=4,
        help='VLM评估的并行worker数 (默认: 4)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 批量评估所有Baseline (VLM评估)")
    print("="*80)
    
    # 输出目录
    if args.output_dir:
        output_base_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = Path(RESULTS_BASE_DIR) / f"batch_evaluation_{timestamp}"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 输出目录: {output_base_dir}")
    
    if args.settings_only:
        print(f"📋 模式: 只评估Setting")
        print(f"📋 支持的Setting: {', '.join(sorted(SUPPORTED_SETTINGS))}")
    elif args.room_types:
        print(f"📋 模式: 指定房间类型/Setting")
        print(f"📋 指定的类型: {', '.join(args.room_types)}")
    else:
        print(f"📋 模式: 评估所有房间类型和Setting")
        print(f"📋 支持的房间类型: {', '.join(sorted(SUPPORTED_ROOM_TYPES))}")
        print(f"📋 支持的Setting: {', '.join(sorted(SUPPORTED_SETTINGS))}")
    
    # 发现所有baseline
    all_baselines = discover_baselines()
    if not all_baselines:
        print("❌ 未发现任何baseline")
        return
    
    # 过滤baseline
    if args.baselines:
        baselines = [b for b in args.baselines if b in all_baselines]
        missing = set(args.baselines) - set(baselines)
        if missing:
            print(f"⚠️  以下baseline不存在: {', '.join(missing)}")
        if not baselines:
            print("❌ 没有有效的baseline")
            return
    else:
        baselines = all_baselines
    
    print(f"\n📦 评估 {len(baselines)} 个baseline: {', '.join(baselines)}")
    
    # 收集所有评估任务
    evaluation_tasks = []
    for baseline in baselines:
        room_types = discover_room_types(
            baseline, 
            settings_only=args.settings_only,
            room_types_filter=args.room_types
        )
        print(f"🏠 {baseline}: {len(room_types)} 个场景类别 - {[r[0] for r in room_types]}")
        for room_original, room_normalized in room_types:
            evaluation_tasks.append((baseline, room_original, room_normalized))
    
    print(f"\n📋 总共需要评估: {len(evaluation_tasks)} 个场景")
    
    if len(evaluation_tasks) == 0:
        print("❌ 没有需要评估的场景")
        return
    
    # === 运行所有VLM评估 ===
    print(f"\n{'='*80}")
    print(f"🎨 VLM评估 (共 {len(evaluation_tasks)} 个场景)")
    print(f"{'='*80}")
    
    all_results = []
    for baseline, room_original, room_normalized in evaluation_tasks:
        # 运行VLM评估
        output_dir, vlm_success, vlm_file = evaluate_baseline_room_vlm(
            baseline, room_original, room_normalized, str(output_base_dir),
            max_workers=args.max_workers
        )
        
        # 合并结果
        result = combine_evaluation_results(
            baseline, room_original, room_normalized, vlm_file
        )
        all_results.append(result)
    
    # 生成汇总报告
    summary_file = output_base_dir / "summary_report.json"
    create_summary_report(all_results, str(summary_file))
    
    print(f"\n{'='*80}")
    print(f"✨ 所有评估完成!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
