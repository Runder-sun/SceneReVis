#!/usr/bin/env python3
"""
批量处理场景脚本 - 处理llmscene数据集中的对话文件
支持新的multi_turn格式和旧的intermediate格式
为每个对话创建单独的文件夹，包含修改后的JSON和渲染图片

新格式特点：
- multi_turn_*.json文件
- 包含多个步骤的images数组
- messages中可能有多个<initial_scene>标签
- 为每个初始场景渲染图片，按顺序更新images数组

旧格式特点：
- intermediate_*.json文件  
- 单个image字段
- conversation中只有一个<initial_scene>
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import pickle

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, skip
    pass

# 设置3D模型路径环境变量
if not os.environ.get('PTH_3DFUTURE_ASSETS'):
    # 默认3D模型路径
    default_3d_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
    if Path(default_3d_path).exists():
        os.environ['PTH_3DFUTURE_ASSETS'] = default_3d_path
        print(f"Set PTH_3DFUTURE_ASSETS to: {default_3d_path}")
    else:
        print(f"Warning: 3D model path not found: {default_3d_path}")

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from image_merger import ImageMerger, merge_rendered_views

def load_progress_file(progress_file_path):
    """加载进度文件，返回已处理的文件集合"""
    if Path(progress_file_path).exists():
        try:
            with open(progress_file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load progress file {progress_file_path}: {e}")
            return set()
    return set()

def save_progress_file(progress_file_path, processed_files):
    """保存进度文件"""
    try:
        with open(progress_file_path, 'wb') as f:
            pickle.dump(processed_files, f)
    except Exception as e:
        print(f"Warning: Could not save progress file {progress_file_path}: {e}")

def is_already_processed(json_file_path, output_base_dir, input_base_dir=None):
    """检查文件是否已经被处理过"""
    json_file = Path(json_file_path)
    
    # 计算输出目录路径（与process_single_conversation中的逻辑一致）
    if input_base_dir:
        input_base = Path(input_base_dir)
        try:
            relative_dir = json_file.parent.relative_to(input_base)
            output_dir = Path(output_base_dir) / relative_dir / json_file.stem
        except ValueError:
            output_dir = Path(output_base_dir) / json_file.parent.name / json_file.stem
    else:
        output_dir = Path(output_base_dir) / json_file.parent.name / json_file.stem
    
    # 检查是否存在更新后的JSON文件和至少一个合并的图片
    updated_json_path = output_dir / f"{json_file.stem}.json"
    if not updated_json_path.exists():
        return False
    
    # 检查是否存在合并的图片文件
    merged_images = list(output_dir.glob("*_merged.png"))
    return len(merged_images) > 0

def process_file_wrapper(args):
    """多进程处理的包装函数"""
    json_file_path, output_base_dir, input_base_dir, verbose = args
    
    try:
        success = process_single_conversation(
            json_file_path=json_file_path,
            output_base_dir=output_base_dir,
            input_base_dir=input_base_dir,
            verbose=verbose
        )
        return json_file_path, success, None
    except Exception as e:
        return json_file_path, False, str(e)

def extract_scenes_from_conversation(conversation_data):
    """从对话数据中提取所有场景（支持initial_scene和current_scene）"""
    scenes = []
    
    # 检查是否为新格式（multi_turn）
    if "messages" in conversation_data:
        # 新格式：从messages中提取所有scene标签
        for message in conversation_data.get("messages", []):
            content = message.get("content", "")
            # 查找所有可能的场景标签
            scene_tags = ["<initial_scene>", "<current_scene>"]
            
            for tag in scene_tags:
                if tag in content:
                    # 查找所有的场景标签
                    start_pos = 0
                    while True:
                        start_tag = content.find(tag, start_pos)
                        if start_tag == -1:
                            break
                        
                        # 找到JSON开始位置
                        json_start = content.find("```json\n", start_tag) + 8
                        if json_start == 7:  # 没找到```json\n
                            start_pos = start_tag + len(tag)
                            continue
                        
                        # 找到JSON结束位置
                        json_end = content.find("\n```", json_start)
                        if json_end == -1:
                            start_pos = start_tag + len(tag)
                            continue
                        
                        try:
                            json_str = content[json_start:json_end]
                            scene_data = json.loads(json_str)
                            scenes.append(scene_data)
                        except json.JSONDecodeError:
                            pass
                        
                        start_pos = json_end + 4
    else:
        # 旧格式：从conversation中提取场景（保持向后兼容）
        for item in conversation_data.get("conversation", []):
            if item.get("role") == "user":
                content = item.get("content", "")
                # 查找旧格式的场景标签
                scene_tags = ["<initial_scene>", "<current_scene>"]
                
                for tag in scene_tags:
                    if tag in content:
                        start = content.find("```json\n") + 8
                        end = content.find("\n```", start)
                        if start > 7 and end > start:
                            try:
                                json_str = content[start:end]
                                scene_data = json.loads(json_str)
                                scenes.append(scene_data)
                                break  # 旧格式只有一个场景
                            except json.JSONDecodeError:
                                continue
    
    return scenes

def normalize_scene_format(scene: Dict[str, Any]) -> Dict[str, Any]:
    """
    将场景标准化为带groups的格式
    支持两种输入格式:
    1. 原格式: {"groups": [...], "room_envelope": {...}}
    2. SSR格式: {"objects": [...], "bounds_top": [...], "bounds_bottom": [...], "room_type": "...", "room_id": "..."}
    """
    if "groups" in scene:
        # 已经是groups格式，直接返回
        return scene
    elif "objects" in scene:
        # SSR格式，需要转换为groups格式
        normalized_scene = {
            "groups": [{
                "group_name": "scene_objects",
                "group_type": "objects", 
                "objects": scene["objects"]
            }],
            "room_envelope": {
                "bounds_top": scene.get("bounds_top", []),
                "bounds_bottom": scene.get("bounds_bottom", [])
            }
        }
        
        # 保留其他字段
        for key in ["room_type", "room_id"]:
            if key in scene:
                normalized_scene[key] = scene[key]
        
        return normalized_scene
    else:
        # 未知格式，直接返回
        return scene

def render_with_blender(scene_data, output_dir, scene_id, verbose=False):
    """使用Blender渲染场景"""
    try:
        # 标准化场景格式为groups格式（Blender渲染器需要这种格式）
        normalized_scene = normalize_scene_format(scene_data)
        
        import tempfile
        import json
        import os
        import sys
        
        # 导入Blender渲染函数 - 尝试不同的导入路径
        render_scene_with_bpy = None
        try:
            from utils.main_bpy import render_scene_with_bpy
        except ImportError:
            try:
                # 尝试相对导入
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
                from main_bpy import render_scene_with_bpy
            except ImportError:
                try:
                    # 尝试绝对路径导入
                    sys.path.append('/path/to/SceneReVis/utils')
                    from main_bpy import render_scene_with_bpy
                except ImportError:
                    if verbose:
                        print("  Error: Cannot import render_scene_with_bpy")
                    return False
        
        # 设置环境变量控制Blender的verbose输出
        old_verbose = os.environ.get('BPY_VERBOSE', '0')
        old_3d_assets = os.environ.get('PTH_3DFUTURE_ASSETS', '')
        
        os.environ['BPY_VERBOSE'] = '1' if verbose else '0'
        
        # 确保3D模型路径环境变量被传递
        if not os.environ.get('PTH_3DFUTURE_ASSETS'):
            default_3d_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
            if Path(default_3d_path).exists():
                os.environ['PTH_3DFUTURE_ASSETS'] = default_3d_path
                if verbose:
                    print(f"  Set PTH_3DFUTURE_ASSETS to: {default_3d_path}")
        
        if verbose:
            print(f"  PTH_3DFUTURE_ASSETS: {os.environ.get('PTH_3DFUTURE_ASSETS')}")
        
        try:
            if verbose:
                print(f"  Rendering {scene_id} with Blender...")
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # 创建临时场景文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(scene_data, temp_file, ensure_ascii=False, indent=2)
                temp_scene_path = temp_file.name
            
            try:
                # 如果不是verbose模式，重定向stdout和stderr来抑制Blender输出
                if not verbose:
                    # 保存原始stdout和stderr
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    # 创建null设备来丢弃输出
                    with open(os.devnull, 'w') as devnull:
                        sys.stdout = devnull
                        sys.stderr = devnull
                        
                        try:
                            # 渲染场景
                            render_scene_with_bpy(temp_scene_path, output_path)
                        finally:
                            # 恢复stdout和stderr
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
                else:
                    # verbose模式下正常输出
                    render_scene_with_bpy(temp_scene_path, output_path)
                
                # Blender输出结构是 output_path/top/frame.png 和 output_path/diag/frame.png
                blender_top_file = output_path / "top" / "frame.png"
                blender_diag_file = output_path / "diag" / "frame.png"
                
                # 重命名为所需格式
                target_top_file = output_path / f"{scene_id}_top.png"
                target_diag_file = output_path / f"{scene_id}_diagonal.png"
                
                if blender_top_file.exists() and blender_diag_file.exists():
                    # 移动和重命名文件
                    blender_top_file.rename(target_top_file)
                    blender_diag_file.rename(target_diag_file)
                    
                    # 删除空的子目录
                    (output_path / "top").rmdir()
                    (output_path / "diag").rmdir()
                    
                    if verbose:
                        print(f"  Blender rendering completed")
                    return True
                else:
                    if verbose:
                        print("  Blender rendering failed: output files not found")
                    return False
                    
            finally:
                # 清理临时文件
                import os
                os.unlink(temp_scene_path)
        finally:
            # 恢复原来的环境变量
            os.environ['BPY_VERBOSE'] = old_verbose
            if old_3d_assets:
                os.environ['PTH_3DFUTURE_ASSETS'] = old_3d_assets
            elif 'PTH_3DFUTURE_ASSETS' in os.environ and not old_3d_assets:
                # 如果原来没有设置，现在有了，保持不变（不删除）
                pass
            
    except Exception as e:
        if verbose:
            print(f"  Blender rendering failed: {e}")
            import traceback
            traceback.print_exc()
        return False

def process_single_conversation(json_file_path, output_base_dir, input_base_dir=None, verbose=False):
    """处理单个对话文件"""
    json_file = Path(json_file_path)
    
    # 计算相对于输入基目录的相对路径，用于创建输出目录结构
    if input_base_dir:
        input_base = Path(input_base_dir)
        try:
            # 获取相对路径（不包含文件名）
            relative_dir = json_file.parent.relative_to(input_base)
            # 使用相对路径创建输出目录，加上文件名（不含扩展名）作为最终目录
            output_dir = Path(output_base_dir) / relative_dir / json_file.stem
        except ValueError:
            # 如果无法计算相对路径，则使用原有逻辑
            output_dir = Path(output_base_dir) / json_file.parent.name / json_file.stem
    else:
        # 保持原有逻辑作为后备方案
        output_dir = Path(output_base_dir) / json_file.parent.name / json_file.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    conversation_id = json_file.stem
    
    if verbose:
        print(f"Processing: {json_file}")
        print(f"Output dir: {output_dir}")
    
    try:
        # 加载对话JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)
        
        # 提取所有场景数据（支持initial_scene和current_scene）
        scenes = extract_scenes_from_conversation(conversation_data)
        
        if not scenes:
            if verbose:
                print(f"  Error: Could not find any scene in {json_file}")
            return False
        
        # 显示场景信息
        if verbose:
            print(f"  Found {len(scenes)} scenes")
            for i, scene in enumerate(scenes):
                if 'groups' in scene:
                    objects_count = sum(len(group.get('objects', [])) for group in scene.get('groups', []))
                    room_type = scene.get('room_type', 'unknown')
                    groups_count = len(scene.get('groups', []))
                    print(f"    Scene {i}: {room_type} with {groups_count} groups, {objects_count} total objects")
                else:
                    objects_count = len(scene.get('objects', []))
                    room_type = scene.get('room_type', 'unknown')
                    print(f"    Scene {i}: {room_type} with {objects_count} objects (old format)")
        
        # 渲染每个场景并收集图片路径
        rendered_images = []
        
        for i, scene in enumerate(scenes):
            scene_name = f"{conversation_id}_step_{i}"
            
            # 渲染场景 - 只使用Blender渲染器
            render_success = render_with_blender(scene, str(output_dir), scene_name, verbose)
            
            if not render_success:
                if verbose:
                    print(f"  Blender rendering failed for scene {i}")
                return False
            
            # 查找渲染输出文件 - Blender输出直接在主目录中
            top_source = output_dir / f"{scene_name}_top.png"
            diag_source = output_dir / f"{scene_name}_diagonal.png"
            
            # 合并图像
            if top_source.exists() and diag_source.exists():
                merged_output = output_dir / f"{scene_name}_merged.png"
                if verbose:
                    print(f"  Merging rendered views for scene {i}...")
                
                # 使用包含边界框的合并函数
                merge_rendered_views(
                    str(top_source),
                    str(diag_source),
                    str(merged_output)
                )
                
                # 添加到图片列表（使用绝对路径）
                rendered_images.append(str(merged_output.resolve()))
                
                if verbose:
                    print(f"  Merged image saved to: {merged_output}")
                
                # 删除原始渲染图片（不再需要）
                if top_source.exists():
                    top_source.unlink()
                if diag_source.exists():
                    diag_source.unlink()
                    
            else:
                if verbose:
                    print(f"  Error: Missing rendered views for scene {i}")
                return False
        
        # 清理子目录（如果存在）
        top_dir = output_dir / "top"
        diag_dir = output_dir / "diag"
        if top_dir.exists():
            shutil.rmtree(top_dir)
        if diag_dir.exists():
            shutil.rmtree(diag_dir)
        
        # 更新对话JSON中的图像路径
        if "images" in conversation_data:
            # 新格式：更新images数组
            old_images = conversation_data.get("images", [])
            conversation_data["images"] = rendered_images
            if verbose:
                print(f"  Updated images array: {len(old_images)} -> {len(rendered_images)} images")
        else:
            # 旧格式：更新单个image字段（向后兼容）
            old_image_path = conversation_data.get("image", "")
            if rendered_images:
                conversation_data["image"] = rendered_images[0]
                if verbose:
                    print(f"  Updated image path from {old_image_path} to {rendered_images[0]}")
        
        # 保存更新后的对话JSON文件
        updated_json_path = output_dir / f"{conversation_id}.json"
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, ensure_ascii=False, indent=2)
        
        if verbose:
            print(f"  Updated conversation file: {updated_json_path}")
            print(f"✓ Successfully processed {conversation_id}")
            for i, img_path in enumerate(rendered_images):
                print(f"  ✓ Scene {i} image: {img_path}")
            print(f"  ✓ Updated JSON: {updated_json_path}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"  Error processing {conversation_id}: {e}")
            import traceback
            traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch process scenes with resume and multiprocessing support')
    parser.add_argument('--input-dir', default='/path/to/datasets/llmscene/intermediate_data_v2', help='Input directory containing scene folders')
    parser.add_argument('--output-dir', default='/path/to/datasets/llmscene/rendered_outputs/multi_turn_v2', help='Output directory for results')
    parser.add_argument('--max-workers', type=int, default=32, help='Maximum number of worker processes')
    parser.add_argument('--limit', type=int, help='Limit number of scenes to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--progress-file', default=None, help='Custom progress file path')
    parser.add_argument('--skip-processed', action='store_true', help='Skip files that are already processed (based on output files)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置进度文件路径
    if args.progress_file:
        progress_file_path = Path(args.progress_file)
    else:
        progress_file_path = output_dir / "processing_progress.pkl"
    
    # 递归查找所有JSON文件（包括嵌套子目录）
    input_dir = Path(args.input_dir)
    
    if args.verbose:
        print("Scanning for JSON files...")
    
    # 使用递归glob查找所有JSON文件
    all_json_files = list(input_dir.rglob("*.json"))
    
    if args.verbose:
        print(f"Found {len(all_json_files)} total JSON files")
    
    # 加载进度（如果启用断点续传）
    processed_files = set()
    if args.resume:
        processed_files = load_progress_file(progress_file_path)
        if processed_files:
            print(f"Loaded progress: {len(processed_files)} files already processed")
    
    # 过滤已处理的文件
    remaining_files = []
    for json_file in all_json_files:
        json_file_str = str(json_file)
        
        # 检查进度文件中的记录
        if args.resume and json_file_str in processed_files:
            if args.verbose:
                print(f"Skipping (in progress file): {json_file}")
            continue
        
        # 检查输出文件是否已存在
        if args.skip_processed and is_already_processed(json_file_str, str(output_dir), str(input_dir)):
            if args.verbose:
                print(f"Skipping (already processed): {json_file}")
            processed_files.add(json_file_str)  # 添加到进度中
            continue
        
        remaining_files.append(json_file_str)
    
    # 应用限制
    if args.limit:
        remaining_files = remaining_files[:args.limit]
    
    total_files = len(remaining_files)
    already_processed = len(all_json_files) - total_files
    
    print(f"Files to process: {total_files}")
    if already_processed > 0:
        print(f"Already processed: {already_processed}")
    print(f"Using {args.max_workers} worker processes")
    print("Renderer: Blender (only)")
    
    if total_files == 0:
        print("No files to process!")
        return 0, 0, 0
    
    # 准备任务参数
    task_args = [
        (json_file, str(output_dir), str(input_dir), args.verbose)
        for json_file in remaining_files
    ]
    
    # 统计变量
    processed_count = 0
    success_count = 0
    error_count = 0
    
    # 多进程处理
    start_time = time.time()
    
    if args.max_workers == 1:
        # 单进程模式（用于调试）
        print("Running in single-process mode...")
        with tqdm(total=total_files, desc="Processing JSON files", unit="file") as pbar:
            for task_arg in task_args:
                json_file_path, success, error = process_file_wrapper(task_arg)
                processed_count += 1
                
                if success:
                    success_count += 1
                    processed_files.add(json_file_path)
                else:
                    error_count += 1
                    if error and args.verbose:
                        tqdm.write(f"Error processing {json_file_path}: {error}")
                
                # 定期保存进度
                if processed_count % 10 == 0:
                    save_progress_file(progress_file_path, processed_files)
                
                pbar.set_postfix({
                    'Success': success_count,
                    'Errors': error_count,
                    'Rate': f"{processed_count/(time.time()-start_time):.2f}/s"
                })
                pbar.update(1)
    else:
        # 多进程模式
        print("Running in multi-process mode...")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(process_file_wrapper, task_arg): task_arg[0]
                for task_arg in task_args
            }
            
            # 使用tqdm显示进度
            with tqdm(total=total_files, desc="Processing JSON files", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    json_file_path = future_to_file[future]
                    
                    try:
                        json_file_path, success, error = future.result()
                        processed_count += 1
                        
                        if success:
                            success_count += 1
                            processed_files.add(json_file_path)
                        else:
                            error_count += 1
                            if error and args.verbose:
                                tqdm.write(f"Error processing {json_file_path}: {error}")
                        
                        # 定期保存进度
                        if processed_count % 10 == 0:
                            save_progress_file(progress_file_path, processed_files)
                        
                        # 更新进度条显示
                        elapsed_time = time.time() - start_time
                        rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                        pbar.set_postfix({
                            'Success': success_count,
                            'Errors': error_count,
                            'Rate': f"{rate:.2f}/s"
                        })
                        
                    except Exception as e:
                        error_count += 1
                        if args.verbose:
                            tqdm.write(f"Exception processing {json_file_path}: {e}")
                    
                    pbar.update(1)
    
    # 最终保存进度
    save_progress_file(progress_file_path, processed_files)
    
    # 最终统计
    elapsed_time = time.time() - start_time
    print(f"\nBatch processing completed!")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Average rate: {processed_count/elapsed_time:.2f} files/second" if elapsed_time > 0 else "N/A")
    print(f"Total scenes processed: {processed_count}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {success_count/processed_count*100:.1f}%" if processed_count > 0 else "No scenes processed")
    print(f"Progress saved to: {progress_file_path}")
    
    if not args.verbose and error_count > 0:
        print(f"\nRun with --verbose to see error details")
    
    return processed_count, success_count, error_count

if __name__ == "__main__":
    exit(main())
