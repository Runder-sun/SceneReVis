#!/usr/bin/env python3
"""Blender 版本的示例脚本：参考 main.py 的流程，用 bpy 渲染 top 与 diag 两张图。

运行方式（推荐在 Blender 内）：
  blender --background --python main_bpy.py -- \
      --scene /abs/path/to/scene.json \
      --out ./eval/viz/misc/test-bpy

若你已在 Blender 内部交互 Python Console:
  import json
  import pathlib
  from main_bpy import render_scene_with_bpy
  render_scene_with_bpy('/path/scene.json', pathlib.Path('./eval/viz/misc/test-bpy'))

说明：
 - 此脚本不依赖 ReSpace 的模型推理，只读取已有场景 JSON。
 - 使用 src/blender_renderer.render_scene_frame_bpy_inproc 生成 top / diag 两张图。
 - 若需要真实资产网格，请扩展 blender_renderer 中的 _bpy_create_simple_object。
"""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

def _should_print_debug():
    """Check if debug prints should be shown based on environment variable"""
    return os.environ.get('BPY_VERBOSE', '0') == '1'

def _debug_print(*args, **kwargs):
    """Print debug message only if verbose mode is enabled"""
    if _should_print_debug():
        print(*args, **kwargs)

try:
    from blender_renderer import render_scene_frame_bpy_inproc
except Exception as e:  # noqa: BLE001
    print('[FATAL] 无法导入 Blender 渲染模块 (blender_renderer.py):', e, file=sys.stderr)
    print('请确保：\n  1) 在 Blender 自带 Python 环境中运行 (blender --background ...)\n  2) 工作目录包含 blender_renderer.py\n  3) PYTHONPATH 包含项目根目录', file=sys.stderr)
    sys.exit(2)

# 封装核心逻辑，便于交互式调用
def render_scene_with_bpy(scene_path: str | Path, output_dir: Path):
    scene_path = Path(scene_path)
    if not scene_path.is_file():
        raise FileNotFoundError(f'Scene JSON 不存在: {scene_path}')
    with open(scene_path, 'r') as f:
        scene = json.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    _debug_print(f'📁 加载场景: {scene_path.name}')
    
    # Support both old and new scene formats for display
    if 'groups' in scene:
        objects_count = sum(len(group.get('objects', [])) for group in scene.get('groups', []))
        _debug_print(f'   room_type: {scene.get("room_type", "unknown")} | groups: {len(scene.get("groups", []))} | total objects: {objects_count}')
    else:
        _debug_print(f'   room_type: {scene.get("room_type", "unknown")} | objects: {len(scene.get("objects", []))}')
    
    _debug_print('🧱 开始使用 Blender (bpy) 渲染 (top / diag)...')
    paths = render_scene_frame_bpy_inproc(scene, output_dir)
    _debug_print('✅ 渲染完成:')
    for p in paths:
        _debug_print('   ->', p)
    _debug_print('📁 输出目录:', output_dir)


def parse_args():
    ap = argparse.ArgumentParser(description='Blender in-process rendering demo for ReSpace scene JSON')
    ap.add_argument('--scene', required=False, default='/path/to/workspace/respace/dataset-ssr3dfront/scenes/0a8d471a-2587-458a-9214-586e003e9cf9-3a529582-5a95-4018-9c87-c4a2691dc2f2.json', help='场景 JSON 路径')
    ap.add_argument('--out', required=False, default='./eval/viz/misc/test-bpy', help='输出根目录 (内部会创建 top/ diag)')
    
    # 在 Blender 环境中，sys.argv 会包含 Blender 的参数
    # 我们只需要处理 -- 后面的参数
    import sys
    if '--' in sys.argv:
        # 找到 -- 的位置，只解析 -- 后面的参数
        try:
            dash_index = sys.argv.index('--')
            args_to_parse = sys.argv[dash_index + 1:]
        except ValueError:
            args_to_parse = sys.argv[1:]
    else:
        args_to_parse = sys.argv[1:]
    
    return ap.parse_args(args_to_parse)


def main():
    args = parse_args()
    render_scene_with_bpy(args.scene, Path(args.out))
    print('\n🎉 Blender 渲染示例完成!')


if __name__ == '__main__':
    main()
