#!/usr/bin/env python3
"""Blender version of the example script: follows the main.py workflow, using bpy to render top and diag views.

Usage (recommended inside Blender):
  blender --background --python main_bpy.py -- \
      --scene /abs/path/to/scene.json \
      --out ./eval/viz/misc/test-bpy

If you are already in Blender's interactive Python Console:
  import json
  import pathlib
  from main_bpy import render_scene_with_bpy
  render_scene_with_bpy('/path/scene.json', pathlib.Path('./eval/viz/misc/test-bpy'))

Notes:
 - This script does not depend on ReSpace's model inference; it only reads existing scene JSON.
 - Uses src/blender_renderer.render_scene_frame_bpy_inproc to generate top / diag views.
 - If real asset meshes are needed, extend _bpy_create_simple_object in blender_renderer.
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
    print('[FATAL] Cannot import Blender rendering module (blender_renderer.py):', e, file=sys.stderr)
    print('Please ensure:\n  1) Running in Blender\'s built-in Python environment (blender --background ...)\n  2) Working directory contains blender_renderer.py\n  3) PYTHONPATH includes the project root directory', file=sys.stderr)
    sys.exit(2)

# Wrap core logic for interactive invocation
def render_scene_with_bpy(scene_path: str | Path, output_dir: Path):
    scene_path = Path(scene_path)
    if not scene_path.is_file():
        raise FileNotFoundError(f'Scene JSON not found: {scene_path}')
    with open(scene_path, 'r') as f:
        scene = json.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    _debug_print(f'📁 Loading scene: {scene_path.name}')
    
    # Support both old and new scene formats for display
    if 'groups' in scene:
        objects_count = sum(len(group.get('objects', [])) for group in scene.get('groups', []))
        _debug_print(f'   room_type: {scene.get("room_type", "unknown")} | groups: {len(scene.get("groups", []))} | total objects: {objects_count}')
    else:
        _debug_print(f'   room_type: {scene.get("room_type", "unknown")} | objects: {len(scene.get("objects", []))}')
    
    _debug_print('🧱 Starting Blender (bpy) rendering (top / diag)...')
    paths = render_scene_frame_bpy_inproc(scene, output_dir)
    _debug_print('✅ Rendering complete:')
    for p in paths:
        _debug_print('   ->', p)
    _debug_print('📁 Output directory:', output_dir)


def parse_args():
    ap = argparse.ArgumentParser(description='Blender in-process rendering demo for ReSpace scene JSON')
    ap.add_argument('--scene', required=False, default='/path/to/workspace/respace/dataset-ssr3dfront/scenes/0a8d471a-2587-458a-9214-586e003e9cf9-3a529582-5a95-4018-9c87-c4a2691dc2f2.json', help='Path to scene JSON')
    ap.add_argument('--out', required=False, default='./eval/viz/misc/test-bpy', help='Output root directory (top/ and diag/ will be created inside)')
    
    # In Blender environment, sys.argv includes Blender's arguments
    # We only need to process arguments after --
    import sys
    if '--' in sys.argv:
        # Find the position of --, only parse arguments after it
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
    print('\n🎉 Blender rendering demo complete!')


if __name__ == '__main__':
    main()
