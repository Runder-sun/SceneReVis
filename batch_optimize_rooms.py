#!/usr/bin/env python
"""Batch optimize bedroom and dining_room scenes."""
import os
import sys
import glob
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

# Get the current Python executable path
PYTHON_EXECUTABLE = sys.executable

def batch_optimize(input_dir, output_dir, format_type='ours'):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all scene files
    scene_files = sorted(list(input_path.glob("*_final_scene.json")))
    
    # Skip evaluation_results.json
    scene_files = [f for f in scene_files if 'evaluation_results' not in f.name]
    
    print(f"Found {len(scene_files)} scenes to optimize in {input_dir}")

    success_count = 0
    fail_count = 0
    
    for scene_file in tqdm(scene_files, desc=f"Optimizing {input_path.name}"):
        output_file = output_path / scene_file.name
        
        # Skip if already optimized
        if output_file.exists():
            print(f"Skipping {scene_file.name} (already exists)")
            success_count += 1
            continue
        
        cmd = [
            PYTHON_EXECUTABLE,
            "/path/to/SceneReVis/optimize_scene.py",
            "--scene_file", str(scene_file),
            "--output_file", str(output_file),
            "--format", format_type
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"\nError optimizing {scene_file.name}: {e}")
            print(f"Stderr: {e.stderr}")
            fail_count += 1
        except subprocess.TimeoutExpired:
            print(f"\nTimeout optimizing {scene_file.name}")
            fail_count += 1
        except Exception as e:
            print(f"\nUnexpected error for {scene_file.name}: {e}")
            fail_count += 1
    
    print(f"\nCompleted: {success_count} success, {fail_count} failed")
    return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='Batch optimize scenes')
    parser.add_argument('--rooms', nargs='+', default=['bedroom', 'dining_room'],
                       help='Room types to optimize')
    parser.add_argument('--base_dir', type=str, 
                       default='/path/to/SceneReVis/output/rl_ood_B200_v6_e3_s80',
                       help='Base output directory')
    parser.add_argument('--format', type=str, default='ours',
                       help='Scene format (ours or respace)')
    args = parser.parse_args()
    
    total_success = 0
    total_fail = 0
    
    for room in args.rooms:
        input_dir = os.path.join(args.base_dir, room, 'final_scenes_collection')
        output_dir = os.path.join(args.base_dir, room, 'optimized_scenes')
        
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Processing {room}")
        print(f"{'='*60}")
        
        success, fail = batch_optimize(input_dir, output_dir, args.format)
        total_success += success
        total_fail += fail
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_success} success, {total_fail} failed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
