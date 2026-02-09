#!/usr/bin/env python3
import os
import json
import glob
import subprocess
import sys
import shutil
from pathlib import Path

# Add current directory to path to import image_merger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from image_merger import merge_rendered_views_with_annotations

# Configuration
PROJECT_ROOT = Path("/path/to/SceneReVis")
SCENES_DIR = PROJECT_ROOT / "baseline/IDesign/output_batch"
PROMPTS_DIR = PROJECT_ROOT / "test/split_prompts"
OUTPUT_DIR = PROJECT_ROOT / "baseline/IDesign/results/rendered_images"
UTILS_DIR = PROJECT_ROOT / "utils"
MAIN_BPY = UTILS_DIR / "main_bpy.py"

def load_prompts():
    prompts_map = {}
    if not PROMPTS_DIR.exists():
        print(f"Prompts directory not found: {PROMPTS_DIR}")
        return prompts_map
        
    for prompt_file in PROMPTS_DIR.glob("*.txt"):
        room_type = prompt_file.stem # e.g. "bedroom"
        with open(prompt_file, "r") as f:
            # Store prompts with their 1-based index
            prompts = [line.strip() for line in f if line.strip()]
            prompts_map[room_type] = prompts
    return prompts_map

def find_prompt_index(prompt_text, room_type, prompts_map):
    if room_type not in prompts_map:
        return None
    
    target = prompt_text.strip()
    prompts = prompts_map[room_type]
    
    try:
        return prompts.index(target) + 1
    except ValueError:
        # Try to find by matching the beginning if exact match fails
        # (Sometimes prompts in JSON might be truncated or slightly different?)
        # But based on previous check, it seems exact match works.
        # Let's try a more robust check just in case
        for i, p in enumerate(prompts):
            if p == target:
                return i + 1
            # If target is a substring of p or vice versa (handling potential truncation)
            if len(target) > 50 and (target in p or p in target):
                return i + 1
        return None

def render_scene(scene_path, output_path):
    # Create temp dir for blender output
    temp_out = output_path.parent / f"temp_{output_path.stem}"
    if temp_out.exists():
        shutil.rmtree(temp_out)
    temp_out.mkdir(parents=True, exist_ok=True)
    
    # Run blender
    # Ensure PYTHONPATH includes utils dir so main_bpy can import blender_renderer
    env = os.environ.copy()
    env["PYTHONPATH"] = str(UTILS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    
    cmd = [
        "blender",
        "--background",
        "--python", str(MAIN_BPY),
        "--",
        "--scene", str(scene_path),
        "--out", str(temp_out)
    ]
    
    print(f"Rendering {scene_path.name}...")
    try:
        # Capture output to avoid cluttering terminal, print only on error
        result = subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error rendering {scene_path.name}:")
        print(e.stdout.decode())
        print(e.stderr.decode())
        return False
        
    # Merge images
    # main_bpy.py -> render_scene_frame_bpy_inproc uses default filename='frame'
    top_view = temp_out / "top" / "frame.png"
    diag_view = temp_out / "diag" / "frame.png"
    
    if top_view.exists() and diag_view.exists():
        try:
            merge_rendered_views_with_annotations(top_view, diag_view, output_path)
            print(f"Saved merged image to {output_path}")
            # Cleanup temp
            shutil.rmtree(temp_out)
            return True
        except Exception as e:
            print(f"Error merging images: {e}")
            return False
    else:
        print(f"Missing render outputs for {scene_path.name}")
        if not top_view.exists(): print(f"  Missing: {top_view}")
        if not diag_view.exists(): print(f"  Missing: {diag_view}")
        return False

def main():
    prompts_map = load_prompts()
    print(f"Loaded prompts for: {list(prompts_map.keys())}")
    
    if not SCENES_DIR.exists():
        print(f"Scenes directory not found: {SCENES_DIR}")
        return

    # Walk through scenes
    for room_dir in SCENES_DIR.iterdir():
        if not room_dir.is_dir():
            continue
            
        room_type_key = room_dir.name # e.g. "bedroom"
        
        # Handle potential mismatch in naming (e.g. "living_room" vs "livingroom")
        # The prompt files are like "living_room.txt"
        # The folders are like "living_room"
        # So it should match.
        
        if room_type_key not in prompts_map:
            print(f"Warning: No prompts found for room type {room_type_key}")
            continue
            
        output_room_dir = OUTPUT_DIR / room_type_key
        output_room_dir.mkdir(parents=True, exist_ok=True)
        
        json_files = list(room_dir.glob("*.json"))
        print(f"Found {len(json_files)} scenes in {room_type_key}")
        
        for scene_file in json_files:
            if scene_file.name == "evaluation_results.json":
                continue
                
            try:
                # IDesign uses numeric filenames like 0000.json, 0001.json, etc.
                # Extract the index from the filename directly
                file_stem = scene_file.stem  # e.g. "0000", "0001", etc.
                
                try:
                    # Convert filename to 1-based index (0000.json -> prompt_1.png)
                    index = int(file_stem) + 1
                except ValueError:
                    # If filename is not numeric, try the old method
                    with open(scene_file, "r") as f:
                        data = json.load(f)
                        prompt_text = data.get("room_type", "")
                    index = find_prompt_index(prompt_text, room_type_key, prompts_map)
                    if index is None:
                        print(f"Could not find prompt index for {scene_file.name}")
                        continue
                    
                output_filename = f"prompt_{index}.png"
                output_path = output_room_dir / output_filename
                
                if output_path.exists():
                    print(f"Skipping {output_filename}, already exists.")
                    continue
                    
                render_scene(scene_file, output_path)
                
            except Exception as e:
                print(f"Failed to process {scene_file.name}: {e}")

if __name__ == "__main__":
    main()
