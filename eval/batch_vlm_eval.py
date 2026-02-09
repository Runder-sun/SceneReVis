#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
from pathlib import Path
from vlm_scene_eval import run_evaluation
import concurrent.futures

# Configuration
BASE_RENDER_DIR = "/path/to/SceneReVis/baseline/IDesign/results/rendered_images"
BASE_PROMPTS_DIR = "/path/to/workspace/split_prompts"
ROOM_TYPES = [
    "bedroom",
    "dining_room",
    "entertainment_room",
    "gym",
    "living_room",
    "office",
    "study_room"
]

def evaluate_room_type(room_type, max_workers=4):
    print(f"\n{'='*50}")
    print(f"Evaluating Room Type: {room_type} (Workers: {max_workers})")
    print(f"{'='*50}")
    render_dir = os.path.join(BASE_RENDER_DIR, room_type)
    prompts_file = os.path.join(BASE_PROMPTS_DIR, f"{room_type}.txt")
    output_file = os.path.join(render_dir, "vlm_evaluation_results.json")
    if not os.path.exists(render_dir):
        print(f"Skipping {room_type}: Render directory not found at {render_dir}")
        return None
    if not os.path.exists(prompts_file):
        print(f"Skipping {room_type}: Prompts file not found at {prompts_file}")
        return None
    try:
        results = run_evaluation(
            render_dir=render_dir,
            prompts_file=prompts_file,
            output_file=output_file,
            max_workers=max_workers,
            verbose=False,
            resume=True
        )
        if results and 'average_scores' in results:
            stats = results['average_scores']
            stats['room_type'] = room_type
            stats['count'] = results.get('total_scenes', 0)
            return stats
        else:
            print(f"No results returned for {room_type}")
            return None
    except Exception as e:
        print(f"Error evaluating {room_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    all_stats = []
    
    print(f"Starting batch evaluation for {len(ROOM_TYPES)} room types...")
    
    # Configurable workers per room
    WORKERS_PER_ROOM = 4
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_room = {executor.submit(evaluate_room_type, room_type, WORKERS_PER_ROOM): room_type for room_type in ROOM_TYPES}
        for future in concurrent.futures.as_completed(future_to_room):
            room_type = future_to_room[future]
            stats = future.result()
            if stats:
                all_stats.append(stats)

    # Generate Summary Report
    if all_stats:
        print("\n\n")
        print("="*80)
        print("FINAL COMPREHENSIVE REPORT")
        print("="*80)
        
        df = pd.DataFrame(all_stats)
        
        # Reorder columns
        cols = ['room_type', 'count'] + [c for c in df.columns if c not in ['room_type', 'count']]
        df = df[cols]
        
        # Print per-room breakdown
        print("\nPer-Room Type Breakdown:")
        try:
            print(df.to_markdown(index=False, floatfmt=".2f"))
        except ImportError:
            print(df.to_string(index=False, float_format="%.2f"))
        
        # Calculate overall averages
        numeric_cols = [c for c in df.columns if c not in ['room_type', 'count']]
        overall_avg = df[numeric_cols].mean()
        
        print("\nOverall Averages across all room types:")
        try:
            print(overall_avg.to_markdown(floatfmt=".2f"))
        except ImportError:
            print(overall_avg.to_string(float_format="%.2f"))
        
        # Save summary to CSV
        summary_path = os.path.join(BASE_RENDER_DIR, "comprehensive_vlm_eval_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSummary report saved to: {summary_path}")
        
    else:
        print("\nNo stats collected.")

if __name__ == "__main__":
    main()
