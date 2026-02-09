#!/usr/bin/env python3
"""
Enhanced scene validation script with full voxel-based validation.
This script provides more accurate detection of overlaps and out-of-bounds objects.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
from dotenv import load_dotenv

# Add src to Python path to import eval functions
sys.path.append('/path/to/workspace/respace/src')

from eval import eval_scene

def validate_scenes_full_voxel(scenes_dir, output_dir, voxel_size=0.05, total_loss_threshold=0.00001):
    """
    Full voxel-based validation using eval_scene function.
    This is more accurate but slower than the simple bounding box validation.
    """
    # Get all scene files
    scene_files = list(Path(scenes_dir).glob("*.json"))
    if not scene_files:
        print(f"No scene files found in {scenes_dir}")
        return
    
    print(f"Found {len(scene_files)} scene files for full voxel validation")
    
    # Results storage
    valid_scenes = []
    detailed_results = []
    
    # Statistics
    stats = {
        'total': len(scene_files),
        'valid': 0,
        'invalid_overlap_only': 0,
        'invalid_bounds_only': 0,
        'invalid_both': 0,
        'errors': 0
    }
    
    print(f"Running full voxel-based validation (voxel_size={voxel_size}, threshold={total_loss_threshold})...")
    
    for scene_file in tqdm(scene_files, desc="Full validation"):
        scene_name = scene_file.stem
        
        try:
            # Load scene
            with open(scene_file, 'r') as f:
                scene = json.load(f)
            
            # Validate required fields
            required_fields = ['bounds_top', 'bounds_bottom', 'objects']
            if not all(field in scene for field in required_fields):
                detailed_results.append({
                    'scene': scene_name,
                    'valid': False,
                    'reason': 'Missing required fields',
                    'metrics': None
                })
                stats['errors'] += 1
                continue
            
            if not scene.get('objects'):
                valid_scenes.append(scene_name)
                detailed_results.append({
                    'scene': scene_name,
                    'valid': True,
                    'reason': 'No objects to validate',
                    'metrics': {'total_pbl_loss': 0, 'total_oob_loss': 0, 'total_mbl_loss': 0}
                })
                stats['valid'] += 1
                continue
            
            # Run full evaluation with voxel-based collision detection
            metrics = eval_scene(scene, 
                               is_debug=False, 
                               voxel_size=voxel_size, 
                               total_loss_threshold=total_loss_threshold)
            
            is_valid = metrics.get('is_valid_scene_pbl', False)
            total_loss = metrics.get('total_pbl_loss', 0)
            oob_loss = metrics.get('total_oob_loss', 0)
            mbl_loss = metrics.get('total_mbl_loss', 0)
            
            # Determine the reason for invalidity
            if is_valid:
                reason = 'Valid'
                valid_scenes.append(scene_name)
                stats['valid'] += 1
            else:
                if oob_loss > 0 and mbl_loss > 0:
                    reason = f'Both overlap and bounds violations (oob={oob_loss:.3f}, mbl={mbl_loss:.3f})'
                    stats['invalid_both'] += 1
                elif oob_loss > 0:
                    reason = f'Out of bounds violations (oob_loss={oob_loss:.3f})'
                    stats['invalid_bounds_only'] += 1
                elif mbl_loss > 0:
                    reason = f'Object overlaps (mbl_loss={mbl_loss:.3f})'
                    stats['invalid_overlap_only'] += 1
                else:
                    reason = f'High total loss ({total_loss:.3f})'
                    stats['errors'] += 1
            
            detailed_results.append({
                'scene': scene_name,
                'valid': is_valid,
                'reason': reason,
                'metrics': {
                    'total_pbl_loss': float(total_loss),
                    'total_oob_loss': float(oob_loss),
                    'total_mbl_loss': float(mbl_loss),
                    'is_valid_scene_pbl': is_valid
                }
            })
            
        except Exception as e:
            detailed_results.append({
                'scene': scene_name,
                'valid': False,
                'reason': f'Processing error: {str(e)}',
                'metrics': None
            })
            stats['errors'] += 1
    
    # Print statistics
    print("\n" + "="*70)
    print("FULL VOXEL-BASED VALIDATION RESULTS")
    print("="*70)
    print(f"Total scenes processed: {stats['total']}")
    print(f"Valid scenes: {stats['valid']} ({stats['valid']/stats['total']*100:.1f}%)")
    print(f"Invalid scenes breakdown:")
    print(f"  - Overlap only: {stats['invalid_overlap_only']}")
    print(f"  - Out of bounds only: {stats['invalid_bounds_only']}")  
    print(f"  - Both issues: {stats['invalid_both']}")
    print(f"  - Processing errors: {stats['errors']}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save valid scenes list
    valid_scenes_file = os.path.join(output_dir, "valid_scenes_voxel.txt")
    with open(valid_scenes_file, 'w') as f:
        for scene_name in sorted(valid_scenes):
            f.write(f"{scene_name}\n")
    
    # Save detailed results
    results_file = os.path.join(output_dir, "scene_validation_voxel_results.json")
    results = {
        'validation_mode': 'full_voxel',
        'parameters': {
            'voxel_size': voxel_size,
            'total_loss_threshold': total_loss_threshold
        },
        'statistics': stats,
        'valid_scenes': sorted(valid_scenes),
        'detailed_results': detailed_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nValid scenes saved to: {valid_scenes_file}")
    print(f"Number of valid scenes: {len(valid_scenes)}")
    print(f"Detailed results saved to: {results_file}")
    
    return valid_scenes, stats

def compare_validation_results(simple_file, voxel_file, output_dir):
    """Compare results from simple and voxel-based validation."""
    
    # Load simple validation results
    with open(simple_file, 'r') as f:
        simple_scenes = set(line.strip() for line in f)
    
    # Load voxel validation results  
    with open(voxel_file, 'r') as f:
        voxel_scenes = set(line.strip() for line in f)
    
    # Compute differences
    only_simple = simple_scenes - voxel_scenes
    only_voxel = voxel_scenes - simple_scenes
    common = simple_scenes & voxel_scenes
    
    print("\n" + "="*60)
    print("VALIDATION METHOD COMPARISON")
    print("="*60)
    print(f"Simple validation valid scenes: {len(simple_scenes)}")
    print(f"Voxel validation valid scenes: {len(voxel_scenes)}")
    print(f"Common valid scenes: {len(common)}")
    print(f"Only valid in simple: {len(only_simple)}")
    print(f"Only valid in voxel: {len(only_voxel)}")
    
    # Save comparison results
    comparison_file = os.path.join(output_dir, "validation_comparison.json")
    comparison = {
        'simple_valid_count': len(simple_scenes),
        'voxel_valid_count': len(voxel_scenes),
        'common_valid_count': len(common),
        'only_simple_count': len(only_simple),
        'only_voxel_count': len(only_voxel),
        'common_valid_scenes': sorted(list(common)),
        'only_simple_valid': sorted(list(only_simple)),
        'only_voxel_valid': sorted(list(only_voxel))
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison results saved to: {comparison_file}")
    
    return comparison

def main():
    """Main function to run full voxel-based validation."""
    # Load environment variables
    load_dotenv("/path/to/workspace/respace/.env")
    
    scenes_dir = "/path/to/workspace/respace/dataset-ssr3dfront/scenes"
    output_dir = "/path/to/workspace/respace"
    
    # Run full voxel-based validation
    valid_scenes, stats = validate_scenes_full_voxel(
        scenes_dir=scenes_dir,
        output_dir=output_dir,
        voxel_size=0.05,  # 5cm voxels
        total_loss_threshold=0.00001
    )
    
    # Compare with simple validation if both exist
    simple_file = os.path.join(output_dir, "valid_scenes.txt")
    voxel_file = os.path.join(output_dir, "valid_scenes_voxel.txt")
    
    if os.path.exists(simple_file) and os.path.exists(voxel_file):
        compare_validation_results(simple_file, voxel_file, output_dir)

if __name__ == "__main__":
    main()
