#!/usr/bin/env python
"""Batch optimize bedroom and dining_room scenes - integrated version without subprocess."""
import os
import sys
import json
import glob
import argparse
import numpy as np
import trimesh
import copy
import time
import random
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Azure OpenAI imports
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

# Azure OpenAI配置
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
AZURE_OPENAI_SCOPE = "YOUR_AZURE_OPENAI_SCOPE"

def setup_azure_client():
    """Setup Azure OpenAI client with proper authentication."""
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential()
    )
    token_provider = get_bearer_token_provider(credential, AZURE_OPENAI_SCOPE)
    
    client = AzureOpenAI(
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider
    )
    return client

# Add eval directory to path to import myeval
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

from eval.myeval import (
    parse_scene_data,
    create_room_mesh,
    create_floor_polygon,
    find_objaverse_glb,
    get_object_field,
    check_object_out_of_bounds
)


class SceneOptimizer:
    def __init__(self, scene_json, models_path, format_type='ours', client=None):
        self.scene_json = scene_json
        self.models_path = models_path
        self.format_type = format_type
        self.client = client
        
        # Parse initial data
        self.bounds_bottom, self.bounds_top, self.objects_data = parse_scene_data(self.scene_json, format_type)
        
        # Create room mesh (static)
        self.room_mesh = create_room_mesh(self.bounds_bottom, self.bounds_top)
        self.floor_polygon = create_floor_polygon(self.bounds_bottom)
        
        # Calculate room height range
        bounds_bottom_arr = np.array(self.bounds_bottom)
        bounds_top_arr = np.array(self.bounds_top)
        self.room_height_min = bounds_bottom_arr[:, 1].min()
        self.room_height_max = bounds_top_arr[:, 1].max()
        self.room_center = np.mean(bounds_bottom_arr, axis=0)
        
        # Cache for base meshes (untransformed)
        self.mesh_cache = {}
        
        # Object importance cache
        self.object_importance = {}

    def get_base_mesh(self, obj_data):
        asset_source = obj_data.get('asset_source', '3d-future')
        jid = get_object_field(obj_data, 'jid', self.format_type)
        uid = obj_data.get('uid')
        
        if asset_source == 'objaverse':
            asset_id = uid
        else:
            asset_id = jid
            
        cache_key = (asset_source, asset_id)
        if cache_key in self.mesh_cache:
            return self.mesh_cache[cache_key]
            
        # Load mesh
        model_path = None
        if asset_source == 'objaverse':
            if uid:
                model_path = find_objaverse_glb(uid)
                if model_path:
                    model_path = str(model_path)
        else:
            model_path = os.path.join(self.models_path, jid, 'raw_model.glb')
            
        mesh = None
        if model_path and os.path.exists(model_path):
            try:
                loaded = trimesh.load(model_path, force='mesh')
                if isinstance(loaded, trimesh.Scene):
                    if len(loaded.geometry) > 0:
                        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                else:
                    mesh = loaded
            except Exception as e:
                pass
        
        if mesh is None:
            # Fallback box
            target_size = get_object_field(obj_data, 'size', self.format_type)
            mesh = trimesh.creation.box(extents=target_size)
            
        self.mesh_cache[cache_key] = mesh.copy()
        return self.mesh_cache[cache_key]

    def get_transformed_mesh(self, obj_data):
        mesh = self.get_base_mesh(obj_data).copy()
        
        target_size = get_object_field(obj_data, 'size', self.format_type)
        original_size = mesh.extents
        target_size_array = np.array(target_size)
        scale_factors = target_size_array / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        pos = np.array(obj_data['pos'])
        rot_xyzw = obj_data['rot']
        
        rotation = R.from_quat(rot_xyzw)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = pos
        
        bounds = mesh.bounds
        bottom_center_pivot = np.array([
            (bounds[0, 0] + bounds[1, 0]) / 2,
            bounds[0, 1],
            (bounds[0, 2] + bounds[1, 2]) / 2
        ])
        
        center_transform = np.eye(4)
        center_transform[:3, 3] = -bottom_center_pivot
        
        mesh.apply_transform(center_transform)
        mesh.apply_transform(transform_matrix)
        
        return mesh

    def get_scene_meshes(self):
        meshes = {}
        for i, obj in enumerate(self.objects_data):
            jid = get_object_field(obj, 'jid', self.format_type)
            key = f"{i}_{jid}"
            meshes[key] = self.get_transformed_mesh(obj)
        return meshes

    def check_physics(self):
        meshes = self.get_scene_meshes()
        
        # Collisions
        manager = trimesh.collision.CollisionManager()
        for name, mesh in meshes.items():
            manager.add_object(name, mesh)
            
        is_collision, contact_data = manager.in_collision_internal(return_data=True)
        collision_tolerance = 0.01
        actual_collisions = [d for d in contact_data if d.depth > collision_tolerance]
        
        colliding_indices = set()
        for contact in actual_collisions:
            for name in contact.names:
                idx = int(name.split('_')[0])
                colliding_indices.add(idx)
                
        # OOB
        oob_indices = set()
        for name, mesh in meshes.items():
            idx = int(name.split('_')[0])
            is_oob, _ = check_object_out_of_bounds(
                mesh, self.room_mesh, self.floor_polygon, 
                self.room_height_min, self.room_height_max
            )
            if is_oob:
                oob_indices.add(idx)
                
        return colliding_indices, oob_indices, manager

    def consult_gpt(self, indices_to_check):
        """使用GPT判断物体重要性，如果没有client则使用规则判断"""
        unknown_indices = [i for i in indices_to_check if i not in self.object_importance]
        if not unknown_indices:
            return
        
        # 如果没有 GPT client，使用基于规则的判断
        if self.client is None:
            # 简单规则：根据物体描述判断重要性
            key_keywords = ['bed', 'sofa', 'couch', 'table', 'dining', 'desk', 'wardrobe', 'cabinet', 
                           'chair', 'tv', 'television', 'refrigerator', 'stove', 'sink']
            for i in unknown_indices:
                obj = self.objects_data[i]
                desc = get_object_field(obj, 'desc', self.format_type).lower()
                # 检查是否包含关键物品关键词
                is_key = any(kw in desc for kw in key_keywords)
                self.object_importance[i] = 'Key' if is_key else 'Non-Key'
            return
            
        objects_desc = []
        for i in unknown_indices:
            obj = self.objects_data[i]
            jid = get_object_field(obj, 'jid', self.format_type)
            desc = get_object_field(obj, 'desc', self.format_type)
            objects_desc.append(f"ID {i}: {desc} (JID: {jid})")
            
        prompt = f"""
You are an interior design expert. I have a list of objects in a room.
I need to optimize the layout by removing some objects to fix collisions or out-of-bounds issues.
However, I must KEEP the KEY/ESSENTIAL objects for this room type.
Please classify each object as 'Key' (must keep) or 'Non-Key' (can delete).

Objects:
{chr(10).join(objects_desc)}

Output format (one per line):
ID 0: Key
ID 1: Non-Key
...
"""
        try:
            response = self.client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            content = response.choices[0].message.content
            
            for line in content.split('\n'):
                if ':' in line:
                    parts = line.split(':')
                    id_part = parts[0].strip()
                    type_part = parts[1].strip()
                    if id_part.startswith('ID '):
                        try:
                            idx = int(id_part[3:])
                            if 'Non-Key' in type_part:
                                self.object_importance[idx] = 'Non-Key'
                            else:
                                self.object_importance[idx] = 'Key'
                        except:
                            pass
        except Exception as e:
            for i in unknown_indices:
                self.object_importance[i] = 'Key'

    def optimize(self, max_steps=20):
        for step in range(max_steps):
            colliding_indices, oob_indices, manager = self.check_physics()
            problematic_indices = colliding_indices.union(oob_indices)
            
            if not problematic_indices:
                break
                
            self.consult_gpt(problematic_indices)
            
            indices_to_delete = set()
            sorted_indices = sorted(list(problematic_indices), reverse=True)
            
            for idx in sorted_indices:
                importance = self.object_importance.get(idx, 'Key')
                
                if importance == 'Non-Key':
                    indices_to_delete.add(idx)
                else:
                    obj = self.objects_data[idx]
                    
                    if idx in oob_indices:
                        current_pos = np.array(obj['pos'])
                        direction = self.room_center - current_pos
                        direction[1] = 0 
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                            new_pos = current_pos + direction * 0.2
                            obj['pos'] = new_pos.tolist()
                            
                    elif idx in colliding_indices:
                        jid = get_object_field(obj, 'jid', self.format_type)
                        name = f"{idx}_{jid}"
                        
                        # Remove current object from manager to test against others
                        if manager.remove_object(name):
                            solved = False
                            original_pos = copy.deepcopy(obj['pos'])
                            original_rot = copy.deepcopy(obj['rot'])
                            
                            # Strategy 1: Position fine-tuning (Try 10 times)
                            for _ in range(10):
                                offset = (np.random.rand(3) - 0.5) * 0.2  # +/- 0.1m
                                offset[1] = 0
                                test_pos = (np.array(original_pos) + offset).tolist()
                                obj['pos'] = test_pos
                                
                                test_mesh = self.get_transformed_mesh(obj)
                                if not manager.in_collision_single(test_mesh):
                                    solved = True
                                    manager.add_object(name, test_mesh)
                                    break
                            
                            if not solved:
                                # Revert position to try rotation
                                obj['pos'] = original_pos
                                
                                # Strategy 2: Rotation (-5 to 5 degrees) (Try 10 times)
                                for _ in range(10):
                                    current_rot = R.from_quat(original_rot)
                                    angle = random.uniform(-5, 5)
                                    random_rot = R.from_euler('y', angle, degrees=True)
                                    new_rot = current_rot * random_rot
                                    obj['rot'] = new_rot.as_quat().tolist()
                                    
                                    test_mesh = self.get_transformed_mesh(obj)
                                    if not manager.in_collision_single(test_mesh):
                                        solved = True
                                        manager.add_object(name, test_mesh)
                                        break
                                        
                            if not solved:
                                # Both failed. Apply a random perturbation to escape local optimum
                                obj['pos'] = original_pos
                                obj['rot'] = original_rot
                                
                                current_pos = np.array(obj['pos'])
                                offset = (np.random.rand(3) - 0.5) * 0.2
                                offset[1] = 0
                                obj['pos'] = (current_pos + offset).tolist()
                                
                                manager.add_object(name, self.get_transformed_mesh(obj))
                        else:
                            # Fallback if name mismatch (should not happen)
                            current_pos = np.array(obj['pos'])
                            offset = (np.random.rand(3) - 0.5) * 0.2
                            offset[1] = 0
                            obj['pos'] = (current_pos + offset).tolist()

            if indices_to_delete:
                self.objects_data = [obj for i, obj in enumerate(self.objects_data) if i not in indices_to_delete]
                self.object_importance = {}
                
        return self.objects_data

    def get_optimized_json(self):
        new_json = copy.deepcopy(self.scene_json)
        
        if 'groups' in new_json:
            del new_json['groups']
        
        if self.format_type == 'respace' and 'scene' in new_json and 'objects' in new_json['scene']:
             new_json['scene']['objects'] = self.objects_data
        else:
             new_json['objects'] = self.objects_data
            
        return new_json


def optimize_single_scene(args):
    """优化单个场景文件（用于多进程）"""
    scene_file, output_file, format_type, models_path = args
    
    try:
        with open(scene_file, 'r') as f:
            scene_json = json.load(f)
        
        # 不使用 GPT，直接基于规则优化
        optimizer = SceneOptimizer(scene_json, models_path, format_type, client=None)
        optimizer.optimize()
        optimized_json = optimizer.get_optimized_json()
        
        with open(output_file, 'w') as f:
            json.dump(optimized_json, f, indent=2)
        
        return (True, scene_file.name, None)
    except Exception as e:
        return (False, scene_file.name, str(e))


def batch_optimize_parallel(input_dir, output_dir, format_type, models_path, num_workers=None):
    """并行优化一个房间类型中的所有场景"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(list(input_path.glob("*_final_scene.json")))
    scene_files = [f for f in scene_files if 'evaluation_results' not in f.name]
    
    print(f"Found {len(scene_files)} scenes to optimize in {input_dir}")
    
    if len(scene_files) == 0:
        return 0, 0

    # 准备任务参数
    tasks = []
    for scene_file in scene_files:
        output_file = output_path / scene_file.name
        if not output_file.exists():
            tasks.append((scene_file, output_file, format_type, models_path))
    
    already_done = len(scene_files) - len(tasks)
    print(f"  {already_done} already optimized, {len(tasks)} to process")
    
    if len(tasks) == 0:
        return already_done, 0

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), len(tasks), 8)
    
    success_count = already_done
    fail_count = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(optimize_single_scene, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Optimizing {input_path.name}"):
            success, filename, error = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"\n  Error: {filename}: {error}")
    
    print(f"\nCompleted: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def batch_optimize(input_dir, output_dir, format_type, client, models_path):
    """原始的顺序优化函数（保留兼容性）"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scene_files = sorted(list(input_path.glob("*_final_scene.json")))
    scene_files = [f for f in scene_files if 'evaluation_results' not in f.name]
    
    print(f"Found {len(scene_files)} scenes to optimize in {input_dir}")

    success_count = 0
    fail_count = 0
    
    for scene_file in tqdm(scene_files, desc=f"Optimizing {input_path.name}"):
        output_file = output_path / scene_file.name
        
        if output_file.exists():
            success_count += 1
            continue
        
        try:
            with open(scene_file, 'r') as f:
                scene_json = json.load(f)
            
            optimizer = SceneOptimizer(scene_json, models_path, format_type, client)
            optimizer.optimize()
            optimized_json = optimizer.get_optimized_json()
            
            with open(output_file, 'w') as f:
                json.dump(optimized_json, f, indent=2)
            
            success_count += 1
        except Exception as e:
            print(f"\nError optimizing {scene_file.name}: {e}")
            fail_count += 1
    
    print(f"\nCompleted: {success_count} success, {fail_count} failed")
    return success_count, fail_count


def process_room_type(args):
    """处理单个房间类型（用于房间类型级别的并行）"""
    room, base_dir, format_type, models_path, num_scene_workers = args
    
    input_dir = os.path.join(base_dir, room, 'final_scenes_collection')
    output_dir = os.path.join(base_dir, room, 'optimized_scenes')
    
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")
        return room, 0, 0
    
    print(f"\n{'='*60}")
    print(f"Processing {room}")
    print(f"{'='*60}")
    
    success, fail = batch_optimize_parallel(input_dir, output_dir, format_type, models_path, num_scene_workers)
    return room, success, fail


def main():
    parser = argparse.ArgumentParser(description='Batch optimize scenes')
    parser.add_argument('--rooms', nargs='+', default=['bedroom', 'dining_room'],
                       help='Room types to optimize')
    parser.add_argument('--base_dir', type=str, 
                       default='/path/to/SceneReVis/output/rl_ood_B200_v6_e3_s80',
                       help='Base output directory')
    parser.add_argument('--format', type=str, default='respace',
                       help='Scene format (ours or respace)')
    parser.add_argument('--models_path', type=str,
                       default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                       help='Path to 3D models')
    parser.add_argument('--parallel_rooms', action='store_true',
                       help='Process multiple room types in parallel')
    parser.add_argument('--room_workers', type=int, default=None,
                       help='Number of room types to process in parallel')
    parser.add_argument('--scene_workers', type=int, default=4,
                       help='Number of scenes to optimize in parallel per room type')
    args = parser.parse_args()
    
    print(f"Parallel mode: rooms={args.parallel_rooms}, scene_workers={args.scene_workers}")
    
    total_success = 0
    total_fail = 0
    
    if args.parallel_rooms and len(args.rooms) > 1:
        # 并行处理多个房间类型
        room_workers = args.room_workers or len(args.rooms)
        print(f"Processing {len(args.rooms)} room types in parallel with {room_workers} workers")
        
        tasks = [(room, args.base_dir, args.format, args.models_path, args.scene_workers) 
                 for room in args.rooms]
        
        with ProcessPoolExecutor(max_workers=room_workers) as executor:
            futures = {executor.submit(process_room_type, task): task[0] for task in tasks}
            
            for future in as_completed(futures):
                room, success, fail = future.result()
                total_success += success
                total_fail += fail
                print(f"\n[{room}] Done: {success} success, {fail} failed")
    else:
        # 顺序处理房间类型，但并行处理每个房间内的场景
        for room in args.rooms:
            input_dir = os.path.join(args.base_dir, room, 'final_scenes_collection')
            output_dir = os.path.join(args.base_dir, room, 'optimized_scenes')
            
            if not os.path.exists(input_dir):
                print(f"Directory not found: {input_dir}")
                continue
                
            print(f"\n{'='*60}")
            print(f"Processing {room}")
            print(f"{'='*60}")
            
            success, fail = batch_optimize_parallel(input_dir, output_dir, args.format, 
                                                    args.models_path, args.scene_workers)
            total_success += success
            total_fail += fail
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_success} success, {total_fail} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
