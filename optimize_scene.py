import os
import sys
import json
import argparse
import numpy as np
import trimesh
import copy
import time
import random
from scipy.spatial.transform import Rotation as R
from pathlib import Path

# Add eval directory to path to import myeval
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

try:
    from eval.myeval import (
        parse_scene_data,
        create_room_mesh,
        create_floor_polygon,
        find_objaverse_glb,
        get_object_field,
        check_object_out_of_bounds
    )
except ImportError:
    # Fallback if running from eval directory
    sys.path.append(os.path.dirname(__file__))
    from myeval import (
        parse_scene_data,
        create_room_mesh,
        create_floor_polygon,
        find_objaverse_glb,
        get_object_field,
        check_object_out_of_bounds
    )

from infer import setup_azure_client, AZURE_OPENAI_DEPLOYMENT_NAME

class SceneOptimizer:
    def __init__(self, scene_file, models_path, format_type='ours', client=None):
        self.scene_file = scene_file
        self.models_path = models_path
        self.format_type = format_type
        self.client = client
        
        # Load scene data
        with open(scene_file, 'r') as f:
            self.scene_json = json.load(f)
            
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
        self.mesh_cache = {} # (asset_source, asset_id) -> mesh
        
        # Object importance cache
        self.object_importance = {} # jid -> 'Key' or 'Non-Key'

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
                print(f"Error loading mesh {model_path}: {e}")
        
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
            # Use index to ensure uniqueness
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
        # Filter indices that are not yet classified
        # Note: Since we might delete objects, indices shift. 
        # But we rebuild the list and clear cache if we delete.
        # So here we just check if we have info for current indices.
        # Actually, if we clear cache, we re-ask. That's fine.
        
        unknown_indices = [i for i in indices_to_check if i not in self.object_importance]
        if not unknown_indices:
            return
            
        # Prepare prompt
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
            
            # Parse response
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
            print(f"GPT Error: {e}")
            # Default to Key if error
            for i in unknown_indices:
                self.object_importance[i] = 'Key'

    def optimize(self, max_steps=20):
        print(f"Starting optimization for {len(self.objects_data)} objects...")
        
        for step in range(max_steps):
            colliding_indices, oob_indices, manager = self.check_physics()
            problematic_indices = colliding_indices.union(oob_indices)
            
            if not problematic_indices:
                print(f"Step {step}: No issues found! Optimization complete.")
                break
                
            print(f"Step {step}: Found {len(colliding_indices)} colliding, {len(oob_indices)} OOB.")
            
            # Consult GPT
            self.consult_gpt(problematic_indices)
            
            # Actions
            indices_to_delete = set()
            
            # Sort indices descending to delete safely later
            sorted_indices = sorted(list(problematic_indices), reverse=True)
            
            for idx in sorted_indices:
                importance = self.object_importance.get(idx, 'Key')
                
                if importance == 'Non-Key':
                    indices_to_delete.add(idx)
                    print(f"  Deleting Non-Key object {idx}")
                else:
                    # Try to move Key object
                    obj = self.objects_data[idx]
                    
                    if idx in oob_indices:
                        # Move towards center
                        current_pos = np.array(obj['pos'])
                        # Move only X, Z
                        direction = self.room_center - current_pos
                        direction[1] = 0 
                        if np.linalg.norm(direction) > 1e-6:
                            direction = direction / np.linalg.norm(direction)
                            # Move by 0.2m
                            new_pos = current_pos + direction * 0.2
                            obj['pos'] = new_pos.tolist()
                            print(f"  Moving Key object {idx} (OOB) towards center")
                            
                    elif idx in colliding_indices:
                        # Try to resolve collision with fine-tuning
                        jid = get_object_field(obj, 'jid', self.format_type)
                        name = f"{idx}_{jid}"
                        
                        # Remove current object from manager to test against others
                        if manager.remove_object(name):
                            solved = False
                            original_pos = copy.deepcopy(obj['pos'])
                            original_rot = copy.deepcopy(obj['rot'])
                            
                            # Strategy 1: Position fine-tuning (Try 10 times)
                            for _ in range(10):
                                offset = (np.random.rand(3) - 0.5) * 0.2 # +/- 0.1m
                                offset[1] = 0
                                test_pos = (np.array(original_pos) + offset).tolist()
                                obj['pos'] = test_pos
                                
                                test_mesh = self.get_transformed_mesh(obj)
                                if not manager.in_collision_single(test_mesh):
                                    solved = True
                                    print(f"  Resolved collision for {idx} via position shift")
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
                                        print(f"  Resolved collision for {idx} via rotation")
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
                                
                                print(f"  Could not resolve {idx} cleanly, applying random perturbation")
                                manager.add_object(name, self.get_transformed_mesh(obj))
                        else:
                            # Fallback if name mismatch (should not happen)
                            current_pos = np.array(obj['pos'])
                            offset = (np.random.rand(3) - 0.5) * 0.2
                            offset[1] = 0
                            obj['pos'] = (current_pos + offset).tolist()
                            print(f"  Perturbing Key object {idx} (Collision) - Manager fallback")

            # Apply deletions
            if indices_to_delete:
                self.objects_data = [obj for i, obj in enumerate(self.objects_data) if i not in indices_to_delete]
                self.object_importance = {} # Clear cache as indices changed
                
        return self.objects_data

    def save_scene(self, output_path):
        new_json = copy.deepcopy(self.scene_json)
        
        # Flatten structure to 'objects' list
        if 'groups' in new_json:
            del new_json['groups']
        
        # Handle respace format structure if needed
        if self.format_type == 'respace' and 'scene' in new_json and 'objects' in new_json['scene']:
             new_json['scene']['objects'] = self.objects_data
        else:
             new_json['objects'] = self.objects_data
            
        with open(output_path, 'w') as f:
            json.dump(new_json, f, indent=2)
        print(f"Saved optimized scene to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Optimize scene layout to fix collisions and OOB.')
    parser.add_argument('--scene_file', type=str, required=True, help='Path to input scene JSON')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output scene JSON')
    parser.add_argument('--models_path', type=str, 
                       default='/path/to/datasets/3d-front/3D-FUTURE-model/',
                       help='Path to 3D models')
    parser.add_argument('--format', type=str, default='ours', choices=['ours', 'respace'],
                       help='Scene format')
    
    args = parser.parse_args()
    
    client = setup_azure_client()
    
    optimizer = SceneOptimizer(args.scene_file, args.models_path, args.format, client)
    optimizer.optimize()
    optimizer.save_scene(args.output_file)

if __name__ == "__main__":
    main()
