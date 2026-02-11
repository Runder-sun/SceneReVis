import os
import json
import re
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import uuid
import atexit
import sys
import numpy as np
import copy
import logging
import random

# Set swift and vllm log level to WARNING to reduce output
logging.getLogger("swift").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Add eval directory to path for physics optimization
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

# Import helper functions for physics optimization (not importing SceneOptimizer to avoid circular imports)
try:
    from eval.myeval import parse_scene_data, create_room_mesh, create_floor_polygon, get_object_field
    PHYSICS_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import eval utils: {e}. Physics optimization will be disabled.")
    PHYSICS_OPTIMIZATION_AVAILABLE = False

# Azure OpenAI imports
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Azure OpenAI configuration
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
AZURE_OPENAI_SCOPE = "YOUR_AZURE_OPENAI_SCOPE"

# Set necessary environment variables for asset retrieval
os.environ['PTH_3DFUTURE_ASSETS'] = '/path/to/datasets/3d-front/3D-FUTURE-model'
os.environ['PTH_INVALID_ROOMS'] = './metadata/invalid_threed_front_rooms.txt'
os.environ['PTH_ASSETS_METADATA'] = './metadata/model_info_3dfuture_assets.json'
os.environ['PTH_ASSETS_METADATA_SCALED'] = './metadata/model_info_3dfuture_assets_scaled.json'
os.environ['PTH_ASSETS_METADATA_SIMPLE_DESCS'] = './metadata/model_info_3dfuture_assets_simple_descs.json'
os.environ['PTH_ASSETS_METADATA_PROMPTS'] = './metadata/model_info_3dfuture_assets_prompts.json'
os.environ['PTH_ASSETS_EMBED'] = './metadata/model_info_3dfuture_assets_embeds.pickle'

# When possible, patch modelscope's dynamic import functions to avoid triggering extra imports and exceptions on interpreter exit
try:
    import modelscope
    # Set try_import_from_hf to a safe no-op implementation (no exceptions thrown)
    if hasattr(modelscope, 'try_import_from_hf'):
        modelscope.try_import_from_hf = lambda *a, **kw: None
    # Compatible with some versions' internal hook names
    if hasattr(modelscope, '_extra_import_func'):
        modelscope._extra_import_func = lambda name: None
    print('Patched modelscope dynamic import hooks to be safe for atexit')
except Exception:
    # If modelscope is not installed or patching fails, don't interrupt execution
    pass

from swift.llm import (
    PtEngine, VllmEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from swift.plugin import InferStats
from typing import List

# Add cleanup function for exit
def cleanup_on_exit():
    """Cleanup function called on program exit"""
    try:
        # Clean up modules that may cause conflicts
        import sys
        conflicting_modules = []
        
        # Collect modules that need to be cleaned up
        for module_name in list(sys.modules.keys()):
            if any(pattern in module_name.lower() for pattern in [
                'addon_utils', 'bpy.ops', 'blender', 
                'modelscope.utils.import_utils'
            ]):
                conflicting_modules.append(module_name)
        
        # Safely remove these modules
        for module_name in conflicting_modules:
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            except:
                pass
                
        # Try to clean up bpy scene data
        if 'bpy' in sys.modules:
            try:
                import bpy
                bpy.ops.wm.read_factory_settings(use_empty=True)
            except:
                pass
    except:
        pass

# Register exit cleanup function in a safer way
import weakref
def safe_cleanup():
    try:
        cleanup_on_exit()
    except:
        pass

atexit.register(safe_cleanup)


# ============== Physics Optimization: Rule-based Scene Optimizer ==============

class RuleBasedSceneOptimizer:
    """
    Rule-based scene optimizer for resolving object collision and out-of-bounds issues.
    Does not rely on LLM; adjusts positions and removes conflicting objects through rules.
    """
    
    def __init__(self, scene_file, models_path, format_type='ours', client=None):
        if not PHYSICS_OPTIMIZATION_AVAILABLE:
            raise ImportError("Physics optimization modules not available")
        
        self.scene_file = scene_file
        self.models_path = models_path
        self.format_type = format_type
        self.client = client  # Kept for compatibility but not used
        
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
        self.mesh_cache = {} 
        self.indices_to_delete = set()
        
        # Track deleted object categories (for feedback to the model)
        self.deleted_objects_desc = []  # Store descriptions of deleted objects
    
    def get_object_category(self, idx):
        """Get object category description (for feedback)"""
        obj = self.objects_data[idx]
        desc = get_object_field(obj, 'desc', self.format_type)
        if not desc:
            return 'unknown object'
        # Extract short category name (usually the last one or two words or main noun of the description)
        # Simple approach: return the first few words of the description
        words = desc.split()
        if len(words) <= 3:
            return desc
        # Try to extract core noun (usually at the end)
        return ' '.join(words[-3:]).strip()

    def get_object_volume(self, idx):
        """Calculate bounding box volume as a proxy for size"""
        obj = self.objects_data[idx]
        size = get_object_field(obj, 'size', self.format_type)
        return size[0] * size[1] * size[2]

    def get_transformed_mesh(self, obj):
        """Get transformed mesh for an object - uses actual models or fallback box"""
        import trimesh
        from scipy.spatial.transform import Rotation as R
        
        jid = get_object_field(obj, 'jid', self.format_type)
        size = get_object_field(obj, 'size', self.format_type)
        pos = obj.get('pos', [0, 0, 0])
        rot = obj.get('rot', [0, 0, 0, 1])  # Default quaternion [x, y, z, w]
        
        # Ensure size is valid
        try:
            size = [float(s) for s in size]
            if any(s <= 0 for s in size):
                size = [0.1, 0.1, 0.1]  # Default small size
        except (TypeError, ValueError):
            size = [0.1, 0.1, 0.1]
        
        # Ensure pos is valid
        try:
            pos = [float(p) for p in pos]
        except (TypeError, ValueError):
            pos = [0, 0, 0]
        
        # Handle rotation - supports both quaternion and Euler angle formats
        try:
            if rot is None:
                rotation = R.from_quat([0, 0, 0, 1])  # Identity quaternion
            elif isinstance(rot, (int, float)):
                # Single value: Y-axis rotation (radians)
                rotation = R.from_euler('y', float(rot), degrees=False)
            elif isinstance(rot, list):
                if len(rot) == 4:
                    # Quaternion format [x, y, z, w]
                    rot = [float(r) for r in rot]
                    rotation = R.from_quat(rot)
                elif len(rot) == 3:
                    # Euler angle format [rx, ry, rz] (radians)
                    rot = [float(r) for r in rot]
                    rotation = R.from_euler('xyz', rot, degrees=False)
                else:
                    rotation = R.from_quat([0, 0, 0, 1])
            else:
                rotation = R.from_quat([0, 0, 0, 1])
        except Exception:
            rotation = R.from_quat([0, 0, 0, 1])
        
        # Create box mesh (simplified model)
        box = trimesh.creation.box(extents=size)
        
        # Move box so bottom center is at origin (similar to optimize_scene.py's approach)
        bounds = box.bounds
        bottom_center_pivot = np.array([
            (bounds[0, 0] + bounds[1, 0]) / 2,
            bounds[0, 1],
            (bounds[0, 2] + bounds[1, 2]) / 2
        ])
        center_transform = np.eye(4)
        center_transform[:3, 3] = -bottom_center_pivot
        box.apply_transform(center_transform)
        
        # Build transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = pos
        
        # Apply transformation
        box.apply_transform(transform_matrix)
        
        return box

    def check_physics(self):
        """Check for collisions and out-of-bounds objects"""
        import trimesh
        from shapely.geometry import Point, box as shapely_box
        
        manager = trimesh.collision.CollisionManager()
        colliding_indices = set()
        oob_indices = set()
        
        # Add all objects to collision manager and check OOB
        for idx, obj in enumerate(self.objects_data):
            jid = get_object_field(obj, 'jid', self.format_type)
            mesh = self.get_transformed_mesh(obj)
            name = f"{idx}_{jid}"
            manager.add_object(name, mesh)
            
            # Check out of bounds using object bounding box corners (not just center)
            pos = obj.get('pos', [0, 0, 0])
            size = get_object_field(obj, 'size', self.format_type)
            try:
                pos = [float(p) for p in pos]
                size = [float(s) for s in size]
                
                # Calculate bounding box corners in XZ plane (floor projection)
                half_x = size[0] / 2
                half_z = size[2] / 2
                corners = [
                    Point(pos[0] - half_x, pos[2] - half_z),  # bottom-left
                    Point(pos[0] + half_x, pos[2] - half_z),  # bottom-right
                    Point(pos[0] - half_x, pos[2] + half_z),  # top-left
                    Point(pos[0] + half_x, pos[2] + half_z),  # top-right
                ]
                
                # Check if any corner is outside the floor polygon
                for corner in corners:
                    if not self.floor_polygon.contains(corner):
                        oob_indices.add(idx)
                        break
            except Exception:
                pass  # Skip OOB check if position/size is invalid
        
        # Check internal collisions
        is_collision, contact_data = manager.in_collision_internal(return_data=True)
        collision_tolerance = 0.01
        
        if is_collision:
            for contact in contact_data:
                if contact.depth > collision_tolerance:
                    names = list(contact.names)
                    try:
                        idx1 = int(names[0].split('_')[0])
                        idx2 = int(names[1].split('_')[0])
                        colliding_indices.add(idx1)
                        colliding_indices.add(idx2)
                    except:
                        pass
        
        return colliding_indices, oob_indices, manager

    def resolve_oob(self, idx):
        """Move out-of-bounds object towards room center"""
        obj = self.objects_data[idx]
        current_pos = np.array(obj['pos'])
        # Move only X, Z towards center
        direction = self.room_center - current_pos
        direction[1] = 0 
        if np.linalg.norm(direction) > 1e-6:
            direction = direction / np.linalg.norm(direction)
            # Move by 0.2m
            new_pos = current_pos + direction * 0.2
            obj['pos'] = new_pos.tolist()
            print(f"  Moving object {idx} (OOB) towards center")

    def resolve_collision(self, idx, manager):
        """Try to resolve collision by moving or rotating object"""
        from scipy.spatial.transform import Rotation as R
        import copy
        
        obj = self.objects_data[idx]
        jid = get_object_field(obj, 'jid', self.format_type)
        name = f"{idx}_{jid}"
        
        original_pos = copy.deepcopy(obj['pos'])
        original_rot = copy.deepcopy(obj['rot'])
        
        # Remove current object from manager to test against others
        if manager.remove_object(name):
            solved = False
            
            # Strategy 1: Position fine-tuning (Try 10 times with random offsets)
            for _ in range(10):
                offset = (np.random.rand(3) - 0.5) * 0.2  # +/- 0.1m
                offset[1] = 0  # Don't move vertically
                test_pos = (np.array(original_pos) + offset).tolist()
                obj['pos'] = test_pos
                
                test_mesh = self.get_transformed_mesh(obj)
                if not manager.in_collision_single(test_mesh):
                    solved = True
                    print(f"  Resolved collision for {idx} via position shift")
                    manager.add_object(name, test_mesh)
                    return True
            
            if not solved:
                # Revert position to try rotation
                obj['pos'] = original_pos
                
                # Strategy 2: Rotation (-5 to 5 degrees) (Try 10 times)
                # Determine original rotation
                if isinstance(original_rot, list) and len(original_rot) == 4:
                    current_rot = R.from_quat(original_rot)
                elif isinstance(original_rot, list) and len(original_rot) == 3:
                    current_rot = R.from_euler('xyz', original_rot, degrees=False)
                else:
                    current_rot = R.from_quat([0, 0, 0, 1])
                
                for _ in range(10):
                    angle = random.uniform(-5, 5)
                    random_rot = R.from_euler('y', angle, degrees=True)
                    new_rot = current_rot * random_rot
                    obj['rot'] = new_rot.as_quat().tolist()
                    
                    test_mesh = self.get_transformed_mesh(obj)
                    if not manager.in_collision_single(test_mesh):
                        solved = True
                        print(f"  Resolved collision for {idx} via rotation")
                        manager.add_object(name, test_mesh)
                        return True
            
            # If failed, restore original (will be deleted later)
            obj['pos'] = original_pos
            obj['rot'] = original_rot
            manager.add_object(name, self.get_transformed_mesh(obj))
            return False
        return False

    def optimize(self, max_steps=5):
        """Run optimization to resolve collisions and OOB issues"""
        print(f"Starting rule-based optimization for {len(self.objects_data)} objects...")
        
        for step in range(max_steps):
            self.indices_to_delete = set()
            colliding_indices, oob_indices, manager = self.check_physics()
            problematic_indices = colliding_indices.union(oob_indices)
            
            if not problematic_indices:
                print(f"Step {step}: No issues found! Optimization complete.")
                break
                
            print(f"Step {step}: Found {len(colliding_indices)} colliding, {len(oob_indices)} OOB.")
            
            # Handle OOB first - move all OOB objects towards center
            for idx in oob_indices:
                self.resolve_oob(idx)
            
            # Get collision pairs from manager
            is_collision, contact_data = manager.in_collision_internal(return_data=True)
            collision_tolerance = 0.01
            conflicts = []
            for contact in contact_data:
                if contact.depth > collision_tolerance:
                    names = list(contact.names)
                    try:
                        idx1 = int(names[0].split('_')[0])
                        idx2 = int(names[1].split('_')[0])
                        conflicts.append((idx1, idx2))
                    except:
                        pass
            
            processed_indices = set()
            
            # Process each collision pair - delete smaller object if cannot resolve
            for idx1, idx2 in conflicts:
                if idx1 in self.indices_to_delete or idx2 in self.indices_to_delete:
                    continue
                
                # Choose smaller object to try to resolve
                vol1 = self.get_object_volume(idx1)
                vol2 = self.get_object_volume(idx2)
                target_idx = idx1 if vol1 < vol2 else idx2
                
                if target_idx in processed_indices:
                    continue
                
                processed_indices.add(target_idx)
                
                # Try to resolve collision, delete if cannot
                if not self.resolve_collision(target_idx, manager):
                    category = self.get_object_category(target_idx)
                    print(f"  Deleting object {target_idx} ({category}) (Collision unresolved)")
                    self.deleted_objects_desc.append(category)
                    self.indices_to_delete.add(target_idx)

            # Apply deletions
            if self.indices_to_delete:
                # Sort descending to delete correctly
                for idx in sorted(list(self.indices_to_delete), reverse=True):
                    del self.objects_data[idx]
                print(f"  Deleted {len(self.indices_to_delete)} objects in step {step}")
                
        return self.objects_data
    
    def get_deleted_objects_feedback(self):
        """Get feedback information about deleted objects"""
        if not self.deleted_objects_desc:
            return ""
        # Deduplicate
        unique_deleted = list(set(self.deleted_objects_desc))
        return f"[Physics Optimization] The following objects were removed due to collision/out-of-bounds: {', '.join(unique_deleted)}. Consider adding them back in better positions."


def apply_physics_optimization(scene_data: dict, models_path: str, max_steps: int = 5, azure_client=None) -> tuple:
    """
    Apply physics optimization to the scene, resolving collision and out-of-bounds issues.
    
    Args:
        scene_data: Scene data dictionary
        models_path: Path to 3D models
        max_steps: Maximum number of optimization steps
        azure_client: Azure OpenAI client for GPT consultation (optional)
        
    Returns:
        tuple: (optimized scene data, deleted objects feedback string)
    """
    if not PHYSICS_OPTIMIZATION_AVAILABLE:
        print("⚠ Physics optimization not available (missing dependencies)")
        return scene_data, ""
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        # Determine format type
        format_type = 'ours'
        if 'room_envelope' not in scene_data and 'bounds_bottom' in scene_data:
            format_type = 'respace'
        
        # If no client provided, try to create one
        if azure_client is None:
            try:
                azure_client = setup_azure_client()
                print("  Initialized Azure client for GPT-based object importance classification")
            except Exception as e:
                print(f"  Warning: Could not initialize Azure client: {e}")
                print("  Will use fallback strategy (keep all objects when resolving conflicts)")
        
        # Initialize optimizer and run (pass client)
        optimizer = RuleBasedSceneOptimizer(temp_path, models_path, format_type=format_type, client=azure_client)
        optimized_objects = optimizer.optimize(max_steps=max_steps)
        
        # Get feedback for deleted objects
        deleted_feedback = optimizer.get_deleted_objects_feedback()
        if deleted_feedback:
            print(f"  Deleted objects feedback: {deleted_feedback}")
        
        # Update scene data
        result = copy.deepcopy(scene_data)
        result['objects'] = optimized_objects
        if 'groups' in result:
            del result['groups']
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        print(f"✓ Physics optimization completed: {len(optimized_objects)} objects remaining")
        return result, deleted_feedback
        
    except Exception as e:
        print(f"⚠ Physics optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return scene_data, ""


def setup_azure_client() -> AzureOpenAI:
    """Create AzureOpenAI client using Azure CLI or managed identity tokens."""
    scope = AZURE_OPENAI_SCOPE
    credential = get_bearer_token_provider(
        ChainedTokenCredential(
            AzureCliCredential(),
            ManagedIdentityCredential(),
        ),
        scope,
    )
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=credential,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    return client


def generate_vlm_layout_feedback_azure(image_path: str, user_requirement: str, azure_client: AzureOpenAI = None) -> str:
    """
    Use Azure GPT-5.1 to generate brief VLM layout feedback (excluding physics collision/out-of-bounds issues)
    
    Args:
        image_path: Path to rendered image
        user_requirement: User requirement
        azure_client: Azure OpenAI client instance (creates a new one if None)
        
    Returns:
        Brief layout feedback text, or empty string if no issues or failure
    """
    import base64
    
    if azure_client is None:
        azure_client = setup_azure_client()
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Warning: Image not found for VLM feedback: {image_path}")
        return ""
    
    # Convert image to base64
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img_base64 = f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
    except Exception as e:
        print(f"Warning: Failed to read image for VLM feedback: {e}")
        return ""
    
    prompt_layout_feedback = f"""You are an interior design expert. Analyze this room rendering (left: top view, right: diagonal view).

User requirement: {user_requirement}

**YOUR TASK: Identify the TOP 1-2 MOST IMPORTANT layout issues (excluding collision/out-of-bounds problems).**

Focus ONLY on these issues:
1. **Core furniture placement**: Is the bed/sofa against a wall? Is it properly oriented?
2. **Spatial distribution**: Are objects clustered in one area? Is the layout balanced?
3. **Missing essentials**: Is key furniture for this room type missing?

**OUTPUT FORMAT** (one line, max 50 words):
If issues found: "[Layout] issue1; issue2. Suggestion."
If no issues: Leave empty.

**IMPORTANT**: Do NOT mention collision or out-of-bounds issues - they are handled separately.

Example outputs:
- "[Layout] Sofa in center instead of against wall; furniture clustered in corner. Move sofa to south wall and distribute items evenly."
- "[Layout] Missing bed for bedroom. Add a bed against the east or west wall."
- "" (if layout is good)"""

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_layout_feedback},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=100,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        # If VLM returns "no issues" or similar content, return empty string
        if any(phrase in result.lower() for phrase in ["no issue", "looks good", "well-designed", "properly placed", "no problems"]):
            return ""
        
        print(f"VLM layout feedback: {result}")
        return result
        
    except Exception as e:
        print(f"Warning: Azure VLM layout feedback generation failed: {e}")
        return ""


# Import rendering and asset retrieval related modules
render_scene_with_bpy = None
AssetRetrievalModule = None

try:
    import sys
    # Add utils path
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    
    # Import smart Blender wrapper (switched to external Blender process rendering to isolate bpy)
    from blender_wrapper import render_scene_blender_external as render_scene_with_bpy
    print("Successfully imported Blender external-process rendering wrapper")
    # No longer patching bpy.addon_utils in main process; external process runs in its own Python environment
    
except Exception as e:
    print(f"Warning: Could not import Blender rendering wrapper: {e}")
    
    # Fallback rendering function
    def render_scene_with_bpy(scene_data, output_dir):
        """Simplest fallback rendering function"""
        print(f"Using simple fallback rendering for scene in {output_dir}")
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create simple placeholder files
        (output_path / 'top').mkdir(exist_ok=True)
        (output_path / 'diag').mkdir(exist_ok=True)
        
        # Create empty image files as placeholders
        for view in ['top', 'diag']:
            placeholder_file = output_path / view / 'frame.png'
            with open(placeholder_file, 'w') as f:
                f.write("placeholder")
        
        return str(output_path)

try:
    # Use advanced asset retrieval module from sample.py
    from utils.sample import AssetRetrievalModule
    # Initialization parameters match the test code in sample.py
    asset_retrieval_module = AssetRetrievalModule(
        lambd=0.7, 
        sigma=0.05, 
        temp=0.1, 
        top_p=0.95, 
        top_k=20, 
        asset_size_threshold=0.5, 
        rand_seed=42, 
        do_print=True
    )
    print("Successfully imported advanced asset retrieval module from sample.py")
except Exception as e:
    print(f"Warning: Could not import advanced asset retrieval module: {e}")
    print("  Will use placeholder asset IDs instead")
    asset_retrieval_module = None

# Import Objaverse retrieval module
objaverse_retrieval_module = None
try:
    from utils.objaverse_retriever import ObjaverseRetriever
    print("Successfully imported Objaverse retrieval module")
except Exception as e:
    print(f"Warning: Could not import Objaverse retrieval module: {e}")
    ObjaverseRetriever = None

# Import 3D visualization module
try:
    from utils.visualization_3d import render_with_visualization
    print("Successfully imported 3D visualization module")
except Exception as e:
    print(f"Warning: Could not import 3D visualization module: {e}")
    render_with_visualization = None

# Import scene editor
try:
    from utils.scene_editor import apply_tool_calls
    print("Successfully imported scene_editor module")
except Exception as e:
    print(f"Warning: Could not import scene_editor module: {e}")
    apply_tool_calls = None

# Import format conversion functions and physics evaluation
try:
    from utils.RL_utils import convert_flat_to_grouped, convert_grouped_to_flat, TrimeshPhysicsMetrics, generate_physics_feedback
    print("Successfully imported format conversion functions and physics utils from RL_utils")
except Exception as e:
    print(f"Warning: Could not import format conversion functions: {e}")
except Exception as e:
    print(f"Warning: Could not import format conversion functions: {e}")
    # If import fails, define fallback versions
    def convert_flat_to_grouped(scene):
        if 'groups' in scene:
            return scene
        if 'objects' not in scene:
            return scene
        grouped_scene = {
            'room_type': scene.get('room_type', 'unknown'),
            'room_id': scene.get('room_id', 'room_001'),
        }
        if 'room_envelope' in scene:
            grouped_scene['room_envelope'] = scene['room_envelope']
        elif 'bounds_top' in scene and 'bounds_bottom' in scene:
            grouped_scene['room_envelope'] = {
                'bounds_top': scene['bounds_top'],
                'bounds_bottom': scene['bounds_bottom']
            }
        grouped_scene['groups'] = [
            {
                'group_name': 'main_group',
                'group_type': 'functional_area',
                'description': 'Main functional area containing all objects',
                'objects': scene['objects']
            }
        ]
        return grouped_scene
    
    def convert_grouped_to_flat(scene):
        if 'groups' not in scene:
            return scene
        flat_scene = {
            'room_type': scene.get('room_type', 'unknown'),
            'room_id': scene.get('room_id', 'room_001'),
        }
        if 'room_envelope' in scene:
            flat_scene['bounds_top'] = scene['room_envelope'].get('bounds_top', [])
            flat_scene['bounds_bottom'] = scene['room_envelope'].get('bounds_bottom', [])
        all_objects = []
        for group in scene.get('groups', []):
            all_objects.extend(group.get('objects', []))
        flat_scene['objects'] = all_objects
        return flat_scene




def read_initial_scene_json(json_file_path):
    """Read JSON file and return formatted string"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        return json.dumps(scene_data, indent=2)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return "{}"

def extract_tool_calls_from_response(response_text):
    """Extract <tool_calls> content from model response"""
    pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            tool_calls_json = match.group(1).strip()
            tool_calls = json.loads(tool_calls_json)
            return tool_calls
        except json.JSONDecodeError as e:
            print(f"Error parsing tool_calls JSON: {e}")
            return None
    else:
        print("No <tool_calls> found in response")
        return None

def extract_create_scene_from_response(response_text):
    """Extract <create_scene> content from model response
    
    Supports two formats:
    1. Grouped format (with room_envelope and groups fields)
    2. Flat format (with bounds_top/bounds_bottom and objects fields)
    """
    pattern = r'<create_scene>\s*```json\s*(.*?)\s*```\s*</create_scene>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            scene_json_str = match.group(1).strip()
            scene_data = json.loads(scene_json_str)
            
            # Validate basic fields
            if 'room_type' in scene_data and 'room_id' in scene_data:
                # Both formats are valid
                if 'groups' in scene_data or 'objects' in scene_data:
                    format_type = "grouped" if 'groups' in scene_data else "flat"
                    print(f"Successfully extracted scene in {format_type} format")
                    return scene_data
            
            print("Warning: Scene data missing required fields (room_type, room_id, and groups/objects)")
            return scene_data  # Return but emit warning
        except json.JSONDecodeError as e:
            print(f"Error parsing create_scene JSON: {e}")
            return None
    else:
        print("No <create_scene> found in response")
        return None

def extract_conclusion_from_response(response_text):
    """Extract <conclusion> content from model response"""
    pattern = r'<conclusion>\s*(.*?)\s*</conclusion>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        conclusion = match.group(1).strip()
        return conclusion
    else:
        print("No <conclusion> found in response")
        return None

def extract_final_scene_from_response(response_text):
    """Extract <final_scene> content from model response (kept as fallback)"""
    pattern = r'<final_scene>\s*```json\s*(.*?)\s*```\s*</final_scene>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            scene_json = match.group(1).strip()
            scene_data = json.loads(scene_json)
            return scene_data
        except json.JSONDecodeError as e:
            print(f"Error parsing final_scene JSON: {e}")
            return None
    else:
        print("No <final_scene> found in response")
        return None


def smart_truncate_conversation_history(
    conversation_history: list,
    current_user_message: str,
    current_image_path: str,
    engine,
    max_model_len: int = 40960,
    max_tokens: int = 16384,
    system_prompt: str = None
) -> list:
    """
    Smart truncation of conversation history to ensure enough remaining token space for response generation.
    
    When max_model_len - num_tokens < max_tokens, removes turns from the beginning
    until there is enough space for generation.
    
    Args:
        conversation_history: Complete conversation history list [(user_msg, assistant_msg), ...]
        current_user_message: Current turn's user message
        current_image_path: Current image path
        engine: VllmEngine instance (for getting tokenizer)
        max_model_len: Maximum model context length
        max_tokens: Maximum tokens to reserve for generation
        system_prompt: System prompt (optional)
    
    Returns:
        Truncated conversation history list
    """
    if not conversation_history:
        return []
    
    # Get tokenizer
    tokenizer = engine.tokenizer
    
    def estimate_tokens(messages: list, image_path: str = None) -> int:
        """Estimate token count for a list of messages"""
        total_text = ""
        
        # Add system prompt
        if system_prompt:
            total_text += system_prompt + "\n"
        
        # Add all message text
        for msg in messages:
            content = msg.get('content', '')
            total_text += content + "\n"
        
        # Encode text using tokenizer
        try:
            tokens = tokenizer.encode(total_text, add_special_tokens=True)
            text_tokens = len(tokens)
        except Exception as e:
            # If encoding fails, use rough estimate (about 1.5 tokens per character)
            text_tokens = int(len(total_text) * 1.5)
        
        # Image token estimation (Qwen2.5-VL uses about 1280 tokens per image)
        image_tokens = 1280 if image_path else 0
        
        return text_tokens + image_tokens
    
    # Build complete message list for estimation
    def build_messages(history: list) -> list:
        messages = []
        for hist_user_msg, hist_assistant_msg in history:
            messages.append({'role': 'user', 'content': hist_user_msg})
            messages.append({'role': 'assistant', 'content': hist_assistant_msg})
        messages.append({'role': 'user', 'content': current_user_message})
        return messages
    
    truncated_history = list(conversation_history)
    
    # Progressively remove oldest conversation turns until there is enough space
    while truncated_history:
        messages = build_messages(truncated_history)
        num_tokens = estimate_tokens(messages, current_image_path)
        remaining_tokens = max_model_len - num_tokens
        
        if remaining_tokens >= max_tokens:
            # Enough space
            if len(truncated_history) < len(conversation_history):
                removed_count = len(conversation_history) - len(truncated_history)
                print(f"⚠ Smart truncation: removed {removed_count} oldest turns "
                      f"(tokens: {num_tokens}, remaining: {remaining_tokens}, required: {max_tokens})")
            break
        else:
            # Not enough space, remove the oldest conversation turn
            if len(truncated_history) > 0:
                truncated_history.pop(0)
            else:
                # No more history to delete
                print(f"⚠ Warning: Even without history, tokens ({num_tokens}) may exceed limit. "
                      f"Remaining: {remaining_tokens}, required: {max_tokens}")
                break
    
    # If all history is deleted and still not enough, print warning
    if not truncated_history and conversation_history:
        messages = build_messages([])
        num_tokens = estimate_tokens(messages, current_image_path)
        remaining_tokens = max_model_len - num_tokens
        if remaining_tokens < max_tokens:
            print(f"⚠ Warning: max_model_len({max_model_len}) - num_tokens({num_tokens}) = {remaining_tokens} < max_tokens({max_tokens})")
    
    return truncated_history

def generate_empty_room_with_model(room_prompt: str, engine, request_config, output_path: str = None) -> tuple[Dict[str, Any], tuple[str, str]]:
    """Generate empty room structure using fine-tuned model
    
    Returns:
        tuple: (room_data, (user_message, assistant_message)) or (None, None) if failed
    """
    
    print(f"Generating empty room with fine-tuned model...")
    print(f"Room prompt: {room_prompt}")
    
    # Build messages - only user's text requirement, no images
    user_message = room_prompt
    messages = [
        {
            'role': 'user',
            'content': user_message
        }
    ]
    
    # Create inference request (no images needed)
    infer_requests = [
        InferRequest(messages=messages, images=None),
    ]
    
    try:
        # Execute inference
        print("Requesting model to generate initial scene structure...")
        resp_list = engine.infer(infer_requests, request_config)
        response = resp_list[0].choices[0].message.content
        
        print(f"Model response length: {len(response)} characters")
        
        # Save model response (for debugging)
        if output_path:
            response_file = Path(output_path).parent / "initial_scene_generation_response.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Model response saved to: {response_file}")
        
        # Extract <create_scene> content
        room_data = extract_create_scene_from_response(response)
        
        if room_data is None:
            print("Failed to extract scene data from model response")
            return None, None
        
        # Validate the generated room structure
        has_groups_or_objects = 'groups' in room_data or 'objects' in room_data
        if not has_groups_or_objects:
            print(f"Warning: Generated room has neither 'groups' nor 'objects' field")
            return None, None
        
        print(f"✓ Successfully generated room structure:")
        print(f"  Room type: {room_data.get('room_type', 'N/A')}")
        if 'groups' in room_data:
            print(f"  Number of groups: {len(room_data.get('groups', []))}")
        elif 'objects' in room_data:
            print(f"  Number of objects: {len(room_data.get('objects', []))}")
        
        # **Important**: Convert to flat format before saving
        room_data_to_save = convert_grouped_to_flat(room_data) if 'groups' in room_data else room_data
        
        # Save to file (if output path specified)
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(room_data_to_save, f, indent=2, ensure_ascii=False)
            print(f"Empty room saved to: {output_file} (flat format)")
        
        # Return flat format scene data and conversation history
        return room_data_to_save, (user_message, response)
        
    except Exception as e:
        print(f"Error generating room with model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_empty_room(room_prompt: str, output_path: str = None) -> Dict[str, Any]:
    """Generate empty room structure using Azure OpenAI (room boundaries only, no objects)"""
    
    # Set up Azure OpenAI client
    try:
        client = setup_azure_client()
    except Exception as e:
        print(f"Failed to setup Azure OpenAI client: {e}")
        return None
    
    # Build system prompt
    system_prompt = """You are an expert interior designer and room layout specialist. Your task is to create an empty room structure based on user requirements.

You must respond with a JSON structure that includes:
1. room_type: The type of room (bedroom, living_room, kitchen, etc.)
2. room_id: A unique identifier for the room
3. bounds_top and bounds_bottom: The physical boundaries of the room
4. objects: An empty array (objects will be added later)

Guidelines:
- Room dimensions should be realistic for the specified room type
- bounds_top and bounds_bottom define the room shape using 4-8 corner points
- For rectangular rooms: use 4 corner points
- For L-shaped rooms: use 6 corner points
- For T-shaped or more complex rooms: use up to 8 corner points
- bounds_bottom has y=0.0 (floor level), bounds_top has y=2.6 (ceiling level)
- Points MUST be ordered clockwise or counter-clockwise around the room perimeter
- The polygon formed by the points must be simple (non-self-intersecting)

ROOM SIZE REQUIREMENTS (STRICTLY FOLLOW):
- bedroom: 10-20 square meters (e.g., 3.5m x 4m = 14 sqm, or 4m x 5m = 20 sqm)
- livingroom/living_room: 15-35 square meters (e.g., 4m x 5m = 20 sqm, or 5m x 6m = 30 sqm)
- diningroom/dining_room: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- studyroom/study_room/office: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- Calculate floor area as the polygon area formed by the corner points

Example 1 - Rectangular room (4 points):
{
  "room_type": "bedroom",
  "room_id": "Bedroom-1234",
  "bounds_top": [
    [-2.35, 2.6, 2.05],
    [2.35, 2.6, 2.05], 
    [2.35, 2.6, -2.05],
    [-2.35, 2.6, -2.05]
  ],
  "bounds_bottom": [
    [-2.35, 0.0, 2.05],
    [2.35, 0.0, 2.05],
    [2.35, 0.0, -2.05], 
    [-2.35, 0.0, -2.05]
  ],
  "objects": []
}

Example 2 - L-shaped room (6 points, clockwise from top-left):
{
  "room_type": "livingroom",
  "room_id": "LivingRoom-5678",
  "bounds_top": [
    [-3.0, 2.6, 2.5],
    [1.0, 2.6, 2.5],
    [1.0, 2.6, 0.0],
    [3.0, 2.6, 0.0],
    [3.0, 2.6, -2.5],
    [-3.0, 2.6, -2.5]
  ],
  "bounds_bottom": [
    [-3.0, 0.0, 2.5],
    [1.0, 0.0, 2.5],
    [1.0, 0.0, 0.0],
    [3.0, 0.0, 0.0],
    [3.0, 0.0, -2.5],
    [-3.0, 0.0, -2.5]
  ],
  "objects": []
}

Respond ONLY with valid JSON, no additional text."""

    # Build user prompt
    user_prompt = f"""Create an empty room structure for: {room_prompt}

Please generate a room that would be suitable for this description. Keep the objects list empty."""

    try:
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_completion_tokens=2000
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        print(f"Generated room response length: {len(response_content)} characters")
        
        # Try to parse JSON response
        try:
            # If response contains ```json markers, extract the JSON part
            if "```json" in response_content:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            room_data = json.loads(response_content)
            
            # Validate generated room structure - check for bounds
            if 'bounds_top' not in room_data or 'bounds_bottom' not in room_data:
                print(f"Warning: Generated room missing bounds_top or bounds_bottom")
                return None
            
            # Ensure objects field exists
            if 'objects' not in room_data:
                room_data['objects'] = []
            
            # **Important**: Convert to flat format before saving
            room_data_to_save = convert_grouped_to_flat(room_data) if 'groups' in room_data else room_data
            
            # Save to file (if output path specified)
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(room_data_to_save, f, indent=2, ensure_ascii=False)
                print(f"Empty room saved to: {output_file} (flat format)")
            
            return room_data_to_save
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response content: {response_content[:500]}...")
            return None
            
    except Exception as e:
        print(f"Error calling Azure OpenAI API: {e}")
        return None


def generate_scene_with_objects(room_prompt: str, output_path: str = None, 
                                 asset_retrieval_module=None, asset_source='3d-future',
                                 objaverse_retriever=None) -> Dict[str, Any]:
    """Generate a complete scene with objects using Azure OpenAI (flat format)
    
    Args:
        room_prompt: User's room requirement description
        output_path: Output file path (optional)
        asset_retrieval_module: 3D-FUTURE asset retrieval module
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retrieval module
        
    Returns:
        Dict: Complete scene data with objects (flat format), assets already retrieved
    """
    
    # Set up Azure OpenAI client
    try:
        client = setup_azure_client()
    except Exception as e:
        print(f"Failed to setup Azure OpenAI client: {e}")
        return None
    
    # Build system prompt - generate complete scene with objects
    system_prompt = """You are an expert interior designer and room layout specialist. Your task is to create a complete room layout with furniture based on user requirements.

You must respond with a JSON structure in FLAT FORMAT that includes:
1. room_type: The type of room (bedroom, livingroom, diningroom, studyroom, etc.)
2. room_id: A unique identifier for the room (e.g., "Bedroom-1234")
3. bounds_top: 4-8 corner points of the ceiling (4 for rectangular, 6 for L-shaped, up to 8 for complex shapes)
4. bounds_bottom: Same corner points as bounds_top but with y=0.0
5. objects: An array of furniture objects, each with:
   - desc: Detailed description of the object (style, material, color, design features)
   - size: [width_x, height_y, depth_z] in meters (estimated realistic size)
   - pos: [x, y, z] position in room coordinates (y=0.0 for floor-standing objects)
   - rot: [x, y, z, w] quaternion rotation (use [0,0,0,1] for no rotation, [0,0.70711,0,0.70711] for 90° Y-axis rotation)

ROOM SHAPE GUIDELINES:
- For rectangular rooms: use 4 corner points
- For L-shaped rooms: use 6 corner points
- For T-shaped or more complex rooms: use up to 8 corner points
- Points MUST be ordered clockwise or counter-clockwise around the room perimeter
- The polygon formed by the points must be simple (non-self-intersecting)

ROOM SIZE REQUIREMENTS (STRICTLY FOLLOW):
- bedroom: 10-20 square meters (e.g., 3.5m x 4m = 14 sqm, or 4m x 5m = 20 sqm)
- livingroom: 15-35 square meters (e.g., 4m x 5m = 20 sqm, or 5m x 6m = 30 sqm)
- diningroom: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- studyroom/office: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- Calculate floor area as the polygon area formed by the corner points

IMPORTANT GUIDELINES:
- bounds_bottom has y=0.0 (floor level), bounds_top has y=2.6 (ceiling level)
- Place objects logically: beds against walls, sofas facing TV, desks near windows
- Avoid overlapping objects - leave adequate spacing between furniture
- Objects should be within room bounds (inside the polygon formed by corner points)
- For L-shaped rooms, utilize both sections of the room effectively
- **ONLY include floor-standing furniture** (objects placed directly on the floor)
- **DO NOT include**: wall-mounted items (paintings, mirrors on walls), tabletop items (decorations, lamps on tables), ceiling fixtures, rugs, carpets, or floor mats
- For all objects, pos[1] (y) should be 0.0 since they are all floor-standing
- Include 5-10 appropriate floor-standing objects based on room type

ROTATION GUIDE:
- [0, 0, 0, 1]: No rotation (facing +Z direction)
- [0, 0.70711, 0, 0.70711]: 90° clockwise around Y-axis (facing +X)
- [0, 1, 0, 0]: 180° around Y-axis (facing -Z)
- [0, -0.70711, 0, 0.70711]: 90° counter-clockwise around Y-axis (facing -X)

OBJECT DESCRIPTION GUIDE:
Write detailed descriptions that help identify the exact furniture piece:
- Include style: modern, minimalist, traditional, industrial, scandinavian, etc.
- Include material: wood, leather, fabric, metal, glass, etc.
- Include color: specific colors like "walnut brown", "charcoal gray", "cream white"
- Include design features: "with tapered legs", "open shelving", "curved backrest"

Example 1 - Rectangular room with objects:
{
  "room_type": "bedroom",
  "room_id": "Bedroom-1234",
  "bounds_top": [[-2.5, 2.6, 2.0], [2.5, 2.6, 2.0], [2.5, 2.6, -2.0], [-2.5, 2.6, -2.0]],
  "bounds_bottom": [[-2.5, 0.0, 2.0], [2.5, 0.0, 2.0], [2.5, 0.0, -2.0], [-2.5, 0.0, -2.0]],
  "objects": [
    {"desc": "Modern queen-sized bed with gray fabric headboard", "size": [1.6, 0.5, 2.0], "pos": [-1.5, 0.0, 0.0], "rot": [0, 0.70711, 0, 0.70711]}
  ]
}

Example 2 - L-shaped room with objects (6 points, clockwise):
{
  "room_type": "livingroom",
  "room_id": "LivingRoom-5678",
  "bounds_top": [
    [-3.0, 2.6, 2.5], [1.0, 2.6, 2.5], [1.0, 2.6, 0.0],
    [3.0, 2.6, 0.0], [3.0, 2.6, -2.5], [-3.0, 2.6, -2.5]
  ],
  "bounds_bottom": [
    [-3.0, 0.0, 2.5], [1.0, 0.0, 2.5], [1.0, 0.0, 0.0],
    [3.0, 0.0, 0.0], [3.0, 0.0, -2.5], [-3.0, 0.0, -2.5]
  ],
  "objects": [
    {"desc": "Modern L-shaped sectional sofa in charcoal gray", "size": [2.8, 0.85, 1.6], "pos": [-1.5, 0.0, 1.5], "rot": [0, 0, 0, 1]},
    {"desc": "Minimalist TV stand in walnut wood", "size": [1.8, 0.5, 0.4], "pos": [2.0, 0.0, -1.5], "rot": [0, 0.70711, 0, 0.70711]}
  ]
}

Respond ONLY with valid JSON, no additional text."""

    # Build user prompt
    user_prompt = f"""Create a complete furnished room for: {room_prompt}

Please generate a realistic room layout with appropriate furniture. Include detailed descriptions for each piece of furniture to enable accurate asset retrieval."""

    try:
        print(f"Generating scene with GPT for: {room_prompt}")
        
        # Call Azure OpenAI API
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_completion_tokens=4000
        )
        
        # Extract response content
        response_content = response.choices[0].message.content.strip()
        print(f"Generated scene response length: {len(response_content)} characters")
        
        # Try to parse JSON response
        try:
            # If response contains ```json markers, extract the JSON part
            if "```json" in response_content:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            scene_data = json.loads(response_content)
            
            # Validate generated scene structure
            if 'bounds_top' not in scene_data or 'bounds_bottom' not in scene_data:
                print(f"Warning: Generated scene missing bounds_top or bounds_bottom")
                return None
            
            if 'objects' not in scene_data or not scene_data['objects']:
                print(f"Warning: Generated scene has no objects")
                return None
            
            # Convert to flat format (if needed)
            if 'groups' in scene_data:
                scene_data = convert_grouped_to_flat(scene_data)
            
            print(f"GPT generated scene with {len(scene_data.get('objects', []))} objects")
            
            # Perform asset retrieval
            print("Starting asset retrieval for generated objects...")
            scene_data = retrieve_and_update_assets(
                scene_data, 
                asset_retrieval_module=asset_retrieval_module,
                asset_source=asset_source,
                objaverse_retriever=objaverse_retriever
            )
            
            # Save to file (if output path specified)
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(scene_data, f, indent=2, ensure_ascii=False)
                print(f"Complete scene saved to: {output_file}")
            
            return scene_data
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Response content: {response_content[:500]}...")
            return None
            
    except Exception as e:
        print(f"Error calling Azure OpenAI API: {e}")
        import traceback
        traceback.print_exc()
        return None


def retrieve_and_update_assets(scene_data, asset_retrieval_module=None, 
                                asset_source='3d-future', objaverse_retriever=None) -> Dict[str, Any]:
    """Retrieve asset IDs for objects in the scene and replace original size with retrieved_size
    
    Args:
        scene_data: Scene data (flat format)
        asset_retrieval_module: 3D-FUTURE asset retrieval module
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retrieval module
        
    Returns:
        Dict: Updated scene data with uid and updated size
    """
    if not scene_data or 'objects' not in scene_data:
        return scene_data
    
    print(f"Retrieving assets for {len(scene_data['objects'])} objects (source: {asset_source})...")
    
    updated_scene = None
    
    # Select retrieval strategy based on asset source
    if asset_source == 'objaverse' and objaverse_retriever:
        try:
            updated_scene = objaverse_retriever.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Asset retrieval completed using Objaverse")
        except Exception as e:
            print(f"Error during Objaverse asset retrieval: {e}")
            print("Falling back to 3D-FUTURE retrieval...")
    
    elif asset_source == 'auto' and objaverse_retriever and asset_retrieval_module:
        # Hybrid mode
        try:
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            # Check for objects that failed retrieval
            needs_objaverse = False
            for obj in updated_scene.get('objects', []):
                if not obj.get('jid') or obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>']:
                    needs_objaverse = True
                    break
            if needs_objaverse:
                print("Some assets not found in 3D-FUTURE, trying Objaverse...")
                updated_scene = objaverse_retriever.sample_all_assets(updated_scene, is_greedy_sampling=True)
            print("Asset retrieval completed using hybrid mode")
        except Exception as e:
            print(f"Error during auto asset retrieval: {e}")
    
    elif asset_source == '3d-future' and asset_retrieval_module:
        try:
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Asset retrieval completed using 3D-FUTURE")
        except Exception as e:
            print(f"Error during 3D-FUTURE asset retrieval: {e}")
    
    if updated_scene is None:
        print("Warning: Asset retrieval failed, returning original scene")
        return scene_data
    
    # Replace original size with retrieved_size
    objects_updated = 0
    for obj in updated_scene.get('objects', []):
        if 'retrieved_size' in obj and obj['retrieved_size']:
            original_size = obj.get('size', [])
            obj['size'] = obj['retrieved_size']
            objects_updated += 1
            if objects_updated <= 3:  # Only print the first 3 as examples
                print(f"  Updated size for '{obj.get('desc', 'Unknown')[:40]}...': {original_size} -> {obj['size']}")
    
    if objects_updated > 3:
        print(f"  ... and {objects_updated - 3} more objects updated")
    
    print(f"Updated sizes for {objects_updated}/{len(updated_scene.get('objects', []))} objects")
    
    return updated_scene

def apply_tool_calls_to_scene(initial_scene, tool_calls):
    """Use scene_editor's apply_tool_calls function to modify the scene
    
    This function automatically handles format conversion:
    1. If input is in flat format (without groups), first convert to grouped format
    2. Apply tool_calls to modify the scene  
    3. **Always return flat format** (without groups)
    """
    if not apply_tool_calls:
        print("Warning: scene_editor not available, using fallback")
        return initial_scene
    
    try:
        # Detect input format
        is_flat_format = 'objects' in initial_scene and 'groups' not in initial_scene
        
        # If in flat format (without groups), first convert to grouped format
        if is_flat_format:
            print("Detected flat format (without groups), converting to grouped format for editing...")
            scene_for_editing = convert_flat_to_grouped(initial_scene)
        else:
            scene_for_editing = initial_scene
        
        # Use scene_editor module's apply_tool_calls function
        edited_scene = apply_tool_calls(scene_for_editing, tool_calls)
        
        # **Key**: Always convert back to flat format before returning
        final_scene = convert_grouped_to_flat(edited_scene) if 'groups' in edited_scene else edited_scene
        print("Returning scene in flat format (without groups)")
        
        return final_scene
    except Exception as e:
        print(f"Error applying tool calls: {e}")
        import traceback
        traceback.print_exc()
        return initial_scene

def check_and_retrieve_assets(scene_data, asset_retrieval_module=None, asset_source='3d-future', objaverse_retriever=None):
    """Check if the scene has assets that need retrieval, and perform retrieval
    
    Supports two formats:
    1. Grouped format (groups -> objects)
    2. Flat format (direct objects array)
    
    Parameters:
    - scene_data: Scene data
    - asset_retrieval_module: 3D-FUTURE asset retrieval module
    - asset_source: Asset source ('3d-future', 'objaverse', 'auto')
    - objaverse_retriever: Objaverse asset retrieval module
    """
    if not scene_data:
        return scene_data
    
    print(f"Checking for assets that need retrieval (source: {asset_source})...")
    
    # Select retrieval strategy based on asset source
    if asset_source == 'objaverse' and objaverse_retriever:
        try:
            # Use Objaverse retrieval module
            updated_scene = objaverse_retriever.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Assets retrieval completed using Objaverse")
            return updated_scene
        except Exception as e:
            print(f"Error during Objaverse asset retrieval: {e}")
            # Fall back to 3D-FUTURE
            print("Falling back to 3D-FUTURE retrieval...")
    
    elif asset_source == 'auto' and objaverse_retriever and asset_retrieval_module:
        # Hybrid mode: try 3D-FUTURE, fall back to Objaverse
        try:
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            
            # Check for objects that failed retrieval (still have <NEED_RETRIEVAL>)
            needs_objaverse = False
            objects_list = []
            if 'groups' in updated_scene:
                for group in updated_scene.get('groups', []):
                    objects_list.extend(group.get('objects', []))
            elif 'objects' in updated_scene:
                objects_list = updated_scene.get('objects', [])
            
            for obj in objects_list:
                if obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>'] or not obj.get('jid'):
                    needs_objaverse = True
                    break
            
            if needs_objaverse:
                print("Some assets not found in 3D-FUTURE, trying Objaverse...")
                updated_scene = objaverse_retriever.sample_all_assets(updated_scene, is_greedy_sampling=True)
            
            print("Assets retrieval completed using hybrid mode (3D-FUTURE + Objaverse)")
            return updated_scene
        except Exception as e:
            print(f"Error during auto asset retrieval: {e}")
    
    # Default: use 3D-FUTURE retrieval module
    if asset_retrieval_module:
        try:
            # Use sample_all_assets method, consistent with the logic in respace.py
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Assets retrieval completed using sample_all_assets (3D-FUTURE)")
            return updated_scene
        except Exception as e:
            print(f"Error during asset retrieval: {e}")
            # Continue to fallback logic
    
    # Fallback logic: manually handle <NEED_RETRIEVAL>
    print("Using fallback asset retrieval logic...")
    modified = False
    
    # Handle grouped format
    if 'groups' in scene_data:
        for group in scene_data.get('groups', []):
            for obj in group.get('objects', []):
                if obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>'] or not obj.get('jid'):
                    print(f"Need to retrieve asset for object: {obj.get('desc', 'Unknown')}")
                    # Generate a placeholder jid
                    obj['jid'] = str(uuid.uuid4())
                    print(f"Using placeholder jid: {obj['jid']}")
                    modified = True
    
    # Handle flat format (direct objects array)
    elif 'objects' in scene_data:
        for obj in scene_data.get('objects', []):
            if obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>'] or not obj.get('jid'):
                print(f"Need to retrieve asset for object: {obj.get('desc', 'Unknown')}")
                # Generate a placeholder jid
                obj['jid'] = str(uuid.uuid4())
                print(f"Using placeholder jid: {obj['jid']}")
                modified = True
    
    if modified:
        print("Fallback assets retrieval completed")
    else:
        print("No assets needed retrieval")
    
    return scene_data


def _predownload_objaverse_glbs(scene_data):
    """Pre-download Objaverse GLB files before rendering
    
    This way the Blender subprocess only needs to read from local cache, without needing the objaverse package installed
    """
    try:
        from utils.objaverse_glb_manager import get_objaverse_glb_path
    except ImportError:
        print("Warning: objaverse_glb_manager not available, skipping GLB pre-download")
        return
    
    # Extract all Objaverse UIDs that need to be downloaded
    objects_list = []
    if 'groups' in scene_data:
        for group in scene_data.get('groups', []):
            objects_list.extend(group.get('objects', []))
    elif 'objects' in scene_data:
        objects_list = scene_data.get('objects', [])
    
    uids_to_download = []
    for obj in objects_list:
        if obj.get('asset_source') == 'objaverse' and obj.get('uid'):
            uids_to_download.append(obj['uid'])
    
    if not uids_to_download:
        return
    
    print(f"Pre-downloading {len(uids_to_download)} Objaverse GLB files...")
    
    downloaded = 0
    for uid in uids_to_download:
        try:
            glb_path = get_objaverse_glb_path(uid, download_if_missing=True)
            if glb_path and os.path.exists(glb_path):
                downloaded += 1
                print(f"  ✓ Downloaded: {uid[:16]}... ({os.path.getsize(glb_path) / 1024:.1f} KB)")
            else:
                print(f"  ✗ Failed to download: {uid[:16]}...")
        except Exception as e:
            print(f"  ✗ Error downloading {uid[:16]}...: {e}")
    
    print(f"Pre-download complete: {downloaded}/{len(uids_to_download)} files")


def render_scene_to_image(scene_data, output_dir, iteration, enable_visualization=False):
    """Render scene using Blender and return merged image path
    
    Args:
        scene_data: Scene data
        output_dir: Output directory
        iteration: Iteration number
        enable_visualization: Whether to enable 3D visualization guide lines
    """
    try:
        print(f"Rendering scene for iteration {iteration}...")
        
        # Pre-download Objaverse GLB files (download in main process to avoid Blender subprocess needing the objaverse package)
        _predownload_objaverse_glbs(scene_data)
        
        # Create temporary output directory
        temp_output_dir = Path(output_dir) / f"temp_render_{iteration}"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables to control Blender output
        os.environ['BPY_VERBOSE'] = '0'  # Reduce output
        # Don't force placeholder usage, let Blender renderer try to load real 3D models
        os.environ['BPY_USE_PLACEHOLDER_ONLY'] = '0'
        # Set 3D visualization environment variables
        os.environ['BPY_ENABLE_VISUALIZATION'] = '1' if enable_visualization else '0'
        
        # Use Blender rendering wrapper
        if render_scene_with_bpy:
            scene_id = f"scene_iter_{iteration}"
            render_result = render_scene_with_bpy(scene_data, temp_output_dir, scene_id)
            print(f"Blender rendering completed: {render_result}")
        else:
            print("No rendering function available, creating placeholder")
            
        # Find generated image files
        top_file = temp_output_dir / "top" / "frame.png"
        diag_file = temp_output_dir / "diag" / "frame.png"
        
        if top_file.exists() and diag_file.exists():
            try:
                # Use the image merge function with bounding boxes and labels
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "image_merger", 
                    str(Path(__file__).parent / 'utils' / 'image_merger.py')
                )
                image_merger = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(image_merger)
                merge_rendered_views_with_annotations = image_merger.merge_rendered_views_with_annotations
                
                # Save merged image
                merged_path = Path(output_dir) / f"merged_iter_{iteration}.png"
                merge_rendered_views_with_annotations(str(top_file), str(diag_file), str(merged_path))
                
                print(f"Rendered and merged image saved to: {merged_path}")
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_output_dir)
                except:
                    pass
                
                return str(merged_path)
                
            except Exception as img_error:
                print(f"Error processing rendered images: {img_error}")
                import traceback
                traceback.print_exc()
                # Continue to creating placeholder image
        
        # If rendering failed or image processing failed, create placeholder image
        print("Creating placeholder image...")
        try:
            from PIL import Image, ImageDraw
            
            # Create placeholder image
            img = Image.new('RGB', (1024, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # Count objects
            objects_count = 0
            if 'groups' in scene_data:
                objects_count = sum(len(group.get('objects', [])) for group in scene_data.get('groups', []))
            elif 'objects' in scene_data:
                objects_count = len(scene_data.get('objects', []))
            
            # Draw information
            text = f"Iteration {iteration}\n{objects_count} objects/groups\nPlaceholder Image"
            draw.text((50, 200), text, fill='black')
            
            # Save placeholder image
            placeholder_path = Path(output_dir) / f"placeholder_iter_{iteration}.png"
            img.save(placeholder_path)
            
            print(f"Created placeholder image: {placeholder_path}")
            return str(placeholder_path)
            
        except Exception as placeholder_error:
            print(f"Error creating placeholder image: {placeholder_error}")
            return './test/init.png'
            
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
        return './test/init.png'

def create_iteration_summary_image(output_dir, all_image_paths, num_iterations):
    """Composite all iteration images into one large image"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        print("Creating iteration summary image...")
        
        # Filter out existing image files
        valid_images = []
        for i, img_path in enumerate(all_image_paths):
            if Path(img_path).exists():
                valid_images.append((i, img_path))
            else:
                print(f"Warning: Image {img_path} does not exist")
        
        if not valid_images:
            print("No valid images found for summary")
            return None
        
        # Load first image to get dimensions
        first_img = Image.open(valid_images[0][1])
        img_width, img_height = first_img.size
        
        # Calculate grid layout - as close to square as possible
        import math
        cols = math.ceil(math.sqrt(len(valid_images)))
        rows = math.ceil(len(valid_images) / cols)
        
        # Calculate composite image dimensions
        margin = 20
        label_height = 40
        cell_width = img_width + margin
        cell_height = img_height + label_height + margin
        
        total_width = cols * cell_width + margin
        total_height = rows * cell_height + margin + 60  # Extra space for title
        
        # Create large image - use RGBA mode to support alpha channel, fully transparent background
        summary_img = Image.new('RGBA', (total_width, total_height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(summary_img)
        
        # Try to load font
        try:
            # Try to use system font
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            try:
                # Fallback font
                font_title = ImageFont.load_default()
                font_label = ImageFont.load_default()
            except:
                font_title = None
                font_label = None
        
        # Draw title
        title = f"Scene Generation Progress ({len(valid_images)} iterations)"
        if font_title:
            title_bbox = draw.textbbox((0, 0), title, font=font_title)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (total_width - title_width) // 2
        else:
            title_x = total_width // 2 - len(title) * 6
        
        draw.text((title_x, 20), title, fill=(0, 0, 0, 255), font=font_title)
        
        # Draw each iteration's image
        for idx, (iter_num, img_path) in enumerate(valid_images):
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            x = col * cell_width + margin
            y = row * cell_height + margin + 60  # 60 is the title area height
            
            # Load and paste image
            try:
                img = Image.open(img_path)
                # Convert to RGBA mode to ensure transparency support
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # Resize if image dimensions don't match
                if img.size != (img_width, img_height):
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # Use alpha composite to paste image, preserving transparency
                summary_img.paste(img, (x, y), img)
                
                # Add label
                if iter_num == 0:
                    label = "Initial Scene"
                else:
                    label = f"Iteration {iter_num}"
                
                label_y = y + img_height + 5
                if font_label:
                    label_bbox = draw.textbbox((0, 0), label, font=font_label)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_x = x + (img_width - label_width) // 2
                else:
                    label_x = x + img_width // 2 - len(label) * 4
                
                draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=font_label)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Draw placeholder
                draw.rectangle([x, y, x + img_width, y + img_height], fill=(211, 211, 211, 255), outline=(128, 128, 128, 255))
                error_text = f"Error: {iter_num}"
                draw.text((x + 10, y + img_height // 2), error_text, fill=(255, 0, 0, 255), font=font_label)
        
        # Save composite image - save as PNG format preserving alpha channel
        output_path = Path(output_dir)
        summary_path = output_path / "iteration_summary.png"
        summary_img.save(summary_path, "PNG", optimize=True)
        
        print(f"✓ Iteration summary image saved to: {summary_path}")
        return str(summary_path)
        
    except Exception as e:
        print(f"Error creating iteration summary image: {e}")
        import traceback
        traceback.print_exc()
        return None

def iterative_scene_generation(initial_scene_path, user_prompt, engine, request_config, 
                             asset_retrieval_module=None, num_iterations=10, output_dir="./output",
                             initial_conversation=None, asset_source='3d-future', objaverse_retriever=None,
                             enable_visualization=False, enable_physics_feedback=False, enable_vlm_feedback=False,
                             enable_physics_optimization=False, physics_opt_steps=5, models_path=None):
    """Execute iterative scene generation
    
    Args:
        initial_conversation: tuple (user_message, assistant_message) from initial scene generation
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retriever instance
        enable_visualization: Whether to enable 3D visualization guide lines
        enable_physics_feedback: Whether to enable physics feedback injection
        enable_vlm_feedback: Whether to enable VLM layout feedback injection
        enable_physics_optimization: Whether to enable physics optimization (collision/out-of-bounds fix)
        physics_opt_steps: Maximum physics optimization steps
        models_path: Path to 3D models
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read initial scene
    with open(initial_scene_path, 'r', encoding='utf-8') as f:
        current_scene = json.load(f)
    
    # Save all image paths for final composition
    all_image_paths = []
    
    # Render initial scene as first image instead of using hardcoded path
    print("Rendering initial scene...")
    current_image_path = render_scene_to_image(current_scene, output_path, 0, enable_visualization=enable_visualization)
    all_image_paths.append(current_image_path)
    print(f"Initial scene rendered to: {current_image_path}")
    
    print(f"Starting iterative scene generation with {num_iterations} iterations...")
    print(f"Initial prompt: {user_prompt}")
    
    # Save all generated scenes
    all_scenes = []
    
    # Save complete conversation history including user requests and model responses for each turn
    conversation_history = []
    
    # If there is an initial conversation (from scene generation), add to history
    if initial_conversation is not None:
        conversation_history.append(initial_conversation)
        print(f"Added initial scene generation conversation to history")
    
    # Initialize components needed for feedback generation
    last_feedback = ""  # Store feedback generated from the previous round
    trimesh_metrics_instance = None
    azure_client_for_feedback = None
    
    # Try to initialize physics evaluator
    try:
        trimesh_metrics_instance = TrimeshPhysicsMetrics(verbose=False)
        print("Initialized TrimeshPhysicsMetrics for feedback generation")
    except Exception as e:
        print(f"Warning: Could not initialize TrimeshPhysicsMetrics: {e}")
    
    # Try to initialize Azure client (for VLM layout feedback)
    try:
        azure_client_for_feedback = setup_azure_client()
        print("Initialized Azure client for VLM layout feedback")
    except Exception as e:
        print(f"Warning: Could not initialize Azure client for feedback: {e}")
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # Build current iteration's user content
        current_scene_json = json.dumps(current_scene, indent=2)
        
        # Build base user content (including feedback if available)
        if iteration == 0:
            # First iteration: initial request (no feedback)
            base_user_content = f'{user_prompt}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        else:
            # Subsequent turns: continue optimizing the scene, including feedback
            feedback_section = ""
            if last_feedback:
                feedback_section = f'\n<feedback>\n{last_feedback}\n</feedback>\n'
                print(f"Injecting feedback: {last_feedback[:100]}...")
            base_user_content = f'Please continue to improve the scene based on the original request: "{user_prompt}"{feedback_section}\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        
        # Build complete message list including conversation history
        messages = []
        
        # Add current turn's user message
        current_user_message = f'<image>{base_user_content}'
        
        # Use smart truncation: based on token count rather than fixed turns
        # Configuration: max_model_len=40960, max_tokens=16384
        truncated_history = smart_truncate_conversation_history(
            conversation_history=conversation_history,
            current_user_message=current_user_message,
            current_image_path=current_image_path,
            engine=engine,
            max_model_len=40960,
            max_tokens=request_config.max_tokens if request_config.max_tokens else 16384
        )
        
        # If there is conversation history, add to message list
        if truncated_history:
            # Add historical conversations
            for hist_user_msg, hist_assistant_msg in truncated_history:
                messages.append({'role': 'user', 'content': hist_user_msg})
                messages.append({'role': 'assistant', 'content': hist_assistant_msg})
        
        messages.append({'role': 'user', 'content': current_user_message})
        
        # Create inference request
        infer_requests = [
            InferRequest(messages=messages,
                        images=[current_image_path]),
        ]
        
        # Execute inference
        print("Generating response from model...")
        resp_list = engine.infer(infer_requests, request_config)
        response = resp_list[0].choices[0].message.content
        
        print(f"Response length: {len(response)} characters")
        
        # Save response
        with open(output_path / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        
        # Add current turn's conversation to history
        conversation_history.append((current_user_message, response))
        
        # First try to extract tool_calls
        tool_calls = extract_tool_calls_from_response(response)
        final_scene = None
        
        if tool_calls is not None:
            print(f"Extracted {len(tool_calls)} tool calls")
            
            # Check for terminate tool call
            has_terminate = False
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get('name') == 'terminate':
                    has_terminate = True
                    terminate_reason = tool_call.get('arguments', {}).get('reason', 'No reason provided')
                    print(f"🛑 Terminate tool detected: {terminate_reason}")
                    break
            
            # If terminate tool found, stop iterations
            if has_terminate:
                print(f"Stopping iterations early due to terminate tool call")
                # **Important**: Convert to flat format before saving
                scene_to_save = convert_grouped_to_flat(current_scene) if 'groups' in current_scene else current_scene
                
                # Apply physics optimization (collision/out-of-bounds fix) - also optimize on last round
                if enable_physics_optimization:
                    print(f"Applying final physics optimization before saving...")
                    scene_to_save, _ = apply_physics_optimization(
                        scene_to_save, 
                        models_path=models_path or "", 
                        max_steps=physics_opt_steps
                    )
                
                scene_file_path = output_path / f"scene_iter_{iteration + 1}_final.json"
                with open(scene_file_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_to_save, f, indent=2, ensure_ascii=False)
                print(f"Final scene saved to: {scene_file_path}")
                break
            
            # Use scene_editor to apply tool calls to generate final scene
            final_scene = apply_tool_calls_to_scene(current_scene, tool_calls)
            print("Applied tool calls to generate final scene")
        else:
            # If no tool_calls found, try to extract final_scene (as fallback)
            print("No tool_calls found, trying to extract final_scene as fallback")
            final_scene = extract_final_scene_from_response(response)
        
        if final_scene is None:
            print(f"⚠ No executable commands found in iteration {iteration + 1}. Ending generation and saving current state.")
            
            # Save current state as final
            scene_to_save = convert_grouped_to_flat(current_scene) if 'groups' in current_scene else current_scene
            
            # Apply physics optimization (collision/out-of-bounds fix) - also optimize on last round
            if enable_physics_optimization:
                print(f"Applying final physics optimization before saving...")
                scene_to_save, _ = apply_physics_optimization(
                    scene_to_save, 
                    models_path=models_path or "", 
                    max_steps=physics_opt_steps
                )
            
            scene_file_path = output_path / f"scene_iter_{iteration + 1}_final.json"
            with open(scene_file_path, 'w', encoding='utf-8') as f:
                json.dump(scene_to_save, f, indent=2, ensure_ascii=False)
            print(f"Final scene saved to: {scene_file_path}")
            
            break
        
        print(f"Extracted final_scene with {len(final_scene.get('groups', final_scene.get('objects', [])))} objects/groups")
        
        # Retrieve required assets
        final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module, 
                                                 asset_source=asset_source, 
                                                 objaverse_retriever=objaverse_retriever)
        
        # Apply physics optimization (collision/out-of-bounds fix)
        physics_deleted_feedback = ""
        if enable_physics_optimization:
            print(f"Applying physics optimization...")
            final_scene, physics_deleted_feedback = apply_physics_optimization(
                final_scene, 
                models_path=models_path or "", 
                max_steps=physics_opt_steps
            )
        
        # **Important**: Convert to flat format (without groups) before saving
        final_scene_to_save = convert_grouped_to_flat(final_scene) if 'groups' in final_scene else final_scene
        
        # Save current scene (flat format)
        scene_file_path = output_path / f"scene_iter_{iteration + 1}.json"
        with open(scene_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_scene_to_save, f, indent=2, ensure_ascii=False)
        
        all_scenes.append(final_scene_to_save)
        
        # Render scene as image (using flat format)
        current_image_path = render_scene_to_image(final_scene_to_save, output_path, iteration + 1, enable_visualization=enable_visualization)
        all_image_paths.append(current_image_path)
        
        # Update current scene for next iteration (using flat format)
        current_scene = final_scene_to_save
        
        # ===== Generate feedback for next round =====
        # Only generate feedback on non-last rounds
        if iteration < num_iterations - 1:
            feedback_parts = []
            
            # 0. Physics optimization deleted objects feedback (add first)
            if physics_deleted_feedback:
                feedback_parts.append(physics_deleted_feedback)
                print(f"Physics deleted feedback: {physics_deleted_feedback}")
            
            # 1. Physics feedback (from trimesh)
            if enable_physics_feedback and trimesh_metrics_instance is not None:
                try:
                    # Use grouped format for physics evaluation
                    scene_for_eval = convert_flat_to_grouped(final_scene_to_save) if 'objects' in final_scene_to_save else final_scene_to_save
                    trimesh_metrics = trimesh_metrics_instance.evaluate_scene(scene_for_eval, format_type='ours')
                    physics_feedback = generate_physics_feedback(trimesh_metrics, top_k=3)
                    if physics_feedback:
                        feedback_parts.append(physics_feedback)
                        print(f"Physics feedback: {physics_feedback}")
                except Exception as e:
                    print(f"Warning: Physics feedback generation failed: {e}")
            
            # 2. VLM layout feedback (from Azure GPT-5.1)
            if enable_vlm_feedback and azure_client_for_feedback is not None and current_image_path and Path(current_image_path).exists():
                try:
                    layout_feedback = generate_vlm_layout_feedback_azure(
                        image_path=current_image_path,
                        user_requirement=user_prompt,
                        azure_client=azure_client_for_feedback
                    )
                    if layout_feedback:
                        feedback_parts.append(layout_feedback)
                        print(f"Layout feedback: {layout_feedback}")
                except Exception as e:
                    print(f"Warning: VLM layout feedback generation failed: {e}")
            
            # Combine feedback
            if feedback_parts:
                last_feedback = " ".join(feedback_parts)
                print(f"Combined feedback for next iteration: {last_feedback}")
            else:
                last_feedback = ""
        
        print(f"Iteration {iteration + 1} completed successfully")
    
    # Save complete conversation history to file
    if conversation_history:
        with open(output_path / "conversation_history.txt", 'w', encoding='utf-8') as f:
            for i, (user_msg, assistant_msg) in enumerate(conversation_history, 1):
                f.write(f"=== Iteration {i} ===\n")
                f.write(f"User: {user_msg}\n\n")
                f.write(f"Assistant: {assistant_msg}\n\n")
                f.write("-" * 80 + "\n\n")
        print(f"Saved {len(conversation_history)} conversation turns to conversation_history.txt")
    
    # Generate iteration process composite image
    print(f"\n{'='*50}")
    print("CREATING ITERATION SUMMARY")
    print(f"{'='*50}")
    summary_image_path = create_iteration_summary_image(output_path, all_image_paths, num_iterations)
    
    print(f"\n{'='*50}")
    print("ITERATIVE GENERATION COMPLETED")
    print(f"{'='*50}")
    print(f"Total iterations: {len(all_scenes)}")
    print(f"Output directory: {output_path}")
    print(f"Final scene: {output_path / f'scene_iter_{len(all_scenes)}.json'}")
    print(f"Conversation history: {output_path / 'conversation_history.txt'}")
    if summary_image_path:
        print(f"Iteration summary: {summary_image_path}")
    
    return all_scenes

def parse_prompt_with_scene(prompt_line):
    """
    Parse a prompt line that might contain <current_scene>...json...</current_scene>.
    Returns:
        tuple: (cleaned_prompt, scene_data_dict or None)
    """
    import re
    import json
    
    # Check for <current_scene> block with markdown code block
    match = re.search(r'<current_scene>\s*```json\s*({.*?})\s*```\s*</current_scene>', prompt_line, re.DOTALL)
    if not match:
        # Try without markdown code block markers
        match = re.search(r'<current_scene>\s*({.*?})\s*</current_scene>', prompt_line, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            scene_data = json.loads(json_str)
            # Remove the scene block from prompt
            cleaned_prompt = re.sub(r'<current_scene>.*?</current_scene>', '', prompt_line, flags=re.DOTALL).strip()
            return cleaned_prompt, scene_data
        except json.JSONDecodeError:
            print("Warning: Failed to parse JSON from <current_scene> block")
            return prompt_line, None
            
    return prompt_line, None

def batch_iterative_scene_generation_parallel(prompts: List[str], engine, request_config, 
                                              initial_scene_path: str = None,
                                              asset_retrieval_module=None, 
                                              num_iterations: int = 3,
                                              output_base_dir: str = "./output/batch",
                                              generate_room: bool = False,
                                              use_model_for_creation: bool = False,
                                              use_gpt_with_objects: bool = False,
                                              asset_source: str = '3d-future',
                                              objaverse_retriever=None,
                                              enable_visualization: bool = False,
                                              max_batch_size: int = 4,
                                              max_history_turns: int = 4,
                                              enable_physics_feedback: bool = False,
                                              enable_vlm_feedback: bool = False,
                                              enable_physics_optimization: bool = False,
                                              physics_opt_steps: int = 5,
                                              models_path: str = None,
                                              original_indices: List[int] = None):
    """Parallel batch iterative scene generation - process all prompts simultaneously at each iteration step
    
    Args:
        prompts: List of prompts
        engine: Inference engine
        request_config: Request configuration
        initial_scene_path: Initial scene JSON file path (if not generating empty room)
        asset_retrieval_module: Asset retrieval module
        num_iterations: Number of iterations per prompt
        output_base_dir: Base output directory
        generate_room: Whether to generate empty room
        use_model_for_creation: Whether to use model to generate initial scene
        use_gpt_with_objects: Whether to use GPT to generate complete scene with objects
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retriever instance
        enable_visualization: Whether to enable 3D visualization guide lines
        max_batch_size: Maximum batch size for parallel inference to prevent OOM
        max_history_turns: Maximum number of conversation history turns to keep
        enable_physics_feedback: Whether to enable physics feedback injection
        enable_vlm_feedback: Whether to enable VLM layout feedback injection
        enable_physics_optimization: Whether to enable physics optimization (collision/out-of-bounds fix)
        physics_opt_steps: Maximum physics optimization steps
        models_path: Path to 3D models
        original_indices: Original prompt index list (1-indexed), for maintaining index consistency when skipping completed scenes
    
    Returns:
        List of results for all scenes
    """
    # If no original indices provided, use default 1 to N
    if original_indices is None:
        original_indices = list(range(1, len(prompts) + 1))
    
    print(f"\n{'='*60}")
    print(f"PARALLEL BATCH ITERATIVE SCENE GENERATION")
    print(f"Processing {len(prompts)} prompts in parallel with {num_iterations} iterations each")
    print(f"{'='*60}\n")
    
    # Initialize components needed for feedback generation
    trimesh_metrics_instance = None
    azure_client_for_feedback = None
    
    # Try to initialize physics evaluator
    try:
        trimesh_metrics_instance = TrimeshPhysicsMetrics(verbose=False)
        print("Initialized TrimeshPhysicsMetrics for feedback generation")
    except Exception as e:
        print(f"Warning: Could not initialize TrimeshPhysicsMetrics: {e}")
    
    # Try to initialize Azure client (for VLM layout feedback)
    try:
        azure_client_for_feedback = setup_azure_client()
        print("Initialized Azure client for VLM layout feedback")
    except Exception as e:
        print(f"Warning: Could not initialize Azure client for feedback: {e}")
    
    # Create collection folder (create in advance, collect immediately upon completion)
    output_base_path = Path(output_base_dir)
    final_scenes_dir = output_base_path / "final_scenes_collection"
    final_renders_dir = output_base_path / "final_renders_collection"
    final_scenes_dir.mkdir(parents=True, exist_ok=True)
    final_renders_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created collection directories:")
    print(f"  Final scenes: {final_scenes_dir}")
    print(f"  Final renders: {final_renders_dir}")
    
    # For tracking collection count
    collected_scenes_count = 0
    collected_renders_count = 0
    
    # Prepare output directories and initialization data for each prompt
    prompt_contexts = []
    
    for list_idx, prompt in enumerate(prompts):
        # Use original index instead of list index
        original_idx = original_indices[list_idx]
        
        # Parse prompt for embedded scene
        prompt_text, embedded_scene = parse_prompt_with_scene(prompt)
        
        prompt_output_dir = Path(output_base_dir) / f"prompt_{original_idx}"
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prompt to file
        with open(prompt_output_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        context = {
            "idx": original_idx,  # Use original index
            "prompt": prompt_text,
            "output_dir": prompt_output_dir,
            "status": "initializing",
            "conversation_history": [],
            "current_scene": None,
            "current_image_path": None,
            "all_scenes": [],
            "all_image_paths": [],
            "error": None,
            "last_feedback": "",  # Store feedback generated from previous round
            "embedded_scene": embedded_scene  # Store embedded scene
        }
        prompt_contexts.append(context)
    
    # Step 1: Generate or load initial scenes in parallel
    print(f"\n{'='*60}")
    print(f"STEP 0: Initializing scenes for {len(prompts)} prompts")
    print(f"{'='*60}\n")
    
    if use_gpt_with_objects:
        # Use GPT to generate complete scene with objects
        print("Generating complete scenes with GPT (including furniture objects)...")
        
        for ctx in prompt_contexts:
            if ctx.get("embedded_scene"):
                # Use embedded scene directly
                ctx["current_scene"] = ctx["embedded_scene"]
                ctx["status"] = "initialized"
                print(f"✓ Prompt {ctx['idx']}: Using embedded scene from input")
                
                # Save the embedded scene
                scene_path = ctx["output_dir"] / "embedded_initial_scene.json"
                with open(scene_path, 'w', encoding='utf-8') as f:
                    json.dump(ctx["embedded_scene"], f, indent=2, ensure_ascii=False)
            else:
                # Use GPT to generate complete scene with objects
                print(f"Generating complete scene with GPT for prompt {ctx['idx']}...")
                try:
                    generated_scene_path = ctx["output_dir"] / "generated_scene_with_objects.json"
                    
                    scene_data = generate_scene_with_objects(
                        ctx["prompt"], 
                        str(generated_scene_path),
                        asset_retrieval_module=asset_retrieval_module,
                        asset_source=asset_source,
                        objaverse_retriever=objaverse_retriever
                    )
                    
                    if scene_data is not None:
                        ctx["current_scene"] = scene_data
                        ctx["status"] = "initialized"
                        print(f"✓ Prompt {ctx['idx']}: Complete scene with objects generated")
                    else:
                        # Fall back to generating empty room
                        print(f"⚠ Prompt {ctx['idx']}: GPT scene generation failed, falling back to empty room")
                        generated_room_path = ctx["output_dir"] / "generated_empty_room.json"
                        room_data = generate_empty_room(ctx["prompt"], str(generated_room_path))
                        if room_data is not None:
                            ctx["current_scene"] = room_data
                            ctx["status"] = "initialized"
                            print(f"✓ Prompt {ctx['idx']}: Fallback empty room generated")
                        else:
                            ctx["status"] = "failed"
                            ctx["error"] = "Failed to generate scene"
                            print(f"✗ Prompt {ctx['idx']}: Failed to generate scene")
                except Exception as e:
                    ctx["status"] = "failed"
                    ctx["error"] = f"Scene generation error: {str(e)}"
                    print(f"✗ Prompt {ctx['idx']}: Scene generation error - {e}")
    
    elif generate_room:
        # Generate empty rooms in parallel
        print("Generating empty rooms in parallel...")
        
        # Identify contexts that need generation vs those with embedded scenes
        contexts_to_generate = []
        infer_requests = []
        
        for ctx in prompt_contexts:
            if ctx.get("embedded_scene"):
                # Use embedded scene directly
                ctx["current_scene"] = ctx["embedded_scene"]
                ctx["status"] = "initialized"
                print(f"✓ Prompt {ctx['idx']}: Using embedded scene from input")
                
                # Save the embedded scene
                scene_path = ctx["output_dir"] / "embedded_initial_scene.json"
                with open(scene_path, 'w', encoding='utf-8') as f:
                    json.dump(ctx["embedded_scene"], f, indent=2, ensure_ascii=False)
            else:
                # Needs generation
                contexts_to_generate.append(ctx)
                messages = [{'role': 'user', 'content': ctx["prompt"]}]
                infer_requests.append(InferRequest(messages=messages, images=None))
        
        if contexts_to_generate:
            try:
                resp_list = engine.infer(infer_requests, request_config)
                
                for ctx, resp in zip(contexts_to_generate, resp_list):
                    response = resp.choices[0].message.content
                    room_data = extract_create_scene_from_response(response)
                    
                    if room_data is not None:
                        # Save generated room
                        generated_room_path = ctx["output_dir"] / "generated_empty_room.json"
                        with open(generated_room_path, 'w', encoding='utf-8') as f:
                            json.dump(room_data, f, indent=2, ensure_ascii=False)
                        
                        ctx["current_scene"] = room_data
                        ctx["conversation_history"].append((ctx["prompt"], response))
                        ctx["status"] = "initialized"
                        print(f"✓ Prompt {ctx['idx']}: Empty room generated")
                    else:
                        ctx["status"] = "failed"
                        ctx["error"] = "Failed to generate empty room"
                        print(f"✗ Prompt {ctx['idx']}: Failed to generate empty room")
            except Exception as e:
                print(f"Error during parallel room generation: {e}")
                for ctx in contexts_to_generate:
                    if ctx["status"] == "initializing":
                        ctx["status"] = "failed"
                        ctx["error"] = f"Room generation error: {str(e)}"
    else:
        # Use the same initial scene
        if not initial_scene_path or not Path(initial_scene_path).exists():
            print(f"Error: Initial scene file not found: {initial_scene_path}")
            return []
        
        with open(initial_scene_path, 'r', encoding='utf-8') as f:
            initial_scene = json.load(f)
        
        for ctx in prompt_contexts:
            ctx["current_scene"] = initial_scene.copy()
            ctx["status"] = "initialized"
        print(f"✓ Loaded initial scene for all {len(prompts)} prompts")
    
    # Render initial scenes for all prompts
    print("\nRendering initial scenes...")
    for ctx in prompt_contexts:
        if ctx["status"] == "initialized":
            try:
                ctx["current_image_path"] = render_scene_to_image(
                    ctx["current_scene"], 
                    ctx["output_dir"], 
                    0,
                    enable_visualization=enable_visualization
                )
                ctx["all_image_paths"].append(ctx["current_image_path"])
                print(f"✓ Prompt {ctx['idx']}: Initial scene rendered")
            except Exception as e:
                print(f"✗ Prompt {ctx['idx']}: Rendering failed - {e}")
                ctx["status"] = "failed"
                ctx["error"] = f"Rendering error: {str(e)}"
    
    # Parallel iterative optimization
    active_contexts = [ctx for ctx in prompt_contexts if ctx["status"] == "initialized"]
    
    for iteration in range(num_iterations):
        if not active_contexts:
            print("\nNo active prompts remaining, stopping iterations")
            break
        
        print(f"\n{'='*60}")
        print(f"PARALLEL ITERATION {iteration + 1}/{num_iterations}")
        print(f"Processing {len(active_contexts)} active prompts")
        print(f"{'='*60}\n")
        
        # Prepare inference requests for all active prompts
        infer_requests = []
        for ctx in active_contexts:
            current_scene_json = json.dumps(ctx["current_scene"], indent=2)
            
            if iteration == 0:
                # First round: no feedback
                base_user_content = f'{ctx["prompt"]}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
            else:
                # Subsequent turns: include feedback (if available)
                feedback_section = ""
                if ctx.get("last_feedback"):
                    feedback_section = f'\n<feedback>\n{ctx["last_feedback"]}\n</feedback>\n'
                base_user_content = f'Please continue to improve the scene based on the original request: "{ctx["prompt"]}"{feedback_section}\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
            
            messages = []
            # Add current turn's user message
            current_user_message = f'<image>{base_user_content}'
            
            # Use smart truncation: based on token count rather than fixed turns
            conv_history = smart_truncate_conversation_history(
                conversation_history=ctx["conversation_history"],
                current_user_message=current_user_message,
                current_image_path=ctx["current_image_path"],
                engine=engine,
                max_model_len=40960,
                max_tokens=request_config.max_tokens if request_config.max_tokens else 16384
            )
            
            for hist_user_msg, hist_assistant_msg in conv_history:
                messages.append({'role': 'user', 'content': hist_user_msg})
                messages.append({'role': 'assistant', 'content': hist_assistant_msg})
            
            messages.append({'role': 'user', 'content': current_user_message})
            
            infer_requests.append(InferRequest(
                messages=messages,
                images=[ctx["current_image_path"]]
            ))
        
        # Execute inference in parallel (batch processing to avoid OOM)
        print(f"Executing parallel inference for {len(infer_requests)} prompts (batch size: {max_batch_size})...")
        try:
            # Batch inference
            all_responses = []
            for batch_start in range(0, len(infer_requests), max_batch_size):
                batch_end = min(batch_start + max_batch_size, len(infer_requests))
                batch_requests = infer_requests[batch_start:batch_end]
                print(f"  Processing batch {batch_start//max_batch_size + 1}: requests {batch_start+1}-{batch_end}")
                batch_resp_list = engine.infer(batch_requests, request_config)
                all_responses.extend(batch_resp_list)
            resp_list = all_responses
            
            # Process each response
            newly_inactive = []
            for ctx, resp in zip(active_contexts, resp_list):
                response = resp.choices[0].message.content
                
                # Save response
                with open(ctx["output_dir"] / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
                    f.write(response)
                
                # Update conversation history
                ctx["conversation_history"].append((
                    f'<image>{base_user_content}',
                    response
                ))
                
                # Extract tool_calls
                tool_calls = extract_tool_calls_from_response(response)
                
                if tool_calls is not None:
                    # Check for terminate tool
                    has_terminate = any(
                        isinstance(tc, dict) and tc.get('name') == 'terminate' 
                        for tc in tool_calls
                    )
                    
                    if has_terminate:
                        print(f"✓ Prompt {ctx['idx']}: Terminated early (scene complete)")
                        ctx["status"] = "completed"
                        newly_inactive.append(ctx)
                        
                        # Save final scene
                        scene_file_path = ctx["output_dir"] / f"scene_iter_{iteration + 1}_final.json"
                        with open(scene_file_path, 'w', encoding='utf-8') as f:
                            json.dump(ctx["current_scene"], f, indent=2, ensure_ascii=False)
                        continue
                    
                    # Apply tool calls
                    final_scene = apply_tool_calls_to_scene(ctx["current_scene"], tool_calls)
                else:
                    final_scene = extract_final_scene_from_response(response)
                
                if final_scene is None:
                    print(f"✗ Prompt {ctx['idx']}: Failed to extract scene")
                    ctx["status"] = "failed"
                    ctx["error"] = f"Failed to extract scene at iteration {iteration + 1}"
                    newly_inactive.append(ctx)
                    continue
                
                # Retrieve assets
                final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module,
                                                        asset_source=asset_source,
                                                        objaverse_retriever=objaverse_retriever)
                
                # Apply physics optimization (collision/out-of-bounds fix)
                physics_deleted_feedback = ""
                if enable_physics_optimization:
                    print(f"✓ Prompt {ctx['idx']}: Applying physics optimization...")
                    final_scene, physics_deleted_feedback = apply_physics_optimization(
                        final_scene, 
                        models_path=models_path or "", 
                        max_steps=physics_opt_steps
                    )
                
                # Save scene
                scene_file_path = ctx["output_dir"] / f"scene_iter_{iteration + 1}.json"
                with open(scene_file_path, 'w', encoding='utf-8') as f:
                    json.dump(final_scene, f, indent=2, ensure_ascii=False)
                
                ctx["all_scenes"].append(final_scene)
                
                # Render scene
                ctx["current_image_path"] = render_scene_to_image(
                    final_scene, 
                    ctx["output_dir"], 
                    iteration + 1,
                    enable_visualization=enable_visualization
                )
                ctx["all_image_paths"].append(ctx["current_image_path"])
                
                # Update current scene
                ctx["current_scene"] = final_scene
                
                # ===== Generate feedback for next round =====
                # Only generate feedback on non-last rounds
                if iteration < num_iterations - 1:
                    feedback_parts = []
                    
                    # 0. Physics optimization deleted objects feedback (add first)
                    if physics_deleted_feedback:
                        feedback_parts.append(physics_deleted_feedback)
                    
                    # 1. Physics feedback (from trimesh)
                    if enable_physics_feedback and trimesh_metrics_instance is not None:
                        try:
                            # Use grouped format for physics evaluation
                            scene_for_eval = convert_flat_to_grouped(final_scene) if 'objects' in final_scene else final_scene
                            trimesh_metrics = trimesh_metrics_instance.evaluate_scene(scene_for_eval, format_type='ours')
                            physics_feedback = generate_physics_feedback(trimesh_metrics, top_k=3)
                            if physics_feedback:
                                feedback_parts.append(physics_feedback)
                        except Exception as e:
                            print(f"⚠ Prompt {ctx['idx']}: Physics feedback failed - {e}")
                    
                    # 2. VLM layout feedback (from Azure GPT-5.1)
                    if enable_vlm_feedback and azure_client_for_feedback is not None and ctx["current_image_path"] and Path(ctx["current_image_path"]).exists():
                        try:
                            layout_feedback = generate_vlm_layout_feedback_azure(
                                image_path=ctx["current_image_path"],
                                user_requirement=ctx["prompt"],
                                azure_client=azure_client_for_feedback
                            )
                            if layout_feedback:
                                feedback_parts.append(layout_feedback)
                        except Exception as e:
                            print(f"⚠ Prompt {ctx['idx']}: VLM feedback failed - {e}")
                    
                    # Combine feedback
                    if feedback_parts:
                        ctx["last_feedback"] = " ".join(feedback_parts)
                    else:
                        ctx["last_feedback"] = ""
                
                print(f"✓ Prompt {ctx['idx']}: Iteration {iteration + 1} completed")
            
            # Remove no longer active contexts
            for ctx in newly_inactive:
                active_contexts.remove(ctx)
                
        except Exception as e:
            print(f"Error during parallel iteration {iteration + 1}: {e}")
            import traceback
            traceback.print_exc()
            
            for ctx in active_contexts:
                ctx["status"] = "failed"
                ctx["error"] = f"Parallel inference error at iteration {iteration + 1}: {str(e)}"
            break
    
    # Mark prompts that completed all iterations
    for ctx in active_contexts:
        if ctx["status"] == "initialized":
            ctx["status"] = "completed"
    
    # Generate iteration summary images and immediately collect final_scene and renders
    print(f"\n{'='*60}")
    print("GENERATING SUMMARIES AND COLLECTING FINAL OUTPUTS")
    print(f"{'='*60}\n")
    
    for ctx in prompt_contexts:
        if ctx["status"] in ["completed", "initialized"] and ctx["all_image_paths"]:
            try:
                summary_image_path = create_iteration_summary_image(
                    ctx["output_dir"], 
                    ctx["all_image_paths"], 
                    len(ctx["all_scenes"])
                )
                print(f"✓ Prompt {ctx['idx']}: Summary image created")
            except Exception as e:
                print(f"✗ Prompt {ctx['idx']}: Failed to create summary - {e}")
            
            # Immediately collect final_scene and renders
            try:
                # Collect final scene JSON
                scene_files = sorted(ctx["output_dir"].glob("scene_iter_*.json"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if scene_files:
                    final_scene_file = scene_files[-1]
                    destination_scene = final_scenes_dir / f"prompt_{ctx['idx']}_final_scene.json"
                    shutil.copy2(final_scene_file, destination_scene)
                    collected_scenes_count += 1
                    print(f"✓ Prompt {ctx['idx']}: Collected final scene")
                
                # Collect final render images
                render_files = sorted(ctx["output_dir"].glob("merged_iter_*.png"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if render_files:
                    final_render_file = render_files[-1]
                    destination_render = final_renders_dir / f"prompt_{ctx['idx']}_final_render.png"
                    shutil.copy2(final_render_file, destination_render)
                    collected_renders_count += 1
                    print(f"✓ Prompt {ctx['idx']}: Collected final render")
            except Exception as collect_e:
                print(f"⚠ Prompt {ctx['idx']}: Failed to collect files - {collect_e}")
    
    # Save conversation history
    for ctx in prompt_contexts:
        if ctx["conversation_history"]:
            with open(ctx["output_dir"] / "conversation_history.txt", 'w', encoding='utf-8') as f:
                for i, (user_msg, assistant_msg) in enumerate(ctx["conversation_history"], 1):
                    f.write(f"=== Iteration {i} ===\n")
                    f.write(f"User: {user_msg}\n\n")
                    f.write(f"Assistant: {assistant_msg}\n\n")
                    f.write("-" * 80 + "\n\n")
    
    # Collect results
    all_results = []
    for ctx in prompt_contexts:
        if ctx["status"] in ["completed", "initialized"]:
            all_results.append({
                "prompt": ctx["prompt"],
                "status": "success",
                "output_dir": str(ctx["output_dir"]),
                "num_scenes": len(ctx["all_scenes"])
            })
        else:
            all_results.append({
                "prompt": ctx["prompt"],
                "status": "failed" if ctx["status"] == "failed" else "error",
                "error": ctx.get("error", "Unknown error")
            })
    
    # Summarize collection results
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    successful_count = len([r for r in all_results if r['status'] == 'success'])
    print(f"Collected {collected_scenes_count}/{successful_count} final scenes")
    print(f"Collected {collected_renders_count}/{successful_count} final renders")
    print(f"Final scenes directory: {final_scenes_dir}")
    print(f"Final renders directory: {final_renders_dir}")
    
    # Save summary
    summary_file = output_base_path / "batch_summary.json"
    summary_data = {
        "total_prompts": len(prompts),
        "successful": successful_count,
        "failed": len([r for r in all_results if r["status"] != "success"]),
        "iterations_per_prompt": num_iterations,
        "final_scenes_collected": collected_scenes_count,
        "final_renders_collected": collected_renders_count,
        "final_scenes_directory": str(final_scenes_dir),
        "final_renders_directory": str(final_renders_dir),
        "parallel_processing": True,
        "results": all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("PARALLEL BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total prompts: {summary_data['total_prompts']}")
    print(f"Successful: {summary_data['successful']}")
    print(f"Failed: {summary_data['failed']}")
    print(f"Final scenes collected: {summary_data['final_scenes_collected']}")
    print(f"Final renders collected: {summary_data['final_renders_collected']}")
    print(f"Summary saved to: {summary_file}")
    
    return all_results

def batch_iterative_scene_generation(prompts: List[str], engine, request_config, 
                                     initial_scene_path: str = None,
                                     asset_retrieval_module=None, 
                                     num_iterations: int = 3,
                                     output_base_dir: str = "./output/batch",
                                     generate_room: bool = False,
                                     use_model_for_creation: bool = False,
                                     use_gpt_with_objects: bool = False,
                                     asset_source: str = '3d-future',
                                     objaverse_retriever=None,
                                     enable_visualization: bool = False,
                                     enable_physics_feedback: bool = False,
                                     enable_vlm_feedback: bool = False,
                                     enable_physics_optimization: bool = False,
                                     physics_opt_steps: int = 5,
                                     models_path: str = None,
                                     original_indices: List[int] = None):
    """Batch iterative scene generation - execute full iterative scene generation for multiple prompts
    
    Args:
        prompts: List of prompts
        engine: Inference engine
        request_config: Request configuration
        initial_scene_path: Initial scene JSON file path (if not generating empty room)
        asset_retrieval_module: Asset retrieval module
        num_iterations: Number of iterations per prompt
        output_base_dir: Base output directory
        generate_room: Whether to generate empty room
        use_model_for_creation: Whether to use model to generate initial scene
        use_gpt_with_objects: Whether to use GPT to generate complete scene with objects
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retriever instance
        enable_visualization: Whether to enable 3D visualization guide lines
        enable_physics_feedback: Whether to enable physics feedback injection
        enable_vlm_feedback: Whether to enable VLM layout feedback injection
        enable_physics_optimization: Whether to enable physics optimization (collision/out-of-bounds fix)
        physics_opt_steps: Maximum physics optimization steps
        models_path: Path to 3D models
        original_indices: Original prompt index list (1-indexed), for maintaining index consistency when skipping completed scenes
    
    Returns:
        List of results for all scenes
    """
    # If no original indices provided, use default 1 to N
    if original_indices is None:
        original_indices = list(range(1, len(prompts) + 1))
    
    print(f"\n{'='*50}")
    print(f"BATCH ITERATIVE SCENE GENERATION")
    print(f"Processing {len(prompts)} prompts with {num_iterations} iterations each")
    print(f"{'='*50}\n")
    
    # Create collection folder (create in advance, collect immediately upon completion)
    output_base_path = Path(output_base_dir)
    final_scenes_dir = output_base_path / "final_scenes_collection"
    final_renders_dir = output_base_path / "final_renders_collection"
    final_scenes_dir.mkdir(parents=True, exist_ok=True)
    final_renders_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created collection directories:")
    print(f"  Final scenes: {final_scenes_dir}")
    print(f"  Final renders: {final_renders_dir}")
    
    all_results = []
    collected_scenes_count = 0
    collected_renders_count = 0
    
    for list_idx, prompt in enumerate(prompts):
        # Use original index instead of list index
        idx = original_indices[list_idx]
        
        # Parse prompt for embedded scene
        prompt_text, embedded_scene = parse_prompt_with_scene(prompt)
        
        print(f"\n{'#'*60}")
        print(f"# BATCH ITEM {list_idx + 1}/{len(prompts)} (Original Index: {idx})")
        print(f"# Prompt: {prompt_text}")
        print(f"{'#'*60}\n")
        
        # Create independent output directory for each prompt
        prompt_output_dir = Path(output_base_dir) / f"prompt_{idx}"
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save current prompt to file
        with open(prompt_output_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        try:
            # Determine initial scene path
            scene_path = initial_scene_path
            initial_conversation = None
            
            # If embedded scene exists, use it
            if embedded_scene:
                print(f"Using embedded scene from input for prompt {idx}...")
                embedded_scene_path = prompt_output_dir / "embedded_initial_scene.json"
                with open(embedded_scene_path, 'w', encoding='utf-8') as f:
                    json.dump(embedded_scene, f, indent=2, ensure_ascii=False)
                scene_path = str(embedded_scene_path)
                print(f"✓ Embedded scene saved: {scene_path}")
            
            # If using GPT to generate complete scene with objects
            elif use_gpt_with_objects:
                print(f"Generating complete scene with GPT for prompt {idx}...")
                generated_scene_path = prompt_output_dir / "generated_scene_with_objects.json"
                
                scene_data = generate_scene_with_objects(
                    prompt_text, 
                    str(generated_scene_path),
                    asset_retrieval_module=asset_retrieval_module,
                    asset_source=asset_source,
                    objaverse_retriever=objaverse_retriever
                )
                
                if scene_data is not None:
                    scene_path = str(generated_scene_path)
                    print(f"✓ Complete scene with objects generated: {scene_path}")
                else:
                    print(f"✗ Failed to generate scene with objects, falling back to empty room")
                    # Fall back to generating empty room
                    generated_room_path = prompt_output_dir / "generated_empty_room.json"
                    room_data = generate_empty_room(prompt_text, str(generated_room_path))
                    if room_data is not None:
                        scene_path = str(generated_room_path)
                        print(f"✓ Fallback empty room generated: {scene_path}")
                    else:
                        if not scene_path or not Path(scene_path).exists():
                            print(f"Error: No valid scene file available for prompt {idx}")
                            all_results.append({
                                "prompt": prompt_text,
                                "status": "failed",
                                "error": "No valid scene file"
                            })
                            continue
                
            # If empty room generation is needed
            elif generate_room:
                print(f"Generating empty room for prompt {idx}...")
                generated_room_path = prompt_output_dir / "generated_empty_room.json"
                
                if use_model_for_creation:
                    room_data, initial_conversation = generate_empty_room_with_model(
                        prompt_text, 
                        engine, 
                        request_config,
                        str(generated_room_path)
                    )
                else:
                    room_data = generate_empty_room(prompt_text, str(generated_room_path))
                
                if room_data is not None:
                    scene_path = str(generated_room_path)
                    print(f"✓ Empty room generated: {scene_path}")
                else:
                    print(f"✗ Failed to generate room, using default scene")
                    if not scene_path or not Path(scene_path).exists():
                        print(f"Error: No valid scene file available for prompt {idx}")
                        all_results.append({
                            "prompt": prompt_text,
                            "status": "failed",
                            "error": "No valid scene file"
                        })
                        continue
            
            # Check if scene file exists
            if not scene_path or not Path(scene_path).exists():
                print(f"Error: Scene file not found: {scene_path}")
                all_results.append({
                    "prompt": prompt_text,
                    "status": "failed",
                    "error": f"Scene file not found: {scene_path}"
                })
                continue
            
            # Execute iterative scene generation
            scenes = iterative_scene_generation(
                initial_scene_path=scene_path,
                user_prompt=prompt_text,
                engine=engine,
                request_config=request_config,
                asset_retrieval_module=asset_retrieval_module,
                num_iterations=num_iterations,
                output_dir=str(prompt_output_dir),
                initial_conversation=initial_conversation,
                asset_source=asset_source,
                objaverse_retriever=objaverse_retriever,
                enable_visualization=enable_visualization,
                enable_physics_feedback=enable_physics_feedback,
                enable_vlm_feedback=enable_vlm_feedback,
                enable_physics_optimization=enable_physics_optimization,
                physics_opt_steps=physics_opt_steps,
                models_path=models_path
            )
            
            all_results.append({
                "prompt": prompt_text,
                "status": "success",
                "output_dir": str(prompt_output_dir),
                "num_scenes": len(scenes)
            })
            
            print(f"\n✓ Prompt {idx} completed successfully!")
            print(f"  Generated {len(scenes)} scenes")
            print(f"  Output: {prompt_output_dir}")
            
            # Immediately collect final_scene and renders
            try:
                # Collect final scene JSON
                scene_files = sorted(prompt_output_dir.glob("scene_iter_*.json"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if scene_files:
                    final_scene_file = scene_files[-1]
                    destination_scene = final_scenes_dir / f"prompt_{idx}_final_scene.json"
                    shutil.copy2(final_scene_file, destination_scene)
                    collected_scenes_count += 1
                    print(f"  ✓ Collected final scene: {destination_scene.name}")
                
                # Collect final render images
                render_files = sorted(prompt_output_dir.glob("merged_iter_*.png"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if render_files:
                    final_render_file = render_files[-1]
                    destination_render = final_renders_dir / f"prompt_{idx}_final_render.png"
                    shutil.copy2(final_render_file, destination_render)
                    collected_renders_count += 1
                    print(f"  ✓ Collected final render: {destination_render.name}")
            except Exception as collect_e:
                print(f"  ⚠ Warning: Failed to collect files for prompt {idx}: {collect_e}")
            
        except Exception as e:
            print(f"\n✗ Error processing prompt {idx}: {e}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                "prompt": prompt,
                "status": "error",
                "error": str(e)
            })
    
    # Summarize collection results
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    successful_count = len([r for r in all_results if r['status'] == 'success'])
    print(f"Collected {collected_scenes_count}/{successful_count} final scenes")
    print(f"Collected {collected_renders_count}/{successful_count} final renders")
    print(f"Final scenes directory: {final_scenes_dir}")
    print(f"Final renders directory: {final_renders_dir}")
    
    # Save batch processing summary results
    print(f"\n{'='*60}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    output_base_path = Path(output_base_dir)
    summary_file = output_base_path / "batch_summary.json"
    
    summary_data = {
        "total_prompts": len(prompts),
        "successful": len([r for r in all_results if r["status"] == "success"]),
        "failed": len([r for r in all_results if r["status"] != "success"]),
        "iterations_per_prompt": num_iterations,
        "final_scenes_collected": collected_scenes_count,
        "final_renders_collected": collected_renders_count,
        "final_scenes_directory": str(final_scenes_dir),
        "final_renders_directory": str(final_renders_dir),
        "results": all_results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"Total prompts: {summary_data['total_prompts']}")
    print(f"Successful: {summary_data['successful']}")
    print(f"Failed: {summary_data['failed']}")
    print(f"Final scenes collected: {summary_data['final_scenes_collected']}")
    print(f"Final renders collected: {summary_data['final_renders_collected']}")
    print(f"Summary saved to: {summary_file}")
    
    return all_results

def load_prompts_from_file(file_path: str, max_prompts: int = None) -> List[str]:
    """Read prompts from txt file
    
    Args:
        file_path: Path to txt file, one prompt per line
        max_prompts: Maximum number of prompts to read, None for all
    
    Returns:
        List of prompts
    """
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    prompts.append(line)
                    # If limit reached, stop reading
                    if max_prompts is not None and len(prompts) >= max_prompts:
                        break
        
        if max_prompts is not None and len(prompts) > max_prompts:
            prompts = prompts[:max_prompts]
        
        print(f"✓ Loaded {len(prompts)} prompts from {file_path}")
        if max_prompts is not None:
            print(f"  (Limited to first {max_prompts} prompts)")
    except Exception as e:
        print(f"Error loading prompts from file: {e}")
        import traceback
        traceback.print_exc()
    return prompts


def find_existing_scenes(output_base_dir: str, total_prompts: int) -> set:
    """Detect completed scenes, return set of completed prompt indices
    
    Args:
        output_base_dir: Base output directory
        total_prompts: Total number of prompts
        
    Returns:
        Set of completed prompt indices (1-indexed)
    """
    existing_indices = set()
    output_path = Path(output_base_dir)
    
    # Check completed scenes in final_scenes_collection directory
    final_scenes_dir = output_path / "final_scenes_collection"
    if final_scenes_dir.exists():
        for scene_file in final_scenes_dir.glob("prompt_*_final_scene.json"):
            # Extract index from filename, format: prompt_{idx}_final_scene.json
            try:
                filename = scene_file.stem  # prompt_1_final_scene
                parts = filename.split('_')
                if len(parts) >= 2:
                    idx = int(parts[1])
                    if 1 <= idx <= total_prompts:
                        existing_indices.add(idx)
            except (ValueError, IndexError):
                continue
    
    # Also check each prompt_X directory for generated scene files
    for idx in range(1, total_prompts + 1):
        if idx in existing_indices:
            continue
        prompt_dir = output_path / f"prompt_{idx}"
        if prompt_dir.exists():
            # Check for final iteration scene files
            scene_files = list(prompt_dir.glob("scene_iter_*.json"))
            if scene_files:
                # Find maximum iteration number
                max_iter = max(
                    int(f.stem.replace("scene_iter_", "")) 
                    for f in scene_files
                )
                # If there are scenes with high iteration numbers, consider it completed
                if max_iter >= 1:
                    existing_indices.add(idx)
    
    return existing_indices


def filter_prompts_by_existing(prompts: List[str], output_base_dir: str, skip_existing: bool = False) -> tuple:
    """Filter prompts based on completed scenes
    
    Args:
        prompts: Original list of prompts
        output_base_dir: Base output directory
        skip_existing: Whether to skip completed scenes
        
    Returns:
        tuple: (filtered prompts list, filtered prompt index list (1-indexed), number of skipped prompts)
    """
    if not skip_existing:
        return prompts, list(range(1, len(prompts) + 1)), 0
    
    existing_indices = find_existing_scenes(output_base_dir, len(prompts))
    
    if not existing_indices:
        print(f"No existing scenes found, processing all {len(prompts)} prompts")
        return prompts, list(range(1, len(prompts) + 1)), 0
    
    filtered_prompts = []
    filtered_indices = []
    
    for idx, prompt in enumerate(prompts, 1):
        if idx not in existing_indices:
            filtered_prompts.append(prompt)
            filtered_indices.append(idx)
    
    skipped_count = len(existing_indices)
    print(f"\n{'='*60}")
    print(f"SKIP EXISTING SCENES MODE")
    print(f"{'='*60}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Already completed: {skipped_count}")
    print(f"Remaining to process: {len(filtered_prompts)}")
    if existing_indices:
        sorted_existing = sorted(existing_indices)
        if len(sorted_existing) <= 20:
            print(f"Skipped prompt indices: {sorted_existing}")
        else:
            print(f"Skipped prompt indices: {sorted_existing[:10]} ... {sorted_existing[-10:]}")
    print(f"{'='*60}\n")
    
    return filtered_prompts, filtered_indices, skipped_count


# Initialize asset retrieval module (if available)
asset_retrieval_module = None
try:
    # Use advanced asset retrieval module from sample.py
    from utils.sample import AssetRetrievalModule
    # Initialization parameters match the test code in sample.py
    asset_retrieval_module = AssetRetrievalModule(
        lambd=0.5, 
        sigma=0.05, 
        temp=0.2, 
        top_p=0.95, 
        top_k=20, 
        asset_size_threshold=0.5, 
        rand_seed=1234, 
        do_print=True
    )
    print("✓ Advanced asset retrieval module initialized successfully")
except Exception as e:
    print(f"Could not initialize advanced asset retrieval module: {e}")
    print("Will use fallback asset retrieval logic")

# Add PIL import for image processing
try:
    from PIL import Image
    print("PIL imported successfully")
except ImportError:
    print("Warning: PIL not available, rendering will be limited")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Iterative Scene Generation')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations (default: 10)')
    parser.add_argument('--prompt', default='I want a modern bedroom.', 
                       help='User prompt for scene generation')
    parser.add_argument('--scene', default='./test/empty_livingroom.json', help='Initial scene JSON file')
    parser.add_argument('--output', default='./output/sft_50k_e3', help='Output directory')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode without actual model inference')
    parser.add_argument('--generate-room', action='store_true', default=True, help='Generate empty room structure before iterative generation')
    parser.add_argument('--use-model-for-creation', action='store_true', default=True,  help='Use fine-tuned model instead of GPT-4o to generate initial scene')
    parser.add_argument('--use-gpt-with-objects', action='store_true', default=False, help='Use GPT to generate complete scene with furniture objects (includes asset retrieval)')
    parser.add_argument('--room-prompt', default=None, help='Prompt for generating empty room (overrides --prompt if provided)')
    # Add batch inference arguments
    parser.add_argument('--batch-mode', action='store_true', help='Run in batch mode - process multiple prompts from file')
    parser.add_argument('--prompts-file', default="/path/to/datasets/llmscene/sft/test_prompt_3dfront_v3.txt", help='Path to txt file containing prompts (one per line)')
    parser.add_argument('--max-prompts', type=int, default=None, help='Maximum number of prompts to process (default: process all)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing - process all prompts simultaneously at each iteration (faster)')
    # Asset source selection arguments
    parser.add_argument('--asset-source', choices=['3d-future', 'objaverse', 'auto'], default='auto',
                       help='Asset source for retrieval: 3d-future (default), objaverse, or auto (hybrid)')
    # 3D visualization arguments
    parser.add_argument('--enable-viz', action='store_true', 
                       help='Enable 3D visualization with auxiliary lines (bbox, arrows, coordinate grid)')
    parser.add_argument('--disable-viz', action='store_true',
                       help='Explicitly disable 3D visualization')
    # Model path arguments
    parser.add_argument('--model', type=str, default="/path/to/SceneReVis/ckpt/rl_ood_B200_v6_e3_s80",
                       help='Path to the model checkpoint directory')
    parser.add_argument('--lora-checkpoint', type=str, default=None,
                       help='Path to the LoRA checkpoint directory (optional)')
    # Multi-GPU parallel arguments
    parser.add_argument('--tensor-parallel', type=int, default=1,
                       help='Number of GPUs for tensor parallelism (default: 1, use all available GPUs with -1)')
    # Conversation history limit arguments
    parser.add_argument('--max-history-turns', type=int, default=8,
                       help='Maximum number of conversation history turns to keep (default: 4)')
    # Batch size limit arguments
    parser.add_argument('--max-batch-size', type=int, default=4,
                       help='Maximum batch size for parallel inference to prevent OOM (default: 4)')
    # Feedback injection control arguments
    parser.add_argument('--enable-physics-feedback', action='store_true', default=False,
                       help='Enable physics feedback injection into user prompts (default: disabled)')
    parser.add_argument('--enable-vlm-feedback', action='store_true', default=False,
                       help='Enable VLM (GPT-5.1) layout feedback injection into user prompts (default: disabled)')
    # Physics optimization control arguments (collision detection and out-of-bounds fix)
    parser.add_argument('--enable-physics-optimization', action='store_true', default=False,
                       help='Enable physics optimization after each iteration (resolve collisions and out-of-bounds)')
    parser.add_argument('--physics-opt-steps', type=int, default=5,
                       help='Maximum steps for physics optimization (default: 5)')
    parser.add_argument('--models-path', type=str, default=None,
                       help='Path to 3D models directory for physics optimization')
    # Skip completed scenes arguments
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='Skip prompts that already have generated final scenes in output directory')
    
    args = parser.parse_args()

    # Initialize Objaverse retriever (if needed)
    objaverse_retriever = None
    if args.asset_source in ['objaverse', 'auto']:
        if ObjaverseRetriever is not None:
            try:
                objaverse_retriever = ObjaverseRetriever(
                    retrieval_threshold=28.0,
                    use_text=True,
                    do_print=True
                )
                print("✓ Objaverse retrieval module initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Objaverse retrieval module: {e}")
                if args.asset_source == 'objaverse':
                    print("  Falling back to 3d-future")
                    args.asset_source = '3d-future'
        else:
            print("Warning: ObjaverseRetriever not available")
            if args.asset_source == 'objaverse':
                print("  Falling back to 3d-future")
                args.asset_source = '3d-future'

    # Determine whether to enable visualization
    enable_visualization = args.enable_viz and not args.disable_viz
    if enable_visualization and render_with_visualization is None:
        print("Warning: 3D visualization requested but module not available")
        enable_visualization = False
    
    print(f"Asset source: {args.asset_source}")
    print(f"3D visualization: {'enabled' if enable_visualization else 'disabled'}")
    print(f"Physics feedback: {'enabled' if args.enable_physics_feedback else 'disabled'}")
    print(f"VLM layout feedback: {'enabled' if args.enable_vlm_feedback else 'disabled'}")
    print(f"Skip existing: {'enabled' if args.skip_existing else 'disabled'}")

    # Model configuration
    global engine, request_config
    model = args.model
    lora_checkpoint = args.lora_checkpoint
    template_type = None  # None: use the corresponding model's default template_type
    
    print(f"Model path: {model}")
    if lora_checkpoint:
        print(f"LoRA checkpoint: {lora_checkpoint}")
    default_system = """### Role and Core Directive

You are an AI spatial layout planner. Your core task is to analyze and optimize indoor scenes, ensuring they are physically valid and functionally efficient.

### Core Capabilities

Your primary responsibility is to **diagnose and correct** problems in the current scene.

1.  **Analyze and Identify Problems**: Based on the input rendered image and scene JSON, proactively identify three types of issues:

      * **Physical Conflicts**: Objects overlapping with each other or extending beyond the defined room boundaries.
      * **Poor Layout**: Furniture placement that obstructs main traffic flows, object orientation or grouping that is not functionally logical, or imbalanced use of space.

2.  **Resolve and Optimize**: Once problems are identified, you must use the available tools in `tool_calls` to automatically correct them, aiming to create a scene that is free of physical conflicts, has clear circulation, and a rational functional layout.

### Common Objects Reference

Here are some common objects found in various room types to guide your scene generation and optimization:

*   **Living Room**: Sofa, Coffee Table, TV Stand, Armchair, Bookshelf, Floor Lamp, Rug, Side Table, Plant.
*   **Bedroom**: Bed, Nightstand, Wardrobe, Dresser, Desk, Chair, Mirror, Lamp, Rug.
*   **Dining Room**: Dining Table, Dining Chair, Sideboard, Chandelier, Rug, Cabinet.
*   **Office**: Desk, Office Chair, Conference Table, Filing Cabinet, Whiteboard, Sofa, Plant.
*   **Study Room**: Desk, Office Chair, Bookshelf, Filing Cabinet, Lamp, Armchair, Rug.
*   **Gym**: Treadmill, Exercise Bike, Dumbbell Rack, Yoga Mat, Bench, Mirror, Gym Ball.
*   **Entertainment Room**: Sofa, TV Stand, Pool Table, Ping Pong Table, Gaming Desk, Gaming Chair, Karaoke Machine, Speaker, Bar Counter.

### Scene Analysis and Spatial Rules

Your input will include a rendered image and the scene's JSON data. The rendered image displays two key views side-by-side, which you must use in combination for a comprehensive judgment:

  * **Left side: Top-down view** - Used for precisely judging relative positions, spacing, overlaps, and boundary compliance. This is the primary basis for detecting physical conflicts.
  * **Right side: Diagonal perspective view** - Used for understanding the space's 3D feel, the actual appearance of furniture, and the harmony and functionality of the overall layout. This is the primary basis for judging layout quality.

**Mandatory Execution Requirements:**
You must analyze the scene by combining the visual image and JSON data, and strictly adhere to the following rules for corrections:

1.  **Fix Physical Conflicts**: No objects are ever allowed to overlap or extend beyond the room boundaries (defined by `bounds_top` and `bounds_bottom`). Upon detecting such issues, you must immediately use tools like `move_object` to correct them.
2.  **Optimize Functional Layout**: Based on your understanding of both views, adjust furniture positions to ensure clear traffic flow, functional soundness, and spatial balance.
3.  **Validate All Operations**: Before every tool call, you must mentally pre-calculate its final state during your thinking process to ensure it does not create new conflicts or layout problems.
4.  **Empty Scene Strategy**: If the scene is empty or lacks essential furniture, prioritize adding all necessary objects first to establish the functional base, then refine their positions and layout in subsequent steps.

### Output Format Requirements

You must differentiate your output format based on the task type:

**1. When Editing a Scene (using `tool_calls`):**
You must strictly follow the `<think>`,  `<tool_calls>` order.

Format template:
```xml
<think>
[Your detailed analysis and reasoning process here]
- Analyze the rendered image (top-down and perspective views)
- Identify any physical conflicts or layout issues
- Calculate object boundaries and validate positions
- Determine the necessary corrections
</think>

<tool_calls>
[
  {
    "id": "tool_1",
    "name": "tool_name",
    "arguments": {
      "param1": "value1",
      "param2": [array_values]
    }
  },
  {
    "id": "tool_2",
    "name": "tool_name",
    "arguments": {
      "param": "value"
    }
  }
]
</tool_calls>
```

**2. When Creating an Initial Scene (using `create_scene`):**
Directly output the `<create_scene>` tag **without** `<think>`.

Format template:
```xml
<create_scene>
```json
{
  "bounds_top": [...],
  "bounds_bottom": [...],
  "room_type": "bedroom",
  "room_id": "...",
  "objects": []
}
```
</create_scene>
```

### Available Tools
**1. add_object**: Add a new furniture piece.
* `object_description` (string)
* `position` (array)
* `rotation` (array)
* `size` (array)

**2. remove_object**: Remove an existing object.
* `jid/uid` (string)

**3. move_object**: Change an object's position.
* `jid/uid` (string)
* `new_position` (array)

**4. rotate_object**: Change an object's rotation.
* `jid/uid` (string)
* `new_rotation` (array)

**5. scale_object**: Change an object's size.
* `jid/uid` (string)
* `new_size` (array)

**6. replace_object**: Replace an existing object.
* `jid/uid_to_replace` (string)
* `new_object_description` (string)

**7. terminate**: End the editing session.
* `reason` (string)"""


    # Load model and conversation template - using vLLM engine
    infer_backend = 'vllm'  # Use vLLM backend for inference

    # Determine tensor parallel size
    import torch
    if args.tensor_parallel == -1:
        # Auto-detect available GPU count
        tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-detected {tensor_parallel_size} GPUs for tensor parallelism")
    else:
        tensor_parallel_size = args.tensor_parallel
    
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Max history turns: {args.max_history_turns}")
    print(f"Max batch size: {args.max_batch_size}")

    if infer_backend == 'vllm':
        # Set seed for reproducibility, add tensor_parallel_size for multi-GPU support
        # First get tokenizer and template to set default_system
        from swift.llm import get_model_tokenizer, get_template
        _, tokenizer = get_model_tokenizer(model, model_type="qwen2_5_vl", load_model=False)
        template = get_template("qwen2_5_vl", tokenizer, default_system=default_system)
        
        engine = VllmEngine(
            model, 
            max_model_len=40960, 
            model_type="qwen2_5_vl", 
            seed=42,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.5,
            template=template
        )
    else:
        # If other backends are needed, they can be added here
        model, tokenizer = get_model_tokenizer(model, model_type="qwen2_5_vl")
        template_type = template_type or model.model_meta.template
        template = get_template(template_type, tokenizer, default_system=default_system)
        engine = PtEngine.from_model_template(model, template, max_batch_size=64)

    # temperature=0 + seed ensures deterministic output
    request_config = RequestConfig(max_tokens=16384, temperature=0, seed=42)
    
    # Batch mode - execute full iterative scene generation for multiple prompts
    if args.batch_mode:
        mode_name = "PARALLEL BATCH MODE" if args.parallel else "SEQUENTIAL BATCH MODE"
        print("="*60)
        print(f"{mode_name} - Multiple Prompts Iterative Scene Generation")
        print("="*60)
        
        if not args.prompts_file:
            print("Error: --prompts-file is required for batch mode")
            return
        
        # Load prompts from file
        prompts = load_prompts_from_file(args.prompts_file, max_prompts=args.max_prompts)
        
        if not prompts:
            print("Error: No prompts loaded from file")
            return
        
        # Filter completed scenes
        filtered_prompts, original_indices, skipped_count = filter_prompts_by_existing(
            prompts, args.output, skip_existing=args.skip_existing
        )
        
        if not filtered_prompts:
            print("All prompts have been processed. Nothing to do.")
            print("Remove --skip-existing flag to regenerate all scenes.")
            return
        
        print(f"\nConfiguration:")
        print(f"  Prompts file: {args.prompts_file}")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Skipped (already done): {skipped_count}")
        print(f"  To process: {len(filtered_prompts)}")
        if args.max_prompts:
            print(f"  Max prompts limit: {args.max_prompts}")
        print(f"  Iterations per prompt: {args.iterations}")
        print(f"  Initial scene: {args.scene}")
        print(f"  Output base dir: {args.output}")
        print(f"  Generate room: {args.generate_room}")
        print(f"  Use model for creation: {args.use_model_for_creation}")
        print(f"  Use GPT with objects: {args.use_gpt_with_objects}")
        print(f"  Parallel processing: {args.parallel}")
        print(f"  Skip existing: {args.skip_existing}")
        print(f"  Test mode: {args.test_mode}")
        
        if args.test_mode:
            print("\nWarning: Test mode is not supported in batch mode")
            print("Running with actual model inference...")
        
        # Choose different processing function based on parallel flag
        if args.parallel:
            # Parallel processing - process all prompts simultaneously at each iteration step
            results = batch_iterative_scene_generation_parallel(
                prompts=filtered_prompts,
                engine=engine,
                request_config=request_config,
                initial_scene_path=args.scene if not args.generate_room and not args.use_gpt_with_objects else None,
                asset_retrieval_module=asset_retrieval_module,
                num_iterations=args.iterations,
                output_base_dir=args.output,
                generate_room=args.generate_room,
                use_model_for_creation=args.use_model_for_creation,
                use_gpt_with_objects=args.use_gpt_with_objects,
                asset_source=args.asset_source,
                objaverse_retriever=objaverse_retriever,
                enable_visualization=enable_visualization,
                max_batch_size=args.max_batch_size,
                max_history_turns=args.max_history_turns,
                enable_physics_feedback=args.enable_physics_feedback,
                enable_vlm_feedback=args.enable_vlm_feedback,
                enable_physics_optimization=args.enable_physics_optimization,
                physics_opt_steps=args.physics_opt_steps,
                models_path=args.models_path if hasattr(args, 'models_path') else None,
                original_indices=original_indices
            )
        else:
            # Serial processing - process one prompt at a time
            results = batch_iterative_scene_generation(
                prompts=filtered_prompts,
                engine=engine,
                request_config=request_config,
                initial_scene_path=args.scene if not args.generate_room and not args.use_gpt_with_objects else None,
                asset_retrieval_module=asset_retrieval_module,
                num_iterations=args.iterations,
                output_base_dir=args.output,
                generate_room=args.generate_room,
                use_model_for_creation=args.use_model_for_creation,
                use_gpt_with_objects=args.use_gpt_with_objects,
                asset_source=args.asset_source,
                objaverse_retriever=objaverse_retriever,
                enable_visualization=enable_visualization,
                enable_physics_feedback=args.enable_physics_feedback,
                enable_vlm_feedback=args.enable_vlm_feedback,
                enable_physics_optimization=args.enable_physics_optimization,
                physics_opt_steps=args.physics_opt_steps,
                models_path=args.models_path if hasattr(args, 'models_path') else None,
                original_indices=original_indices
            )
        
        print(f"\n{'='*60}")
        print(f"{mode_name} COMPLETED")
        print(f"{'='*60}")
        print(f"Processed {len(results)} prompts")
        print(f"Results saved to: {args.output}")
        
        return
    
    # Single prompt iterative scene generation mode (original functionality)
    print("Starting iterative scene generation system...")
    
    print(f"Configuration:")
    print(f"  Initial scene: {args.scene}")
    print(f"  User prompt: {args.prompt}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Output dir: {args.output}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  Generate room: {args.generate_room}")
    print(f"  Use model for creation: {args.use_model_for_creation}")
    print(f"  Use GPT with objects: {args.use_gpt_with_objects}")
    if args.room_prompt:
        print(f"  Room prompt: {args.room_prompt}")
    
    # If scene generation is needed, generate scene structure first
    initial_scene_path = args.scene
    initial_conversation = None  # For storing initial scene generation conversation
    
    if args.use_gpt_with_objects:
        # Use GPT to generate complete scene with objects
        print("\n" + "="*50)
        print("GENERATING COMPLETE SCENE WITH GPT")
        print("="*50)
        
        room_prompt = args.room_prompt or args.prompt
        print(f"Scene generation prompt: {room_prompt}")
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_scene_path = output_dir / "generated_scene_with_objects.json"
        
        print("Using Azure OpenAI (GPT) to generate complete scene with objects...")
        scene_data = generate_scene_with_objects(
            room_prompt, 
            str(generated_scene_path),
            asset_retrieval_module=asset_retrieval_module,
            asset_source=args.asset_source,
            objaverse_retriever=objaverse_retriever
        )
        
        if scene_data is not None:
            print(f"✓ Complete scene with objects generated successfully: {generated_scene_path}")
            initial_scene_path = str(generated_scene_path)
        else:
            print("✗ Failed to generate complete scene, falling back to empty room")
            # Fall back to generating empty room
            generated_room_path = output_dir / "generated_empty_room.json"
            room_data = generate_empty_room(room_prompt, str(generated_room_path))
            if room_data is not None:
                initial_scene_path = str(generated_room_path)
                print(f"✓ Fallback empty room generated: {generated_room_path}")
            else:
                print(f"Fallback to: {args.scene}")
    
    elif args.generate_room:
        print("\n" + "="*50)
        print("GENERATING EMPTY ROOM STRUCTURE")
        print("="*50)
        
        # Use room_prompt or default prompt
        room_prompt = args.room_prompt or args.prompt
        print(f"Room generation prompt: {room_prompt}")
        
        # Generate empty room and save
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_room_path = output_dir / "generated_empty_room.json"
        
        # Choose method based on use_model_for_creation option
        if args.use_model_for_creation:
            print("Using fine-tuned model to generate initial scene...")
            room_data, initial_conversation = generate_empty_room_with_model(
                room_prompt, 
                engine, 
                request_config,
                str(generated_room_path)
            )
        else:
            print("Using Azure OpenAI (GPT-4o) to generate initial scene...")
            room_data = generate_empty_room(room_prompt, str(generated_room_path))
        
        if room_data is not None:
            print(f"✓ Empty room generated successfully: {generated_room_path}")
            # Use generated room as initial scene
            initial_scene_path = str(generated_room_path)
        else:
            print("✗ Failed to generate empty room, using original scene file")
            print(f"Fallback to: {args.scene}")
    
    # Check if initial scene file exists
    if not Path(initial_scene_path).exists():
        print(f"Error: Initial scene file not found: {initial_scene_path}")
        return
    
    if args.test_mode:
        print("Running in test mode (no model inference)...")
        # Use test function
        test_iterative_generation(initial_scene_path, args.prompt, args.iterations, args.output, 
                                  asset_retrieval_module, asset_source=args.asset_source,
                                  objaverse_retriever=objaverse_retriever, 
                                  enable_visualization=enable_visualization)
    else:
        # Execute actual iterative generation
        try:
            all_scenes = iterative_scene_generation(
                initial_scene_path=initial_scene_path,
                user_prompt=args.prompt,
                engine=engine,
                request_config=request_config,
                asset_retrieval_module=asset_retrieval_module,
                num_iterations=args.iterations,
                output_dir=args.output,
                initial_conversation=initial_conversation,
                asset_source=args.asset_source,
                objaverse_retriever=objaverse_retriever,
                enable_visualization=enable_visualization,
                enable_physics_feedback=args.enable_physics_feedback,
                enable_vlm_feedback=args.enable_vlm_feedback,
                enable_physics_optimization=args.enable_physics_optimization,
                physics_opt_steps=args.physics_opt_steps,
                models_path=args.models_path if hasattr(args, 'models_path') else None
            )
            
            print(f"\n🎉 Generation completed successfully!")
            print(f"Generated {len(all_scenes)} scenes")
            print(f"Check output directory: {args.output}")
            
        except Exception as e:
            print(f"Error during iterative generation: {e}")
            import traceback
            traceback.print_exc()

def test_iterative_generation(initial_scene_path, user_prompt, num_iterations, output_dir, 
                              asset_retrieval_module=None, asset_source='3d-future',
                              objaverse_retriever=None, enable_visualization=False):
    """Test mode iterative generation (without using actual model)
    
    Args:
        asset_source: Asset source ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse asset retriever instance
        enable_visualization: Whether to enable 3D visualization guide lines
    """
    
    def mock_model_response(iteration, current_scene, conversation_history):
        """Simulate model response, considering conversation history to avoid repetition"""
        
        # Return terminate tool call on 3rd iteration, testing early stop functionality
        if iteration == 3:
            tool_calls = [
                {
                    "id": f"tool_{iteration}",
                    "name": "terminate",
                    "arguments": {
                        "reason": "Room design is now complete with a good balance of furniture and functionality. The space includes seating, table, and lighting which meets the user's requirements."
                    }
                }
            ]
            
            response = f"""I'll evaluate the current scene to determine if it meets the design requirements.

<think>
Looking at the current scene, we have added a modern sofa for seating and a coffee table. The room now has the essential furniture pieces for a functional living space. The layout is balanced and meets the basic requirements for a comfortable living room. I believe the design is complete and ready for use.
</think>

<conclusion>
The room design is now complete with all essential furniture pieces properly arranged for optimal functionality and aesthetics.
</conclusion>

<tool_calls>
{json.dumps(tool_calls, indent=2)}
</tool_calls>

The scene now provides a complete and functional living space that meets the user's requirements."""
            
            return response
        
        # Adjust response based on conversation history
        if iteration == 1:
            furniture_type = "modern sofa"
            action_desc = "adding a comfortable seating area"
        elif iteration == 2:
            furniture_type = "coffee table"
            action_desc = "complementing the sofa with a central table"
        elif iteration == 4:
            furniture_type = "floor lamp"
            action_desc = "improving lighting for the seating area"
        else:
            furniture_type = f"decorative item {iteration}"
            action_desc = f"adding finishing touches with decorative elements"
        
        # Simulate tool calls
        tool_calls = [
            {
                "id": f"tool_{iteration}",
                "name": "add_object",
                "arguments": {
                    "object_description": furniture_type,
                    "position": [iteration * 0.8, 0, iteration * 0.5],
                    "rotation": [0, 0, 0, 1],
                    "size": [1.2, 0.8, 1.0],
                    "group_name": f"Added {furniture_type.title()}"
                }
            }
        ]
        
        response = f"""I'll continue improving the room design based on our previous work.

<think>
Looking at the current scene and considering what we've done in previous iterations, I should focus on {action_desc}. The space needs better {furniture_type} placement to enhance functionality and aesthetics.
</think>

<conclusion>
In iteration {iteration}, {action_desc} to create a more cohesive and functional living space.
</conclusion>

<tool_calls>
{json.dumps(tool_calls, indent=2)}
</tool_calls>

This iteration builds upon our previous improvements to create a more complete living environment."""
        
        return response
    
    # Read initial scene
    with open(initial_scene_path, 'r', encoding='utf-8') as f:
        current_scene = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Render initial scene as first image
    print("Rendering initial scene...")
    initial_image_path = render_scene_to_image(current_scene, output_path, 0, enable_visualization=enable_visualization)
    print(f"Initial scene rendered to: {initial_image_path}")
    
    print(f"Starting test iterative generation with {num_iterations} iterations...")
    
    all_scenes = []
    
    # Save complete conversation history including user requests and model responses for each turn
    conversation_history = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"TEST ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # Build current iteration's user content
        current_scene_json = json.dumps(current_scene, indent=2)
        
        # Build base user content
        if iteration == 0:
            # First round: initial request
            base_user_content = f'{user_prompt}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        else:
            # Subsequent turns: continue optimizing scene
            base_user_content = f'Please continue to improve the scene based on the original request: "{user_prompt}"\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        
        current_user_message = f'<image>{base_user_content}'
        
        print(f"Current scene has {len(current_scene.get('groups', current_scene.get('objects', [])))} objects/groups")
        
        # Simulate model response (considering conversation history)
        response = mock_model_response(iteration + 1, current_scene, conversation_history)
        
        # Save response
        with open(output_path / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        
        # Add current turn's conversation to history
        conversation_history.append((current_user_message, response))
        
        # First try to extract tool_calls
        tool_calls = extract_tool_calls_from_response(response)
        final_scene = None
        
        if tool_calls is not None:
            print(f"Extracted {len(tool_calls)} tool calls")
            
            # Check for terminate tool call
            has_terminate = False
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get('name') == 'terminate':
                    has_terminate = True
                    terminate_reason = tool_call.get('arguments', {}).get('reason', 'No reason provided')
                    print(f"🛑 Terminate tool detected: {terminate_reason}")
                    break
            
            # If terminate tool found, stop iterations
            if has_terminate:
                print(f"Stopping test iterations early due to terminate tool call")
                # Save current scene as final scene
                scene_file_path = output_path / f"scene_iter_{iteration + 1}_final.json"
                with open(scene_file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_scene, f, indent=2, ensure_ascii=False)
                print(f"Final scene saved to: {scene_file_path}")
                break
            
            # Use scene_editor to apply tool calls to generate final scene
            final_scene = apply_tool_calls_to_scene(current_scene, tool_calls)
            print("Applied tool calls to generate final scene")
        else:
            # If no tool_calls found, try to extract final_scene (as fallback)
            print("No tool_calls found, trying to extract final_scene as fallback")
            final_scene = extract_final_scene_from_response(response)
        
        if final_scene is None:
            print(f"Failed to extract scene data from iteration {iteration + 1}")
            break
        
        print(f"Extracted final_scene with {len(final_scene.get('groups', final_scene.get('objects', [])))} objects/groups")
        
        # Retrieve required assets
        final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module,
                                                 asset_source=asset_source,
                                                 objaverse_retriever=objaverse_retriever)
        
        # Save current scene
        scene_file_path = output_path / f"scene_iter_{iteration + 1}.json"
        with open(scene_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_scene, f, indent=2, ensure_ascii=False)
        
        all_scenes.append(final_scene)
        
        # Render scene as image
        image_path = render_scene_to_image(final_scene, output_path, iteration + 1, enable_visualization=enable_visualization)
        print(f"Rendered image: {image_path}")
        
        # Update current scene for next iteration
        current_scene = final_scene
        
        print(f"Iteration {iteration + 1} completed successfully")
    
    # Save complete conversation history to file
    if conversation_history:
        with open(output_path / "conversation_history.txt", 'w', encoding='utf-8') as f:
            for i, (user_msg, assistant_msg) in enumerate(conversation_history, 1):
                f.write(f"=== Iteration {i} ===\n")
                f.write(f"User: {user_msg}\n\n")
                f.write(f"Assistant: {assistant_msg}\n\n")
                f.write("-" * 80 + "\n\n")
        print(f"Saved {len(conversation_history)} conversation turns to conversation_history.txt")
    
    print(f"\n{'='*50}")
    print("TEST GENERATION COMPLETED")
    print(f"{'='*50}")
    print(f"Total iterations: {len(all_scenes)}")
    print(f"Output directory: {output_path}")
    print(f"Conversation history: {output_path / 'conversation_history.txt'}")
    
    return all_scenes

# Further patch modelscope internal import_utils module's __getattr__, return False for __addon_enabled__ to avoid triggering remote imports
try:
    import importlib
    import sys
    
    # Patch modelscope.utils.import_utils
    imp_mod = importlib.import_module('modelscope.utils.import_utils')
    
    # Save original __getattr__
    _original_getattr = getattr(imp_mod, '__getattr__', None)
    
    def _safe_import_utils_getattr(name):
        # Return safe values for Blender addon related attributes
        if name in ('__addon_enabled__', '__addon_dependencies__', 'bl_info', 'register', 'unregister'):
            return False
        # For other attributes, try to return None instead of raising exceptions
        if _original_getattr:
            try:
                return _original_getattr(name)
            except ImportError:
                return None
        return None
    
    # Replace module-level __getattr__ with safe implementation
    imp_mod.__getattr__ = _safe_import_utils_getattr
    
    # Also patch modelscope main module
    modelscope_mod = sys.modules.get('modelscope')
    if modelscope_mod:
        _original_modelscope_getattr = getattr(modelscope_mod, '__getattr__', None)
        def _safe_modelscope_getattr(name):
            if name in ('__addon_enabled__', '__addon_dependencies__', 'bl_info', 'register', 'unregister'):
                return False
            if _original_modelscope_getattr:
                try:
                    return _original_modelscope_getattr(name)
                except ImportError:
                    return None
            raise AttributeError(f"module 'modelscope' has no attribute '{name}'")
        modelscope_mod.__getattr__ = _safe_modelscope_getattr
    
    print('Patched modelscope modules to be safe for Blender atexit')
except Exception as e:
    pass

if __name__ == "__main__":
    main()
