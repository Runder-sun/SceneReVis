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

# 设置 swift 和 vllm 的日志级别为 WARNING，减少输出
logging.getLogger("swift").setLevel(logging.WARNING)
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Add eval directory to path for physics optimization
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))

# Import helper functions for physics optimization (不导入 SceneOptimizer 以避免循环导入)
try:
    from eval.myeval import parse_scene_data, create_room_mesh, create_floor_polygon, get_object_field
    PHYSICS_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import eval utils: {e}. Physics optimization will be disabled.")
    PHYSICS_OPTIMIZATION_AVAILABLE = False

# Azure OpenAI imports
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, ManagedIdentityCredential, ChainedTokenCredential, get_bearer_token_provider

# 单卡配置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Azure OpenAI配置
AZURE_OPENAI_ENDPOINT = "YOUR_AZURE_OPENAI_ENDPOINT"
AZURE_OPENAI_DEPLOYMENT_NAME = "YOUR_DEPLOYMENT_NAME"
AZURE_OPENAI_API_VERSION = "2025-03-01-preview"
AZURE_OPENAI_SCOPE = "YOUR_AZURE_OPENAI_SCOPE"

# 设置必要的环境变量用于资产检索
# 自适应路径检测：优先使用远程服务器路径，回退到本地路径
def _find_3dfuture_assets_path():
    """查找 3D-FUTURE 资产路径"""
    possible_paths = [
        '/path/to/datasets/3d-front/3D-FUTURE-model',  # 远程服务器（优先）
        '/path/to/datasets/3d-front/3D-FUTURE-model',  # 本地开发
        '/path/to/amlt/datasets/3d-front/3D-FUTURE-model',  # AMLT 环境
        os.path.expanduser('~/datasets/3D-FUTURE-model'),
        './datasets/3D-FUTURE-model',
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found 3D-FUTURE assets at: {path}")
            return path
    print(f"Warning: 3D-FUTURE assets not found, using default: {possible_paths[0]}")
    return possible_paths[0]

os.environ['PTH_3DFUTURE_ASSETS'] = _find_3dfuture_assets_path()
os.environ['PTH_INVALID_ROOMS'] = './metadata/invalid_threed_front_rooms.txt'
os.environ['PTH_ASSETS_METADATA'] = './metadata/model_info_3dfuture_assets.json'
os.environ['PTH_ASSETS_METADATA_SCALED'] = './metadata/model_info_3dfuture_assets_scaled.json'
os.environ['PTH_ASSETS_METADATA_SIMPLE_DESCS'] = './metadata/model_info_3dfuture_assets_simple_descs.json'
os.environ['PTH_ASSETS_METADATA_PROMPTS'] = './metadata/model_info_3dfuture_assets_prompts.json'
os.environ['PTH_ASSETS_EMBED'] = './metadata/model_info_3dfuture_assets_embeds.pickle'

# 在可能的情况下，修补 modelscope 的动态导入函数，以避免在解释器退出时触发额外导入和异常
try:
    import modelscope
    # 将 try_import_from_hf 设置为安全的空实现（不抛异常）
    if hasattr(modelscope, 'try_import_from_hf'):
        modelscope.try_import_from_hf = lambda *a, **kw: None
    # 兼容某些版本的内部钩子名
    if hasattr(modelscope, '_extra_import_func'):
        modelscope._extra_import_func = lambda name: None
    print('Patched modelscope dynamic import hooks to be safe for atexit')
except Exception:
    # 如果 modelscope 未安装或修补失败，不要中断运行
    pass

from swift.llm import (
    PtEngine, VllmEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
from swift.plugin import InferStats
from typing import List

# 添加退出时的清理函数
def cleanup_on_exit():
    """程序退出时的清理函数"""
    try:
        # 清理可能导致冲突的模块
        import sys
        conflicting_modules = []
        
        # 收集需要清理的模块
        for module_name in list(sys.modules.keys()):
            if any(pattern in module_name.lower() for pattern in [
                'addon_utils', 'bpy.ops', 'blender', 
                'modelscope.utils.import_utils'
            ]):
                conflicting_modules.append(module_name)
        
        # 安全地移除这些模块
        for module_name in conflicting_modules:
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            except:
                pass
                
        # 尝试清理bpy场景数据
        if 'bpy' in sys.modules:
            try:
                import bpy
                bpy.ops.wm.read_factory_settings(use_empty=True)
            except:
                pass
    except:
        pass

# 注册退出清理函数，使用更安全的方式
import weakref
def safe_cleanup():
    try:
        cleanup_on_exit()
    except:
        pass

atexit.register(safe_cleanup)


# ============== 物理优化：规则式场景优化器 ==============

class RuleBasedSceneOptimizer:
    """
    规则式场景优化器，用于解决物体碰撞和出界问题。
    不依赖 LLM，通过规则进行位置调整和删除冲突物体。
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
        
        # 记录被删除的物品类别（用于反馈给模型）
        self.deleted_objects_desc = []  # 存储被删除物品的描述
    
    def get_object_category(self, idx):
        """获取物体的类别描述（用于反馈）"""
        obj = self.objects_data[idx]
        desc = get_object_field(obj, 'desc', self.format_type)
        if not desc:
            return 'unknown object'
        # 提取简短类别名（通常是描述的最后一两个单词或主要名词）
        # 简单处理：返回描述的前几个单词
        words = desc.split()
        if len(words) <= 3:
            return desc
        # 尝试提取核心名词（通常在末尾）
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
        rot = obj.get('rot', [0, 0, 0, 1])  # 默认四元数 [x, y, z, w]
        
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
        
        # 处理旋转 - 支持四元数和欧拉角两种格式
        try:
            if rot is None:
                rotation = R.from_quat([0, 0, 0, 1])  # 单位四元数
            elif isinstance(rot, (int, float)):
                # 单个值：Y轴旋转（弧度）
                rotation = R.from_euler('y', float(rot), degrees=False)
            elif isinstance(rot, list):
                if len(rot) == 4:
                    # 四元数格式 [x, y, z, w]
                    rot = [float(r) for r in rot]
                    rotation = R.from_quat(rot)
                elif len(rot) == 3:
                    # 欧拉角格式 [rx, ry, rz]（弧度）
                    rot = [float(r) for r in rot]
                    rotation = R.from_euler('xyz', rot, degrees=False)
                else:
                    rotation = R.from_quat([0, 0, 0, 1])
            else:
                rotation = R.from_quat([0, 0, 0, 1])
        except Exception:
            rotation = R.from_quat([0, 0, 0, 1])
        
        # 创建 box mesh（简化模型）
        box = trimesh.creation.box(extents=size)
        
        # 将 box 移动到底部中心为原点（类似 optimize_scene.py 的处理）
        bounds = box.bounds
        bottom_center_pivot = np.array([
            (bounds[0, 0] + bounds[1, 0]) / 2,
            bounds[0, 1],
            (bounds[0, 2] + bounds[1, 2]) / 2
        ])
        center_transform = np.eye(4)
        center_transform[:3, 3] = -bottom_center_pivot
        box.apply_transform(center_transform)
        
        # 构建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation.as_matrix()
        transform_matrix[:3, 3] = pos
        
        # 应用变换
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
        """获取被删除物品的反馈信息"""
        if not self.deleted_objects_desc:
            return ""
        # 去重
        unique_deleted = list(set(self.deleted_objects_desc))
        return f"[Physics Optimization] The following objects were removed due to collision/out-of-bounds: {', '.join(unique_deleted)}. Consider adding them back in better positions."


def apply_physics_optimization(scene_data: dict, models_path: str, max_steps: int = 5, azure_client=None) -> tuple:
    """
    对场景应用物理优化，解决碰撞和出界问题。
    
    Args:
        scene_data: 场景数据字典
        models_path: 3D模型路径
        max_steps: 最大优化步数
        azure_client: Azure OpenAI client，用于 GPT 咨询（可选）
        
    Returns:
        tuple: (优化后的场景数据, 删除物品反馈字符串)
    """
    if not PHYSICS_OPTIMIZATION_AVAILABLE:
        print("⚠ Physics optimization not available (missing dependencies)")
        return scene_data, ""
    
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scene_data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        # 确定格式类型
        format_type = 'ours'
        if 'room_envelope' not in scene_data and 'bounds_bottom' in scene_data:
            format_type = 'respace'
        
        # 如果没有提供 client，尝试创建一个
        if azure_client is None:
            try:
                azure_client = setup_azure_client()
                print("  Initialized Azure client for GPT-based object importance classification")
            except Exception as e:
                print(f"  Warning: Could not initialize Azure client: {e}")
                print("  Will use fallback strategy (keep all objects when resolving conflicts)")
        
        # 初始化优化器并运行（传递 client）
        optimizer = RuleBasedSceneOptimizer(temp_path, models_path, format_type=format_type, client=azure_client)
        optimized_objects = optimizer.optimize(max_steps=max_steps)
        
        # 获取删除物品的反馈
        deleted_feedback = optimizer.get_deleted_objects_feedback()
        if deleted_feedback:
            print(f"  Deleted objects feedback: {deleted_feedback}")
        
        # 更新场景数据
        result = copy.deepcopy(scene_data)
        result['objects'] = optimized_objects
        if 'groups' in result:
            del result['groups']
        
        # 清理临时文件
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
    使用 Azure GPT-5.1 生成简短的VLM布局反馈（不包含物理碰撞/出界问题）
    
    参数:
        image_path: 渲染图路径
        user_requirement: 用户需求
        azure_client: Azure OpenAI 客户端实例（如果为None则创建新的）
        
    返回:
        简短的布局反馈文本，如果没有问题或失败则返回空字符串
    """
    import base64
    
    if azure_client is None:
        azure_client = setup_azure_client()
    
    # 检查图像是否存在
    if not Path(image_path).exists():
        print(f"Warning: Image not found for VLM feedback: {image_path}")
        return ""
    
    # 将图像转换为base64
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
        
        # 如果VLM返回"no issues"类似的内容，返回空字符串
        if any(phrase in result.lower() for phrase in ["no issue", "looks good", "well-designed", "properly placed", "no problems"]):
            return ""
        
        print(f"VLM layout feedback: {result}")
        return result
        
    except Exception as e:
        print(f"Warning: Azure VLM layout feedback generation failed: {e}")
        return ""


# 导入渲染和资产检索相关模块
render_scene_with_bpy = None
AssetRetrievalModule = None

try:
    import sys
    # 添加utils路径
    utils_path = os.path.join(os.path.dirname(__file__), 'utils')
    if utils_path not in sys.path:
        sys.path.append(utils_path)
    
    # 导入智能Blender包装器（改为使用外部 Blender 进程渲染以隔离 bpy）
    from blender_wrapper import render_scene_blender_external as render_scene_with_bpy
    print("Successfully imported Blender external-process rendering wrapper")
    # 不再在主进程修补 bpy.addon_utils；外部进程会在自己的 Python 环境中运行
    
except Exception as e:
    print(f"Warning: Could not import Blender rendering wrapper: {e}")
    
    # Fallback渲染函数
    def render_scene_with_bpy(scene_data, output_dir):
        """最简单的fallback渲染函数"""
        print(f"Using simple fallback rendering for scene in {output_dir}")
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建简单的占位符文件
        (output_path / 'top').mkdir(exist_ok=True)
        (output_path / 'diag').mkdir(exist_ok=True)
        
        # 创建空图片文件作为占位符
        for view in ['top', 'diag']:
            placeholder_file = output_path / view / 'frame.png'
            with open(placeholder_file, 'w') as f:
                f.write("placeholder")
        
        return str(output_path)

try:
    # 使用 sample.py 中的高级资产检索模块
    from utils.sample import AssetRetrievalModule
    # 初始化参数与 sample.py 中的测试代码保持一致
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

# 导入 Objaverse 检索模块
objaverse_retrieval_module = None
try:
    from utils.objaverse_retriever import ObjaverseRetriever
    print("Successfully imported Objaverse retrieval module")
except Exception as e:
    print(f"Warning: Could not import Objaverse retrieval module: {e}")
    ObjaverseRetriever = None

# 导入 3D 可视化模块
try:
    from utils.visualization_3d import render_with_visualization
    print("Successfully imported 3D visualization module")
except Exception as e:
    print(f"Warning: Could not import 3D visualization module: {e}")
    render_with_visualization = None

# 导入场景编辑器
try:
    from utils.scene_editor import apply_tool_calls
    print("Successfully imported scene_editor module")
except Exception as e:
    print(f"Warning: Could not import scene_editor module: {e}")
    apply_tool_calls = None

# 导入格式转换函数和物理评估
try:
    from utils.RL_utils import convert_flat_to_grouped, convert_grouped_to_flat, TrimeshPhysicsMetrics, generate_physics_feedback
    print("Successfully imported format conversion functions and physics utils from RL_utils")
except Exception as e:
    print(f"Warning: Could not import format conversion functions: {e}")
except Exception as e:
    print(f"Warning: Could not import format conversion functions: {e}")
    # 如果导入失败，定义回退版本
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
    """读取JSON文件并返回格式化的字符串"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
        return json.dumps(scene_data, indent=2)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return "{}"

def extract_tool_calls_from_response(response_text):
    """从模型响应中提取<tool_calls>内容"""
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
    """从模型响应中提取<create_scene>内容
    
    支持两种格式：
    1. 带groups的格式 (有room_envelope和groups字段)
    2. 不带groups的格式 (有bounds_top/bounds_bottom和objects字段)
    """
    pattern = r'<create_scene>\s*```json\s*(.*?)\s*```\s*</create_scene>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        try:
            scene_json_str = match.group(1).strip()
            scene_data = json.loads(scene_json_str)
            
            # 验证基本字段
            if 'room_type' in scene_data and 'room_id' in scene_data:
                # 两种格式都有效
                if 'groups' in scene_data or 'objects' in scene_data:
                    format_type = "grouped" if 'groups' in scene_data else "flat"
                    print(f"Successfully extracted scene in {format_type} format")
                    return scene_data
            
            print("Warning: Scene data missing required fields (room_type, room_id, and groups/objects)")
            return scene_data  # 返回但发出警告
        except json.JSONDecodeError as e:
            print(f"Error parsing create_scene JSON: {e}")
            return None
    else:
        print("No <create_scene> found in response")
        return None

def extract_conclusion_from_response(response_text):
    """从模型响应中提取<conclusion>内容"""
    pattern = r'<conclusion>\s*(.*?)\s*</conclusion>'
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        conclusion = match.group(1).strip()
        return conclusion
    else:
        print("No <conclusion> found in response")
        return None

def extract_final_scene_from_response(response_text):
    """从模型响应中提取<final_scene>内容（保留作为备用）"""
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
    智能截断对话历史，确保剩余 token 空间足够生成响应。
    
    当 max_model_len - num_tokens < max_tokens 时，从前面删除轮次，
    直到有足够的空间用于生成。
    
    Args:
        conversation_history: 完整的对话历史列表 [(user_msg, assistant_msg), ...]
        current_user_message: 当前轮次的用户消息
        current_image_path: 当前图片路径
        engine: VllmEngine 实例（用于获取 tokenizer）
        max_model_len: 模型最大上下文长度
        max_tokens: 生成时需要保留的最大 token 数
        system_prompt: 系统提示词（可选）
    
    Returns:
        截断后的对话历史列表
    """
    if not conversation_history:
        return []
    
    # 获取 tokenizer
    tokenizer = engine.tokenizer
    
    def estimate_tokens(messages: list, image_path: str = None) -> int:
        """估算消息列表的 token 数量"""
        total_text = ""
        
        # 添加系统提示词
        if system_prompt:
            total_text += system_prompt + "\n"
        
        # 添加所有消息文本
        for msg in messages:
            content = msg.get('content', '')
            total_text += content + "\n"
        
        # 使用 tokenizer 编码文本
        try:
            tokens = tokenizer.encode(total_text, add_special_tokens=True)
            text_tokens = len(tokens)
        except Exception as e:
            # 如果编码失败，使用粗略估算（每个字符约1.5个token）
            text_tokens = int(len(total_text) * 1.5)
        
        # 图片 token 估算（Qwen2.5-VL 每张图片约 1280 tokens）
        image_tokens = 1280 if image_path else 0
        
        return text_tokens + image_tokens
    
    # 构建完整消息列表用于估算
    def build_messages(history: list) -> list:
        messages = []
        for hist_user_msg, hist_assistant_msg in history:
            messages.append({'role': 'user', 'content': hist_user_msg})
            messages.append({'role': 'assistant', 'content': hist_assistant_msg})
        messages.append({'role': 'user', 'content': current_user_message})
        return messages
    
    truncated_history = list(conversation_history)
    
    # 逐步删除最早的对话轮次，直到有足够空间
    while truncated_history:
        messages = build_messages(truncated_history)
        num_tokens = estimate_tokens(messages, current_image_path)
        remaining_tokens = max_model_len - num_tokens
        
        if remaining_tokens >= max_tokens:
            # 空间足够
            if len(truncated_history) < len(conversation_history):
                removed_count = len(conversation_history) - len(truncated_history)
                print(f"⚠ Smart truncation: removed {removed_count} oldest turns "
                      f"(tokens: {num_tokens}, remaining: {remaining_tokens}, required: {max_tokens})")
            break
        else:
            # 空间不足，删除最早的一轮对话
            if len(truncated_history) > 0:
                truncated_history.pop(0)
            else:
                # 已经没有历史可删除了
                print(f"⚠ Warning: Even without history, tokens ({num_tokens}) may exceed limit. "
                      f"Remaining: {remaining_tokens}, required: {max_tokens}")
                break
    
    # 如果删除了所有历史仍然不够，打印警告
    if not truncated_history and conversation_history:
        messages = build_messages([])
        num_tokens = estimate_tokens(messages, current_image_path)
        remaining_tokens = max_model_len - num_tokens
        if remaining_tokens < max_tokens:
            print(f"⚠ Warning: max_model_len({max_model_len}) - num_tokens({num_tokens}) = {remaining_tokens} < max_tokens({max_tokens})")
    
    return truncated_history

def generate_empty_room_with_model(room_prompt: str, engine, request_config, output_path: str = None) -> tuple[Dict[str, Any], tuple[str, str]]:
    """使用微调模型生成空房间结构
    
    Returns:
        tuple: (room_data, (user_message, assistant_message)) 或 (None, None) 如果失败
    """
    
    print(f"Generating empty room with fine-tuned model...")
    print(f"Room prompt: {room_prompt}")
    
    # 构建消息 - 只有用户的文本需求，没有图片
    user_message = room_prompt
    messages = [
        {
            'role': 'user',
            'content': user_message
        }
    ]
    
    # 创建推理请求（不需要图片）
    infer_requests = [
        InferRequest(messages=messages, images=None),
    ]
    
    try:
        # 执行推理
        print("Requesting model to generate initial scene structure...")
        resp_list = engine.infer(infer_requests, request_config)
        response = resp_list[0].choices[0].message.content
        
        print(f"Model response length: {len(response)} characters")
        
        # 保存模型响应（用于调试）
        if output_path:
            response_file = Path(output_path).parent / "initial_scene_generation_response.txt"
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Model response saved to: {response_file}")
        
        # 提取<create_scene>内容
        room_data = extract_create_scene_from_response(response)
        
        if room_data is None:
            print("Failed to extract scene data from model response")
            return None, None
        
        # 验证生成的房间结构
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
        
        # **重要**：转换为flat格式后再保存
        room_data_to_save = convert_grouped_to_flat(room_data) if 'groups' in room_data else room_data
        
        # 保存到文件（如果指定了输出路径）
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(room_data_to_save, f, indent=2, ensure_ascii=False)
            print(f"Empty room saved to: {output_file} (flat format)")
        
        # 返回flat格式的场景数据和对话历史
        return room_data_to_save, (user_message, response)
        
    except Exception as e:
        print(f"Error generating room with model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_empty_room(room_prompt: str, output_path: str = None) -> Dict[str, Any]:
    """使用Azure OpenAI生成空房间结构（仅包含房间边界，不包含物体）"""
    
    # 设置Azure OpenAI客户端
    try:
        client = setup_azure_client()
    except Exception as e:
        print(f"Failed to setup Azure OpenAI client: {e}")
        return None
    
    # 构建系统提示
    system_prompt = """You are an expert interior designer and room layout specialist. Your task is to create an empty room structure based on user requirements.

You must respond with a JSON structure that includes:
1. room_type: The type of room (bedroom, living_room, kitchen, etc.)
2. room_id: A unique identifier for the room
3. bounds_top and bounds_bottom: The physical boundaries of the room
4. objects: An empty array (objects will be added later)

Guidelines:
- Room dimensions should be realistic for the specified room type
- bounds_top and bounds_bottom should define a rectangular room with 4 corner points
- Each bounds array contains 4 points: [[x1,y,z1], [x2,y,z2], [x3,y,z3], [x4,y,z4]]
- bounds_bottom has y=0.0 (floor level), bounds_top has y=2.6 (ceiling level)
- Points should form a rectangle in clockwise or counter-clockwise order

ROOM SIZE REQUIREMENTS (STRICTLY FOLLOW):
- bedroom: 10-20 square meters (e.g., 3.5m x 4m = 14 sqm, or 4m x 5m = 20 sqm)
- livingroom/living_room: 15-35 square meters (e.g., 4m x 5m = 20 sqm, or 5m x 6m = 30 sqm)
- diningroom/dining_room: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- studyroom/study_room/office: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- Calculate floor area as: width (x-axis) × depth (z-axis)

Example format:
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

Respond ONLY with valid JSON, no additional text."""

    # 构建用户提示
    user_prompt = f"""Create an empty room structure for: {room_prompt}

Please generate a room that would be suitable for this description. Keep the objects list empty."""

    try:
        # 调用Azure OpenAI API
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_completion_tokens=2000
        )
        
        # 提取响应内容
        response_content = response.choices[0].message.content.strip()
        print(f"Generated room response length: {len(response_content)} characters")
        
        # 尝试解析JSON响应
        try:
            # 如果响应包含```json标记，提取其中的JSON部分
            if "```json" in response_content:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            room_data = json.loads(response_content)
            
            # 验证生成的房间结构 - 检查是否有bounds
            if 'bounds_top' not in room_data or 'bounds_bottom' not in room_data:
                print(f"Warning: Generated room missing bounds_top or bounds_bottom")
                return None
            
            # 确保有objects字段
            if 'objects' not in room_data:
                room_data['objects'] = []
            
            # **重要**：转换为flat格式后再保存
            room_data_to_save = convert_grouped_to_flat(room_data) if 'groups' in room_data else room_data
            
            # 保存到文件（如果指定了输出路径）
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
    """使用Azure OpenAI生成包含物体的完整场景（扁平格式）
    
    Args:
        room_prompt: 用户的房间需求描述
        output_path: 输出文件路径（可选）
        asset_retrieval_module: 3D-FUTURE资产检索模块
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse资产检索模块
        
    Returns:
        Dict: 包含物体的完整场景数据（扁平格式），物体已完成资产检索
    """
    
    # 设置Azure OpenAI客户端
    try:
        client = setup_azure_client()
    except Exception as e:
        print(f"Failed to setup Azure OpenAI client: {e}")
        return None
    
    # 构建系统提示 - 生成带物体的完整场景
    system_prompt = """You are an expert interior designer and room layout specialist. Your task is to create a complete room layout with furniture based on user requirements.

You must respond with a JSON structure in FLAT FORMAT that includes:
1. room_type: The type of room (bedroom, livingroom, diningroom, studyroom, etc.)
2. room_id: A unique identifier for the room (e.g., "Bedroom-1234")
3. bounds_top: 4 corner points of the ceiling [[x1,y,z1], [x2,y,z2], [x3,y,z3], [x4,y,z4]]
4. bounds_bottom: 4 corner points of the floor (same x,z as bounds_top, but y=0.0)
5. objects: An array of furniture objects, each with:
   - desc: Detailed description of the object (style, material, color, design features)
   - size: [width_x, height_y, depth_z] in meters (estimated realistic size)
   - pos: [x, y, z] position in room coordinates (y=0.0 for floor-standing objects)
   - rot: [x, y, z, w] quaternion rotation (use [0,0,0,1] for no rotation, [0,0.70711,0,0.70711] for 90° Y-axis rotation)

ROOM SIZE REQUIREMENTS (STRICTLY FOLLOW):
- bedroom: 10-20 square meters (e.g., 3.5m x 4m = 14 sqm, or 4m x 5m = 20 sqm)
- livingroom: 15-35 square meters (e.g., 4m x 5m = 20 sqm, or 5m x 6m = 30 sqm)
- diningroom: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- studyroom/office: 10-25 square meters (e.g., 3m x 4m = 12 sqm, or 4m x 5m = 20 sqm)
- Calculate floor area as: width (x-axis) × depth (z-axis)

IMPORTANT GUIDELINES:
- bounds_bottom has y=0.0 (floor level), bounds_top has y=2.6 (ceiling level)
- Place objects logically: beds against walls, sofas facing TV, desks near windows
- Avoid overlapping objects - leave adequate spacing between furniture
- Objects should be within room bounds (check x, z coordinates against bounds)
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

Example object:
{
  "desc": "Modern minimalist queen-sized bed with upholstered gray fabric headboard and wooden frame",
  "size": [1.6, 0.5, 2.0],
  "pos": [-1.5, 0.0, 0.0],
  "rot": [0, 0.70711, 0, 0.70711]
}

Respond ONLY with valid JSON, no additional text."""

    # 构建用户提示
    user_prompt = f"""Create a complete furnished room for: {room_prompt}

Please generate a realistic room layout with appropriate furniture. Include detailed descriptions for each piece of furniture to enable accurate asset retrieval."""

    try:
        print(f"Generating scene with GPT for: {room_prompt}")
        
        # 调用Azure OpenAI API
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_completion_tokens=4000
        )
        
        # 提取响应内容
        response_content = response.choices[0].message.content.strip()
        print(f"Generated scene response length: {len(response_content)} characters")
        
        # 尝试解析JSON响应
        try:
            # 如果响应包含```json标记，提取其中的JSON部分
            if "```json" in response_content:
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            scene_data = json.loads(response_content)
            
            # 验证生成的场景结构
            if 'bounds_top' not in scene_data or 'bounds_bottom' not in scene_data:
                print(f"Warning: Generated scene missing bounds_top or bounds_bottom")
                return None
            
            if 'objects' not in scene_data or not scene_data['objects']:
                print(f"Warning: Generated scene has no objects")
                return None
            
            # 转换为flat格式（如果需要）
            if 'groups' in scene_data:
                scene_data = convert_grouped_to_flat(scene_data)
            
            print(f"GPT generated scene with {len(scene_data.get('objects', []))} objects")
            
            # 进行资产检索
            print("Starting asset retrieval for generated objects...")
            scene_data = retrieve_and_update_assets(
                scene_data, 
                asset_retrieval_module=asset_retrieval_module,
                asset_source=asset_source,
                objaverse_retriever=objaverse_retriever
            )
            
            # 保存到文件（如果指定了输出路径）
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
    """为场景中的物体检索资产ID，并用retrieved_size替换原始size
    
    Args:
        scene_data: 场景数据（扁平格式）
        asset_retrieval_module: 3D-FUTURE资产检索模块
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse资产检索模块
        
    Returns:
        Dict: 更新后的场景数据，包含uid和更新后的size
    """
    if not scene_data or 'objects' not in scene_data:
        return scene_data
    
    print(f"Retrieving assets for {len(scene_data['objects'])} objects (source: {asset_source})...")
    
    updated_scene = None
    
    # 根据资产来源选择检索策略
    if asset_source == 'objaverse' and objaverse_retriever:
        try:
            updated_scene = objaverse_retriever.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Asset retrieval completed using Objaverse")
        except Exception as e:
            print(f"Error during Objaverse asset retrieval: {e}")
            print("Falling back to 3D-FUTURE retrieval...")
    
    elif asset_source == 'auto' and objaverse_retriever and asset_retrieval_module:
        # 混合模式
        try:
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            # 检查是否有未检索到的物体
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
    
    # 用retrieved_size替换原始size
    objects_updated = 0
    for obj in updated_scene.get('objects', []):
        if 'retrieved_size' in obj and obj['retrieved_size']:
            original_size = obj.get('size', [])
            obj['size'] = obj['retrieved_size']
            objects_updated += 1
            if objects_updated <= 3:  # 只打印前3个作为示例
                print(f"  Updated size for '{obj.get('desc', 'Unknown')[:40]}...': {original_size} -> {obj['size']}")
    
    if objects_updated > 3:
        print(f"  ... and {objects_updated - 3} more objects updated")
    
    print(f"Updated sizes for {objects_updated}/{len(updated_scene.get('objects', []))} objects")
    
    return updated_scene

def apply_tool_calls_to_scene(initial_scene, tool_calls):
    """使用 scene_editor 的 apply_tool_calls 函数来修改场景
    
    这个函数会自动处理格式转换：
    1. 如果输入是不带groups的格式，先转换为带groups的格式
    2. 应用tool_calls修改场景  
    3. **始终返回flat格式**（不带groups）
    """
    if not apply_tool_calls:
        print("Warning: scene_editor not available, using fallback")
        return initial_scene
    
    try:
        # 检测输入格式
        is_flat_format = 'objects' in initial_scene and 'groups' not in initial_scene
        
        # 如果是不带groups的格式，先转换为带groups的格式
        if is_flat_format:
            print("Detected flat format (without groups), converting to grouped format for editing...")
            scene_for_editing = convert_flat_to_grouped(initial_scene)
        else:
            scene_for_editing = initial_scene
        
        # 使用 scene_editor 模块的 apply_tool_calls 函数
        edited_scene = apply_tool_calls(scene_for_editing, tool_calls)
        
        # **关键**：始终转换回flat格式返回
        final_scene = convert_grouped_to_flat(edited_scene) if 'groups' in edited_scene else edited_scene
        print("Returning scene in flat format (without groups)")
        
        return final_scene
    except Exception as e:
        print(f"Error applying tool calls: {e}")
        import traceback
        traceback.print_exc()
        return initial_scene

def check_and_retrieve_assets(scene_data, asset_retrieval_module=None, asset_source='3d-future', objaverse_retriever=None):
    """检查场景中是否有需要检索的资产，并进行检索
    
    支持两种格式：
    1. 带groups的格式 (groups -> objects)
    2. 不带groups的格式 (直接的objects数组)
    
    参数：
    - scene_data: 场景数据
    - asset_retrieval_module: 3D-FUTURE 资产检索模块
    - asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
    - objaverse_retriever: Objaverse 资产检索模块
    """
    if not scene_data:
        return scene_data
    
    print(f"Checking for assets that need retrieval (source: {asset_source})...")
    
    # 根据资产来源选择检索策略
    if asset_source == 'objaverse' and objaverse_retriever:
        try:
            # 使用 Objaverse 检索模块
            updated_scene = objaverse_retriever.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Assets retrieval completed using Objaverse")
            return updated_scene
        except Exception as e:
            print(f"Error during Objaverse asset retrieval: {e}")
            # 回退到 3D-FUTURE
            print("Falling back to 3D-FUTURE retrieval...")
    
    elif asset_source == 'auto' and objaverse_retriever and asset_retrieval_module:
        # 混合模式：尝试 3D-FUTURE，失败则使用 Objaverse
        try:
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            
            # 检查是否有检索失败的物体（保持 <NEED_RETRIEVAL>）
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
    
    # 默认：使用 3D-FUTURE 检索模块
    if asset_retrieval_module:
        try:
            # 使用sample_all_assets方法，这与respace.py中的逻辑一致
            updated_scene = asset_retrieval_module.sample_all_assets(scene_data, is_greedy_sampling=True)
            print("Assets retrieval completed using sample_all_assets (3D-FUTURE)")
            return updated_scene
        except Exception as e:
            print(f"Error during asset retrieval: {e}")
            # 继续到后备逻辑
    
    # 后备逻辑：手动处理<NEED_RETRIEVAL>
    print("Using fallback asset retrieval logic...")
    modified = False
    
    # 处理带groups的格式
    if 'groups' in scene_data:
        for group in scene_data.get('groups', []):
            for obj in group.get('objects', []):
                if obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>'] or not obj.get('jid'):
                    print(f"Need to retrieve asset for object: {obj.get('desc', 'Unknown')}")
                    # 生成一个占位符jid
                    obj['jid'] = str(uuid.uuid4())
                    print(f"Using placeholder jid: {obj['jid']}")
                    modified = True
    
    # 处理不带groups的格式（直接的objects数组）
    elif 'objects' in scene_data:
        for obj in scene_data.get('objects', []):
            if obj.get('jid') in ['<NEED_RETRIEVAL>', '<NEED_RETRIVEAL>'] or not obj.get('jid'):
                print(f"Need to retrieve asset for object: {obj.get('desc', 'Unknown')}")
                # 生成一个占位符jid
                obj['jid'] = str(uuid.uuid4())
                print(f"Using placeholder jid: {obj['jid']}")
                modified = True
    
    if modified:
        print("Fallback assets retrieval completed")
    else:
        print("No assets needed retrieval")
    
    return scene_data


def _predownload_objaverse_glbs(scene_data):
    """在渲染前预先下载 Objaverse GLB 文件
    
    这样 Blender 子进程只需要从本地缓存读取，不需要安装 objaverse 包
    """
    try:
        from utils.objaverse_glb_manager import get_objaverse_glb_path
    except ImportError:
        print("Warning: objaverse_glb_manager not available, skipping GLB pre-download")
        return
    
    # 提取所有需要下载的 Objaverse UIDs
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
    """使用Blender渲染场景并返回合并图片路径
    
    Args:
        scene_data: 场景数据
        output_dir: 输出目录
        iteration: 迭代次数
        enable_visualization: 是否启用3D可视化辅助线
    """
    try:
        print(f"Rendering scene for iteration {iteration}...")
        
        # 预先下载 Objaverse GLB 文件（在主进程中下载，避免 Blender 子进程需要安装 objaverse 包）
        _predownload_objaverse_glbs(scene_data)
        
        # 创建临时输出目录
        temp_output_dir = Path(output_dir) / f"temp_render_{iteration}"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置环境变量控制Blender输出
        os.environ['BPY_VERBOSE'] = '0'  # 减少输出
        # 不强制使用占位符，让Blender渲染器尝试加载真实的3D模型
        os.environ['BPY_USE_PLACEHOLDER_ONLY'] = '0'
        # 设置3D可视化环境变量
        os.environ['BPY_ENABLE_VISUALIZATION'] = '1' if enable_visualization else '0'
        
        # 使用Blender渲染包装器
        if render_scene_with_bpy:
            scene_id = f"scene_iter_{iteration}"
            render_result = render_scene_with_bpy(scene_data, temp_output_dir, scene_id)
            print(f"Blender rendering completed: {render_result}")
        else:
            print("No rendering function available, creating placeholder")
            
        # 查找生成的图片文件
        top_file = temp_output_dir / "top" / "frame.png"
        diag_file = temp_output_dir / "diag" / "frame.png"
        
        if top_file.exists() and diag_file.exists():
            try:
                # 使用带边界框和标签的图像合并函数
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "image_merger", 
                    str(Path(__file__).parent / 'utils' / 'image_merger.py')
                )
                image_merger = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(image_merger)
                merge_rendered_views_with_annotations = image_merger.merge_rendered_views_with_annotations
                
                # 保存合并图片
                merged_path = Path(output_dir) / f"merged_iter_{iteration}.png"
                merge_rendered_views_with_annotations(str(top_file), str(diag_file), str(merged_path))
                
                print(f"Rendered and merged image saved to: {merged_path}")
                
                # 清理临时目录
                try:
                    shutil.rmtree(temp_output_dir)
                except:
                    pass
                
                return str(merged_path)
                
            except Exception as img_error:
                print(f"Error processing rendered images: {img_error}")
                import traceback
                traceback.print_exc()
                # 继续到创建占位符图片
        
        # 如果渲染失败或图片处理失败，创建占位符图片
        print("Creating placeholder image...")
        try:
            from PIL import Image, ImageDraw
            
            # 创建占位符图片
            img = Image.new('RGB', (1024, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # 计算对象数量
            objects_count = 0
            if 'groups' in scene_data:
                objects_count = sum(len(group.get('objects', [])) for group in scene_data.get('groups', []))
            elif 'objects' in scene_data:
                objects_count = len(scene_data.get('objects', []))
            
            # 绘制信息
            text = f"Iteration {iteration}\n{objects_count} objects/groups\nPlaceholder Image"
            draw.text((50, 200), text, fill='black')
            
            # 保存占位符图片
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
    """将所有迭代的图片合成为一张大图"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        print("Creating iteration summary image...")
        
        # 过滤出存在的图片文件
        valid_images = []
        for i, img_path in enumerate(all_image_paths):
            if Path(img_path).exists():
                valid_images.append((i, img_path))
            else:
                print(f"Warning: Image {img_path} does not exist")
        
        if not valid_images:
            print("No valid images found for summary")
            return None
        
        # 加载第一张图片来获取尺寸
        first_img = Image.open(valid_images[0][1])
        img_width, img_height = first_img.size
        
        # 计算网格布局 - 尽量接近正方形
        import math
        cols = math.ceil(math.sqrt(len(valid_images)))
        rows = math.ceil(len(valid_images) / cols)
        
        # 计算合成图片的尺寸
        margin = 20
        label_height = 40
        cell_width = img_width + margin
        cell_height = img_height + label_height + margin
        
        total_width = cols * cell_width + margin
        total_height = rows * cell_height + margin + 60  # 额外空间给标题
        
        # 创建大图 - 使用RGBA模式支持透明通道，背景完全透明
        summary_img = Image.new('RGBA', (total_width, total_height), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(summary_img)
        
        # 尝试加载字体
        try:
            # 尝试使用系统字体
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            try:
                # 备用字体
                font_title = ImageFont.load_default()
                font_label = ImageFont.load_default()
            except:
                font_title = None
                font_label = None
        
        # 绘制标题
        title = f"Scene Generation Progress ({len(valid_images)} iterations)"
        if font_title:
            title_bbox = draw.textbbox((0, 0), title, font=font_title)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = (total_width - title_width) // 2
        else:
            title_x = total_width // 2 - len(title) * 6
        
        draw.text((title_x, 20), title, fill=(0, 0, 0, 255), font=font_title)
        
        # 绘制每个迭代的图片
        for idx, (iter_num, img_path) in enumerate(valid_images):
            row = idx // cols
            col = idx % cols
            
            # 计算位置
            x = col * cell_width + margin
            y = row * cell_height + margin + 60  # 60是标题区域的高度
            
            # 加载并粘贴图片
            try:
                img = Image.open(img_path)
                # 转换为RGBA模式以确保透明度支持
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                # 如果图片尺寸不匹配，调整大小
                if img.size != (img_width, img_height):
                    img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # 使用alpha合成粘贴图片，保持透明度
                summary_img.paste(img, (x, y), img)
                
                # 添加标签
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
                # 绘制占位符
                draw.rectangle([x, y, x + img_width, y + img_height], fill=(211, 211, 211, 255), outline=(128, 128, 128, 255))
                error_text = f"Error: {iter_num}"
                draw.text((x + 10, y + img_height // 2), error_text, fill=(255, 0, 0, 255), font=font_label)
        
        # 保存合成图片 - 保存为PNG格式并保留透明通道
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
    """执行迭代式场景生成
    
    Args:
        initial_conversation: tuple (user_message, assistant_message) from initial scene generation
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse 资产检索器实例
        enable_visualization: 是否启用3D可视化辅助线
        enable_physics_feedback: 是否启用物理反馈注入
        enable_vlm_feedback: 是否启用VLM布局反馈注入
        enable_physics_optimization: 是否启用物理优化（碰撞/出界修复）
        physics_opt_steps: 物理优化最大步数
        models_path: 3D模型路径
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 读取初始场景
    with open(initial_scene_path, 'r', encoding='utf-8') as f:
        current_scene = json.load(f)
    
    # 保存所有图片路径用于最后的合成
    all_image_paths = []
    
    # 渲染初始场景作为第一张图片，而不是使用硬编码路径
    print("Rendering initial scene...")
    current_image_path = render_scene_to_image(current_scene, output_path, 0, enable_visualization=enable_visualization)
    all_image_paths.append(current_image_path)
    print(f"Initial scene rendered to: {current_image_path}")
    
    print(f"Starting iterative scene generation with {num_iterations} iterations...")
    print(f"Initial prompt: {user_prompt}")
    
    # 保存所有生成的场景
    all_scenes = []
    
    # 保存完整的对话历史，包含每轮的用户请求和模型响应
    conversation_history = []
    
    # 如果有初始对话（来自场景生成），添加到历史中
    if initial_conversation is not None:
        conversation_history.append(initial_conversation)
        print(f"Added initial scene generation conversation to history")
    
    # 初始化反馈生成所需的组件
    last_feedback = ""  # 存储上一轮生成的反馈
    trimesh_metrics_instance = None
    azure_client_for_feedback = None
    
    # 尝试初始化物理评估器
    try:
        trimesh_metrics_instance = TrimeshPhysicsMetrics(verbose=False)
        print("Initialized TrimeshPhysicsMetrics for feedback generation")
    except Exception as e:
        print(f"Warning: Could not initialize TrimeshPhysicsMetrics: {e}")
    
    # 尝试初始化Azure客户端（用于VLM布局反馈）
    try:
        azure_client_for_feedback = setup_azure_client()
        print("Initialized Azure client for VLM layout feedback")
    except Exception as e:
        print(f"Warning: Could not initialize Azure client for feedback: {e}")
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # 构建当前迭代的用户内容
        current_scene_json = json.dumps(current_scene, indent=2)
        
        # 构建基础用户内容（包含反馈，如果有的话）
        if iteration == 0:
            # 第一轮：初始请求（无反馈）
            base_user_content = f'{user_prompt}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        else:
            # 后续轮次：继续优化场景，包含反馈
            feedback_section = ""
            if last_feedback:
                feedback_section = f'\n<feedback>\n{last_feedback}\n</feedback>\n'
                print(f"Injecting feedback: {last_feedback[:100]}...")
            base_user_content = f'Please continue to improve the scene based on the original request: "{user_prompt}"{feedback_section}\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        
        # 构建完整的消息列表，包含历史对话
        messages = []
        
        # 添加当前轮次的用户消息
        current_user_message = f'<image>{base_user_content}'
        
        # 使用智能截断：基于 token 数量而非固定轮次
        # 配置：max_model_len=40960, max_tokens=16384
        truncated_history = smart_truncate_conversation_history(
            conversation_history=conversation_history,
            current_user_message=current_user_message,
            current_image_path=current_image_path,
            engine=engine,
            max_model_len=40960,
            max_tokens=request_config.max_tokens if request_config.max_tokens else 16384
        )
        
        # 如果有对话历史，添加到消息列表中
        if truncated_history:
            # 添加历史对话
            for hist_user_msg, hist_assistant_msg in truncated_history:
                messages.append({'role': 'user', 'content': hist_user_msg})
                messages.append({'role': 'assistant', 'content': hist_assistant_msg})
        
        messages.append({'role': 'user', 'content': current_user_message})
        
        # 创建推理请求
        infer_requests = [
            InferRequest(messages=messages,
                        images=[current_image_path]),
        ]
        
        # 执行推理
        print("Generating response from model...")
        resp_list = engine.infer(infer_requests, request_config)
        response = resp_list[0].choices[0].message.content
        
        print(f"Response length: {len(response)} characters")
        
        # 保存响应
        with open(output_path / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        
        # 将当前轮次的对话添加到历史中
        conversation_history.append((current_user_message, response))
        
        # 首先尝试提取 tool_calls
        tool_calls = extract_tool_calls_from_response(response)
        final_scene = None
        
        if tool_calls is not None:
            print(f"Extracted {len(tool_calls)} tool calls")
            
            # 检查是否有terminate工具调用
            has_terminate = False
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get('name') == 'terminate':
                    has_terminate = True
                    terminate_reason = tool_call.get('arguments', {}).get('reason', 'No reason provided')
                    print(f"🛑 Terminate tool detected: {terminate_reason}")
                    break
            
            # 如果发现terminate工具，停止迭代
            if has_terminate:
                print(f"Stopping iterations early due to terminate tool call")
                # **重要**：转换为flat格式后再保存
                scene_to_save = convert_grouped_to_flat(current_scene) if 'groups' in current_scene else current_scene
                
                # 应用物理优化（碰撞/出界修复）- 最后一轮也要优化
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
            
            # 使用 scene_editor 应用工具调用来生成最终场景
            final_scene = apply_tool_calls_to_scene(current_scene, tool_calls)
            print("Applied tool calls to generate final scene")
        else:
            # 如果没有找到 tool_calls，尝试提取 final_scene（作为备用）
            print("No tool_calls found, trying to extract final_scene as fallback")
            final_scene = extract_final_scene_from_response(response)
        
        if final_scene is None:
            print(f"⚠ No executable commands found in iteration {iteration + 1}. Ending generation and saving current state.")
            
            # 保存当前状态为 final
            scene_to_save = convert_grouped_to_flat(current_scene) if 'groups' in current_scene else current_scene
            
            # 应用物理优化（碰撞/出界修复）- 最后一轮也要优化
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
        
        # 检索需要的资产
        final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module, 
                                                 asset_source=asset_source, 
                                                 objaverse_retriever=objaverse_retriever)
        
        # 应用物理优化（碰撞/出界修复）
        physics_deleted_feedback = ""
        if enable_physics_optimization:
            print(f"Applying physics optimization...")
            final_scene, physics_deleted_feedback = apply_physics_optimization(
                final_scene, 
                models_path=models_path or "", 
                max_steps=physics_opt_steps
            )
        
        # **重要**：转换为flat格式（不带groups）后再保存
        final_scene_to_save = convert_grouped_to_flat(final_scene) if 'groups' in final_scene else final_scene
        
        # 保存当前场景（flat格式）
        scene_file_path = output_path / f"scene_iter_{iteration + 1}.json"
        with open(scene_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_scene_to_save, f, indent=2, ensure_ascii=False)
        
        all_scenes.append(final_scene_to_save)
        
        # 渲染场景为图片（使用flat格式）
        current_image_path = render_scene_to_image(final_scene_to_save, output_path, iteration + 1, enable_visualization=enable_visualization)
        all_image_paths.append(current_image_path)
        
        # 更新当前场景为下一次迭代做准备（使用flat格式）
        current_scene = final_scene_to_save
        
        # ===== 生成反馈用于下一轮 =====
        # 只在非最后一轮生成反馈
        if iteration < num_iterations - 1:
            feedback_parts = []
            
            # 0. 物理优化删除物品反馈（优先添加）
            if physics_deleted_feedback:
                feedback_parts.append(physics_deleted_feedback)
                print(f"Physics deleted feedback: {physics_deleted_feedback}")
            
            # 1. 物理反馈（来自trimesh）
            if enable_physics_feedback and trimesh_metrics_instance is not None:
                try:
                    # 使用grouped格式进行物理评估
                    scene_for_eval = convert_flat_to_grouped(final_scene_to_save) if 'objects' in final_scene_to_save else final_scene_to_save
                    trimesh_metrics = trimesh_metrics_instance.evaluate_scene(scene_for_eval, format_type='ours')
                    physics_feedback = generate_physics_feedback(trimesh_metrics, top_k=3)
                    if physics_feedback:
                        feedback_parts.append(physics_feedback)
                        print(f"Physics feedback: {physics_feedback}")
                except Exception as e:
                    print(f"Warning: Physics feedback generation failed: {e}")
            
            # 2. VLM布局反馈（来自Azure GPT-5.1）
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
            
            # 组合反馈
            if feedback_parts:
                last_feedback = " ".join(feedback_parts)
                print(f"Combined feedback for next iteration: {last_feedback}")
            else:
                last_feedback = ""
        
        print(f"Iteration {iteration + 1} completed successfully")
    
    # 保存完整的对话历史到文件
    if conversation_history:
        with open(output_path / "conversation_history.txt", 'w', encoding='utf-8') as f:
            for i, (user_msg, assistant_msg) in enumerate(conversation_history, 1):
                f.write(f"=== Iteration {i} ===\n")
                f.write(f"User: {user_msg}\n\n")
                f.write(f"Assistant: {assistant_msg}\n\n")
                f.write("-" * 80 + "\n\n")
        print(f"Saved {len(conversation_history)} conversation turns to conversation_history.txt")
    
    # 生成迭代过程合成图
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
    """并行批量迭代场景生成 - 在每个迭代步骤中同时处理所有prompt
    
    Args:
        prompts: prompts列表
        engine: 推理引擎
        request_config: 请求配置
        initial_scene_path: 初始场景JSON文件路径（如果不生成空房间）
        asset_retrieval_module: 资产检索模块
        num_iterations: 每个prompt的迭代次数
        output_base_dir: 输出基础目录
        generate_room: 是否生成空房间
        use_model_for_creation: 是否使用模型生成初始场景
        use_gpt_with_objects: 是否使用GPT生成带物体的完整场景
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse 资产检索器实例
        enable_visualization: 是否启用3D可视化辅助线
        max_batch_size: 并行推理的最大批处理大小，防止OOM
        max_history_turns: 保留的最大对话历史轮数
        enable_physics_feedback: 是否启用物理反馈注入
        enable_vlm_feedback: 是否启用VLM布局反馈注入
        enable_physics_optimization: 是否启用物理优化（碰撞/出界修复）
        physics_opt_steps: 物理优化最大步数
        models_path: 3D模型路径
        original_indices: 原始prompt索引列表(1-indexed)，用于跳过已完成场景时保持索引一致
    
    Returns:
        所有场景的结果列表
    """
    # 如果没有提供原始索引，使用默认的1到N
    if original_indices is None:
        original_indices = list(range(1, len(prompts) + 1))
    
    print(f"\n{'='*60}")
    print(f"PARALLEL BATCH ITERATIVE SCENE GENERATION")
    print(f"Processing {len(prompts)} prompts in parallel with {num_iterations} iterations each")
    print(f"{'='*60}\n")
    
    # 初始化反馈生成所需的组件
    trimesh_metrics_instance = None
    azure_client_for_feedback = None
    
    # 尝试初始化物理评估器
    try:
        trimesh_metrics_instance = TrimeshPhysicsMetrics(verbose=False)
        print("Initialized TrimeshPhysicsMetrics for feedback generation")
    except Exception as e:
        print(f"Warning: Could not initialize TrimeshPhysicsMetrics: {e}")
    
    # 尝试初始化Azure客户端（用于VLM布局反馈）
    try:
        azure_client_for_feedback = setup_azure_client()
        print("Initialized Azure client for VLM layout feedback")
    except Exception as e:
        print(f"Warning: Could not initialize Azure client for feedback: {e}")
    
    # 创建收集文件夹（提前创建，每完成一个就立即收集）
    output_base_path = Path(output_base_dir)
    final_scenes_dir = output_base_path / "final_scenes_collection"
    final_renders_dir = output_base_path / "final_renders_collection"
    final_scenes_dir.mkdir(parents=True, exist_ok=True)
    final_renders_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created collection directories:")
    print(f"  Final scenes: {final_scenes_dir}")
    print(f"  Final renders: {final_renders_dir}")
    
    # 用于跟踪收集数量
    collected_scenes_count = 0
    collected_renders_count = 0
    
    # 为每个prompt准备输出目录和初始化数据
    prompt_contexts = []
    
    for list_idx, prompt in enumerate(prompts):
        # 使用原始索引而不是列表索引
        original_idx = original_indices[list_idx]
        
        # Parse prompt for embedded scene
        prompt_text, embedded_scene = parse_prompt_with_scene(prompt)
        
        prompt_output_dir = Path(output_base_dir) / f"prompt_{original_idx}"
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存prompt到文件
        with open(prompt_output_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        context = {
            "idx": original_idx,  # 使用原始索引
            "prompt": prompt_text,
            "output_dir": prompt_output_dir,
            "status": "initializing",
            "conversation_history": [],
            "current_scene": None,
            "current_image_path": None,
            "all_scenes": [],
            "all_image_paths": [],
            "error": None,
            "last_feedback": "",  # 存储上一轮生成的反馈
            "embedded_scene": embedded_scene  # Store embedded scene
        }
        prompt_contexts.append(context)
    
    # 第一步：并行生成或加载初始场景
    print(f"\n{'='*60}")
    print(f"STEP 0: Initializing scenes for {len(prompts)} prompts")
    print(f"{'='*60}\n")
    
    if use_gpt_with_objects:
        # 使用GPT生成带物体的完整场景
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
                # 使用GPT生成带物体的完整场景
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
                        # 回退到生成空房间
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
        # 并行生成空房间
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
                        # 保存生成的房间
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
        # 使用相同的初始场景
        if not initial_scene_path or not Path(initial_scene_path).exists():
            print(f"Error: Initial scene file not found: {initial_scene_path}")
            return []
        
        with open(initial_scene_path, 'r', encoding='utf-8') as f:
            initial_scene = json.load(f)
        
        for ctx in prompt_contexts:
            ctx["current_scene"] = initial_scene.copy()
            ctx["status"] = "initialized"
        print(f"✓ Loaded initial scene for all {len(prompts)} prompts")
    
    # 为所有prompt渲染初始场景
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
    
    # 并行迭代优化
    active_contexts = [ctx for ctx in prompt_contexts if ctx["status"] == "initialized"]
    
    for iteration in range(num_iterations):
        if not active_contexts:
            print("\nNo active prompts remaining, stopping iterations")
            break
        
        print(f"\n{'='*60}")
        print(f"PARALLEL ITERATION {iteration + 1}/{num_iterations}")
        print(f"Processing {len(active_contexts)} active prompts")
        print(f"{'='*60}\n")
        
        # 准备所有活跃prompt的推理请求
        infer_requests = []
        for ctx in active_contexts:
            current_scene_json = json.dumps(ctx["current_scene"], indent=2)
            
            if iteration == 0:
                # 第一轮：无反馈
                base_user_content = f'{ctx["prompt"]}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
            else:
                # 后续轮次：包含反馈（如果有）
                feedback_section = ""
                if ctx.get("last_feedback"):
                    feedback_section = f'\n<feedback>\n{ctx["last_feedback"]}\n</feedback>\n'
                base_user_content = f'Please continue to improve the scene based on the original request: "{ctx["prompt"]}"{feedback_section}\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
            
            messages = []
            # 添加当前轮次的用户消息
            current_user_message = f'<image>{base_user_content}'
            
            # 使用智能截断：基于 token 数量而非固定轮次
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
        
        # 并行执行推理（分批处理避免OOM）
        print(f"Executing parallel inference for {len(infer_requests)} prompts (batch size: {max_batch_size})...")
        try:
            # 分批推理
            all_responses = []
            for batch_start in range(0, len(infer_requests), max_batch_size):
                batch_end = min(batch_start + max_batch_size, len(infer_requests))
                batch_requests = infer_requests[batch_start:batch_end]
                print(f"  Processing batch {batch_start//max_batch_size + 1}: requests {batch_start+1}-{batch_end}")
                batch_resp_list = engine.infer(batch_requests, request_config)
                all_responses.extend(batch_resp_list)
            resp_list = all_responses
            
            # 处理每个响应
            newly_inactive = []
            for ctx, resp in zip(active_contexts, resp_list):
                response = resp.choices[0].message.content
                
                # 保存响应
                with open(ctx["output_dir"] / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
                    f.write(response)
                
                # 更新对话历史
                ctx["conversation_history"].append((
                    f'<image>{base_user_content}',
                    response
                ))
                
                # 提取tool_calls
                tool_calls = extract_tool_calls_from_response(response)
                
                if tool_calls is not None:
                    # 检查terminate工具
                    has_terminate = any(
                        isinstance(tc, dict) and tc.get('name') == 'terminate' 
                        for tc in tool_calls
                    )
                    
                    if has_terminate:
                        print(f"✓ Prompt {ctx['idx']}: Terminated early (scene complete)")
                        ctx["status"] = "completed"
                        newly_inactive.append(ctx)
                        
                        # 保存最终场景
                        scene_file_path = ctx["output_dir"] / f"scene_iter_{iteration + 1}_final.json"
                        with open(scene_file_path, 'w', encoding='utf-8') as f:
                            json.dump(ctx["current_scene"], f, indent=2, ensure_ascii=False)
                        continue
                    
                    # 应用工具调用
                    final_scene = apply_tool_calls_to_scene(ctx["current_scene"], tool_calls)
                else:
                    final_scene = extract_final_scene_from_response(response)
                
                if final_scene is None:
                    print(f"✗ Prompt {ctx['idx']}: Failed to extract scene")
                    ctx["status"] = "failed"
                    ctx["error"] = f"Failed to extract scene at iteration {iteration + 1}"
                    newly_inactive.append(ctx)
                    continue
                
                # 检索资产
                final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module,
                                                        asset_source=asset_source,
                                                        objaverse_retriever=objaverse_retriever)
                
                # 应用物理优化（碰撞/出界修复）
                physics_deleted_feedback = ""
                if enable_physics_optimization:
                    print(f"✓ Prompt {ctx['idx']}: Applying physics optimization...")
                    final_scene, physics_deleted_feedback = apply_physics_optimization(
                        final_scene, 
                        models_path=models_path or "", 
                        max_steps=physics_opt_steps
                    )
                
                # 保存场景
                scene_file_path = ctx["output_dir"] / f"scene_iter_{iteration + 1}.json"
                with open(scene_file_path, 'w', encoding='utf-8') as f:
                    json.dump(final_scene, f, indent=2, ensure_ascii=False)
                
                ctx["all_scenes"].append(final_scene)
                
                # 渲染场景
                ctx["current_image_path"] = render_scene_to_image(
                    final_scene, 
                    ctx["output_dir"], 
                    iteration + 1,
                    enable_visualization=enable_visualization
                )
                ctx["all_image_paths"].append(ctx["current_image_path"])
                
                # 更新当前场景
                ctx["current_scene"] = final_scene
                
                # ===== 生成反馈用于下一轮 =====
                # 只在非最后一轮生成反馈
                if iteration < num_iterations - 1:
                    feedback_parts = []
                    
                    # 0. 物理优化删除物品反馈（优先添加）
                    if physics_deleted_feedback:
                        feedback_parts.append(physics_deleted_feedback)
                    
                    # 1. 物理反馈（来自trimesh）
                    if enable_physics_feedback and trimesh_metrics_instance is not None:
                        try:
                            # 使用grouped格式进行物理评估
                            scene_for_eval = convert_flat_to_grouped(final_scene) if 'objects' in final_scene else final_scene
                            trimesh_metrics = trimesh_metrics_instance.evaluate_scene(scene_for_eval, format_type='ours')
                            physics_feedback = generate_physics_feedback(trimesh_metrics, top_k=3)
                            if physics_feedback:
                                feedback_parts.append(physics_feedback)
                        except Exception as e:
                            print(f"⚠ Prompt {ctx['idx']}: Physics feedback failed - {e}")
                    
                    # 2. VLM布局反馈（来自Azure GPT-5.1）
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
                    
                    # 组合反馈
                    if feedback_parts:
                        ctx["last_feedback"] = " ".join(feedback_parts)
                    else:
                        ctx["last_feedback"] = ""
                
                print(f"✓ Prompt {ctx['idx']}: Iteration {iteration + 1} completed")
            
            # 移除不再活跃的contexts
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
    
    # 标记完成所有迭代的prompts
    for ctx in active_contexts:
        if ctx["status"] == "initialized":
            ctx["status"] = "completed"
    
    # 生成迭代汇总图并立即收集final_scene和渲染图
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
            
            # 立即收集final_scene和渲染图
            try:
                # 收集最终场景JSON
                scene_files = sorted(ctx["output_dir"].glob("scene_iter_*.json"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if scene_files:
                    final_scene_file = scene_files[-1]
                    destination_scene = final_scenes_dir / f"prompt_{ctx['idx']}_final_scene.json"
                    shutil.copy2(final_scene_file, destination_scene)
                    collected_scenes_count += 1
                    print(f"✓ Prompt {ctx['idx']}: Collected final scene")
                
                # 收集最终渲染图
                render_files = sorted(ctx["output_dir"].glob("merged_iter_*.png"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if render_files:
                    final_render_file = render_files[-1]
                    destination_render = final_renders_dir / f"prompt_{ctx['idx']}_final_render.png"
                    shutil.copy2(final_render_file, destination_render)
                    collected_renders_count += 1
                    print(f"✓ Prompt {ctx['idx']}: Collected final render")
            except Exception as collect_e:
                print(f"⚠ Prompt {ctx['idx']}: Failed to collect files - {collect_e}")
    
    # 保存对话历史
    for ctx in prompt_contexts:
        if ctx["conversation_history"]:
            with open(ctx["output_dir"] / "conversation_history.txt", 'w', encoding='utf-8') as f:
                for i, (user_msg, assistant_msg) in enumerate(ctx["conversation_history"], 1):
                    f.write(f"=== Iteration {i} ===\n")
                    f.write(f"User: {user_msg}\n\n")
                    f.write(f"Assistant: {assistant_msg}\n\n")
                    f.write("-" * 80 + "\n\n")
    
    # 收集结果
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
    
    # 汇总收集结果
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    successful_count = len([r for r in all_results if r['status'] == 'success'])
    print(f"Collected {collected_scenes_count}/{successful_count} final scenes")
    print(f"Collected {collected_renders_count}/{successful_count} final renders")
    print(f"Final scenes directory: {final_scenes_dir}")
    print(f"Final renders directory: {final_renders_dir}")
    
    # 保存汇总
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
    """批量迭代场景生成 - 对多个prompts分别执行完整的迭代场景生成
    
    Args:
        prompts: prompts列表
        engine: 推理引擎
        request_config: 请求配置
        initial_scene_path: 初始场景JSON文件路径（如果不生成空房间）
        asset_retrieval_module: 资产检索模块
        num_iterations: 每个prompt的迭代次数
        output_base_dir: 输出基础目录
        generate_room: 是否生成空房间
        use_model_for_creation: 是否使用模型生成初始场景
        use_gpt_with_objects: 是否使用GPT生成带物体的完整场景
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse 资产检索器实例
        enable_visualization: 是否启用3D可视化辅助线
        enable_physics_feedback: 是否启用物理反馈注入
        enable_vlm_feedback: 是否启用VLM布局反馈注入
        enable_physics_optimization: 是否启用物理优化（碰撞/出界修复）
        physics_opt_steps: 物理优化最大步数
        models_path: 3D模型路径
        original_indices: 原始prompt索引列表(1-indexed)，用于跳过已完成场景时保持索引一致
    
    Returns:
        所有场景的结果列表
    """
    # 如果没有提供原始索引，使用默认的1到N
    if original_indices is None:
        original_indices = list(range(1, len(prompts) + 1))
    
    print(f"\n{'='*50}")
    print(f"BATCH ITERATIVE SCENE GENERATION")
    print(f"Processing {len(prompts)} prompts with {num_iterations} iterations each")
    print(f"{'='*50}\n")
    
    # 创建收集文件夹（提前创建，每完成一个就立即收集）
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
        # 使用原始索引而不是列表索引
        idx = original_indices[list_idx]
        
        # Parse prompt for embedded scene
        prompt_text, embedded_scene = parse_prompt_with_scene(prompt)
        
        print(f"\n{'#'*60}")
        print(f"# BATCH ITEM {list_idx + 1}/{len(prompts)} (Original Index: {idx})")
        print(f"# Prompt: {prompt_text}")
        print(f"{'#'*60}\n")
        
        # 为每个prompt创建独立的输出目录
        prompt_output_dir = Path(output_base_dir) / f"prompt_{idx}"
        prompt_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存当前prompt到文件
        with open(prompt_output_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt_text)
        
        try:
            # 确定初始场景路径
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
            
            # 如果使用GPT生成带物体的完整场景
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
                    # 回退到生成空房间
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
                
            # 如果需要生成空房间
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
            
            # 检查场景文件是否存在
            if not scene_path or not Path(scene_path).exists():
                print(f"Error: Scene file not found: {scene_path}")
                all_results.append({
                    "prompt": prompt_text,
                    "status": "failed",
                    "error": f"Scene file not found: {scene_path}"
                })
                continue
            
            # 执行迭代场景生成
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
            
            # 立即收集final_scene和渲染图
            try:
                # 收集最终场景JSON
                scene_files = sorted(prompt_output_dir.glob("scene_iter_*.json"), key=lambda p: int(re.search(r'iter_(\d+)', p.name).group(1)))
                if scene_files:
                    final_scene_file = scene_files[-1]
                    destination_scene = final_scenes_dir / f"prompt_{idx}_final_scene.json"
                    shutil.copy2(final_scene_file, destination_scene)
                    collected_scenes_count += 1
                    print(f"  ✓ Collected final scene: {destination_scene.name}")
                
                # 收集最终渲染图
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
    
    # 汇总收集结果
    print(f"\n{'='*60}")
    print("COLLECTION SUMMARY")
    print(f"{'='*60}")
    successful_count = len([r for r in all_results if r['status'] == 'success'])
    print(f"Collected {collected_scenes_count}/{successful_count} final scenes")
    print(f"Collected {collected_renders_count}/{successful_count} final renders")
    print(f"Final scenes directory: {final_scenes_dir}")
    print(f"Final renders directory: {final_renders_dir}")
    
    # 保存批量处理的汇总结果
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
    """从txt文件读取prompts
    
    Args:
        file_path: txt文件路径,每行一个prompt
        max_prompts: 最多读取的prompt数量，None表示读取全部
    
    Returns:
        prompts列表
    """
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    prompts.append(line)
                    # 如果达到限制数量，停止读取
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
    """检测已完成的场景，返回已完成的prompt索引集合
    
    Args:
        output_base_dir: 输出基础目录
        total_prompts: 总prompt数量
        
    Returns:
        已完成的prompt索引集合 (1-indexed)
    """
    existing_indices = set()
    output_path = Path(output_base_dir)
    
    # 检查 final_scenes_collection 目录中的已完成场景
    final_scenes_dir = output_path / "final_scenes_collection"
    if final_scenes_dir.exists():
        for scene_file in final_scenes_dir.glob("prompt_*_final_scene.json"):
            # 从文件名提取索引，格式为 prompt_{idx}_final_scene.json
            try:
                filename = scene_file.stem  # prompt_1_final_scene
                parts = filename.split('_')
                if len(parts) >= 2:
                    idx = int(parts[1])
                    if 1 <= idx <= total_prompts:
                        existing_indices.add(idx)
            except (ValueError, IndexError):
                continue
    
    # 也检查各个 prompt_X 目录中是否有生成的场景文件
    for idx in range(1, total_prompts + 1):
        if idx in existing_indices:
            continue
        prompt_dir = output_path / f"prompt_{idx}"
        if prompt_dir.exists():
            # 检查是否有最终迭代的场景文件
            scene_files = list(prompt_dir.glob("scene_iter_*.json"))
            if scene_files:
                # 找到最大迭代号
                max_iter = max(
                    int(f.stem.replace("scene_iter_", "")) 
                    for f in scene_files
                )
                # 如果有较高迭代号的场景，认为已完成
                if max_iter >= 1:
                    existing_indices.add(idx)
    
    return existing_indices


def filter_prompts_by_existing(prompts: List[str], output_base_dir: str, skip_existing: bool = False) -> tuple:
    """根据已完成的场景过滤prompts
    
    Args:
        prompts: 原始prompts列表
        output_base_dir: 输出基础目录
        skip_existing: 是否跳过已完成的场景
        
    Returns:
        tuple: (过滤后的prompts列表, 过滤后的prompt索引列表(1-indexed), 跳过的prompt数量)
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


# 初始化资产检索模块（如果可用）
asset_retrieval_module = None
try:
    # 使用 sample.py 中的高级资产检索模块
    from utils.sample import AssetRetrievalModule
    # 初始化参数与 sample.py 中的测试代码保持一致
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

# 添加PIL导入用于图片处理
try:
    from PIL import Image
    print("PIL imported successfully")
except ImportError:
    print("Warning: PIL not available, rendering will be limited")

def main():
    """主函数"""
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
    # 添加批量推理参数
    parser.add_argument('--batch-mode', action='store_true', help='Run in batch mode - process multiple prompts from file')
    parser.add_argument('--prompts-file', default="/path/to/datasets/llmscene/sft/test_prompt_3dfront_v3.txt", help='Path to txt file containing prompts (one per line)')
    parser.add_argument('--max-prompts', type=int, default=None, help='Maximum number of prompts to process (default: process all)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing - process all prompts simultaneously at each iteration (faster)')
    # 资产来源选择参数
    parser.add_argument('--asset-source', choices=['3d-future', 'objaverse', 'auto'], default='auto',
                       help='Asset source for retrieval: 3d-future (default), objaverse, or auto (hybrid)')
    # 3D可视化参数
    parser.add_argument('--enable-viz', action='store_true', 
                       help='Enable 3D visualization with auxiliary lines (bbox, arrows, coordinate grid)')
    parser.add_argument('--disable-viz', action='store_true', default=True,
                       help='Explicitly disable 3D visualization')
    # 模型路径参数
    parser.add_argument('--model', type=str, default="/path/to/SceneReVis/ckpt/rl_ood_B200_v6_e3_s80",
                       help='Path to the model checkpoint directory')
    parser.add_argument('--lora-checkpoint', type=str, default=None,
                       help='Path to the LoRA checkpoint directory (optional)')
    # 多卡并行参数
    parser.add_argument('--tensor-parallel', type=int, default=1,
                       help='Number of GPUs for tensor parallelism (default: 1, use all available GPUs with -1)')
    # 对话历史限制参数
    parser.add_argument('--max-history-turns', type=int, default=8,
                       help='Maximum number of conversation history turns to keep (default: 4)')
    # 批处理大小限制参数
    parser.add_argument('--max-batch-size', type=int, default=4,
                       help='Maximum batch size for parallel inference to prevent OOM (default: 4)')
    # 反馈注入控制参数
    parser.add_argument('--enable-physics-feedback', action='store_true', default=False,
                       help='Enable physics feedback injection into user prompts (default: disabled)')
    parser.add_argument('--enable-vlm-feedback', action='store_true', default=False,
                       help='Enable VLM (GPT-5.1) layout feedback injection into user prompts (default: disabled)')
    # 物理优化控制参数（碰撞检测和出界修复）
    parser.add_argument('--enable-physics-optimization', action='store_true', default=False,
                       help='Enable physics optimization after each iteration (resolve collisions and out-of-bounds)')
    parser.add_argument('--physics-opt-steps', type=int, default=5,
                       help='Maximum steps for physics optimization (default: 5)')
    parser.add_argument('--models-path', type=str, default=None,
                       help='Path to 3D models directory for physics optimization')
    # 跳过已完成场景参数
    parser.add_argument('--skip-existing', action='store_true', default=False,
                       help='Skip prompts that already have generated final scenes in output directory')
    
    args = parser.parse_args()

    # 初始化 Objaverse 检索器（如果需要）
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

    # 确定是否启用可视化
    enable_visualization = args.enable_viz and not args.disable_viz
    if enable_visualization and render_with_visualization is None:
        print("Warning: 3D visualization requested but module not available")
        enable_visualization = False
    
    print(f"Asset source: {args.asset_source}")
    print(f"3D visualization: {'enabled' if enable_visualization else 'disabled'}")
    print(f"Physics feedback: {'enabled' if args.enable_physics_feedback else 'disabled'}")
    print(f"VLM layout feedback: {'enabled' if args.enable_vlm_feedback else 'disabled'}")
    print(f"Skip existing: {'enabled' if args.skip_existing else 'disabled'}")

    # 模型配置
    global engine, request_config
    model = args.model
    lora_checkpoint = args.lora_checkpoint
    template_type = None  # None: 使用对应模型默认的template_type
    
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


    # 加载模型和对话模板 - 使用vLLM引擎
    infer_backend = 'vllm'  # 使用vLLM后端进行推理

    # 确定tensor parallel大小
    import torch
    if args.tensor_parallel == -1:
        # 自动检测可用GPU数量
        tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-detected {tensor_parallel_size} GPUs for tensor parallelism")
    else:
        tensor_parallel_size = args.tensor_parallel
    
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Max history turns: {args.max_history_turns}")
    print(f"Max batch size: {args.max_batch_size}")

    if infer_backend == 'vllm':
        # 设置seed确保可复现性，添加tensor_parallel_size支持多卡
        # 先获取tokenizer和template以设置default_system
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
        # 如果需要使用其他后端,可以在这里添加
        model, tokenizer = get_model_tokenizer(model, model_type="qwen2_5_vl")
        template_type = template_type or model.model_meta.template
        template = get_template(template_type, tokenizer, default_system=default_system)
        engine = PtEngine.from_model_template(model, template, max_batch_size=64)

    # temperature=0 + seed 确保确定性输出
    request_config = RequestConfig(max_tokens=16384, temperature=0, seed=42)
    
    # 批量模式 - 对多个prompts分别执行完整的迭代场景生成
    if args.batch_mode:
        mode_name = "PARALLEL BATCH MODE" if args.parallel else "SEQUENTIAL BATCH MODE"
        print("="*60)
        print(f"{mode_name} - Multiple Prompts Iterative Scene Generation")
        print("="*60)
        
        if not args.prompts_file:
            print("Error: --prompts-file is required for batch mode")
            return
        
        # 从文件加载prompts
        prompts = load_prompts_from_file(args.prompts_file, max_prompts=args.max_prompts)
        
        if not prompts:
            print("Error: No prompts loaded from file")
            return
        
        # 过滤已完成的场景
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
        
        # 根据是否并行选择不同的处理函数
        if args.parallel:
            # 并行处理 - 在每个迭代步骤中同时处理所有prompt
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
            # 串行处理 - 一个prompt完成后再处理下一个
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
    
    # 单个prompt的迭代场景生成模式（原有功能）
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
    
    # 如果需要生成场景，先生成场景结构
    initial_scene_path = args.scene
    initial_conversation = None  # 用于存储初始场景生成的对话
    
    if args.use_gpt_with_objects:
        # 使用GPT生成带物体的完整场景
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
            # 回退到生成空房间
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
        
        # 使用room_prompt或默认的prompt
        room_prompt = args.room_prompt or args.prompt
        print(f"Room generation prompt: {room_prompt}")
        
        # 生成空房间并保存
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_room_path = output_dir / "generated_empty_room.json"
        
        # 根据use_model_for_creation选项决定使用哪种方法
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
            # 使用生成的房间作为初始场景
            initial_scene_path = str(generated_room_path)
        else:
            print("✗ Failed to generate empty room, using original scene file")
            print(f"Fallback to: {args.scene}")
    
    # 检查初始场景文件是否存在
    if not Path(initial_scene_path).exists():
        print(f"Error: Initial scene file not found: {initial_scene_path}")
        return
    
    if args.test_mode:
        print("Running in test mode (no model inference)...")
        # 使用测试函数
        test_iterative_generation(initial_scene_path, args.prompt, args.iterations, args.output, 
                                  asset_retrieval_module, asset_source=args.asset_source,
                                  objaverse_retriever=objaverse_retriever, 
                                  enable_visualization=enable_visualization)
    else:
        # 执行实际的迭代生成
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
    """测试模式的迭代生成（不使用实际模型）
    
    Args:
        asset_source: 资产来源 ('3d-future', 'objaverse', 'auto')
        objaverse_retriever: Objaverse 资产检索器实例
        enable_visualization: 是否启用3D可视化辅助线
    """
    
    def mock_model_response(iteration, current_scene, conversation_history):
        """模拟模型响应，考虑对话历史避免重复"""
        
        # 在第3次迭代时返回terminate工具调用，测试提前停止功能
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
        
        # 根据历史对话调整响应
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
        
        # 模拟工具调用
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
    
    # 读取初始场景
    with open(initial_scene_path, 'r', encoding='utf-8') as f:
        current_scene = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 渲染初始场景作为第一张图片
    print("Rendering initial scene...")
    initial_image_path = render_scene_to_image(current_scene, output_path, 0, enable_visualization=enable_visualization)
    print(f"Initial scene rendered to: {initial_image_path}")
    
    print(f"Starting test iterative generation with {num_iterations} iterations...")
    
    all_scenes = []
    
    # 保存完整的对话历史，包含每轮的用户请求和模型响应
    conversation_history = []
    
    for iteration in range(num_iterations):
        print(f"\n{'='*50}")
        print(f"TEST ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*50}")
        
        # 构建当前迭代的用户内容
        current_scene_json = json.dumps(current_scene, indent=2)
        
        # 构建基础用户内容
        if iteration == 0:
            # 第一轮：初始请求
            base_user_content = f'{user_prompt}\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        else:
            # 后续轮次：继续优化场景
            base_user_content = f'Please continue to improve the scene based on the original request: "{user_prompt}"\n\n<current_scene>\n```json\n{current_scene_json}\n```\n</current_scene>'
        
        current_user_message = f'<image>{base_user_content}'
        
        print(f"Current scene has {len(current_scene.get('groups', current_scene.get('objects', [])))} objects/groups")
        
        # 模拟模型响应（考虑对话历史）
        response = mock_model_response(iteration + 1, current_scene, conversation_history)
        
        # 保存响应
        with open(output_path / f"response_iter_{iteration + 1}.txt", 'w', encoding='utf-8') as f:
            f.write(response)
        
        # 将当前轮次的对话添加到历史中
        conversation_history.append((current_user_message, response))
        
        # 首先尝试提取 tool_calls
        tool_calls = extract_tool_calls_from_response(response)
        final_scene = None
        
        if tool_calls is not None:
            print(f"Extracted {len(tool_calls)} tool calls")
            
            # 检查是否有terminate工具调用
            has_terminate = False
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get('name') == 'terminate':
                    has_terminate = True
                    terminate_reason = tool_call.get('arguments', {}).get('reason', 'No reason provided')
                    print(f"🛑 Terminate tool detected: {terminate_reason}")
                    break
            
            # 如果发现terminate工具，停止迭代
            if has_terminate:
                print(f"Stopping test iterations early due to terminate tool call")
                # 保存当前场景作为最终场景
                scene_file_path = output_path / f"scene_iter_{iteration + 1}_final.json"
                with open(scene_file_path, 'w', encoding='utf-8') as f:
                    json.dump(current_scene, f, indent=2, ensure_ascii=False)
                print(f"Final scene saved to: {scene_file_path}")
                break
            
            # 使用 scene_editor 应用工具调用来生成最终场景
            final_scene = apply_tool_calls_to_scene(current_scene, tool_calls)
            print("Applied tool calls to generate final scene")
        else:
            # 如果没有找到 tool_calls，尝试提取 final_scene（作为备用）
            print("No tool_calls found, trying to extract final_scene as fallback")
            final_scene = extract_final_scene_from_response(response)
        
        if final_scene is None:
            print(f"Failed to extract scene data from iteration {iteration + 1}")
            break
        
        print(f"Extracted final_scene with {len(final_scene.get('groups', final_scene.get('objects', [])))} objects/groups")
        
        # 检索需要的资产
        final_scene = check_and_retrieve_assets(final_scene, asset_retrieval_module,
                                                 asset_source=asset_source,
                                                 objaverse_retriever=objaverse_retriever)
        
        # 保存当前场景
        scene_file_path = output_path / f"scene_iter_{iteration + 1}.json"
        with open(scene_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_scene, f, indent=2, ensure_ascii=False)
        
        all_scenes.append(final_scene)
        
        # 渲染场景为图片
        image_path = render_scene_to_image(final_scene, output_path, iteration + 1, enable_visualization=enable_visualization)
        print(f"Rendered image: {image_path}")
        
        # 更新当前场景为下一次迭代做准备
        current_scene = final_scene
        
        print(f"Iteration {iteration + 1} completed successfully")
    
    # 保存完整的对话历史到文件
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

# 进一步修补 modelscope 内部的 import_utils 模块的 __getattr__，对 __addon_enabled__ 返回 False 避免触发远程导入
try:
    import importlib
    import sys
    
    # 修补 modelscope.utils.import_utils
    imp_mod = importlib.import_module('modelscope.utils.import_utils')
    
    # 保存原始的 __getattr__
    _original_getattr = getattr(imp_mod, '__getattr__', None)
    
    def _safe_import_utils_getattr(name):
        # 对 Blender addon 相关的属性返回安全值
        if name in ('__addon_enabled__', '__addon_dependencies__', 'bl_info', 'register', 'unregister'):
            return False
        # 对其他属性，尝试返回 None 而不是抛出异常
        if _original_getattr:
            try:
                return _original_getattr(name)
            except ImportError:
                return None
        return None
    
    # 将模块级的 __getattr__ 替换为安全实现
    imp_mod.__getattr__ = _safe_import_utils_getattr
    
    # 同时修补 modelscope 主模块
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
