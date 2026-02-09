#!/usr/bin/env python3
"""
SceneEditingInteraction - 场景编辑交互类
用于VERL框架中的场景编辑强化学习任务
"""

import os
import sys
import json
import asyncio
import logging
import base64
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from io import BytesIO

# 类级别的信号量（延迟初始化）
# 不在模块级别创建，避免事件循环绑定问题
_RENDER_SEMAPHORE = None

# 添加RL_utils所在路径
# 更可靠的路径查找: 从当前文件向上找到项目根目录
current_file = Path(__file__).resolve()
# 假设项目结构: llmscene/verl/verl/interactions/scene_editing_interaction.py
# 向上4层到 llmscene，然后找 utils
project_root = current_file.parent.parent.parent.parent
utils_path = project_root / "utils"

if not utils_path.exists():
    raise RuntimeError(f"Utils path not found: {utils_path}")

if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))  # 使用 insert(0) 优先使用这个路径

from verl.interactions.base import BaseInteraction
from RL_utils import edit_and_render_scene, VoxelReward, TrimeshPhysicsMetrics, convert_flat_to_grouped, convert_grouped_to_flat, generate_physics_feedback

# 导入Objaverse资产元数据获取函数
try:
    from objaverse_retriever import get_bbox_dims as objaverse_get_bbox_dims, ObjaverseRetriever
except ImportError:
    objaverse_get_bbox_dims = None
    ObjaverseRetriever = None

# 定义各房间类型的必备基础物体（如果缺少这些物体，直接评分-1）
# 注意：key 使用标准化名称，matching_patterns 用于从用户输入中匹配
ROOM_TYPE_ESSENTIAL_OBJECTS = {
    "bedroom": {
        "essential": ["bed", "wardrobe", "nightstand"],
        "matching_patterns": ["bedroom", "bed room"],
        "aliases": {
            "bed": ["bed", "double bed", "single bed", "queen bed", "king bed", "twin bed"],
            "wardrobe": ["wardrobe", "closet", "armoire", "clothes cabinet"],
            "nightstand": ["nightstand", "bedside table", "night table", "bedside cabinet"]
        }
    },
    "living room": {
        "essential": ["sofa", "coffee table", "tv stand"],
        "matching_patterns": ["living room", "livingroom", "living-room", "lounge"],
        "aliases": {
            "sofa": ["sofa", "couch", "sectional", "loveseat"],
            "coffee table": ["coffee table", "center table"],
            "tv stand": ["tv stand", "tv cabinet", "tv unit", "entertainment center", "media console"]
        }
    },
    "study room": {
        "essential": ["desk", "chair", "bookshelf"],
        "matching_patterns": ["study room", "studyroom", "study-room", "study"],
        "aliases": {
            "desk": ["desk", "office desk", "writing desk", "work desk", "study desk"],
            "chair": ["chair", "office chair", "desk chair", "swivel chair"],
            "bookshelf": ["bookshelf", "bookcase", "book shelf", "shelving unit"]
        }
    },
    "office": {
        "essential": ["desk", "chair", "filing cabinet"],
        "matching_patterns": ["office", "home office", "work room", "workroom"],
        "aliases": {
            "desk": ["desk", "office desk", "writing desk", "work desk", "executive desk"],
            "chair": ["chair", "office chair", "desk chair", "executive chair"],
            "filing cabinet": ["filing cabinet", "file cabinet", "storage cabinet", "drawer unit"]
        }
    },
    "gym": {
        "essential": ["treadmill", "weight_equipment"],  # 大型健身器械是必备的，哑铃太小不算
        "matching_patterns": ["gym", "fitness room", "fitnessroom", "workout room", "exercise room", "home gym"],
        "aliases": {
            "treadmill": ["treadmill", "running machine", "exercise bike", "elliptical", "stationary bike", "rowing machine"],
            "weight_equipment": ["weight bench", "bench press", "power rack", "squat rack", "cable machine", "smith machine", "lat pulldown", "leg press", "multi-gym", "home gym"]
        }
    },
    "dining room": {
        "essential": ["dining table", "dining chair", "sideboard"],
        "matching_patterns": ["dining room", "diningroom", "dining-room", "dining"],
        "aliases": {
            "dining table": ["dining table", "dinner table", "eating table"],
            "dining chair": ["dining chair", "chair"],
            "sideboard": ["sideboard", "buffet", "credenza", "cabinet"]
        }
    },
    # 娱乐室类型多样（音乐室、桌游室、台球室、电子游戏室等），无法定义统一的必备物体
    # 交由VLM根据用户需求动态判断
    "entertainment room": {
        "essential": [],  # 不设固定必备物体，由VLM动态判断
        "matching_patterns": [
            # 通用娱乐室
            "entertainment room", "entertainmentroom", "entertainment", 
            "game room", "gameroom", "gaming room",
            "recreation room", "rec room", "play room", "playroom",
            # 家庭影院
            "home theater", "home theatre", "home cinema", "theater room", "theatre room",
            "cinema room", "movie room", "media room",
            # 音乐相关
            "music room", "musicroom", "ktv", "karaoke room", "karaoke",
            "piano room", "studio",
            # 球类运动
            "billiard room", "billiards room", "pool room", "snooker room",
            "ping pong room", "pingpong room", "table tennis room",
            # 桌游
            "board game room", "boardgame room", "card room", "poker room",
            "mahjong room", "chess room",
            # 电子游戏
            "video game room", "videogame room", "arcade room", "esports room",
            # 其他
            "bar room", "lounge room", "party room"
        ],
        "aliases": {}
    }
}


def match_room_type(user_input: str) -> Optional[str]:
    """
    从用户输入中匹配房间类型
    
    参数:
        user_input: 用户需求描述或房间类型字符串
        
    返回:
        匹配到的标准房间类型名称，或 None
    """
    if not user_input:
        return None
    
    user_input_lower = user_input.lower().strip()
    
    # 遍历所有房间类型，检查 matching_patterns
    for room_type, config in ROOM_TYPE_ESSENTIAL_OBJECTS.items():
        patterns = config.get("matching_patterns", [room_type])
        for pattern in patterns:
            if pattern in user_input_lower:
                return room_type
    
    return None


class SceneEditingInteraction(BaseInteraction):
    """
    场景编辑交互类
    
    处理场景编辑任务的交互流程，包括：
    1. 解析LLM输出的tool_calls
    2. 调用场景编辑和渲染
    3. 计算基于体素评估的物理奖励
    4. 管理多轮交互状态
    5. 使用VLM judge进行场景质量评估
    """
    
    @staticmethod
    def extract_create_scene_from_response(response_text: str) -> Optional[Dict[str, Any]]:
        """从模型响应中提取<create_scene>内容"""
        import re
        pattern = r'<create_scene>\s*```json\s*(.*?)\s*```\s*</create_scene>'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            try:
                scene_json_str = match.group(1).strip()
                scene_data = json.loads(scene_json_str)
                return scene_data
            except json.JSONDecodeError as e:
                print(f"Error parsing create_scene JSON: {e}", file=sys.stderr, flush=True)
                return None
        else:
            print("No <create_scene> found in response", file=sys.stderr, flush=True)
            return None
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """将图像文件转换为base64字符串"""
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
    
    @staticmethod
    def extract_think_content(response_text: str) -> Optional[str]:
        """从模型响应中提取<think>内容"""
        import re
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化场景编辑交互
        
        参数:
            config: 配置字典，包含以下键：
                - max_turns: 最大交互轮数 (默认10)
                - models_base_path: 3D模型基础路径
                - voxel_size: 体素大小 (默认0.05)
                - reward_threshold: PBL损失阈值 (默认1e-5)
                - output_dir: 输出目录 (默认"./scene_editing_output")
                - verbose: 是否输出详细日志 (默认False)
                - paths: 统一路径配置块（可选）
        """
        # ========== 初始化 PathConfig 单例（统一路径配置）==========
        try:
            from path_config import PathConfig
            path_config = PathConfig.init_from_config(config)
            print(f"✓ PathConfig initialized: {path_config}")
        except Exception as e:
            print(f"⚠ PathConfig initialization failed: {e}, using fallback paths")
            path_config = None
        
        # 创建自定义日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        
        # 优先使用 PathConfig，然后使用硬编码回退
        if path_config and path_config.logs_dir:
            log_dir = Path(path_config.logs_dir) / "interaction_logs"
        elif Path("/path/to/logs").exists():
            log_dir = Path("/path/to/logs/interaction_logs")
        else:
            log_dir = Path("./logs/interaction_logs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"scene_editing_interaction_{timestamp}_pid{pid}.log"
        # self.log_file = Path(f"/path/to/data/logs/scene_editing_interaction_{timestamp}_pid{pid}.log")

        # 设置日志记录器
        self.logger = logging.getLogger(f"scene_editing_{timestamp}_{pid}")
        self.logger.setLevel(logging.DEBUG)
        
        # 文件 handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 记录初始化开始
        self.logger.info("="*80)
        self.logger.info(f"SceneEditingInteraction.__init__ called")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Process ID: {pid}")
        self.logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")
        if path_config:
            self.logger.info(f"PathConfig: {path_config}")
        
        # 打印到控制台（可能被 Ray 捕获）
        print(f"✓ SceneEditingInteraction log file: {self.log_file}")
        
        super().__init__(config)
        self.name = "scene_editing"
        self.max_turns = config.get("max_turns", 10)
        self.output_dir = Path(config.get("output_dir", "./scene_editing_output"))
        self.verbose = config.get("verbose", False)
        
        # 物理评估模式：'voxel' 或 'trimesh'（默认使用trimesh）
        self.physics_mode = config.get("physics_mode", "trimesh")
        
        # 资产来源配置：是否使用 Objaverse（默认 True，使用 Objaverse）
        self.use_objaverse = config.get("use_objaverse", True)
        
        # VLM Judge 配置
        self.vlm_judge_enabled = config.get("vlm_judge_enabled", True)
        self.vlm_judge_url = config.get("vlm_judge_url", "http://localhost:8000/v1/chat/completions")
        self.vlm_judge_model = config.get("vlm_judge_model", "Qwen/Qwen2.5-VL-72B-Instruct")
        self.vlm_judge_timeout = config.get("vlm_judge_timeout", 60)
        
        # 信号量配置（限制并发渲染/重型任务的数量）
        self.semaphore_limit = config.get("semaphore_limit", 32)
        
        # 反馈注入配置
        feedback_config = config.get("feedback_injection", {})
        self.feedback_injection_enabled = feedback_config.get("enabled", True)
        self.physics_feedback_enabled = feedback_config.get("physics_feedback_enabled", True)
        self.layout_feedback_enabled = feedback_config.get("layout_feedback_enabled", True)
        
        self.logger.info(f"name: {self.name}")
        self.logger.info(f"max_turns: {self.max_turns}")
        self.logger.info(f"output_dir: {self.output_dir}")
        self.logger.info(f"verbose: {self.verbose}")
        self.logger.info(f"physics_mode: {self.physics_mode}")
        self.logger.info(f"use_objaverse: {self.use_objaverse}")
        self.logger.info(f"vlm_judge_enabled: {self.vlm_judge_enabled}")
        self.logger.info(f"feedback_injection_enabled: {self.feedback_injection_enabled}")
        self.logger.info(f"physics_feedback_enabled: {self.physics_feedback_enabled}")
        self.logger.info(f"layout_feedback_enabled: {self.layout_feedback_enabled}")
        if self.vlm_judge_enabled:
            self.logger.info(f"vlm_judge_url: {self.vlm_judge_url}")
            self.logger.info(f"vlm_judge_model: {self.vlm_judge_model}")
        
        # 初始化体素奖励计算器 - 优先使用 PathConfig
        # 当使用 Objaverse 模式时，3D-FUTURE 路径是可选的
        models_base_path = None
        if path_config and path_config.future3d_models_dir:
            models_base_path = path_config.future3d_models_dir
            self.logger.info(f"Using PathConfig models_base_path: {models_base_path}")
        elif not self.use_objaverse:
            # 只有在非 Objaverse 模式下才强制要求 3D-FUTURE 路径
            models_base_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
            # 回退到本地路径
            if not Path(models_base_path).exists():
                alt_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
                if Path(alt_path).exists():
                    models_base_path = alt_path
                    self.logger.info(f"Using alternative models_base_path: {models_base_path}")
        else:
            # Objaverse 模式：3D-FUTURE 路径可选，尝试查找但不强制
            for candidate in [
                "/path/to/datasets/3d-front/3D-FUTURE-model",
                "/path/to/datasets/3d-front/3D-FUTURE-model"
            ]:
                if Path(candidate).exists():
                    models_base_path = candidate
                    self.logger.info(f"Found optional 3D-FUTURE path: {models_base_path}")
                    break
            if not models_base_path:
                self.logger.info("3D-FUTURE path not found, using Objaverse-only mode")
        
        # 根据physics_mode初始化对应的评估器
        if self.physics_mode == "voxel":
            self.logger.info(f"Initializing VoxelReward with models_base_path: {models_base_path}")
            
            self.voxel_reward = VoxelReward(
                models_base_path=models_base_path,
                voxel_size=config.get("voxel_size", 0.05),
                reward_threshold=config.get("reward_threshold", 1e-5),
                verbose=self.verbose
            )
            self.trimesh_metrics = None
            
            self.logger.info("VoxelReward initialized successfully")
            
        elif self.physics_mode == "trimesh":
            self.logger.info(f"Initializing TrimeshPhysicsMetrics with models_base_path: {models_base_path}")
            
            self.voxel_reward = None
            self.trimesh_metrics = TrimeshPhysicsMetrics(
                models_base_path=models_base_path,
                verbose=self.verbose
            )
            
            self.logger.info("TrimeshPhysicsMetrics initialized successfully")
            
        else:
            raise ValueError(f"Invalid physics_mode: {self.physics_mode}. Must be 'voxel' or 'trimesh'")
        
        # 存储每个实例的状态
        self._instance_dict = {}
        
        # 支撑类型缓存：存储物体描述到支撑类型的映射，避免重复LLM调用
        self.support_type_cache = {}
        
        # Objaverse资产数据库（延迟初始化，用于获取物体真实尺寸）
        self._objaverse_retriever = None
        self._objaverse_database = None
        
        if self.verbose:
            print(f"✓ SceneEditingInteraction initialized")
            print(f"  Max turns: {self.max_turns}")
            print(f"  Output dir: {self.output_dir}")
            print(f"  Models path: {models_base_path}")
        
        self.logger.info("SceneEditingInteraction.__init__ completed")
        self.logger.info("="*80)
    
    async def _get_render_semaphore(self):
        """
        获取渲染信号量（延迟初始化，避免在模块导入时绑定事件循环）
        
        使用类级别的信号量，在第一次调用时创建。
        这样可以确保信号量在正确的事件循环中创建。
        
        Returns:
            asyncio.Semaphore 实例
        """
        global _RENDER_SEMAPHORE
        if _RENDER_SEMAPHORE is None:
            _RENDER_SEMAPHORE = asyncio.Semaphore(self.semaphore_limit)
            self.logger.info(f"Created render semaphore with limit: {self.semaphore_limit}")
        return _RENDER_SEMAPHORE
    
    def _get_objaverse_database(self) -> Optional[Dict]:
        """
        获取Objaverse资产数据库（延迟初始化）
        
        Returns:
            资产数据库字典，或None（如果加载失败）
        """
        if self._objaverse_database is not None:
            return self._objaverse_database
        
        if ObjaverseRetriever is None:
            self.logger.warning("ObjaverseRetriever not available, cannot get asset database")
            return None
        
        try:
            if self._objaverse_retriever is None:
                self._objaverse_retriever = ObjaverseRetriever(do_print=False)
            self._objaverse_database = self._objaverse_retriever.database
            self.logger.info(f"Loaded Objaverse database with {len(self._objaverse_database)} assets")
            return self._objaverse_database
        except Exception as e:
            self.logger.warning(f"Failed to load Objaverse database: {e}")
            return None
    
    def _get_asset_real_size(self, uid: str) -> Optional[Dict[str, float]]:
        """
        通过资产ID获取物体的真实尺寸（仿照 Holodeck 的 get_bbox_dims 方法）
        
        数据路径：annotations[uid]["thor_metadata"]["assetMetadata"]["boundingBox"]
        boundingBox 格式：{min: {x, y, z}, max: {x, y, z}}
        
        参数:
            uid: Objaverse资产UID
            
        返回:
            包含 x, y, z 尺寸的字典（单位：米），或None（如果失败）
        """
        if not uid:
            return None
        
        database = self._get_objaverse_database()
        if database is None or uid not in database:
            return None
        
        try:
            obj_data = database[uid]
            
            # 获取 assetMetadata（仿照 Holodeck 的 get_asset_metadata）
            if "assetMetadata" in obj_data:
                asset_metadata = obj_data["assetMetadata"]
            elif "thor_metadata" in obj_data:
                asset_metadata = obj_data["thor_metadata"].get("assetMetadata")
            else:
                self.logger.debug(f"No assetMetadata found for uid {uid}")
                return None
            
            if not asset_metadata or "boundingBox" not in asset_metadata:
                self.logger.debug(f"No boundingBox in assetMetadata for uid {uid}")
                return None
            
            # 获取 boundingBox 尺寸（仿照 Holodeck 的 get_bbox_dims）
            bbox_info = asset_metadata["boundingBox"]
            
            if "x" in bbox_info:
                return bbox_info
            if "size" in bbox_info:
                return bbox_info["size"]
            
            # 从 min/max 计算尺寸
            mins = bbox_info["min"]
            maxs = bbox_info["max"]
            return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}
            
        except Exception as e:
            self.logger.warning(f"Failed to get bbox dims for uid {uid}: {e}")
            return None

    async def _call_vlm_judge(
        self,
        image_path: str,
        prompt: str,
        max_retries: int = 3,
        max_tokens: int = 50,
        return_text: bool = False
    ) -> Optional[float] | Optional[str]:
        """
        调用VLM judge进行评分或获取文本响应
        
        参数:
            image_path: 渲染图像路径
            prompt: 评分提示
            max_retries: 最大重试次数
            max_tokens: 最大生成token数（评分用50，分析/描述用更大值）
            return_text: 如果为True，返回原始文本而非解析评分
            
        返回:
            如果return_text=False: 评分 (-1.0, -0.5, 0.0, 0.5, 1.0) 或 None（如果失败）
            如果return_text=True: 原始文本响应 或 None（如果失败）
        """
        if not self.vlm_judge_enabled:
            self.logger.info("VLM judge disabled, skipping")
            return None
        
        if not Path(image_path).exists():
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        # 将图像转换为base64
        try:
            img_base64 = self.image_to_base64(image_path)
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            return None
        
        # 构建请求
        data = {
            "model": self.vlm_judge_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_base64
                            }
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        # 重试逻辑
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    self.vlm_judge_url,
                    json=data,
                    timeout=self.vlm_judge_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # 如果需要返回文本，直接返回
                    if return_text:
                        self.logger.info(f"VLM judge text response (first 200 chars): {content[:200]}...")
                        return content
                    
                    # 解析评分（支持5级评分：-1.0, -0.5, 0.0, 0.5, 1.0）
                    import re
                    # 首先尝试匹配带小数的分数
                    match = re.search(r'(-?[01])\.([05])', content)
                    if match:
                        score = float(f"{match.group(1)}.{match.group(2)}")
                        self.logger.info(f"VLM judge score: {score}, response: {content}")
                        return score
                    # 如果没有小数，尝试匹配整数
                    match = re.search(r'(-1|0|1)', content)
                    if match:
                        score = float(match.group(1))
                        self.logger.info(f"VLM judge score: {score}, response: {content}")
                        return score
                    else:
                        self.logger.warning(f"Failed to parse score from response: {content}")
                        
                else:
                    self.logger.warning(f"VLM judge request failed with status {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"VLM judge attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # 等待1秒后重试
        
        self.logger.error("VLM judge failed after all retries")
        return None
    
    async def _vlm_analyze_scene_problems(
        self,
        image_path: str,
        scene_json_str: str,
        user_requirement: str
    ) -> Optional[str]:
        """
        第一阶段：让VLM独立分析场景中存在的问题（用于中间轮次）
        
        参数:
            image_path: 渲染图路径
            scene_json_str: 场景JSON字符串
            user_requirement: 用户需求
            
        返回:
            VLM识别的问题列表文本，或None（如果失败）
        """
        prompt_analyze = f"""You are a highly critical interior design expert. Carefully examine this scene rendering (left: top view, right: diagonal view).

**IMPORTANT - USE VISUAL ANNOTATIONS IN THE IMAGE:**
- **Floor coordinate grid**: The TOP VIEW (left) shows a coordinate grid on the floor. Use this to precisely locate object positions and room boundaries.
- **Bounding boxes**: Each object has a colored bounding box (bbox) drawn around it. Use these to identify object boundaries, check for overlaps/collisions, and verify spatial relationships.

User requirement: {user_requirement}

Current scene JSON (with complete object positions, sizes, rotations):
```json
{scene_json_str}
```

**YOUR TASK: Independently analyze and list ALL problems in the current scene.**

Examine the scene carefully and identify problems in these THREE categories:

## Problem Categories:

**1. Physical Bugs (物理类问题)**:
- Object Overlap/Collision: Two or more objects occupying the same space (check bboxes intersecting in TOP VIEW)
- Out of Bounds: Objects extending beyond room boundaries (check bboxes vs floor grid)
- Floating Objects: Objects not properly supported (check DIAGONAL VIEW)
- Example: "PHYSICAL: The coffee table bbox overlaps with sofa bbox by approximately 0.3m"
- Example: "PHYSICAL: The wardrobe extends 0.5m beyond the north wall boundary"

**2. Layout Rationality Bugs (场景合理性问题)**:
- Core Furniture Misplacement: Bed/sofa not against wall, in room center
- Missing Essential Items: Room lacks core furniture for its type (bedroom needs bed, living room needs sofa)
- Improper Orientation: Furniture facing wrong direction (sofa facing wall instead of TV area)
- Example: "RATIONALITY: The bed is placed in the center of the room, not against any wall"
- Example: "RATIONALITY: The sofa is facing the corner wall instead of the open area"

**3. Spatial Distribution Bugs (空间分布问题)**:
- Clustering: All furniture crowded in one corner/side of the room
- Large Empty Areas: More than 40% of room completely empty
- Unbalanced Layout: One half crowded, other half empty
- Example: "DISTRIBUTION: All furniture is clustered in the northeast corner, leaving 70% of the room empty"
- Example: "DISTRIBUTION: Objects are all along the south wall, north half is completely empty"

**OUTPUT FORMAT:**
List each problem found with its category prefix. Be specific about locations and measurements.
If a category has no problems, write "[CATEGORY]: No issues found"

**PHYSICAL BUGS:**
[List all physical problems found, or "No issues found"]

**RATIONALITY BUGS:**
[List all rationality problems found, or "No issues found"]

**DISTRIBUTION BUGS:**
[List all distribution problems found, or "No issues found"]

**SEVERITY SUMMARY:**
[Rate overall severity: CRITICAL / MODERATE / MINOR / NONE]
[Brief summary of the most important issues to fix]"""
        
        return await self._call_vlm_judge(
            image_path, 
            prompt_analyze, 
            max_tokens=800, 
            return_text=True
        )
    
    async def _vlm_generate_layout_feedback(
        self,
        image_path: str,
        user_requirement: str
    ) -> str:
        """
        生成简短的VLM布局反馈（不包含物理碰撞/出界问题，这些由trimesh处理）
        
        参数:
            image_path: 渲染图路径
            user_requirement: 用户需求
            
        返回:
            简短的布局反馈文本，如果没有问题或失败则返回空字符串
        """
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
            result = await self._call_vlm_judge(
                image_path, 
                prompt_layout_feedback, 
                max_tokens=100, 
                return_text=True
            )
            
            if result:
                # 清理结果，确保简洁
                result = result.strip()
                # 如果VLM返回"no issues"类似的内容，返回空字符串
                if any(phrase in result.lower() for phrase in ["no issue", "looks good", "well-designed", "properly placed"]):
                    return ""
                return result
            return ""
        except Exception as e:
            self.logger.warning(f"VLM layout feedback generation failed: {e}")
            return ""
    
    def _extract_tool_calls_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        从消息历史中提取最新的工具调用列表
        
        参数:
            messages: 消息列表
            
        返回:
            工具调用列表，每个元素包含 name 和 arguments
        """
        import re
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str) and "<tool_calls>" in content and "</tool_calls>" in content:
                    match = re.search(r'<tool_calls>\s*(.*?)\s*</tool_calls>', content, re.DOTALL)
                    if match:
                        try:
                            tool_calls_str = match.group(1).strip()
                            tool_calls = json.loads(tool_calls_str)
                            if isinstance(tool_calls, list):
                                return tool_calls
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse tool_calls JSON from messages")
                            return []
        return []
    
    async def _vlm_evaluate_new_objects_relevance(
        self,
        tool_calls: List[Dict],
        user_requirement: str,
        room_type: str = ""
    ) -> Dict[str, Any]:
        """
        评估本轮add/replace的新物体是否与场景需求相关
        
        只针对add_object和replace_object操作中的新物体进行评估
        
        参数:
            tool_calls: 工具调用列表
            user_requirement: 用户需求描述
            room_type: 房间类型
            
        返回:
            包含评分和详细信息的字典：
            - score: 评分 (1.0, 0.0, -1.0)
            - relevant_objects: 相关物体列表
            - irrelevant_objects: 无关物体列表
        """
        result = {
            "score": 0.0,
            "relevant_objects": [],
            "irrelevant_objects": []
        }
        
        if not self.vlm_judge_enabled:
            return result
        
        # 提取add_object和replace_object中的新物体描述
        new_objects = []
        for tool_call in tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})
            
            if name == "add_object":
                obj_desc = args.get("object_description", "")
                if obj_desc:
                    new_objects.append(obj_desc)
            elif name == "replace_object":
                new_desc = args.get("new_object_description", "")
                if new_desc:
                    new_objects.append(new_desc)
        
        if not new_objects:
            # 没有add/replace操作，返回中性分数
            return result
        
        new_objects_str = "\n".join([f"- {obj}" for obj in new_objects])
        
        prompt = f"""You are an expert interior designer. Evaluate whether the following NEW OBJECTS being added to the scene are RELEVANT and APPROPRIATE for the room type and user requirement.

**ROOM TYPE:** {room_type if room_type else "(Infer from user requirement)"}

**USER REQUIREMENT:**
{user_requirement}

**NEW OBJECTS BEING ADDED:**
{new_objects_str}

**YOUR TASK:**
For each object, determine if it is:
1. **RELEVANT**: The object belongs in this room type and matches the user's requirement
2. **IRRELEVANT**: The object does NOT belong in this room type or contradicts the requirement

**Examples of IRRELEVANT objects:**
- A toilet in a bedroom or living room
- A stove in a bedroom
- Exercise equipment in a dining room (unless user requested it)
- Excessive duplicates (5 identical chairs when only 2 are needed)

**OUTPUT FORMAT (JSON only):**
```json
{{
    "relevant_objects": ["object1", "object2"],
    "irrelevant_objects": ["object3"]
}}
```

If ALL objects are relevant, irrelevant_objects should be empty [].
If ALL objects are irrelevant, relevant_objects should be empty []."""

        try:
            data = {
                "model": self.vlm_judge_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 200,
                "temperature": 0.1
            }
            
            response = await asyncio.to_thread(
                requests.post,
                self.vlm_judge_url,
                json=data,
                timeout=self.vlm_judge_timeout
            )
            
            if response.status_code != 200:
                self.logger.warning(f"New objects relevance evaluation failed with status {response.status_code}")
                return result
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                eval_json = json.loads(json_match.group(1))
            else:
                eval_json = json.loads(content)
            
            relevant = eval_json.get("relevant_objects", [])
            irrelevant = eval_json.get("irrelevant_objects", [])
            
            result["relevant_objects"] = relevant
            result["irrelevant_objects"] = irrelevant
            
            # 计算评分
            total = len(new_objects)
            irrelevant_count = len(irrelevant)
            
            if irrelevant_count == 0:
                # 全部相关
                result["score"] = 1.0
            elif irrelevant_count >= total:
                # 全部无关
                result["score"] = -1.0
            else:
                # 部分相关：根据比例计算
                # 无关比例 > 50% 则负分，否则正分
                irrelevant_ratio = irrelevant_count / total
                if irrelevant_ratio > 0.5:
                    result["score"] = -0.5
                else:
                    result["score"] = 0.5
            
            self.logger.info(f"New objects relevance: score={result['score']}, relevant={relevant}, irrelevant={irrelevant}")
            
        except Exception as e:
            self.logger.warning(f"New objects relevance evaluation failed: {e}")
            result["score"] = 0.0
        
        return result
    
    async def _vlm_evaluate_object_size_proportion(
        self,
        image_path: str,
        scene_data: Dict[str, Any],
        user_requirement: str
    ) -> Dict[str, Any]:
        """
        评估场景中物体的尺寸和比例是否合理（程序化评估，使用资产真实尺寸）
        
        通过资产ID获取真实尺寸，与常见物体的合理尺寸范围进行比较
        
        参数:
            image_path: 渲染图路径（保留参数但不使用，因为改为程序化评估）
            scene_data: 场景JSON数据
            user_requirement: 用户需求（保留参数但不使用）
            
        返回:
            包含评分和详细信息的字典：
            - score: 评分 (-1.0, -0.5, 0.0, 0.5, 1.0)
            - issues: 发现的尺寸/比例问题列表
        """
        result = {
            "score": 1.0,  # 默认满分，有问题时扣分
            "issues": []
        }
        
        # 定义各类物体的合理尺寸范围（单位：米）
        # 格式：关键词 -> (min_width, max_width, min_height, max_height, min_depth, max_depth)
        SIZE_STANDARDS = {
            # 床类
            "bed": (1.2, 2.5, 0.3, 0.8, 1.8, 2.2),
            "double bed": (1.4, 2.2, 0.3, 0.8, 1.9, 2.2),
            "single bed": (0.9, 1.2, 0.3, 0.8, 1.8, 2.1),
            # 座椅类
            "sofa": (1.5, 3.5, 0.6, 1.2, 0.7, 1.2),
            "couch": (1.5, 3.5, 0.6, 1.2, 0.7, 1.2),
            "chair": (0.4, 0.7, 0.7, 1.2, 0.4, 0.7),
            "office chair": (0.5, 0.8, 0.9, 1.3, 0.5, 0.7),
            "armchair": (0.6, 1.0, 0.7, 1.1, 0.6, 1.0),
            # 桌类
            "desk": (0.8, 2.0, 0.7, 0.85, 0.5, 0.9),
            "table": (0.6, 2.5, 0.4, 0.9, 0.6, 1.5),
            "coffee table": (0.6, 1.5, 0.3, 0.6, 0.4, 1.0),
            "dining table": (0.8, 2.5, 0.7, 0.85, 0.8, 1.5),
            "nightstand": (0.35, 0.6, 0.4, 0.7, 0.35, 0.55),
            "bedside table": (0.35, 0.6, 0.4, 0.7, 0.35, 0.55),
            # 柜类
            "wardrobe": (0.8, 2.5, 1.8, 2.5, 0.5, 0.7),
            "closet": (0.8, 2.5, 1.8, 2.5, 0.5, 0.7),
            "cabinet": (0.4, 1.5, 0.6, 2.2, 0.3, 0.7),
            "bookshelf": (0.6, 1.5, 1.2, 2.2, 0.25, 0.45),
            "bookcase": (0.6, 1.5, 1.2, 2.2, 0.25, 0.45),
            "tv stand": (0.8, 2.0, 0.4, 0.7, 0.35, 0.55),
            "sideboard": (1.0, 2.2, 0.7, 1.0, 0.4, 0.6),
            # 健身器材
            "treadmill": (0.7, 1.0, 1.2, 1.6, 1.5, 2.2),
            "exercise bike": (0.5, 0.7, 1.0, 1.5, 1.0, 1.5),
            "weight bench": (0.5, 0.8, 0.4, 0.6, 1.2, 1.8),
            "power rack": (1.0, 1.5, 2.0, 2.5, 1.2, 1.8),
            # 其他
            "lamp": (0.15, 0.5, 0.3, 1.8, 0.15, 0.5),
            "floor lamp": (0.25, 0.5, 1.2, 1.9, 0.25, 0.5),
            "tv": (0.8, 2.0, 0.5, 1.2, 0.05, 0.2),
            "mirror": (0.3, 1.5, 0.5, 2.0, 0.02, 0.1),
            "rug": (1.0, 4.0, 0.01, 0.05, 1.0, 3.0),
            "plant": (0.2, 0.8, 0.3, 2.0, 0.2, 0.8),
        }
        
        # 提取所有物体的尺寸信息
        objects_info = []
        
        def extract_object_info(obj):
            """从单个物体中提取尺寸信息，优先使用资产真实尺寸"""
            desc = obj.get('desc', obj.get('object_description', 'unknown'))
            specified_size = obj.get('size', [1, 1, 1])
            uid = obj.get('uid', None)
            
            # 优先通过 uid 从数据库查询真实尺寸（ground truth）
            # retrieved_size 可能是归一化后的 [1,1,1]，不可靠
            real_size = None
            size_source = "specified"
            
            if uid:
                real_bbox = self._get_asset_real_size(uid)
                if real_bbox:
                    # 检查是否是有效的真实尺寸（非默认值）
                    x, y, z = real_bbox.get('x', 1.0), real_bbox.get('y', 1.0), real_bbox.get('z', 1.0)
                    if not (x == 1.0 and y == 1.0 and z == 1.0):
                        real_size = [x, y, z]
                        size_source = "ground_truth"
                        self.logger.debug(f"Got ground truth size for '{desc}': {real_size}")
            
            if real_size is None:
                real_size = specified_size
                size_source = "specified"
            
            return {
                'desc': desc,
                'size': real_size,
                'specified_size': specified_size,
                'size_source': size_source,
                'uid': uid
            }
        
        if 'groups' in scene_data and scene_data['groups']:
            for group in scene_data['groups']:
                for obj in group.get('objects', []):
                    objects_info.append(extract_object_info(obj))
        elif 'objects' in scene_data:
            for obj in scene_data['objects']:
                objects_info.append(extract_object_info(obj))
        
        if not objects_info:
            return result
        
        # 程序化检查每个物体的尺寸
        critical_issues = 0  # 严重问题数量
        minor_issues = 0     # 轻微问题数量
        
        for obj_info in objects_info:
            desc = obj_info['desc'].lower()
            size = obj_info['size']  # [width, height, depth]
            
            # 只有当尺寸来源是 "specified"（用户指定）时，才需要用标准范围检查
            # 如果尺寸来自 "ground_truth"（retrieved_size 或 uid 查询），直接信任
            if obj_info['size_source'] == "specified":
                # 尝试匹配物体类型
                matched_standard = None
                for keyword, standard in SIZE_STANDARDS.items():
                    if keyword in desc:
                        matched_standard = (keyword, standard)
                        break
                
                if matched_standard:
                    keyword, (min_w, max_w, min_h, max_h, min_d, max_d) = matched_standard
                    w, h, d = size[0], size[1], size[2]
                    
                    issues_for_obj = []
                    
                    # 检查宽度
                    if w < min_w * 0.5:  # 太小（低于最小值的50%）
                        issues_for_obj.append(f"width {w:.2f}m is too small (expected {min_w:.1f}-{max_w:.1f}m)")
                        critical_issues += 1
                    elif w < min_w * 0.8:  # 略小
                        issues_for_obj.append(f"width {w:.2f}m is slightly small")
                        minor_issues += 1
                    elif w > max_w * 2.0:  # 太大
                        issues_for_obj.append(f"width {w:.2f}m is too large (expected {min_w:.1f}-{max_w:.1f}m)")
                        critical_issues += 1
                    elif w > max_w * 1.3:  # 略大
                        issues_for_obj.append(f"width {w:.2f}m is slightly large")
                        minor_issues += 1
                    
                    # 检查高度
                    if h < min_h * 0.5:
                        issues_for_obj.append(f"height {h:.2f}m is too small (expected {min_h:.1f}-{max_h:.1f}m)")
                        critical_issues += 1
                    elif h < min_h * 0.8:
                        minor_issues += 1
                    elif h > max_h * 2.0:
                        issues_for_obj.append(f"height {h:.2f}m is too large (expected {min_h:.1f}-{max_h:.1f}m)")
                        critical_issues += 1
                    elif h > max_h * 1.3:
                        minor_issues += 1
                    
                    # 检查深度
                    if d < min_d * 0.5:
                        issues_for_obj.append(f"depth {d:.2f}m is too small (expected {min_d:.1f}-{max_d:.1f}m)")
                        critical_issues += 1
                    elif d > max_d * 2.0:
                        issues_for_obj.append(f"depth {d:.2f}m is too large (expected {min_d:.1f}-{max_d:.1f}m)")
                        critical_issues += 1
                    
                    if issues_for_obj:
                        result["issues"].append(f"{obj_info['desc']}: {'; '.join(issues_for_obj)}")
            else:
                # 尺寸来自资产数据库，是真实尺寸，记录为已验证
                self.logger.debug(f"Object '{obj_info['desc']}' has verified size from {obj_info['size_source']}: {size}")
            
            # 检查用户指定尺寸与资产真实尺寸的差异（仅当有真实尺寸时）
            if obj_info['size_source'] != "specified":
                specified = obj_info['specified_size']
                real = obj_info['size']
                if specified != [1, 1, 1]:  # 非默认值
                    diff_ratio = max(
                        abs(specified[0] - real[0]) / max(real[0], 0.01),
                        abs(specified[1] - real[1]) / max(real[1], 0.01),
                        abs(specified[2] - real[2]) / max(real[2], 0.01)
                    )
                    if diff_ratio > 1.0:  # 超过100%的差异
                        result["issues"].append(
                            f"{obj_info['desc']}: user specified size {[round(s,2) for s in specified]} differs greatly "
                            f"from real asset size {[round(s,2) for s in real]} (ground truth)"
                        )
                        critical_issues += 1
                    elif diff_ratio > 0.5:  # 超过50%的差异
                        result["issues"].append(
                            f"{obj_info['desc']}: user specified size {[round(s,2) for s in specified]} differs "
                            f"from real asset size {[round(s,2) for s in real]} (ground truth)"
                        )
                        minor_issues += 1
        
        # 根据问题数量计算评分
        if critical_issues >= 3:
            result["score"] = -1.0
        elif critical_issues >= 2:
            result["score"] = -0.5
        elif critical_issues >= 1:
            result["score"] = 0.0
        elif minor_issues >= 3:
            result["score"] = 0.0
        elif minor_issues >= 1:
            result["score"] = 0.5
        else:
            result["score"] = 1.0
        
        self.logger.info(f"Object size evaluation (programmatic): score={result['score']}, "
                        f"critical_issues={critical_issues}, minor_issues={minor_issues}, issues={result['issues']}")
        
        return result

    async def _vlm_extract_and_evaluate_key_objects(
        self,
        image_path: str,
        scene_summary: str,
        user_requirement: str,
        room_type: str = "",
        instance_id: str = None
    ) -> Dict[str, Any]:
        """
        从用户需求中提取关键物体并评估场景是否包含这些物体
        
        步骤:
        1. 根据房间类型和用户指令，让VLM提取3-5个关键物体和1个最关键的物体（仅首次调用，后续使用缓存）
        2. 让VLM判断当前场景是否包含这些物体
        3. 返回评分和详细信息
        
        参数:
            image_path: 渲染图路径
            scene_summary: 场景物体摘要（由_extract_objects_summary提取）
            user_requirement: 用户需求描述
            room_type: 房间类型（如bedroom, living room等）
            instance_id: 实例ID，用于缓存关键物体列表
            
        返回:
            包含评分和详细信息的字典：
            - score: 评分 (1.0, 0.0, -1.0)
            - key_objects: 关键物体列表
            - most_critical_object: 最关键的物体
            - found_objects: 在场景中找到的物体列表
            - missing_objects: 在场景中缺失的物体列表
        """
        result = {
            "score": 0.0,
            "key_objects": [],
            "most_critical_object": "",
            "found_objects": [],
            "missing_objects": [],
            "essential_objects_missing": [],  # 新增：缺失的基础必备物体
            "essential_objects_found": []     # 新增：找到的基础必备物体
        }
        
        if not self.vlm_judge_enabled:
            self.logger.info("VLM judge disabled, skipping key objects evaluation")
            return result
        
        # ========== 新增：检查房间类型的基础必备物体 ==========
        # 标准化房间类型（使用 match_room_type 函数增强匹配）
        normalized_room_type = ""
        
        # 首先尝试从传入的 room_type 参数匹配
        if room_type:
            normalized_room_type = match_room_type(room_type) or ""
        
        # 如果没有匹配到，尝试从用户需求中推断房间类型
        if not normalized_room_type:
            normalized_room_type = match_room_type(user_requirement) or ""
        
        # 获取该房间类型的基础物体定义
        essential_config = ROOM_TYPE_ESSENTIAL_OBJECTS.get(normalized_room_type, None)
        
        self.logger.info(f"Room type: '{normalized_room_type}', essential config: {essential_config is not None}")
        
        # 第一步：检查缓存，如果有缓存则直接使用
        most_critical = None
        key_objects = None
        
        if instance_id and instance_id in self._instance_dict:
            instance = self._instance_dict[instance_id]
            cached_key_objects = instance.get("cached_key_objects")
            cached_most_critical = instance.get("cached_most_critical_object")
            
            if cached_key_objects and cached_most_critical:
                most_critical = cached_most_critical
                key_objects = cached_key_objects
                result["most_critical_object"] = most_critical
                result["key_objects"] = key_objects
                self.logger.info(f"Using cached key objects: {key_objects}, most critical: {most_critical}")
        
        # 如果没有缓存，调用VLM提取关键物体
        if not most_critical or not key_objects:
            prompt_extract = f"""You are an expert interior designer. Based on the room type and user requirement, identify the MANDATORY OBJECTS that MUST be present in this room.

**ROOM TYPE:** {room_type if room_type else "(Infer from user requirement)"}

**USER REQUIREMENT:**
{user_requirement}

**YOUR TASK: Generate a list of 5-15 mandatory objects.**

1. Identify 5-15 objects that are ESSENTIAL for this room.
2. **IMPORTANT:** If the room typically requires multiple instances of an item (e.g., dining chairs, nightstands), LIST THEM MULTIPLE TIMES.
   - Example for Dining Room: ["dining table", "dining chair", "dining chair", "dining chair", "dining chair", "sideboard"]
   - Example for Living Room: ["sofa", "coffee table", "TV stand", "TV", "armchair", "floor lamp"]
   - Example for Bedroom: ["double bed", "nightstand", "nightstand", "wardrobe", "lamp", "lamp"]
   - Example for Study Room: ["desk", "office chair", "bookshelf", "desk lamp", "armchair"]
   - Example for Office: ["large desk", "office chair", "guest chair", "guest chair", "filing cabinet", "bookshelf"]
   - Example for Gym: ["treadmill", "exercise bike", "weight bench", "dumbbell set", "yoga mat", "mirror"]
   - Example for Entertainment Room: ["billiard table", "bar stool", "bar stool", "sofa", "TV", "gaming console", "speaker"]

**Guidelines:**
- Include the core furniture for the room type (bed, sofa, table, etc.)
- Include specific items requested by the user.
- Include necessary functional items (toilet, stove, etc.)
- The list size MUST be between 5 and 15 items.

**OUTPUT FORMAT (JSON only, no other text):**
```json
{{
    "mandatory_objects": ["<object1>", "<object2>", "<object3>", ...]
}}
```"""
        
        try:
            # 调用VLM提取关键物体（不需要图像）
            data = {
                "model": self.vlm_judge_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_extract
                    }
                ],
                "max_tokens": 300,
                "temperature": 0.1
            }
            
            response = await asyncio.to_thread(
                requests.post,
                self.vlm_judge_url,
                json=data,
                timeout=self.vlm_judge_timeout
            )
            
            if response.status_code != 200:
                self.logger.warning(f"Key objects extraction request failed with status {response.status_code}")
                return result
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                extracted_json = json.loads(json_match.group(1))
            else:
                # 尝试直接解析
                extracted_json = json.loads(content)
            
            key_objects = extracted_json.get("mandatory_objects", [])
            # 兼容旧格式
            if not key_objects:
                key_objects = extracted_json.get("key_objects", [])
            
            # 确保列表不为空
            if not key_objects:
                self.logger.warning("Failed to extract mandatory objects from VLM response")
                return result
            
            # 简单的most_critical逻辑（取第一个）
            most_critical = key_objects[0] if key_objects else ""
            
            result["most_critical_object"] = most_critical
            result["key_objects"] = key_objects
            self.logger.info(f"Extracted mandatory objects: {key_objects}")
            
            # 缓存关键物体列表，避免后续重复调用VLM
            if instance_id and instance_id in self._instance_dict:
                self._instance_dict[instance_id]["cached_key_objects"] = key_objects
                self._instance_dict[instance_id]["cached_most_critical_object"] = most_critical
                self.logger.info(f"Cached key objects for instance {instance_id}")
            
        except Exception as e:
            self.logger.warning(f"Key objects extraction failed: {e}")
            return result
        
        # 第二步：检查场景是否包含这些物体
        key_objects_str = str(key_objects)
        
        prompt_evaluate = f"""You are an expert interior designer evaluating a room scene.

**IMPORTANT - USE VISUAL ANNOTATIONS IN THE IMAGE:**
- **Floor coordinate grid**: The TOP VIEW (left) shows a coordinate grid on the floor.
- **Bounding boxes**: Each object has a colored bounding box (bbox) drawn around it.

**MANDATORY OBJECTS LIST (Target):**
{key_objects_str}

**CURRENT SCENE OBJECTS (extracted summary):**
{scene_summary}

**YOUR TASK: Check how many of the mandatory objects are present in the scene.**

Examine both the image and the object summary.
For each item in the Mandatory Objects List, check if a corresponding object exists in the scene.
- If the list has "chair", "chair", "chair", you need to find 3 separate chairs.
- Be flexible with naming (e.g., "couch" matches "sofa").

**OUTPUT FORMAT (JSON only, no other text):**
```json
{{
    "found_count": <number of items found>,
    "total_count": <total number of items in mandatory list>,
    "found_objects": ["<list of matched objects>"],
    "missing_objects": ["<list of missing objects>"]
}}
```"""
        
        try:
            # 调用VLM评估（需要图像）
            score_result = await self._call_vlm_judge(
                image_path,
                prompt_evaluate,
                max_tokens=300,
                return_text=True
            )
            
            if not score_result:
                self.logger.warning("Key objects evaluation VLM call returned None")
                return result
            
            # 解析JSON响应
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', score_result, re.DOTALL)
            if json_match:
                eval_json = json.loads(json_match.group(1))
            else:
                # 尝试直接解析
                eval_json = json.loads(score_result)
            
            found_count = eval_json.get("found_count", 0)
            total_count = eval_json.get("total_count", len(key_objects))
            found_objects = eval_json.get("found_objects", [])
            missing_objects = eval_json.get("missing_objects", [])
            
            result["found_objects"] = found_objects
            result["missing_objects"] = missing_objects
            
            # ========== 新增：检查基础必备物体 ==========
            essential_missing = []
            essential_found = []
            
            if essential_config:
                essential_items = essential_config.get("essential", [])
                aliases_map = essential_config.get("aliases", {})
                
                # 将场景中找到的物体描述转为小写进行匹配
                found_objects_lower = [obj.lower() for obj in found_objects]
                scene_summary_lower = scene_summary.lower() if scene_summary else ""
                
                for essential_item in essential_items:
                    # 获取该物体的所有别名
                    item_aliases = aliases_map.get(essential_item, [essential_item])
                    
                    # 检查是否在找到的物体中或场景摘要中
                    found = False
                    for alias in item_aliases:
                        alias_lower = alias.lower()
                        # 检查found_objects
                        for found_obj in found_objects_lower:
                            if alias_lower in found_obj or found_obj in alias_lower:
                                found = True
                                break
                        if not found:
                            # 检查场景摘要
                            if alias_lower in scene_summary_lower:
                                found = True
                        if found:
                            break
                    
                    if found:
                        essential_found.append(essential_item)
                    else:
                        essential_missing.append(essential_item)
                
                result["essential_objects_found"] = essential_found
                result["essential_objects_missing"] = essential_missing
                
                self.logger.info(f"Essential objects check - Found: {essential_found}, Missing: {essential_missing}")
            
            # ========== 计算评分（结合基础物体检查） ==========
            # 如果缺失基础必备物体，直接判定为-1
            if essential_missing:
                result["score"] = -1.0
                self.logger.info(f"Key objects score: -1.0 (Missing ESSENTIAL objects: {essential_missing})")
            elif total_count == 0:
                result["score"] = 0.0
            else:
                ratio = found_count / total_count
                if ratio >= 0.99:  # 允许微小误差，视为全部包含
                    result["score"] = 1.0
                    self.logger.info(f"Key objects score: 1.0 (Found {found_count}/{total_count}, ratio={ratio:.2f})")
                elif ratio > 0.5:
                    result["score"] = 0.0
                    self.logger.info(f"Key objects score: 0.0 (Found {found_count}/{total_count}, ratio={ratio:.2f})")
                else:
                    result["score"] = -1.0
                    self.logger.info(f"Key objects score: -1.0 (Found {found_count}/{total_count}, ratio={ratio:.2f})")
            
        except Exception as e:
            self.logger.warning(f"Key objects evaluation failed: {e}")
            # 如果评估失败，返回中性分数
            result["score"] = 0.0
        
        return result
    
    async def _vlm_describe_scene(
        self,
        image_path: str,
        scene_json_str: str,
        user_requirement: str
    ) -> Optional[str]:
        """
        第一阶段：让VLM详细描述场景（用于最终轮次）
        
        参数:
            image_path: 渲染图路径
            scene_json_str: 场景JSON字符串
            user_requirement: 用户需求
            
        返回:
            详细场景描述文本，或None（如果失败）
        """
        prompt_describe = f"""You are a highly critical interior design expert. Carefully examine this final scene rendering (left: top view, right: diagonal view).

**IMPORTANT - USE VISUAL ANNOTATIONS IN THE IMAGE:**
- **Floor coordinate grid**: The TOP VIEW (left) shows a coordinate grid on the floor with axis labels. Use this to precisely describe object positions.
- **Bounding boxes**: Each object has a colored bounding box (bbox) drawn around it. Use these to describe object sizes and spatial relationships.

User requirement: {user_requirement}

Scene JSON for reference:
```json
{scene_json_str}
```

**YOUR TASK: Provide a comprehensive, detailed description of this scene.**

Describe the scene thoroughly, covering ALL of the following aspects:

## 1. OBJECT INVENTORY
List every visible object in the scene:
- Object name/type
- Approximate position in the room (use coordinate references from TOP VIEW)
- Size (small/medium/large)
- Color/material if visible

## 2. SPATIAL LAYOUT
- Room dimensions and shape (from floor grid)
- How objects are distributed across the room
- Which areas are occupied vs empty
- Overall layout pattern (clustered, spread out, L-shaped arrangement, etc.)

## 3. OBJECT RELATIONSHIPS
- Which objects are grouped together (functional groups)
- Distances between key furniture pieces
- Facing directions (what faces what)
- Supporting relationships (what's on top of what)

## 4. ORIENTATIONS
- Direction each major furniture piece faces
- Whether orientations make functional sense
- Any furniture facing walls or corners inappropriately

## 5. WALL PROXIMITY
- Which furniture is against walls (and which walls)
- Which furniture is floating in open space
- Distance estimates from walls for key pieces

## 6. IDENTIFIED PROBLEMS
Based on your detailed analysis, list any problems found:

**Physical Issues:**
- Any overlapping objects (bboxes intersecting)
- Any out-of-bounds objects
- Any floating objects

**Rationality Issues:**
- Any misplaced core furniture (bed/sofa not against wall)
- Any missing essential items (e.g., no toilet in bathroom, no stove in kitchen)
- Any wrong orientations

**Distribution Issues:**
- Any clustering problems
- Any large empty areas
- Any layout imbalance

## 7. OVERALL ASSESSMENT
- Does the scene fulfill the user's requirement?
- What works well in this design?
- What are the most critical issues to address?
- Overall quality rating: EXCELLENT / GOOD / ACCEPTABLE / POOR / VERY POOR

Be thorough and specific in your description. Reference the coordinate grid and bounding boxes for precise locations."""
        
        return await self._call_vlm_judge(
            image_path, 
            prompt_describe, 
            max_tokens=1500, 
            return_text=True
        )
    
    async def _vlm_generate_expected_scene_graph(
        self,
        user_requirement: str,
        room_type: str = ""
    ) -> Optional[str]:
        """
        让VLM根据用户需求和房间类型生成期望的场景图
        
        描述场景中应该包含的10-30个关键物体及其空间关系约束
        
        参数:
            user_requirement: 用户需求描述
            room_type: 房间类型（如bedroom, living room等）
            
        返回:
            期望场景图的自由文本描述，或None（如果失败）
        """
        prompt_expected_graph = f"""You are an expert interior designer. Based on the user's requirement and room type, generate an EXPECTED SCENE GRAPH that describes what a well-designed room SHOULD contain and how objects SHOULD be arranged.

**USER REQUIREMENT:**
{user_requirement}

**ROOM TYPE (if identifiable):**
{room_type if room_type else "Infer from the user requirement"}

**YOUR TASK: Generate an Expected Scene Graph**

Describe the IDEAL scene that should be created, including:

## 1. KEY OBJECTS (10-30 essential furniture items)
List the important furniture that MUST be present for this room type:
- For Bedroom: bed, wardrobe/closet, nightstand(s), dresser, etc.
- For Living Room: sofa, coffee table, TV stand, TV, armchair(s), etc.
- For Dining Room: dining table, dining chairs, sideboard, etc.
- For Study/Office: desk, office chair, bookshelf, lamp, etc.
- For Bathroom: toilet, sink/vanity, bathtub/shower, mirror, etc.
- For Kitchen: cabinets, stove/oven, refrigerator, sink, counter, etc.
- For Gym: treadmill, weights, bench, yoga mat, mirror, etc.

## 2. CRITICAL SPATIAL CONSTRAINTS (Relationship Rules)
Describe the spatial relationships that MUST be satisfied:

**Wall Placement Constraints:**
- Which furniture MUST be against a wall (e.g., bed, sofa, wardrobe, TV stand)
- Which walls are preferred (e.g., bed headboard against solid wall, not window wall)

**Proximity Constraints:**
- What should be NEXT TO what (e.g., nightstand next to bed, coffee table in front of sofa)
- What should be NEAR what (e.g., desk near window for natural light)

**Support Constraints:**
- What should be ON TOP OF what (e.g., TV on TV stand, lamp on nightstand, books on bookshelf)

**Facing/Orientation Constraints:**
- What should FACE what (e.g., sofa facing TV, desk chair facing desk)
- What should NOT face (e.g., bed should not directly face door)

**Functional Grouping:**
- Which objects form functional groups (e.g., dining table + chairs, desk + chair + lamp)
- How groups should be spatially organized

## 3. LAYOUT PRINCIPLES
- How should furniture be distributed in the room (balanced, not clustered)
- What areas should remain clear for circulation
- Overall spatial organization strategy

Be specific and practical. Focus on the MOST IMPORTANT constraints that define a functional, well-designed room."""

        # 这里不需要图像，只需要文本生成
        # 使用一个简单的文本请求
        if not self.vlm_judge_enabled:
            self.logger.info("VLM judge disabled, skipping expected scene graph generation")
            return None
        
        import requests
        
        data = {
            "model": self.vlm_judge_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_expected_graph
                }
            ],
            "max_tokens": 1200,
            "temperature": 0.1
        }
        
        try:
            response = await asyncio.to_thread(
                requests.post,
                self.vlm_judge_url,
                json=data,
                timeout=self.vlm_judge_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                self.logger.info(f"Expected scene graph generated: {content[:300]}...")
                return content
            else:
                self.logger.warning(f"Expected scene graph request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Expected scene graph generation failed: {e}")
            return None
    
    async def _vlm_generate_actual_scene_graph(
        self,
        image_path: str,
        scene_json_str: str,
        user_requirement: str
    ) -> Optional[str]:
        """
        让VLM根据渲染图和场景JSON生成实际场景的场景图
        
        描述当前场景中物体的实际位置和空间关系
        
        参数:
            image_path: 渲染图路径
            scene_json_str: 场景JSON字符串
            user_requirement: 用户需求
            
        返回:
            实际场景图的自由文本描述，或None（如果失败）
        """
        prompt_actual_graph = f"""You are an expert interior designer analyzing a rendered scene. Generate an ACTUAL SCENE GRAPH that describes the current state of the room.

**IMPORTANT - USE VISUAL ANNOTATIONS IN THE IMAGE:**
- **Floor coordinate grid**: The TOP VIEW (left) shows a coordinate grid. Use this to describe precise positions.
- **Bounding boxes**: Each object has a colored bounding box. Use these to identify objects and their boundaries.

**USER REQUIREMENT (for context):**
{user_requirement}

**SCENE JSON (for object details):**
```json
{scene_json_str}
```

**YOUR TASK: Generate an Actual Scene Graph**

Describe the CURRENT state of the scene, focusing on:

## 1. OBJECTS PRESENT
List all visible key furniture items in the scene:
- Object type/name
- Approximate position in the room

## 2. ACTUAL SPATIAL RELATIONSHIPS

**Wall Placement (What IS against walls):**
- Which furniture is currently against which wall
- Which furniture is NOT against any wall (floating in open space)
- Example: "Bed is against the north wall", "Sofa is floating in the center, not against any wall"

**Proximity Relationships (What IS next to what):**
- Which objects are adjacent to each other
- Example: "Nightstand is next to the bed on the left side", "Coffee table is in front of the sofa"

**Support Relationships (What IS on top of what):**
- Which objects are placed on other objects
- Example: "Lamp is on the nightstand", "TV is on the TV stand"

**Facing/Orientation (What IS facing what):**
- Direction each major furniture faces
- Example: "Sofa is facing the TV", "Desk is facing the window"

**Functional Groups (How objects ARE grouped):**
- Which objects form groups in the current layout
- Example: "Dining set: table with 4 chairs around it", "Bedroom set: bed with nightstands on both sides"

## 3. SPATIAL DISTRIBUTION
- How is furniture actually distributed in the room
- Are there empty areas or crowded areas
- Is the layout balanced or unbalanced

Be objective and accurate. Describe what you SEE, not what should be."""

        return await self._call_vlm_judge(
            image_path,
            prompt_actual_graph,
            max_tokens=1200,
            return_text=True
        )
    
    async def _vlm_compare_scene_graphs(
        self,
        image_path: str,
        expected_graph: str,
        actual_graph: str,
        user_requirement: str
    ) -> Optional[float]:
        """
        让VLM对比期望场景图和实际场景图，评估约束满足程度
        
        参数:
            image_path: 渲染图路径（用于视觉验证）
            expected_graph: 期望场景图描述
            actual_graph: 实际场景图描述
            user_requirement: 用户需求
            
        返回:
            场景图约束满足度评分 (-1.0, -0.5, 0.0, 0.5, 1.0)
        """
        prompt_compare = f"""You are an expert interior designer evaluating how well a generated scene satisfies the expected spatial constraints.

**USER REQUIREMENT:**
{user_requirement}

**EXPECTED SCENE GRAPH (What SHOULD be):**
```
{expected_graph}
```

**ACTUAL SCENE GRAPH (What IS):**
```
{actual_graph}
```

**YOUR TASK: Compare the two scene graphs and evaluate constraint satisfaction.**

Carefully compare the EXPECTED constraints with the ACTUAL relationships:

## Evaluation Criteria:

**1. Wall Placement Constraints:**
- Are objects that SHOULD be against walls actually against walls?
- Critical: bed, sofa, wardrobe, TV stand MUST be against walls

**2. Proximity Constraints:**
- Are objects that SHOULD be next to each other actually adjacent?
- Example: nightstand next to bed, coffee table near sofa

**3. Support Constraints:**
- Are objects that SHOULD be on other objects actually placed on them?
- Example: TV on TV stand, lamp on table

**4. Facing/Orientation Constraints:**
- Are objects facing the correct direction?
- Example: sofa facing TV, chairs facing table

**5. Functional Grouping:**
- Are functional groups properly formed and positioned?

## 5-LEVEL SCORING:

**1.0** = EXCELLENT - All or nearly all critical constraints satisfied
- All key furniture against appropriate walls
- All proximity relationships correct
- All support relationships correct
- Orientations make functional sense

**0.5** = GOOD - Most constraints satisfied, minor issues
- Most wall placements correct, maybe 1 item not ideal
- Most proximities correct
- Minor orientation issues

**0.0** = PARTIAL - About half of constraints satisfied
- Some critical furniture correctly placed, some not
- Mixed results on proximity and support
- Several orientation issues

**-0.5** = POOR - Most constraints NOT satisfied
- Key furniture (bed/sofa) not against walls
- Multiple proximity violations
- Poor functional grouping

**-1.0** = FAILED - Almost no constraints satisfied
- CRITICAL: Essential objects from Expected Graph are MISSING in Actual Graph
- Critical furniture floating in room center
- No proper adjacencies or groupings
- Completely dysfunctional layout

**IMPORTANT:**
- Wall placement for bed/sofa is CRITICAL - missing this is a major failure
- Consider both what's present AND how it's arranged
- Use the image to verify the actual relationships described

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""

        return await self._call_vlm_judge(
            image_path,
            prompt_compare,
            max_tokens=50,
            return_text=False
        )
    
    async def _judge_intermediate_turn(
        self,
        prev_image_path: str,
        current_image_path: str,
        user_requirement: str,
        prev_scene_summary: str = "",
        current_scene_summary: str = ""
    ) -> Dict[str, float]:
        """
        中间轮次的VLM评分：对比上一轮和本轮的渲染图，判断场景是否变得更好
        
        参数:
            prev_image_path: 上一轮渲染图路径（已拼接的俯视图+对角视图）
            current_image_path: 本轮渲染图路径（已拼接的俯视图+对角视图）
            user_requirement: 用户需求/房间描述
            prev_scene_summary: 上一轮场景的物体摘要（描述、位置、大小）
            current_scene_summary: 当前场景的物体摘要（描述、位置、大小）
            
        返回:
            包含一个维度评分的字典：
            - scene_improvement: 场景是否改进 (-1.0, -0.5, 0.0, 0.5, 1.0)
        """
        self.logger.info("Starting intermediate turn VLM judge evaluation (scene improvement comparison)")
        
        scores = {
            "scene_improvement": 0.0
        }
        
        # 检查两张图像是否都存在
        if not prev_image_path or not Path(prev_image_path).exists():
            self.logger.warning(f"Previous image not found: {prev_image_path}")
            return scores
        if not current_image_path or not Path(current_image_path).exists():
            self.logger.warning(f"Current image not found: {current_image_path}")
            return scores
        
        # 将两张图像转换为base64
        try:
            prev_img_base64 = self.image_to_base64(prev_image_path)
            curr_img_base64 = self.image_to_base64(current_image_path)
        except Exception as e:
            self.logger.error(f"Failed to convert images to base64: {e}")
            return scores
        
        # 构建场景物体摘要部分（如果提供）
        scene_comparison_section = ""
        if prev_scene_summary or current_scene_summary:
            scene_comparison_section = f"""

**SCENE OBJECTS COMPARISON (for reference):**

--- PREVIOUS SCENE (IMAGE 1) ---
{prev_scene_summary if prev_scene_summary else "No data available"}

--- CURRENT SCENE (IMAGE 2) ---
{current_scene_summary if current_scene_summary else "No data available"}

Use this to verify:
- What objects were added, removed, or moved
- Position changes (pos values)
- Whether objects are within room bounds
"""
        
        # 构建对比评估的prompt
        prompt_compare_improvement = f"""You are an expert interior designer evaluating whether a scene has IMPROVED after editing.

**IMPORTANT: You are given TWO images:**
- **IMAGE 1 (First image)**: The PREVIOUS scene (before editing)
- **IMAGE 2 (Second image)**: The CURRENT scene (after editing)

Each image shows a room rendering with top view (left half) and diagonal view (right half).

**ROOM DESCRIPTION / USER REQUIREMENT:**
{user_requirement}{scene_comparison_section}

**YOUR TASK: Compare the two scenes and determine if the CURRENT scene (IMAGE 2) is BETTER than the PREVIOUS scene (IMAGE 1).**

Look for ANY improvement in these aspects:
- Physical: Fewer collisions/overlaps? Fewer objects out of bounds?
- Layout: Better furniture positioning? Core furniture (bed/sofa) against walls?
- Completeness: More appropriate furniture? Room more complete?
- Distribution: Better spatial balance? Less clustering?

## 3-LEVEL SCORING:

**1** = IMPROVED (Any positive change counts!)
- Collisions or out-of-bounds reduced
- Furniture added or better positioned
- Layout more balanced or functional
- Any visible improvement, even small ones

**0** = CHANGED BUT NOT IMPROVED
- Scene has visible changes
- But quality is similar to before (improvements and regressions cancel out)
- Horizontal movement without clear benefit

**-1** = WORSE OR NO CHANGE
- Scene looks exactly the same (no effort made)
- OR scene has gotten worse (more collisions, worse layout, furniture removed badly)
- Wasted editing opportunity

**IMPORTANT:**
- Be ENCOURAGING: Even small improvements deserve score 1
- If objects moved and collisions reduced → score 1
- If new furniture added appropriately → score 1
- Only give 0 if changes are truly neutral
- Give -1 if NO change or scene got worse

Output ONLY one number: -1, 0, or 1"""

        # 构建带两张图像的请求
        import requests
        
        data = {
            "model": self.vlm_judge_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_compare_improvement},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": prev_img_base64
                            }
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": curr_img_base64
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        # 调用VLM进行评分
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    self.vlm_judge_url,
                    json=data,
                    timeout=self.vlm_judge_timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result['choices'][0]['message']['content']
                    
                    # 解析评分（优先匹配3档整数评分：-1, 0, 1）
                    import re
                    # 先尝试匹配整数评分（-1, 0, 1）
                    match = re.search(r'(-1|0|1)(?:\s|$|[,.])', content)
                    if match:
                        score = float(match.group(1))
                        scores["scene_improvement"] = score
                        self.logger.info(f"Intermediate turn VLM score: {score}, response: {content}")
                        break
                    # 兼容旧的小数评分格式（-1.0, -0.5, 0.5, 1.0）
                    match = re.search(r'(-?[01])\.([05])', content)
                    if match:
                        score = float(f"{match.group(1)}.{match.group(2)}")
                        # 将旧格式映射到新的3档评分
                        if score >= 0.5:
                            score = 1.0
                        elif score <= -0.5:
                            score = -1.0
                        else:
                            score = 0.0
                        scores["scene_improvement"] = score
                        self.logger.info(f"Intermediate turn VLM score: {score}, response: {content}")
                        break
                    else:
                        self.logger.warning(f"Failed to parse score from response: {content}")
                else:
                    self.logger.warning(f"VLM judge request failed with status {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"VLM judge attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
        
        self.logger.info(f"Intermediate turn scores (scene improvement): {scores}")
        return scores
    
    def _extract_objects_summary(self, scene_data: Dict[str, Any]) -> str:
        """
        从场景中提取物体摘要信息（用于VLM评估，节省token）
        
        只提取关键信息：物体描述、位置、大小、旋转
        
        参数:
            scene_data: 场景JSON数据
            
        返回:
            物体摘要字符串（紧凑格式）
        """
        try:
            objects_list = []
            
            # 从groups结构中提取物体
            if 'groups' in scene_data and scene_data['groups']:
                for group in scene_data['groups']:
                    if 'objects' in group and group['objects']:
                        for obj in group['objects']:
                            obj_info = {
                                'desc': obj.get('desc', obj.get('object_description', 'unknown')),
                                'pos': obj.get('pos', obj.get('position', [0, 0, 0])),
                                'size': obj.get('size', [1, 1, 1]),
                                'rot': obj.get('rot', obj.get('rotation', [0, 0, 0, 1]))
                            }
                            objects_list.append(obj_info)
            
            # 也检查flat格式的objects字段
            elif 'objects' in scene_data and scene_data['objects']:
                for obj in scene_data['objects']:
                    obj_info = {
                        'desc': obj.get('desc', obj.get('object_description', 'unknown')),
                        'pos': obj.get('pos', obj.get('position', [0, 0, 0])),
                        'size': obj.get('size', [1, 1, 1]),
                        'rot': obj.get('rot', obj.get('rotation', [0, 0, 0, 1]))
                    }
                    objects_list.append(obj_info)
            
            if not objects_list:
                return "No objects in scene"
            
            # 生成紧凑格式的摘要
            summary_lines = [f"Total objects: {len(objects_list)}"]
            for i, obj in enumerate(objects_list, 1):
                # 格式化位置和大小为简短的字符串（保留2位小数）
                pos = obj['pos']
                size = obj['size']
                pos_str = f"[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
                size_str = f"[{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]"
                summary_lines.append(f"{i}. {obj['desc']}: pos={pos_str}, size={size_str}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.warning(f"Failed to extract objects summary: {e}")
            return "Error extracting objects summary"

    async def _judge_final_turn(
        self,
        image_path: str,
        user_requirement: str,
        current_scene: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        最后一轮的VLM评分：合并评估（减少VLM调用次数）
        
        阶段1：让VLM详细描述场景（包括物体清单、位置、朝向、关系、问题）
        阶段2：一次性评估三个维度：合理性、需求匹配度、场景图约束
        
        注意：已删除aesthetics（美观度）评估
        
        参数:
            image_path: 最终渲染图路径（左侧顶视图，右侧对角视图）
            user_requirement: 用户需求
            current_scene: 当前场景JSON数据
            
        返回:
            包含三个维度评分的字典：
            - rationality: 场景合理性 (-1.0, -0.5, 0.0, 0.5, 1.0) - 物体完整性、空间分布、布局真实性
            - scene_graph: 场景图约束满足度 (-1.0, -0.5, 0.0, 0.5, 1.0) - 空间关系约束
            - requirement_match: 与用户需求的匹配度 (-1.0, -0.5, 0.0, 0.5, 1.0)
        """
        self.logger.info("Starting final turn VLM judge evaluation (consolidated)")
        
        scores = {
            "rationality": 0.0,
            "scene_graph": 0.0,
            "requirement_match": 0.0
        }
        
        # 将场景JSON转换为格式化字符串
        scene_json_str = json.dumps(current_scene, indent=2, ensure_ascii=False)
        
        # ========== 第一阶段：VLM详细描述场景 ==========
        self.logger.info("Phase 1: VLM comprehensive scene description")
        scene_description = await self._vlm_describe_scene(
            image_path, scene_json_str, user_requirement
        )
        
        if scene_description is None:
            self.logger.warning("VLM scene description failed, falling back to JSON-only evaluation")
            scene_description = "Scene description unavailable. Please evaluate based on the image and JSON data only."
        else:
            self.logger.info(f"VLM scene description completed: {scene_description[:500]}...")
        
        # 从场景中提取房间类型
        room_type = current_scene.get('room_type', '')
        
        # ========== 第二阶段：合并评估三个维度（一次VLM调用） ==========
        self.logger.info("Phase 2: Consolidated evaluation (rationality + requirement_match + scene_graph)")
        
        prompt_consolidated = f"""You are a highly critical interior design expert. Evaluate this room scene on THREE dimensions.

**USER REQUIREMENT:**
{user_requirement}

**ROOM TYPE:** {room_type if room_type else "(Infer from requirement)"}

**EXPERT SCENE DESCRIPTION:**
```
{scene_description}
```

**EVALUATE THREE DIMENSIONS:**

## DIMENSION 1: RATIONALITY (场景合理性)
Check these aspects:
1. **Object Completeness**: Are essential furniture items present for this room type?
   - Bedroom: bed, wardrobe, nightstand
   - Living room: sofa, coffee table, TV stand
   - Dining room: dining table, chairs
   - Bathroom: toilet, sink, bathtub/shower
   - Kitchen: stove, fridge, sink, cabinets
   - Gym: exercise equipment (treadmill/weights)
2. **Spatial Distribution**: Is furniture well-distributed (not all in one corner)?
3. **Layout Realism**: Does it look like a real, livable room?
4. **Object Sizes**: Are sizes realistic (no miniature furniture or giant accessories)?

## DIMENSION 2: REQUIREMENT MATCH (需求匹配度)
Check these aspects:
1. **Explicit Requirements**: Are all user-requested items present?
2. **Implicit Requirements**: For the room type, are standard essentials present?
3. **Relevance**: Are objects appropriate for this room? Any irrelevant objects (e.g., toilet in bedroom)?
4. **Style/Theme**: Does it match any requested style?

## DIMENSION 3: SCENE GRAPH (场景图约束)
Check spatial relationships:
1. **Functional Groupings**: Are related objects grouped properly (e.g., nightstands near bed)?
2. **Orientations**: Do objects face sensible directions (e.g., sofa facing TV)?
3. **Accessibility**: Can furniture be accessed reasonably?
4. **Wall Proximity**: Are wall-appropriate items (bed, sofa) against walls?

**SCORING GUIDELINES (for each dimension):**
- **1.0** = Excellent (meets all criteria)
- **0.5** = Good (minor issues only)
- **0.0** = Borderline (some issues but acceptable)
- **-0.5** = Poor (significant issues)
- **-1.0** = Failed (critical issues like missing essential furniture or many irrelevant objects)

**OUTPUT FORMAT (JSON only, no other text):**
```json
{{
    "rationality": <score>,
    "requirement_match": <score>,
    "scene_graph": <score>
}}
```

Scores must be one of: -1.0, -0.5, 0.0, 0.5, or 1.0"""

        try:
            result = await self._call_vlm_judge(
                image_path,
                prompt_consolidated,
                max_tokens=200,
                return_text=True
            )
            
            if result:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
                if json_match:
                    eval_json = json.loads(json_match.group(1))
                else:
                    eval_json = json.loads(result)
                
                scores["rationality"] = float(eval_json.get("rationality", 0.0))
                scores["requirement_match"] = float(eval_json.get("requirement_match", 0.0))
                scores["scene_graph"] = float(eval_json.get("scene_graph", 0.0))
                
                self.logger.info(f"Consolidated VLM scores: {scores}")
            else:
                self.logger.warning("Consolidated VLM evaluation returned None")
                
        except Exception as e:
            self.logger.error(f"Consolidated VLM evaluation failed: {e}", exc_info=True)
            # 如果合并评估失败，尝试回退到单独评估
            self.logger.info("Falling back to separate evaluations...")
            
            # 单独评估合理性
            prompt_rationality = f"""Evaluate scene RATIONALITY only.
User requirement: {user_requirement}
Scene description: {scene_description}

Check: object completeness, spatial distribution, layout realism, object sizes.
Score: 1.0=excellent, 0.5=good, 0.0=borderline, -0.5=poor, -1.0=failed

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""
            
            score_rationality = await self._call_vlm_judge(image_path, prompt_rationality)
            if score_rationality is not None:
                scores["rationality"] = float(score_rationality)
            
            # 单独评估需求匹配
            prompt_match = f"""Evaluate REQUIREMENT MATCH only.
User requirement: {user_requirement}
Scene description: {scene_description}

Check: explicit requirements, implicit requirements, object relevance, style match.
Score: 1.0=excellent, 0.5=good, 0.0=borderline, -0.5=poor, -1.0=failed

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""
            
            score_match = await self._call_vlm_judge(image_path, prompt_match)
            if score_match is not None:
                scores["requirement_match"] = float(score_match)
            
            # 单独评估场景图约束
            prompt_graph = f"""Evaluate SCENE GRAPH constraints only.
User requirement: {user_requirement}
Scene description: {scene_description}

Check: functional groupings, orientations, accessibility, wall proximity.
Score: 1.0=excellent, 0.5=good, 0.0=borderline, -0.5=poor, -1.0=failed

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""
            
            score_graph = await self._call_vlm_judge(image_path, prompt_graph)
            if score_graph is not None:
                scores["scene_graph"] = float(score_graph)
        
        self.logger.info(f"Final turn scores: {scores}")
        return scores
    
    async def start_interaction(
        self, 
        instance_id: Optional[str] = None, 
        **kwargs
    ) -> str:
        """
        开始一个新的交互实例
        
        参数:
            instance_id: 实例ID，如果为None则自动生成
            **kwargs: 额外参数，包括：
                - initial_scene: 初始场景JSON数据
                - max_turns: 覆盖默认最大轮数
                
        返回:
            实例ID
        """
        import sys
        print(f"[SCENE_EDITING] start_interaction called with kwargs: {list(kwargs.keys())}",   
            file=sys.stderr, flush=True)
        self.logger.info("="*80)
        self.logger.info(f"start_interaction called with instance_id={instance_id}")
        
        if instance_id is None:
            import uuid
            instance_id = str(uuid.uuid4())
            self.logger.info(f"Generated new instance_id: {instance_id}")
        
        # 获取初始场景（第一轮会从模型响应提取，所以可以为空）
        initial_scene = kwargs.get("initial_scene", {})
        
        # 记录初始场景状态
        if initial_scene:
            self.logger.info(f"Initial scene provided with {len(initial_scene.get('objects', initial_scene.get('groups', [])))} objects/groups")
        else:
            self.logger.info("No initial scene provided, will extract from model response in first turn")
        
        # 创建实例输出目录
        instance_output_dir = self.output_dir / instance_id
        instance_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory: {instance_output_dir}")
        
        # 初始化实例状态
        self._instance_dict[instance_id] = {  
            "current_scene": initial_scene,  
            "turn_count": 0,  
            "max_turns": kwargs.get("max_turns", self.max_turns),  
            "output_dir": instance_output_dir,  
            "history": [],  # 存储每轮的操作历史  
            "rewards": [],  # 存储每轮的总奖励值  
            "reward_components": [],  # 新增 - 存储每轮的命名奖励组件  
            "total_reward": 0.0,
            "user_requirement": kwargs.get("user_requirement", ""),  # 保存用户需求
            # 关键物体缓存（避免每轮重复调用VLM提取）
            "cached_key_objects": None,  # 缓存的关键物体列表
            "cached_most_critical_object": None  # 缓存的最关键物体
        }
        
        # 保存初始场景（如果提供了）
        if initial_scene:
            initial_scene_path = instance_output_dir / "initial_scene.json"
            with open(initial_scene_path, 'w', encoding='utf-8') as f:
                json.dump(initial_scene, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved provided initial scene to: {initial_scene_path}")
        else:
            self.logger.info("No initial scene to save, will be generated by model in first turn")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Started interaction: {instance_id}")
            print(f"  Output dir: {instance_output_dir}")
            print(f"  Max turns: {self._instance_dict[instance_id]['max_turns']}")
            print(f"{'='*60}")
        
        self.logger.info(f"start_interaction completed for instance {instance_id}")
        self.logger.info("="*80)
        
        return instance_id
    
    def _check_room_shape_reward(self, bounds_bottom: List, bounds_top: List) -> Tuple[float, str]:
        """
        根据房间顶点数计算房间形状奖励（只检查顶点个数，只接受偶数顶点）
        
        奖励规则：
        - 4个顶点：1.0（最佳）
        - 6个顶点：0.5
        - 8个顶点：0.5
        - 其他顶点数：-1.0
        
        同时验证每个点的有效性：每个点必须是包含至少3个坐标的列表
        
        Args:
            bounds_bottom: 房间底部边界点列表
            bounds_top: 房间顶部边界点列表
        
        Returns:
            (reward, message)
        """
        try:
            import numpy as np
            
            # 首先验证每个点的有效性（必须是恰好包含3个坐标的列表）
            def validate_points(points, name):
                if not points:
                    return False, f"{name} is empty"
                for i, p in enumerate(points):
                    if not isinstance(p, (list, tuple)):
                        return False, f"{name}[{i}] is not a list/tuple: {type(p)}"
                    if len(p) != 3:
                        return False, f"{name}[{i}] must have exactly 3 coordinates, got {len(p)}"
                return True, "OK"
            
            valid_bottom, msg_bottom = validate_points(bounds_bottom, "bounds_bottom")
            if not valid_bottom:
                self.logger.warning(f"Invalid bounds_bottom: {msg_bottom}")
                return -1.0, f"Invalid bounds_bottom: {msg_bottom}"
            
            valid_top, msg_top = validate_points(bounds_top, "bounds_top")
            if not valid_top:
                self.logger.warning(f"Invalid bounds_top: {msg_top}")
                return -1.0, f"Invalid bounds_top: {msg_top}"
            
            # 转换为numpy数组
            bounds_bottom = np.array(bounds_bottom)
            bounds_top = np.array(bounds_top)
            
            num_bottom = len(bounds_bottom)
            num_top = len(bounds_top)
            
            # 顶点数不一致
            if num_bottom != num_top:
                return -1.0, f"Bottom ({num_bottom}) and top ({num_top}) have different number of points"
            
            # 计算房间尺寸用于日志
            x_coords = bounds_bottom[:, 0]
            z_coords = bounds_bottom[:, 2]
            x_size = x_coords.max() - x_coords.min()
            z_size = z_coords.max() - z_coords.min()
            
            # 只接受偶数顶点：4、6、8
            if num_bottom == 4:
                self.logger.info(f"Room has 4 vertices (best): {x_size:.2f}m × {z_size:.2f}m, reward=1.0")
                return 1.0, "4 vertices (best)"
            elif num_bottom == 6:
                self.logger.info(f"Room has 6 vertices: {x_size:.2f}m × {z_size:.2f}m, reward=0.5")
                return 0.5, "6 vertices, acceptable"
            elif num_bottom == 8:
                self.logger.info(f"Room has 8 vertices: {x_size:.2f}m × {z_size:.2f}m, reward=0.5")
                return 0.5, "8 vertices, acceptable"
            else:
                self.logger.warning(f"Room has {num_bottom} vertices (invalid, only 4/6/8 accepted): {x_size:.2f}m × {z_size:.2f}m, reward=-1.0")
                return -1.0, f"Invalid vertex count: {num_bottom} (only 4/6/8 accepted)"
            
        except Exception as e:
            return -1.0, f"Error checking room shape: {str(e)}"
    
    def _check_room_is_rectangle(self, bounds_bottom: List, bounds_top: List) -> Tuple[bool, str]:
        """
        检查房间是否有合法的顶点数（向后兼容的包装方法）
        
        只接受偶数顶点：4、6、8
        
        Args:
            bounds_bottom: 房间底部边界点列表
            bounds_top: 房间顶部边界点列表
        
        Returns:
            (is_valid, error_message)
        """
        reward, message = self._check_room_shape_reward(bounds_bottom, bounds_top)
        # 奖励 >= 0 认为是可接受的房间形状
        return reward >= 0.0, message
    
    def _check_room_type_reward(self, scene_data: Dict[str, Any], user_requirement: str) -> float:
        """
        检查生成的room_type和room_id是否与用户需求匹配
        
        Args:
            scene_data: 场景JSON数据，包含room_type和room_id字段
            user_requirement: 用户需求描述
        
        Returns:
            1.0: 两者都匹配或无法判断
            0.0: 只有一个匹配
            -1.0: 两者都不匹配
        """
        # 定义支持的房间类型（标准化为小写用于匹配）
        valid_room_types = [
            "living room", "livingroom",
            "bedroom",
            "kitchen",
            "bathroom",
            "dining room", "diningroom",
            "office",
            "study room", "studyroom", "study",
            "nursery",
            "game room", "gameroom",
            "gym",
            "home theater", "hometheater", "theater",
            "library"
        ]
        
        # 房间类型的别名映射（用于匹配不同表达方式）
        room_type_aliases = {
            "living room": ["living room", "livingroom", "living-room", "lounge"],
            "bedroom": ["bedroom", "bed room", "bed-room"],
            "kitchen": ["kitchen"],
            "bathroom": ["bathroom", "bath room", "bath-room", "restroom", "washroom"],
            "dining room": ["dining room", "diningroom", "dining-room"],
            "office": ["office", "home office", "workspace", "work room"],
            "study room": ["study room", "studyroom", "study-room", "study"],
            "nursery": ["nursery", "baby room", "children's room", "kids room"],
            "game room": ["game room", "gameroom", "game-room", "gaming room", "playroom"],
            "gym": ["gym", "fitness room", "workout room", "exercise room"],
            "home theater": ["home theater", "hometheater", "home-theater", "theater", "theatre", "media room"],
            "library": ["library", "reading room", "book room"]
        }
        
        # 如果没有用户需求，无法判断，返回1.0
        if not user_requirement:
            self.logger.info("No user requirement provided, cannot check room type")
            return 1.0
        
        # 从用户需求中提取期望的房间类型
        user_requirement_lower = user_requirement.lower()
        expected_room_type = None
        
        for canonical_type, aliases in room_type_aliases.items():
            for alias in aliases:
                if alias in user_requirement_lower:
                    expected_room_type = canonical_type
                    break
            if expected_room_type:
                break
        
        # 如果无法从用户需求中识别房间类型，返回1.0
        if not expected_room_type:
            self.logger.info(f"Could not identify room type from user requirement: {user_requirement[:100]}...")
            return 1.0
        
        self.logger.info(f"Expected room type from user requirement: {expected_room_type}")
        
        # 获取场景中的room_type和room_id
        generated_room_type = scene_data.get("room_type", "").lower().strip()
        generated_room_id = scene_data.get("room_id", "").lower().strip()
        
        self.logger.info(f"Generated room_type: '{generated_room_type}', room_id: '{generated_room_id}'")
        
        # 检查room_type是否匹配
        room_type_match = False
        if expected_room_type in room_type_aliases:
            for alias in room_type_aliases[expected_room_type]:
                # 检查生成的room_type是否包含期望类型的别名
                if alias.replace(" ", "") in generated_room_type.replace(" ", ""):
                    room_type_match = True
                    break
        
        # 检查room_id是否匹配（room_id通常包含房间类型信息，如"LivingRoom-1003"）
        room_id_match = False
        if expected_room_type in room_type_aliases:
            for alias in room_type_aliases[expected_room_type]:
                # 去掉空格后比较
                alias_no_space = alias.replace(" ", "").lower()
                if alias_no_space in generated_room_id.replace(" ", "").replace("-", "").replace("_", ""):
                    room_id_match = True
                    break
        
        self.logger.info(f"Room type match: {room_type_match}, Room ID match: {room_id_match}")
        
        # 计算奖励
        if room_type_match and room_id_match:
            return 1.0
        elif room_type_match or room_id_match:
            return 0.0
        else:
            return -1.0

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """
        生成响应并执行场景编辑
        
        参数:
            instance_id: 实例ID
            messages: 消息历史列表
            **kwargs: 额外参数，包括：
                - tool_calls: 工具调用列表
                - retrieve_assets: 是否进行资产检索 (默认True)
                
        返回:
            (is_done, response_text, reward, metadata) 元组：
            - is_done: 是否完成交互
            - response_text: 响应文本
            - reward: 本轮奖励
            - metadata: 元数据（包含渲染图像等）
        """
        self.logger.info("="*80)
        self.logger.info(f"generate_response called for instance {instance_id}")
        
        if instance_id not in self._instance_dict:
            self.logger.error(f"Instance {instance_id} not found")
            raise ValueError(f"Instance {instance_id} not found")
        
        instance = self._instance_dict[instance_id]
        instance["turn_count"] += 1
        current_turn = instance["turn_count"]
        
        self.logger.info(f"Turn {current_turn}/{instance['max_turns']}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Turn {current_turn}/{instance['max_turns']} - Instance: {instance_id}")
            print(f"{'='*60}")
        
        # 第一轮特殊处理：提取模型生成的初始场景
        if current_turn == 1:
            self.logger.info("First turn: extracting initial scene from model response")
            
            # 从最后一条assistant消息中提取<create_scene>
            initial_scene = None
            format_conversion_success = False  # 标记格式转换是否成功
            
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<create_scene>" in content:
                        initial_scene = self.extract_create_scene_from_response(content)
                        if initial_scene:
                            self.logger.info(f"Extracted initial scene from model response")
                            
                            # ===== 格式转换逻辑 =====
                            # 检测并转换场景格式
                            try:
                                # 保存原始模型输出（用于日志）
                                original_format = "grouped" if "groups" in initial_scene else "flat"
                                self.logger.info(f"Model output format: {original_format}")
                                
                                # 如果是flat格式，转换为grouped格式（用于内部处理）
                                if "objects" in initial_scene and "groups" not in initial_scene:
                                    self.logger.info("Converting flat format to grouped format for internal processing")
                                    initial_scene = convert_flat_to_grouped(initial_scene)
                                    format_conversion_success = True
                                    self.logger.info("Format conversion successful")
                                elif "groups" in initial_scene:
                                    # 已经是grouped格式，不需要转换
                                    format_conversion_success = True
                                    self.logger.info("Scene already in grouped format")
                                else:
                                    # 无法识别的格式
                                    self.logger.error("Unknown scene format: missing both 'groups' and 'objects' fields")
                                    format_conversion_success = False
                                
                            except Exception as e:
                                self.logger.error(f"Format conversion failed: {e}", exc_info=True)
                                format_conversion_success = False
                            
                            # 存储格式转换状态到实例（用于奖励计算）
                            instance["format_conversion_success"] = format_conversion_success
                            
                            self.logger.info(f"Scene has {len(initial_scene.get('groups', []))} groups")
                            
                            # 更新实例的当前场景为转换后的grouped格式
                            instance["current_scene"] = initial_scene
                            
                            # 保存模型生成的初始场景（grouped格式）
                            model_initial_scene_path = instance["output_dir"] / "model_generated_initial_scene.json"
                            with open(model_initial_scene_path, 'w', encoding='utf-8') as f:
                                json.dump(initial_scene, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"Saved model-generated initial scene to: {model_initial_scene_path}")
                            
                            if self.verbose:
                                print(f"✓ Extracted initial scene from model response")
                                print(f"  Format conversion: {'✓' if format_conversion_success else '✗'}")
                                print(f"  Groups: {len(initial_scene.get('groups', []))}")
                            break
                    break  # 只检查最后一条assistant消息
            
            if not initial_scene:
                error_msg = "第一轮：模型未生成有效的初始场景（缺少<create_scene>标签）"
                self.logger.error(error_msg)
                if self.verbose:
                    print(f"Error: {error_msg}")
                # 格式转换失败
                instance["format_conversion_success"] = False
                return True, error_msg, -1.0, {}
            
            # ===== 在渲染前验证 bounds 有效性 =====
            # 检查 bounds_bottom 和 bounds_top 是否有效，无效则直接终止对话
            room_envelope = initial_scene.get("room_envelope", {})
            bounds_bottom = room_envelope.get("bounds_bottom", initial_scene.get("bounds_bottom", []))
            bounds_top = room_envelope.get("bounds_top", initial_scene.get("bounds_top", []))
            
            # 验证 bounds 有效性
            room_shape_reward, shape_msg = self._check_room_shape_reward(bounds_bottom, bounds_top)
            invalid_bounds = room_shape_reward < 0
            if invalid_bounds:
                # 房间角点数不在规定范围内，记录警告但不终止对话
                # 设置该轮 reward 为 -1，但继续进行下一轮次
                self.logger.warning(f"第一轮：场景 bounds 数据无效 - {shape_msg}，将继续下一轮次")
                if self.verbose:
                    print(f"Warning: 场景 bounds 数据无效 - {shape_msg}，继续下一轮次")
            else:
                self.logger.info(f"Bounds validation passed: {shape_msg}")
            
            # 第一轮不执行tool_calls，直接渲染初始场景并返回
            self.logger.info("First turn: rendering initial scene without tool_calls")
            
            try:
                # 渲染初始场景（不需要tool_calls）
                from RL_utils import render_scene_quick
                
                # 设置输出路径
                img_output_path = instance["output_dir"] / f"turn_{current_turn:03d}_initial_merged.png"
                
                # 使用信号量限制并发渲染
                render_semaphore = await self._get_render_semaphore()
                async with render_semaphore:
                    img_path_result = await asyncio.to_thread(
                        render_scene_quick,
                        scene_data=initial_scene,
                        output_path=str(img_output_path),
                        return_image=True,
                        verbose=self.verbose,
                        fast_mode=True  # 启用快速渲染模式（512x512, 8采样）
                    )
                
                # render_scene_quick 返回 (path, image) 当 return_image=True
                if isinstance(img_path_result, tuple):
                    img_path, img = img_path_result
                else:
                    img_path = img_path_result
                    img = None
                
                self.logger.info(f"Initial scene rendered: {img_path}")
                
                # 如果没有图像对象，尝试加载
                if img is None and img_path and Path(img_path).exists():
                    try:
                        img = Image.open(img_path)
                        self.logger.info(f"Loaded initial scene image from: {img_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load image: {e}")
                
            except Exception as e:
                error_msg = f"初始场景渲染失败: {str(e)}"
                self.logger.error(f"Rendering error: {error_msg}", exc_info=True)
                if self.verbose:
                    print(f"Error: {error_msg}")
                return True, error_msg, -1.0, {}
            
            # 计算初始场景的奖励
            # 如果 bounds 无效，直接设置 reward 为 -1
            if invalid_bounds:
                reward = -1.0
                self.logger.info(f"Initial scene reward set to -1.0 due to invalid bounds: {shape_msg}")
            else:
                self.logger.info("Calculating initial scene reward")
                reward = await self.calculate_score(instance_id, messages=messages)
                self.logger.info(f"Initial scene reward: {reward}")
            
            # 存储奖励
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # 记录历史
            history_entry = {
                "turn": current_turn,
                "tool_calls": [],  # 第一轮没有tool_calls
                "reward": reward,
                "is_terminated": False,
                "img_path": img_path,
                "note": "Initial scene generation",
                "format_conversion_success": format_conversion_success
            }
            if invalid_bounds:
                history_entry["invalid_bounds"] = True
                history_entry["bounds_error"] = shape_msg
            instance["history"].append(history_entry)
            
            # ===== 返回flat格式给模型 =====
            # 将grouped格式转换回flat格式
            try:
                flat_scene_for_model = convert_grouped_to_flat(initial_scene)
                self.logger.info("Converted scene back to flat format for model output")
            except Exception as e:
                self.logger.error(f"Failed to convert to flat format for output: {e}")
                flat_scene_for_model = initial_scene  # 回退到原始格式
            
            # 构建响应
            response_text = "<image>\n"
            user_requirement = instance.get("user_requirement", "")
            if user_requirement:
                response_text += f"{user_requirement}\n"
            response_text += f"<current_scene>\n```json\n{json.dumps(flat_scene_for_model, indent=2, ensure_ascii=False)}\n```\n</current_scene>"
            
            metadata = {
                "image": [img] if img else [],
            }
            
            self.logger.info(f"First turn completed. Reward: {reward}")
            self.logger.info("="*80)
            
            return False, response_text, reward, metadata
        
        # 第二轮及以后：正常的tool_calls处理流程
        self.logger.info("Turn >= 2: processing tool_calls")
        
        # 提取tool_calls（从kwargs或最后一条消息）
        tool_calls = kwargs.get("tool_calls", [])
        if not tool_calls and messages:
            # 尝试从最后一条assistant消息中提取tool_calls
            # tool_calls使用<tool_calls></tool_calls>标签包裹
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<tool_calls>" in content and "</tool_calls>" in content:
                        try:
                            # 提取<tool_calls>...</tool_calls>之间的内容
                            import re
                            match = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL)
                            if match:
                                tool_calls_str = match.group(1).strip()
                                # 解析JSON
                                tool_calls = json.loads(tool_calls_str)
                                self.logger.info(f"Extracted tool_calls from assistant message: {len(tool_calls)} calls")
                                break
                        except (json.JSONDecodeError, AttributeError) as e:
                            self.logger.warning(f"Failed to parse tool_calls from message: {e}")
                            self.logger.warning(f"Content: {content[:200]}...")
                    break  # 只检查最后一条assistant消息
        
        self.logger.info(f"Tool calls: {json.dumps(tool_calls, indent=2, default=str)}")
        
        if not tool_calls:
            # 没有tool_calls，返回错误
            self.logger.warning("No tool_calls provided")
            return False, "错误：未提供tool_calls", -1.0, {}
        
        # 执行场景编辑和渲染
        self.logger.info("Starting edit_and_render_scene")
        
        # 用于记录工具调用是否成功
        tool_execution_success = True
        tool_execution_error = None
        
        try:
            # 使用信号量限制并发渲染
            render_semaphore = await self._get_render_semaphore()
            async with render_semaphore:
                edited_scene, img_path, img, is_terminated = await asyncio.to_thread(
                    edit_and_render_scene,
                    scene_data=instance["current_scene"],
                    tool_calls=tool_calls,
                    output_dir=instance["output_dir"],
                    scene_id=f"turn_{current_turn:03d}",
                    retrieve_assets=kwargs.get("retrieve_assets", True),
                    use_objaverse=self.use_objaverse,  # 使用配置的资产来源
                    return_image=True,
                    verbose=self.verbose,
                    fast_mode=True  # 启用快速渲染模式（512x512, 8采样，约2-3倍加速）
                )
            self.logger.info(f"edit_and_render_scene completed. Image: {img_path}")
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            # 捕获特定的预期错误
            tool_execution_success = False
            tool_execution_error = str(e)
            error_msg = f"场景编辑失败: {str(e)}"
            self.logger.error(f"Expected error: {error_msg}")
            if self.verbose:
                print(f"Error: {error_msg}")
            
            # 工具调用失败，但仍然计算奖励（包含工具执行失败的惩罚）
            # 使用当前场景（未修改）计算奖励
            reward = await self.calculate_score(
                instance_id, 
                messages=messages, 
                tool_execution_success=False
            )
            
            # 存储奖励
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # 记录失败的工具调用到历史
            instance["history"].append({
                "turn": current_turn,
                "tool_calls": tool_calls,
                "reward": reward,
                "is_terminated": True,
                "error": error_msg,
                "tool_execution_success": False
            })
            
            return True, error_msg, reward, {}
        except Exception as e:
            # 未预期的错误
            tool_execution_success = False
            tool_execution_error = str(e)
            error_msg = f"场景编辑遇到未预期错误: {str(e)}"
            self.logger.error(f"Unexpected error: {error_msg}", exc_info=True)
            if self.verbose:
                print(f"Unexpected Error: {error_msg}")
                import traceback
                traceback.print_exc()
            
            # 未预期错误，仍然计算奖励（包含工具执行失败的惩罚）
            reward = await self.calculate_score(
                instance_id, 
                messages=messages, 
                tool_execution_success=False
            )
            
            # 存储奖励
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # 记录失败的工具调用到历史
            instance["history"].append({
                "turn": current_turn,
                "tool_calls": tool_calls,
                "reward": reward,
                "is_terminated": True,
                "error": error_msg,
                "tool_execution_success": False
            })
            
            return True, error_msg, reward, {}
        
        # 保存编辑前的场景（用于VLM judge评估）
        scene_before_edit = instance["current_scene"]
        
        # 更新当前场景（保持为grouped格式）
        instance["current_scene"] = edited_scene
        self.logger.info("Current scene updated")
        
        # 计算奖励，传递工具执行状态、终止状态、编辑前的场景和当前图像路径
        self.logger.info("Calculating reward")
        reward = await self.calculate_score(
            instance_id, 
            messages=messages, 
            tool_execution_success=tool_execution_success,
            is_terminated=is_terminated,
            scene_before_edit=scene_before_edit,  # 传递编辑前的场景
            current_img_path=img_path  # 传递本轮生成的图像路径
        )  
        self.logger.info(f"Reward calculated: {reward}")
        
        # 存储奖励和组件  
        instance["rewards"].append(reward)    
        instance["total_reward"] += reward
        
        
        # 记录历史
        instance["history"].append({
            "turn": current_turn,
            "tool_calls": tool_calls,
            "reward": reward,
            "is_terminated": is_terminated,
            "img_path": img_path,
            "tool_execution_success": tool_execution_success
        })
        
        # 判断是否终止
        is_done = False
        
        # ===== 返回flat格式给模型 =====
        # 将grouped格式转换回flat格式
        try:
            flat_scene_for_model = convert_grouped_to_flat(edited_scene)
            self.logger.info("Converted edited scene back to flat format for model output")
        except Exception as e:
            self.logger.error(f"Failed to convert to flat format for output: {e}")
            flat_scene_for_model = edited_scene  # 回退到原始格式
        
        # 构建响应：四部分格式（含反馈）
        # 第一部分: <image> 标签
        response_text = "<image>\n"
        # response_text = ""
        
        # 第二部分: 用户需求（从 kwargs 或实例状态获取）
        user_requirement = instance.get("user_requirement", "")
        if user_requirement:
            response_text += f"{user_requirement}\n"
        
        # 第三部分: 反馈（如果有）- 在用户需求后、场景之前
        last_feedback = instance.get("last_feedback", "")
        if last_feedback:
            response_text += f"<feedback>\n{last_feedback}\n</feedback>\n"
            self.logger.info(f"Injected feedback into response: {last_feedback[:100]}...")
        
        # 第四部分: <current_scene>json</current_scene> (使用flat格式)
        response_text += f"<current_scene>\n```json\n{json.dumps(flat_scene_for_model, indent=2, ensure_ascii=False)}\n```\n</current_scene>"
        
        if is_terminated:
            is_done = True
            response_text = "Interaction terminated by terminate call"
            self.logger.info("Interaction terminated by terminate call")
        elif current_turn >= instance["max_turns"]:
            is_done = True
            response_text = "Interaction terminated by max_turns"
            self.logger.info("Interaction terminated by max_turns")
        
        if is_done:
            if self.verbose:
                print(f"\n✓ Interaction completed")
                print(f"  Total reward: {instance['total_reward']:.4f}")
                print(f"{'='*60}")
        
        # 容错机制：如果 img 为 None，尝试从 img_path 加载
        if img is None and img_path:
            try:
                from PIL import Image
                img = Image.open(img_path)
                self.logger.info(f"Loaded image from path: {img_path}")
                if self.verbose:
                    print(f"✓ Loaded image from path: {img_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load image from {img_path}: {e}")
                if self.verbose:
                    print(f"Warning: Failed to load image from {img_path}: {e}")
        
        # 准备元数据
        metadata = {
            "image": [img] if img else [],  # PIL Image对象列表
        }
        
        self.logger.info(f"generate_response completed. is_done={is_done}, reward={reward}")
        self.logger.info("="*80)
        
        return is_done, response_text, reward, metadata
    
    def _calculate_format_reward(self, messages: List[Dict[str, Any]], turn: int, instance_id: str) -> float:
        """
        计算格式reward
        
        Args:
            messages: 消息历史
            turn: 当前轮次
            instance_id: 实例ID，用于获取当前场景状态和格式转换状态
        
        第一轮：
        - 必须包含<create_scene>...</create_scene>，且不应有<think>和<conclusion>
        - JSON格式必须正确
        - 必须是空场景（groups为空列表或包含空的objects列表）
        - 格式转换必须成功（如果模型输出flat格式，转换为grouped格式必须成功）
        
        其他轮：
        - 必须按顺序包含<think>、<tool_calls>
        - tool_calls的JSON格式必须正确
        - 工具调用必须包含有效的工具名称和必需参数
        - 涉及jid的操作必须使用当前场景中存在的jid
        
        Returns:
            格式正确返回1.0，部分错误返回0到-1之间的值，完全错误或格式转换失败返回-1.0
        """
        import re
        
        # ===== 首先检查格式转换是否成功 =====
        if instance_id in self._instance_dict:
            format_conversion_success = self._instance_dict[instance_id].get("format_conversion_success", True)
            if not format_conversion_success:
                self.logger.warning(f"Turn {turn}: Format conversion failed, returning -1.0")
                return -1.0
        
        # 获取最后一条assistant消息
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
        
        if not assistant_msg:
            return -1.0
        
        # 定义有效的工具名称和必需参数
        # 注意：需要物体ID的工具可以接受 jid（3D-FUTURE）或 uid（Objaverse）
        # 这里列出的是必需参数，实际验证时会检查 jid 或 uid 是否存在
        valid_tools = {
            "add_object": ["object_description", "position", "rotation", "size"],
            "remove_object": [],  # jid 或 uid，在下面单独验证
            "move_object": ["new_position"],  # jid 或 uid，在下面单独验证
            "rotate_object": ["new_rotation"],  # jid 或 uid，在下面单独验证
            "scale_object": ["new_size"],  # jid 或 uid，在下面单独验证
            "replace_object": ["new_object_description"],  # jid_to_replace 或 uid_to_replace，在下面单独验证
            "terminate": ["reason"]
        }
        
        # 需要物体ID的工具（支持 jid 或 uid）
        # 格式: tool_name -> (jid参数名, uid参数名)
        tools_requiring_object_id = {
            "remove_object": ("jid", "uid"),
            "move_object": ("jid", "uid"),
            "rotate_object": ("jid", "uid"),
            "scale_object": ("jid", "uid"),
            "replace_object": ("jid_to_replace", "uid_to_replace")
        }
        
        if turn == 1:
            # ===== 第一轮：检查初始场景创建格式 =====
            has_create_scene = "<create_scene>" in assistant_msg and "</create_scene>" in assistant_msg
            has_think = "<think>" in assistant_msg
            has_conclusion = "<conclusion>" in assistant_msg
            
            # 检查1：不应有think和conclusion
            if has_think or has_conclusion:
                self.logger.warning("Turn 1: Found <think> or <conclusion> tags (should not exist)")
                return -1.0
            
            # 检查2：必须有create_scene标签
            if not has_create_scene:
                self.logger.warning("Turn 1: Missing <create_scene> tags")
                return -1.0
            
            # 检查3：提取并验证JSON格式
            try:
                pattern = r'<create_scene>\s*```json\s*(.*?)\s*```\s*</create_scene>'
                match = re.search(pattern, assistant_msg, re.DOTALL)
                if not match:
                    self.logger.warning("Turn 1: JSON code block not found or malformed")
                    return -1.0
                
                scene_json_str = match.group(1).strip()
                scene_data = json.loads(scene_json_str)
                
                # 检查4：必须是空场景（如果有groups字段）
                groups = scene_data.get("groups", [])
                total_objects = 0
                if groups:
                    for group in groups:
                        if isinstance(group, dict) and "objects" in group and group["objects"]:
                            total_objects += len(group["objects"])
                
                # 也检查flat格式的objects字段
                if "objects" in scene_data and scene_data["objects"]:
                    total_objects += len(scene_data["objects"])
                
                if total_objects > 0:
                    self.logger.warning(f"Turn 1: Initial scene should be empty, but has {total_objects} objects")
                    return -1.0
                
                # 检查5：验证房间形状（只检查顶点数）
                # 支持两种格式：room_envelope.bounds_bottom 或直接 bounds_bottom
                room_envelope = scene_data.get("room_envelope", {})
                bounds_bottom = room_envelope.get("bounds_bottom", scene_data.get("bounds_bottom", []))
                bounds_top = room_envelope.get("bounds_top", scene_data.get("bounds_top", []))
                
                if not bounds_bottom or not bounds_top:
                    self.logger.warning("Turn 1: Missing bounds_bottom or bounds_top in room_envelope")
                    return -1.0
                
                # 检查房间形状并根据顶点数返回不同奖励
                # 4顶点=1.0, 5顶点=0.5, 6顶点=0.0, 其他=-1.0
                room_shape_reward, shape_msg = self._check_room_shape_reward(bounds_bottom, bounds_top)
                self.logger.info(f"Turn 1: Room shape check - {shape_msg}, reward={room_shape_reward}")
                
                # 检查6：验证room_type和room_id是否与用户需求匹配
                # 从instance获取用户需求
                user_requirement = ""
                if instance_id in self._instance_dict:
                    user_requirement = self._instance_dict[instance_id].get("user_requirement", "")
                
                room_type_reward = self._check_room_type_reward(scene_data, user_requirement)
                self.logger.info(f"Turn 1: Room type check - reward={room_type_reward}")
                
                # 综合奖励：房间形状奖励 + room_type奖励
                # room_type_reward: -1.0（两者都不对）, 0.0（一个不对）, 1.0（都对或无法判断）
                if room_type_reward < 0:
                    # 两者都不对，直接返回-1
                    self.logger.warning("Turn 1: Both room_type and room_id mismatch with user requirement")
                    return -1.0
                elif room_type_reward == 0:
                    # 一个不对，返回0
                    self.logger.warning("Turn 1: Either room_type or room_id mismatch with user requirement")
                    return 0.0
                else:
                    # 都对或无法判断，再检查房间面积
                    # 只惩罚过大的房间（>30m²），其他情况不影响格式奖励
                    try:
                        import numpy as np
                        bounds_np = np.array(bounds_bottom)
                        x_size = abs(bounds_np[:, 0].max() - bounds_np[:, 0].min())
                        z_size = abs(bounds_np[:, 2].max() - bounds_np[:, 2].min())
                        area = x_size * z_size
                        self.logger.info(f"Turn 1: Room area = {area:.2f} m²")
                        
                        if area > 30:
                            self.logger.warning(f"Turn 1: Room area ({area:.2f}m²) > 30m², returning -1.0")
                            return -1.0
                    except Exception as area_e:
                        self.logger.warning(f"Turn 1: Room area calculation failed: {area_e}")
                        # 面积计算失败不影响格式奖励
                    
                    return room_shape_reward
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Turn 1: JSON parsing error: {e}")
                return -1.0
            except Exception as e:
                self.logger.warning(f"Turn 1: Unexpected error during format check: {e}")
                return -1.0
                
        else:
            # ===== 其他轮：检查编辑操作格式 =====
            has_think = "<think>" in assistant_msg and "</think>" in assistant_msg
            has_tool_calls = "<tool_calls>" in assistant_msg and "</tool_calls>" in assistant_msg
            
            # 检查1：必须包含think和tool_calls标签
            if not (has_think and has_tool_calls):
                missing = []
                if not has_think: missing.append("think")
                if not has_tool_calls: missing.append("tool_calls")
                self.logger.warning(f"Turn {turn}: Missing tags: {missing}")
                return -1.0
            
            # 检查2：检查顺序（think应该在tool_calls之前）
            think_pos = assistant_msg.find("<think>")
            tool_calls_pos = assistant_msg.find("<tool_calls>")
            
            if not (think_pos < tool_calls_pos):
                self.logger.warning(f"Turn {turn}: Wrong tag order (should be think->tool_calls)")
                return -0.8
            
            # 检查3：提取并验证tool_calls的JSON格式
            try:
                pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
                match = re.search(pattern, assistant_msg, re.DOTALL)
                if not match:
                    self.logger.warning(f"Turn {turn}: tool_calls content not found")
                    return -1.0
                
                tool_calls_str = match.group(1).strip()
                tool_calls = json.loads(tool_calls_str)
                
                # 检查4：必须是列表
                if not isinstance(tool_calls, list):
                    self.logger.warning(f"Turn {turn}: tool_calls must be a list")
                    return -1.0
                
                # 检查5：不能为空（除非是terminate）
                if len(tool_calls) == 0:
                    self.logger.warning(f"Turn {turn}: tool_calls list is empty")
                    return -1.0
                
                # 获取当前场景中的所有物体ID（jid 和 uid）
                # 使用传入的instance_id，确保并发安全
                if instance_id not in self._instance_dict:
                    self.logger.warning(f"Turn {turn}: instance_id {instance_id} not found in _instance_dict")
                    return -1.0
                
                current_scene = self._instance_dict[instance_id].get("current_scene", {})
                valid_jids = set()  # 3D-FUTURE 物体 ID
                valid_uids = set()  # Objaverse 物体 ID
                valid_object_ids = set()  # 所有物体 ID（jid + uid）
                
                if current_scene and 'groups' in current_scene:
                    for group in current_scene.get('groups', []):
                        if 'objects' in group:
                            for obj in group.get('objects', []):
                                if 'jid' in obj and obj['jid']:
                                    valid_jids.add(obj['jid'])
                                    valid_object_ids.add(obj['jid'])
                                if 'uid' in obj and obj['uid']:
                                    valid_uids.add(obj['uid'])
                                    valid_object_ids.add(obj['uid'])
                
                self.logger.debug(f"Turn {turn}: Found {len(valid_jids)} jids, {len(valid_uids)} uids in current scene")
                
                # 检查6：验证每个工具调用
                penalty = 0.0
                for i, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] is not a dict")
                        penalty += 0.1
                        continue
                    
                    # 检查必需字段
                    if "name" not in tool_call:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] missing 'name' field")
                        penalty += 0.1
                        continue
                    
                    tool_name = tool_call["name"]
                    
                    # 检查工具名称是否有效
                    if tool_name not in valid_tools:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] has invalid tool name '{tool_name}'")
                        penalty += 0.15
                        continue
                    
                    # 检查arguments字段
                    if "arguments" not in tool_call:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] missing 'arguments' field")
                        penalty += 0.1
                        continue
                    
                    arguments = tool_call["arguments"]
                    if not isinstance(arguments, dict):
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] arguments is not a dict")
                        penalty += 0.1
                        continue
                    
                    # 检查必需参数
                    required_params = valid_tools[tool_name]
                    missing_params = [p for p in required_params if p not in arguments]
                    if missing_params:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] ({tool_name}) missing required params: {missing_params}")
                        penalty += 0.1
                    
                    # 检查需要物体ID的工具（支持 jid 或 uid）
                    if tool_name in tools_requiring_object_id:
                        jid_param, uid_param = tools_requiring_object_id[tool_name]
                        has_jid = jid_param in arguments and arguments[jid_param]
                        has_uid = uid_param in arguments and arguments[uid_param]
                        
                        # 必须提供 jid 或 uid 中的一个
                        if not has_jid and not has_uid:
                            self.logger.warning(f"Turn {turn}: tool_call[{i}] ({tool_name}) missing object ID (need '{jid_param}' or '{uid_param}')")
                            penalty += 0.15
                        else:
                            # 验证提供的ID是否存在于当前场景
                            id_valid = False
                            id_value = None
                            id_type = None
                            
                            if has_jid:
                                id_value = arguments[jid_param]
                                id_type = "jid"
                                id_valid = id_value in valid_jids
                            elif has_uid:
                                id_value = arguments[uid_param]
                                id_type = "uid"
                                id_valid = id_value in valid_uids
                            
                            if not id_valid:
                                self.logger.warning(f"Turn {turn}: tool_call[{i}] ({tool_name}) uses non-existent {id_type} '{id_value}'")
                                if id_type == "jid":
                                    self.logger.warning(f"  Valid jids: {list(valid_jids)[:5]}{'...' if len(valid_jids) > 5 else ''}")
                                else:
                                    self.logger.warning(f"  Valid uids: {list(valid_uids)[:5]}{'...' if len(valid_uids) > 5 else ''}")
                                penalty += 0.2  # ID不存在是严重错误，给较大惩罚
                
                # 计算最终分数
                final_score = 1.0 - min(penalty, 1.0)
                return max(final_score, -1.0)
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Turn {turn}: tool_calls JSON parsing error: {e}")
                return -0.9
            except Exception as e:
                self.logger.warning(f"Turn {turn}: Unexpected error during format check: {e}")
                return -1.0
    
    def _calculate_object_count_reward(self, scene_data: Dict[str, Any], is_final_turn: bool = False) -> Tuple[float, bool]:
        """
        计算物体数量reward
        
        新的奖励规则：
        - 物体数量 >= 15：reward = 1.0
        - 物体数量 5-15：线性插值从 0.0 到 +1.0（5个=0，15个=1）
        - 物体数量 4-5：线性插值从 -1.0 到 0.0（4个=-1，5个=0）
        - 物体数量 < 4（仅最后一轮）：直接返回全局 -1 的总奖励
        - 物体数量 < 4（非最后一轮）：reward = -1.0
        
        Args:
            scene_data: 场景数据
            is_final_turn: 是否是最后一轮
        
        Returns:
            (reward, should_override_total): 
            - reward: 物体数量奖励值
            - should_override_total: 如果为True，表示应该直接将总奖励设为-1
        """
        # 统计物体总数
        total_objects = 0
        if 'groups' in scene_data and scene_data['groups']:
            for group in scene_data['groups']:
                if 'objects' in group and group['objects']:
                    total_objects += len(group['objects'])
        elif 'objects' in scene_data and scene_data['objects']:
            # 兼容直接objects格式
            total_objects = len(scene_data['objects'])
        
        self.logger.info(f"Object count: {total_objects} (is_final_turn={is_final_turn})")

        # 特殊规则：最后一轮物体数量少于4个，直接返回-1并标记覆盖总奖励
        if is_final_turn and total_objects < 4:
            self.logger.warning(f"Final turn: Object count ({total_objects}) < 4, overriding total reward to -1")
            return -1.0, True
        
        # 定义奖励区间
        if total_objects >= 15:
            # 物体数量充足，满分
            return 1.0, False
        elif 5 <= total_objects < 15:
            # 5-15个物体，线性插值从 0.0 到 +1.0
            # total_objects=5 -> 0.0, total_objects=15 -> +1.0
            return (total_objects - 5) / 10.0, False
        elif 4 <= total_objects < 5:
            # 4-5个物体，线性插值从 -1.0 到 0.0
            # total_objects=4 -> -1.0, total_objects=5 -> 0.0
            return -1.0 + (total_objects - 4) / 1.0, False
        else:  # total_objects < 4
            # 非最后一轮，物体太少，返回 -1.0 但不覆盖总奖励
            return -1.0, False
    
    def _get_room_boundaries(self, scene_data: Dict[str, Any]) -> Dict[str, float]:
        """
        从room_envelope提取房间边界
        
        Returns:
            包含floor_y, ceiling_y, x_min, x_max, z_min, z_max的字典
        """
        room_envelope = scene_data.get('room_envelope', {})
        bounds_bottom = room_envelope.get('bounds_bottom', [])
        bounds_top = room_envelope.get('bounds_top', [])
        
        if not bounds_bottom or not bounds_top:
            raise ValueError("Missing room_envelope bounds")
        
        import numpy as np
        bounds_bottom = np.array(bounds_bottom)
        bounds_top = np.array(bounds_top)
        
        return {
            'floor_y': bounds_bottom[0][1],
            'ceiling_y': bounds_top[0][1],
            'x_min': bounds_bottom[:, 0].min(),
            'x_max': bounds_bottom[:, 0].max(),
            'z_min': bounds_bottom[:, 2].min(),
            'z_max': bounds_bottom[:, 2].max()
        }
    
    async def _classify_object_support_type(self, obj_desc: str) -> str:
        """
        分类物体的支撑类型（优先关键词匹配，fallback到LLM）
        
        Args:
            obj_desc: 物体描述文本
        
        Returns:
            支撑类型：'floor', 'surface', 'ceiling', 'wall'
        """
        # 检查缓存
        if obj_desc in self.support_type_cache:
            return self.support_type_cache[obj_desc]
        
        desc_lower = obj_desc.lower()
        
        # 英文关键词匹配
        floor_keywords = ['chair', 'sofa', 'bed', 'table', 'desk', 'cabinet', 'shelf',
                         'wardrobe', 'dresser', 'nightstand', 'stool', 'bench', 'ottoman',
                         'couch', 'armchair', 'bookcase', 'sideboard', 'credenza', 'console',
                         'stand', 'cart', 'rack']
        
        surface_keywords = ['lamp', 'vase', 'book', 'clock', 'plant', 'decor', 'decoration',
                           'sculpture', 'figurine', 'bowl', 'tray', 'candle', 'photo frame',
                           'picture frame', 'pot', 'jar', 'bottle', 'box', 'container',
                           'accessory', 'ornament', 'knick-knack', 'display']
        
        ceiling_keywords = ['chandelier', 'pendant', 'ceiling lamp', 'ceiling light',
                          'hanging light', 'suspended light', 'drop light', 'lantern']
        
        wall_keywords = ['painting', 'mirror', 'wall lamp', 'sconce', 'picture', 'artwork',
                        'wall art', 'frame', 'wall shelf', 'wall cabinet', 'wall clock',
                        'wall decor', 'tapestry', 'poster', 'photograph']
        
        # 关键词匹配
        for keyword in ceiling_keywords:
            if keyword in desc_lower:
                self.support_type_cache[obj_desc] = 'ceiling'
                return 'ceiling'
        
        for keyword in wall_keywords:
            if keyword in desc_lower:
                self.support_type_cache[obj_desc] = 'wall'
                return 'wall'
        
        for keyword in surface_keywords:
            if keyword in desc_lower:
                self.support_type_cache[obj_desc] = 'surface'
                return 'surface'
        
        for keyword in floor_keywords:
            if keyword in desc_lower:
                self.support_type_cache[obj_desc] = 'floor'
                return 'floor'
        
        # 未匹配到关键词，调用LLM判断
        try:
            prompt = f"""Classify the support type needed for this furniture/object: "{obj_desc}"

Support types:
- floor: Objects that stand on the floor (chairs, tables, cabinets, etc.)
- surface: Small objects placed on table/shelf surfaces (lamps, vases, books, etc.)
- ceiling: Objects hanging from ceiling (chandeliers, pendant lights, etc.)
- wall: Objects mounted on walls (paintings, mirrors, wall lamps, etc.)

Reply with only one word: floor, surface, ceiling, or wall"""
            
            # 使用VLM API进行文本分类（不需要图像）
            # 构建纯文本请求
            data = {
                "model": self.vlm_judge_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = await asyncio.to_thread(
                requests.post,
                self.vlm_judge_url,
                json=data,
                timeout=self.vlm_judge_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].lower().strip()
                
                # 提取支撑类型
                for support_type in ['ceiling', 'wall', 'surface', 'floor']:
                    if support_type in content:
                        self.support_type_cache[obj_desc] = support_type
                        self.logger.info(f"LLM classified '{obj_desc}' as '{support_type}'")
                        return support_type
        
        except Exception as e:
            self.logger.warning(f"LLM classification failed for '{obj_desc}': {e}")
        
        # 默认fallback到floor
        self.support_type_cache[obj_desc] = 'floor'
        return 'floor'
    
    def _find_support_surfaces(self, all_objects: List[Dict]) -> List[Tuple]:
        """
        识别潜在的支撑面（桌子、架子等）
        
        Returns:
            List of (jid, top_y, bbox_bounds, obj)
            bbox_bounds = (x_min, x_max, z_min, z_max)
        """
        surfaces = []
        
        surface_keywords = ['table', 'desk', 'shelf', 'cabinet', 'counter', 'nightstand',
                          'bookcase', 'stand', 'dresser', 'sideboard', 'console',
                          'credenza', 'buffet', 'bar']
        
        for obj in all_objects:
            desc = obj.get('desc', '').lower()
            size = obj.get('size', [0, 0, 0])
            pos = obj.get('pos', [0, 0, 0])
            jid = obj.get('jid', '')
            
            # 关键词匹配或尺寸启发式
            is_surface = False
            
            # 关键词检查
            for keyword in surface_keywords:
                if keyword in desc:
                    is_surface = True
                    break
            
            # 尺寸启发式：高度 < 1.2m 且 面积 > 0.2m²
            if not is_surface and size[1] < 1.2 and size[0] * size[2] > 0.2:
                is_surface = True
            
            if is_surface:
                top_y = pos[1] + size[1]
                # 计算bbox边界
                x_min = pos[0] - size[0] / 2
                x_max = pos[0] + size[0] / 2
                z_min = pos[2] - size[2] / 2
                z_max = pos[2] + size[2] / 2
                
                surfaces.append((jid, top_y, (x_min, x_max, z_min, z_max), obj))
        
        return surfaces
    
    async def _calculate_support_reward(self, scene_data: Dict[str, Any]) -> float:
        """
        计算支撑reward - 评估物体是否有合理的支撑
        
        策略:
        1. 识别房间边界（地板、天花板、墙面）
        2. 识别潜在支撑面（桌子、架子等）
        3. 对每个物体分类支撑类型并验证位置
        4. 计算无支撑物体的比例
        
        Returns:
            支撑合理返回1.0，不合理返回负值
            - 0% unsupported: 1.0
            - 0-10% unsupported: 0 to 1.0 (linear)
            - >10% unsupported: negative reward
        """
        try:
            # 获取房间边界
            boundaries = self._get_room_boundaries(scene_data)
            floor_y = boundaries['floor_y']
            ceiling_y = boundaries['ceiling_y']
            x_min, x_max = boundaries['x_min'], boundaries['x_max']
            z_min, z_max = boundaries['z_min'], boundaries['z_max']
            
            # 提取所有物体
            all_objects = []
            for group in scene_data.get('groups', []):
                all_objects.extend(group.get('objects', []))
            
            if not all_objects:
                return 0.0
            
            # 识别支撑面
            support_surfaces = self._find_support_surfaces(all_objects)
            
            # 验证每个物体的支撑
            unsupported_objects = []
            keyword_matched = 0
            llm_called = 0
            
            for obj in all_objects:
                desc = obj.get('desc', '')
                pos = obj.get('pos', [0, 0, 0])
                size = obj.get('size', [0, 0, 0])
                jid = obj.get('jid', '')
                
                # 物体底部和顶部Y坐标
                bottom_y = pos[1]
                top_y = pos[1] + size[1]
                
                # 分类支撑类型
                cache_before = len(self.support_type_cache)
                support_type = await self._classify_object_support_type(desc)
                cache_after = len(self.support_type_cache)
                
                if cache_after == cache_before:
                    keyword_matched += 1
                else:
                    # 检查是否是新的LLM调用
                    if desc not in self.support_type_cache or cache_after > cache_before:
                        llm_called += 1
                
                # 根据类型验证支撑
                is_supported = False
                support_reason = ""
                
                tolerance_y = 0.05  # 5cm tolerance for Y-axis
                tolerance_wall = 0.15  # 15cm tolerance for wall proximity
                
                if support_type == 'floor':
                    # 检查是否在地板上
                    if abs(bottom_y - floor_y) < tolerance_y:
                        is_supported = True
                        support_reason = "on floor"
                
                elif support_type == 'surface':
                    # 检查是否在某个支撑面上
                    for surf_jid, surf_top_y, surf_bounds, surf_obj in support_surfaces:
                        if abs(bottom_y - surf_top_y) < tolerance_y:
                            # 检查XZ平面重叠
                            obj_x_min = pos[0] - size[0] / 2
                            obj_x_max = pos[0] + size[0] / 2
                            obj_z_min = pos[2] - size[2] / 2
                            obj_z_max = pos[2] + size[2] / 2
                            
                            surf_x_min, surf_x_max, surf_z_min, surf_z_max = surf_bounds
                            
                            # 检查重叠
                            x_overlap = not (obj_x_max < surf_x_min or obj_x_min > surf_x_max)
                            z_overlap = not (obj_z_max < surf_z_min or obj_z_min > surf_z_max)
                            
                            if x_overlap and z_overlap:
                                is_supported = True
                                support_reason = f"on surface ({surf_obj.get('desc', 'unknown')[:30]})"
                                break
                
                elif support_type == 'ceiling':
                    # 检查是否接近天花板（允许物体悬挂在天花板下方）
                    # 物体顶部应该在天花板高度的80%-100%范围内
                    if top_y >= ceiling_y * 0.8 and top_y <= ceiling_y + tolerance_y:
                        is_supported = True
                        support_reason = "hanging from ceiling"
                
                elif support_type == 'wall':
                    # 检查是否贴近墙面
                    near_x_wall = (abs(pos[0] - x_min) < tolerance_wall or 
                                  abs(pos[0] - x_max) < tolerance_wall)
                    near_z_wall = (abs(pos[2] - z_min) < tolerance_wall or 
                                  abs(pos[2] - z_max) < tolerance_wall)
                    
                    if near_x_wall or near_z_wall:
                        is_supported = True
                        support_reason = "mounted on wall"
                
                if not is_supported:
                    unsupported_objects.append({
                        'jid': jid,
                        'desc': desc[:50],
                        'type': support_type,
                        'pos': pos
                    })
            
            # 计算无支撑物体比例
            total_objects = len(all_objects)
            unsupported_count = len(unsupported_objects)
            unsupported_rate = (unsupported_count / total_objects * 100) if total_objects > 0 else 0
            
            # 计算奖励
            if unsupported_rate == 0:
                reward = 1.0
            elif unsupported_rate <= 10:
                reward = 1.0 - unsupported_rate / 10.0
            else:
                reward = -1.0 * (unsupported_rate - 10) / 90.0
            
            # 打印详细信息
            if self.verbose:
                print(f"\n  [Support Reward Details]", file=sys.stderr, flush=True)
                print(f"    Total objects: {total_objects}", file=sys.stderr, flush=True)
                print(f"    Unsupported: {unsupported_count} ({unsupported_rate:.1f}%)", file=sys.stderr, flush=True)
                print(f"    Keyword matched: {keyword_matched}", file=sys.stderr, flush=True)
                print(f"    LLM called: {llm_called}", file=sys.stderr, flush=True)
                print(f"    Cache size: {len(self.support_type_cache)}", file=sys.stderr, flush=True)
                print(f"    Support surfaces found: {len(support_surfaces)}", file=sys.stderr, flush=True)
                
                if unsupported_objects:
                    print(f"    Unsupported objects:", file=sys.stderr, flush=True)
                    for obj_info in unsupported_objects[:5]:  # 最多显示5个
                        print(f"      - {obj_info['desc']} (type: {obj_info['type']})", file=sys.stderr, flush=True)
                    if len(unsupported_objects) > 5:
                        print(f"      ... and {len(unsupported_objects) - 5} more", file=sys.stderr, flush=True)
            
            self.logger.info(f"Support reward: {reward:.4f} (unsupported: {unsupported_count}/{total_objects})")
            
            return reward
            
        except Exception as e:
            self.logger.error(f"Support reward calculation failed: {e}", exc_info=True)
            if self.verbose:
                print(f"  [Support Reward] Calculation failed: {e}", file=sys.stderr, flush=True)
            return 0.0
    
    def _calculate_room_area_reward(self, scene_data: Dict[str, Any]) -> float:
        """
        计算房间面积reward（仅用于第一轮）
        
        注意：只对长方形房间计算面积奖励，非长方形房间直接返回负奖励。
        这是因为异型房间的包围盒面积会高估实际面积，导致奖励信号不准确。
        
        房间面积在20-40平米时给正奖励：
        - 最优区间：25-35平米，reward = 1.0
        - 边界区间：20-25和35-40，线性衰减到0.5
        - 超出范围：给负奖励
        
        Returns:
            面积合理返回0.5-1.0，不合理返回负值
        """
        try:
            room_envelope = scene_data.get('room_envelope', {})
            bounds_bottom = room_envelope.get('bounds_bottom', [])
            bounds_top = room_envelope.get('bounds_top', [])
            
            if not bounds_bottom or not bounds_top:
                self.logger.warning("Room area calculation: missing bounds")
                return -1.0
            
            # 首先检查是否为长方形，非长方形直接返回负奖励
            is_rectangle, rect_error = self._check_room_is_rectangle(bounds_bottom, bounds_top)
            if not is_rectangle:
                self.logger.warning(f"Room area calculation: room is not a rectangle - {rect_error}")
                return -1.0  # 非长方形房间，面积奖励直接为负
            
            # 计算房间尺寸（只有长方形才能准确计算）
            import numpy as np
            bounds_bottom = np.array(bounds_bottom)
            bounds_top = np.array(bounds_top)
            
            # 取X和Z轴的范围（对于长方形，这就是准确的尺寸）
            x_size = abs(bounds_bottom[:, 0].max() - bounds_bottom[:, 0].min())
            z_size = abs(bounds_bottom[:, 2].max() - bounds_bottom[:, 2].min())
            
            # 计算面积（平方米）
            area = x_size * z_size
            
            self.logger.info(f"Room area: {area:.2f} m² (size: {x_size:.2f}m × {z_size:.2f}m)")
            
            # 定义奖励区间（连续曲线，覆盖-1到1完整范围）
            # 最优区间: 20-30m², reward=1.0
            # 边界区间: 15-20和30-35m², reward从0.5线性到1.0
            # 过渡区间: 10-15和35-40m², reward从0线性到0.5
            # 惩罚区间: <10或>40m², reward从0线性到-1.0
            if 20 <= area <= 30:
                # 最优区间
                return 1.0
            elif 15 <= area < 20:
                # 下边界区间，线性插值 0.5 -> 1.0
                return 0.5 + 0.5 * (area - 15) / 5
            elif 30 < area <= 35:
                # 上边界区间，线性插值 1.0 -> 0.5
                return 1.0 - 0.5 * (area - 30) / 5
            elif 10 <= area < 15:
                # 下过渡区间，线性插值 0 -> 0.5
                return 0.5 * (area - 10) / 5
            elif 35 < area <= 40:
                # 上过渡区间，线性插值 0.5 -> 0
                return 0.5 - 0.5 * (area - 35) / 5
            elif area < 10:
                # 房间太小，负奖励（线性插值 0 -> -1.0）
                # area=10 -> 0, area=0 -> -1.0
                return max(-1.0, -0.1 * (10 - area))
            else:  # area > 40
                # 房间太大，负奖励（线性插值 0 -> -1.0）
                # area=40 -> 0, area=50 -> -1.0
                return max(-1.0, -0.1 * (area - 40))
                
        except Exception as e:
            self.logger.warning(f"Room area calculation failed: {e}")
            return -1.0
    
    async def calculate_score(self, instance_id: str, **kwargs) -> float:  
        """
        计算综合奖励，包括：
        1. 物理有效性奖励（体素评估或trimesh评估）
        2. 格式奖励
        3. 物体数量奖励
        4. 工具执行成功奖励 (仅第二轮及以后)
        5. VLM judge评分 (中间轮次和最后一轮)
        
        最终归一化到合理范围
        """  
        instance = self._instance_dict[instance_id]  
        current_scene = instance["current_scene"]  
        turn = instance["turn_count"]
        messages = kwargs.get("messages", [])
        tool_execution_success = kwargs.get("tool_execution_success", None)
        is_terminated = kwargs.get("is_terminated", False)  # 新增：获取terminate状态
        scene_before_edit = kwargs.get("scene_before_edit", None)  # 编辑前的场景
        current_img_path = kwargs.get("current_img_path", None)  # 本轮生成的图像路径
        
        # 判断是否是最后一轮（与is_done逻辑一致：terminate或达到max_turns）
        max_turns = instance.get("max_turns", self.max_turns)
        is_final_turn = is_terminated or (turn >= max_turns)
        
        # 获取VLM judge需要的场景和图像
        vlm_image_path = None
        vlm_prev_image_path = None  # 中间轮次需要上一轮的图像用于对比
        vlm_scene = None
        
        # 从history中获取图像路径
        # 注意：calculate_score是在history.append之前调用的，所以history[-1]是上一轮
        if turn == 1:
            # 第1轮没有VLM评估
            vlm_image_path = None
            vlm_prev_image_path = None
            vlm_scene = None
            self.logger.info("Turn 1: No VLM evaluation")
        elif is_final_turn:
            # 最后一轮：使用本轮生成的图像和编辑后的场景
            vlm_image_path = current_img_path
            vlm_prev_image_path = None  # 最后一轮不需要对比
            vlm_scene = current_scene  # 最后一轮评估编辑后的场景
            self.logger.info(f"Final turn {turn}: Using current image {vlm_image_path} and edited scene")
        else:
            # 中间轮次：需要上一轮的图像和本轮的图像进行对比
            vlm_image_path = current_img_path  # 本轮生成的图像
            vlm_scene = scene_before_edit  # 使用编辑前的场景
            
            if turn == 2:
                # 第2轮：上一轮是第1轮的图像（初始场景）
                if instance["history"] and len(instance["history"]) >= 1:
                    first_turn_history = instance["history"][0]
                    vlm_prev_image_path = first_turn_history.get("img_path", None)
                    self.logger.info(f"Turn 2: Comparing turn 1 image {vlm_prev_image_path} with current {vlm_image_path}")
            else:
                # 第3轮及以后的中间轮：使用上一轮的图像
                if instance["history"] and len(instance["history"]) >= 1:
                    prev_turn_history = instance["history"][-1]
                    vlm_prev_image_path = prev_turn_history.get("img_path", None)
                    self.logger.info(f"Turn {turn}: Comparing previous turn image {vlm_prev_image_path} with current {vlm_image_path}")
        
        # 定义权重（根据是否有工具执行调整）
        if turn == 1:
            # 第一轮：初始场景生成，包含格式、房间面积、分组数奖励
            weights = {
                "physics": 0.0,
                "format": 1.0,
                "object_count": 0.0,
                "collision_rate": 0.0,
                "oob_rate": 0.0,
                "penetration_depth": 0.0,  # 穿透深度体积奖励
                "oob_volume": 0.0,  # 出界体积奖励
                "support": 0.0,
                "room_area": 0.0,
                "tool_execution": 0.0,
                # VLM judge (第一轮不使用)
                "vlm_problem_identification": 0.0,
                "vlm_action_reasonableness": 0.0,
                "vlm_scene_improvement": 0.0,  # 场景改进度评估
                "vlm_key_objects": 0.0,  # 关键物体评估
                "vlm_rationality": 0.0,
                "vlm_aesthetics": 0.0,
                "vlm_requirement_match": 0.0,
                "vlm_scene_graph": 0.0  # 场景图约束评估
            }
        else:
            # 使用前面计算的is_final_turn判断（已包含terminate和max_turns两种情况）
            
            if is_final_turn:
                # ========== 最后一轮：分层评估体系 ==========
                # 格式层 (0.1) + 物体层 (0.3) + 场景层 (0.6)
                # 物体层：key_objects (0.10) + size_proportion (0.10) + object_count (0.10)
                # 场景层：物理 (0.30) + VLM (0.30)
                #   - 物理: collision + oob + support (trimesh) 或 voxel physics
                #   - VLM: rationality + requirement_match + scene_graph (删除了aesthetics)
                if self.physics_mode == "voxel":
                    weights = {
                        "physics": 0.30,  # 体素物理评估（场景层物理部分）
                        "format": 0.10,  # 格式层
                        # 物体层 (0.3)
                        "object_count": 0.0,  # 权重移至vlm_key_objects
                        "vlm_key_objects": 0.20,  # 关键物体匹配度（物体层，含数量权重）
                        "vlm_size_proportion": 0.10,  # 尺寸比例合理性（物体层）
                        # 物理指标（voxel模式不使用单独指标）
                        "collision_rate": 0.0,
                        "oob_rate": 0.0,
                        "penetration_depth": 0.0,
                        "oob_volume": 0.0,
                        "support": 0.0,
                        "room_area": 0.0,
                        "tool_execution": 0.0,
                        # VLM场景层 (0.3)
                        "vlm_problem_identification": 0.0,
                        "vlm_action_reasonableness": 0.0,
                        "vlm_scene_improvement": 0.0,
                        "vlm_rationality": 0.10,  # 场景层VLM
                        "vlm_aesthetics": 0.0,  # 已删除
                        "vlm_requirement_match": 0.10,  # 场景层VLM
                        "vlm_scene_graph": 0.10  # 场景层VLM
                    }
                else:  # trimesh
                    weights = {
                        "physics": 0.0,
                        "format": 0.10,  # 格式层
                        # 物体层 (0.3)
                        "object_count": 0.0,  # 权重移至vlm_key_objects
                        "vlm_key_objects": 0.20,  # 关键物体匹配度（物体层，含数量权重）
                        "vlm_size_proportion": 0.10,  # 尺寸比例合理性（物体层）
                        # 场景层物理部分 (0.3)
                        "collision_rate": 0.08,
                        "oob_rate": 0.07,
                        "penetration_depth": 0.05,
                        "oob_volume": 0.05,
                        "support": 0.05,
                        "room_area": 0.0,
                        "tool_execution": 0.0,
                        # VLM场景层 (0.3)
                        "vlm_problem_identification": 0.0,
                        "vlm_action_reasonableness": 0.0,
                        "vlm_scene_improvement": 0.0,
                        "vlm_rationality": 0.10,  # 场景层VLM
                        "vlm_aesthetics": 0.0,  # 已删除
                        "vlm_requirement_match": 0.10,  # 场景层VLM
                        "vlm_scene_graph": 0.10  # 场景层VLM
                    }
            else:
                # 中间轮次：评估格式 + 场景改进度 + 碰撞率/出界率 + 体积奖励 + 关键物体
                # 权重总计: 0.10 + 0.1*4 + 0.25*2 = 0.10 + 0.40 + 0.50 = 1.0
                weights = {
                    "physics": 0.0,  # 不评估
                    "format": 0.10,  # 格式正确性
                    "object_count": 0.0,  # 不评估
                    "collision_rate": 0.1,  # 碰撞率（中间轮次启用）
                    "oob_rate": 0.1,  # 出界率（中间轮次启用）
                    "penetration_depth": 0.1,  # 穿透深度体积奖励（中间轮次启用）
                    "oob_volume": 0.1,  # 出界体积奖励（中间轮次启用）
                    "support": 0.0,
                    "room_area": 0.0,
                    "tool_execution": 0.0,
                    # VLM judge 中间评分
                    "vlm_problem_identification": 0.0,  # 已废弃
                    "vlm_action_reasonableness": 0.0,  # 已废弃
                    "vlm_scene_improvement": 0.25,  # 场景改进度评估（VLM评估的一半）
                    "vlm_key_objects": 0.25,  # 关键物体评估（VLM评估的一半，占总权重0.25）
                    "vlm_rationality": 0.0,
                    "vlm_aesthetics": 0.0,
                    "vlm_requirement_match": 0.0,
                    "vlm_scene_graph": 0.0  # 场景图约束评估（中间轮次不使用）
                }
        
        rewards = {}
        
        # 1. 计算物理有效性奖励（仅在最后一轮且voxel模式下计算）
        
        # 先统计物体数量，用于物理评估的前置检查
        total_objects_for_physics = 0
        if 'groups' in current_scene and current_scene['groups']:
            for group in current_scene['groups']:
                if 'objects' in group and group['objects']:
                    total_objects_for_physics += len(group['objects'])
        elif 'objects' in current_scene and current_scene['objects']:
            total_objects_for_physics = len(current_scene['objects'])
        
        if self.physics_mode == "voxel" and self.voxel_reward is not None and is_final_turn:
            # 物体少于2个时直接返回-1（没有评估的必要）
            if total_objects_for_physics < 2:
                self.logger.warning(f"Object count ({total_objects_for_physics}) < 2, skipping voxel physics evaluation")
                rewards["physics"] = -1.0
            else:
                # 只在最后一轮计算体素物理评估
                try:  
                    # 使用VoxelReward计算
                    physics_reward, metrics = await asyncio.to_thread(  
                        self.voxel_reward.compute_reward,  
                        current_scene,  
                        format_type='ours'  
                    )  
                    
                    if isinstance(physics_reward, (list, tuple)):  
                        physics_reward = float(physics_reward[0]) if physics_reward else 0.0  
                    physics_reward = float(physics_reward)
                    rewards["physics"] = physics_reward
                    
                    # 保存物理评估结果  
                    metrics_path = instance["output_dir"] / f"turn_{turn:03d}_voxel_metrics.json"  
                    with open(metrics_path, 'w', encoding='utf-8') as f:  
                        json.dump(metrics, f, indent=2)  
                        
                except Exception as e:  
                    if self.verbose:  
                        print(f"Warning: VoxelReward calculation failed: {e}")
                    self.logger.warning(f"VoxelReward calculation failed: {e}")
                    rewards["physics"] = -1.0
        else:
            rewards["physics"] = 0.0
        
        # 2. 计算格式奖励
        try:
            format_reward = self._calculate_format_reward(messages, turn, instance_id)
            rewards["format"] = format_reward
        except Exception as e:
            if self.verbose:
                print(f"Warning: Format reward calculation failed: {e}")
            rewards["format"] = -1.0
        
        # 3. 计算物体数量奖励（仅在最后一轮计算）
        # 标记是否需要覆盖总奖励为-1（当最后一轮物体数量<5时）
        should_override_total_reward = False
        
        if is_final_turn:
            try:
                object_count_reward, should_override_total_reward = self._calculate_object_count_reward(current_scene, is_final_turn=True)
                rewards["object_count"] = object_count_reward
                if should_override_total_reward:
                    self.logger.warning(f"Object count < 5 in final turn, will override total reward to -1")
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Object count reward calculation failed: {e}")
                rewards["object_count"] = 0.0
        else:
            # 中间轮次不计算物体数量
            rewards["object_count"] = 0.0
        
        # 4. 计算碰撞率、出界率和体积奖励（中间轮次和最后一轮在trimesh模式下计算）
        # 体积奖励阈值设计：
        # - 穿透深度: 0m=+1.0, 0.05m(5cm)=0.0, 0.2m(20cm)=-1.0
        # - 出界体积: 0m³=+1.0, 0.1m³=0.0, 0.5m³=-1.0
        if self.physics_mode == "trimesh" and self.trimesh_metrics is not None and turn > 1:
            # 物体少于3个时直接返回-1（没有评估的必要）
            if total_objects_for_physics < 3:
                self.logger.warning(f"Object count ({total_objects_for_physics}) < 3, skipping trimesh physics evaluation")
                rewards["collision_rate"] = -1.0
                rewards["oob_rate"] = -1.0
                rewards["penetration_depth"] = -1.0
                rewards["oob_volume"] = -1.0
                instance["_trimesh_metrics_for_feedback"] = None  # 确保设置为None
            else:
                try:
                    trimesh_reward, trimesh_metrics = await asyncio.to_thread(
                        self.trimesh_metrics.compute_reward,
                        current_scene,
                        format_type='ours'
                    )
                    
                    # 提取各项指标
                    collision_rate = trimesh_metrics['collision_rate']
                    oob_rate = trimesh_metrics['out_of_bounds_rate']
                    total_penetration_depth = trimesh_metrics.get('total_penetration_depth', 0.0)
                    total_oob_volume = trimesh_metrics.get('total_oob_volume', 0.0)
                    
                    # ===== 碰撞率奖励（基于SFT基线45%调整） =====
                    # 阈值设计：
                    # - 碰撞率 ≤ 20%: +1.0 到 +0.5（优秀）
                    # - 碰撞率 20%-45%: +0.5 到 0.0（SFT基线为零点）
                    # - 碰撞率 > 45%: 0.0 到 -1.0（比SFT差）
                    if collision_rate <= 20:
                        # 优秀区间：0% -> +1.0, 20% -> +0.5
                        collision_reward = 1.0 - 0.5 * (collision_rate / 20.0)
                    elif collision_rate <= 45:
                        # 良好区间（SFT基线为零点）：20% -> +0.5, 45% -> 0.0
                        collision_reward = 0.5 - 0.5 * (collision_rate - 20) / 25.0
                    else:
                        # 差于SFT：45% -> 0.0, 100% -> -1.0
                        collision_reward = -1.0 * (collision_rate - 45) / 55.0
                    
                    # ===== 出界率奖励（基于SFT基线30%调整） =====
                    # 阈值设计：
                    # - 出界率 ≤ 10%: +1.0 到 +0.5（优秀）
                    # - 出界率 10%-30%: +0.5 到 0.0（SFT基线为零点）
                    # - 出界率 > 30%: 0.0 到 -1.0（比SFT差）
                    if oob_rate <= 10:
                        # 优秀区间：0% -> +1.0, 10% -> +0.5
                        oob_reward = 1.0 - 0.5 * (oob_rate / 10.0)
                    elif oob_rate <= 30:
                        # 良好区间（SFT基线为零点）：10% -> +0.5, 30% -> 0.0
                        oob_reward = 0.5 - 0.5 * (oob_rate - 10) / 20.0
                    else:
                        # 差于SFT：30% -> 0.0, 100% -> -1.0
                        oob_reward = -1.0 * (oob_rate - 30) / 70.0
                    
                    # ===== 穿透深度体积奖励（归一化） =====
                    # 阈值设计（考虑累计值，多个碰撞对的总穿透深度）：
                    # 0m=+1.0, 0.1m=+0.5, 0.3m=0.0, 0.6m=-0.5, 1.0m=-1.0
                    # 场景参考：5个碰撞对各2cm=0.1m，各6cm=0.3m
                    if total_penetration_depth == 0:
                        penetration_reward = 1.0
                    elif total_penetration_depth <= 0.1:  # 10cm以内（轻微碰撞）
                        # 线性从1.0降到0.5
                        penetration_reward = 1.0 - 0.5 * (total_penetration_depth / 0.1)
                    elif total_penetration_depth <= 0.3:  # 10cm到30cm（中等碰撞）
                        # 线性从0.5降到0.0
                        penetration_reward = 0.5 - 0.5 * (total_penetration_depth - 0.1) / 0.2
                    elif total_penetration_depth <= 0.6:  # 30cm到60cm（较严重碰撞）
                        # 线性从0.0降到-0.5
                        penetration_reward = -0.5 * (total_penetration_depth - 0.3) / 0.3
                    elif total_penetration_depth <= 1.0:  # 60cm到1m（严重碰撞）
                        # 线性从-0.5降到-1.0
                        penetration_reward = -0.5 - 0.5 * (total_penetration_depth - 0.6) / 0.4
                    else:  # 超过1m
                        penetration_reward = -1.0
                    
                    # ===== 出界体积奖励（归一化） =====
                    # 阈值设计（考虑累计值，多个物体的总出界体积）：
                    # 0m³=+1.0, 0.2m³=+0.5, 0.5m³=0.0, 1.0m³=-0.5, 2.0m³=-1.0
                    # 场景参考：一把椅子体积约0.1m³，一个沙发约0.5m³
                    if total_oob_volume == 0:
                        oob_volume_reward = 1.0
                    elif total_oob_volume <= 0.2:  # 0.2立方米以内（轻微出界）
                        # 线性从1.0降到0.5
                        oob_volume_reward = 1.0 - 0.5 * (total_oob_volume / 0.2)
                    elif total_oob_volume <= 0.5:  # 0.2到0.5立方米（中等出界）
                        # 线性从0.5降到0.0
                        oob_volume_reward = 0.5 - 0.5 * (total_oob_volume - 0.2) / 0.3
                    elif total_oob_volume <= 1.0:  # 0.5到1立方米（较严重出界）
                        # 线性从0.0降到-0.5
                        oob_volume_reward = -0.5 * (total_oob_volume - 0.5) / 0.5
                    elif total_oob_volume <= 2.0:  # 1到2立方米（严重出界）
                        # 线性从-0.5降到-1.0
                        oob_volume_reward = -0.5 - 0.5 * (total_oob_volume - 1.0) / 1.0
                    else:  # 超过2立方米
                        oob_volume_reward = -1.0
                    
                    rewards["collision_rate"] = collision_reward
                    rewards["oob_rate"] = oob_reward
                    rewards["penetration_depth"] = penetration_reward
                    rewards["oob_volume"] = oob_volume_reward
                    
                    # 保存trimesh物理评估结果
                    trimesh_metrics_path = instance["output_dir"] / f"turn_{turn:03d}_trimesh_metrics.json"
                    with open(trimesh_metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(trimesh_metrics, f, indent=2)
                    
                    # 存储trimesh_metrics供后续反馈生成使用
                    instance["_trimesh_metrics_for_feedback"] = trimesh_metrics
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Trimesh metrics calculation failed: {e}")
                    self.logger.warning(f"Trimesh metrics calculation failed: {e}")
                    rewards["collision_rate"] = -1.0
                    rewards["oob_rate"] = -1.0
                    rewards["penetration_depth"] = -1.0
                    rewards["oob_volume"] = -1.0
                    instance["_trimesh_metrics_for_feedback"] = None
        else:
            rewards["collision_rate"] = 0.0
            rewards["oob_rate"] = 0.0
            rewards["penetration_depth"] = 0.0
            rewards["oob_volume"] = 0.0
            instance["_trimesh_metrics_for_feedback"] = None
        
        # 4.5. 计算支撑奖励（仅在最后一轮且trimesh模式下计算）
        if is_final_turn and self.physics_mode == "trimesh":
            try:
                support_reward = await self._calculate_support_reward(current_scene)
                rewards["support"] = support_reward
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Support reward calculation failed: {e}", file=sys.stderr, flush=True)
                self.logger.warning(f"Support reward calculation failed: {e}")
                rewards["support"] = 0.0
        else:
            rewards["support"] = 0.0
        
        # 5. 房间形状奖励已合并到格式奖励中（第一轮的format reward包含了形状奖励）
        rewards["room_shape"] = 0.0
        
        # 房间面积奖励已禁用
        rewards["room_area"] = 0.0
        
        # 6. 计算工具执行成功奖励（仅第二轮及以后）
        if turn > 1:
            if tool_execution_success is True:
                rewards["tool_execution"] = 1.0
            elif tool_execution_success is False:
                rewards["tool_execution"] = -1.0
            else:
                # 如果没有提供状态，默认为成功
                rewards["tool_execution"] = 1.0
        else:
            # 第一轮没有工具执行
            rewards["tool_execution"] = 0.0
        
        # 8. VLM Judge 评分
        # 初始化所有VLM评分为0
        rewards["vlm_problem_identification"] = 0.0
        rewards["vlm_action_reasonableness"] = 0.0
        rewards["vlm_scene_improvement"] = 0.0  # 中间轮次的场景改进度评估
        rewards["vlm_key_objects"] = 0.0  # 物体层：关键物体匹配度
        rewards["vlm_size_proportion"] = 0.0  # 物体层：尺寸比例合理性
        rewards["vlm_rationality"] = 0.0
        rewards["vlm_aesthetics"] = 0.0  # 已删除，保持为0
        rewards["vlm_requirement_match"] = 0.0
        rewards["vlm_scene_graph"] = 0.0  # 场景图约束评估
        
        # 熔断标志：物体层不合格时，场景层直接设为-1
        object_level_fused = False
        
        # VLM Judge评估
        if turn > 1 and self.vlm_judge_enabled and vlm_image_path and Path(vlm_image_path).exists():
            user_requirement = instance.get("user_requirement", "")
            
            if is_final_turn:
                # ========== 最后一轮：分层评估 ==========
                self.logger.info("Final turn: Layered evaluation (Object Level -> Scene Level)")
                self.logger.info(f"  Image: {vlm_image_path}")
                self.logger.info(f"  Scene: edited scene with {len(vlm_scene.get('groups', []))} groups")
                
                # ----- 物体层评估 (0.3) -----
                self.logger.info("=== Object Level Evaluation ===")
                
                # 1. 关键物体匹配度评估
                try:
                    current_scene_summary = self._extract_objects_summary(current_scene)
                    room_type = current_scene.get('room_type', '')
                    key_objects_result = await self._vlm_extract_and_evaluate_key_objects(
                        image_path=vlm_image_path,
                        scene_summary=current_scene_summary,
                        user_requirement=user_requirement,
                        room_type=room_type,
                        instance_id=instance_id
                    )
                    rewards["vlm_key_objects"] = float(key_objects_result.get("score", 0))
                    self.logger.info(f"Key objects score: {rewards['vlm_key_objects']}")
                except Exception as e:
                    self.logger.warning(f"Key objects evaluation failed: {e}")
                    rewards["vlm_key_objects"] = 0.0
                
                # 2. 尺寸比例合理性评估
                try:
                    size_proportion_result = await self._vlm_evaluate_object_size_proportion(
                        image_path=vlm_image_path,
                        scene_data=current_scene,
                        user_requirement=user_requirement
                    )
                    rewards["vlm_size_proportion"] = float(size_proportion_result.get("score", 0))
                    self.logger.info(f"Size proportion score: {rewards['vlm_size_proportion']}, issues: {size_proportion_result.get('issues', [])}")
                except Exception as e:
                    self.logger.warning(f"Size proportion evaluation failed: {e}")
                    rewards["vlm_size_proportion"] = 0.0
                
                # 3. 物体数量评估（已在前面计算）
                # rewards["object_count"] 已计算
                
                # 如果物体数量过少（<4），强制设置关键物体得分为-1，确保物体层熔断
                if should_override_total_reward:
                    rewards["vlm_key_objects"] = -1.0
                    self.logger.warning("Object count < 4, forcing vlm_key_objects to -1.0")
                
                # ----- 物体层熔断判断 -----
                # 物体层加权得分 = key_objects * 0.20 + size_proportion * 0.10 + object_count * 0.0
                object_level_score = (
                    rewards["vlm_key_objects"] * weights.get("vlm_key_objects", 0.20) +
                    rewards["vlm_size_proportion"] * weights.get("vlm_size_proportion", 0.10) +
                    rewards["object_count"] * weights.get("object_count", 0.0)
                )
                self.logger.info(f"Object level weighted score: {object_level_score:.4f}")
                
                if object_level_score < 0:
                    # 物体层不合格，熔断！场景层直接设为-1
                    object_level_fused = True
                    self.logger.warning(f"Object level FUSED: score={object_level_score:.4f} < 0, scene level will be set to -1")
                    rewards["vlm_rationality"] = -1.0
                    rewards["vlm_requirement_match"] = -1.0
                    rewards["vlm_scene_graph"] = -1.0
                    # 物理指标也设为-1
                    if self.physics_mode == "trimesh":
                        rewards["collision_rate"] = -1.0
                        rewards["oob_rate"] = -1.0
                        rewards["penetration_depth"] = -1.0
                        rewards["oob_volume"] = -1.0
                        rewards["support"] = -1.0
                    elif self.physics_mode == "voxel":
                        rewards["physics"] = -1.0
                else:
                    # ----- 场景层评估 (0.6) -----
                    self.logger.info("=== Scene Level Evaluation ===")
                    
                    # 场景层VLM评估（已合并为一次调用）
                    try:
                        vlm_scores = await self._judge_final_turn(
                            image_path=vlm_image_path,
                            user_requirement=user_requirement,
                            current_scene=vlm_scene
                        )
                        rewards["vlm_rationality"] = float(vlm_scores.get("rationality", 0))
                        rewards["vlm_requirement_match"] = float(vlm_scores.get("requirement_match", 0))
                        rewards["vlm_scene_graph"] = float(vlm_scores.get("scene_graph", 0))
                        
                        # 保存VLM评分结果
                        vlm_scores_path = instance["output_dir"] / f"turn_{turn:03d}_vlm_final_scores.json"
                        vlm_scores["key_objects_score"] = rewards["vlm_key_objects"]
                        vlm_scores["size_proportion_score"] = rewards["vlm_size_proportion"]
                        vlm_scores["object_level_fused"] = object_level_fused
                        with open(vlm_scores_path, 'w', encoding='utf-8') as f:
                            json.dump(vlm_scores, f, indent=2)
                        
                    except Exception as e:
                        self.logger.error(f"VLM judge final evaluation failed: {e}", exc_info=True)
                        if self.verbose:
                            print(f"Warning: VLM judge final evaluation failed: {e}")
            else:
                # ========== 中间轮次：场景改进度 + 物体相关性评估 ==========
                self.logger.info("Intermediate turn: calling VLM judge for scene improvement comparison")
                self.logger.info(f"  Previous image: {vlm_prev_image_path}")
                self.logger.info(f"  Current image: {vlm_image_path}")
                
                # 检查两张图像是否都存在
                if vlm_prev_image_path and Path(vlm_prev_image_path).exists():
                    try:
                        # 提取上一轮和当前轮的物体摘要供VLM参考
                        prev_scene_summary = ""
                        current_scene_summary = ""
                        
                        # scene_before_edit 是编辑前的场景（上一轮状态）
                        if scene_before_edit:
                            try:
                                prev_scene_summary = self._extract_objects_summary(scene_before_edit)
                            except Exception as e:
                                self.logger.warning(f"Failed to extract prev scene summary: {e}")
                        
                        # current_scene 是编辑后的场景（当前轮状态）
                        if current_scene:
                            try:
                                current_scene_summary = self._extract_objects_summary(current_scene)
                            except Exception as e:
                                self.logger.warning(f"Failed to extract current scene summary: {e}")
                        
                        vlm_scores = await self._judge_intermediate_turn(
                            prev_image_path=vlm_prev_image_path,  # 上一轮的图像
                            current_image_path=vlm_image_path,  # 本轮的图像
                            user_requirement=user_requirement,
                            prev_scene_summary=prev_scene_summary,  # 上一轮物体摘要
                            current_scene_summary=current_scene_summary  # 当前轮物体摘要
                        )
                        rewards["vlm_scene_improvement"] = float(vlm_scores.get("scene_improvement", 0))
                        
                        # ===== 物体相关性评估（根据工具调用类型选择策略） =====
                        # 提取本轮工具调用
                        current_tool_calls = self._extract_tool_calls_from_messages(messages)
                        
                        # 检查是否有add_object或replace_object操作
                        has_add_or_replace = any(
                            tc.get("name") in ["add_object", "replace_object"]
                            for tc in current_tool_calls
                        )
                        
                        if has_add_or_replace:
                            # 有add/replace操作：评估新增物体是否与场景需求相关
                            self.logger.info("Intermediate turn: Has add/replace operations, evaluating new objects relevance")
                            try:
                                room_type = current_scene.get('room_type', '')
                                relevance_result = await self._vlm_evaluate_new_objects_relevance(
                                    tool_calls=current_tool_calls,
                                    user_requirement=user_requirement,
                                    room_type=room_type
                                )
                                rewards["vlm_key_objects"] = float(relevance_result.get("score", 0))
                                
                                vlm_scores["new_objects_relevance"] = relevance_result
                                vlm_scores["key_objects_score"] = rewards["vlm_key_objects"]
                                
                                self.logger.info(f"New objects relevance: score={rewards['vlm_key_objects']}, "
                                               f"relevant={relevance_result.get('relevant_objects', [])}, "
                                               f"irrelevant={relevance_result.get('irrelevant_objects', [])}")
                            except Exception as e:
                                self.logger.warning(f"New objects relevance evaluation failed: {e}")
                                rewards["vlm_key_objects"] = 0.0
                        else:
                            # 无add/replace操作：评估当前场景是否包含关键物体
                            self.logger.info("Intermediate turn: No add/replace operations, evaluating key objects presence")
                            try:
                                room_type = current_scene.get('room_type', '')
                                key_objects_result = await self._vlm_extract_and_evaluate_key_objects(
                                    image_path=vlm_image_path,
                                    scene_summary=current_scene_summary,
                                    user_requirement=user_requirement,
                                    room_type=room_type,
                                    instance_id=instance_id
                                )
                                rewards["vlm_key_objects"] = float(key_objects_result.get("score", 0))
                                
                                vlm_scores["key_objects"] = key_objects_result.get("key_objects", [])
                                vlm_scores["most_critical_object"] = key_objects_result.get("most_critical_object", "")
                                vlm_scores["found_objects"] = key_objects_result.get("found_objects", [])
                                vlm_scores["missing_objects"] = key_objects_result.get("missing_objects", [])
                                vlm_scores["key_objects_score"] = rewards["vlm_key_objects"]
                                
                                self.logger.info(f"Key objects evaluation: score={rewards['vlm_key_objects']}, "
                                               f"found={key_objects_result.get('found_objects', [])}, "
                                               f"missing={key_objects_result.get('missing_objects', [])}")
                            except Exception as e:
                                self.logger.warning(f"Key objects evaluation failed: {e}")
                                rewards["vlm_key_objects"] = 0.0
                        
                        # 保存VLM评分结果
                        vlm_scores_path = instance["output_dir"] / f"turn_{turn:03d}_vlm_intermediate_scores.json"
                        with open(vlm_scores_path, 'w', encoding='utf-8') as f:
                            json.dump(vlm_scores, f, indent=2)
                        
                    except Exception as e:
                        self.logger.error(f"VLM judge intermediate evaluation failed: {e}", exc_info=True)
                        if self.verbose:
                            print(f"Warning: VLM judge intermediate evaluation failed: {e}")
                else:
                    self.logger.warning(f"Previous image not found: {vlm_prev_image_path}")
        elif turn > 1 and self.vlm_judge_enabled:
            self.logger.warning(f"VLM judge skipped: vlm_image_path={vlm_image_path}, exists={Path(vlm_image_path).exists() if vlm_image_path else False}")
        
        # 计算加权总奖励
        total_reward = (
            weights["physics"] * rewards["physics"] +
            weights["format"] * rewards["format"] +
            weights["object_count"] * rewards["object_count"] +
            weights["collision_rate"] * rewards["collision_rate"] +
            weights["oob_rate"] * rewards["oob_rate"] +
            weights["penetration_depth"] * rewards["penetration_depth"] +
            weights["oob_volume"] * rewards["oob_volume"] +
            weights["support"] * rewards["support"] +
            weights["room_area"] * rewards["room_area"] +
            weights["tool_execution"] * rewards["tool_execution"] +
            weights["vlm_problem_identification"] * rewards["vlm_problem_identification"] +
            weights["vlm_action_reasonableness"] * rewards["vlm_action_reasonableness"] +
            weights.get("vlm_scene_improvement", 0.0) * rewards["vlm_scene_improvement"] +  # 场景改进度
            weights.get("vlm_key_objects", 0.0) * rewards["vlm_key_objects"] +  # 物体层：关键物体
            weights.get("vlm_size_proportion", 0.0) * rewards["vlm_size_proportion"] +  # 物体层：尺寸比例
            weights["vlm_rationality"] * rewards["vlm_rationality"] +
            weights["vlm_aesthetics"] * rewards["vlm_aesthetics"] +  # 已废弃，保持为0
            weights["vlm_requirement_match"] * rewards["vlm_requirement_match"] +
            weights["vlm_scene_graph"] * rewards["vlm_scene_graph"]  # 场景图约束评估
        )
        
        # 特殊规则：如果最后一轮物体数量<4，直接覆盖总奖励为-1
        if should_override_total_reward:
            self.logger.warning(f"Overriding total reward from {total_reward:.4f} to -1.0 due to object count < 4")
            print(f"WARNING: Object count < 4 in final turn, overriding total reward to -1.0", file=sys.stderr, flush=True)
            total_reward = -1.0
        
        # 调试输出
        print(f"DEBUG [Interaction Turn {turn}] (physics_mode={self.physics_mode}, is_final={is_final_turn}, fused={object_level_fused}): Reward breakdown:",   
            file=sys.stderr, flush=True)
        
        # 显示非零权重的评估指标
        if turn == 1:
            print(f"  Format: {rewards['format']:.4f} (weight: {weights['format']})",
                file=sys.stderr, flush=True)
            print(f"  Room Area: {rewards['room_area']:.4f} (weight: {weights['room_area']})",
                file=sys.stderr, flush=True)
        elif is_final_turn:
            # 最后一轮：分层显示
            print(f"  === Format Layer (0.1) ===", file=sys.stderr, flush=True)
            print(f"  Format: {rewards['format']:.4f} (weight: {weights['format']})",
                file=sys.stderr, flush=True)
            
            print(f"  === Object Layer (0.3) ===", file=sys.stderr, flush=True)
            print(f"  Object Count: {rewards['object_count']:.4f} (weight: {weights['object_count']})",
                file=sys.stderr, flush=True)
            print(f"  VLM Key Objects: {rewards['vlm_key_objects']:.4f} (weight: {weights.get('vlm_key_objects', 0.0)})",
                file=sys.stderr, flush=True)
            print(f"  VLM Size Proportion: {rewards['vlm_size_proportion']:.4f} (weight: {weights.get('vlm_size_proportion', 0.0)})",
                file=sys.stderr, flush=True)
            if object_level_fused:
                print(f"  ** OBJECT LEVEL FUSED - Scene level set to -1 **", file=sys.stderr, flush=True)
            
            print(f"  === Scene Layer (0.6) ===", file=sys.stderr, flush=True)
            if self.physics_mode == "voxel":
                print(f"  Physics (Voxel): {rewards['physics']:.4f} (weight: {weights['physics']})",
                    file=sys.stderr, flush=True)
            elif self.physics_mode == "trimesh":
                print(f"  Collision Rate: {rewards['collision_rate']:.4f} (weight: {weights['collision_rate']})",
                    file=sys.stderr, flush=True)
                print(f"  OOB Rate: {rewards['oob_rate']:.4f} (weight: {weights['oob_rate']})",
                    file=sys.stderr, flush=True)
                print(f"  Penetration Depth: {rewards['penetration_depth']:.4f} (weight: {weights['penetration_depth']})",
                    file=sys.stderr, flush=True)
                print(f"  OOB Volume: {rewards['oob_volume']:.4f} (weight: {weights['oob_volume']})",
                    file=sys.stderr, flush=True)
                print(f"  Support: {rewards['support']:.4f} (weight: {weights['support']})",
                    file=sys.stderr, flush=True)
            
            # VLM Scene Layer
            print(f"  === VLM Scene Layer ===", file=sys.stderr, flush=True)
            print(f"  VLM Rationality: {rewards['vlm_rationality']:.4f} (weight: {weights['vlm_rationality']})",
                file=sys.stderr, flush=True)
            print(f"  VLM Requirement Match: {rewards['vlm_requirement_match']:.4f} (weight: {weights['vlm_requirement_match']})",
                file=sys.stderr, flush=True)
            print(f"  VLM Scene Graph: {rewards['vlm_scene_graph']:.4f} (weight: {weights['vlm_scene_graph']})",
                file=sys.stderr, flush=True)
        else:
            # 中间轮次：显示格式、碰撞率、出界率、体积奖励和场景改进度评分
            print(f"  Format: {rewards['format']:.4f} (weight: {weights['format']})",
                file=sys.stderr, flush=True)
            if self.physics_mode == "trimesh":
                print(f"  Collision Rate: {rewards['collision_rate']:.4f} (weight: {weights['collision_rate']})",
                    file=sys.stderr, flush=True)
                print(f"  OOB Rate: {rewards['oob_rate']:.4f} (weight: {weights['oob_rate']})",
                    file=sys.stderr, flush=True)
                print(f"  Penetration Depth: {rewards['penetration_depth']:.4f} (weight: {weights['penetration_depth']})",
                    file=sys.stderr, flush=True)
                print(f"  OOB Volume: {rewards['oob_volume']:.4f} (weight: {weights['oob_volume']})",
                    file=sys.stderr, flush=True)
            print(f"  VLM Scene Improvement: {rewards['vlm_scene_improvement']:.4f} (weight: {weights.get('vlm_scene_improvement', 0.0)})",
                file=sys.stderr, flush=True)
            print(f"  VLM Key Objects: {rewards['vlm_key_objects']:.4f} (weight: {weights.get('vlm_key_objects', 0.0)})",
                file=sys.stderr, flush=True)
        
        print(f"  Total: {total_reward:.4f}",
            file=sys.stderr, flush=True)
        
        # 存储详细奖励组件到实例状态  
        reward_component = {
            f"turn_{turn}_format_reward": rewards["format"],
            f"turn_{turn}_object_count_reward": rewards["object_count"],
            f"turn_{turn}_total_reward": total_reward
        }
        
        if turn == 1:
            reward_component[f"turn_{turn}_room_area_reward"] = rewards["room_area"]
        
        if turn > 1:
            if self.physics_mode == "voxel":
                reward_component[f"turn_{turn}_physics_reward"] = rewards["physics"]
            elif self.physics_mode == "trimesh":
                reward_component[f"turn_{turn}_collision_rate_reward"] = rewards["collision_rate"]
                reward_component[f"turn_{turn}_oob_rate_reward"] = rewards["oob_rate"]
                reward_component[f"turn_{turn}_penetration_depth_reward"] = rewards["penetration_depth"]
                reward_component[f"turn_{turn}_oob_volume_reward"] = rewards["oob_volume"]
                reward_component[f"turn_{turn}_support_reward"] = rewards["support"]
            reward_component[f"turn_{turn}_tool_execution_reward"] = rewards["tool_execution"]
            
            # VLM judge 奖励组件
            
            if is_final_turn:
                reward_component[f"turn_{turn}_vlm_key_objects"] = rewards["vlm_key_objects"]  # Object Layer: 关键物体
                reward_component[f"turn_{turn}_vlm_size_proportion"] = rewards["vlm_size_proportion"]  # Object Layer: 物体尺寸
                reward_component[f"turn_{turn}_vlm_rationality"] = rewards["vlm_rationality"]  # Scene Layer: 合理性
                reward_component[f"turn_{turn}_vlm_requirement_match"] = rewards["vlm_requirement_match"]  # Scene Layer: 需求匹配
                reward_component[f"turn_{turn}_vlm_scene_graph"] = rewards["vlm_scene_graph"]  # Scene Layer: 场景图约束
            else:
                reward_component[f"turn_{turn}_vlm_scene_improvement"] = rewards["vlm_scene_improvement"]  # 场景改进度评估
                reward_component[f"turn_{turn}_vlm_key_objects"] = rewards["vlm_key_objects"]  # 关键物体评估
        
        instance.setdefault("reward_components", []).append(reward_component)
        
        # ===== 生成反馈用于下一轮 =====
        # 只在中间轮次生成反馈（最后一轮不需要反馈）
        # 可通过配置开关控制是否启用反馈注入
        if turn > 1 and not is_final_turn and self.feedback_injection_enabled:
            feedback_parts = []
            
            # 1. 物理反馈（来自trimesh）- 受 physics_feedback_enabled 控制
            if self.physics_feedback_enabled:
                trimesh_metrics_for_feedback = instance.get("_trimesh_metrics_for_feedback")
                if trimesh_metrics_for_feedback:
                    physics_feedback = generate_physics_feedback(trimesh_metrics_for_feedback, top_k=3)
                    if physics_feedback:
                        feedback_parts.append(physics_feedback)
                    self.logger.info(f"Physics feedback: {physics_feedback}")
            else:
                self.logger.info("Physics feedback disabled by config")
            
            # 2. VLM布局反馈（只在有图像时生成）- 受 layout_feedback_enabled 控制
            if self.layout_feedback_enabled and vlm_image_path and Path(vlm_image_path).exists():
                user_requirement = instance.get("user_requirement", "")
                try:
                    layout_feedback = await self._vlm_generate_layout_feedback(
                        image_path=vlm_image_path,
                        user_requirement=user_requirement
                    )
                    if layout_feedback:
                        feedback_parts.append(layout_feedback)
                    self.logger.info(f"Layout feedback: {layout_feedback}")
                except Exception as e:
                    self.logger.warning(f"VLM layout feedback generation failed: {e}")
            elif not self.layout_feedback_enabled:
                self.logger.info("Layout feedback disabled by config")
            
            # 组合反馈
            if feedback_parts:
                combined_feedback = " ".join(feedback_parts)
                instance["last_feedback"] = combined_feedback
                self.logger.info(f"Combined feedback for next turn: {combined_feedback}")
            else:
                instance["last_feedback"] = ""
        else:
            if not self.feedback_injection_enabled:
                self.logger.info("Feedback injection disabled by config")
            instance["last_feedback"] = ""
        
        return total_reward
    
    async def finalize_interaction(
        self,
        instance_id: str,
        **kwargs
    ) -> None:
        """结束交互实例，清理资源，并返回奖励信息"""
        
        self.logger.info("="*80)
        self.logger.info(f"finalize_interaction called for instance {instance_id}")
        
        if instance_id not in self._instance_dict:
            self.logger.warning(f"Instance {instance_id} not found in finalize")
            if self.verbose:
                print(f"Warning: Instance {instance_id} not found in finalize")
            return {"reward_scores": {}}
        
        instance = self._instance_dict[instance_id]
        
        # 保存交互摘要
        summary = {
            "instance_id": instance_id,
            "total_turns": instance["turn_count"],
            "max_turns": instance["max_turns"],
            "total_reward": instance["total_reward"],
            "average_reward": instance["total_reward"] / max(instance["turn_count"], 1),
            "history": instance["history"]
        }
        
        summary_path = instance["output_dir"] / "interaction_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved interaction summary to: {summary_path}")
        self.logger.info(f"Total turns: {instance['turn_count']}")
        self.logger.info(f"Total reward: {instance['total_reward']:.4f}")
        self.logger.info(f"Average reward: {summary['average_reward']:.4f}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Finalized interaction: {instance_id}")
            print(f"  Total turns: {instance['turn_count']}")
            print(f"  Total reward: {instance['total_reward']:.4f}")
            print(f"  Average reward: {summary['average_reward']:.4f}")
            print(f"  Summary saved to: {summary_path}")
            print(f"{'='*60}")
        
        # 合并所有 turn 的奖励组件到一个字典
        reward_scores = {}
        for components in instance.get("reward_components", []):
            reward_scores.update(components)
        final_scene = instance["current_scene"]
        
        self.logger.info(f"Reward scores: {json.dumps(reward_scores, indent=2)}")
        self.logger.info(f"Final scene has {len(final_scene.get('objects', []))} objects")
        
        # 清理实例
        # 删除实例输出目录及其内容

        del self._instance_dict[instance_id]
        
        self.logger.info(f"Cleaned up instance {instance_id}")
        self.logger.info("finalize_interaction completed")
        self.logger.info("="*80)
        