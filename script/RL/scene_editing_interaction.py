#!/usr/bin/env python3
"""
SceneEditingInteraction - Scene Editing Interaction Class
Used for scene editing reinforcement learning tasks in the VERL framework
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

# Class-level semaphore (lazy initialization)
# Not created at module level to avoid event loop binding issues
_RENDER_SEMAPHORE = None

# Add path where RL_utils is located
# More reliable path finding: navigate up from current file to project root
current_file = Path(__file__).resolve()
# Assumed project structure: llmscene/verl/verl/interactions/scene_editing_interaction.py
# Go up 4 levels to llmscene, then find utils
project_root = current_file.parent.parent.parent.parent
utils_path = project_root / "utils"

if not utils_path.exists():
    raise RuntimeError(f"Utils path not found: {utils_path}")

if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))  # Use insert(0) to prioritize this path

from verl.interactions.base import BaseInteraction
from RL_utils import edit_and_render_scene, VoxelReward, TrimeshPhysicsMetrics, convert_flat_to_grouped, convert_grouped_to_flat, generate_physics_feedback

# Import Objaverse asset metadata retrieval functions
try:
    from objaverse_retriever import get_bbox_dims as objaverse_get_bbox_dims, ObjaverseRetriever
except ImportError:
    objaverse_get_bbox_dims = None
    ObjaverseRetriever = None

# Define essential base objects for each room type (if these objects are missing, score is directly -1)
# Note: keys use standardized names, matching_patterns are used to match from user input
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
        "essential": ["treadmill", "weight_equipment"],  # Large gym equipment is essential; dumbbells are too small to count
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
    # Entertainment rooms vary widely (music room, board game room, billiard room, video game room, etc.), cannot define uniform essential objects
    # Delegate to VLM to judge dynamically based on user requirements
    "entertainment room": {
        "essential": [],  # No fixed essential objects, VLM judges dynamically
        "matching_patterns": [
            # General entertainment room
            "entertainment room", "entertainmentroom", "entertainment", 
            "game room", "gameroom", "gaming room",
            "recreation room", "rec room", "play room", "playroom",
            # Home theater
            "home theater", "home theatre", "home cinema", "theater room", "theatre room",
            "cinema room", "movie room", "media room",
            # Music related
            "music room", "musicroom", "ktv", "karaoke room", "karaoke",
            "piano room", "studio",
            # Ball sports
            "billiard room", "billiards room", "pool room", "snooker room",
            "ping pong room", "pingpong room", "table tennis room",
            # Board games
            "board game room", "boardgame room", "card room", "poker room",
            "mahjong room", "chess room",
            # Video games
            "video game room", "videogame room", "arcade room", "esports room",
            # Other
            "bar room", "lounge room", "party room"
        ],
        "aliases": {}
    }
}


def match_room_type(user_input: str) -> Optional[str]:
    """
    Match room type from user input
    
    Args:
        user_input: User requirement description or room type string
        
    Returns:
        Matched standard room type name, or None
    """
    if not user_input:
        return None
    
    user_input_lower = user_input.lower().strip()
    
    # Iterate over all room types, check matching_patterns
    for room_type, config in ROOM_TYPE_ESSENTIAL_OBJECTS.items():
        patterns = config.get("matching_patterns", [room_type])
        for pattern in patterns:
            if pattern in user_input_lower:
                return room_type
    
    return None


class SceneEditingInteraction(BaseInteraction):
    """
    Scene Editing Interaction Class
    
    Handles the interaction flow for scene editing tasks, including:
    1. Parsing tool_calls from LLM output
    2. Invoking scene editing and rendering
    3. Computing physics rewards based on voxel evaluation
    4. Managing multi-turn interaction state
    5. Using VLM judge for scene quality evaluation
    """
    
    @staticmethod
    def extract_create_scene_from_response(response_text: str) -> Optional[Dict[str, Any]]:
        """Extract <create_scene> content from model response"""
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
        """Convert image file to base64 string"""
        with open(image_path, 'rb') as f:
            img_data = f.read()
        return f"data:image/png;base64,{base64.b64encode(img_data).decode()}"
    
    @staticmethod
    def extract_think_content(response_text: str) -> Optional[str]:
        """Extract <think> content from model response"""
        import re
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize scene editing interaction
        
        Args:
            config: Configuration dictionary containing the following keys:
                - max_turns: Maximum number of interaction turns (default 10)
                - models_base_path: Base path for 3D models
                - voxel_size: Voxel size (default 0.05)
                - reward_threshold: PBL loss threshold (default 1e-5)
                - output_dir: Output directory (default "./scene_editing_output")
                - verbose: Whether to output detailed logs (default False)
                - paths: Unified path configuration block (optional)
        """
        # ========== Initialize PathConfig singleton (unified path configuration) ==========
        try:
            from path_config import PathConfig
            path_config = PathConfig.init_from_config(config)
            print(f"✓ PathConfig initialized: {path_config}")
        except Exception as e:
            print(f"⚠ PathConfig initialization failed: {e}, using fallback paths")
            path_config = None
        
        # Create custom log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        
        # Prefer PathConfig, then use hardcoded fallback
        if path_config and path_config.logs_dir:
            log_dir = Path(path_config.logs_dir) / "interaction_logs"
        elif Path("/path/to/logs").exists():
            log_dir = Path("/path/to/logs/interaction_logs")
        else:
            log_dir = Path("./logs/interaction_logs")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / f"scene_editing_interaction_{timestamp}_pid{pid}.log"
        # self.log_file = Path(f"/path/to/data/logs/scene_editing_interaction_{timestamp}_pid{pid}.log")

        # Set up logger
        self.logger = logging.getLogger(f"scene_editing_{timestamp}_{pid}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Log format
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Log initialization start
        self.logger.info("="*80)
        self.logger.info(f"SceneEditingInteraction.__init__ called")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"Process ID: {pid}")
        self.logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")
        if path_config:
            self.logger.info(f"PathConfig: {path_config}")
        
        # Print to console (may be captured by Ray)
        print(f"✓ SceneEditingInteraction log file: {self.log_file}")
        
        super().__init__(config)
        self.name = "scene_editing"
        self.max_turns = config.get("max_turns", 10)
        self.output_dir = Path(config.get("output_dir", "./scene_editing_output"))
        self.verbose = config.get("verbose", False)
        
        # Physics evaluation mode: 'voxel' or 'trimesh' (default: trimesh)
        self.physics_mode = config.get("physics_mode", "trimesh")
        
        # Asset source configuration: whether to use Objaverse (default True, use Objaverse)
        self.use_objaverse = config.get("use_objaverse", True)
        
        # VLM Judge configuration
        self.vlm_judge_enabled = config.get("vlm_judge_enabled", True)
        self.vlm_judge_url = config.get("vlm_judge_url", "http://localhost:8000/v1/chat/completions")
        self.vlm_judge_model = config.get("vlm_judge_model", "Qwen/Qwen2.5-VL-72B-Instruct")
        self.vlm_judge_timeout = config.get("vlm_judge_timeout", 60)
        
        # Semaphore configuration (limit concurrent rendering/heavy tasks)
        self.semaphore_limit = config.get("semaphore_limit", 32)
        
        # Feedback injection configuration
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
        
        # Initialize voxel reward calculator - prefer PathConfig
        # When using Objaverse mode, 3D-FUTURE path is optional
        models_base_path = None
        if path_config and path_config.future3d_models_dir:
            models_base_path = path_config.future3d_models_dir
            self.logger.info(f"Using PathConfig models_base_path: {models_base_path}")
        elif not self.use_objaverse:
            # Only require 3D-FUTURE path in non-Objaverse mode
            models_base_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
            # Fallback to local path
            if not Path(models_base_path).exists():
                alt_path = "/path/to/datasets/3d-front/3D-FUTURE-model"
                if Path(alt_path).exists():
                    models_base_path = alt_path
                    self.logger.info(f"Using alternative models_base_path: {models_base_path}")
        else:
            # Objaverse mode: 3D-FUTURE path is optional, try to find but don't enforce
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
        
        # Initialize the corresponding evaluator based on physics_mode
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
        
        # Store state for each instance
        self._instance_dict = {}
        
        # Support type cache: stores mapping from object description to support type, avoiding repeated LLM calls
        self.support_type_cache = {}
        
        # Objaverse asset database (lazy initialization, for getting real object sizes)
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
        Get render semaphore (lazy initialization, avoids binding to event loop at module import time)
        
        Uses a class-level semaphore, created on first call.
        This ensures the semaphore is created in the correct event loop.
        
        Returns:
            asyncio.Semaphore instance
        """
        global _RENDER_SEMAPHORE
        if _RENDER_SEMAPHORE is None:
            _RENDER_SEMAPHORE = asyncio.Semaphore(self.semaphore_limit)
            self.logger.info(f"Created render semaphore with limit: {self.semaphore_limit}")
        return _RENDER_SEMAPHORE
    
    def _get_objaverse_database(self) -> Optional[Dict]:
        """
        Get Objaverse asset database (lazy initialization)
        
        Returns:
            Asset database dictionary, or None (if loading failed)
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
        Get real size of an object by asset ID (following Holodeck's get_bbox_dims method)
        
        Data path: annotations[uid]["thor_metadata"]["assetMetadata"]["boundingBox"]
        boundingBox format: {min: {x, y, z}, max: {x, y, z}}
        
        Args:
            uid: Objaverse asset UID
            
        Returns:
            Dictionary containing x, y, z dimensions (unit: meters), or None (if failed)
        """
        if not uid:
            return None
        
        database = self._get_objaverse_database()
        if database is None or uid not in database:
            return None
        
        try:
            obj_data = database[uid]
            
            # Get assetMetadata (following Holodeck's get_asset_metadata)
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
            
            # Get boundingBox dimensions (following Holodeck's get_bbox_dims)
            bbox_info = asset_metadata["boundingBox"]
            
            if "x" in bbox_info:
                return bbox_info
            if "size" in bbox_info:
                return bbox_info["size"]
            
            # Calculate dimensions from min/max
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
        Call VLM judge for scoring or getting text response
        
        Args:
            image_path: Rendered image path
            prompt: Scoring prompt
            max_retries: Maximum number of retries
            max_tokens: Maximum generation tokens (50 for scoring, larger values for analysis/description)
            return_text: If True, return raw text instead of parsed score
            
        Returns:
            If return_text=False: Score (-1.0, -0.5, 0.0, 0.5, 1.0) or None (if failed)
            If return_text=True: Raw text response or None (if failed)
        """
        if not self.vlm_judge_enabled:
            self.logger.info("VLM judge disabled, skipping")
            return None
        
        if not Path(image_path).exists():
            self.logger.error(f"Image not found: {image_path}")
            return None
        
        # Convert image to base64
        try:
            img_base64 = self.image_to_base64(image_path)
        except Exception as e:
            self.logger.error(f"Failed to convert image to base64: {e}")
            return None
        
        # Build request
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
        
        # Retry logic
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
                    
                    # If text return is needed, return directly
                    if return_text:
                        self.logger.info(f"VLM judge text response (first 200 chars): {content[:200]}...")
                        return content
                    
                    # Parse score (supports 5-level scoring: -1.0, -0.5, 0.0, 0.5, 1.0)
                    import re
                    # First try to match decimal scores
                    match = re.search(r'(-?[01])\.([05])', content)
                    if match:
                        score = float(f"{match.group(1)}.{match.group(2)}")
                        self.logger.info(f"VLM judge score: {score}, response: {content}")
                        return score
                    # If no decimal, try matching integers
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
                    await asyncio.sleep(1)  # Wait 1 second before retrying
        
        self.logger.error("VLM judge failed after all retries")
        return None
    
    async def _vlm_analyze_scene_problems(
        self,
        image_path: str,
        scene_json_str: str,
        user_requirement: str
    ) -> Optional[str]:
        """
        Phase 1: Let VLM independently analyze problems in the scene (used for intermediate turns)
        
        Args:
            image_path: Rendered image path
            scene_json_str: Scene JSON string
            user_requirement: User requirement
            
        Returns:
            Text listing problems identified by VLM, or None (if failed)
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

**1. Physical Bugs (Physical Issues)**:
- Object Overlap/Collision: Two or more objects occupying the same space (check bboxes intersecting in TOP VIEW)
- Out of Bounds: Objects extending beyond room boundaries (check bboxes vs floor grid)
- Floating Objects: Objects not properly supported (check DIAGONAL VIEW)
- Example: "PHYSICAL: The coffee table bbox overlaps with sofa bbox by approximately 0.3m"
- Example: "PHYSICAL: The wardrobe extends 0.5m beyond the north wall boundary"

**2. Layout Rationality Bugs (Scene Rationality Issues)**:
- Core Furniture Misplacement: Bed/sofa not against wall, in room center
- Missing Essential Items: Room lacks core furniture for its type (bedroom needs bed, living room needs sofa)
- Improper Orientation: Furniture facing wrong direction (sofa facing wall instead of TV area)
- Example: "RATIONALITY: The bed is placed in the center of the room, not against any wall"
- Example: "RATIONALITY: The sofa is facing the corner wall instead of the open area"

**3. Spatial Distribution Bugs (Spatial Distribution Issues)**:
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
        Generate brief VLM layout feedback (excluding collision/out-of-bounds issues, which are handled by trimesh)
        
        Args:
            image_path: Rendered image path
            user_requirement: User requirement
            
        Returns:
            Brief layout feedback text, or empty string if no issues or failed
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
                # Clean result, ensure concise
                result = result.strip()
                # If VLM returns "no issues" or similar content, return empty string
                if any(phrase in result.lower() for phrase in ["no issue", "looks good", "well-designed", "properly placed"]):
                    return ""
                return result
            return ""
        except Exception as e:
            self.logger.warning(f"VLM layout feedback generation failed: {e}")
            return ""
    
    def _extract_tool_calls_from_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Extract the latest tool call list from message history
        
        Args:
            messages: Message list
            
        Returns:
            Tool call list, each element containing name and arguments
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
        Evaluate whether new objects added/replaced in this turn are relevant to the scene requirements
        
        Only evaluates new objects in add_object and replace_object operations
        
        Args:
            tool_calls: Tool call list
            user_requirement: User requirement description
            room_type: Room type
            
        Returns:
            Dictionary containing score and details:
            - score: Score (1.0, 0.0, -1.0)
            - relevant_objects: List of relevant objects
            - irrelevant_objects: List of irrelevant objects
        """
        result = {
            "score": 0.0,
            "relevant_objects": [],
            "irrelevant_objects": []
        }
        
        if not self.vlm_judge_enabled:
            return result
        
        # Extract new object descriptions from add_object and replace_object
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
            # No add/replace operations, return neutral score
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
            
            # Parse JSON response
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
            
            # Calculate score
            total = len(new_objects)
            irrelevant_count = len(irrelevant)
            
            if irrelevant_count == 0:
                # All relevant
                result["score"] = 1.0
            elif irrelevant_count >= total:
                # All irrelevant
                result["score"] = -1.0
            else:
                # Partially relevant: calculate based on ratio
                # If irrelevant ratio > 50%, negative score; otherwise positive score
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
        Evaluate whether object sizes and proportions in the scene are reasonable (programmatic evaluation using real asset sizes)
        
        Retrieves real sizes via asset ID and compares with reasonable size ranges for common objects
        
        Args:
            image_path: Rendered image path (parameter preserved but unused, as evaluation is now programmatic)
            scene_data: Scene JSON data
            user_requirement: User requirement (parameter preserved but unused)
            
        Returns:
            Dictionary containing score and details:
            - score: Score (-1.0, -0.5, 0.0, 0.5, 1.0)
            - issues: List of size/proportion issues found
        """
        result = {
            "score": 1.0,  # Default full score, deducted when issues found
            "issues": []
        }
        
        # Define reasonable size ranges for various object types (unit: meters)
        # Format: keyword -> (min_width, max_width, min_height, max_height, min_depth, max_depth)
        SIZE_STANDARDS = {
            # Beds
            "bed": (1.2, 2.5, 0.3, 0.8, 1.8, 2.2),
            "double bed": (1.4, 2.2, 0.3, 0.8, 1.9, 2.2),
            "single bed": (0.9, 1.2, 0.3, 0.8, 1.8, 2.1),
            # Seating
            "sofa": (1.5, 3.5, 0.6, 1.2, 0.7, 1.2),
            "couch": (1.5, 3.5, 0.6, 1.2, 0.7, 1.2),
            "chair": (0.4, 0.7, 0.7, 1.2, 0.4, 0.7),
            "office chair": (0.5, 0.8, 0.9, 1.3, 0.5, 0.7),
            "armchair": (0.6, 1.0, 0.7, 1.1, 0.6, 1.0),
            # Tables
            "desk": (0.8, 2.0, 0.7, 0.85, 0.5, 0.9),
            "table": (0.6, 2.5, 0.4, 0.9, 0.6, 1.5),
            "coffee table": (0.6, 1.5, 0.3, 0.6, 0.4, 1.0),
            "dining table": (0.8, 2.5, 0.7, 0.85, 0.8, 1.5),
            "nightstand": (0.35, 0.6, 0.4, 0.7, 0.35, 0.55),
            "bedside table": (0.35, 0.6, 0.4, 0.7, 0.35, 0.55),
            # Cabinets/Storage
            "wardrobe": (0.8, 2.5, 1.8, 2.5, 0.5, 0.7),
            "closet": (0.8, 2.5, 1.8, 2.5, 0.5, 0.7),
            "cabinet": (0.4, 1.5, 0.6, 2.2, 0.3, 0.7),
            "bookshelf": (0.6, 1.5, 1.2, 2.2, 0.25, 0.45),
            "bookcase": (0.6, 1.5, 1.2, 2.2, 0.25, 0.45),
            "tv stand": (0.8, 2.0, 0.4, 0.7, 0.35, 0.55),
            "sideboard": (1.0, 2.2, 0.7, 1.0, 0.4, 0.6),
            # Gym equipment
            "treadmill": (0.7, 1.0, 1.2, 1.6, 1.5, 2.2),
            "exercise bike": (0.5, 0.7, 1.0, 1.5, 1.0, 1.5),
            "weight bench": (0.5, 0.8, 0.4, 0.6, 1.2, 1.8),
            "power rack": (1.0, 1.5, 2.0, 2.5, 1.2, 1.8),
            # Others
            "lamp": (0.15, 0.5, 0.3, 1.8, 0.15, 0.5),
            "floor lamp": (0.25, 0.5, 1.2, 1.9, 0.25, 0.5),
            "tv": (0.8, 2.0, 0.5, 1.2, 0.05, 0.2),
            "mirror": (0.3, 1.5, 0.5, 2.0, 0.02, 0.1),
            "rug": (1.0, 4.0, 0.01, 0.05, 1.0, 3.0),
            "plant": (0.2, 0.8, 0.3, 2.0, 0.2, 0.8),
        }
        
        # Extract size information for all objects
        objects_info = []
        
        def extract_object_info(obj):
            """Extract size info from a single object, preferring real asset sizes"""
            desc = obj.get('desc', obj.get('object_description', 'unknown'))
            specified_size = obj.get('size', [1, 1, 1])
            uid = obj.get('uid', None)
            
            # Prefer looking up real size from database via uid (ground truth)
            # retrieved_size may be normalized to [1,1,1], unreliable
            real_size = None
            size_source = "specified"
            
            if uid:
                real_bbox = self._get_asset_real_size(uid)
                if real_bbox:
                    # Check if it's a valid real size (not default value)
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
        
        # Programmatic check of each object's size
        critical_issues = 0  # Number of critical issues
        minor_issues = 0     # Number of minor issues
        
        for obj_info in objects_info:
            desc = obj_info['desc'].lower()
            size = obj_info['size']  # [width, height, depth]
            
            # Only check with standard ranges when size source is "specified" (user-specified)
            # If size comes from "ground_truth" (retrieved_size or uid query), trust it directly
            if obj_info['size_source'] == "specified":
                # Try to match object type
                matched_standard = None
                for keyword, standard in SIZE_STANDARDS.items():
                    if keyword in desc:
                        matched_standard = (keyword, standard)
                        break
                
                if matched_standard:
                    keyword, (min_w, max_w, min_h, max_h, min_d, max_d) = matched_standard
                    w, h, d = size[0], size[1], size[2]
                    
                    issues_for_obj = []
                    
                    # Check width
                    if w < min_w * 0.5:  # Too small (below 50% of minimum)
                        issues_for_obj.append(f"width {w:.2f}m is too small (expected {min_w:.1f}-{max_w:.1f}m)")
                        critical_issues += 1
                    elif w < min_w * 0.8:  # Slightly small
                        issues_for_obj.append(f"width {w:.2f}m is slightly small")
                        minor_issues += 1
                    elif w > max_w * 2.0:  # Too large
                        issues_for_obj.append(f"width {w:.2f}m is too large (expected {min_w:.1f}-{max_w:.1f}m)")
                        critical_issues += 1
                    elif w > max_w * 1.3:  # Slightly large
                        issues_for_obj.append(f"width {w:.2f}m is slightly large")
                        minor_issues += 1
                    
                    # Check height
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
                    
                    # Check depth
                    if d < min_d * 0.5:
                        issues_for_obj.append(f"depth {d:.2f}m is too small (expected {min_d:.1f}-{max_d:.1f}m)")
                        critical_issues += 1
                    elif d > max_d * 2.0:
                        issues_for_obj.append(f"depth {d:.2f}m is too large (expected {min_d:.1f}-{max_d:.1f}m)")
                        critical_issues += 1
                    
                    if issues_for_obj:
                        result["issues"].append(f"{obj_info['desc']}: {'; '.join(issues_for_obj)}")
            else:
                # Size comes from asset database, is real size, logged as verified
                self.logger.debug(f"Object '{obj_info['desc']}' has verified size from {obj_info['size_source']}: {size}")
            
            # Check difference between user-specified size and real asset size (only when real size is available)
            if obj_info['size_source'] != "specified":
                specified = obj_info['specified_size']
                real = obj_info['size']
                if specified != [1, 1, 1]:  # Non-default value
                    diff_ratio = max(
                        abs(specified[0] - real[0]) / max(real[0], 0.01),
                        abs(specified[1] - real[1]) / max(real[1], 0.01),
                        abs(specified[2] - real[2]) / max(real[2], 0.01)
                    )
                    if diff_ratio > 1.0:  # Over 100% difference
                        result["issues"].append(
                            f"{obj_info['desc']}: user specified size {[round(s,2) for s in specified]} differs greatly "
                            f"from real asset size {[round(s,2) for s in real]} (ground truth)"
                        )
                        critical_issues += 1
                    elif diff_ratio > 0.5:  # Over 50% difference
                        result["issues"].append(
                            f"{obj_info['desc']}: user specified size {[round(s,2) for s in specified]} differs "
                            f"from real asset size {[round(s,2) for s in real]} (ground truth)"
                        )
                        minor_issues += 1
        
        # Calculate score based on number of issues
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
        Extract key objects from user requirements and evaluate whether the scene contains them
        
        Steps:
        1. Based on room type and user instructions, let VLM extract 3-5 key objects and 1 most critical object (only on first call, cached afterwards)
        2. Let VLM determine whether the current scene contains these objects
        3. Return score and details
        
        Args:
            image_path: Rendered image path
            scene_summary: Scene object summary (extracted by _extract_objects_summary)
            user_requirement: User requirement description
            room_type: Room type (e.g., bedroom, living room, etc.)
            instance_id: Instance ID, used for caching key objects list
            
        Returns:
            Dictionary containing score and details:
            - score: Score (1.0, 0.0, -1.0)
            - key_objects: Key objects list
            - most_critical_object: Most critical object
            - found_objects: Objects found in the scene
            - missing_objects: Objects missing from the scene
        """
        result = {
            "score": 0.0,
            "key_objects": [],
            "most_critical_object": "",
            "found_objects": [],
            "missing_objects": [],
            "essential_objects_missing": [],  # New: missing essential base objects
            "essential_objects_found": []     # New: found essential base objects
        }
        
        if not self.vlm_judge_enabled:
            self.logger.info("VLM judge disabled, skipping key objects evaluation")
            return result
        
        # ========== New: Check essential base objects for room type ==========
        # Standardize room type (enhanced matching using match_room_type function)
        normalized_room_type = ""
        
        # First try to match from the passed room_type parameter
        if room_type:
            normalized_room_type = match_room_type(room_type) or ""
        
        # If no match, try to infer room type from user requirement
        if not normalized_room_type:
            normalized_room_type = match_room_type(user_requirement) or ""
        
        # Get the essential objects definition for this room type
        essential_config = ROOM_TYPE_ESSENTIAL_OBJECTS.get(normalized_room_type, None)
        
        self.logger.info(f"Room type: '{normalized_room_type}', essential config: {essential_config is not None}")
        
        # Step 1: Check cache, use cached data if available
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
        
        # If no cache, call VLM to extract key objects
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
            # Call VLM to extract key objects (no image needed)
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
            
            # Parse JSON response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                extracted_json = json.loads(json_match.group(1))
            else:
                # Try direct parsing
                extracted_json = json.loads(content)
            
            key_objects = extracted_json.get("mandatory_objects", [])
            # Compatible with old format
            if not key_objects:
                key_objects = extracted_json.get("key_objects", [])
            
            # Ensure list is not empty
            if not key_objects:
                self.logger.warning("Failed to extract mandatory objects from VLM response")
                return result
            
            # Simple most_critical logic (take the first one)
            most_critical = key_objects[0] if key_objects else ""
            
            result["most_critical_object"] = most_critical
            result["key_objects"] = key_objects
            self.logger.info(f"Extracted mandatory objects: {key_objects}")
            
            # Cache key objects list to avoid repeated VLM calls
            if instance_id and instance_id in self._instance_dict:
                self._instance_dict[instance_id]["cached_key_objects"] = key_objects
                self._instance_dict[instance_id]["cached_most_critical_object"] = most_critical
                self.logger.info(f"Cached key objects for instance {instance_id}")
            
        except Exception as e:
            self.logger.warning(f"Key objects extraction failed: {e}")
            return result
        
        # Step 2: Check if scene contains these objects
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
            # Call VLM for evaluation (image needed)
            score_result = await self._call_vlm_judge(
                image_path,
                prompt_evaluate,
                max_tokens=300,
                return_text=True
            )
            
            if not score_result:
                self.logger.warning("Key objects evaluation VLM call returned None")
                return result
            
            # Parse JSON response
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', score_result, re.DOTALL)
            if json_match:
                eval_json = json.loads(json_match.group(1))
            else:
                # Try direct parsing
                eval_json = json.loads(score_result)
            
            found_count = eval_json.get("found_count", 0)
            total_count = eval_json.get("total_count", len(key_objects))
            found_objects = eval_json.get("found_objects", [])
            missing_objects = eval_json.get("missing_objects", [])
            
            result["found_objects"] = found_objects
            result["missing_objects"] = missing_objects
            
            # ========== New: Check essential base objects ==========
            essential_missing = []
            essential_found = []
            
            if essential_config:
                essential_items = essential_config.get("essential", [])
                aliases_map = essential_config.get("aliases", {})
                
                # Convert found objects descriptions to lowercase for matching
                found_objects_lower = [obj.lower() for obj in found_objects]
                scene_summary_lower = scene_summary.lower() if scene_summary else ""
                
                for essential_item in essential_items:
                    # Get all aliases for this object
                    item_aliases = aliases_map.get(essential_item, [essential_item])
                    
                    # Check if it exists in found objects or scene summary
                    found = False
                    for alias in item_aliases:
                        alias_lower = alias.lower()
                        # Check found_objects
                        for found_obj in found_objects_lower:
                            if alias_lower in found_obj or found_obj in alias_lower:
                                found = True
                                break
                        if not found:
                            # Check scene summary
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
            
            # ========== Calculate score (combined with essential objects check) ==========
            # If essential base objects are missing, directly set to -1
            if essential_missing:
                result["score"] = -1.0
                self.logger.info(f"Key objects score: -1.0 (Missing ESSENTIAL objects: {essential_missing})")
            elif total_count == 0:
                result["score"] = 0.0
            else:
                ratio = found_count / total_count
                if ratio >= 0.99:  # Allow minor error, treat as all included
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
            # If evaluation failed, return neutral score
            result["score"] = 0.0
        
        return result
    
    async def _vlm_describe_scene(
        self,
        image_path: str,
        scene_json_str: str,
        user_requirement: str
    ) -> Optional[str]:
        """
        Phase 1: Let VLM describe the scene in detail (used for final turn)
        
        Args:
            image_path: Rendered image path
            scene_json_str: Scene JSON string
            user_requirement: User requirement
            
        Returns:
            Detailed scene description text, or None (if failed)
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
        Let VLM generate an expected scene graph based on user requirements and room type
        
        Describes 10-30 key objects and their spatial relationship constraints that the scene should contain
        
        Args:
            user_requirement: User requirement description
            room_type: Room type (e.g., bedroom, living room, etc.)
            
        Returns:
            Free-text description of the expected scene graph, or None (if failed)
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

        # No image needed here, only text generation
        # Use a simple text request
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
        Let VLM generate an actual scene graph based on rendered image and scene JSON
        
        Describes the actual positions and spatial relationships of objects in the current scene
        
        Args:
            image_path: Rendered image path
            scene_json_str: Scene JSON string
            user_requirement: User requirement
            
        Returns:
            Free-text description of the actual scene graph, or None (if failed)
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
        Let VLM compare expected and actual scene graphs to evaluate constraint satisfaction
        
        Args:
            image_path: Rendered image path (for visual verification)
            expected_graph: Expected scene graph description
            actual_graph: Actual scene graph description
            user_requirement: User requirement
            
        Returns:
            Scene graph constraint satisfaction score (-1.0, -0.5, 0.0, 0.5, 1.0)
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
        Intermediate turn VLM scoring: compare previous and current turn renderings to determine if the scene has improved
        
        Args:
            prev_image_path: Previous turn rendering path (concatenated top view + diagonal view)
            current_image_path: Current turn rendering path (concatenated top view + diagonal view)
            user_requirement: User requirement/room description
            prev_scene_summary: Previous turn scene object summary (description, position, size)
            current_scene_summary: Current scene object summary (description, position, size)
            
        Returns:
            Dictionary containing one dimension score:
            - scene_improvement: Whether the scene has improved (-1.0, -0.5, 0.0, 0.5, 1.0)
        """
        self.logger.info("Starting intermediate turn VLM judge evaluation (scene improvement comparison)")
        
        scores = {
            "scene_improvement": 0.0
        }
        
        # Check if both images exist
        if not prev_image_path or not Path(prev_image_path).exists():
            self.logger.warning(f"Previous image not found: {prev_image_path}")
            return scores
        if not current_image_path or not Path(current_image_path).exists():
            self.logger.warning(f"Current image not found: {current_image_path}")
            return scores
        
        # Convert both images to base64
        try:
            prev_img_base64 = self.image_to_base64(prev_image_path)
            curr_img_base64 = self.image_to_base64(current_image_path)
        except Exception as e:
            self.logger.error(f"Failed to convert images to base64: {e}")
            return scores
        
        # Build scene objects summary section (if provided)
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
        
        # Build comparison evaluation prompt
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

        # Build request with two images
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
        
        # Call VLM for scoring
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
                    
                    # Parse score (prefer 3-level integer score: -1, 0, 1)
                    import re
                    # First try to match integer score (-1, 0, 1)
                    match = re.search(r'(-1|0|1)(?:\s|$|[,.])', content)
                    if match:
                        score = float(match.group(1))
                        scores["scene_improvement"] = score
                        self.logger.info(f"Intermediate turn VLM score: {score}, response: {content}")
                        break
                    # Compatible with old decimal format (-1.0, -0.5, 0.5, 1.0)
                    match = re.search(r'(-?[01])\.([05])', content)
                    if match:
                        score = float(f"{match.group(1)}.{match.group(2)}")
                        # Map old format to new 3-level scoring
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
        Extract object summary information from the scene (for VLM evaluation, saving tokens)
        
        Only extracts key information: object description, position, size, rotation
        
        Args:
            scene_data: Scene JSON data
            
        Returns:
            Object summary string (compact format)
        """
        try:
            objects_list = []
            
            # Extract objects from groups structure
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
            
            # Also check flat format objects field
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
            
            # Generate compact format summary
            summary_lines = [f"Total objects: {len(objects_list)}"]
            for i, obj in enumerate(objects_list, 1):
                # Format position and size as compact strings (2 decimal places)
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
        Final turn VLM scoring: consolidated evaluation (reduce VLM call count)
        
        Phase 1: Let VLM describe the scene in detail (including object inventory, positions, orientations, relationships, problems)
        Phase 2: Evaluate three dimensions in one call: rationality, requirement match, scene graph constraints
        
        Note: aesthetics evaluation has been removed
        
        Args:
            image_path: Final rendering path (left: top view, right: diagonal view)
            user_requirement: User requirement
            current_scene: Current scene JSON data
            
        Returns:
            Dictionary containing three dimension scores:
            - rationality: Scene rationality (-1.0, -0.5, 0.0, 0.5, 1.0) - object completeness, spatial distribution, layout realism
            - scene_graph: Scene graph constraint satisfaction (-1.0, -0.5, 0.0, 0.5, 1.0) - spatial relationship constraints
            - requirement_match: Match with user requirements (-1.0, -0.5, 0.0, 0.5, 1.0)
        """
        self.logger.info("Starting final turn VLM judge evaluation (consolidated)")
        
        scores = {
            "rationality": 0.0,
            "scene_graph": 0.0,
            "requirement_match": 0.0
        }
        
        # Convert scene JSON to formatted string
        scene_json_str = json.dumps(current_scene, indent=2, ensure_ascii=False)
        
        # ========== Phase 1: VLM detailed scene description ==========
        self.logger.info("Phase 1: VLM comprehensive scene description")
        scene_description = await self._vlm_describe_scene(
            image_path, scene_json_str, user_requirement
        )
        
        if scene_description is None:
            self.logger.warning("VLM scene description failed, falling back to JSON-only evaluation")
            scene_description = "Scene description unavailable. Please evaluate based on the image and JSON data only."
        else:
            self.logger.info(f"VLM scene description completed: {scene_description[:500]}...")
        
        # Extract room type from scene
        room_type = current_scene.get('room_type', '')
        
        # ========== Phase 2: Consolidated evaluation of three dimensions (single VLM call) ==========
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

## DIMENSION 1: RATIONALITY (Scene Rationality)
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

## DIMENSION 2: REQUIREMENT MATCH (Requirement Match)
Check these aspects:
1. **Explicit Requirements**: Are all user-requested items present?
2. **Implicit Requirements**: For the room type, are standard essentials present?
3. **Relevance**: Are objects appropriate for this room? Any irrelevant objects (e.g., toilet in bedroom)?
4. **Style/Theme**: Does it match any requested style?

## DIMENSION 3: SCENE GRAPH (Scene Graph Constraints)
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
            # If consolidated evaluation failed, try fallback to separate evaluations
            self.logger.info("Falling back to separate evaluations...")
            
            # Separately evaluate rationality
            prompt_rationality = f"""Evaluate scene RATIONALITY only.
User requirement: {user_requirement}
Scene description: {scene_description}

Check: object completeness, spatial distribution, layout realism, object sizes.
Score: 1.0=excellent, 0.5=good, 0.0=borderline, -0.5=poor, -1.0=failed

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""
            
            score_rationality = await self._call_vlm_judge(image_path, prompt_rationality)
            if score_rationality is not None:
                scores["rationality"] = float(score_rationality)
            
            # Separately evaluate requirement match
            prompt_match = f"""Evaluate REQUIREMENT MATCH only.
User requirement: {user_requirement}
Scene description: {scene_description}

Check: explicit requirements, implicit requirements, object relevance, style match.
Score: 1.0=excellent, 0.5=good, 0.0=borderline, -0.5=poor, -1.0=failed

Output ONLY one number: -1.0, -0.5, 0.0, 0.5, or 1.0"""
            
            score_match = await self._call_vlm_judge(image_path, prompt_match)
            if score_match is not None:
                scores["requirement_match"] = float(score_match)
            
            # Separately evaluate scene graph constraints
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
        Start a new interaction instance
        
        Args:
            instance_id: Instance ID, auto-generated if None
            **kwargs: Additional arguments, including:
                - initial_scene: Initial scene JSON data
                - max_turns: Override default maximum turns
                
        Returns:
            Instance ID
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
        
        # Get initial scene (first turn will extract from model response, so can be empty)
        initial_scene = kwargs.get("initial_scene", {})
        
        # Log initial scene state
        if initial_scene:
            self.logger.info(f"Initial scene provided with {len(initial_scene.get('objects', initial_scene.get('groups', [])))} objects/groups")
        else:
            self.logger.info("No initial scene provided, will extract from model response in first turn")
        
        # Create instance output directory
        instance_output_dir = self.output_dir / instance_id
        instance_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created output directory: {instance_output_dir}")
        
        # Initialize instance state
        self._instance_dict[instance_id] = {  
            "current_scene": initial_scene,  
            "turn_count": 0,  
            "max_turns": kwargs.get("max_turns", self.max_turns),  
            "output_dir": instance_output_dir,  
            "history": [],  # Store operation history for each turn  
            "rewards": [],  # Store total reward values for each turn  
            "reward_components": [],  # New - store named reward components for each turn  
            "total_reward": 0.0,
            "user_requirement": kwargs.get("user_requirement", ""),  # Save user requirement
            # Key objects cache (avoid repeated VLM calls per turn)
            "cached_key_objects": None,  # Cached key objects list
            "cached_most_critical_object": None  # Cached most critical object
        }
        
        # Save initial scene (if provided)
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
        Calculate room shape reward based on number of vertices (only checks vertex count, only accepts even vertex counts)
        
        Reward rules:
        - 4 vertices: 1.0 (best)
        - 6 vertices: 0.5
        - 8 vertices: 0.5
        - Other vertex counts: -1.0
        
        Also validates each point: each point must be a list containing at least 3 coordinates
        
        Args:
            bounds_bottom: List of room bottom boundary points
            bounds_top: List of room top boundary points
        
        Returns:
            (reward, message)
        """
        try:
            import numpy as np
            
            # First validate each point (must be a list/tuple with exactly 3 coordinates)
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
            
            # Convert to numpy arrays
            bounds_bottom = np.array(bounds_bottom)
            bounds_top = np.array(bounds_top)
            
            num_bottom = len(bounds_bottom)
            num_top = len(bounds_top)
            
            # Vertex count mismatch
            if num_bottom != num_top:
                return -1.0, f"Bottom ({num_bottom}) and top ({num_top}) have different number of points"
            
            # Calculate room dimensions for logging
            x_coords = bounds_bottom[:, 0]
            z_coords = bounds_bottom[:, 2]
            x_size = x_coords.max() - x_coords.min()
            z_size = z_coords.max() - z_coords.min()
            
            # Only accept even vertex counts: 4, 6, 8
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
        Check if room has a valid vertex count (backward-compatible wrapper method)
        
        Only accepts even vertex counts: 4, 6, 8
        
        Args:
            bounds_bottom: List of room bottom boundary points
            bounds_top: List of room top boundary points
        
        Returns:
            (is_valid, error_message)
        """
        reward, message = self._check_room_shape_reward(bounds_bottom, bounds_top)
        # Reward >= 0 is considered an acceptable room shape
        return reward >= 0.0, message
    
    def _check_room_type_reward(self, scene_data: Dict[str, Any], user_requirement: str) -> float:
        """
        Check if generated room_type and room_id match user requirements
        
        Args:
            scene_data: Scene JSON data containing room_type and room_id fields
            user_requirement: User requirement description
        
        Returns:
            1.0: Both match or cannot determine
            0.0: Only one matches
            -1.0: Neither matches
        """
        # Define supported room types (standardized to lowercase for matching)
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
        
        # Room type alias mapping (for matching different expressions)
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
        
        # If no user requirement, cannot determine, return 1.0
        if not user_requirement:
            self.logger.info("No user requirement provided, cannot check room type")
            return 1.0
        
        # Extract expected room type from user requirement
        user_requirement_lower = user_requirement.lower()
        expected_room_type = None
        
        for canonical_type, aliases in room_type_aliases.items():
            for alias in aliases:
                if alias in user_requirement_lower:
                    expected_room_type = canonical_type
                    break
            if expected_room_type:
                break
        
        # If cannot identify room type from user requirement, return 1.0
        if not expected_room_type:
            self.logger.info(f"Could not identify room type from user requirement: {user_requirement[:100]}...")
            return 1.0
        
        self.logger.info(f"Expected room type from user requirement: {expected_room_type}")
        
        # Get room_type and room_id from scene
        generated_room_type = scene_data.get("room_type", "").lower().strip()
        generated_room_id = scene_data.get("room_id", "").lower().strip()
        
        self.logger.info(f"Generated room_type: '{generated_room_type}', room_id: '{generated_room_id}'")
        
        # Check if room_type matches
        room_type_match = False
        if expected_room_type in room_type_aliases:
            for alias in room_type_aliases[expected_room_type]:
                # Check if generated room_type contains alias of expected type
                if alias.replace(" ", "") in generated_room_type.replace(" ", ""):
                    room_type_match = True
                    break
        
        # Check if room_id matches (room_id usually contains room type info, e.g., "LivingRoom-1003")
        room_id_match = False
        if expected_room_type in room_type_aliases:
            for alias in room_type_aliases[expected_room_type]:
                # Compare after removing spaces
                alias_no_space = alias.replace(" ", "").lower()
                if alias_no_space in generated_room_id.replace(" ", "").replace("-", "").replace("_", ""):
                    room_id_match = True
                    break
        
        self.logger.info(f"Room type match: {room_type_match}, Room ID match: {room_id_match}")
        
        # Calculate reward
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
        Generate response and execute scene editing
        
        Args:
            instance_id: Instance ID
            messages: Message history list
            **kwargs: Additional arguments, including:
                - tool_calls: Tool call list
                - retrieve_assets: Whether to perform asset retrieval (default True)
                
        Returns:
            (is_done, response_text, reward, metadata) tuple:
            - is_done: Whether the interaction is complete
            - response_text: Response text
            - reward: Reward for this turn
            - metadata: Metadata (including rendered images, etc.)
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
        
        # First turn special handling: extract the model-generated initial scene
        if current_turn == 1:
            self.logger.info("First turn: extracting initial scene from model response")
            
            # Extract <create_scene> from the last assistant message
            initial_scene = None
            format_conversion_success = False  # Flag for whether format conversion succeeded
            
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<create_scene>" in content:
                        initial_scene = self.extract_create_scene_from_response(content)
                        if initial_scene:
                            self.logger.info(f"Extracted initial scene from model response")
                            
                            # ===== Format Conversion Logic =====
                            # Detect and convert scene format
                            try:
                                # Save original model output (for logging)
                                original_format = "grouped" if "groups" in initial_scene else "flat"
                                self.logger.info(f"Model output format: {original_format}")
                                
                                # If flat format, convert to grouped format (for internal processing)
                                if "objects" in initial_scene and "groups" not in initial_scene:
                                    self.logger.info("Converting flat format to grouped format for internal processing")
                                    initial_scene = convert_flat_to_grouped(initial_scene)
                                    format_conversion_success = True
                                    self.logger.info("Format conversion successful")
                                elif "groups" in initial_scene:
                                    # Already in grouped format, no conversion needed
                                    format_conversion_success = True
                                    self.logger.info("Scene already in grouped format")
                                else:
                                    # Unrecognizable format
                                    self.logger.error("Unknown scene format: missing both 'groups' and 'objects' fields")
                                    format_conversion_success = False
                                
                            except Exception as e:
                                self.logger.error(f"Format conversion failed: {e}", exc_info=True)
                                format_conversion_success = False
                            
                            # Store format conversion state in instance (for reward calculation)
                            instance["format_conversion_success"] = format_conversion_success
                            
                            self.logger.info(f"Scene has {len(initial_scene.get('groups', []))} groups")
                            
                            # Update instance's current scene to the converted grouped format
                            instance["current_scene"] = initial_scene
                            
                            # Save model-generated initial scene (grouped format)
                            model_initial_scene_path = instance["output_dir"] / "model_generated_initial_scene.json"
                            with open(model_initial_scene_path, 'w', encoding='utf-8') as f:
                                json.dump(initial_scene, f, indent=2, ensure_ascii=False)
                            self.logger.info(f"Saved model-generated initial scene to: {model_initial_scene_path}")
                            
                            if self.verbose:
                                print(f"✓ Extracted initial scene from model response")
                                print(f"  Format conversion: {'✓' if format_conversion_success else '✗'}")
                                print(f"  Groups: {len(initial_scene.get('groups', []))}")
                            break
                    break  # Only check the last assistant message
            
            if not initial_scene:
                error_msg = "Turn 1: Model did not generate a valid initial scene (missing <create_scene> tag)"
                self.logger.error(error_msg)
                if self.verbose:
                    print(f"Error: {error_msg}")
                # Format conversion failed
                instance["format_conversion_success"] = False
                return True, error_msg, -1.0, {}
            
            # ===== Validate bounds before rendering =====
            # Check if bounds_bottom and bounds_top are valid; terminate conversation if invalid
            room_envelope = initial_scene.get("room_envelope", {})
            bounds_bottom = room_envelope.get("bounds_bottom", initial_scene.get("bounds_bottom", []))
            bounds_top = room_envelope.get("bounds_top", initial_scene.get("bounds_top", []))
            
            # Validate bounds
            room_shape_reward, shape_msg = self._check_room_shape_reward(bounds_bottom, bounds_top)
            invalid_bounds = room_shape_reward < 0
            if invalid_bounds:
                # Room vertex count is out of the allowed range, log warning but do not terminate
                # Set this turn's reward to -1, but continue to the next turn
                self.logger.warning(f"Turn 1: Scene bounds data invalid - {shape_msg}, continuing to next turn")
                if self.verbose:
                    print(f"Warning: Scene bounds data invalid - {shape_msg}, continuing to next turn")
            else:
                self.logger.info(f"Bounds validation passed: {shape_msg}")
            
            # First turn does not execute tool_calls; directly render the initial scene and return
            self.logger.info("First turn: rendering initial scene without tool_calls")
            
            try:
                # Render initial scene (no tool_calls needed)
                from RL_utils import render_scene_quick
                
                # Set output path
                img_output_path = instance["output_dir"] / f"turn_{current_turn:03d}_initial_merged.png"
                
                # Use semaphore to limit concurrent rendering
                render_semaphore = await self._get_render_semaphore()
                async with render_semaphore:
                    img_path_result = await asyncio.to_thread(
                        render_scene_quick,
                        scene_data=initial_scene,
                        output_path=str(img_output_path),
                        return_image=True,
                        verbose=self.verbose,
                        fast_mode=True  # Enable fast rendering mode (512x512, 8 samples)
                    )
                
                # render_scene_quick returns (path, image) when return_image=True
                if isinstance(img_path_result, tuple):
                    img_path, img = img_path_result
                else:
                    img_path = img_path_result
                    img = None
                
                self.logger.info(f"Initial scene rendered: {img_path}")
                
                # If no image object, try to load it
                if img is None and img_path and Path(img_path).exists():
                    try:
                        img = Image.open(img_path)
                        self.logger.info(f"Loaded initial scene image from: {img_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load image: {e}")
                
            except Exception as e:
                error_msg = f"Initial scene rendering failed: {str(e)}"
                self.logger.error(f"Rendering error: {error_msg}", exc_info=True)
                if self.verbose:
                    print(f"Error: {error_msg}")
                return True, error_msg, -1.0, {}
            
            # Calculate initial scene reward
            # If bounds are invalid, set reward to -1 directly
            if invalid_bounds:
                reward = -1.0
                self.logger.info(f"Initial scene reward set to -1.0 due to invalid bounds: {shape_msg}")
            else:
                self.logger.info("Calculating initial scene reward")
                reward = await self.calculate_score(instance_id, messages=messages)
                self.logger.info(f"Initial scene reward: {reward}")
            
            # Store rewards
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # Record history
            history_entry = {
                "turn": current_turn,
                "tool_calls": [],  # First turn has no tool_calls
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
            
            # ===== Return flat format to model =====
            # Convert grouped format back to flat format
            try:
                flat_scene_for_model = convert_grouped_to_flat(initial_scene)
                self.logger.info("Converted scene back to flat format for model output")
            except Exception as e:
                self.logger.error(f"Failed to convert to flat format for output: {e}")
                flat_scene_for_model = initial_scene  # Fall back to original format
            
            # Build response
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
        
        # Second turn and onwards: normal tool_calls processing flow
        self.logger.info("Turn >= 2: processing tool_calls")
        
        # Extract tool_calls (from kwargs or last message)
        tool_calls = kwargs.get("tool_calls", [])
        if not tool_calls and messages:
            # Try to extract tool_calls from the last assistant message
            # tool_calls are wrapped in <tool_calls></tool_calls> tags
            for msg in reversed(messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<tool_calls>" in content and "</tool_calls>" in content:
                        try:
                            # Extract content between <tool_calls>...</tool_calls>
                            import re
                            match = re.search(r'<tool_calls>(.*?)</tool_calls>', content, re.DOTALL)
                            if match:
                                tool_calls_str = match.group(1).strip()
                                # Parse JSON
                                tool_calls = json.loads(tool_calls_str)
                                self.logger.info(f"Extracted tool_calls from assistant message: {len(tool_calls)} calls")
                                break
                        except (json.JSONDecodeError, AttributeError) as e:
                            self.logger.warning(f"Failed to parse tool_calls from message: {e}")
                            self.logger.warning(f"Content: {content[:200]}...")
                    break  # Only check the last assistant message
        
        self.logger.info(f"Tool calls: {json.dumps(tool_calls, indent=2, default=str)}")
        
        if not tool_calls:
            # No tool_calls, return error
            self.logger.warning("No tool_calls provided")
            return False, "Error: No tool_calls provided", -1.0, {}
        
        # Execute scene editing and rendering
        self.logger.info("Starting edit_and_render_scene")
        
        # Track whether tool calls succeeded
        tool_execution_success = True
        tool_execution_error = None
        
        try:
            # Use semaphore to limit concurrent rendering
            render_semaphore = await self._get_render_semaphore()
            async with render_semaphore:
                edited_scene, img_path, img, is_terminated = await asyncio.to_thread(
                    edit_and_render_scene,
                    scene_data=instance["current_scene"],
                    tool_calls=tool_calls,
                    output_dir=instance["output_dir"],
                    scene_id=f"turn_{current_turn:03d}",
                    retrieve_assets=kwargs.get("retrieve_assets", True),
                    use_objaverse=self.use_objaverse,  # Use configured asset source
                    return_image=True,
                    verbose=self.verbose,
                    fast_mode=True  # Enable fast rendering mode (512x512, 8 samples, ~2-3x speedup)
                )
            self.logger.info(f"edit_and_render_scene completed. Image: {img_path}")
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as e:
            # Catch specific expected errors
            tool_execution_success = False
            tool_execution_error = str(e)
            error_msg = f"Scene editing failed: {str(e)}"
            self.logger.error(f"Expected error: {error_msg}")
            if self.verbose:
                print(f"Error: {error_msg}")
            
            # Tool call failed, but still calculate reward (includes tool execution failure penalty)
            # Use current scene (unmodified) to calculate reward
            reward = await self.calculate_score(
                instance_id, 
                messages=messages, 
                tool_execution_success=False
            )
            
            # Store reward
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # Record failed tool call to history
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
            # Unexpected error
            tool_execution_success = False
            tool_execution_error = str(e)
            error_msg = f"Scene editing encountered unexpected error: {str(e)}"
            self.logger.error(f"Unexpected error: {error_msg}", exc_info=True)
            if self.verbose:
                print(f"Unexpected Error: {error_msg}")
                import traceback
                traceback.print_exc()
            
            # Unexpected error, still calculate reward (includes tool execution failure penalty)
            reward = await self.calculate_score(
                instance_id, 
                messages=messages, 
                tool_execution_success=False
            )
            
            # Store rewards
            instance["rewards"].append(reward)
            instance["total_reward"] += reward
            
            # Record failed tool calls to history
            instance["history"].append({
                "turn": current_turn,
                "tool_calls": tool_calls,
                "reward": reward,
                "is_terminated": True,
                "error": error_msg,
                "tool_execution_success": False
            })
            
            return True, error_msg, reward, {}
        
        # Save the scene before editing (for VLM judge evaluation)
        scene_before_edit = instance["current_scene"]
        
        # Update current scene (keep in grouped format)
        instance["current_scene"] = edited_scene
        self.logger.info("Current scene updated")
        
        # Calculate reward, passing tool execution status, termination status, scene before edit, and current image path
        self.logger.info("Calculating reward")
        reward = await self.calculate_score(
            instance_id, 
            messages=messages, 
            tool_execution_success=tool_execution_success,
            is_terminated=is_terminated,
            scene_before_edit=scene_before_edit,  # Pass the scene before editing
            current_img_path=img_path  # Pass current turn's generated image path
        )  
        self.logger.info(f"Reward calculated: {reward}")
        
        # Store rewards and components  
        instance["rewards"].append(reward)    
        instance["total_reward"] += reward
        
        
        # Record history
        instance["history"].append({
            "turn": current_turn,
            "tool_calls": tool_calls,
            "reward": reward,
            "is_terminated": is_terminated,
            "img_path": img_path,
            "tool_execution_success": tool_execution_success
        })
        
        # Determine whether to terminate
        is_done = False
        
        # ===== Return flat format to model =====
        # Convert grouped format back to flat format
        try:
            flat_scene_for_model = convert_grouped_to_flat(edited_scene)
            self.logger.info("Converted edited scene back to flat format for model output")
        except Exception as e:
            self.logger.error(f"Failed to convert to flat format for output: {e}")
            flat_scene_for_model = edited_scene  # Fall back to original format
        
        # Build response: four-part format (with feedback)
        # Part 1: <image> tag
        response_text = "<image>\n"
        # response_text = ""
        
        # Part 2: User requirement (from kwargs or instance state)
        user_requirement = instance.get("user_requirement", "")
        if user_requirement:
            response_text += f"{user_requirement}\n"
        
        # Part 3: Feedback (if available) - after user requirement, before scene
        last_feedback = instance.get("last_feedback", "")
        if last_feedback:
            response_text += f"<feedback>\n{last_feedback}\n</feedback>\n"
            self.logger.info(f"Injected feedback into response: {last_feedback[:100]}...")
        
        # Part 4: <current_scene>json</current_scene> (using flat format)
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
        
        # Fault tolerance: if img is None, try to load from img_path
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
        
        # Prepare metadata
        metadata = {
            "image": [img] if img else [],  # List of PIL Image objects
        }
        
        self.logger.info(f"generate_response completed. is_done={is_done}, reward={reward}")
        self.logger.info("="*80)
        
        return is_done, response_text, reward, metadata
    
    def _calculate_format_reward(self, messages: List[Dict[str, Any]], turn: int, instance_id: str) -> float:
        """
        Calculate format reward
        
        Args:
            messages: Message history
            turn: Current turn number
            instance_id: Instance ID, used to get current scene state and format conversion state
        
        Turn 1:
        - Must contain <create_scene>...</create_scene>, and should not have <think> or <conclusion>
        - JSON format must be correct
        - Must be an empty scene (groups is an empty list or contains empty objects list)
        - Format conversion must succeed (if model outputs flat format, conversion to grouped format must succeed)
        
        Other turns:
        - Must contain <think> and <tool_calls> in order
        - tool_calls JSON format must be correct
        - Tool calls must contain valid tool names and required parameters
        - Operations involving jid must use jid existing in the current scene
        
        Returns:
            Correct format returns 1.0, partial errors return values between 0 and -1, completely wrong or failed format conversion returns -1.0
        """
        import re
        
        # ===== First check if format conversion succeeded =====
        if instance_id in self._instance_dict:
            format_conversion_success = self._instance_dict[instance_id].get("format_conversion_success", True)
            if not format_conversion_success:
                self.logger.warning(f"Turn {turn}: Format conversion failed, returning -1.0")
                return -1.0
        
        # Get the last assistant message
        assistant_msg = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
        
        if not assistant_msg:
            return -1.0
        
        # Define valid tool names and required parameters
        # Note: Tools requiring object IDs can accept jid (3D-FUTURE) or uid (Objaverse)
        # Listed here are required parameters; validation will check if jid or uid exists
        valid_tools = {
            "add_object": ["object_description", "position", "rotation", "size"],
            "remove_object": [],  # jid or uid, validated separately below
            "move_object": ["new_position"],  # jid or uid, validated separately below
            "rotate_object": ["new_rotation"],  # jid or uid, validated separately below
            "scale_object": ["new_size"],  # jid or uid, validated separately below
            "replace_object": ["new_object_description"],  # jid_to_replace or uid_to_replace, validated separately below
            "terminate": ["reason"]
        }
        
        # Tools requiring object ID (supports jid or uid)
        # Format: tool_name -> (jid_param_name, uid_param_name)
        tools_requiring_object_id = {
            "remove_object": ("jid", "uid"),
            "move_object": ("jid", "uid"),
            "rotate_object": ("jid", "uid"),
            "scale_object": ("jid", "uid"),
            "replace_object": ("jid_to_replace", "uid_to_replace")
        }
        
        if turn == 1:
            # ===== Turn 1: Check initial scene creation format =====
            has_create_scene = "<create_scene>" in assistant_msg and "</create_scene>" in assistant_msg
            has_think = "<think>" in assistant_msg
            has_conclusion = "<conclusion>" in assistant_msg
            
            # Check 1: Should not have think or conclusion
            if has_think or has_conclusion:
                self.logger.warning("Turn 1: Found <think> or <conclusion> tags (should not exist)")
                return -1.0
            
            # Check 2: Must have create_scene tags
            if not has_create_scene:
                self.logger.warning("Turn 1: Missing <create_scene> tags")
                return -1.0
            
            # Check 3: Extract and validate JSON format
            try:
                pattern = r'<create_scene>\s*```json\s*(.*?)\s*```\s*</create_scene>'
                match = re.search(pattern, assistant_msg, re.DOTALL)
                if not match:
                    self.logger.warning("Turn 1: JSON code block not found or malformed")
                    return -1.0
                
                scene_json_str = match.group(1).strip()
                scene_data = json.loads(scene_json_str)
                
                # Check 4: Must be an empty scene (if groups field exists)
                groups = scene_data.get("groups", [])
                total_objects = 0
                if groups:
                    for group in groups:
                        if isinstance(group, dict) and "objects" in group and group["objects"]:
                            total_objects += len(group["objects"])
                
                # Also check flat format objects field
                if "objects" in scene_data and scene_data["objects"]:
                    total_objects += len(scene_data["objects"])
                
                if total_objects > 0:
                    self.logger.warning(f"Turn 1: Initial scene should be empty, but has {total_objects} objects")
                    return -1.0
                
                # Check 5: Validate room shape (only check vertex count)
                # Supports two formats: room_envelope.bounds_bottom or direct bounds_bottom
                room_envelope = scene_data.get("room_envelope", {})
                bounds_bottom = room_envelope.get("bounds_bottom", scene_data.get("bounds_bottom", []))
                bounds_top = room_envelope.get("bounds_top", scene_data.get("bounds_top", []))
                
                if not bounds_bottom or not bounds_top:
                    self.logger.warning("Turn 1: Missing bounds_bottom or bounds_top in room_envelope")
                    return -1.0
                
                # Check room shape and return different rewards based on vertex count
                # 4 vertices=1.0, 5 vertices=0.5, 6 vertices=0.0, other=-1.0
                room_shape_reward, shape_msg = self._check_room_shape_reward(bounds_bottom, bounds_top)
                self.logger.info(f"Turn 1: Room shape check - {shape_msg}, reward={room_shape_reward}")
                
                # Check 6: Validate that room_type and room_id match the user requirement
                # Get user requirement from instance
                user_requirement = ""
                if instance_id in self._instance_dict:
                    user_requirement = self._instance_dict[instance_id].get("user_requirement", "")
                
                room_type_reward = self._check_room_type_reward(scene_data, user_requirement)
                self.logger.info(f"Turn 1: Room type check - reward={room_type_reward}")
                
                # Combined reward: room shape reward + room_type reward
                # room_type_reward: -1.0 (both wrong), 0.0 (one wrong), 1.0 (both correct or unable to determine)
                if room_type_reward < 0:
                    # Both wrong, return -1 directly
                    self.logger.warning("Turn 1: Both room_type and room_id mismatch with user requirement")
                    return -1.0
                elif room_type_reward == 0:
                    # One is wrong, return 0
                    self.logger.warning("Turn 1: Either room_type or room_id mismatch with user requirement")
                    return 0.0
                else:
                    # Both correct or unable to determine, then check room area
                    # Only penalize rooms that are too large (>30m²), other cases do not affect format reward
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
                        # Area calculation failure does not affect format reward
                    
                    return room_shape_reward
                
            except json.JSONDecodeError as e:
                self.logger.warning(f"Turn 1: JSON parsing error: {e}")
                return -1.0
            except Exception as e:
                self.logger.warning(f"Turn 1: Unexpected error during format check: {e}")
                return -1.0
                
        else:
            # ===== Other turns: Check editing operation format =====
            has_think = "<think>" in assistant_msg and "</think>" in assistant_msg
            has_tool_calls = "<tool_calls>" in assistant_msg and "</tool_calls>" in assistant_msg
            
            # Check 1: Must contain both think and tool_calls tags
            if not (has_think and has_tool_calls):
                missing = []
                if not has_think: missing.append("think")
                if not has_tool_calls: missing.append("tool_calls")
                self.logger.warning(f"Turn {turn}: Missing tags: {missing}")
                return -1.0
            
            # Check 2: Check order (think should come before tool_calls)
            think_pos = assistant_msg.find("<think>")
            tool_calls_pos = assistant_msg.find("<tool_calls>")
            
            if not (think_pos < tool_calls_pos):
                self.logger.warning(f"Turn {turn}: Wrong tag order (should be think->tool_calls)")
                return -0.8
            
            # Check 3: Extract and validate tool_calls JSON format
            try:
                pattern = r'<tool_calls>\s*(.*?)\s*</tool_calls>'
                match = re.search(pattern, assistant_msg, re.DOTALL)
                if not match:
                    self.logger.warning(f"Turn {turn}: tool_calls content not found")
                    return -1.0
                
                tool_calls_str = match.group(1).strip()
                tool_calls = json.loads(tool_calls_str)
                
                # Check 4: Must be a list
                if not isinstance(tool_calls, list):
                    self.logger.warning(f"Turn {turn}: tool_calls must be a list")
                    return -1.0
                
                # Check 5: Cannot be empty (unless it's terminate)
                if len(tool_calls) == 0:
                    self.logger.warning(f"Turn {turn}: tool_calls list is empty")
                    return -1.0
                
                # Get all object IDs in the current scene (jid and uid)
                # Use the provided instance_id to ensure concurrency safety
                if instance_id not in self._instance_dict:
                    self.logger.warning(f"Turn {turn}: instance_id {instance_id} not found in _instance_dict")
                    return -1.0
                
                current_scene = self._instance_dict[instance_id].get("current_scene", {})
                valid_jids = set()  # 3D-FUTURE object IDs
                valid_uids = set()  # Objaverse object IDs
                valid_object_ids = set()  # All object IDs (jid + uid)
                
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
                
                # Check 6: Validate each tool call
                penalty = 0.0
                for i, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] is not a dict")
                        penalty += 0.1
                        continue
                    
                    # Check required fields
                    if "name" not in tool_call:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] missing 'name' field")
                        penalty += 0.1
                        continue
                    
                    tool_name = tool_call["name"]
                    
                    # Check if tool name is valid
                    if tool_name not in valid_tools:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] has invalid tool name '{tool_name}'")
                        penalty += 0.15
                        continue
                    
                    # Check arguments field
                    if "arguments" not in tool_call:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] missing 'arguments' field")
                        penalty += 0.1
                        continue
                    
                    arguments = tool_call["arguments"]
                    if not isinstance(arguments, dict):
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] arguments is not a dict")
                        penalty += 0.1
                        continue
                    
                    # Check required parameters
                    required_params = valid_tools[tool_name]
                    missing_params = [p for p in required_params if p not in arguments]
                    if missing_params:
                        self.logger.warning(f"Turn {turn}: tool_call[{i}] ({tool_name}) missing required params: {missing_params}")
                        penalty += 0.1
                    
                    # Check tools that require object IDs (supports jid or uid)
                    if tool_name in tools_requiring_object_id:
                        jid_param, uid_param = tools_requiring_object_id[tool_name]
                        has_jid = jid_param in arguments and arguments[jid_param]
                        has_uid = uid_param in arguments and arguments[uid_param]
                        
                        # Must provide either jid or uid
                        if not has_jid and not has_uid:
                            self.logger.warning(f"Turn {turn}: tool_call[{i}] ({tool_name}) missing object ID (need '{jid_param}' or '{uid_param}')")
                            penalty += 0.15
                        else:
                            # Validate that the provided ID exists in the current scene
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
                                penalty += 0.2  # Non-existent ID is a serious error, apply larger penalty
                
                # Calculate final score
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
        Calculate object count reward
        
        Reward rules:
        - Object count >= 15: reward = 1.0
        - Object count 5-15: linear interpolation from 0.0 to +1.0 (5=0, 15=1)
        - Object count 4-5: linear interpolation from -1.0 to 0.0 (4=-1, 5=0)
        - Object count < 4 (final turn only): directly return global -1 total reward
        - Object count < 4 (non-final turn): reward = -1.0
        
        Args:
            scene_data: Scene data
            is_final_turn: Whether this is the final turn
        
        Returns:
            (reward, should_override_total): 
            - reward: Object count reward value
            - should_override_total: If True, total reward should be overridden to -1
        """
        # Count total objects
        total_objects = 0
        if 'groups' in scene_data and scene_data['groups']:
            for group in scene_data['groups']:
                if 'objects' in group and group['objects']:
                    total_objects += len(group['objects'])
        elif 'objects' in scene_data and scene_data['objects']:
            # Compatible with direct objects format
            total_objects = len(scene_data['objects'])
        
        self.logger.info(f"Object count: {total_objects} (is_final_turn={is_final_turn})")

        # Special rule: if final turn and object count < 4, return -1 and mark to override total reward
        if is_final_turn and total_objects < 4:
            self.logger.warning(f"Final turn: Object count ({total_objects}) < 4, overriding total reward to -1")
            return -1.0, True
        
        # Define reward intervals
        if total_objects >= 15:
            # Sufficient objects, full score
            return 1.0, False
        elif 5 <= total_objects < 15:
            # 5-15 objects, linear interpolation from 0.0 to +1.0
            # total_objects=5 -> 0.0, total_objects=15 -> +1.0
            return (total_objects - 5) / 10.0, False
        elif 4 <= total_objects < 5:
            # 4-5 objects, linear interpolation from -1.0 to 0.0
            # total_objects=4 -> -1.0, total_objects=5 -> 0.0
            return -1.0 + (total_objects - 4) / 1.0, False
        else:  # total_objects < 4
            # Non-final turn, too few objects, return -1.0 but do not override total reward
            return -1.0, False
    
    def _get_room_boundaries(self, scene_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract room boundaries from room_envelope
        
        Returns:
            Dictionary containing floor_y, ceiling_y, x_min, x_max, z_min, z_max
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
        Classify object support type (keyword matching first, fallback to LLM)
        
        Args:
            obj_desc: Object description text
        
        Returns:
            Support type: 'floor', 'surface', 'ceiling', 'wall'
        """
        # Check cache
        if obj_desc in self.support_type_cache:
            return self.support_type_cache[obj_desc]
        
        desc_lower = obj_desc.lower()
        
        # English keyword matching
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
        
        # Keyword matching
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
        
        # No keyword matched, call LLM for classification
        try:
            prompt = f"""Classify the support type needed for this furniture/object: "{obj_desc}"

Support types:
- floor: Objects that stand on the floor (chairs, tables, cabinets, etc.)
- surface: Small objects placed on table/shelf surfaces (lamps, vases, books, etc.)
- ceiling: Objects hanging from ceiling (chandeliers, pendant lights, etc.)
- wall: Objects mounted on walls (paintings, mirrors, wall lamps, etc.)

Reply with only one word: floor, surface, ceiling, or wall"""
            
            # Use VLM API for text classification (no image needed)
            # Build text-only request
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
                
                # Extract support type
                for support_type in ['ceiling', 'wall', 'surface', 'floor']:
                    if support_type in content:
                        self.support_type_cache[obj_desc] = support_type
                        self.logger.info(f"LLM classified '{obj_desc}' as '{support_type}'")
                        return support_type
        
        except Exception as e:
            self.logger.warning(f"LLM classification failed for '{obj_desc}': {e}")
        
        # Default fallback to floor
        self.support_type_cache[obj_desc] = 'floor'
        return 'floor'
    
    def _find_support_surfaces(self, all_objects: List[Dict]) -> List[Tuple]:
        """
        Identify potential support surfaces (tables, shelves, etc.)
        
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
            
            # Keyword matching or size heuristic
            is_surface = False
            
            # Keyword check
            for keyword in surface_keywords:
                if keyword in desc:
                    is_surface = True
                    break
            
            # Size heuristic: height < 1.2m and area > 0.2m²
            if not is_surface and size[1] < 1.2 and size[0] * size[2] > 0.2:
                is_surface = True
            
            if is_surface:
                top_y = pos[1] + size[1]
                # Calculate bbox boundaries
                x_min = pos[0] - size[0] / 2
                x_max = pos[0] + size[0] / 2
                z_min = pos[2] - size[2] / 2
                z_max = pos[2] + size[2] / 2
                
                surfaces.append((jid, top_y, (x_min, x_max, z_min, z_max), obj))
        
        return surfaces
    
    async def _calculate_support_reward(self, scene_data: Dict[str, Any]) -> float:
        """
        Calculate support reward - evaluate whether objects have reasonable support
        
        Strategy:
        1. Identify room boundaries (floor, ceiling, walls)
        2. Identify potential support surfaces (tables, shelves, etc.)
        3. Classify support type for each object and verify position
        4. Calculate the ratio of unsupported objects
        
        Returns:
            Returns 1.0 if support is reasonable, negative value if unreasonable
            - 0% unsupported: 1.0
            - 0-10% unsupported: 0 to 1.0 (linear)
            - >10% unsupported: negative reward
        """
        try:
            # Get room boundaries
            boundaries = self._get_room_boundaries(scene_data)
            floor_y = boundaries['floor_y']
            ceiling_y = boundaries['ceiling_y']
            x_min, x_max = boundaries['x_min'], boundaries['x_max']
            z_min, z_max = boundaries['z_min'], boundaries['z_max']
            
            # Extract all objects
            all_objects = []
            for group in scene_data.get('groups', []):
                all_objects.extend(group.get('objects', []))
            
            if not all_objects:
                return 0.0
            
            # Identify support surfaces
            support_surfaces = self._find_support_surfaces(all_objects)
            
            # Verify support for each object
            unsupported_objects = []
            keyword_matched = 0
            llm_called = 0
            
            for obj in all_objects:
                desc = obj.get('desc', '')
                pos = obj.get('pos', [0, 0, 0])
                size = obj.get('size', [0, 0, 0])
                jid = obj.get('jid', '')
                
                # Object bottom and top Y coordinates
                bottom_y = pos[1]
                top_y = pos[1] + size[1]
                
                # Classify support type
                cache_before = len(self.support_type_cache)
                support_type = await self._classify_object_support_type(desc)
                cache_after = len(self.support_type_cache)
                
                if cache_after == cache_before:
                    keyword_matched += 1
                else:
                    # Check if it's a new LLM call
                    if desc not in self.support_type_cache or cache_after > cache_before:
                        llm_called += 1
                
                # Verify support based on type
                is_supported = False
                support_reason = ""
                
                tolerance_y = 0.05  # 5cm tolerance for Y-axis
                tolerance_wall = 0.15  # 15cm tolerance for wall proximity
                
                if support_type == 'floor':
                    # Check if on the floor
                    if abs(bottom_y - floor_y) < tolerance_y:
                        is_supported = True
                        support_reason = "on floor"
                
                elif support_type == 'surface':
                    # Check if on a support surface
                    for surf_jid, surf_top_y, surf_bounds, surf_obj in support_surfaces:
                        if abs(bottom_y - surf_top_y) < tolerance_y:
                            # Check XZ plane overlap
                            obj_x_min = pos[0] - size[0] / 2
                            obj_x_max = pos[0] + size[0] / 2
                            obj_z_min = pos[2] - size[2] / 2
                            obj_z_max = pos[2] + size[2] / 2
                            
                            surf_x_min, surf_x_max, surf_z_min, surf_z_max = surf_bounds
                            
                            # Check overlap
                            x_overlap = not (obj_x_max < surf_x_min or obj_x_min > surf_x_max)
                            z_overlap = not (obj_z_max < surf_z_min or obj_z_min > surf_z_max)
                            
                            if x_overlap and z_overlap:
                                is_supported = True
                                support_reason = f"on surface ({surf_obj.get('desc', 'unknown')[:30]})"
                                break
                
                elif support_type == 'ceiling':
                    # Check if close to ceiling (allow objects hanging below ceiling)
                    # Object top should be within 80%-100% of ceiling height
                    if top_y >= ceiling_y * 0.8 and top_y <= ceiling_y + tolerance_y:
                        is_supported = True
                        support_reason = "hanging from ceiling"
                
                elif support_type == 'wall':
                    # Check if close to a wall
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
            
            # Calculate unsupported object ratio
            total_objects = len(all_objects)
            unsupported_count = len(unsupported_objects)
            unsupported_rate = (unsupported_count / total_objects * 100) if total_objects > 0 else 0
            
            # Calculate reward
            if unsupported_rate == 0:
                reward = 1.0
            elif unsupported_rate <= 10:
                reward = 1.0 - unsupported_rate / 10.0
            else:
                reward = -1.0 * (unsupported_rate - 10) / 90.0
            
            # Print detailed information
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
                    for obj_info in unsupported_objects[:5]:  # Show at most 5
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
        Calculate room area reward (only used for the first turn)
        
        Note: Area reward is only calculated for rectangular rooms; non-rectangular rooms directly return negative reward.
        This is because the bounding box area of irregular rooms overestimates actual area, leading to inaccurate reward signals.
        
        Positive reward when room area is 20-40 sqm:
        - Optimal range: 25-35 sqm, reward = 1.0
        - Boundary range: 20-25 and 35-40, linear decay to 0.5
        - Out of range: negative reward
        
        Returns:
            Returns 0.5-1.0 if area is reasonable, negative value if unreasonable
        """
        try:
            room_envelope = scene_data.get('room_envelope', {})
            bounds_bottom = room_envelope.get('bounds_bottom', [])
            bounds_top = room_envelope.get('bounds_top', [])
            
            if not bounds_bottom or not bounds_top:
                self.logger.warning("Room area calculation: missing bounds")
                return -1.0
            
            # First check if room is rectangular; non-rectangular rooms directly return negative reward
            is_rectangle, rect_error = self._check_room_is_rectangle(bounds_bottom, bounds_top)
            if not is_rectangle:
                self.logger.warning(f"Room area calculation: room is not a rectangle - {rect_error}")
                return -1.0  # Non-rectangular room, area reward is directly negative
            
            # Calculate room dimensions (only accurate for rectangular rooms)
            import numpy as np
            bounds_bottom = np.array(bounds_bottom)
            bounds_top = np.array(bounds_top)
            
            # Get X and Z axis ranges (for rectangles, this gives accurate dimensions)
            x_size = abs(bounds_bottom[:, 0].max() - bounds_bottom[:, 0].min())
            z_size = abs(bounds_bottom[:, 2].max() - bounds_bottom[:, 2].min())
            
            # Calculate area (square meters)
            area = x_size * z_size
            
            self.logger.info(f"Room area: {area:.2f} m² (size: {x_size:.2f}m × {z_size:.2f}m)")
            
            # Define reward intervals (continuous curve, covering full -1 to 1 range)
            # Optimal range: 20-30m², reward=1.0
            # Boundary range: 15-20 and 30-35m², reward linearly from 0.5 to 1.0
            # Transition range: 10-15 and 35-40m², reward linearly from 0 to 0.5
            # Penalty range: <10 or >40m², reward linearly from 0 to -1.0
            if 20 <= area <= 30:
                # Optimal range
                return 1.0
            elif 15 <= area < 20:
                # Lower boundary range, linear interpolation 0.5 -> 1.0
                return 0.5 + 0.5 * (area - 15) / 5
            elif 30 < area <= 35:
                # Upper boundary range, linear interpolation 1.0 -> 0.5
                return 1.0 - 0.5 * (area - 30) / 5
            elif 10 <= area < 15:
                # Lower transition range, linear interpolation 0 -> 0.5
                return 0.5 * (area - 10) / 5
            elif 35 < area <= 40:
                # Upper transition range, linear interpolation 0.5 -> 0
                return 0.5 - 0.5 * (area - 35) / 5
            elif area < 10:
                # Room too small, negative reward (linear interpolation 0 -> -1.0)
                # area=10 -> 0, area=0 -> -1.0
                return max(-1.0, -0.1 * (10 - area))
            else:  # area > 40
                # Room too large, negative reward (linear interpolation 0 -> -1.0)
                # area=40 -> 0, area=50 -> -1.0
                return max(-1.0, -0.1 * (area - 40))
                
        except Exception as e:
            self.logger.warning(f"Room area calculation failed: {e}")
            return -1.0
    
    async def calculate_score(self, instance_id: str, **kwargs) -> float:  
        """
        Calculate comprehensive reward, including:
        1. Physics validity reward (voxel evaluation or trimesh evaluation)
        2. Format reward
        3. Object count reward
        4. Tool execution success reward (only from turn 2 onward)
        5. VLM judge scoring (intermediate turns and final turn)
        
        Normalize to reasonable range at the end
        """  
        instance = self._instance_dict[instance_id]  
        current_scene = instance["current_scene"]  
        turn = instance["turn_count"]
        messages = kwargs.get("messages", [])
        tool_execution_success = kwargs.get("tool_execution_success", None)
        is_terminated = kwargs.get("is_terminated", False)  # New: get terminate status
        scene_before_edit = kwargs.get("scene_before_edit", None)  # Scene before editing
        current_img_path = kwargs.get("current_img_path", None)  # Image path generated this turn
        
        # Determine if this is the final turn (consistent with is_done logic: terminate or reaching max_turns)
        max_turns = instance.get("max_turns", self.max_turns)
        is_final_turn = is_terminated or (turn >= max_turns)
        
        # Get scene and image needed for VLM judge
        vlm_image_path = None
        vlm_prev_image_path = None  # Intermediate turns need previous turn's image for comparison
        vlm_scene = None
        
        # Get image paths from history
        # Note: calculate_score is called before history.append, so history[-1] is the previous turn
        if turn == 1:
            # Turn 1 has no VLM evaluation
            vlm_image_path = None
            vlm_prev_image_path = None
            vlm_scene = None
            self.logger.info("Turn 1: No VLM evaluation")
        elif is_final_turn:
            # Final turn: use current turn's generated image and edited scene
            vlm_image_path = current_img_path
            vlm_prev_image_path = None  # Final turn doesn't need comparison
            vlm_scene = current_scene  # Final turn evaluates edited scene
            self.logger.info(f"Final turn {turn}: Using current image {vlm_image_path} and edited scene")
        else:
            # Intermediate turn: need previous turn's image and current turn's image for comparison
            vlm_image_path = current_img_path  # Image generated this turn
            vlm_scene = scene_before_edit  # Use scene before editing
            
            if turn == 2:
                # Turn 2: previous turn is turn 1's image (initial scene)
                if instance["history"] and len(instance["history"]) >= 1:
                    first_turn_history = instance["history"][0]
                    vlm_prev_image_path = first_turn_history.get("img_path", None)
                    self.logger.info(f"Turn 2: Comparing turn 1 image {vlm_prev_image_path} with current {vlm_image_path}")
            else:
                # Turn 3 and later intermediate turns: use previous turn's image
                if instance["history"] and len(instance["history"]) >= 1:
                    prev_turn_history = instance["history"][-1]
                    vlm_prev_image_path = prev_turn_history.get("img_path", None)
                    self.logger.info(f"Turn {turn}: Comparing previous turn image {vlm_prev_image_path} with current {vlm_image_path}")
        
        # Define weights (adjusted based on whether tool execution exists)
        if turn == 1:
            # Turn 1: initial scene generation, includes format, room area, group count rewards
            weights = {
                "physics": 0.0,
                "format": 1.0,
                "object_count": 0.0,
                "collision_rate": 0.0,
                "oob_rate": 0.0,
                "penetration_depth": 0.0,  # Penetration depth volume reward
                "oob_volume": 0.0,  # Out-of-bounds volume reward
                "support": 0.0,
                "room_area": 0.0,
                "tool_execution": 0.0,
                # VLM judge (not used in turn 1)
                "vlm_problem_identification": 0.0,
                "vlm_action_reasonableness": 0.0,
                "vlm_scene_improvement": 0.0,  # Scene improvement evaluation
                "vlm_key_objects": 0.0,  # Key objects evaluation
                "vlm_rationality": 0.0,
                "vlm_aesthetics": 0.0,
                "vlm_requirement_match": 0.0,
                "vlm_scene_graph": 0.0  # Scene graph constraint evaluation
            }
        else:
            # Use previously computed is_final_turn (already includes both terminate and max_turns cases)
            
            if is_final_turn:
                # ========== Final turn: Layered evaluation system ==========
                # Format layer (0.1) + Object layer (0.3) + Scene layer (0.6)
                # Object layer: key_objects (0.10) + size_proportion (0.10) + object_count (0.10)
                # Scene layer: Physics (0.30) + VLM (0.30)
                #   - Physics: collision + oob + support (trimesh) or voxel physics
                #   - VLM: rationality + requirement_match + scene_graph (removed aesthetics)
                if self.physics_mode == "voxel":
                    weights = {
                        "physics": 0.30,  # Voxel physics evaluation (scene layer physics part)
                        "format": 0.10,  # Format layer
                        # Object layer (0.3)
                        "object_count": 0.0,  # Weight moved to vlm_key_objects
                        "vlm_key_objects": 0.20,  # Key objects match (object layer, includes count weight)
                        "vlm_size_proportion": 0.10,  # Size proportion reasonableness (object layer)
                        # Physics metrics (not used individually in voxel mode)
                        "collision_rate": 0.0,
                        "oob_rate": 0.0,
                        "penetration_depth": 0.0,
                        "oob_volume": 0.0,
                        "support": 0.0,
                        "room_area": 0.0,
                        "tool_execution": 0.0,
                        # VLM scene layer (0.3)
                        "vlm_problem_identification": 0.0,
                        "vlm_action_reasonableness": 0.0,
                        "vlm_scene_improvement": 0.0,
                        "vlm_rationality": 0.10,  # Scene layer VLM
                        "vlm_aesthetics": 0.0,  # Removed
                        "vlm_requirement_match": 0.10,  # Scene layer VLM
                        "vlm_scene_graph": 0.10  # Scene layer VLM
                    }
                else:  # trimesh
                    weights = {
                        "physics": 0.0,
                        "format": 0.10,  # Format layer
                        # Object layer (0.3)
                        "object_count": 0.0,  # Weight moved to vlm_key_objects
                        "vlm_key_objects": 0.20,  # Key objects match (object layer, includes count weight)
                        "vlm_size_proportion": 0.10,  # Size proportion reasonableness (object layer)
                        # Scene layer physics part (0.3)
                        "collision_rate": 0.08,
                        "oob_rate": 0.07,
                        "penetration_depth": 0.05,
                        "oob_volume": 0.05,
                        "support": 0.05,
                        "room_area": 0.0,
                        "tool_execution": 0.0,
                        # VLM scene layer (0.3)
                        "vlm_problem_identification": 0.0,
                        "vlm_action_reasonableness": 0.0,
                        "vlm_scene_improvement": 0.0,
                        "vlm_rationality": 0.10,  # Scene layer VLM
                        "vlm_aesthetics": 0.0,  # Removed
                        "vlm_requirement_match": 0.10,  # Scene layer VLM
                        "vlm_scene_graph": 0.10  # Scene layer VLM
                    }
            else:
                # Intermediate turns: evaluate format + scene improvement + collision rate/OOB rate + volume rewards + key objects
                # Weight total: 0.10 + 0.1*4 + 0.25*2 = 0.10 + 0.40 + 0.50 = 1.0
                weights = {
                    "physics": 0.0,  # Not evaluated
                    "format": 0.10,  # Format correctness
                    "object_count": 0.0,  # Not evaluated
                    "collision_rate": 0.1,  # Collision rate (enabled for intermediate turns)
                    "oob_rate": 0.1,  # OOB rate (enabled for intermediate turns)
                    "penetration_depth": 0.1,  # Penetration depth volume reward (enabled for intermediate turns)
                    "oob_volume": 0.1,  # OOB volume reward (enabled for intermediate turns)
                    "support": 0.0,
                    "room_area": 0.0,
                    "tool_execution": 0.0,
                    # VLM judge intermediate scoring
                    "vlm_problem_identification": 0.0,  # Deprecated
                    "vlm_action_reasonableness": 0.0,  # Deprecated
                    "vlm_scene_improvement": 0.25,  # Scene improvement evaluation (half of VLM evaluation)
                    "vlm_key_objects": 0.25,  # Key objects evaluation (half of VLM evaluation, 0.25 of total weight)
                    "vlm_rationality": 0.0,
                    "vlm_aesthetics": 0.0,
                    "vlm_requirement_match": 0.0,
                    "vlm_scene_graph": 0.0  # Scene graph constraint evaluation (not used for intermediate turns)
                }
        
        rewards = {}
        
        # 1. Calculate physics validity reward (only in final turn and voxel mode)
        
        # First count objects for physics evaluation pre-check
        total_objects_for_physics = 0
        if 'groups' in current_scene and current_scene['groups']:
            for group in current_scene['groups']:
                if 'objects' in group and group['objects']:
                    total_objects_for_physics += len(group['objects'])
        elif 'objects' in current_scene and current_scene['objects']:
            total_objects_for_physics = len(current_scene['objects'])
        
        if self.physics_mode == "voxel" and self.voxel_reward is not None and is_final_turn:
            # Fewer than 2 objects, directly return -1 (no need for evaluation)
            if total_objects_for_physics < 2:
                self.logger.warning(f"Object count ({total_objects_for_physics}) < 2, skipping voxel physics evaluation")
                rewards["physics"] = -1.0
            else:
                # Only compute voxel physics evaluation in the final turn
                try:  
                    # Use VoxelReward to compute
                    physics_reward, metrics = await asyncio.to_thread(  
                        self.voxel_reward.compute_reward,  
                        current_scene,  
                        format_type='ours'  
                    )  
                    
                    if isinstance(physics_reward, (list, tuple)):  
                        physics_reward = float(physics_reward[0]) if physics_reward else 0.0  
                    physics_reward = float(physics_reward)
                    rewards["physics"] = physics_reward
                    
                    # Save physics evaluation results  
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
        
        # 2. Calculate format reward
        try:
            format_reward = self._calculate_format_reward(messages, turn, instance_id)
            rewards["format"] = format_reward
        except Exception as e:
            if self.verbose:
                print(f"Warning: Format reward calculation failed: {e}")
            rewards["format"] = -1.0
        
        # 3. Calculate object count reward (only in final turn)
        # Flag whether to override total reward to -1 (when final turn object count < 5)
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
            # Intermediate turns don't calculate object count
            rewards["object_count"] = 0.0
        
        # 4. Calculate collision rate, OOB rate, and volume rewards (intermediate and final turns in trimesh mode)
        # Volume reward threshold design:
        # - Penetration depth: 0m=+1.0, 0.05m(5cm)=0.0, 0.2m(20cm)=-1.0
        # - OOB volume: 0m³=+1.0, 0.1m³=0.0, 0.5m³=-1.0
        if self.physics_mode == "trimesh" and self.trimesh_metrics is not None and turn > 1:
            # Fewer than 3 objects, directly return -1 (no need for evaluation)
            if total_objects_for_physics < 3:
                self.logger.warning(f"Object count ({total_objects_for_physics}) < 3, skipping trimesh physics evaluation")
                rewards["collision_rate"] = -1.0
                rewards["oob_rate"] = -1.0
                rewards["penetration_depth"] = -1.0
                rewards["oob_volume"] = -1.0
                instance["_trimesh_metrics_for_feedback"] = None  # Ensure set to None
            else:
                try:
                    trimesh_reward, trimesh_metrics = await asyncio.to_thread(
                        self.trimesh_metrics.compute_reward,
                        current_scene,
                        format_type='ours'
                    )
                    
                    # Extract individual metrics
                    collision_rate = trimesh_metrics['collision_rate']
                    oob_rate = trimesh_metrics['out_of_bounds_rate']
                    total_penetration_depth = trimesh_metrics.get('total_penetration_depth', 0.0)
                    total_oob_volume = trimesh_metrics.get('total_oob_volume', 0.0)
                    
                    # ===== Collision rate reward (adjusted based on SFT baseline 45%) =====
                    # Threshold design:
                    # - Collision rate ≤ 20%: +1.0 to +0.5 (excellent)
                    # - Collision rate 20%-45%: +0.5 to 0.0 (SFT baseline as zero point)
                    # - Collision rate > 45%: 0.0 to -1.0 (worse than SFT)
                    if collision_rate <= 20:
                        # Excellent range: 0% -> +1.0, 20% -> +0.5
                        collision_reward = 1.0 - 0.5 * (collision_rate / 20.0)
                    elif collision_rate <= 45:
                        # Good range (SFT baseline as zero point): 20% -> +0.5, 45% -> 0.0
                        collision_reward = 0.5 - 0.5 * (collision_rate - 20) / 25.0
                    else:
                        # Worse than SFT: 45% -> 0.0, 100% -> -1.0
                        collision_reward = -1.0 * (collision_rate - 45) / 55.0
                    
                    # ===== OOB rate reward (adjusted based on SFT baseline 30%) =====
                    # Threshold design:
                    # - OOB rate ≤ 10%: +1.0 to +0.5 (excellent)
                    # - OOB rate 10%-30%: +0.5 to 0.0 (SFT baseline as zero point)
                    # - OOB rate > 30%: 0.0 to -1.0 (worse than SFT)
                    if oob_rate <= 10:
                        # Excellent range: 0% -> +1.0, 10% -> +0.5
                        oob_reward = 1.0 - 0.5 * (oob_rate / 10.0)
                    elif oob_rate <= 30:
                        # Good range (SFT baseline as zero point): 10% -> +0.5, 30% -> 0.0
                        oob_reward = 0.5 - 0.5 * (oob_rate - 10) / 20.0
                    else:
                        # Worse than SFT: 30% -> 0.0, 100% -> -1.0
                        oob_reward = -1.0 * (oob_rate - 30) / 70.0
                    
                    # ===== Penetration depth volume reward (normalized) =====
                    # Threshold design (considering cumulative values, total penetration depth of multiple collision pairs):
                    # 0m=+1.0, 0.1m=+0.5, 0.3m=0.0, 0.6m=-0.5, 1.0m=-1.0
                    # Scene reference: 5 collision pairs each 2cm=0.1m, each 6cm=0.3m
                    if total_penetration_depth == 0:
                        penetration_reward = 1.0
                    elif total_penetration_depth <= 0.1:  # Within 10cm (minor collision)
                        # Linear from 1.0 down to 0.5
                        penetration_reward = 1.0 - 0.5 * (total_penetration_depth / 0.1)
                    elif total_penetration_depth <= 0.3:  # 10cm to 30cm (moderate collision)
                        # Linear from 0.5 down to 0.0
                        penetration_reward = 0.5 - 0.5 * (total_penetration_depth - 0.1) / 0.2
                    elif total_penetration_depth <= 0.6:  # 30cm to 60cm (fairly severe collision)
                        # Linear from 0.0 down to -0.5
                        penetration_reward = -0.5 * (total_penetration_depth - 0.3) / 0.3
                    elif total_penetration_depth <= 1.0:  # 60cm to 1m (severe collision)
                        # Linear from -0.5 down to -1.0
                        penetration_reward = -0.5 - 0.5 * (total_penetration_depth - 0.6) / 0.4
                    else:  # Over 1m
                        penetration_reward = -1.0
                    
                    # ===== OOB volume reward (normalized) =====
                    # Threshold design (considering cumulative values, total OOB volume of multiple objects):
                    # 0m³=+1.0, 0.2m³=+0.5, 0.5m³=0.0, 1.0m³=-0.5, 2.0m³=-1.0
                    # Scene reference: a chair volume ≈ 0.1m³, a sofa ≈ 0.5m³
                    if total_oob_volume == 0:
                        oob_volume_reward = 1.0
                    elif total_oob_volume <= 0.2:  # Within 0.2 cubic meters (minor OOB)
                        # Linear from 1.0 down to 0.5
                        oob_volume_reward = 1.0 - 0.5 * (total_oob_volume / 0.2)
                    elif total_oob_volume <= 0.5:  # 0.2 to 0.5 cubic meters (moderate OOB)
                        # Linear from 0.5 down to 0.0
                        oob_volume_reward = 0.5 - 0.5 * (total_oob_volume - 0.2) / 0.3
                    elif total_oob_volume <= 1.0:  # 0.5 to 1 cubic meter (fairly severe OOB)
                        # Linear from 0.0 down to -0.5
                        oob_volume_reward = -0.5 * (total_oob_volume - 0.5) / 0.5
                    elif total_oob_volume <= 2.0:  # 1 to 2 cubic meters (severe OOB)
                        # Linear from -0.5 down to -1.0
                        oob_volume_reward = -0.5 - 0.5 * (total_oob_volume - 1.0) / 1.0
                    else:  # Over 2 cubic meters
                        oob_volume_reward = -1.0
                    
                    rewards["collision_rate"] = collision_reward
                    rewards["oob_rate"] = oob_reward
                    rewards["penetration_depth"] = penetration_reward
                    rewards["oob_volume"] = oob_volume_reward
                    
                    # Save trimesh physics evaluation results
                    trimesh_metrics_path = instance["output_dir"] / f"turn_{turn:03d}_trimesh_metrics.json"
                    with open(trimesh_metrics_path, 'w', encoding='utf-8') as f:
                        json.dump(trimesh_metrics, f, indent=2)
                    
                    # Store trimesh_metrics for subsequent feedback generation
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
        
        # 4.5. Calculate support reward (only in final turn and trimesh mode)
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
        
        # 5. Room shape reward merged into format reward (turn 1 format reward includes shape reward)
        rewards["room_shape"] = 0.0
        
        # Room area reward disabled
        rewards["room_area"] = 0.0
        
        # 6. Calculate tool execution success reward (only from turn 2 onward)
        if turn > 1:
            if tool_execution_success is True:
                rewards["tool_execution"] = 1.0
            elif tool_execution_success is False:
                rewards["tool_execution"] = -1.0
            else:
                # If status not provided, default to success
                rewards["tool_execution"] = 1.0
        else:
            # Turn 1 has no tool execution
            rewards["tool_execution"] = 0.0
        
        # 8. VLM Judge scoring
        # Initialize all VLM scores to 0
        rewards["vlm_problem_identification"] = 0.0
        rewards["vlm_action_reasonableness"] = 0.0
        rewards["vlm_scene_improvement"] = 0.0  # Intermediate turn scene improvement evaluation
        rewards["vlm_key_objects"] = 0.0  # Object layer: key objects match
        rewards["vlm_size_proportion"] = 0.0  # Object layer: size proportion reasonableness
        rewards["vlm_rationality"] = 0.0
        rewards["vlm_aesthetics"] = 0.0  # Removed, kept at 0
        rewards["vlm_requirement_match"] = 0.0
        rewards["vlm_scene_graph"] = 0.0  # Scene graph constraint evaluation
        
        # Fuse flag: when object layer fails, scene layer is directly set to -1
        object_level_fused = False
        
        # VLM Judge evaluation
        if turn > 1 and self.vlm_judge_enabled and vlm_image_path and Path(vlm_image_path).exists():
            user_requirement = instance.get("user_requirement", "")
            
            if is_final_turn:
                # ========== Final turn: layered evaluation ==========
                self.logger.info("Final turn: Layered evaluation (Object Level -> Scene Level)")
                self.logger.info(f"  Image: {vlm_image_path}")
                self.logger.info(f"  Scene: edited scene with {len(vlm_scene.get('groups', []))} groups")
                
                # ----- Object layer evaluation (0.3) -----
                self.logger.info("=== Object Level Evaluation ===")
                
                # 1. Key objects match evaluation
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
                
                # 2. Size proportion reasonableness evaluation
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
                
                # 3. Object count evaluation (already calculated above)
                # rewards["object_count"] already calculated
                
                # If object count is too low (<4), force key objects score to -1 to ensure object layer fuses
                if should_override_total_reward:
                    rewards["vlm_key_objects"] = -1.0
                    self.logger.warning("Object count < 4, forcing vlm_key_objects to -1.0")
                
                # ----- Object layer fuse judgment -----
                # Object layer weighted score = key_objects * 0.20 + size_proportion * 0.10 + object_count * 0.0
                object_level_score = (
                    rewards["vlm_key_objects"] * weights.get("vlm_key_objects", 0.20) +
                    rewards["vlm_size_proportion"] * weights.get("vlm_size_proportion", 0.10) +
                    rewards["object_count"] * weights.get("object_count", 0.0)
                )
                self.logger.info(f"Object level weighted score: {object_level_score:.4f}")
                
                if object_level_score < 0:
                    # Object layer failed, fuse! Scene layer directly set to -1
                    object_level_fused = True
                    self.logger.warning(f"Object level FUSED: score={object_level_score:.4f} < 0, scene level will be set to -1")
                    rewards["vlm_rationality"] = -1.0
                    rewards["vlm_requirement_match"] = -1.0
                    rewards["vlm_scene_graph"] = -1.0
                    # Physics metrics also set to -1
                    if self.physics_mode == "trimesh":
                        rewards["collision_rate"] = -1.0
                        rewards["oob_rate"] = -1.0
                        rewards["penetration_depth"] = -1.0
                        rewards["oob_volume"] = -1.0
                        rewards["support"] = -1.0
                    elif self.physics_mode == "voxel":
                        rewards["physics"] = -1.0
                else:
                    # ----- Scene layer evaluation (0.6) -----
                    self.logger.info("=== Scene Level Evaluation ===")
                    
                    # Scene layer VLM evaluation (merged into a single call)
                    try:
                        vlm_scores = await self._judge_final_turn(
                            image_path=vlm_image_path,
                            user_requirement=user_requirement,
                            current_scene=vlm_scene
                        )
                        rewards["vlm_rationality"] = float(vlm_scores.get("rationality", 0))
                        rewards["vlm_requirement_match"] = float(vlm_scores.get("requirement_match", 0))
                        rewards["vlm_scene_graph"] = float(vlm_scores.get("scene_graph", 0))
                        
                        # Save VLM scoring results
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
                # ========== Intermediate turns: scene improvement + object relevance evaluation ==========
                self.logger.info("Intermediate turn: calling VLM judge for scene improvement comparison")
                self.logger.info(f"  Previous image: {vlm_prev_image_path}")
                self.logger.info(f"  Current image: {vlm_image_path}")
                
                # Check if both images exist
                if vlm_prev_image_path and Path(vlm_prev_image_path).exists():
                    try:
                        # Extract previous and current turn object summaries for VLM reference
                        prev_scene_summary = ""
                        current_scene_summary = ""
                        
                        # scene_before_edit is the scene before editing (previous turn state)
                        if scene_before_edit:
                            try:
                                prev_scene_summary = self._extract_objects_summary(scene_before_edit)
                            except Exception as e:
                                self.logger.warning(f"Failed to extract prev scene summary: {e}")
                        
                        # current_scene is the scene after editing (current turn state)
                        if current_scene:
                            try:
                                current_scene_summary = self._extract_objects_summary(current_scene)
                            except Exception as e:
                                self.logger.warning(f"Failed to extract current scene summary: {e}")
                        
                        vlm_scores = await self._judge_intermediate_turn(
                            prev_image_path=vlm_prev_image_path,  # Previous turn's image
                            current_image_path=vlm_image_path,  # Current turn's image
                            user_requirement=user_requirement,
                            prev_scene_summary=prev_scene_summary,  # Previous turn object summary
                            current_scene_summary=current_scene_summary  # Current turn object summary
                        )
                        rewards["vlm_scene_improvement"] = float(vlm_scores.get("scene_improvement", 0))
                        
                        # ===== Object relevance evaluation (select strategy based on tool call type) =====
                        # Extract current turn's tool calls
                        current_tool_calls = self._extract_tool_calls_from_messages(messages)
                        
                        # Check if there are add_object or replace_object operations
                        has_add_or_replace = any(
                            tc.get("name") in ["add_object", "replace_object"]
                            for tc in current_tool_calls
                        )
                        
                        if has_add_or_replace:
                            # Has add/replace operations: evaluate whether new objects are relevant to scene requirements
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
                            # No add/replace operations: evaluate whether current scene contains key objects
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
                        
                        # Save VLM scoring results
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
        
        # Calculate weighted total reward
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
            weights.get("vlm_scene_improvement", 0.0) * rewards["vlm_scene_improvement"] +  # Scene improvement
            weights.get("vlm_key_objects", 0.0) * rewards["vlm_key_objects"] +  # Object layer: key objects
            weights.get("vlm_size_proportion", 0.0) * rewards["vlm_size_proportion"] +  # Object layer: size proportion
            weights["vlm_rationality"] * rewards["vlm_rationality"] +
            weights["vlm_aesthetics"] * rewards["vlm_aesthetics"] +  # Deprecated, kept at 0
            weights["vlm_requirement_match"] * rewards["vlm_requirement_match"] +
            weights["vlm_scene_graph"] * rewards["vlm_scene_graph"]  # Scene graph constraint evaluation
        )
        
        # Special rule: if object count < 4 in final turn, override total reward to -1
        if should_override_total_reward:
            self.logger.warning(f"Overriding total reward from {total_reward:.4f} to -1.0 due to object count < 4")
            print(f"WARNING: Object count < 4 in final turn, overriding total reward to -1.0", file=sys.stderr, flush=True)
            total_reward = -1.0
        
        # Debug output
        print(f"DEBUG [Interaction Turn {turn}] (physics_mode={self.physics_mode}, is_final={is_final_turn}, fused={object_level_fused}): Reward breakdown:",   
            file=sys.stderr, flush=True)
        
        # Show non-zero weight evaluation metrics
        if turn == 1:
            print(f"  Format: {rewards['format']:.4f} (weight: {weights['format']})",
                file=sys.stderr, flush=True)
            print(f"  Room Area: {rewards['room_area']:.4f} (weight: {weights['room_area']})",
                file=sys.stderr, flush=True)
        elif is_final_turn:
            # Final turn: layered display
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
            # Intermediate turns: show format, collision rate, OOB rate, volume rewards and scene improvement scores
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
        
        # Store detailed reward components to instance state  
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
            
            # VLM judge reward components
            
            if is_final_turn:
                reward_component[f"turn_{turn}_vlm_key_objects"] = rewards["vlm_key_objects"]  # Object Layer: key objects
                reward_component[f"turn_{turn}_vlm_size_proportion"] = rewards["vlm_size_proportion"]  # Object Layer: object size
                reward_component[f"turn_{turn}_vlm_rationality"] = rewards["vlm_rationality"]  # Scene Layer: rationality
                reward_component[f"turn_{turn}_vlm_requirement_match"] = rewards["vlm_requirement_match"]  # Scene Layer: requirement match
                reward_component[f"turn_{turn}_vlm_scene_graph"] = rewards["vlm_scene_graph"]  # Scene Layer: scene graph constraint
            else:
                reward_component[f"turn_{turn}_vlm_scene_improvement"] = rewards["vlm_scene_improvement"]  # Scene improvement evaluation
                reward_component[f"turn_{turn}_vlm_key_objects"] = rewards["vlm_key_objects"]  # Key objects evaluation
        
        instance.setdefault("reward_components", []).append(reward_component)
        
        # ===== Generate feedback for next turn =====
        # Only generate feedback in intermediate turns (final turn doesn't need feedback)
        # Can be controlled by config switch to enable/disable feedback injection
        if turn > 1 and not is_final_turn and self.feedback_injection_enabled:
            feedback_parts = []
            
            # 1. Physics feedback (from trimesh) - controlled by physics_feedback_enabled
            if self.physics_feedback_enabled:
                trimesh_metrics_for_feedback = instance.get("_trimesh_metrics_for_feedback")
                if trimesh_metrics_for_feedback:
                    physics_feedback = generate_physics_feedback(trimesh_metrics_for_feedback, top_k=3)
                    if physics_feedback:
                        feedback_parts.append(physics_feedback)
                    self.logger.info(f"Physics feedback: {physics_feedback}")
            else:
                self.logger.info("Physics feedback disabled by config")
            
            # 2. VLM layout feedback (only generated when image is available) - controlled by layout_feedback_enabled
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
            
            # Combine feedback
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
        """End interaction instance, clean up resources, and return reward information"""
        
        self.logger.info("="*80)
        self.logger.info(f"finalize_interaction called for instance {instance_id}")
        
        if instance_id not in self._instance_dict:
            self.logger.warning(f"Instance {instance_id} not found in finalize")
            if self.verbose:
                print(f"Warning: Instance {instance_id} not found in finalize")
            return {"reward_scores": {}}
        
        instance = self._instance_dict[instance_id]
        
        # Save interaction summary
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
        
        # Merge all turn reward components into one dictionary
        reward_scores = {}
        for components in instance.get("reward_components", []):
            reward_scores.update(components)
        final_scene = instance["current_scene"]
        
        self.logger.info(f"Reward scores: {json.dumps(reward_scores, indent=2)}")
        self.logger.info(f"Final scene has {len(final_scene.get('objects', []))} objects")
        
        # Clean up instance
        # Delete instance output directory and its contents

        del self._instance_dict[instance_id]
        
        self.logger.info(f"Cleaned up instance {instance_id}")
        self.logger.info("finalize_interaction completed")
        self.logger.info("="*80)
        