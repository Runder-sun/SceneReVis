#!/usr/bin/env python3
"""
PathConfig - Unified Path Configuration Manager

Centralized path management for the llmscene project, supporting three initialization methods:
1. Read from a YAML configuration file
2. Read from environment variables
3. Use default hardcoded paths (fallback chain)

Priority: Config file (YAML paths block) > Environment variables > Default hardcoded paths

Usage:
    # Method 1: Initialize from a config dictionary (for RL training)
    from path_config import PathConfig
    PathConfig.init_from_config(config_dict)
    
    # Method 2: Initialize from a YAML file
    PathConfig.init_from_yaml("/path/to/config.yaml")
    
    # Method 3: Auto-initialize (from environment variables and defaults)
    paths = PathConfig.get_instance()
    
    # Get paths
    glb_cache = paths.objaverse_glb_cache_dir
    blender_exe = paths.blender_executable
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List


class PathConfig:
    """
    Unified path configuration manager (singleton pattern)
    
    Attributes:
        objaverse_base_dir: Objaverse assets base directory
        objaverse_glb_cache_dir: Objaverse GLB cache directory
        objaverse_assets_version: Objaverse assets version (default 2023_09_23)
        
        future3d_models_dir: 3D-FUTURE model directory
        future3d_metadata_json: 3D-FUTURE metadata JSON path
        future3d_metadata_scaled_json: 3D-FUTURE scaled metadata JSON path
        future3d_embeddings_pkl: 3D-FUTURE embeddings pickle path
        
        blender_executable: Blender executable path
        blender_candidates: Blender candidate paths list
        
        logs_dir: Logs directory
        output_dir: Output directory
    """
    
    _instance: Optional['PathConfig'] = None
    _lock = threading.Lock()
    
    # ========== Environment Detection ==========
    @staticmethod
    def _detect_environment() -> str:
        """
        Detect the current runtime environment
        
        Returns:
            'azure': Azure cloud environment (/path/to/storage exists)
            'local': Local development environment
        """
        # Check for Azure characteristic path
        if os.path.exists('/path/to/storage'):
            return 'azure'
        # Check AMLT environment variables
        if os.environ.get('AMLT_JOB_ID') or os.environ.get('AZUREML_RUN_ID'):
            return 'azure'
        return 'local'
    
    # ========== Default Path Definitions (dynamically selected based on environment) ==========
    @classmethod
    def _get_objaverse_candidates(cls) -> List[str]:
        """Get Objaverse path candidate list, with priority adjusted by environment"""
        env = cls._detect_environment()
        
        azure_paths = [
            "/path/to/datasets/objathor-assets",
        ]
        
        local_paths = [
            "/path/to/data/datasets/objathor-assets",
            "/path/to/datasets/objathor-assets",
            os.path.expanduser("~/.objathor-assets"),
            os.path.expanduser("~/.objaverse"),
        ]
        
        if env == 'azure':
            return azure_paths + local_paths
        else:
            return local_paths + azure_paths
    
    @classmethod
    def _get_3dfuture_candidates(cls) -> List[str]:
        """Get 3D-FUTURE path candidate list, with priority adjusted by environment"""
        env = cls._detect_environment()
        
        azure_paths = [
            "/path/to/datasets/3d-front/3D-FUTURE-model",
        ]
        
        local_paths = [
            "/path/to/datasets/3d-front/3D-FUTURE-model",
            "/path/to/data/datasets/3d-front/3D-FUTURE-model",
            os.path.expanduser("~/datasets/3D-FUTURE-model"),
        ]
        
        if env == 'azure':
            return azure_paths + local_paths
        else:
            return local_paths + azure_paths
    
    @classmethod
    def _get_logs_candidates(cls) -> List[str]:
        """Get log path candidate list, with priority adjusted by environment"""
        env = cls._detect_environment()
        
        azure_paths = [
            "/path/to/logs",
        ]
        
        local_paths = [
            "./logs/interaction_logs",
            os.path.expanduser("~/llmscene_logs"),
        ]
        
        if env == 'azure':
            return azure_paths + local_paths
        else:
            return local_paths + azure_paths
    
    # Blender paths (typically not affected by environment)
    _BLENDER_CANDIDATES = [
        os.path.expanduser("~/.local/bin/blender"),
        os.path.expanduser("~/.local/blender/blender-4.0.2-linux-x64/blender"),
        os.path.expanduser("~/.local/blender/blender-4.0.2/blender"),
        os.path.expanduser("~/.local/blender/blender-3.6.0/blender"),
        "/usr/bin/blender",
        "/snap/bin/blender",
        "/usr/local/bin/blender",
        "/opt/blender/blender",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize path configuration
        
        Args:
            config: Configuration dictionary, may contain a 'paths' key
        """
        self._config = config or {}
        paths_config = self._config.get('paths', {})
        
        # Detect runtime environment
        self._detected_env = self._detect_environment()
        
        # ========== Objaverse Paths ==========
        objaverse_config = paths_config.get('objaverse', {})
        
        # Base directory - use dynamic candidate list
        self.objaverse_base_dir = self._resolve_path(
            config_value=objaverse_config.get('base_dir'),
            env_var='OBJATHOR_ASSETS_BASE_DIR',
            candidates=self._get_objaverse_candidates(),
            create_if_missing=False
        )
        
        # GLB cache directory
        glb_cache_config = objaverse_config.get('glb_cache_dir')
        if glb_cache_config:
            self.objaverse_glb_cache_dir = glb_cache_config
        elif os.environ.get('OBJAVERSE_GLB_CACHE_DIR'):
            self.objaverse_glb_cache_dir = os.environ['OBJAVERSE_GLB_CACHE_DIR']
        elif self.objaverse_base_dir:
            self.objaverse_glb_cache_dir = os.path.join(self.objaverse_base_dir, 'glbs')
        else:
            self.objaverse_glb_cache_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
        
        # Assets version
        self.objaverse_assets_version = (
            objaverse_config.get('assets_version') or
            os.environ.get('ASSETS_VERSION', '2023_09_23')
        )
        
        # Holodeck version
        self.holodeck_base_version = (
            objaverse_config.get('holodeck_version') or
            os.environ.get('HD_BASE_VERSION', '2023_09_23')
        )
        
        # ========== 3D-FUTURE Paths ==========
        future3d_config = paths_config.get('3d_future', {})
        
        # Model directory - use dynamic candidate list
        self.future3d_models_dir = self._resolve_path(
            config_value=future3d_config.get('models_dir'),
            env_var='PTH_3DFUTURE_ASSETS',
            candidates=self._get_3dfuture_candidates(),
            create_if_missing=False
        )
        
        # Metadata file paths
        self.future3d_metadata_json = (
            future3d_config.get('metadata_json') or
            os.environ.get('PTH_ASSETS_METADATA') or
            './metadata/model_info_3dfuture_assets.json'
        )
        
        self.future3d_metadata_scaled_json = (
            future3d_config.get('metadata_scaled_json') or
            os.environ.get('PTH_ASSETS_METADATA_SCALED') or
            './metadata/model_info_3dfuture_assets_scaled.json'
        )
        
        self.future3d_embeddings_pkl = (
            future3d_config.get('embeddings_pkl') or
            os.environ.get('PTH_ASSETS_EMBED') or
            './metadata/model_info_3dfuture_assets_embeds.pickle'
        )
        
        # ========== Blender Paths ==========
        blender_config = paths_config.get('blender', {})
        
        # Blender executable
        self.blender_executable = self._resolve_blender_executable(
            config_value=blender_config.get('executable'),
            additional_candidates=blender_config.get('candidates', [])
        )
        
        # Save candidate paths for later use
        self.blender_candidates = (
            blender_config.get('candidates', []) + self._BLENDER_CANDIDATES
        )
        
        # ========== Logs and Output Paths ==========
        # Use dynamic candidate list
        self.logs_dir = self._resolve_path(
            config_value=paths_config.get('logs_dir'),
            env_var='LLMSCENE_LOGS_DIR',
            candidates=self._get_logs_candidates(),
            create_if_missing=True
        )
        
        self.output_dir = (
            paths_config.get('output_dir') or
            os.environ.get('LLMSCENE_OUTPUT_DIR') or
            './scene_editing_output'
        )
    
    def _resolve_path(
        self,
        config_value: Optional[str],
        env_var: str,
        candidates: List[str],
        create_if_missing: bool = False
    ) -> Optional[str]:
        """
        Resolve a path by priority: config value > environment variable > candidate path chain
        
        Args:
            config_value: Value from the configuration file
            env_var: Environment variable name
            candidates: List of candidate paths
            create_if_missing: Whether to create the path if it does not exist
            
        Returns:
            Resolved path, or None if none exist
        """
        # 1. Config file value
        if config_value:
            if os.path.exists(config_value) or create_if_missing:
                if create_if_missing and not os.path.exists(config_value):
                    Path(config_value).mkdir(parents=True, exist_ok=True)
                return config_value
        
        # 2. Environment variable
        env_value = os.environ.get(env_var)
        if env_value and os.path.exists(env_value):
            return env_value
        
        # 3. Candidate path chain
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        # 4. If creation is allowed, use the first candidate path
        if create_if_missing and candidates:
            first_candidate = candidates[0]
            Path(first_candidate).mkdir(parents=True, exist_ok=True)
            return first_candidate
        
        return None
    
    def _resolve_blender_executable(
        self,
        config_value: Optional[str],
        additional_candidates: List[str]
    ) -> Optional[str]:
        """
        Resolve the Blender executable path
        
        Args:
            config_value: Value from the configuration file
            additional_candidates: Additional candidate paths
            
        Returns:
            Blender executable path, or None if none exist
        """
        # 1. Config file value
        if config_value and os.path.exists(config_value):
            return config_value
        
        # 2. Environment variable
        env_value = os.environ.get('BLENDER_EXECUTABLE')
        if env_value and os.path.exists(env_value):
            return env_value
        
        # 3. which blender
        try:
            from shutil import which
            detected = which('blender')
            if detected:
                return detected
        except Exception:
            pass
        
        # 4. Candidate path chain
        all_candidates = additional_candidates + self._BLENDER_CANDIDATES
        for candidate in all_candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None
    
    # ========== Convenience Methods ==========
    
    def get_objaverse_paths(self) -> Dict[str, str]:
        """
        Get all Objaverse-related paths (compatible with _get_objathor_paths in objaverse_retriever.py)
        
        Returns:
            Dictionary containing all Objaverse paths
        """
        base_dir = self.objaverse_base_dir or ""
        versioned_dir = os.path.join(base_dir, self.objaverse_assets_version) if base_dir else ""
        
        return {
            "base_dir": base_dir,
            "assets_dir": os.path.join(versioned_dir, "assets") if versioned_dir else "",
            "features_dir": os.path.join(versioned_dir, "features") if versioned_dir else "",
            "annotations_path": os.path.join(versioned_dir, "annotations.json.gz") if versioned_dir else "",
            "holodeck_base_dir": os.path.join(base_dir, "holodeck", self.holodeck_base_version) if base_dir else "",
            "thor_features_dir": os.path.join(base_dir, "holodeck", self.holodeck_base_version, "thor_object_data") if base_dir else "",
            "thor_annotations_path": os.path.join(base_dir, "holodeck", self.holodeck_base_version, "thor_object_data", "annotations.json.gz") if base_dir else "",
            "glb_cache_dir": self.objaverse_glb_cache_dir or "",
        }
    
    def get_glb_cache_dirs(self) -> List[str]:
        """
        Get the list of GLB cache directories (for searching)
        
        Returns:
            List of available GLB cache directories
        """
        dirs = []
        
        # Primary cache directory
        if self.objaverse_glb_cache_dir and os.path.exists(self.objaverse_glb_cache_dir):
            dirs.append(self.objaverse_glb_cache_dir)
        
        # Fallback cache directories
        fallback_dirs = [
            os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs"),
        ]
        for d in fallback_dirs:
            if os.path.exists(d) and d not in dirs:
                dirs.append(d)
        
        return dirs
    
    def to_env_dict(self) -> Dict[str, str]:
        """
        Convert configuration to an environment variable dictionary (for subprocesses)
        
        Returns:
            Environment variable dictionary
        """
        env = {}
        
        if self.objaverse_base_dir:
            env['OBJATHOR_ASSETS_BASE_DIR'] = self.objaverse_base_dir
        if self.objaverse_glb_cache_dir:
            env['OBJAVERSE_GLB_CACHE_DIR'] = self.objaverse_glb_cache_dir
        if self.objaverse_assets_version:
            env['ASSETS_VERSION'] = self.objaverse_assets_version
        if self.holodeck_base_version:
            env['HD_BASE_VERSION'] = self.holodeck_base_version
        
        if self.future3d_models_dir:
            env['PTH_3DFUTURE_ASSETS'] = self.future3d_models_dir
        if self.future3d_metadata_json:
            env['PTH_ASSETS_METADATA'] = self.future3d_metadata_json
        if self.future3d_metadata_scaled_json:
            env['PTH_ASSETS_METADATA_SCALED'] = self.future3d_metadata_scaled_json
        if self.future3d_embeddings_pkl:
            env['PTH_ASSETS_EMBED'] = self.future3d_embeddings_pkl
        
        if self.blender_executable:
            env['BLENDER_EXECUTABLE'] = self.blender_executable
        
        if self.logs_dir:
            env['LLMSCENE_LOGS_DIR'] = self.logs_dir
        
        return env
    
    def __repr__(self) -> str:
        return (
            f"PathConfig(\n"
            f"  detected_environment={self._detected_env}\n"
            f"  objaverse_base_dir={self.objaverse_base_dir}\n"
            f"  objaverse_glb_cache_dir={self.objaverse_glb_cache_dir}\n"
            f"  future3d_models_dir={self.future3d_models_dir}\n"
            f"  blender_executable={self.blender_executable}\n"
            f"  logs_dir={self.logs_dir}\n"
            f")"
        )
    
    def print_environment_info(self) -> None:
        """Print environment detection information and path configuration"""
        print(f"\n{'='*60}")
        print(f"PathConfig Environment Detection")
        print(f"{'='*60}")
        print(f"Detected environment: {self._detected_env}")
        print(f"  /path/to/storage exists: {os.path.exists('/path/to/storage')}")
        print(f"  AMLT_JOB_ID: {os.environ.get('AMLT_JOB_ID', 'not set')}")
        print(f"  AZUREML_RUN_ID: {os.environ.get('AZUREML_RUN_ID', 'not set')}")
        print(f"\nResolved Paths:")
        print(f"  Objaverse base: {self.objaverse_base_dir}")
        print(f"  Objaverse GLB cache: {self.objaverse_glb_cache_dir}")
        print(f"  3D-FUTURE models: {self.future3d_models_dir}")
        print(f"  Blender executable: {self.blender_executable}")
        print(f"  Logs dir: {self.logs_dir}")
        print(f"{'='*60}\n")
    
    # ========== Singleton Pattern Methods ==========
    
    @classmethod
    def init_from_config(cls, config: Dict[str, Any]) -> 'PathConfig':
        """
        Initialize the singleton from a configuration dictionary
        
        Args:
            config: Configuration dictionary (typically from SceneEditingInteraction's config)
            
        Returns:
            PathConfig instance
        """
        with cls._lock:
            cls._instance = cls(config)
            print(f"[PathConfig] Initialized from config: {cls._instance}")
            return cls._instance
    
    @classmethod
    def init_from_yaml(cls, yaml_path: str) -> 'PathConfig':
        """
        Initialize the singleton from a YAML file
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            PathConfig instance
        """
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Extract paths from the interaction config
        if 'interaction' in config:
            for interaction in config['interaction']:
                if 'config' in interaction:
                    return cls.init_from_config(interaction['config'])
        
        return cls.init_from_config(config)
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> 'PathConfig':
        """
        Get the singleton instance, creating it if it does not exist
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            PathConfig instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    print(f"[PathConfig] Auto-initialized: {cls._instance}")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)"""
        with cls._lock:
            cls._instance = None


# ========== Convenience Functions (backward compatible) ==========

def get_objaverse_paths() -> Dict[str, str]:
    """
    Get Objaverse paths (backward compatible with objaverse_retriever.py)
    """
    return PathConfig.get_instance().get_objaverse_paths()


def get_objaverse_glb_cache_dir() -> Optional[str]:
    """
    Get the Objaverse GLB cache directory
    """
    return PathConfig.get_instance().objaverse_glb_cache_dir


def get_3dfuture_models_dir() -> Optional[str]:
    """
    Get the 3D-FUTURE model directory
    """
    return PathConfig.get_instance().future3d_models_dir


def get_blender_executable() -> Optional[str]:
    """
    Get the Blender executable path
    """
    return PathConfig.get_instance().blender_executable


# ========== Test Code ==========

if __name__ == "__main__":
    # Test auto-initialization
    print("=== Testing PathConfig ===")
    
    config = PathConfig.get_instance()
    print(config)
    
    print("\n=== Objaverse Paths ===")
    for key, value in config.get_objaverse_paths().items():
        print(f"  {key}: {value}")
    
    print("\n=== GLB Cache Dirs ===")
    for d in config.get_glb_cache_dirs():
        print(f"  {d}")
    
    print("\n=== Environment Variables ===")
    for key, value in config.to_env_dict().items():
        print(f"  {key}={value}")
