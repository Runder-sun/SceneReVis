#!/usr/bin/env python3
"""
PathConfig - 统一路径配置管理器

用于 llmscene 项目中所有路径的集中管理，支持三种初始化方式：
1. 从 YAML 配置文件读取
2. 从环境变量读取
3. 使用默认硬编码路径（回退链）

优先级：配置文件 (YAML paths 块) > 环境变量 > 默认硬编码路径

使用方法：
    # 方式1：从配置字典初始化（用于 RL 训练）
    from path_config import PathConfig
    PathConfig.init_from_config(config_dict)
    
    # 方式2：从 YAML 文件初始化
    PathConfig.init_from_yaml("/path/to/config.yaml")
    
    # 方式3：自动初始化（从环境变量和默认值）
    paths = PathConfig.get_instance()
    
    # 获取路径
    glb_cache = paths.objaverse_glb_cache_dir
    blender_exe = paths.blender_executable
"""

import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List


class PathConfig:
    """
    统一的路径配置管理器（单例模式）
    
    Attributes:
        objaverse_base_dir: Objaverse 资产基础目录
        objaverse_glb_cache_dir: Objaverse GLB 缓存目录
        objaverse_assets_version: Objaverse 资产版本（默认 2023_09_23）
        
        future3d_models_dir: 3D-FUTURE 模型目录
        future3d_metadata_json: 3D-FUTURE 元数据 JSON 路径
        future3d_metadata_scaled_json: 3D-FUTURE 缩放元数据 JSON 路径
        future3d_embeddings_pkl: 3D-FUTURE 嵌入 pickle 路径
        
        blender_executable: Blender 可执行文件路径
        blender_candidates: Blender 备选路径列表
        
        logs_dir: 日志目录
        output_dir: 输出目录
    """
    
    _instance: Optional['PathConfig'] = None
    _lock = threading.Lock()
    
    # ========== 环境检测 ==========
    @staticmethod
    def _detect_environment() -> str:
        """
        检测当前运行环境
        
        Returns:
            'azure': Azure 云端环境（/path/to/storage 存在）
            'local': 本地开发环境
        """
        # 检查 Azure 特征路径
        if os.path.exists('/path/to/storage'):
            return 'azure'
        # 检查 AMLT 环境变量
        if os.environ.get('AMLT_JOB_ID') or os.environ.get('AZUREML_RUN_ID'):
            return 'azure'
        return 'local'
    
    # ========== 默认路径定义（根据环境动态选择）==========
    @classmethod
    def _get_objaverse_candidates(cls) -> List[str]:
        """获取 Objaverse 路径候选列表，根据环境调整优先级"""
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
        """获取 3D-FUTURE 路径候选列表，根据环境调整优先级"""
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
        """获取日志路径候选列表，根据环境调整优先级"""
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
    
    # Blender 路径（通常不受环境影响）
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
        初始化路径配置
        
        Args:
            config: 配置字典，可以包含 'paths' 键
        """
        self._config = config or {}
        paths_config = self._config.get('paths', {})
        
        # 检测运行环境并打印
        self._detected_env = self._detect_environment()
        
        # ========== Objaverse 路径 ==========
        objaverse_config = paths_config.get('objaverse', {})
        
        # 基础目录 - 使用动态候选列表
        self.objaverse_base_dir = self._resolve_path(
            config_value=objaverse_config.get('base_dir'),
            env_var='OBJATHOR_ASSETS_BASE_DIR',
            candidates=self._get_objaverse_candidates(),
            create_if_missing=False
        )
        
        # GLB 缓存目录
        glb_cache_config = objaverse_config.get('glb_cache_dir')
        if glb_cache_config:
            self.objaverse_glb_cache_dir = glb_cache_config
        elif os.environ.get('OBJAVERSE_GLB_CACHE_DIR'):
            self.objaverse_glb_cache_dir = os.environ['OBJAVERSE_GLB_CACHE_DIR']
        elif self.objaverse_base_dir:
            self.objaverse_glb_cache_dir = os.path.join(self.objaverse_base_dir, 'glbs')
        else:
            self.objaverse_glb_cache_dir = os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs")
        
        # 资产版本
        self.objaverse_assets_version = (
            objaverse_config.get('assets_version') or
            os.environ.get('ASSETS_VERSION', '2023_09_23')
        )
        
        # Holodeck 版本
        self.holodeck_base_version = (
            objaverse_config.get('holodeck_version') or
            os.environ.get('HD_BASE_VERSION', '2023_09_23')
        )
        
        # ========== 3D-FUTURE 路径 ==========
        future3d_config = paths_config.get('3d_future', {})
        
        # 模型目录 - 使用动态候选列表
        self.future3d_models_dir = self._resolve_path(
            config_value=future3d_config.get('models_dir'),
            env_var='PTH_3DFUTURE_ASSETS',
            candidates=self._get_3dfuture_candidates(),
            create_if_missing=False
        )
        
        # 元数据文件路径
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
        
        # ========== Blender 路径 ==========
        blender_config = paths_config.get('blender', {})
        
        # Blender 可执行文件
        self.blender_executable = self._resolve_blender_executable(
            config_value=blender_config.get('executable'),
            additional_candidates=blender_config.get('candidates', [])
        )
        
        # 保存候选路径供后续使用
        self.blender_candidates = (
            blender_config.get('candidates', []) + self._BLENDER_CANDIDATES
        )
        
        # ========== 日志和输出路径 ==========
        # 使用动态候选列表
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
        解析路径，按优先级：配置值 > 环境变量 > 候选路径链
        
        Args:
            config_value: 配置文件中的值
            env_var: 环境变量名
            candidates: 候选路径列表
            create_if_missing: 如果路径不存在是否创建
            
        Returns:
            解析后的路径，如果都不存在则返回 None
        """
        # 1. 配置文件值
        if config_value:
            if os.path.exists(config_value) or create_if_missing:
                if create_if_missing and not os.path.exists(config_value):
                    Path(config_value).mkdir(parents=True, exist_ok=True)
                return config_value
        
        # 2. 环境变量
        env_value = os.environ.get(env_var)
        if env_value and os.path.exists(env_value):
            return env_value
        
        # 3. 候选路径链
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        
        # 4. 如果允许创建，使用第一个候选路径
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
        解析 Blender 可执行文件路径
        
        Args:
            config_value: 配置文件中的值
            additional_candidates: 额外的候选路径
            
        Returns:
            Blender 可执行文件路径，如果都不存在则返回 None
        """
        # 1. 配置文件值
        if config_value and os.path.exists(config_value):
            return config_value
        
        # 2. 环境变量
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
        
        # 4. 候选路径链
        all_candidates = additional_candidates + self._BLENDER_CANDIDATES
        for candidate in all_candidates:
            if os.path.exists(candidate):
                return candidate
        
        return None
    
    # ========== 便捷方法 ==========
    
    def get_objaverse_paths(self) -> Dict[str, str]:
        """
        获取 Objaverse 相关的所有路径（兼容 objaverse_retriever.py 的 _get_objathor_paths）
        
        Returns:
            包含所有 Objaverse 路径的字典
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
        获取 GLB 缓存目录列表（用于搜索）
        
        Returns:
            可用的 GLB 缓存目录列表
        """
        dirs = []
        
        # 主缓存目录
        if self.objaverse_glb_cache_dir and os.path.exists(self.objaverse_glb_cache_dir):
            dirs.append(self.objaverse_glb_cache_dir)
        
        # 备选缓存目录
        fallback_dirs = [
            os.path.expanduser("~/.objaverse/hf-objaverse-v1/glbs"),
        ]
        for d in fallback_dirs:
            if os.path.exists(d) and d not in dirs:
                dirs.append(d)
        
        return dirs
    
    def to_env_dict(self) -> Dict[str, str]:
        """
        将配置转换为环境变量字典（用于子进程）
        
        Returns:
            环境变量字典
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
        """打印环境检测信息和路径配置"""
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
    
    # ========== 单例模式方法 ==========
    
    @classmethod
    def init_from_config(cls, config: Dict[str, Any]) -> 'PathConfig':
        """
        从配置字典初始化单例
        
        Args:
            config: 配置字典（通常来自 SceneEditingInteraction 的 config）
            
        Returns:
            PathConfig 实例
        """
        with cls._lock:
            cls._instance = cls(config)
            print(f"[PathConfig] Initialized from config: {cls._instance}")
            return cls._instance
    
    @classmethod
    def init_from_yaml(cls, yaml_path: str) -> 'PathConfig':
        """
        从 YAML 文件初始化单例
        
        Args:
            yaml_path: YAML 配置文件路径
            
        Returns:
            PathConfig 实例
        """
        import yaml
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 从 interaction 配置中提取 paths
        if 'interaction' in config:
            for interaction in config['interaction']:
                if 'config' in interaction:
                    return cls.init_from_config(interaction['config'])
        
        return cls.init_from_config(config)
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> 'PathConfig':
        """
        获取单例实例，如果不存在则创建
        
        Args:
            config: 可选的配置字典
            
        Returns:
            PathConfig 实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
                    print(f"[PathConfig] Auto-initialized: {cls._instance}")
        return cls._instance
    
    @classmethod
    def reset(cls):
        """重置单例（用于测试）"""
        with cls._lock:
            cls._instance = None


# ========== 便捷函数（向后兼容）==========

def get_objaverse_paths() -> Dict[str, str]:
    """
    获取 Objaverse 路径（向后兼容 objaverse_retriever.py）
    """
    return PathConfig.get_instance().get_objaverse_paths()


def get_objaverse_glb_cache_dir() -> Optional[str]:
    """
    获取 Objaverse GLB 缓存目录
    """
    return PathConfig.get_instance().objaverse_glb_cache_dir


def get_3dfuture_models_dir() -> Optional[str]:
    """
    获取 3D-FUTURE 模型目录
    """
    return PathConfig.get_instance().future3d_models_dir


def get_blender_executable() -> Optional[str]:
    """
    获取 Blender 可执行文件路径
    """
    return PathConfig.get_instance().blender_executable


# ========== 测试代码 ==========

if __name__ == "__main__":
    # 测试自动初始化
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
