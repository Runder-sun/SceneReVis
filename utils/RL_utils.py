#!/usr/bin/env python3
"""
RL_utils.py - 强化学习工具模块
提供场景渲染的封装函数，用于RL训练过程中的场景可视化
"""

import os
import sys
import json
import shutil
import traceback
import threading
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple

# 导入PIL用于图像处理
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, image processing will be limited")

# 导入体素评估相关依赖
try:
    import numpy as np
    import trimesh
    import copy
    from scipy.spatial.transform import Rotation as R
    from shapely.geometry import Polygon
    VOXEL_EVAL_AVAILABLE = True
except ImportError as e:
    VOXEL_EVAL_AVAILABLE = False
    print(f"Warning: Voxel evaluation dependencies not available: {e}")


# ============================================================================
# 场景格式转换函数
# 用于在flat格式（直接objects数组）和grouped格式（groups包含objects）之间转换
# ============================================================================

def convert_flat_to_grouped(scene: Dict[str, Any]) -> Dict[str, Any]:
    """将不带groups的场景格式转换为带groups的格式
    
    Args:
        scene: 不带groups的场景数据，包含直接的objects数组
        
    Returns:
        带groups的场景数据
    """
    # 如果已经有groups字段，直接返回
    if 'groups' in scene:
        return scene
    
    # 如果没有objects字段，也直接返回
    if 'objects' not in scene:
        return scene
    
    # 创建新的场景数据，保留原有的room信息
    grouped_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # 处理room_envelope或bounds字段
    if 'room_envelope' in scene:
        grouped_scene['room_envelope'] = scene['room_envelope']
    elif 'bounds_top' in scene and 'bounds_bottom' in scene:
        # 将旧格式的bounds转换为room_envelope
        grouped_scene['room_envelope'] = {
            'bounds_top': scene['bounds_top'],
            'bounds_bottom': scene['bounds_bottom']
        }
    
    # 将所有objects放入一个默认组
    grouped_scene['groups'] = [
        {
            'group_name': 'main_group',
            'group_type': 'functional_area',
            'description': 'Main functional area containing all objects',
            'objects': scene['objects']
        }
    ]
    
    print(f"Converted flat format to grouped format: {len(scene['objects'])} objects → 1 group")
    return grouped_scene


def convert_grouped_to_flat(scene: Dict[str, Any]) -> Dict[str, Any]:
    """将带groups的场景格式转换回不带groups的格式
    
    Args:
        scene: 带groups的场景数据
        
    Returns:
        不带groups的场景数据，包含直接的objects数组
    """
    # 如果没有groups字段，直接返回
    if 'groups' not in scene:
        return scene
    
    # 创建新的场景数据，保留原有的room信息
    flat_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # 处理room_envelope或bounds字段
    if 'room_envelope' in scene:
        # 提取bounds到顶层
        flat_scene['bounds_top'] = scene['room_envelope'].get('bounds_top', [])
        flat_scene['bounds_bottom'] = scene['room_envelope'].get('bounds_bottom', [])
    
    # 收集所有组中的objects到一个数组
    all_objects = []
    for group in scene.get('groups', []):
        all_objects.extend(group.get('objects', []))
    
    flat_scene['objects'] = all_objects
    
    print(f"Converted grouped format to flat format: {len(scene.get('groups', []))} groups → {len(all_objects)} objects")
    return flat_scene


# ============================================================================
# 全局单例模式和线程锁
# 用于解决多线程环境下的线程安全问题
# ============================================================================

# AssetRetrievalModule全局单例（解决CLIP模型线程安全）- 用于 3D-FUTURE
_GLOBAL_ASSET_RETRIEVAL = None
_ASSET_RETRIEVAL_LOCK = threading.Lock()

# ObjaverseRetriever全局单例 - 用于 Objaverse
_GLOBAL_OBJAVERSE_RETRIEVAL = None
_OBJAVERSE_RETRIEVAL_LOCK = threading.Lock()

# Blender渲染全局锁（解决Blender进程并发冲突）
_BLENDER_RENDER_LOCK = threading.Lock()


def _get_global_objaverse_retrieval(params: Optional[Dict[str, Any]] = None):
    """
    获取全局 ObjaverseRetriever 单例
    线程安全，确保在每个进程中只初始化一次
    
    参数:
        params: ObjaverseRetriever 初始化参数
    
    返回:
        ObjaverseRetriever 实例，如果初始化失败则返回 None
    """
    global _GLOBAL_OBJAVERSE_RETRIEVAL
    
    # 快速路径：如果已经初始化，直接返回
    if _GLOBAL_OBJAVERSE_RETRIEVAL is not None:
        return _GLOBAL_OBJAVERSE_RETRIEVAL
    
    # 加锁初始化
    with _OBJAVERSE_RETRIEVAL_LOCK:
        # 双重检查：可能在等待锁期间已被其他线程初始化
        if _GLOBAL_OBJAVERSE_RETRIEVAL is not None:
            return _GLOBAL_OBJAVERSE_RETRIEVAL
        
        try:
            # 确保当前目录在 sys.path 中
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            from objaverse_retriever import ObjaverseRetriever
            
            # 默认参数
            default_params = {
                'retrieval_threshold': 25.0,
                'do_print': False
            }
            if params:
                default_params.update(params)
            
            _GLOBAL_OBJAVERSE_RETRIEVAL = ObjaverseRetriever(**default_params)
            print(f"[RL_utils] ObjaverseRetriever initialized successfully")
            
        except ImportError as e:
            print(f"[RL_utils] Warning: Could not import ObjaverseRetriever: {e}")
            _GLOBAL_OBJAVERSE_RETRIEVAL = None
        except Exception as e:
            print(f"[RL_utils] Warning: Failed to initialize ObjaverseRetriever: {e}")
            _GLOBAL_OBJAVERSE_RETRIEVAL = None
    
    return _GLOBAL_OBJAVERSE_RETRIEVAL


def _get_global_asset_retrieval(params: Optional[Dict[str, Any]] = None):
    """
    获取全局AssetRetrievalModule单例
    线程安全，确保在每个进程中只初始化一次
    
    参数:
        params: AssetRetrievalModule初始化参数
    
    返回:
        AssetRetrievalModule实例，如果初始化失败则返回None
    """
    global _GLOBAL_ASSET_RETRIEVAL
    
    # 快速路径：如果已经初始化，直接返回
    if _GLOBAL_ASSET_RETRIEVAL is not None:
        return _GLOBAL_ASSET_RETRIEVAL
    
    # 加锁初始化
    with _ASSET_RETRIEVAL_LOCK:
        # 双重检查：可能在等待锁期间已被其他线程初始化
        if _GLOBAL_ASSET_RETRIEVAL is not None:
            return _GLOBAL_ASSET_RETRIEVAL
        
        try:
            # 确保当前目录在sys.path中
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            # 设置AssetRetrievalModule所需的环境变量
            project_root = Path(__file__).parent.parent  # llmscene目录
            
            # 定义可能的metadata路径
            metadata_paths = [
                project_root / "metadata" / "model_info_3dfuture_assets.json",
                Path("/path/to/SceneReVis/metadata/model_info_3dfuture_assets.json"),
                Path("/path/to/amlt/metadata/model_info_3dfuture_assets.json"),
                Path("/workspace/code/metadata/model_info_3dfuture_assets.json"),
            ]
            
            metadata_scaled_paths = [
                project_root / "metadata" / "model_info_3dfuture_assets_scaled.json",
                Path("/path/to/SceneReVis/metadata/model_info_3dfuture_assets_scaled.json"),
                Path("/path/to/amlt/metadata/model_info_3dfuture_assets_scaled.json"),
                Path("/workspace/code/metadata/model_info_3dfuture_assets_scaled.json"),
            ]
            
            embed_paths = [
                project_root / "metadata" / "all_assets_embed.pkl",
                project_root / "metadata" / "model_info_3dfuture_assets_embeds.pickle",
                Path("/path/to/SceneReVis/metadata/model_info_3dfuture_assets_embeds.pickle"),
                Path("/path/to/amlt/metadata/model_info_3dfuture_assets_embeds.pickle"),
                Path("/workspace/code/metadata/model_info_3dfuture_assets_embeds.pickle"),
            ]
            
            # 查找存在的文件
            metadata_file = None
            for path in metadata_paths:
                if path.exists():
                    metadata_file = str(path)
                    break
            
            metadata_scaled_file = None
            for path in metadata_scaled_paths:
                if path.exists():
                    metadata_scaled_file = str(path)
                    break
            
            embed_file = None
            for path in embed_paths:
                if path.exists():
                    embed_file = str(path)
                    break
            
            # 检查是否所有文件都找到
            if not metadata_file or not metadata_scaled_file or not embed_file:
                missing = []
                if not metadata_file:
                    missing.append("PTH_ASSETS_METADATA")
                if not metadata_scaled_file:
                    missing.append("PTH_ASSETS_METADATA_SCALED")
                if not embed_file:
                    missing.append("PTH_ASSETS_EMBED")
                
                print(f"Warning: Missing asset files for: {', '.join(missing)}")
                print(f"  Searched in: {project_root / 'metadata'}")
                return None
            
            # 设置环境变量
            os.environ['PTH_ASSETS_METADATA'] = metadata_file
            os.environ['PTH_ASSETS_METADATA_SCALED'] = metadata_scaled_file
            os.environ['PTH_ASSETS_EMBED'] = embed_file
            
            # 导入AssetRetrievalModule
            from sample import AssetRetrievalModule
            
            # 默认参数
            default_params = {
                'lambd': 0.7,
                'sigma': 0.05,
                'temp': 0.1,
                'top_p': 0.95,
                'top_k': 20,
                'asset_size_threshold': 0.5,
                'rand_seed': 42,
                'do_print': False
            }
            
            # 合并用户参数
            if params:
                default_params.update(params)
            
            # 初始化全局单例
            _GLOBAL_ASSET_RETRIEVAL = AssetRetrievalModule(**default_params)
            
            print(f"✓ Global AssetRetrievalModule initialized (thread-safe)")
            print(f"  Metadata: {metadata_file}")
            print(f"  Embeds: {embed_file}")
            
            return _GLOBAL_ASSET_RETRIEVAL
            
        except Exception as e:
            print(f"Error initializing global AssetRetrievalModule: {e}")
            traceback.print_exc()
            return None


class SceneRenderer:
    """
    场景渲染器类
    封装了从JSON场景数据到渲染图像的完整流程
    """
    
    def __init__(self, 
                 asset_path: Optional[str] = None,
                 use_placeholder: bool = False,
                 verbose: bool = False,
                 temp_dir: Optional[str] = None,
                 use_render_lock: bool = False,
                 fast_mode: bool = True,
                 enable_visualization: bool = True):
        """
        初始化场景渲染器
        
        参数:
            asset_path: 3D资产目录路径 (3D-FUTURE-model)
            use_placeholder: 是否使用占位符渲染（不加载真实3D模型）
            verbose: 是否输出详细日志
            temp_dir: 临时文件目录，如果为None则使用系统默认临时目录
            use_render_lock: 是否使用全局渲染锁（多线程环境推荐启用）
            fast_mode: 快速渲染模式（512x512, 16采样, 约2-3倍速度提升）
            enable_visualization: 是否启用3D辅助线可视化（bbox、箭头、坐标网格）
        """
        self.verbose = verbose
        self.use_placeholder = use_placeholder
        self.temp_dir = temp_dir or './temp_render'
        self.use_render_lock = use_render_lock
        self.fast_mode = fast_mode
        self.enable_visualization = enable_visualization
        
        # 设置资产路径
        self.asset_path = asset_path or self._find_asset_path()
        if self.asset_path:
            os.environ['PTH_3DFUTURE_ASSETS'] = self.asset_path
            if self.verbose:
                print(f"Using asset path: {self.asset_path}")
        
        # 设置环境变量
        os.environ['BPY_VERBOSE'] = '1' if verbose else '0'
        os.environ['BPY_USE_PLACEHOLDER_ONLY'] = '1' if use_placeholder else '0'
        os.environ['BPY_FAST_MODE'] = '1' if fast_mode else '0'
        os.environ['BPY_ENABLE_VISUALIZATION'] = '1' if enable_visualization else '0'
        
        if self.verbose and fast_mode:
            print("✓ Fast rendering mode enabled (512x512, 16 samples, ~2-3x speedup)")
        if self.verbose and enable_visualization:
            print("✓ 3D visualization enabled (bbox, arrows, coordinate grid with labels)")
        
        # 尝试导入渲染函数
        self.render_func = self._import_render_function()
        self.merge_func = self._import_merge_function()
        
    def _find_asset_path(self) -> Optional[str]:
        """查找3D资产目录，优先使用 PathConfig"""
        if os.environ.get('PTH_3DFUTURE_ASSETS'):
            return os.environ['PTH_3DFUTURE_ASSETS']
        
        # Try PathConfig first
        try:
            from path_config import PathConfig
            config = PathConfig.get_instance()
            if config.future3d_models_dir and os.path.exists(config.future3d_models_dir):
                return config.future3d_models_dir
        except ImportError:
            pass
        
        # 尝试常见的资产路径
        possible_paths = [
            '/path/to/datasets/3d-front/3D-FUTURE-model',
            '/path/to/datasets/3d-front/3D-FUTURE-model',
            '/path/to/datasets/3D-FUTURE-model',
            '/path/to/workspace/respace/assets',
            os.path.expanduser('~/datasets/3D-FUTURE-model'),
            './datasets/3D-FUTURE-model',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        if self.verbose:
            print("Warning: No 3D asset directory found")
        return None
    
    def _import_render_function(self):
        """导入Blender渲染函数"""
        try:
            # 添加当前目录到sys.path（因为文件已在utils目录中）
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            # 导入Blender包装器
            from blender_wrapper import render_scene_blender_external
            if self.verbose:
                print("✓ Successfully imported Blender external rendering wrapper")
            return render_scene_blender_external
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not import Blender renderer: {e}")
            return None
    
    def _import_merge_function(self):
        """导入图像合并函数"""
        try:
            import importlib.util
            # 文件已在utils目录中，直接引用同目录下的image_merger.py
            utils_path = Path(__file__).parent / 'image_merger.py'
            
            spec = importlib.util.spec_from_file_location("image_merger", str(utils_path))
            image_merger = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(image_merger)
            
            if self.verbose:
                print("✓ Successfully imported image merger")
            return image_merger.merge_rendered_views_with_annotations
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not import image merger: {e}")
            return None
    
    def render_scene(self, 
                    scene_data: Union[Dict[str, Any], str, Path],
                    output_path: Optional[Union[str, Path]] = None,
                    scene_id: str = "scene",
                    return_image: bool = False,
                    cleanup_temp: bool = True) -> Union[str, Tuple]:
        """
        渲染场景并返回图像路径或图像对象（线程安全包装器）
        
        参数:
            scene_data: 场景JSON数据（字典）或JSON文件路径
            output_path: 输出图像路径，如果为None则自动生成
            scene_id: 场景ID，用于命名和日志
            return_image: 是否返回PIL Image对象（而不仅仅是路径）
            cleanup_temp: 是否清理临时文件
            
        返回:
            如果return_image=False: 返回图像路径(str)
            如果return_image=True: 返回(图像路径, PIL.Image对象)元组
        """
        if self.use_render_lock:
            # 使用全局锁保护Blender渲染（多线程环境）
            with _BLENDER_RENDER_LOCK:
                return self._render_scene_impl(scene_data, output_path, scene_id, return_image, cleanup_temp)
        else:
            # 不使用锁（单线程环境或调试模式）
            return self._render_scene_impl(scene_data, output_path, scene_id, return_image, cleanup_temp)
    
    def _render_scene_impl(self, 
                          scene_data: Union[Dict[str, Any], str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          scene_id: str = "scene",
                          return_image: bool = False,
                          cleanup_temp: bool = True) -> Union[str, Tuple]:
        """
        渲染场景的实际实现（内部方法）
        
        参数:
            scene_data: 场景JSON数据（字典）或JSON文件路径
            output_path: 输出图像路径，如果为None则自动生成
            scene_id: 场景ID，用于命名和日志
            return_image: 是否返回PIL Image对象（而不仅仅是路径）
            cleanup_temp: 是否清理临时文件
            
        返回:
            如果return_image=False: 返回图像路径(str)
            如果return_image=True: 返回(图像路径, PIL.Image对象)元组
        """
        try:
            # 解析场景数据
            if isinstance(scene_data, (str, Path)):
                scene_path = Path(scene_data)
                if not scene_path.exists():
                    raise FileNotFoundError(f"Scene file not found: {scene_path}")
                with open(scene_path, 'r', encoding='utf-8') as f:
                    scene_dict = json.load(f)
            elif isinstance(scene_data, dict):
                scene_dict = scene_data
            else:
                raise ValueError(f"Invalid scene_data type: {type(scene_data)}")
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Rendering scene: {scene_id}")
                print(f"{'='*60}")
            
            # 设置输出路径
            if output_path is None:
                output_path = Path(self.temp_dir) / f"{scene_id}_rendered.png"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建临时渲染目录（多线程环境下确保唯一性）
            import time
            thread_id = threading.get_ident()
            timestamp = int(time.time() * 1000000)  # 微秒级时间戳
            temp_render_dir = Path(self.temp_dir) / f"render_{scene_id}_{thread_id}_{timestamp}"
            temp_render_dir.mkdir(parents=True, exist_ok=True)
            
            # 执行Blender渲染
            if self.render_func:
                try:
                    render_result = self.render_func(
                        scene_dict, 
                        temp_render_dir, 
                        scene_id, 
                        enable_visualization=self.enable_visualization,
                        fast_mode=self.fast_mode
                    )
                    if self.verbose:
                        print(f"Blender rendering completed: {render_result}")
                except Exception as e:
                    if self.verbose:
                        print(f"Blender rendering failed: {e}")
                        traceback.print_exc()
                    render_result = None
            else:
                render_result = None
            
            # 查找生成的图片文件
            top_file = temp_render_dir / "top" / "frame.png"
            diag_file = temp_render_dir / "diag" / "frame.png"
            
            # 合并图像或创建占位符
            if top_file.exists() and diag_file.exists() and self.merge_func:
                try:
                    # 合并俯视图和斜视图
                    self.merge_func(str(top_file), str(diag_file), str(output_path))
                    
                    if self.verbose:
                        print(f"✓ Rendered and merged image saved to: {output_path}")
                    
                    # 清理临时文件
                    if cleanup_temp:
                        try:
                            shutil.rmtree(temp_render_dir)
                        except Exception as cleanup_error:
                            if self.verbose:
                                print(f"Warning: Failed to cleanup temp dir: {cleanup_error}")
                    
                except Exception as merge_error:
                    if self.verbose:
                        print(f"Warning: Image merge failed: {merge_error}")
                        traceback.print_exc()
                    # 继续到占位符创建
                    output_path = self._create_placeholder_image(scene_dict, output_path, scene_id)
            else:
                # 创建占位符图像
                if self.verbose:
                    print("Warning: Rendered images not found, creating placeholder")
                output_path = self._create_placeholder_image(scene_dict, output_path, scene_id)
            
            # 返回结果
            output_path_str = str(output_path)
            
            if return_image:
                if PIL_AVAILABLE and output_path.exists():
                    img = Image.open(output_path)
                    return output_path_str, img
                else:
                    if self.verbose:
                        print("Warning: Cannot load image, returning path only")
                    return output_path_str, None
            else:
                return output_path_str
                
        except Exception as e:
            if self.verbose:
                print(f"Error in render_scene: {e}")
                traceback.print_exc()
            
            # 返回占位符
            fallback_path = self._create_placeholder_image(
                scene_dict if 'scene_dict' in locals() else {},
                output_path if output_path else Path(self.temp_dir) / f"{scene_id}_error.png",
                scene_id
            )
            
            if return_image:
                if PIL_AVAILABLE and Path(fallback_path).exists():
                    return str(fallback_path), Image.open(fallback_path)
                else:
                    return str(fallback_path), None
            else:
                return str(fallback_path)
    
    def _create_placeholder_image(self, 
                                  scene_dict: Dict[str, Any], 
                                  output_path: Path,
                                  scene_id: str) -> Path:
        """创建占位符图像"""
        try:
            if not PIL_AVAILABLE:
                if self.verbose:
                    print("Error: PIL not available, cannot create placeholder")
                return output_path
            
            # 创建占位符图片
            img = Image.new('RGB', (1024, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # 计算对象数量
            objects_count = 0
            if 'groups' in scene_dict:
                objects_count = sum(len(group.get('objects', [])) 
                                  for group in scene_dict.get('groups', []))
            elif 'objects' in scene_dict:
                objects_count = len(scene_dict.get('objects', []))
            
            # 获取房间类型
            room_type = scene_dict.get('room_type', 'Unknown')
            
            # 尝试加载字体
            font = None
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                pass
            
            # 绘制信息
            text_lines = [
                f"Scene: {scene_id}",
                f"Room Type: {room_type}",
                f"Objects: {objects_count}",
                "",
                "Placeholder Image",
                "(Rendering not available)"
            ]
            
            y_offset = 150
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (1024 - text_width) // 2
                draw.text((x, y_offset), line, fill='black', font=font)
                y_offset += 35
            
            # 保存占位符图片
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path)
            
            if self.verbose:
                print(f"✓ Created placeholder image: {output_path}")
            
            return output_path
            
        except Exception as e:
            if self.verbose:
                print(f"Error creating placeholder image: {e}")
                traceback.print_exc()
            return output_path


# 便捷函数：快速渲染场景
def render_scene_quick(scene_data: Union[Dict, str, Path],
                      output_path: Optional[str] = None,
                      return_image: bool = False,
                      verbose: bool = False,
                      fast_mode: bool = True) -> Union[str, Tuple]:
    """
    快速渲染场景的便捷函数
    
    参数:
        scene_data: 场景JSON数据（字典）或JSON文件路径
        output_path: 输出图像路径，如果为None则自动生成
        return_image: 是否返回PIL Image对象
        verbose: 是否输出详细日志
        fast_mode: 快速渲染模式（512x512, 16采样, 约2-3倍速度提升）
        
    返回:
        如果return_image=False: 返回图像路径(str)
        如果return_image=True: 返回(图像路径, PIL.Image对象)元组
        
    示例:
        >>> # 渲染JSON文件（快速模式）
        >>> img_path = render_scene_quick("./scene.json")
        >>> 
        >>> # 渲染JSON字典并获取图像对象
        >>> scene_dict = {...}
        >>> img_path, img = render_scene_quick(scene_dict, return_image=True)
        >>> 
        >>> # 高质量模式
        >>> img_path = render_scene_quick(scene_dict, output_path="./my_scene.png", fast_mode=False)
    """
    renderer = SceneRenderer(verbose=verbose, fast_mode=fast_mode)
    return renderer.render_scene(
        scene_data=scene_data,
        output_path=output_path,
        return_image=return_image
    )


# 便捷函数：批量渲染多个场景
def render_scenes_batch(scene_data_list: list,
                       output_dir: Union[str, Path],
                       scene_ids: Optional[list] = None,
                       return_images: bool = False,
                       verbose: bool = False) -> list:
    """
    批量渲染多个场景
    
    参数:
        scene_data_list: 场景数据列表（每个元素是字典或文件路径）
        output_dir: 输出目录
        scene_ids: 场景ID列表，如果为None则自动生成
        return_images: 是否返回图像对象
        verbose: 是否输出详细日志
        
    返回:
        渲染结果列表，每个元素是图像路径或(路径, 图像)元组
        
    示例:
        >>> scenes = [scene1_dict, scene2_dict, scene3_dict]
        >>> results = render_scenes_batch(scenes, "./output")
        >>> for img_path in results:
        ...     print(f"Rendered: {img_path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建渲染器
    renderer = SceneRenderer(verbose=verbose)
    
    # 生成场景ID
    if scene_ids is None:
        scene_ids = [f"scene_{i:03d}" for i in range(len(scene_data_list))]
    
    if len(scene_ids) != len(scene_data_list):
        raise ValueError(f"scene_ids length ({len(scene_ids)}) must match scene_data_list length ({len(scene_data_list)})")
    
    results = []
    for i, (scene_data, scene_id) in enumerate(zip(scene_data_list, scene_ids)):
        if verbose:
            print(f"\nRendering {i+1}/{len(scene_data_list)}: {scene_id}")
        
        output_path = output_dir / f"{scene_id}.png"
        result = renderer.render_scene(
            scene_data=scene_data,
            output_path=output_path,
            scene_id=scene_id,
            return_image=return_images
        )
        results.append(result)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch rendering completed: {len(results)} scenes")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
    
    return results


# 便捷函数：从文件加载并渲染
def render_scene_from_file(json_file_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          return_image: bool = False,
                          verbose: bool = False) -> Union[str, Tuple]:
    """
    从JSON文件加载场景并渲染
    
    参数:
        json_file_path: 场景JSON文件路径
        output_path: 输出图像路径，如果为None则使用与JSON文件相同的目录
        return_image: 是否返回PIL Image对象
        verbose: 是否输出详细日志
        
    返回:
        如果return_image=False: 返回图像路径(str)
        如果return_image=True: 返回(图像路径, PIL.Image对象)元组
        
    示例:
        >>> img_path = render_scene_from_file("./scenes/livingroom.json")
        >>> img_path, img = render_scene_from_file("./scenes/bedroom.json", return_image=True)
    """
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Scene file not found: {json_path}")
    
    # 如果没有指定输出路径，使用与JSON文件相同的目录
    if output_path is None:
        output_path = json_path.parent / f"{json_path.stem}_rendered.png"
    
    return render_scene_quick(
        scene_data=json_path,
        output_path=output_path,
        return_image=return_image,
        verbose=verbose
    )


if __name__ == "__main__":
    """
    测试和示例代码
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Scene Rendering Utility for RL')
    parser.add_argument('--scene', type=str, required=True, 
                       help='Path to scene JSON file or JSON string')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image path (default: auto-generated)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--placeholder', action='store_true',
                       help='Use placeholder rendering (no real 3D models)')
    parser.add_argument('--test', action='store_true',
                       help='Run test with sample scene')
    
    args = parser.parse_args()
    
    if args.test:
        print("Running test with sample scene...")
        
        # 创建示例场景
        test_scene = {
            "room_type": "livingroom",
            "room_id": "test_room",
            "room_envelope": {
                "bounds_top": [
                    [-3.0, 2.6, 3.0],
                    [3.0, 2.6, 3.0],
                    [3.0, 2.6, -3.0],
                    [-3.0, 2.6, -3.0]
                ],
                "bounds_bottom": [
                    [-3.0, 0.0, 3.0],
                    [3.0, 0.0, 3.0],
                    [3.0, 0.0, -3.0],
                    [-3.0, 0.0, -3.0]
                ]
            },
            "groups": [
                {
                    "group_name": "Seating Area",
                    "group_type": "seating",
                    "description": "Main seating arrangement",
                    "objects": [
                        {
                            "desc": "modern sofa",
                            "jid": "test-sofa-001",
                            "pos": [0, 0, 0],
                            "size": [2.0, 0.8, 0.9],
                            "rot": [0, 0, 0, 1]
                        },
                        {
                            "desc": "coffee table",
                            "jid": "test-table-001",
                            "pos": [0, 0, 1.5],
                            "size": [1.2, 0.4, 0.8],
                            "rot": [0, 0, 0, 1]
                        }
                    ]
                }
            ]
        }
        
        print("\n" + "="*60)
        print("Test Scene Data:")
        print(json.dumps(test_scene, indent=2))
        print("="*60)
        
        # 渲染测试场景
        result = render_scene_quick(
            scene_data=test_scene,
            output_path=args.output or "./test_render_output.png",
            return_image=True,
            verbose=True
        )
        
        if isinstance(result, tuple):
            img_path, img = result
            print(f"\n✓ Test completed successfully!")
            print(f"  Image path: {img_path}")
            if img:
                print(f"  Image size: {img.size}")
                print(f"  Image mode: {img.mode}")
        else:
            print(f"\n✓ Test completed!")
            print(f"  Image path: {result}")
    
    else:
        # 渲染指定的场景文件
        print(f"Rendering scene: {args.scene}")
        
        renderer = SceneRenderer(
            verbose=args.verbose,
            use_placeholder=args.placeholder
        )
        
        result = renderer.render_scene(
            scene_data=args.scene,
            output_path=args.output,
            return_image=True
        )
        
        if isinstance(result, tuple):
            img_path, img = result
            print(f"\n✓ Rendering completed!")
            print(f"  Output: {img_path}")
            if img:
                print(f"  Size: {img.size}")
        else:
            print(f"\n✓ Rendering completed!")
            print(f"  Output: {result}")


# ============================================================================
# 场景编辑功能
# ============================================================================

class SceneEditor:
    """
    场景编辑器类
    封装了场景编辑和资产检索的完整流程
    
    支持两种资产来源：
    - 3D-FUTURE: 默认，使用 AssetRetrievalModule
    - Objaverse: 使用 ObjaverseRetriever，需设置 use_objaverse=True
    """
    
    def __init__(self, 
                 asset_retrieval_params: Optional[Dict[str, Any]] = None,
                 use_objaverse: bool = True,
                 verbose: bool = False):
        """
        初始化场景编辑器
        
        参数:
            asset_retrieval_params: 资产检索模块参数，如果为None则使用默认参数
            use_objaverse: 是否使用 Objaverse 资产检索（默认 False，使用 3D-FUTURE）
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        self.use_objaverse = use_objaverse
        
        # 导入场景编辑器模块
        self._import_scene_editor()
        
        # 根据 use_objaverse 选择资产检索模块
        if use_objaverse:
            # 使用 Objaverse 检索器
            self.asset_retrieval_module = _get_global_objaverse_retrieval(asset_retrieval_params)
            if self.verbose and self.asset_retrieval_module:
                print("✓ Using ObjaverseRetriever for asset retrieval")
        else:
            # 使用 3D-FUTURE 检索器（默认）
            self.asset_retrieval_module = _get_global_asset_retrieval(asset_retrieval_params)
            if self.verbose and self.asset_retrieval_module:
                print("✓ Using AssetRetrievalModule (3D-FUTURE) for asset retrieval")
    
    def _import_scene_editor(self):
        """导入场景编辑器模块"""
        try:
            # 添加当前目录到sys.path（因为文件已在utils目录中）
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            from scene_editor import apply_tool_calls
            self.apply_tool_calls = apply_tool_calls
            
            if self.verbose:
                print("✓ Successfully imported scene_editor module")
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not import scene_editor: {e}")
            self.apply_tool_calls = None
    
    def edit_scene(self,
                  scene_data: Union[Dict[str, Any], str, Path],
                  tool_calls: list,
                  retrieve_assets: bool = True,
                  output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        编辑场景：应用tool_calls并进行资产检索
        
        参数:
            scene_data: 初始场景数据（字典）或JSON文件路径
            tool_calls: 工具调用列表，每个调用包含name和arguments
            retrieve_assets: 是否进行资产检索
            output_path: 输出场景JSON文件路径（可选）
            
        返回:
            编辑后的场景数据
            
        示例:
            >>> editor = SceneEditor()
            >>> tool_calls = [
            ...     {
            ...         "name": "add_object",
            ...         "arguments": {
            ...             "object_description": "modern sofa",
            ...             "position": [0, 0, 0],
            ...             "size": [2.0, 0.8, 0.9],
            ...             "rotation": [0, 0, 0, 1],
            ...             "group_name": "Seating Area"
            ...         }
            ...     }
            ... ]
            >>> new_scene = editor.edit_scene(initial_scene, tool_calls)
        """
        try:
            # 解析场景数据
            if isinstance(scene_data, (str, Path)):
                scene_path = Path(scene_data)
                if not scene_path.exists():
                    raise FileNotFoundError(f"Scene file not found: {scene_path}")
                with open(scene_path, 'r', encoding='utf-8') as f:
                    scene_dict = json.load(f)
            elif isinstance(scene_data, dict):
                scene_dict = scene_data
            else:
                raise ValueError(f"Invalid scene_data type: {type(scene_data)}")
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Scene Editing with {len(tool_calls)} tool calls")
                print(f"{'='*60}")
            
            # 1. 应用工具调用
            if self.apply_tool_calls and tool_calls:
                if self.verbose:
                    print(f"\nApplying {len(tool_calls)} tool calls...")
                
                edited_scene = self.apply_tool_calls(scene_dict, tool_calls)
                
                if self.verbose:
                    print(f"✓ Tool calls applied successfully")
            else:
                if not self.apply_tool_calls:
                    if self.verbose:
                        print("Warning: scene_editor not available, skipping tool calls")
                edited_scene = scene_dict
            
            # 2. 资产检索
            if retrieve_assets and self.asset_retrieval_module:
                if self.verbose:
                    print(f"\nPerforming asset retrieval...")
                
                try:
                    final_scene = self.asset_retrieval_module.sample_all_assets(
                        edited_scene,
                        is_greedy_sampling=True
                    )
                    
                    if self.verbose:
                        print(f"✓ Asset retrieval completed")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Asset retrieval failed: {e}")
                        traceback.print_exc()
                    final_scene = edited_scene
            else:
                if retrieve_assets and not self.asset_retrieval_module:
                    if self.verbose:
                        print("Warning: Asset retrieval module not available")
                final_scene = edited_scene
            
            # 3. 保存到文件（如果指定了输出路径）
            if output_path:
                output_file = Path(output_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_scene, f, indent=2, ensure_ascii=False)
                
                if self.verbose:
                    print(f"\n✓ Scene saved to: {output_file}")
            
            if self.verbose:
                print(f"\n{'='*60}")
                print("Scene editing completed successfully")
                print(f"{'='*60}")
            
            return final_scene
            
        except Exception as e:
            if self.verbose:
                print(f"Error in edit_scene: {e}")
                traceback.print_exc()
            raise


# 便捷函数：快速编辑场景
def edit_scene_quick(scene_data: Union[Dict, str, Path],
                    tool_calls: list,
                    retrieve_assets: bool = True,
                    use_objaverse: bool = False,
                    output_path: Optional[str] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    快速编辑场景的便捷函数
    
    参数:
        scene_data: 场景数据（字典）或JSON文件路径
        tool_calls: 工具调用列表
        retrieve_assets: 是否进行资产检索
        use_objaverse: 是否使用 Objaverse 资产检索（默认 False，使用 3D-FUTURE）
        output_path: 输出JSON文件路径（可选）
        verbose: 是否输出详细日志
        
    返回:
        编辑后的场景数据
        
    示例:
        >>> tool_calls = [
        ...     {"name": "add_object", "arguments": {...}},
        ...     {"name": "move_object", "arguments": {...}}
        ... ]
        >>> new_scene = edit_scene_quick(scene_dict, tool_calls)
    """
    editor = SceneEditor(verbose=verbose, use_objaverse=use_objaverse)
    return editor.edit_scene(
        scene_data=scene_data,
        tool_calls=tool_calls,
        retrieve_assets=retrieve_assets,
        output_path=output_path
    )


# 便捷函数：编辑并渲染场景
def edit_and_render_scene(scene_data: Union[Dict, str, Path],
                         tool_calls: list,
                         output_dir: Union[str, Path],
                         scene_id: str = "edited_scene",
                         retrieve_assets: bool = True,
                         use_objaverse: bool = False,
                         return_image: bool = False,
                         verbose: bool = False,
                         fast_mode: bool = True) -> Union[Tuple[Dict[str, Any], str, bool], 
                                                         Tuple[Dict[str, Any], str, Optional[Image.Image], bool]]:
    """
    编辑场景并渲染为图像（线程安全模式）
    
    线程安全保证：
    - AssetRetrievalModule使用全局单例（进程内唯一CLIP模型实例）
    - Blender渲染使用全局锁（串行化子进程调用）
    - 临时目录使用thread_id + timestamp确保唯一性
    
    参数:
        scene_data: 初始场景数据
        tool_calls: 工具调用列表
        output_dir: 输出目录
        scene_id: 场景ID
        retrieve_assets: 是否进行资产检索
        use_objaverse: 是否使用 Objaverse 资产检索（默认 False，使用 3D-FUTURE）
        return_image: 是否返回PIL Image对象
        verbose: 是否输出详细日志
        fast_mode: 快速渲染模式（512x512, 16采样, 约2-3倍速度提升）
        
    返回:
        如果return_image=False: (编辑后的场景数据, 渲染图像路径, 是否终止) 元组
        如果return_image=True: (编辑后的场景数据, 渲染图像路径, PIL.Image对象, 是否终止) 元组
        其中 is_terminated 为 True 表示 tool_calls 中包含 "terminate" 调用
        
    示例:
        >>> tool_calls = [{"name": "add_object", "arguments": {...}}]
        >>> # 只返回路径（快速模式）
        >>> new_scene, img_path, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", fast_mode=True
        ... )
        >>> # 使用 Objaverse 资产
        >>> new_scene, img_path, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", use_objaverse=True
        ... )
        >>> # 返回路径和图像对象
        >>> new_scene, img_path, img, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", return_image=True
        ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 检查是否有 terminate 调用
    is_terminated = any(call.get('name') == 'terminate' for call in tool_calls)
    
    # 1. 编辑场景
    if verbose:
        print(f"\n{'='*60}")
        print(f"Edit and Render: {scene_id}")
        print(f"  Fast mode: {fast_mode}")
        print(f"  Use Objaverse: {use_objaverse}")
        print(f"{'='*60}")
    
    editor = SceneEditor(verbose=verbose, use_objaverse=use_objaverse)
    edited_scene = editor.edit_scene(
        scene_data=scene_data,
        tool_calls=tool_calls,
        retrieve_assets=retrieve_assets,
        output_path=output_path / f"{scene_id}.json"
    )
    
    # 2. 渲染场景（使用快速模式并启用3D可视化）
    renderer = SceneRenderer(verbose=verbose, use_render_lock=False, fast_mode=fast_mode, enable_visualization=True)
    render_result = renderer.render_scene(
        scene_data=edited_scene,
        output_path=output_path / f"{scene_id}.png",
        scene_id=scene_id,
        return_image=return_image
    )
    
    if return_image:
        img_path, img = render_result
        if verbose:
            print(f"\n✓ Edit and render completed!")
            print(f"  Scene JSON: {output_path / f'{scene_id}.json'}")
            print(f"  Image: {img_path}")
            if img:
                print(f"  Image size: {img.size}")
            if is_terminated:
                print(f"  ⚠ Terminated: Tool calls contain 'terminate'")
        return edited_scene, img_path, img, is_terminated
    else:
        img_path = render_result
        if verbose:
            print(f"\n✓ Edit and render completed!")
            print(f"  Scene JSON: {output_path / f'{scene_id}.json'}")
            print(f"  Image: {img_path}")
            if is_terminated:
                print(f"  ⚠ Terminated: Tool calls contain 'terminate'")
        return edited_scene, img_path, is_terminated


# ========================================
# Voxel-based Reward Computation
# ========================================

def _find_objaverse_glb(uid: str) -> Optional[Path]:
    """
    查找 Objaverse GLB 文件路径。
    
    优先使用 PathConfig 统一配置，然后按以下顺序搜索:
    1. PathConfig 配置的缓存目录
    2. OBJAVERSE_GLB_CACHE_DIR 环境变量指定的路径
    3. /path/to/datasets/objathor-assets/glbs (云存储)
    4. ~/.objaverse/hf-objaverse-v1/glbs (本地缓存)
    
    Args:
        uid: Objaverse 资产 UID
        
    Returns:
        GLB 文件路径，如果未找到则返回 None
    """
    import os
    
    if not uid or len(uid) < 2:
        return None
    
    # GLB 缓存目录列表（按优先级排序）
    cache_dirs = []
    
    # Try PathConfig first
    try:
        from path_config import PathConfig
        config = PathConfig.get_instance()
        if config.objaverse_glb_cache_dir:
            cache_dirs.append(Path(config.objaverse_glb_cache_dir))
    except ImportError:
        pass
    
    # 添加环境变量指定的路径
    env_cache = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
    if env_cache:
        cache_dirs.append(Path(env_cache) / "glbs")
    
    # 添加云存储路径
    cache_dirs.append(Path("/path/to/datasets/objathor-assets/glbs"))
    
    # 添加本地缓存
    cache_dirs.append(Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs")
    
    # 使用 uid[:2] 直接构建路径（优化：避免遍历所有子目录）
    subdir_name = uid[:2]
    
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        
        # 直接查找 uid[:2] 子目录下的 GLB 文件
        candidate = cache_dir / subdir_name / f"{uid}.glb"
        if candidate.is_file():
            return candidate
    
    return None


class VoxelReward:
    """
    基于体素评估的奖励计算类
    使用体素化方法评估场景的物理合理性（出界和碰撞），并计算强化学习奖励
    """
    
    def __init__(self, 
                 models_base_path: str,
                 voxel_size: float = 0.05,
                 reward_threshold: float = 1e-5,
                 verbose: bool = False):
        """
        初始化体素奖励计算器
        
        参数:
            models_base_path: 3D模型文件基础路径 (3D-FUTURE-model目录)
            voxel_size: 体素大小，单位为米 (默认0.05m = 5cm)
            reward_threshold: PBL损失阈值，低于此值给予正奖励 (默认1e-5)
            verbose: 是否输出详细日志
        """
        if not VOXEL_EVAL_AVAILABLE:
            raise ImportError(
                "Voxel evaluation dependencies not available. "
                "Please install: numpy, trimesh, scipy, shapely"
            )
        
        self.models_base_path = Path(models_base_path)
        self.voxel_size = voxel_size
        self.reward_threshold = reward_threshold
        self.verbose = verbose
        
        if not self.models_base_path.exists():
            raise ValueError(f"Models base path does not exist: {models_base_path}")
        
        if self.verbose:
            print(f"VoxelReward initialized:")
            print(f"  Models path: {self.models_base_path}")
            print(f"  Voxel size: {self.voxel_size}m")
            print(f"  Reward threshold: {self.reward_threshold}")
    
    def _parse_scene_data(self, scene_json: Dict, format_type: str = 'ours'):
        """
        解析场景JSON数据，支持两种格式
        
        参数:
            scene_json: 场景JSON数据
            format_type: 'ours' 或 'respace'
        
        返回:
            bounds_bottom, bounds_top, all_objects
        """
        if format_type == 'respace':
            bounds_bottom = scene_json.get('bounds_bottom', [])
            bounds_top = scene_json.get('bounds_top', [])
            all_objects = scene_json.get('objects', [])
        else:
            # ours格式：使用room_envelope和groups结构
            if 'room_envelope' not in scene_json:
                raise ValueError("场景JSON缺少'room_envelope'字段")
            
            bounds_bottom = scene_json['room_envelope']['bounds_bottom']
            bounds_top = scene_json['room_envelope']['bounds_top']
            all_objects = []
            if 'groups' in scene_json and scene_json['groups'] is not None:
                for group in scene_json['groups']:
                    if 'objects' in group and group['objects'] is not None:
                        all_objects.extend(group['objects'])
        
        return bounds_bottom, bounds_top, all_objects
    
    def _create_floor_plan_polygon(self, bounds_bottom):
        """从房间底部边界创建地板多边形"""
        points = [(pt[0], pt[2]) for pt in bounds_bottom]
        return Polygon(points)
    
    def _create_room_mesh(self, bounds_bottom, bounds_top, floor_plan_polygon):
        """创建房间mesh"""
        num_verts = len(bounds_bottom)
        all_vertices = np.array(bounds_bottom + bounds_top)
        
        # 使用trimesh创建地板三角形
        vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
        # 移除无效面
        idxs = []
        for i, row in enumerate(floor_faces):
            if np.any(row == num_verts):
                idxs.append(i)
        floor_faces = np.delete(floor_faces, idxs, axis=0)
        
        # 创建天花板面
        ceiling_faces = floor_faces + num_verts
        
        # 创建侧面
        side_faces = []
        for i in range(num_verts):
            next_i = (i + 1) % num_verts
            side_faces.append([i, next_i, i + num_verts])
            side_faces.append([next_i, next_i + num_verts, i + num_verts])
        side_faces = np.array(side_faces)
        
        all_faces = np.concatenate((floor_faces, ceiling_faces, side_faces), axis=0)
        room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        
        trimesh.repair.fix_normals(room_mesh)
        return room_mesh
    
    def _prepare_asset_voxel(self, obj, voxel_size):
        """
        准备对象的体素表示
        关键：先旋转mesh，再体素化，使用中心底部作为锚点
        
        支持两种资产来源:
        - 3D-FUTURE: 使用 'jid' 字段，从 models_base_path/{jid}/raw_model.glb 加载
        - Objaverse: 使用 'uid' 字段，从 GLB 缓存加载
        """
        # 确定资产来源和模型路径
        asset_source = obj.get('asset_source', '3d-future')
        model_path = None
        asset_id = None
        
        if asset_source == 'objaverse':
            # Objaverse 资产：使用 uid 和 GLB 缓存
            uid = obj.get('uid')
            if uid:
                asset_id = uid
                model_path = _find_objaverse_glb(uid)
                if model_path and self.verbose:
                    print(f"Found Objaverse GLB for {uid[:16]}...: {model_path}")
        else:
            # 3D-FUTURE 资产：使用 jid
            jid = obj.get('jid', 'N/A')
            asset_id = jid
            model_path = self.models_base_path / jid / 'raw_model.glb'
            if not model_path.exists():
                model_path = None
        
        if model_path is None or not model_path.exists():
            # 使用边界框创建替代mesh
            if self.verbose:
                print(f"Model not found for {asset_id}, using box placeholder")
            mesh = trimesh.creation.box(extents=obj['size'])
        else:
            try:
                asset_scene = trimesh.load(str(model_path))
                if isinstance(asset_scene, trimesh.Scene):
                    mesh = asset_scene.to_geometry()
                else:
                    mesh = asset_scene
            except Exception as e:
                if self.verbose:
                    print(f"加载mesh失败 {asset_id}: {e}")
                mesh = trimesh.creation.box(extents=obj['size'])
        
        # ==== Objaverse 坐标系校正 ====
        # Objaverse GLB 模型需要重置初始旋转（与 blender_renderer.py 保持一致）
        # 这确保物理计算与渲染结果一致
        if asset_source == 'objaverse' and model_path is not None:
            # Objaverse 模型：重置到标准朝向（无需额外旋转校正）
            # 渲染时使用 rotation_euler = (0, 0, 0)，这里保持 mesh 原样即可
            # 因为 trimesh 加载的 GLB 已经是正确朝向
            if self.verbose:
                print(f"Objaverse asset {asset_id}: using standard orientation (no correction needed)")
        
        # 1. 应用缩放到目标尺寸
        original_size = mesh.extents
        target_size = obj['size']
        scale_factors = target_size / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        # 2. 应用旋转（不包括平移）
        rot_xyzw = obj['rot']
        rotation = R.from_quat(rot_xyzw)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()
        mesh.apply_transform(rotation_matrix)
        
        # 3. 体素化旋转后的mesh
        try:
            asset_voxels = mesh.voxelized(pitch=voxel_size).fill()
            asset_voxel_matrix = asset_voxels.matrix
        except Exception as e:
            if self.verbose:
                print(f"体素化失败 {obj.get('desc', 'Unknown')}: {e}")
            return None, None, mesh
        
        # 4. 计算体素空间的位置偏移
        pos = obj['pos']
        asset_pos_voxels = np.floor(np.array(pos) / voxel_size)
        
        # 物体体素矩阵的锚点：X轴中心，Y轴底部，Z轴中心
        asset_start_voxels = np.array([
            asset_voxel_matrix.shape[0] // 2,
            0,
            asset_voxel_matrix.shape[2] // 2
        ])
        
        # 计算从原点的偏移
        asset_shift_from_origin = asset_pos_voxels - asset_start_voxels
        
        return asset_voxel_matrix, asset_shift_from_origin, mesh
    
    def _occupancy_overlap(self, voxel_matrix_a, voxel_matrix_b, offset_b):
        """计算两个体素矩阵的重叠"""
        overlap_matrix = copy.deepcopy(voxel_matrix_a).astype(int)
        
        for i in range(voxel_matrix_b.shape[0]):
            for j in range(voxel_matrix_b.shape[1]):
                for k in range(voxel_matrix_b.shape[2]):
                    if voxel_matrix_b[i, j, k]:
                        shifted_pos = (i + offset_b[0], j + offset_b[1], k + offset_b[2])
                        if (0 <= shifted_pos[0] < overlap_matrix.shape[0] and 
                            0 <= shifted_pos[1] < overlap_matrix.shape[1] and 
                            0 <= shifted_pos[2] < overlap_matrix.shape[2]):
                            overlap_matrix[shifted_pos[0], shifted_pos[1], shifted_pos[2]] += 1
        
        return (overlap_matrix == 2)
    
    def _compute_voxel_oob(self, obj, room_origin_shift, room_voxel_matrix, voxel_volume):
        """计算基于体素的出界体积"""
        asset_voxel_matrix, asset_shift_from_origin, mesh = self._prepare_asset_voxel(obj, self.voxel_size)
        
        if asset_voxel_matrix is None:
            return 0.0
        
        asset_offset = np.floor(room_origin_shift + asset_shift_from_origin).astype(int)
        
        inside_voxels = self._occupancy_overlap(room_voxel_matrix, asset_voxel_matrix, asset_offset)
        num_asset_voxels = np.sum(asset_voxel_matrix)
        asset_volume = num_asset_voxels * voxel_volume
        
        num_inside_voxels = np.sum(inside_voxels)
        inside_volume = num_inside_voxels * voxel_volume
        
        voxel_oob = asset_volume - inside_volume
        return max(0.0, voxel_oob)
    
    def _compute_voxel_collision(self, obj_x, obj_y, voxel_volume):
        """计算基于体素的碰撞体积"""
        asset_voxel_matrix_x, asset_shift_x, mesh_x = self._prepare_asset_voxel(obj_x, self.voxel_size)
        asset_voxel_matrix_y, asset_shift_y, mesh_y = self._prepare_asset_voxel(obj_y, self.voxel_size)
        
        if asset_voxel_matrix_x is None or asset_voxel_matrix_y is None:
            return 0.0
        
        offset = np.floor(asset_shift_y - asset_shift_x).astype(int)
        inside_voxels = self._occupancy_overlap(asset_voxel_matrix_x, asset_voxel_matrix_y, offset)
        
        num_inside_voxels = np.sum(inside_voxels)
        intersection_volume = num_inside_voxels * voxel_volume
        
        return intersection_volume
    
    def evaluate_scene(self, 
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Dict[str, Any]:
        """
        评估场景的物理合理性
        
        参数:
            scene_data: 场景数据，可以是字典、JSON文件路径或Path对象
            format_type: 场景格式类型，'ours' 或 'respace'
        
        返回:
            包含评估指标的字典，包含：
            - total_oob_loss: 总出界体积损失
            - total_mbl_loss: 总碰撞体积损失 (Mesh-Based Loss)
            - total_pbl_loss: 总物理损失 (Physics-Based Loss = OOB + MBL)
            - num_oob_objects: 出界物体数量
            - num_collision_pairs: 碰撞物体对数量
            - voxel_size: 使用的体素大小
        """
        # 加载场景数据
        if isinstance(scene_data, (str, Path)):
            with open(scene_data, 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
        else:
            scene_json = scene_data
        
        # 提取场景数据
        bounds_bottom, bounds_top, all_objects = self._parse_scene_data(scene_json, format_type)
        
        if len(all_objects) == 0:
            return {
                'total_oob_loss': 0.0,
                'total_mbl_loss': 0.0,
                'total_pbl_loss': 0.0,
                'num_oob_objects': 0,
                'num_collision_pairs': 0,
                'voxel_size': self.voxel_size
            }
        
        floor_plan_polygon = self._create_floor_plan_polygon(bounds_bottom)
        
        # 创建房间mesh并体素化
        room_mesh = self._create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
        room_voxels = room_mesh.voxelized(pitch=self.voxel_size).fill()
        room_voxel_matrix = room_voxels.matrix
        room_size_voxels = np.ceil(abs(room_mesh.bounds[0] - room_mesh.bounds[1]) / self.voxel_size)
        room_origin_shift = np.array([room_size_voxels[0] / 2.0, 0, room_size_voxels[2] / 2.0])
        
        voxel_volume = self.voxel_size ** 3
        
        # 计算指标
        voxel_oobs = []
        voxel_collisions = []
        
        for i, obj in enumerate(all_objects):
            # 体素出界
            voxel_oob = self._compute_voxel_oob(obj, room_origin_shift, room_voxel_matrix, voxel_volume)
            voxel_oobs.append(voxel_oob)
            
            # 与其他对象的碰撞
            for j, other_obj in enumerate(all_objects[i+1:], i+1):
                voxel_collision = self._compute_voxel_collision(obj, other_obj, voxel_volume)
                voxel_collisions.append(voxel_collision)
        
        # 汇总指标
        total_oob_loss = sum(voxel_oobs)
        total_mbl_loss = sum(voxel_collisions)
        total_pbl_loss = total_oob_loss + total_mbl_loss
        
        metrics = {
            'total_oob_loss': total_oob_loss,
            'total_mbl_loss': total_mbl_loss,
            'total_pbl_loss': total_pbl_loss,
            'num_oob_objects': sum(1 for oob in voxel_oobs if oob > 1e-6),
            'num_collision_pairs': sum(1 for col in voxel_collisions if col > 1e-6),
            'voxel_size': self.voxel_size
        }
        
        if self.verbose:
            print(f"\n场景评估结果:")
            print(f"  OOB损失: {total_oob_loss:.6f}")
            print(f"  MBL损失: {total_mbl_loss:.6f}")
            print(f"  PBL损失: {total_pbl_loss:.6f}")
            print(f"  出界物体: {metrics['num_oob_objects']}/{len(all_objects)}")
            print(f"  碰撞对数: {metrics['num_collision_pairs']}")
        
        return metrics
    
    def compute_reward(self, 
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Tuple[float, Dict[str, Any]]:
        """
        计算场景的强化学习奖励
        
        参数:
            scene_data: 场景数据，可以是字典、JSON文件路径或Path对象
            format_type: 场景格式类型，'ours' 或 'respace'
        
        返回:
            (reward, metrics) 元组：
            - reward: 奖励值
              - PBL损失 > 0.1: -1.0
              - PBL损失在 [1e-5, 0.1] 之间: 线性插值从 -1.0 到 +1.0
              - PBL损失 < 1e-5: +1.0
            - metrics: 详细评估指标字典
        """
        # 评估场景
        metrics = self.evaluate_scene(scene_data, format_type)
        
        # 根据PBL损失计算奖励（分段线性）
        pbl_loss = metrics['total_pbl_loss']
        
        if pbl_loss < self.reward_threshold:
            # 非常好：PBL损失 < 1e-5
            reward = 1.0
        elif pbl_loss <= 0.1:
            # 中等：在 [1e-5, 0.1] 之间线性插值
            # reward = -1.0 + 2.0 * (0.1 - pbl_loss) / (0.1 - 1e-5)
            # 简化：pbl_loss 从 1e-5 增长到 0.1，reward 从 1.0 降到 -1.0
            reward = 1.0 - 2.0 * (pbl_loss - self.reward_threshold) / (0.1 - self.reward_threshold)
        else:
            # 差：PBL损失 > 0.1
            reward = -1.0
        
        if self.verbose:
            print(f"\n奖励计算:")
            print(f"  PBL损失: {pbl_loss:.6e}")
            print(f"  阈值: {self.reward_threshold:.6e}")
            print(f"  奖励: {reward:.4f}")
        
        return reward, metrics


# ========================================
# Trimesh-based Physics Metrics (参考myeval实现)
# ========================================

class TrimeshPhysicsMetrics:
    """
    基于Trimesh碰撞检测的物理指标计算类
    完全参考myeval.py的实现方式，使用trimesh.collision.CollisionManager
    而不是体素化方法
    
    此版本支持异型房间（L型、T型等）的精确出界检测，
    并删除了碰撞检测的容差
    """
    
    def __init__(self,
                 models_base_path: str = None,
                 verbose: bool = False):
        """
        初始化Trimesh物理指标计算器
        
        参数:
            models_base_path: 3D模型文件基础路径 (3D-FUTURE-model目录)
                             当使用 Objaverse 资产时可以为 None
            verbose: 是否输出详细日志
        """
        self.verbose = verbose
        
        # models_base_path 现在是可选的
        # 只在使用 3D-FUTURE 资产时才需要
        if models_base_path:
            self.models_base_path = Path(models_base_path)
            if not self.models_base_path.exists():
                # 改为警告而非错误，允许在 Objaverse 模式下继续运行
                if self.verbose:
                    print(f"Warning: Models base path does not exist: {self.models_base_path}")
                    print(f"  3D-FUTURE assets will use placeholder boxes")
                # 设为 None 表示不可用
                self.models_base_path = None
        else:
            self.models_base_path = None
        
        if self.verbose:
            print(f"TrimeshPhysicsMetrics initialized:")
            if self.models_base_path:
                print(f"  Models path: {self.models_base_path}")
            else:
                print(f"  Models path: None (Objaverse mode or 3D-FUTURE unavailable)")
    
    def _create_floor_polygon(self, bounds_bottom):
        """
        从房间底部边界创建地板多边形（支持异型房间）
        
        Args:
            bounds_bottom: 底部边界顶点列表 [[x1,y1,z1], [x2,y2,z2], ...]
        
        Returns:
            shapely.geometry.Polygon: 地板多边形（使用X和Z坐标）
        """
        from shapely.geometry import Polygon
        # 使用X和Z坐标创建2D多边形（Y是高度方向）
        points = [(pt[0], pt[2]) for pt in bounds_bottom]
        return Polygon(points)
    
    def _create_room_mesh(self, bounds_bottom, bounds_top):
        """
        从多边形边界创建真实的房间mesh（支持异型房间）
        
        Args:
            bounds_bottom: 底部边界顶点列表
            bounds_top: 顶部边界顶点列表
        
        Returns:
            trimesh.Trimesh: 房间mesh对象
        """
        bounds_bottom = np.array(bounds_bottom)
        bounds_top = np.array(bounds_top)
        
        # 创建地板多边形
        floor_polygon = self._create_floor_polygon(bounds_bottom.tolist())
        
        num_verts = len(bounds_bottom)
        all_vertices = np.concatenate([bounds_bottom, bounds_top], axis=0)
        
        # 使用trimesh三角化地板多边形
        try:
            vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="triangle")
        except Exception:
            try:
                vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="earcut")
            except Exception:
                floor_faces = np.array([[0, i, i+1] for i in range(1, num_verts-1)])
        
        # 移除无效面
        valid_mask = np.all(floor_faces < num_verts, axis=1)
        floor_faces = floor_faces[valid_mask]
        
        # 创建天花板面
        ceiling_faces = floor_faces + num_verts
        ceiling_faces = ceiling_faces[:, ::-1]
        
        # 创建侧面
        side_faces = []
        for i in range(num_verts):
            next_i = (i + 1) % num_verts
            side_faces.append([i, next_i, i + num_verts])
            side_faces.append([next_i, next_i + num_verts, i + num_verts])
        side_faces = np.array(side_faces)
        
        # 合并所有面
        all_faces = np.concatenate([floor_faces, ceiling_faces, side_faces], axis=0)
        
        # 创建mesh并修复法线
        room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        trimesh.repair.fix_normals(room_mesh)
        
        return room_mesh
    
    def _check_object_out_of_bounds(self, obj_mesh, room_mesh, floor_polygon, 
                                    room_height_min, room_height_max, num_samples=500):
        """
        使用mesh containment检测物体是否出界（支持异型房间和高度检测）
        
        Args:
            obj_mesh: 物体的trimesh对象
            room_mesh: 房间的trimesh对象
            floor_polygon: 房间地板的shapely多边形
            room_height_min: 房间最低高度
            room_height_max: 房间最高高度
            num_samples: 采样点数量
        
        Returns:
            (is_oob: bool, oob_volume: float)
        """
        from shapely.geometry import Point
        
        try:
            # 在物体mesh上采样点
            sample_points = obj_mesh.sample(num_samples)
            
            # 检测2D出界（XZ平面）
            # 注意：shapely的contains()对于边界上的点返回False，这会导致贴墙放置的家具被误判为出界
            # 解决方案：给多边形添加微小缓冲区(1mm)，使边界上的点被正确判定为在内部
            buffered_polygon = floor_polygon.buffer(0.001)  # 1mm缓冲区
            points_2d = [Point(pt[0], pt[2]) for pt in sample_points]
            inside_2d = np.array([buffered_polygon.contains(p) for p in points_2d])
            
            # 检测高度出界（Y方向）- 同样添加微小容差
            height_tolerance = 0.001  # 1mm容差
            inside_height = (sample_points[:, 1] >= room_height_min - height_tolerance) & (sample_points[:, 1] <= room_height_max + height_tolerance)
            
            # 综合判断
            inside = inside_2d & inside_height
            
            oob_ratio = 1.0 - (inside.sum() / len(inside))
            
            if oob_ratio > 0.01:  # 超过1%的点在外面
                is_oob = True
                obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
                oob_volume = obj_volume * oob_ratio
            else:
                is_oob = False
                oob_volume = 0.0
            
            return is_oob, oob_volume
            
        except Exception as e:
            if self.verbose:
                print(f"警告: 出界检测采样失败: {e}，使用边界框备用方案")
            # 备用方案
            obj_bounds = obj_mesh.bounds
            room_bounds = room_mesh.bounds
            
            if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
                obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
                return True, obj_volume * 0.1
            return False, 0.0
    def _parse_scene_data(self, scene_json: Dict, format_type: str = 'ours'):
        """
        解析场景JSON数据，支持两种格式
        
        参数:
            scene_json: 场景JSON数据
            format_type: 'ours' 或 'respace'
        
        返回:
            bounds_bottom, bounds_top, all_objects
        """
        if format_type == 'respace':
            bounds_bottom = scene_json.get('bounds_bottom', [])
            bounds_top = scene_json.get('bounds_top', [])
            all_objects_data = scene_json.get('objects', [])
        else:
            # ours格式：使用room_envelope和groups结构
            # 也支持flat格式（直接有objects字段）
            if 'room_envelope' in scene_json:
                bounds_bottom = scene_json['room_envelope']['bounds_bottom']
                bounds_top = scene_json['room_envelope']['bounds_top']
            else:
                bounds_bottom = scene_json.get('bounds_bottom', [])
                bounds_top = scene_json.get('bounds_top', [])
            
            all_objects_data = []
            if 'groups' in scene_json and scene_json['groups'] is not None:
                for group in scene_json['groups']:
                    if 'objects' in group and group['objects'] is not None:
                        all_objects_data.extend(group['objects'])
            elif 'objects' in scene_json:
                # 支持flat格式
                all_objects_data = scene_json['objects']
        
        return bounds_bottom, bounds_top, all_objects_data
    
    def _load_and_transform_mesh(self, obj_data: Dict, format_type: str = 'ours') -> Optional['trimesh.Trimesh']:
        """
        加载并变换物体的mesh（参考myeval的实现）
        
        支持两种资产来源:
        - 3D-FUTURE: 使用 'jid' 字段，从 models_base_path/{jid}/raw_model.glb 加载
        - Objaverse: 使用 'uid' 字段，从 GLB 缓存加载
        
        参数:
            obj_data: 物体数据字典
            format_type: 'ours' 或 'respace'
        
        返回:
            变换后的trimesh对象，如果失败则返回None
        """
        try:
            # 获取目标尺寸
            if format_type == 'respace':
                target_size = obj_data.get('sampled_asset_size', obj_data.get('size', [1, 1, 1]))
            else:
                target_size = obj_data.get('size', [1, 1, 1])
            
            # 确定资产来源和模型路径
            asset_source = obj_data.get('asset_source', '3d-future')
            model_path = None
            asset_id = None
            
            if asset_source == 'objaverse':
                # Objaverse 资产：使用 uid 和 GLB 缓存
                uid = obj_data.get('uid')
                if uid:
                    asset_id = uid
                    model_path = _find_objaverse_glb(uid)
                    if model_path and self.verbose:
                        print(f"Found Objaverse GLB for {uid[:16]}...: {model_path}")
            else:
                # 3D-FUTURE 资产：使用 jid
                if format_type == 'respace':
                    jid = obj_data.get('sampled_asset_jid', obj_data.get('jid', 'N/A'))
                else:
                    jid = obj_data.get('jid', 'N/A')
                asset_id = jid
                # 只有在 models_base_path 可用时才尝试加载 3D-FUTURE 模型
                if self.models_base_path is not None:
                    model_path = self.models_base_path / jid / 'raw_model.glb'
                else:
                    model_path = None
                if not model_path.exists():
                    model_path = None
            
            # 加载模型
            if model_path is None or not model_path.exists():
                if self.verbose:
                    print(f"Warning: Model not found for {asset_id}, using box placeholder")
                mesh = trimesh.creation.box(extents=target_size)
            else:
                loaded = trimesh.load(str(model_path))
                # 处理 trimesh.load 返回 Scene 对象的情况
                if isinstance(loaded, trimesh.Scene):
                    # 将场景中所有几何体合并为一个 mesh
                    if len(loaded.geometry) > 0:
                        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                    else:
                        if self.verbose:
                            print(f"Warning: Model file {asset_id} is empty scene, using box placeholder")
                        mesh = trimesh.creation.box(extents=target_size)
                else:
                    mesh = loaded
            
            # ==== Objaverse 坐标系校正 ====
            # Objaverse GLB 模型需要重置初始旋转（与 blender_renderer.py 保持一致）
            # 这确保物理计算与渲染结果一致
            if asset_source == 'objaverse' and model_path is not None and model_path.exists():
                # Objaverse 模型：重置到标准朝向（无需额外旋转校正）
                # 渲染时使用 rotation_euler = (0, 0, 0)，这里保持 mesh 原样即可
                # 因为 trimesh 加载的 GLB 已经是正确朝向
                if self.verbose:
                    print(f"Objaverse asset {asset_id}: using standard orientation (no correction needed)")
            
            # 1. 缩放到目标尺寸
            original_size = mesh.extents
            target_size_array = np.array(target_size)
            scale_factors = target_size_array / (original_size + 1e-6)
            mesh.apply_scale(scale_factors)
            
            # 2. 旋转
            pos = obj_data['pos']
            rot_xyzw = obj_data['rot']
            
            rotation = R.from_quat(rot_xyzw)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation.as_matrix()
            transform_matrix[:3, 3] = pos
            
            # 3. 计算底部中心作为锚点
            bounds = mesh.bounds
            bottom_center_pivot = np.array([
                (bounds[0, 0] + bounds[1, 0]) / 2,  # X轴中心
                bounds[0, 1],                        # Y轴底部
                (bounds[0, 2] + bounds[1, 2]) / 2    # Z轴中心
            ])
            
            # 移动到原点
            center_transform = np.eye(4)
            center_transform[:3, 3] = -bottom_center_pivot
            
            # 应用完整变换
            mesh.apply_transform(center_transform)
            mesh.apply_transform(transform_matrix)
            
            return mesh
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading/transforming mesh: {e}")
            return None
    
    def evaluate_scene(self,
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Dict[str, Any]:
        """
        评估场景的物理指标（完全参考myeval实现）
        
        参数:
            scene_data: 场景数据，可以是字典、JSON文件路径或Path对象
            format_type: 场景格式类型，'ours' 或 'respace'
        
        返回:
            包含以下指标的字典：
            - collision_free_rate: 无碰撞率 (%)
            - num_colliding_pairs: 碰撞对数量
            - collision_rate: 碰撞率 (%)
            - mean_penetration_depth: 平均穿透深度 (m)
            - valid_placement_rate: 有效放置率 (%)
            - num_oob_objects: 出界物体数量
            - out_of_bounds_rate: 出界率 (%)
            - mean_oob_volume: 平均出界体积 (m³)
        """
        # 加载场景数据
        if isinstance(scene_data, (str, Path)):
            with open(scene_data, 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
        else:
            scene_json = scene_data
        
        # 解析场景数据
        bounds_bottom, bounds_top, all_objects_data = self._parse_scene_data(scene_json, format_type)
        
        if len(all_objects_data) == 0:
            return {
                'collision_free_rate': 100.0,
                'num_colliding_pairs': 0,
                'collision_rate': 0.0,
                'mean_penetration_depth': 0.0,
                'valid_placement_rate': 100.0,
                'num_oob_objects': 0,
                'out_of_bounds_rate': 0.0,
                'mean_oob_volume': 0.0
            }
        
        total_objects = len(all_objects_data)
        
        # 创建房间mesh和地板多边形（支持异型房间）
        bounds_bottom_arr = np.array(bounds_bottom)
        bounds_top_arr = np.array(bounds_top)
        
        room_mesh = self._create_room_mesh(bounds_bottom, bounds_top)
        floor_polygon = self._create_floor_polygon(bounds_bottom)
        
        # 计算房间高度范围
        room_height_min = bounds_bottom_arr[:, 1].min()
        room_height_max = bounds_top_arr[:, 1].max()
        room_bounds = room_mesh.bounds
        
        # 加载并变换所有物体的mesh
        # 同时构建 name_to_meta 映射，用于生成可读的反馈
        furniture_objects = {}
        name_to_meta = {}  # 存储 name -> {idx, jid, uid, desc, asset_source} 的映射
        
        for i, obj_data in enumerate(all_objects_data):
            mesh = self._load_and_transform_mesh(obj_data, format_type)
            if mesh is not None:
                # 优先使用 uid（Objaverse），否则使用 jid（3D-FUTURE）
                uid = obj_data.get('uid', '')
                jid = obj_data.get('jid', f'object_{i}')
                desc = obj_data.get('desc', obj_data.get('category', 'unknown'))
                asset_source = obj_data.get('asset_source', '3d-future')
                
                # 使用更短但唯一的命名
                if uid:
                    obj_name = f"obj_{i}_{uid[:8]}"
                else:
                    obj_name = f"obj_{i}_{jid[:8]}"
                
                furniture_objects[obj_name] = mesh
                name_to_meta[obj_name] = {
                    'idx': i,
                    'jid': jid,
                    'uid': uid,
                    'desc': desc,
                    'asset_source': asset_source
                }
        
        if len(furniture_objects) == 0:
            return {
                'collision_free_rate': 100.0,
                'num_colliding_pairs': 0,
                'collision_rate': 0.0,
                'mean_penetration_depth': 0.0,
                'valid_placement_rate': 100.0,
                'num_oob_objects': 0,
                'out_of_bounds_rate': 0.0,
                'mean_oob_volume': 0.0
            }
        
        # ===== 碰撞检测 =====
        manager = trimesh.collision.CollisionManager()
        for name, mesh in furniture_objects.items():
            manager.add_object(name, mesh)
        
        # 获取所有碰撞数据
        is_collision, contact_data = manager.in_collision_internal(return_data=True)
        
        # 所有穿透深度 > 0 的都视为碰撞（删除容差）
        actual_collisions_data = [d for d in contact_data if d.depth > 0]
        
        num_colliding_pairs = len(actual_collisions_data)
        
        # 计算涉及碰撞的物体数量，并构建碰撞对列表
        colliding_objects = set()
        collision_pairs_raw = []  # 存储原始碰撞对信息
        
        for contact in actual_collisions_data:
            if hasattr(contact, 'names') and contact.names is not None:
                # contact.names 可能是 list, tuple, frozenset, set 等类型
                names_list = list(contact.names) if hasattr(contact.names, '__iter__') else []
                
                if len(names_list) >= 2:
                    name_a, name_b = names_list[0], names_list[1]
                    colliding_objects.add(name_a)
                    colliding_objects.add(name_b)
                    collision_pairs_raw.append({
                        'name_a': name_a,
                        'name_b': name_b,
                        'depth': contact.depth
                    })
                elif len(names_list) > 0:
                    colliding_objects.update(names_list)
        
        num_colliding_objects = len(colliding_objects)
        collision_rate = (num_colliding_objects / total_objects * 100) if total_objects > 0 else 0.0
        
        # 计算平均穿透深度
        total_penetration_depth = 0
        if num_colliding_pairs > 0:
            for contact in actual_collisions_data:
                # 累加完整的穿透深度（删除容差减法）
                total_penetration_depth += contact.depth
        
        mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0
        
        # ===== 出界检测（支持异型房间）=====
        num_oob_objects = 0
        total_oob_volume = 0
        oob_objects_raw = []  # 存储出界物体信息
        
        for name, mesh in furniture_objects.items():
            # 使用精确的多边形+高度检测
            is_oob, oob_volume = self._check_object_out_of_bounds(
                mesh, room_mesh, floor_polygon, room_height_min, room_height_max
            )
            if is_oob:
                num_oob_objects += 1
                total_oob_volume += oob_volume
                # 计算出界比例
                obj_volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else np.prod(mesh.extents)
                oob_ratio = oob_volume / obj_volume if obj_volume > 0 else 0.0
                oob_objects_raw.append({
                    'name': name,
                    'oob_volume': oob_volume,
                    'oob_ratio': min(oob_ratio, 1.0)  # 限制在 0-1
                })
        
        mean_oob_volume = (total_oob_volume / num_oob_objects) if num_oob_objects > 0 else 0
        out_of_bounds_rate = (num_oob_objects / total_objects * 100) if total_objects > 0 else 0.0
        
        # ===== 构建碰撞对列表（带元数据，按深度排序）=====
        collision_pairs = []
        for pair in sorted(collision_pairs_raw, key=lambda x: x['depth'], reverse=True):
            meta_a = name_to_meta.get(pair['name_a'], {})
            meta_b = name_to_meta.get(pair['name_b'], {})
            collision_pairs.append({
                'obj_a_desc': meta_a.get('desc', 'unknown'),
                'obj_a_id': meta_a.get('uid') or meta_a.get('jid', 'unknown'),
                'obj_b_desc': meta_b.get('desc', 'unknown'),
                'obj_b_id': meta_b.get('uid') or meta_b.get('jid', 'unknown'),
                'depth': pair['depth'],
                'depth_cm': round(pair['depth'] * 100, 1)  # 转换为厘米
            })
        
        # ===== 构建出界物体列表（带元数据，按出界比例排序）=====
        oob_objects = []
        for obj in sorted(oob_objects_raw, key=lambda x: x['oob_ratio'], reverse=True):
            meta = name_to_meta.get(obj['name'], {})
            oob_objects.append({
                'desc': meta.get('desc', 'unknown'),
                'obj_id': meta.get('uid') or meta.get('jid', 'unknown'),
                'oob_volume': obj['oob_volume'],
                'oob_ratio': obj['oob_ratio'],
                'oob_percent': round(obj['oob_ratio'] * 100, 1)  # 转换为百分比
            })
        
        # 构建结果
        metrics = {
            'collision_free_rate': 100.0 if num_colliding_pairs == 0 else 0.0,
            'num_colliding_pairs': num_colliding_pairs,
            'collision_rate': collision_rate,
            'mean_penetration_depth': mean_penetration_depth,
            'total_penetration_depth': total_penetration_depth,  # 总穿透深度（用于体积奖励）
            'valid_placement_rate': 100.0 if num_oob_objects == 0 else 0.0,
            'num_oob_objects': num_oob_objects,
            'out_of_bounds_rate': out_of_bounds_rate,
            'mean_oob_volume': mean_oob_volume,
            'total_oob_volume': total_oob_volume,  # 总出界体积（用于体积奖励）
            # 新增：碰撞对和出界物体详细列表
            'collision_pairs': collision_pairs,  # List[{obj_a_desc, obj_a_id, obj_b_desc, obj_b_id, depth, depth_cm}]
            'oob_objects': oob_objects  # List[{desc, obj_id, oob_volume, oob_ratio, oob_percent}]
        }
        
        if self.verbose:
            print(f"\nTrimesh物理指标评估结果:")
            print(f"  碰撞率: {collision_rate:.2f}%")
            print(f"  碰撞对数: {num_colliding_pairs}")
            print(f"  平均穿透深度: {mean_penetration_depth:.6f}m")
            print(f"  出界率: {out_of_bounds_rate:.2f}%")
            print(f"  出界物体: {num_oob_objects}/{total_objects}")
        
        return metrics
    
    def compute_reward(self,
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Tuple[float, Dict[str, Any]]:
        """
        基于物理指标计算奖励
        
        奖励计算策略（基于SFT基线调整）：
        
        碰撞率（SFT基线45%为零点）：
        - 碰撞率 ≤ 20%: +1.0 到 +0.5（优秀）
        - 碰撞率 20%-45%: +0.5 到 0.0（SFT基线为零点）
        - 碰撞率 > 45%: 0.0 到 -1.0（比SFT差）
        
        出界率（SFT基线30%为零点）：
        - 出界率 ≤ 10%: +1.0 到 +0.5（优秀）
        - 出界率 10%-30%: +0.5 到 0.0（SFT基线为零点）
        - 出界率 > 30%: 0.0 到 -1.0（比SFT差）
        
        最终奖励 = (碰撞奖励 + 出界奖励) / 2
        
        参数:
            scene_data: 场景数据
            format_type: 场景格式类型
        
        返回:
            (reward, metrics) 元组
        """
        metrics = self.evaluate_scene(scene_data, format_type)
        
        collision_rate = metrics['collision_rate']
        oob_rate = metrics['out_of_bounds_rate']
        
        # 碰撞奖励（基于SFT基线45%调整）
        if collision_rate <= 20:
            # 优秀区间：0% -> +1.0, 20% -> +0.5
            collision_reward = 1.0 - 0.5 * (collision_rate / 20.0)
        elif collision_rate <= 45:
            # 良好区间（SFT基线为零点）：20% -> +0.5, 45% -> 0.0
            collision_reward = 0.5 - 0.5 * (collision_rate - 20) / 25.0
        else:
            # 差于SFT：45% -> 0.0, 100% -> -1.0
            collision_reward = -1.0 * (collision_rate - 45) / 55.0
        
        # 出界奖励（基于SFT基线30%调整）
        if oob_rate <= 10:
            # 优秀区间：0% -> +1.0, 10% -> +0.5
            oob_reward = 1.0 - 0.5 * (oob_rate / 10.0)
        elif oob_rate <= 30:
            # 良好区间（SFT基线为零点）：10% -> +0.5, 30% -> 0.0
            oob_reward = 0.5 - 0.5 * (oob_rate - 10) / 20.0
        else:
            # 差于SFT：30% -> 0.0, 100% -> -1.0
            oob_reward = -1.0 * (oob_rate - 30) / 70.0
        
        # 综合奖励
        reward = (collision_reward + oob_reward) / 2.0
        
        if self.verbose:
            print(f"\n奖励计算:")
            print(f"  碰撞奖励: {collision_reward:.4f} (碰撞率: {collision_rate:.2f}%)")
            print(f"  出界奖励: {oob_reward:.4f} (出界率: {oob_rate:.2f}%)")
            print(f"  综合奖励: {reward:.4f}")
        
        return reward, metrics


def generate_physics_feedback(metrics: Dict[str, Any], top_k: int = 3) -> str:
    """
    根据物理评估指标生成简洁的反馈文本
    
    参数:
        metrics: TrimeshPhysicsMetrics.evaluate_scene 返回的指标字典
        top_k: 最多报告的碰撞对/出界物体数量
    
    返回:
        物理反馈文本，如果没有问题则返回空字符串
    """
    feedback_parts = []
    
    # 处理碰撞问题
    collision_pairs = metrics.get('collision_pairs', [])
    if collision_pairs:
        # 按穿透深度排序，取最严重的top_k个
        sorted_collisions = sorted(collision_pairs, key=lambda x: x.get('depth', 0), reverse=True)[:top_k]
        collision_strs = []
        for cp in sorted_collisions:
            obj_a = cp.get('obj_a_desc') or cp.get('obj_a_id', 'unknown')
            obj_b = cp.get('obj_b_desc') or cp.get('obj_b_id', 'unknown')
            depth_cm = cp.get('depth_cm', 0)
            collision_strs.append(f"'{obj_a}' and '{obj_b}' (depth: {depth_cm:.1f}cm)")
        
        if len(collision_pairs) > top_k:
            collision_strs.append(f"and {len(collision_pairs) - top_k} more")
        
        feedback_parts.append(f"Collisions: {'; '.join(collision_strs)}. Separate these objects to avoid overlap.")
    
    # 处理出界问题
    oob_objects = metrics.get('oob_objects', [])
    if oob_objects:
        # 按出界比例排序，取最严重的top_k个
        sorted_oob = sorted(oob_objects, key=lambda x: x.get('oob_ratio', 0), reverse=True)[:top_k]
        oob_strs = []
        for obj in sorted_oob:
            desc = obj.get('desc') or obj.get('obj_id', 'unknown')
            oob_pct = obj.get('oob_percent', 0)
            oob_strs.append(f"'{desc}' ({oob_pct:.0f}% outside)")
        
        if len(oob_objects) > top_k:
            oob_strs.append(f"and {len(oob_objects) - top_k} more")
        
        feedback_parts.append(f"Out of bounds: {'; '.join(oob_strs)}. Move these objects inside the room boundaries.")
    
    if not feedback_parts:
        return ""
    
    return "[Physics] " + " ".join(feedback_parts)
