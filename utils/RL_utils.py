#!/usr/bin/env python3
"""
RL_utils.py - Reinforcement Learning Utility Module
Provides wrapper functions for scene rendering and visualization during RL training
"""

import os
import sys
import json
import shutil
import traceback
import threading
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple

# Import PIL for image processing
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, image processing will be limited")

# Import voxel evaluation dependencies
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
# Scene Format Conversion Functions
# Convert between flat format (direct objects array) and grouped format (groups containing objects)
# ============================================================================

def convert_flat_to_grouped(scene: Dict[str, Any]) -> Dict[str, Any]:
    """Convert scene format without groups to grouped format
    
    Args:
        scene: Scene data without groups, containing a direct objects array
        
    Returns:
        Scene data with groups
    """
    # If groups field already exists, return directly
    if 'groups' in scene:
        return scene
    
    # If objects field doesn't exist, return directly
    if 'objects' not in scene:
        return scene
    
    # Create new scene data, preserving original room information
    grouped_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # Handle room_envelope or bounds fields
    if 'room_envelope' in scene:
        grouped_scene['room_envelope'] = scene['room_envelope']
    elif 'bounds_top' in scene and 'bounds_bottom' in scene:
        # Convert old format bounds to room_envelope
        grouped_scene['room_envelope'] = {
            'bounds_top': scene['bounds_top'],
            'bounds_bottom': scene['bounds_bottom']
        }
    
    # Put all objects into a single default group
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
    """Convert grouped scene format back to flat format without groups
    
    Args:
        scene: Scene data with groups
        
    Returns:
        Scene data without groups, containing a direct objects array
    """
    # If groups field doesn't exist, return directly
    if 'groups' not in scene:
        return scene
    
    # Create new scene data, preserving original room information
    flat_scene = {
        'room_type': scene.get('room_type', 'unknown'),
        'room_id': scene.get('room_id', 'room_001'),
    }
    
    # Handle room_envelope or bounds fields
    if 'room_envelope' in scene:
        # Extract bounds to top level
        flat_scene['bounds_top'] = scene['room_envelope'].get('bounds_top', [])
        flat_scene['bounds_bottom'] = scene['room_envelope'].get('bounds_bottom', [])
    
    # Collect all objects from all groups into a single array
    all_objects = []
    for group in scene.get('groups', []):
        all_objects.extend(group.get('objects', []))
    
    flat_scene['objects'] = all_objects
    
    print(f"Converted grouped format to flat format: {len(scene.get('groups', []))} groups → {len(all_objects)} objects")
    return flat_scene


# ============================================================================
# Global Singleton Pattern and Thread Locks
# Used to solve thread safety issues in multi-threaded environments
# ============================================================================

# AssetRetrievalModule global singleton (resolves CLIP model thread safety) - for 3D-FUTURE
_GLOBAL_ASSET_RETRIEVAL = None
_ASSET_RETRIEVAL_LOCK = threading.Lock()

# ObjaverseRetriever global singleton - for Objaverse
_GLOBAL_OBJAVERSE_RETRIEVAL = None
_OBJAVERSE_RETRIEVAL_LOCK = threading.Lock()

# Blender render global lock (resolves Blender process concurrency conflicts)
_BLENDER_RENDER_LOCK = threading.Lock()


def _get_global_objaverse_retrieval(params: Optional[Dict[str, Any]] = None):
    """
    Get global ObjaverseRetriever singleton
    Thread-safe, ensures initialization only once per process
    
    Args:
        params: ObjaverseRetriever initialization parameters
    
    Returns:
        ObjaverseRetriever instance, or None if initialization fails
    """
    global _GLOBAL_OBJAVERSE_RETRIEVAL
    
    # Fast path: if already initialized, return directly
    if _GLOBAL_OBJAVERSE_RETRIEVAL is not None:
        return _GLOBAL_OBJAVERSE_RETRIEVAL
    
    # Lock for initialization
    with _OBJAVERSE_RETRIEVAL_LOCK:
        # Double-check: may have been initialized by another thread while waiting for lock
        if _GLOBAL_OBJAVERSE_RETRIEVAL is not None:
            return _GLOBAL_OBJAVERSE_RETRIEVAL
        
        try:
            # Ensure current directory is in sys.path
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            from objaverse_retriever import ObjaverseRetriever
            
            # Default parameters
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
    Get global AssetRetrievalModule singleton
    Thread-safe, ensures initialization only once per process
    
    Args:
        params: AssetRetrievalModule initialization parameters
    
    Returns:
        AssetRetrievalModule instance, or None if initialization fails
    """
    global _GLOBAL_ASSET_RETRIEVAL
    
    # Fast path: if already initialized, return directly
    if _GLOBAL_ASSET_RETRIEVAL is not None:
        return _GLOBAL_ASSET_RETRIEVAL
    
    # Lock for initialization
    with _ASSET_RETRIEVAL_LOCK:
        # Double-check: may have been initialized by another thread while waiting for lock
        if _GLOBAL_ASSET_RETRIEVAL is not None:
            return _GLOBAL_ASSET_RETRIEVAL
        
        try:
            # Ensure current directory is in sys.path
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            # Set environment variables required by AssetRetrievalModule
            project_root = Path(__file__).parent.parent  # llmscene directory
            
            # Define possible metadata paths
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
            
            # Find existing files
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
            
            # Check if all files are found
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
            
            # Set environment variables
            os.environ['PTH_ASSETS_METADATA'] = metadata_file
            os.environ['PTH_ASSETS_METADATA_SCALED'] = metadata_scaled_file
            os.environ['PTH_ASSETS_EMBED'] = embed_file
            
            # Import AssetRetrievalModule
            from sample import AssetRetrievalModule
            
            # Default parameters
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
            
            # Merge user parameters
            if params:
                default_params.update(params)
            
            # Initialize global singleton
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
    Scene Renderer class
    Encapsulates the complete pipeline from JSON scene data to rendered images
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
        Initialize scene renderer
        
        Args:
            asset_path: Path to 3D asset directory (3D-FUTURE-model)
            use_placeholder: Whether to use placeholder rendering (without loading real 3D models)
            verbose: Whether to output detailed logs
            temp_dir: Temporary file directory, uses system default if None
            use_render_lock: Whether to use global render lock (recommended for multi-threaded environments)
            fast_mode: Fast rendering mode (512x512, 16 samples, ~2-3x speedup)
            enable_visualization: Whether to enable 3D helper visualization (bbox, arrows, coordinate grid)
        """
        self.verbose = verbose
        self.use_placeholder = use_placeholder
        self.temp_dir = temp_dir or './temp_render'
        self.use_render_lock = use_render_lock
        self.fast_mode = fast_mode
        self.enable_visualization = enable_visualization
        
        # Set up asset path
        self.asset_path = asset_path or self._find_asset_path()
        if self.asset_path:
            os.environ['PTH_3DFUTURE_ASSETS'] = self.asset_path
            if self.verbose:
                print(f"Using asset path: {self.asset_path}")
        
        # Set environment variables
        os.environ['BPY_VERBOSE'] = '1' if verbose else '0'
        os.environ['BPY_USE_PLACEHOLDER_ONLY'] = '1' if use_placeholder else '0'
        os.environ['BPY_FAST_MODE'] = '1' if fast_mode else '0'
        os.environ['BPY_ENABLE_VISUALIZATION'] = '1' if enable_visualization else '0'
        
        if self.verbose and fast_mode:
            print("✓ Fast rendering mode enabled (512x512, 16 samples, ~2-3x speedup)")
        if self.verbose and enable_visualization:
            print("✓ 3D visualization enabled (bbox, arrows, coordinate grid with labels)")
        
        # Try to import rendering function
        self.render_func = self._import_render_function()
        self.merge_func = self._import_merge_function()
        
    def _find_asset_path(self) -> Optional[str]:
        """Find 3D asset directory, prioritizing PathConfig"""
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
        
        # Try common asset paths
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
        """Import Blender rendering function"""
        try:
            # Add current directory to sys.path (since file is in utils directory)
            utils_path = os.path.dirname(__file__)
            if utils_path not in sys.path:
                sys.path.append(utils_path)
            
            # Import Blender wrapper
            from blender_wrapper import render_scene_blender_external
            if self.verbose:
                print("✓ Successfully imported Blender external rendering wrapper")
            return render_scene_blender_external
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not import Blender renderer: {e}")
            return None
    
    def _import_merge_function(self):
        """Import image merge function"""
        try:
            import importlib.util
            # File is in utils directory, directly reference image_merger.py in the same directory
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
        Render scene and return image path or image object (thread-safe wrapper)
        
        Args:
            scene_data: Scene JSON data (dict) or JSON file path
            output_path: Output image path, auto-generated if None
            scene_id: Scene ID, used for naming and logging
            return_image: Whether to return a PIL Image object (not just the path)
            cleanup_temp: Whether to clean up temporary files
            
        Returns:
            If return_image=False: Returns image path (str)
            If return_image=True: Returns (image path, PIL.Image object) tuple
        """
        if self.use_render_lock:
            # Use global lock to protect Blender rendering (multi-threaded environment)
            with _BLENDER_RENDER_LOCK:
                return self._render_scene_impl(scene_data, output_path, scene_id, return_image, cleanup_temp)
        else:
            # No lock (single-threaded environment or debug mode)
            return self._render_scene_impl(scene_data, output_path, scene_id, return_image, cleanup_temp)
    
    def _render_scene_impl(self, 
                          scene_data: Union[Dict[str, Any], str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          scene_id: str = "scene",
                          return_image: bool = False,
                          cleanup_temp: bool = True) -> Union[str, Tuple]:
        """
        Actual implementation of scene rendering (internal method)
        
        Args:
            scene_data: Scene JSON data (dict) or JSON file path
            output_path: Output image path, auto-generated if None
            scene_id: Scene ID, used for naming and logging
            return_image: Whether to return a PIL Image object (not just the path)
            cleanup_temp: Whether to clean up temporary files
            
        Returns:
            If return_image=False: Returns image path (str)
            If return_image=True: Returns (image path, PIL.Image object) tuple
        """
        try:
            # Parse scene data
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
            
            # Set output path
            if output_path is None:
                output_path = Path(self.temp_dir) / f"{scene_id}_rendered.png"
            else:
                output_path = Path(output_path)
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary render directory (ensure uniqueness in multi-threaded environment)
            import time
            thread_id = threading.get_ident()
            timestamp = int(time.time() * 1000000)  # Microsecond-level timestamp
            temp_render_dir = Path(self.temp_dir) / f"render_{scene_id}_{thread_id}_{timestamp}"
            temp_render_dir.mkdir(parents=True, exist_ok=True)
            
            # Execute Blender rendering
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
            
            # Find generated image files
            top_file = temp_render_dir / "top" / "frame.png"
            diag_file = temp_render_dir / "diag" / "frame.png"
            
            # Merge images or create placeholder
            if top_file.exists() and diag_file.exists() and self.merge_func:
                try:
                    # Merge top-down view and diagonal view
                    self.merge_func(str(top_file), str(diag_file), str(output_path))
                    
                    if self.verbose:
                        print(f"✓ Rendered and merged image saved to: {output_path}")
                    
                    # Clean up temporary files
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
                    # Continue to placeholder creation
                    output_path = self._create_placeholder_image(scene_dict, output_path, scene_id)
            else:
                # Create placeholder image
                if self.verbose:
                    print("Warning: Rendered images not found, creating placeholder")
                output_path = self._create_placeholder_image(scene_dict, output_path, scene_id)
            
            # Return results
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
            
            # Return placeholder
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
        """Create placeholder image"""
        try:
            if not PIL_AVAILABLE:
                if self.verbose:
                    print("Error: PIL not available, cannot create placeholder")
                return output_path
            
            # Create placeholder image
            img = Image.new('RGB', (1024, 512), color='lightgray')
            draw = ImageDraw.Draw(img)
            
            # Count objects
            objects_count = 0
            if 'groups' in scene_dict:
                objects_count = sum(len(group.get('objects', [])) 
                                  for group in scene_dict.get('groups', []))
            elif 'objects' in scene_dict:
                objects_count = len(scene_dict.get('objects', []))
            
            # Get room type
            room_type = scene_dict.get('room_type', 'Unknown')
            
            # Try to load font
            font = None
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                pass
            
            # Draw information
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
            
            # Save placeholder image
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


# Convenience function: quickly render a scene
def render_scene_quick(scene_data: Union[Dict, str, Path],
                      output_path: Optional[str] = None,
                      return_image: bool = False,
                      verbose: bool = False,
                      fast_mode: bool = True) -> Union[str, Tuple]:
    """
    Convenience function for quick scene rendering
    
    Args:
        scene_data: Scene JSON data (dict) or JSON file path
        output_path: Output image path, auto-generated if None
        return_image: Whether to return a PIL Image object
        verbose: Whether to output detailed logs
        fast_mode: Fast rendering mode (512x512, 16 samples, ~2-3x speedup)
        
    Returns:
        If return_image=False: Returns image path (str)
        If return_image=True: Returns (image path, PIL.Image object) tuple
        
    Examples:
        >>> # Render JSON file (fast mode)
        >>> img_path = render_scene_quick("./scene.json")
        >>> 
        >>> # Render JSON dict and get image object
        >>> scene_dict = {...}
        >>> img_path, img = render_scene_quick(scene_dict, return_image=True)
        >>> 
        >>> # High quality mode
        >>> img_path = render_scene_quick(scene_dict, output_path="./my_scene.png", fast_mode=False)
    """
    renderer = SceneRenderer(verbose=verbose, fast_mode=fast_mode)
    return renderer.render_scene(
        scene_data=scene_data,
        output_path=output_path,
        return_image=return_image
    )


# Convenience function: batch render multiple scenes
def render_scenes_batch(scene_data_list: list,
                       output_dir: Union[str, Path],
                       scene_ids: Optional[list] = None,
                       return_images: bool = False,
                       verbose: bool = False) -> list:
    """
    Batch render multiple scenes
    
    Args:
        scene_data_list: List of scene data (each element is a dict or file path)
        output_dir: Output directory
        scene_ids: List of scene IDs, auto-generated if None
        return_images: Whether to return image objects
        verbose: Whether to output detailed logs
        
    Returns:
        List of render results, each element is an image path or (path, image) tuple
        
    Examples:
        >>> scenes = [scene1_dict, scene2_dict, scene3_dict]
        >>> results = render_scenes_batch(scenes, "./output")
        >>> for img_path in results:
        ...     print(f"Rendered: {img_path}")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create renderer
    renderer = SceneRenderer(verbose=verbose)
    
    # Generate scene IDs
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


# Convenience function: load from file and render
def render_scene_from_file(json_file_path: Union[str, Path],
                          output_path: Optional[Union[str, Path]] = None,
                          return_image: bool = False,
                          verbose: bool = False) -> Union[str, Tuple]:
    """
    Load scene from JSON file and render
    
    Args:
        json_file_path: Path to scene JSON file
        output_path: Output image path; if None, uses the same directory as the JSON file
        return_image: Whether to return a PIL Image object
        verbose: Whether to output detailed logs
        
    Returns:
        If return_image=False: returns the image path (str)
        If return_image=True: returns a (image_path, PIL.Image) tuple
        
    Examples:
        >>> img_path = render_scene_from_file("./scenes/livingroom.json")
        >>> img_path, img = render_scene_from_file("./scenes/bedroom.json", return_image=True)
    """
    json_path = Path(json_file_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Scene file not found: {json_path}")
    
    # If no output path specified, use same directory as JSON file
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
    Test and example code
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
        
        # Create sample scene
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
        
        # Render test scene
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
        # Render the specified scene file
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
# Scene Editing Features
# ============================================================================

class SceneEditor:
    """
    Scene Editor Class
    Encapsulates the complete pipeline for scene editing and asset retrieval
    
    Supports two asset sources:
    - 3D-FUTURE: Default, uses AssetRetrievalModule
    - Objaverse: Uses ObjaverseRetriever, requires use_objaverse=True
    """
    
    def __init__(self, 
                 asset_retrieval_params: Optional[Dict[str, Any]] = None,
                 use_objaverse: bool = True,
                 verbose: bool = False):
        """
        Initialize scene editor
        
        Args:
            asset_retrieval_params: Asset retrieval module parameters, uses defaults if None
            use_objaverse: Whether to use Objaverse asset retrieval (default False, uses 3D-FUTURE)
            verbose: Whether to output detailed logs
        """
        self.verbose = verbose
        self.use_objaverse = use_objaverse
        
        # Import scene editor module
        self._import_scene_editor()
        
        # Select asset retrieval module based on use_objaverse
        if use_objaverse:
            # Use Objaverse retriever
            self.asset_retrieval_module = _get_global_objaverse_retrieval(asset_retrieval_params)
            if self.verbose and self.asset_retrieval_module:
                print("✓ Using ObjaverseRetriever for asset retrieval")
        else:
            # Use 3D-FUTURE retriever (default)
            self.asset_retrieval_module = _get_global_asset_retrieval(asset_retrieval_params)
            if self.verbose and self.asset_retrieval_module:
                print("✓ Using AssetRetrievalModule (3D-FUTURE) for asset retrieval")
    
    def _import_scene_editor(self):
        """Import scene editor module"""
        try:
            # Add current directory to sys.path (since file is in utils directory)
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
        Edit scene: apply tool_calls and perform asset retrieval
        
        Args:
            scene_data: Initial scene data (dict) or JSON file path
            tool_calls: List of tool calls, each containing name and arguments
            retrieve_assets: Whether to perform asset retrieval
            output_path: Output scene JSON file path (optional)
            
        Returns:
            Edited scene data
            
        Examples:
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
            # Parse scene data
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
            
            # 1. Apply tool calls
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
            
            # 2. Asset retrieval
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
            
            # 3. Save to file (if output path is specified)
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


# Convenience function: quickly edit a scene
def edit_scene_quick(scene_data: Union[Dict, str, Path],
                    tool_calls: list,
                    retrieve_assets: bool = True,
                    use_objaverse: bool = False,
                    output_path: Optional[str] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Convenience function for quick scene editing
    
    Args:
        scene_data: Scene data (dict) or JSON file path
        tool_calls: List of tool calls
        retrieve_assets: Whether to perform asset retrieval
        use_objaverse: Whether to use Objaverse asset retrieval (default False, uses 3D-FUTURE)
        output_path: Output JSON file path (optional)
        verbose: Whether to output detailed logs
        
    Returns:
        Edited scene data
        
    Examples:
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


# Convenience function: edit and render scene
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
    Edit scene and render to image (thread-safe mode)
    
    Thread safety guarantees:
    - AssetRetrievalModule uses global singleton (single CLIP model instance per process)
    - Blender rendering uses global lock (serialized subprocess calls)
    - Temporary directories use thread_id + timestamp to ensure uniqueness
    
    Args:
        scene_data: Initial scene data
        tool_calls: List of tool calls
        output_dir: Output directory
        scene_id: Scene ID
        retrieve_assets: Whether to perform asset retrieval
        use_objaverse: Whether to use Objaverse asset retrieval (default False, uses 3D-FUTURE)
        return_image: Whether to return a PIL Image object
        verbose: Whether to output detailed logs
        fast_mode: Fast rendering mode (512x512, 16 samples, ~2-3x speedup)
        
    Returns:
        If return_image=False: (edited scene data, render image path, is_terminated) tuple
        If return_image=True: (edited scene data, render image path, PIL.Image object, is_terminated) tuple
        where is_terminated is True if tool_calls contain a "terminate" call
        
    Examples:
        >>> tool_calls = [{"name": "add_object", "arguments": {...}}]
        >>> # Return path only (fast mode)
        >>> new_scene, img_path, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", fast_mode=True
        ... )
        >>> # Use Objaverse assets
        >>> new_scene, img_path, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", use_objaverse=True
        ... )
        >>> # Return path and image object
        >>> new_scene, img_path, img, is_terminated = edit_and_render_scene(
        ...     scene_dict, tool_calls, "./output", return_image=True
        ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if there is a terminate call
    is_terminated = any(call.get('name') == 'terminate' for call in tool_calls)
    
    # 1. Edit scene
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
    
    # 2. Render scene (using fast mode with 3D visualization enabled)
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
    Find the path to an Objaverse GLB file.
    
    Prioritizes PathConfig unified configuration, then searches in the following order:
    1. Cache directory configured by PathConfig
    2. Path specified by OBJAVERSE_GLB_CACHE_DIR environment variable
    3. /path/to/datasets/objathor-assets/glbs (cloud storage)
    4. ~/.objaverse/hf-objaverse-v1/glbs (local cache)
    
    Args:
        uid: Objaverse asset UID
        
    Returns:
        GLB file path, or None if not found
    """
    import os
    
    if not uid or len(uid) < 2:
        return None
    
    # GLB cache directories list (sorted by priority)
    cache_dirs = []
    
    # Try PathConfig first
    try:
        from path_config import PathConfig
        config = PathConfig.get_instance()
        if config.objaverse_glb_cache_dir:
            cache_dirs.append(Path(config.objaverse_glb_cache_dir))
    except ImportError:
        pass
    
    # Add path specified by environment variable
    env_cache = os.environ.get("OBJAVERSE_GLB_CACHE_DIR")
    if env_cache:
        cache_dirs.append(Path(env_cache) / "glbs")
    
    # Add cloud storage path
    cache_dirs.append(Path("/path/to/datasets/objathor-assets/glbs"))
    
    # Add local cache
    cache_dirs.append(Path.home() / ".objaverse" / "hf-objaverse-v1" / "glbs")
    
    # Use uid[:2] to directly build the path (optimization: avoid traversing all subdirectories)
    subdir_name = uid[:2]
    
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        
        # Directly find GLB file in uid[:2] subdirectory
        candidate = cache_dir / subdir_name / f"{uid}.glb"
        if candidate.is_file():
            return candidate
    
    return None


class VoxelReward:
    """
    Voxel-based reward computation class
    Uses voxelization to evaluate scene physical plausibility (out-of-bounds and collisions),
    and computes reinforcement learning rewards
    """
    
    def __init__(self, 
                 models_base_path: str,
                 voxel_size: float = 0.05,
                 reward_threshold: float = 1e-5,
                 verbose: bool = False):
        """
        Initialize voxel reward calculator
        
        Args:
            models_base_path: Base path for 3D model files (3D-FUTURE-model directory)
            voxel_size: Voxel size in meters (default 0.05m = 5cm)
            reward_threshold: PBL loss threshold, positive reward given below this value (default 1e-5)
            verbose: Whether to output detailed logs
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
        Parse scene JSON data, supports two formats
        
        Args:
            scene_json: Scene JSON data
            format_type: 'ours' or 'respace'
        
        Returns:
            bounds_bottom, bounds_top, all_objects
        """
        if format_type == 'respace':
            bounds_bottom = scene_json.get('bounds_bottom', [])
            bounds_top = scene_json.get('bounds_top', [])
            all_objects = scene_json.get('objects', [])
        else:
            # ours format: uses room_envelope and groups structure
            if 'room_envelope' not in scene_json:
                raise ValueError("Scene JSON missing 'room_envelope' field")
            
            bounds_bottom = scene_json['room_envelope']['bounds_bottom']
            bounds_top = scene_json['room_envelope']['bounds_top']
            all_objects = []
            if 'groups' in scene_json and scene_json['groups'] is not None:
                for group in scene_json['groups']:
                    if 'objects' in group and group['objects'] is not None:
                        all_objects.extend(group['objects'])
        
        return bounds_bottom, bounds_top, all_objects
    
    def _create_floor_plan_polygon(self, bounds_bottom):
        """Create floor polygon from room bottom boundary"""
        points = [(pt[0], pt[2]) for pt in bounds_bottom]
        return Polygon(points)
    
    def _create_room_mesh(self, bounds_bottom, bounds_top, floor_plan_polygon):
        """Create room mesh"""
        num_verts = len(bounds_bottom)
        all_vertices = np.array(bounds_bottom + bounds_top)
        
        # Use trimesh to create floor triangles
        vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_plan_polygon, engine="triangle")
        # Remove invalid faces
        idxs = []
        for i, row in enumerate(floor_faces):
            if np.any(row == num_verts):
                idxs.append(i)
        floor_faces = np.delete(floor_faces, idxs, axis=0)
        
        # Create ceiling faces
        ceiling_faces = floor_faces + num_verts
        
        # Create side faces
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
        Prepare voxel representation of an object
        Key: rotate mesh first, then voxelize, using center-bottom as anchor
        
        Supports two asset sources:
        - 3D-FUTURE: Uses 'jid' field, loads from models_base_path/{jid}/raw_model.glb
        - Objaverse: Uses 'uid' field, loads from GLB cache
        """
        # Determine asset source and model path
        asset_source = obj.get('asset_source', '3d-future')
        model_path = None
        asset_id = None
        
        if asset_source == 'objaverse':
            # Objaverse asset: use uid and GLB cache
            uid = obj.get('uid')
            if uid:
                asset_id = uid
                model_path = _find_objaverse_glb(uid)
                if model_path and self.verbose:
                    print(f"Found Objaverse GLB for {uid[:16]}...: {model_path}")
        else:
            # 3D-FUTURE asset: use jid
            jid = obj.get('jid', 'N/A')
            asset_id = jid
            model_path = self.models_base_path / jid / 'raw_model.glb'
            if not model_path.exists():
                model_path = None
        
        if model_path is None or not model_path.exists():
            # Use bounding box as placeholder mesh
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
                    print(f"Failed to load mesh {asset_id}: {e}")
                mesh = trimesh.creation.box(extents=obj['size'])
        
        # ==== Objaverse coordinate system correction ====
        # Objaverse GLB models need initial rotation reset (consistent with blender_renderer.py)
        # This ensures physics calculations match rendering results
        if asset_source == 'objaverse' and model_path is not None:
            # Objaverse model: reset to standard orientation (no extra rotation correction needed)
            # Rendering uses rotation_euler = (0, 0, 0), so keep mesh as-is
            # Because trimesh-loaded GLB is already in the correct orientation
            if self.verbose:
                print(f"Objaverse asset {asset_id}: using standard orientation (no correction needed)")
        
        # 1. Apply scaling to target size
        original_size = mesh.extents
        target_size = obj['size']
        scale_factors = target_size / (original_size + 1e-6)
        mesh.apply_scale(scale_factors)
        
        # 2. Apply rotation (without translation)
        rot_xyzw = obj['rot']
        rotation = R.from_quat(rot_xyzw)
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation.as_matrix()
        mesh.apply_transform(rotation_matrix)
        
        # 3. Voxelize the rotated mesh
        try:
            asset_voxels = mesh.voxelized(pitch=voxel_size).fill()
            asset_voxel_matrix = asset_voxels.matrix
        except Exception as e:
            if self.verbose:
                print(f"Voxelization failed for {obj.get('desc', 'Unknown')}: {e}")
            return None, None, mesh
        
        # 4. Compute position offset in voxel space
        pos = obj['pos']
        asset_pos_voxels = np.floor(np.array(pos) / voxel_size)
        
        # Anchor point of object voxel matrix: X-axis center, Y-axis bottom, Z-axis center
        asset_start_voxels = np.array([
            asset_voxel_matrix.shape[0] // 2,
            0,
            asset_voxel_matrix.shape[2] // 2
        ])
        
        # Compute offset from origin
        asset_shift_from_origin = asset_pos_voxels - asset_start_voxels
        
        return asset_voxel_matrix, asset_shift_from_origin, mesh
    
    def _occupancy_overlap(self, voxel_matrix_a, voxel_matrix_b, offset_b):
        """Compute overlap between two voxel matrices"""
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
        """Compute voxel-based out-of-bounds volume"""
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
        """Compute voxel-based collision volume"""
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
        Evaluate the physical plausibility of a scene
        
        Args:
            scene_data: Scene data, can be a dict, JSON file path, or Path object
            format_type: Scene format type, 'ours' or 'respace'
        
        Returns:
            Dictionary containing evaluation metrics:
            - total_oob_loss: Total out-of-bounds volume loss
            - total_mbl_loss: Total collision volume loss (Mesh-Based Loss)
            - total_pbl_loss: Total physics loss (Physics-Based Loss = OOB + MBL)
            - num_oob_objects: Number of out-of-bounds objects
            - num_collision_pairs: Number of colliding object pairs
            - voxel_size: Voxel size used
        """
        # Load scene data
        if isinstance(scene_data, (str, Path)):
            with open(scene_data, 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
        else:
            scene_json = scene_data
        
        # Extract scene data
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
        
        # Create room mesh and voxelize
        room_mesh = self._create_room_mesh(bounds_bottom, bounds_top, floor_plan_polygon)
        room_voxels = room_mesh.voxelized(pitch=self.voxel_size).fill()
        room_voxel_matrix = room_voxels.matrix
        room_size_voxels = np.ceil(abs(room_mesh.bounds[0] - room_mesh.bounds[1]) / self.voxel_size)
        room_origin_shift = np.array([room_size_voxels[0] / 2.0, 0, room_size_voxels[2] / 2.0])
        
        voxel_volume = self.voxel_size ** 3
        
        # Compute metrics
        voxel_oobs = []
        voxel_collisions = []
        
        for i, obj in enumerate(all_objects):
            # Voxel out-of-bounds
            voxel_oob = self._compute_voxel_oob(obj, room_origin_shift, room_voxel_matrix, voxel_volume)
            voxel_oobs.append(voxel_oob)
            
            # Collisions with other objects
            for j, other_obj in enumerate(all_objects[i+1:], i+1):
                voxel_collision = self._compute_voxel_collision(obj, other_obj, voxel_volume)
                voxel_collisions.append(voxel_collision)
        
        # Aggregate metrics
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
            print(f"\nScene evaluation results:")
            print(f"  OOB loss: {total_oob_loss:.6f}")
            print(f"  MBL loss: {total_mbl_loss:.6f}")
            print(f"  PBL loss: {total_pbl_loss:.6f}")
            print(f"  OOB objects: {metrics['num_oob_objects']}/{len(all_objects)}")
            print(f"  Collision pairs: {metrics['num_collision_pairs']}")
        
        return metrics
    
    def compute_reward(self, 
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Tuple[float, Dict[str, Any]]:
        """
        Compute reinforcement learning reward for the scene
        
        Args:
            scene_data: Scene data, can be a dict, JSON file path, or Path object
            format_type: Scene format type, 'ours' or 'respace'
        
        Returns:
            (reward, metrics) tuple:
            - reward: Reward value
              - PBL loss > 0.1: -1.0
              - PBL loss in [1e-5, 0.1]: linear interpolation from -1.0 to +1.0
              - PBL loss < 1e-5: +1.0
            - metrics: Detailed evaluation metrics dictionary
        """
        # Evaluate scene
        metrics = self.evaluate_scene(scene_data, format_type)
        
        # Compute reward based on PBL loss (piecewise linear)
        pbl_loss = metrics['total_pbl_loss']
        
        if pbl_loss < self.reward_threshold:
            # Very good: PBL loss < 1e-5
            reward = 1.0
        elif pbl_loss <= 0.1:
            # Medium: linear interpolation in [1e-5, 0.1]
            # reward = -1.0 + 2.0 * (0.1 - pbl_loss) / (0.1 - 1e-5)
            # Simplified: pbl_loss grows from 1e-5 to 0.1, reward goes from 1.0 to -1.0
            reward = 1.0 - 2.0 * (pbl_loss - self.reward_threshold) / (0.1 - self.reward_threshold)
        else:
            # Poor: PBL loss > 0.1
            reward = -1.0
        
        if self.verbose:
            print(f"\nReward calculation:")
            print(f"  PBL loss: {pbl_loss:.6e}")
            print(f"  Threshold: {self.reward_threshold:.6e}")
            print(f"  Reward: {reward:.4f}")
        
        return reward, metrics


# ========================================
# Trimesh-based Physics Metrics (reference myeval implementation)
# ========================================

class TrimeshPhysicsMetrics:
    """
    Trimesh collision detection-based physics metrics calculation class.
    Fully references myeval.py implementation, using trimesh.collision.CollisionManager
    instead of voxelization methods.
    
    This version supports irregular rooms (L-shaped, T-shaped, etc.) for precise
    out-of-bounds detection, and removes collision detection tolerance.
    """
    
    def __init__(self,
                 models_base_path: str = None,
                 verbose: bool = False):
        """
        Initialize Trimesh physics metrics calculator
        
        Args:
            models_base_path: Base path for 3D model files (3D-FUTURE-model directory)
                             Can be None when using Objaverse assets
            verbose: Whether to output detailed logs
        """
        self.verbose = verbose
        
        # models_base_path is now optional
        # Only needed when using 3D-FUTURE assets
        if models_base_path:
            self.models_base_path = Path(models_base_path)
            if not self.models_base_path.exists():
                # Changed to warning instead of error, allowing continued operation in Objaverse mode
                if self.verbose:
                    print(f"Warning: Models base path does not exist: {self.models_base_path}")
                    print(f"  3D-FUTURE assets will use placeholder boxes")
                # Set to None to indicate unavailable
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
        Create floor polygon from room bottom bounds (supports irregular rooms)
        
        Args:
            bounds_bottom: List of bottom boundary vertices [[x1,y1,z1], [x2,y2,z2], ...]
        
        Returns:
            shapely.geometry.Polygon: Floor polygon (using X and Z coordinates)
        """
        from shapely.geometry import Polygon
        # Use X and Z coordinates to create 2D polygon (Y is the height axis)
        points = [(pt[0], pt[2]) for pt in bounds_bottom]
        return Polygon(points)
    
    def _create_room_mesh(self, bounds_bottom, bounds_top):
        """
        Create actual room mesh from polygon boundaries (supports irregular rooms)
        
        Args:
            bounds_bottom: List of bottom boundary vertices
            bounds_top: List of top boundary vertices
        
        Returns:
            trimesh.Trimesh: Room mesh object
        """
        bounds_bottom = np.array(bounds_bottom)
        bounds_top = np.array(bounds_top)
        
        # Create floor polygon
        floor_polygon = self._create_floor_polygon(bounds_bottom.tolist())
        
        num_verts = len(bounds_bottom)
        all_vertices = np.concatenate([bounds_bottom, bounds_top], axis=0)
        
        # Triangulate floor polygon using trimesh
        try:
            vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="triangle")
        except Exception:
            try:
                vtx, floor_faces = trimesh.creation.triangulate_polygon(floor_polygon, engine="earcut")
            except Exception:
                floor_faces = np.array([[0, i, i+1] for i in range(1, num_verts-1)])
        
        # Remove invalid faces
        valid_mask = np.all(floor_faces < num_verts, axis=1)
        floor_faces = floor_faces[valid_mask]
        
        # Create ceiling faces
        ceiling_faces = floor_faces + num_verts
        ceiling_faces = ceiling_faces[:, ::-1]
        
        # Create side faces
        side_faces = []
        for i in range(num_verts):
            next_i = (i + 1) % num_verts
            side_faces.append([i, next_i, i + num_verts])
            side_faces.append([next_i, next_i + num_verts, i + num_verts])
        side_faces = np.array(side_faces)
        
        # Merge all faces
        all_faces = np.concatenate([floor_faces, ceiling_faces, side_faces], axis=0)
        
        # Create mesh and fix normals
        room_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces)
        trimesh.repair.fix_normals(room_mesh)
        
        return room_mesh
    
    def _check_object_out_of_bounds(self, obj_mesh, room_mesh, floor_polygon, 
                                    room_height_min, room_height_max, num_samples=500):
        """
        Use mesh containment to detect if an object is out of bounds (supports irregular rooms and height detection)
        
        Args:
            obj_mesh: Object's trimesh object
            room_mesh: Room's trimesh object
            floor_polygon: Room floor's shapely polygon
            room_height_min: Minimum room height
            room_height_max: Maximum room height
            num_samples: Number of sample points
        
        Returns:
            (is_oob: bool, oob_volume: float)
        """
        from shapely.geometry import Point
        
        try:
            # Sample points on the object mesh
            sample_points = obj_mesh.sample(num_samples)
            
            # Detect 2D out-of-bounds (XZ plane)
            # Note: shapely's contains() returns False for points on the boundary, causing wall-adjacent furniture to be falsely flagged
            # Solution: add a tiny buffer (1mm) to the polygon so boundary points are correctly classified as inside
            buffered_polygon = floor_polygon.buffer(0.001)  # 1mm buffer
            points_2d = [Point(pt[0], pt[2]) for pt in sample_points]
            inside_2d = np.array([buffered_polygon.contains(p) for p in points_2d])
            
            # Detect height out-of-bounds (Y direction) - also add a tiny tolerance
            height_tolerance = 0.001  # 1mm tolerance
            inside_height = (sample_points[:, 1] >= room_height_min - height_tolerance) & (sample_points[:, 1] <= room_height_max + height_tolerance)
            
            # Combined judgment
            inside = inside_2d & inside_height
            
            oob_ratio = 1.0 - (inside.sum() / len(inside))
            
            if oob_ratio > 0.01:  # More than 1% of points are outside
                is_oob = True
                obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
                oob_volume = obj_volume * oob_ratio
            else:
                is_oob = False
                oob_volume = 0.0
            
            return is_oob, oob_volume
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: OOB detection sampling failed: {e}, using bounding box fallback")
            # Fallback method
            obj_bounds = obj_mesh.bounds
            room_bounds = room_mesh.bounds
            
            if (obj_bounds[0] < room_bounds[0]).any() or (obj_bounds[1] > room_bounds[1]).any():
                obj_volume = obj_mesh.volume if hasattr(obj_mesh, 'volume') and obj_mesh.volume > 0 else np.prod(obj_mesh.extents)
                return True, obj_volume * 0.1
            return False, 0.0
    def _parse_scene_data(self, scene_json: Dict, format_type: str = 'ours'):
        """
        Parse scene JSON data, supports two formats
        
        Args:
            scene_json: Scene JSON data
            format_type: 'ours' or 'respace'
        
        Returns:
            bounds_bottom, bounds_top, all_objects
        """
        if format_type == 'respace':
            bounds_bottom = scene_json.get('bounds_bottom', [])
            bounds_top = scene_json.get('bounds_top', [])
            all_objects_data = scene_json.get('objects', [])
        else:
            # ours format: uses room_envelope and groups structure
            # Also supports flat format (with direct objects field)
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
                # Support flat format
                all_objects_data = scene_json['objects']
        
        return bounds_bottom, bounds_top, all_objects_data
    
    def _load_and_transform_mesh(self, obj_data: Dict, format_type: str = 'ours') -> Optional['trimesh.Trimesh']:
        """
        Load and transform object mesh (references myeval implementation)
        
        Supports two asset sources:
        - 3D-FUTURE: Uses 'jid' field, loads from models_base_path/{jid}/raw_model.glb
        - Objaverse: Uses 'uid' field, loads from GLB cache
        
        Args:
            obj_data: Object data dictionary
            format_type: 'ours' or 'respace'
        
        Returns:
            Transformed trimesh object, or None on failure
        """
        try:
            # Get target size
            if format_type == 'respace':
                target_size = obj_data.get('sampled_asset_size', obj_data.get('size', [1, 1, 1]))
            else:
                target_size = obj_data.get('size', [1, 1, 1])
            
            # Determine asset source and model path
            asset_source = obj_data.get('asset_source', '3d-future')
            model_path = None
            asset_id = None
            
            if asset_source == 'objaverse':
                # Objaverse asset: use uid and GLB cache
                uid = obj_data.get('uid')
                if uid:
                    asset_id = uid
                    model_path = _find_objaverse_glb(uid)
                    if model_path and self.verbose:
                        print(f"Found Objaverse GLB for {uid[:16]}...: {model_path}")
            else:
                # 3D-FUTURE asset: use jid
                if format_type == 'respace':
                    jid = obj_data.get('sampled_asset_jid', obj_data.get('jid', 'N/A'))
                else:
                    jid = obj_data.get('jid', 'N/A')
                asset_id = jid
                # Only try loading 3D-FUTURE models when models_base_path is available
                if self.models_base_path is not None:
                    model_path = self.models_base_path / jid / 'raw_model.glb'
                else:
                    model_path = None
                if not model_path.exists():
                    model_path = None
            
            # Load model
            if model_path is None or not model_path.exists():
                if self.verbose:
                    print(f"Warning: Model not found for {asset_id}, using box placeholder")
                mesh = trimesh.creation.box(extents=target_size)
            else:
                loaded = trimesh.load(str(model_path))
                # Handle case where trimesh.load returns a Scene object
                if isinstance(loaded, trimesh.Scene):
                    # Merge all geometries in the scene into a single mesh
                    if len(loaded.geometry) > 0:
                        mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
                    else:
                        if self.verbose:
                            print(f"Warning: Model file {asset_id} is empty scene, using box placeholder")
                        mesh = trimesh.creation.box(extents=target_size)
                else:
                    mesh = loaded
            
            # ==== Objaverse coordinate system correction ====
            # Objaverse GLB models need initial rotation reset (consistent with blender_renderer.py)
            # This ensures physics calculations match rendering results
            if asset_source == 'objaverse' and model_path is not None and model_path.exists():
                # Objaverse model: reset to standard orientation (no extra rotation correction needed)
                # Rendering uses rotation_euler = (0, 0, 0), so keep mesh as-is
                # Because trimesh-loaded GLB is already in the correct orientation
                if self.verbose:
                    print(f"Objaverse asset {asset_id}: using standard orientation (no correction needed)")
            
            # 1. Scale to target size
            original_size = mesh.extents
            target_size_array = np.array(target_size)
            scale_factors = target_size_array / (original_size + 1e-6)
            mesh.apply_scale(scale_factors)
            
            # 2. Rotate
            pos = obj_data['pos']
            rot_xyzw = obj_data['rot']
            
            rotation = R.from_quat(rot_xyzw)
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = rotation.as_matrix()
            transform_matrix[:3, 3] = pos
            
            # 3. Compute bottom center as anchor point
            bounds = mesh.bounds
            bottom_center_pivot = np.array([
                (bounds[0, 0] + bounds[1, 0]) / 2,  # X-axis center
                bounds[0, 1],                        # Y-axis bottom
                (bounds[0, 2] + bounds[1, 2]) / 2    # Z-axis center
            ])
            
            # Move to origin
            center_transform = np.eye(4)
            center_transform[:3, 3] = -bottom_center_pivot
            
            # Apply full transform
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
        Evaluate scene physics metrics (fully references myeval implementation)
        
        Args:
            scene_data: Scene data, can be a dict, JSON file path, or Path object
            format_type: Scene format type, 'ours' or 'respace'
        
        Returns:
            Dictionary containing the following metrics:
            - collision_free_rate: Collision-free rate (%)
            - num_colliding_pairs: Number of colliding pairs
            - collision_rate: Collision rate (%)
            - mean_penetration_depth: Mean penetration depth (m)
            - valid_placement_rate: Valid placement rate (%)
            - num_oob_objects: Number of out-of-bounds objects
            - out_of_bounds_rate: Out-of-bounds rate (%)
            - mean_oob_volume: Mean out-of-bounds volume (m³)
        """
        # Load scene data
        if isinstance(scene_data, (str, Path)):
            with open(scene_data, 'r', encoding='utf-8') as f:
                scene_json = json.load(f)
        else:
            scene_json = scene_data
        
        # Parse scene data
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
        
        # Create room mesh and floor polygon (supports irregular rooms)
        bounds_bottom_arr = np.array(bounds_bottom)
        bounds_top_arr = np.array(bounds_top)
        
        room_mesh = self._create_room_mesh(bounds_bottom, bounds_top)
        floor_polygon = self._create_floor_polygon(bounds_bottom)
        
        # Compute room height range
        room_height_min = bounds_bottom_arr[:, 1].min()
        room_height_max = bounds_top_arr[:, 1].max()
        room_bounds = room_mesh.bounds
        
        # Load and transform all object meshes
        # Also build name_to_meta mapping for generating readable feedback
        furniture_objects = {}
        name_to_meta = {}  # Store name -> {idx, jid, uid, desc, asset_source} mapping
        
        for i, obj_data in enumerate(all_objects_data):
            mesh = self._load_and_transform_mesh(obj_data, format_type)
            if mesh is not None:
                # Prefer uid (Objaverse), otherwise use jid (3D-FUTURE)
                uid = obj_data.get('uid', '')
                jid = obj_data.get('jid', f'object_{i}')
                desc = obj_data.get('desc', obj_data.get('category', 'unknown'))
                asset_source = obj_data.get('asset_source', '3d-future')
                
                # Use shorter but unique naming
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
        
        # ===== Collision Detection =====
        manager = trimesh.collision.CollisionManager()
        for name, mesh in furniture_objects.items():
            manager.add_object(name, mesh)
        
        # Get all collision data
        is_collision, contact_data = manager.in_collision_internal(return_data=True)
        
        # All penetration depths > 0 are treated as collisions (tolerance removed)
        actual_collisions_data = [d for d in contact_data if d.depth > 0]
        
        num_colliding_pairs = len(actual_collisions_data)
        
        # Compute number of colliding objects and build collision pair list
        colliding_objects = set()
        collision_pairs_raw = []  # Store raw collision pair info
        
        for contact in actual_collisions_data:
            if hasattr(contact, 'names') and contact.names is not None:
                # contact.names can be list, tuple, frozenset, set, etc.
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
        
        # Compute mean penetration depth
        total_penetration_depth = 0
        if num_colliding_pairs > 0:
            for contact in actual_collisions_data:
                # Accumulate full penetration depth (tolerance subtraction removed)
                total_penetration_depth += contact.depth
        
        mean_penetration_depth = (total_penetration_depth / num_colliding_pairs) if num_colliding_pairs > 0 else 0
        
        # ===== Out-of-Bounds Detection (supports irregular rooms) =====
        num_oob_objects = 0
        total_oob_volume = 0
        oob_objects_raw = []  # Store OOB object info
        
        for name, mesh in furniture_objects.items():
            # Use precise polygon + height detection
            is_oob, oob_volume = self._check_object_out_of_bounds(
                mesh, room_mesh, floor_polygon, room_height_min, room_height_max
            )
            if is_oob:
                num_oob_objects += 1
                total_oob_volume += oob_volume
                # Compute out-of-bounds ratio
                obj_volume = mesh.volume if hasattr(mesh, 'volume') and mesh.volume > 0 else np.prod(mesh.extents)
                oob_ratio = oob_volume / obj_volume if obj_volume > 0 else 0.0
                oob_objects_raw.append({
                    'name': name,
                    'oob_volume': oob_volume,
                    'oob_ratio': min(oob_ratio, 1.0)  # Clamp to 0-1
                })
        
        mean_oob_volume = (total_oob_volume / num_oob_objects) if num_oob_objects > 0 else 0
        out_of_bounds_rate = (num_oob_objects / total_objects * 100) if total_objects > 0 else 0.0
        
        # ===== Build collision pairs list (with metadata, sorted by depth) =====
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
                'depth_cm': round(pair['depth'] * 100, 1)  # Convert to centimeters
            })
        
        # ===== Build OOB objects list (with metadata, sorted by OOB ratio) =====
        oob_objects = []
        for obj in sorted(oob_objects_raw, key=lambda x: x['oob_ratio'], reverse=True):
            meta = name_to_meta.get(obj['name'], {})
            oob_objects.append({
                'desc': meta.get('desc', 'unknown'),
                'obj_id': meta.get('uid') or meta.get('jid', 'unknown'),
                'oob_volume': obj['oob_volume'],
                'oob_ratio': obj['oob_ratio'],
                'oob_percent': round(obj['oob_ratio'] * 100, 1)  # Convert to percentage
            })
        
        # Build results
        metrics = {
            'collision_free_rate': 100.0 if num_colliding_pairs == 0 else 0.0,
            'num_colliding_pairs': num_colliding_pairs,
            'collision_rate': collision_rate,
            'mean_penetration_depth': mean_penetration_depth,
            'total_penetration_depth': total_penetration_depth,  # Total penetration depth (for volume reward)
            'valid_placement_rate': 100.0 if num_oob_objects == 0 else 0.0,
            'num_oob_objects': num_oob_objects,
            'out_of_bounds_rate': out_of_bounds_rate,
            'mean_oob_volume': mean_oob_volume,
            'total_oob_volume': total_oob_volume,  # Total OOB volume (for volume reward)
            # New: detailed collision pair and OOB object lists
            'collision_pairs': collision_pairs,  # List[{obj_a_desc, obj_a_id, obj_b_desc, obj_b_id, depth, depth_cm}]
            'oob_objects': oob_objects  # List[{desc, obj_id, oob_volume, oob_ratio, oob_percent}]
        }
        
        if self.verbose:
            print(f"\nTrimesh physics metrics evaluation results:")
            print(f"  Collision rate: {collision_rate:.2f}%")
            print(f"  Colliding pairs: {num_colliding_pairs}")
            print(f"  Mean penetration depth: {mean_penetration_depth:.6f}m")
            print(f"  Out-of-bounds rate: {out_of_bounds_rate:.2f}%")
            print(f"  OOB objects: {num_oob_objects}/{total_objects}")
        
        return metrics
    
    def compute_reward(self,
                      scene_data: Union[Dict, str, Path],
                      format_type: str = 'ours') -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward based on physics metrics
        
        Reward calculation strategy (adjusted based on SFT baseline):
        
        Collision rate (SFT baseline 45% as zero point):
        - Collision rate ≤ 20%: +1.0 to +0.5 (excellent)
        - Collision rate 20%-45%: +0.5 to 0.0 (SFT baseline as zero point)
        - Collision rate > 45%: 0.0 to -1.0 (worse than SFT)
        
        Out-of-bounds rate (SFT baseline 30% as zero point):
        - OOB rate ≤ 10%: +1.0 to +0.5 (excellent)
        - OOB rate 10%-30%: +0.5 to 0.0 (SFT baseline as zero point)
        - OOB rate > 30%: 0.0 to -1.0 (worse than SFT)
        
        Final reward = (collision_reward + oob_reward) / 2
        
        Args:
            scene_data: Scene data
            format_type: Scene format type
        
        Returns:
            (reward, metrics) tuple
        """
        metrics = self.evaluate_scene(scene_data, format_type)
        
        collision_rate = metrics['collision_rate']
        oob_rate = metrics['out_of_bounds_rate']
        
        # Collision reward (adjusted based on SFT baseline 45%)
        if collision_rate <= 20:
            # Excellent range: 0% -> +1.0, 20% -> +0.5
            collision_reward = 1.0 - 0.5 * (collision_rate / 20.0)
        elif collision_rate <= 45:
            # Good range (SFT baseline as zero point): 20% -> +0.5, 45% -> 0.0
            collision_reward = 0.5 - 0.5 * (collision_rate - 20) / 25.0
        else:
            # Worse than SFT: 45% -> 0.0, 100% -> -1.0
            collision_reward = -1.0 * (collision_rate - 45) / 55.0
        
        # OOB reward (adjusted based on SFT baseline 30%)
        if oob_rate <= 10:
            # Excellent range: 0% -> +1.0, 10% -> +0.5
            oob_reward = 1.0 - 0.5 * (oob_rate / 10.0)
        elif oob_rate <= 30:
            # Good range (SFT baseline as zero point): 10% -> +0.5, 30% -> 0.0
            oob_reward = 0.5 - 0.5 * (oob_rate - 10) / 20.0
        else:
            # Worse than SFT: 30% -> 0.0, 100% -> -1.0
            oob_reward = -1.0 * (oob_rate - 30) / 70.0
        
        # Combined reward
        reward = (collision_reward + oob_reward) / 2.0
        
        if self.verbose:
            print(f"\nReward calculation:")
            print(f"  Collision reward: {collision_reward:.4f} (collision rate: {collision_rate:.2f}%)")
            print(f"  OOB reward: {oob_reward:.4f} (OOB rate: {oob_rate:.2f}%)")
            print(f"  Combined reward: {reward:.4f}")
        
        return reward, metrics


def generate_physics_feedback(metrics: Dict[str, Any], top_k: int = 3) -> str:
    """
    Generate concise feedback text based on physics evaluation metrics
    
    Args:
        metrics: Metrics dictionary returned by TrimeshPhysicsMetrics.evaluate_scene
        top_k: Maximum number of collision pairs/OOB objects to report
    
    Returns:
        Physics feedback text, or empty string if there are no issues
    """
    feedback_parts = []
    
    # Handle collision issues
    collision_pairs = metrics.get('collision_pairs', [])
    if collision_pairs:
        # Sort by penetration depth, take the top_k most severe
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
    
    # Handle out-of-bounds issues
    oob_objects = metrics.get('oob_objects', [])
    if oob_objects:
        # Sort by OOB ratio, take the top_k most severe
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
