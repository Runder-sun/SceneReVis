#!/usr/bin/env python3
"""
Blender渲染包装器，兼容没有bpy的环境
"""
import os
import json
from pathlib import Path

def render_scene_blender_external(scene_data, output_dir, scene_id="scene", enable_visualization=None, fast_mode=None):
    """
    使用外部Blender进程渲染场景
    当bpy不可用时的备选方案
    
    参数:
        scene_data: 场景数据字典
        output_dir: 输出目录
        scene_id: 场景ID
        enable_visualization: 是否启用3D辅助线可视化（bbox、箭头、坐标网格等）
                            如果为None，则从环境变量BPY_ENABLE_VISUALIZATION读取
        fast_mode: 是否启用快速渲染模式（512x512, 16采样）
                   如果为None，则从环境变量BPY_FAST_MODE读取
    """
    try:
        import subprocess
        import tempfile
        import json
        import os
        from pathlib import Path
        
        print(f"\n=== Blender Rendering (external process) for {scene_id} ===")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建临时场景文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(scene_data, temp_file, ensure_ascii=False, indent=2)
            temp_scene_path = temp_file.name
        
        try:
            # 构建Blender命令
            utils_dir = Path(__file__).parent
            main_bpy_path = utils_dir / 'main_bpy.py'
            
            if not main_bpy_path.exists():
                print(f"main_bpy.py not found at {main_bpy_path}")
                return create_placeholder_images(output_path, scene_id)
            
            # Determine Blender executable: prefer PathConfig, then BLENDER_EXECUTABLE, else try PATH, else check common locations
            blender_cmd = None
            
            # 1. Try PathConfig first (unified configuration)
            try:
                from path_config import PathConfig
                blender_cmd = PathConfig.get_instance().blender_executable
                if blender_cmd:
                    print(f"[PathConfig] Using Blender: {blender_cmd}")
            except ImportError:
                pass
            
            # 2. Environment variable
            if not blender_cmd:
                blender_cmd = os.environ.get('BLENDER_EXECUTABLE')
            
            # 3. Try which command
            if not blender_cmd:
                try:
                    from shutil import which
                    detected = which('blender')
                    if detected:
                        blender_cmd = detected
                    else:
                        # 常见路径候选（包括用户安装路径）
                        candidates = [
                            os.path.expanduser('~/.local/bin/blender'),
                            os.path.expanduser('~/.local/blender/blender-4.0.2/blender'),
                            os.path.expanduser('~/.local/blender/blender-3.6.0/blender'),
                            '/usr/bin/blender',
                            '/snap/bin/blender',
                            '/usr/local/bin/blender',
                            '/opt/blender/blender',
                            '/path/to/home/.local/bin/blender',
                            '/Applications/Blender.app/Contents/MacOS/Blender'
                        ]
                        found = next((p for p in candidates if os.path.exists(p)), None)
                        if found:
                            blender_cmd = found
                            print(f"Found Blender at: {blender_cmd}")
                        else:
                            # 尝试自动安装
                            print("⚠ Blender not found! Attempting auto-installation...")
                            install_script = Path(__file__).parent.parent / 'quick_install_blender.sh'
                            if install_script.exists():
                                try:
                                    import subprocess
                                    result = subprocess.run(['bash', str(install_script)], 
                                                          capture_output=True, text=True, timeout=300)
                                    if result.returncode == 0:
                                        # 再次尝试查找
                                        detected = which('blender')
                                        if detected:
                                            blender_cmd = detected
                                            print(f"✓ Auto-installed Blender at: {blender_cmd}")
                                        else:
                                            blender_cmd = 'blender'  # 最后回退
                                    else:
                                        print(f"Auto-installation failed: {result.stderr}")
                                        blender_cmd = 'blender'
                                except Exception as e:
                                    print(f"Auto-installation error: {e}")
                                    blender_cmd = 'blender'
                            else:
                                blender_cmd = 'blender'
                except Exception:
                    blender_cmd = 'blender'

            if blender_cmd != 'blender':
                print(f"Using Blender executable: {blender_cmd}")
            else:
                print("No Blender executable auto-detected; will try 'blender' from PATH or likely fail")

            cmd = [
                blender_cmd, '--background', '--python', str(main_bpy_path),
                '--', '--scene', temp_scene_path, '--out', str(output_path)
            ]
            
            # 设置环境变量
            env = os.environ.copy()
            # 启用详细输出来调试
            env['BPY_VERBOSE'] = '1'
            # 不使用占位符，渲染真实的3D模型和纹理
            env['BPY_USE_PLACEHOLDER_ONLY'] = '0'
            
            # 启用快速渲染模式（512x512, 8采样）
            if fast_mode is None:
                # 从环境变量读取
                fast_mode = os.environ.get('BPY_FAST_MODE', '1') == '1'  # 默认启用快速模式
            if fast_mode:
                env['BPY_FAST_MODE'] = '1'
                print("✓ Fast rendering mode enabled (512x512, 8 samples)")
            else:
                env['BPY_FAST_MODE'] = '0'
            
            # 启用3D可视化（bbox、箭头、坐标网格等）
            if enable_visualization is None:
                # 从环境变量读取
                enable_visualization = os.environ.get('BPY_ENABLE_VISUALIZATION', '0') == '1'
            if enable_visualization:
                env['BPY_ENABLE_VISUALIZATION'] = '1'
                print("✓ 3D visualization enabled (bbox, arrows, coordinate grid with labels)")
            else:
                env['BPY_ENABLE_VISUALIZATION'] = '0'
            # 设置3D资产路径（如果可用）- 优先使用 PathConfig
            if not env.get('PTH_3DFUTURE_ASSETS'):
                # Try PathConfig first
                try:
                    from path_config import PathConfig
                    config = PathConfig.get_instance()
                    if config.future3d_models_dir and os.path.exists(config.future3d_models_dir):
                        env['PTH_3DFUTURE_ASSETS'] = config.future3d_models_dir
                        print(f"[PathConfig] Using 3D-FUTURE asset path: {config.future3d_models_dir}")
                except ImportError:
                    pass
                
                # Fallback to hardcoded paths
                if not env.get('PTH_3DFUTURE_ASSETS'):
                    possible_asset_paths = [
                        '/path/to/datasets/3d-front/3D-FUTURE-model',
                        '/path/to/datasets/3D-FUTURE-model',
                        '/path/to/workspace/respace/assets',
                        os.path.expanduser('~/datasets/3D-FUTURE-model')
                    ]
                    for asset_path in possible_asset_paths:
                        if os.path.exists(asset_path):
                            env['PTH_3DFUTURE_ASSETS'] = asset_path
                            print(f"Using asset path: {asset_path}")
                            break
                
                # 验证资产路径包含GLB文件
                if env.get('PTH_3DFUTURE_ASSETS'):
                    test_glb_count = 0
                    try:
                        for item in os.listdir(env['PTH_3DFUTURE_ASSETS'])[:10]:  # 只检查前10个
                            glb_path = os.path.join(env['PTH_3DFUTURE_ASSETS'], item, 'raw_model.glb')
                            if os.path.exists(glb_path):
                                test_glb_count += 1
                        print(f"Verified: Found {test_glb_count}/10 test GLB models in asset directory")
                    except:
                        pass
            
            # 设置 Objaverse GLB 缓存路径（如果可用）- 优先使用 PathConfig
            if not env.get('OBJAVERSE_GLB_CACHE_DIR'):
                # Try PathConfig first
                try:
                    from path_config import PathConfig
                    config = PathConfig.get_instance()
                    if config.objaverse_glb_cache_dir:
                        # Use parent dir (objathor-assets), not glbs subdir
                        cache_base = os.path.dirname(config.objaverse_glb_cache_dir)
                        if os.path.exists(cache_base):
                            env['OBJAVERSE_GLB_CACHE_DIR'] = cache_base
                            print(f"[PathConfig] Using Objaverse GLB cache: {cache_base}")
                except ImportError:
                    pass
                
                # Fallback to hardcoded paths
                if not env.get('OBJAVERSE_GLB_CACHE_DIR'):
                    possible_objaverse_paths = [
                        '/path/to/data/datasets/objathor-assets',
                        '/path/to/datasets/objathor-assets',
                        os.path.expanduser('~/.objaverse')
                    ]
                    for objaverse_path in possible_objaverse_paths:
                        glb_dir = os.path.join(objaverse_path, 'glbs')
                        if os.path.exists(glb_dir) and os.path.isdir(glb_dir):
                            env['OBJAVERSE_GLB_CACHE_DIR'] = objaverse_path
                            print(f"Using Objaverse GLB cache: {objaverse_path}")
                            break
            
            # 确保 Blender 子进程的 PYTHONPATH 包含项目根和 utils 目录，以便 main_bpy.py 能找到 blender_renderer
            existing_py = env.get('PYTHONPATH', '')
            utils_dir = os.path.dirname(os.path.abspath(__file__))  # /path/to/SceneReVis/utils
            extra_paths = [
                utils_dir,  # utils 目录包含 blender_renderer.py
                '/path/to/workspace/respace/src',
                '/path/to/workspace/respace',
                '/path/to/SceneReVis'
            ]
            combined = ':'.join([p for p in [existing_py] + extra_paths if p])
            env['PYTHONPATH'] = combined
            
            print(f"Running Blender command: {' '.join(cmd)}")
            # Use errors='replace' to avoid UnicodeDecodeError if Blender outputs non-UTF8 bytes
            result = subprocess.run(cmd, capture_output=True, text=True, errors='replace', env=env, timeout=180)
            
            # 显示Blender的详细输出（用于调试）
            if result.stdout:
                print("=== Blender STDOUT ===")
                print(result.stdout)
                print("=== End STDOUT ===")
            if result.stderr:
                print("=== Blender STDERR ===")
                print(result.stderr)
                print("=== End STDERR ===")
            
            if result.returncode == 0:
                # 检查输出文件
                top_file = output_path / "top" / "frame.png"
                diag_file = output_path / "diag" / "frame.png"
                
                if top_file.exists() and diag_file.exists():
                    print("Blender rendering completed successfully")
                    return str(output_path)
                else:
                    print("Blender completed but output files not found")
                    return create_placeholder_images(output_path, scene_id)
            else:
                print(f"Blender failed with return code {result.returncode}")
                return create_placeholder_images(output_path, scene_id)
                
        finally:
            # 调试：暂时保留临时文件
            print(f"DEBUG: Temporary scene file saved at: {temp_scene_path}")
            # 清理临时文件 (暂时禁用用于调试)
            # try:
            #     os.unlink(temp_scene_path)
            # except:
            #     pass
                
    except Exception as e:
        print(f"Error in external Blender rendering: {e}")
        return create_placeholder_images(output_path, scene_id)

def create_placeholder_images(output_path, scene_id="scene"):
    """创建占位符图片"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # 创建必要的目录
        (output_path / 'top').mkdir(parents=True, exist_ok=True)
        (output_path / 'diag').mkdir(parents=True, exist_ok=True)
        
        for view in ['top', 'diag']:
            # 创建图片
            img = Image.new('RGB', (1024, 1024), color='lightblue' if view == 'top' else 'lightgreen')
            draw = ImageDraw.Draw(img)
            
            # 绘制文字
            text = f"{view.upper()} View\n{scene_id}\nPlaceholder Image"
            text_bbox = draw.textbbox((0, 0), text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 居中文字
            x = (1024 - text_width) // 2
            y = (1024 - text_height) // 2
            draw.text((x, y), text, fill='black')
            
            # 保存图片
            view_path = output_path / view / 'frame.png'
            img.save(view_path)
            print(f"Created placeholder image: {view_path}")
        
        return str(output_path)
        
    except Exception as e:
        print(f"Error creating placeholder images: {e}")
        return str(output_path)

def render_scene_with_bpy_wrapper(scene_data, output_dir, scene_id="scene"):
    """
    Smart Blender rendering wrapper with real asset loading and multiple fallback levels
    """
    try:
        import bpy
        import tempfile
        import json
        import os
        from pathlib import Path
        from PIL import Image
        
        print(f"\n=== Blender Rendering (bpy internal) ===")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables for proper asset loading
        os.environ['BPY_VERBOSE'] = '1'  # Enable debug output
        
        # Set up asset path environment variable if not already set
        # This should point to your 3D assets directory
        if not os.environ.get('PTH_3DFUTURE_ASSETS'):
            # Try to find assets directory relative to project
            possible_paths = [
                '/path/to/workspace/respace/assets',
                '/path/to/datasets/3d-front/3D-FUTURE-model',
                '/path/to/workspace/assets',
            ]
            for path in possible_paths:
                if Path(path).exists():
                    os.environ['PTH_3DFUTURE_ASSETS'] = path
                    print(f"Using assets path: {path}")
                    break
            else:
                print("Warning: No 3D assets directory found, will use placeholders")
        
        # Create temp scene file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(scene_data, f, ensure_ascii=False, indent=2)
            temp_scene_path = f.name
        
        try:
            # Import and use the real blender renderer with asset loading
            from blender_renderer import render_scene_frame_bpy_inproc
            
            print("Using real Blender renderer with 3D assets and textures...")
            
            # Enable real asset loading (not placeholders)
            os.environ['BPY_USE_PLACEHOLDER_ONLY'] = '0'
            
            # Configure texture loading
            os.environ['BPY_TEXTURE_CANDIDATES'] = 'texture.png,texture.jpg,image.png,image.jpg,albedo.png,diffuse.png'
            os.environ['BPY_FORCE_REAPPLY_TEXTURE'] = '1'  # Force texture application
            
            generated_paths = render_scene_frame_bpy_inproc(scene_data, output_path)
            
            # Check if both views were generated
            top_file = output_path / "top" / "frame.png"
            diag_file = output_path / "diag" / "frame.png"
            
            if top_file.exists() and diag_file.exists():
                # Rename to expected format
                final_top = output_path / f"{scene_id}_top.png"
                final_diag = output_path / f"{scene_id}_diagonal.png"
                
                top_file.rename(final_top)
                diag_file.rename(final_diag)
                
                # Clean up empty directories
                (output_path / "top").rmdir()
                (output_path / "diag").rmdir()
                
                print(f"✅ Blender render with real assets completed: {final_top}, {final_diag}")
                return True
            else:
                print("❌ Blender render failed - output files not found")
                return False
                
        finally:
            # Clean up temp file
            import os
            os.unlink(temp_scene_path)
            
    except Exception as e:
        print(f"❌ Blender internal rendering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 测试函数
    test_scene = {
        "room_type": "livingroom",
        "room_envelope": {
            "bounds_bottom": [[-2, 0, -2], [2, 0, -2], [2, 0, 2], [-2, 0, 2]]
        },
        "groups": [
            {
                "group_name": "Test Group",
                "objects": [
                    {
                        "desc": "test object",
                        "jid": "test-id",
                        "pos": [0, 0, 0],
                        "size": [1, 1, 1],
                        "rot": [0, 0, 0, 1]
                    }
                ]
            }
        ]
    }
    
    result = render_scene_with_bpy_wrapper(test_scene, "./test_render")
    print(f"Rendering result: {result}")
