#!/usr/bin/env python3
"""
图像合并工具：将俯视图和对角视图合并为一张图片，带边界框和标签
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Union

def merge_rendered_views_with_annotations(top_view_path: Union[str, Path], 
                                        diagonal_view_path: Union[str, Path], 
                                        output_path: Union[str, Path]):
    """
    合并俯视图和斜视图到一张图片中，左右排列，保持透明背景，并添加边界框和标签
    """
    # 加载图片并保持RGBA模式
    top_img = Image.open(top_view_path)
    diag_img = Image.open(diagonal_view_path)
    
    # 确保图片都是RGBA模式以保持透明度
    if top_img.mode != 'RGBA':
        top_img = top_img.convert('RGBA')
    if diag_img.mode != 'RGBA':
        diag_img = diag_img.convert('RGBA')
    
    # 获取图片尺寸
    top_width, top_height = top_img.size
    diag_width, diag_height = diag_img.size
    
    # 添加边框和标签的空间
    border_width = 4
    label_height = 30
    padding = 10
    
    # 计算合并后的尺寸（包含边框和标签空间）
    view_width = max(top_width, diag_width) + 2 * border_width + 2 * padding
    view_height = max(top_height, diag_height) + 2 * border_width + label_height + 2 * padding
    merged_width = 2 * view_width
    merged_height = view_height
    
    # 创建新的RGBA图片（透明背景）
    merged_img = Image.new('RGBA', (merged_width, merged_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(merged_img)
    
    # 尝试加载字体，如果失败则使用默认字体
    font = None
    try:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
                break
    except:
        pass
    
    if not font:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 绘制左边的俯视图区域
    left_x = padding
    left_y = padding + label_height
    
    # 绘制左边的边界框（红色）
    draw.rectangle([left_x - border_width, left_y - border_width, 
                   left_x + top_width + border_width, left_y + top_height + border_width], 
                   outline=(255, 0, 0, 255), width=border_width)
    
    # 添加左边的标签（红色）
    if font:
        draw.text((left_x, padding), "Top View", fill=(255, 0, 0, 255), font=font)
    
    # 粘贴俯视图
    merged_img.paste(top_img, (left_x, left_y), top_img if top_img.mode == 'RGBA' else None)
    
    # 绘制右边的对角视图区域
    right_x = view_width + padding
    right_y = padding + label_height
    
    # 绘制右边的边界框（蓝色）
    draw.rectangle([right_x - border_width, right_y - border_width, 
                   right_x + diag_width + border_width, right_y + diag_height + border_width], 
                   outline=(0, 0, 255, 255), width=border_width)
    
    # 添加右边的标签（蓝色）
    if font:
        draw.text((right_x, padding), "Diagonal View", fill=(0, 0, 255, 255), font=font)
    
    # 粘贴对角视图
    merged_img.paste(diag_img, (right_x, right_y), diag_img if diag_img.mode == 'RGBA' else None)
    
    # 保存图片
    # 如果需要RGB模式（不支持透明度），转换背景为白色
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        # JPEG不支持透明度，创建白色背景
        rgb_img = Image.new('RGB', merged_img.size, (255, 255, 255))
        rgb_img.paste(merged_img, mask=merged_img.split()[-1] if merged_img.mode == 'RGBA' else None)
        rgb_img.save(output_path, quality=95)
    else:
        # PNG支持透明度
        merged_img.save(output_path)
    
    return True

def simple_merge_views(top_view_path: Union[str, Path], 
                      diagonal_view_path: Union[str, Path], 
                      output_path: Union[str, Path]):
    """
    简单的图像合并，左右排列，不添加边界框和标签
    """
    from PIL import Image
    
    top_img = Image.open(top_view_path)
    diag_img = Image.open(diagonal_view_path)
    
    # 调整大小使其一致
    target_size = (512, 512)
    top_img = top_img.resize(target_size)
    diag_img = diag_img.resize(target_size)
    
    # 水平合并
    merged_img = Image.new('RGB', (1024, 512))
    merged_img.paste(top_img, (0, 0))
    merged_img.paste(diag_img, (512, 0))
    
    merged_img.save(output_path)
    return True
