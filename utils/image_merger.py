#!/usr/bin/env python3
"""
Image merging tool: combines top view and diagonal view into a single image with bounding boxes and labels.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Union

def merge_rendered_views_with_annotations(top_view_path: Union[str, Path], 
                                        diagonal_view_path: Union[str, Path], 
                                        output_path: Union[str, Path]):
    """
    Merge top view and diagonal view into a single image, side by side, preserving transparent background, with bounding boxes and labels.
    """
    # Load images and keep RGBA mode
    top_img = Image.open(top_view_path)
    diag_img = Image.open(diagonal_view_path)
    
    # Ensure images are in RGBA mode to preserve transparency
    if top_img.mode != 'RGBA':
        top_img = top_img.convert('RGBA')
    if diag_img.mode != 'RGBA':
        diag_img = diag_img.convert('RGBA')
    
    # Get image dimensions
    top_width, top_height = top_img.size
    diag_width, diag_height = diag_img.size
    
    # Space for borders and labels
    border_width = 4
    label_height = 30
    padding = 10
    
    # Calculate merged dimensions (including borders and label space)
    view_width = max(top_width, diag_width) + 2 * border_width + 2 * padding
    view_height = max(top_height, diag_height) + 2 * border_width + label_height + 2 * padding
    merged_width = 2 * view_width
    merged_height = view_height
    
    # Create new RGBA image (transparent background)
    merged_img = Image.new('RGBA', (merged_width, merged_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(merged_img)
    
    # Try to load font, fall back to default if failed
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
    
    # Draw left top view area
    left_x = padding
    left_y = padding + label_height
    
    # Draw left bounding box (red)
    draw.rectangle([left_x - border_width, left_y - border_width, 
                   left_x + top_width + border_width, left_y + top_height + border_width], 
                   outline=(255, 0, 0, 255), width=border_width)
    
    # Add left label (red)
    if font:
        draw.text((left_x, padding), "Top View", fill=(255, 0, 0, 255), font=font)
    
    # Paste top view
    merged_img.paste(top_img, (left_x, left_y), top_img if top_img.mode == 'RGBA' else None)
    
    # Draw right diagonal view area
    right_x = view_width + padding
    right_y = padding + label_height
    
    # Draw right bounding box (blue)
    draw.rectangle([right_x - border_width, right_y - border_width, 
                   right_x + diag_width + border_width, right_y + diag_height + border_width], 
                   outline=(0, 0, 255, 255), width=border_width)
    
    # Add right label (blue)
    if font:
        draw.text((right_x, padding), "Diagonal View", fill=(0, 0, 255, 255), font=font)
    
    # Paste diagonal view
    merged_img.paste(diag_img, (right_x, right_y), diag_img if diag_img.mode == 'RGBA' else None)
    
    # Save image
    # If RGB mode is needed (no transparency support), convert background to white
    if str(output_path).lower().endswith(('.jpg', '.jpeg')):
        # JPEG doesn't support transparency, create white background
        rgb_img = Image.new('RGB', merged_img.size, (255, 255, 255))
        rgb_img.paste(merged_img, mask=merged_img.split()[-1] if merged_img.mode == 'RGBA' else None)
        rgb_img.save(output_path, quality=95)
    else:
        # PNG supports transparency
        merged_img.save(output_path)
    
    return True

def simple_merge_views(top_view_path: Union[str, Path], 
                      diagonal_view_path: Union[str, Path], 
                      output_path: Union[str, Path]):
    """
    Simple image merge, side by side, without bounding boxes or labels.
    """
    from PIL import Image
    
    top_img = Image.open(top_view_path)
    diag_img = Image.open(diagonal_view_path)
    
    # Resize to consistent dimensions
    target_size = (512, 512)
    top_img = top_img.resize(target_size)
    diag_img = diag_img.resize(target_size)
    
    # Merge horizontally
    merged_img = Image.new('RGB', (1024, 512))
    merged_img.paste(top_img, (0, 0))
    merged_img.paste(diag_img, (512, 0))
    
    merged_img.save(output_path)
    return True
