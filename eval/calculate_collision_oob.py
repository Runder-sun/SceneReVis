#!/usr/bin/env python3
"""
计算场景的碰撞率(Collision Rate)和出界率(Out-of-Bounds Rate)

碰撞率 = 一个场景中碰撞物体数量 / 所有物体数量
出界率 = 一个场景中出界物体数量 / 所有物体数量
"""

import os
import json
import math
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def get_rotation_matrix_z(angle_deg: float) -> np.ndarray:
    """获取绕Z轴旋转的旋转矩阵"""
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])


def get_oriented_bbox_corners(position: List[float], 
                               size: List[float], 
                               rotation_z: float) -> np.ndarray:
    """
    获取带旋转的2D包围盒的四个角点
    
    Args:
        position: [x, y, z] 物体中心位置
        size: [width, depth, height] 或 [x, y, z] 物体尺寸
        rotation_z: 绕Z轴的旋转角度(度)
    
    Returns:
        4个角点的坐标 shape: (4, 2)
    """
    # 取XY平面的尺寸的一半
    half_w = size[0] / 2
    half_d = size[1] / 2
    
    # 局部坐标系下的四个角点
    corners_local = np.array([
        [-half_w, -half_d],
        [half_w, -half_d],
        [half_w, half_d],
        [-half_w, half_d]
    ])
    
    # 旋转矩阵
    rot_mat = get_rotation_matrix_z(rotation_z)
    
    # 旋转角点
    corners_rotated = corners_local @ rot_mat.T
    
    # 平移到世界坐标
    center = np.array([position[0], position[1]])
    corners_world = corners_rotated + center
    
    return corners_world


def project_polygon_onto_axis(corners: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """将多边形投影到轴上，返回最小和最大值"""
    projections = corners @ axis
    return projections.min(), projections.max()


def check_obb_collision(corners1: np.ndarray, corners2: np.ndarray) -> bool:
    """
    使用分离轴定理(SAT)检测两个OBB是否碰撞
    
    Returns:
        True 如果碰撞，False 如果不碰撞
    """
    def get_axes(corners: np.ndarray) -> List[np.ndarray]:
        """获取多边形的两条边的法向量作为分离轴"""
        axes = []
        for i in range(2):  # 只需要两条边的法向量（矩形）
            edge = corners[(i + 1) % 4] - corners[i]
            # 法向量（垂直于边）
            normal = np.array([-edge[1], edge[0]])
            # 归一化
            length = np.linalg.norm(normal)
            if length > 1e-10:
                normal = normal / length
                axes.append(normal)
        return axes
    
    # 获取两个矩形的分离轴
    axes = get_axes(corners1) + get_axes(corners2)
    
    for axis in axes:
        min1, max1 = project_polygon_onto_axis(corners1, axis)
        min2, max2 = project_polygon_onto_axis(corners2, axis)
        
        # 检查是否有间隙（增加一个小的容差）
        tolerance = 0.01  # 1cm的容差
        if max1 < min2 - tolerance or max2 < min1 - tolerance:
            return False  # 存在分离轴，不碰撞
    
    return True  # 在所有轴上都有重叠，碰撞


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    判断点是否在多边形内部（射线法）
    """
    n = len(polygon)
    inside = False
    x, y = point
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def check_out_of_bounds(corners: np.ndarray, boundary_vertices: List[List[float]]) -> bool:
    """
    检查物体是否出界
    
    Args:
        corners: 物体的四个角点
        boundary_vertices: 房间边界顶点
    
    Returns:
        True 如果出界，False 如果在边界内
    """
    # 提取边界的XY坐标
    boundary_xy = np.array([[v[0], v[1]] for v in boundary_vertices])
    
    # 计算边界的AABB
    min_x = boundary_xy[:, 0].min()
    max_x = boundary_xy[:, 0].max()
    min_y = boundary_xy[:, 1].min()
    max_y = boundary_xy[:, 1].max()
    
    # 检查物体的每个角点是否在边界内
    tolerance = 0.001  # 1mm的容差
    for corner in corners:
        # 简单的AABB检查（假设边界是矩形）
        if (corner[0] < min_x - tolerance or corner[0] > max_x + tolerance or
            corner[1] < min_y - tolerance or corner[1] > max_y + tolerance):
            return True
    
    return False


def get_object_size(asset_info: Dict) -> List[float]:
    """
    从asset信息中获取物体尺寸
    """
    # 优先使用assetMetadata中的boundingBox
    if 'assetMetadata' in asset_info and 'boundingBox' in asset_info['assetMetadata']:
        bb = asset_info['assetMetadata']['boundingBox']
        if isinstance(bb, dict):
            return [bb.get('x', 0.5), bb.get('y', 0.5), bb.get('z', 0.5)]
    
    # 其次使用_original_y_up_data中的_matched_size
    if '_original_y_up_data' in asset_info:
        y_up_data = asset_info['_original_y_up_data']
        if '_matched_size' in y_up_data:
            matched = y_up_data['_matched_size']
            # 需要转换坐标系：Y-up to Z-up
            return [matched[0], matched[2], matched[1]]
        if 'size' in y_up_data:
            size = y_up_data['size']
            return [size[0], size[2], size[1]]
    
    # 使用annotations中的尺寸（单位是cm，需要转换为m）
    if 'annotations' in asset_info:
        ann = asset_info['annotations']
        width = ann.get('width', 50) / 100.0
        depth = ann.get('depth', 50) / 100.0
        height = ann.get('height', 50) / 100.0
        return [width, depth, height]
    
    # 默认尺寸
    return [0.5, 0.5, 0.5]


def analyze_scene(layout_path: str, scene_config_path: str) -> Dict:
    """
    分析单个场景的碰撞和出界情况
    
    Returns:
        包含分析结果的字典
    """
    try:
        with open(layout_path, 'r') as f:
            layout = json.load(f)
        with open(scene_config_path, 'r') as f:
            scene_config = json.load(f)
    except Exception as e:
        return {'error': str(e)}
    
    # 获取边界
    boundary = scene_config.get('boundary', {})
    floor_vertices = boundary.get('floor_vertices', [])
    if not floor_vertices:
        return {'error': 'No floor vertices found'}
    
    # 获取assets信息
    assets = scene_config.get('assets', {})
    
    # 收集所有物体信息
    objects = []
    for obj_key, obj_layout in layout.items():
        position = obj_layout.get('position', [0, 0, 0])
        rotation = obj_layout.get('rotation', [0, 0, 0])
        rotation_z = rotation[2] if len(rotation) > 2 else 0
        
        # 获取物体尺寸
        asset_info = assets.get(obj_key, {})
        size = get_object_size(asset_info)
        
        # 计算OBB角点
        corners = get_oriented_bbox_corners(position, size, rotation_z)
        
        objects.append({
            'key': obj_key,
            'position': position,
            'rotation_z': rotation_z,
            'size': size,
            'corners': corners
        })
    
    total_objects = len(objects)
    if total_objects == 0:
        return {
            'total_objects': 0,
            'collision_count': 0,
            'oob_count': 0,
            'collision_rate': 0.0,
            'oob_rate': 0.0,
            'collision_pairs': [],
            'oob_objects': []
        }
    
    # 检测碰撞
    collision_objects = set()
    collision_pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if check_obb_collision(objects[i]['corners'], objects[j]['corners']):
                collision_objects.add(objects[i]['key'])
                collision_objects.add(objects[j]['key'])
                collision_pairs.append((objects[i]['key'], objects[j]['key']))
    
    # 检测出界
    oob_objects = []
    for obj in objects:
        if check_out_of_bounds(obj['corners'], floor_vertices):
            oob_objects.append(obj['key'])
    
    collision_count = len(collision_objects)
    oob_count = len(oob_objects)
    
    return {
        'total_objects': total_objects,
        'collision_count': collision_count,
        'oob_count': oob_count,
        'collision_rate': collision_count / total_objects if total_objects > 0 else 0.0,
        'oob_rate': oob_count / total_objects if total_objects > 0 else 0.0,
        'collision_pairs': collision_pairs,
        'oob_objects': oob_objects
    }


def analyze_setting(setting_dir: str) -> Dict:
    """
    分析一个setting目录下所有场景
    """
    setting_path = Path(setting_dir)
    
    results = {
        'setting': setting_path.name,
        'scenes': {},
        'summary': {}
    }
    
    # 查找所有prompt_*目录
    prompt_dirs = sorted([d for d in setting_path.iterdir() 
                         if d.is_dir() and d.name.startswith('prompt_')],
                        key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0)
    
    total_objects = 0
    total_collision_objects = 0
    total_oob_objects = 0
    valid_scenes = 0
    
    for prompt_dir in prompt_dirs:
        layout_path = prompt_dir / 'layout.json'
        scene_config_path = prompt_dir / 'scene_config.json'
        
        if layout_path.exists() and scene_config_path.exists():
            scene_result = analyze_scene(str(layout_path), str(scene_config_path))
            results['scenes'][prompt_dir.name] = scene_result
            
            if 'error' not in scene_result:
                valid_scenes += 1
                total_objects += scene_result['total_objects']
                total_collision_objects += scene_result['collision_count']
                total_oob_objects += scene_result['oob_count']
    
    # 计算汇总统计
    results['summary'] = {
        'valid_scenes': valid_scenes,
        'total_scenes': len(prompt_dirs),
        'total_objects': total_objects,
        'total_collision_objects': total_collision_objects,
        'total_oob_objects': total_oob_objects,
        'avg_collision_rate': total_collision_objects / total_objects if total_objects > 0 else 0.0,
        'avg_oob_rate': total_oob_objects / total_objects if total_objects > 0 else 0.0,
    }
    
    # 计算每个场景的平均碰撞率和出界率
    scene_collision_rates = []
    scene_oob_rates = []
    for scene_name, scene_result in results['scenes'].items():
        if 'error' not in scene_result and scene_result['total_objects'] > 0:
            scene_collision_rates.append(scene_result['collision_rate'])
            scene_oob_rates.append(scene_result['oob_rate'])
    
    if scene_collision_rates:
        results['summary']['mean_scene_collision_rate'] = sum(scene_collision_rates) / len(scene_collision_rates)
        results['summary']['mean_scene_oob_rate'] = sum(scene_oob_rates) / len(scene_oob_rates)
    else:
        results['summary']['mean_scene_collision_rate'] = 0.0
        results['summary']['mean_scene_oob_rate'] = 0.0
    
    return results


def main():
    parser = argparse.ArgumentParser(description='计算场景的碰撞率和出界率')
    parser.add_argument('--settings-dirs', nargs='+', 
                        help='Setting目录列表')
    parser.add_argument('--base-dir', type=str,
                        default='/path/to/SceneReVis/baseline/LayoutVLM/results',
                        help='基础目录')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径')
    args = parser.parse_args()
    
    if args.settings_dirs:
        settings_dirs = args.settings_dirs
    else:
        # 默认处理三个setting
        settings_dirs = [
            os.path.join(args.base_dir, 'benchmark_unified_setting1'),
            os.path.join(args.base_dir, 'benchmark_unified_setting2'),
            os.path.join(args.base_dir, 'benchmark_unified_setting3'),
        ]
    
    all_results = {}
    
    print("=" * 80)
    print("场景碰撞率和出界率分析")
    print("=" * 80)
    
    for setting_dir in settings_dirs:
        if os.path.exists(setting_dir):
            print(f"\n处理: {setting_dir}")
            results = analyze_setting(setting_dir)
            all_results[results['setting']] = results
            
            summary = results['summary']
            print(f"\n  {results['setting']} 汇总:")
            print(f"    有效场景数: {summary['valid_scenes']}/{summary['total_scenes']}")
            print(f"    总物体数: {summary['total_objects']}")
            print(f"    碰撞物体数: {summary['total_collision_objects']}")
            print(f"    出界物体数: {summary['total_oob_objects']}")
            print(f"    整体碰撞率: {summary['avg_collision_rate']:.4f} ({summary['avg_collision_rate']*100:.2f}%)")
            print(f"    整体出界率: {summary['avg_oob_rate']:.4f} ({summary['avg_oob_rate']*100:.2f}%)")
            print(f"    场景平均碰撞率: {summary['mean_scene_collision_rate']:.4f} ({summary['mean_scene_collision_rate']*100:.2f}%)")
            print(f"    场景平均出界率: {summary['mean_scene_oob_rate']:.4f} ({summary['mean_scene_oob_rate']*100:.2f}%)")
        else:
            print(f"\n警告: 目录不存在 - {setting_dir}")
    
    # 打印对比表格
    print("\n" + "=" * 80)
    print("各Setting对比:")
    print("=" * 80)
    print(f"{'Setting':<30} {'场景数':<10} {'物体数':<10} {'碰撞率':<15} {'出界率':<15}")
    print("-" * 80)
    for setting_name, results in all_results.items():
        s = results['summary']
        print(f"{setting_name:<30} {s['valid_scenes']:<10} {s['total_objects']:<10} "
              f"{s['avg_collision_rate']*100:.2f}%{'':<8} {s['avg_oob_rate']*100:.2f}%")
    print("=" * 80)
    
    # 保存结果
    if args.output:
        # 将numpy数组转换为列表以便JSON序列化
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(all_results)
        with open(args.output, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
