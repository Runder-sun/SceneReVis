import json
import os
import shutil

# 读取评估结果
results_file = "/path/to/datasets/ssr/scenes/evaluation_results.json"
scenes_dir = "/path/to/datasets/ssr/scenes"
output_dir = "/path/to/datasets/ssr/scenes_filtered_v2"

print("正在读取评估结果...")
with open(results_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 筛选条件
collision_rate_threshold = 10.0  # 碰撞率 < 10%
oob_rate_threshold = 30.0        # 出界率 < 30%
min_objects = 4                 # 物体数量 > 4
high_object_threshold = 20      # 高物体数量阈值（>=20个物体直接保留）
zero_collision_min_objects = 5  # 零碰撞场景的最低物体数量

filtered_scenes = []
room_type_stats = {}  # 统计各类型房间数量
object_count_stats = {}  # 统计不同物体数量的场景
stats = {
    'total': len(results['individual_results']),
    'collision_pass': 0,
    'oob_pass': 0,
    'objects_pass': 0,
    'all_pass': 0,
    'high_objects_pass': 0,      # 高物体数量直接通过
    'zero_collision_pass': 0     # 零碰撞通过
}

print(f"\n开始筛选场景...")
print(f"筛选条件:")
print(f"  条件1: 碰撞率 < {collision_rate_threshold}% 且 越界率 < {oob_rate_threshold}% 且 物体数量 > {min_objects}")
print(f"  条件2: 物体数量 >= {high_object_threshold} (直接保留)")
print(f"  条件3: 碰撞率 = 0% 且 物体数量 > {zero_collision_min_objects}")
print()

for scene_result in results['individual_results']:
    filename = scene_result['file']
    metrics = scene_result['metrics']
    
    # 获取场景的物体数量和房间类型(需要读取原始JSON文件)
    scene_path = os.path.join(scenes_dir, filename)
    with open(scene_path, 'r', encoding='utf-8') as f:
        scene_data = json.load(f)
    
    num_objects = len(scene_data.get('objects', []))
    room_type = scene_data.get('room_type', 'unknown')  # 获取房间类型
    
    # 检查条件
    collision_rate = metrics.get('Collision Rate (%)', 100)
    oob_rate = metrics.get('Out-of-Bounds Rate (%)', 100)
    
    collision_pass = collision_rate < collision_rate_threshold
    oob_pass = oob_rate < oob_rate_threshold
    objects_pass = num_objects > min_objects
    
    # 新增条件
    high_objects_pass = num_objects >= high_object_threshold  # 20个物体以上
    zero_collision_pass = collision_rate == 0 and num_objects > zero_collision_min_objects  # 零碰撞且物体>5
    
    if collision_pass:
        stats['collision_pass'] += 1
    if oob_pass:
        stats['oob_pass'] += 1
    if objects_pass:
        stats['objects_pass'] += 1
    if high_objects_pass:
        stats['high_objects_pass'] += 1
    if zero_collision_pass:
        stats['zero_collision_pass'] += 1
    
    # 满足以下任一条件即可保留:
    # 1. 原有条件: 碰撞率<10% 且 越界率<30% 且 物体>4
    # 2. 新条件1: 物体数量>=20
    # 3. 新条件2: 碰撞率=0% 且 物体>5
    original_pass = collision_pass and oob_pass and objects_pass
    should_keep = original_pass or high_objects_pass or zero_collision_pass
    
    if should_keep:
        stats['all_pass'] += 1
        
        # 统计房间类型
        if room_type not in room_type_stats:
            room_type_stats[room_type] = 0
        room_type_stats[room_type] += 1
        
        # 统计物体数量
        if num_objects not in object_count_stats:
            object_count_stats[num_objects] = 0
        object_count_stats[num_objects] += 1
        
        filtered_scenes.append({
            'file': filename,
            'num_objects': num_objects,
            'room_type': room_type,  # 添加房间类型
            'collision_rate': collision_rate,
            'oob_rate': oob_rate,
            'metrics': metrics
        })
        
        # 复制文件到输出目录
        src_path = scene_path
        dst_path = os.path.join(output_dir, filename)
        shutil.copy2(src_path, dst_path)

# 按物体数量排序
filtered_scenes.sort(key=lambda x: x['num_objects'], reverse=True)

# 打印统计信息
print("=" * 80)
print("筛选结果统计:")
print(f"总场景数: {stats['total']}")
print(f"碰撞率达标 (<{collision_rate_threshold}%): {stats['collision_pass']} ({stats['collision_pass']/stats['total']*100:.1f}%)")
print(f"越界率达标 (<{oob_rate_threshold}%): {stats['oob_pass']} ({stats['oob_pass']/stats['total']*100:.1f}%)")
print(f"物体数达标 (>{min_objects}): {stats['objects_pass']} ({stats['objects_pass']/stats['total']*100:.1f}%)")
print(f"高物体数量 (>={high_object_threshold}): {stats['high_objects_pass']} ({stats['high_objects_pass']/stats['total']*100:.1f}%)")
print(f"零碰撞且物体>{zero_collision_min_objects}: {stats['zero_collision_pass']} ({stats['zero_collision_pass']/stats['total']*100:.1f}%)")
print(f"最终保留: {stats['all_pass']} ({stats['all_pass']/stats['total']*100:.1f}%)")
print()

# 打印房间类型统计
print("=" * 80)
print("符合条件的场景按房间类型统计:")
print()
for room_type, count in sorted(room_type_stats.items(), key=lambda x: x[1], reverse=True):
    print(f"  {room_type}: {count} ({count/stats['all_pass']*100:.1f}%)")
print()

# 计算平均物体数量
total_objects = sum(scene['num_objects'] for scene in filtered_scenes)
avg_objects = total_objects / len(filtered_scenes) if filtered_scenes else 0

# 打印物体数量统计
print("=" * 80)
print("符合条件的场景按物体数量统计:")
print()

# 按物体数量分组统计
object_ranges = {
    '5-10': 0,
    '11-15': 0,
    '16-20': 0,
    '21-25': 0,
    '26-30': 0,
    '31+': 0
}

for num_objects, count in object_count_stats.items():
    if num_objects <= 10:
        object_ranges['5-10'] += count
    elif num_objects <= 15:
        object_ranges['11-15'] += count
    elif num_objects <= 20:
        object_ranges['16-20'] += count
    elif num_objects <= 25:
        object_ranges['21-25'] += count
    elif num_objects <= 30:
        object_ranges['26-30'] += count
    else:
        object_ranges['31+'] += count

print(f"平均物体数量: {avg_objects:.2f}")
print()
print("物体数量分布:")
for range_name, count in object_ranges.items():
    if count > 0:
        print(f"  {range_name}个物体: {count} ({count/stats['all_pass']*100:.1f}%)")
print()

# 显示最常见的物体数量 (Top 10)
print("最常见的物体数量 (前10):")
for num_objects, count in sorted(object_count_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {num_objects}个物体: {count} ({count/stats['all_pass']*100:.1f}%)")
print()

# 保存筛选结果
output_json = os.path.join(output_dir, "filtered_results.json")
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump({
        'filter_criteria': {
            'collision_rate_max': collision_rate_threshold,
            'oob_rate_max': oob_rate_threshold,
            'min_objects': min_objects,
            'high_object_threshold': high_object_threshold,
            'zero_collision_min_objects': zero_collision_min_objects
        },
        'statistics': stats,
        'room_type_statistics': room_type_stats,  # 添加房间类型统计
        'object_count_statistics': object_count_stats,  # 添加物体数量统计
        'average_objects': avg_objects,  # 添加平均物体数量
        'filtered_scenes': filtered_scenes
    }, f, indent=2, ensure_ascii=False)

print(f"筛选后的场景已复制到: {output_dir}")
print(f"详细结果已保存到: {output_json}")
print()

# 显示前10个最好的场景
print("=" * 80)
print("前10个符合条件的场景(按物体数量排序):")
print()
for i, scene in enumerate(filtered_scenes[:10], 1):
    print(f"{i}. {scene['file']}")
    print(f"   房间类型: {scene['room_type']}")
    print(f"   物体数量: {scene['num_objects']}")
    print(f"   碰撞率: {scene['collision_rate']:.1f}%")
    print(f"   越界率: {scene['oob_rate']:.1f}%")
    print(f"   无碰撞率: {scene['metrics'].get('Collision-Free Rate (%)', 0):.1f}%")
    print(f"   有效放置率: {scene['metrics'].get('Valid Placement Rate (%)', 0):.1f}%")
    print()
