#!/bin/bash
# 批量运行BBOX评估脚本测试四个场景文件夹

cd /path/to/SceneReVis/eval

echo "=============================================="
echo "开始BBOX评估测试 (无容差)"
echo "=============================================="

# 场景文件夹列表
SCENE_DIRS=(
    "/path/to/data/eval/output/rl_ood_B200_v6_e3_s80/setting1/final_scenes_collection"
    "/path/to/data/eval/output/rl_ood_B200_v6_e3_s80/setting2/final_scenes_collection"
    "/path/to/data/eval/output/rl_ood_B200_v6_e3_s80/setting3/final_scenes_collection"
    "/path/to/SceneReVis/output/rl_ood_B200_v6_e3_s80/dining_room/final_scenes_collection"
)

NAMES=("setting1" "setting2" "setting3" "dining_room")

for i in "${!SCENE_DIRS[@]}"; do
    echo ""
    echo "=============================================="
    echo "测试 ${NAMES[$i]}: ${SCENE_DIRS[$i]}"
    echo "=============================================="
    echo ""
    
    python myeval_bbox.py --scenes_dir "${SCENE_DIRS[$i]}" --format respace
    
    echo ""
    echo "----------------------------------------------"
done

echo ""
echo "=============================================="
echo "所有测试完成!"
echo "=============================================="
