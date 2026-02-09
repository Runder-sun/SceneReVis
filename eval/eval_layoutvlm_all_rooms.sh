#!/bin/bash
# 批量评估 LayoutVLM 的 bedroom, living_room, dining_room, study_room

BASE_RENDER="/path/to/SceneReVis/baseline/results/LayoutVLM/render"
BASE_JSON="/path/to/SceneReVis/baseline/results/LayoutVLM/json"
BASE_PROMPTS="/path/to/SceneReVis/test/split_prompts"
EVAL_SCRIPT="/path/to/SceneReVis/eval/vlm_scene_eval.py"

cd /path/to/SceneReVis/eval

ROOMS=("bedroom" "living_room" "dining_room" "study_room")

for room in "${ROOMS[@]}"; do
    echo "========================================"
    echo "Evaluating: $room"
    echo "========================================"
    
    python "$EVAL_SCRIPT" \
        --render-dir "$BASE_RENDER/$room" \
        --prompts-file "$BASE_PROMPTS/$room.txt" \
        --json-dir "$BASE_JSON/$room" \
        --output "$BASE_RENDER/$room/vlm_evaluation_results.json" \
        --verbose
    
    echo ""
    echo "Completed: $room"
    echo ""
done

echo "All evaluations completed!"
