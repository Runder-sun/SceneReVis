#!/bin/bash

# Batch run script for testing different prompts with iterative scene generation
# This script reads prompts from extracted_user_instructions.txt
# to evaluate the scene generation system
#
# Usage:
#   ./batch_run.sh                    # Process all prompts
#   MAX_PROMPTS=50 ./batch_run.sh     # Process only first 50 prompts
#   MAX_PROMPTS=300 ./batch_run.sh    # Process only first 300 prompts

# Set base output directory
BASE_OUTPUT_DIR="./output/batch_evaluation_rl_scene_v2"
mkdir -p "$BASE_OUTPUT_DIR"

# Create a centralized directory for all final scene JSON files
FINAL_SCENES_DIR="$BASE_OUTPUT_DIR/final_scenes_collection"
mkdir -p "$FINAL_SCENES_DIR"

# Generate timestamp for unique naming
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Set number of iterations per run
ITERATIONS=10

# Set maximum number of prompts to process (set to 0 or leave empty for all prompts)
# You can override this by setting MAX_PROMPTS environment variable
# Example: MAX_PROMPTS=50 ./batch_run.sh
MAX_PROMPTS=${MAX_PROMPTS:-0}  # Default to 0 (process all)

# Voxel evaluation settings
MODELS_BASE_PATH="/path/to/datasets/3d-front/3D-FUTURE-model/"
VOXEL_SIZE=0.05
SCENE_FORMAT="ours"

# Function to ask user for prompt limit if not set
ask_prompt_limit() {
    local total_prompts=$1
    echo "" >&2
    echo "Total prompts available: $total_prompts" >&2
    echo "Options:" >&2
    echo "  1) Process all prompts ($total_prompts)" >&2
    echo "  2) Process first 50 prompts" >&2
    echo "  3) Process first 100 prompts" >&2
    echo "  4) Process first 300 prompts" >&2
    echo "  5) Enter custom number" >&2
    echo "" >&2
    read -p "Select option (1-5): " choice
    
    case $choice in
        1) echo "0" ;;
        2) echo "50" ;;
        3) echo "100" ;;
        4) echo "300" ;;
        5) 
            read -p "Enter number of prompts to process: " custom_num
            if [[ $custom_num =~ ^[0-9]+$ ]] && [ $custom_num -gt 0 ] && [ $custom_num -le $total_prompts ]; then
                echo "$custom_num"
            else
                echo "Invalid number. Processing all prompts." >&2
                echo "0"
            fi
            ;;
        *)
            echo "Invalid choice. Processing all prompts." >&2
            echo "0"
            ;;
    esac
}

# Path to the extracted user instructions file
USER_INSTRUCTIONS_FILE="/path/to/datasets/llmscene/sft/test_prompt_v2.txt"

# Check if the user instructions file exists
if [ ! -f "$USER_INSTRUCTIONS_FILE" ]; then
    echo "Error: User instructions file not found: $USER_INSTRUCTIONS_FILE"
    echo "Please run the extract_user_instructions.py script first to generate the file."
    exit 1
fi

# Read all prompts from the user instructions file
echo "Reading prompts from: $USER_INSTRUCTIONS_FILE"
mapfile -t ALL_PROMPTS < "$USER_INSTRUCTIONS_FILE"

# Determine how many prompts to process
TOTAL_PROMPTS=${#ALL_PROMPTS[@]}

# If MAX_PROMPTS is not set (0), ask user interactively
if [ "$MAX_PROMPTS" -eq 0 ]; then
    MAX_PROMPTS=$(ask_prompt_limit $TOTAL_PROMPTS)
fi

if [ "$MAX_PROMPTS" -gt 0 ] && [ "$MAX_PROMPTS" -lt "$TOTAL_PROMPTS" ]; then
    PROMPTS=("${ALL_PROMPTS[@]:0:$MAX_PROMPTS}")
    echo "Processing limited set: $MAX_PROMPTS out of $TOTAL_PROMPTS prompts"
else
    PROMPTS=("${ALL_PROMPTS[@]}")
    echo "Processing all prompts: $TOTAL_PROMPTS prompts"
fi

# Generate corresponding output directory names (just numbers)
declare -a OUTPUT_NAMES=()
for i in $(seq 1 ${#PROMPTS[@]}); do
    OUTPUT_NAMES+=("$i")
done

echo "Starting batch evaluation with ${#PROMPTS[@]} different prompts..."
if [ "$MAX_PROMPTS" -gt 0 ] && [ "$MAX_PROMPTS" -lt "$TOTAL_PROMPTS" ]; then
    echo "Prompt limit: $MAX_PROMPTS (out of $TOTAL_PROMPTS total prompts available)"
else
    echo "Prompt limit: All prompts ($TOTAL_PROMPTS)"
fi
echo "Base output directory: $BASE_OUTPUT_DIR"
echo "Final scenes collection: $FINAL_SCENES_DIR"
echo "Timestamp: $TIMESTAMP"
echo "Iterations per run: $ITERATIONS"
echo "================================"

# Function to process a single prompt
process_prompt() {
    local i=$1
    local PROMPT=$2
    local OUTPUT_NAME=$3
    local OUTPUT_DIR="$BASE_OUTPUT_DIR/$OUTPUT_NAME"
    local TOTAL=${#PROMPTS[@]}
    
    echo ""
    echo "[$((i+1))/$TOTAL] Running prompt: $OUTPUT_NAME"
    echo "Prompt: $PROMPT"
    echo "Output: $OUTPUT_DIR"
    echo "----------------------------------------"
    
    # Run the inference with room generation
    python infer_batch.py \
        --iterations "$ITERATIONS" \
        --prompt "$PROMPT" \
        --output "$OUTPUT_DIR" \
        --generate-room \
        --use-model-for-creation
    
    local exit_code=$?
    
    # Check if the run was successful
    if [ $exit_code -eq 0 ]; then
        echo "[$OUTPUT_NAME] ✓ Successfully completed"
        
        # Copy final scene JSON files to centralized collection with unique naming
        FINAL_SCENE_FILES=$(find "$OUTPUT_DIR" -name "scene_iter_*.json" | sort -V | tail -1)
        if [ -n "$FINAL_SCENE_FILES" ] && [ -f "$FINAL_SCENE_FILES" ]; then
            # Create unique filename: timestamp_序号_final.json (only scene JSON in collection)
            UNIQUE_FILENAME="${TIMESTAMP}_${OUTPUT_NAME}_final.json"
            cp "$FINAL_SCENE_FILES" "$FINAL_SCENES_DIR/$UNIQUE_FILENAME"
            echo "[$OUTPUT_NAME] ✓ Final scene copied: $UNIQUE_FILENAME"
        else
            echo "[$OUTPUT_NAME] ⚠ Warning: No final scene JSON file found"
        fi
        
        # Check if summary image was created
        if [ -f "$OUTPUT_DIR/iteration_summary.png" ]; then
            echo "[$OUTPUT_NAME] ✓ Summary image created"
        else
            echo "[$OUTPUT_NAME] ⚠ Warning: No summary image found"
        fi
    else
        echo "[$OUTPUT_NAME] ✗ Failed to complete (exit code: $exit_code)"
    fi
    
    echo "[$OUTPUT_NAME] ----------------------------------------"
    
    return $exit_code
}

# Run all prompts sequentially
echo "Running sequentially (1 job at a time)..."

for i in "${!PROMPTS[@]}"; do
    process_prompt "$i" "${PROMPTS[$i]}" "${OUTPUT_NAMES[$i]}"
done

# After all processing is complete, continue with summary generation
echo ""
echo "All scene generation tasks completed!"
echo "Collecting results and generating summary..."

# Temporary: Re-scan the results since they were processed in parallel
# This ensures we capture all results correctly
for i in "${!PROMPTS[@]}"; do
    # This loop is just for verification after parallel processing
    # The actual file copying was done in process_prompt function
    continue
done

echo ""
echo "================================"
echo "Batch evaluation completed!"
echo "Results saved in: $BASE_OUTPUT_DIR"
echo "Final scene JSON files collected in: $FINAL_SCENES_DIR"

# Count collected final scene files
COLLECTED_FILES=$(find "$FINAL_SCENES_DIR" -name "*_final.json" 2>/dev/null | wc -l)
echo "Total final scene files collected: $COLLECTED_FILES"

# Create a summary report
SUMMARY_FILE="$BASE_OUTPUT_DIR/batch_summary.txt"
echo "Batch Evaluation Summary" > "$SUMMARY_FILE"
echo "========================" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "Source file: $USER_INSTRUCTIONS_FILE" >> "$SUMMARY_FILE"
echo "Total prompts: ${#PROMPTS[@]}" >> "$SUMMARY_FILE"
echo "Iterations per run: $ITERATIONS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Check results for each run
for i in "${!PROMPTS[@]}"; do
    OUTPUT_NAME="${OUTPUT_NAMES[$i]}"
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$OUTPUT_NAME"
    
    echo "[$((i+1))] $OUTPUT_NAME" >> "$SUMMARY_FILE"
    echo "    Prompt: ${PROMPTS[$i]}" >> "$SUMMARY_FILE"
    echo "    Output: $OUTPUT_DIR" >> "$SUMMARY_FILE"
    
    if [ -d "$OUTPUT_DIR" ]; then
        SCENE_FILES=$(find "$OUTPUT_DIR" -name "scene_iter_*.json" | wc -l)
        echo "    Status: ✓ Completed ($SCENE_FILES scene files)" >> "$SUMMARY_FILE"
        
        if [ -f "$OUTPUT_DIR/iteration_summary.png" ]; then
            echo "    Summary image: ✓ Available" >> "$SUMMARY_FILE"
        else
            echo "    Summary image: ✗ Missing" >> "$SUMMARY_FILE"
        fi
        
        if [ -f "$OUTPUT_DIR/conversation_history.txt" ]; then
            echo "    Conversation history: ✓ Available" >> "$SUMMARY_FILE"
        else
            echo "    Conversation history: ✗ Missing" >> "$SUMMARY_FILE"
        fi
        
        # Check if final scene was collected
        FINAL_SCENE_COLLECTED=$(find "$FINAL_SCENES_DIR" -name "${TIMESTAMP}_${OUTPUT_NAME}_final.json" 2>/dev/null | wc -l)
        if [ $FINAL_SCENE_COLLECTED -gt 0 ]; then
            echo "    Final scene collected: ✓ Available" >> "$SUMMARY_FILE"
        else
            echo "    Final scene collected: ✗ Missing" >> "$SUMMARY_FILE"
        fi
    else
        echo "    Status: ✗ Failed (no output directory)" >> "$SUMMARY_FILE"
    fi
    echo "" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"
echo "Final Scene Collection Summary:" >> "$SUMMARY_FILE"
echo "==============================" >> "$SUMMARY_FILE"
echo "Collection directory: $FINAL_SCENES_DIR" >> "$SUMMARY_FILE"
echo "Total files collected: $COLLECTED_FILES" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# List all collected final scene files
if [ $COLLECTED_FILES -gt 0 ]; then
    echo "Collected final scene files:" >> "$SUMMARY_FILE"
    find "$FINAL_SCENES_DIR" -name "*_final.json" -exec basename {} \; | sort >> "$SUMMARY_FILE"
fi

echo "Summary report created: $SUMMARY_FILE"
echo "Use 'cat $SUMMARY_FILE' to view the detailed results"

# ================================
# VOXEL EVALUATION
# ================================

if [ $COLLECTED_FILES -gt 0 ]; then
    echo ""
    echo "================================"
    echo "STARTING VOXEL-BASED EVALUATION"
    echo "================================"
    echo "Evaluating $COLLECTED_FILES scenes from: $FINAL_SCENES_DIR"
    echo "Scene format: $SCENE_FORMAT"
    echo "Voxel size: $VOXEL_SIZE meters"
    echo "Models path: $MODELS_BASE_PATH"
    
    # Define voxel evaluation output file
    VOXEL_EVAL_OUTPUT="$BASE_OUTPUT_DIR/voxel_evaluation_results_${TIMESTAMP}.json"
    
    # Run voxel evaluation
    echo "Running voxel evaluation script..."
    python eval/voxel_eval.py \
        --format "$SCENE_FORMAT" \
        --scenes_dir "$FINAL_SCENES_DIR" \
        --models_path "$MODELS_BASE_PATH" \
        --output_file "$VOXEL_EVAL_OUTPUT" \
        --voxel_size "$VOXEL_SIZE"
    
    # Check if voxel evaluation was successful
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Voxel evaluation completed successfully!"
        echo "Results saved to: $VOXEL_EVAL_OUTPUT"
        
        # Append voxel evaluation summary to the batch summary file
        echo "" >> "$SUMMARY_FILE"
        echo "Voxel-Based Evaluation Summary:" >> "$SUMMARY_FILE"
        echo "==============================" >> "$SUMMARY_FILE"
        echo "Evaluation timestamp: $TIMESTAMP" >> "$SUMMARY_FILE"
        echo "Scenes evaluated: $COLLECTED_FILES" >> "$SUMMARY_FILE"
        echo "Voxel size: $VOXEL_SIZE meters" >> "$SUMMARY_FILE"
        echo "Results file: $VOXEL_EVAL_OUTPUT" >> "$SUMMARY_FILE"
        echo "" >> "$SUMMARY_FILE"
        
        # Extract key metrics from voxel evaluation results if possible
        if [ -f "$VOXEL_EVAL_OUTPUT" ]; then
            echo "Key metrics from voxel evaluation:" >> "$SUMMARY_FILE"
            # Use python to extract summary metrics
            python -c "
import json
import sys
try:
    with open('$VOXEL_EVAL_OUTPUT', 'r') as f:
        data = json.load(f)
    summary = data.get('batch_evaluation_summary', {})
    print('  Total OOB loss: {:.2f} ± {:.2f}'.format(
        summary.get('mean_total_oob_loss_scaled', 0),
        summary.get('std_total_oob_loss_scaled', 0)))
    print('  Total MBL loss: {:.2f} ± {:.2f}'.format(
        summary.get('mean_total_mbl_loss_scaled', 0),
        summary.get('std_total_mbl_loss_scaled', 0)))
    print('  Total PBL loss: {:.2f} ± {:.2f}'.format(
        summary.get('mean_total_pbl_loss_scaled', 0),
        summary.get('std_total_pbl_loss_scaled', 0)))
    print('  Valid scene ratio: {:.2f}'.format(
        summary.get('valid_scene_ratio_pbl', 0)))
    print('  Valid scenes: {}/{}'.format(
        summary.get('valid_scenes_count', 0),
        summary.get('successful_scenes', 0)))
except Exception as e:
    print('  Error extracting metrics: {}'.format(str(e)))
" >> "$SUMMARY_FILE"
        fi
        
        echo "" >> "$SUMMARY_FILE"
        echo "For detailed voxel evaluation results, see: $VOXEL_EVAL_OUTPUT" >> "$SUMMARY_FILE"
        
    else
        echo "✗ Voxel evaluation failed!"
        echo "" >> "$SUMMARY_FILE"
        echo "Voxel-Based Evaluation:" >> "$SUMMARY_FILE"
        echo "Status: ✗ FAILED" >> "$SUMMARY_FILE"
    fi
    
    echo "================================"
    echo "ALL TASKS COMPLETED"
    echo "================================"
    echo "Scene generation: ✓ $COLLECTED_FILES scenes"
    echo "Batch summary: $SUMMARY_FILE"
    if [ -f "$VOXEL_EVAL_OUTPUT" ]; then
        echo "Voxel evaluation: ✓ $VOXEL_EVAL_OUTPUT"
    else
        echo "Voxel evaluation: ✗ Failed"
    fi
    
else
    echo ""
    echo "⚠ Warning: No final scenes were collected, skipping voxel evaluation"
    echo "" >> "$SUMMARY_FILE"
    echo "Voxel-Based Evaluation:" >> "$SUMMARY_FILE"
    echo "Status: SKIPPED (no scenes to evaluate)" >> "$SUMMARY_FILE"
fi

echo ""
echo "Complete batch summary saved to: $SUMMARY_FILE"