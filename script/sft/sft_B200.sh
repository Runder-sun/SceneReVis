# Run with 16 GPUs (MI200)
# Configuration optimized for 30k+ length
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset '/path/to/workspace/datasets/sft/train_dataset_v3_create.jsonl' \
    --split_dataset_ratio 0.05 \
    --train_type full \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --packing true \
    --padding_free true \
    --attn_impl flash_attn \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 50 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 51200 \
    --truncation_strategy left \
    --output_dir /path/to/workspace/sft_output_b200_8card_50k \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --save_only_model true \
    --deepspeed zero3 \
    --gradient_checkpointing true \
    --report_to swanlab \
    --swanlab_project llmscene \
    --swanlab_token YOUR_SWANLAB_TOKEN