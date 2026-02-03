#!/bin/bash


OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/exp6
mkdir -p ${OUTPUT_DIR_FT}


NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
PORT=${MASTER_PORT:-29513}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

torchrun --nproc_per_node 8 \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    train.py \
    --output_dir "./log/Qwen3-VL-4B-exp6-16cards" \
    --model_name_or_path "Qwen3-VL-4B-exp1-stage2" \
    --dataset_name "mixture_with_caption" \
    --deepspeed scripts/zero2.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name Qwen3-VL-4B-sft \
    --save_steps 1000 \
    --save_total_limit 2 \
    --max_grad_norm 5 \
    --dataloader_prefetch_factor 2 \
    --dataloader_num_workers 2 \
    --freeze_vision_modules true \
    --per_image_train_text_batch_size 10 \
    --num_classes 100 \
    --ddp_timeout 7200 \
    --use_task_prompt true \
    --use_global_caption true \
    --use_two_tokens 2 \
    --use_two_captions true \
    2>&1 | tee -a ${OUTPUT_DIR_FT}/16cards_log_node_$RANK.txt && echo "Done."

