#!/bin/bash


OUTPUT_DIR=output
OUTPUT_DIR_FT=${OUTPUT_DIR}/ObjEmbed-2B-32cards
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
    --output_dir "./log/ObjEmbed-2B-32cards" \
    --model_name_or_path "fushh7/WeDetect-Ref-2B-stage2" \
    --dataset_path "YOUR_DATASET_PATH"
    --dataset_name "mixture_with_caption" \
    --deepspeed scripts/zero2.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --gradient_checkpointing true \
    --num_train_epochs 2 \
    --run_name ObjEmbed-2B-32cards \
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
    2>&1 | tee -a ${OUTPUT_DIR_FT}/log_node_$RANK.txt && echo "Done."

