#!/bin/bash

# 定义SNR值和对应的GPU ID（避免内存冲突）
snr_values=(-4 -2 0 2 4)
gpu_ids=(0 1 2 6 7)  # 交替使用GPU

echo "开始并行训练所有SNR配置..."

for i in "${!snr_values[@]}"; do
    echo "启动 SNR=${snr_values[i]} 在 GPU ${gpu_ids[i]}"
    python run.py \
        --split_dir ./data_split \
        --model ACDAE \
        --batch_size 64 \
        --epochs 100 \
        --lr 1e-3 \
        --noise_type em \
        --snr_db "${snr_values[i]}" \
        --gpu_id "${gpu_ids[i]}" \
        --checkpoint_dir ./checkpoints \
        --mode train &
    sleep 5  # 间隔5秒启动，避免冲突
done

# 等待所有后台任务完成
wait
echo "所有ACDAE训练任务已完成!"