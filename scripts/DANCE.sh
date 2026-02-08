#!/bin/bash

i=5
snr_values=-4
gpu_ids=(0 1 4 5 6 7)  

echo "starting..."

# for i in "${!snr_values[@]}"; do
    echo "启动 SNR=${snr_values} 在 GPU ${gpu_ids[i]}"
    python run.py \
        --split_dir ./data_split \
        --model DANCE \
        --batch_size 64 \
        --epochs 80 \
        --lr 1e-3 \
        --noise_type emb \
        --snr_db "${snr_values}" \
        --gpu_id "${gpu_ids[i]}" \
        --checkpoint_dir ./checkpoints \
        --mode train 
    # sleep 5  
# done


wait
echo "所有DANCE训练任务已完成!"

