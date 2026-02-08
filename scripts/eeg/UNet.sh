#!/bin/bash


i=1
snr_values=-4
gpu_ids=(0 4 5 6)  

echo "starting..."

# for i in "${!snr_values[@]}"; do
    echo "启动 SNR=${snr_values} 在 GPU ${gpu_ids[i]}"
    python run2.py \
        --split_dir ./DeepSeparator/data \
        --model ACDAE \
        --batch_size 128 \
        --epochs 60 \
        --lr 1e-4 \
        --noise_type EMG \
        --snr_db "${snr_values}" \
        --gpu_id "${gpu_ids[i]}" \
        --checkpoint_dir ./checkpoints \
        --mode train 
    # sleep 5  
# done


wait
echo "所有UNet训练任务已完成!"