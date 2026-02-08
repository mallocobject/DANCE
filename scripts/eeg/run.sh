#!/usr/bin/env bash

MODELS=("DANCE" "U-Net" "DACNN" "ACDAE" "RALENet")
RUNS=5
MODE=train
NOISE_TYPES=("EOG" "EMG" "EOGEMG")
NOISE_TYPES_LIMITED=("EOGEMG")
SNR_DBS_ALL=(-4 -2 0 2 4)
SNR_DBS_LIMITED=(-4)
BATCH=128
EPOCHS=60
LR=1e-4
GPU_ID=7    # 可用 GPU 列表，按需修改

OUT_DIR=results
LOG_FILE="${OUT_DIR}/all_results5.log"

# 创建输出目录
mkdir -p "${OUT_DIR}"
mkdir -p checkpoints

echo "开始所有实验..."

# 清空日志文件
> "${LOG_FILE}"

计算总任务数
total_tasks=0
for model in "${MODELS[@]}"; do
    for noise_type in "${NOISE_TYPES[@]}"; do
        if [ "${noise_type}" = "EOGEMG" ]; then
            total_tasks=$((total_tasks + ${#SNR_DBS_ALL[@]} * RUNS))
        else
            total_tasks=$((total_tasks + ${#SNR_DBS_LIMITED[@]} * RUNS))
        fi
    done
done

current_task=1

for model in "${MODELS[@]}"; do
    for noise_type in "${NOISE_TYPES[@]}"; do
        # 根据噪声类型选择SNR数组
        if [ "${noise_type}" = "EOGEMG" ]; then
            snr_array=("${SNR_DBS_ALL[@]}")
        else
            snr_array=("${SNR_DBS_LIMITED[@]}")
        fi
        
        for snr_db in "${snr_array[@]}"; do
            for ((run=1; run<=RUNS; run++)); do
                echo "==========================================" | tee -a "${LOG_FILE}"
                echo "[${current_task}/${total_tasks}] 模型: ${model}, 噪声: ${noise_type}, SNR: ${snr_db}dB, 第 ${run} 次运行" | tee -a "${LOG_FILE}"
                echo "==========================================" | tee -a "${LOG_FILE}"
                
                timestamp=$(date "+%Y%m%d_%H%M%S")

                if [ "${noise_type}" = "EOGEMG" ]; then
                    EPOCHS=120
                else
                    EPOCHS=60
                fi
                
                # 执行训练
                python run2.py \
                    --split_dir ./DeepSeparator/data \
                    --model "${model}" \
                    --batch_size "${BATCH}" \
                    --epochs "${EPOCHS}" \
                    --lr "${LR}" \
                    --noise_type "${noise_type}" \
                    --snr_db "${snr_db}" \
                    --gpu_id "${GPU_ID}" \
                    --checkpoint_dir "./checkpoints/${model}/${noise_type}/${snr_db}dB/run_${run}" \
                    --mode "${MODE}" \
                    2>&1 | tee -a "${LOG_FILE}"
                
                echo "完成: ${model} - ${noise_type} - ${snr_db}dB - 第 ${run} 次运行" | tee -a "${LOG_FILE}"
                echo "" | tee -a "${LOG_FILE}"
                
                ((current_task++))
            done
        done
    done
done

echo "==========================================" | tee -a "${LOG_FILE}"
echo "所有实验完成! 总任务数: ${total_tasks}" | tee -a "${LOG_FILE}"
echo "日志已保存到: ${LOG_FILE}" | tee -a "${LOG_FILE}"