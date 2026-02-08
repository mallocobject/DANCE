#!/usr/bin/env bash

MODELS=("DANCE")
RUNS=5
MODE=train
NOISE_TYPES=("bw" "em" "ma" "emb")
NOISE_TYPES_LIMITED=("emb")
SNR_DBS_ALL=(-4 -2 0 2 4)  # emb 使用的所有SNR
SNR_DBS_LIMITED=(-4)       # bw/em/ma 使用的SNR
BATCH=64
EPOCHS=80
LR=1e-3
GPU_ID=0    # 可用 GPU 列表，按需修改
PLUGIN_TYPE=("SE" "CBAM" "ECA")

OUT_DIR=results
LOG_FILE="${OUT_DIR}/all_results4.log"

# 创建输出目录
mkdir -p "${OUT_DIR}"
mkdir -p checkpoints

echo "开始所有实验..."

# 清空日志文件
> "${LOG_FILE}"

for plugin_type in "${PLUGIN_TYPE[@]}"; do
    for snr in "${SNR_DBS_ALL[@]}"; do
        if [ "${plugin_type}" = "DANCE" ]; then
            total_tasks=$((total_tasks + ${#NOISE_TYPES[@]} * RUNS))
        else
            total_tasks=$((total_tasks + ${#NOISE_TYPES_LIMITED[@]} * RUNS))
        fi
    done
done

current_task=1

for plugin_type in "${PLUGIN_TYPE[@]}"; do
    for snr_db in "${SNR_DBS_ALL[@]}"; do
        # 根据噪声类型选择SNR数组
        if [ "${plugin_type}" = "DANCE" ]; then
            noise_array=("${NOISE_TYPES[@]}")
        else
            noise_array=("${NOISE_TYPES_LIMITED[@]}")
        fi
        
        for noise_type in "${noise_array[@]}"; do
            for ((run=1; run<=RUNS; run++)); do
                echo "==========================================" | tee -a "${LOG_FILE}"
                echo "[${current_task}/${total_tasks}] 模型: ${plugin_type}, 噪声: ${noise_type}, SNR: ${snr_db}dB, 第 ${run} 次运行" | tee -a "${LOG_FILE}"
                echo "==========================================" | tee -a "${LOG_FILE}"
                
                timestamp=$(date "+%Y%m%d_%H%M%S")
                
                # 执行训练
                python run.py \
                    --split_dir ./data_split \
                    --model DANCE \
                    --plugin "${plugin_type}" \
                    --batch_size "${BATCH}" \
                    --epochs "${EPOCHS}" \
                    --lr "${LR}" \
                    --noise_type "${noise_type}" \
                    --snr_db "${snr_db}" \
                    --gpu_id "${GPU_ID}" \
                    --checkpoint_dir "./checkpoints/${plugin_type}/${noise_type}/${snr_db}dB/run_${run}" \
                    --mode "${MODE}" \
                    2>&1 | tee -a "${LOG_FILE}"
                
                echo "完成: ${plugin_type} - ${noise_type} - ${snr_db}dB - 第 ${run} 次运行" | tee -a "${LOG_FILE}"
                echo "" | tee -a "${LOG_FILE}"
                
                ((current_task++))
            done
        done
    done
done

echo "==========================================" | tee -a "${LOG_FILE}"
echo "所有实验完成! 总任务数: ${total_tasks}" | tee -a "${LOG_FILE}"
echo "日志已保存到: ${LOG_FILE}" | tee -a "${LOG_FILE}"