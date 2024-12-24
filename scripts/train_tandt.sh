#!/bin/bash

# 检查输入参数
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <algorithm>"
    echo "Algorithm options: baseline, baseline_aug, baseline_loss, baseline_aug_loss"
    exit 1
fi

# 获取算法类型
ALGORITHM=$1

# 定义根目录
ROOT_DIR="/home/stq/CV/3DGS/Topology-GS"

# 数据集和场景定义
declare -A datasets
datasets[tandt]="train truck"

# GPU 数量
num_gpus=8
gpu_index=0

# 遍历所有数据集和场景
for dataset in "${!datasets[@]}"; do
    for scene in ${datasets[$dataset]}; do
        # 根据算法类型构建配置文件路径
        case $ALGORITHM in
            baseline)
                CONFIG_FILE="${ROOT_DIR}/experiments/scaffold/${dataset}/${scene}/baseline.yaml"
                ;;
            baseline_aug)
                CONFIG_FILE="${ROOT_DIR}/experiments/topology/${dataset}/${scene}/baseline_aug.yaml"
                ;;
            baseline_loss)
                CONFIG_FILE="${ROOT_DIR}/experiments/topology/${dataset}/${scene}/baseline_loss.yaml"
                ;;
            baseline_aug_loss)
                CONFIG_FILE="${ROOT_DIR}/experiments/topology/${dataset}/${scene}/baseline_aug_loss.yaml"
                ;;
            *)
                echo "Unknown algorithm: $ALGORITHM"
                exit 1
                ;;
        esac

        # 设置 spath 和 opath
        SPATH="../data/${dataset}/${scene}"
        OPATH="../output/${dataset}/${ALGORITHM}/${scene}"

        # 打印修改后的配置文件内容
        echo "YAML content of ${dataset}/${scene}:"
        cat "$CONFIG_FILE"
        echo "-----------------------"

        # 计算当前任务应使用的 GPU
        current_gpu=$((gpu_index % num_gpus))

        # 使用子进程在后台运行训练脚本
        (
            CUDA_VISIBLE_DEVICES=$current_gpu LD_LIBRARY_PATH=foobar python "${ROOT_DIR}/train.py" --cfg "$CONFIG_FILE" --spath "$SPATH" --opath "$OPATH"
        ) &

        # 更新 GPU 索引
        ((gpu_index++))

        # 每次启动训练之间暂停 16 秒
        sleep 16
    done
done

# 等待所有后台进程完成
wait
echo "All jobs are completed."
