#!/bin/bash

# RT分布匹配改进实验批量运行脚本
# 每个实验使用nohup后台运行，合上笔记本也不会中断

EXPERIMENTS_DIR="outputs/experiments/mnist_convlstm"
DATA_PATH="data/raw/rtnet/behavioral data.csv"
EPOCHS=30
BATCH_SIZE=64

echo "=============================================="
echo "RT Distribution Matching Experiments"
echo "=============================================="
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=============================================="

# 方案2: RT分布匹配损失
echo ""
echo "Starting Exp02: RT Distribution Loss..."
mkdir -p "$EXPERIMENTS_DIR/exp02_distribution_loss"
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
    --data_path "$DATA_PATH" \
    --output_dir "$EXPERIMENTS_DIR/exp02_distribution_loss" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_rt_loss \
    --rt_dist_weight 0.5 \
    --experiment_name exp02_distribution_loss \
    > "$EXPERIMENTS_DIR/exp02_distribution_loss/training.log" 2>&1 &
echo "Exp02 started. PID: $!"
sleep 2

# 方案3: 偏度惩罚
echo ""
echo "Starting Exp03: Skewness Penalty..."
mkdir -p "$EXPERIMENTS_DIR/exp03_skewness_penalty"
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
    --data_path "$DATA_PATH" \
    --output_dir "$EXPERIMENTS_DIR/exp03_skewness_penalty" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_rt_loss \
    --skewness_weight 0.1 \
    --experiment_name exp03_skewness_penalty \
    > "$EXPERIMENTS_DIR/exp03_skewness_penalty/training.log" 2>&1 &
echo "Exp03 started. PID: $!"
sleep 2

# 方案4: 噪声参数正则化
echo ""
echo "Starting Exp04: Noise Regularization..."
mkdir -p "$EXPERIMENTS_DIR/exp04_noise_regularization"
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
    --data_path "$DATA_PATH" \
    --output_dir "$EXPERIMENTS_DIR/exp04_noise_regularization" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_rt_loss \
    --min_noise_std 0.05 \
    --experiment_name exp04_noise_regularization \
    > "$EXPERIMENTS_DIR/exp04_noise_regularization/training.log" 2>&1 &
echo "Exp04 started. PID: $!"
sleep 2

# 方案5: 调整阈值初始化
echo ""
echo "Starting Exp05: Threshold Initialization..."
mkdir -p "$EXPERIMENTS_DIR/exp05_threshold_init"
nohup python -m src.experiments.mnist_convlstm.04_train_rt_distribution \
    --data_path "$DATA_PATH" \
    --output_dir "$EXPERIMENTS_DIR/exp05_threshold_init" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --use_rt_loss \
    --initial_threshold 2.0 \
    --experiment_name exp05_threshold_init \
    > "$EXPERIMENTS_DIR/exp05_threshold_init/training.log" 2>&1 &
echo "Exp05 started. PID: $!"
sleep 2

echo ""
echo "=============================================="
echo "All experiments started!"
echo "=============================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f $EXPERIMENTS_DIR/exp*/training.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep 04_train_rt_distribution"
echo ""
