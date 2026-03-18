#!/bin/bash
# VAM ConvLSTM Experiment Runner
# This script runs experiments for VAM Lost in Migration task

set -e

PROJECT_ROOT="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie"
DATA_DIR="VAM_Lost-in-Migration"
OUTPUT_BASE="outputs/experiments/vam_convlstm"

cd $PROJECT_ROOT

echo "=============================================="
echo "VAM ConvLSTM Experiments"
echo "=============================================="

# Experiment 1: Train from scratch with RT supervision
echo ""
echo "Experiment 1: Training from scratch..."
echo ""

EXP_DIR="${OUTPUT_BASE}/exp01_from_scratch"
mkdir -p $EXP_DIR

python src/experiments/vam_convlstm/train_vam.py \
    --data_dir $DATA_DIR \
    --output_dir $EXP_DIR \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --use_rt_loss \
    --num_filter 16 \
    --kernel_size 3 \
    --time_steps 20 \
    --device auto \
    --random_seed 42 \
    --train_ratio 0.8 \
    --max_trials_per_user 25000 \
    2>&1 | tee $EXP_DIR/training.log

echo ""
echo "Experiment 1 complete!"
echo ""

# Experiment 2: Fine-tune from MNIST pretrained model
echo ""
echo "Experiment 2: Fine-tuning from MNIST pretrained model..."
echo ""

PRETRAINED_MODEL="outputs/experiments/mnist_convlstm/learnable_noise_ep100/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp.pth"

if [ -f "$PRETRAINED_MODEL" ]; then
    EXP_DIR="${OUTPUT_BASE}/exp02_finetune_mnist"
    mkdir -p $EXP_DIR

    python src/experiments/vam_convlstm/train_vam.py \
        --data_dir $DATA_DIR \
        --output_dir $EXP_DIR \
        --epochs 50 \
        --batch_size 64 \
        --lr 0.0001 \
        --use_rt_loss \
        --num_filter 16 \
        --kernel_size 3 \
        --time_steps 20 \
        --device auto \
        --random_seed 42 \
        --train_ratio 0.8 \
        --max_trials_per_user 25000 \
        --pretrained_model $PRETRAINED_MODEL \
        2>&1 | tee $EXP_DIR/training.log

    echo ""
    echo "Experiment 2 complete!"
else
    echo "Pretrained model not found: $PRETRAINED_MODEL"
    echo "Skipping Experiment 2"
fi

# Experiment 3: Fine-tune with frozen encoder
echo ""
echo "Experiment 3: Fine-tuning with frozen encoder..."
echo ""

if [ -f "$PRETRAINED_MODEL" ]; then
    EXP_DIR="${OUTPUT_BASE}/exp03_finetune_frozen"
    mkdir -p $EXP_DIR

    python src/experiments/vam_convlstm/train_vam.py \
        --data_dir $DATA_DIR \
        --output_dir $EXP_DIR \
        --epochs 50 \
        --batch_size 64 \
        --lr 0.0001 \
        --use_rt_loss \
        --num_filter 16 \
        --kernel_size 3 \
        --time_steps 20 \
        --device auto \
        --random_seed 42 \
        --train_ratio 0.8 \
        --max_trials_per_user 25000 \
        --pretrained_model $PRETRAINED_MODEL \
        --freeze_encoder \
        2>&1 | tee $EXP_DIR/training.log

    echo ""
    echo "Experiment 3 complete!"
else
    echo "Pretrained model not found: $PRETRAINED_MODEL"
    echo "Skipping Experiment 3"
fi

echo ""
echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
