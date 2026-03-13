#!/bin/bash

# Run training with nohup to prevent interruption when lid is closed

cd "$(dirname "$0")"

nohup python src/experiments/mnist_convlstm/02_train_model.py \
    --data_path data/RTNet_Dataset/behavioral\ data.csv \
    --epochs 50 \
    --batch_size 64 \
    --use_rt_loss \
    --output_dir output_convlstm_v2 \
    > training_nohup.log 2>&1 &

echo "Training started with nohup. Check training_nohup.log for progress."
echo "To monitor progress: tail -f training_nohup.log"
echo "To stop training: pkill -f 'python.*02_train_model.py'"
