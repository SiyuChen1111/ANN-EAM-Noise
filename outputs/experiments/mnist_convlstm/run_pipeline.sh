#!/bin/bash
# Automated experiment pipeline for RT prediction improvement

PROJECT_ROOT="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie"
EXP1_DIR="$PROJECT_ROOT/outputs/experiments/mnist_convlstm/exp02_timesteps100_quick"
EXP2_DIR="$PROJECT_ROOT/outputs/experiments/mnist_convlstm/exp03_log_norm_quick"
FINAL_DIR="$PROJECT_ROOT/outputs/experiments/mnist_convlstm/exp04_final"

LOG_FILE="$EXP1_DIR/training.log"

echo "=== RT Prediction Experiment Pipeline ==="
echo "Started at: $(date)"
echo ""

# Function to check if training is complete
check_training_complete() {
    local log_file=$1
    if grep -q "Training Complete" "$log_file" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Function to extract RT correlation from results
get_rt_correlation() {
    local results_file=$1
    if [ -f "$results_file" ]; then
        python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$results_file')
corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0,1]
print(f'{corr:.4f}')
"
    else
        echo "0.0"
    fi
}

# Function to analyze RT statistics
analyze_rt_stats() {
    local results_file=$1
    python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$results_file')
model_mean = df['rt_pred_seconds'].mean()
model_std = df['rt_pred_seconds'].std()
human_mean = df['rt_human_seconds'].mean()
human_std = df['rt_human_seconds'].std()
ratio = model_mean / human_mean
print(f'Model RT: {model_mean:.3f} +/- {model_std:.3f}')
print(f'Human RT: {human_mean:.3f} +/- {human_std:.3f}')
print(f'Ratio (model/human): {ratio:.2f}')
print(f'RT Correlation: {np.corrcoef(df[\"rt_pred_seconds\"], df[\"rt_human_seconds\"])[0,1]:.4f}')
"
}

# Wait for Experiment 1 to complete
echo "=== Waiting for Experiment 1 (time_steps=100) to complete ==="
while ! check_training_complete "$LOG_FILE"; do
    LAST_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1)
    echo "[$(date '+%H:%M:%S')] $LAST_EPOCH still running..."
    sleep 120
done

echo ""
echo "=== Experiment 1 Completed ==="
echo "Finished at: $(date)"
echo ""

# Find and analyze results
RESULTS_FILE=$(find "$EXP1_DIR" -name "*_results.csv" | head -1)
if [ -f "$RESULTS_FILE" ]; then
    echo "=== Experiment 1 Results ==="
    analyze_rt_stats "$RESULTS_FILE"
    echo ""
    
    # Get RT ratio
    RT_RATIO=$(python3 -c "
import pandas as pd
df = pd.read_csv('$RESULTS_FILE')
print(f'{df[\"rt_pred_seconds\"].mean() / df[\"rt_human_seconds\"].mean():.2f}')
")
    
    # Decision logic: if ratio < 2.0, consider it successful
    if (( $(echo "$RT_RATIO < 2.0" | bc -l) )); then
        echo "=== Experiment 1 SUCCESSFUL (RT ratio: $RT_RATIO) ==="
        echo "Proceeding with 100 epoch training..."
        
        # Run 100 epoch training with best config
        mkdir -p "$FINAL_DIR"
        cd "$PROJECT_ROOT"
        PYTHONPATH="$PROJECT_ROOT" nohup python src/experiments/mnist_convlstm/02_train_model.py \
            --data_path "data/raw/rtnet/behavioral data.csv" \
            --output_dir "$FINAL_DIR" \
            --epochs 100 \
            --batch_size 64 \
            --lr 0.001 \
            --use_rt_loss \
            --time_steps 100 \
            --num_filter 16 \
            --kernel_size 3 \
            --fixed_noise \
            --device auto \
            > "$FINAL_DIR/training.log" 2>&1 &
        
        echo "100 epoch training started. PID: $!"
        echo "Monitor with: tail -f $FINAL_DIR/training.log"
    else
        echo "=== Experiment 1 NOT optimal (RT ratio: $RT_RATIO) ==="
        echo "Starting Experiment 2 (log normalization)..."
        
        # Run Experiment 2
        mkdir -p "$EXP2_DIR"
        cd "$PROJECT_ROOT"
        PYTHONPATH="$PROJECT_ROOT" nohup python src/experiments/mnist_convlstm/train_model_log.py \
            --data_path "data/raw/rtnet/behavioral data.csv" \
            --output_dir "$EXP2_DIR" \
            --epochs 10 \
            --batch_size 64 \
            --lr 0.001 \
            --use_rt_loss \
            --time_steps 100 \
            --num_filter 16 \
            --kernel_size 3 \
            --fixed_noise \
            --device auto \
            > "$EXP2_DIR/training.log" 2>&1 &
        
        EXP2_PID=$!
        echo "Experiment 2 started. PID: $EXP2_PID"
        
        # Wait for Experiment 2
        echo "Waiting for Experiment 2 to complete..."
        while ! check_training_complete "$EXP2_DIR/training.log"; do
            sleep 120
        done
        
        # Analyze Experiment 2
        RESULTS_FILE_2=$(find "$EXP2_DIR" -name "*_results.csv" | head -1)
        if [ -f "$RESULTS_FILE_2" ]; then
            echo ""
            echo "=== Experiment 2 Results ==="
            analyze_rt_stats "$RESULTS_FILE_2"
            
            # Compare and choose best
            RT_RATIO_2=$(python3 -c "
import pandas as pd
df = pd.read_csv('$RESULTS_FILE_2')
print(f'{df[\"rt_pred_seconds\"].mean() / df[\"rt_human_seconds\"].mean():.2f}')
")
            
            # Run 100 epoch with best approach
            if (( $(echo "$RT_RATIO_2 < $RT_RATIO" | bc -l) )); then
                echo "Experiment 2 is better. Running 100 epoch with log normalization..."
                BEST_SCRIPT="train_model_log.py"
                BEST_DIR="$FINAL_DIR"
            else
                echo "Experiment 1 is better. Running 100 epoch with time_steps=100..."
                BEST_SCRIPT="02_train_model.py"
                BEST_DIR="$FINAL_DIR"
            fi
            
            mkdir -p "$BEST_DIR"
            cd "$PROJECT_ROOT"
            PYTHONPATH="$PROJECT_ROOT" nohup python "src/experiments/mnist_convlstm/$BEST_SCRIPT" \
                --data_path "data/raw/rtnet/behavioral data.csv" \
                --output_dir "$BEST_DIR" \
                --epochs 100 \
                --batch_size 64 \
                --lr 0.001 \
                --use_rt_loss \
                --time_steps 100 \
                --num_filter 16 \
                --kernel_size 3 \
                --fixed_noise \
                --device auto \
                > "$BEST_DIR/training.log" 2>&1 &
            
            echo "100 epoch training started. PID: $!"
        fi
    fi
else
    echo "ERROR: Results file not found!"
fi

echo ""
echo "=== Pipeline Complete ==="
echo "Finished at: $(date)"
