#!/bin/bash
# Automated experiment pipeline with auto-tuning

PROJECT_ROOT="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie"
EXP08_DIR="$PROJECT_ROOT/outputs/experiments/mnist_convlstm/exp08_balanced"
LOG_FILE="$EXP08_DIR/training.log"

echo "=== RT Prediction Auto-Tuning Pipeline ==="
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

# Function to analyze RT results
analyze_rt_results() {
    local results_file=$1
    python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$results_file')
model_mean = df['rt_pred_seconds'].mean()
human_mean = df['rt_human_seconds'].mean()
ratio = model_mean / human_mean
corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0,1]
print(f'{ratio:.2f},{corr:.4f}')
"
}

# Wait for training to complete
echo "Waiting for Exp08 training to complete..."
while ! check_training_complete "$LOG_FILE"; do
    LAST_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1)
    echo "[$(date '+%H:%M:%S')] $LAST_EPOCH still running..."
    sleep 180
done

echo ""
echo "=== Exp08 Training Complete ==="
echo "Finished at: $(date)"

# Find results file
RESULTS_FILE=$(find "$EXP08_DIR" -name "*_results.csv" 2>/dev/null | head -1)

if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "=== Analyzing Results ==="
    
    # Get RT ratio and correlation
    RESULTS=$(analyze_rt_results "$RESULTS_FILE")
    RT_RATIO=$(echo "$RESULTS" | cut -d',' -f1)
    RT_CORR=$(echo "$RESULTS" | cut -d',' -f2)
    
    echo "RT Ratio: $RT_RATIO"
    echo "RT Correlation: $RT_CORR"
    
    # Decision logic
    IS_GOOD=$(python3 -c "
ratio = float('$RT_RATIO')
corr = float('$RT_CORR')
# Good if ratio < 1.5 and correlation > 0.1
if ratio < 1.5 and corr > 0.1:
    print('good')
else:
    print('needs_improvement')
")
    
    if [ "$IS_GOOD" = "good" ]; then
        echo ""
        echo "=== SUCCESS! Results are good ==="
        echo "RT Ratio: $RT_RATIO (< 1.5)"
        echo "RT Correlation: $RT_CORR (> 0.1)"
        echo ""
        echo "Generating visualizations..."
        
        # Generate visualizations
        cd "$PROJECT_ROOT"
        python3 src/utils/visualize_rt_quick.py
        
        echo ""
        echo "=== Pipeline Complete ==="
    else
        echo ""
        echo "=== Results need improvement ==="
        echo "RT Ratio: $RT_RATIO (target: < 1.5)"
        echo "RT Correlation: $RT_CORR (target: > 0.1)"
        echo ""
        echo "Starting Exp09 with adjusted parameters..."
        
        # Create Exp09 with stronger RT constraints
        EXP09_DIR="$PROJECT_ROOT/outputs/experiments/mnist_convlstm/exp09_stronger_rt"
        mkdir -p "$EXP09_DIR"
        
        cd "$PROJECT_ROOT"
        PYTHONPATH="$PROJECT_ROOT" nohup python src/experiments/mnist_convlstm/train_model_balanced.py \
            --data_path "data/raw/rtnet/behavioral data.csv" \
            --output_dir "$EXP09_DIR" \
            --epochs 70 \
            --batch_size 64 \
            --lr 0.001 \
            --use_rt_loss \
            --rt_loss_weight 3.0 \
            --speed_penalty 0.2 \
            --time_steps 20 \
            --num_filter 16 \
            --kernel_size 3 \
            --fixed_noise \
            --device auto \
            > "$EXP09_DIR/training.log" 2>&1 &
        
        echo "Exp09 started with:"
        echo "  - RT Loss Weight: 3.0 (increased from 2.0)"
        echo "  - Speed Penalty: 0.2 (increased from 0.1)"
        echo "  - PID: $!"
        
        # Wait for Exp09
        echo ""
        echo "Waiting for Exp09 to complete..."
        while ! check_training_complete "$EXP09_DIR/training.log"; do
            sleep 180
        done
        
        # Analyze Exp09
        RESULTS_FILE_9=$(find "$EXP09_DIR" -name "*_results.csv" 2>/dev/null | head -1)
        if [ -f "$RESULTS_FILE_9" ]; then
            RESULTS_9=$(analyze_rt_results "$RESULTS_FILE_9")
            RT_RATIO_9=$(echo "$RESULTS_9" | cut -d',' -f1)
            RT_CORR_9=$(echo "$RESULTS_9" | cut -d',' -f2)
            
            echo ""
            echo "=== Exp09 Results ==="
            echo "RT Ratio: $RT_RATIO_9"
            echo "RT Correlation: $RT_CORR_9"
        fi
        
        echo ""
        echo "=== Pipeline Complete ==="
        echo "Compare Exp08 and Exp09 results to choose best model"
    fi
else
    echo "ERROR: Results file not found!"
fi

echo ""
echo "Finished at: $(date)"
