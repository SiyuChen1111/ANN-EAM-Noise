#!/bin/bash
# Progress monitoring script for 100-epoch training

LOG_FILE="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp04_final/training.log"

echo "=== 100-Epoch Training Progress Monitor ==="
echo "Log file: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "ERROR: Log file not found!"
    exit 1
fi

# Check if training completed
if grep -q "Training Complete" "$LOG_FILE" 2>/dev/null; then
    echo "✅ TRAINING COMPLETED!"
    echo ""
    
    # Find results file
    RESULTS_FILE=$(find "/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp04_final" -name "*_results.csv" | head -1)
    if [ -f "$RESULTS_FILE" ]; then
        echo "=== Final Results ==="
        python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$RESULTS_FILE')
model_mean = df['rt_pred_seconds'].mean()
model_std = df['rt_pred_seconds'].std()
human_mean = df['rt_human_seconds'].mean()
human_std = df['rt_human_seconds'].std()
ratio = model_mean / human_mean
corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0,1]
print(f'Model RT: {model_mean:.3f} ± {model_std:.3f} seconds')
print(f'Human RT: {human_mean:.3f} ± {human_std:.3f} seconds')
print(f'RT Ratio (model/human): {ratio:.2f}')
print(f'RT Correlation: {corr:.4f}')
print(f'')
print(f'Improvement from baseline (ratio=3.0):')
print(f'  RT gap reduced by: {((3.0 - ratio) / 2.0 * 100):.1f}%')
"
    fi
    exit 0
fi

# Get current progress
CURRENT_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_FILE" 2>/dev/null | tail -1)
TOTAL_EPOCHS=100

if [ -n "$CURRENT_EPOCH" ]; then
    EPOCH_NUM=$(echo "$CURRENT_EPOCH" | grep -o "[0-9]*" | head -1)
    PROGRESS=$(echo "scale=1; $EPOCH_NUM * 100 / $TOTAL_EPOCHS" | bc)
    
    echo "Current Progress: $CURRENT_EPOCH ($PROGRESS%)"
    echo ""
    
    # Get latest metrics
    LAST_ACC=$(grep -o "acc_correct=[0-9.]*" "$LOG_FILE" | tail -1 | cut -d'=' -f2)
    LAST_CORR=$(grep -o "corr=[-0-9.]*" "$LOG_FILE" | tail -1 | cut -d'=' -f2)
    
    echo "Latest Metrics:"
    echo "  Accuracy: ${LAST_ACC:-N/A}"
    echo "  RT Correlation: ${LAST_CORR:-N/A}"
    echo ""
    
    # Estimate remaining time (assuming ~9 min per epoch)
    REMAINING=$((TOTAL_EPOCHS - EPOCH_NUM))
    REMAINING_MINUTES=$((REMAINING * 9))
    REMAINING_HOURS=$((REMAINING_MINUTES / 60))
    REMAINING_MINS=$((REMAINING_MINUTES % 60))
    
    echo "Estimated Remaining Time: ${REMAINING_HOURS}h ${REMAINING_MINS}m"
    echo ""
fi

# Show last few lines of log
echo "=== Last 10 lines of training log ==="
tail -10 "$LOG_FILE" | grep -o "Epoch.*" | tail -5
