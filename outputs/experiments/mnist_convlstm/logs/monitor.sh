#!/bin/bash
# Continuous monitoring script for RT experiments

LOG_DIR="/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm"
PROGRESS_LOG="$LOG_DIR/logs/progress_2026-03-15.md"

echo "=== RT Experiment Monitor ==="
echo "Started at: $(date)"
echo ""

while true; do
    # Check if both experiments are complete
    EXP05_COMPLETE=$(grep -c "Training Complete" "$LOG_DIR/exp05_log_norm_quick/training.log" 2>/dev/null || echo "0")
    EXP06_COMPLETE=$(grep -c "Training Complete" "$LOG_DIR/exp06_timesteps50_quick/training.log" 2>/dev/null || echo "0")
    
    if [ "$EXP05_COMPLETE" -ge 1 ] && [ "$EXP06_COMPLETE" -ge 1 ]; then
        echo "=== BOTH EXPERIMENTS COMPLETE ==="
        echo "Finished at: $(date)"
        
        # Extract results
        echo "" >> "$PROGRESS_LOG"
        echo "### $(date '+%Y-%m-%d %H:%M') - Experiments Complete" >> "$PROGRESS_LOG"
        echo "" >> "$PROGRESS_LOG"
        
        # Get Exp05 results
        EXP05_RESULTS=$(find "$LOG_DIR/exp05_log_norm_quick" -name "*_results.csv" 2>/dev/null | head -1)
        if [ -f "$EXP05_RESULTS" ]; then
            echo "#### Exp05 (Log Normalization) Results:" >> "$PROGRESS_LOG"
            python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$EXP05_RESULTS')
model_mean = df['rt_pred_seconds'].mean()
human_mean = df['rt_human_seconds'].mean()
ratio = model_mean / human_mean
corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0,1]
print(f'- RT Ratio: {ratio:.2f}x')
print(f'- Model RT: {model_mean:.3f}s, Human RT: {human_mean:.3f}s')
print(f'- RT Correlation: {corr:.4f}')
" >> "$PROGRESS_LOG" 2>/dev/null
        fi
        
        # Get Exp06 results
        EXP06_RESULTS=$(find "$LOG_DIR/exp06_timesteps50_quick" -name "*_results.csv" 2>/dev/null | head -1)
        if [ -f "$EXP06_RESULTS" ]; then
            echo "" >> "$PROGRESS_LOG"
            echo "#### Exp06 (time_steps=50) Results:" >> "$PROGRESS_LOG"
            python3 -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$EXP06_RESULTS')
model_mean = df['rt_pred_seconds'].mean()
human_mean = df['rt_human_seconds'].mean()
ratio = model_mean / human_mean
corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0,1]
print(f'- RT Ratio: {ratio:.2f}x')
print(f'- Model RT: {model_mean:.3f}s, Human RT: {human_mean:.3f}s')
print(f'- RT Correlation: {corr:.4f}')
" >> "$PROGRESS_LOG" 2>/dev/null
        fi
        
        break
    fi
    
    # Get current progress
    EXP05_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_DIR/exp05_log_norm_quick/training.log" 2>/dev/null | tail -1)
    EXP06_EPOCH=$(grep -o "Epoch [0-9]*/[0-9]*" "$LOG_DIR/exp06_timesteps50_quick/training.log" 2>/dev/null | tail -1)
    
    echo "[$(date '+%H:%M:%S')] Exp05: $EXP05_EPOCH | Exp06: $EXP06_EPOCH"
    
    sleep 300  # Check every 5 minutes
done

echo ""
echo "Monitor finished."
