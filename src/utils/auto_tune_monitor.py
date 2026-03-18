"""
Auto-tuning script for RT prediction model.
Monitors training, analyzes results, and automatically runs next experiment if needed.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

PROJECT_ROOT = "/Users/siyu/Documents/GitHub/ANN-EAM-Nosie"
EXP08_DIR = os.path.join(PROJECT_ROOT, "outputs/experiments/mnist_convlstm/exp08_balanced")

def check_training_complete(log_path):
    """Check if training has completed."""
    if not os.path.exists(log_path):
        return False
    with open(log_path, 'r') as f:
        content = f.read()
    return "Training Complete!" in content

def get_current_epoch(log_path):
    """Get current epoch from log."""
    if not os.path.exists(log_path):
        return 0
    with open(log_path, 'r') as f:
        content = f.read()
    import re
    matches = re.findall(r'Epoch (\d+)/\d+', content)
    if matches:
        return int(matches[-1])
    return 0

def analyze_results(results_path):
    """Analyze RT results and return metrics."""
    df = pd.read_csv(results_path)
    
    model_mean = df['rt_pred_seconds'].mean()
    model_std = df['rt_pred_seconds'].std()
    human_mean = df['rt_human_seconds'].mean()
    human_std = df['rt_human_seconds'].std()
    ratio = model_mean / human_mean
    corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
    accuracy = (df['pred_label'] == df['true_label']).mean()
    
    return {
        'model_rt_mean': model_mean,
        'model_rt_std': model_std,
        'human_rt_mean': human_mean,
        'human_rt_std': human_std,
        'rt_ratio': ratio,
        'rt_correlation': corr,
        'accuracy': accuracy
    }

def main():
    log_path = os.path.join(EXP08_DIR, "training.log")
    
    print("=" * 60)
    print("RT Prediction Auto-Tuning Monitor")
    print("=" * 60)
    print(f"\nMonitoring: {log_path}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Wait for training to complete
    print("\nWaiting for training to complete...")
    last_epoch = 0
    while not check_training_complete(log_path):
        current_epoch = get_current_epoch(log_path)
        if current_epoch != last_epoch:
            print(f"  Current: Epoch {current_epoch}/70")
            last_epoch = current_epoch
        time.sleep(60)
    
    print("\n✅ Training completed!")
    
    # Find results file
    results_files = [f for f in os.listdir(EXP08_DIR) if f.endswith('_results.csv')]
    if not results_files:
        print("ERROR: No results file found!")
        return
    
    results_path = os.path.join(EXP08_DIR, results_files[0])
    print(f"\nAnalyzing results: {results_path}")
    
    metrics = analyze_results(results_path)
    
    print("\n" + "=" * 60)
    print("Exp08 (Balanced Loss) Final Results")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Model RT: {metrics['model_rt_mean']:.3f} ± {metrics['model_rt_std']:.3f} seconds")
    print(f"Human RT: {metrics['human_rt_mean']:.3f} ± {metrics['human_rt_std']:.3f} seconds")
    print(f"RT Ratio (model/human): {metrics['rt_ratio']:.2f}x")
    print(f"RT Correlation: {metrics['rt_correlation']:.4f}")
    
    # Decision logic
    print("\n" + "=" * 60)
    print("Analysis & Next Steps")
    print("=" * 60)
    
    if metrics['rt_ratio'] < 1.5:
        print("✅ SUCCESS! RT ratio is acceptable (< 1.5x)")
        print("   Model is ready for final visualization.")
    elif metrics['rt_ratio'] < 2.0:
        print("⚠️ MODERATE improvement. RT ratio is between 1.5x and 2.0x")
        print("   Suggestions for further improvement:")
        print("   1. Increase speed_penalty to 0.2")
        print("   2. Increase rt_loss_weight to 3.0")
        print("   3. Try dynamic RT loss weight scheduling")
    else:
        print("❌ RT ratio is still high (> 2.0x)")
        print("   Need more aggressive tuning:")
        print("   1. Increase speed_penalty to 0.3")
        print("   2. Increase rt_loss_weight to 5.0")
        print("   3. Consider early stopping based on RT ratio")
    
    print(f"\nFinished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
