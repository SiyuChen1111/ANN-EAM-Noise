"""
Visualize experiment comparison: ACC and RT across different experiments.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def extract_epoch_metrics_from_log(log_path):
    """Extract accuracy and RT metrics from training log."""
    epochs = []
    accuracies = []
    rt_correlations = []
    
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()
    
    # Find all epoch lines with accuracy
    pattern = r'Epoch (\d+)/\d+.*acc_correct=([0-9.]+).*corr=([-0-9.]+)'
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch = int(match[0])
        acc = float(match[1])
        corr = float(match[2])
        epochs.append(epoch)
        accuracies.append(acc)
        rt_correlations.append(corr)
    
    if len(epochs) == 0:
        return None
    
    df = pd.DataFrame({
        'epoch': epochs,
        'accuracy': accuracies,
        'rt_correlation': rt_correlations
    })
    
    # Average by epoch
    df = df.groupby('epoch').agg({
        'accuracy': 'mean',
        'rt_correlation': 'mean'
    }).reset_index()
    
    return df

def main():
    OUTPUT_DIR = 'outputs/experiments/mnist_convlstm'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Experiment configurations
    experiments = {
        'Exp07 (t=20, rt_w=1.0)': {
            'results': 'outputs/experiments/mnist_convlstm/exp07_log_norm_full/convlstm_log_t20_rt_sup_log_human_resp_results.csv',
            'log': 'outputs/experiments/mnist_convlstm/exp07_log_norm_full/training.log',
            'time_steps': 20,
            'rt_loss_weight': 1.0,
            'color': '#1f77b4'
        },
        'Exp08 (t=20, rt_w=2.0)': {
            'results': 'outputs/experiments/mnist_convlstm/exp08_balanced/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv',
            'log': 'outputs/experiments/mnist_convlstm/exp08_balanced/training.log',
            'time_steps': 20,
            'rt_loss_weight': 2.0,
            'color': '#ff7f0e'
        },
        'Exp10 (t=25, rt_w=2.0)': {
            'results': 'outputs/experiments/mnist_convlstm/exp10_t25_rt2/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv',
            'log': 'outputs/experiments/mnist_convlstm/exp10_t25_rt2/training.log',
            'time_steps': 25,
            'rt_loss_weight': 2.0,
            'color': '#2ca02c'
        },
    }
    
    # Collect final results
    final_results = []
    for exp_name, config in experiments.items():
        if os.path.exists(config['results']):
            df = pd.read_csv(config['results'])
            model_acc = (df['pred_label'] == df['true_label']).mean() * 100
            human_acc = df['correct'].mean() * 100
            model_rt = df['rt_pred_seconds'].mean()
            human_rt = df['rt_human_seconds'].mean()
            rt_ratio = model_rt / human_rt
            rt_corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
            
            final_results.append({
                'experiment': exp_name,
                'time_steps': config['time_steps'],
                'rt_loss_weight': config['rt_loss_weight'],
                'model_acc': model_acc,
                'human_acc': human_acc,
                'model_rt': model_rt,
                'human_rt': human_rt,
                'rt_ratio': rt_ratio,
                'rt_corr': rt_corr,
                'color': config['color']
            })
    
    df_final = pd.DataFrame(final_results)
    
    # Figure 1: Final Results Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy Comparison
    ax = axes[0]
    x = np.arange(len(df_final))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_final['model_acc'], width, label='Model ACC', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_final['human_acc'], width, label='Human ACC', color='#1f77b4', alpha=0.8)
    
    ax.axhline(y=70.44, color='gray', linestyle='--', linewidth=1, label='Human baseline (70.44%)')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('A. Model vs Human Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nrt_w={row['rt_loss_weight']}" for _, row in df_final.iterrows()])
    ax.legend(frameon=False)
    ax.set_ylim(0, 100)
    
    # Plot 2: RT Comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, df_final['model_rt'], width, label='Model RT', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, df_final['human_rt'], width, label='Human RT', color='#1f77b4', alpha=0.8)
    
    ax.axhline(y=0.942, color='gray', linestyle='--', linewidth=1, label='Human baseline (0.942s)')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('RT (seconds)')
    ax.set_title('B. Model vs Human RT')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nrt_w={row['rt_loss_weight']}" for _, row in df_final.iterrows()])
    ax.legend(frameon=False)
    
    # Plot 3: RT Ratio
    ax = axes[2]
    bars = ax.bar(x, df_final['rt_ratio'], color=df_final['color'], alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Perfect match (1.0x)')
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1, label='Target (1.5x)')
    
    for i, (bar, ratio) in enumerate(zip(bars, df_final['rt_ratio'])):
        ax.annotate(f'{ratio:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('RT Ratio (Model/Human)')
    ax.set_title('C. RT Ratio (lower is better)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nrt_w={row['rt_loss_weight']}" for _, row in df_final.iterrows()])
    ax.legend(frameon=False)
    ax.set_ylim(0, 2.5)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'experiment_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Experiment comparison saved to: {save_path}")
    
    # Figure 2: Training Progress (Epoch vs ACC/RT)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot training curves for each experiment
    for exp_name, config in experiments.items():
        epoch_df = extract_epoch_metrics_from_log(config['log'])
        if epoch_df is not None:
            # Accuracy over epochs
            axes[0].plot(epoch_df['epoch'], epoch_df['accuracy'] * 100, 
                        label=exp_name, color=config['color'], linewidth=1.5, alpha=0.8)
            
            # RT correlation over epochs
            axes[1].plot(epoch_df['epoch'], epoch_df['rt_correlation'], 
                        label=exp_name, color=config['color'], linewidth=1.5, alpha=0.8)
    
    # Add human baseline
    axes[0].axhline(y=70.44, color='gray', linestyle='--', linewidth=1, label='Human baseline (70.44%)')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    axes[0].set_xlabel('Training Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('A. Accuracy Over Training')
    axes[0].legend(frameon=False, loc='lower right')
    axes[0].set_xlim(1, 100)
    axes[0].set_ylim(0, 100)
    
    axes[1].set_xlabel('Training Epoch')
    axes[1].set_ylabel('RT Correlation')
    axes[1].set_title('B. RT Correlation Over Training')
    axes[1].legend(frameon=False, loc='upper right')
    axes[1].set_xlim(1, 100)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_progress.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Training progress saved to: {save_path}")
    
    # Figure 3: ACC vs RT Tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in df_final.iterrows():
        ax.scatter(row['rt_ratio'], row['model_acc'], s=200, c=row['color'], 
                   label=row['experiment'], alpha=0.8, edgecolors='white', linewidth=2)
        ax.annotate(f"t={row['time_steps']}", 
                    (row['rt_ratio'], row['model_acc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, label='Perfect RT match')
    ax.axhline(y=70.44, color='gray', linestyle=':', linewidth=1, label='Human accuracy')
    
    ax.set_xlabel('RT Ratio (Model/Human)')
    ax.set_ylabel('Model Accuracy (%)')
    ax.set_title('Speed-Accuracy Tradeoff Across Experiments')
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlim(1.0, 2.5)
    ax.set_ylim(60, 85)
    
    # Add arrow showing improvement direction
    ax.annotate('', xy=(1.2, 78), xytext=(2.0, 75),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.6, 76.5, 'Better', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'acc_rt_tradeoff.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"ACC-RT tradeoff saved to: {save_path}")
    
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)

if __name__ == '__main__':
    main()
