"""
Quick RT Distribution Visualization for Exp07 model.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def main():
    RESULTS_PATH = 'outputs/experiments/mnist_convlstm/exp07_log_norm_full/convlstm_log_t20_rt_sup_log_human_resp_results.csv'
    OUTPUT_DIR = 'outputs/experiments/mnist_convlstm/exp07_log_norm_full/analysis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading results...")
    df = pd.read_csv(RESULTS_PATH)
    print(f"Loaded {len(df)} trials")
    
    model_mean = df['rt_pred_seconds'].mean()
    model_std = df['rt_pred_seconds'].std()
    human_mean = df['rt_human_seconds'].mean()
    human_std = df['rt_human_seconds'].std()
    ratio = model_mean / human_mean
    corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
    
    print(f"\n=== RT Statistics ===")
    print(f"Model RT: {model_mean:.3f} ± {model_std:.3f} seconds")
    print(f"Human RT: {human_mean:.3f} ± {human_std:.3f} seconds")
    print(f"RT Ratio (model/human): {ratio:.2f}x")
    print(f"RT Correlation: {corr:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    sns.kdeplot(data=df['rt_human_seconds'], ax=ax, label='Human', color='#1f77b4', linewidth=2, alpha=0.7)
    sns.kdeplot(data=df['rt_pred_seconds'], ax=ax, label='Model', color='#ff7f0e', linewidth=2, alpha=0.7)
    ax.axvline(x=human_mean, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'Human μ={human_mean:.2f}s')
    ax.axvline(x=model_mean, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'Model μ={model_mean:.2f}s')
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('A. Overall RT Distribution')
    ax.legend(frameon=False, loc='upper right')
    
    ax = axes[0, 1]
    ax.scatter(df['rt_human_seconds'], df['rt_pred_seconds'], alpha=0.3, s=10, c='#1f77b4')
    max_rt = max(df['rt_human_seconds'].max(), df['rt_pred_seconds'].max())
    ax.plot([0, max_rt], [0, max_rt], 'r--', linewidth=1.5, label='Perfect match')
    ax.set_xlabel('Human RT (seconds)')
    ax.set_ylabel('Model RT (seconds)')
    ax.set_title(f'B. RT Correlation (r={corr:.3f})')
    ax.legend(frameon=False)
    
    ax = axes[1, 0]
    correct_mask = df['correct'] == 1
    model_correct_rt = df.loc[correct_mask, 'rt_pred_seconds']
    model_error_rt = df.loc[~correct_mask, 'rt_pred_seconds']
    
    sns.kdeplot(data=model_correct_rt, ax=ax, label=f'Correct (n={len(model_correct_rt)})',
                color='#2ca02c', linewidth=2, alpha=0.7)
    if len(model_error_rt) > 0:
        sns.kdeplot(data=model_error_rt, ax=ax, label=f'Error (n={len(model_error_rt)})',
                    color='#d62728', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('C. Model RT: Correct vs Error Trials')
    ax.legend(frameon=False)
    
    ax = axes[1, 1]
    human_correct_rt = df.loc[correct_mask, 'rt_human_seconds']
    human_error_rt = df.loc[~correct_mask, 'rt_human_seconds']
    
    sns.kdeplot(data=human_correct_rt, ax=ax, label=f'Correct (n={len(human_correct_rt)})',
                color='#2ca02c', linewidth=2, alpha=0.7)
    if len(human_error_rt) > 0:
        sns.kdeplot(data=human_error_rt, ax=ax, label=f'Error (n={len(human_error_rt)})',
                    color='#d62728', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('D. Human RT: Correct vs Error Trials')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'rt_distribution_quick.pdf')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"\nRT distribution visualization saved to: {save_path}")
    print(f"PNG version saved to: {save_path.replace('.pdf', '.png')}")

if __name__ == '__main__':
    main()
