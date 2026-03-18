"""
Analyze ConvLSTM model results and generate APA-formatted visualizations.

This script analyzes the output from the ConvLSTM model training and generates:
1. RT distribution comparison (Model vs Human) for each stimulus
2. Correct vs Error RT distribution comparison
3. Accuracy comparison per stimulus
4. Statistical significance tests
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

sys.path.insert(0, 'scripts')

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
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
})

def load_results(results_path):
    """Load results from CSV file."""
    df = pd.read_csv(results_path)
    return df

def compute_statistics(df):
    """Compute statistics for model vs human comparison."""
    stats_dict = {}
    
    stats_dict['overall_accuracy'] = (df['pred_label'] == df['true_label']).mean()
    stats_dict['human_accuracy'] = df['correct'].mean()
    
    stats_dict['rt_correlation'] = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
    
    correct_mask = df['correct'] == 1
    stats_dict['model_rt_correct'] = df.loc[correct_mask, 'rt_pred_seconds'].mean()
    stats_dict['model_rt_error'] = df.loc[~correct_mask, 'rt_pred_seconds'].mean()
    stats_dict['human_rt_correct'] = df.loc[correct_mask, 'rt_human_seconds'].mean()
    stats_dict['human_rt_error'] = df.loc[~correct_mask, 'rt_human_seconds'].mean()
    
    correct_rt = df.loc[correct_mask, 'rt_pred_seconds'].values
    error_rt = df.loc[~correct_mask, 'rt_pred_seconds'].values
    if len(error_rt) > 0:
        t_stat, p_value = stats.ttest_ind(correct_rt, error_rt)
        stats_dict['model_rt_ttest'] = {'t': t_stat, 'p': p_value}
    
    human_correct_rt = df.loc[correct_mask, 'rt_human_seconds'].values
    human_error_rt = df.loc[~correct_mask, 'rt_human_seconds'].values
    if len(human_error_rt) > 0:
        t_stat, p_value = stats.ttest_ind(human_correct_rt, human_error_rt)
        stats_dict['human_rt_ttest'] = {'t': t_stat, 'p': p_value}
    
    return stats_dict

def compute_per_stimulus_stats(df):
    """Compute statistics for each stimulus."""
    stim_stats = []
    
    for stim in sorted(df['true_label'].unique()):
        mask = df['true_label'] == stim
        stim_df = df[mask]
        
        model_acc = (stim_df['pred_label'] == stim_df['true_label']).mean()
        human_acc = stim_df['correct'].mean()
        
        model_rt = stim_df['rt_pred_seconds']
        human_rt = stim_df['rt_human_seconds']
        
        if len(model_rt) > 1 and len(human_rt) > 1:
            rt_corr = np.corrcoef(model_rt, human_rt)[0, 1]
        else:
            rt_corr = 0
        
        correct_mask = stim_df['correct'] == 1
        model_rt_correct = stim_df.loc[correct_mask, 'rt_pred_seconds']
        model_rt_error = stim_df.loc[~correct_mask, 'rt_pred_seconds']
        human_rt_correct = stim_df.loc[correct_mask, 'rt_human_seconds']
        human_rt_error = stim_df.loc[~correct_mask, 'rt_human_seconds']
        
        model_skew = stats.skew(model_rt) if len(model_rt) > 2 else 0
        human_skew = stats.skew(human_rt) if len(human_rt) > 2 else 0
        
        stim_stats.append({
            'stimulus': stim + 1,
            'n_trials': len(stim_df),
            'model_accuracy': model_acc,
            'human_accuracy': human_acc,
            'rt_correlation': rt_corr,
            'model_rt_mean': model_rt.mean(),
            'model_rt_std': model_rt.std(),
            'human_rt_mean': human_rt.mean(),
            'human_rt_std': human_rt.std(),
            'model_rt_correct_mean': model_rt_correct.mean() if len(model_rt_correct) > 0 else np.nan,
            'model_rt_error_mean': model_rt_error.mean() if len(model_rt_error) > 0 else np.nan,
            'human_rt_correct_mean': human_rt_correct.mean() if len(human_rt_correct) > 0 else np.nan,
            'human_rt_error_mean': human_rt_error.mean() if len(human_rt_error) > 0 else np.nan,
            'model_skewness': model_skew,
            'human_skewness': human_skew,
        })
    
    return pd.DataFrame(stim_stats)

def plot_rt_distribution_comparison(df, stim_stats, save_path):
    """Plot RT distribution comparison for all stimuli in one large figure."""
    n_stim = len(stim_stats)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, stim in enumerate(sorted(df['true_label'].unique())):
        mask = df['true_label'] == stim
        stim_df = df[mask]
        
        model_rt = stim_df['rt_pred_seconds']
        human_rt = stim_df['rt_human_seconds']
        
        ax = axes[i]
        
        sns.kdeplot(data=human_rt, ax=ax, label='Human', color='#1f77b4', linewidth=1.5, alpha=0.7)
        sns.kdeplot(data=model_rt, ax=ax, label='Model', color='#ff7f0e', linewidth=1.5, alpha=0.7)
        
        row = stim_stats[stim_stats['stimulus'] == stim + 1].iloc[0]
        ax.set_xlabel('RT (seconds)')
        ax.set_ylabel('Density')
        ax.set_title(f'Stimulus {stim + 1} (n={len(stim_df)})\n'
                    f'Model: μ={row["model_rt_mean"]:.3f}s, σ={row["model_rt_std"]:.3f}s\n'
                    f'Human: μ={row["human_rt_mean"]:.3f}s, σ={row["human_rt_std"]:.3f}s')
        ax.legend(frameon=False, loc='upper right')
    
    for i in range(n_stim, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"RT distribution comparison saved to: {save_path}")

def plot_correct_error_rt_comparison(df, save_path):
    """Plot correct vs error RT distribution comparison (similar to Fig. 4e)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    correct_mask = df['correct'] == 1
    
    ax = axes[0]
    model_correct_rt = df.loc[correct_mask, 'rt_pred_seconds']
    model_error_rt = df.loc[~correct_mask, 'rt_pred_seconds']
    
    sns.kdeplot(data=model_correct_rt, ax=ax, label=f'Correct (n={len(model_correct_rt)})', 
                color='#2ca02c', linewidth=1.5, alpha=0.7)
    if len(model_error_rt) > 0:
        sns.kdeplot(data=model_error_rt, ax=ax, label=f'Error (n={len(model_error_rt)})', 
                    color='#d62728', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('A. Model RT Distribution')
    ax.legend(frameon=False, loc='upper right')
    
    ax = axes[1]
    human_correct_rt = df.loc[correct_mask, 'rt_human_seconds']
    human_error_rt = df.loc[~correct_mask, 'rt_human_seconds']
    
    sns.kdeplot(data=human_correct_rt, ax=ax, label=f'Correct (n={len(human_correct_rt)})', 
                color='#2ca02c', linewidth=1.5, alpha=0.7)
    if len(human_error_rt) > 0:
        sns.kdeplot(data=human_error_rt, ax=ax, label=f'Error (n={len(human_error_rt)})', 
                    color='#d62728', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('B. Human RT Distribution')
    ax.legend(frameon=False, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Correct/Error RT comparison saved to: {save_path}")

def plot_accuracy_comparison(stim_stats, save_path):
    """Plot accuracy comparison per stimulus."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(stim_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, stim_stats['model_accuracy'] * 100, width, 
                   label='Model', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, stim_stats['human_accuracy'] * 100, width, 
                   label='Human', color='#1f77b4', alpha=0.8)
    
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model vs Human Accuracy by Stimulus')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{int(s)}' for s in stim_stats['stimulus']])
    ax.legend(frameon=False)
    ax.set_ylim(0, 105)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Accuracy comparison saved to: {save_path}")

def plot_rt_stats_comparison(stim_stats, save_path):
    """Plot RT statistics comparison per stimulus."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(len(stim_stats))
    width = 0.35
    
    ax = axes[0]
    ax.bar(x - width/2, stim_stats['model_rt_mean'], width, 
           yerr=stim_stats['model_rt_std'], label='Model', color='#ff7f0e', alpha=0.8,
           capsize=3)
    ax.bar(x + width/2, stim_stats['human_rt_mean'], width, 
           yerr=stim_stats['human_rt_std'], label='Human', color='#1f77b4', alpha=0.8,
           capsize=3)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('RT (seconds)')
    ax.set_title('A. Mean RT by Stimulus')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{int(s)}' for s in stim_stats['stimulus']])
    ax.legend(frameon=False)
    
    ax = axes[1]
    ax.bar(x - width/2, stim_stats['model_skewness'], width, 
           label='Model', color='#ff7f0e', alpha=0.8)
    ax.bar(x + width/2, stim_stats['human_skewness'], width, 
           label='Human', color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Stimulus')
    ax.set_ylabel('Skewness')
    ax.set_title('B. RT Distribution Skewness')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{int(s)}' for s in stim_stats['stimulus']])
    ax.legend(frameon=False)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"RT stats comparison saved to: {save_path}")

def print_statistical_report(df, stim_stats, overall_stats):
    """Print statistical report."""
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS REPORT")
    print("="*70)
    
    print("\n### Overall Statistics ###")
    print(f"Overall Model Accuracy: {overall_stats['overall_accuracy']*100:.2f}%")
    print(f"Overall Human Accuracy: {overall_stats['human_accuracy']*100:.2f}%")
    print(f"RT Correlation (Model vs Human): r = {overall_stats['rt_correlation']:.4f}")
    
    print("\n### RT by Correctness ###")
    print(f"Model RT - Correct: {overall_stats['model_rt_correct']:.4f} s")
    print(f"Model RT - Error: {overall_stats['model_rt_error']:.4f} s")
    if 'model_rt_ttest' in overall_stats:
        t = overall_stats['model_rt_ttest']['t']
        p = overall_stats['model_rt_ttest']['p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"Model RT t-test: t = {t:.3f}, p = {p:.4f} {sig}")
    
    print(f"\nHuman RT - Correct: {overall_stats['human_rt_correct']:.4f} s")
    print(f"Human RT - Error: {overall_stats['human_rt_error']:.4f} s")
    if 'human_rt_ttest' in overall_stats:
        t = overall_stats['human_rt_ttest']['t']
        p = overall_stats['human_rt_ttest']['p']
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"Human RT t-test: t = {t:.3f}, p = {p:.4f} {sig}")
    
    print("\n### Per-Stimulus Statistics ###")
    print("-"*70)
    print(f"{'Stim':<6} {'N':<8} {'Model Acc':<12} {'Human Acc':<12} {'RT Corr':<10} {'Model Skew':<12} {'Human Skew':<12}")
    print("-"*70)
    for _, row in stim_stats.iterrows():
        print(f"{int(row['stimulus']):<6} {int(row['n_trials']):<8} "
              f"{row['model_accuracy']*100:>10.1f}% {row['human_accuracy']*100:>10.1f}% "
              f"{row['rt_correlation']:>8.3f} {row['model_skewness']:>10.3f} {row['human_skewness']:>10.3f}")
    print("-"*70)
    
    print("\n### Statistical Significance Legend ###")
    print("* p < 0.05, ** p < 0.01, *** p < 0.001")

def main():
    RESULTS_PATH = 'outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_results.csv'
    OUTPUT_DIR = 'outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/analysis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading results...")
    df = load_results(RESULTS_PATH)
    print(f"Loaded {len(df)} trials")
    
    print("\nComputing statistics...")
    overall_stats = compute_statistics(df)
    stim_stats = compute_per_stimulus_stats(df)
    
    print("\nGenerating visualizations...")
    
    plot_rt_distribution_comparison(
        df, stim_stats,
        os.path.join(OUTPUT_DIR, 'rt_distribution_comparison.pdf')
    )
    
    plot_correct_error_rt_comparison(
        df,
        os.path.join(OUTPUT_DIR, 'correct_error_rt_comparison.pdf')
    )
    
    plot_accuracy_comparison(
        stim_stats,
        os.path.join(OUTPUT_DIR, 'accuracy_comparison.pdf')
    )
    
    plot_rt_stats_comparison(
        stim_stats,
        os.path.join(OUTPUT_DIR, 'rt_stats_comparison.pdf')
    )
    
    print_statistical_report(df, stim_stats, overall_stats)
    
    stats_path = os.path.join(OUTPUT_DIR, 'stimulus_statistics.csv')
    stim_stats.to_csv(stats_path, index=False)
    print(f"\nStimulus statistics saved to: {stats_path}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)

if __name__ == '__main__':
    main()
