"""
Unified Analysis Script for ConvLSTM RT Prediction Model

This script consolidates all experiment analysis functionality into a single tool.
It can analyze any experiment by providing the results CSV path.

Features:
1. RT distribution comparison (Model vs Human) for each stimulus
2. Correct vs Error RT distribution comparison
3. Accuracy comparison per stimulus
4. RT statistics comparison
5. RT scatter plot with correlation
6. Difficulty analysis (if 'difficulty' column exists)
7. Speed-accuracy trade-off analysis
8. Statistical report generation
9. Per-stimulus statistics CSV export

Usage:
    python unified_analysis.py <results_csv_path> [output_dir]
    python unified_analysis.py --exp <experiment_name>

Examples:
    python unified_analysis.py outputs/experiments/mnist_convlstm/exp11_t40/results.csv
    python unified_analysis.py --exp exp11_t40
    python unified_analysis.py results.csv ./analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
import argparse

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


def add_significance_bar(ax, x1, x2, y, h, text, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=10, fontweight='bold')


def compute_statistics(df):
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
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_correct_error_rt_comparison(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    correct_mask = df['correct'] == 1
    
    ax = axes[0]
    model_correct_rt = df.loc[correct_mask, 'rt_pred_seconds']
    model_error_rt = df.loc[~correct_mask, 'rt_pred_seconds']
    
    sns.kdeplot(data=model_correct_rt, ax=ax, label=f'Correct (n={len(model_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    if len(model_error_rt) > 0:
        sns.kdeplot(data=model_error_rt, ax=ax, label=f'Error (n={len(model_error_rt)})', 
                    color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=model_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    if len(model_error_rt) > 0:
        ax.axvline(x=model_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('A. Model RT: Correct vs Error')
    ax.legend(frameon=False)
    
    if len(model_error_rt) > 0:
        t_stat, p_val = stats.ttest_ind(model_correct_rt, model_error_rt)
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        ax.text(0.95, 0.95, f't = {t_stat:.2f}, p {sig_text}', transform=ax.transAxes, 
                ha='right', va='top', fontsize=10)
    
    ax = axes[1]
    human_correct_rt = df.loc[correct_mask, 'rt_human_seconds']
    human_error_rt = df.loc[~correct_mask, 'rt_human_seconds']
    
    sns.kdeplot(data=human_correct_rt, ax=ax, label=f'Correct (n={len(human_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    if len(human_error_rt) > 0:
        sns.kdeplot(data=human_error_rt, ax=ax, label=f'Error (n={len(human_error_rt)})', 
                    color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=human_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    if len(human_error_rt) > 0:
        ax.axvline(x=human_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('B. Human RT: Correct vs Error')
    ax.legend(frameon=False)
    
    if len(human_error_rt) > 0:
        t_stat, p_val = stats.ttest_ind(human_correct_rt, human_error_rt)
        ax.text(0.95, 0.95, f't = {t_stat:.2f}, p ***', transform=ax.transAxes, 
                ha='right', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_accuracy_comparison(stim_stats, save_path):
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
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rt_stats_comparison(stim_stats, save_path):
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
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_rt_scatter(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.scatter(df['rt_human_seconds'], df['rt_pred_seconds'], alpha=0.3, s=10, c='#1f77b4')
    max_rt = max(df['rt_human_seconds'].max(), df['rt_pred_seconds'].max())
    ax.plot([0, max_rt], [0, max_rt], 'r--', linewidth=1.5, label='Perfect match')
    ax.set_xlabel('Human RT (seconds)')
    ax.set_ylabel('Model RT (seconds)')
    corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
    ax.set_title(f'A. RT Correlation (r={corr:.3f})')
    ax.legend(frameon=False)
    
    ax = axes[1]
    correct_mask = df['correct'] == 1
    ax.scatter(df.loc[correct_mask, 'rt_human_seconds'], 
               df.loc[correct_mask, 'rt_pred_seconds'], 
               alpha=0.3, s=10, c='#2ca02c', label='Correct')
    ax.scatter(df.loc[~correct_mask, 'rt_human_seconds'], 
               df.loc[~correct_mask, 'rt_pred_seconds'], 
               alpha=0.3, s=10, c='#d62728', label='Error')
    ax.plot([0, max_rt], [0, max_rt], 'k--', linewidth=1.5, label='Perfect match')
    ax.set_xlabel('Human RT (seconds)')
    ax.set_ylabel('Model RT (seconds)')
    ax.set_title('B. RT Correlation by Correctness')
    ax.legend(frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_difficulty_analysis(df, save_path):
    if 'difficulty' not in df.columns:
        print("Warning: 'difficulty' column not found, skipping difficulty analysis")
        return
    
    easy_df = df[df['difficulty'] == 'easy']
    difficult_df = df[df['difficulty'] == 'difficult']
    
    if len(easy_df) == 0 or len(difficult_df) == 0:
        print("Warning: No easy/difficult data found, skipping difficulty analysis")
        return
    
    easy_model_acc = (easy_df['pred_label'] == easy_df['true_label']).mean() * 100
    easy_human_acc = easy_df['correct'].mean() * 100
    diff_model_acc = (difficult_df['pred_label'] == difficult_df['true_label']).mean() * 100
    diff_human_acc = difficult_df['correct'].mean() * 100
    
    easy_model_rt = easy_df['rt_pred_seconds'].mean()
    easy_human_rt = easy_df['rt_human_seconds'].mean()
    diff_model_rt = difficult_df['rt_pred_seconds'].mean()
    diff_human_rt = difficult_df['rt_human_seconds'].mean()
    
    easy_rt_ratio = easy_model_rt / easy_human_rt
    diff_rt_ratio = diff_model_rt / diff_human_rt
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [easy_model_acc, diff_model_acc], width, label='Model', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, [easy_human_acc, diff_human_acc], width, label='Human', color='#1f77b4', alpha=0.8)
    
    for bar, val in zip(bars1, [easy_model_acc, diff_model_acc]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, [easy_human_acc, diff_human_acc]):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('A. Accuracy: Model vs Human by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 110)
    
    ax = axes[1]
    bars1 = ax.bar(x - width/2, [easy_model_rt, diff_model_rt], width, label='Model', color='#ff7f0e', alpha=0.8)
    bars2 = ax.bar(x + width/2, [easy_human_rt, diff_human_rt], width, label='Human', color='#1f77b4', alpha=0.8)
    
    for bar, val in zip(bars1, [easy_model_rt, diff_model_rt]):
        ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, [easy_human_rt, diff_human_rt]):
        ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('RT (seconds)')
    ax.set_title('B. RT: Model vs Human by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False)
    
    ax = axes[2]
    bars = ax.bar(x, [easy_rt_ratio, diff_rt_ratio], color=['#2ca02c', '#d62728'], alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Perfect match')
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1, label='Target (1.5x)')
    
    for bar, val in zip(bars, [easy_rt_ratio, diff_rt_ratio]):
        ax.annotate(f'{val:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('RT Ratio (Model/Human)')
    ax.set_title('C. RT Ratio by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def plot_speed_accuracy_tradeoff(df, save_path):
    df['model_correct'] = (df['pred_label'] == df['true_label']).astype(int)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    ax = axes[0]
    model_correct_rt = df[df['model_correct'] == 1]['rt_pred_seconds']
    model_error_rt = df[df['model_correct'] == 0]['rt_pred_seconds']
    
    sns.kdeplot(data=model_correct_rt, ax=ax, label=f'Correct (n={len(model_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    sns.kdeplot(data=model_error_rt, ax=ax, label=f'Error (n={len(model_error_rt)})', 
                color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=model_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    ax.axvline(x=model_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('A. Model RT: Correct vs Error')
    ax.legend(frameon=False)
    
    t_stat, p_val = stats.ttest_ind(model_correct_rt, model_error_rt)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.text(0.95, 0.95, f't = {t_stat:.2f}, p {sig_text}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10)
    
    ax = axes[1]
    human_correct_rt = df[df['correct'] == 1]['rt_human_seconds']
    human_error_rt = df[df['correct'] == 0]['rt_human_seconds']
    
    sns.kdeplot(data=human_correct_rt, ax=ax, label=f'Correct (n={len(human_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    sns.kdeplot(data=human_error_rt, ax=ax, label=f'Error (n={len(human_error_rt)})', 
                color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=human_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    ax.axvline(x=human_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('B. Human RT: Correct vs Error')
    ax.legend(frameon=False)
    
    t_stat, p_val = stats.ttest_ind(human_correct_rt, human_error_rt)
    ax.text(0.95, 0.95, f't = {t_stat:.2f}, p ***', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10)
    
    ax = axes[2]
    df['rt_bin'] = pd.qcut(df['rt_pred_seconds'], q=5, labels=['Fastest', 'Fast', 'Medium', 'Slow', 'Slowest'])
    
    rt_bin_stats = df.groupby('rt_bin').agg({
        'model_correct': 'mean',
        'rt_pred_seconds': 'mean'
    }).reset_index()
    
    ax.plot(rt_bin_stats['rt_pred_seconds'], rt_bin_stats['model_correct'] * 100, 
            'o-', color='#ff7f0e', linewidth=2, markersize=10, label='Model')
    
    human_bin_stats = df.groupby('rt_bin').agg({
        'correct': 'mean',
        'rt_human_seconds': 'mean'
    }).reset_index()
    
    ax.plot(human_bin_stats['rt_human_seconds'], human_bin_stats['correct'] * 100, 
            's--', color='#1f77b4', linewidth=2, markersize=10, label='Human')
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('C. Speed-Accuracy Trade-off')
    ax.legend(frameon=False)
    ax.set_ylim(50, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")


def print_statistical_report(df, stim_stats, overall_stats, experiment_name=""):
    print("\n" + "="*70)
    print(f"STATISTICAL ANALYSIS REPORT{f' - {experiment_name}' if experiment_name else ''}")
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
    parser = argparse.ArgumentParser(description='Unified Analysis for ConvLSTM RT Prediction Model')
    parser.add_argument('results_path', nargs='?', help='Path to results CSV file')
    parser.add_argument('output_dir', nargs='?', help='Output directory for analysis')
    parser.add_argument('--exp', type=str, help='Experiment name (e.g., exp11_t40)')
    parser.add_argument('--base_dir', type=str, 
                        default='outputs/experiments/mnist_convlstm',
                        help='Base directory for experiments')
    
    args = parser.parse_args()
    
    if args.exp:
        exp_dir = os.path.join(args.base_dir, args.exp)
        results_files = [f for f in os.listdir(exp_dir) if f.endswith('_results.csv')]
        if results_files:
            args.results_path = os.path.join(exp_dir, results_files[0])
            args.output_dir = os.path.join(exp_dir, 'analysis')
        else:
            print(f"Error: No results.csv found in {exp_dir}")
            sys.exit(1)
    
    if not args.results_path:
        print("Usage: python unified_analysis.py <results_csv_path> [output_dir]")
        print("       python unified_analysis.py --exp <experiment_name>")
        print("\nExample:")
        print("  python unified_analysis.py outputs/experiments/mnist_convlstm/exp11_t40/results.csv")
        print("  python unified_analysis.py --exp exp11_t40")
        sys.exit(1)
    
    if not args.output_dir:
        args.output_dir = os.path.dirname(args.results_path) + '/analysis'
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    experiment_name = os.path.basename(os.path.dirname(args.results_path))
    
    print(f"Loading results from: {args.results_path}")
    df = pd.read_csv(args.results_path)
    print(f"Loaded {len(df)} trials")
    
    print("\nComputing statistics...")
    overall_stats = compute_statistics(df)
    stim_stats = compute_per_stimulus_stats(df)
    
    print("\nGenerating visualizations...")
    
    plot_rt_distribution_comparison(
        df, stim_stats,
        os.path.join(args.output_dir, 'rt_distribution_comparison.pdf')
    )
    
    plot_correct_error_rt_comparison(
        df,
        os.path.join(args.output_dir, 'correct_error_rt_comparison.pdf')
    )
    
    plot_accuracy_comparison(
        stim_stats,
        os.path.join(args.output_dir, 'accuracy_comparison.pdf')
    )
    
    plot_rt_stats_comparison(
        stim_stats,
        os.path.join(args.output_dir, 'rt_stats_comparison.pdf')
    )
    
    plot_rt_scatter(
        df,
        os.path.join(args.output_dir, 'rt_scatter.pdf')
    )
    
    plot_difficulty_analysis(
        df,
        os.path.join(args.output_dir, 'difficulty_analysis.pdf')
    )
    
    plot_speed_accuracy_tradeoff(
        df,
        os.path.join(args.output_dir, 'speed_accuracy_tradeoff.pdf')
    )
    
    print_statistical_report(df, stim_stats, overall_stats, experiment_name)
    
    stats_path = os.path.join(args.output_dir, 'stimulus_statistics.csv')
    stim_stats.to_csv(stats_path, index=False)
    print(f"\nStimulus statistics saved to: {stats_path}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nGenerated 7 visualizations in: {args.output_dir}")
    print("  1. rt_distribution_comparison.pdf - RT distribution by stimulus")
    print("  2. correct_error_rt_comparison.pdf - Correct vs Error RT")
    print("  3. accuracy_comparison.pdf - Accuracy by stimulus")
    print("  4. rt_stats_comparison.pdf - RT statistics by stimulus")
    print("  5. rt_scatter.pdf - RT correlation scatter plot")
    print("  6. difficulty_analysis.pdf - Performance by difficulty (if available)")
    print("  7. speed_accuracy_tradeoff.pdf - Speed-accuracy trade-off")


if __name__ == '__main__':
    main()
