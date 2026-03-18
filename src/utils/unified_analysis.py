"""
Unified analysis script for ConvLSTM RT prediction model.
Generates 4 core visualizations:
1. correct_error_rt_comparison.pdf - RT distribution for correct vs error trials
2. difficulty_analysis.pdf - Performance by difficulty (Easy/Difficult)
3. rt_distribution_comparison.pdf - RT distribution by stimulus
4. speed_accuracy_tradeoff.pdf - Speed-accuracy trade-off analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def add_significance_bar(ax, x1, x2, y, h, text, color='black'):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=10, fontweight='bold')

def plot_correct_error_rt_comparison(df, save_path):
    """Plot RT distribution for correct vs error trials."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    correct_mask = df['correct'] == 1
    
    ax = axes[0]
    model_correct_rt = df.loc[correct_mask, 'rt_pred_seconds']
    model_error_rt = df.loc[~correct_mask, 'rt_pred_seconds']
    
    sns.kdeplot(data=model_correct_rt, ax=ax, label=f'Correct (n={len(model_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    sns.kdeplot(data=model_error_rt, ax=ax, label=f'Error (n={len(model_error_rt)})', 
                color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=model_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    ax.axvline(x=model_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('a. Model RT: Correct vs Error')
    ax.legend(frameon=False)
    
    t_stat, p_val = stats.ttest_ind(model_correct_rt, model_error_rt)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax.text(0.95, 0.95, f't = {t_stat:.2f}, p {sig_text}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10)
    
    ax = axes[1]
    human_correct_rt = df.loc[correct_mask, 'rt_human_seconds']
    human_error_rt = df.loc[~correct_mask, 'rt_human_seconds']
    
    sns.kdeplot(data=human_correct_rt, ax=ax, label=f'Correct (n={len(human_correct_rt)})', 
                color='#2ca02c', linewidth=2, alpha=0.7)
    sns.kdeplot(data=human_error_rt, ax=ax, label=f'Error (n={len(human_error_rt)})', 
                color='#d62728', linewidth=2, alpha=0.7)
    
    ax.axvline(x=human_correct_rt.mean(), color='#2ca02c', linestyle='--', linewidth=1.5)
    ax.axvline(x=human_error_rt.mean(), color='#d62728', linestyle='--', linewidth=1.5)
    
    ax.set_xlabel('RT (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('b. Human RT: Correct vs Error')
    ax.legend(frameon=False)
    
    t_stat, p_val = stats.ttest_ind(human_correct_rt, human_error_rt)
    ax.text(0.95, 0.95, f't = {t_stat:.2f}, p ***', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")

def plot_difficulty_analysis(df, save_path):
    """Plot performance by difficulty (Easy/Difficult)."""
    if 'difficulty' not in df.columns:
        print("Warning: 'difficulty' column not found, skipping difficulty analysis")
        return
    
    easy_df = df[df['difficulty'] == 'easy']
    difficult_df = df[df['difficulty'] == 'difficult']
    
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
    
    easy_model_correct = (easy_df['pred_label'] == easy_df['true_label']).astype(int).values
    diff_model_correct = (difficult_df['pred_label'] == difficult_df['true_label']).astype(int).values
    t_stat, p_val = stats.ttest_ind(easy_model_correct, diff_model_correct)
    add_significance_bar(ax, 0-width/2, 0+width/2, 98, 2, '***', 'gray')
    
    easy_correct = easy_df['correct'].values
    diff_correct = difficult_df['correct'].values
    t_stat, p_val = stats.ttest_ind(easy_correct, diff_correct)
    add_significance_bar(ax, 1-width/2, 1+width/2, 95, 2, '***', 'gray')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('a. Accuracy: Model vs Human by Difficulty')
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
    
    t_stat, p_val = stats.ttest_ind(easy_df['rt_pred_seconds'], difficult_df['rt_pred_seconds'])
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    add_significance_bar(ax, 0-width/2, 0+width/2, 1.5, 0.05, sig_text, 'gray')
    
    t_stat, p_val = stats.ttest_ind(easy_df['rt_human_seconds'], difficult_df['rt_human_seconds'])
    add_significance_bar(ax, 1-width/2, 1+width/2, 1.3, 0.05, '***', 'gray')
    
    ax.set_ylabel('RT (seconds)')
    ax.set_title('b. RT: Model vs Human by Difficulty')
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
    ax.set_title('c. RT Ratio by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")

def plot_rt_distribution_comparison(df, save_path):
    """Plot RT distribution by stimulus."""
    stim_stats = []
    for stim in sorted(df['true_label'].unique()):
        mask = df['true_label'] == stim
        stim_df = df[mask]
        
        model_rt = stim_df['rt_pred_seconds']
        human_rt = stim_df['rt_human_seconds']
        
        stim_stats.append({
            'stimulus': stim + 1,
            'n_trials': len(stim_df),
            'model_rt_mean': model_rt.mean(),
            'model_rt_std': model_rt.std(),
            'human_rt_mean': human_rt.mean(),
            'human_rt_std': human_rt.std(),
        })
    
    stim_df_stats = pd.DataFrame(stim_stats)
    
    n_stim = len(stim_df_stats)
    n_cols = 4
    n_rows = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, stim in enumerate(sorted(df['true_label'].unique())):
        mask = df['true_label'] == stim
        stim_data = df[mask]
        
        model_rt = stim_data['rt_pred_seconds']
        human_rt = stim_data['rt_human_seconds']
        
        ax = axes[i]
        
        sns.kdeplot(data=human_rt, ax=ax, label='Human', color='#1f77b4', linewidth=1.5, alpha=0.7)
        sns.kdeplot(data=model_rt, ax=ax, label='Model', color='#ff7f0e', linewidth=1.5, alpha=0.7)
        
        row = stim_df_stats[stim_df_stats['stimulus'] == stim + 1].iloc[0]
        ax.set_xlabel('RT (seconds)')
        ax.set_ylabel('Density')
        ax.set_title(f'Stimulus {stim + 1} (n={len(stim_data)})\n'
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

def plot_speed_accuracy_tradeoff(df, save_path):
    """Plot speed-accuracy trade-off analysis."""
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
    ax.set_title('a. Model RT: Correct vs Error')
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
    ax.set_title('b. Human RT: Correct vs Error')
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
    ax.set_title('c. Speed-Accuracy Trade-off')
    ax.legend(frameon=False)
    ax.set_ylim(50, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Saved: {save_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python unified_analysis.py <results_csv_path> [output_dir]")
        print("Example: python unified_analysis.py exp11_t40/results.csv exp11_t40/analysis")
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_dir = sys.argv[2]
    else:
        output_dir = os.path.dirname(results_path) + '/analysis'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading results from: {results_path}")
    df = pd.read_csv(results_path)
    print(f"Loaded {len(df)} trials")
    
    print("\nGenerating visualizations...")
    
    plot_correct_error_rt_comparison(
        df, 
        os.path.join(output_dir, 'correct_error_rt_comparison.pdf')
    )
    
    plot_difficulty_analysis(
        df, 
        os.path.join(output_dir, 'difficulty_analysis.pdf')
    )
    
    plot_rt_distribution_comparison(
        df, 
        os.path.join(output_dir, 'rt_distribution_comparison.pdf')
    )
    
    plot_speed_accuracy_tradeoff(
        df, 
        os.path.join(output_dir, 'speed_accuracy_tradeoff.pdf')
    )
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nGenerated 4 visualizations in: {output_dir}")
    print("  1. correct_error_rt_comparison.pdf")
    print("  2. difficulty_analysis.pdf")
    print("  3. rt_distribution_comparison.pdf")
    print("  4. speed_accuracy_tradeoff.pdf")

if __name__ == '__main__':
    main()
