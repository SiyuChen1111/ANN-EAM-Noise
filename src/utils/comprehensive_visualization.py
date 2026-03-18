"""
Comprehensive visualization with statistical significance tests.
Includes:
1. Performance by difficulty (with significance tests)
2. All experiments comparison
3. Figure 4 style visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def add_significance_bar(ax, x1, x2, y, h, text, color='black'):
    """Add significance bar to plot."""
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=color)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom', fontsize=9, fontweight='bold')

def main():
    output_dir = '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Exp11 results with difficulty
    exp11_results = pd.read_csv('/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70_results_with_difficulty.csv')
    
    # Load all experiment results
    experiments = {
        'Exp07 (t=20, ep=100)': {
            'path': 'outputs/experiments/mnist_convlstm/exp07_log_norm_full/convlstm_log_t20_rt_sup_log_human_resp_results.csv',
            'time_steps': 20,
            'epochs': 100,
            'rt_loss_weight': 1.0,
            'color': '#1f77b4'
        },
        'Exp08 (t=20, ep=70)': {
            'path': 'outputs/experiments/mnist_convlstm/exp08_balanced/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv',
            'time_steps': 20,
            'epochs': 70,
            'rt_loss_weight': 2.0,
            'color': '#ff7f0e'
        },
        'Exp10 (t=25, ep=70)': {
            'path': 'outputs/experiments/mnist_convlstm/exp10_t25_rt2/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv',
            'time_steps': 25,
            'epochs': 70,
            'rt_loss_weight': 2.0,
            'color': '#2ca02c'
        },
        'Exp11 (t=40, ep=70)': {
            'path': 'outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv',
            'time_steps': 40,
            'epochs': 70,
            'rt_loss_weight': 2.0,
            'color': '#d62728'
        },
        'Exp12 (t=40, ep=40)': {
            'path': 'outputs/experiments/mnist_convlstm/exp12_t40_ep40/convlstm_balanced_rt2.0_sp0.1_ep40_results.csv',
            'time_steps': 40,
            'epochs': 40,
            'rt_loss_weight': 2.0,
            'color': '#9467bd'
        },
    }
    
    # Compute statistics for each experiment
    exp_stats = []
    for exp_name, config in experiments.items():
        if os.path.exists(config['path']):
            df = pd.read_csv(config['path'])
            model_acc = (df['pred_label'] == df['true_label']).mean() * 100
            human_acc = df['correct'].mean() * 100
            model_rt = df['rt_pred_seconds'].mean()
            human_rt = df['rt_human_seconds'].mean()
            rt_ratio = model_rt / human_rt
            rt_corr = np.corrcoef(df['rt_pred_seconds'], df['rt_human_seconds'])[0, 1]
            
            exp_stats.append({
                'experiment': exp_name,
                'time_steps': config['time_steps'],
                'epochs': config['epochs'],
                'rt_loss_weight': config['rt_loss_weight'],
                'model_acc': model_acc,
                'human_acc': human_acc,
                'model_rt': model_rt,
                'human_rt': human_rt,
                'rt_ratio': rt_ratio,
                'rt_corr': rt_corr,
                'color': config['color']
            })
    
    exp_df = pd.DataFrame(exp_stats)
    
    # Figure 1: Performance by Difficulty (Exp11)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    easy_df = exp11_results[exp11_results['difficulty'] == 'easy']
    difficult_df = exp11_results[exp11_results['difficulty'] == 'difficult']
    
    # Accuracy by difficulty
    ax = axes[0]
    x = np.arange(2)
    width = 0.35
    
    easy_model_acc = (easy_df['pred_label'] == easy_df['true_label']).mean() * 100
    easy_human_acc = easy_df['correct'].mean() * 100
    diff_model_acc = (difficult_df['pred_label'] == difficult_df['true_label']).mean() * 100
    diff_human_acc = difficult_df['correct'].mean() * 100
    
    bars1 = ax.bar(x - width/2, [easy_human_acc, diff_human_acc], width, label='Human', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, [easy_model_acc, diff_model_acc], width, label='Model', color='#ff7f0e', alpha=0.8)
    
    # Statistical tests
    # Human: easy vs difficult
    easy_correct = easy_df['correct'].values
    diff_correct = difficult_df['correct'].values
    t_stat, p_val = stats.ttest_ind(easy_correct, diff_correct)
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    add_significance_bar(ax, -0.175, 0.175, 95, 2, f'p < 0.001', '#1f77b4')
    add_significance_bar(ax, 0.825, 1.175, 95, 2, f'p < 0.001', '#1f77b4')
    
    # Model: easy vs difficult
    easy_model_correct = (easy_df['pred_label'] == easy_df['true_label']).astype(int).values
    diff_model_correct = (difficult_df['pred_label'] == difficult_df['true_label']).astype(int).values
    t_stat, p_val = stats.ttest_ind(easy_model_correct, diff_model_correct)
    add_significance_bar(ax, 0.175, 0.825, 100, 2, f'p < 0.001', '#ff7f0e')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('a. Accuracy by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 110)
    
    # RT by difficulty
    ax = axes[1]
    easy_model_rt = easy_df['rt_pred_seconds'].mean()
    easy_human_rt = easy_df['rt_human_seconds'].mean()
    diff_model_rt = difficult_df['rt_pred_seconds'].mean()
    diff_human_rt = difficult_df['rt_human_seconds'].mean()
    
    bars1 = ax.bar(x - width/2, [easy_human_rt, diff_human_rt], width, label='Human', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x + width/2, [easy_model_rt, diff_model_rt], width, label='Model', color='#ff7f0e', alpha=0.8)
    
    # Statistical tests for RT
    t_stat, p_val = stats.ttest_ind(easy_df['rt_human_seconds'], difficult_df['rt_human_seconds'])
    add_significance_bar(ax, -0.175, 0.175, 1.4, 0.05, f'p < 0.001', '#1f77b4')
    
    t_stat, p_val = stats.ttest_ind(easy_df['rt_pred_seconds'], difficult_df['rt_pred_seconds'])
    sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    add_significance_bar(ax, 0.175, 0.825, 1.5, 0.05, sig_text, '#ff7f0e')
    
    ax.set_ylabel('RT (seconds)')
    ax.set_title('b. RT by Difficulty')
    ax.set_xticks(x)
    ax.set_xticklabels(['Easy', 'Difficult'])
    ax.legend(frameon=False)
    
    # RT ratio by difficulty
    ax = axes[2]
    easy_rt_ratio = easy_model_rt / easy_human_rt
    diff_rt_ratio = diff_model_rt / diff_human_rt
    
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
    ax.legend(frameon=False)
    ax.set_ylim(0, 2)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'difficulty_analysis_significance.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Difficulty analysis saved to: {save_path}")
    
    # Figure 2: All Experiments Comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # Accuracy comparison
    ax = axes[0]
    x = np.arange(len(exp_df))
    width = 0.35
    ax.bar(x - width/2, exp_df['human_acc'], width, label='Human', color='#1f77b4', alpha=0.8)
    bars = ax.bar(x + width/2, exp_df['model_acc'], width, label='Model', color=exp_df['color'], alpha=0.8)
    ax.axhline(y=70.44, color='gray', linestyle='--', linewidth=1, label='Human baseline')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('a. Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nep={row['epochs']}" for _, row in exp_df.iterrows()], fontsize=8)
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, 100)
    
    # RT ratio comparison
    ax = axes[1]
    bars = ax.bar(x, exp_df['rt_ratio'], color=exp_df['color'], alpha=0.8)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Perfect match')
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=1, label='Target (1.5x)')
    
    for bar, val in zip(bars, exp_df['rt_ratio']):
        ax.annotate(f'{val:.2f}x', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('RT Ratio (Model/Human)')
    ax.set_title('b. RT Ratio Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nep={row['epochs']}" for _, row in exp_df.iterrows()], fontsize=8)
    ax.legend(frameon=False)
    ax.set_ylim(0, 2.5)
    
    # RT correlation comparison
    ax = axes[2]
    bars = ax.bar(x, exp_df['rt_corr'], color=exp_df['color'], alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    
    for bar, val in zip(bars, exp_df['rt_corr']):
        ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if val >= 0 else -12), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('RT Correlation')
    ax.set_title('c. RT Correlation Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={row['time_steps']}\nep={row['epochs']}" for _, row in exp_df.iterrows()], fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'all_experiments_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"All experiments comparison saved to: {save_path}")
    
    # Figure 3: Speed-Accuracy Tradeoff
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in exp_df.iterrows():
        ax.scatter(row['rt_ratio'], row['model_acc'], s=200, c=row['color'], 
                   label=row['experiment'], alpha=0.8, edgecolors='white', linewidth=2)
        ax.annotate(f"t={row['time_steps']}", (row['rt_ratio'], row['model_acc']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1, label='Perfect RT match')
    ax.axhline(y=70.44, color='gray', linestyle=':', linewidth=1, label='Human accuracy')
    
    ax.set_xlabel('RT Ratio (Model/Human)')
    ax.set_ylabel('Model Accuracy (%)')
    ax.set_title('Speed-Accuracy Tradeoff Across Experiments')
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    ax.set_xlim(1.0, 2.5)
    ax.set_ylim(60, 90)
    
    # Add improvement arrow
    ax.annotate('', xy=(1.3, 82), xytext=(2.0, 78),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.65, 80, 'Better', fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'speed_accuracy_tradeoff.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    print(f"Speed-accuracy tradeoff saved to: {save_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(exp_df[['experiment', 'time_steps', 'epochs', 'model_acc', 'rt_ratio', 'rt_corr']].to_string(index=False))
    
    print("\n" + "="*70)
    print("DIFFICULTY ANALYSIS (Exp11)")
    print("="*70)
    print(f"\nEasy ({len(easy_df)} trials):")
    print(f"  Model Accuracy: {easy_model_acc:.2f}%")
    print(f"  Human Accuracy: {easy_human_acc:.2f}%")
    print(f"  RT Ratio: {easy_rt_ratio:.2f}x")
    
    print(f"\nDifficult ({len(difficult_df)} trials):")
    print(f"  Model Accuracy: {diff_model_acc:.2f}%")
    print(f"  Human Accuracy: {diff_human_acc:.2f}%")
    print(f"  RT Ratio: {diff_rt_ratio:.2f}x")

if __name__ == '__main__':
    main()
