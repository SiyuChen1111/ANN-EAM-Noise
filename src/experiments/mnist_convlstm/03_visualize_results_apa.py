import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Calibri'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

APA_COLORS = {
    'blue': '#4472C4',
    'orange': '#ED7D31',
    'gray': '#A5A5A5',
    'gold': '#FFC000',
    'light_blue': '#5B9BD5',
    'green': '#70AD47',
    'red': '#C00000',
    'purple': '#7030A0',
}

OKABE_ITO_COLORS = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#999999']

def plot_training_curves_apa(log_path, output_path):
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    epochs = []
    losses = []
    accs_correct = []
    accs_response = []
    rt_corrs = []
    
    for line in lines:
        if 'Epoch' in line and 'loss=' in line:
            try:
                parts = line.split(',')
                for p in parts:
                    if 'loss=' in p:
                        loss = float(p.split('loss=')[1].strip())
                        losses.append(loss)
                    if 'acc_correct=' in p:
                        acc = float(p.split('acc_correct=')[1].strip())
                        accs_correct.append(acc)
                    if 'acc_response=' in p:
                        acc = float(p.split('acc_response=')[1].strip())
                        accs_response.append(acc)
                    if 'corr=' in p:
                        corr = float(p.split('corr=')[1].strip())
                        rt_corrs.append(corr)
            except:
                continue
    
    if len(losses) == 0:
        print("No training data found in log file")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    ax1 = axes[0, 0]
    ax1.plot(losses, color=APA_COLORS['blue'], linewidth=1.5)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('A. Training Loss', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = axes[0, 1]
    ax2.plot(accs_correct, color=APA_COLORS['green'], linewidth=1.5, label='Correct Label')
    ax2.plot(accs_response, color=APA_COLORS['orange'], linewidth=1.5, label='Human Response')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('B. Accuracy', fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.05)
    ax2.legend(frameon=False, loc='lower right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = axes[1, 0]
    ax3.plot(rt_corrs, color=APA_COLORS['purple'], linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('RT Correlation')
    ax3.set_title('C. RT Correlation', fontweight='bold', loc='left')
    ax3.set_ylim(-0.5, 0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax4 = axes[1, 1]
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed_loss = pd.Series(losses).rolling(window=window).mean()
        smoothed_acc = pd.Series(accs_correct).rolling(window=window).mean()
        ax4.plot(smoothed_loss, color=APA_COLORS['blue'], linewidth=1.5, label='Loss (smoothed)')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(smoothed_acc, color=APA_COLORS['green'], linewidth=1.5, label='Accuracy (smoothed)')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Loss', color=APA_COLORS['blue'])
        ax4_twin.set_ylabel('Accuracy', color=APA_COLORS['green'])
        ax4.tick_params(axis='y', labelcolor=APA_COLORS['blue'])
        ax4_twin.tick_params(axis='y', labelcolor=APA_COLORS['green'])
        ax4.set_title('D. Smoothed Metrics', fontweight='bold', loc='left')
        ax4.spines['top'].set_visible(False)
        ax4_twin.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {output_path}")

def plot_rt_distribution_apa(results_path, output_path):
    df = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    
    ax1 = axes[0]
    rt_pred = df['rt_pred_normalized'].values
    rt_human = df['rt_human_normalized'].values
    
    ax1.scatter(rt_human, rt_pred, alpha=0.3, s=8, color=APA_COLORS['blue'], edgecolors='none')
    
    mask = ~np.isnan(rt_human) & ~np.isnan(rt_pred)
    if mask.sum() > 1:
        z = np.polyfit(rt_human[mask], rt_pred[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(rt_human[mask].min(), rt_human[mask].max(), 100)
        ax1.plot(x_line, p(x_line), color=APA_COLORS['red'], linewidth=2, label='Linear fit')
        
        corr, p_val = stats.pearsonr(rt_human[mask], rt_pred[mask])
        ax1.set_title(f'A. Model vs. Human RT\nr = {corr:.3f}, p < .001' if p_val < .001 else f'A. Model vs. Human RT\nr = {corr:.3f}, p = {p_val:.3f}', 
                     fontweight='bold', loc='left')
    
    ax1.set_xlabel('Human RT (normalized)')
    ax1.set_ylabel('Model RT (normalized)')
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    
    ax2 = axes[1]
    correct_rt = df[df['correct'] == True]['rt_pred_normalized'].values
    incorrect_rt = df[df['correct'] == False]['rt_pred_normalized'].values
    
    bins = np.linspace(0, 1, 25)
    ax2.hist(correct_rt, bins=bins, alpha=0.7, label=f'Correct (n={len(correct_rt):,})', 
             color=APA_COLORS['green'], density=True, edgecolor='white', linewidth=0.5)
    if len(incorrect_rt) > 0:
        ax2.hist(incorrect_rt, bins=bins, alpha=0.7, label=f'Incorrect (n={len(incorrect_rt):,})', 
                 color=APA_COLORS['red'], density=True, edgecolor='white', linewidth=0.5)
    
    ax2.set_xlabel('RT (normalized)')
    ax2.set_ylabel('Density')
    ax2.set_title('B. RT Distribution by Accuracy', fontweight='bold', loc='left')
    ax2.legend(frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RT distribution saved to: {output_path}")

def plot_confusion_matrix_apa(results_path, output_path):
    df = pd.read_csv(results_path)
    
    true_labels = df['true_label'].values
    pred_labels = df['pred_label'].values
    
    n_classes = max(true_labels.max(), pred_labels.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title('A. Confusion Matrix (Counts)', fontweight='bold', loc='left')
    
    ax2 = axes[1]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2, 
                vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('B. Confusion Matrix (Normalized)', fontweight='bold', loc='left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def plot_rt_by_digit_apa(results_path, output_path):
    df = pd.read_csv(results_path)
    
    unique_labels = sorted(df['true_label'].unique())
    n_stim = len(unique_labels)
    
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    
    for i, stim in enumerate(unique_labels):
        ax = axes[i]
        mask = df['true_label'] == stim
        stim_rt_pred = df.loc[mask, 'rt_pred_normalized'].values
        stim_rt_human = df.loc[mask, 'rt_human_normalized'].values
        
        ax.scatter(stim_rt_human, stim_rt_pred, alpha=0.3, s=6, color=APA_COLORS['blue'], edgecolors='none')
        
        valid = ~np.isnan(stim_rt_human) & ~np.isnan(stim_rt_pred)
        if valid.sum() > 1:
            corr, p_val = stats.pearsonr(stim_rt_human[valid], stim_rt_pred[valid])
        else:
            corr = 0
        
        ax.set_xlabel('Human RT')
        ax.set_ylabel('Model RT')
        ax.set_title(f'Digit {stim} (n={len(stim_rt_pred):,})\nr = {corr:.3f}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for i in range(n_stim, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RT by digit saved to: {output_path}")

def plot_accuracy_by_rt_bin_apa(results_path, output_path):
    df = pd.read_csv(results_path)
    
    df['rt_bin'] = pd.cut(df['rt_human_normalized'], bins=5, labels=['Very Fast', 'Fast', 'Medium', 'Slow', 'Very Slow'])
    
    accuracy_by_bin = df.groupby('rt_bin').apply(
        lambda x: pd.Series({
            'accuracy': (x['pred_label'] == x['true_label']).mean(),
            'count': len(x),
            'std': (x['pred_label'] == x['true_label']).std()
        })
    ).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bars = ax.bar(range(len(accuracy_by_bin)), accuracy_by_bin['accuracy'], 
                  color=OKABE_ITO_COLORS[:5], edgecolor='black', linewidth=0.8)
    
    ax.errorbar(range(len(accuracy_by_bin)), accuracy_by_bin['accuracy'], 
                yerr=accuracy_by_bin['std'] / np.sqrt(accuracy_by_bin['count']),
                fmt='none', ecolor='black', capsize=4, capthick=1)
    
    ax.set_xticks(range(len(accuracy_by_bin)))
    ax.set_xticklabels(accuracy_by_bin['rt_bin'])
    ax.set_xlabel('Human RT Bin')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy by Human RT Bin', fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for i, (bar, count) in enumerate(zip(bars, accuracy_by_bin['count'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Accuracy by RT bin saved to: {output_path}")

def plot_model_summary_apa(results_path, output_path):
    df = pd.read_csv(results_path)
    
    fig = plt.figure(figsize=(12, 10))
    
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    correct_rt = df[df['correct'] == True]['rt_pred_normalized']
    incorrect_rt = df[df['correct'] == False]['rt_pred_normalized']
    ax1.boxplot([correct_rt, incorrect_rt], labels=['Correct', 'Incorrect'],
                patch_artist=True, boxprops=dict(facecolor=APA_COLORS['light_blue']))
    ax1.set_ylabel('Model RT (normalized)')
    ax1.set_title('A. RT by Correctness', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    accuracy_by_digit = df.groupby('true_label').apply(lambda x: (x['pred_label'] == x['true_label']).mean())
    ax2.bar(accuracy_by_digit.index, accuracy_by_digit.values, color=OKABE_ITO_COLORS[:8], edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('True Digit')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('B. Accuracy by Digit', fontweight='bold', loc='left')
    ax2.set_ylim(0, 1.05)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df['rt_human_normalized'], bins=30, alpha=0.7, label='Human', color=APA_COLORS['orange'], density=True)
    ax3.hist(df['rt_pred_normalized'], bins=30, alpha=0.7, label='Model', color=APA_COLORS['blue'], density=True)
    ax3.set_xlabel('RT (normalized)')
    ax3.set_ylabel('Density')
    ax3.set_title('C. RT Distributions', fontweight='bold', loc='left')
    ax3.legend(frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    ax4 = fig.add_subplot(gs[1, :])
    rt_pred = df['rt_pred_normalized'].values
    rt_human = df['rt_human_normalized'].values
    ax4.scatter(rt_human, rt_pred, alpha=0.2, s=4, color=APA_COLORS['blue'], edgecolors='none')
    
    mask = ~np.isnan(rt_human) & ~np.isnan(rt_pred)
    if mask.sum() > 1:
        z = np.polyfit(rt_human[mask], rt_pred[mask], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax4.plot(x_line, p(x_line), color=APA_COLORS['red'], linewidth=2, label=f'Linear fit: y = {z[0]:.2f}x + {z[1]:.2f}')
        corr, p_val = stats.pearsonr(rt_human[mask], rt_pred[mask])
        ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes, fontsize=12, 
                verticalalignment='top', fontweight='bold')
    
    ax4.set_xlabel('Human RT (normalized)')
    ax4.set_ylabel('Model RT (normalized)')
    ax4.set_title('D. Model vs. Human RT Correlation', fontweight='bold', loc='left')
    ax4.legend(frameon=False, loc='lower right')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.5)
    
    ax5 = fig.add_subplot(gs[2, :])
    
    summary_text = f"""
    Model Performance Summary
    {'─'*60}
    
    Overall Accuracy (vs. correct label):  {(df['pred_label'] == df['true_label']).mean()*100:.2f}%
    Overall Accuracy (vs. human response):  {(df['pred_label'] == df['human_response']).mean()*100:.2f}%
    
    RT Correlation:  {stats.pearsonr(rt_human[mask], rt_pred[mask])[0]:.4f}
    
    RT Statistics (Model):
      Correct trials:   {correct_rt.mean():.4f} ± {correct_rt.std():.4f} (n={len(correct_rt):,})
      Incorrect trials: {incorrect_rt.mean():.4f} ± {incorrect_rt.std():.4f} (n={len(incorrect_rt):,})
    
    RT Statistics (Human):
      Correct trials:   {df[df['correct']==True]['rt_human_normalized'].mean():.4f} ± {df[df['correct']==True]['rt_human_normalized'].std():.4f}
      Incorrect trials: {df[df['correct']==False]['rt_human_normalized'].mean():.4f} ± {df[df['correct']==False]['rt_human_normalized'].std():.4f}
    
    Total trials: {len(df):,}
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes, fontsize=10, 
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax5.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model summary saved to: {output_path}")

def main():
    output_dir = 'output_convlstm_v2'
    log_path = 'training_nohup_100ep.log'
    results_path = os.path.join(output_dir, 'convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_results.csv')
    
    os.makedirs('figures_apa', exist_ok=True)
    
    print("Generating APA-style visualizations...")
    print("="*60)
    
    plot_training_curves_apa(log_path, 'figures_apa/fig1_training_curves.png')
    
    plot_rt_distribution_apa(results_path, 'figures_apa/fig2_rt_distribution.png')
    
    plot_confusion_matrix_apa(results_path, 'figures_apa/fig3_confusion_matrix.png')
    
    plot_rt_by_digit_apa(results_path, 'figures_apa/fig4_rt_by_digit.png')
    
    plot_accuracy_by_rt_bin_apa(results_path, 'figures_apa/fig5_accuracy_by_rt_bin.png')
    
    plot_model_summary_apa(results_path, 'figures_apa/fig6_model_summary.png')
    
    print("\n" + "="*60)
    print("All APA-style figures generated successfully!")
    print("Output directory: figures_apa/")
    print("="*60)

if __name__ == '__main__':
    main()
