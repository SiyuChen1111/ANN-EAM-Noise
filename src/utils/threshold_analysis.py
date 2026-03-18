"""
Visualize the concept of adaptive threshold and evidence accumulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def main():
    output_dir = '/Users/siyu/Documents/GitHub/ANN-EAM-Nosie/outputs/experiments/mnist_convlstm/exp11_t40/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    time_steps = 40
    
    # Plot 1: Global Threshold (Current Model)
    ax = axes[0, 0]
    
    drift_easy = 0.15
    evidence_easy = np.cumsum(np.random.normal(drift_easy, 0.1, time_steps))
    
    drift_diff = 0.08
    evidence_diff = np.cumsum(np.random.normal(drift_diff, 0.15, time_steps))
    
    threshold_global = 4.28
    
    ax.plot(range(time_steps), evidence_easy, 'g-', linewidth=2, label='Easy (high drift)')
    ax.plot(range(time_steps), evidence_diff, 'r-', linewidth=2, label='Difficult (low drift)')
    ax.axhline(y=threshold_global, color='blue', linestyle='--', linewidth=2, label='Global Threshold')
    
    dt_easy = np.argmax(evidence_easy >= threshold_global) if np.any(evidence_easy >= threshold_global) else time_steps
    dt_diff = np.argmax(evidence_diff >= threshold_global) if np.any(evidence_diff >= threshold_global) else time_steps
    
    ax.axvline(x=dt_easy, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=dt_diff, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Accumulated Evidence')
    ax.set_title(f'a. Global Threshold (Current Model)\nEasy RT={dt_easy}, Difficult RT={dt_diff}')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, time_steps)
    
    # Plot 2: Adaptive Threshold
    ax = axes[0, 1]
    
    threshold_easy = 3.0
    threshold_diff = 5.5
    
    ax.plot(range(time_steps), evidence_easy, 'g-', linewidth=2, label='Easy (high drift)')
    ax.plot(range(time_steps), evidence_diff, 'r-', linewidth=2, label='Difficult (low drift)')
    ax.axhline(y=threshold_easy, color='green', linestyle='--', linewidth=2, label='Easy Threshold (low)')
    ax.axhline(y=threshold_diff, color='red', linestyle='--', linewidth=2, label='Difficult Threshold (high)')
    
    dt_easy_adapt = np.argmax(evidence_easy >= threshold_easy) if np.any(evidence_easy >= threshold_easy) else time_steps
    dt_diff_adapt = np.argmax(evidence_diff >= threshold_diff) if np.any(evidence_diff >= threshold_diff) else time_steps
    
    ax.axvline(x=dt_easy_adapt, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=dt_diff_adapt, color='red', linestyle=':', alpha=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Accumulated Evidence')
    ax.set_title(f'b. Adaptive Threshold (Human-like)\nEasy RT={dt_easy_adapt}, Difficult RT={dt_diff_adapt}')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, time_steps)
    
    # Plot 3: Speed-Accuracy Trade-off
    ax = axes[0, 2]
    
    thresholds = np.linspace(2, 8, 50)
    rts_easy = []
    rts_diff = []
    acc_easy = []
    acc_diff = []
    
    for thresh in thresholds:
        n_sim = 100
        
        rt_easy = []
        correct_easy = []
        for _ in range(n_sim):
            ev = np.cumsum(np.random.normal(0.15, 0.1, time_steps))
            dt = np.argmax(ev >= thresh) if np.any(ev >= thresh) else time_steps
            rt_easy.append(dt)
            correct_easy.append(1 if np.random.random() < 0.7 + 0.03 * thresh else 0)
        rts_easy.append(np.mean(rt_easy))
        acc_easy.append(np.mean(correct_easy))
        
        rt_diff = []
        correct_diff = []
        for _ in range(n_sim):
            ev = np.cumsum(np.random.normal(0.08, 0.15, time_steps))
            dt = np.argmax(ev >= thresh) if np.any(ev >= thresh) else time_steps
            rt_diff.append(dt)
            correct_diff.append(1 if np.random.random() < 0.5 + 0.03 * thresh else 0)
        rts_diff.append(np.mean(rt_diff))
        acc_diff.append(np.mean(correct_diff))
    
    ax.plot(rts_easy, acc_easy, 'g-', linewidth=2, label='Easy')
    ax.plot(rts_diff, acc_diff, 'r-', linewidth=2, label='Difficult')
    ax.scatter([rts_easy[20]], [acc_easy[20]], color='green', s=100, zorder=5, marker='o')
    ax.scatter([rts_diff[30]], [acc_diff[30]], color='red', s=100, zorder=5, marker='o')
    
    ax.set_xlabel('RT (time steps)')
    ax.set_ylabel('Accuracy')
    ax.set_title('c. Speed-Accuracy Trade-off')
    ax.legend(frameon=False)
    
    # Plot 4: RT Comparison
    ax = axes[1, 0]
    
    x = np.arange(4)
    width = 0.6
    
    model_easy_rt = 1.18
    model_diff_rt = 1.20
    human_easy_rt = 0.89
    human_diff_rt = 0.99
    
    bars = ax.bar(x, [model_easy_rt, model_diff_rt, human_easy_rt, human_diff_rt], 
                  color=['#2ca02c', '#d62728', '#2ca02c', '#d62728'], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Model\nEasy', 'Model\nDifficult', 'Human\nEasy', 'Human\nDifficult'])
    ax.set_ylabel('RT (seconds)')
    ax.set_title('d. RT Comparison: Model vs Human')
    
    ax.annotate('', xy=(0, model_easy_rt), xytext=(1, model_diff_rt),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(0.5, 1.25, 'Δ=0.02s\n(n.s.)', ha='center', fontsize=9)
    
    ax.annotate('', xy=(2, human_easy_rt), xytext=(3, human_diff_rt),
                arrowprops=dict(arrowstyle='<->', color='gray'))
    ax.text(2.5, 1.05, 'Δ=0.10s\n(***)', ha='center', fontsize=9)
    
    # Plot 5: Threshold Learning
    ax = axes[1, 1]
    
    epochs = np.arange(1, 71)
    threshold_evolution = 6.0 - 1.72 * (1 - np.exp(-epochs/20))
    
    ax.plot(epochs, threshold_evolution, 'b-', linewidth=2)
    ax.axhline(y=6.0, color='gray', linestyle='--', label='Initial (6.0)')
    ax.axhline(y=4.28, color='red', linestyle='--', label='Final (4.28)')
    
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Threshold Value')
    ax.set_title('e. Threshold Learning Process')
    ax.legend(frameon=False)
    
    # Plot 6: Summary
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = """
Evidence Accumulation Model Analysis
====================================

CURRENT MODEL (Global Threshold):
- Single threshold: 4.28
- All trials use same threshold
- RT difference: 0.02s (not significant)

HUMAN BEHAVIOR (Adaptive Threshold):
- Easy: Low threshold -> Fast decision
- Difficult: High threshold -> Slow decision
- RT difference: 0.10s (significant)

KEY INSIGHT:
The model learns a global optimal threshold
that balances ALL trials together, but
doesn't capture difficulty-specific
decision strategies.

IMPROVEMENT DIRECTIONS:
1. Stimulus-specific threshold (8 thresholds)
2. Difficulty-conditioned threshold (2 thresholds)
3. Adaptive threshold (input-dependent)
"""
    
    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'threshold_analysis.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', format='png')
    plt.close()
    
    print(f"Threshold analysis saved to: {save_path}")

if __name__ == '__main__':
    main()
