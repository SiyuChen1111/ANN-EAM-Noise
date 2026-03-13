# ConvLSTM with Learnable Noise Parameters (100 Epochs)

## Experiment Overview

This experiment trained a ConvLSTM-based evidence accumulation model with learnable noise parameters to predict both human responses and reaction times on an MNIST classification task.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | ConvLSTM |
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Time Steps | 20 |
| RT Supervision | Enabled |
| Learnable Noise | Yes |
| Target | Human Response (including errors) |
| Initial noise_std | 0.1 |
| Initial mask_p | 0.3 |

## Results

| Metric | Value |
|--------|-------|
| Accuracy (vs correct label) | 78.44% |
| Accuracy (vs human response) | 63.66% |
| RT Correlation | 0.1105 |
| Learned Threshold | 0.0149 |
| Learned noise_std | 0.0004 |
| Learned mask_p | 0.3000 |

## Critical Issue: RT Distribution Mismatch

**Problem**: The model's RT distribution does NOT match human RT distribution.

### Observations

1. **Human RT Distribution**: 
   - Right-skewed (typical of decision-making)
   - Mean: ~0.91s (correct), ~1.01s (error)
   - Shows characteristic positive skewness

2. **Model RT Distribution**:
   - Nearly uniform/normal distribution
   - Mean: ~2.9s (correct), ~2.9s (error)
   - Lacks the right-skewed shape
   - Model RT is ~3x slower than human RT

3. **Learned Noise Parameters**:
   - `noise_std` collapsed to near 0 (0.0004)
   - `mask_p` remained at initial value (0.3)

### Root Cause

- The model optimizes for accuracy at the expense of RT distribution matching
- MSE loss alone is insufficient to enforce RT distribution shape
- Noise parameters need different initialization or constraints

### Potential Solutions

1. Add KL divergence loss for RT distribution matching
2. Add skewness penalty to encourage right-skewed RT
3. Different noise parameter constraints
4. Adjust threshold initialization
5. Multi-task learning balance adjustment

## Files

```
learnable_noise_ep100/
├── figures_apa/                          # APA-style visualization figures
│   ├── fig1_training_curves.png
│   ├── fig2_rt_distribution.png
│   ├── fig3_confusion_matrix.png
│   ├── fig4_rt_by_digit.png
│   ├── fig5_accuracy_by_rt_bin.png
│   └── fig6_model_summary.png
├── convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp.pth              # Model weights
├── convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_results.csv      # Detailed results
├── convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_training_curves.png
├── convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_rt_distribution.png
└── training.log                          # Training log
```

## Related Files

- Training script: `src/experiments/mnist_convlstm/02_train_model.py`
- Visualization script: `src/experiments/mnist_convlstm/03_visualize_results_apa.py`
- Data preprocessing: `src/data/preprocess_mnist_behavioral.py`
- Work log: `logs/2026-03-13_learnable_noise_training.md`

## Next Steps

1. Implement RT distribution matching loss
2. Add skewness constraint
3. Experiment with different noise parameter initializations
4. Compare with fixed noise parameters baseline

---
*Experiment date: 2026-03-13*
