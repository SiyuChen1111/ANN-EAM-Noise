# RT Prediction Model Progress Log
# Date: 2026-03-16 to 2026-03-17
# ============================================

## Project Overview
**Goal**: Build a ConvLSTM model that predicts both decision and reaction time (RT) for MNIST digit classification, matching human behavioral data.

**Dataset**: MNIST Behavioral Dataset with human RT labels
- Source: `data/raw/rtnet/behavioral data.csv`
- Training samples: ~48,900
- Test samples: ~12,200
- RT range: 0.296s - 4.995s

---

## Model Architecture
- **Type**: ConvLSTM with Evidence Accumulation
- **Input**: 28x28 grayscale MNIST images
- **ConvLSTM**: 16 filters, kernel size 3
- **Output**: 8 classes (digits 1-8), RT prediction
- **Decision Mechanism**: Evidence accumulation with threshold
- **Threshold**: Learnable parameter (nn.Parameter)

---

## Experiment History

### Exp11: Time Steps = 40 (70 epochs) ✅ **BEST MODEL**
**Date**: 2026-03-15
**Configuration**: t=40, log norm, epochs=70, rt_loss_weight=2.0, speed_penalty=0.1
**Results**:
- Accuracy: **80.9%** ✅ (exceeds human 70.44%)
- RT Ratio: **1.27x** ✅ (best result!)
- RT Correlation: **0.029** ✅ (positive correlation!)
- Final Threshold: **4.28** (learned from 6.0)

**Key Findings**:
1. Time steps = 40 provides better decision time resolution
2. Model reaches human accuracy at epoch 40
3. Best RT ratio and correlation among all experiments

---

### Exp12: Time Steps = 40, Epochs = 40
**Date**: 2026-03-16
**Configuration**: t=40, log norm, epochs=40, rt_loss_weight=2.0, speed_penalty=0.1
**Results**:
- Accuracy: **69.17%** (close to human 70.44%)
- RT Ratio: **1.33x**
- RT Correlation: **0.024**

**Hypothesis Test**: Does early stopping improve RT ratio?
- **Conclusion**: Full training (70 epochs) works better
- Exp11 (70 ep): Accuracy 80.9%, RT ratio 1.27x
- Exp12 (40 ep): Accuracy 69.2%, RT ratio 1.33x

---

## New Analysis (2026-03-16 to 2026-03-17)

### 1. Difficulty Analysis (Exp11)

**Method**: Re-evaluated model with difficulty labels (Easy/Difficult)

**Results**:

| Difficulty | Trials | Model Accuracy | Human Accuracy | RT Ratio |
|------------|--------|----------------|----------------|----------|
| **Easy** | 6,128 | **91.68%** | 81.32% | 1.33x |
| **Difficult** | 6,142 | **69.86%** | 59.56% | **1.21x** ✅ |

**Key Findings**:
1. Difficult trials have **lower RT ratio** (1.21x vs 1.33x)
2. Model outperforms humans in both conditions
3. All comparisons are statistically significant (p < 0.001)

---

### 2. Speed-Accuracy Trade-off Analysis (Exp11)

**Method**: Analyzed RT for correct vs error trials

**Results**:

| Metric | Model | Human |
|--------|-------|-------|
| Correct RT | 1.192s | 0.915s |
| Error RT | 1.185s | 1.009s |
| **Difference** | **-0.007s** | **+0.095s** |

**Speed-Accuracy Correlation**:
- Model: r = **-0.252** (faster → more accurate)
- Human: r = **0.060** (weak positive)

**Key Finding**: 
**Model does NOT capture human-like speed-accuracy trade-off**
- Human: Error trials are SLOWER (need more time for difficult stimuli)
- Model: Error trials have similar RT to correct trials

---

### 3. Threshold Analysis

**Global Threshold Learning**:
- Initial: 6.0
- Final: **4.28**
- Change: -29% (lowered to speed up decisions)

**Loss Function Components**:
```
total_loss = label_loss + rt_loss_weight × rt_loss + speed_penalty × rt_mean
           = CrossEntropy + 2.0 × MSE(RT) + 0.1 × mean(RT)
```

**Why threshold lowered**:
- RT loss (weight=2.0) + Speed penalty (0.1) > Classification loss pull
- Model learned to lower threshold to match human RT

**Problem with Global Threshold**:
- Single threshold for ALL trials
- Cannot adapt to different difficulty levels
- Cannot capture human-like decision strategies

---

## Key Insights

### Why Model Doesn't Show Speed-Accuracy Trade-off?

**Human Behavior**:
```
Evidence Accumulation: S(t) = S(t-1) + v + noise

Easy trials:    High drift rate (v) + Low threshold → Fast decision
Difficult trials: Low drift rate (v) + High threshold → Slow decision
```

**Model Behavior**:
```
Easy trials:    High drift rate (v) + Same threshold (4.28)
Difficult trials: Low drift rate (v) + Same threshold (4.28)

Result: RT difference is minimal (0.02s)
```

### Root Cause
The model learns a **global optimal threshold** that balances ALL trials together, but doesn't capture **difficulty-specific decision strategies**.

---

## Improvement Directions

### 1. Difficulty-Conditioned Threshold (Recommended)
- Learn 2 thresholds: one for Easy, one for Difficult
- Implementation: Add difficulty embedding to threshold prediction
- Expected: Better speed-accuracy trade-off

### 2. Stimulus-Specific Threshold
- Learn 8 thresholds: one per stimulus class
- More fine-grained control
- May capture stimulus-specific strategies

### 3. Adaptive Threshold (Most Flexible)
- Input-dependent threshold prediction
- Most similar to human behavior
- Hardest to train

---

## Final Results Comparison

| Experiment | Time Steps | Epochs | Accuracy | RT Ratio | RT Corr | Threshold |
|------------|------------|--------|----------|----------|---------|-----------|
| Exp07 | 20 | 100 | 77.9% | 2.03x | 0.001 | - |
| Exp08 | 20 | 70 | 78.8% | 1.66x | -0.007 | - |
| Exp10 | 25 | 70 | 72.7% | 1.55x | -0.005 | - |
| **Exp11** | **40** | **70** | **80.9%** | **1.27x** | **0.029** | **4.28** |
| Exp12 | 40 | 40 | 69.2% | 1.33x | 0.024 | - |

---

## Generated Visualizations

### Exp11 Analysis
- `difficulty_analysis.pdf` - Performance by difficulty with significance tests
- `speed_accuracy_tradeoff.pdf` - Speed-accuracy trade-off analysis
- `threshold_analysis.pdf` - Threshold learning and evidence accumulation
- `figure4_style/figure4_style.pdf` - Rafiei et al. 2024 style visualization

### Exp12 Analysis
- `difficulty_analysis.pdf` - Performance by difficulty

### Overall Comparison
- `all_experiments_comparison.pdf` - All experiments comparison
- `speed_accuracy_tradeoff.pdf` - Speed-accuracy tradeoff across experiments

---

## Files Structure
```
outputs/experiments/mnist_convlstm/
├── exp07_log_norm_full/           # 100 epochs baseline
├── exp08_balanced/                # 70 epochs, rt_w=2.0
├── exp10_t25_rt2/                 # t=25, 70 epochs
│   └── analysis/
├── exp11_t40/                     # t=40, 70 epochs ✅ BEST
│   ├── analysis/
│   │   ├── difficulty_analysis.pdf
│   │   ├── speed_accuracy_tradeoff.pdf
│   │   ├── threshold_analysis.pdf
│   │   └── figure4_style/
│   └── *_results_with_difficulty.csv
├── exp12_t40_ep40/                # t=40, 40 epochs
│   ├── analysis/
│   │   └── difficulty_analysis.pdf
│   └── *_results_with_difficulty.csv
└── logs/
    ├── model_progress_2026-03-16.md
    ├── model_progress_2026-03-17.md
    ├── all_experiments_comparison.pdf
    └── difficulty_analysis_significance.pdf
```

---

## Next Steps

1. **Implement Difficulty-Conditioned Threshold**
   - Modify model to predict threshold based on difficulty
   - Train and compare with current best model

2. **Explore Stimulus-Specific Threshold**
   - Add stimulus embedding to threshold prediction
   - May capture more nuanced decision strategies

3. **Analyze Evidence Accumulation Dynamics**
   - Visualize evidence trajectories for different conditions
   - Understand how evidence quality affects decisions

---

*Last Updated: 2026-03-17*
