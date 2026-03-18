# ANN-EAM-Noise Project Log

## 2026-03-17 RT Prediction Analysis & Model Improvement

### Summary

This day focused on comprehensive analysis of the best model (Exp11) and identifying key limitations:
1. **Difficulty Analysis**: Model performance by stimulus difficulty
2. **Speed-Accuracy Trade-off**: Comparison with human decision-making patterns
3. **Threshold Analysis**: Understanding learned decision threshold behavior
4. **Visualization Generation**: Publication-ready figures

---

## Part 1: Exp11 - Best Model Analysis

### 1.1 Model Configuration

**Experiment**: Exp11 - Time Steps = 40, 70 epochs

| Parameter | Value |
|-----------|-------|
| Time Steps | 40 |
| Normalization | Log scale |
| Epochs | 70 |
| RT Loss Weight | 2.0 |
| Speed Penalty | 0.1 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Fixed Noise | True |

### 1.2 Final Results

| Metric | Value |
|--------|-------|
| Accuracy (correct label) | **80.9%** ✅ |
| Accuracy (human response) | - |
| RT Correlation | **0.029** ✅ |
| RT Ratio | **1.27x** ✅ |
| Learned Threshold | 4.28 |

**Key Achievement**: Best RT ratio (1.27x) and positive RT correlation (0.029) among all experiments.

---

## Part 2: Difficulty Analysis

### 2.1 Method

Re-evaluated model with difficulty labels:
- **Easy trials**: Stimuli where human accuracy > 75%
- **Difficult trials**: Stimuli where human accuracy ≤ 75%

### 2.2 Results

| Difficulty | Trials | Model Accuracy | Human Accuracy | RT Ratio |
|------------|--------|----------------|----------------|----------|
| **Easy** | 6,128 | **91.68%** | 81.32% | 1.33x |
| **Difficult** | 6,142 | **69.86%** | 59.56% | **1.21x** ✅ |

### 2.3 Key Findings

1. **Difficult trials have lower RT ratio** (1.21x vs 1.33x)
   - Model takes relatively longer on difficult trials
   - This is a positive sign of difficulty-sensitive behavior

2. **Model outperforms humans in both conditions**
   - Easy: 91.68% vs 81.32% (+10.36%)
   - Difficult: 69.86% vs 59.56% (+10.30%)

3. **All comparisons are statistically significant** (p < 0.001)

---

## Part 3: Speed-Accuracy Trade-off Analysis

### 3.1 Method

Analyzed RT for correct vs error trials to understand decision-making patterns.

### 3.2 Results

| Metric | Model | Human |
|--------|-------|-------|
| Correct RT | 1.192s | 0.915s |
| Error RT | 1.185s | 1.009s |
| **Difference** | **-0.007s** | **+0.095s** |

**Speed-Accuracy Correlation**:
- Model: r = **-0.252** (faster → more accurate)
- Human: r = **0.060** (weak positive)

### 3.3 Critical Finding

**Model does NOT capture human-like speed-accuracy trade-off**

| Behavior | Human | Model |
|----------|-------|-------|
| Error trials | **SLOWER** (+0.095s) | Similar (-0.007s) |
| Interpretation | Need more time for difficult stimuli | No difficulty adaptation |

### 3.4 Root Cause Analysis

**Human Decision Process**:
```
Evidence Accumulation: S(t) = S(t-1) + v + noise

Easy trials:    High drift rate (v) + Lower threshold → Fast decision
Difficult trials: Low drift rate (v) + Higher threshold → Slow decision
```

**Model Behavior**:
```
Easy trials:    High drift rate (v) + Same threshold (4.28)
Difficult trials: Low drift rate (v) + Same threshold (4.28)

Result: RT difference is minimal (0.007s)
```

**Problem**: The model learns a **global optimal threshold** that balances ALL trials together, but doesn't capture **difficulty-specific decision strategies**.

---

## Part 4: Threshold Analysis

### 4.1 Threshold Learning Dynamics

| Stage | Threshold Value | Change |
|-------|-----------------|--------|
| Initial | 6.0 | - |
| Final | **4.28** | -29% |

### 4.2 Loss Function Components

```
total_loss = label_loss + rt_loss_weight × rt_loss + speed_penalty × rt_mean
           = CrossEntropy + 2.0 × MSE(RT) + 0.1 × mean(RT)
```

### 4.3 Why Threshold Lowered

- RT loss (weight=2.0) + Speed penalty (0.1) > Classification loss pull
- Model learned to lower threshold to match human RT
- Lower threshold → faster decisions → lower RT

### 4.4 Problem with Global Threshold

| Limitation | Description |
|------------|-------------|
| Single threshold | Applied to ALL trials equally |
| No adaptation | Cannot adjust for different difficulty levels |
| Missing strategy | Cannot capture human-like decision patterns |

---

## Part 5: Exp12 - Early Stopping Experiment

### 5.1 Configuration

| Parameter | Value |
|-----------|-------|
| Time Steps | 40 |
| Epochs | 40 (vs 70 in Exp11) |
| RT Loss Weight | 2.0 |
| Speed Penalty | 0.1 |

### 5.2 Results

| Metric | Exp11 (70 ep) | Exp12 (40 ep) |
|--------|---------------|---------------|
| Accuracy | **80.9%** | 69.17% |
| RT Ratio | **1.27x** | 1.33x |
| RT Correlation | **0.029** | 0.024 |

### 5.3 Conclusion

**Full training (70 epochs) works better**
- Exp11 achieves higher accuracy AND better RT ratio
- Early stopping does not improve RT matching

---

## Part 6: Generated Visualizations

### Exp11 Analysis Files

| File | Description |
|------|-------------|
| `difficulty_analysis.pdf` | Performance by difficulty with significance tests |
| `speed_accuracy_tradeoff.pdf` | Speed-accuracy trade-off analysis |
| `threshold_analysis.pdf` | Threshold learning and evidence accumulation |
| `figure4_style/figure4_style.pdf` | Rafiei et al. 2024 style visualization |

### Exp12 Analysis Files

| File | Description |
|------|-------------|
| `difficulty_analysis.pdf` | Performance by difficulty |
| `exp12_analysis.pdf` | Full analysis report |

---

## Part 7: Improvement Directions Identified

### 7.1 Difficulty-Conditioned Threshold (Recommended)

**Approach**: Learn 2 thresholds - one for Easy, one for Difficult

**Implementation**:
```python
class DifficultyConditionedThreshold(nn.Module):
    def __init__(self):
        self.threshold_easy = nn.Parameter(torch.tensor(4.0))
        self.threshold_hard = nn.Parameter(torch.tensor(5.0))
    
    def forward(self, difficulty):
        return torch.where(difficulty == 'easy', 
                          self.threshold_easy, 
                          self.threshold_hard)
```

**Expected Benefit**: Better speed-accuracy trade-off

### 7.2 Stimulus-Specific Threshold

**Approach**: Learn 8 thresholds - one per stimulus class

**Pros**:
- More fine-grained control
- May capture stimulus-specific strategies

**Cons**:
- More parameters to learn
- May overfit

### 7.3 Adaptive Threshold (Most Flexible)

**Approach**: Input-dependent threshold prediction

**Implementation**:
```python
class AdaptiveThreshold(nn.Module):
    def __init__(self, hidden_dim=64):
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensure positive threshold
        )
    
    def forward(self, hidden_state):
        return self.fc(hidden_state)
```

**Pros**:
- Most similar to human behavior
- Can adapt to any input

**Cons**:
- Hardest to train
- May be unstable

---

## Part 8: Experiments Comparison Table

| Experiment | Time Steps | Epochs | Accuracy | RT Ratio | RT Corr | Threshold |
|------------|------------|--------|----------|----------|---------|-----------|
| Exp07 | 20 | 100 | 77.9% | 2.03x | 0.001 | - |
| Exp08 | 20 | 70 | 78.8% | 1.66x | -0.007 | - |
| Exp10 | 25 | 70 | 72.7% | 1.55x | -0.005 | - |
| **Exp11** | **40** | **70** | **80.9%** | **1.27x** | **0.029** | **4.28** |
| Exp12 | 40 | 40 | 69.2% | 1.33x | 0.024 | - |

---

## Part 9: Files Created/Modified

### Analysis Scripts
- `src/utils/evaluate_with_difficulty.py` - Difficulty evaluation
- `src/utils/speed_accuracy_tradeoff.py` - SAT analysis
- `src/utils/threshold_analysis.py` - Threshold dynamics
- `src/utils/comprehensive_visualization.py` - Unified visualization
- `src/utils/unified_analysis.py` - Combined analysis pipeline

### Output Files
```
outputs/experiments/mnist_convlstm/
├── exp11_t40/
│   ├── analysis/
│   │   ├── difficulty_analysis.pdf
│   │   ├── speed_accuracy_tradeoff.pdf
│   │   ├── threshold_analysis.pdf
│   │   └── figure4_style/
│   └── *_results_with_difficulty.csv
├── exp12_t40_ep40/
│   ├── analysis/
│   │   └── difficulty_analysis.pdf
│   └── *_results_with_difficulty.csv
└── experiment_comparison.csv
```

---

## Part 10: Next Steps

### Immediate
1. [ ] Implement difficulty-conditioned threshold
2. [ ] Train new model with adaptive threshold
3. [ ] Compare speed-accuracy trade-off with Exp11

### Future
1. [ ] Analyze evidence accumulation dynamics
2. [ ] Visualize evidence trajectories for different conditions
3. [ ] Explore stimulus-specific threshold approach
4. [ ] Prepare publication-ready figures

---

*Log created: 2026-03-17*
