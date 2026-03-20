# ANN-EAM-Noise Project Log

## 2026-03-19 to 2026-03-20: SAT Model Development & Code Organization

### Summary

This period focused on two major areas:
1. **SAT (Speed-Accuracy Trade-off) Model Development** - Implementing and improving the 4-parameter SAT model
2. **Code Organization** - Consolidating analysis scripts and creating documentation

---

## Part 1: SAT Model Development

### 1.1 Problem Identification

From previous experiments (Exp11, Exp12), we identified a critical limitation:

**Model does NOT capture human-like speed-accuracy trade-off**

| Behavior | Human | Model (Exp11) |
|----------|-------|---------------|
| Error trials | SLOWER (+0.095s) | Similar (-0.007s) |
| Root cause | Difficulty-specific thresholds | Global threshold (4.28) |

### 1.2 SAT Model Architecture

Implemented `RTify_ConvLSTM_SAT` with condition-specific thresholds:

```python
class RTify_ConvLSTM_SAT(RTify_ConvLSTM):
    def __init__(self, ...):
        super().__init__(...)
        del self.threshold  # Remove single threshold
        
        self.threshold_speed = nn.Parameter(torch.tensor(4.28))
        self.threshold_accuracy = nn.Parameter(torch.tensor(4.28))
        
        self.sat_mapping = {
            'speed focus': 0,
            'accuracy focus': 1,
        }
```

### 1.3 Development Phases

#### Phase 1: Basic SAT Model (exp_sat)

**Configuration:**
- Transfer learning from Exp11 weights
- Learnable thresholds for speed/accuracy conditions
- 70 epochs training

**Results:**
| Metric | Value | Issue |
|--------|-------|-------|
| Accuracy | ~19% | ❌ Far below human (70%) |
| RT Correlation | 0.17-0.20 | ✅ Improved |
| Threshold evolution | Unexpected | ❌ Not differentiating |

**Problem:** Model learned human responses (including errors) but accuracy dropped dramatically.

#### Phase 2: 4-Parameter SAT Model

**Configuration:**
- `threshold_speed`: Learnable
- `threshold_accuracy`: Learnable
- `speed_penalty_speed`: Fixed (0.3)
- `speed_penalty_accuracy`: Fixed (0.08)

**Rationale:**
- Fixed penalty coefficients based on human data
- Speed condition: Higher penalty → faster decisions
- Accuracy condition: Lower penalty → more accurate decisions

**Status:** Training initiated

#### Phase 3: Improved Loss Design (exp_sat_improved_loss)

**Problem Analysis:**

1. **Threshold directly affects decision_logits**
   ```python
   decision_time = DiffDecision.apply(s_accumulated - threshold, ...)
   soft_index = exp(-0.5 * (decision_time - t)**2 / sigma**2)
   decision_logits = (logit_trajectory * soft_index).sum()
   ```
   - Threshold too low → Early decision → Poor logits quality
   - Threshold too high → Timeout → Forced decision

2. **Current loss doesn't distinguish conditions**
   ```python
   # Old: Same weights for all conditions
   loss = label_loss + rt_loss_weight * rt_loss + speed_penalty * rt
   ```

**Proposed Solution:**

| Condition | Label Weight | RT Weight | Speed Penalty |
|-----------|--------------|-----------|---------------|
| Speed | 1.0 | 3.0 | +0.2 (penalize long RT) |
| Accuracy | 2.0 | 1.0 | -0.05 (encourage longer RT) |

**Threshold Regularization:**
```python
th_diff = threshold_accuracy - threshold_speed
th_diff_loss = torch.relu(2.0 - th_diff)  # Ensure difference >= 2
```

**Status:** Plan created, test training (10 epochs) initiated

---

## Part 2: Code Organization

### 2.1 Analysis Scripts Consolidation

**Before:** 11 separate analysis scripts
```
src/utils/
├── analyze_exp05.py
├── analyze_exp07.py
├── analyze_exp10.py
├── analyze_exp11.py
├── analyze_exp12.py
├── analyze_exp12_full.py
├── analyze_results.py
├── analyze_difficulty.py
├── difficulty_analysis_exp11.py
├── difficulty_analysis_exp12.py
└── unified_analysis.py
```

**After:** 1 unified script
```
src/utils/
├── unified_analysis.py  ← Enhanced with all features
└── README.md            ← New documentation
```

**Features in unified_analysis.py:**
1. RT distribution comparison by stimulus
2. Correct vs Error RT comparison
3. Accuracy comparison
4. RT statistics comparison
5. RT scatter plot with correlation
6. Difficulty analysis (if data has 'difficulty' column)
7. Speed-accuracy trade-off analysis
8. Statistical report generation
9. Per-stimulus CSV export

**Usage:**
```bash
# By experiment name
python -m src.utils.unified_analysis --exp exp11_t40

# By results path
python -m src.utils.unified_analysis path/to/results.csv

# With custom output
python -m src.utils.unified_analysis results.csv ./analysis
```

### 2.2 Transfer Scripts Consolidation

**Before:** 3 scripts
```
src/experiments/mnist_convlstm/
├── transfer_to_sat.py
├── transfer_to_sat_fixed.py
└── transfer_to_4param_sat.py
```

**After:** 1 script
```
src/experiments/mnist_convlstm/
└── transfer_to_4param_sat.py  ← Primary script
```

### 2.3 Documentation Created

| Location | File | Purpose |
|----------|------|---------|
| `src/experiments/mnist_convlstm/` | README.txt | Experiment directory guide |
| `src/utils/` | README.md | Unified analysis tool usage |

### 2.4 Archive Created

```
src/experiments/mnist_convlstm/archive/
├── 04_train_rt_distribution.py
├── train_two_stage.py
└── train_model_with_noise.py
```

---

## Part 3: Experiment Results Summary

### 3.1 Best Model (Exp11)

| Metric | Model | Human | Notes |
|--------|-------|-------|-------|
| Overall Accuracy | 80.9% | 70.4% | ✅ Exceeds human |
| Easy Accuracy | 91.7% | 81.3% | - |
| Difficult Accuracy | 69.9% | 59.6% | - |
| RT Ratio (Overall) | 1.27x | - | ✅ Best result |
| RT Ratio (Easy) | 1.33x | - | - |
| RT Ratio (Difficult) | 1.21x | - | Better match |
| RT Correlation | 0.029 | - | ✅ Positive |
| Final Threshold | 4.28 | - | Learned from 6.0 |

### 3.2 Experiments Comparison

| Experiment | Time Steps | Epochs | Accuracy | RT Ratio | RT Corr | Status |
|------------|------------|--------|----------|----------|---------|--------|
| Exp07 | 20 | 100 | 77.9% | 2.03x | 0.001 | Done |
| Exp08 | 20 | 70 | 78.8% | 1.66x | -0.007 | Done |
| Exp10 | 25 | 70 | 72.7% | 1.55x | -0.005 | Done |
| **Exp11** | **40** | **70** | **80.9%** | **1.27x** | **0.029** | **Best** |
| Exp12 | 40 | 40 | 69.2% | 1.33x | 0.024 | Done |
| Exp_SAT | 40 | 70 | ~19% | - | 0.17-0.20 | Issues |
| Exp_SAT_Improved | 40 | 10 | - | - | - | Running |

---

## Part 4: Key Insights

### 4.1 Why Global Threshold Fails

**Human Behavior:**
```
Easy trials:    High drift rate + Lower threshold → Fast decision
Difficult trials: Low drift rate + Higher threshold → Slow decision
```

**Model Behavior:**
```
Easy trials:    High drift rate + Same threshold (4.28)
Difficult trials: Low drift rate + Same threshold (4.28)
Result: RT difference is minimal (0.007s)
```

### 4.2 SAT Condition Mapping

Based on human behavioral data:
- Speed focus: RT=0.855s, Acc=69.2%
- Accuracy focus: RT=1.045s, Acc=71.2%
- RT difference: 0.189s (speed condition is faster)

### 4.3 Threshold Learning Dynamics

- Initial: 6.0
- Final: 4.28
- Change: -29%

**Why lowered:** RT loss (weight=2.0) + Speed penalty (0.1) > Classification loss pull

---

## Part 5: Files Created/Modified

### 2026-03-19

**Created:**
- `src/utils/unified_analysis.py` - Consolidated analysis script
- `src/utils/README.md` - Analysis tool documentation
- `src/experiments/mnist_convlstm/README.txt` - Experiment directory guide

**Deleted:**
- 10 redundant analysis scripts
- 2 redundant transfer scripts

**Archived:**
- 3 experimental training scripts

### 2026-03-20

**Created:**
- `src/experiments/mnist_convlstm/train_sat_improved_loss.py` - Improved loss training
- `outputs/experiments/mnist_convlstm/exp_sat_improved_loss/SAT_Improved_Loss_Plan.md` - Plan document

**Updated:**
- `outputs/experiments/mnist_convlstm/logs/model_progress_2026-03-19.md`

---

## Part 6: Current Project Structure

```
ANN-EAM-Nosie/
├── src/
│   ├── data/                          # Data preprocessing
│   ├── experiments/
│   │   └── mnist_convlstm/
│   │       ├── archive/               # Archived scripts
│   │       ├── 02_train_model.py      # Base model
│   │       ├── train_sat_4param.py    # 4-param SAT
│   │       ├── train_sat_improved_loss.py  # Improved loss
│   │       ├── transfer_to_4param_sat.py   # Weight transfer
│   │       └── README.txt             # Documentation
│   ├── models/
│   │   └── convlstm_sat.py            # SAT model definition
│   └── utils/
│       ├── unified_analysis.py        # Unified analysis ★
│       └── README.md                  # Documentation
│
├── outputs/experiments/mnist_convlstm/
│   ├── exp11_t40/                     # Best model
│   ├── exp12_t40_ep40/
│   ├── exp_sat/
│   ├── exp_sat_improved_loss/         # Current work
│   └── logs/
│
└── logs/                              # Project logs
    ├── 2026-03-19_progress_log.md     # This file
    └── free-thinking/                 # Analysis notes
```

---

## Part 7: Next Steps

### Immediate

1. [ ] Monitor improved loss training results
2. [ ] Analyze threshold differentiation
3. [ ] Compare Speed vs Accuracy condition performance

### Short-term

1. [ ] Adjust hyperparameters if needed
2. [ ] Run full training (40-70 epochs) with best configuration
3. [ ] Generate comparison visualizations

### Long-term

1. [ ] Explore stimulus-specific thresholds
2. [ ] Implement adaptive threshold network
3. [ ] Consider reinforcement learning framework
4. [ ] Prepare publication-ready figures

---

## Part 8: Technical Reference

### 8.1 Model Architecture

```
Input: MNIST image [B, 1, 28, 28]
      ↓
ConvLSTM (16 filters, kernel=3)
      ↓
AdaptivePool + FC → logit_trajectory [B, T, 8]
      ↓
Evidence Module (MLP) → s_traj [B, T]
      ↓
Noise (Gaussian + Dropout)
      ↓
Evidence Accumulation → s_accumulated
      ↓
DiffDecision(threshold) → decision_time
      ↓
Output: decision_logits, rt_normalized
```

### 8.2 Loss Functions

**Standard:**
```python
loss = label_loss + rt_loss_weight * rt_loss + speed_penalty * rt_mean
```

**Improved (Proposed):**
```python
loss = (
    w_label_speed * label_loss_speed +
    w_label_acc * label_loss_acc +
    w_rt_speed * rt_loss_speed +
    w_rt_acc * rt_loss_acc +
    speed_penalty_loss +
    th_diff_weight * th_diff_loss
)
```

---

*Log created: 2026-03-20*
*Related: `outputs/experiments/mnist_convlstm/logs/model_progress_2026-03-19.md`*
