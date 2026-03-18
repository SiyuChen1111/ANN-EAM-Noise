# RT Prediction Improvement Experiment Log

## Experiment Overview
**Goal**: Reduce the gap between model RT and human RT  
**Baseline**: Model RT ~2.7s vs Human RT ~0.9s (ratio ~3.0x)  
**Date**: 2026-03-15

---

## Experiment Summary

| Experiment | Configuration | Status | RT Ratio | RT Correlation |
|------------|---------------|--------|----------|----------------|
| Baseline | time_steps=20, linear norm | Done | 3.0x | - |
| Exp01 | time_steps=100, 10ep | Done | **1.69x** | -0.016 |
| Exp05 | time_steps=20, log norm | Running | - | - |
| Exp06 | time_steps=50, linear norm | Running | - | - |

---

## Experiment 1: Increase Time Steps (time_steps=100) ✅

### Configuration
- **Time Steps**: 100 (baseline: 20)
- **Epochs**: 10 (quick test)
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **RT Supervision**: Yes
- **Fixed Noise**: True (noise_std=0.5, mask_p=0.4)

### Results (10 epochs)
| Metric | Model | Human | Ratio |
|--------|-------|-------|-------|
| RT Mean | 1.59s | 0.94s | **1.69x** |
| RT Std | 0.29s | 0.43s | - |
| RT Correlation | -0.016 | - | - |

### Conclusion
✅ **SUCCESS** - RT ratio reduced from 3.0x to 1.69x
- Improvement: 43.5% reduction in RT gap
- But training time is too long (~15 hours for 100 epochs)

---

## Experiment 5: Log Normalization (time_steps=20)

### Configuration
- **Time Steps**: 20 (same as baseline)
- **Normalization**: Log scale (instead of linear)
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **RT Supervision**: Yes
- **Fixed Noise**: True

### Rationale
- RT distribution is right-skewed
- Log normalization provides better resolution for low RT values
- Should help model learn RT patterns better

### Status
🔄 **Running** - PID: 6977
- Started: 2026-03-15
- Progress: Epoch 1/10 (~42%)

### Results
(To be filled after training completes)

---

## Experiment 6: Moderate Time Steps (time_steps=50)

### Configuration
- **Time Steps**: 50 (compromise between 20 and 100)
- **Normalization**: Linear (same as baseline)
- **Epochs**: 10
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **RT Supervision**: Yes
- **Fixed Noise**: True

### Rationale
- Faster training than time_steps=100
- Better resolution than time_steps=20
- Output range: [0.02, 1.0] → min RT ~0.39s

### Status
🔄 **Running** - PID: 7661
- Started: 2026-03-15
- Progress: Epoch 1/10 (~13%)

### Results
(To be filled after training completes)

---

## Technical Analysis

### Root Cause of RT Gap
1. **Original Problem**: Model output range [0.05, 1.0] could not match human RT distribution
2. **Discretization**: With time_steps=20, model can only output 20 discrete RT values
3. **Minimum RT**: time_steps=20 → min output 0.05 → 0.53s after denormalization
4. **Human RT min**: 0.296s, so model cannot produce fast enough responses

### Solution Approaches
| Approach | Output Range | Min RT | Training Speed |
|----------|--------------|--------|----------------|
| Baseline (t=20) | [0.05, 1.0] | 0.53s | Fast |
| Log norm (t=20) | [0.05, 1.0] | 0.53s | Fast |
| t=50 | [0.02, 1.0] | 0.39s | Medium |
| t=100 | [0.01, 1.0] | 0.34s | Slow |

---

## Files Generated
- Exp01: `outputs/experiments/mnist_convlstm/exp02_timesteps100_quick/`
- Exp05: `outputs/experiments/mnist_convlstm/exp05_log_norm_quick/`
- Exp06: `outputs/experiments/mnist_convlstm/exp06_timesteps50_quick/`

## How to Check Progress
```bash
# Check all experiments
tail -5 outputs/experiments/mnist_convlstm/exp*/training.log

# Check specific experiment
tail -f outputs/experiments/mnist_convlstm/exp05_log_norm_quick/training.log
```

---

## Next Steps
1. Wait for Exp05 and Exp06 to complete
2. Compare RT ratios across all experiments
3. Select best approach based on:
   - RT ratio (lower is better)
   - RT correlation (higher is better)
   - Training time (faster is better)
4. Run full 100-epoch training with best configuration
