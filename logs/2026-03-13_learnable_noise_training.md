# ANN-EAM-Noise Project Log

## 2026-03-13 Training with Learnable Noise Parameters

### Summary

Today's work focused on training the ConvLSTM model with learnable noise parameters and 100 epochs. The model was trained to learn human responses (including errors) rather than just correct labels.

### 1. File Structure Reorganization

#### Problem Identified
- Data preprocessing file was named `01_preprocess_mnist_behavioral.py` (starting with a number)
- Python cannot import modules starting with numbers
- Import paths were broken after folder structure changes

#### Solution Applied
- Renamed `src/data/01_preprocess_mnist_behavioral.py` → `src/data/preprocess_mnist_behavioral.py`
- Updated `src/data/__init__.py` with correct import
- Fixed import statements in training scripts

### 2. Training Configuration

**File**: `src/experiments/mnist_convlstm/02_train_model.py`

**Training Parameters**:
| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Time Steps | 20 |
| RT Supervision | Enabled |
| Learnable Noise | Yes |
| Target | Human Response (including errors) |

**Model Architecture**:
- ConvLSTM with 16 filters, kernel size 3
- Learnable noise parameters: `noise_std`, `mask_p`
- Differentiable decision time computation

### 3. Training Results

**Final Performance**:
| Metric | Value |
|--------|-------|
| Accuracy (vs correct label) | 78.44% |
| Accuracy (vs human response) | 63.66% |
| RT Correlation | 0.1105 |
| Learned Threshold | 0.0149 |
| Learned noise_std | 0.0004 |
| Learned mask_p | 0.3000 |

**RT Statistics (Normalized)**:
| Condition | Mean | Std | N |
|-----------|------|-----|---|
| Correct trials | 0.5604 | 0.1224 | 8,610 |
| Incorrect trials | 0.5702 | 0.1164 | 3,614 |

### 4. Critical Issue Identified: RT Distribution Mismatch

**Problem**: After making noise parameters learnable, the model's RT distribution does NOT match human RT distribution.

**Observations**:
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
   - This suggests the model learned to minimize noise, not use it meaningfully

**Root Cause Analysis**:
- The model may be optimizing for accuracy at the expense of RT distribution matching
- The RT loss (MSE) may not be sufficient to enforce RT distribution shape
- The noise parameters might need different initialization or constraints
- The threshold learning dynamics may need adjustment

**Potential Solutions to Explore**:
1. **Distribution Matching Loss**: Add KL divergence or Wasserstein distance between model and human RT distributions
2. **Skewness Penalty**: Add a loss term to encourage right-skewed RT distribution
3. **Noise Parameter Constraints**: Different parameterization or regularization
4. **Multi-task Learning Balance**: Adjust weight between accuracy loss and RT loss
5. **Threshold Initialization**: Start with higher threshold to encourage longer RTs
6. **Architecture Changes**: Consider different evidence accumulation mechanisms

### 5. Visualization Generated

**Script**: `src/experiments/mnist_convlstm/03_visualize_results_apa.py`

**Generated Figures** (APA-compliant, saved in `figures_apa/`):
1. `fig1_training_curves.png` - Training loss, accuracy, and RT correlation over iterations
2. `fig2_rt_distribution.png` - Model vs human RT scatter plot and distribution by accuracy
3. `fig3_confusion_matrix.png` - Confusion matrix (counts and normalized)
4. `fig4_rt_by_digit.png` - RT correlation for each digit (8 subplots)
5. `fig5_accuracy_by_rt_bin.png` - Accuracy grouped by human RT bins
6. `fig6_model_summary.png` - Comprehensive model performance summary

**APA Style Features**:
- Sans-serif fonts (Arial/Calibri)
- Colorblind-friendly palettes (Okabe-Ito, APA default)
- 300 DPI resolution
- Clean layout (no top/right spines)
- Proper figure numbering and titles

### 6. Bug Fixes

**Issue 1: `plt.show()` Blocking**
- Problem: `plt.show()` blocks execution in nohup background mode
- Solution: Changed all `plt.show()` to `plt.close()` in training script

**Issue 2: MNIST Test Labels Missing**
- Problem: MNIST test labels file not found in `data/mnist-data/`
- Solution: Copied from `data/raw/mnist/MNIST/raw/`

**Issue 3: Data Path**
- Problem: Training script couldn't find behavioral data
- Solution: Updated path to `data/raw/rtnet/behavioral data.csv`

### 7. Output Files

**Model Weights**:
`output_convlstm_v2/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp.pth`

**Results CSV**:
`output_convlstm_v2/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp_results.csv`

**Training Log**:
`training_nohup_100ep.log`

### 8. Next Steps

**Immediate Priorities**:
1. [ ] Implement RT distribution matching loss (KL divergence or histogram matching)
2. [ ] Add skewness constraint to encourage right-skewed RT distribution
3. [ ] Experiment with different noise parameter initializations
4. [ ] Adjust threshold learning rate or initialization
5. [ ] Compare with fixed noise parameters baseline

**Long-term Goals**:
1. [ ] Achieve RT distribution shape similar to human (right-skewed)
2. [ ] Improve RT correlation to > 0.3
3. [ ] Match human accuracy (~70%) while maintaining RT similarity
4. [ ] Analyze error patterns and compare with human errors

### 9. Technical Notes

**Training Duration**: ~8-10 hours (100 epochs)

**Hardware**: Apple Silicon (MPS acceleration)

**Dataset Statistics**:
- Total trials: 61,347
- Training: 49,077 (80%)
- Test: 12,270 (20%)
- Correct trials: 43,096 (70.3%)
- Error trials: 18,251 (29.7%)

---
*Log created: 2026-03-13*
