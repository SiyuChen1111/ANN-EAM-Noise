# VAM Lost in Migration - ConvLSTM RT Prediction Plan

## Project Overview

This document provides the experimental plan for training ConvLSTM-based reaction time (RT) prediction models on the VAM Lost in Migration dataset.

**Model Type**: ConvLSTM with Evidence Accumulation  
**Task**: Joint prediction of decision (4-class direction classification) and reaction time  
**Dataset**: VAM Lost in Migration (Flanker task variant)

---

## 1. Dataset Summary

### 1.1 Data Overview

| Property | Value |
|----------|-------|
| **Task Type** | Flanker task (identify center bird direction) |
| **Total Users** | 75 |
| **Total Trials** | 3,229,416 |
| **Avg Trials/User** | 43,059 |
| **Direction Classes** | 4 (L/R/U/D) |
| **Input Channels** | 3 (RGB) |
| **Image Size** | 128×128 (preprocessed) |

### 1.2 RT Statistics

| Metric | Value |
|--------|-------|
| **Min RT** | 6 ms |
| **Max RT** | 43,282 ms |
| **Mean RT** | 697 ms |
| **Median RT** | 666 ms |
| **Std RT** | 216 ms |
| **25th Percentile** | 576 ms |
| **75th Percentile** | 777 ms |
| **99th Percentile** | 1,294 ms |

### 1.3 RT Distribution

| RT Range | Count | Percentage |
|----------|-------|------------|
| 0-250 ms | 9,491 | 0.29% |
| 250-500 ms | 230,885 | 7.15% |
| **500-750 ms** | **2,005,801** | **62.11%** |
| 750-1000 ms | 810,635 | 25.10% |
| 1000-1500 ms | 158,497 | 4.91% |
| 1500+ ms | 14,507 | 0.44% |

### 1.4 Comparison with MNIST

| Metric | VAM | MNIST |
|--------|-----|-------|
| Mean RT | 697 ms | 950 ms |
| Median RT | 666 ms | 833 ms |
| Std RT | 216 ms | 512 ms |
| CV (Std/Mean) | 0.31 | 0.54 |
| Distribution | Moderately right-skewed | Strongly right-skewed |

---

## 2. RT Normalization

### 2.1 Filtering Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_rt` | 250 ms | Minimum RT threshold |
| `rt_filter` | (0.2, 5.0) s | RT range for training |
| `max_trials_per_user` | 25,000 | Cap per user |

### 2.2 Log Normalization (Recommended ✅)

**Rationale**: 
- VAM RT distribution is right-skewed (median < mean)
- 87% of trials in 0.5-1.0s range where log provides 1.5-3.0x resolution advantage
- Consistent with MNIST methodology for cross-dataset comparison

```python
# Log normalization parameters
rt_min = 0.2   # seconds (filtered minimum)
rt_max = 5.0   # seconds (filtered maximum)

# Log transformation
log_rt_min = np.log(rt_min)    # ≈ -1.609
log_rt_max = np.log(rt_max)    # ≈ 1.609
log_rt_range = log_rt_max - log_rt_min  # ≈ 3.219

# Normalization
rt_normalized = (np.log(rt) - log_rt_min) / log_rt_range

# Denormalization
log_rt = rt_normalized * log_rt_range + log_rt_min
rt_seconds = np.exp(log_rt)
```

### 2.3 Resolution Comparison

| RT (s) | Linear Resolution | Log Resolution | Log Advantage |
|--------|-------------------|----------------|---------------|
| 0.3 | 2.1% | 10.4% | **5.0x** |
| 0.5 | 2.1% | 6.2% | **3.0x** |
| 0.7 | 2.1% | 4.4% | **2.1x** |
| 1.0 | 2.1% | 3.1% | 1.5x |
| 2.0 | 2.1% | 1.6% | 0.7x |

**Key insight**: Log normalization provides higher resolution in the 0.5-1.0s range where most VAM trials concentrate.

### 2.4 Linear Normalization (Baseline)

```python
# Linear normalization (alternative)
rt_min = 0.2   # seconds
rt_max = 5.0   # seconds
rt_range = rt_max - rt_min  # 4.8 seconds

rt_normalized = (rt - rt_min) / rt_range
rt_seconds = rt_normalized * rt_range + rt_min
```

---

## 3. Model Architecture

### 3.1 ConvLSTM Layer

| Parameter | Value | Description |
|-----------|-------|-------------|
| `input_channel` | 3 | RGB input |
| `num_filter` | 16 | Number of ConvLSTM filters |
| `kernel_size` | 3 | Convolution kernel size |
| `stride` | 1 | Convolution stride |
| `padding` | 1 | Zero-padding |

### 3.2 Output Layers

| Layer | Configuration | Output Shape |
|-------|---------------|--------------|
| AdaptiveAvgPool2D | output_size=(1,1) | (B, 16, 1, 1) |
| FC (decision) | Linear(16, 4) | (B, 4) |
| Evidence Network | Linear(16→16→1) + Tanh | (B, T) |

### 3.3 Decision Mechanism

| Parameter | Value | Description |
|-----------|-------|-------------|
| `time_steps` | 20 | Number of discrete time steps |
| `threshold` | 6.0 (learnable) | Evidence accumulation threshold |
| `sigma` | 2.0 | Soft decision weighting parameter |

**Evidence Accumulation Formula**:
```
s(t) = cumsum(evidence(t))
decision_time = argmax(s(t) > threshold)
rt_output = (decision_time + 1) / time_steps
```

---

## 4. Noise Parameters

### 4.1 Noise Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `noise_position` | `'evidence'` | Where to inject noise |
| `learnable_noise` | False (exp01) / True (exp02+) | Noise parameter type |

### 4.2 Fixed Noise Values (exp01)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `evidence_noise_std` | **0.5** | Gaussian noise standard deviation |
| `evidence_mask_p` | **0.4** | Dropout probability (40% evidence masked) |
| `evidence_dropout_rescale` | False | No rescaling after dropout |

### 4.3 Learnable Noise (exp02+)

| Parameter | Initial Value | Constraint |
|-----------|---------------|------------|
| `noise_std` | ~0.1 | softplus (positive) |
| `mask_p` | ~0.3 | sigmoid ([0, 1]) |

---

## 5. Training Parameters

### 5.1 Optimizer

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 0 (default) |
| Betas | (0.9, 0.999) (default) |

### 5.2 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 100 | Number of training epochs |
| `batch_size` | 64 | Samples per batch |
| `train_ratio` | 0.8 | Fraction for training |
| `random_seed` | 42 | Reproducibility seed |

### 5.3 Loss Functions

| Loss | Weight | Formula |
|------|--------|---------|
| Decision Loss | 1.0 | CrossEntropyLoss |
| RT Loss | 1.0 | MSELoss (if `use_rt_loss=True`) |

**Total Loss**:
```python
loss = decision_loss + rt_loss
```

### 5.4 Label Learning

| Parameter | Value | Description |
|-----------|-------|-------------|
| `learn_human_response` | True | Learn human response (including errors) |
| `learn_correct_label` | False | Alternative: learn ground truth |

---

## 6. Data Parameters

### 6.1 Dataset Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | VAM Lost in Migration |
| Source | `VAM_Lost-in-Migration/` |
| Image Size | 128×128 (RGB) |
| Classes | 4 (L/R/U/D) |

### 6.2 Data Loading

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_trials_per_user` | 25,000 | Cap per user for balance |
| `min_rt` | 250 ms | Minimum RT filter |
| `precompute_images` | True | Cache preprocessed images |
| `num_workers` | 0 | DataLoader workers |
| `pin_memory` | False | Memory pinning |

### 6.3 Data Transforms

```python
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # No normalization needed (RGB, 0-1 range)
])
```

---

## 7. Experiment Design

### 7.1 Recommended Configuration

**直接使用 Log 归一化 + Fixed Noise 参数（与 MNIST 一致）**

| Parameter | Value | Description |
|-----------|-------|-------------|
| RT Normalization | **Log** | 推荐方案 |
| `learnable_noise` | **False** | 使用固定噪声参数 |
| `evidence_noise_std` | **0.5** | 与 MNIST 一致 |
| `evidence_mask_p` | **0.4** | 与 MNIST 一致 |
| `time_steps` | 20 | 与 MNIST 一致 |
| `num_filter` | 16 | 与 MNIST 一致 |
| `kernel_size` | 3 | 与 MNIST 一致 |

### 7.2 Training Command

**快速测试（单用户）**：

```bash
python src/experiments/vam_convlstm/train_vam.py \
    --data_dir "VAM_Lost-in-Migration" \
    --output_dir "outputs/experiments/vam_convlstm/exp01_log_norm" \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --use_rt_loss \
    --time_steps 20 \
    --num_filter 16 \
    --kernel_size 3 \
    --fixed_noise \
    --evidence_noise_std 0.5 \
    --evidence_mask_p 0.4 \
    --users 677 \
    --device auto
```

**完整训练（所有用户）**：

```bash
python src/experiments/vam_convlstm/train_vam.py \
    --data_dir "VAM_Lost-in-Migration" \
    --output_dir "outputs/experiments/vam_convlstm/exp01_log_norm_full" \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.001 \
    --use_rt_loss \
    --time_steps 20 \
    --num_filter 16 \
    --kernel_size 3 \
    --fixed_noise \
    --evidence_noise_std 0.5 \
    --evidence_mask_p 0.4 \
    --device auto
```

### 7.3 Optional: Fine-tune from MNIST

如果需要从 MNIST 预训练模型微调：

```bash
python src/experiments/vam_convlstm/train_vam.py \
    --data_dir "VAM_Lost-in-Migration" \
    --output_dir "outputs/experiments/vam_convlstm/exp02_finetune_mnist" \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --use_rt_loss \
    --time_steps 20 \
    --num_filter 16 \
    --kernel_size 3 \
    --fixed_noise \
    --evidence_noise_std 0.5 \
    --evidence_mask_p 0.4 \
    --pretrained_model "outputs/experiments/mnist_convlstm/exp01_fixed_noise_ep100/convlstm_nf16_ks3_ep100_bs64_lr0.001_t20_rt_sup_human_resp.pth" \
    --device auto
```

---

## 8. Evaluation Metrics

### 8.1 Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy (correct) | Model vs ground truth label | >80% |
| Accuracy (response) | Model vs human response | >75% |
| RT Correlation | Pearson r (model vs human RT) | >0.1 |
| RT Ratio | Model RT mean / Human RT mean | ~1.0 |

### 8.2 Behavioral Metrics

| Metric | Description | Expected Pattern |
|--------|-------------|------------------|
| Flanker Effect | RT difference (incongruent - congruent) | Positive |
| Error Effect | RT difference (incorrect - correct) | Positive |
| Direction Balance | RT by direction | Similar across L/R/U/D |

### 8.3 Analysis Plots

1. **Training curves**: RT loss, label loss, accuracy, correlation
2. **RT distribution**: Model vs human, by correctness, by congruency
3. **Scatter plot**: Model RT vs Human RT with correlation
4. **Behavioral analysis**: Flanker effect, error patterns

---

## 9. Hardware & Speed

### 9.1 Device Configuration

| Parameter | Value |
|-----------|-------|
| `device` | auto (MPS/CUDA/CPU) |
| `num_workers` | 0 (DataLoader) |
| `pin_memory` | False |

### 9.2 Estimated Training Time

| Configuration | Time/Epoch | 100 Epochs |
|---------------|------------|------------|
| MPS, t=20 | ~15 min | ~25 hours |
| CUDA, t=20 | ~10 min | ~17 hours |
| CPU, t=20 | ~60 min | ~100 hours |

**Note**: VAM has more data than MNIST (1.87M training trials vs ~49K), so training takes longer.

---

## 10. File Structure

```
project/
├── VAM_Lost-in-Migration/
│   ├── gameplay_data/           # User trial data (CSV)
│   ├── graphics/                # Bird images and background
│   ├── metadata.csv             # User demographics
│   └── processed_cache/         # Preprocessed image cache
├── src/
│   ├── experiments/vam_convlstm/
│   │   ├── train_vam.py         # Training script
│   │   └── config.py            # Configuration
│   └── data/
│       └── preprocess_vam_data.py  # Dataset class (with log norm)
├── outputs/experiments/vam_convlstm/
│   └── exp01_log_norm/          # Training results
└── docs/
    └── VAM_PLAN.md              # This document
```

---

## 11. Key Differences from MNIST

| Aspect | MNIST | VAM |
|--------|-------|-----|
| Input channels | 1 (grayscale) | 3 (RGB) |
| Output classes | 8 (digits 1-8) | 4 (directions L/R/U/D) |
| Mean RT | 950 ms | 697 ms |
| RT variability | High (CV=0.54) | Moderate (CV=0.31) |
| Data size | 61K trials | 3.2M trials |
| Special variables | - | Congruency (flanker effect) |
| Pretrained transfer | N/A | MNIST → VAM (1→3 channels) |

---

## 12. Implementation Checklist

### 12.1 Before Training

- [ ] Verify VAM data directory structure
- [ ] Check graphics files (bird images, background)
- [ ] Confirm metadata.csv exists
- [ ] Set random seed for reproducibility

### 12.2 Data Preprocessing

- [ ] Update `preprocess_vam_data.py` to use log normalization
- [ ] Verify RT normalization: `rt_normalized = (log(rt) - log_min) / log_range`
- [ ] Test denormalization: `rt = exp(normalized * log_range + log_min)`

### 12.3 Model Training

- [ ] Run exp01_log_norm with fixed noise parameters
- [ ] Monitor RT ratio (target: ~1.0)
- [ ] Check behavioral patterns (flanker effect)

### 12.4 Evaluation

- [ ] Generate training curves
- [ ] Create RT distribution plots
- [ ] Analyze flanker effect (congruency)
- [ ] Compare with MNIST results

---

## 13. Expected Outcomes

### 13.1 RT Prediction

| Metric | Target | Notes |
|--------|--------|-------|
| RT Ratio | ~1.0-1.2x | Log normalization should improve calibration |
| RT Correlation | >0.1 | Model-human RT correlation |
| Accuracy (correct) | >80% | Classification accuracy vs ground truth |
| Accuracy (response) | >75% | Classification accuracy vs human response |

### 13.2 Behavioral Patterns

| Pattern | Expected Result |
|---------|-----------------|
| Flanker effect | Model should show longer RT for incongruent trials |
| Error patterns | Model should capture human error distribution |
| Direction effects | Minimal differences across directions |

---

## References

1. **VAM Paper**: Jaffe, P. I., et al. (2024). An image-computable model of speeded decision-making. *eLife* 13, RP98351.

2. **Flanker Task**: Eriksen, B. A., & Eriksen, C. W. (1974). Effects of noise letters upon the identification of a target letter in a nonsearch task. *Perception & Psychophysics*, 16(1), 143-149.

3. **Evidence Accumulation**: Gold, J. I., & Shadlen, M. N. (2007). The neural basis of decision making. *Annual Review of Neuroscience*, 30, 535-574.

4. **ConvLSTM**: Shi, X., et al. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. NeurIPS.

---

*Last Updated: 2026-03-15*
