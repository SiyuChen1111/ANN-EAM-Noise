# ANN-EAM-Noise Project Log

## 2026-03-12 Project Reorganization Log

### Completed Tasks

#### 1. Project File Structure Reorganization
Organized project files into appropriate folders based on functionality:

| File Type | Target Folder | Files |
|-----------|---------------|-------|
| Python Scripts | `scripts/` | `1_train_rt_matched.py`, `preprocess_mnist_behavioral.py`, `train_mnist_alexnet_lstm.py`, `train_mnist_convlstm.py` |
| Markdown Drafts | `drafts/` | `EAM_证据累积模型综述.md`, `FS-Net模型架构与运作说明.md`, `MNIST.md` |
| Jupyter Notebooks | `notebooks/` | `RTNet.ipynb`, `example.ipynb`, `run_mnist_convlstm.ipynb`, `run_mnist_model.ipynb`, `train.ipynb` |
| Datasets | `data/` | `RTNet_Dataset/`, `mnist-data/` |

#### 2. Created Folder Structure

```
ANN-EAM-Nosie/
├── data/                          # Data files
│   ├── RTNet_Dataset/             # RTNet behavioral dataset
│   ├── datasets/                  # Additional datasets (reserved)
│   ├── generators/                # Data generators (reserved)
│   └── mnist-data/                # MNIST raw data
│
├── drafts/                        # Draft documents (work in progress)
│   ├── EAM_证据累积模型综述.md
│   ├── FS-Net模型架构与运作说明.md
│   └── MNIST.md
│
├── logs/                          # Training logs
│
├── models/                        # Model code
│   ├── eam/                       # EAM model implementation
│   ├── complete_models/           # Complete models (reserved)
│   └── encoders/                  # Encoders (reserved)
│
├── notebooks/                     # Jupyter notebooks
│   ├── RTNet.ipynb
│   ├── example.ipynb
│   ├── run_mnist_convlstm.ipynb
│   ├── run_mnist_model.ipynb
│   └── train.ipynb
│
├── outputs/                       # Output results
│   ├── analysis/                  # Analysis results
│   ├── logs/                      # Output logs
│   ├── models/                    # Saved model weights
│   └── plots/                     # Plot files
│
├── references/                    # Reference materials
│
├── scripts/                       # Python scripts
│   ├── 1_train_rt_matched.py
│   ├── preprocess_mnist_behavioral.py
│   ├── train_mnist_alexnet_lstm.py
│   └── train_mnist_convlstm.py
│
├── skills/                        # AI skill reference resources
│   ├── AI-Python-for-Deep-Learning/
│   └── skills-main/
│
├── training/                      # Training configs (reserved)
│
└── utils/                         # Utility functions (reserved)
```

#### 3. Model Architecture Overview

The project includes two main model architectures:

1. **ConvLSTM Model** (`train_mnist_convlstm.py`)
   - Convolutional LSTM-based evidence accumulation model
   - Supports noise injection (evidence noise, dropout)
   - Differentiable decision time computation
   - Supports RT-supervised training

2. **AlexNet-LSTM Model** (`train_mnist_alexnet_lstm.py`)
   - Uses pretrained AlexNet as feature extractor
   - LSTM for temporal evidence accumulation
   - Supports freezing/fine-tuning encoder
   - Includes best model saving mechanism

#### 4. Data Description

- **RTNet_Dataset**: Contains behavioral data (`behavioral data.csv`), column info (`column info.docx`), and experiment images (`experiment_images.mat`)
- **MNIST Data**: Standard MNIST handwritten digit dataset

### Pending Tasks

- [x] Run model training and record results
- [ ] Add more model variants
- [ ] Improve documentation
- [ ] Add unit tests

---

## 2026-03-12 Model Training & Analysis Updates

### 1. ConvLSTM Model Training (First Version)

**File**: `scripts/train_mnist_convlstm.py`

**Training Configuration**:
- Epochs: 50
- Batch Size: 64
- Learning Rate: 0.001
- Time Steps: 20
- RT Supervision: Enabled (`--use_rt_loss`)
- Noise: Fixed (std=0.1, mask_p=0.3)

**Results** (saved in `output_mnist_convlstm/`):
- Model Accuracy: 99.67%
- Human Accuracy: 70.44%
- RT Correlation: r = 0.0419
- Learned Threshold: 0.8762

**Key Issues Identified**:
1. Model accuracy far exceeds human accuracy (not learning human errors)
2. Model RT distribution not right-skewed like human RT
3. Model RT ~3x slower than human RT
4. RT correlation very low

### 2. Analysis Script Created

**File**: `analyze_results.py`

**Features**:
- RT distribution comparison (Model vs Human) for each stimulus
- Correct vs Error RT distribution comparison (similar to Fig. 4e from Rafiei et al. 2024)
- Accuracy comparison per stimulus
- RT statistics comparison (mean and skewness)
- Statistical significance tests (t-test)

**Output** (saved in `output_mnist_convlstm/analysis/`):
- `rt_distribution_comparison.pdf` - RT distributions for all 8 stimuli
- `correct_error_rt_comparison.pdf` - Correct vs Error RT comparison
- `accuracy_comparison.pdf` - Accuracy by stimulus
- `rt_stats_comparison.pdf` - RT mean and skewness comparison
- `stimulus_statistics.csv` - Detailed statistics

### 3. Training Script Modifications

#### 3.1 Learnable Noise Parameters

**Modified**: `scripts/train_mnist_convlstm.py`

**Changes**:
- Added `learnable_noise` parameter to `RTify_ConvLSTM` class
- `noise_std` and `mask_p` are now learnable by default
- Constraints implemented:
  - `noise_std`: via softplus (ensures ≥ 0)
  - `mask_p`: via sigmoid (ensures ∈ [0, 1])
- Initial values: std=0.1, mask_p=0.3

**New CLI Arguments**:
```bash
--fixed_noise          # Use fixed noise parameters (disable learning)
--learn_correct_label  # Learn correct label instead of human response
```

**Code Changes**:
```python
# RTify_ConvLSTM class
if learnable_noise:
    self._noise_std_raw = nn.Parameter(torch.tensor(0.1).log())
    self._mask_p_raw = nn.Parameter(torch.tensor(0.3).logit())
else:
    self._fixed_noise_std = evidence_noise_std
    self._fixed_mask_p = evidence_mask_p

@property
def noise_std(self):
    if self.learnable_noise:
        return torch.nn.functional.softplus(self._noise_std_raw)
    return self._fixed_noise_std

@property
def mask_p(self):
    if self.learnable_noise:
        return torch.sigmoid(self._mask_p_raw)
    return self._fixed_mask_p
```

#### 3.2 Learning Human Response (Including Errors)

**Critical Change**: Model now learns human's actual response (including errors) instead of correct labels.

**Before**:
```python
labels = batch['label']  # Correct stimulus label
label_loss = label_criterion(decision_logits, labels)
```

**After**:
```python
response = batch['response']  # Human's actual response (may be wrong)
target = response if learn_human_response else labels
label_loss = label_criterion(decision_logits, target)
```

**Expected Impact**:
- Model accuracy should match human accuracy (~70%)
- Better fit to human RT distribution
- Model learns human error patterns

**Evaluation Metrics Updated**:
- `accuracy_correct`: Model prediction vs correct label
- `accuracy_response`: Model prediction vs human response

**Output File Naming**:
- Now includes training mode: `_human_resp` or `_correct_label`
- Example: `convlstm_nf16_ks3_ep50_bs64_lr0.001_t20_rt_sup_human_resp`

### 4. Visualization Code Updates

**Modified**: `analyze_results.py`

**Changes**:
- All RT values now displayed in **seconds** (not normalized)
- Uses `rt_pred_seconds` and `rt_human_seconds` columns
- Removed `set_xlim(0, 1)` constraints
- Updated axis labels to "RT (seconds)"

**Before vs After**:
| Metric | Before (normalized) | After (seconds) |
|--------|---------------------|-----------------|
| Model RT (Correct) | 0.5681 | 2.97 s |
| Model RT (Error) | 0.5593 | 2.92 s |
| Human RT (Correct) | 0.1314 | 0.91 s |
| Human RT (Error) | 0.1518 | 1.01 s |

### 5. Bug Fixes

**File**: `scripts/preprocess_mnist_behavioral.py`

**Fixed**: Error trials count calculation
```python
# Before (incorrect - bitwise NOT on int)
(~self.filtered_data['correct']).sum()

# After (correct - comparison)
(self.filtered_data['correct'] == 0).sum()
```

### 6. Dataset Information

**Dataset**: `data/RTNet_Dataset/behavioral data.csv`

**Key Columns**:
- `stim`: Stimulus label (1-8, mapped to 0-7 for model)
- `response`: Human's response (1-8, may differ from stim)
- `correct`: Whether human response was correct (0 or 1)
- `resp_rt`: Human reaction time in seconds
- `mnist_index`: Index into MNIST dataset

**Statistics**:
- Total trials: ~15,000
- Correct trials: ~10,500 (70%)
- Error trials: ~4,500 (30%)
- RT range: 0.2 - 5.0 seconds

### Next Steps

- [ ] Train model with learnable noise parameters
- [ ] Train model learning human response (including errors)
- [ ] Compare results with previous version
- [ ] Analyze if model RT distribution better matches human

---
*Log updated: 2026-03-12*

### Notes

- All empty folders have `.gitkeep` files for Git tracking
- `skills/` folder contains AI-assisted development reference resources, can be removed or archived as needed
- `drafts/` folder contains work-in-progress documents, not final versions

---
*Log created: 2026-03-12*
