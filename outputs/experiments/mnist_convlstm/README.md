# MNIST ConvLSTM Experiment Results

This directory contains all outputs from the MNIST ConvLSTM experiment.

## Directory Structure

```
mnist_convlstm/
├── models/              # Trained model files
│   ├── best_model.pth  # Best performing model
│   └── checkpoints/    # Training checkpoints
├── figures/             # Generated visualizations
│   ├── training_curves_apa.png
│   ├── rt_distribution_apa.png
│   └── rt_by_stimulus_apa.png
├── logs/                # Training logs
├── analysis/            # Analysis results
│   ├── accuracy_comparison.pdf
│   ├── rt_distribution_comparison.pdf
│   └── stimulus_statistics.csv
├── results.csv          # Main results file
└── README.md            # This file
```

## Model Files

### best_model.pth

**Description**: Best performing model checkpoint

**Performance**:
- Validation accuracy: [Value]
- Validation loss: [Value]

**Usage**:
```python
import torch
model = torch.load('models/best_model.pth')
```

## Figures

All figures are formatted according to APA style guidelines.

### 1. Training Curves

**File**: `training_curves_apa.png`

**Description**: Shows training and validation loss/accuracy over epochs.

### 2. RT Distribution

**File**: `rt_distribution_apa.png`

**Description**: Distribution of reaction times across the dataset.

### 3. RT by Stimulus

**File**: `rt_by_stimulus_apa.png`

**Description**: Reaction times grouped by stimulus type.

## Results Summary

**Main Results** (`results.csv`):

| Metric | Value |
|--------|-------|
| Test Accuracy | [Value] |
| Average RT | [Value] |
| RT SD | [Value] |

## Analysis

The `analysis/` directory contains detailed statistical analysis:

- **accuracy_comparison.pdf**: Comparison of accuracy across conditions
- **rt_distribution_comparison.pdf**: RT distribution analysis
- **stimulus_statistics.csv**: Per-stimulus performance metrics

## Reproducibility

To reproduce these results:

```bash
# From project root
cd src/experiments/mnist_convlstm
python 02_train_model.py
python 03_evaluate_model.py
```

## Notes

- All visualizations use APA-compliant formatting
- Color palettes are colorblind-friendly
- Figures are saved at 300 DPI for publication quality
