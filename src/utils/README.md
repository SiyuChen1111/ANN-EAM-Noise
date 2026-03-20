#   Unified Analysis Tool

A comprehensive analysis script for ConvLSTM RT (Reaction Time) Prediction Model results.

## Features

This script consolidates all experiment analysis functionality into a single tool:

1. **RT Distribution Comparison** - Model vs Human RT distribution by stimulus
2. **Correct vs Error RT Comparison** - RT distribution for correct and error trials
3. **Accuracy Comparison** - Model vs Human accuracy by stimulus
4. **RT Statistics Comparison** - Mean RT and skewness by stimulus
5. **RT Scatter Plot** - Correlation between model and human RT
6. **Difficulty Analysis** - Performance breakdown by Easy/Difficult conditions (if available)
7. **Speed-Accuracy Trade-off** - Analysis of RT vs accuracy relationship
8. **Statistical Report** - Comprehensive statistics with significance tests
9. **CSV Export** - Per-stimulus statistics exported to CSV

## Usage

### Basic Usage

```bash
# Method 1: Specify results CSV path directly
python -m src.utils.unified_analysis outputs/experiments/mnist_convlstm/exp11_t40/results.csv

# Method 2: Use experiment name shortcut
python -m src.utils.unified_analysis --exp exp11_t40

# Method 3: Specify custom output directory
python -m src.utils.unified_analysis results.csv ./analysis_output
```

### Command Line Arguments

| Argument       | Description                                                                      |
| -------------- | -------------------------------------------------------------------------------- |
| `results_path` | Path to results CSV file (optional if using --exp)                               |
| `output_dir`   | Output directory for analysis (optional, defaults to `results_path/../analysis`) |
| `--exp`        | Experiment name (e.g., `exp11_t40`)                                              |
| `--base_dir`   | Base directory for experiments (default: `outputs/experiments/mnist_convlstm`)   |

### Examples

```bash
# Analyze exp11_t40 experiment
python -m src.utils.unified_analysis --exp exp11_t40

# Analyze exp12_t40_ep40 experiment
python -m src.utils.unified_analysis --exp exp12_t40_ep40

# Analyze with custom base directory
python -m src.utils.unified_analysis --exp exp11_t40 --base_dir /path/to/experiments

# Analyze specific results file
python -m src.utils.unified_analysis \
    outputs/experiments/mnist_convlstm/exp11_t40/convlstm_balanced_rt2.0_sp0.1_ep70_results.csv
```

## Input Requirements

The input CSV file should contain the following columns:

| Column             | Description                                         |
| ------------------ | --------------------------------------------------- |
| `true_label`       | Ground truth label                                  |
| `pred_label`       | Model prediction                                    |
| `correct`          | Human correctness (1=correct, 0=error)              |
| `rt_pred_seconds`  | Model predicted RT in seconds                       |
| `rt_human_seconds` | Human RT in seconds                                 |
| `difficulty`       | (Optional) Difficulty level ('easy' or 'difficult') |

## Output

### Generated Visualizations

The script generates 7 PDF/PNG visualizations:

| File                              | Description                              |
| --------------------------------- | ---------------------------------------- |
| `rt_distribution_comparison.pdf`  | RT distribution by stimulus (8 subplots) |
| `correct_error_rt_comparison.pdf` | Correct vs Error RT distribution         |
| `accuracy_comparison.pdf`         | Accuracy comparison bar chart            |
| `rt_stats_comparison.pdf`         | Mean RT and skewness by stimulus         |
| `rt_scatter.pdf`                  | RT correlation scatter plot              |
| `difficulty_analysis.pdf`         | Performance by difficulty (if available) |
| `speed_accuracy_tradeoff.pdf`     | Speed-accuracy trade-off analysis        |

### Statistical Report

The script prints a comprehensive statistical report including:

- Overall accuracy (Model vs Human)
- RT correlation coefficient
- RT by correctness with t-test results
- Per-stimulus statistics table

### CSV Export

- `stimulus_statistics.csv` - Detailed per-stimulus statistics

## Output Directory Structure

```
experiment_dir/
├── results.csv
└── analysis/
    ├── rt_distribution_comparison.pdf
    ├── rt_distribution_comparison.png
    ├── correct_error_rt_comparison.pdf
    ├── correct_error_rt_comparison.png
    ├── accuracy_comparison.pdf
    ├── accuracy_comparison.png
    ├── rt_stats_comparison.pdf
    ├── rt_stats_comparison.png
    ├── rt_scatter.pdf
    ├── rt_scatter.png
    ├── difficulty_analysis.pdf
    ├── difficulty_analysis.png
    ├── speed_accuracy_tradeoff.pdf
    ├── speed_accuracy_tradeoff.png
    └── stimulus_statistics.csv
```

## Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

## Statistical Tests

The script performs the following statistical tests:

- **Independent t-test**: Comparing RT between correct and error trials
- **Correlation analysis**: Pearson correlation between model and human RT
- **Skewness calculation**: Distribution shape analysis

### Significance Levels

| Symbol | p-value   |
| ------ | --------- |
| `*`    | p < 0.05  |
| `**`   | p < 0.01  |
| `***`  | p < 0.001 |

## Notes

- If the `difficulty` column is not present in the input data, the difficulty analysis will be skipped
- All visualizations are saved in both PDF and PNG formats
- The script automatically creates the output directory if it doesn't exist

## Related Files

- `src/experiments/mnist_convlstm/train_sat_4param.py` - Training script for 4-parameter SAT model
- `src/models/convlstm_sat.py` - SAT model definition
- `src/data/preprocess_mnist_behavioral_log.py` - Data preprocessing

