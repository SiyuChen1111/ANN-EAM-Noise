# ANN-EAM-Noise

A research project implementing Evidence Accumulation Models (EAM) with neural networks for modeling decision-making processes and reaction times.

## Project Overview

This project combines cognitive modeling (Evidence Accumulation Models) with deep learning approaches to model and predict human decision-making behavior, including reaction times and accuracy patterns.

### Key Features

- 🧠 **Evidence Accumulation Modeling**: Implementation of EAM for cognitive modeling
- 🔬 **Neural Network Integration**: Deep learning models (ConvLSTM, AlexNet-LSTM) for behavioral prediction
- 📊 **APA-Compliant Visualization**: Publication-ready figures following APA guidelines
- 🔄 **Reproducible Research**: TIER Protocol-compliant project structure
- ⚡ **SAT (Speed-Accuracy Trade-off)**: Condition-specific decision thresholds

## Project Structure

```
ANN-EAM-Nosie/
├── data/                      # Data files
│   ├── raw/                  # Original, unmodified data
│   │   ├── mnist/           # MNIST dataset
│   │   └── rtnet/           # RTNet behavioral data
│   ├── processed/            # Processed datasets
│   └── generators/           # Data generation scripts
│
├── src/                      # Source code
│   ├── data/                # Data processing scripts
│   ├── models/              # Model definitions
│   │   └── convlstm_sat.py  # SAT model implementation
│   ├── experiments/         # Experiment scripts
│   │   └── mnist_convlstm/  # MNIST ConvLSTM experiments
│   └── utils/               # Utility functions
│       └── unified_analysis.py  # Unified analysis tool
│
├── outputs/                  # Experiment outputs
│   └── experiments/
│       └── mnist_convlstm/  # Experiment results
│           └── logs/        # Progress logs
│
├── drafts/                   # Work-in-progress
│   ├── ideas/               # Ideas and notes
│   └── papers/              # Draft manuscripts
│
├── logs/                    # Project progress logs
├── references/              # Reference materials
├── scripts/                 # Utility scripts
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone [repository-url]
cd ANN-EAM-Nosie
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running Experiments

**Best Model (Exp11)**:
```bash
cd src/experiments/mnist_convlstm
python 02_train_model.py --epochs 70 --time_steps 40 --rt_loss_weight 2.0 --speed_penalty 0.1
```

**SAT Model Training**:
```bash
python train_sat_4param.py --pretrained_path path/to/exp11_model.pth
```

**Analysis**:
```bash
# By experiment name
python -m src.utils.unified_analysis --exp exp11_t40

# By results path
python -m src.utils.unified_analysis path/to/results.csv
```

Results will be saved in `outputs/experiments/mnist_convlstm/`.

## Key Components

### 1. Evidence Accumulation Model (EAM)

The EAM module implements cognitive models of decision-making:

- Drift rate modeling
- Decision threshold mechanisms
- Reaction time prediction
- Speed-accuracy trade-off

**Location**: `src/models/eam/`

### 2. Neural Network Models

#### ConvLSTM (Primary Model)

Convolutional LSTM for sequential image processing:

- Spatial feature extraction (16 filters, kernel=3)
- Temporal dynamics modeling
- Evidence accumulation mechanism
- Differentiable decision function
- Reaction time supervision

#### ConvLSTM-SAT

Extended ConvLSTM with Speed-Accuracy Trade-off:

- Condition-specific thresholds (speed vs accuracy)
- Learnable threshold parameters
- Transfer learning from base model

### 3. Unified Analysis Tool

Consolidated analysis pipeline for all experiments:

```bash
python -m src.utils.unified_analysis --exp <experiment_name>
```

Generates:
- RT distribution comparison
- Correct vs Error RT analysis
- Accuracy comparison by stimulus
- Difficulty analysis
- Speed-accuracy trade-off analysis
- Statistical reports

### 4. Visualization

APA-compliant visualization tools:

- Colorblind-friendly palettes
- 300 DPI output quality
- Publication-ready figures

## Best Model Results (Exp11)

**Status**: ✅ Best performing model

| Metric | Model | Human | Notes |
|--------|-------|-------|-------|
| Overall Accuracy | **80.9%** | 70.4% | Exceeds human |
| RT Ratio | **1.27x** | - | Best result |
| RT Correlation | **0.029** | - | Positive |
| Final Threshold | 4.28 | - | Learned |

### Difficulty Analysis

| Difficulty | Model Accuracy | Human Accuracy | RT Ratio |
|------------|----------------|----------------|----------|
| Easy | 91.7% | 81.3% | 1.33x |
| Difficult | 69.9% | 59.6% | 1.21x |

### Speed-Accuracy Trade-off

| Behavior | Human | Model |
|----------|-------|-------|
| Error RT | SLOWER (+0.095s) | Similar (-0.007s) |
| Root Cause | Difficulty-specific thresholds | Global threshold |

**Finding**: Model captures RT matching but not human-like speed-accuracy trade-off.

## Experiments Summary

| Experiment | Time Steps | Epochs | Accuracy | RT Ratio | RT Corr | Status |
|------------|------------|--------|----------|----------|---------|--------|
| Exp07 | 20 | 100 | 77.9% | 2.03x | 0.001 | Done |
| Exp08 | 20 | 70 | 78.8% | 1.66x | -0.007 | Done |
| Exp10 | 25 | 70 | 72.7% | 1.55x | -0.005 | Done |
| **Exp11** | **40** | **70** | **80.9%** | **1.27x** | **0.029** | **Best** |
| Exp12 | 40 | 40 | 69.2% | 1.33x | 0.024 | Done |
| Exp_SAT | 40 | 70 | ~19% | - | 0.17-0.20 | Issues |
| Exp_SAT_Improved | 40 | 10 | - | - | - | Running |

## SAT Model Development

### Current Status

The SAT (Speed-Accuracy Trade-off) model is under development to capture human-like decision strategies:

1. **Basic SAT Model**: Implemented condition-specific thresholds
2. **4-Parameter SAT**: Added fixed penalty coefficients
3. **Improved Loss Design**: Condition-specific loss weights (in progress)

### Research Plans

Located in `drafts/ideas/`:
- `SAT_4param_research_plan.md` - 4-parameter SAT research plan
- `SAT_Improved_Loss_Plan.md` - Improved loss design plan

## Documentation

### Experiment Logs

Located in `logs/`:
- `2026-03-19_progress_log.md` - Latest progress
- `2026-03-17_progress_log.md` - RT analysis and model improvement
- `2026-03-14_progress_log.md` - Progress from 2026-03-14 to 2026-03-16

### Experiment-Specific Documentation

- `src/experiments/mnist_convlstm/README.txt` - Experiment directory guide
- `src/utils/README.md` - Unified analysis tool usage

## Reproducibility

This project follows the **TIER Protocol 4.0** for reproducible research:

✅ **Sufficiency**: All data and code included  
✅ **Soup-to-nuts**: Complete pipeline from raw data to results  
✅ **Portability**: Relative paths, no hardcoded locations  
✅ **One-click reproducibility**: Scripts reproduce all results

### Reproducing Results

1. Ensure data is in `data/raw/`
2. Run processing scripts in `src/data/`
3. Run experiment scripts in `src/experiments/`
4. Analyze results with `python -m src.utils.unified_analysis`

## Technologies

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization

## Contributing

This is a research project. For questions or collaboration:

- Open an issue on GitHub
- Contact: [Your contact information]

## License

[Add license information]

## Citation

If you use this code, please cite:

```bibtex
@misc{ann_eam_noise_2026,
  title={ANN-EAM-Noise: Neural Network Evidence Accumulation Models},
  author={[Your Name]},
  year={2026},
  publisher={GitHub},
  url={[repository-url]}
}
```

## Acknowledgments

- TIER Protocol for reproducibility guidelines
- APA Publication Manual for visualization standards
- Alós-Ferrer & Garagnani (2026) for SAT theoretical framework
- Rafiei et al. (2024) for RTNet reference

---

**Last Updated**: 2026-03-20

**Project Status**: 🔄 Active Development (SAT Model)
