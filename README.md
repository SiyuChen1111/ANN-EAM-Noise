# ANN-EAM-Noise

A research project implementing Evidence Accumulation Models (EAM) with neural networks for modeling decision-making processes and reaction times.

## Project Overview

This project combines cognitive modeling (Evidence Accumulation Models) with deep learning approaches to model and predict human decision-making behavior, including reaction times and accuracy patterns.

### Key Features

- 🧠 **Evidence Accumulation Modeling**: Implementation of EAM for cognitive modeling
- 🔬 **Neural Network Integration**: Deep learning models (ConvLSTM, AlexNet-LSTM) for behavioral prediction
- 📊 **APA-Compliant Visualization**: Publication-ready figures following APA guidelines
- 🔄 **Reproducible Research**: TIER Protocol-compliant project structure

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
│   │   ├── eam/            # Evidence Accumulation Model
│   │   └── encoders/       # Encoder architectures
│   ├── experiments/         # Experiment scripts
│   │   ├── mnist_convlstm/ # MNIST ConvLSTM experiment
│   │   └── mnist_alexnet_lstm/
│   └── utils/               # Utility functions
│
├── notebooks/                # Jupyter notebooks
│   ├── exploration/         # Exploratory analysis
│   └── experiments/         # Experiment notebooks
│
├── outputs/                  # Experiment outputs
│   ├── experiments/         # Results by experiment
│   └── analysis/            # Cross-experiment analysis
│
├── docs/                     # Documentation
│   ├── data/                # Data documentation
│   ├── models/              # Model documentation
│   ├── experiments/         # Experiment logs
│   └── usage/               # Usage guides
│
├── drafts/                   # Work-in-progress
│   ├── ideas/               # Ideas and notes
│   └── papers/              # Draft manuscripts
│
├── .trae/                    # Trae IDE configuration
│   └── skills/              # Agent skills
│       ├── apa-visualization/
│       └── project-tier-structure/
│
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

**MNIST ConvLSTM Experiment**:
```bash
cd src/experiments/mnist_convlstm
python 02_train_model.py
python 03_evaluate_model.py
```

Results will be saved in `outputs/experiments/mnist_convlstm/`.

For detailed instructions, see the [Quick Start Guide](docs/usage/quick_start.md).

## Key Components

### 1. Evidence Accumulation Model (EAM)

The EAM module implements cognitive models of decision-making:

- Drift rate modeling
- Decision threshold mechanisms
- Reaction time prediction

**Location**: `src/models/eam/`

**Documentation**: [Model Architecture](docs/models/architecture.md)

### 2. Neural Network Models

#### ConvLSTM

Convolutional LSTM for sequential image processing:

- Spatial feature extraction
- Temporal dynamics modeling
- Reaction time supervision

#### AlexNet-LSTM

AlexNet feature extractor combined with LSTM:

- Pre-trained feature extraction
- Sequential processing
- Behavioral prediction

### 3. Data Processing

Data pipeline from raw to analysis-ready:

```
Raw Data → Processing Scripts → Processed Data → Model Training
```

**Scripts**: `src/data/`

**Documentation**: [Data Sources](docs/data/data_sources.md)

### 4. Visualization

APA-compliant visualization tools for publication-ready figures:

- Bar charts, line graphs, scatter plots
- Colorblind-friendly palettes
- 300 DPI output quality

**Skill**: `.trae/skills/apa-visualization/`

## Experiments

### MNIST ConvLSTM with Learnable Noise (100 Epochs)

**Status**: ✅ Completed (2026-03-13)

**Objective**: Train ConvLSTM with learnable noise parameters to predict human responses and RTs

**Key Results**:
| Metric | Value |
|--------|-------|
| Accuracy (vs correct label) | 78.44% |
| Accuracy (vs human response) | 63.66% |
| RT Correlation | 0.1105 |

**Critical Issue**: RT distribution does NOT match human RT distribution
- Model RT: ~2.9s (uniform distribution)
- Human RT: ~0.91s (right-skewed)
- Noise parameters collapsed to near-zero

**Location**: `outputs/experiments/mnist_convlstm/learnable_noise_ep100/`

**Documentation**: [Experiment README](outputs/experiments/mnist_convlstm/learnable_noise_ep100/README.md)

### MNIST ConvLSTM (Initial)

**Status**: ✅ Completed

**Objective**: Train ConvLSTM on MNIST for reaction time prediction

**Results**:
- Model: `outputs/experiments/mnist_convlstm/models/convlstm_model.pth`
- Figures: `outputs/experiments/mnist_convlstm/analysis/`
- Results: `outputs/experiments/mnist_convlstm/results.csv`

**Documentation**: [Experiment README](src/experiments/mnist_convlstm/README.md)

### MNIST AlexNet-LSTM

**Status**: 🔄 In Progress

**Objective**: Compare AlexNet-LSTM performance with ConvLSTM

**Location**: `src/experiments/mnist_alexnet_lstm/`

## Documentation

### For Users

- [Quick Start Guide](docs/usage/quick_start.md) - Get started quickly
- [Data Sources](docs/data/data_sources.md) - Data documentation
- [Model Architecture](docs/models/architecture.md) - Model details

### For Developers

- [Project Structure](#project-structure) - Directory organization
- [Adding New Models](docs/models/architecture.md#implementation-guidelines) - Development guide
- [Code Documentation](src/models/README.md) - Code structure

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
4. Check outputs in `outputs/experiments/`

## Technologies

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Jupyter** - Interactive analysis

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
- [Other acknowledgments]

---

**Last Updated**: 2026-03-13

**Project Status**: 🔄 Active Development
