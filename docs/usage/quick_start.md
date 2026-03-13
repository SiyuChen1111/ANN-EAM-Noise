# Quick Start Guide

This guide will help you quickly set up and run experiments in the ANN-EAM-Noise project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for version control)

## Installation

### 1. Clone the Repository

```bash
git clone [repository-url]
cd ANN-EAM-Nosie
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
ANN-EAM-Nosie/
├── data/               # Data files
│   ├── raw/           # Original data (read-only)
│   └── processed/     # Processed datasets
├── src/               # Source code
│   ├── data/         # Data processing scripts
│   ├── models/       # Model definitions
│   ├── experiments/  # Experiment scripts
│   └── utils/        # Utility functions
├── notebooks/         # Jupyter notebooks
├── outputs/           # Experiment outputs
├── docs/              # Documentation
└── README.md          # Project overview
```

## Running Your First Experiment

### MNIST ConvLSTM Experiment

1. **Prepare Data** (if not already prepared):

```bash
cd src/data
python 01_preprocess_mnist_behavioral.py
```

2. **Train Model**:

```bash
cd ../experiments/mnist_convlstm
python 02_train_model.py
```

3. **Evaluate Model**:

```bash
python 03_evaluate_model.py
```

4. **View Results**:

Results are saved in `outputs/experiments/mnist_convlstm/`:
- `models/`: Trained model checkpoints
- `figures/`: Visualization plots
- `results.csv`: Performance metrics

### Using Jupyter Notebooks

For interactive exploration:

```bash
cd notebooks/experiments
jupyter notebook
```

Open any `.ipynb` file to run experiments interactively.

## Common Tasks

### Running a New Experiment

1. Create experiment directory:

```bash
mkdir -p src/experiments/my_experiment
```

2. Create experiment scripts:
   - `01_prepare_data.py`: Data preparation
   - `02_train_model.py`: Model training
   - `03_evaluate_model.py`: Model evaluation

3. Create output directory:

```bash
mkdir -p outputs/experiments/my_experiment/{models,figures,logs}
```

4. Run the experiment:

```bash
cd src/experiments/my_experiment
python 02_train_model.py
python 03_evaluate_model.py
```

### Adding a New Model

1. Create model file:

```bash
touch src/models/my_model.py
```

2. Implement model class:

```python
import torch.nn as nn

class MyModel(nn.Module):
    """
    Description of your model.
    
    Args:
        param1 (type): Description
    """
    def __init__(self, param1):
        super().__init__()
        # Define layers
    
    def forward(self, x):
        # Define forward pass
        return output
```

3. Import in experiments:

```python
from src.models.my_model import MyModel
```

### Working with Data

**Load raw data**:

```python
from torchvision import datasets

mnist = datasets.MNIST('data/raw/mnist', download=True)
```

**Load processed data**:

```python
import pandas as pd

data = pd.read_csv('data/processed/datasets/my_data.csv')
```

**Save processed data**:

```python
data.to_csv('data/processed/datasets/my_processed_data.csv', index=False)
```

## Visualization

### Creating APA-Formatted Figures

The project includes an APA visualization skill for creating publication-ready figures:

```python
from src.utils.visualization import create_apa_bar_chart

create_apa_bar_chart(
    data=[85, 92, 78],
    labels=['Group A', 'Group B', 'Group C'],
    ylabel='Accuracy (%)',
    fig_num=1,
    title='Model Performance Comparison',
    note='N = 100 per group.'
)
```

## Troubleshooting

### Common Issues

**Issue**: Module not found error

**Solution**: Ensure you're in the project root directory and virtual environment is activated:
```bash
cd /path/to/ANN-EAM-Nosie
source venv/bin/activate
```

**Issue**: Data not found

**Solution**: Check that data files exist in `data/raw/` or run data preparation scripts.

**Issue**: CUDA out of memory

**Solution**: Reduce batch size in training script or use CPU:
```python
device = torch.device('cpu')
```

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Code Comments**: Read docstrings in source code
- **Issues**: Open an issue on GitHub
- **Examples**: See `notebooks/experiments/` for examples

## Next Steps

1. Explore the [Model Architecture Documentation](../models/architecture.md)
2. Read about [Data Sources](../data/data_sources.md)
3. Review existing experiments in `src/experiments/`
4. Start your own experiment!

## Tips for Success

- ✅ Always work in a virtual environment
- ✅ Document your experiments in README files
- ✅ Use version control (Git) for code changes
- ✅ Keep raw data unchanged
- ✅ Save model checkpoints regularly
- ✅ Use descriptive names for experiments and files
- ✅ Follow the project structure conventions
