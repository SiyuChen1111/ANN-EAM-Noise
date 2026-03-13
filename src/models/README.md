# Models

This directory contains model definitions and architectures.

## Directory Structure

```
models/
├── eam/            # Evidence Accumulation Model
│   └── __init__.py
├── encoders/       # Encoder architectures
│   └── __init__.py
└── README.md       # This file
```

## Model Architectures

### 1. Evidence Accumulation Model (EAM)

**Location**: `eam/`

**Description**: 
Evidence Accumulation Model for modeling decision-making processes. This model simulates how evidence accumulates over time to reach a decision threshold.

**Key Components**:
- Evidence accumulation mechanism
- Decision threshold
- Response time modeling

**Architecture Details**:
- Input: [Describe input format]
- Hidden layers: [Describe architecture]
- Output: [Describe output format]

**Parameters**:
- Threshold parameter: θ
- Drift rate: v
- Non-decision time: Ter

### 2. Encoders

**Location**: `encoders/`

**Description**: 
Encoder architectures for feature extraction from input data.

**Available Encoders**:
- [List encoder types here]

## Model Training

Models are trained using scripts in `src/experiments/`:

```bash
# Example: Train ConvLSTM model on MNIST
cd src/experiments/mnist_convlstm
python 02_train_model.py
```

## Model Checkpoints

Trained model checkpoints are saved in `outputs/experiments/[experiment_name]/models/`:
- `best_model.pth`: Best performing model
- `checkpoints/`: Periodic training checkpoints

## Adding New Models

When adding new models:

1. Create a new subdirectory under `models/`
2. Implement model class with clear docstrings
3. Add `__init__.py` to make it importable
4. Update this README with model documentation
5. Document model architecture, parameters, and usage

## Model Documentation Standards

Each model should include:

```python
class MyModel(nn.Module):
    """
    Brief description of the model.
    
    Args:
        param1 (type): Description of parameter
        param2 (type): Description of parameter
    
    Architecture:
        - Layer 1: [description]
        - Layer 2: [description]
    
    Example:
        >>> model = MyModel(param1=value1, param2=value2)
        >>> output = model(input_data)
    """
```
