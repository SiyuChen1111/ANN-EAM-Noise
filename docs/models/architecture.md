# Model Architecture

This document describes the model architectures used in the ANN-EAM-Noise project.

## Table of Contents

1. [Evidence Accumulation Model (EAM)](#evidence-accumulation-model-eam)
2. [ConvLSTM Architecture](#convlstm-architecture)
3. [AlexNet-LSTM Architecture](#alexnet-lstm-architecture)

---

## Evidence Accumulation Model (EAM)

### Overview

The Evidence Accumulation Model (EAM) is a cognitive modeling framework that simulates decision-making processes by modeling how evidence accumulates over time until reaching a decision threshold.

### Theoretical Background

EAM is based on the drift-diffusion model and related sequential sampling models. It posits that:

1. Decision-makers accumulate evidence over time
2. Evidence accumulates until reaching a threshold
3. The time to reach threshold determines reaction time
4. The threshold reached determines the decision

### Key Components

#### 1. Drift Rate (v)

- **Description**: Rate of evidence accumulation
- **Typical Range**: 0.1 - 1.0
- **Interpretation**: Higher drift rate = faster/more accurate decisions

#### 2. Decision Threshold (θ)

- **Description**: Amount of evidence needed to make a decision
- **Typical Range**: 0.5 - 2.0
- **Interpretation**: Higher threshold = slower but more accurate decisions

#### 3. Non-decision Time (Ter)

- **Description**: Time for perceptual encoding and motor response
- **Typical Range**: 200 - 400 ms
- **Interpretation**: Baseline reaction time independent of decision process

#### 4. Starting Point (z)

- **Description**: Initial bias in evidence accumulation
- **Typical Range**: 0 - θ
- **Interpretation**: Prior bias toward one decision

### Mathematical Formulation

The evidence accumulation process can be described as:

```
dX(t) = v·dt + σ·dW(t)
```

Where:
- X(t) = accumulated evidence at time t
- v = drift rate
- σ = noise parameter
- W(t) = Wiener process

### Implementation

**Location**: `src/models/eam/`

```python
class EAM(nn.Module):
    """
    Evidence Accumulation Model implementation.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        threshold (float): Decision threshold
        dt (float): Time step size
    
    Architecture:
        - Feature encoder: [Describe encoder]
        - Accumulation layer: [Describe accumulation mechanism]
        - Decision layer: [Describe decision mechanism]
    """
```

---

## ConvLSTM Architecture

### Overview

Convolutional LSTM combines convolutional layers with LSTM for processing sequential image data.

### Architecture Details

**Experiment**: MNIST ConvLSTM

**Location**: `src/experiments/mnist_convlstm/`

### Network Structure

```
Input: (batch, time, channels, height, width)
    ↓
ConvLSTM Layer 1
    - Filters: 16
    - Kernel size: 3×3
    - Hidden state: 16 channels
    ↓
ConvLSTM Layer 2 (optional)
    ↓
Output Layer
    - Fully connected
    - Output: (batch, num_classes)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| num_filters | 16 | Number of convolutional filters |
| kernel_size | 3 | Convolutional kernel size |
| time_steps | 20 | Number of time steps |
| batch_size | 64 | Training batch size |
| learning_rate | 0.001 | Adam optimizer learning rate |
| epochs | 50 | Number of training epochs |

### Key Features

1. **Spatial Feature Extraction**: Convolutional layers extract spatial features
2. **Temporal Dynamics**: LSTM captures temporal dependencies
3. **Reaction Time Modeling**: Time steps model evidence accumulation

### Training Configuration

```python
model = ConvLSTM(
    input_size=(28, 28),
    input_dim=1,
    hidden_dim=16,
    kernel_size=(3, 3),
    num_layers=1,
    batch_first=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

---

## AlexNet-LSTM Architecture

### Overview

AlexNet-LSTM combines AlexNet feature extraction with LSTM for sequential processing.

### Architecture Details

**Experiment**: MNIST AlexNet-LSTM

**Location**: `src/experiments/mnist_alexnet_lstm/`

### Network Structure

```
Input: (batch, channels, height, width)
    ↓
AlexNet Feature Extractor
    - Conv1: 96 filters, 11×11 kernel
    - Conv2: 256 filters, 5×5 kernel
    - Conv3: 384 filters, 3×3 kernel
    - Conv4: 384 filters, 3×3 kernel
    - Conv5: 256 filters, 3×3 kernel
    ↓
LSTM Layer
    - Hidden size: [Specify]
    - Num layers: [Specify]
    ↓
Output Layer
    - Fully connected
    - Output: (batch, num_classes)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| [Add parameters] | | |

---

## Model Comparison

| Model | Parameters | Training Time | Accuracy | RT Prediction |
|-------|-----------|---------------|----------|---------------|
| ConvLSTM | ~50K | ~30 min | [Value] | [Value] |
| AlexNet-LSTM | ~[Value] | ~[Value] | [Value] | [Value] |

---

## Implementation Guidelines

### Adding New Models

1. Create model class in `src/models/[model_name]/`
2. Inherit from `nn.Module`
3. Implement `__init__()` and `forward()` methods
4. Add comprehensive docstrings
5. Create corresponding experiment directory in `src/experiments/`
6. Update this documentation

### Model Documentation Standards

```python
class MyModel(nn.Module):
    """
    Brief description of the model.
    
    Args:
        param1 (type): Description
        param2 (type): Description
    
    Architecture:
        - Layer 1: Description
        - Layer 2: Description
    
    Example:
        >>> model = MyModel(param1=value1)
        >>> output = model(input_data)
    
    References:
        - [Paper citation]
    """
```

---

## References

1. Ratcliff, R., & McKoon, G. (2008). The diffusion decision model: theory and data. *Neural Computation*.

2. Shi, X., et al. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *NIPS*.

3. Krizhevsky, A., et al. (2012). ImageNet classification with deep convolutional neural networks. *NIPS*.
