# MNIST ConvLSTM Experiment

This experiment trains a Convolutional LSTM model on the MNIST dataset for behavioral modeling.

## Experiment Overview

**Objective**: Train a ConvLSTM model to predict reaction times and accuracy on MNIST digit classification.

**Date Started**: 2026-03-12

**Status**: Completed

## Directory Structure

```
mnist_convlstm/
├── 02_train_model.py      # Training script
├── 03_evaluate_model.py   # Evaluation script
└── README.md              # This file
```

## Model Configuration

**Architecture**: ConvLSTM

**Hyperparameters**:
- Number of filters: 16
- Kernel size: 3
- Epochs: 50
- Batch size: 64
- Learning rate: 0.001
- Time steps: 20

**Training Configuration**:
- Optimizer: Adam
- Loss function: [Specify loss function]
- Device: [CPU/GPU]

## Results

**Training Performance**:
- Final training loss: [Value]
- Final training accuracy: [Value]

**Test Performance**:
- Test accuracy: [Value]
- Average reaction time: [Value]

**Output Files**:
- Model: `outputs/experiments/mnist_convlstm/models/convlstm_model.pth`
- Figures: `outputs/experiments/mnist_convlstm/figures/`
  - Training curves
  - RT distribution
  - Accuracy comparison
- Results: `outputs/experiments/mnist_convlstm/results.csv`

## How to Run

### Training

```bash
cd src/experiments/mnist_convlstm
python 02_train_model.py
```

### Evaluation

```bash
python 03_evaluate_model.py
```

## Key Findings

[Document key findings and insights from this experiment]

## Notes

- Model trained on MNIST dataset
- Reaction time supervision applied
- Results saved with APA-formatted visualizations

## Related Experiments

- MNIST AlexNet-LSTM: `../mnist_alexnet_lstm/`

## References

[Add relevant references]
