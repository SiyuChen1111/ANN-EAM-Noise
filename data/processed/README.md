# Processed Data

This directory contains processed and analysis-ready data files.

## Directory Structure

```
processed/
├── datasets/       # Processed datasets
└── README.md       # This file
```

## Data Processing Pipeline

Data processing is performed by scripts in `src/data/`:

1. **01_preprocess_mnist_behavioral.py**: Preprocesses MNIST behavioral data
   - Input: `data/raw/mnist/`
   - Output: `data/processed/datasets/`

## Data Transformations

Document all transformations applied to raw data:

### MNIST Behavioral Data

**Processing Steps**:
1. Load raw MNIST images and labels
2. Apply preprocessing transformations
3. Generate behavioral data format
4. Save processed dataset

**Output Files**:
- [List processed files here]

## Important Notes

- All processed data should be reproducible from raw data using scripts
- Document any parameters or configurations used in processing
- Include sample sizes and any filtering criteria

## Regenerating Processed Data

To regenerate all processed data:

```bash
cd src/data
python 01_preprocess_mnist_behavioral.py
```
