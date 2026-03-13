# Data Sources

This directory contains original, unmodified data files used in the project.

## Directory Structure

```
raw/
├── mnist/          # MNIST handwritten digit dataset
├── rtnet/          # RTNet behavioral dataset
└── README.md       # This file
```

## Datasets

### 1. MNIST Dataset

**Location**: `raw/mnist/`

**Description**: The MNIST database of handwritten digits with 60,000 training examples and 10,000 test examples.

**Source**: http://yann.lecun.com/exdb/mnist/

**Files**:
- `train-images-idx3-ubyte`: Training set images
- `train-labels-idx1-ubyte`: Training set labels
- `t10k-images-idx3-ubyte`: Test set images
- `t10k-labels-idx1-ubyte`: Test set labels

**Date Acquired**: 2026-03

**License**: Public domain

### 2. RTNet Dataset

**Location**: `raw/rtnet/`

**Description**: Behavioral data from RTNet experiments including reaction time data and experimental stimuli.

**Files**:
- `behavioral data.csv`: Behavioral measurements
- `column info.docx`: Column descriptions
- `experiment_images.mat`: Experimental stimuli images

**Date Acquired**: 2026-03

**Usage Restrictions**: [Add any restrictions here]

## Important Notes

⚠️ **CRITICAL**: Never modify files in this directory. All data transformations should be done through scripts in `src/data/` and saved to `data/processed/`.

## Adding New Data

When adding new datasets:

1. Create a new subdirectory under `raw/`
2. Place original, unmodified data files
3. Update this README with dataset information
4. Document the data source, acquisition date, and any usage restrictions
