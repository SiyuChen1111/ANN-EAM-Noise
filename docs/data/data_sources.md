# Data Sources

This document provides detailed information about all data sources used in the ANN-EAM-Noise project.

## Table of Contents

1. [MNIST Dataset](#mnist-dataset)
2. [RTNet Dataset](#rtnet-dataset)
3. [Data Usage Guidelines](#data-usage-guidelines)

---

## MNIST Dataset

### Overview

The MNIST (Modified National Institute of Standards and Technology) database is a large database of handwritten digits commonly used for training various image processing systems.

### Dataset Details

**Official Name**: The MNIST Database of Handwritten Digit Images

**Source**: http://yann.lecun.com/exdb/mnist/

**Creators**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges

**Date Acquired**: March 2026

**Version**: Original version

### Dataset Statistics

- **Training Set**: 60,000 examples
- **Test Set**: 10,000 examples
- **Image Size**: 28 × 28 pixels
- **Color Space**: Grayscale
- **Classes**: 10 (digits 0-9)

### File Structure

```
data/raw/mnist/
├── train-images-idx3-ubyte    # Training images
├── train-labels-idx1-ubyte    # Training labels
├── t10k-images-idx3-ubyte     # Test images
└── t10k-labels-idx1-ubyte     # Test labels
```

### License

**License Type**: Public Domain

**Usage Restrictions**: None. The dataset is freely available for any use.

### Citation

If you use this dataset, please cite:

```bibtex
@article{lecun1998mnist,
  title={The MNIST database of handwritten digit images for machine learning research},
  author={LeCun, Yann and Cortes, Corinna and Burges, Christopher JC},
  journal={IEEE Signal Processing Magazine},
  volume={29},
  number={6},
  pages={141--142},
  year={1998},
  publisher={IEEE}
}
```

### Related Publications

- LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.

---

## RTNet Dataset

### Overview

The RTNet dataset contains behavioral data from cognitive experiments, including reaction times and experimental stimuli.

### Dataset Details

**Official Name**: RTNet Behavioral Dataset

**Source**: [Add source information]

**Date Acquired**: March 2026

### Dataset Contents

**Files**:
- `behavioral data.csv`: Behavioral measurements including reaction times and accuracy
- `column info.docx`: Column descriptions and variable definitions
- `experiment_images.mat`: Experimental stimuli images in MATLAB format

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| [Add variables] | | |

### License

**License Type**: [Add license information]

**Usage Restrictions**: [Add any restrictions]

### Citation

```bibtex
[Add citation information]
```

---

## Data Usage Guidelines

### Data Integrity

1. **Never modify raw data**: Files in `data/raw/` should remain unchanged
2. **Document all transformations**: All processing steps must be scripted and documented
3. **Version control**: Keep track of data versions and processing dates

### Data Processing Workflow

```
Raw Data (data/raw/)
    ↓
Processing Scripts (src/data/)
    ↓
Processed Data (data/processed/)
    ↓
Model Training (src/experiments/)
```

### Adding New Data

When adding new datasets:

1. Create a new subdirectory under `data/raw/`
2. Place original, unmodified data files
3. Document the data source in this file
4. Create processing scripts in `src/data/`
5. Update the data dictionary

### Data Backup

- Raw data is backed up in the project backup archive
- Processed data can be regenerated from raw data using scripts
- Model outputs are saved in `outputs/experiments/`

---

## Contact

For questions about data usage or to report issues:

- Project Maintainer: [Your Name]
- Email: [Your Email]
- GitHub Issues: [Project Repository URL]
