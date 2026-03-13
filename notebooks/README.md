# Jupyter Notebooks

This directory contains Jupyter notebooks for interactive analysis and experimentation.

## Directory Structure

```
notebooks/
├── exploration/     # Exploratory data analysis
│   └── data_exploration.ipynb
├── experiments/     # Experiment notebooks
│   ├── RTNet.ipynb
│   ├── example.ipynb
│   ├── run_mnist_convlstm.ipynb
│   ├── run_mnist_model.ipynb
│   └── train.ipynb
└── README.md        # This file
```

## Notebook Categories

### Exploration Notebooks

**Purpose**: Data exploration and visualization

**Contents**:
- Data distribution analysis
- Feature inspection
- Preliminary visualizations

### Experiment Notebooks

**Purpose**: Running and documenting experiments

**Contents**:
- Model training
- Hyperparameter tuning
- Results visualization
- Performance analysis

## Best Practices

### Notebook Organization

1. **Clear Structure**: Use markdown cells to organize sections
2. **Documentation**: Add explanations for each code cell
3. **Reproducibility**: Set random seeds and document versions
4. **Clean Output**: Clear output before committing

### Notebook Template

```markdown
# Experiment Title

## Overview
Brief description of the experiment.

## Setup
Import libraries and set configurations.

## Data Loading
Load and inspect data.

## Model Definition
Define or import model.

## Training
Train the model with documentation.

## Evaluation
Evaluate and visualize results.

## Conclusions
Summary of findings.
```

### Version Control

**Important**: Jupyter notebooks can be difficult to version control due to their JSON format.

**Recommendations**:
- Clear all output before committing
- Use `.gitignore` to exclude checkpoint files
- Consider using Jupytext for plain-text version

## Running Notebooks

### Start Jupyter

```bash
cd notebooks
jupyter notebook
```

### Run in Virtual Environment

Ensure your virtual environment is activated:

```bash
source venv/bin/activate
jupyter notebook
```

## Converting Notebooks to Scripts

For better reproducibility, consider converting notebooks to Python scripts:

```bash
jupyter nbconvert --to script notebooks/experiments/my_notebook.ipynb
```

Save the script in `src/experiments/[experiment_name]/`.

## Notebook Dependencies

All notebooks should use the project's virtual environment with dependencies from `requirements.txt`.

**Key Libraries**:
- Jupyter
- IPython
- Matplotlib
- NumPy
- Pandas
- PyTorch

## Tips

- ✅ Use relative paths for data files
- ✅ Document all parameters and configurations
- ✅ Save important results to `outputs/`
- ✅ Keep notebooks focused on single topics
- ✅ Use meaningful notebook names
