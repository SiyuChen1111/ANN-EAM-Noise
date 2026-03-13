---
name: "project-tier-structure"
description: "Standardizes project folder structure based on TIER Protocol 4.0 for reproducibility. Invoke when user needs to organize project files, create reproducible documentation, or set up a new research project."
---

# Project TIER Structure

This skill helps you implement the TIER Protocol 4.0 to create standardized, reproducible project folder structures for research and data analysis projects.

## Overview

The TIER Protocol (Teaching Integrity in Empirical Research) specifies the contents and organization of reproduction documentation for projects involving computations with statistical data. Documentation that meets these specifications contains all the data, scripts, and supporting information necessary to enable reproducibility.

## Core Standards of Reproducibility

When implementing this protocol, ensure the following standards:

1. **Sufficiency**: The documentation should contain everything necessary to reproduce the results
2. **Soup-to-nuts**: Include scripts for both data processing and analysis
3. **Portability**: Any user should be able to run the scripts on their own computer
4. **(Almost) one-click reproducibility**: Executing a single Master Script should reproduce all computations

## Standard Folder Structure

Create the following hierarchy for a TIER-compliant project:

```
project-root/
├── README.md                    # Project overview and instructions
├── master-script.ext            # Master script to run all analyses
├── paper/                       # Final report and write-up
│   ├── paper.pdf               # Final paper/report
│   ├── paper-source.ext        # Source document (e.g., .tex, .docx, .Rmd)
│   └── figures/                # Figures used in the paper
│       └── *.png|*.pdf
├── original-data/              # Original, unmodified data files
│   ├── README.md               # Data source documentation
│   └── *.csv|*.dta|*.xlsx      # Original data files
├── command-files/              # All analysis scripts
│   ├── data-processing/        # Scripts to process raw data
│   │   ├── 01_import_data.ext
│   │   ├── 02_clean_data.ext
│   │   └── 03_merge_data.ext
│   ├── analysis/               # Scripts for statistical analysis
│   │   ├── 01_descriptive.ext
│   │   ├── 02_regression.ext
│   │   └── 03_robustness.ext
│   └── output/                 # Generated outputs
│       ├── tables/
│       └── figures/
├── analysis-data/              # Processed/analysis-ready data
│   ├── README.md               # Documentation of data transformations
│   └── *.csv|*.dta|*.rds       # Processed datasets
├── metadata/                   # Supporting documentation
│   ├── codebook.md             # Variable descriptions
│   ├── data-dictionary.md      # Data definitions
│   └── methodology.md          # Methodological notes
└── supplementary/              # Additional materials
    ├── literature/
    ├── notes/
    └── archive/
```

## Implementation Steps

### Step 1: Initialize Project Structure

When setting up a new project:

1. Create the root project folder
2. Generate all required subdirectories
3. Create README.md with project overview
4. Set up the master script template

### Step 2: Organize Original Data

1. Place all original, unmodified data files in `original-data/`
2. Create a README documenting:
   - Data sources
   - Date of acquisition
   - Any licensing or usage restrictions
   - Original file formats

**Critical Rule**: Never modify files in `original-data/`. All transformations should be done through scripts.

### Step 3: Create Command Files

1. Number scripts sequentially to indicate execution order
2. Each script should:
   - Use relative paths (for portability)
   - Be self-documenting with clear comments
   - Save outputs to appropriate directories
   - Not contain hardcoded absolute paths

### Step 4: Document Data Processing

1. All data transformations must be scripted
2. Save intermediate and final processed data in `analysis-data/`
3. Document all transformations in `analysis-data/README.md`

### Step 5: Create Master Script

The master script should:
- Set the working directory (relative to script location)
- Execute all processing scripts in correct order
- Execute all analysis scripts
- Generate all outputs

Example structure:
```r
# Master Script
# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Data Processing
source("command-files/data-processing/01_import_data.R")
source("command-files/data-processing/02_clean_data.R")
source("command-files/data-processing/03_merge_data.R")

# Analysis
source("command-files/analysis/01_descriptive.R")
source("command-files/analysis/02_regression.R")
source("command-files/analysis/03_robustness.R")
```

## Path Portability Guidelines

To ensure portability across systems:

1. **Always use relative paths**: Scripts should reference files relative to the project root
2. **Set working directory at script start**: Use the script's location as reference
3. **Avoid absolute paths**: Never hardcode user-specific paths

Example of portable path handling:
```r
# Good: Relative path
data <- read.csv("original-data/survey-data.csv")

# Bad: Absolute path
data <- read.csv("/Users/username/project/original-data/survey-data.csv")
```

## File Naming Conventions

1. Use lowercase with hyphens for folder names: `original-data`, `command-files`
2. Number scripts sequentially: `01_import.R`, `02_clean.R`
3. Use descriptive names that indicate content
4. Include date in original data files if relevant: `survey-2024-03.csv`

## Validation Checklist

Before considering a project TIER-compliant, verify:

- [ ] All original data files are in `original-data/` and unmodified
- [ ] All data processing is scripted (no manual edits)
- [ ] Master script runs all analyses in correct order
- [ ] All paths are relative (portable)
- [ ] README files document data sources and transformations
- [ ] Codebook describes all variables
- [ ] Results can be reproduced by running only the master script

## Adaptability

The TIER Protocol is flexible and can be adapted for:

- **Small projects**: Simplify structure, may not need all subdirectories
- **Large projects**: Add additional subdirectories as needed
- **Different software**: Works with R, Stata, Python, or any scriptable software
- **Dynamic documents**: Adapt for R Markdown, Jupyter Notebooks, etc.

## Common Use Cases

Invoke this skill when:
- Starting a new research project
- Reorganizing an existing project for reproducibility
- Preparing documentation for peer review or publication
- Teaching reproducible research practices
- Collaborating on data analysis projects

## Example Workflow

When a user asks to set up a TIER-compliant project:

1. Ask about project type (research paper, homework, lab exercise)
2. Determine required complexity of folder structure
3. Create the appropriate directory structure
4. Generate template README files
5. Create master script template
6. Provide guidance on organizing existing materials
