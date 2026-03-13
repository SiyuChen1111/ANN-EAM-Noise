# Project TIER Structure Skill

A Trae IDE skill that implements the TIER Protocol 4.0 for creating standardized, reproducible project folder structures for research and data analysis projects.

## Purpose

This skill helps researchers, students, and data analysts organize their projects according to the TIER Protocol (Teaching Integrity in Empirical Research). The TIER Protocol specifies the contents and organization of reproduction documentation for projects involving computations with statistical data.

### Key Benefits

- **Reproducibility**: Ensures all computations can be reproduced by others
- **Standardization**: Provides a consistent folder structure across projects
- **Portability**: Projects work on any system without modification
- **Documentation**: Forces proper documentation of data sources and transformations
- **One-click execution**: A single master script reproduces all results

### Core Standards

The skill enforces four key standards of reproducibility:

1. **Sufficiency**: Contains everything necessary to reproduce results
2. **Soup-to-nuts**: Covers both data processing and analysis
3. **Portability**: Works on any computer
4. **(Almost) one-click reproducibility**: Single master script execution

## Usage

### Invocation

This skill is automatically invoked when you:
- Ask to organize project files
- Request help creating reproducible documentation
- Want to set up a new research project
- Need to prepare documentation for peer review or publication

### Example Prompts

```
"Help me set up a TIER-compliant project structure"
"Organize my research project folders according to TIER Protocol"
"Create a reproducible project structure for my data analysis"
"Set up documentation for my thesis project"
```

## Standard Folder Structure

When you invoke this skill, it creates the following hierarchy:

```
project-root/
├── README.md                    # Project overview and instructions
├── master-script.ext            # Master script to run all analyses
├── paper/                       # Final report and write-up
│   ├── paper.pdf               # Final paper/report
│   ├── paper-source.ext        # Source document
│   └── figures/                # Figures used in the paper
├── original-data/              # Original, unmodified data files
│   ├── README.md               # Data source documentation
│   └── *.csv|*.dta|*.xlsx      # Original data files
├── command-files/              # All analysis scripts
│   ├── data-processing/        # Scripts to process raw data
│   ├── analysis/               # Scripts for statistical analysis
│   └── output/                 # Generated outputs
├── analysis-data/              # Processed/analysis-ready data
│   ├── README.md               # Documentation of transformations
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

## Usage Examples

### Example 1: Setting Up a New Research Project

**User prompt:**
```
I'm starting a new research project analyzing survey data. 
Help me set up a TIER-compliant folder structure.
```

**Skill response:**
The skill will:
1. Create the complete folder hierarchy
2. Generate template README files
3. Create a master script template
4. Provide guidance on organizing your materials

### Example 2: Organizing an Existing Project

**User prompt:**
```
I have a messy project folder with data files and scripts everywhere. 
Can you help me reorganize it according to TIER Protocol?
```

**Skill response:**
The skill will:
1. Analyze your current folder structure
2. Identify data files, scripts, and outputs
3. Reorganize into TIER-compliant structure
4. Create documentation for each component
5. Generate a master script to tie everything together

### Example 3: Preparing for Publication

**User prompt:**
```
I need to prepare reproducibility documentation for my journal submission.
```

**Skill response:**
The skill will:
1. Verify all original data is properly stored
2. Ensure all processing is scripted
3. Validate that master script runs successfully
4. Check all paths are portable (relative)
5. Generate required documentation files

### Example 4: Creating a Master Script

**User prompt:**
```
Create a master script for my R project that processes data from 
original-data/ and runs all analyses.
```

**Skill response:**
The skill will generate a master script like:

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

cat("All analyses completed successfully!\n")
```

## Key Rules

### Critical Guidelines

1. **Never modify original data**: Files in `original-data/` must remain unchanged
2. **Script everything**: All data transformations must be in scripts, not manual
3. **Use relative paths**: Never hardcode absolute paths for portability
4. **Number scripts sequentially**: Use prefixes like `01_`, `02_`, etc.
5. **Document everything**: Each folder should have a README explaining its contents

### Path Portability Example

```r
# Good: Relative path (portable)
data <- read.csv("original-data/survey-data.csv")

# Bad: Absolute path (not portable)
data <- read.csv("/Users/username/project/original-data/survey-data.csv")
```

## Validation Checklist

Before considering a project complete, verify:

- [ ] All original data files are in `original-data/` and unmodified
- [ ] All data processing is scripted (no manual edits)
- [ ] Master script runs all analyses in correct order
- [ ] All paths are relative (portable)
- [ ] README files document data sources and transformations
- [ ] Codebook describes all variables
- [ ] Results can be reproduced by running only the master script

## Adaptability

The TIER Protocol is flexible and can be adapted for:

| Project Type | Adaptation |
|--------------|------------|
| Small homework | Simplified structure, fewer subdirectories |
| Full research paper | Complete structure as specified |
| Lab exercise | Focus on data and scripts, minimal documentation |
| Large collaborative project | Add subdirectories for each contributor |

### Software Compatibility

Works with any scriptable software:
- **R**: `.R` scripts, R Markdown
- **Python**: `.py` scripts, Jupyter Notebooks
- **Stata**: `.do` files
- **Julia**: `.jl` scripts
- **MATLAB**: `.m` files

## References

- [TIER Protocol 4.0](https://www.projecttier.org/tier-protocol/protocol-4-0/)
- [Project TIER](https://www.projecttier.org/)
- [Open Science Framework TIER Template](https://www.projecttier.org/tier-protocol/open-science-framework/)

## License

This skill is based on the TIER Protocol developed by Project TIER (Teaching Integrity in Empirical Research).
