# Project Logs

This directory contains detailed logs of project development and experiments.

## Log Files

| Date | File | Description |
|------|------|-------------|
| 2026-03-19 | [2026-03-19_progress_log.md](2026-03-19_progress_log.md) | SAT model development & code organization |
| 2026-03-17 | [2026-03-17_progress_log.md](2026-03-17_progress_log.md) | RT prediction analysis & model improvement |
| 2026-03-14 | [2026-03-14_to_2026-03-16_progress_log.md](2026-03-14_to_2026-03-16_progress_log.md) | Progress from 2026-03-14 to 2026-03-16 |
| 2026-03-13 | [2026-03-13_learnable_noise_training.md](2026-03-13_learnable_noise_training.md) | Training with learnable noise parameters (100 epochs) |
| 2026-03-12 | [2026-03-12_project_reorganization.md](2026-03-12_project_reorganization.md) | Project structure reorganization and initial training |

## Free-Thinking Notes

Located in `free-thinking/` directory:
- `sat_modeling_analysis_2026-03-19.md` - SAT modeling analysis
- `sat_modeling_analysis_2026-03-19_v2.md` - SAT modeling analysis v2

## Log Format

Each log file should contain:

1. **Summary**: Brief overview of the day's work
2. **Tasks Completed**: List of completed tasks with details
3. **Issues Identified**: Problems encountered and solutions
4. **Results**: Key results and metrics
5. **Next Steps**: Planned future work

## Key Issues Tracking

### RT Distribution Mismatch (2026-03-13)

**Status**: 🟡 Partially Resolved

**Problem**: Model RT distribution does not match human RT distribution after making noise parameters learnable.

**Solution Applied**:
- Used log normalization for RT values
- Increased time steps to 40 (Exp11)
- Fixed noise parameters (noise_std=0.5, mask_p=0.4)

**Current Result** (Exp11):
- RT Ratio: 1.27x (improved from 3.0x)
- RT Correlation: 0.029 (positive)

---

### Speed-Accuracy Trade-off (2026-03-17)

**Status**: 🟡 In Progress

**Problem**: Model does NOT capture human-like speed-accuracy trade-off.

**Details**:
- Human: Error trials are SLOWER (+0.095s)
- Model: Error trials have similar RT (-0.007s)
- Root cause: Global threshold for all trials

**Solution Being Implemented**:
- 4-parameter SAT model with condition-specific thresholds
- Improved loss design with condition-specific weights
- Threshold differentiation regularization

---

### SAT Model Accuracy Drop (2026-03-19)

**Status**: 🔴 Active Investigation

**Problem**: SAT model accuracy dropped to ~19% (vs human 70%)

**Details**:
- RT correlation improved (0.17-0.20)
- Threshold evolution did not meet expectations
- Model learned human responses including errors

**Current Approach**:
- Improved loss design with condition-specific weights
- Threshold differentiation regularization
- Test training in progress

---
*Last updated: 2026-03-20*
