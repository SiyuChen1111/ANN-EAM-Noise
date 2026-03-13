# Project Logs

This directory contains detailed logs of project development and experiments.

## Log Files

| Date | File | Description |
|------|------|-------------|
| 2026-03-13 | [learnable_noise_training.md](2026-03-13_learnable_noise_training.md) | Training with learnable noise parameters (100 epochs) |
| 2026-03-12 | [project_reorganization.md](2026-03-12_project_reorganization.md) | Project structure reorganization and initial training |

## Log Format

Each log file should contain:

1. **Summary**: Brief overview of the day's work
2. **Tasks Completed**: List of completed tasks with details
3. **Issues Identified**: Problems encountered and solutions
4. **Results**: Key results and metrics
5. **Next Steps**: Planned future work

## Key Issues Tracking

### RT Distribution Mismatch (2026-03-13)

**Status**: 🔴 Unresolved

**Problem**: Model RT distribution does not match human RT distribution after making noise parameters learnable.

**Details**:
- Model RT: ~2.9s, nearly uniform distribution
- Human RT: ~0.91s, right-skewed distribution
- Learned noise_std collapsed to 0.0004

**Potential Solutions**:
1. Add KL divergence loss for RT distribution
2. Add skewness penalty
3. Different noise parameter initialization
4. Adjust threshold learning

---
*Last updated: 2026-03-13*
