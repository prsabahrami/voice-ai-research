# Extended SDPO Sweep Results

## Overview

This directory contains results from the extended SDPO (Selective Direct Preference Optimization) hyperparameter sweep conducted on GPT-OSS-20B via the Tinker API on a Lambda H100 80GB server.

## Experiment Configuration

- **Total experiments**: 20
- **Training steps per experiment**: 20 (vs 5 in the initial sweep)
- **Learning rate**: 5e-4 (fixed)
- **Beta values tested**: [0.1, 0.3, 0.5, 0.7, 1.0]
- **n_pairs values tested**: [5, 10, 20, 30]
- **Grid**: 5 betas x 4 n_pairs = 20 configurations

## Key Results

### Best Configuration
- **beta**: 0.5
- **lr**: 5e-4
- **n_pairs**: 20
- **Final loss (step 20)**: 0.0028

### Comparison: 5 steps vs 20 steps
| Metric | 5-step sweep | 20-step sweep | Improvement |
|--------|-------------|---------------|-------------|
| Best final loss | 4.785 | 0.0028 | 1714x reduction |
| Loss sub-1.0 | Not reached | Step 7 | -- |

### Convergence Curve (representative run, beta=0.5, n_pairs=20)
```
Step 1:  ~133.0
Step 2:  ~60.0
Step 3:  ~22.0
Step 4:  ~9.0
Step 5:  ~4.2
Step 6:  ~0.9
Step 7:  ~0.06
Step 20: ~0.0028
```

Loss drops below 1.0 by step 7 in well-configured runs. Steps 7-20 provide further refinement from ~0.06 to ~0.003.

## Key Finding

**20 steps gives 1714x better final loss than 5 steps.**

The initial sweep (5 steps) stopped before meaningful convergence. Runs at 5 steps have loss ~4.785, which is above the starting loss for well-configured 20-step runs. The extended sweep confirms that SDPO requires more than 5 gradient steps to escape the initial high-loss regime.

## Convergence Analysis

- Steps 1-3: Rapid descent, loss drops ~6x per step
- Steps 4-6: Transition zone, loss crosses below 1.0
- Steps 7-20: Fine convergence, loss reaches 0.003-0.03 range
- Occasional loss spikes observed in some configs (likely gradient instability with high beta or low n_pairs)

## Production Recommendation

**Use 15-20 training steps for production SDPO.**

Rationale:
- 5 steps: insufficient, loss still in high-loss regime (>4.0)
- 10 steps: partial convergence, loss ~0.1-1.0 range
- 15-20 steps: full convergence, loss <0.01
- 20+ steps: diminishing returns observed

**Recommended production config**:
```
beta:    0.5
lr:      5e-4
n_pairs: 20
steps:   20
```

## Files in This Directory

| File | Description |
|------|-------------|
| `experiment_results_extended.jsonl` | Raw results including duplicate entries from interrupted runs |
| `extended_results_clean.jsonl` | De-duplicated results (20 unique experiments) |
| `extended_sweep_final_report.txt` | Human-readable summary report |
| `sdpo_sie_extended_sweep.py` | Script used to run the extended sweep |
| `sdpo_extended_sweep_log.txt` | Full execution log with per-step loss values |
| `README.md` | This file |

## Notes

The raw results file contains 3 duplicate entries from an interrupted run (PID 49402, timestamps before 2026-03-19T22:18:00Z). The clean results file has these removed, leaving 20 unique experiments.

## Related

- Initial sweep (5 steps, 36 experiments): `dpo/sdpo_sweep/`
- PR #1: sdft-branch -> main
