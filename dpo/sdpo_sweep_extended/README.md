# Extended SDPO Hyperparameter Sweep

## Overview

This directory contains results from an extended SDPO (Selective Direct Preference Optimization)
hyperparameter sweep using GPT-OSS-20B via the Tinker API on an H100 80GB GPU.

The sweep was conducted as a follow-up to the original 5-step sweep (see `dpo/sdpo_sweep/`),
extending training to 20 steps per experiment to characterize convergence behavior and identify
production-ready hyperparameter configurations.

## Sweep Configuration

| Parameter     | Values                     |
|---------------|----------------------------|
| beta          | 0.1, 0.2, 0.3, 0.4, 0.5   |
| n_pairs       | 10, 20, 30, 50             |
| learning_rate | 5e-4 (fixed)               |
| steps         | 20 (fixed)                 |
| total runs    | 20 (5 betas x 4 n_pairs)   |
| failures      | 0                          |

## Key Results

### Best Configuration

**beta=0.5, lr=5e-4, n_pairs=20** -> final_loss=0.0028

All 20 experiments converged successfully. Initial loss was ~133.37 in all cases.

### Top 10 Configurations

| Rank | Experiment         | beta | n_pairs | Final Loss |
|------|--------------------|------|---------|------------|
| 1    | sdpo_sie_ext_018   | 0.5  | 20      | 0.0028     |
| 2    | sdpo_sie_ext_001   | 0.1  | 10      | 0.0180     |
| 3    | sdpo_sie_ext_010   | 0.3  | 20      | 0.0273     |
| 4    | sdpo_sie_ext_003   | 0.1  | 30      | 0.0308     |
| 5    | sdpo_sie_ext_011   | 0.3  | 30      | 0.0433     |
| 6    | sdpo_sie_ext_008   | 0.2  | 50      | 0.0470     |
| 7    | sdpo_sie_ext_016   | 0.4  | 50      | 0.0481     |
| 8    | sdpo_sie_ext_005   | 0.2  | 10      | 0.0622     |
| 9    | sdpo_sie_ext_017   | 0.5  | 10      | 0.1186     |
| 10   | sdpo_sie_ext_015   | 0.4  | 30      | 0.1277     |

### Full Results by Beta Group

| beta | n_pairs | Final Loss | Reduction |
|------|---------|------------|-----------|
| 0.1  | 10      | 0.0180     | 100.0%    |
| 0.1  | 20      | 2.6271     | 98.0%     |
| 0.1  | 30      | 0.0308     | 100.0%    |
| 0.1  | 50      | 0.2962     | 99.8%     |
| 0.2  | 10      | 0.0622     | 100.0%    |
| 0.2  | 20      | 0.4036     | 99.7%     |
| 0.2  | 30      | 0.2958     | 99.8%     |
| 0.2  | 50      | 0.0470     | 100.0%    |
| 0.3  | 10      | 0.7760     | 99.4%     |
| 0.3  | 20      | 0.0273     | 100.0%    |
| 0.3  | 30      | 0.0433     | 100.0%    |
| 0.3  | 50      | 0.5893     | 99.6%     |
| 0.4  | 10      | 0.1863     | 99.9%     |
| 0.4  | 20      | 1.1105     | 99.2%     |
| 0.4  | 30      | 0.1277     | 99.9%     |
| 0.4  | 50      | 0.0481     | 100.0%    |
| 0.5  | 10      | 0.1186     | 99.9%     |
| 0.5  | 20      | 0.0028     | 100.0%    |
| 0.5  | 30      | 1.1847     | 99.1%     |
| 0.5  | 50      | 0.5627     | 99.6%     |

## Convergence Analysis

Loss trajectory (representative run, best config beta=0.5, n_pairs=20):

```
Step  1: ~133
Step  2: ~60
Step  3: ~22
Step  4: ~9
Step  5: ~4.2   <- original sweep stopped here (best was 4.785)
Step  6: ~0.9
Step  7: ~0.06  <- sub-1.0 threshold crossed
Step 10: ~0.02
Step 15: ~0.005
Step 20:  0.0028
```

**Key finding: Loss drops below 1.0 by step 7.** The original 5-step sweep terminated
at step 5, before the model had the opportunity to enter the low-loss regime.

## Comparison vs Original 5-Step Sweep

| Sweep       | Best config                           | Final loss |
|-------------|---------------------------------------|------------|
| Original    | beta=0.3, lr=5e-4, n_pairs=30         | 4.785      |
| Extended    | beta=0.5, lr=5e-4, n_pairs=20         | 0.0028     |
| Improvement | 4x more steps (5 -> 20)               | **1714x**  |

20 training steps yields a 1714x improvement in final loss over 5 steps.

## Production Recommendation

**Use 15-20 training steps for production SDPO.** The critical convergence event
(loss crossing below 1.0) occurs around step 7 for the best configurations. Training
for only 5 steps leaves the model in a high-loss region before this transition.

- Minimum viable: 10 steps (most configs reach sub-1.0 loss)
- Recommended: 15-20 steps (captures full convergence)
- Best hyperparams: beta=0.5, lr=5e-4, n_pairs=20

## Files

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `experiment_results_extended.jsonl` | Raw results from all 20 experiments (20 steps each) |
| `extended_results_clean.jsonl`      | Deduplicated results (duplicate entries removed)    |
| `extended_sweep_final_report.txt`   | Full analysis report with rankings and comparison   |
| `sdpo_sie_extended_sweep.py`        | Python sweep script used to run experiments         |
| `sdpo_extended_sweep_log.txt`       | Full execution log from sweep run                   |

## Data Quality Note

The raw results file contains ~3 duplicate entries from an accidentally-started
duplicate process (timestamps 22:16-22:17 UTC on 2026-03-19). The `extended_results_clean.jsonl`
file has these removed. All 20 unique experiments use the latest-timestamp entry per
experiment ID, all from the canonical run (PID 50111, started 22:17 UTC).
