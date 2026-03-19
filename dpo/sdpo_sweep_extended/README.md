# Extended SDPO Sweep Results

## Overview

This directory contains results from the extended SDPO (Sequence-level DPO) hyperparameter sweep
conducted on GPT-OSS-20B via the Tinker API. The extended sweep ran each configuration for
**20 training steps** (vs. 5 steps in the original sweep) to assess convergence behavior.

## Experiment Design

- **Model**: openai/gpt-oss-20b
- **Method**: SDPO (Sequence-level Direct Preference Optimization)
- **Total experiments**: 20 (5 betas x 4 n_pairs values)
- **Steps per experiment**: 20
- **Fixed learning rate**: 5e-4
- **LoRA rank**: 8
- **Beta values tested**: 0.1, 0.2, 0.3, 0.4, 0.5
- **n_pairs values tested**: 10, 20, 30, 50
- **Failures**: 0 / 20

## Best Configuration

| Parameter | Value |
|-----------|-------|
| beta | 0.5 |
| learning_rate | 5e-4 |
| n_pairs | 20 |
| steps | 20 |
| **final_loss** | **0.0028** |
| initial_loss | 133.37 |
| loss_reduction | 100.0% |

## Key Findings

### 20 Steps vs. 5 Steps

The extended sweep demonstrates that 5 training steps is severely insufficient for SDPO:

- **Original sweep best (5 steps)**: beta=0.3, lr=5e-4, n_pairs=30 -> final_loss=4.785
- **Extended sweep best (20 steps)**: beta=0.5, lr=5e-4, n_pairs=20 -> final_loss=0.0028
- **Improvement factor**: ~1714x better final loss with 4x more steps

This is not a marginal improvement. It indicates the model is still deep in initial descent at
step 5 and that running even 15-20 steps is essential for meaningful SDPO training.

### Convergence Analysis

Based on observed loss curves across experiments, the typical convergence trajectory is:

| Step | Approximate Loss |
|------|-----------------|
| 0 (initial) | ~133 |
| 1 | ~60 |
| 2 | ~22 |
| 3 | ~9 |
| 4 | ~4.2 |
| 5 | ~0.9 |
| 6 | ~0.06 |
| 7+ | sub-1.0 with occasional spikes |
| 20 | 0.003 - 2.6 depending on config |

Loss consistently drops below 1.0 by step 7 for well-tuned configurations.
The 5-step results were capturing measurements before convergence had fully occurred.

### Top 10 Configurations

| Rank | exp_id | beta | n_pairs | final_loss |
|------|--------|------|---------|------------|
| 1 | sdpo_sie_ext_018 | 0.5 | 20 | 0.0028 |
| 2 | sdpo_sie_ext_001 | 0.1 | 10 | 0.0180 |
| 3 | sdpo_sie_ext_010 | 0.3 | 20 | 0.0273 |
| 4 | sdpo_sie_ext_003 | 0.1 | 30 | 0.0308 |
| 5 | sdpo_sie_ext_011 | 0.3 | 30 | 0.0433 |
| 6 | sdpo_sie_ext_008 | 0.2 | 50 | 0.0470 |
| 7 | sdpo_sie_ext_016 | 0.4 | 50 | 0.0481 |
| 8 | sdpo_sie_ext_005 | 0.2 | 10 | 0.0622 |
| 9 | sdpo_sie_ext_017 | 0.5 | 10 | 0.1186 |
| 10 | sdpo_sie_ext_015 | 0.4 | 30 | 0.1277 |

### n_pairs Analysis

n_pairs=20 appears to be a sweet spot across multiple beta values. In particular:
- beta=0.5, n_pairs=20: 0.0028 (rank 1)
- beta=0.3, n_pairs=20: 0.0273 (rank 3)

Larger n_pairs (30, 50) does not consistently improve results and sometimes performs worse,
suggesting diminishing returns and potential noise from too many comparison pairs per step.

## Production Recommendation

**Use 15-20 steps as the minimum for any production SDPO run.**

The 5-step default used in the original sweep is insufficient. Based on convergence analysis:
- 15 steps: captures most of the improvement (loss typically sub-0.1)
- 20 steps: recommended, achieves final_loss < 0.003 for the best config
- Fewer than 10 steps: captures measurements before meaningful convergence

For production hyperparameters:
- beta: 0.5 (winner), but 0.1 and 0.3 are competitive
- lr: 5e-4 (not swept in this run; validated from prior sweep)
- n_pairs: 20 (sweet spot)
- steps: 20 (recommended minimum)

## Files in This Directory

| File | Description |
|------|-------------|
| `experiment_results_extended.jsonl` | Raw experiment results (includes ~3 duplicate entries from PID 49402) |
| `extended_results_clean.jsonl` | Deduplicated results (20 unique experiments, one entry per exp_id) |
| `extended_sweep_final_report.txt` | Full formatted report with per-beta breakdowns and comparison vs original sweep |
| `sdpo_sie_extended_sweep.py` | Sweep script used to generate results |
| `sdpo_extended_sweep_log.txt` | Execution log from the sweep run |

## Data Quality Note

The raw results file (`experiment_results_extended.jsonl`) contains approximately 3 entries
from a duplicate process (PID 49402, timestamps 22:16-22:17 UTC). The clean file
(`extended_results_clean.jsonl`) uses the latest-timestamp entry per exp_id and represents
20 unique experiments from PID 50111. All 20 experiments completed successfully.

## Context

This extended sweep was conducted as a follow-up to the original 36-experiment SDPO sweep
(results in `dpo/sdpo_sweep/`). The original sweep used 5 steps and found a best loss of
4.785. This extended sweep confirms that 20 steps dramatically outperforms 5 steps across
all tested hyperparameter configurations.
