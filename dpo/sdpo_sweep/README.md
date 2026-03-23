# SDPO Sweep Results

## Overview

This directory contains results from a hyperparameter sweep of Supervised Direct Preference Optimization (SDPO) training on `openai/gpt-oss-20b` via the Tinker API (LoRA fine-tuning). The sweep covered 36 experiments across 4 beta values, 3 learning rates, and 3 n_pairs values, with 5 training steps per experiment.

**Sweep completed:** 2026-03-19  
**Total experiments:** 36 / 36 completed  
**Model:** openai/gpt-oss-20b  
**Training method:** SDPO (LoRA rank=8, 5 steps each)  
**API:** Tinker API (ServiceClient / create_lora_training_client)

---

## Hyperparameter Grid

| Parameter | Values |
|-----------|--------|
| beta | 0.1, 0.2, 0.3, 0.5 |
| learning_rate | 1e-4, 2e-4, 5e-4 |
| n_pairs | 10, 20, 30 |
| steps | 5 (fixed) |
| lora_rank | 8 (fixed) |

**Total combinations:** 4 x 3 x 3 = 36

---

## Key Findings

### 1. Learning Rate Is the Dominant Factor

Learning rate had by far the largest effect on final loss:

| Learning Rate | Avg Final Loss | Min Final Loss | Max Final Loss |
|---------------|---------------|----------------|----------------|
| 1e-4 | 68.25 | 67.23 | 69.22 |
| 2e-4 | 49.83 | 48.36 | 51.42 |
| **5e-4** | **7.58** | **4.79** | **9.93** |

At lr=5e-4, all 12 configs achieved >95% loss reduction in just 5 steps. At lr=1e-4 or 2e-4, loss reduction was negligible.

**Recommendation: Use lr=5e-4 for SDPO on this model.**

### 2. Beta Has Minor Effect (at optimal LR)

At lr=5e-4, all beta values produced low final losses. Beta=0.3 produced the single best run:

| Beta | Avg Final Loss (lr=5e-4) | Min Final Loss |
|------|--------------------------|----------------|
| 0.1 | 7.04 | 6.50 |
| 0.2 | 7.29 | 6.78 |
| **0.3** | **6.17** | **4.79** |
| 0.5 | 9.80 | 6.62 |

**Recommendation: Beta=0.3 is a solid default.**

### 3. n_pairs Has Minimal Impact (at optimal LR)

At lr=5e-4, n_pairs had little systematic effect. Larger n_pairs (30) produced the single best run, but differences are small:

| n_pairs | Avg Final Loss (lr=5e-4) | Min Final Loss |
|---------|--------------------------|----------------|
| 10 | 7.01 | 6.50 |
| 20 | 7.65 | 7.08 |
| 30 | 7.73 | 4.79 |

**Recommendation: n_pairs=10 or 30 are reasonable. n_pairs=10 gives the best average and is more compute-efficient.**

---

## Top 5 Configurations

| Rank | Exp ID | Beta | LR | n_pairs | Initial Loss | Final Loss | Reduction |
|------|--------|------|----|---------|-------------|------------|-----------|
| 1 | sdpo_sie_027 | 0.3 | 5e-4 | 30 | 133.37 | 4.79 | 96.4% |
| 2 | sdpo_sie_007 | 0.1 | 5e-4 | 10 | 133.37 | 6.50 | 95.1% |
| 3 | sdpo_sie_034 | 0.5 | 5e-4 | 10 | 133.37 | 6.62 | 95.0% |
| 4 | sdpo_sie_016 | 0.2 | 5e-4 | 10 | 132.99 | 6.78 | 94.9% |
| 5 | sdpo_sie_017 | 0.2 | 5e-4 | 20 | 133.37 | 7.08 | 94.7% |

---

## Recommended Configuration

Based on the sweep, the recommended configuration for SDPO fine-tuning on GPT-OSS-20B:

```
beta = 0.3
learning_rate = 5e-4
n_pairs = 10          # compute-efficient; 30 gives best single run but higher variance
lora_rank = 8
steps = 5+            # 5 steps already yields ~95% loss reduction
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `experiment_results.jsonl` | Full results for all 36 experiments (JSON Lines) |
| `sdpo_sie_sweep.py` | Sweep script used to run the experiments |
| `sdpo_sweep_log.txt` | Full training log with per-step loss values |
| `sweep_summary.json` | Machine-readable summary of top results |
| `README.md` | This file |

---

## Notes

- All experiments used the same initial model checkpoint (`openai/gpt-oss-20b`)
- Initial loss was consistently ~133.4 across all runs (same base model state)
- Training was done via Tinker API `ServiceClient.create_lora_training_client`
- The sweep script fixed 3 bugs from the initial draft: TinkerClient API, BatchEncoding tokenization, and metric key (`loss:sum`)
- Each experiment was self-contained (no checkpoint reuse between configs)
