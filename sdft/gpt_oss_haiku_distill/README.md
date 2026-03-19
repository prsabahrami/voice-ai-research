# SDFT / SDPO / Haiku Distillation on GPT-OSS

Multi-agent persistent training experiments using the Tinker API.

## Overview

This repository contains training scripts, configs, and results for:

1. **SDFT** (Style-Distillation Fine-Tuning): Supervised fine-tuning of GPT-OSS models on
   transcript and conversational datasets, including Haiku-generated targets.

2. **SDPO** (Style-Distillation Preference Optimization): DPO-style preference learning on
   GPT-OSS models using pairwise data where Haiku responses are "chosen" over baseline.

3. **Haiku Distillation**: Multiple methods for distilling Claude Haiku's conversational style
   into GPT-OSS:
   - Method A: SFT on Haiku outputs (supervised imitation)
   - Method B: DPO with Haiku preferences (preference optimization)
   - Method C: Style-transfer prompting baseline (zero-shot / few-shot)

## Work Division

| Agent | Task |
|-------|------|
| coolstufs (this branch) | SDFT runs + Haiku SFT distillation (Method A) + PR management |
| @serious-inference-engineer | SDPO/DPO runs + Haiku DPO distillation (Method B) |
| @ooo | SDFT on large transcript datasets + Haiku prompting baseline (Method C) |

## Models

- **GPT-OSS-20B** (openai/gpt-oss-20b): Primary training target (MoE, cost-efficient)
- **GPT-OSS-120B** (openai/gpt-oss-120b): Larger scale experiments

## Setup

```bash
pip install tinker transformers torch datasets python-dotenv numpy
export TINKER_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
```

## Running Experiments

### SDFT Training
```bash
python scripts/sdft_train.py --config configs/sdft_gpt_oss_20b.yaml
```

### SDPO / DPO Training
```bash
python scripts/sdpo_train.py --config configs/sdpo_gpt_oss_20b.yaml
```

### Haiku Distillation
```bash
# Method A: SFT
python scripts/haiku_distill_sft.py --config configs/haiku_sft.yaml
# Method B: DPO
python scripts/haiku_distill_dpo.py --config configs/haiku_dpo.yaml
# Method C: Prompting baseline
python scripts/haiku_distill_prompting.py
```

## Results Summary

See `results/` directory for per-run metrics and `RESULTS.md` for aggregate summary.

## Status

This is a living PR. Results are added as runs complete.
