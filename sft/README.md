# SFT Baseline Experiments

**Owner:** ooo
**Branch:** sft-branch

## Objective

Establish a Supervised Fine-Tuning baseline on LJSpeech-1.1 using an OSS voice AI model.
This baseline provides the reference checkpoint for DPO and SDFT experiments.

## Getting Started

```bash
git checkout sft-branch
cd sft/
```

## Expected Deliverables

- `train_sft.py` -- training script
- `config/` -- training configs
- `checkpoints/` -- saved model checkpoints (or pointer to Lambda path)
- `results/` -- RTF, MOS, WER evaluation results
- `README.md` -- update with your results

## Key Paths on Lambda H100

- Dataset: `/home/ubuntu/datasets/LJSpeech-1.1/`
- Venv: `/home/ubuntu/voice_ai_venv`
- SFT dir: `/home/ubuntu/voice_ai_sft_baseline/`
