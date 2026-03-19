# SDFT Experiments

**Owner:** coolstufs
**Branch:** sdft-branch

## Objective

Implement Self-Distillation Fine-Tuning (SDFT) using the SFT baseline as the teacher model.
Compare quality and latency against the SFT baseline.

## Getting Started

```bash
git checkout sdft-branch
cd sdft/
```

## Expected Deliverables

- `train_sdft.py` -- SDFT training script
- `config/` -- training configs
- `checkpoints/` -- saved model checkpoints (or pointer to Lambda path)
- `results/` -- RTF, MOS, WER evaluation results
- `README.md` -- update with your results

## Key Paths on Lambda H100

- Dataset: `/home/ubuntu/datasets/LJSpeech-1.1/`
- Venv: `/home/ubuntu/voice_ai_venv`
- SDFT dir: `/home/ubuntu/voice_ai_sdft/`
