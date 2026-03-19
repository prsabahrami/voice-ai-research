# DPO Experiments

**Owner:** serious-inference-engineer
**Branch:** dpo-branch

## Objective

Apply Direct Preference Optimization (DPO) using the SFT checkpoint as the reference model.
Measure quality improvements (UTMOS, WER, speaker similarity) and inference latency (RTF).

## Getting Started

```bash
git checkout dpo-branch
cd dpo/
```

## Expected Deliverables

- `train_dpo.py` -- DPO training script (scaffolded at `/home/ubuntu/voice_ai_dpo/train_dpo.py`)
- `benchmark_rtf.py` -- real-time factor benchmark
- `config/` -- training configs
- `results/` -- RTF, MOS, WER evaluation results before/after DPO
- `README.md` -- update with your results

## Key Paths on Lambda H100

- Dataset: `/home/ubuntu/datasets/LJSpeech-1.1/`
- Venv: `/home/ubuntu/voice_ai_venv`
- DPO dir: `/home/ubuntu/voice_ai_dpo/`
- Scripts: `train_dpo.py`, `benchmark_rtf.py`
