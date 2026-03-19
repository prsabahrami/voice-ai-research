# SFT Baseline Experiments

## Overview
Supervised Fine-Tuning (SFT) baseline for voice AI using Orpheus-3B on LJSpeech.

## Model
- **Primary**: `unsloth/orpheus-3b-0.1-ft` (Orpheus-3B, publicly accessible, LLaMA-based TTS)
- **Proxy**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (for pipeline validation)

## Dataset
- LJSpeech-1.1: 13,100 utterances, ~24h, single speaker (Linda Johnson)
- Path on Lambda: `/home/ubuntu/datasets/LJSpeech-1.1/`

## Training Setup
- Hardware: NVIDIA H100 80GB
- Method: LoRA/PEFT (r=32, alpha=64) via TRL SFTTrainer
- Batch: 4 * 4 grad_accum = effective batch 16
- LR: 2e-4 cosine with 100 warmup steps
- Epochs: 3
- Precision: bf16

## Usage

```bash
# On Lambda H100:
source /home/ubuntu/voice_ai_venv/bin/activate
python sft/train_sft.py \
  --model_name orpheus \
  --output_dir /home/ubuntu/voice_ai_sft_baseline

# Watch training:
tmux attach -t sft_training
```

## Outputs
- `/home/ubuntu/voice_ai_sft_baseline/sft_config.json` - training config
- `/home/ubuntu/voice_ai_sft_baseline/loss_curve.json` - training loss curve
- `/home/ubuntu/voice_ai_sft_baseline/eval_metrics.json` - evaluation metrics
- `/home/ubuntu/voice_ai_sft_baseline/final_model/` - LoRA adapter weights
- `/home/ubuntu/voice_ai_sft_baseline/checkpoint-*/` - intermediate checkpoints

## Checkpoint for SDFT/DPO
The SFT checkpoint at `/home/ubuntu/voice_ai_sft_baseline/final_model/` is the baseline
that SDFT and DPO experiments build upon.
