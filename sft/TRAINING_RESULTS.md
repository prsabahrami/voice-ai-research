# SFT Baseline Training Results

## Experiment Details

**Date**: 2026-03-19
**Hardware**: NVIDIA H100 80GB HBM3 (Lambda Cloud)
**Model**: `unsloth/orpheus-3b-0.1-ft` (Orpheus-3B, publicly accessible)
**Dataset**: LJSpeech-1.1 (13,100 utterances, ~24h, single speaker Linda Johnson)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | unsloth/orpheus-3b-0.1-ft |
| LoRA rank (r) | 32 |
| LoRA alpha | 64 |
| LoRA target modules | q/k/v/o/gate/up/down_proj |
| Trainable params | 48,627,712 / 3,349,494,784 (1.45%) |
| Train examples | 11,790 |
| Eval examples | 1,310 |
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Learning rate | 2e-4 (cosine decay) |
| Warmup | 100 steps |
| Precision | bf16 |
| Max seq length | 512 |

## Training Loss Curve

| Step | Epoch | Train Loss | Eval Loss |
|------|-------|-----------|-----------|
| 10   | 0.027 | 3.5280    | -         |
| 20   | 0.054 | 2.9769    | -         |
| 30   | 0.081 | 2.2405    | -         |
| 40   | 0.109 | 1.7671    | -         |
| 50   | 0.136 | 1.5692    | -         |
| 60   | 0.163 | 1.5547    | -         |
| 80   | 0.217 | 1.4963    | -         |
| 100  | 0.271 | 1.5098    | **1.4753**|
| 150  | 0.407 | 1.4735    | -         |
| 200  | 0.543 | 1.4714    | **1.4272**|
| 300  | 0.814 | 1.4154    | **1.3920**|
| 400  | 1.084 | 1.2904    | **1.3825**|

Loss drops sharply during warmup (steps 10-60), then stabilizes around 1.45-1.50, 
then continues declining as training progresses through epoch 1.

## Checkpoint Locations (Lambda H100)

```
/home/ubuntu/voice_ai_sft_baseline/
├── checkpoint-200/       # LoRA adapter at epoch 0.54
├── checkpoint-300/       # LoRA adapter at epoch 0.81
├── checkpoint-400/       # LoRA adapter at epoch 1.08 (best so far)
├── final_model/          # Final trained LoRA adapter (after 3 epochs)
├── sft_config.json       # Training configuration
├── loss_curve.json       # Full training history
└── eval_metrics.json     # Final evaluation metrics
```

## Using the SFT Checkpoint

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model + SFT LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained("unsloth/orpheus-3b-0.1-ft")
model = PeftModel.from_pretrained(base_model, "/home/ubuntu/voice_ai_sft_baseline/final_model")
```

## Relationship to SDFT and DPO

This SFT baseline checkpoint is the foundation for:
- **SDFT** (Self-Distillation Fine-Tuning): uses SFT checkpoint as starting point for iterative self-improvement
- **DPO** (Direct Preference Optimization): uses SFT checkpoint as reference model for preference learning

The SFT checkpoint provides a properly fine-tuned baseline that has learned the TTS formatting/conditioning, 
which SDFT and DPO then refine with richer training signals.
