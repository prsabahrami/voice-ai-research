# SFT Baseline Training Results - COMPLETE

## Experiment Details

**Date**: 2026-03-19  
**Completion time**: 12:52 UTC  
**Hardware**: NVIDIA H100 80GB HBM3 (Lambda Cloud, 192.222.55.210)  
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
| Effective batch size | 32 |
| Learning rate | 2e-4 (cosine decay) |
| Warmup | 100 steps |
| Precision | bf16 |
| Max seq length | 512 |
| Total steps | 1,107 |

## Final Training Metrics

| Metric | Value |
|--------|-------|
| Training runtime | 2495.9s (41.6 min) |
| Avg train loss | 1.319 |
| Samples/second | 14.17 |
| Mean token accuracy | 0.783 |
| Total FLOPs | 4.12e16 |

## Eval Loss Trajectory

| Step | Epoch | Eval Loss | Notes |
|------|-------|-----------|-------|
| 100  | 0.27  | 1.4753    | End of warmup phase |
| 200  | 0.54  | 1.4272    | Steady improvement |
| 300  | 0.81  | 1.3920    | |
| 400  | 1.08  | 1.3825    | Epoch 1 complete |
| 500  | 1.36  | 1.3749    | |
| 600  | 1.63  | 1.3613    | |
| **700** | **1.90** | **1.3553** | **BEST - checkpoint selected for final_model/** |
| 800  | 2.17  | 1.3866    | Overfitting begins (train loss 1.10 but eval up) |
| 900  | 2.44  | 1.3846    | |
| 1000 | 2.71  | 1.3839    | |
| 1100 | 2.98  | 1.3833    | Near epoch 3 end |

Best checkpoint: step 700 (epoch 1.90), eval_loss=1.3553

## Checkpoint Locations (Lambda H100)

```
/home/ubuntu/voice_ai_sft_baseline/
├── final_model/          <- Best model (checkpoint-700 weights, load_best_model_at_end=True)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── chat_template.jinja
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── checkpoint-700/       <- Best checkpoint (eval_loss=1.3553)
├── checkpoint-1100/      <- Latest checkpoint
├── checkpoint-1107/      <- Final training checkpoint
├── training_metrics.json <- Full training history JSON
└── sft_config.json       <- Configuration used

```

## Key Finding: Overfitting After Epoch 2

Train loss dropped to 1.085 by epoch 3 but eval loss stabilized at ~1.383 after peak at step 700. This is expected for SFT on a small, single-speaker dataset. The final_model/ contains the best epoch-1.9 weights.

## Using the SFT Checkpoint

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model + best SFT LoRA adapter
base = AutoModelForCausalLM.from_pretrained(
    "unsloth/orpheus-3b-0.1-ft",
    torch_dtype="bfloat16",
    device_map="auto"
)
model = PeftModel.from_pretrained(
    base, 
    "/home/ubuntu/voice_ai_sft_baseline/final_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ubuntu/voice_ai_sft_baseline/final_model"
)
```

## Downstream Use

- **DPO**: Use `final_model/` as reference model (best eval_loss=1.3553, checkpoint-700 weights)
- **SDFT**: Use `final_model/` as starting point for iterative self-improvement rounds
