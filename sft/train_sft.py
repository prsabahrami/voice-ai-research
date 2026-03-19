#!/usr/bin/env python3
"""
SFT Baseline Training Script for Orpheus-3B on LJSpeech
Model: unsloth/orpheus-3b-0.1-ft (publicly accessible, same architecture as canopylabs/orpheus-3b-0.1-pretrained)
Uses TRL SFTTrainer with LoRA/PEFT for memory efficiency on H100 80GB

Usage:
  source /home/ubuntu/voice_ai_venv/bin/activate
  python sft/train_sft.py --output_dir /home/ubuntu/voice_ai_sft_baseline

  # Use TinyLlama for pipeline testing (no HF token needed):
  python sft/train_sft.py --model_name tinyllama --output_dir /home/ubuntu/voice_ai_sft_baseline
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================
# Model configs
# ============================================================

MODEL_CONFIGS = {
    "orpheus": {
        "id": "unsloth/orpheus-3b-0.1-ft",
        "description": "Orpheus-3B (unsloth fork, publicly accessible, no HF gating)",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
        "batch_size": 4,
        "max_seq_length": 512,
    },
    "tinyllama": {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "description": "TinyLlama 1.1B (proxy model for pipeline validation)",
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
        "batch_size": 8,
        "max_seq_length": 512,
    },
}

DATASET_PATH = "/home/ubuntu/datasets/LJSpeech-1.1"
OUTPUT_DIR = "/home/ubuntu/voice_ai_sft_baseline"
HF_CACHE_DIR = "/home/ubuntu/.cache/huggingface"

GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 100
LOGGING_STEPS = 25
SAVE_STEPS = 250
EVAL_STEPS = 250
EVAL_SPLIT = 0.05

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


def format_tts_example(text: str, model_key: str) -> str:
    """Format example for TTS-style SFT training."""
    text = text.strip()
    if model_key == "orpheus":
        # Orpheus/LLaMA chat format for TTS
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"<available_voice>tara</available_voice><|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Convert to speech: {text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
        )
    else:
        # Generic chat format
        return (
            f"<|system|>\nYou are a TTS assistant for voice synthesis training.</s>\n"
            f"<|user|>\nPrepare speech: {text}</s>\n"
            f"<|assistant|>\n{text}</s>"
        )


def load_ljspeech(metadata_path: str, model_key: str,
                  max_samples: Optional[int], eval_split: float):
    """Load and format LJSpeech dataset."""
    from datasets import Dataset
    
    logger.info(f"Loading LJSpeech from {metadata_path}")
    records = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                text = parts[2].strip()
                if text:
                    records.append({"raw": text})
    
    logger.info(f"Loaded {len(records)} LJSpeech records")
    if max_samples:
        records = records[:max_samples]
    
    def format_fn(ex):
        return {"text": format_tts_example(ex["raw"], model_key)}
    
    ds = Dataset.from_list(records)
    ds = ds.map(format_fn, remove_columns=["raw"])
    split = ds.train_test_split(test_size=eval_split, seed=42)
    logger.info(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split["train"], split["test"]


def main():
    parser = argparse.ArgumentParser(description="SFT baseline for Orpheus-3B on LJSpeech")
    parser.add_argument("--model_name", default="orpheus",
                        choices=["orpheus", "tinyllama"])
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_cfg = MODEL_CONFIGS[args.model_name]
    model_id = model_cfg["id"]
    
    logger.info(f"Model: {model_id}")
    logger.info(f"Description: {model_cfg['description']}")
    
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=LORA_ALPHA,
        target_modules=model_cfg["lora_target_modules"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Dataset
    train_ds, eval_ds = load_ljspeech(
        os.path.join(args.dataset_path, "metadata.csv"),
        args.model_name, args.max_samples, EVAL_SPLIT
    )
    
    # Model
    logger.info(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    logger.info(f"Loading model {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, cache_dir=HF_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    logger.info(f"Model loaded. Params: {model.num_parameters():,}")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # SFT config
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=model_cfg["batch_size"],
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        max_seq_length=model_cfg["max_seq_length"],
        dataset_text_field="text",
        report_to="none",
        dataloader_num_workers=4,
        seed=42,
        logging_first_step=True,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    
    # Save training config
    config = {
        "model_key": args.model_name,
        "model_id": model_id,
        "description": model_cfg["description"],
        "dataset": "LJSpeech-1.1",
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "lora_r": args.lora_r,
        "lora_alpha": LORA_ALPHA,
        "num_epochs": args.num_epochs,
        "batch_size": model_cfg["batch_size"],
        "grad_accum": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch": model_cfg["batch_size"] * GRADIENT_ACCUMULATION_STEPS,
        "lr": LEARNING_RATE,
        "bf16": True,
        "max_seq_len": model_cfg["max_seq_length"],
    }
    with open(os.path.join(args.output_dir, "sft_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    logger.info("=" * 60)
    logger.info("STARTING SFT TRAINING")
    logger.info("=" * 60)
    
    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    logger.info(f"Training complete in {(time.time()-t0)/3600:.2f}h")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Saved to {final_path}")
    
    metrics = trainer.evaluate()
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "loss_curve.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    logger.info(f"Final eval: {metrics}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
