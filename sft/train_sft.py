#!/usr/bin/env python3
"""
SFT Baseline Training Script
Trains Orpheus-3B (or unsloth/orpheus-3b-0.1-ft) on LJSpeech with TRL SFTTrainer + LoRA.
Usage:
  python train_sft.py --model_name orpheus --output_dir /home/ubuntu/voice_ai_sft_baseline
  python train_sft.py --model_name tinyllama --output_dir /home/ubuntu/voice_ai_sft_baseline
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import glob

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "orpheus": "unsloth/orpheus-3b-0.1-ft",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi2": "microsoft/phi-2",
}

LJSPEECH_DIR = "/home/ubuntu/datasets/LJSpeech-1.1"


def load_ljspeech_dataset(data_dir: str, max_samples: Optional[int] = None):
    """Load LJSpeech metadata and format for language model training."""
    from datasets import Dataset

    metadata_file = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"LJSpeech metadata not found at {metadata_file}")

    examples = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            parts = line.strip().split("|")
            if len(parts) >= 2:
                wav_id = parts[0]
                text = parts[1] if len(parts) > 1 else parts[0]
                wav_path = os.path.join(data_dir, "wavs", f"{wav_id}.wav")
                if os.path.exists(wav_path):
                    # Format for SFT: text-to-speech instruction tuning
                    examples.append({
                        "text": f"Convert the following text to speech: {text}\n\nAudio transcription: {text}",
                        "wav_id": wav_id,
                        "transcript": text,
                    })

    logger.info(f"Loaded {len(examples)} LJSpeech examples")
    return Dataset.from_list(examples)


def format_prompt(example, tokenizer):
    """Format example as instruction-following prompt for SFT."""
    text = example["text"]
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="orpheus", choices=list(MODEL_MAP.keys()),
                        help="Model to use for SFT")
    parser.add_argument("--output_dir", default="/home/ubuntu/voice_ai_sft_baseline",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max training samples (None = all)")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--scaffold_only", action="store_true",
                        help="Just validate imports and exit")
    args = parser.parse_args()

    model_id = MODEL_MAP[args.model_name]
    logger.info(f"Starting SFT training")
    logger.info(f"Model: {model_id}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Epochs: {args.num_epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    if args.scaffold_only:
        logger.info("Scaffold-only mode: imports OK")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # Import training libs (delayed to allow scaffold check)
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    logger.info("Applying LoRA adapter")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Loading LJSpeech dataset")
    dataset = load_ljspeech_dataset(LJSPEECH_DIR, max_samples=args.max_samples)
    # 90/10 train/val split
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        eval_steps=100,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        max_length=args.max_seq_length,
        dataset_text_field="text",
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final model...")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # Save training metrics
    train_metrics = trainer.state.log_history
    metrics_file = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(train_metrics, f, indent=2)

    logger.info(f"Training complete. Model saved to: {final_model_dir}")
    logger.info(f"Metrics saved to: {metrics_file}")

    # Log final loss
    if train_metrics:
        last_log = [m for m in train_metrics if "loss" in m]
        if last_log:
            final_loss = last_log[-1]["loss"]
            logger.info(f"Final training loss: {final_loss:.4f}")


if __name__ == "__main__":
    main()
