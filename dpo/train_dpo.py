#!/usr/bin/env python3
"""
DPO Training for Voice AI (Orpheus-3B).
Loads the SFT LoRA checkpoint on top of the base model.
Uses text-level preference pairs from LJSpeech.
"""
import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LJSPEECH_DIR = "/home/ubuntu/datasets/LJSpeech-1.1"
BASE_MODEL = "unsloth/orpheus-3b-0.1-ft"


def corrupt_text(text: str, corruption_rate: float = 0.20) -> str:
    """Create a degraded version of text for rejected completions."""
    words = text.split()
    if len(words) <= 1:
        return text + " ..."
    corrupted = []
    for word in words:
        r = random.random()
        if r < corruption_rate / 2:
            continue  # drop word
        elif r < corruption_rate:
            corrupted.append(word[:max(1, len(word)//2)])  # truncate word
        else:
            corrupted.append(word)
    if not corrupted:
        return words[0]
    return " ".join(corrupted)


def load_ljspeech_pairs(data_dir: str, max_samples: int = None, seed: int = 42):
    """Load LJSpeech and create preference pairs for DPO."""
    random.seed(seed)
    metadata_file = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"LJSpeech metadata not found at {metadata_file}")

    pairs = []
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                text = parts[1].strip()
                if text and len(text) > 10:
                    prompt = f"Speak this text clearly: {text}"
                    chosen = f"{text}"
                    rejected = f"{corrupt_text(text)}"
                    pairs.append({
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

    if max_samples:
        random.shuffle(pairs)
        pairs = pairs[:max_samples]

    logger.info(f"Built {len(pairs)} preference pairs")
    return pairs


def load_model_with_adapter(model_path: str, base_model: str):
    """Load model, detecting if model_path is a LoRA adapter or full model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        logger.info(f"Detected LoRA adapter at {model_path}")
        logger.info(f"Loading base model: {base_model}")
        from peft import PeftModel
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, model_path)
        model.config.use_cache = False
        return model, tokenizer
    else:
        logger.info(f"Loading full model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model.config.use_cache = False
        return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=BASE_MODEL)
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--output_dir", default="/home/ubuntu/voice_ai_dpo/dpo_output")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=1500)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    logger.info(f"DPO v2 Training")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_with_adapter(args.model_path, args.base_model)

    # Load reference model (frozen base)
    logger.info(f"Loading reference model: {args.base_model}")
    from transformers import AutoModelForCausalLM
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Apply additional LoRA for DPO
    logger.info("Applying DPO LoRA")
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    # Only apply new LoRA if model doesn't already have one
    try:
        model = get_peft_model(model, lora_config)
    except Exception as e:
        logger.warning(f"Could not apply additional LoRA (model may already be PEFT): {e}")
    
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()

    # Build dataset
    logger.info("Building preference dataset")
    pairs = load_ljspeech_pairs(LJSPEECH_DIR, max_samples=args.max_samples)
    split_idx = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_idx]
    eval_pairs = pairs[split_idx:]

    from datasets import Dataset
    train_dataset = Dataset.from_list(train_pairs)
    eval_dataset = Dataset.from_list(eval_pairs)
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # DPO config
    from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    dpo_config = TRLDPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
        max_length=args.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training")
    trainer.train()

    logger.info("Saving DPO model")
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = trainer.state.log_history
    with open(os.path.join(args.output_dir, "dpo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Log final metrics
    loss_logs = [m for m in metrics if "loss" in m]
    if loss_logs:
        logger.info(f"Final loss: {loss_logs[-1].get('loss', 'N/A')}")

    logger.info(f"DPO complete. Model at {final_dir}")


if __name__ == "__main__":
    main()
