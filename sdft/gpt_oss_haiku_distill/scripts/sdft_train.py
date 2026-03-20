#!/usr/bin/env python3
"""
SDFT: Style-Distillation Fine-Tuning on GPT-OSS
Persistent training session using the Tinker API.

Usage:
    python scripts/sdft_train.py --config configs/sdft_gpt_oss_20b.yaml
"""

import os
import sys
import json
import time
import random
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

import tinker
from tinker import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ---- Data utilities --------------------------------------------------------

def build_transcript_dataset(tokenizer, n_train=2000, n_eval=200):
    """
    Build a transcript-style conversational dataset.
    Uses HuggingFace datasets (UltraChat subset) as base, optionally augmented
    with Haiku-generated completions via Anthropic API.
    """
    from datasets import load_dataset

    logger.info("Loading UltraChat conversation dataset...")
    try:
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
        examples = []
        for i, ex in enumerate(ds):
            if i >= n_train + n_eval:
                break
            examples.append(ex)
        logger.info(f"Loaded {len(examples)} raw examples")
    except Exception as e:
        logger.warning(f"Could not load UltraChat: {e}. Using synthetic data.")
        examples = _build_synthetic_dataset(n_train + n_eval)

    train_data, eval_data = [], []
    for i, ex in enumerate(examples):
        datum = _process_example(ex, tokenizer)
        if datum is None:
            continue
        if i < n_train:
            train_data.append(datum)
        else:
            eval_data.append(datum)

    logger.info(f"Dataset ready: {len(train_data)} train, {len(eval_data)} eval")
    return train_data, eval_data


def _process_example(ex, tokenizer):
    """Convert a conversational example to a Tinker Datum."""
    try:
        messages = ex.get("messages", [])
        if not messages:
            return None

        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
        # Loss weights: 1 for assistant tokens, 0 for system/user
        loss_weights = _get_assistant_loss_weights(messages, tokens, tokenizer)

        return types.Datum(
            tokens=tokens,
            loss_weights=loss_weights,
        )
    except Exception:
        return None


def _get_assistant_loss_weights(messages, tokens, tokenizer):
    """Binary loss weights - 1 for assistant response tokens, 0 elsewhere."""
    weights = [0.0] * len(tokens)

    # Re-render each prefix to find boundary positions
    prefix_tokens_len = 0
    for msg in messages:
        prefix = messages[: messages.index(msg)]
        prefix_tok = tokenizer.apply_chat_template(prefix, add_generation_prompt=(msg["role"] == "assistant"))
        if msg["role"] == "assistant":
            start = len(prefix_tok)
            end = len(tokenizer.apply_chat_template(messages[: messages.index(msg) + 1]))
            for j in range(start, min(end, len(weights))):
                weights[j] = 1.0
        prefix_tokens_len = len(tokenizer.apply_chat_template(messages[: messages.index(msg) + 1]))

    return weights


def _build_synthetic_dataset(n):
    """Fallback synthetic dataset for testing."""
    topics = [
        "machine learning", "climate change", "cooking pasta", "quantum computing",
        "ancient history", "music theory", "software engineering", "poetry",
        "economics", "travel tips"
    ]
    examples = []
    for i in range(n):
        topic = random.choice(topics)
        examples.append({
            "messages": [
                {"role": "user", "content": f"Can you explain {topic} to me?"},
                {"role": "assistant", "content": f"Sure! {topic.capitalize()} is a fascinating area. Let me walk you through the key concepts in a clear and engaging way..."}
            ]
        })
    return examples


# ---- Training loop ---------------------------------------------------------

def run_sdft(config: dict, results_dir: Path):
    """Run SDFT training loop with Tinker API."""
    model_name = config.get("model", "openai/gpt-oss-20b")
    rank = config.get("lora_rank", 16)
    n_steps = config.get("n_steps", 500)
    batch_size = config.get("batch_size", 8)
    grad_accum = config.get("grad_accum", 4)
    lr = config.get("lr", 2e-4)
    eval_every = config.get("eval_every", 50)
    n_train = config.get("n_train_examples", 2000)
    n_eval = config.get("n_eval_examples", 200)
    run_name = config.get("run_name", f"sdft_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logger.info(f"Starting SDFT run: {run_name}")
    logger.info(f"Model: {model_name}, rank={rank}, steps={n_steps}, lr={lr}")

    # Tinker client setup
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=rank,
    )
    tokenizer = training_client.get_tokenizer()

    # Dataset
    train_data, eval_data = build_transcript_dataset(tokenizer, n_train, n_eval)
    logger.info(f"Dataset: {len(train_data)} train, {len(eval_data)} eval")

    # Metrics tracking
    metrics = {
        "run_name": run_name,
        "model": model_name,
        "config": config,
        "steps": [],
        "eval_steps": [],
        "start_time": datetime.now().isoformat(),
    }
    metrics_path = results_dir / f"{run_name}_metrics.json"

    def save_metrics():
        metrics_path.write_text(json.dumps(metrics, indent=2))

    # Training loop
    futures = []
    step = 0

    logger.info("Beginning SDFT training loop...")
    while step < n_steps:
        # Sample a mini-batch
        batch = random.choices(train_data, k=batch_size)
        batch_datums = [d for d in batch if d is not None]

        if not batch_datums:
            continue

        # Forward-backward pass
        fb_future = training_client.forward_backward(
            data=batch_datums,
            loss=types.Loss.NLL,
        )
        futures.append(fb_future)

        # Gradient accumulation
        if len(futures) >= grad_accum:
            # Optimizer step
            fb_results = [f.result() for f in futures]
            train_loss = np.mean([r.loss for r in fb_results if r.loss is not None])
            futures = []

            opt_future = training_client.optim_step(lr=lr)
            opt_future.result()

            step += 1
            step_metric = {"step": step, "train_loss": float(train_loss), "timestamp": time.time()}
            metrics["steps"].append(step_metric)
            logger.info(f"Step {step}/{n_steps} | train_loss={train_loss:.4f}")

            # Periodic eval
            if step % eval_every == 0:
                logger.info(f"Running eval at step {step}...")
                eval_batch = random.choices(eval_data, k=min(32, len(eval_data)))
                eval_datums = [d for d in eval_batch if d is not None]
                eval_future = training_client.forward_backward(
                    data=eval_datums,
                    loss=types.Loss.NLL,
                )
                eval_result = eval_future.result()
                eval_loss = eval_result.loss if eval_result.loss else float("nan")
                logger.info(f"  eval_loss={eval_loss:.4f}")

                # Sample from model
                sample_prompt = [{"role": "user", "content": "Tell me about the importance of clean code."}]
                sample_tokens = tokenizer.apply_chat_template(sample_prompt, add_generation_prompt=True)
                sampling_client = training_client.save_weights_and_get_sampling_client(name=f"eval_sampler_step{step}")
                sample_out = sampling_client.sample(tokens=sample_tokens, max_tokens=150)
                sample_text = tokenizer.decode(sample_out.tokens)

                eval_metric = {
                    "step": step,
                    "eval_loss": float(eval_loss),
                    "sample": sample_text[:300],
                    "timestamp": time.time(),
                }
                metrics["eval_steps"].append(eval_metric)
                logger.info(f"  sample (first 150 chars): {sample_text[:150]!r}")

                save_metrics()

                # Save checkpoint
                ckpt_future = training_client.save_state(name=f"{run_name}_step{step}")
                ckpt_future.result()
                logger.info(f"  Checkpoint saved: {run_name}_step{step}")

    # Final save
    metrics["end_time"] = datetime.now().isoformat()
    metrics["final_train_loss"] = float(train_loss) if step > 0 else None
    save_metrics()

    final_ckpt = training_client.save_state(name=f"{run_name}_final")
    final_ckpt.result()
    logger.info(f"SDFT run complete: {run_name}")
    logger.info(f"Metrics saved to: {metrics_path}")

    return metrics


# ---- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDFT training on GPT-OSS via Tinker")
    parser.add_argument("--config", type=str, help="YAML config path")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--results-dir", type=str, default="results/sdft")
    args = parser.parse_args()

    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "model": args.model,
            "n_steps": args.steps,
            "lora_rank": 16,
            "batch_size": 8,
            "grad_accum": 4,
            "lr": 2e-4,
            "eval_every": 50,
            "n_train_examples": 2000,
            "n_eval_examples": 200,
            "run_name": f"sdft_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_sdft(config, results_dir)


if __name__ == "__main__":
    main()
