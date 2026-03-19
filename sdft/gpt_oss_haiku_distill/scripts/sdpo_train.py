#!/usr/bin/env python3
"""
SDPO: Style-Distillation Preference Optimization on GPT-OSS
DPO-based training using Haiku-style responses as "chosen" vs baseline GPT-OSS as "rejected".

Usage:
    python scripts/sdpo_train.py --config configs/sdpo_gpt_oss_20b.yaml
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

import anthropic
import tinker
from tinker import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---- Haiku response generation ---------------------------------------------

def generate_haiku_response(prompt: str, client: anthropic.Anthropic) -> str:
    """Generate a Haiku-style response using Claude Haiku."""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        logger.warning(f"Haiku generation failed: {e}")
        return None


def generate_baseline_response(prompt: str, client: anthropic.Anthropic) -> str:
    """Generate a baseline response using Claude Haiku without style guidance."""
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system="Give a brief, plain, factual response without conversational warmth.",
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        logger.warning(f"Baseline generation failed: {e}")
        return None


# ---- Preference dataset construction ---------------------------------------

CONVERSATION_PROMPTS = [
    "What's the best way to learn a new programming language?",
    "Can you help me understand why my Python code is slow?",
    "What makes a good conversation?",
    "How do I deal with imposter syndrome as a software developer?",
    "What's the difference between machine learning and deep learning?",
    "Can you help me write a cover letter for a software engineering job?",
    "Why is sleep so important?",
    "How can I get better at debugging code?",
    "What are some good habits for remote work?",
    "Can you explain recursion in simple terms?",
    "What's a healthy work-life balance in tech?",
    "How do I learn to enjoy reading technical papers?",
    "What makes a great engineering team?",
    "How should I approach a technical interview?",
    "Can you explain what a transformer model is?",
    "What's the intuition behind gradient descent?",
    "How do I become a better communicator at work?",
    "What are the most important skills for a data scientist?",
    "Can you walk me through system design concepts?",
    "How do I stay motivated when working on long projects?",
    "What's the best approach to code review?",
    "How do open-source communities work?",
    "Can you explain Docker and containers simply?",
    "What should I know about database design?",
    "How do I handle disagreements with teammates professionally?",
    "What makes a codebase maintainable?",
    "Can you explain async programming?",
    "How do I evaluate if a ML model is actually good?",
    "What's the best way to document code?",
    "How should a developer approach learning new frameworks?",
]


def build_preference_dataset(n_pairs: int, anthropic_client, cache_path: Path):
    """Build preference pairs: chosen=Haiku-style, rejected=baseline."""
    if cache_path.exists():
        logger.info(f"Loading cached preference dataset from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    logger.info(f"Building preference dataset with {n_pairs} pairs...")
    pairs = []

    prompts = (CONVERSATION_PROMPTS * (n_pairs // len(CONVERSATION_PROMPTS) + 1))[:n_pairs]

    for i, prompt in enumerate(prompts):
        chosen = generate_haiku_response(prompt, anthropic_client)
        rejected = generate_baseline_response(prompt, anthropic_client)

        if chosen and rejected:
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
            logger.info(f"  Pair {i+1}/{n_pairs}: prompt={prompt[:40]!r}...")

        time.sleep(0.1)  # rate limiting

    cache_path.write_text(json.dumps(pairs, indent=2))
    logger.info(f"Preference dataset saved to {cache_path} ({len(pairs)} pairs)")
    return pairs


def encode_dpo_pair(pair: dict, tokenizer) -> tuple:
    """Encode a preference pair into DPO-format token sequences."""
    messages_chosen = [
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["chosen"]},
    ]
    messages_rejected = [
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["rejected"]},
    ]

    tokens_chosen = tokenizer.apply_chat_template(messages_chosen, add_generation_prompt=False)
    tokens_rejected = tokenizer.apply_chat_template(messages_rejected, add_generation_prompt=False)

    # Loss weights: 1 on assistant tokens
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True
    )
    n_prompt = len(prompt_only)

    lw_chosen = [0.0] * n_prompt + [1.0] * (len(tokens_chosen) - n_prompt)
    lw_rejected = [0.0] * n_prompt + [1.0] * (len(tokens_rejected) - n_prompt)

    datum_chosen = types.Datum(tokens=tokens_chosen, loss_weights=lw_chosen)
    datum_rejected = types.Datum(tokens=tokens_rejected, loss_weights=lw_rejected)

    return datum_chosen, datum_rejected


# ---- DPO training loop ------------------------------------------------------

def run_sdpo(config: dict, results_dir: Path):
    """Run SDPO (DPO) training loop with Tinker API."""
    model_name = config.get("model", "openai/gpt-oss-20b")
    rank = config.get("lora_rank", 16)
    n_steps = config.get("n_steps", 300)
    batch_size = config.get("batch_size", 4)
    lr = config.get("lr", 1e-4)
    beta = config.get("dpo_beta", 0.1)
    eval_every = config.get("eval_every", 50)
    n_pairs = config.get("n_preference_pairs", 200)
    run_name = config.get("run_name", f"sdpo_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logger.info(f"Starting SDPO run: {run_name}")
    logger.info(f"Model: {model_name}, rank={rank}, steps={n_steps}, beta={beta}")

    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Build preference dataset
    data_cache = results_dir / f"preference_pairs_{n_pairs}.json"
    pairs = build_preference_dataset(n_pairs, anthropic_client, data_cache)
    n_train = int(0.9 * len(pairs))
    train_pairs, eval_pairs = pairs[:n_train], pairs[n_train:]

    # Tinker client setup
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=model_name,
        rank=rank,
    )
    tokenizer = training_client.get_tokenizer()

    # Encode preference pairs
    logger.info("Encoding preference pairs...")
    train_encoded = [encode_dpo_pair(p, tokenizer) for p in train_pairs]
    eval_encoded = [encode_dpo_pair(p, tokenizer) for p in eval_pairs]
    train_encoded = [(c, r) for c, r in train_encoded if c and r]

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

    # DPO training loop
    logger.info("Beginning SDPO training loop...")
    step = 0
    train_loss = float("nan")

    while step < n_steps:
        batch = random.choices(train_encoded, k=batch_size)

        # DPO loss: minimize NLL on chosen - beta * NLL on rejected
        chosen_batch = [c for c, r in batch]
        rejected_batch = [r for c, r in batch]

        # Forward-backward on chosen (minimize)
        fb_chosen = training_client.forward_backward(
            data=chosen_batch,
            loss=types.Loss.NLL,
        )
        # Forward-backward on rejected (maximize = minimize negative)
        # Use negative loss weight trick via DPO loss if supported, else approximate
        fb_rejected = training_client.forward_backward(
            data=rejected_batch,
            loss=types.Loss.NLL,
            loss_scale=-beta,  # Scale rejected gradient negatively (DPO approximation)
        )

        chosen_result = fb_chosen.result()
        rejected_result = fb_rejected.result()

        opt_future = training_client.optim_step(lr=lr)
        opt_future.result()

        chosen_loss = chosen_result.loss if chosen_result.loss else float("nan")
        rejected_loss = rejected_result.loss if rejected_result.loss else float("nan")
        train_loss = chosen_loss - beta * rejected_loss

        step += 1
        step_metric = {
            "step": step,
            "train_loss": float(train_loss),
            "chosen_loss": float(chosen_loss),
            "rejected_loss": float(rejected_loss),
            "timestamp": time.time(),
        }
        metrics["steps"].append(step_metric)
        logger.info(f"Step {step}/{n_steps} | loss={train_loss:.4f} chosen={chosen_loss:.4f} rej={rejected_loss:.4f}")

        if step % eval_every == 0:
            logger.info(f"Eval at step {step}...")
            # Sample from model and compare to Haiku
            test_prompt = random.choice(CONVERSATION_PROMPTS)
            prompt_msgs = [{"role": "user", "content": test_prompt}]
            prompt_tokens = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
            sampling_client = training_client.save_weights_and_get_sampling_client(name=f"sdpo_sampler_step{step}")
            sample_out = sampling_client.sample(tokens=prompt_tokens, max_tokens=150)
            sample_text = tokenizer.decode(sample_out.tokens)

            haiku_ref = generate_haiku_response(test_prompt, anthropic_client)

            eval_metric = {
                "step": step,
                "prompt": test_prompt,
                "model_output": sample_text[:300],
                "haiku_reference": haiku_ref[:300] if haiku_ref else None,
                "timestamp": time.time(),
            }
            metrics["eval_steps"].append(eval_metric)
            logger.info(f"  Model: {sample_text[:100]!r}")
            logger.info(f"  Haiku: {haiku_ref[:100]!r}" if haiku_ref else "  Haiku: N/A")

            save_metrics()
            training_client.save_state(name=f"{run_name}_step{step}").result()

    metrics["end_time"] = datetime.now().isoformat()
    metrics["final_train_loss"] = float(train_loss)
    save_metrics()
    training_client.save_state(name=f"{run_name}_final").result()
    logger.info(f"SDPO run complete: {run_name}")

    return metrics


# ---- Entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--results-dir", type=str, default="results/sdpo")
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
            "batch_size": 4,
            "lr": 1e-4,
            "dpo_beta": 0.1,
            "eval_every": 50,
            "n_preference_pairs": 200,
        }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_sdpo(config, results_dir)


if __name__ == "__main__":
    main()
