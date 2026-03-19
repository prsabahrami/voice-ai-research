#!/usr/bin/env python3
"""
Haiku Distillation Method A: SFT on Haiku-generated outputs.
Train GPT-OSS to imitate Claude Haiku's conversational style via supervised fine-tuning.

Haiku style characteristics:
- Warm, conversational tone
- Clear structure with natural flow
- Helpful without being overly formal
- Concise but complete
- Uses natural transitions and connectives
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

import anthropic
import tinker
from tinker import types

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HAIKU_STYLE_SYSTEM_PROMPT = """You are Claude, made by Anthropic. You are helpful, harmless, and honest. 
Your responses are warm and conversational - you engage naturally with the person you're talking to.
You explain things clearly without being overly formal. You use a friendly tone that feels like talking 
to a knowledgeable friend."""

# Wide variety of conversational prompts for distillation
DISTILLATION_PROMPTS = [
    # Technical help
    "Can you help me debug this Python error: TypeError: 'NoneType' object is not subscriptable?",
    "What's the difference between a list and a tuple in Python?",
    "How does garbage collection work in modern programming languages?",
    "Can you explain what REST APIs are and why they're useful?",
    "I'm confused about when to use async/await in JavaScript. Can you help?",
    "What's the best way to structure a machine learning project?",
    # Conversational / personal
    "I've been feeling overwhelmed at work lately. Any advice?",
    "What are some good techniques for staying focused while working from home?",
    "I want to learn something new this year. What would you recommend?",
    "How do you approach a big project that feels intimidating?",
    # Explanatory
    "Can you explain what inflation actually means in simple terms?",
    "How does the immune system work?",
    "What is consciousness? Can AI be conscious?",
    "Why do we dream?",
    "How does the internet actually work at a fundamental level?",
    # Creative
    "Can you help me come up with a name for my startup that does sustainable packaging?",
    "I need to write a thank-you note to a mentor. Any suggestions?",
    "Help me brainstorm ideas for a weekend project involving woodworking.",
    # Meta / philosophical  
    "What makes a good teacher?",
    "What do you think is the most important skill to have today?",
    "Why is reading important even when we have the internet?",
    # Multi-turn follow-ups (simulated)
    "Earlier you mentioned gradient descent. Can you go deeper on momentum?",
    "What about Adam optimizer compared to SGD?",
    "Going back to what you said about REST - what about GraphQL instead?",
]


def generate_haiku_pairs(n_pairs: int, cache_path: Path):
    """Generate (prompt, haiku_response) pairs using Claude Haiku."""
    if cache_path.exists():
        logger.info(f"Loading cached Haiku pairs from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    pairs = []

    prompts = (DISTILLATION_PROMPTS * (n_pairs // len(DISTILLATION_PROMPTS) + 1))[:n_pairs]

    logger.info(f"Generating {n_pairs} Haiku-style responses...")
    for i, prompt in enumerate(prompts):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=400,
                system=HAIKU_STYLE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text
            pairs.append({"prompt": prompt, "response": text})
            logger.info(f"  {i+1}/{n_pairs}: {prompt[:50]!r} -> {text[:60]!r}...")
            time.sleep(0.05)
        except Exception as e:
            logger.warning(f"  Failed on {i}: {e}")

    cache_path.write_text(json.dumps(pairs, indent=2))
    logger.info(f"Saved {len(pairs)} pairs to {cache_path}")
    return pairs


def build_sft_datum(pair: dict, tokenizer) -> types.Datum:
    """Convert a (prompt, response) pair to a Tinker SFT datum."""
    messages = [
        {"role": "user", "content": pair["prompt"]},
        {"role": "assistant", "content": pair["response"]},
    ]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)

    # Loss weights: 1 on assistant response tokens only
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True
    )
    n_prompt = len(prompt_tokens)
    loss_weights = [0.0] * n_prompt + [1.0] * max(0, len(tokens) - n_prompt)

    return types.Datum(tokens=tokens, loss_weights=loss_weights)


def run_haiku_sft_distill(config: dict, results_dir: Path):
    """Run Haiku SFT distillation on GPT-OSS."""
    model_name = config.get("model", "openai/gpt-oss-20b")
    rank = config.get("lora_rank", 16)
    n_steps = config.get("n_steps", 300)
    batch_size = config.get("batch_size", 8)
    lr = config.get("lr", 2e-4)
    eval_every = config.get("eval_every", 50)
    n_pairs = config.get("n_pairs", 500)
    run_name = config.get("run_name", f"haiku_sft_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logger.info(f"=== Haiku SFT Distillation: {run_name} ===")

    # Generate Haiku training data
    cache_path = results_dir / f"haiku_pairs_{n_pairs}.json"
    pairs = generate_haiku_pairs(n_pairs, cache_path)
    n_train = int(0.85 * len(pairs))
    train_pairs, eval_pairs = pairs[:n_train], pairs[n_train:]

    # Tinker setup
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=model_name, rank=rank)
    tokenizer = training_client.get_tokenizer()

    # Encode data
    train_data = [build_sft_datum(p, tokenizer) for p in train_pairs]
    eval_data = [build_sft_datum(p, tokenizer) for p in eval_pairs]
    train_data = [d for d in train_data if d is not None]
    eval_data = [d for d in eval_data if d is not None]

    metrics = {
        "run_name": run_name, "model": model_name, "config": config,
        "method": "sft_on_haiku_outputs",
        "steps": [], "eval_steps": [],
        "start_time": datetime.now().isoformat(),
    }
    metrics_path = results_dir / f"{run_name}_metrics.json"

    # Training loop
    step = 0
    train_loss = float("nan")
    grad_accum = config.get("grad_accum", 4)
    futures = []

    logger.info(f"Training: {len(train_data)} examples, {n_steps} steps")
    while step < n_steps:
        batch = random.choices(train_data, k=batch_size)
        fb = training_client.forward_backward(data=batch, loss=types.Loss.NLL)
        futures.append(fb)

        if len(futures) >= grad_accum:
            results = [f.result() for f in futures]
            train_loss = np.mean([r.loss for r in results if r.loss is not None])
            futures = []

            training_client.optim_step(lr=lr).result()
            step += 1

            metrics["steps"].append({"step": step, "train_loss": float(train_loss), "t": time.time()})
            logger.info(f"Step {step}/{n_steps} | train_loss={train_loss:.4f}")

            if step % eval_every == 0:
                # Eval: sample from model, compare to Haiku reference
                test_pair = random.choice(eval_pairs)
                prompt_tokens = tokenizer.apply_chat_template(
                    [{"role": "user", "content": test_pair["prompt"]}], add_generation_prompt=True
                )
                sampler = training_client.save_weights_and_get_sampling_client(name=f"haiku_sft_sampler_step{step}")
                out = sampler.sample(tokens=prompt_tokens, max_tokens=200)
                model_text = tokenizer.decode(out.tokens)

                metrics["eval_steps"].append({
                    "step": step,
                    "prompt": test_pair["prompt"],
                    "model_output": model_text[:400],
                    "haiku_reference": test_pair["response"][:400],
                    "t": time.time(),
                })
                logger.info(f"  Prompt: {test_pair['prompt'][:60]!r}")
                logger.info(f"  GPT-OSS (distilled): {model_text[:120]!r}")
                logger.info(f"  Haiku target:        {test_pair['response'][:120]!r}")

                metrics_path.write_text(json.dumps(metrics, indent=2))
                training_client.save_state(name=f"{run_name}_step{step}").result()

    metrics["end_time"] = datetime.now().isoformat()
    metrics["final_train_loss"] = float(train_loss)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    training_client.save_state(name=f"{run_name}_final").result()
    logger.info(f"Haiku SFT distillation complete: {run_name}")
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n-pairs", type=int, default=500)
    parser.add_argument("--results-dir", type=str, default="results/haiku_distill")
    args = parser.parse_args()

    config = {
        "model": args.model, "n_steps": args.steps, "n_pairs": args.n_pairs,
        "lora_rank": 16, "batch_size": 8, "grad_accum": 4, "lr": 2e-4, "eval_every": 50,
        "run_name": f"haiku_sft_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_haiku_sft_distill(config, results_dir)


if __name__ == "__main__":
    main()
