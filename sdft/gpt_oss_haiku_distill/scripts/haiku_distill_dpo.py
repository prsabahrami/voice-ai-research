#!/usr/bin/env python3
"""
Haiku Distillation Method B: DPO with Haiku preferences.
Train GPT-OSS to prefer Haiku-style responses over generic ones using DPO.
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

# Haiku-style system prompt
HAIKU_SYSTEM = """You are a warm, conversational AI assistant. Be helpful and engaging, 
use natural language, and make the conversation feel personal and flowing."""

# Flat/baseline system prompt
BASELINE_SYSTEM = """Answer the question directly and factually. Keep responses brief and impersonal."""

PROMPTS = [
    "What's the best way to stay productive when you're feeling unmotivated?",
    "Can you explain neural networks to a complete beginner?",
    "I made a mistake at work and I'm not sure how to handle it.",
    "What are some good books to read if I want to learn about economics?",
    "Why is exercise so important for mental health?",
    "How do you think about making difficult decisions?",
    "Can you help me understand what blockchain actually does?",
    "What's the most interesting thing about the universe?",
    "I want to start writing more. Where do I begin?",
    "What makes a good API design?",
    "How should I approach learning something completely new?",
    "What are the key differences between Python and JavaScript?",
    "How does memory work in humans?",
    "What makes code readable?",
    "Can you explain compound interest simply?",
    "What's the best way to give feedback to someone?",
    "How do I know if I'm making progress in learning a skill?",
    "What's the relationship between sleep and memory?",
    "Why is diversity important in teams?",
    "How do you handle not knowing the answer to something?",
]


def build_dpo_preference_data(n_pairs: int, cache_path: Path):
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    if cache_path.exists():
        logger.info(f"Loading DPO preference data from {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    pairs = []
    prompts = (PROMPTS * (n_pairs // len(PROMPTS) + 1))[:n_pairs]

    for i, prompt in enumerate(prompts):
        try:
            # Chosen: Haiku-style response
            chosen_resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                system=HAIKU_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            chosen = chosen_resp.content[0].text

            # Rejected: flat/baseline response
            rejected_resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                system=BASELINE_SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            rejected = rejected_resp.content[0].text

            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            logger.info(f"  Pair {i+1}/{n_pairs}: chosen[0:60]={chosen[:60]!r}")
            time.sleep(0.1)
        except Exception as e:
            logger.warning(f"Failed pair {i}: {e}")

    cache_path.write_text(json.dumps(pairs, indent=2))
    logger.info(f"Saved {len(pairs)} DPO pairs to {cache_path}")
    return pairs


def encode_dpo_datum(pair: dict, tokenizer):
    """Encode a DPO preference pair."""
    def make_datum(response):
        msgs = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": response},
        ]
        tokens = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
        prompt_toks = tokenizer.apply_chat_template(
            [{"role": "user", "content": pair["prompt"]}], add_generation_prompt=True
        )
        n_prompt = len(prompt_toks)
        lw = [0.0] * n_prompt + [1.0] * max(0, len(tokens) - n_prompt)
        return types.Datum(tokens=tokens, loss_weights=lw)

    return make_datum(pair["chosen"]), make_datum(pair["rejected"])


def run_haiku_dpo_distill(config: dict, results_dir: Path):
    model_name = config.get("model", "openai/gpt-oss-20b")
    rank = config.get("lora_rank", 16)
    n_steps = config.get("n_steps", 300)
    batch_size = config.get("batch_size", 4)
    lr = config.get("lr", 1e-4)
    beta = config.get("dpo_beta", 0.1)
    eval_every = config.get("eval_every", 50)
    n_pairs = config.get("n_pairs", 300)
    run_name = config.get("run_name", f"haiku_dpo_{model_name.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    logger.info(f"=== Haiku DPO Distillation: {run_name} ===")

    cache_path = results_dir / f"haiku_dpo_pairs_{n_pairs}.json"
    pairs = build_dpo_preference_data(n_pairs, cache_path)
    n_train = int(0.85 * len(pairs))
    train_pairs, eval_pairs = pairs[:n_train], pairs[n_train:]

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(base_model=model_name, rank=rank)
    tokenizer = training_client.get_tokenizer()

    train_encoded = [encode_dpo_datum(p, tokenizer) for p in train_pairs]
    eval_encoded = [encode_dpo_datum(p, tokenizer) for p in eval_pairs]

    metrics = {
        "run_name": run_name, "model": model_name, "config": config,
        "method": "dpo_haiku_preferences",
        "beta": beta, "steps": [], "eval_steps": [],
        "start_time": datetime.now().isoformat(),
    }
    metrics_path = results_dir / f"{run_name}_metrics.json"

    step = 0
    train_loss = float("nan")

    while step < n_steps:
        batch = random.choices(train_encoded, k=batch_size)
        chosen_batch = [c for c, r in batch]
        rejected_batch = [r for c, r in batch]

        fb_chosen = training_client.forward_backward(data=chosen_batch, loss=types.Loss.NLL)
        fb_rejected = training_client.forward_backward(data=rejected_batch, loss=types.Loss.NLL, loss_scale=-beta)

        c_result = fb_chosen.result()
        r_result = fb_rejected.result()
        training_client.optim_step(lr=lr).result()

        c_loss = c_result.loss or float("nan")
        r_loss = r_result.loss or float("nan")
        train_loss = c_loss - beta * r_loss
        step += 1

        metrics["steps"].append({
            "step": step, "train_loss": float(train_loss),
            "chosen_nll": float(c_loss), "rejected_nll": float(r_loss), "t": time.time()
        })
        logger.info(f"Step {step}/{n_steps} | dpo_loss={train_loss:.4f} chosen_nll={c_loss:.4f} rej_nll={r_loss:.4f}")

        if step % eval_every == 0:
            test_pair = random.choice(eval_pairs)
            prompt_toks = tokenizer.apply_chat_template(
                [{"role": "user", "content": test_pair["prompt"]}], add_generation_prompt=True
            )
            sampler = training_client.save_weights_and_get_sampling_client(name=f"haiku_dpo_sampler_step{step}")
            out = sampler.sample(tokens=prompt_toks, max_tokens=200)
            model_text = tokenizer.decode(out.tokens)

            metrics["eval_steps"].append({
                "step": step,
                "prompt": test_pair["prompt"],
                "model_output": model_text[:400],
                "haiku_chosen": test_pair["chosen"][:400],
                "baseline_rejected": test_pair["rejected"][:400],
                "t": time.time(),
            })
            logger.info(f"  Prompt: {test_pair['prompt'][:60]!r}")
            logger.info(f"  Model (DPO): {model_text[:120]!r}")
            logger.info(f"  Chosen ref:  {test_pair['chosen'][:120]!r}")
            metrics_path.write_text(json.dumps(metrics, indent=2))
            training_client.save_state(name=f"{run_name}_step{step}").result()

    metrics["end_time"] = datetime.now().isoformat()
    metrics["final_loss"] = float(train_loss)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    training_client.save_state(name=f"{run_name}_final").result()
    logger.info(f"Haiku DPO distillation complete: {run_name}")
    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n-pairs", type=int, default=300)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--results-dir", default="results/haiku_distill")
    args = parser.parse_args()

    config = {
        "model": args.model, "n_steps": args.steps, "n_pairs": args.n_pairs,
        "lora_rank": 16, "batch_size": 4, "lr": 1e-4, "dpo_beta": args.beta,
        "eval_every": 50,
        "run_name": f"haiku_dpo_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_haiku_dpo_distill(config, results_dir)


if __name__ == "__main__":
    main()
