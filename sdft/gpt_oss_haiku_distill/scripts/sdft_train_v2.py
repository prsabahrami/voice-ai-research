#!/usr/bin/env python3
"""
SDFT: Style-Distillation Fine-Tuning on GPT-OSS-20B via Tinker API v0.16.1
Persistent training session.

Correct API usage:
  Datum(model_input=ModelInput.from_ints(input_tokens),  # tokens[:-1]
        loss_fn_inputs={"target_tokens": target_list,     # tokens[1:]
                        "weights": weight_list})           # 0=prompt, 1=response
"""

import os, sys, json, time, random, logging
import numpy as np
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SDFT] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sdft_training.log"),
    ]
)
logger = logging.getLogger("sdft")

import tinker
from tinker import types

RESULTS_DIR = Path("results/sdft")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("data/haiku_sft_train_batch1.json")

MODEL = "openai/gpt-oss-20b"
RANK = 16
LR = 2e-4
BATCH_SIZE = 4
EVAL_EVERY = 25
MAX_STEPS = 200
RUN_NAME = f"sdft_gpt_oss_20b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
METRICS_PATH = RESULTS_DIR / f"{RUN_NAME}_metrics.json"

logger.info(f"SDFT run: {RUN_NAME}")
logger.info(f"Model: {MODEL}, rank={RANK}, lr={LR}, steps={MAX_STEPS}")

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model=MODEL, rank=RANK)
tokenizer = tc.get_tokenizer()
logger.info("Tinker client ready")

def make_datum(pair):
    """Build cross-entropy Datum with causal LM loss on response tokens only."""
    msgs = [{"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["response"]}]
    prompt_msgs = [{"role": "user", "content": pair["prompt"]}]
    
    full_enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
    prompt_enc = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
    
    full_tokens = list(full_enc["input_ids"])
    n_prompt = len(list(prompt_enc["input_ids"]))
    
    # Causal LM: predict next token from current
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    
    # Weights: 0 for prompt tokens, 1 for response tokens
    weights = [0.0 if i < n_prompt - 1 else 1.0 for i in range(len(input_tokens))]
    
    if sum(weights) == 0:
        return None
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights}
    )

with open(DATA_PATH) as f:
    raw_pairs = json.load(f)
logger.info(f"Loaded {len(raw_pairs)} training pairs")

train_data = [make_datum(p) for p in raw_pairs]
train_data = [d for d in train_data if d is not None]
logger.info(f"Encoded {len(train_data)} valid datums")

metrics = {
    "run_name": RUN_NAME, "model": MODEL, "lora_rank": RANK, "lr": LR,
    "n_train_examples": len(train_data),
    "steps": [], "eval_steps": [],
    "start_time": datetime.now().isoformat(),
}

def save_metrics():
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

def compute_loss(fwdbwd_result, batch):
    """Compute weighted average loss per token."""
    logprobs = np.concatenate([out["logprobs"].tolist() for out in fwdbwd_result.loss_fn_outputs])
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    w_sum = weights.sum()
    return float(-np.dot(logprobs, weights) / w_sum) if w_sum > 0 else float("nan")

step = 0
train_loss = float("nan")
logger.info("Starting SDFT training loop...")

while step < MAX_STEPS:
    batch = random.choices(train_data, k=BATCH_SIZE)
    fb = tc.forward_backward(data=batch, loss_fn="cross_entropy")
    opt = tc.optim_step(types.AdamParams(learning_rate=LR))
    
    fb_result = fb.result()
    opt.result()
    step += 1
    
    train_loss = compute_loss(fb_result, batch)
    metrics["steps"].append({"step": step, "train_loss": train_loss, "t": time.time()})
    logger.info(f"Step {step}/{MAX_STEPS} | train_loss={train_loss:.4f}")
    
    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        # Save checkpoint
        ckpt = tc.save_state(name=f"{RUN_NAME}_step{step}")
        
        # Sample
        test_prompt = "Can you explain what machine learning is in simple terms?"
        prompt_tok = tokenizer.encode(test_prompt, add_special_tokens=True)
        prompt_mi = types.ModelInput.from_ints(tokens=prompt_tok)
        sp = types.SamplingParams(max_tokens=100, temperature=0.7)
        
        sampler = tc.save_weights_and_get_sampling_client(name=f"sdft_eval_s{step}")
        sample_fut = sampler.sample(prompt=prompt_mi, sampling_params=sp, num_samples=1)
        sample_res = sample_fut.result()
        sample_text = tokenizer.decode(sample_res.sequences[0].tokens)
        
        metrics["eval_steps"].append({
            "step": step, "prompt": test_prompt,
            "model_output": sample_text[:300], "t": time.time()
        })
        logger.info(f"  Sample: {sample_text[:120]!r}")
        ckpt.result()
        logger.info(f"  Checkpoint: {RUN_NAME}_step{step}")
        save_metrics()

metrics["end_time"] = datetime.now().isoformat()
metrics["final_train_loss"] = train_loss
save_metrics()
tc.save_state(name=f"{RUN_NAME}_final").result()
logger.info(f"SDFT complete. Final loss: {train_loss:.4f}")
