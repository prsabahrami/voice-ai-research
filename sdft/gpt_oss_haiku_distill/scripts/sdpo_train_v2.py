#!/usr/bin/env python3
"""
SDPO: Style-Distillation Preference Optimization on GPT-OSS-20B
DPO-style training: chosen=Haiku-style, rejected=flat baseline.
Uses Tinker API v0.16.1 cross_entropy loss with negative loss scaling.
"""

import os, sys, json, time, random, logging
import numpy as np
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [SDPO] %(message)s")
logger = logging.getLogger("sdpo")

import tinker
from tinker import types
import anthropic

RESULTS_DIR = Path("results/sdpo")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "openai/gpt-oss-20b"
RANK = 16
LR = 1e-4
BATCH_SIZE = 2  # pairs
BETA = 0.1  # DPO temperature
MAX_STEPS = 150
EVAL_EVERY = 25
RUN_NAME = f"sdpo_gpt_oss_20b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
METRICS_PATH = RESULTS_DIR / f"{RUN_NAME}_metrics.json"

# Load preference pairs
DATA_PATH = Path("data/haiku_dpo_preference_pairs_batch1.json")
with open(DATA_PATH) as f:
    pairs = json.load(f)
logger.info(f"Loaded {len(pairs)} preference pairs")

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model=MODEL, rank=RANK)
tokenizer = tc.get_tokenizer()
logger.info("Tinker client ready")

def make_datum(pair, response_key):
    """Build Datum for chosen or rejected response."""
    msgs = [{"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair[response_key]}]
    prompt_msgs = [{"role": "user", "content": pair["prompt"]}]
    
    full_enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)
    prompt_enc = tokenizer.apply_chat_template(prompt_msgs, add_generation_prompt=True)
    full_tokens = list(full_enc["input_ids"])
    n_prompt = len(list(prompt_enc["input_ids"]))
    
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    weights = [0.0 if i < n_prompt - 1 else 1.0 for i in range(len(input_tokens))]
    
    if sum(weights) == 0:
        return None
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights}
    )

# Encode all preference pairs
encoded_pairs = []
for p in pairs:
    c = make_datum(p, "chosen")
    r = make_datum(p, "rejected")
    if c and r:
        encoded_pairs.append((c, r))
logger.info(f"Encoded {len(encoded_pairs)} preference pairs")

metrics = {
    "run_name": RUN_NAME, "model": MODEL, "method": "sdpo_dpo",
    "beta": BETA, "steps": [], "eval_steps": [],
    "start_time": datetime.now().isoformat(),
}

def compute_nll(fb_result, batch):
    logprobs = np.concatenate([out["logprobs"].tolist() for out in fb_result.loss_fn_outputs])
    weights = np.concatenate([d.loss_fn_inputs["weights"].tolist() for d in batch])
    w_sum = weights.sum()
    return float(-np.dot(logprobs, weights) / w_sum) if w_sum > 0 else float("nan")

step = 0
logger.info("Starting SDPO training loop...")

while step < MAX_STEPS:
    batch_pairs = random.choices(encoded_pairs, k=BATCH_SIZE)
    chosen_batch = [c for c, r in batch_pairs]
    rejected_batch = [r for c, r in batch_pairs]
    
    # DPO approximation: forward-backward on chosen + rejected with negative scale
    # Minimize chosen NLL + maximize rejected NLL (by using loss_fn_config)
    fb_chosen = tc.forward_backward(data=chosen_batch, loss_fn="cross_entropy")
    fb_rejected = tc.forward_backward(
        data=rejected_batch, loss_fn="cross_entropy",
        loss_fn_config={"scale": -BETA}  # Negate the rejected gradient
    )
    opt = tc.optim_step(types.AdamParams(learning_rate=LR))
    
    c_result = fb_chosen.result()
    r_result = fb_rejected.result()
    opt.result()
    step += 1
    
    c_nll = compute_nll(c_result, chosen_batch)
    r_nll = compute_nll(r_result, rejected_batch)
    dpo_loss = c_nll - BETA * r_nll
    
    metrics["steps"].append({
        "step": step, "dpo_loss": dpo_loss,
        "chosen_nll": c_nll, "rejected_nll": r_nll, "t": time.time()
    })
    logger.info(f"Step {step}/{MAX_STEPS} | dpo_loss={dpo_loss:.4f} chosen={c_nll:.4f} rejected={r_nll:.4f}")
    
    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        test_pair = random.choice(pairs)
        prompt_tok = tokenizer.encode(test_pair["prompt"], add_special_tokens=True)
        sp = types.SamplingParams(max_tokens=100, temperature=0.7)
        sampler = tc.save_weights_and_get_sampling_client(name=f"sdpo_eval_s{step}")
        out = sampler.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tok),
            sampling_params=sp, num_samples=1
        ).result()
        model_text = tokenizer.decode(out.sequences[0].tokens)
        
        metrics["eval_steps"].append({
            "step": step,
            "prompt": test_pair["prompt"],
            "model_output": model_text[:300],
            "haiku_chosen": test_pair["chosen"][:300],
            "flat_rejected": test_pair["rejected"][:300],
            "t": time.time(),
        })
        logger.info(f"  Model:   {model_text[:100]!r}")
        logger.info(f"  Chosen:  {test_pair['chosen'][:100]!r}")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))
        tc.save_state(name=f"{RUN_NAME}_step{step}").result()

metrics["end_time"] = datetime.now().isoformat()
metrics["final_dpo_loss"] = dpo_loss
METRICS_PATH.write_text(json.dumps(metrics, indent=2))
tc.save_state(name=f"{RUN_NAME}_final").result()
logger.info(f"SDPO complete. Final dpo_loss: {dpo_loss:.4f}")
