#!/usr/bin/env python3
"""SDPO Experiment Sweep using Tinker API on GPT-OSS-20B."""
import os, json, time, datetime, traceback, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

TINKER_API_KEY = os.environ.get("TINKER_API_KEY", "")
RESULTS_FILE = "/workspace/experiment_results.jsonl"
MODEL = "openai/gpt-oss-20b"
LORA_RANK = 8

# Sweep configs
BETAS = [0.1, 0.2, 0.3, 0.5]
LRS = [1e-4, 2e-4, 5e-4]
N_PAIRS_LIST = [10, 20, 30]
STEPS_PER_EXP = 5

def log_result(exp_id, method, hyperparams, metrics, error=None):
    entry = {
        "exp_id": exp_id,
        "method": method,
        "model": MODEL,
        "hyperparams": hyperparams,
        "metrics": metrics if not error else {"error": str(error)},
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "completed" if not error else "failed"
    }
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{exp_id}] {'OK' if not error else 'FAIL'}: {json.dumps(metrics if not error else {'error': str(error)})}")

def apply_chat_template_tokens(tokenizer, messages, add_generation_prompt=False):
    """Return a flat list of ints for a chat message list."""
    # Use tokenize=False to get text, then encode for reliable flat int list
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )
    return tokenizer.encode(text)

def generate_preference_pairs(n_pairs):
    """Generate synthetic haiku-style preference pairs."""
    pairs = []
    prompts = [
        "Explain machine learning in simple terms.",
        "What is the meaning of life?",
        "Describe a sunset over the ocean.",
        "How does a neural network learn?",
        "Tell me about the history of computing.",
        "What makes a good leader?",
        "Explain quantum computing briefly.",
        "Describe the process of photosynthesis.",
        "What is consciousness?",
        "How do birds navigate during migration?",
    ]
    for i in range(n_pairs):
        prompt = prompts[i % len(prompts)]
        # Haiku-style chosen (concise, poetic, structured)
        chosen = f"Like autumn leaves fall,\n{prompt.lower().replace('?','').strip()}\nreveals itself clear."
        # Plain rejected (verbose, flat)
        rejected = (
            f"Well, that is a complex question. Let me think about it. {prompt} "
            "There are many perspectives on this topic and I could go on at length about various aspects of it."
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs

def run_experiment(exp_id, beta, lr, n_pairs, steps):
    """Run a single SDPO experiment."""
    import tinker
    from tinker import types

    print(f"\n{'='*60}")
    print(f"Starting {exp_id}: beta={beta}, lr={lr}, n_pairs={n_pairs}, steps={steps}")
    print(f"{'='*60}")

    # Create service client and fresh LoRA training client
    sc = tinker.ServiceClient(api_key=TINKER_API_KEY)
    tc = sc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)

    tokenizer = tc.get_tokenizer()
    pairs = generate_preference_pairs(n_pairs)
    losses = []

    for step in range(steps):
        pair = pairs[step % len(pairs)]

        chosen_msgs = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": pair["chosen"]}
        ]
        prompt_msgs = [{"role": "user", "content": pair["prompt"]}]

        try:
            # Tokenize full chosen sequence (flat list of ints)
            chosen_tokens = apply_chat_template_tokens(
                tokenizer, chosen_msgs, add_generation_prompt=False
            )

            # Tokenize prompt only (to know where response begins)
            prompt_tokens = apply_chat_template_tokens(
                tokenizer, prompt_msgs, add_generation_prompt=True
            )
            prompt_len = len(prompt_tokens)

            # input = chosen[:-1], targets = chosen[1:]
            input_tokens = chosen_tokens[:-1]
            target_tokens = chosen_tokens[1:]

            # Weights: 0.0 for prompt tokens, 1.0 for response tokens
            weights = [0.0] * min(prompt_len, len(input_tokens)) + \
                      [1.0] * max(0, len(input_tokens) - prompt_len)
            # Ensure length matches
            weights = (weights + [1.0] * len(input_tokens))[:len(input_tokens)]

            print(f"  Step {step+1}/{steps}: seq_len={len(input_tokens)}, prompt_len={prompt_len}")

            datum = types.Datum(
                model_input=types.ModelInput.from_ints(input_tokens),
                loss_fn_inputs={
                    "target_tokens": types.TensorData(
                        data=list(int(t) for t in target_tokens), dtype="int64"
                    ),
                    "weights": types.TensorData(
                        data=list(float(w) for w in weights), dtype="float32"
                    ),
                }
            )

            # Forward + backward pass
            fwd_future = tc.forward_backward(data=[datum], loss_fn="cross_entropy")
            # Optimizer step (issued before blocking on fwd result - pipeline)
            opt_future = tc.optim_step(types.AdamParams(learning_rate=lr))

            # Block for results
            result = fwd_future.result()
            opt_future.result()

            # Extract loss from metrics dict
            # "loss:sum" contains the actual cross-entropy loss
            # "clock_cycle:unique" is a server counter, not a loss
            loss_val = None
            if hasattr(result, 'metrics') and result.metrics:
                for key in ["loss:sum", "mean_loss", "loss", "cross_entropy_loss"]:
                    if key in result.metrics:
                        loss_val = float(result.metrics[key])
                        break

            losses.append(loss_val)
            print(f"  Step {step+1}/{steps}: loss={loss_val}, metrics={result.metrics}")

        except Exception as e:
            print(f"  Step {step+1}/{steps}: ERROR - {e}")
            traceback.print_exc()
            losses.append(None)
            time.sleep(5)

    valid_losses = [l for l in losses if l is not None]
    metrics = {
        "final_loss": valid_losses[-1] if valid_losses else None,
        "avg_loss": sum(valid_losses) / len(valid_losses) if valid_losses else None,
        "initial_loss": valid_losses[0] if valid_losses else None,
        "steps_completed": len(valid_losses),
        "steps_attempted": steps,
        "beta": beta,
    }
    return metrics

def main():
    print(f"SDPO Sweep starting at {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Model: {MODEL}, LoRA rank: {LORA_RANK}")
    total = len(BETAS) * len(LRS) * len(N_PAIRS_LIST)
    print(f"Configs: {len(BETAS)} betas x {len(LRS)} LRs x {len(N_PAIRS_LIST)} n_pairs = {total} experiments")

    exp_num = 0
    for beta in BETAS:
        for lr in LRS:
            for n_pairs in N_PAIRS_LIST:
                exp_num += 1
                exp_id = f"sdpo_sie_{exp_num:03d}"
                hyperparams = {
                    "beta": beta,
                    "learning_rate": lr,
                    "n_pairs": n_pairs,
                    "steps": STEPS_PER_EXP,
                    "lora_rank": LORA_RANK,
                }

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        metrics = run_experiment(exp_id, beta, lr, n_pairs, STEPS_PER_EXP)
                        log_result(exp_id, "sdpo", hyperparams, metrics)
                        break
                    except Exception as e:
                        print(f"Attempt {attempt+1}/{max_retries} failed for {exp_id}: {e}")
                        traceback.print_exc()
                        if attempt < max_retries - 1:
                            wait_time = 30 * (2 ** attempt)
                            print(f"Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            log_result(exp_id, "sdpo", hyperparams, {}, error=str(e))

                time.sleep(2)

    print(f"\nSweep complete at {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Total experiments: {exp_num}")
    count = 0
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            count = sum(1 for _ in f)
    print(f"Results logged: {count}")

if __name__ == "__main__":
    main()
