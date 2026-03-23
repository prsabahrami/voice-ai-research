import os, json, time, datetime, sys, traceback

# Load API key
env_lines = open("/workspace/.env").read().splitlines()
api_key = None
for line in env_lines:
    if line.startswith("TINKER_API_KEY="):
        api_key = line.split("=", 1)[1]
        break

from tinker import ServiceClient, types

# CORRECT client creation: base_model, NO lora_config
sc = ServiceClient(api_key=api_key)
tc = sc.create_lora_training_client(base_model="openai/gpt-oss-20b")
print("Training client created successfully")

# Training data
SDFT_DATA = [
    ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from experience without explicit programming."),
    ("Explain neural networks briefly.", "Neural networks are computing systems of interconnected nodes that process information using connectionist approaches."),
    ("What is gradient descent?", "Gradient descent iteratively adjusts parameters by moving in the direction of steepest decrease of a loss function."),
    ("Define overfitting.", "Overfitting occurs when a model memorizes training noise rather than learning the underlying pattern."),
    ("What is transfer learning?", "Transfer learning reuses a model trained on one task as the starting point for a different but related task."),
    ("Explain attention mechanism.", "Attention allows models to dynamically focus on relevant input parts when producing output."),
    ("What is a transformer?", "A transformer uses self-attention to process sequential data in parallel rather than sequentially."),
    ("Define reinforcement learning.", "Reinforcement learning trains agents to make decisions by rewarding desired and punishing undesired behaviors."),
    ("What is batch normalization?", "Batch normalization normalizes layer inputs across a mini-batch, stabilizing training."),
    ("Explain dropout.", "Dropout randomly deactivates neurons during training to prevent co-adaptation and reduce overfitting."),
]

HAIKU_DATA = [
    ("What is machine learning?", "Patterns in the noise / algorithms learn to see / what humans once taught"),
    ("Explain neural networks.", "Layers upon layers / signals flow like morning light / through connected nodes"),
    ("What is gradient descent?", "Downhill step by step / the mountain of loss grows small / toward the valley floor"),
    ("Define overfitting.", "Too close to the tree / the forest fades from clear sight / memorized not learned"),
    ("What is transfer learning?", "Knowledge carries forth / one domain lights another / wisdom shared reborn"),
]

def make_datum(prompt, response):
    # Simple tokenization using raw bytes (avoid HF tokenizer dependency)
    text = f"User: {prompt}\nAssistant: {response}"
    tokens = list(text.encode("utf-8"))[:200]  # cap at 200 tokens
    prompt_text = f"User: {prompt}\nAssistant: "
    prompt_len = len(prompt_text.encode("utf-8"))
    
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = [0.0] * min(prompt_len - 1, len(input_tokens)) + [1.0] * max(0, len(input_tokens) - prompt_len + 1)
    
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": target_tokens,
            "weights": weights,
        }
    )

results = []
exp_count = 0

def run_experiment(exp_id, method, data_pairs, lr, n_steps, n_examples):
    global exp_count
    batch = [make_datum(p, r) for p, r in data_pairs[:n_examples]]
    losses = []
    try:
        for step in range(n_steps):
            # CRITICAL: call .result() on the future
            result = tc.forward_backward(data=batch, loss_fn="cross_entropy").result()
            loss_val = result.metrics.get("loss:sum", result.metrics.get("loss", 0))
            losses.append(float(loss_val) if loss_val else 0)
            tc.optim_step(types.AdamParams(learning_rate=lr)).result()
        
        entry = {
            "exp_id": exp_id, "method": method, "model": "openai/gpt-oss-20b",
            "hyperparams": {"lr": lr, "n_steps": n_steps, "n_examples": n_examples},
            "metrics": {"losses": losses, "final_loss": losses[-1] if losses else None},
            "timestamp": datetime.datetime.utcnow().isoformat(), "status": "success"
        }
    except Exception as e:
        entry = {
            "exp_id": exp_id, "method": method, "model": "openai/gpt-oss-20b",
            "hyperparams": {"lr": lr, "n_steps": n_steps, "n_examples": n_examples},
            "metrics": {"error": str(e)},
            "timestamp": datetime.datetime.utcnow().isoformat(), "status": "error"
        }
        traceback.print_exc()
    
    with open("/workspace/experiment_results.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[{exp_id}] lr={lr} steps={n_steps} ex={n_examples} -> {entry['metrics'].get('final_loss', entry['metrics'].get('error', 'N/A'))}")
    return entry

# SDFT experiments (18 configs: 3 LR x 3 steps x 2 sizes)
print("\n=== SDFT EXPERIMENTS ===")
for lr in [1e-4, 2e-4, 5e-4]:
    for n_steps in [5, 10, 20]:
        for n_examples in [5, 10]:
            exp_count += 1
            exp_id = f"sdft_miniQuant_{exp_count:03d}"
            entry = run_experiment(exp_id, "sdft", SDFT_DATA, lr, n_steps, n_examples)
            results.append(entry)

# Haiku SFT experiments (6 configs: 3 LR x 2 steps)
print("\n=== HAIKU SFT DISTILLATION ===")
for lr in [1e-4, 2e-4, 5e-4]:
    for n_steps in [10, 20]:
        exp_count += 1
        exp_id = f"haiku_sft_miniQuant_{exp_count:03d}"
        entry = run_experiment(exp_id, "haiku_sft", HAIKU_DATA, lr, n_steps, len(HAIKU_DATA))
        results.append(entry)

# Summary
successes = [r for r in results if r["status"] == "success"]
errors = [r for r in results if r["status"] == "error"]
print(f"\n=== SWEEP COMPLETE ===")
print(f"Total: {len(results)} | Success: {len(successes)} | Errors: {len(errors)}")
if successes:
    best = min(successes, key=lambda x: x["metrics"]["final_loss"])
    print(f"Best: {best['exp_id']} final_loss={best['metrics']['final_loss']}")
print(f"Results: /workspace/experiment_results.jsonl")
