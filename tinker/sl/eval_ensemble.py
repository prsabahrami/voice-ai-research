"""Evaluate ensemble of 2 SFT models via combined majority vote.

Usage: python eval_ensemble.py <sampler1_path> <sampler2_path> [num_samples_per_model]
"""

import json
import re
import sys
from collections import Counter
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(completion):
    boxed_matches = _BOXED_RE.findall(completion)
    if boxed_matches:
        return boxed_matches[-1].strip()
    num_matches = _NUM_RE.findall(completion.replace(",", ""))
    if num_matches:
        return num_matches[-1].strip()
    return None


def check_correct(answer, ground_truth):
    if answer is None:
        return False
    expected = _normalize_number(ground_truth)
    predicted = _normalize_number(answer)
    if expected is not None and predicted is not None:
        return abs(predicted - expected) < 1e-6
    return answer.strip() == ground_truth.strip()


def main():
    if len(sys.argv) < 3:
        print("Usage: python eval_ensemble.py <sampler1> <sampler2> [n_per_model]")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]
    n_per_model = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(f"Ensemble: 2 models, {n_per_model} samples each = {n_per_model*2} total")

    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                eval_prompts.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    service_client = tinker.ServiceClient()
    sc1 = service_client.create_sampling_client(base_model=MODEL, model_path=path1)
    sc2 = service_client.create_sampling_client(base_model=MODEL, model_path=path2)

    stop_sequences = []
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)
    for st in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if st not in stop_sequences:
            stop_sequences.append(st)

    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=stop_sequences)

    # Submit all requests
    futures = []
    for item in eval_prompts:
        messages = [{"role": "user", "content": item["prompt"]}]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)])
        f1 = sc1.sample(prompt=mi, num_samples=n_per_model, sampling_params=sp)
        f2 = sc2.sample(prompt=mi, num_samples=n_per_model, sampling_params=sp)
        futures.append((f1, f2, item))

    correct_mv = 0
    correct_any = 0

    for i, (f1, f2, item) in enumerate(futures):
        r1, r2 = f1.result(), f2.result()
        answers = []
        any_correct = False

        for r in [r1, r2]:
            for seq in r.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                answer = extract_answer(text)
                if answer is not None:
                    answers.append(answer)
                if check_correct(answer, item["ground_truth"]):
                    any_correct = True

        if answers:
            normalized = []
            for a in answers:
                n = _normalize_number(a)
                normalized.append(f"{n:.6f}" if n is not None else a.strip())
            majority = Counter(normalized).most_common(1)[0][0]
            if check_correct(majority, item["ground_truth"]):
                correct_mv += 1

        if any_correct:
            correct_any += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(eval_prompts)}] mv: {correct_mv/(i+1):.1%} | any: {correct_any/(i+1):.1%}")

    n = len(eval_prompts)
    print(f"\n{'='*60}")
    print(f"ENSEMBLE MV@{n_per_model*2} ({n_per_model} per model, temp={TEMPERATURE})")
    print(f"{'='*60}")
    print(f"majority_accuracy: {correct_mv/n:.4f} ({correct_mv}/{n})")
    print(f"any_correct: {correct_any/n:.4f} ({correct_any}/{n})")


if __name__ == "__main__":
    main()
