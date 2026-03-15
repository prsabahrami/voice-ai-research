"""Identify which eval problems the model consistently gets wrong.

Runs MV@5 multiple times and tracks per-problem failure rates.

Usage: python error_analysis.py <sampler_weights_path> [num_runs]
"""

import json
import re
import sys
from collections import Counter, defaultdict
import tinker
from tinker import types
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3-8B"
MAX_TOKENS = 4096
TEMPERATURE = 0.5
EVAL_PROMPTS_PATH = "../../tinker/rl/eval_prompts.jsonl"
NUM_SAMPLES = 5

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s):
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_answer(completion):
    boxed = _BOXED_RE.findall(completion)
    if boxed:
        return boxed[-1].strip()
    nums = _NUM_RE.findall(completion.replace(",", ""))
    if nums:
        return nums[-1].strip()
    return None


def check_correct(answer, gt):
    if answer is None:
        return False
    exp = _normalize_number(gt)
    pred = _normalize_number(answer)
    if exp is not None and pred is not None:
        return abs(pred - exp) < 1e-6
    return answer.strip() == gt.strip()


def main():
    sampler_path = sys.argv[1]
    num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    eval_prompts = []
    with open(EVAL_PROMPTS_PATH) as f:
        for line in f:
            if line.strip():
                eval_prompts.append(json.loads(line))

    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(base_model=MODEL, model_path=sampler_path)

    stop_seqs = [tokenizer.eos_token] if tokenizer.eos_token else []
    for st in ["<|im_end|>", "<|eot_id|>", "</s>"]:
        if st not in stop_seqs:
            stop_seqs.append(st)

    sp = types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, stop=stop_seqs)

    # Track per-problem results across runs
    problem_correct = defaultdict(int)  # problem_idx → number of runs correct (MV)
    problem_any = defaultdict(int)      # problem_idx → number of runs with any correct

    for run in range(num_runs):
        futures = []
        for item in eval_prompts:
            msgs = [{"role": "user", "content": item["prompt"]}]
            toks = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            mi = types.ModelInput(chunks=[types.EncodedTextChunk(tokens=toks)])
            f = sampling_client.sample(prompt=mi, num_samples=NUM_SAMPLES, sampling_params=sp)
            futures.append((f, item))

        for i, (f, item) in enumerate(futures):
            result = f.result()
            answers = []
            any_ok = False
            for seq in result.sequences:
                text = tokenizer.decode(seq.tokens, skip_special_tokens=True)
                ans = extract_answer(text)
                if ans:
                    answers.append(ans)
                if check_correct(ans, item["ground_truth"]):
                    any_ok = True

            # MV
            if answers:
                norm = []
                for a in answers:
                    n = _normalize_number(a)
                    norm.append(f"{n:.6f}" if n is not None else a.strip())
                mv = Counter(norm).most_common(1)[0][0]
                if check_correct(mv, item["ground_truth"]):
                    problem_correct[i] += 1
            if any_ok:
                problem_any[i] += 1

        mv_acc = sum(1 for i in range(len(eval_prompts)) if problem_correct[i] > run) / len(eval_prompts)
        print(f"Run {run+1}/{num_runs}: MV@5 = {mv_acc:.1%}", flush=True)

    # Report
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS ({num_runs} runs, MV@{NUM_SAMPLES}, temp={TEMPERATURE})")
    print(f"{'='*60}")

    never_correct = []
    sometimes_wrong = []
    always_correct = []

    for i, item in enumerate(eval_prompts):
        mv_count = problem_correct[i]
        any_count = problem_any[i]
        if mv_count == 0:
            never_correct.append((i, item, any_count))
        elif mv_count < num_runs:
            sometimes_wrong.append((i, item, mv_count, any_count))
        else:
            always_correct.append(i)

    print(f"\nAlways correct (MV): {len(always_correct)}/{len(eval_prompts)}")
    print(f"Sometimes wrong: {len(sometimes_wrong)}")
    print(f"Never correct (MV): {len(never_correct)}")

    if never_correct:
        print(f"\n--- NEVER CORRECT (MV) ---")
        for i, item, any_ct in never_correct:
            print(f"  #{i}: any_correct={any_ct}/{num_runs} | GT={item['ground_truth']}")
            print(f"    {item['prompt'][:120]}...")

    if sometimes_wrong:
        print(f"\n--- SOMETIMES WRONG ---")
        for i, item, mv_ct, any_ct in sometimes_wrong:
            print(f"  #{i}: mv_correct={mv_ct}/{num_runs}, any={any_ct}/{num_runs} | GT={item['ground_truth']}")
            print(f"    {item['prompt'][:120]}...")


if __name__ == "__main__":
    main()
