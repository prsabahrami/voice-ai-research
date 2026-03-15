"""
Generate ~500 hard math problems for RL training from the MATH dataset.

- Levels 4 and 5 only (supplement with level 3 if needed to hit 500)
- Numeric answers only (parseable as float after extracting from \\boxed{})
- No overlap with eval_prompts.jsonl
- Output: prompts_new.jsonl with {"prompt": "...", "ground_truth": "N.0"} per line
"""

import json
import math
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
EVAL_PATH = SCRIPT_DIR / "eval_prompts.jsonl"
OUT_PATH = SCRIPT_DIR / "prompts_new.jsonl"
TARGET = 500

# ---------------------------------------------------------------------------
# Load eval prompts to exclude (match on raw problem text)
# ---------------------------------------------------------------------------
eval_prompts: set[str] = set()
with open(EVAL_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            eval_prompts.add(obj["prompt"])

print(f"Loaded {len(eval_prompts)} eval prompts to exclude.")

# ---------------------------------------------------------------------------
# Brace-balanced \\boxed{} extractor
# Handles nested braces like \\boxed{\\frac{1}{2}}
# ---------------------------------------------------------------------------
def extract_last_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...} in text, handling nested braces."""
    # Find all positions of \boxed{
    starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not starts:
        return None

    last_start = starts[-1]
    # Walk forward from the opening brace to find the matching close brace
    open_brace = last_start + len("\\boxed")  # points to '{'
    depth = 0
    i = open_brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i]  # content between braces
        i += 1
    return None  # unclosed brace


# ---------------------------------------------------------------------------
# Numeric normalization — mirrors reward.py
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _normalize_number(s: str) -> float | None:
    """Try to parse a string as a number, stripping whitespace/commas."""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def extract_numeric_answer(solution: str) -> float | None:
    """
    Extract a numeric answer from a MATH solution.
    Uses brace-balanced \\boxed{} extraction. NO fallback to last number in
    solution — we want strictly numeric boxed answers.

    Strategy (applied to the boxed content only):
      1. Direct float parse (handles integers, decimals, negatives)
      2. Simple \\frac{a}{b} pattern (both a and b must be numeric)
      3. Strip LaTeX wrappers and retry float parse
      4. Single number anywhere in boxed content
    Returns float or None.
    """
    boxed = extract_last_boxed(solution)
    if boxed is None:
        return None

    content = boxed.strip()

    # 1. Direct parse
    val = _normalize_number(content)
    if val is not None:
        return val

    # 2. Simple \frac{a}{b} — both parts must be numeric
    frac_match = re.fullmatch(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}", content
    )
    if frac_match:
        num = _normalize_number(frac_match.group(1))
        den = _normalize_number(frac_match.group(2))
        if num is not None and den is not None and den != 0:
            return num / den

    # 3. Strip known single-arg LaTeX wrappers (\text{...}, \sqrt{...}, etc.)
    #    Only strip if the result still looks purely numeric
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^{}]*)\}", r"\1", content)
    cleaned = re.sub(r"\\[a-zA-Z]+\b", "", cleaned)
    # Also strip currency symbols and %
    cleaned = re.sub(r"[$%]", "", cleaned)
    val = _normalize_number(cleaned)
    if val is not None:
        return val

    # 4. Only ONE number in boxed content (no tuples, intervals, expressions)
    nums = _NUM_RE.findall(content.replace(",", ""))
    if len(nums) == 1:
        val = _normalize_number(nums[0])
        if val is not None:
            return val

    # Not a numeric answer — skip this problem
    return None


# ---------------------------------------------------------------------------
# Load MATH dataset
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed.", file=sys.stderr)
    sys.exit(1)

print("Loading MATH dataset (chiayewken/competition_math)...")
ds = load_dataset("chiayewken/competition_math")

all_problems = []
for split in ds:
    for item in ds[split]:
        all_problems.append(item)

print(f"Total problems in dataset: {len(all_problems)}")

# ---------------------------------------------------------------------------
# Filter and collect
# ---------------------------------------------------------------------------
def make_prompt(problem_text: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        + problem_text.strip()
    )


collected: list[dict] = []
skipped_non_numeric = 0
skipped_eval_overlap = 0
skipped_duplicate = 0

seen_problems: set[str] = set()

# Process by level priority: 5 & 4 first, then 3 if needed
for target_levels in [["Level 5", "Level 4"], ["Level 3"]]:
    if len(collected) >= TARGET:
        break

    level_label = "+".join(target_levels)
    batch = [p for p in all_problems if p.get("level", "") in target_levels]
    print(f"\nLevel {level_label}: {len(batch)} problems available")

    numeric_count = 0
    for item in batch:
        if len(collected) >= TARGET:
            break

        problem_text = item.get("problem", "").strip()
        solution = item.get("solution", "").strip()

        if not problem_text or not solution:
            continue

        # Skip duplicates within this run
        if problem_text in seen_problems:
            skipped_duplicate += 1
            continue

        # Extract numeric answer — strict (boxed only, no fallback)
        answer_float = extract_numeric_answer(solution)
        if answer_float is None:
            skipped_non_numeric += 1
            continue

        # Sanity: must be finite
        if not math.isfinite(answer_float):
            skipped_non_numeric += 1
            continue

        # Build the prompt
        prompt = make_prompt(problem_text)

        # Skip eval overlaps
        if prompt in eval_prompts:
            skipped_eval_overlap += 1
            continue

        seen_problems.add(problem_text)
        numeric_count += 1
        collected.append({
            "prompt": prompt,
            "ground_truth": str(float(answer_float)),
        })

    print(f"  -> Accepted {numeric_count} numeric problems from this level group")

print(f"\n--- Collection Summary ---")
print(f"Collected                    : {len(collected)}")
print(f"Skipped (non-numeric answer) : {skipped_non_numeric}")
print(f"Skipped (eval overlap)       : {skipped_eval_overlap}")
print(f"Skipped (duplicate)          : {skipped_duplicate}")

if len(collected) < TARGET:
    print(f"WARNING: Only collected {len(collected)} problems (target was {TARGET}).")

# ---------------------------------------------------------------------------
# Spot-check: verify 20 random samples are sensible
# ---------------------------------------------------------------------------
import random
random.seed(42)
sample = random.sample(collected, min(20, len(collected)))
print("\nSpot-check (20 random samples):")
for entry in sample[:5]:
    print(f"  gt={entry['ground_truth']:>10}  prompt[:70]: {entry['prompt'][entry['prompt'].index(chr(10))+2:][:70]!r}")

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
with open(OUT_PATH, "w") as f:
    for item in collected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved {len(collected)} problems to {OUT_PATH}")

# Print first 3 entries
print("\nFirst 3 entries:")
for entry in collected[:3]:
    body = entry["prompt"].split("\n\n", 1)[1] if "\n\n" in entry["prompt"] else entry["prompt"]
    print(f"  ground_truth: {entry['ground_truth']}")
    print(f"  problem[:100]: {body[:100]!r}")
    print()
