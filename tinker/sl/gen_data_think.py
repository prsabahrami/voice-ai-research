"""Reformat MATH solutions to use Qwen3-8B's <think>...</think> format.

Takes reasoning steps and puts them inside <think> blocks, with the
final \\boxed{answer} outside.

Run: python gen_data_think.py
"""

import json
import re

def find_boxed(text: str) -> list[tuple[int, int]]:
    """Find all \\boxed{...} spans, handling nested braces."""
    spans = []
    i = 0
    while i < len(text):
        idx = text.find("\\boxed{", i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        j = idx + 6  # position of '{'
        while j < len(text):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    spans.append((idx, j + 1))
                    break
            j += 1
        i = j + 1 if j < len(text) else len(text)
    return spans

data = []
with open("data.jsonl") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

reformatted = []
skipped = 0
for item in data:
    solution = item["response"]

    # Find the last \boxed{...} match
    boxed_spans = find_boxed(solution)
    if not boxed_spans:
        skipped += 1
        continue

    start, end = boxed_spans[-1]
    boxed_str = solution[start:end]  # e.g. \boxed{42} or \boxed{\frac{1}{2}}

    # Everything before the boxed answer is reasoning
    reasoning = solution[:start].rstrip()

    # Format: <think>reasoning</think>\n\nThe answer is \boxed{...}
    new_response = f"<think>\n{reasoning}\n</think>\n\nThe answer is ${boxed_str}$."

    reformatted.append({"prompt": item["prompt"], "response": new_response})

print(f"Reformatted {len(reformatted)}/{len(data)} examples ({skipped} skipped, no boxed)")

# Write
with open("data.jsonl", "w") as f:
    for item in reformatted:
        f.write(json.dumps(item) + "\n")

print(f"Wrote {len(reformatted)} examples to data.jsonl")

# Show samples
for item in reformatted[:2]:
    print(f"\nPROMPT: {item['prompt'][:100]}")
    print(f"RESPONSE: {item['response'][:400]}")
    print("---")
