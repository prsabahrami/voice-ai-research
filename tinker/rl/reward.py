"""
Reward function for GRPO training.

CONTRACT:
    compute_reward(completion: str, ground_truth: str) -> float
    - Must return a float (typically 0.0 to 1.0)
    - Must be deterministic
    - Must NEVER crash (catch all exceptions, return 0.0)

EXPERIMENT: Binary reward + format_coef trick (from tinker-cookbook).
Small format penalty encourages consistent \\boxed{} usage.
reward = format_coef * (correct_format - 1) + correct_answer
"""

import re

# Match \boxed{...}
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
# Fallback: extract last number
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

# Format coefficient (from tinker-cookbook ProblemEnv)
FORMAT_COEF = 0.1


def _normalize_number(s: str) -> float | None:
    """Try to parse a string as a number, stripping whitespace/commas."""
    s = s.strip().replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def compute_reward(completion: str, ground_truth: str) -> float:
    """
    Binary reward with format bonus (format_coef trick).

    correct_format=True, correct_answer=True  → 1.0
    correct_format=False, correct_answer=True → 0.9
    correct_format=True, correct_answer=False → 0.0
    correct_format=False, correct_answer=False → -0.1
    """
    try:
        expected = _normalize_number(ground_truth)
        if expected is None:
            return 0.0

        # Check format: does completion contain \boxed{}?
        boxed_matches = _BOXED_RE.findall(completion)
        correct_format = len(boxed_matches) > 0

        # Check answer correctness
        correct_answer = False
        if boxed_matches:
            predicted = _normalize_number(boxed_matches[-1])
            if predicted is not None and abs(predicted - expected) < 1e-6:
                correct_answer = True
        else:
            # Fallback: last number
            num_matches = _NUM_RE.findall(completion.replace(",", ""))
            if num_matches:
                predicted = _normalize_number(num_matches[-1])
                if predicted is not None and abs(predicted - expected) < 1e-6:
                    correct_answer = True

        # format_coef trick: reward = format_coef * (correct_format - 1) + correct_answer
        reward = FORMAT_COEF * (int(correct_format) - 1) + int(correct_answer)
        return reward
    except Exception:
        return 0.0
