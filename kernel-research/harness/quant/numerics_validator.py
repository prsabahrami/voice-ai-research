"""
numerics_validator.py
=====================
Correctness validation for quantized kernel outputs.

Compares a baseline FP16/FP32 tensor against a quantized output tensor and
reports comprehensive error statistics used to decide whether a kernel meets
production acceptance criteria.

Usage:
    from numerics_validator import validate_quantized_output

    metrics = validate_quantized_output(baseline_fp16, quantized_output)
    if metrics["passed"]:
        print("Kernel passes correctness check")
    else:
        print("FAIL:", metrics)

Author: miniQuant kernel-research branch
"""

import math
import warnings
from typing import Dict, Union

import torch


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MAX_ABS_ERROR_THRESHOLD = 1e-4   # flag if max absolute error exceeds this
RELATIVE_ERROR_THRESHOLD = 1e-2  # 1% relative error warning threshold
KL_DIV_THRESHOLD = 1e-3          # KL divergence warning threshold


# ---------------------------------------------------------------------------
# Core validator
# ---------------------------------------------------------------------------

def validate_quantized_output(
    baseline: torch.Tensor,
    quantized: torch.Tensor,
    threshold_max_abs: float = MAX_ABS_ERROR_THRESHOLD,
    threshold_rel: float = RELATIVE_ERROR_THRESHOLD,
    threshold_kl: float = KL_DIV_THRESHOLD,
    eps: float = 1e-8,
) -> Dict:
    """Compare quantized kernel output against a floating-point baseline.

    Metrics computed:
      - max_abs_error    : max |baseline - quantized|
      - mean_abs_error   : mean |baseline - quantized|
      - relative_error   : mean |baseline - quantized| / (mean |baseline| + eps)
      - kl_divergence    : KL(softmax(baseline) || softmax(quantized)) over last dim
      - cosine_similarity: mean cosine similarity between baseline and quantized rows
      - snr_db           : signal-to-noise ratio in decibels

    Flags:
      - passed           : True iff max_abs_error <= threshold_max_abs
      - warn_relative    : True iff relative_error > threshold_rel
      - warn_kl          : True iff kl_divergence > threshold_kl
    """
    if baseline.shape != quantized.shape:
        raise ValueError(
            f"Shape mismatch: baseline {tuple(baseline.shape)} vs quantized {tuple(quantized.shape)}"
        )

    ref = baseline.detach().float().cpu()
    out = quantized.detach().float().cpu()

    diff = (ref - out).abs()

    max_abs_error = diff.max().item()
    mean_abs_error = diff.mean().item()
    median_abs_error = diff.median().item()

    ref_norm = ref.abs().mean().item()
    relative_error = mean_abs_error / (ref_norm + eps)

    signal_power = (ref ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    if noise_power < 1e-30:
        snr_db = float("inf")
    else:
        snr_db = 10.0 * math.log10(signal_power / noise_power + 1e-30)

    flat_ref = ref.reshape(-1, ref.shape[-1]) if ref.dim() > 1 else ref.unsqueeze(0)
    flat_out = out.reshape(-1, out.shape[-1]) if out.dim() > 1 else out.unsqueeze(0)
    dot = (flat_ref * flat_out).sum(dim=-1)
    norm_ref = flat_ref.norm(dim=-1).clamp(min=eps)
    norm_out = flat_out.norm(dim=-1).clamp(min=eps)
    cos_sim = (dot / (norm_ref * norm_out)).mean().item()

    kl_div = _kl_divergence(ref, out, eps=eps)

    passed = max_abs_error <= threshold_max_abs
    warn_relative = relative_error > threshold_rel
    warn_kl = (kl_div < float("inf")) and (kl_div > threshold_kl)

    metrics = {
        "max_abs_error":     max_abs_error,
        "mean_abs_error":    mean_abs_error,
        "median_abs_error":  median_abs_error,
        "relative_error":    relative_error,
        "snr_db":            snr_db,
        "cosine_similarity": cos_sim,
        "kl_divergence":     kl_div,
        "tensor_shape":      list(baseline.shape),
        "num_elements":      int(baseline.numel()),
        "threshold_max_abs": threshold_max_abs,
        "threshold_rel":     threshold_rel,
        "threshold_kl":      threshold_kl,
        "passed":            passed,
        "warn_relative":     warn_relative,
        "warn_kl":           warn_kl,
        "summary": (
            f"max_abs={max_abs_error:.4e}  mean_abs={mean_abs_error:.4e}  "
            f"rel={relative_error:.4e}  snr={snr_db:.1f}dB  "
            f"cos={cos_sim:.4f}  kl={kl_div:.4e}  "
            f"{'PASS' if passed else 'FAIL'}"
        ),
    }
    return metrics


def _kl_divergence(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    if p_logits.shape[-1] < 2:
        return float("inf")

    D = p_logits.shape[-1]
    p_flat = p_logits.reshape(-1, D)
    q_flat = q_logits.reshape(-1, D)

    p_prob = torch.softmax(p_flat.double(), dim=-1).clamp(min=eps)
    q_prob = torch.softmax(q_flat.double(), dim=-1).clamp(min=eps)

    kl = (p_prob * (p_prob / q_prob).log()).sum(dim=-1)
    return kl.mean().item()


def validate_all(
    baseline: torch.Tensor,
    outputs: Dict,
    **kwargs,
) -> Dict:
    """Validate multiple quantized outputs against a single baseline."""
    results = {}
    for name, out in outputs.items():
        try:
            results[name] = validate_quantized_output(baseline, out, **kwargs)
        except Exception as e:
            results[name] = {"error": str(e), "passed": False}
    return results


def _self_test() -> None:
    torch.manual_seed(0)
    ref = torch.randn(4, 64)

    m = validate_quantized_output(ref, ref.clone())
    assert m["max_abs_error"] == 0.0
    assert m["passed"]
    print(f"  [PASS] perfect match: {m['summary']}")

    noise = torch.randn_like(ref) * 0.01
    m2 = validate_quantized_output(ref, ref + noise)
    print(f"  [INFO] small noise:   {m2['summary']}")

    large_noise = torch.randn_like(ref) * 1.0
    m3 = validate_quantized_output(ref, ref + large_noise)
    assert not m3["passed"]
    print(f"  [PASS] large noise correctly flagged: {m3['summary']}")

    results = validate_all(ref, {"int8": ref + noise, "fp8": ref + noise * 0.1})
    for k, v in results.items():
        print(f"  [BATCH] {k}: {v['summary']}")


if __name__ == "__main__":
    _self_test()
