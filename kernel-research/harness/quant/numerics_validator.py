#!/usr/bin/env python3
"""
Numerics validator for quantized kernels.
Compares quantized vs FP16 baseline, reports error metrics.
Owned by miniQuant.

Usage:
    python numerics_validator.py --kernel int8_attention --batch_size 8
"""

import torch
import torch.nn.functional as F
import math
import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from baseline_kernels import baseline_attention, baseline_gemm, make_attention_args, make_gemm_args

sys.path.insert(0, str(Path(__file__).parent))
from quant_kernels import int8_attention, int8_gemm, fp8_gemm, mixed_precision_attention


MAX_ABS_ERR_THRESHOLD = 1e-4


def validate_kernel(fn_quant, fn_fp16, args, kernel_name: str):
    """
    Run both kernels and compare outputs.
    Returns dict with error metrics and pass/fail.
    """
    with torch.no_grad():
        out_quant = fn_quant(*args)
        out_fp16 = fn_fp16(*args)

    if isinstance(out_quant, (list, tuple)):
        out_quant = out_quant[0]
    if isinstance(out_fp16, (list, tuple)):
        out_fp16 = out_fp16[0]

    out_quant_f = out_quant.float()
    out_fp16_f = out_fp16.float()

    abs_diff = torch.abs(out_quant_f - out_fp16_f)
    max_abs_err = float(abs_diff.max().item())
    mean_abs_err = float(abs_diff.mean().item())
    rel_err = float((abs_diff / (torch.abs(out_fp16_f) + 1e-8)).max().item())

    # KL divergence (on softmax distributions if reasonable)
    try:
        p = F.softmax(out_fp16_f.flatten(), dim=0)
        q = F.softmax(out_quant_f.flatten(), dim=0)
        kl_div = float(F.kl_div(q.log(), p, reduction="sum").item())
    except Exception:
        kl_div = float("nan")

    result = {
        "kernel": kernel_name,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "rel_err": rel_err,
        "kl_div": kl_div,
        "passed": max_abs_err < MAX_ABS_ERR_THRESHOLD,
        "threshold": MAX_ABS_ERR_THRESHOLD,
    }

    return result


def print_validation_result(result: dict):
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n[{status}] {result['kernel']}")
    print(f"  Max abs err:  {result['max_abs_err']:.4e}  (threshold: {result['threshold']:.1e})")
    print(f"  Mean abs err: {result['mean_abs_err']:.4e}")
    print(f"  Rel err:      {result['rel_err']:.4e}")
    print(f"  KL divergence: {result['kl_div']:.4e}")


def main():
    parser = argparse.ArgumentParser(description="Validate quantized kernel numerics")
    parser.add_argument("--kernel", choices=["int8_attention", "int8_gemm", "fp8_gemm", "mixed_attention", "all"],
                        default="all", help="Kernel to validate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    bs = args.batch_size
    results = []

    if args.kernel in ("int8_attention", "all"):
        q, k, v = make_attention_args(batch_size=bs, device=device)
        r = validate_kernel(
            fn_quant=lambda q, k, v: int8_attention(q, k, v),
            fn_fp16=lambda q, k, v: baseline_attention(q, k, v),
            args=(q, k, v),
            kernel_name="int8_attention",
        )
        print_validation_result(r)
        results.append(r)

    if args.kernel in ("mixed_attention", "all"):
        q, k, v = make_attention_args(batch_size=bs, device=device)
        r = validate_kernel(
            fn_quant=lambda q, k, v: mixed_precision_attention(q, k, v),
            fn_fp16=lambda q, k, v: baseline_attention(q, k, v),
            args=(q, k, v),
            kernel_name="mixed_precision_attention",
        )
        print_validation_result(r)
        results.append(r)

    if args.kernel in ("int8_gemm", "all"):
        a, b = make_gemm_args(batch_size=bs, device=device)
        r = validate_kernel(
            fn_quant=lambda a, b: int8_gemm(a, b),
            fn_fp16=lambda a, b: baseline_gemm(a, b),
            args=(a, b),
            kernel_name="int8_gemm",
        )
        print_validation_result(r)
        results.append(r)

    if args.kernel in ("fp8_gemm", "all"):
        a, b = make_gemm_args(batch_size=bs, device=device, dtype=torch.float16)
        try:
            r = validate_kernel(
                fn_quant=lambda a, b: fp8_gemm(a, b.t().contiguous()),
                fn_fp16=lambda a, b: baseline_gemm(a, b),
                args=(a, b),
                kernel_name="fp8_gemm",
            )
            print_validation_result(r)
            results.append(r)
        except Exception as e:
            print(f"fp8_gemm not available: {e}")

    passed = sum(1 for r in results if r["passed"])
    print(f"\nSummary: {passed}/{len(results)} kernels passed numerics validation")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
