#!/usr/bin/env python3
"""
Hypothesis validator - callable by any branch to pre-validate a result
before it goes to the main eval harness.

Usage:
    python hypothesis_validator.py --hypothesis_id h001 --speedup 1.15 \
        --max_abs_err 0.00005 --batch_sizes 1 8 32 --variance_ratio 1.02

Or import and call directly:
    from hypothesis_validator import validate_claim
    result = validate_claim(hypothesis_id="h001", speedup=1.15, ...)
"""

import argparse
import json
import sys

SPEEDUP_THRESHOLD = 1.10
VARIANCE_THRESHOLD = 1.05
MAX_ABS_ERR_FP16 = 1e-4
REQUIRED_BATCH_SIZES = [1, 8, 32]


def validate_claim(
    hypothesis_id: str,
    speedup: float,
    max_abs_err: float,
    batch_sizes: list,
    variance_ratio: float = None,
    notes: str = "",
) -> dict:
    """
    Pre-validate a claim before submitting to eval harness.
    Returns dict with passed/failed breakdown.
    """
    checks = {
        "hypothesis_id": hypothesis_id,
        "speedup_ok": speedup >= SPEEDUP_THRESHOLD,
        "speedup_value": speedup,
        "speedup_threshold": SPEEDUP_THRESHOLD,
        "correctness_ok": max_abs_err < MAX_ABS_ERR_FP16,
        "max_abs_err": max_abs_err,
        "err_threshold": MAX_ABS_ERR_FP16,
        "batch_sizes_ok": all(bs in batch_sizes for bs in REQUIRED_BATCH_SIZES),
        "batch_sizes": batch_sizes,
        "required_batch_sizes": REQUIRED_BATCH_SIZES,
        "variance_ok": (variance_ratio is None) or (variance_ratio < VARIANCE_THRESHOLD),
        "variance_ratio": variance_ratio,
        "variance_threshold": VARIANCE_THRESHOLD,
        "notes": notes,
    }
    checks["passed"] = all([
        checks["speedup_ok"],
        checks["correctness_ok"],
        checks["batch_sizes_ok"],
        checks["variance_ok"],
    ])
    return checks


def print_claim_result(checks: dict):
    status = "PRE-VALIDATED" if checks["passed"] else "PRE-VALIDATION FAILED"
    print(f"\n[{status}] Hypothesis: {checks['hypothesis_id']}")
    print(f"  Speedup:       {checks['speedup_value']:.3f}x (need >={checks['speedup_threshold']}) {'OK' if checks['speedup_ok'] else 'FAIL'}")
    print(f"  Max abs err:   {checks['max_abs_err']:.2e} (need <{checks['err_threshold']:.0e}) {'OK' if checks['correctness_ok'] else 'FAIL'}")
    print(f"  Batch sizes:   {checks['batch_sizes']} (need {checks['required_batch_sizes']}) {'OK' if checks['batch_sizes_ok'] else 'FAIL'}")
    print(f"  Variance ratio: {checks['variance_ratio']} (need <{checks['variance_threshold']}) {'OK' if checks['variance_ok'] else 'FAIL'}")
    if checks.get("notes"):
        print(f"  Notes: {checks['notes']}")
    if checks["passed"]:
        print("  -> READY for eval harness certification")
    else:
        print("  -> Fix failures before submitting to eval harness")


def main():
    parser = argparse.ArgumentParser(description="Pre-validate a kernel optimization claim")
    parser.add_argument("--hypothesis_id", "-H", required=True)
    parser.add_argument("--speedup", "-s", type=float, required=True)
    parser.add_argument("--max_abs_err", "-e", type=float, required=True)
    parser.add_argument("--batch_sizes", "-b", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--variance_ratio", "-v", type=float, default=None)
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args()

    checks = validate_claim(
        hypothesis_id=args.hypothesis_id,
        speedup=args.speedup,
        max_abs_err=args.max_abs_err,
        batch_sizes=args.batch_sizes,
        variance_ratio=args.variance_ratio,
        notes=args.notes,
    )
    print_claim_result(checks)
    return 0 if checks["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
