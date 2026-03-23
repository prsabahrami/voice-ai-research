#!/usr/bin/env python3
"""
Eval harness owned by serious-inference-engineer.
This is the gatekeeper for all claimed results.

Any branch that wants to certify a result must call hypothesis_validator.py first.
Final results in results.jsonl are only considered valid after passing this harness.

Usage:
    python eval_harness.py --run_all                    # validate all results in results.jsonl
    python eval_harness.py --hypothesis_id h001          # validate specific hypothesis
    python eval_harness.py --certify h001                # certify h001 as production-ready
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
from harness import (
    SPEEDUP_THRESHOLD,
    VARIANCE_THRESHOLD,
    MAX_ABS_ERR_FP16,
    REQUIRED_BATCH_SIZES,
)

RESULTS_PATH = Path("/home/ubuntu/kernel-research/results/results.jsonl")
CERTIFIED_PATH = Path("/home/ubuntu/kernel-research/results/certified.jsonl")


def load_results(hypothesis_id=None):
    if not RESULTS_PATH.exists():
        return []
    results = []
    with open(RESULTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if hypothesis_id is None or r.get("hypothesis_id") == hypothesis_id:
                        results.append(r)
                except json.JSONDecodeError:
                    pass
    return results


def validate_result(result: dict) -> dict:
    """
    Re-validate a result record against all success criteria.
    Returns the record with a validation summary appended.
    """
    checks = {}

    # Speedup
    speedup = result.get("speedup", 0)
    checks["speedup_ok"] = speedup >= SPEEDUP_THRESHOLD
    checks["speedup_value"] = speedup

    # Variance (p99/p50 ratio)
    # If we have batch_results, check per-batch variance
    variance_ratio = None
    optimized_us = result.get("optimized_us", 0)
    if "batch_results" in result:
        for bs, br in result["batch_results"].items():
            opt_stats = br.get("optimized", {})
            p50 = opt_stats.get("p50_us", 1)
            p99 = opt_stats.get("p99_us", 1)
            ratio = p99 / max(p50, 1e-9)
            if variance_ratio is None or ratio > variance_ratio:
                variance_ratio = ratio
    checks["variance_ok"] = (variance_ratio is None) or (variance_ratio < VARIANCE_THRESHOLD)
    checks["variance_ratio"] = variance_ratio

    # Correctness
    max_abs_err = result.get("correctness_max_abs_err", float("inf"))
    checks["correctness_ok"] = max_abs_err < MAX_ABS_ERR_FP16
    checks["max_abs_err"] = max_abs_err

    # Batch sizes
    batch_sizes_tested = result.get("batch_sizes_tested", [])
    required_covered = all(bs in batch_sizes_tested for bs in REQUIRED_BATCH_SIZES)
    checks["batch_sizes_ok"] = required_covered
    checks["batch_sizes_tested"] = batch_sizes_tested

    # Overall pass
    checks["passed_all"] = all([
        checks["speedup_ok"],
        checks["variance_ok"],
        checks["correctness_ok"],
        checks["batch_sizes_ok"],
    ])

    return checks


def print_validation(result: dict, checks: dict):
    hid = result.get("hypothesis_id", "?")
    method = result.get("method", "?")
    status = "CERTIFIED" if checks["passed_all"] else "FAILED"
    print(f"\n[{status}] {hid}: {method}")
    print(f"  Speedup:     {checks['speedup_value']:.3f}x  (need >= {SPEEDUP_THRESHOLD})  {'OK' if checks['speedup_ok'] else 'FAIL'}")
    print(f"  Variance:    {checks['variance_ratio']}  (need < {VARIANCE_THRESHOLD})  {'OK' if checks['variance_ok'] else 'FAIL'}")
    print(f"  Max abs err: {checks['max_abs_err']:.2e}  (need < {MAX_ABS_ERR_FP16:.0e})  {'OK' if checks['correctness_ok'] else 'FAIL'}")
    print(f"  Batch sizes: {checks['batch_sizes_tested']}  (need {REQUIRED_BATCH_SIZES})  {'OK' if checks['batch_sizes_ok'] else 'FAIL'}")


def certify_result(result: dict, checks: dict):
    """Write a certified result to certified.jsonl."""
    CERTIFIED_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **result,
        "certified_at": datetime.now(timezone.utc).isoformat(),
        "validation_checks": checks,
        "certified_by": "serious-inference-engineer",
    }
    with open(CERTIFIED_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  -> Certified and written to {CERTIFIED_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Eval harness for kernel optimization results")
    parser.add_argument("--run_all", action="store_true", help="Validate all results")
    parser.add_argument("--hypothesis_id", type=str, help="Validate specific hypothesis")
    parser.add_argument("--certify", type=str, help="Certify a passing hypothesis")
    args = parser.parse_args()

    if args.certify:
        results = load_results(args.certify)
        if not results:
            print(f"No results found for {args.certify}")
            sys.exit(1)
        # Use the latest result
        result = results[-1]
        checks = validate_result(result)
        print_validation(result, checks)
        if checks["passed_all"]:
            certify_result(result, checks)
        else:
            print("  -> Cannot certify: criteria not met")
            sys.exit(1)
        return

    if args.run_all or args.hypothesis_id:
        results = load_results(args.hypothesis_id)
        if not results:
            print("No results found")
            sys.exit(0)

        passed = 0
        for result in results:
            checks = validate_result(result)
            print_validation(result, checks)
            if checks["passed_all"]:
                passed += 1

        print(f"\nSummary: {passed}/{len(results)} results pass all criteria")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
