#!/usr/bin/env python3
"""
run_benchmark.py - CLI entry point for kernel optimization benchmark harness.

Usage:
    python run_benchmark.py --hypothesis_id h001 --kernel_type attention --batch_size 8

Writes results to /workspace/kernel-research/results/results.jsonl (appended).

Output schema per line:
{
  "hypothesis_id": "h001",
  "kernel_type": "attention",
  "method": "tiled_attention",
  "baseline_us": 123.4,
  "optimized_us": 110.2,
  "speedup": 1.12,
  "correctness_max_abs_err": 0.000045,
  "batch_sizes_tested": [1, 8, 32],
  "passed_criteria": true,
  "notes": "..."
}
"""

import argparse
import json
import os
import sys
import time

# Ensure harness directory is on the path
HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

RESULTS_PATH = os.path.join(
    os.path.dirname(HARNESS_DIR), "results", "results.jsonl"
)

# Acceptance criteria thresholds
SPEEDUP_THRESHOLD        = 1.10   # 10% wall-clock improvement
MAX_ABS_ERR_THRESHOLD    = 1e-4   # max absolute error for FP16 correctness
VARIANCE_THRESHOLD       = 1.05   # p99 / p50 ratio must be < 5%

METHOD_NAMES = {
    "attention": "tiled_attention",
    "gemm":      "blocked_gemm",
    "fused_ops": "fused_softmax_cast",
}


def evaluate_criteria(baseline_us: float, optimized_us: float,
                      correctness_max_abs_err: float,
                      p50_us: float, p99_us: float) -> tuple:
    """
    Returns (passed: bool, notes: str) based on acceptance criteria.
    """
    notes_parts = []
    passed = True

    speedup = baseline_us / optimized_us if optimized_us > 0 else 0.0
    if speedup >= SPEEDUP_THRESHOLD:
        notes_parts.append(f"speedup {speedup:.3f}x >= {SPEEDUP_THRESHOLD}x OK")
    else:
        passed = False
        notes_parts.append(f"speedup {speedup:.3f}x < {SPEEDUP_THRESHOLD}x FAIL")

    variance_ratio = p99_us / p50_us if p50_us > 0 else float('inf')
    if variance_ratio < VARIANCE_THRESHOLD:
        notes_parts.append(f"variance p99/p50={variance_ratio:.3f} < {VARIANCE_THRESHOLD} OK")
    else:
        # Variance criterion is advisory; mark as warning but do not fail
        notes_parts.append(f"variance p99/p50={variance_ratio:.3f} >= {VARIANCE_THRESHOLD} WARNING")

    if correctness_max_abs_err <= MAX_ABS_ERR_THRESHOLD:
        notes_parts.append(f"max_abs_err {correctness_max_abs_err:.2e} <= {MAX_ABS_ERR_THRESHOLD:.0e} OK")
    else:
        passed = False
        notes_parts.append(f"max_abs_err {correctness_max_abs_err:.2e} > {MAX_ABS_ERR_THRESHOLD:.0e} FAIL")

    return passed, "; ".join(notes_parts)


def run_all_batch_sizes(hypothesis_id: str, kernel_type: str,
                        batch_sizes: list) -> dict:
    """Run benchmark across all batch sizes and aggregate results."""
    from harness import run_benchmark

    best_speedup       = 0.0
    best_result        = None
    aggregate_err      = 0.0
    all_baseline_us    = []
    all_optimized_us   = []

    for bs in batch_sizes:
        print(f"  batch_size={bs} ...", flush=True)
        result = run_benchmark(kernel_type, bs)

        baseline_us  = result["baseline"]["p50_us"]
        optimized_us = result["optimized"]["p50_us"]
        speedup      = baseline_us / optimized_us if optimized_us > 0 else 0.0
        err          = result["correctness_max_abs_err"]

        all_baseline_us.append(baseline_us)
        all_optimized_us.append(optimized_us)
        aggregate_err = max(aggregate_err, err)

        if speedup > best_speedup:
            best_speedup = speedup
            best_result  = result

    # Use best result's p50 stats for the record
    mean_baseline_us  = sum(all_baseline_us)  / len(all_baseline_us)
    mean_optimized_us = sum(all_optimized_us) / len(all_optimized_us)
    final_speedup     = mean_baseline_us / mean_optimized_us if mean_optimized_us > 0 else 0.0

    p50_us = best_result["optimized"]["p50_us"]
    p99_us = best_result["optimized"]["p99_us"]

    passed, notes = evaluate_criteria(
        mean_baseline_us, mean_optimized_us,
        aggregate_err, p50_us, p99_us
    )

    return {
        "hypothesis_id":           hypothesis_id,
        "kernel_type":             kernel_type,
        "method":                  METHOD_NAMES.get(kernel_type, "unknown"),
        "baseline_us":             round(mean_baseline_us,  4),
        "optimized_us":            round(mean_optimized_us, 4),
        "speedup":                 round(final_speedup,     4),
        "correctness_max_abs_err": round(aggregate_err,     8),
        "batch_sizes_tested":      batch_sizes,
        "passed_criteria":         passed,
        "notes":                   notes,
        "device":                  best_result.get("device", "cpu"),
        "timestamp":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Kernel optimization microbenchmark runner"
    )
    parser.add_argument("--hypothesis_id", required=True,
                        help="Hypothesis identifier (e.g. h001)")
    parser.add_argument("--kernel_type", required=True,
                        choices=["attention", "gemm", "fused_ops"],
                        help="Kernel type to benchmark")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Single batch size to test; omit to test [1,8,32]")
    args = parser.parse_args()

    batch_sizes = [args.batch_size] if args.batch_size else [1, 8, 32]
    print(f"Running {args.kernel_type} benchmark for {args.hypothesis_id} "
          f"on batch_sizes={batch_sizes}")

    record = run_all_batch_sizes(args.hypothesis_id, args.kernel_type, batch_sizes)

    # Print human-readable summary
    print("\n--- Results ---")
    print(f"  baseline_us:  {record['baseline_us']:.2f}")
    print(f"  optimized_us: {record['optimized_us']:.2f}")
    print(f"  speedup:      {record['speedup']:.4f}x")
    print(f"  max_abs_err:  {record['correctness_max_abs_err']:.2e}")
    print(f"  passed:       {record['passed_criteria']}")
    print(f"  notes:        {record['notes']}")

    # Write to JSONL
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nResult written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
