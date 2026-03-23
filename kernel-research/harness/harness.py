#!/usr/bin/env python3
"""
Shared benchmark harness for kernel optimization research sprint.
Owned by coolstufs. Used by all tracks.

Usage:
    from harness import benchmark_kernel

    result = benchmark_kernel(
        fn_optimized=my_kernel,
        fn_baseline=baseline_kernel,
        args=(x, y),
        n_warmup=10,
        n_timed=100,
        hypothesis_id="h001",
        kernel_type="GEMM",
        method="tiled GEMM with shared memory",
    )
"""

import time
import json
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Any, Tuple


RESULTS_PATH = Path("/home/ubuntu/kernel-research/results/results.jsonl")

# Success criteria
SPEEDUP_THRESHOLD = 1.10        # 10% improvement
VARIANCE_THRESHOLD = 1.05       # p99/p50 < 1.05
MAX_ABS_ERR_FP16 = 1e-4         # max absolute error for FP16
REQUIRED_BATCH_SIZES = [1, 8, 32]


def time_kernel_us(fn: Callable, args: Tuple, n_warmup: int = 10, n_timed: int = 100) -> List[float]:
    """Run fn(*args) and return list of latencies in microseconds."""
    device = None
    for a in args:
        if isinstance(a, torch.Tensor):
            device = a.device
            break

    # Warmup
    for _ in range(n_warmup):
        out = fn(*args)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times_us = []
    for _ in range(n_timed):
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = fn(*args)
            torch.cuda.synchronize()
        else:
            start = time.perf_counter()
            out = fn(*args)
        end = time.perf_counter()
        times_us.append((end - start) * 1e6)

    return times_us


def compute_percentiles(times_us: List[float]) -> dict:
    arr = np.array(times_us)
    return {
        "p50_us": float(np.percentile(arr, 50)),
        "p90_us": float(np.percentile(arr, 90)),
        "p99_us": float(np.percentile(arr, 99)),
        "mean_us": float(np.mean(arr)),
        "std_us": float(np.std(arr)),
        "variance_ratio": float(np.percentile(arr, 99) / max(np.percentile(arr, 50), 1e-9)),
    }


def check_correctness(
    fn_optimized: Callable,
    fn_baseline: Callable,
    args: Tuple,
    dtype: torch.dtype = torch.float16,
) -> dict:
    """Compare outputs and return max absolute error."""
    with torch.no_grad():
        out_opt = fn_optimized(*args)
        out_base = fn_baseline(*args)

    if isinstance(out_opt, (list, tuple)):
        out_opt = out_opt[0]
    if isinstance(out_base, (list, tuple)):
        out_base = out_base[0]

    out_opt = out_opt.float()
    out_base = out_base.float()

    max_abs_err = float(torch.max(torch.abs(out_opt - out_base)).item())
    rel_err = float(torch.max(torch.abs(out_opt - out_base) / (torch.abs(out_base) + 1e-8)).item())

    return {
        "max_abs_err": max_abs_err,
        "rel_err": rel_err,
        "passed_correctness": max_abs_err < MAX_ABS_ERR_FP16,
    }


def benchmark_kernel(
    fn_optimized: Callable,
    fn_baseline: Callable,
    args: Tuple,
    hypothesis_id: str,
    kernel_type: str,
    method: str,
    batch_sizes: Optional[List[int]] = None,
    n_warmup: int = 10,
    n_timed: int = 100,
    agent: str = "unknown",
    notes: str = "",
    make_args_for_batch: Optional[Callable] = None,
) -> dict:
    """
    Full benchmark: correctness check + timing across batch sizes.

    If make_args_for_batch is provided, it will be called as make_args_for_batch(batch_size)
    to get args for each batch size. Otherwise, uses the provided args for all batch sizes.
    """
    if batch_sizes is None:
        batch_sizes = REQUIRED_BATCH_SIZES

    # Correctness check
    corr = check_correctness(fn_optimized, fn_baseline, args)

    # Timing per batch size
    batch_results = {}
    all_passed = corr["passed_correctness"]

    for bs in batch_sizes:
        if make_args_for_batch is not None:
            bs_args = make_args_for_batch(bs)
        else:
            bs_args = args

        times_baseline = time_kernel_us(fn_baseline, bs_args, n_warmup, n_timed)
        times_opt = time_kernel_us(fn_optimized, bs_args, n_warmup, n_timed)

        stats_base = compute_percentiles(times_baseline)
        stats_opt = compute_percentiles(times_opt)

        speedup = stats_base["p50_us"] / max(stats_opt["p50_us"], 1e-9)
        passed_speedup = speedup >= SPEEDUP_THRESHOLD
        passed_variance = stats_opt["variance_ratio"] < VARIANCE_THRESHOLD

        batch_results[bs] = {
            "baseline": stats_base,
            "optimized": stats_opt,
            "speedup": speedup,
            "passed_speedup": passed_speedup,
            "passed_variance": passed_variance,
        }

        if not (passed_speedup and passed_variance):
            all_passed = False

    # Build result record
    best_batch = list(batch_results.keys())[0]
    result = {
        "hypothesis_id": hypothesis_id,
        "kernel_type": kernel_type,
        "method": method,
        "baseline_us": batch_results[best_batch]["baseline"]["p50_us"],
        "optimized_us": batch_results[best_batch]["optimized"]["p50_us"],
        "speedup": batch_results[best_batch]["speedup"],
        "correctness_max_abs_err": corr["max_abs_err"],
        "batch_sizes_tested": batch_sizes,
        "passed_criteria": all_passed and corr["passed_correctness"],
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "batch_results": batch_results,
        "correctness": corr,
    }

    # Write to results.jsonl
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "a") as f:
        # Write compact version (without full batch_results)
        compact = {k: v for k, v in result.items() if k not in ("batch_results",)}
        f.write(json.dumps(compact) + "\n")

    return result


def print_result(result: dict):
    """Pretty-print a benchmark result."""
    print(f"\n{'='*60}")
    print(f"Hypothesis: {result['hypothesis_id']} | {result['kernel_type']} | {result['method']}")
    print(f"Speedup: {result['speedup']:.3f}x | Max abs err: {result['correctness_max_abs_err']:.2e}")
    print(f"Passed criteria: {result['passed_criteria']}")
    print(f"Baseline: {result['baseline_us']:.1f} us | Optimized: {result['optimized_us']:.1f} us")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Harness loaded. Import benchmark_kernel to use.")
    print(f"Results will be written to: {RESULTS_PATH}")
    print(f"Success criteria: speedup>={SPEEDUP_THRESHOLD}x, variance_ratio<{VARIANCE_THRESHOLD}, max_abs_err<{MAX_ABS_ERR_FP16}")
