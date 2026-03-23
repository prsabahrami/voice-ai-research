"""
benchmark_quant.py
==================
Full benchmark harness for INT8/FP8/mixed-precision quantization kernels.

Execution model:
  - 10 warmup iterations (not timed)
  - 100 timed iterations
  - Reports p50, p90, p99 latency in microseconds
  - Tests:
      * GEMM: M=N=K=2048
      * Attention: B in [1,8,32], S=512, D=64

Results written as JSONL to /workspace/kernel-research/results/quant_results.jsonl.

Author: miniQuant kernel-research branch
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

QUANT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = Path("/workspace/kernel-research/results")
RESULTS_FILE = RESULTS_DIR / "quant_results.jsonl"

sys.path.insert(0, str(QUANT_DIR))

from quant_kernels import (
    CUDA_AVAILABLE,
    FP8_SUPPORTED,
    SCALED_MM_AVAILABLE,
    DEVICE,
    fp16_gemm,
    fp16_attention,
    int8_gemm,
    int8_attention,
    fp8_gemm,
    fp8_attention,
    mixed_precision_gemm,
)
from numerics_validator import validate_quantized_output

WARMUP_ITERS = 10
TIMED_ITERS = 100
BATCH_SIZES = [1, 8, 32]
SEQ_LEN = 512
D_MODEL = 64
GEMM_DIM = 2048

SPEEDUP_THRESHOLD = 1.10
MAX_ABS_ERR_THRESHOLD = 1e-4
VARIANCE_THRESHOLD = 0.05


def _timed_run(fn, warmup=WARMUP_ITERS, repeats=TIMED_ITERS):
    result = None
    if CUDA_AVAILABLE:
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        times_us = []
        for _ in range(repeats):
            start_ev.record()
            result = fn()
            end_ev.record()
            torch.cuda.synchronize()
            times_us.append(start_ev.elapsed_time(end_ev) * 1000.0)
    else:
        for _ in range(warmup):
            result = fn()
        times_us = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            times_us.append((t1 - t0) * 1e6)
    return times_us, result


def _percentiles(times_us):
    arr = sorted(times_us)
    n = len(arr)
    return {
        "p50_us": arr[n // 2],
        "p90_us": arr[int(n * 0.90)],
        "p99_us": arr[int(n * 0.99)],
        "mean_us": sum(arr) / n,
        "min_us": arr[0],
        "max_us": arr[-1],
    }


def benchmark_gemm(hypothesis_id="quant-gemm"):
    records = []
    M = N = K = GEMM_DIM
    dev = DEVICE

    print("")
    print(f"--- GEMM benchmarks (M=N=K={M}) ---")
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)

    print("  Running fp16_gemm baseline...")
    base_times, base_out = _timed_run(lambda: fp16_gemm(A, B)[0])
    base_stats = _percentiles(base_times)
    baseline_us = base_stats["p50_us"]
    print(f"    fp16_gemm p50={baseline_us:.1f}us")

    kernels = [
        ("int8_gemm",            lambda: int8_gemm(A, B)),
        ("fp8_gemm",             lambda: fp8_gemm(A, B)),
        ("mixed_precision_gemm", lambda: mixed_precision_gemm(A, B)),
    ]

    for method, fn in kernels:
        print(f"  Running {method}...")
        try:
            times, out = _timed_run(fn)
            stats = _percentiles(times)
            opt_us = stats["p50_us"]
            speedup = baseline_us / opt_us

            val = validate_quantized_output(base_out, out, threshold_max_abs=MAX_ABS_ERR_THRESHOLD)
            max_abs_err = val["max_abs_error"]
            variance_ratio = stats["p99_us"] / stats["p50_us"] - 1.0

            passed = speedup >= SPEEDUP_THRESHOLD and val["passed"] and variance_ratio <= VARIANCE_THRESHOLD
            notes_parts = []
            if speedup < SPEEDUP_THRESHOLD:
                notes_parts.append(f"speedup {speedup:.2f}x below 1.10x threshold")
            if not val["passed"]:
                notes_parts.append(f"max_abs_err {max_abs_err:.2e} exceeds {MAX_ABS_ERR_THRESHOLD:.0e}")
            if variance_ratio > VARIANCE_THRESHOLD:
                notes_parts.append(f"timing variance {variance_ratio*100:.1f}% exceeds 5%")
            if not notes_parts:
                notes_parts.append("all criteria met")

            record = {
                "hypothesis_id": hypothesis_id,
                "kernel_type": "gemm",
                "method": method,
                "baseline_us": round(baseline_us, 2),
                "optimized_us": round(opt_us, 2),
                "speedup": round(speedup, 4),
                "correctness_max_abs_err": round(max_abs_err, 8),
                "batch_sizes_tested": [1],
                "passed_criteria": passed,
                "notes": "; ".join(notes_parts),
                "p50_us": round(stats["p50_us"], 2),
                "p90_us": round(stats["p90_us"], 2),
                "p99_us": round(stats["p99_us"], 2),
                "mean_us": round(stats["mean_us"], 2),
            }
            records.append(record)
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {method}: speedup={speedup:.3f}x  p50={opt_us:.1f}us  max_abs_err={max_abs_err:.2e}")
        except Exception as e:
            records.append({
                "hypothesis_id": hypothesis_id,
                "kernel_type": "gemm",
                "method": method,
                "baseline_us": round(baseline_us, 2),
                "optimized_us": None,
                "speedup": None,
                "correctness_max_abs_err": None,
                "batch_sizes_tested": [1],
                "passed_criteria": False,
                "notes": f"ERROR: {e}",
            })
            print(f"    [ERROR] {method}: {e}")
    return records


def benchmark_attention(hypothesis_id="quant-attention"):
    records = []
    dev = DEVICE
    S = SEQ_LEN
    D = D_MODEL

    print("")
    print(f"--- Attention benchmarks (S={S}, D={D}) ---")

    for B in BATCH_SIZES:
        print("")
        print(f"  Batch size {B}:")
        Q = torch.randn(B, S, D, device=dev)
        K = torch.randn(B, S, D, device=dev)
        V = torch.randn(B, S, D, device=dev)

        base_times, base_out = _timed_run(lambda: fp16_attention(Q, K, V)[0])
        base_stats = _percentiles(base_times)
        baseline_us = base_stats["p50_us"]
        print(f"    fp16_attention p50={baseline_us:.1f}us")

        kernels = [
            ("int8_attention",  lambda: int8_attention(Q, K, V)),
            ("fp8_attention",   lambda: fp8_attention(Q, K, V)),
        ]

        for method, fn in kernels:
            print(f"    Running {method}...")
            try:
                times, out = _timed_run(fn)
                stats = _percentiles(times)
                opt_us = stats["p50_us"]
                speedup = baseline_us / opt_us

                val = validate_quantized_output(base_out, out, threshold_max_abs=MAX_ABS_ERR_THRESHOLD)
                max_abs_err = val["max_abs_error"]
                variance_ratio = stats["p99_us"] / stats["p50_us"] - 1.0

                passed = speedup >= SPEEDUP_THRESHOLD and val["passed"] and variance_ratio <= VARIANCE_THRESHOLD
                notes_parts = []
                if speedup < SPEEDUP_THRESHOLD:
                    notes_parts.append(f"speedup {speedup:.2f}x below 1.10x threshold")
                if not val["passed"]:
                    notes_parts.append(f"max_abs_err {max_abs_err:.2e} exceeds {MAX_ABS_ERR_THRESHOLD:.0e}")
                if variance_ratio > VARIANCE_THRESHOLD:
                    notes_parts.append(f"timing variance {variance_ratio*100:.1f}% exceeds 5%")
                if not notes_parts:
                    notes_parts.append("all criteria met")

                record = {
                    "hypothesis_id": f"{hypothesis_id}-B{B}",
                    "kernel_type": "attention",
                    "method": method,
                    "baseline_us": round(baseline_us, 2),
                    "optimized_us": round(opt_us, 2),
                    "speedup": round(speedup, 4),
                    "correctness_max_abs_err": round(max_abs_err, 8),
                    "batch_sizes_tested": [B],
                    "passed_criteria": passed,
                    "notes": "; ".join(notes_parts),
                    "p50_us": round(stats["p50_us"], 2),
                    "p90_us": round(stats["p90_us"], 2),
                    "p99_us": round(stats["p99_us"], 2),
                    "mean_us": round(stats["mean_us"], 2),
                    "seq_len": S,
                    "d_model": D,
                    "batch_size": B,
                }
                records.append(record)
                status = "PASS" if passed else "FAIL"
                print(f"      [{status}] {method} B={B}: speedup={speedup:.3f}x  p50={opt_us:.1f}us  max_abs_err={max_abs_err:.2e}")
            except Exception as e:
                records.append({
                    "hypothesis_id": f"{hypothesis_id}-B{B}",
                    "kernel_type": "attention",
                    "method": method,
                    "baseline_us": round(baseline_us, 2),
                    "optimized_us": None,
                    "speedup": None,
                    "correctness_max_abs_err": None,
                    "batch_sizes_tested": [B],
                    "passed_criteria": False,
                    "notes": f"ERROR: {e}",
                    "batch_size": B,
                })
                print(f"      [ERROR] {method} B={B}: {e}")
    return records


def write_results(records):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nResults written to {RESULTS_FILE} ({len(records)} records)")


def main():
    print("=" * 60)
    print("miniQuant Benchmark Harness")
    print(f"Device: {DEVICE}")
    if CUDA_AVAILABLE:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory // 2**30} GB)")
        print(f"Compute capability: {props.major}.{props.minor}")
    print(f"FP8 supported: {FP8_SUPPORTED}")
    print(f"torch._scaled_mm available: {SCALED_MM_AVAILABLE}")
    print("=" * 60)

    all_records = []
    all_records.extend(benchmark_gemm(hypothesis_id="H100-quant-gemm"))
    all_records.extend(benchmark_attention(hypothesis_id="H100-quant-attn"))
    write_results(all_records)

    print("")
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for r in all_records if r.get("passed_criteria"))
    print(f"Kernels tested: {len(all_records)}  Passed: {passed_count}  Failed: {len(all_records) - passed_count}")
    print()
    header = f"{'method':<30} {'speedup':>8} {'p50_us':>8} {'max_abs_err':>14} {'pass':>5}"
    print(header)
    print("-" * len(header))
    for r in all_records:
        su = f"{r['speedup']:.3f}x" if r.get("speedup") else "N/A"
        p50 = f"{r['p50_us']:.1f}" if r.get("p50_us") else "N/A"
        err = f"{r['correctness_max_abs_err']:.2e}" if r.get("correctness_max_abs_err") is not None else "N/A"
        ok = "Y" if r.get("passed_criteria") else "N"
        tag = r.get("method", "?") + (f" B={r['batch_size']}" if "batch_size" in r else "")
        print(f"{tag:<30} {su:>8} {p50:>8} {err:>14} {ok:>5}")


if __name__ == "__main__":
    main()
