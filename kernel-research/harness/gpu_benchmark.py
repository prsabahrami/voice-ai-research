#!/usr/bin/env python3
"""
gpu_benchmark.py - GPU benchmark runner for H100-specific kernels (Lambda deployment).

Usage:
    python gpu_benchmark.py                          # all kernels, batch_sizes [1,8,32]
    python gpu_benchmark.py --kernel triton_flash    # single kernel
    python gpu_benchmark.py --batch_size 8           # single batch size

Output schema per record:
{
  "hypothesis_id":          "gpu-triton-flash",
  "kernel_name":            "triton_flash_attention",
  "kernel_type":            "attention",
  "batch_size":             8,
  "baseline_name":          "sdpa",
  "baseline_p50_us":        123.4,
  "kernel_p50_us":          110.2,
  "speedup":                1.12,
  "correctness_max_abs_err": 0.000045,
  "peak_memory_bytes":      12345678,
  "device":                 "cuda:0",
  "passed_criteria":        true,
  "notes":                  "speedup 1.12x >= 1.10x OK; ...",
  "timestamp":              "2026-03-23T21:00:00Z"
}
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, Any, List, Optional

import torch

HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if HARNESS_DIR not in sys.path:
    sys.path.insert(0, HARNESS_DIR)

RESULTS_DIR  = os.path.join(os.path.dirname(HARNESS_DIR), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.jsonl")

from harness import (
    warmup,
    _time_fn_gpu,
    _time_fn_cpu,
    compute_latency_stats,
    correctness_check,
    WARMUP_ITERS,
    TIMED_ITERS,
)

from gpu_kernels import (
    TRITON_AVAILABLE,
    INT8_GEMM_AVAILABLE,
    triton_flash_attention,
    compiled_attention,
    fp16_gemm,
    int8_gemm_scaled,
    compiled_fused_softmax_cast,
)

from baseline_kernels import (
    naive_attention,
    standard_gemm,
    unfused_softmax_cast,
    make_attention_inputs,
    make_gemm_inputs,
    make_fused_ops_inputs,
)

from optimized_kernels import (
    tiled_attention,
    fused_softmax_cast,
)

import torch.nn.functional as F

SPEEDUP_THRESHOLD     = 1.10
MAX_ABS_ERR_FP16      = 1e-2
MAX_ABS_ERR_FP32      = 1e-4
VARIANCE_THRESHOLD    = 1.05

BATCH_SIZES = [1, 8, 32]


def _cuda_available() -> bool:
    return torch.cuda.is_available()

def _reset_cuda_memory() -> None:
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()

def _peak_cuda_memory_bytes() -> int:
    if _cuda_available():
        return torch.cuda.max_memory_allocated()
    return 0


def _benchmark_pair(
    baseline_fn: Callable,
    baseline_args: tuple,
    kernel_fn: Callable,
    kernel_args: tuple,
    use_gpu: bool,
) -> Dict[str, Any]:
    warmup(baseline_fn, *baseline_args)
    warmup(kernel_fn,   *kernel_args)
    time_fn = _time_fn_gpu if use_gpu else _time_fn_cpu
    baseline_times = time_fn(baseline_fn, *baseline_args)
    kernel_times   = time_fn(kernel_fn,   *kernel_args)
    return {
        "baseline": compute_latency_stats(baseline_times),
        "kernel":   compute_latency_stats(kernel_times),
    }


def _evaluate(
    baseline_p50_us: float,
    kernel_p50_us: float,
    max_err: float,
    p50_us: float,
    p99_us: float,
    max_err_threshold: float = MAX_ABS_ERR_FP32,
) -> tuple:
    parts  = []
    passed = True

    speedup = baseline_p50_us / kernel_p50_us if kernel_p50_us > 0 else 0.0
    if speedup >= SPEEDUP_THRESHOLD:
        parts.append(f"speedup {speedup:.3f}x >= {SPEEDUP_THRESHOLD}x OK")
    else:
        passed = False
        parts.append(f"speedup {speedup:.3f}x < {SPEEDUP_THRESHOLD}x FAIL")

    vr = p99_us / p50_us if p50_us > 0 else float("inf")
    if vr < VARIANCE_THRESHOLD:
        parts.append(f"variance p99/p50={vr:.3f} < {VARIANCE_THRESHOLD} OK")
    else:
        parts.append(f"variance p99/p50={vr:.3f} >= {VARIANCE_THRESHOLD} WARNING")

    if max_err <= max_err_threshold:
        parts.append(f"max_abs_err {max_err:.2e} <= {max_err_threshold:.0e} OK")
    else:
        passed = False
        parts.append(f"max_abs_err {max_err:.2e} > {max_err_threshold:.0e} FAIL")

    return passed, "; ".join(parts)


def _make_record(
    hypothesis_id: str,
    kernel_name: str,
    kernel_type: str,
    batch_size: int,
    baseline_name: str,
    stats: Dict[str, Any],
    max_err: float,
    peak_memory_bytes: int,
    device: torch.device,
    max_err_threshold: float = MAX_ABS_ERR_FP32,
) -> Dict[str, Any]:
    baseline_p50 = stats["baseline"]["p50_us"]
    kernel_p50   = stats["kernel"]["p50_us"]
    speedup      = baseline_p50 / kernel_p50 if kernel_p50 > 0 else 0.0

    passed, notes = _evaluate(
        baseline_p50, kernel_p50, max_err,
        stats["kernel"]["p50_us"], stats["kernel"]["p99_us"],
        max_err_threshold=max_err_threshold,
    )

    return {
        "hypothesis_id":            hypothesis_id,
        "kernel_name":              kernel_name,
        "kernel_type":              kernel_type,
        "batch_size":               batch_size,
        "baseline_name":            baseline_name,
        "baseline_p50_us":          round(baseline_p50, 4),
        "baseline_p90_us":          round(stats["baseline"]["p90_us"], 4),
        "baseline_p99_us":          round(stats["baseline"]["p99_us"], 4),
        "kernel_p50_us":            round(kernel_p50, 4),
        "kernel_p90_us":            round(stats["kernel"]["p90_us"], 4),
        "kernel_p99_us":            round(stats["kernel"]["p99_us"], 4),
        "speedup":                  round(speedup, 4),
        "correctness_max_abs_err":  round(max_err, 8),
        "peak_memory_bytes":        peak_memory_bytes,
        "device":                   str(device),
        "passed_criteria":          passed,
        "notes":                    notes,
        "timestamp":                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def bench_triton_flash_attention(batch_size, device, use_gpu):
    if not TRITON_AVAILABLE:
        return {"skipped": True, "reason": "Triton not available"}
    q, k, v = make_attention_inputs(batch_size, seq_len=128, num_heads=8, head_dim=64,
                                    dtype=torch.float16, device=device)
    def _sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)
    _reset_cuda_memory()
    stats = _benchmark_pair(_sdpa, (q, k, v), triton_flash_attention, (q, k, v), use_gpu)
    peak_mem = _peak_cuda_memory_bytes()
    ref_out    = _sdpa(q, k, v)
    triton_out = triton_flash_attention(q, k, v)
    max_err    = correctness_check(ref_out, triton_out)
    return _make_record("gpu-h001-triton-flash", "triton_flash_attention", "attention",
                        batch_size, "sdpa", stats, max_err, peak_mem, device, 1e-2)


def bench_compiled_attention(batch_size, device, use_gpu):
    q, k, v = make_attention_inputs(batch_size, seq_len=128, num_heads=8, head_dim=64,
                                    dtype=torch.float32, device=device)
    try:
        _ = compiled_attention(q, k, v)
        if use_gpu:
            torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": f"torch.compile failed: {exc}"}
    _reset_cuda_memory()
    stats = _benchmark_pair(naive_attention, (q, k, v), compiled_attention, (q, k, v), use_gpu)
    peak_mem = _peak_cuda_memory_bytes()
    ref_out  = naive_attention(q, k, v)
    comp_out = compiled_attention(q, k, v)
    max_err  = correctness_check(ref_out, comp_out)
    time_fn = _time_fn_gpu if use_gpu else _time_fn_cpu
    sdpa_stats = compute_latency_stats(time_fn(F.scaled_dot_product_attention, q, k, v))
    record = _make_record("gpu-h002-compiled-attn", "compiled_attention", "attention",
                          batch_size, "naive_attention", stats, max_err, peak_mem, device)
    record["sdpa_p50_us"] = round(sdpa_stats["p50_us"], 4)
    return record


def bench_fp16_gemm(batch_size, device, use_gpu):
    a_f32, b_f32 = make_gemm_inputs(batch_size, M=1024, K=1024, N=1024,
                                     dtype=torch.float32, device=device)
    a_f16 = a_f32.half()
    b_f16 = b_f32.half()
    def _fp32_gemm(a, b):
        return torch.matmul(a, b)
    _reset_cuda_memory()
    stats = _benchmark_pair(_fp32_gemm, (a_f32, b_f32), fp16_gemm, (a_f16, b_f16), use_gpu)
    peak_mem = _peak_cuda_memory_bytes()
    ref_out  = _fp32_gemm(a_f32, b_f32)
    fp16_out = fp16_gemm(a_f16, b_f16)
    max_err  = correctness_check(ref_out, fp16_out)
    return _make_record("gpu-h003-fp16-gemm", "fp16_gemm", "gemm",
                        batch_size, "standard_gemm_fp32", stats, max_err, peak_mem, device, 1e-1)


def bench_int8_gemm_scaled(batch_size, device, use_gpu):
    if not INT8_GEMM_AVAILABLE:
        return {"skipped": True, "reason": "torch._scaled_mm not available (PyTorch < 2.1)"}
    if not use_gpu:
        return {"skipped": True, "reason": "int8_gemm_scaled requires CUDA"}
    a_f32, b_f32 = make_gemm_inputs(batch_size, M=1024, K=1024, N=1024,
                                     dtype=torch.float32, device=device)
    a_f16 = a_f32.half()
    b_f16 = b_f32.half()
    try:
        _ = int8_gemm_scaled(a_f32, b_f32)
        torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": f"int8_gemm_scaled init failed: {exc}"}
    _reset_cuda_memory()
    stats = _benchmark_pair(fp16_gemm, (a_f16, b_f16), int8_gemm_scaled, (a_f32, b_f32), use_gpu)
    peak_mem = _peak_cuda_memory_bytes()
    ref_out  = torch.matmul(a_f32, b_f32)
    int8_out = int8_gemm_scaled(a_f32, b_f32)
    max_err  = correctness_check(ref_out, int8_out)
    return _make_record("gpu-h004-int8-gemm", "int8_gemm_scaled", "gemm",
                        batch_size, "fp16_gemm", stats, max_err, peak_mem, device, 1.0)


def bench_compiled_fused_softmax_cast(batch_size, device, use_gpu):
    x = make_fused_ops_inputs(batch_size, seq_len=4096, dtype=torch.float32, device=device)
    def _unfused(x):
        return unfused_softmax_cast(x, target_dtype=torch.float16)
    try:
        _ = compiled_fused_softmax_cast(x)
        if use_gpu:
            torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": f"torch.compile fused softmax failed: {exc}"}
    _reset_cuda_memory()
    stats = _benchmark_pair(_unfused, (x,), compiled_fused_softmax_cast, (x,), use_gpu)
    peak_mem = _peak_cuda_memory_bytes()
    ref_out   = _unfused(x)
    fused_out = compiled_fused_softmax_cast(x)
    max_err   = correctness_check(ref_out, fused_out)
    return _make_record("gpu-h005-compiled-fused-softmax", "compiled_fused_softmax_cast", "fused_ops",
                        batch_size, "unfused_softmax_cast", stats, max_err, peak_mem, device)


ALL_BENCHMARKS = {
    "triton_flash":            bench_triton_flash_attention,
    "compiled_attention":      bench_compiled_attention,
    "fp16_gemm":               bench_fp16_gemm,
    "int8_gemm":               bench_int8_gemm_scaled,
    "compiled_fused_softmax":  bench_compiled_fused_softmax_cast,
}


def run_all_benchmarks(kernel_filter=None, batch_sizes=None, write_results=True):
    use_gpu = _cuda_available()
    device  = torch.device("cuda" if use_gpu else "cpu")
    bs_list = batch_sizes or BATCH_SIZES

    if not use_gpu:
        print("WARNING: CUDA is not available. GPU benchmarks will fail or produce meaningless results.")

    benchmarks_to_run = (
        {kernel_filter: ALL_BENCHMARKS[kernel_filter]}
        if kernel_filter and kernel_filter in ALL_BENCHMARKS
        else ALL_BENCHMARKS
    )

    all_records = []

    for bench_name, bench_fn in benchmarks_to_run.items():
        print()
        print("=" * 60)
        print(f"Benchmark: {bench_name}")
        print("=" * 60)

        for bs in bs_list:
            print(f"  batch_size={bs} ...", end=" ", flush=True)
            try:
                record = bench_fn(bs, device, use_gpu)
            except Exception as exc:
                record = {
                    "kernel_name": bench_name, "batch_size": bs,
                    "error": str(exc), "skipped": True,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"ERROR: {exc}")
            else:
                if record.get("skipped"):
                    print(f"SKIPPED -- {record.get('reason', 'unknown')}")
                else:
                    speedup = record.get("speedup", 0.0)
                    passed  = record.get("passed_criteria", False)
                    print(f"speedup={speedup:.3f}x  p50={record.get('kernel_p50_us', 0):.1f}us  "
                          f"err={record.get('correctness_max_abs_err', -1):.2e}  "
                          f"{'PASS' if passed else 'FAIL'}")
            all_records.append(record)
            if write_results and not record.get("skipped"):
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(RESULTS_PATH, "a") as fh:
                    fh.write(json.dumps(record) + "\n")

    return all_records


def print_summary(records):
    print()
    print("=" * 70)
    print("GPU BENCHMARK SUMMARY")
    print("=" * 70)
    fmt = "{:<30} {:>6} {:>12} {:>12} {:>8} {:>6}"
    print(fmt.format("Kernel", "BS", "Baseline us", "Kernel us", "Speedup", "Pass"))
    print("-" * 70)
    for rec in records:
        if rec.get("skipped") or "error" in rec:
            name = rec.get("kernel_name", "?")[:28]
            print(f"  {name:<28} SKIPPED/ERROR")
            continue
        print(fmt.format(
            rec["kernel_name"][:30], rec["batch_size"],
            f"{rec['baseline_p50_us']:.1f}", f"{rec['kernel_p50_us']:.1f}",
            f"{rec['speedup']:.3f}x",
            "PASS" if rec["passed_criteria"] else "FAIL",
        ))
    print("=" * 70)
    print(f"Results written to: {RESULTS_PATH}")


def main():
    parser = argparse.ArgumentParser(description="H100 GPU kernel benchmarks (Lambda deployment)")
    parser.add_argument("--kernel", choices=list(ALL_BENCHMARKS.keys()), default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    batch_sizes = [args.batch_size] if args.batch_size else BATCH_SIZES

    print("GPU Benchmark Runner")
    print(f"Device:          {'cuda' if _cuda_available() else 'cpu (no CUDA!)'}")
    print(f"Triton:          {'available' if TRITON_AVAILABLE else 'NOT available'}")
    print(f"int8 scaled_mm:  {'available' if INT8_GEMM_AVAILABLE else 'NOT available'}")
    print(f"Batch sizes:     {batch_sizes}")

    records = run_all_benchmarks(
        kernel_filter=args.kernel, batch_sizes=batch_sizes, write_results=not args.no_write)
    print_summary(records)


if __name__ == "__main__":
    main()
