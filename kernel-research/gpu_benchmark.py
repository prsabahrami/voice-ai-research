#!/usr/bin/env python3
"""
gpu_benchmark.py - Standalone GPU benchmark runner for H100-specific kernels.

Top-level entry point for Lambda deployment. Imports kernel implementations
from kernel-research/harness/ and runs all GPU benchmarks.

Usage (from kernel-research/ directory on Lambda):
    python gpu_benchmark.py
    python gpu_benchmark.py --kernel triton_flash
    python gpu_benchmark.py --batch_size 8
    python gpu_benchmark.py --no-write

Prerequisites: PyTorch >= 2.1 with CUDA, Triton (for triton_flash), harness/ directory.
"""

import argparse
import json
import os
import sys
import time

import torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.join(SCRIPT_DIR, "harness")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "results.jsonl")

for d in [HARNESS_DIR, SCRIPT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from harness import (
    warmup, _time_fn_gpu, _time_fn_cpu, compute_latency_stats,
    correctness_check, WARMUP_ITERS, TIMED_ITERS,
)
from gpu_kernels import (
    TRITON_AVAILABLE, INT8_GEMM_AVAILABLE,
    triton_flash_attention, compiled_attention,
    fp16_gemm, int8_gemm_scaled, compiled_fused_softmax_cast,
)
from baseline_kernels import (
    naive_attention, standard_gemm, unfused_softmax_cast,
    make_attention_inputs, make_gemm_inputs, make_fused_ops_inputs,
)
import torch.nn.functional as F

SPEEDUP_THRESHOLD  = 1.10
VARIANCE_THRESHOLD = 1.05
BATCH_SIZES = [1, 8, 32]


def _cuda_available():
    return torch.cuda.is_available()

def _reset_cuda_memory():
    if _cuda_available():
        torch.cuda.reset_peak_memory_stats()

def _peak_cuda_memory_bytes():
    return torch.cuda.max_memory_allocated() if _cuda_available() else 0

def _benchmark_pair(baseline_fn, baseline_args, kernel_fn, kernel_args, use_gpu):
    warmup(baseline_fn, *baseline_args)
    warmup(kernel_fn, *kernel_args)
    time_fn = _time_fn_gpu if use_gpu else _time_fn_cpu
    return {
        "baseline": compute_latency_stats(time_fn(baseline_fn, *baseline_args)),
        "kernel":   compute_latency_stats(time_fn(kernel_fn, *kernel_args)),
    }

def _evaluate(baseline_p50, kernel_p50, max_err, p50, p99, max_err_threshold):
    parts = []
    passed = True
    speedup = baseline_p50 / kernel_p50 if kernel_p50 > 0 else 0.0
    if speedup >= SPEEDUP_THRESHOLD:
        parts.append("speedup %.3fx >= %.1fx OK" % (speedup, SPEEDUP_THRESHOLD))
    else:
        passed = False
        parts.append("speedup %.3fx < %.1fx FAIL" % (speedup, SPEEDUP_THRESHOLD))
    vr = p99 / p50 if p50 > 0 else float("inf")
    parts.append("variance p99/p50=%.3f %s" % (vr, "OK" if vr < VARIANCE_THRESHOLD else "WARNING"))
    if max_err <= max_err_threshold:
        parts.append("max_abs_err %.2e <= %.0e OK" % (max_err, max_err_threshold))
    else:
        passed = False
        parts.append("max_abs_err %.2e > %.0e FAIL" % (max_err, max_err_threshold))
    return passed, "; ".join(parts)

def _make_record(hyp_id, kernel_name, kernel_type, batch_size, baseline_name,
                  stats, max_err, peak_mem, device, max_err_threshold=1e-4):
    baseline_p50 = stats["baseline"]["p50_us"]
    kernel_p50   = stats["kernel"]["p50_us"]
    speedup      = baseline_p50 / kernel_p50 if kernel_p50 > 0 else 0.0
    passed, notes = _evaluate(baseline_p50, kernel_p50, max_err,
                               kernel_p50, stats["kernel"]["p99_us"], max_err_threshold)
    return {
        "hypothesis_id":           hyp_id,
        "kernel_name":             kernel_name,
        "kernel_type":             kernel_type,
        "batch_size":              batch_size,
        "baseline_name":           baseline_name,
        "baseline_p50_us":         round(baseline_p50, 4),
        "baseline_p90_us":         round(stats["baseline"]["p90_us"], 4),
        "baseline_p99_us":         round(stats["baseline"]["p99_us"], 4),
        "kernel_p50_us":           round(kernel_p50, 4),
        "kernel_p90_us":           round(stats["kernel"]["p90_us"], 4),
        "kernel_p99_us":           round(stats["kernel"]["p99_us"], 4),
        "speedup":                 round(speedup, 4),
        "correctness_max_abs_err": round(max_err, 8),
        "peak_memory_bytes":       peak_mem,
        "device":                  str(device),
        "passed_criteria":         passed,
        "notes":                   notes,
        "timestamp":               time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def bench_triton_flash_attention(batch_size, device, use_gpu):
    if not TRITON_AVAILABLE:
        return {"skipped": True, "reason": "Triton not available"}
    q, k, v = make_attention_inputs(batch_size, seq_len=128, num_heads=8, head_dim=64,
                                     dtype=torch.float16, device=device)
    def sdpa(q, k, v):
        return F.scaled_dot_product_attention(q, k, v)
    _reset_cuda_memory()
    stats = _benchmark_pair(sdpa, (q, k, v), triton_flash_attention, (q, k, v), use_gpu)
    max_err = correctness_check(sdpa(q, k, v), triton_flash_attention(q, k, v))
    return _make_record("gpu-h001-triton-flash", "triton_flash_attention", "attention",
                         batch_size, "sdpa", stats, max_err, _peak_cuda_memory_bytes(), device, 1e-2)


def bench_compiled_attention(batch_size, device, use_gpu):
    q, k, v = make_attention_inputs(batch_size, seq_len=128, num_heads=8, head_dim=64,
                                     dtype=torch.float32, device=device)
    try:
        _ = compiled_attention(q, k, v)
        if use_gpu:
            torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": "torch.compile failed: %s" % exc}
    _reset_cuda_memory()
    stats = _benchmark_pair(naive_attention, (q, k, v), compiled_attention, (q, k, v), use_gpu)
    max_err = correctness_check(naive_attention(q, k, v), compiled_attention(q, k, v))
    record = _make_record("gpu-h002-compiled-attn", "compiled_attention", "attention",
                           batch_size, "naive_attention", stats, max_err,
                           _peak_cuda_memory_bytes(), device, 1e-4)
    time_fn = _time_fn_gpu if use_gpu else _time_fn_cpu
    record["sdpa_p50_us"] = round(
        compute_latency_stats(time_fn(F.scaled_dot_product_attention, q, k, v))["p50_us"], 4
    )
    return record


def bench_fp16_gemm(batch_size, device, use_gpu):
    a_f32, b_f32 = make_gemm_inputs(batch_size, M=1024, K=1024, N=1024,
                                      dtype=torch.float32, device=device)
    a_f16, b_f16 = a_f32.half(), b_f32.half()
    def fp32_gemm(a, b):
        return torch.matmul(a, b)
    _reset_cuda_memory()
    stats = _benchmark_pair(fp32_gemm, (a_f32, b_f32), fp16_gemm, (a_f16, b_f16), use_gpu)
    max_err = correctness_check(fp32_gemm(a_f32, b_f32), fp16_gemm(a_f16, b_f16))
    return _make_record("gpu-h003-fp16-gemm", "fp16_gemm", "gemm", batch_size,
                         "standard_gemm_fp32", stats, max_err, _peak_cuda_memory_bytes(), device, 1e-1)


def bench_int8_gemm_scaled(batch_size, device, use_gpu):
    if not INT8_GEMM_AVAILABLE:
        return {"skipped": True, "reason": "torch._scaled_mm not available (PyTorch < 2.1)"}
    if not use_gpu:
        return {"skipped": True, "reason": "int8_gemm_scaled requires CUDA"}
    a_f32, b_f32 = make_gemm_inputs(batch_size, M=1024, K=1024, N=1024,
                                      dtype=torch.float32, device=device)
    a_f16, b_f16 = a_f32.half(), b_f32.half()
    try:
        _ = int8_gemm_scaled(a_f32, b_f32)
        torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": "int8_gemm_scaled init failed: %s" % exc}
    _reset_cuda_memory()
    stats = _benchmark_pair(fp16_gemm, (a_f16, b_f16), int8_gemm_scaled, (a_f32, b_f32), use_gpu)
    max_err = correctness_check(torch.matmul(a_f32, b_f32), int8_gemm_scaled(a_f32, b_f32))
    return _make_record("gpu-h004-int8-gemm", "int8_gemm_scaled", "gemm", batch_size,
                         "fp16_gemm", stats, max_err, _peak_cuda_memory_bytes(), device, 1.0)


def bench_compiled_fused_softmax_cast(batch_size, device, use_gpu):
    x = make_fused_ops_inputs(batch_size, seq_len=4096, dtype=torch.float32, device=device)
    def unfused(x):
        return unfused_softmax_cast(x, target_dtype=torch.float16)
    try:
        _ = compiled_fused_softmax_cast(x)
        if use_gpu:
            torch.cuda.synchronize()
    except Exception as exc:
        return {"skipped": True, "reason": "torch.compile fused softmax failed: %s" % exc}
    _reset_cuda_memory()
    stats = _benchmark_pair(unfused, (x,), compiled_fused_softmax_cast, (x,), use_gpu)
    max_err = correctness_check(unfused(x), compiled_fused_softmax_cast(x))
    return _make_record("gpu-h005-compiled-fused-softmax", "compiled_fused_softmax_cast",
                         "fused_ops", batch_size, "unfused_softmax_cast",
                         stats, max_err, _peak_cuda_memory_bytes(), device, 1e-4)


ALL_BENCHMARKS = {
    "triton_flash":           bench_triton_flash_attention,
    "compiled_attention":     bench_compiled_attention,
    "fp16_gemm":              bench_fp16_gemm,
    "int8_gemm":              bench_int8_gemm_scaled,
    "compiled_fused_softmax": bench_compiled_fused_softmax_cast,
}


def run_all_benchmarks(kernel_filter=None, batch_sizes=None, write_results=True):
    use_gpu = _cuda_available()
    device  = torch.device("cuda" if use_gpu else "cpu")
    bs_list = batch_sizes or BATCH_SIZES
    if not use_gpu:
        print("WARNING: CUDA not available. GPU benchmarks require Lambda H100 instance.")
    benchmarks_to_run = (
        {kernel_filter: ALL_BENCHMARKS[kernel_filter]}
        if kernel_filter and kernel_filter in ALL_BENCHMARKS else ALL_BENCHMARKS
    )
    all_records = []
    for bench_name, bench_fn in benchmarks_to_run.items():
        print("\n" + "="*60)
        print("Benchmark: %s" % bench_name)
        print("="*60)
        for bs in bs_list:
            print("  batch_size=%d ..." % bs, end=" ", flush=True)
            try:
                record = bench_fn(bs, device, use_gpu)
            except Exception as exc:
                record = {"kernel_name": bench_name, "batch_size": bs, "error": str(exc),
                          "skipped": True,
                          "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                print("ERROR: %s" % exc)
            else:
                if record.get("skipped"):
                    print("SKIPPED -- %s" % record.get("reason", "unknown"))
                else:
                    su = record.get("speedup", 0.0)
                    ok = record.get("passed_criteria", False)
                    print("speedup=%.3fx  p50=%.1fus  err=%.2e  %s" % (
                        su, record.get("kernel_p50_us", 0),
                        record.get("correctness_max_abs_err", -1),
                        "PASS" if ok else "FAIL"
                    ))
            all_records.append(record)
            if write_results and not record.get("skipped"):
                os.makedirs(RESULTS_DIR, exist_ok=True)
                with open(RESULTS_PATH, "a") as fh:
                    fh.write(json.dumps(record) + "\n")
    return all_records


def print_summary(records):
    print("\n" + "="*70)
    print("GPU BENCHMARK SUMMARY")
    print("="*70)
    fmt = "%-30s %6s %12s %12s %8s %6s"
    print(fmt % ("Kernel", "BS", "Baseline us", "Kernel us", "Speedup", "Pass"))
    print("-" * 70)
    for rec in records:
        if rec.get("skipped") or "error" in rec:
            print("  %-28s SKIPPED/ERROR" % rec.get("kernel_name", "?")[:28])
            continue
        print(fmt % (rec["kernel_name"][:30], rec["batch_size"],
                      "%.1f" % rec["baseline_p50_us"], "%.1f" % rec["kernel_p50_us"],
                      "%.3fx" % rec["speedup"], "PASS" if rec["passed_criteria"] else "FAIL"))
    print("="*70)
    print("Results written to: %s" % RESULTS_PATH)


def main():
    parser = argparse.ArgumentParser(description="H100 GPU kernel benchmarks (Lambda deployment)")
    parser.add_argument("--kernel", choices=list(ALL_BENCHMARKS.keys()), default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    batch_sizes = [args.batch_size] if args.batch_size else BATCH_SIZES
    print("GPU Benchmark Runner")
    print("Device:         %s" % ("cuda" if _cuda_available() else "cpu (no CUDA!)"))
    print("Triton:         %s" % ("available" if TRITON_AVAILABLE else "NOT available"))
    print("int8 scaled_mm: %s" % ("available" if INT8_GEMM_AVAILABLE else "NOT available"))
    print("Batch sizes:    %s" % batch_sizes)
    print("Results file:   %s" % RESULTS_PATH)
    records = run_all_benchmarks(kernel_filter=args.kernel, batch_sizes=batch_sizes,
                                  write_results=not args.no_write)
    print_summary(records)


if __name__ == "__main__":
    main()
