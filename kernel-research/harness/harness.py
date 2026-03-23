"""
harness.py - Core timing harness for kernel optimization research.

Features:
  - GPU warmup (10 iterations) when CUDA available, else CPU warmup
  - 100 timed iterations per benchmark
  - Reports p50, p90, p99 latency in microseconds
  - Correctness check: max absolute error vs reference implementation
  - Supports attention, gemm, and fused_ops kernel types
"""

import time
import math
import statistics
import torch
from typing import Callable, List, Dict, Any, Optional, Tuple


WARMUP_ITERS = 10
TIMED_ITERS = 100

KERNEL_TYPES = ("attention", "gemm", "fused_ops")


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _time_fn_cpu(fn: Callable, *args, n_iter: int = TIMED_ITERS) -> List[float]:
    """Time a function on CPU. Returns list of elapsed times in seconds."""
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _time_fn_gpu(fn: Callable, *args, n_iter: int = TIMED_ITERS) -> List[float]:
    """Time a function on GPU using CUDA events. Returns elapsed times in seconds."""
    times = []
    for _ in range(n_iter):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)  # ms -> s
    return times


def warmup(fn: Callable, *args):
    """Run warmup iterations (GPU sync-aware)."""
    for _ in range(WARMUP_ITERS):
        fn(*args)
    if _cuda_available():
        torch.cuda.synchronize()


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Compute a percentile from sorted data (linear interpolation)."""
    n = len(sorted_data)
    if n == 0:
        return float('nan')
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    frac = idx - lo
    if hi >= n:
        return sorted_data[-1]
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def compute_latency_stats(times_sec: List[float]) -> Dict[str, float]:
    """Given a list of elapsed times in seconds, return p50/p90/p99/mean in microseconds."""
    sorted_t = sorted(times_sec)
    us = [t * 1e6 for t in sorted_t]
    return {
        "p50_us":  _percentile(us, 50),
        "p90_us":  _percentile(us, 90),
        "p99_us":  _percentile(us, 99),
        "mean_us": statistics.mean(us),
    }


def correctness_check(reference_output: torch.Tensor,
                      candidate_output: torch.Tensor) -> float:
    """Returns the max absolute error between reference and candidate outputs."""
    ref = reference_output.detach().float()
    cand = candidate_output.detach().float()
    return float(torch.max(torch.abs(ref - cand)).item())


def _run_attention_benchmark(batch_size: int, device: torch.device,
                              use_gpu: bool) -> Dict[str, Any]:
    from baseline_kernels import naive_attention, make_attention_inputs
    from optimized_kernels import tiled_attention
    q, k, v = make_attention_inputs(batch_size, device=device)
    warmup(naive_attention, q, k, v)
    warmup(tiled_attention, q, k, v)
    if use_gpu:
        base_times = _time_fn_gpu(naive_attention, q, k, v)
        opt_times = _time_fn_gpu(tiled_attention, q, k, v)
    else:
        base_times = _time_fn_cpu(naive_attention, q, k, v)
        opt_times = _time_fn_cpu(tiled_attention, q, k, v)
    ref_out = naive_attention(q, k, v)
    opt_out = tiled_attention(q, k, v)
    max_err = correctness_check(ref_out, opt_out)
    return {
        "kernel_type": "attention",
        "baseline": compute_latency_stats(base_times),
        "optimized": compute_latency_stats(opt_times),
        "correctness_max_abs_err": max_err,
        "device": str(device),
    }


def _run_gemm_benchmark(batch_size: int, device: torch.device,
                        use_gpu: bool) -> Dict[str, Any]:
    from baseline_kernels import standard_gemm, make_gemm_inputs
    from optimized_kernels import blocked_gemm
    a, b = make_gemm_inputs(batch_size, device=device)
    warmup(standard_gemm, a, b)
    warmup(blocked_gemm, a, b)
    if use_gpu:
        base_times = _time_fn_gpu(standard_gemm, a, b)
        opt_times = _time_fn_gpu(blocked_gemm, a, b)
    else:
        base_times = _time_fn_cpu(standard_gemm, a, b)
        opt_times = _time_fn_cpu(blocked_gemm, a, b)
    ref_out = standard_gemm(a, b)
    opt_out = blocked_gemm(a, b)
    max_err = correctness_check(ref_out, opt_out)
    return {
        "kernel_type": "gemm",
        "baseline": compute_latency_stats(base_times),
        "optimized": compute_latency_stats(opt_times),
        "correctness_max_abs_err": max_err,
        "device": str(device),
    }


def _run_fused_ops_benchmark(batch_size: int, device: torch.device,
                              use_gpu: bool) -> Dict[str, Any]:
    from baseline_kernels import unfused_softmax_cast, make_fused_ops_inputs
    from optimized_kernels import fused_softmax_cast
    x = make_fused_ops_inputs(batch_size, device=device)
    warmup(unfused_softmax_cast, x)
    warmup(fused_softmax_cast, x)
    if use_gpu:
        base_times = _time_fn_gpu(unfused_softmax_cast, x)
        opt_times = _time_fn_gpu(fused_softmax_cast, x)
    else:
        base_times = _time_fn_cpu(unfused_softmax_cast, x)
        opt_times = _time_fn_cpu(fused_softmax_cast, x)
    ref_out = unfused_softmax_cast(x)
    opt_out = fused_softmax_cast(x)
    max_err = correctness_check(ref_out, opt_out)
    return {
        "kernel_type": "fused_ops",
        "baseline": compute_latency_stats(base_times),
        "optimized": compute_latency_stats(opt_times),
        "correctness_max_abs_err": max_err,
        "device": str(device),
    }


_KERNEL_RUNNERS = {
    "attention": _run_attention_benchmark,
    "gemm":      _run_gemm_benchmark,
    "fused_ops": _run_fused_ops_benchmark,
}


def run_benchmark(kernel_type: str, batch_size: int,
                  device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Run the benchmark for the given kernel_type and batch_size."""
    if kernel_type not in KERNEL_TYPES:
        raise ValueError(f"Unknown kernel_type {kernel_type!r}. Choose from {KERNEL_TYPES}")
    use_gpu = _cuda_available()
    if device is None:
        device = torch.device("cuda" if use_gpu else "cpu")
    runner = _KERNEL_RUNNERS[kernel_type]
    return runner(batch_size, device, use_gpu)
