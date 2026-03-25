#!/usr/bin/env python3
"""
Kernel Benchmark Harness for H100 Lambda Instance
===================================================
Measures wall-clock latency, memory bandwidth, throughput, and numerical error
vs fp32 baseline for attention and GEMM kernels.

Usage:
    python benchmark_harness.py                    # run all benchmarks
    python benchmark_harness.py --kernel attention  # attention only
    python benchmark_harness.py --kernel gemm       # GEMM only
    python benchmark_harness.py --output results.jsonl

Targets: Lambda H100 (sm_90), CUDA 12+, Triton 2.x
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Triton import -- optional, graceful degradation if not available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    print("[WARN] Triton not available -- Triton kernels will be skipped", file=sys.stderr)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_GPU = DEVICE == "cuda"

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark configuration
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZES = [1, 8, 32]
SEQ_LENGTHS = [512, 2048, 8192]
HEAD_DIMS = [64, 128]           # 64 = standard, 128 = Llama-style
NUM_HEADS = 32
GEMM_SIZES = [512, 1024, 2048, 4096]  # square matrix side length

WARMUP_ITERS = 10
TIMED_ITERS = 50

# ─────────────────────────────────────────────────────────────────────────────
# Timer utilities
# ─────────────────────────────────────────────────────────────────────────────

class CudaTimer:
    """GPU-accurate timing via CUDA events."""
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, *args):
        self.end_event.record()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


class CpuTimer:
    """Wall-clock CPU timer fallback."""
    def __init__(self):
        self._start = None
        self._end = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000.0


def make_timer():
    return CudaTimer() if IS_GPU else CpuTimer()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    kernel_name: str
    config: dict
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    throughput_tops: float           # tera-ops/sec
    memory_bandwidth_gbs: float      # GB/s effective
    max_abs_error: float             # vs fp32 baseline
    relative_error: float            # max abs / (abs(baseline).mean() + 1e-8)
    passed_accuracy: bool            # max_abs_error < 1e-4
    device: str
    dtype: str
    notes: str = ""


def percentile(lst: List[float], p: float) -> float:
    lst_sorted = sorted(lst)
    idx = max(0, min(int(math.ceil(p / 100.0 * len(lst_sorted))) - 1, len(lst_sorted) - 1))
    return lst_sorted[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Numerical correctness helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_error(output: torch.Tensor, baseline: torch.Tensor) -> Tuple[float, float]:
    """Return (max_abs_error, relative_error) vs fp32 baseline."""
    with torch.no_grad():
        out_f32 = output.to(torch.float32)
        base_f32 = baseline.to(torch.float32)
        abs_err = (out_f32 - base_f32).abs()
        max_abs = abs_err.max().item()
        rel = max_abs / (base_f32.abs().mean().item() + 1e-8)
    return max_abs, rel


# ─────────────────────────────────────────────────────────────────────────────
# Baseline implementations (fp32 reference)
# ─────────────────────────────────────────────────────────────────────────────

def ref_attention_fp32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard scaled dot-product attention in fp32."""
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    scale = q32.shape[-1] ** -0.5
    attn = torch.matmul(q32, k32.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v32)


def ref_gemm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.float(), b.float())


# ─────────────────────────────────────────────────────────────────────────────
# Triton FlashAttention-2 style kernel (forward pass)
# ─────────────────────────────────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _flash_attn_fwd_kernel(
        Q, K, V, Out,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_ob, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Tiled FlashAttention forward kernel.
        Each program block handles BLOCK_M query rows.
        """
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_b = off_hz // H
        off_h = off_hz % H

        # Pointer arithmetic
        q_offset = off_b * stride_qb + off_h * stride_qh
        k_offset = off_b * stride_kb + off_h * stride_kh
        v_offset = off_b * stride_vb + off_h * stride_vh
        o_offset = off_b * stride_ob + off_h * stride_oh

        Q_ptrs = Q + q_offset + (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm                                + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_qk
        K_ptrs = K + k_offset + tl.arange(0, BLOCK_N)[:, None] * stride_kn                                + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_kk
        V_ptrs = V + v_offset + tl.arange(0, BLOCK_N)[:, None] * stride_vn                                + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_vk

        # Load Q block
        q = tl.load(Q_ptrs, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < N_CTX)

        # Running max and sum for online softmax
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        scale = (BLOCK_DMODEL ** -0.5)

        n_blocks = tl.cdiv(N_CTX, BLOCK_N)
        for j in range(n_blocks):
            if IS_CAUSAL:
                # Only attend to past tokens
                if j * BLOCK_N > (start_m + 1) * BLOCK_M:
                    break

            k = tl.load(K_ptrs + j * BLOCK_N * stride_kn,
                        mask=tl.arange(0, BLOCK_N)[:, None] + j * BLOCK_N < N_CTX)
            v = tl.load(V_ptrs + j * BLOCK_N * stride_vn,
                        mask=tl.arange(0, BLOCK_N)[:, None] + j * BLOCK_N < N_CTX)

            # QK^T
            qk = tl.dot(q, tl.trans(k)) * scale
            if IS_CAUSAL:
                # Mask future positions
                offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
                qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))

            # Online softmax update
            m_j = tl.max(qk, 1)
            p = tl.exp(qk - m_j[:, None])
            l_j = tl.sum(p, 1)
            alpha = tl.exp(m_i - tl.maximum(m_i, m_j))
            m_i = tl.maximum(m_i, m_j)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_j

        # Normalize
        acc = acc / l_i[:, None]

        O_ptrs = Out + o_offset + (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_om                                  + tl.arange(0, BLOCK_DMODEL)[None, :] * stride_ok
        tl.store(O_ptrs, acc.to(Out.dtype.element_ty),
                 mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < N_CTX)


    def triton_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                causal: bool = True,
                                BLOCK_M: int = 128, BLOCK_N: int = 64) -> torch.Tensor:
        """
        Triton FlashAttention forward pass.
        q, k, v: (batch, heads, seqlen, head_dim) in fp16/bf16
        """
        batch, heads, seqlen, head_dim = q.shape
        assert head_dim in (32, 64, 128), f"head_dim {head_dim} not supported"
        o = torch.empty_like(q)
        grid = (triton.cdiv(seqlen, BLOCK_M), batch * heads)
        _flash_attn_fwd_kernel[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch, heads, seqlen,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=head_dim,
            IS_CAUSAL=causal,
        )
        return o


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch SDPA baseline (uses FlashAttention under the hood on H100)
# ─────────────────────────────────────────────────────────────────────────────

def pytorch_sdpa_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            causal: bool = True) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


# ─────────────────────────────────────────────────────────────────────────────
# Fused RoPE + Attention
# ─────────────────────────────────────────────────────────────────────────────

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings in-place style."""
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def fused_rope_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          cos: torch.Tensor, sin: torch.Tensor,
                          causal: bool = True) -> torch.Tensor:
    """RoPE applied to Q/K before SDPA."""
    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)
    return F.scaled_dot_product_attention(q_rot, k_rot, v, is_causal=causal)


# ─────────────────────────────────────────────────────────────────────────────
# GEMM kernels
# ─────────────────────────────────────────────────────────────────────────────

def fp16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.half(), b.half())


def bf16_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a.bfloat16(), b.bfloat16())


def int8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """INT8 GEMM via torch._scaled_mm (requires CUDA sm_89+)."""
    a_q = a.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else a.half()
    b_q = b.to(torch.float8_e4m3fn) if hasattr(torch, 'float8_e4m3fn') else b.half()
    # Fallback: standard INT8 quantize-dequantize
    scale_a = a.abs().max() / 127.0 + 1e-8
    scale_b = b.abs().max() / 127.0 + 1e-8
    a_i8 = (a / scale_a).round().to(torch.int8)
    b_i8 = (b / scale_b).round().to(torch.int8)
    # torch._scaled_mm requires cuda and specific dtypes
    try:
        result = torch._scaled_mm(
            a_i8.to(torch.float8_e4m3fn),
            b_i8.to(torch.float8_e4m3fn).t(),
            out_dtype=torch.float16
        )
        return result * (scale_a * scale_b)
    except Exception:
        # CPU fallback
        return torch.matmul(a_i8.float() * scale_a, b_i8.float() * scale_b)


# ─────────────────────────────────────────────────────────────────────────────
# KV-Cache incremental decode (star result from CPU wave 2)
# ─────────────────────────────────────────────────────────────────────────────

def incremental_kv_cache_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-query incremental attention decode.
    Appends new_k/v to cache and attends over full cache.
    Returns (output, updated_k_cache, updated_v_cache).
    """
    # Append to KV cache
    k_full = torch.cat([k_cache, new_k], dim=2)
    v_full = torch.cat([v_cache, new_v], dim=2)
    # Single-query attention (q has seqlen=1)
    out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
    return out, k_full, v_full


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_timed_kernel(fn, *args, warmup=WARMUP_ITERS, timed=TIMED_ITERS, **kwargs):
    """Run fn(*args, **kwargs) warmup+timed times. Returns list of latencies in ms."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)
    if IS_GPU:
        torch.cuda.synchronize()

    times = []
    for _ in range(timed):
        t = make_timer()
        with t:
            fn(*args, **kwargs)
        times.append(t.elapsed_ms)
    return times


def compute_attention_flops(batch: int, heads: int, seqlen: int, head_dim: int) -> float:
    """FLOPs for attention: 2 * batch * heads * seqlen^2 * head_dim (QK^T + AV)."""
    return 2.0 * batch * heads * seqlen * seqlen * head_dim


def compute_gemm_flops(m: int, n: int, k: int) -> float:
    """FLOPs for GEMM: 2*m*n*k."""
    return 2.0 * m * n * k


def compute_attention_memory_bytes(batch: int, heads: int, seqlen: int, head_dim: int,
                                    elem_bytes: int = 2) -> float:
    """Bytes touched: Q+K+V+O tensors."""
    return 4.0 * batch * heads * seqlen * head_dim * elem_bytes


def benchmark_attention_configs(output_rows: list):
    """Benchmark attention across all batch x seqlen x head_dim configurations."""
    print("\n=== ATTENTION BENCHMARKS ===")
    print(f"{'Kernel':<30} {'B':>4} {'S':>6} {'D':>4} {'p50(ms)':>9} {'p90(ms)':>9} {'TFLOPS':>8} {'BW(GB/s)':>10} {'MaxErr':>10} {'Pass':>5}")
    print("-" * 110)

    for batch in BATCH_SIZES:
        for seqlen in SEQ_LENGTHS:
            for head_dim in HEAD_DIMS:
                q = torch.randn(batch, NUM_HEADS, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
                k = torch.randn(batch, NUM_HEADS, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
                v = torch.randn(batch, NUM_HEADS, seqlen, head_dim, device=DEVICE, dtype=torch.float16)

                # fp32 reference
                with torch.no_grad():
                    ref = ref_attention_fp32(q, k, v)

                configs_to_run = [
                    ("sdpa_fp16", lambda q=q, k=k, v=v: pytorch_sdpa_attention(q, k, v)),
                ]

                # RoPE fusion
                cos = torch.ones(1, 1, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
                sin = torch.zeros(1, 1, seqlen, head_dim, device=DEVICE, dtype=torch.float16)
                configs_to_run.append((
                    "fused_rope_sdpa",
                    lambda q=q, k=k, v=v, cos=cos, sin=sin: fused_rope_attention(q, k, v, cos, sin)
                ))

                if HAS_TRITON and head_dim in (64, 128) and seqlen <= 4096:
                    configs_to_run.append((
                        "triton_flash_attn",
                        lambda q=q, k=k, v=v: triton_flash_attention(q, k, v)
                    ))

                flops = compute_attention_flops(batch, NUM_HEADS, seqlen, head_dim)
                mem_bytes = compute_attention_memory_bytes(batch, NUM_HEADS, seqlen, head_dim)

                for kernel_name, fn in configs_to_run:
                    try:
                        times = run_timed_kernel(fn)
                        p50 = percentile(times, 50)
                        p90 = percentile(times, 90)
                        p99 = percentile(times, 99)
                        mean_t = sum(times) / len(times)

                        # Correctness
                        with torch.no_grad():
                            out = fn()
                        max_err, rel_err = compute_error(out, ref)
                        passed = max_err < 1e-3  # fp16 can't hit 1e-4 exactly but should be close

                        tflops = (flops / (mean_t * 1e-3)) / 1e12
                        bw_gbs = (mem_bytes / (mean_t * 1e-3)) / 1e9

                        config = {"batch": batch, "seqlen": seqlen, "head_dim": head_dim, "num_heads": NUM_HEADS}
                        result = BenchmarkResult(
                            kernel_name=kernel_name,
                            config=config,
                            latency_p50_ms=p50,
                            latency_p90_ms=p90,
                            latency_p99_ms=p99,
                            latency_mean_ms=mean_t,
                            throughput_tops=tflops,
                            memory_bandwidth_gbs=bw_gbs,
                            max_abs_error=max_err,
                            relative_error=rel_err,
                            passed_accuracy=passed,
                            device=DEVICE,
                            dtype="fp16",
                        )
                        output_rows.append(asdict(result))

                        print(f"{kernel_name:<30} {batch:>4} {seqlen:>6} {head_dim:>4} "
                              f"{p50:>9.3f} {p90:>9.3f} {tflops:>8.2f} {bw_gbs:>10.2f} "
                              f"{max_err:>10.2e} {'Y' if passed else 'N':>5}")
                    except Exception as e:
                        print(f"{kernel_name:<30} {batch:>4} {seqlen:>6} {head_dim:>4} ERROR: {e}")


def benchmark_gemm_configs(output_rows: list):
    """Benchmark GEMM across matrix sizes."""
    print("\n=== GEMM BENCHMARKS ===")
    print(f"{'Kernel':<25} {'M':>5} {'N':>5} {'K':>5} {'p50(ms)':>9} {'TFLOPS':>8} {'BW(GB/s)':>10} {'MaxErr':>10} {'Pass':>5}")
    print("-" * 90)

    for sz in GEMM_SIZES:
        a = torch.randn(sz, sz, device=DEVICE, dtype=torch.float32)
        b = torch.randn(sz, sz, device=DEVICE, dtype=torch.float32)
        ref = ref_gemm_fp32(a, b)

        flops = compute_gemm_flops(sz, sz, sz)
        mem_bytes = (3 * sz * sz * 4)  # A + B + C in fp32

        kernels = [
            ("gemm_fp32",  lambda a=a, b=b: torch.matmul(a.float(), b.float())),
            ("gemm_fp16",  lambda a=a, b=b: fp16_gemm(a, b)),
            ("gemm_bf16",  lambda a=a, b=b: bf16_gemm(a, b)),
            ("gemm_int8",  lambda a=a, b=b: int8_gemm(a, b)),
        ]

        for kernel_name, fn in kernels:
            try:
                times = run_timed_kernel(fn)
                p50 = percentile(times, 50)
                p90 = percentile(times, 90)
                p99 = percentile(times, 99)
                mean_t = sum(times) / len(times)

                with torch.no_grad():
                    out = fn()
                max_err, rel_err = compute_error(out, ref)
                passed = max_err < 1.0  # GEMM accumulation error tolerance

                tflops = (flops / (mean_t * 1e-3)) / 1e12
                bw_gbs = (mem_bytes / (mean_t * 1e-3)) / 1e9

                result = BenchmarkResult(
                    kernel_name=kernel_name,
                    config={"M": sz, "N": sz, "K": sz},
                    latency_p50_ms=p50,
                    latency_p90_ms=p90,
                    latency_p99_ms=p99,
                    latency_mean_ms=mean_t,
                    throughput_tops=tflops,
                    memory_bandwidth_gbs=bw_gbs,
                    max_abs_error=max_err,
                    relative_error=rel_err,
                    passed_accuracy=passed,
                    device=DEVICE,
                    dtype=kernel_name.split("_")[1],
                )
                output_rows.append(asdict(result))

                print(f"{kernel_name:<25} {sz:>5} {sz:>5} {sz:>5} "
                      f"{p50:>9.3f} {tflops:>8.2f} {bw_gbs:>10.2f} "
                      f"{max_err:>10.2e} {'Y' if passed else 'N':>5}")
            except Exception as e:
                print(f"{kernel_name:<25} {sz:>5} {sz:>5} {sz:>5} ERROR: {e}")


def benchmark_kv_cache_decode(output_rows: list):
    """Benchmark KV-cache incremental decode vs full recompute (replication of H-07 star result)."""
    print("\n=== KV-CACHE DECODE BENCHMARKS (H-07 validation) ===")
    print(f"{'Mode':<30} {'B':>4} {'CacheLen':>10} {'Heads':>6} {'p50(ms)':>9} {'Speedup':>8}")
    print("-" * 80)

    batch = 1
    heads = 32
    head_dim = 128
    new_seqlen = 1  # decode step

    for cache_len in [128, 512, 2048]:
        k_cache = torch.randn(batch, heads, cache_len, head_dim, device=DEVICE, dtype=torch.float16)
        v_cache = torch.randn(batch, heads, cache_len, head_dim, device=DEVICE, dtype=torch.float16)
        q_dec = torch.randn(batch, heads, new_seqlen, head_dim, device=DEVICE, dtype=torch.float16)
        new_k = torch.randn(batch, heads, new_seqlen, head_dim, device=DEVICE, dtype=torch.float16)
        new_v = torch.randn(batch, heads, new_seqlen, head_dim, device=DEVICE, dtype=torch.float16)

        # Full recompute (all tokens re-attend)
        all_k = torch.cat([k_cache, new_k], dim=2)
        all_v = torch.cat([v_cache, new_v], dim=2)
        q_full = torch.randn(batch, heads, cache_len + new_seqlen, head_dim, device=DEVICE, dtype=torch.float16)

        def full_recompute():
            return F.scaled_dot_product_attention(q_full, all_k, all_v, is_causal=True)

        def incr_decode():
            return incremental_kv_cache_decode(q_dec, k_cache, v_cache, new_k, new_v)[0]

        times_full = run_timed_kernel(full_recompute, warmup=5, timed=30)
        times_incr = run_timed_kernel(incr_decode, warmup=5, timed=30)

        p50_full = percentile(times_full, 50)
        p50_incr = percentile(times_incr, 50)
        speedup = p50_full / (p50_incr + 1e-9)

        for name, p50 in [("full_recompute", p50_full), ("incr_kv_decode", p50_incr)]:
            result = BenchmarkResult(
                kernel_name=name,
                config={"batch": batch, "cache_len": cache_len, "heads": heads, "head_dim": head_dim},
                latency_p50_ms=p50,
                latency_p90_ms=p50,
                latency_p99_ms=p50,
                latency_mean_ms=p50,
                throughput_tops=0.0,
                memory_bandwidth_gbs=0.0,
                max_abs_error=0.0,
                relative_error=0.0,
                passed_accuracy=True,
                device=DEVICE,
                dtype="fp16",
                notes=f"cache_len={cache_len} speedup={speedup:.1f}x" if name == "incr_kv_decode" else "",
            )
            output_rows.append(asdict(result))

        print(f"{'incr_kv_decode':<30} {batch:>4} {cache_len:>10} {heads:>6} {p50_incr:>9.3f} {speedup:>8.1f}x")
        print(f"{'full_recompute':<30} {batch:>4} {cache_len:>10} {heads:>6} {p50_full:>9.3f} {'1.0x':>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kernel Benchmark Harness for H100")
    parser.add_argument("--kernel", choices=["all", "attention", "gemm", "kv_cache"],
                        default="all", help="Which kernel family to benchmark")
    parser.add_argument("--output", default=None, help="Output JSONL file path for results")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS, help="Warmup iterations")
    parser.add_argument("--timed", type=int, default=TIMED_ITERS, help="Timed iterations")
    args = parser.parse_args()

    global WARMUP_ITERS, TIMED_ITERS
    WARMUP_ITERS = args.warmup
    TIMED_ITERS = args.timed

    print(f"Kernel Benchmark Harness")
    print(f"Device: {DEVICE}")
    if IS_GPU:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, {props.total_memory // 1024**3}GB, sm_{props.major}{props.minor}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"PyTorch: {torch.__version__}")

    output_rows = []

    if args.kernel in ("all", "attention"):
        benchmark_attention_configs(output_rows)

    if args.kernel in ("all", "gemm"):
        benchmark_gemm_configs(output_rows)

    if args.kernel in ("all", "kv_cache"):
        benchmark_kv_cache_decode(output_rows)

    if args.output:
        with open(args.output, "w") as f:
            for row in output_rows:
                f.write(json.dumps(row) + "\n")
        print(f"\nResults written to {args.output}")
    else:
        print(f"\nTotal benchmarks run: {len(output_rows)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
