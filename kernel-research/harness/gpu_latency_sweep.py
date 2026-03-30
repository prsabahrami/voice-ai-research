#!/usr/bin/env python3
"""
gpu_latency_sweep.py
H100 GPU latency benchmark for KV-cache and attention variants.

Measures wall-clock time for:
  (a) naive SDPA (torch.nn.functional.scaled_dot_product_attention)
  (b) flash_attn if available
  (c) triton_block_sparse_attention at 50/80/90% sparsity
  (d) QuantizedKVCache INT8 store + retrieve (GEAR+v4 style)

Sweep: batch={1, 8, 32} x seqlen={512, 2048, 8192}
Logs to results_gpu.jsonl

Usage:
  python3 gpu_latency_sweep.py --output results_gpu.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import numpy as np

_HERE = Path(__file__).parent.resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
WARMUP_ITERS = 20
TIMED_ITERS  = 100

SWEEP_BATCHES  = [1, 8, 32]
SWEEP_SEQLENS  = [512, 2048, 8192]
SWEEP_SPARSITY = [0.50, 0.80, 0.90]

NUM_HEADS = 32
HEAD_DIM  = 128
DTYPE     = torch.float16

# -----------------------------------------------------------------------
# Timing helper
# -----------------------------------------------------------------------
def timed_gpu(fn, warmup=WARMUP_ITERS, iters=TIMED_ITERS):
    """Run fn() on GPU, return list of latency_ms values."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return latencies


def stats(lats):
    lats_s = sorted(lats)
    n = len(lats_s)
    def pct(p):
        i = (n-1)*p/100
        lo, hi = int(i), min(int(i)+1, n-1)
        return lats_s[lo] + (lats_s[hi]-lats_s[lo])*(i-int(i))
    return {
        "mean_ms": round(sum(lats)/n, 4),
        "p50_ms":  round(pct(50), 4),
        "p95_ms":  round(pct(95), 4),
        "p99_ms":  round(pct(99), 4),
        "min_ms":  round(min(lats), 4),
        "max_ms":  round(max(lats), 4),
    }


# -----------------------------------------------------------------------
# Benchmark variants
# -----------------------------------------------------------------------

def bench_naive_sdpa(B, S, H=NUM_HEADS, D=HEAD_DIM, dtype=DTYPE, device="cuda"):
    """Naive SDPA using torch.nn.functional."""
    q = torch.randn(B, H, 1, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
    lats = timed_gpu(fn)
    mem_bytes = (q.nbytes + k.nbytes + v.nbytes)
    return {
        "variant": "naive_sdpa",
        "config": {"batch": B, "seqlen": S, "num_heads": H, "head_dim": D},
        "latency": stats(lats),
        "memory_bytes": mem_bytes,
    }


def bench_flash_attn(B, S, H=NUM_HEADS, D=HEAD_DIM, dtype=DTYPE, device="cuda"):
    """FlashAttention-2 if available, else None."""
    try:
        from flash_attn import flash_attn_func
        # flash_attn expects (B, S, H, D) layout
        q = torch.randn(B, 1, H, D, dtype=dtype, device=device)
        k = torch.randn(B, S, H, D, dtype=dtype, device=device)
        v = torch.randn(B, S, H, D, dtype=dtype, device=device)
        fn = lambda: flash_attn_func(q, k, v, causal=False)
        lats = timed_gpu(fn)
        mem_bytes = (q.nbytes + k.nbytes + v.nbytes)
        return {
            "variant": "flash_attn_v2",
            "config": {"batch": B, "seqlen": S, "num_heads": H, "head_dim": D},
            "latency": stats(lats),
            "memory_bytes": mem_bytes,
        }
    except ImportError:
        return None


def bench_block_sparse(B, S, sparsity, H=NUM_HEADS, D=HEAD_DIM, dtype=DTYPE, device="cuda"):
    """Block-sparse attention via hand-written masking (CPU-side mask, GPU compute)."""
    BLOCK_SIZE = 64
    n_blocks = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
    # Generate sparsity mask
    mask = torch.rand(B, H, 1, n_blocks, device=device) > sparsity  # (B,H,1,NB)
    # Ensure at least 1 block active per row
    mask[:, :, :, 0] = True
    actual_sparsity = 1.0 - float(mask.float().mean())

    q = torch.randn(B, H, 1, D, dtype=dtype, device=device)
    k = torch.randn(B, H, S, D, dtype=dtype, device=device)
    v = torch.randn(B, H, S, D, dtype=dtype, device=device)

    def _forward():
        # Expand mask to full attention mask shape (B, H, 1, S)
        block_mask = mask.repeat_interleave(BLOCK_SIZE, dim=-1)[:, :, :, :S]
        attn_mask = torch.where(block_mask,
                                torch.zeros_like(block_mask, dtype=dtype),
                                torch.full_like(block_mask, float("-inf"), dtype=dtype))
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    lats = timed_gpu(_forward)
    return {
        "variant": f"block_sparse_{int(sparsity*100)}pct",
        "config": {"batch": B, "seqlen": S, "num_heads": H, "head_dim": D,
                   "target_sparsity": sparsity, "actual_sparsity": round(actual_sparsity, 3)},
        "latency": stats(lats),
    }


def bench_quant_kv(B, S, quant_type="int8", H=8, D=64, device="cuda"):
    """Quantized KV-cache benchmark: GPU store+retrieve+attention."""
    try:
        from quantized_kv_cache import QuantizedKVCache
    except ImportError:
        return None

    import numpy as np
    rng = np.random.default_rng(0)
    k_np = rng.normal(0, 1, (B, H, S, D)).astype(np.float32)
    v_np = rng.normal(0, 1, (B, H, S, D)).astype(np.float32)
    q_np = rng.normal(0, 1, (B, H, 1, D)).astype(np.float32)

    # Build cache on CPU (INT8/FP8)
    cache = QuantizedKVCache(B, H, S, D, quant_type=quant_type)
    cache.update(k_np, v_np, start_pos=0)

    # For GPU timing, we move the quantized tensors to GPU and do dequant+attention
    # This simulates the GPU path where quantized weights live in HBM
    k_q_gpu  = torch.from_numpy(cache._k_quant).to(device)   # (B,H,S,D) int8
    v_q_gpu  = torch.from_numpy(cache._v_quant).to(device)   # (B,H,S,D) int8
    k_s_gpu  = torch.from_numpy(cache._k_scales).to(device)  # (B,H,1,1) fp32
    v_s_gpu  = torch.from_numpy(cache._v_scales).to(device)  # (B,H,1,1) fp32
    q_gpu    = torch.from_numpy(q_np).to(device)             # (B,H,1,D)

    def _gpu_dequant_attn():
        # Dequantize on GPU
        k_fp = k_q_gpu.float() * k_s_gpu   # (B,H,S,D)
        v_fp = v_q_gpu.float() * v_s_gpu   # (B,H,S,D)
        # SDPA
        return torch.nn.functional.scaled_dot_product_attention(
            q_gpu.float(), k_fp, v_fp)

    lats = timed_gpu(_gpu_dequant_attn)
    mem_save = cache.memory_savings_pct()

    return {
        "variant": f"quant_kv_{quant_type}",
        "config": {"batch": B, "seqlen": S, "num_heads": H, "head_dim": D,
                   "quant_type": quant_type},
        "latency": stats(lats),
        "memory_savings_pct": round(mem_save * 100, 2),
    }


# -----------------------------------------------------------------------
# FP8 Tensor Core path (H100 specific)
# -----------------------------------------------------------------------
def bench_fp8_tensor_core(B, S, H=NUM_HEADS, D=HEAD_DIM, device="cuda"):
    """FP8 attention via torch._scaled_dot_product_flash_attention with FP8 dtypes.
    H100 SM89+ supports native FP8 (float8_e4m3fn).
    """
    try:
        fp8_dtype = torch.float8_e4m3fn
        q = torch.randn(B, H, 1, D, device=device).to(fp8_dtype)
        k = torch.randn(B, H, S, D, device=device).to(fp8_dtype)
        v = torch.randn(B, H, S, D, device=device).to(fp8_dtype)
        # Scale factors for FP8
        q_scale = torch.tensor(1.0, device=device)
        k_scale = torch.tensor(1.0, device=device)
        v_scale = torch.tensor(1.0, device=device)
        fn = lambda: torch._scaled_dot_product_flash_attention(
            q.view(B, H, 1, D),
            k.view(B, H, S, D),
            v.view(B, H, S, D),
            scale=1.0/D**0.5,
        )
        lats = timed_gpu(fn)
        return {
            "variant": "fp8_tensor_core",
            "config": {"batch": B, "seqlen": S, "num_heads": H, "head_dim": D},
            "latency": stats(lats),
        }
    except Exception as e:
        return {"variant": "fp8_tensor_core", "error": str(e),
                "config": {"batch": B, "seqlen": S}}


# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------
def run_sweep(output_path=None):
    device = "cuda"
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Torch: {torch.__version__}")

    results = []
    out_f = open(output_path, "w") if output_path else None

    def emit(r):
        results.append(r)
        line = json.dumps(r)
        print(f"  {r['variant']} B={r['config']['batch']} S={r['config']['seqlen']}: "
              f"mean={r.get('latency', {}).get('mean_ms', 'ERR')}ms")
        if out_f:
            out_f.write(line + "\n")
            out_f.flush()

    for B in SWEEP_BATCHES:
        for S in SWEEP_SEQLENS:
            print(f"\n--- B={B} S={S} ---")
            # (a) Naive SDPA
            emit(bench_naive_sdpa(B, S))
            # (b) FlashAttention
            r = bench_flash_attn(B, S)
            if r: emit(r)
            # (c) Block-sparse
            for sp in SWEEP_SPARSITY:
                emit(bench_block_sparse(B, S, sp))
            # (d) Quantized KV-cache
            for qt in ["int8", "fp8"]:
                r = bench_quant_kv(B, S, qt)
                if r: emit(r)
            # (e) FP8 Tensor Core
            emit(bench_fp8_tensor_core(B, S))

    if out_f:
        out_f.close()

    print(f"\nTotal: {len(results)} benchmark runs.")
    return results


def main():
    p = argparse.ArgumentParser(description="GPU latency sweep for KV-cache variants")
    p.add_argument("--output", default="results_gpu.jsonl")
    p.add_argument("--batch", type=int, nargs="+", default=SWEEP_BATCHES)
    p.add_argument("--seqlen", type=int, nargs="+", default=SWEEP_SEQLENS)
    args = p.parse_args()
    global SWEEP_BATCHES, SWEEP_SEQLENS
    SWEEP_BATCHES = args.batch
    SWEEP_SEQLENS = args.seqlen
    run_sweep(args.output)


if __name__ == "__main__":
    main()
