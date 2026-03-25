#!/usr/bin/env python3
"""
Attention Benchmark Harness: Naive vs FlashAttention-2 (SDPA) on H100
=====================================================================
Measures: latency (ms), bandwidth (GB/s), peak VRAM (bytes), numerical error vs fp32.
Configs: batch {1, 8, 32} x seqlen {512, 2048, 8192} x dtype {fp16, bf16}
Output: ~/voice-ai-research/kernels/benchmarks/results.jsonl (one JSON object per config)

Usage:
    python bench.py                     # Run on GPU (or CPU fallback)
    python bench.py --output custom.jsonl  # Custom output path
"""

import json
import os
import sys
import time
import gc
import math
import traceback
import argparse
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

# ---- Constants ----
NUM_HEADS = 32
HEAD_DIM = 128
WARMUP_ITERS = 10
BENCH_ITERS = 100   # 100 timed iterations (org convention)
BATCH_SIZES = [1, 8, 32]
SEQ_LENGTHS = [512, 2048, 8192]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

DEFAULT_RESULTS_PATH = os.path.expanduser(
    "~/voice-ai-research/kernels/benchmarks/results.jsonl"
)

# H100 SXM constants
H100_HBM_BW_GBS = 3350.0     # GB/s theoretical peak HBM3
H100_BF16_TFLOPS = 989.4      # TFLOPS BF16 tensor core
H100_FP16_TFLOPS = 989.4      # Same for FP16 tensor core
H100_L2_BYTES = 50 * 1024**2  # 50 MB L2 cache


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: No CUDA device. Running on CPU (timings not meaningful).",
          file=sys.stderr)
    return torch.device("cpu")


# ---- Attention Implementations ----

def naive_attention(q, k, v, scale):
    """Standard scaled dot-product attention (materializes full NxN matrix)."""
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def sdpa_flash_attention(q, k, v, scale):
    """PyTorch SDPA with flash_attention backend (cuDNN FA2/FA3 on H100)."""
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
    ):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)


def sdpa_efficient_attention(q, k, v, scale):
    """PyTorch SDPA with efficient_attention (xformers-like) backend."""
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)


def sdpa_math_attention(q, k, v, scale):
    """PyTorch SDPA with math backend (reference)."""
    with torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.MATH]
    ):
        return F.scaled_dot_product_attention(q, k, v, scale=scale)


def sdpa_auto_attention(q, k, v, scale):
    """PyTorch SDPA auto-dispatch (best available backend)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


# ---- Flash-attn package wrapper (loaded lazily) ----

_flash_attn_func = None


def _load_flash_attn():
    global _flash_attn_func
    if _flash_attn_func is not None:
        return True
    try:
        from flash_attn import flash_attn_func as fa2
        _flash_attn_func = fa2
        return True
    except ImportError:
        return False


def flash_attn_pkg_attention(q, k, v, scale):
    """flash-attn package (requires pip install flash-attn). Input (B,H,N,D)."""
    q_t = q.transpose(1, 2).contiguous()
    k_t = k.transpose(1, 2).contiguous()
    v_t = v.transpose(1, 2).contiguous()
    out = _flash_attn_func(q_t, k_t, v_t, softmax_scale=scale)
    return out.transpose(1, 2)


# ---- Metrics ----

def numerical_error(out, ref_fp32):
    """L-inf and relative L2 error against fp32 reference."""
    out32 = out.float()
    ref = ref_fp32.float()
    diff = (out32 - ref).abs()
    linf = diff.max().item()
    l2_err = torch.norm(diff).item()
    l2_ref = torch.norm(ref).item()
    rel_l2 = l2_err / max(l2_ref, 1e-12)
    return linf, rel_l2


def attention_flops(batch, seqlen, num_heads, head_dim):
    """FLOPs for attention: 4*B*H*N^2*D (QK^T matmul + attn@V matmul)."""
    return 4 * batch * num_heads * seqlen * seqlen * head_dim


def attention_io_bytes(batch, seqlen, num_heads, head_dim, elem_bytes):
    """Minimum memory traffic: read Q,K,V + write O."""
    return 4 * batch * num_heads * seqlen * head_dim * elem_bytes


def arithmetic_intensity(flops, io_bytes):
    """FLOPs per byte of memory traffic."""
    return flops / max(io_bytes, 1) if io_bytes > 0 else float("inf")


def roofline_class(flops, io_bytes, peak_tflops=H100_BF16_TFLOPS,
                   peak_bw_gbs=H100_HBM_BW_GBS):
    """Classify as compute-bound or memory-bound on H100."""
    ai = arithmetic_intensity(flops, io_bytes)
    ridge_point = (peak_tflops * 1e12) / (peak_bw_gbs * 1e9)
    return "compute-bound" if ai >= ridge_point else "memory-bound"


# ---- L2 Cache Flush ----

_flush_buf = None


def flush_l2(device):
    """Write to a large buffer to flush L2 cache."""
    global _flush_buf
    if device.type != "cuda":
        return
    if _flush_buf is None or _flush_buf.device != device:
        _flush_buf = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.float32,
                                 device=device)
    _flush_buf.zero_()


# ---- Benchmark Engine ----

def benchmark_kernel(fn, q, k, v, scale, device,
                     warmup=WARMUP_ITERS, iters=BENCH_ITERS,
                     cold_cache=True):
    """Benchmark with CUDA events. Returns timing dict."""
    if device.type == "cpu":
        for _ in range(warmup):
            _ = fn(q, k, v, scale)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn(q, k, v, scale)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times.sort()
        n = len(times)
        return {
            "median_ms": times[n // 2],
            "mean_ms": sum(times) / n,
            "min_ms": times[0],
            "p90_ms": times[int(0.9 * n)],
            "p99_ms": times[min(int(0.99 * n), n - 1)],
        }

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for _ in range(warmup):
        _ = fn(q, k, v, scale)
    torch.cuda.synchronize()

    for i in range(iters):
        if cold_cache:
            flush_l2(device)
        starts[i].record()
        _ = fn(q, k, v, scale)
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(times)
    return {
        "median_ms": times[n // 2],
        "mean_ms": sum(times) / n,
        "min_ms": times[0],
        "p90_ms": times[int(0.9 * n)],
        "p99_ms": times[min(int(0.99 * n), n - 1)],
    }


def measure_peak_vram(fn, q, k, v, scale, device):
    """Peak VRAM for a single forward pass."""
    if device.type != "cuda":
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    _ = fn(q, k, v, scale)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)


# ---- Single Config Runner ----

def run_config(batch, seqlen, dtype_name, dtype, device, results_fp):
    """Run all kernels for one (batch, seqlen, dtype) config."""
    scale = HEAD_DIM ** -0.5
    elem_bytes = 2
    flops = attention_flops(batch, seqlen, NUM_HEADS, HEAD_DIM)
    io_bytes = attention_io_bytes(batch, seqlen, NUM_HEADS, HEAD_DIM, elem_bytes)
    ai = arithmetic_intensity(flops, io_bytes)
    rc = roofline_class(flops, io_bytes)

    tag = f"B={batch} N={seqlen} {dtype_name} H={NUM_HEADS} D={HEAD_DIM}"
    print(f"\n{'='*72}")
    print(f"  {tag}  |  AI={ai:.1f} FLOP/B  |  {rc}")
    print(f"{'='*72}")

    naive_matrix_gb = batch * NUM_HEADS * seqlen * seqlen * elem_bytes / 1e9
    skip_naive = naive_matrix_gb > 40

    seed = hash((batch, seqlen, dtype_name)) % (2**32)
    torch.manual_seed(seed)
    q = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)

    fp32_ref = None
    if not skip_naive:
        try:
            with torch.no_grad():
                fp32_ref = naive_attention(q.float(), k.float(), v.float(), scale)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print("  fp32 reference OOM -- skipping error metrics for naive")
            fp32_ref = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = []

    kernels = []

    if skip_naive:
        print(f"  SKIP naive: attn matrix ~{naive_matrix_gb:.1f} GB")
        results.append({
            "kernel": "naive_attention", "batch": batch, "seqlen": seqlen,
            "dtype": dtype_name, "num_heads": NUM_HEADS, "head_dim": HEAD_DIM,
            "status": "SKIPPED",
            "reason": f"attn matrix {naive_matrix_gb:.1f} GB exceeds 40 GB limit",
        })
    else:
        kernels.append(("naive_attention", naive_attention))

    flash_ok = False
    if device.type == "cuda":
        try:
            with torch.no_grad():
                _t = sdpa_flash_attention(
                    q[:1, :, :min(64, seqlen), :],
                    k[:1, :, :min(64, seqlen), :],
                    v[:1, :, :min(64, seqlen), :], scale)
            del _t
            flash_ok = True
        except RuntimeError:
            pass

    if flash_ok:
        kernels.append(("sdpa_flash", sdpa_flash_attention))
    else:
        eff_ok = False
        if device.type == "cuda":
            try:
                with torch.no_grad():
                    _t = sdpa_efficient_attention(
                        q[:1, :, :min(64, seqlen), :],
                        k[:1, :, :min(64, seqlen), :],
                        v[:1, :, :min(64, seqlen), :], scale)
                del _t
                eff_ok = True
            except RuntimeError:
                pass
        if eff_ok:
            kernels.append(("sdpa_efficient", sdpa_efficient_attention))
        else:
            kernels.append(("sdpa_auto", sdpa_auto_attention))

    if _load_flash_attn() and device.type == "cuda":
        kernels.append(("flash_attn_pkg", flash_attn_pkg_attention))

    for kname, kfn in kernels:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        try:
            with torch.no_grad():
                timing = benchmark_kernel(kfn, q, k, v, scale, device)
                peak_vram = measure_peak_vram(kfn, q, k, v, scale, device)

                linf, rel_l2 = 0.0, 0.0
                if fp32_ref is not None:
                    out = kfn(q, k, v, scale)
                    linf, rel_l2 = numerical_error(out, fp32_ref)
                    del out

            med_s = timing["median_ms"] / 1000.0
            tflops = (flops / 1e12) / med_s if med_s > 0 else 0
            bw_gbs = (io_bytes / 1e9) / med_s if med_s > 0 else 0
            util_pct = (tflops / H100_BF16_TFLOPS * 100) if tflops > 0 else 0

            rec = {
                "kernel": kname,
                "batch": batch,
                "seqlen": seqlen,
                "dtype": dtype_name,
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
                "status": "OK",
                "median_ms": round(timing["median_ms"], 4),
                "mean_ms": round(timing["mean_ms"], 4),
                "min_ms": round(timing["min_ms"], 4),
                "p90_ms": round(timing["p90_ms"], 4),
                "p99_ms": round(timing["p99_ms"], 4),
                "tflops": round(tflops, 3),
                "bandwidth_gbs": round(bw_gbs, 2),
                "peak_vram_bytes": peak_vram,
                "peak_vram_mb": round(peak_vram / (1024**2), 1),
                "linf_vs_fp32": float(f"{linf:.6e}"),
                "rel_l2_vs_fp32": float(f"{rel_l2:.6e}"),
                "flops": flops,
                "io_bytes": io_bytes,
                "arithmetic_intensity": round(ai, 1),
                "roofline_class": rc,
                "h100_util_pct": round(util_pct, 2),
                "warmup_iters": WARMUP_ITERS,
                "bench_iters": BENCH_ITERS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            results.append(rec)

            print(f"  {kname:<22} med={timing['median_ms']:>9.3f}ms  "
                  f"TFLOPS={tflops:>7.2f}  BW={bw_gbs:>8.1f} GB/s  "
                  f"VRAM={peak_vram/(1024**2):>7.0f}MB  "
                  f"Linf={linf:.2e}  relL2={rel_l2:.2e}  "
                  f"util={util_pct:.1f}%")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  {kname:<22} ERROR: {e}")
            results.append({
                "kernel": kname, "batch": batch, "seqlen": seqlen,
                "dtype": dtype_name, "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM, "status": "OOM", "error": str(e),
            })
            if device.type == "cuda":
                torch.cuda.empty_cache()

    for r in results:
        with open(results_fp, "a") as f:
            f.write(json.dumps(r) + "\n")

    del q, k, v
    if fp32_ref is not None:
        del fp32_ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ---- Environment Info ----

def print_env(device):
    """Print full environment report."""
    print("=" * 72)
    print("  ATTENTION BENCHMARK HARNESS")
    print("  Naive vs FlashAttention-2 (SDPA) -- H100 80GB")
    print("=" * 72)
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        mem = torch.cuda.get_device_properties(device).total_mem
        print(f"  GPU         : {name}")
        print(f"  SM Compute  : {cap[0]}.{cap[1]}")
        print(f"  VRAM        : {mem / (1024**3):.1f} GB")
    try:
        import triton
        print(f"  Triton      : {triton.__version__}")
    except ImportError:
        print("  Triton      : not installed")
    try:
        import flash_attn
        print(f"  flash-attn  : {flash_attn.__version__}")
    except ImportError:
        print("  flash-attn  : not installed")

    if torch.cuda.is_available():
        print("\n  SDPA backends:")
        q_t = torch.randn(1, 1, 64, 128, device=device, dtype=torch.float16)
        for bname, backend in [
            ("FLASH_ATTENTION", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
            ("EFFICIENT_ATTENTION", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
            ("MATH", torch.nn.attention.SDPBackend.MATH),
        ]:
            try:
                with torch.nn.attention.sdpa_kernel([backend]):
                    _ = F.scaled_dot_product_attention(q_t, q_t, q_t)
                print(f"    {bname}: YES")
            except RuntimeError:
                print(f"    {bname}: NO")
        del q_t
        torch.cuda.empty_cache()

    print(f"\n  Config: heads={NUM_HEADS}, head_dim={HEAD_DIM}, "
          f"warmup={WARMUP_ITERS}, iters={BENCH_ITERS}")
    print(f"  Batches: {BATCH_SIZES}")
    print(f"  SeqLens: {SEQ_LENGTHS}")
    print(f"  Dtypes : {list(DTYPES.keys())}")
    print("=" * 72)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Attention benchmark harness")
    parser.add_argument("--output", type=str, default=DEFAULT_RESULTS_PATH,
                        help="Output JSONL path")
    args = parser.parse_args()

    device = get_device()
    print_env(device)

    results_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)

    if os.path.exists(results_path):
        os.remove(results_path)

    all_results = []
    total = len(BATCH_SIZES) * len(SEQ_LENGTHS) * len(DTYPES)
    idx = 0

    for dtype_name, dtype in DTYPES.items():
        for batch in BATCH_SIZES:
            for seqlen in SEQ_LENGTHS:
                idx += 1
                print(f"\n[{idx}/{total}]", end="")
                try:
                    res = run_config(batch, seqlen, dtype_name, dtype,
                                     device, results_path)
                    all_results.extend(res)
                except Exception as e:
                    print(f"\nFATAL: B={batch} N={seqlen} {dtype_name}: {e}")
                    traceback.print_exc()
                    err = {
                        "kernel": "FATAL_ERROR", "batch": batch,
                        "seqlen": seqlen, "dtype": dtype_name,
                        "status": "FATAL_ERROR", "error": str(e),
                    }
                    with open(results_path, "a") as f:
                        f.write(json.dumps(err) + "\n")
                    all_results.append(err)

    # Summary table
    print("\n\n" + "=" * 110)
    print("  SUMMARY TABLE")
    print("=" * 110)
    hdr = (f"{'Kernel':<22} {'B':>3} {'N':>5} {'DT':>4} "
           f"{'Med ms':>9} {'TFLOPS':>7} {'BW GB/s':>8} {'VRAM MB':>8} "
           f"{'Linf':>10} {'relL2':>10} {'Util%':>6}")
    print(hdr)
    print("-" * 110)
    for r in all_results:
        if r.get("status") == "OK":
            print(f"{r['kernel']:<22} {r['batch']:>3} {r['seqlen']:>5} "
                  f"{r['dtype']:>4} {r['median_ms']:>9.3f} "
                  f"{r['tflops']:>7.2f} {r['bandwidth_gbs']:>8.1f} "
                  f"{r['peak_vram_mb']:>8.0f} "
                  f"{r.get('linf_vs_fp32',0):>10.2e} "
                  f"{r.get('rel_l2_vs_fp32',0):>10.2e} "
                  f"{r.get('h100_util_pct',0):>6.1f}")
        elif r.get("status") == "SKIPPED":
            print(f"{r['kernel']:<22} {r['batch']:>3} {r['seqlen']:>5} "
                  f"{r['dtype']:>4} {'SKIPPED':>9}  {r.get('reason','')}")
        else:
            print(f"{r['kernel']:<22} {r.get('batch',''):>3} "
                  f"{r.get('seqlen',''):>5} {r.get('dtype',''):>4} "
                  f"{r.get('status','ERR'):>9}")

    print(f"\nResults: {results_path}")
    print(f"Configs: {idx}, Records: {len(all_results)}")

    # Speedup summary
    print("\n" + "=" * 80)
    print("  SPEEDUP: sdpa_flash / naive_attention")
    print("=" * 80)
    naive_map = {}
    flash_map = {}
    for r in all_results:
        if r.get("status") != "OK":
            continue
        key = (r["batch"], r["seqlen"], r["dtype"])
        if r["kernel"] == "naive_attention":
            naive_map[key] = r
        elif r["kernel"] in ("sdpa_flash", "sdpa_efficient", "sdpa_auto",
                             "flash_attn_pkg"):
            flash_map.setdefault(key, r)

    for key in sorted(naive_map.keys()):
        n = naive_map[key]
        if key in flash_map:
            f = flash_map[key]
            speedup = n["median_ms"] / f["median_ms"] if f["median_ms"] > 0 else 0
            print(f"  B={key[0]:>2} N={key[1]:>5} {key[2]:>4}  "
                  f"naive={n['median_ms']:>8.3f}ms  "
                  f"flash={f['median_ms']:>8.3f}ms  "
                  f"speedup={speedup:>6.2f}x")
        else:
            print(f"  B={key[0]:>2} N={key[1]:>5} {key[2]:>4}  "
                  f"naive={n['median_ms']:>8.3f}ms  flash=N/A")

    for key in sorted(flash_map.keys()):
        if key not in naive_map:
            f = flash_map[key]
            print(f"  B={key[0]:>2} N={key[1]:>5} {key[2]:>4}  "
                  f"naive=SKIPPED  flash={f['median_ms']:>8.3f}ms")


if __name__ == "__main__":
    main()
