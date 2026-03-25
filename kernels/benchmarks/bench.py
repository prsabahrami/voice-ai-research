#!/usr/bin/env python3
"""
Attention Benchmark Harness: Naive vs FlashAttention-2 (SDPA) on H100
Measures: latency (ms), bandwidth (GB/s), peak VRAM (bytes), numerical error vs fp32 baseline.
Configs: batch {1, 8, 32} x seqlen {512, 2048, 8192} x dtype {fp16, bf16}
Output: results.jsonl (one JSON object per config)
"""

import json
import os
import sys
import time
import gc
import traceback

import torch
import torch.nn.functional as F

# ---- Constants ----
NUM_HEADS = 32
HEAD_DIM = 128
WARMUP_ITERS = 10
BENCH_ITERS = 50
BATCH_SIZES = [1, 8, 32]
SEQ_LENGTHS = [512, 2048, 8192]
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

RESULTS_PATH = os.path.expanduser("~/voice-ai-research/kernels/benchmarks/results.jsonl")

# ---- H100 SXM constants ----
H100_HBM_BW_GBS = 3350.0   # GB/s theoretical peak HBM3
H100_BF16_TFLOPS = 989.4    # TFLOPS BF16 tensor core


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: No CUDA device found. Running on CPU (timings meaningless).", file=sys.stderr)
    return torch.device("cpu")


def naive_attention(q, k, v, scale):
    """Standard scaled dot-product attention (materializes full NxN matrix)."""
    # q, k, v: (B, H, N, D)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, N, N)
    attn_weights = torch.softmax(attn_weights, dim=-1)
    out = torch.matmul(attn_weights, v)  # (B, H, N, D)
    return out


def sdpa_flash_attention(q, k, v, scale):
    """PyTorch SDPA with flash_attention backend (dispatches to cuDNN FA2/FA3 on H100)."""
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return out


def sdpa_math_attention(q, k, v, scale):
    """PyTorch SDPA with math backend (reference implementation)."""
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(q, k, v, scale=scale)
    return out


def compute_fp32_reference(q_fp32, k_fp32, v_fp32, scale):
    """Compute fp32 reference for numerical error comparison."""
    with torch.no_grad():
        return naive_attention(q_fp32, k_fp32, v_fp32, scale)


def numerical_error(out, ref_fp32):
    """Compute L-inf and relative L2 error against fp32 reference."""
    out_fp32 = out.float()
    ref = ref_fp32.float()
    diff = (out_fp32 - ref).abs()
    linf = diff.max().item()
    l2_err = torch.norm(diff).item()
    l2_ref = torch.norm(ref).item()
    rel_l2 = l2_err / (l2_ref + 1e-12)
    return linf, rel_l2


def attention_flops(batch, seqlen, num_heads, head_dim):
    """FLOPs for standard attention: 2*B*H*N*N*D (QK^T) + 2*B*H*N*N*D (attn@V) + softmax overhead."""
    # QK^T: 2*B*H*N*D*N = 2*B*H*N^2*D
    # attn@V: 2*B*H*N*N*D = 2*B*H*N^2*D
    # Total dominant: 4*B*H*N^2*D
    return 4 * batch * num_heads * seqlen * seqlen * head_dim


def attention_bytes(batch, seqlen, num_heads, head_dim, dtype_bytes):
    """Memory traffic estimate: read Q,K,V + write O. Minimum for memory-bound analysis."""
    # Q, K, V each: B*H*N*D elements
    # O: B*H*N*D elements
    # Total: 4 * B*H*N*D * dtype_bytes
    return 4 * batch * num_heads * seqlen * head_dim * dtype_bytes


def benchmark_kernel(fn, q, k, v, scale, device, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Benchmark a kernel using CUDA events for accurate GPU timing."""
    if device.type == "cpu":
        # CPU fallback timing
        for _ in range(warmup):
            _ = fn(q, k, v, scale)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = fn(q, k, v, scale)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times.sort()
        return {
            "median_ms": times[len(times)//2],
            "mean_ms": sum(times)/len(times),
            "min_ms": times[0],
            "p90_ms": times[int(0.9*len(times))],
            "p99_ms": times[int(0.99*len(times))],
        }

    # GPU timing with CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    # Warmup
    for _ in range(warmup):
        _ = fn(q, k, v, scale)
    torch.cuda.synchronize()

    # Benchmark
    for i in range(iters):
        start_events[i].record()
        _ = fn(q, k, v, scale)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return {
        "median_ms": times[len(times)//2],
        "mean_ms": sum(times)/len(times),
        "min_ms": times[0],
        "p90_ms": times[int(0.9*len(times))],
        "p99_ms": times[int(0.99*len(times))],
    }


def get_peak_vram(fn, q, k, v, scale, device):
    """Measure peak VRAM usage for a single forward pass."""
    if device.type == "cpu":
        return 0
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    _ = fn(q, k, v, scale)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device)
    return peak


def run_single_config(batch, seqlen, dtype_name, dtype, device, results_file):
    """Run naive + flash attention benchmarks for one config, write results to file."""
    scale = HEAD_DIM ** -0.5
    dtype_bytes = 2  # fp16 and bf16 are both 2 bytes

    print(f"\n{'='*70}")
    print(f"Config: batch={batch}, seqlen={seqlen}, dtype={dtype_name}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
    print(f"{'='*70}")

    # Estimate memory needed for naive attention (full N*N matrix)
    naive_mem_gb = batch * NUM_HEADS * seqlen * seqlen * dtype_bytes / (1024**3)
    skip_naive = naive_mem_gb > 40  # Skip if would use > 40GB for attention matrix alone

    # Generate inputs
    torch.manual_seed(42)
    q = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(batch, NUM_HEADS, seqlen, HEAD_DIM, device=device, dtype=dtype)

    # FP32 reference (on same device, may OOM for large configs)
    fp32_ref = None
    try:
        if not skip_naive:
            q32 = q.float()
            k32 = k.float()
            v32 = v.float()
            with torch.no_grad():
                fp32_ref = compute_fp32_reference(q32, k32, v32, scale)
            del q32, k32, v32
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  FP32 reference OOM: {e}")
        fp32_ref = None

    flops = attention_flops(batch, seqlen, NUM_HEADS, HEAD_DIM)
    mem_bytes = attention_bytes(batch, seqlen, NUM_HEADS, HEAD_DIM, dtype_bytes)

    results = []

    # ---- Naive Attention ----
    if skip_naive:
        print(f"  SKIP naive attention (estimated {naive_mem_gb:.1f} GB for attn matrix)")
        naive_result = {
            "kernel": "naive_attention",
            "batch": batch,
            "seqlen": seqlen,
            "dtype": dtype_name,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "status": "SKIPPED",
            "reason": f"attn matrix would use {naive_mem_gb:.1f} GB",
        }
        results.append(naive_result)
    else:
        try:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                timing = benchmark_kernel(naive_attention, q, k, v, scale, device)
                peak_vram = get_peak_vram(naive_attention, q, k, v, scale, device)

                # Numerical error
                linf, rel_l2 = (0.0, 0.0)
                if fp32_ref is not None:
                    out = naive_attention(q, k, v, scale)
                    linf, rel_l2 = numerical_error(out, fp32_ref)
                    del out

            tflops = (flops / 1e12) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0
            bw_gbs = (mem_bytes / 1e9) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0

            naive_result = {
                "kernel": "naive_attention",
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
                "tflops": round(tflops, 2),
                "bandwidth_gbs": round(bw_gbs, 2),
                "peak_vram_bytes": peak_vram,
                "peak_vram_mb": round(peak_vram / (1024**2), 1),
                "linf_vs_fp32": linf,
                "rel_l2_vs_fp32": rel_l2,
                "flops": flops,
                "mem_bytes_estimate": mem_bytes,
            }
            results.append(naive_result)
            print(f"  Naive:  median={timing['median_ms']:.3f}ms  TFLOPS={tflops:.2f}  BW={bw_gbs:.1f} GB/s  VRAM={peak_vram/(1024**2):.0f}MB  Linf={linf:.2e}  relL2={rel_l2:.2e}")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  Naive attention OOM/error: {e}")
            naive_result = {
                "kernel": "naive_attention",
                "batch": batch,
                "seqlen": seqlen,
                "dtype": dtype_name,
                "num_heads": NUM_HEADS,
                "head_dim": HEAD_DIM,
                "status": "OOM",
                "error": str(e),
            }
            results.append(naive_result)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- SDPA Flash Attention ----
    try:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            # Test if flash backend is available
            flash_available = True
            try:
                test_out = sdpa_flash_attention(q[:1, :, :64, :], k[:1, :, :64, :], v[:1, :, :64, :], scale)
                del test_out
            except RuntimeError as e:
                print(f"  Flash backend not available: {e}")
                flash_available = False

            if flash_available:
                timing = benchmark_kernel(sdpa_flash_attention, q, k, v, scale, device)
                peak_vram = get_peak_vram(sdpa_flash_attention, q, k, v, scale, device)

                linf, rel_l2 = (0.0, 0.0)
                if fp32_ref is not None:
                    out = sdpa_flash_attention(q, k, v, scale)
                    linf, rel_l2 = numerical_error(out, fp32_ref)
                    del out

                tflops = (flops / 1e12) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0
                bw_gbs = (mem_bytes / 1e9) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0

                flash_result = {
                    "kernel": "sdpa_flash_attention",
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
                    "tflops": round(tflops, 2),
                    "bandwidth_gbs": round(bw_gbs, 2),
                    "peak_vram_bytes": peak_vram,
                    "peak_vram_mb": round(peak_vram / (1024**2), 1),
                    "linf_vs_fp32": linf,
                    "rel_l2_vs_fp32": rel_l2,
                    "flops": flops,
                    "mem_bytes_estimate": mem_bytes,
                }
                results.append(flash_result)
                print(f"  Flash:  median={timing['median_ms']:.3f}ms  TFLOPS={tflops:.2f}  BW={bw_gbs:.1f} GB/s  VRAM={peak_vram/(1024**2):.0f}MB  Linf={linf:.2e}  relL2={rel_l2:.2e}")
            else:
                # Fall back to SDPA efficient_attention or math
                print("  Trying SDPA efficient_attention backend...")
                try:
                    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                        test_out = F.scaled_dot_product_attention(q[:1, :, :64, :], k[:1, :, :64, :], v[:1, :, :64, :], scale=scale)
                    del test_out

                    def sdpa_efficient(q, k, v, scale):
                        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]):
                            return F.scaled_dot_product_attention(q, k, v, scale=scale)

                    timing = benchmark_kernel(sdpa_efficient, q, k, v, scale, device)
                    peak_vram = get_peak_vram(sdpa_efficient, q, k, v, scale, device)
                    linf, rel_l2 = (0.0, 0.0)
                    if fp32_ref is not None:
                        out = sdpa_efficient(q, k, v, scale)
                        linf, rel_l2 = numerical_error(out, fp32_ref)
                        del out

                    tflops = (flops / 1e12) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0
                    bw_gbs = (mem_bytes / 1e9) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0

                    flash_result = {
                        "kernel": "sdpa_efficient_attention",
                        "batch": batch,
                        "seqlen": seqlen,
                        "dtype": dtype_name,
                        "num_heads": NUM_HEADS,
                        "head_dim": HEAD_DIM,
                        "status": "OK",
                        "note": "flash backend unavailable, using efficient_attention",
                        "median_ms": round(timing["median_ms"], 4),
                        "mean_ms": round(timing["mean_ms"], 4),
                        "min_ms": round(timing["min_ms"], 4),
                        "p90_ms": round(timing["p90_ms"], 4),
                        "p99_ms": round(timing["p99_ms"], 4),
                        "tflops": round(tflops, 2),
                        "bandwidth_gbs": round(bw_gbs, 2),
                        "peak_vram_bytes": peak_vram,
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "linf_vs_fp32": linf,
                        "rel_l2_vs_fp32": rel_l2,
                        "flops": flops,
                        "mem_bytes_estimate": mem_bytes,
                    }
                    results.append(flash_result)
                    print(f"  Efficient: median={timing['median_ms']:.3f}ms  TFLOPS={tflops:.2f}  BW={bw_gbs:.1f} GB/s  VRAM={peak_vram/(1024**2):.0f}MB")
                except Exception as e2:
                    print(f"  Efficient attention also failed: {e2}")
                    # Try with default SDPA (auto-dispatch)
                    def sdpa_auto(q, k, v, scale):
                        return F.scaled_dot_product_attention(q, k, v, scale=scale)

                    timing = benchmark_kernel(sdpa_auto, q, k, v, scale, device)
                    peak_vram = get_peak_vram(sdpa_auto, q, k, v, scale, device)
                    linf, rel_l2 = (0.0, 0.0)
                    if fp32_ref is not None:
                        out = sdpa_auto(q, k, v, scale)
                        linf, rel_l2 = numerical_error(out, fp32_ref)
                        del out

                    tflops = (flops / 1e12) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0
                    bw_gbs = (mem_bytes / 1e9) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0

                    flash_result = {
                        "kernel": "sdpa_auto",
                        "batch": batch,
                        "seqlen": seqlen,
                        "dtype": dtype_name,
                        "num_heads": NUM_HEADS,
                        "head_dim": HEAD_DIM,
                        "status": "OK",
                        "note": "flash+efficient unavailable, auto backend",
                        "median_ms": round(timing["median_ms"], 4),
                        "mean_ms": round(timing["mean_ms"], 4),
                        "min_ms": round(timing["min_ms"], 4),
                        "p90_ms": round(timing["p90_ms"], 4),
                        "p99_ms": round(timing["p99_ms"], 4),
                        "tflops": round(tflops, 2),
                        "bandwidth_gbs": round(bw_gbs, 2),
                        "peak_vram_bytes": peak_vram,
                        "peak_vram_mb": round(peak_vram / (1024**2), 1),
                        "linf_vs_fp32": linf,
                        "rel_l2_vs_fp32": rel_l2,
                        "flops": flops,
                        "mem_bytes_estimate": mem_bytes,
                    }
                    results.append(flash_result)
                    print(f"  Auto SDPA: median={timing['median_ms']:.3f}ms")

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  SDPA OOM/error: {e}")
        flash_result = {
            "kernel": "sdpa_flash_attention",
            "batch": batch,
            "seqlen": seqlen,
            "dtype": dtype_name,
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "status": "OOM",
            "error": str(e),
        }
        results.append(flash_result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Try flash-attn package if installed ----
    try:
        from flash_attn import flash_attn_func
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        # flash_attn_func expects (B, N, H, D) layout
        q_fa = q.transpose(1, 2).contiguous()  # (B, N, H, D)
        k_fa = k.transpose(1, 2).contiguous()
        v_fa = v.transpose(1, 2).contiguous()

        def fa2_kernel(q, k, v, scale):
            return flash_attn_func(q, k, v, softmax_scale=scale)

        with torch.no_grad():
            timing = benchmark_kernel(fa2_kernel, q_fa, k_fa, v_fa, scale, device)
            peak_vram = get_peak_vram(fa2_kernel, q_fa, k_fa, v_fa, scale, device)

            linf, rel_l2 = (0.0, 0.0)
            if fp32_ref is not None:
                out_fa = fa2_kernel(q_fa, k_fa, v_fa, scale)
                # Convert back to (B, H, N, D) for comparison
                out_fa = out_fa.transpose(1, 2)
                linf, rel_l2 = numerical_error(out_fa, fp32_ref)
                del out_fa

        tflops = (flops / 1e12) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0
        bw_gbs = (mem_bytes / 1e9) / (timing["median_ms"] / 1000.0) if timing["median_ms"] > 0 else 0

        fa2_result = {
            "kernel": "flash_attn_2_package",
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
            "tflops": round(tflops, 2),
            "bandwidth_gbs": round(bw_gbs, 2),
            "peak_vram_bytes": peak_vram,
            "peak_vram_mb": round(peak_vram / (1024**2), 1),
            "linf_vs_fp32": linf,
            "rel_l2_vs_fp32": rel_l2,
            "flops": flops,
            "mem_bytes_estimate": mem_bytes,
        }
        results.append(fa2_result)
        print(f"  FA2pkg: median={timing['median_ms']:.3f}ms  TFLOPS={tflops:.2f}  BW={bw_gbs:.1f} GB/s  VRAM={peak_vram/(1024**2):.0f}MB  Linf={linf:.2e}  relL2={rel_l2:.2e}")
        del q_fa, k_fa, v_fa
    except ImportError:
        pass  # flash-attn package not installed
    except Exception as e:
        print(f"  flash-attn package error: {e}")

    # Write results
    for r in results:
        with open(results_file, "a") as f:
            f.write(json.dumps(r) + "\n")

    # Cleanup
    del q, k, v
    if fp32_ref is not None:
        del fp32_ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


def print_env_info(device):
    """Print environment information."""
    print("=" * 70)
    print("ATTENTION BENCHMARK HARNESS")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(device)}")
        print(f"Device capability: {torch.cuda.get_device_capability(device)}")
        total_mem = torch.cuda.get_device_properties(device).total_mem
        print(f"Total VRAM: {total_mem / (1024**3):.1f} GB")
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("Triton: not installed")
    try:
        import flash_attn
        print(f"flash-attn version: {flash_attn.__version__}")
    except ImportError:
        print("flash-attn: not installed")

    # Check SDPA backends
    if torch.cuda.is_available():
        print("\nSDPA Backend availability:")
        q_test = torch.randn(1, 1, 64, 128, device=device, dtype=torch.float16)
        for backend_name, backend in [
            ("FLASH_ATTENTION", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
            ("EFFICIENT_ATTENTION", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
            ("MATH", torch.nn.attention.SDPBackend.MATH),
        ]:
            try:
                with torch.nn.attention.sdpa_kernel([backend]):
                    _ = F.scaled_dot_product_attention(q_test, q_test, q_test)
                print(f"  {backend_name}: available")
            except RuntimeError:
                print(f"  {backend_name}: NOT available")
        del q_test
        torch.cuda.empty_cache()
    print("=" * 70)


def main():
    device = get_device()
    print_env_info(device)

    # Prepare output directory
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    # Clear previous results
    if os.path.exists(RESULTS_PATH):
        os.remove(RESULTS_PATH)

    all_results = []
    total_configs = len(BATCH_SIZES) * len(SEQ_LENGTHS) * len(DTYPES)
    config_num = 0

    for dtype_name, dtype in DTYPES.items():
        for batch in BATCH_SIZES:
            for seqlen in SEQ_LENGTHS:
                config_num += 1
                print(f"\n[{config_num}/{total_configs}]", end="")
                try:
                    results = run_single_config(batch, seqlen, dtype_name, dtype, device, RESULTS_PATH)
                    all_results.extend(results)
                except Exception as e:
                    print(f"\nFATAL ERROR for config batch={batch} seqlen={seqlen} dtype={dtype_name}: {e}")
                    traceback.print_exc()
                    error_result = {
                        "kernel": "ERROR",
                        "batch": batch,
                        "seqlen": seqlen,
                        "dtype": dtype_name,
                        "status": "FATAL_ERROR",
                        "error": str(e),
                    }
                    with open(RESULTS_PATH, "a") as f:
                        f.write(json.dumps(error_result) + "\n")
                    all_results.append(error_result)

    # Print summary table
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Kernel':<28} {'Batch':>5} {'SeqLen':>6} {'DType':>5} {'Med ms':>8} {'TFLOPS':>7} {'BW GB/s':>8} {'VRAM MB':>8} {'Linf':>10} {'relL2':>10}")
    print("-" * 100)
    for r in all_results:
        if r.get("status") == "OK":
            print(f"{r['kernel']:<28} {r['batch']:>5} {r['seqlen']:>6} {r['dtype']:>5} {r['median_ms']:>8.3f} {r['tflops']:>7.2f} {r['bandwidth_gbs']:>8.1f} {r['peak_vram_mb']:>8.0f} {r.get('linf_vs_fp32', 0):>10.2e} {r.get('rel_l2_vs_fp32', 0):>10.2e}")
        elif r.get("status") == "SKIPPED":
            print(f"{r['kernel']:<28} {r['batch']:>5} {r['seqlen']:>6} {r['dtype']:>5} {'SKIPPED':>8} {r.get('reason', '')}")
        else:
            print(f"{r['kernel']:<28} {r['batch']:>5} {r['seqlen']:>6} {r['dtype']:>5} {r.get('status', 'ERR'):>8}")

    print(f"\nResults written to: {RESULTS_PATH}")
    print(f"Total configs: {config_num}, Total results: {len(all_results)}")


if __name__ == "__main__":
    main()
