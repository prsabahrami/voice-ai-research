"""
quant_kernels.py
================
Production-quality INT8/FP8 quantization kernels for H100 GPU.
Uses torch._scaled_mm (PyTorch 2.1+) for INT8/FP8 GEMM, and Triton-based
patterns for quantized attention.

All functions:
  - Accept standard PyTorch tensors (CUDA expected for real execution)
  - Return (output_tensor, metadata_dict) where metadata contains latency + error stats
  - Gracefully fall back or raise informative errors when GPU features are missing

Author: miniQuant kernel-research branch
Target: Lambda H100 80GB, PyTorch 2.x, Triton
"""

import time
import warnings
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

def _check_cuda(fn_name: str) -> None:
    if not CUDA_AVAILABLE:
        raise RuntimeError(
            f"{fn_name}: CUDA is not available. This kernel requires a CUDA-capable GPU."
        )

def _check_fp8_support() -> bool:
    """Return True if the current device supports FP8 dtypes (H100 / Ada Lovelace)."""
    if not CUDA_AVAILABLE:
        return False
    # FP8 dtypes were added in PyTorch 2.1 alongside CUDA compute capability >= 8.9
    has_dtype = hasattr(torch, "float8_e4m3fn") and hasattr(torch, "float8_e5m2")
    if not has_dtype:
        return False
    cc_major, _ = torch.cuda.get_device_capability()
    return cc_major >= 9  # H100 is sm_90

def _check_scaled_mm_support() -> bool:
    """Return True if torch._scaled_mm is available (PyTorch 2.1+)."""
    return hasattr(torch, "_scaled_mm")

FP8_SUPPORTED = _check_fp8_support()
SCALED_MM_AVAILABLE = _check_scaled_mm_support()

# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _quantize_int8(
    x: torch.Tensor,
    per_tensor: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 quantization.
    
    Returns (quantized_int8, scale_fp32) where scale is the per-tensor or
    per-channel absmax / 127.
    """
    if per_tensor:
        scale = x.abs().max().clamp(min=1e-8) / 127.0
        xq = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return xq, scale.reshape(1)
    else:
        # Per-row (row-wise) scaling
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8) / 127.0
        xq = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return xq, scale.squeeze(-1)


def _quantize_fp8_e4m3(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to float8_e4m3fn (used for forward pass activations/weights)."""
    if not FP8_SUPPORTED:
        raise RuntimeError(
            "FP8 quantization requires PyTorch >= 2.1 and a GPU with compute capability >= 9.0 (H100)."
        )
    # float8_e4m3fn max value is 448.0
    FP8_MAX = 448.0
    scale = x.abs().max().clamp(min=1e-8) / FP8_MAX
    xq = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return xq, scale.reshape(1).to(torch.float32)


def _quantize_fp8_e5m2(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize to float8_e5m2 (used for gradient tensors)."""
    if not FP8_SUPPORTED:
        raise RuntimeError(
            "FP8 (e5m2) quantization requires PyTorch >= 2.1 and a GPU with compute capability >= 9.0 (H100)."
        )
    # float8_e5m2 max value is 57344.0
    FP8_MAX = 57344.0
    scale = x.abs().max().clamp(min=1e-8) / FP8_MAX
    xq = (x / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e5m2)
    return xq, scale.reshape(1).to(torch.float32)


# ---------------------------------------------------------------------------
# Timing utilities
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, warmup: int = 3, repeats: int = 10) -> Tuple[float, object]:
    """Run fn() with CUDA event timing. Returns (median_latency_ms, last_result)."""
    result = None
    if CUDA_AVAILABLE:
        # Warmup
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(repeats):
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[len(times) // 2]
    else:
        # CPU fallback timing
        for _ in range(warmup):
            result = fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            result = fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times.sort()
        median_ms = times[len(times) // 2]
    return median_ms, result


# ---------------------------------------------------------------------------
# FP16 baselines
# ---------------------------------------------------------------------------

def fp16_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """FP16 GEMM baseline using torch.matmul.
    
    Args:
        A: (M, K) float16 tensor
        B: (K, N) float16 tensor

    Returns:
        (C, metadata)  C: (M, N) float32 tensor (accumulated in fp32)
    """
    A = A.to(torch.float16)
    B = B.to(torch.float16)

    def _run():
        return torch.matmul(A, B).to(torch.float32)

    latency_ms, C = _cuda_time_ms(_run)
    return C, {"latency_ms": latency_ms, "dtype": "fp16", "method": "torch.matmul"}


def fp16_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Scaled dot-product attention in FP16 (baseline).
    
    Args:
        Q: (B, S, D)
        K: (B, S, D)
        V: (B, S, D)
        scale: attention scale; defaults to 1/sqrt(D)

    Returns:
        (output, metadata)
    """
    Q = Q.to(torch.float16)
    K = K.to(torch.float16)
    V = V.to(torch.float16)
    d = Q.shape[-1]
    s = scale if scale is not None else (d ** -0.5)

    def _run():
        # Use PyTorch SDPA when available (FlashAttention-2 path on H100)
        if hasattr(F, "scaled_dot_product_attention"):
            return F.scaled_dot_product_attention(Q, K, V, scale=s)
        # Manual fallback
        attn = torch.bmm(Q, K.transpose(-1, -2)) * s
        attn = torch.softmax(attn, dim=-1)
        return torch.bmm(attn, V)

    latency_ms, out = _cuda_time_ms(_run)
    return out.to(torch.float32), {"latency_ms": latency_ms, "dtype": "fp16", "method": "sdpa_fp16"}


# ---------------------------------------------------------------------------
# INT8 kernels
# ---------------------------------------------------------------------------

def int8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict]:
    """INT8 GEMM using torch._scaled_mm (requires PyTorch 2.1+).

    The inputs are quantized symmetrically to INT8, scaled_mm is called with
    the per-tensor scales, and the output is dequantized to out_dtype.

    Args:
        A: (M, K) floating-point tensor (any dtype; will be cast + quantized)
        B: (K, N) floating-point tensor
        out_dtype: output accumulation dtype (float16 or float32)

    Returns:
        (C_dequant, metadata)
    """
    _check_cuda("int8_gemm")

    A_fp = A.to(torch.float32)
    B_fp = B.to(torch.float32)

    A_q, scale_a = _quantize_int8(A_fp)
    B_q, scale_b = _quantize_int8(B_fp)

    A_q = A_q.contiguous()
    B_q = B_q.t().contiguous()  # (N, K)

    scale_a_dev = scale_a.to(DEVICE)
    scale_b_dev = scale_b.to(DEVICE)

    if not SCALED_MM_AVAILABLE:
        warnings.warn(
            "int8_gemm: torch._scaled_mm not available (need PyTorch >= 2.1). "
            "Falling back to dequantize + fp16 matmul."
        )
        A_deq = A_q.to(torch.float16) * scale_a.item()
        B_deq = B_q.t().to(torch.float16) * scale_b.item()

        def _run():
            return torch.matmul(A_deq, B_deq).to(out_dtype)

        latency_ms, C = _cuda_time_ms(_run)
        return C, {
            "latency_ms": latency_ms,
            "dtype": "int8_fallback",
            "method": "dequant_fp16_matmul",
            "scale_a": scale_a.item(),
            "scale_b": scale_b.item(),
        }

    def _run():
        return torch._scaled_mm(
            A_q,
            B_q,
            scale_a=scale_a_dev,
            scale_b=scale_b_dev,
            out_dtype=out_dtype,
        )

    latency_ms, C = _cuda_time_ms(_run)
    return C, {
        "latency_ms": latency_ms,
        "dtype": "int8",
        "method": "torch._scaled_mm",
        "scale_a": scale_a.item(),
        "scale_b": scale_b.item(),
    }


def int8_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Quantized attention: Q,K in INT8, accumulate in FP32, softmax in FP16, V in INT8.

    Implements:
      1. Quantize Q, K to INT8 (per-tensor symmetric)
      2. Compute attention scores: S = Q_q @ K_q^T using torch._scaled_mm -> FP32
      3. Scale attention scores, apply softmax in FP16
      4. Quantize V to INT8, compute output = softmax_weights @ V_q using scaled_mm

    Args:
        Q: (B, S, D) any float dtype
        K: (B, S, D) any float dtype
        V: (B, S, D) any float dtype
        scale: attention scale; defaults to 1/sqrt(D)

    Returns:
        (output, metadata)
    """
    _check_cuda("int8_attention")

    B, S, D = Q.shape
    attn_scale = scale if scale is not None else (D ** -0.5)

    Q_fp = Q.to(torch.float32)
    K_fp = K.to(torch.float32)
    V_fp = V.to(torch.float32)

    # Reshape to 2D for quantization: (B*S, D)
    Q2 = Q_fp.reshape(B * S, D)
    K2 = K_fp.reshape(B * S, D)
    V2 = V_fp.reshape(B * S, D)

    Q_q, scale_q = _quantize_int8(Q2)
    K_q, scale_k = _quantize_int8(K2)
    V_q, scale_v = _quantize_int8(V2)

    scale_q_dev = scale_q.to(DEVICE)
    scale_k_dev = scale_k.to(DEVICE)
    scale_v_dev = scale_v.to(DEVICE)

    if not SCALED_MM_AVAILABLE:
        warnings.warn(
            "int8_attention: torch._scaled_mm not available. Falling back to fp16 attention."
        )
        return fp16_attention(Q, K, V, scale=attn_scale)

    def _run():
        results = []
        for b in range(B):
            q_b = Q_q[b * S : (b + 1) * S].contiguous()  # (S, D)
            k_b = K_q[b * S : (b + 1) * S].t().contiguous()  # (D, S)
            v_b = V_q[b * S : (b + 1) * S].contiguous()  # (S, D)

            scores = torch._scaled_mm(
                q_b,
                k_b,
                scale_a=scale_q_dev,
                scale_b=scale_k_dev,
                out_dtype=torch.float32,
            )  # (S, S)
            scores = scores * attn_scale
            weights = torch.softmax(scores.to(torch.float16), dim=-1).to(torch.float32)

            # Re-quantize softmax weights for the V matmul
            w_q, scale_w = _quantize_int8(weights)
            scale_w_dev = scale_w.to(DEVICE)
            w_q = w_q.contiguous()  # (S, S)
            v_b_t = v_b.t().contiguous()  # (D, S)

            out_b = torch._scaled_mm(
                w_q,
                v_b_t,
                scale_a=scale_w_dev,
                scale_b=scale_v_dev,
                out_dtype=torch.float32,
            )
            results.append(out_b)
        return torch.stack(results, dim=0)  # (B, S, D)

    latency_ms, output = _cuda_time_ms(_run)
    return output, {
        "latency_ms": latency_ms,
        "dtype": "int8",
        "method": "int8_attention_scaled_mm",
        "scale_q": scale_q.item(),
        "scale_k": scale_k.item(),
        "scale_v": scale_v.item(),
    }


# ---------------------------------------------------------------------------
# FP8 kernels
# ---------------------------------------------------------------------------

def fp8_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, Dict]:
    """FP8 GEMM using torch._scaled_mm with Float8 dtypes (H100 native).

    Uses float8_e4m3fn for both operands (standard forward-pass format).

    Args:
        A: (M, K) floating-point tensor
        B: (K, N) floating-point tensor
        out_dtype: output dtype (float16 or float32)

    Returns:
        (C, metadata)
    """
    _check_cuda("fp8_gemm")

    if not FP8_SUPPORTED:
        warnings.warn(
            "fp8_gemm: FP8 not supported on this device (requires H100 / sm_90 + PyTorch >= 2.1). "
            "Falling back to int8_gemm."
        )
        return int8_gemm(A, B, out_dtype=out_dtype)

    A_fp = A.to(torch.float32)
    B_fp = B.to(torch.float32)

    A_q, scale_a = _quantize_fp8_e4m3(A_fp)
    B_q, scale_b = _quantize_fp8_e4m3(B_fp)

    A_q = A_q.contiguous()
    B_q = B_q.t().contiguous()  # (N, K) for scaled_mm

    scale_a_dev = scale_a.to(DEVICE)
    scale_b_dev = scale_b.to(DEVICE)

    def _run():
        return torch._scaled_mm(
            A_q,
            B_q,
            scale_a=scale_a_dev,
            scale_b=scale_b_dev,
            out_dtype=out_dtype,
        )

    latency_ms, C = _cuda_time_ms(_run)
    return C, {
        "latency_ms": latency_ms,
        "dtype": "fp8_e4m3fn",
        "method": "torch._scaled_mm_fp8",
        "scale_a": scale_a.item(),
        "scale_b": scale_b.item(),
    }


def fp8_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: Optional[float] = None,
    use_e5m2_grad: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """FP8 attention: e4m3fn for forward pass, e5m2 for gradient-destined tensors.

    Strategy:
      - Q, K quantized to float8_e4m3fn (forward)
      - V quantized to float8_e4m3fn (forward) or float8_e5m2 if use_e5m2_grad
      - Attention scores accumulated in FP32
      - Softmax in FP16
      - Output matmul (softmax_weights @ V) with FP8

    Args:
        Q, K, V: (B, S, D) float tensors
        scale: attention scale
        use_e5m2_grad: if True, use e5m2 dtype for V (gradient regime)

    Returns:
        (output, metadata)
    """
    _check_cuda("fp8_attention")

    if not FP8_SUPPORTED:
        warnings.warn(
            "fp8_attention: FP8 not supported on this device. Falling back to int8_attention."
        )
        return int8_attention(Q, K, V, scale=scale)

    B, S, D = Q.shape
    attn_scale = scale if scale is not None else (D ** -0.5)

    Q_fp = Q.to(torch.float32)
    K_fp = K.to(torch.float32)
    V_fp = V.to(torch.float32)

    Q2 = Q_fp.reshape(B * S, D)
    K2 = K_fp.reshape(B * S, D)
    V2 = V_fp.reshape(B * S, D)

    Q_q, scale_q = _quantize_fp8_e4m3(Q2)
    K_q, scale_k = _quantize_fp8_e4m3(K2)

    if use_e5m2_grad:
        V_q, scale_v = _quantize_fp8_e5m2(V2)
    else:
        V_q, scale_v = _quantize_fp8_e4m3(V2)

    scale_q_dev = scale_q.to(DEVICE)
    scale_k_dev = scale_k.to(DEVICE)
    scale_v_dev = scale_v.to(DEVICE)

    def _run():
        results = []
        for b in range(B):
            q_b = Q_q[b * S : (b + 1) * S].contiguous()  # (S, D)
            k_b = K_q[b * S : (b + 1) * S].t().contiguous()  # (D, S)
            v_b = V_q[b * S : (b + 1) * S].contiguous()  # (S, D)

            scores = torch._scaled_mm(
                q_b,
                k_b,
                scale_a=scale_q_dev,
                scale_b=scale_k_dev,
                out_dtype=torch.float32,
            )  # (S, S)
            scores = scores * attn_scale
            weights = torch.softmax(scores.to(torch.float16), dim=-1).to(torch.float32)

            # Quantize softmax output for V matmul
            w_q, scale_w = _quantize_fp8_e4m3(weights)
            scale_w_dev = scale_w.to(DEVICE)
            v_b_t = v_b.t().contiguous()  # (D, S)

            out_b = torch._scaled_mm(
                w_q,
                v_b_t,
                scale_a=scale_w_dev,
                scale_b=scale_v_dev,
                out_dtype=torch.float32,
            )
            results.append(out_b)
        return torch.stack(results, dim=0)

    latency_ms, output = _cuda_time_ms(_run)
    dtype_str = "fp8_e4m3fn" if not use_e5m2_grad else "fp8_e4m3fn+e5m2"
    return output, {
        "latency_ms": latency_ms,
        "dtype": dtype_str,
        "method": "fp8_attention_scaled_mm",
        "scale_q": scale_q.item(),
        "scale_k": scale_k.item(),
        "scale_v": scale_v.item(),
    }


# ---------------------------------------------------------------------------
# Mixed-precision GEMM: INT8 compute, FP16 accumulation
# ---------------------------------------------------------------------------

def mixed_precision_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
) -> Tuple[torch.Tensor, Dict]:
    """INT8 compute with FP16 accumulation.

    Quantizes A and B to INT8, performs scaled_mm with out_dtype=float16,
    which gives FP16 accumulated output without a separate dequant step.

    Args:
        A: (M, K) float tensor
        B: (K, N) float tensor

    Returns:
        (C_fp16, metadata)
    """
    _check_cuda("mixed_precision_gemm")

    A_fp = A.to(torch.float32)
    B_fp = B.to(torch.float32)

    A_q, scale_a = _quantize_int8(A_fp)
    B_q, scale_b = _quantize_int8(B_fp)

    A_q = A_q.contiguous()
    B_q = B_q.t().contiguous()

    scale_a_dev = scale_a.to(DEVICE)
    scale_b_dev = scale_b.to(DEVICE)

    if not SCALED_MM_AVAILABLE:
        warnings.warn(
            "mixed_precision_gemm: torch._scaled_mm not available. Falling back to int8_gemm with fp16 out."
        )
        A_deq = A_q.to(torch.float16) * scale_a.item()
        B_deq = B_q.t().to(torch.float16) * scale_b.item()

        def _run_fb():
            return torch.matmul(A_deq, B_deq)

        latency_ms, C = _cuda_time_ms(_run_fb)
        return C, {
            "latency_ms": latency_ms,
            "dtype": "int8_fp16_fallback",
            "method": "dequant_fp16_matmul",
        }

    def _run():
        return torch._scaled_mm(
            A_q,
            B_q,
            scale_a=scale_a_dev,
            scale_b=scale_b_dev,
            out_dtype=torch.float16,
        )

    latency_ms, C = _cuda_time_ms(_run)
    return C, {
        "latency_ms": latency_ms,
        "dtype": "int8_compute_fp16_accum",
        "method": "torch._scaled_mm_fp16_out",
        "scale_a": scale_a.item(),
        "scale_b": scale_b.item(),
    }



# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _self_test() -> None:
    """Minimal smoke test -- prints pass/fail for each kernel."""
    print(f"CUDA available: {CUDA_AVAILABLE}")
    print(f"FP8 supported: {FP8_SUPPORTED}")
    print(f"torch._scaled_mm available: {SCALED_MM_AVAILABLE}")

    dev = DEVICE
    M, K, N = 128, 256, 128
    A = torch.randn(M, K, device=dev)
    B = torch.randn(K, N, device=dev)
    Q = torch.randn(2, 64, 64, device=dev)
    K_ = torch.randn(2, 64, 64, device=dev)
    V = torch.randn(2, 64, 64, device=dev)

    tests = [
        ("fp16_gemm",            lambda: fp16_gemm(A, B)),
        ("fp16_attention",       lambda: fp16_attention(Q, K_, V)),
        ("int8_gemm",            lambda: int8_gemm(A, B)),
        ("int8_attention",       lambda: int8_attention(Q, K_, V)),
        ("fp8_gemm",             lambda: fp8_gemm(A, B)),
        ("fp8_attention",        lambda: fp8_attention(Q, K_, V)),
        ("mixed_precision_gemm", lambda: mixed_precision_gemm(A, B)),
    ]

    for name, fn in tests:
        try:
            out, meta = fn()
            print(f"  [PASS] {name:30s} latency={meta.get('latency_ms', '?'):.3f}ms  shape={tuple(out.shape)}")
        except Exception as e:
            print(f"  [FAIL] {name:30s} {e}")


if __name__ == "__main__":
    _self_test()
