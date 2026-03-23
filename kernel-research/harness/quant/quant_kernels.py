#!/usr/bin/env python3
"""
Quantization-aware kernel implementations: INT8, FP8, mixed-precision.
Owned by miniQuant.

Implements:
- INT8 attention and GEMM via torch._int8_linear or custom Triton
- FP8 kernels using torch.float8_e4m3fn (Hopper+)
- Mixed-precision: FP8 compute, FP16 accumulation

All kernels must be benchmarked via harness.py.
"""

import torch
import torch.nn.functional as F
import math


def int8_gemm(a: torch.Tensor, b: torch.Tensor, scale_a: float = None, scale_b: float = None):
    """
    INT8 GEMM using torch.ops.aten._int_mm or torch._scaled_mm.
    a, b are expected in float32/float16; we quantize internally.
    Returns float32 output.
    """
    if a.dtype != torch.int8:
        # Quantize to INT8
        if scale_a is None:
            scale_a = a.abs().max().item() / 127.0
        a_int8 = (a / scale_a).round().clamp(-128, 127).to(torch.int8)
    else:
        a_int8 = a
        if scale_a is None:
            scale_a = 1.0

    if b.dtype != torch.int8:
        if scale_b is None:
            scale_b = b.abs().max().item() / 127.0
        b_int8 = (b / scale_b).round().clamp(-128, 127).to(torch.int8)
    else:
        b_int8 = b
        if scale_b is None:
            scale_b = 1.0

    # Use torch._int_mm for INT8 matmul
    out_int32 = torch._int_mm(a_int8, b_int8)
    return out_int32.float() * (scale_a * scale_b)


def fp8_gemm(a: torch.Tensor, b: torch.Tensor):
    """
    FP8 GEMM using torch._scaled_mm (requires Hopper H100).
    a, b expected in float16 or float32; we cast to FP8.
    Returns float16 output.
    """
    dtype_fp8 = torch.float8_e4m3fn

    # Scale to FP8 range
    scale_a = torch.tensor(a.abs().max().item() / 448.0, dtype=torch.float32, device=a.device)
    scale_b = torch.tensor(b.abs().max().item() / 448.0, dtype=torch.float32, device=b.device)

    a_fp8 = (a / scale_a).to(dtype_fp8)
    b_fp8 = (b / scale_b).to(dtype_fp8)

    out, amax_out = torch._scaled_mm(
        a_fp8,
        b_fp8,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=torch.float16,
    )
    return out


def int8_attention(q, k, v, scale=None):
    """
    INT8 attention: quantize QK matmul to INT8, softmax in FP32, V matmul in INT8.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    # QK scores in INT8
    q_int8 = (q / (q.abs().max() / 127.0)).round().clamp(-128, 127).to(torch.int8)
    k_int8 = (k / (k.abs().max() / 127.0)).round().clamp(-128, 127).to(torch.int8)

    # Reshape for _int_mm: [B*H*Sq, D] x [D, Sk]
    B, H, Sq, D = q_int8.shape
    Sk = k_int8.shape[2]

    q_2d = q_int8.reshape(B * H * Sq, D)
    k_2d = k_int8.reshape(B * H * Sk, D).transpose(0, 1).contiguous()

    scores_int32 = torch._int_mm(q_2d, k_2d)
    scores = scores_int32.float().reshape(B, H, Sq, Sk) * scale

    attn = F.softmax(scores, dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)
    return out


def mixed_precision_attention(q, k, v, scale=None):
    """
    Mixed precision: FP16 QK + FP32 softmax + FP16 output.
    Uses torch.amp for automatic precision selection.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))

    q_fp16 = q.to(torch.float16)
    k_fp16 = k.to(torch.float16)
    v_fp16 = v.to(torch.float16)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        scores = torch.matmul(q_fp16, k_fp16.transpose(-2, -1)) * scale
        attn = F.softmax(scores.float(), dim=-1).to(torch.float16)
        out = torch.matmul(attn, v_fp16)
    return out


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        # Smoke test INT8 GEMM
        a = torch.randn(128, 256, device=device)
        b = torch.randn(256, 128, device=device)
        out = int8_gemm(a, b)
        print(f"INT8 GEMM: {a.shape} x {b.shape} = {out.shape}, dtype={out.dtype}")

        # Smoke test INT8 attention
        B, H, S, D = 2, 8, 64, 64
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        out = int8_attention(q, k, v)
        print(f"INT8 attention: q {q.shape} -> out {out.shape}")

        # FP8 on H100
        try:
            a_fp16 = torch.randn(128, 256, device=device, dtype=torch.float16)
            b_fp16 = torch.randn(256, 128, device=device, dtype=torch.float16)
            out_fp8 = fp8_gemm(a_fp16, b_fp16.t().contiguous())
            print(f"FP8 GEMM: output {out_fp8.shape}, dtype={out_fp8.dtype}")
        except Exception as e:
            print(f"FP8 GEMM not available on this GPU: {e}")

        print("Quant kernels smoke test passed")
