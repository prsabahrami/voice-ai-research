#!/usr/bin/env python3
"""
Reference baseline implementations for benchmarking.
These are the "ground truth" to beat for each kernel type.
"""

import torch
import torch.nn.functional as F
import math


def baseline_attention(q, k, v, scale=None):
    """Standard scaled dot-product attention (FP16, cuBLAS path)."""
    if scale is None:
        scale = 1.0 / math.sqrt(q.size(-1))
    # q: [batch, heads, seq, head_dim]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def baseline_attention_pytorch(q, k, v, scale=None):
    """PyTorch 2.x scaled_dot_product_attention (uses FlashAttention if available)."""
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def baseline_gemm(a, b):
    """Standard matrix multiply via cuBLAS."""
    return torch.matmul(a, b)


def baseline_gemm_fp16(a, b):
    """FP16 GEMM via cuBLAS."""
    a = a.to(torch.float16)
    b = b.to(torch.float16)
    return torch.matmul(a, b).to(torch.float32)


def baseline_layer_norm(x, weight, bias, eps=1e-5):
    """Standard layer normalization."""
    return F.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)


def baseline_mlp_fused(x, w1, w2, w3=None):
    """Standard 2-layer MLP (no fusion)."""
    h = F.silu(x @ w1.T)
    if w3 is not None:
        h = h * (x @ w3.T)  # SwiGLU variant
    return h @ w2.T


def make_attention_args(batch_size, seq_len=512, n_heads=8, head_dim=64, device="cuda", dtype=torch.float16):
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype)
    return (q, k, v)


def make_gemm_args(batch_size, m=512, n=512, k=512, device="cuda", dtype=torch.float16):
    a = torch.randn(batch_size * m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    return (a, b)


def make_layer_norm_args(batch_size, hidden_size=4096, device="cuda", dtype=torch.float16):
    x = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    weight = torch.ones(hidden_size, device=device, dtype=dtype)
    bias = torch.zeros(hidden_size, device=device, dtype=dtype)
    return (x, weight, bias)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Quick smoke test
        q, k, v = make_attention_args(batch_size=8, device=device)
        out = baseline_attention(q, k, v)
        print(f"Attention baseline: input {q.shape}, output {out.shape}")

        a, b = make_gemm_args(batch_size=8, device=device)
        out = baseline_gemm(a, b)
        print(f"GEMM baseline: {a.shape} x {b.shape} = {out.shape}")

        x, w, bias = make_layer_norm_args(batch_size=8, device=device)
        out = baseline_layer_norm(x, w, bias)
        print(f"LayerNorm baseline: {x.shape} -> {out.shape}")

        print("All baselines OK")
    else:
        print("CUDA not available, baselines require GPU")
