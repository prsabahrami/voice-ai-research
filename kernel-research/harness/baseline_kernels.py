"""
baseline_kernels.py - Reference implementations for benchmark harness.

Provides naive/reference implementations for:
  - Attention (Q @ K^T / sqrt(d), softmax, @ V)
  - GEMM (torch.matmul)
  - Unfused softmax + cast

Both CPU and GPU paths are supported; device selection is automatic.
"""

import math
import torch
import torch.nn.functional as F


def _device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Standard scaled dot-product attention. Inputs: (B, H, S, D) or (B, S, D)."""
    scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def make_attention_inputs(batch_size: int, seq_len: int = 128, num_heads: int = 8,
                          head_dim: int = 64, dtype=torch.float32,
                          device: torch.device = None):
    """Create random Q, K, V tensors for attention benchmarks."""
    if device is None:
        device = _device()
    shape = (batch_size, num_heads, seq_len, head_dim)
    q = torch.randn(*shape, dtype=dtype, device=device)
    k = torch.randn(*shape, dtype=dtype, device=device)
    v = torch.randn(*shape, dtype=dtype, device=device)
    return q, k, v


def standard_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard matrix multiplication via torch.matmul (maps to cuBLAS on GPU)."""
    return torch.matmul(a, b)


def make_gemm_inputs(batch_size: int, M: int = 256, K: int = 512, N: int = 256,
                     dtype=torch.float32, device: torch.device = None):
    """Create random A, B tensors for GEMM benchmarks."""
    if device is None:
        device = _device()
    a = torch.randn(batch_size, M, K, dtype=dtype, device=device)
    b = torch.randn(batch_size, K, N, dtype=dtype, device=device)
    return a, b


def unfused_softmax_cast(x: torch.Tensor, target_dtype=torch.float16) -> torch.Tensor:
    """Baseline: softmax in fp32 then cast to target_dtype -- two separate operations."""
    out_fp32 = F.softmax(x, dim=-1)
    return out_fp32.to(target_dtype)


def make_fused_ops_inputs(batch_size: int, seq_len: int = 256,
                          dtype=torch.float32, device: torch.device = None):
    """Create random logits tensor for softmax+cast benchmarks."""
    if device is None:
        device = _device()
    return torch.randn(batch_size, seq_len, dtype=dtype, device=device)
