"""
optimized_kernels.py - Optimized kernel implementations for benchmark harness.

Provides:
  - tiled_attention: Memory-efficient tiled attention using PyTorch's
    scaled_dot_product_attention (Flash Attention-style fused kernel;
    avoids materializing the full S*S attention matrix).
  - blocked_gemm: Cache-blocked GEMM with configurable tile size along K.
  - fused_softmax_cast: Fused softmax + cast in a single torch pass.

Both CPU and GPU paths are supported.
"""

import math
import torch
import torch.nn.functional as F


def tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    chunk_size: int = 32) -> torch.Tensor:
    """
    Memory-efficient tiled attention using torch SDPA (Flash Attention-style).

    Uses torch.nn.functional.scaled_dot_product_attention which processes
    Q/K/V in tiles to avoid materializing the full S*S score matrix,
    reducing peak memory from O(S^2) to O(S*D).

    Shape: (B, H, S, D) -- same as naive_attention.
    """
    return F.scaled_dot_product_attention(q, k, v)


def tiled_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               chunk_size: int = 32) -> torch.Tensor:
    """
    Pure-Python reference for the tiled/chunked online-softmax algorithm.
    NOT used as the production optimized path due to Python-loop overhead.
    """
    scale = 1.0 / math.sqrt(q.size(-1))
    S = q.size(-2)
    device = q.device
    dtype = q.dtype
    batch_shape = q.shape[:-2]

    m_i = torch.full((*batch_shape,), float('-inf'), device=device, dtype=dtype)
    l_i = torch.zeros(*batch_shape, device=device, dtype=dtype)
    o_i = torch.zeros_like(q)

    for start in range(0, S, chunk_size):
        end = min(start + chunk_size, S)
        k_c = k[..., start:end, :]
        v_c = v[..., start:end, :]
        scores = torch.matmul(q, k_c.transpose(-2, -1)) * scale
        m_new = torch.maximum(m_i, scores.max(dim=-1).values)
        alpha = torch.exp(m_i - m_new)
        exp_s = torch.exp(scores - m_new.unsqueeze(-1))
        l_i = alpha * l_i + exp_s.sum(dim=-1)
        o_i = alpha.unsqueeze(-1) * o_i + torch.matmul(exp_s, v_c)
        m_i = m_new

    return o_i / l_i.unsqueeze(-1)


def blocked_gemm(a: torch.Tensor, b: torch.Tensor,
                 tile_size: int = 64) -> torch.Tensor:
    """
    Cache-blocked matrix multiplication.

    Tiles the K (inner) dimension into chunks of `tile_size` to improve
    cache reuse on CPU L2/L3. On GPU, each tile fits in shared memory.

    Input shapes: (..., M, K) and (..., K, N).
    """
    *batch_dims, M, K = a.shape
    *_, K2, N = b.shape
    assert K == K2, f"Inner dimension mismatch: {K} vs {K2}"

    device = a.device
    dtype = a.dtype
    out = torch.zeros(*batch_dims, M, N, device=device, dtype=dtype)

    for k_start in range(0, K, tile_size):
        k_end = min(k_start + tile_size, K)
        out += torch.matmul(a[..., :, k_start:k_end], b[..., k_start:k_end, :])

    return out


def fused_softmax_cast(x: torch.Tensor,
                       target_dtype=torch.float16) -> torch.Tensor:
    """
    Fused softmax + dtype cast.

    On GPU under torch.compile this fuses into one kernel, reducing HBM
    round-trips from 2 (baseline) to 1.
    """
    return F.softmax(x, dim=-1).to(target_dtype)
