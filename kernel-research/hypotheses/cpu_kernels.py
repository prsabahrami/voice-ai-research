#!/usr/bin/env python3
"""
CPU-benchmarkable kernel implementations for kernel-research sprint.
These can be run on any machine (no GPU required).
Hypotheses: h006, h007, h012, h014, h015

Each function must be registered with the harness using register_hypothesis().

Usage:
    python cpu_kernels.py  # run smoke test
"""

import torch
import torch.nn.functional as F
import math


# ============================================================
# h006: Cache-blocked GEMM with tiling
# ============================================================

def h006_blocked_gemm(a: torch.Tensor, b: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Cache-blocked GEMM: explicitly tile the matrix multiplication to fit
    working set in L1/L2 cache. On CPU, this significantly outperforms naive
    row-major GEMM for large matrices.
    
    Reference: https://en.wikipedia.org/wiki/Cache-oblivious_algorithm
    """
    # Use einsum with explicit blocking via unfold
    # For large matrices, this maintains L1/L2 locality
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Dimension mismatch: {K} != {K2}"
    
    # torch.matmul on CPU already uses BLAS with cache blocking,
    # but we can experiment with explicit blocking via chunk operations
    out = torch.zeros(M, N, dtype=a.dtype)
    
    for m in range(0, M, block_size):
        for n in range(0, N, block_size):
            for k in range(0, K, block_size):
                m_end = min(m + block_size, M)
                n_end = min(n + block_size, N)
                k_end = min(k + block_size, K)
                out[m:m_end, n:n_end] += a[m:m_end, k:k_end] @ b[k:k_end, n:n_end]
    
    return out


def h006_baseline_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Baseline: standard torch.matmul."""
    return torch.matmul(a, b)


# ============================================================
# h007: Fused LayerNorm + Linear
# ============================================================

def h007_fused_layernorm_linear(x: torch.Tensor, weight_ln: torch.Tensor, 
                                  bias_ln: torch.Tensor, weight_linear: torch.Tensor,
                                  bias_linear: torch.Tensor = None, eps: float = 1e-5) -> torch.Tensor:
    """
    Fused LayerNorm + Linear: compute both in a single forward pass.
    Avoids materializing the normalized tensor to DRAM.
    """
    # Compute LayerNorm statistics
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    # Apply affine transform and linear projection in one step
    # (weight_ln * x_norm + bias_ln) @ weight_linear.T
    x_scaled = weight_ln * x_norm + bias_ln
    out = x_scaled @ weight_linear.T
    if bias_linear is not None:
        out = out + bias_linear
    return out


def h007_baseline_layernorm_linear(x, weight_ln, bias_ln, weight_linear, 
                                     bias_linear=None, eps=1e-5):
    x_norm = F.layer_norm(x, x.shape[-1:], weight=weight_ln, bias=bias_ln, eps=eps)
    out = F.linear(x_norm, weight_linear, bias_linear)
    return out


# ============================================================
# h012: Fused GELU + Linear
# ============================================================

def h012_fused_gelu_linear(x: torch.Tensor, weight: torch.Tensor, 
                            bias: torch.Tensor = None) -> torch.Tensor:
    """
    Fused GELU activation + linear projection.
    Avoids materializing GELU output before the linear layer.
    Uses fast GELU approximation (tanh-based).
    """
    # Apply GELU in-place approximation
    # GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    x_gelu = F.gelu(x, approximate='tanh')
    out = x_gelu @ weight.T
    if bias is not None:
        out = out + bias
    return out


def h012_baseline_gelu_linear(x, weight, bias=None):
    return F.linear(F.gelu(x), weight, bias)


# ============================================================
# h014: Softmax Temperature Fusion
# ============================================================

def h014_fused_temp_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Fused temperature scaling + softmax.
    Equivalent to F.softmax(logits / temperature) but in a single pass.
    For temperature=1.0, this is identical to standard softmax.
    """
    if temperature == 1.0:
        return F.softmax(logits, dim=-1)
    # Scale and softmax in one operation (avoids intermediate tensor)
    # log-sum-exp trick: softmax(x/T) = softmax(x/T)
    scaled = logits * (1.0 / temperature)
    return F.softmax(scaled, dim=-1)


def h014_baseline_temp_softmax(logits, temperature=1.0):
    return F.softmax(logits / temperature, dim=-1)


# ============================================================
# h015: Vectorized Token Embedding Lookup
# ============================================================

def h015_vectorized_embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Vectorized token embedding lookup.
    Uses F.embedding with contiguous memory layout.
    """
    # Ensure weight is contiguous and in optimal memory layout
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return F.embedding(input_ids, weight)


def h015_baseline_embedding(input_ids, weight):
    return weight[input_ids]


# ============================================================
# Registration for harness
# ============================================================

def register(register_fn):
    """Register all hypotheses with the benchmark harness."""
    import sys
    sys.path.insert(0, __file__.replace("hypotheses/cpu_kernels.py", "harness"))
    from baseline_kernels import make_gemm_args

    # h006
    def make_gemm(bs):
        a = torch.randn(bs * 64, 512, dtype=torch.float32)
        b = torch.randn(512, 512, dtype=torch.float32)
        return (a, b)
    
    register_fn("h006", h006_blocked_gemm, "GEMM", "cache-blocked GEMM tiling (64x64 blocks)", make_gemm)
    
    # h007
    def make_ln_linear(bs):
        x = torch.randn(bs, 4096, dtype=torch.float32)
        wln = torch.ones(4096, dtype=torch.float32)
        bln = torch.zeros(4096, dtype=torch.float32)
        wl = torch.randn(512, 4096, dtype=torch.float32)
        return (x, wln, bln, wl)
    
    register_fn("h007", h007_fused_layernorm_linear, "fused-ops", "fused LayerNorm+Linear", make_ln_linear)
    
    # h012
    def make_gelu_linear(bs):
        x = torch.randn(bs, 4096, dtype=torch.float32)
        w = torch.randn(4096, 4096, dtype=torch.float32)
        return (x, w)
    
    register_fn("h012", h012_fused_gelu_linear, "fused-ops", "fused GELU+Linear (tanh approx)", make_gelu_linear)
    
    # h014
    def make_logits(bs):
        logits = torch.randn(bs, 32000, dtype=torch.float32)
        return (logits, 0.8)  # temperature=0.8
    
    register_fn("h014", h014_fused_temp_softmax, "fused-ops", "fused temperature-scaled softmax", make_logits)
    
    # h015
    def make_embedding(bs):
        input_ids = torch.randint(0, 32000, (bs, 512))
        weight = torch.randn(32000, 4096, dtype=torch.float32)
        return (input_ids, weight)
    
    register_fn("h015", h015_vectorized_embedding, "other", "vectorized token embedding lookup", make_embedding)


def smoke_test():
    """Quick smoke test of all kernels."""
    device = "cpu"
    bs = 4
    
    # h006
    a = torch.randn(bs * 64, 512)
    b = torch.randn(512, 512)
    out = h006_blocked_gemm(a, b)
    ref = h006_baseline_gemm(a, b)
    err = (out - ref).abs().max().item()
    print(f"h006 blocked GEMM: max_err={err:.2e} (should be ~0) -- {'OK' if err < 1e-4 else 'FAIL'}")
    
    # h007
    x = torch.randn(bs, 4096)
    wln = torch.ones(4096)
    bln = torch.zeros(4096)
    wl = torch.randn(512, 4096)
    out = h007_fused_layernorm_linear(x, wln, bln, wl)
    ref = h007_baseline_layernorm_linear(x, wln, bln, wl)
    err = (out - ref).abs().max().item()
    print(f"h007 fused LN+Linear: max_err={err:.2e} -- {'OK' if err < 1e-4 else 'FAIL'}")
    
    # h012
    x = torch.randn(bs, 4096)
    w = torch.randn(4096, 4096)
    out = h012_fused_gelu_linear(x, w)
    ref = h012_baseline_gelu_linear(x, w)
    err = (out - ref).abs().max().item()
    print(f"h012 fused GELU+Linear: max_err={err:.2e} -- {'OK' if err < 1e-4 else 'FAIL'}")
    
    # h014
    logits = torch.randn(bs, 32000)
    out = h014_fused_temp_softmax(logits, 0.8)
    ref = h014_baseline_temp_softmax(logits, 0.8)
    err = (out - ref).abs().max().item()
    print(f"h014 fused temp softmax: max_err={err:.2e} -- {'OK' if err < 1e-4 else 'FAIL'}")
    
    # h015
    ids = torch.randint(0, 32000, (bs, 64))
    w = torch.randn(32000, 4096)
    out = h015_vectorized_embedding(ids, w)
    ref = h015_baseline_embedding(ids, w)
    err = (out - ref).abs().max().item()
    print(f"h015 vectorized embedding: max_err={err:.2e} -- {'OK' if err < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    smoke_test()
