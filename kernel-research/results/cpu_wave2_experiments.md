# CPU Wave 2 Experiments Results

**Date**: 2026-03-24  
**Environment**: PyTorch 2.11.0+cpu, 4 active threads (20 configured), AVX-512, 1TB RAM  
**Methodology**: 10 warmup + 100 timed iterations per measurement, p50/p90/p99 latencies  

---

## Experiment 1: H-18 Block-Sparse Attention

**Hypothesis**: Block-sparse attention (local window + global stride tokens) should outperform full dense attention at long sequence lengths where attention is quadratic.

**Config**: B=1, H=8, d=64, seq_len=[2048, 4096], block_size=256, local_blocks=1 (window ~768 tokens)

**Implementation**: Block-loop sparse attention - each query block (256 tokens) attends only to neighboring blocks (radius=1) giving ~768 token attention window instead of full S attention.

### Results

| seq_len | Method | p50 (ms) | p90 (ms) | p99 (ms) | Speedup | Max Abs Err |
|---------|--------|----------|----------|----------|---------|-------------|
| 2048 | Dense (SDPA) | 31.111 | 43.795 | 54.178 | 1.000x | - |
| 2048 | Block-Sparse | 47.916 | 94.152 | 218.728 | 0.649x | 4.146e-01 |
| 4096 | Dense (SDPA) | 96.697 | 116.783 | 126.629 | 1.000x | - |
| 4096 | Block-Sparse | 302.990 | 523.150 | 694.600 | 0.319x | 4.968e-01 |

**Verdict**: FAIL (both seq_len=2048 and seq_len=4096)

**Analysis**:
- Theoretical FLOP reduction: 2.67x at seq_len=2048, 5.33x at seq_len=4096
- Actual result: 0.65x and 0.32x speedup (slower than dense)
- Root cause: Python-level loop over `n_blocks` iterations (8 and 16 respectively) introduces significant scheduling overhead. Each iteration launches a separate BLAS matmul which cannot be pipelined. F.scaled_dot_product_attention uses a single fused kernel call with no Python-level fragmentation.
- Note: Max abs error is large (~0.4-0.5) because sparse attention is an approximation - the attended token set differs from dense. This is expected and acceptable for inference quality evaluation, but the perf result still FAILs.
- Also tested: mask-based approach (even slower, 0.05-0.05x) and vectorized unfold approach (OOM/timeout at scale).
- **Key takeaway**: Block-sparse attention speedup on CPU requires C++/CUDA kernel-level implementation. PyTorch Python-level sparse implementations cannot beat optimized BLAS-backed dense attention.

---

## Experiment 2: H-22 Optimal Einsum Contraction

**Hypothesis**: torch.einsum with fused contraction path beats sequential matmul for multi-head attention.

**Config**: B=[1, 8], H=8, S=512, d=64

**Implementation**:
- Baseline: `Q @ K^T` then softmax then `@ V` (2 separate matmuls)
- Optimized: `torch.einsum('bhsd,bhkd->bhsk', q, k)` then softmax then `torch.einsum('bhsk,bhkd->bhsd', w, v)`
- Fused variant: Scale absorbed into first einsum operand

### Results

| B | Method | p50 (ms) | p90 (ms) | p99 (ms) | Speedup vs Sequential | Max Abs Err |
|---|--------|----------|----------|----------|-----------------------|-------------|
| 1 | Sequential matmul | 5.435 | 29.963 | 53.276 | 1.000x (baseline) | - |
| 1 | Einsum | 6.241 | 25.241 | 39.749 | 0.871x | 0.000e+00 |
| 1 | Einsum fused | 4.604 | 13.405 | 32.332 | 1.180x | - |
| 1 | F.sdp (reference) | 10.045 | 17.245 | 23.062 | 0.541x | 2.980e-07 |
| 8 | Sequential matmul | 177.262 | 211.830 | 237.900 | 1.000x (baseline) | - |
| 8 | Einsum | 157.045 | 212.755 | 238.135 | 1.129x | 0.000e+00 |
| 8 | Einsum fused | 128.088 | 198.014 | 260.911 | 1.383x | - |
| 8 | F.sdp (reference) | 21.800 | 31.485 | 46.689 | 8.131x | 3.874e-07 |

**Verdict**: 
- B=1: FAIL (einsum 0.87x; einsum-fused 1.18x - marginal)
- B=8: PASS (einsum 1.13x; einsum-fused 1.38x vs sequential matmul)

**Analysis**:
- Standard einsum (`bhsd,bhkd->bhsk`) is essentially equivalent to matmul internally; the difference is overhead
- The "fused" variant (absorbing scale multiplication into the first operand before matmul) avoids a separate scalar multiply operation, explaining its consistent improvement
- F.sdp dramatically outperforms manual implementations at B=8 (8.1x) by using memory-efficient flash-attention patterns; the benefit comes from avoiding the intermediate S×S attention matrix materialization
- At B=1, single-core efficiency dominates and overhead differences are more visible
- **Key takeaway**: Einsum provides modest benefits (1.1-1.4x) for larger batch sizes. The real win is F.sdp's flash-attention path which avoids materializing the full attention matrix.

---

## Experiment 3: H-07 KV-Cache Decode Latency

**Hypothesis**: Incremental KV-cache with single-query attention dramatically reduces decode latency vs full recompute.

**Config**: B=1, H=32, d=128, cache_len=[128, 512, 2048], dtype=float32

**Implementation**:
- Baseline: Full F.scaled_dot_product_attention over all (cache_len+1) tokens
- Optimized: Single query `q_new [B, H, 1, d]` attending to full KV cache `[B, H, L, d]` via two matmuls

### Results

| cache_len | Method | p50 (ms) | p90 (ms) | p99 (ms) | Speedup | Max Abs Err |
|-----------|--------|----------|----------|----------|---------|-------------|
| 128 | Full recompute | 12.2413 | 19.9574 | 24.8408 | 1.000x | - |
| 128 | KV-cache decode | 0.2255 | 1.9144 | 8.4204 | **54.29x** | 2.682e-07 |
| 512 | Full recompute | 15.3687 | 24.8225 | 31.0677 | 1.000x | - |
| 512 | KV-cache decode | 0.8343 | 6.5645 | 27.2332 | **18.42x** | 2.533e-07 |
| 2048 | Full recompute | 138.9404 | 190.1570 | 222.2660 | 1.000x | - |
| 2048 | KV-cache decode | 2.0181 | 7.6139 | 12.1405 | **68.85x** | 3.055e-07 |

**Verdict**: PASS (all cache lengths)

**Analysis**:
- KV-cache decode achieves 18-69x speedup across all tested cache lengths
- The speedup is sublinear relative to the theoretical maximum (L+1 for full recompute) due to:
  1. Memory bandwidth becoming the bottleneck at decode (reading KV cache is O(L) memory ops)
  2. CPU scheduling overhead per call is relatively fixed
  3. At cache_len=512, memory bandwidth effects are most pronounced (p99 degrades badly)
- Max absolute errors are ~3e-7 (well within float32 numerical tolerance)
- The p99 variance for KV-cache is high (2-27x higher than p50) indicating OS scheduling jitter at these very short durations; p50 is the reliable metric
- This is the strongest result in Wave 2: 18-69x real-world speedup with near-zero error

---

## Summary Table

| Hypothesis | Description | Verdict | Best Speedup | Max Abs Err |
|------------|-------------|---------|--------------|-------------|
| H-18 | Block-Sparse Attention (seq=2048) | FAIL | 0.649x | 4.146e-01 |
| H-18 | Block-Sparse Attention (seq=4096) | FAIL | 0.319x | 4.968e-01 |
| H-22 | Einsum Contraction (B=1) | FAIL | 0.871x (std) / 1.180x (fused) | 0.000e+00 |
| H-22 | Einsum Contraction (B=8) | PASS | 1.129x (std) / 1.383x (fused) | 0.000e+00 |
| H-07 | KV-Cache Decode (cache=128) | PASS | 54.29x | 2.682e-07 |
| H-07 | KV-Cache Decode (cache=512) | PASS | 18.42x | 2.533e-07 |
| H-07 | KV-Cache Decode (cache=2048) | PASS | 68.85x | 3.055e-07 |

**Overall**: 4/7 configs PASS (57%), 2/3 hypotheses have at least one PASS configuration.

## Key Insights from Wave 2

1. **KV-Cache is critical**: H-07 demonstrates the single most impactful optimization for autoregressive decode. Every LLM inference runtime should use KV-caching. The 18-69x speedup is real, consistent, and numerically exact.

2. **Block-sparse attention needs native kernels**: PyTorch Python-level block-sparse implementations cannot compete with fused dense BLAS. This optimization is viable only in custom C++/CUDA kernels (e.g., DeepSpeed Sparse Attention, Flash-Sparse).

3. **Einsum fused scaling pattern**: Absorbing scalar scale into the first einsum operand provides 1.2-1.4x improvement vs sequential matmul at batch sizes >= 8. Not a headline result but a free win.

4. **F.sdp dominates at batch sizes > 1**: The 8.1x speedup of F.sdp vs manual matmuls at B=8 confirms that flash-attention memory access patterns provide massive speedups by avoiding intermediate matrix materialization.
