# coolstufs CPU Wave 1 Results Summary

## Environment

- Platform: CPU-only sandbox (20-core x86_64, 448GB RAM, no GPU)
- PyTorch: 2.11.0+cpu
- Torch SDPA: FlashAttention-2 CPU path

## Results Table

| Hypothesis | Method | Speedup | Notes |
|-----------|--------|---------|-------|
| H01 | Cache-Blocked GEMM Tiling | 0.24x | REFRAMED: Manual Python tiling cannot beat numpy/OpenBLAS. OpenBLAS already tiles optimally. |
| H02 | Fused Softmax+Cast (1D) | 1.3x | WEAK: N=4096: separate 0.56ms vs fused 0.43ms. 60% theoretical BW savings but only 30% actual. |
| H02 Wave2 | Fused Softmax+Cast (2D matrix) | 1.08-1.13x | WEAK: N=512: separate 0.19ms vs fused 0.17ms. |
| H03 | Blocked Attention (numpy) | 3.62x | CONFIRMED: N=256: 1.9x, N=512: 3.62x (peak), N=1024: 2.8x. d_head=64, block_size=64. Memory: 11x reduction at N=2048. |
| H03 Wave2 | torch SDPA vs naive | 4.3-7.47x | STRONG WIN: N=256: 4.3x, N=512: 5.0x, N=1024: 6.1x, N=2048: 7.47x. Max abs err: <1e-5. Baseline: Q@K.T/sqrt(dk)->softmax->@V via torch.matmul+softmax |
| H04 | Memory Layout C vs F | 1.09x | REJECTED: No meaningful gain from Fortran order on this workload. |
| H05 | INT8/FP16 Quantized GEMM | 0.002x | REJECTED: N=256: 0.01x (100x slower), N=512: 0.01x, N=1024: 0.002x (540x slower). numpy has no hardware int8 BLAS path on CPU. GPU Tensor Cores required. |
| H06 | OMP_NUM_THREADS Sweep | 5.78x | CONFIRMED: T=1 baseline, T=4: 3.1x, T=8: 2.5x (ANOMALOUS), T=12: 4.8x, T=16: 5.78x (OPTIMAL), T=20: 5.2x |
| H07 | Vectorized Batched GEMM | 1.4-1.7x | CONFIRMED: B=1: 1.68x, B=4: 1.47x, B=8: 0.77x (regression), B=16: 1.42x, B=32: 1.39x. Eliminates Python loop overhead over heads/batches. |
| H08 | Fused LayerNorm+Linear | 1.15x | WEAK: Only for square projection (d_model=d_out=768, B=64). Mixed results elsewhere. |
| H15 | Batch Size Sweep (Inference) | 1.7x throughput | B*=8 optimal: 29,489 tok/s vs B=1: 17,407 tok/s. Config: 256d, 2L, S=128. Within 50ms SLA. |

## Key Findings

### TIER 1 - ZERO-EFFORT WINS
1. **torch SDPA**: 4.3-7.5x speedup over naive attention (F.scaled_dot_product_attention drop-in)
2. **OMP_NUM_THREADS=16**: 5.78x GEMM speedup (T=8 anomalous regression, T=20 worse than T=16)

### TIER 2 - LOW-EFFORT CODE CHANGES
3. **Vectorized batched GEMM**: 1.4-1.7x faster than Python loop over heads/batches
4. **Optimal batch size B*=8**: 1.7x throughput gain for inference within 50ms SLA

### REJECTED ON CPU
- INT8/FP16 quantization: 100-540x SLOWER without hardware support (Tensor Cores/AMX/VNNI)
- Memory layout optimization: OpenBLAS already handles internally
- LayerNorm+Linear fusion at Python level: minimal gain

## GPU Phase Implications

1. SDPA already beats naive attention massively -- establish as baseline, not target
2. INT8/FP8 quantization is the high-ceiling GPU-only opportunity (H100 Tensor Cores)
3. Thread scheduling matters a lot on CPU; CUDA occupancy is the GPU analog

## GPU Priority Queue (based on CPU evidence)

1. H1/h001: FlashAttention-3 FP8 MLA (1.3-2.0x expected)
2. H3/h003: W4A8 SplitK decode GEMM (2-2.7x expected)
3. H5/h001: FP8 end-to-end pipeline (1.5x, 2x KV capacity)
4. H4/h004: CUDA Graphs persistent decode (15-30% latency)

## SDPA Baseline Clarification

- Baseline: Q @ K.T / sqrt(dk) -> softmax -> @ V using torch.matmul + torch.softmax
- Optimized: F.scaled_dot_product_attention(Q, K, V) (FlashAttention-2 CPU path)
- Range: 4.3x at N=256, 7.47x at N=2048 (CPU, torch 2.11.0+cpu)
- Max absolute error vs naive: below 1e-5 at all sizes

On H100 with actual FlashAttention-2 CUDA kernels, expect 10-50x+ speedup
at longer sequences due to HBM bandwidth savings.

*Results from coolstufs CPU-only benchmarks, March 2026. GPU certification pending Lambda SSH access.*
