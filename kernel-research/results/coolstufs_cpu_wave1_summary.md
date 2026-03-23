# coolstufs CPU Wave 1 Results - Complete Summary

Date: 2026-03-23
Environment: 20-core x86_64, 448GB RAM, torch 2.11.0+cpu, NO GPU

## All Hypothesis Results

### H01 - Cache-Blocked GEMM Tiling
- Status: REFRAMED (0.24x)
- Speedup: 0.24x (slower!)
- Notes: Manual Python tiling CANNOT beat numpy/OpenBLAS. OpenBLAS already tiles optimally internally. Python-level tiling adds overhead without benefit.

### H02 - Fused Softmax+Cast
- Status: WEAK (1.3x 1D, 1.08-1.13x 2D)
- Speedup: 1.3x for 1D (N=4096: separate 0.56ms vs fused 0.43ms)
- 2D: N=512: 1.13x; global-max approximate: 1.2-1.4x but hurts numerical stability
- Notes: 60% theoretical bandwidth savings but only 30% actual. Below 1.10x threshold for 2D.

### H03 - Blocked Attention (numpy) + SDPA
- Status: CONFIRMED - STRONG WIN
- Blocked numpy attention: N=256: 1.9x | N=512: 3.62x (peak) | N=1024: 2.8x
  - d_head=64, block_size=64. Memory: 11x reduction at N=2048 (256MB vs 23MB)
- torch SDPA vs naive: N=256: 4.3x | N=512: 5.0x | N=1024: 6.1x | N=2048: 7.47x
  - Baseline: Q @ K.T / sqrt(dk) -> softmax -> @ V using torch.matmul
  - Max abs error: <1e-5 at all sizes
- THIS IS THE BASELINE FOR GPU BENCHMARKS: Use torch.SDPA, not naive matmul.

### H04 - Memory Layout C vs F
- Status: REJECTED (1.09x)
- Notes: No meaningful gain from F-order on this workload. OpenBLAS handles layout internally.

### H05 - INT8/FP16 Quantized GEMM
- Status: REJECTED (0.002-0.01x = 100-540x SLOWER)
- N=256: 0.01x (100x slower) | N=512: 0.01x | N=1024: 0.002x (540x slower)
- Notes: numpy has no hardware INT8 BLAS path on CPU. GPU Tensor Cores required.
- GPU prediction: INT8 on H100 Tensor Cores = 2-2.7x speedup (Marlin kernel)

### H06 - OMP_NUM_THREADS Sweep
- Status: CONFIRMED (5.78x optimal)
- T=1: 1.0x | T=4: 3.1x | T=8: 2.5x (ANOMALOUS - context switch issue) | T=12: 4.8x | T=16: 5.78x (OPTIMAL) | T=20: 5.2x
- Notes: T=8 shows anomalous regression, likely NUMA boundary issue. T=16 is optimal on 20-core.

### H07 - Vectorized Batched GEMM
- Status: CONFIRMED (1.4-1.7x)
- B=1: 1.68x | B=4: 1.47x | B=8: 0.77x (regression) | B=16: 1.42x | B=32: 1.39x
- Notes: Eliminates Python loop overhead over heads/batches. B=8 regression is anomalous.

### H08 - Fused LayerNorm+Linear
- Status: MARGINAL PASS (1.15x for d_model=d_out=768, B=64)
- Notes: Mixed results elsewhere. Only passes for specific shape configuration.

### H15 - Batch Size Sweep (Inference Throughput)
- Status: COMPLETED
- B*=8 optimal: 29,489 tok/s | B=1: 17,407 tok/s | B=32: higher throughput but exceeds 50ms SLA
- Throughput gain: ~1.7x (B=8 vs B=1)
- Config: 256d, 2L, S=128

## Summary

| ID | Description | Speedup | Status |
|---|---|---|---|
| H01 | Cache-blocked GEMM tiling | 0.24x | REFRAMED |
| H02-1D | Fused softmax+cast 1D | 1.30x | WEAK (marginal) |
| H02-2D | Fused softmax+cast 2D | 1.08-1.13x | WEAK |
| H03-numpy | Blocked attention numpy | 1.9-3.62x | CONFIRMED |
| H03-sdpa | torch SDPA vs naive | 4.3-7.47x | STRONG WIN |
| H04 | Memory layout C vs F | 1.09x | REJECTED |
| H05 | INT8/FP16 CPU | 0.002-0.01x | REJECTED |
| H06 | OMP T=16 | 5.78x | CONFIRMED |
| H07 | Vectorized batched GEMM | 1.4-1.7x | CONFIRMED |
| H08 | Fused LayerNorm+Linear | 1.15x | MARGINAL |
| H15 | Batch size B*=8 | 1.7x throughput | CONFIRMED |

## Key GPU Insights

1. Use torch.SDPA as GPU attention baseline
2. INT8/FP8 quantization is the highest-ceiling GPU optimization
3. Fused operators show marginal CPU gains but substantial GPU gains (fewer HBM round-trips)
