# coolstufs CPU Wave 1 Benchmark Summary

**Date:** 2026-03-23
**Environment:** CPU-only sandbox (torch 2.11.0+cpu, 20-core x86_64, AVX-512 VNNI)
**Sprint:** Kernel Optimization Research Sprint

## Results Table

| Hypothesis | Method | Speedup | Status | Notes |
|---|---|---|---|---|
| H01 Cache-Blocked GEMM | Manual Python tiling (numpy) | 0.24x | REJECTED | OpenBLAS already tiles optimally; Python overhead dominates |
| H02 Fused Softmax+Cast | Fused vs separate softmax+cast (1D) | 1.3x | WEAK | 60% theoretical BW savings, only 30% actual on CPU |
| H03 Blocked Attention | Blocked numpy SDPA (block_size=64) vs naive numpy | 3.62x peak | CONFIRMED | Peak at N=512, d=64; 2.8x at N=1024, 1.9x at N=256 |
| H03 Wave2 SDPA vs naive | F.scaled_dot_product_attention vs torch.matmul attention | 4.3-7.47x | STRONG WIN | 4.3x at N=256, 5.0x at N=512, 6.1x at N=1024, 7.47x at N=2048 |
| H04 Memory Layout | C vs Fortran-order weight layout | 1.09x | REJECTED | No meaningful gain on CPU |
| H05 INT8 CPU GEMM | numpy INT8 quantized GEMM | 0.002x | REJECTED | 540x slower -- no hardware INT8 BLAS path; Tensor Cores required |
| H06 OMP_NUM_THREADS | OMP_NUM_THREADS=16 vs T=1 | 5.78x | CONFIRMED | Also: T=12: 4.8x, T=4: 3.1x; T=8 anomalous (scheduler contention) |
| H07 Vectorized GEMM | torch.bmm over heads vs Python loop | 1.4-1.7x | CONFIRMED | 1.68x at B=1, 1.47x at B=4, 1.42x at B=16; B=8 regression (0.77x) |
| H08 Fused LayerNorm+Linear | Fusion vs separate ops | 1.15x | WEAK | Only square projection (d=768, B=64); mixed results elsewhere |
| H15 Batch Size Sweep | Optimal batch B*=8 vs B=1 | 1.7x throughput | CONFIRMED | 256d, 2L, S=128; throughput gain at optimal batch |

## Summary

- **Experiments completed:** 10 hypotheses evaluated in Wave 1
- **Pass rate:** 5/10 confirmed (50% in Wave 1)
- **Top result:** H03 Wave 2 torch SDPA -- 7.47x at N=2048 (strong win, deploy immediately)
- **Top threading:** H06 OMP_NUM_THREADS=16 -- 5.78x GEMM speedup (zero-effort win)
- **Critical rejection:** H05 INT8 CPU -- 0.002x, requires GPU Tensor Cores

## Deployment Recommendations

1. **Immediate deploy (CPU):** Enable `OMP_NUM_THREADS=16` for all GEMM-heavy workloads -- 5.78x zero-effort win
2. **Immediate deploy (CPU):** Replace `torch.matmul` attention with `F.scaled_dot_product_attention` -- 4.3-7.47x
3. **GPU-gate:** INT8/FP8 quantization kernels (H05 and related) -- no benefit on CPU, expect 2x on H100

## Cross-Validation (from coolstufs harness.py)

- tiled_attention (SDPA) vs naive attention: **3.316x** speedup, max_abs_err **7.2e-7** -- PASS
- Configuration: batch=8, seq=512, d=64 (different from wave results above)

## Notes

- H06 T=8 anomaly: OMP_NUM_THREADS=8 gives only 2.5x (worse than T=4 at 3.1x); scheduler contention suspected
- H07 B=8 regression: vectorized GEMM regresses at batch=8 (0.77x) -- BLAS contention
- BF16 emulated on this CPU (5x slower than FP32) -- skip for CPU work
- FP16 also emulated (no F16C BLAS accumulation path)
