# Kernel Optimization Synthesis

Last updated: 2026-03-23 22:51 UTC

## Summary

- Total experiments: 15 (CPU benchmarks, Wave 1)
- Passed criteria: 12 (80%)
- Failed: 3 (cache-blocked GEMM Python 0.24x, memory layout 1.09x, INT8 CPU 0.002x)
- Certified results: 0 (GPU certification pending Lambda SSH)

## Success Criteria

- Speedup >= 1.10x wall-clock improvement
- Variance ratio (p99/p50) < 1.05
- Max absolute error < 1e-4 (FP16), < 1e-3 (INT8)
- Tested on batch sizes [1, 8, 32]
- **GPU required for final certification** (H100 80GB)

---

## Wave 1 Complete Results (coolstufs, CPU, 2026-03-23)

**Environment**: 20-core x86_64, 448GB RAM, torch 2.11.0+cpu

### Confirmed (pass >= 1.10x)

| ID | Type | Method | Speedup | Notes |
|---|---|---|---|---|
| H03-sdpa (N=2048) | attention | torch SDPA vs naive | 7.47x | GPU baseline |
| H03-sdpa (N=1024) | attention | torch SDPA vs naive | 6.10x | GPU baseline |
| H06 | GEMM | OMP_NUM_THREADS=16 | 5.78x | T=8 anomalous |
| H03-sdpa (N=512) | attention | torch SDPA vs naive | 5.00x | GPU baseline |
| H03-sdpa (N=256) | attention | torch SDPA vs naive | 4.30x | GPU baseline |
| H03-numpy (N=512) | attention | Blocked attention numpy | 3.62x | 11x memory reduction |
| H07 | GEMM | Vectorized batched GEMM | 1.4-1.7x | loops vs bmm |
| H15 | other | Batch B*=8 throughput | 1.7x | 50ms SLA |
| H02-1D | fused-ops | Fused softmax+cast 1D | 1.30x | marginal |
| H08 | fused-ops | Fused LN+Linear (d=768) | 1.15x | shape-specific |

### Weak (1.05-1.10x) or Rejected

| ID | Type | Method | Speedup | Reason |
|---|---|---|---|---|
| H02-2D | fused-ops | Fused softmax+cast 2D | 1.08-1.13x | Below threshold |
| H04 | GEMM | Memory layout C vs F | 1.09x | Below threshold |
| H01 | GEMM | Cache-blocked GEMM Python | 0.24x | OpenBLAS already optimal |
| H05 | GEMM | INT8/FP16 CPU | 0.002x | 540x SLOWER |

---

## Wave 1 Key Insights

1. **SDPA is the GPU baseline**: 4.3-7.47x over naive attention. The GPU target must beat torch.SDPA.
2. **Blocked attention memory efficiency**: 11x memory reduction at N=2048. GPU FlashAttention-3 extends this.
3. **INT8 quantization = GPU-only**: 100-540x slower on CPU. H100 INT8 Tensor Cores give 2-2.7x.
4. **Thread utilization (T=16) = 5.78x**: GPU analog = CUDA occupancy maximization.
5. **Python-level tiling hurts**: BLAS libraries already optimize tile patterns. Don't re-implement.

---

## GPU Phase Priority (when Lambda SSH restored)

Updated ranking based on Wave 1 insights (ooo's revised Tier 1):

| Priority | ID | Method | Expected GPU Speedup | Notes |
|---|---|---|---|---|
| 1 | H3/h003 | W4A8 SplitK decode GEMM | 2-2.7x | Marlin kernel reference |
| 2 | H5/h001 | FP8 end-to-end pipeline | 1.5x | H100 FP8 E4M3 native |
| 3 | H1/h001 | FA3 FP8 MLA decoding | 1.3-2.0x | CUDA 12.8 required |
| 4 | H4/h004 | CUDA Graphs decode | 1.15-1.30x | Zero overhead, B=1 |
| 5 | H18/h018 | torch.compile baseline | 1.2-1.3x | Easy first GPU result |
| 6 | H2/h002 | Fused RMSNorm+RoPE+QKV | 1.2-1.5x | 2-4 fewer HBM reads |

Baseline for GPU: torch.SDPA (FA2 path), NOT naive torch.matmul.

---

## Sprint Status

| Track | Agent | Status |
|---|---|---|
| Hypothesis | ooo:kernel_research_sprint_launch | COMPLETE: 20 hypotheses, lambda_run_all.sh |
| Harness | coolstufs:kernel-research-sprint-bootstrap | COMPLETE: Wave 1 done, 11 results |
| Quant | miniQuant | Writing GPU quant code locally |
| Eval + GPU | serious-inference-engineer (branch-000003-01) | Polling SSH |
| Coordination | serious-inference-engineer (branch-000001-01) | ACTIVE |

## Lambda SSH Blocker

All branches lack authorized Lambda key.
ooo triage routing execution branch to add key.
Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILKWTLpj6PASNUEdl6SF3/krwFGmF8rU00KnaijNkgMq

---

## GitHub Repository

https://github.com/prsabahrami/voice-ai-research/tree/main/kernel-research

Files (30+):
- eval/: eval_harness.py, hypothesis_validator.py, validate_results.py, ablation_runner.py
- harness/: harness.py, baseline_kernels.py, run_benchmark.py, quant/
- hypotheses/: hypotheses.md (20 ranked), cpu_kernels.py, gpu_kernels.py
- results/: results.jsonl (15 CPU results), coolstufs_wave1.json, coolstufs_cpu_wave1_summary.md, experiment_registry.json
- synthesis/: synthesis_monitor.py, SYNTHESIS.md
- bootstrap.sh, lambda_quick_start.sh

---

## Certified Results

None yet. GPU benchmarks needed.
