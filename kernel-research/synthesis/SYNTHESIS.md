# Kernel Optimization Synthesis

Last updated: 2026-03-23 22:36 UTC

## Summary

- Total experiments: 6 (CPU benchmarks, Wave 1)
- Passed criteria: 4 (67%)
- Failed: 2 (memory layout 1.09x, INT8 on CPU 0.01x)
- Certified results: 0 (GPU certification pending Lambda SSH)

## Success Criteria

- Speedup >= 1.10x wall-clock improvement
- Variance ratio (p99/p50) < 1.05
- Max absolute error < 1e-4 (FP16)
- Tested on batch sizes [1, 8, 32]

---

## Wave 1: CPU Benchmarks (coolstufs, 2026-03-23)

**Environment**: 20-core x86_64, 448GB RAM, torch 2.11.0+cpu, NO GPU

### Passed

| Hypothesis | Type | Method | Speedup | Notes |
|---|---|---|---|---|
| h-cpu-gemm-threads | GEMM | OMP_NUM_THREADS=16 | 5.78x | T=8 anomalous; T=20 < T=16 |
| h-cpu-sdpa | attention | torch SDPA vs naive | 5.0x (4.3-7.5x) | Drop-in, use as GPU baseline |
| h-cpu-optimal-batch | fused-ops | B*=8 for 50ms SLA | 2.0x throughput | Batch size tuning |
| h-cpu-batched-gemm | GEMM | Vectorized batched GEMM | 1.55x (1.4-1.7x) | vs Python loop |

### Failed

| Hypothesis | Type | Method | Speedup | Reason |
|---|---|---|---|---|
| h-cpu-memory-layout | GEMM | C vs F order | 1.09x | Below 1.10x threshold; OpenBLAS handles internally |
| h-cpu-int8-rejected | GEMM | INT8 on CPU | 0.01x | 100-540x SLOWER without hardware |

---

## Wave 1 Key Insights

1. **SDPA is a trivial win** (4.3-7.5x over naive attention). Use torch.SDPA as the GPU baseline, NOT naive attention. The target is to beat SDPA.

2. **Quantization on CPU = anti-pattern**. INT8 needs Tensor Cores (H100 gives 2x-2.7x INT8 throughput). GPU-only optimization.

3. **Thread utilization (OMP T=16) = 5.78x**. GPU analog = CUDA occupancy. High-level: use all SMs efficiently.

4. **Batch size tuning matters**. B*=8 gives optimal throughput/latency for 50ms SLA. GPU analog = max concurrent SM utilization.

---

## GPU Phase Priority (when Lambda SSH restored)

Based on ooo's revised Tier 1 (incorporating Wave 1 insights):

| Priority | ooo ID | my ID | Method | Expected Speedup |
|---|---|---|---|---|
| 1 | H3 | h003 | W4A8 SplitK decode GEMM | 2-2.7x |
| 2 | H5 | h001 | FP8 end-to-end pipeline | 1.5x |
| 3 | H1 | h001 | FA3 FP8 MLA decoding | 1.3-2.0x |
| 4 | H4 | h004 | CUDA Graphs persistent decode | 1.15-1.30x |
| 5 | H2 | h002 | Fused RMSNorm+RoPE+QKV | 1.2-1.5x |
| 6 | H18 | h018-compile | torch.compile reduce-overhead | 1.2-1.3x |

Baseline for GPU: torch.SDPA (FA2 path), NOT naive attention.

---

## Hypothesis ID Cross-Reference

| ooo ID | my ID | coolstufs ID | Description |
|---|---|---|---|
| H1 | h001 | -- | FA3 FP8 MLA |
| H2 | h002 | -- | Fused RMSNorm+RoPE+QKV |
| H3 | h003 | -- | W4A8 SplitK GEMM |
| H4 | h004 | -- | CUDA Graphs |
| H5 | h001 | -- | FP8 E2E pipeline |
| H18 | h018-compile | -- | torch.compile baseline |
| -- | h006 | H01 | Cache-blocked GEMM |
| -- | h014 | H02 | Fused softmax+cast |
| -- | h010 | H03 | Chunked attention |
| -- | h007 | H04 | Fused LN+Linear |
| -- | h012 | H05 | Fused GELU+Linear |
| -- | h008 | H06 | Persistent decode attention |
| -- | h011 | H07 | Warp-spec GEMM small M |
| -- | h015 | H08 | Vectorized embeddings |

---

## Sprint Status

| Track | Agent | Status |
|---|---|---|
| Hypothesis | ooo:kernel_research_sprint_launch | DELIVERED: 20 hypotheses, lambda_run_all.sh ready |
| Harness | coolstufs:kernel-research-sprint-bootstrap | Wave 2 COMPLETE: 10 hypotheses, 432-line harness |
| Quant kernels | miniQuant | Acknowledged, spinning up |
| Eval + GPU | serious-inference-engineer (branch-000003-01) | Polling SSH, will run first GPU benchmarks |
| Coordination | serious-inference-engineer (branch-000001-01) | ACTIVE: synthesis, org coordination |

## Lambda SSH Blocker

ooo's triage confirmed the private key is in their sandbox. Execution branch routing in progress.

Public key to add to Lambda:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILKWTLpj6PASNUEdl6SF3/krwFGmF8rU00KnaijNkgMq serious-inference-engineer-eval-harness
```

SSH escalation email sent to zayaan@talkingcomputers.ai at 22:17 UTC.

---

## GitHub Repository

https://github.com/prsabahrami/voice-ai-research/tree/main/kernel-research

Files (25+):
- README.md, bootstrap.sh, lambda_quick_start.sh
- harness/ (harness.py, baseline_kernels.py, run_benchmark.py, quant/)
- eval/ (eval_harness.py, hypothesis_validator.py, validate_results.py, ablation_runner.py)
- synthesis/ (synthesis_monitor.py, SYNTHESIS.md)
- hypotheses/ (hypotheses.md, cpu_kernels.py, gpu_kernels.py)
- results/ (results.jsonl with 6 CPU results)

---

## Certified Results

None yet. GPU certification pending Lambda access.

When GPU results come in, certification criteria:
- speedup >= 1.10x
- variance ratio (p99/p50) < 1.05
- max_abs_err < 1e-4 (FP16), < 1e-3 (INT8)
- tested on H100, batch sizes [1, 8, 32]
