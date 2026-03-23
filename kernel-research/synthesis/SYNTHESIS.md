# Kernel Optimization Synthesis

Last updated: 2026-03-23 22:18 UTC

## Summary

- Total experiments: 4 (CPU benchmarks, Wave 1)
- Passed criteria: 3 (75%)
- Failed: 1 (INT8 on CPU - expected)
- Certified results: 0 (GPU certification pending Lambda access)

## Success Criteria

- Speedup >= 1.10x wall-clock improvement
- Variance ratio (p99/p50) < 1.05
- Max absolute error < 1e-4 (FP16)
- Tested on batch sizes [1, 8, 32]

---

## Wave 1: CPU Benchmarks (coolstufs, 2026-03-23)

### Passed

| Hypothesis | Type | Method | Speedup | Notes |
|---|---|---|---|---|
| h-cpu-sdpa | attention | torch SDPA vs naive | 5.0x (4-7.5x range) | Drop-in replacement! Baseline for GPU phase |
| h-cpu-gemm-threads | GEMM | OMP_NUM_THREADS=16 | 5.78x | T=8 anomalous; T=20 < T=16 |
| h-cpu-batched-gemm | GEMM | Vectorized batched GEMM | 1.55x (1.4-1.7x) | vs Python loop |

### Rejected

| Hypothesis | Type | Method | Speedup | Reason |
|---|---|---|---|---|
| h-cpu-int8-rejected | GEMM | INT8/FP16 on CPU | 0.01x | 100-540x SLOWER without hardware support |

---

## Wave 1 Insights

1. **SDPA is a trivial win** (4-7.5x over naive attention) -- this means naive attention is NOT a fair baseline for GPU benchmarks. Must use PyTorch SDPA or FA2 as baseline.

2. **Quantization on CPU is anti-pattern** -- INT8/FP16 is catastrophically slow on CPU without dedicated hardware (Tensor Cores, AMX, VNNI). On H100, INT8 Tensor Cores give 2x throughput vs FP16. This is a GPU-only optimization.

3. **Thread utilization matters** -- OMP_NUM_THREADS tuning = 5.78x on CPU. GPU analog = CUDA occupancy tuning.

---

## GPU Phase Priority (when Lambda SSH restored)

Based on ooo's Tier 1 hypotheses + Wave 1 CPU insights:

| Priority | Hypothesis | Method | Expected Speedup |
|---|---|---|---|
| 1 | H1/h001 | FA3 FP8 MLA decoding | 1.3-2.0x |
| 2 | H3/h003 | W4A8 SplitK decode GEMM | 2-2.7x |
| 3 | H5 | FP8 end-to-end pipeline | 1.5x |
| 4 | H4/h004 | CUDA Graphs persistent decode | 1.15-1.30x |
| 5 | H2/h002 | Fused RMSNorm+RoPE+QKV | 1.2-1.5x |

NOTE: Baselines for GPU must be FA2/torch.SDPA (not naive attention).

---

## Current Sprint State

### Tracks

#### ooo (Hypothesis)
Status: DELIVERED -- 20 hypotheses in 3 tiers, Welch t-test harness design, roofline classifier

#### coolstufs (Benchmark Harness)
Status: WAVE 2 COMPLETE -- 15 CPU hypotheses, harness.py + synthesis.md in local workspace
Needs: results pushed to GitHub kernel-research/results/results.jsonl

#### miniQuant (Quantization)
Status: ACKNOWLEDGED -- worker spinning up, INT8/FP8 implementation planned

#### serious-inference-engineer (Eval + Coordination)
Status: ACTIVE -- eval_harness.py, hypothesis_validator.py, ablation_runner.py in GitHub
Lambda access: blocked (sent escalation to zayaan@talkingcomputers.ai)

---

## Lambda SSH Blocker

All branches lack the provisioned private key for ubuntu@192.222.55.210.

Public key to add:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAILKWTLpj6PASNUEdl6SF3/krwFGmF8rU00KnaijNkgMq serious-inference-engineer-eval-harness
```

Resolution pending: zayaan@talkingcomputers.ai email sent at 22:17 UTC.

---

## GitHub Repository

https://github.com/prsabahrami/voice-ai-research/tree/main/kernel-research

Committed files (22+):
- README.md, bootstrap.sh
- harness/harness.py, baseline_kernels.py, run_benchmark.py
- harness/quant/quant_kernels.py, numerics_validator.py
- eval/eval_harness.py, hypothesis_validator.py, validate_results.py, ablation_runner.py
- synthesis/synthesis_monitor.py, SYNTHESIS.md (this file)
- hypotheses/hypotheses.md (20 ranked), cpu_kernels.py
- results/results.jsonl (4 CPU results), certified.jsonl (empty)

---

## Certified Results

None yet. GPU benchmarks needed for certification.
Criteria: speedup >= 1.10x, variance < 1.05, max_abs_err < 1e-4, tested on [1,8,32].
