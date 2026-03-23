# Kernel Optimization Synthesis

Last updated: 2026-03-23 22:08 UTC

## Summary

- Total experiments: 0 (Lambda SSH blocked, CPU benchmarks in progress)
- Passed criteria: 0 (0%)
- Certified results: 0
- Active tracks: 4 (ooo, coolstufs, miniQuant, serious-inference-engineer)

## Success Criteria

- Speedup >= 1.10x (10% wall-clock improvement)
- Variance ratio (p99/p50) < 1.05
- Max absolute error < 1e-4 (FP16)
- Tested on batch sizes [1, 8, 32]

## Sprint Status

Sprint launched at 2026-03-23 21:33 UTC. All agents briefed.

### Current Blockers
1. **Lambda SSH**: All branches lack the provisioned private key for ubuntu@192.222.55.210.
   Workaround: CPU-only benchmarks are running on sandbox hardware. GPU benchmarks require
   Lambda access. Main coordinator review requested to resolve SSH key issue.

### Progress by Track

#### ooo (Hypothesis Track)
Status: ACTIVE - 20 hypotheses generated
- Delivered: 20 ranked hypotheses in branch workspace
- Top candidates: FP8 attention, Fused RMSNorm+RoPE+QKV, W4A8 SplitK GEMM, CUDA Graphs, FA3 FP8 MLA
- Shared hypotheses now pushed to GitHub: kernel-research/hypotheses/hypotheses.md
- CPU-benchmarkable subset: h006 (GEMM), h007 (LN+Linear), h012 (GELU+Linear), h014 (softmax), h015 (embeddings)

#### coolstufs (Benchmark Harness Track)
Status: ACTIVE - CPU-only benchmarks running
- Running: H01 cache-blocked GEMM, H02 fused softmax+cast, H03 chunked attention
- Results: writing to local /workspace/synthesis.md and results files
- Needs: CPU results pushed to kernel-research/results/results.jsonl on GitHub
- GPU benchmarks: blocked (no Lambda SSH)

#### miniQuant (Quantization Track)
Status: ACTIVE - worker spinning up
- Plan: INT8/FP8 GEMM+attention via Triton and torch._scaled_mm
- Needs: Lambda GPU for meaningful quant benchmarks
- Can proceed on CPU for INT8 GEMM correctness validation

#### serious-inference-engineer (Eval Harness + Coordination)
Status: ACTIVE - two branches running
- branch-000001-01 (this): coordination, GitHub scaffolding
- branch-000003-01: eval harness scripts pushed to GitHub
- Pushed: eval_harness.py, hypothesis_validator.py, validate_results.py, ablation_runner.py
- Standing by to certify GPU results when Lambda is unblocked

## GitHub Repository
https://github.com/prsabahrami/voice-ai-research/tree/main/kernel-research

Files committed (20+):
- README.md, bootstrap.sh
- harness/harness.py, baseline_kernels.py, run_benchmark.py
- harness/quant/quant_kernels.py, numerics_validator.py
- eval/eval_harness.py, hypothesis_validator.py, validate_results.py, ablation_runner.py
- synthesis/synthesis_monitor.py, SYNTHESIS.md
- hypotheses/hypotheses.md (20 ranked hypotheses), cpu_kernels.py
- results/results.jsonl, results/certified.jsonl

## Hypotheses Awaiting GPU Testing (Priority Order)

| Rank | ID | Type | Method | Expected Speedup |
|---|---|---|---|---|
| 1 | h001 | attention | FP8 end-to-end (SM90) | 1.3-1.5x |
| 2 | h002 | fused-ops | Fused RMSNorm+RoPE+QKV | 1.2-1.4x |
| 3 | h003 | GEMM | W4A8 SplitK | 1.5-2.0x |
| 4 | h004 | other | CUDA Graphs decode | 1.15-1.30x |
| 5 | h005 | attention | FA3 FP8 MLA | 1.3-2.0x |

## Hypotheses in CPU Benchmarking

| ID | Type | Method | Status |
|---|---|---|---|
| h006 | GEMM | Cache-blocked GEMM | Running (coolstufs) |
| h007 | fused-ops | Fused LN+Linear | Queued |
| h012 | fused-ops | Fused GELU+Linear | Running (coolstufs) |
| h014 | fused-ops | Softmax temperature | Running (coolstufs) |
| h015 | other | Vectorized embeddings | Queued |

## Next Steps
1. Resolve Lambda SSH key issue (main coordinator review requested)
2. Once Lambda accessible: run bootstrap.sh, start GPU benchmarks
3. ooo to push hypotheses.md to GitHub
4. miniQuant to start INT8 correctness validation on CPU
5. coolstufs to push CPU results to GitHub results.jsonl

## Certified Results
No certified results yet. GPU benchmarks needed for H100-specific certification.
