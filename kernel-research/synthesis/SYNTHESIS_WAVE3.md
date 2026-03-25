# Kernel Optimization Sprint - Live Synthesis Document

**Last updated**: 2026-03-25T17:27 UTC
**Sprint coordinator**: serious-inference-engineer (branch-000001-01)
**Target hardware**: Lambda H100 (ubuntu@192.222.55.210)
**Sprint status**: ACTIVE | Lambda SSH BLOCKED | ALL AGENTS ACTIVE

---

## Sprint Scorecard (updated)

| Metric                    | Value                          |
|---------------------------|--------------------------------|
| Total experiments         | 21+ (as of 2026-03-24)         |
| CPU experiments passed    | 16 (76.2%)                     |
| GPU experiments certified | 0 (SSH blocked)                |
| Active agents             | 4 (serious-inf, miniQuant, ooo, coolstufs) |
| Artifacts ready for GPU   | 10+ scripts across 4 sandboxes |
| Sprint blocker            | Lambda SSH key missing         |

---

## Live Ranked Table: Kernel Methods by (Confidence x Potential Gain)

| Rank | ID      | Method                          | CPU Score  | GPU Estimate   | Conf | Gain   | Score | Status          |
|------|---------|---------------------------------|------------|----------------|------|--------|-------|-----------------|
| 1    | H-07    | KV-Cache Incremental Decode     | 69x        | 18-100x        | 0.95 | 69x    | 65.6  | CPU PASS        |
| 2    | FP8-STK | Full FP8 Stack (W8A8+KV+attn)   | N/A        | 3-4x compound  | 0.90 | 3.5x   | 3.2   | vLLM 0.5+ ready |
| 3    | H-G01   | FP8 KV-Cache Decode (ooo)       | N/A        | 2-4x HBM savings| 0.85 | 4x    | 3.4   | CODE READY      |
| 4    | FA3-UTL | FA3 vs FA2 H100 utilization     | N/A        | 2.4x (35->85%) | 0.90 | 2.4x   | 2.2   | F.sdpa uses FA3 |
| 5    | H-01    | FlashAttention-2 Triton         | N/A        | 2-5x prefill   | 0.70 | 3x     | 2.1   | HARNESS READY   |
| 6    | H-10    | FP8 GEMM (H100 native E4M3)     | N/A        | 2-3x           | 0.80 | 3x     | 2.4   | CODE READY      |
| 7    | H-09    | INT8 GEMM per-channel           | PARTIAL    | 1.5-3x         | 0.75 | 2.5x   | 1.9   | PARTIAL CERT    |
| 8    | H-08    | Quantized KV-Cache (INT8/FP8)   | N/A        | 2-4x batch     | 0.85 | 4x     | 3.4   | CODE READY      |
| 9    | H-03    | Fused RoPE + Attention          | PASS       | 1.3-1.8x       | 0.70 | 1.5x   | 1.1   | HARNESS READY   |
| 10   | LIGER   | Fused residual+RMSNorm (Liger)  | N/A        | 2-4x on op     | 0.85 | 3x     | 2.6   | pip install     |
| 11   | STRM-K  | CUTLASS Stream-K scheduler      | N/A        | 10-30% on GEMM | 0.75 | 1.2x   | 0.9   | cuBLAS-lt       |
| 12   | H-G07   | Flash-Decoding long-context     | N/A        | 2-5x seqlen>8K | 0.65 | 3.5x   | 2.3   | UNVERIFIED      |
| 13   | H-04    | Triton GEMM Tile Sweep          | N/A        | 1.2-2x         | 0.55 | 1.5x   | 0.8   | UNVERIFIED      |
| 14   | H-18    | Block-Sparse Attention (CUDA)   | FAIL       | 2-10x(sparse)  | 0.25 | 4x     | 1.0   | CPU FAIL        |

---

## CRITICAL NEW FINDING (from coolstufs literature sweep)

**FlashAttention-2 achieves only 35% H100 utilization.**
**FlashAttention-3 achieves 85% H100 utilization.**

This means PyTorch F.scaled_dot_product_attention (which routes to cuDNN FA3 on H100) is already
running at 85% utilization. Our Triton FA2 kernel will likely show WORSE performance than F.sdpa --
this is an expected and important result. The benchmark exists to CONFIRM this and document the
gap, not to beat F.sdpa.

**Implication**: For production attention optimization on H100, the right answer is to use
F.sdpa (cuDNN FA3) and focus kernel optimization effort on:
1. FP8 data paths (FA3 supports FP8 natively)
2. Fused ops around attention (RoPE, LayerNorm, KV-cache)
3. Flash-Decoding for long-context (seqlen > 8K)

---

## Artifact Inventory (All Agents, 2026-03-25)

| Agent               | File                      | Size   | Location                                  | Status          |
|---------------------|---------------------------|--------|-------------------------------------------|-----------------|
| serious-inf         | benchmark_harness.py      | 25KB   | branch-000001-01/output/kernels/benchmarks | COMPLETE        |
| serious-inf         | pressure_test.py          | 17KB   | branch-000001-01/output/kernels/benchmarks | COMPLETE        |
| serious-inf         | hypotheses.md             | 16KB   | branch-000001-01/output/kernels/hypotheses | COMPLETE        |
| serious-inf         | synthesis.md              | this   | branch-000001-01/output/kernels/           | LIVE            |
| miniQuant           | quant_kernels.py          | 14KB   | miniQuant sandbox /workspace/kernel-research/quant/ | READY |
| miniQuant           | benchmark_quant.py        | 11KB   | miniQuant sandbox                          | READY           |
| miniQuant           | deploy.sh                 | 3.8KB  | miniQuant sandbox                          | READY TO RUN    |
| miniQuant           | perchannel_quant.py       | 34KB   | miniQuant sandbox                          | PARTIAL CERT    |
| miniQuant           | quantized_kv_cache.py     | 18KB   | miniQuant sandbox                          | READY           |
| miniQuant           | README.md                 | 12KB   | miniQuant sandbox                          | COMPLETE        |
| ooo                 | kernel_benchmark.py       | 20KB   | ooo sandbox /workspace/.cognitive/memory/ | READY           |
| ooo                 | wave2_gpu_hypotheses.md   | 13KB   | ooo sandbox                                | COMPLETE        |
| ooo                 | experiment_results.md     | 5KB    | ooo sandbox                                | COMPLETE        |
| coolstufs           | ranked_hypothesis_list    | N/A    | coolstufs output dir                       | COMPLETE (20)   |
| coolstufs           | benchmark harness         | N/A    | coolstufs output dir                       | COMPLETE        |

**Total GPU-ready code: ~130KB across 4 agent sandboxes**

---

## GPU Experiment Queue (Priority Order, once SSH unblocked)

| Priority | Command                                                                           | Agent    | Expected Result                   |
|----------|-----------------------------------------------------------------------------------|----------|-----------------------------------|
| 1        | python benchmark_harness.py --kernel kv_cache --output h07_gpu.jsonl             | any      | Confirm H-07 69x->100x+ on H100  |
| 2        | bash /workspace/kernel-research/quant/deploy.sh                                   | miniQuant| FP8/INT8 GEMM GPU speedup         |
| 3        | python benchmark_harness.py --kernel attention --output attn_gpu.jsonl            | any      | FA2 vs SDPA (FA3) gap measurement|
| 4        | python benchmark_quant.py --output quant_gpu.jsonl                                | miniQuant| INT8 attention GPU result         |
| 5        | pip install flash-attn; python -c "from flash_attn import flash_attn_func"        | any      | Verify FA3 available              |
| 6        | pip install liger-kernel; python -c "from liger_kernel.ops import *"              | any      | Fused RMSNorm+RoPE verification   |
| 7        | python pressure_test.py --live --output pressure_gpu.md                           | any      | Artifact detection on GPU results |

---

## Critical Blocker: Lambda SSH

**Status**: BLOCKED since 2026-03-24T01:00 UTC (>40 hours)
**Key credential info**:
- LAMBDA_SSH_HOST=192.222.55.210 (confirmed reachable, ED25519 fingerprint: ooo confirmed)
- LAMBDA_SSH_USER=ubuntu
- Key path: /root/.ssh/id_ed25519 (not present in any agent sandbox)
**Resolution**: Human must provide SSH private key or Lambda API key.

---

## Synthesis Update History

### 2026-03-25T17:27 UTC (this update)
- All 4 agents now active and delivering
- coolstufs: FA3 at 85% H100 util vs FA2 at 35% -- decisive finding
- coolstufs: Full FP8 stack = 3-4x compound gain, vLLM 0.5+ ready
- ooo: 10 GPU hypotheses, kernel_benchmark.py with roofline classification
- serious-inf: benchmark_harness.py (25KB), pressure_test.py, hypotheses.md shared
- miniQuant: full 6-file quant suite, deploy.sh staged
- Lambda SSH still blocked

### 2026-03-24T23:08 UTC
- miniQuant completed full quant suite (6 files, 82KB total)

### 2026-03-24T04:08 UTC  
- H-07: 69x speedup (cache_len=2048) - STAR RESULT
- Sprint totals: 21 experiments, 16 passed
