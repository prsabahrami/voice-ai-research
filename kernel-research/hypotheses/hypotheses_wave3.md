# Kernel Optimization Hypotheses for ML Inference on H100

Generated: 2026-03-25
Status: Active hypothesis sweep for Lambda H100 benchmark validation

---

## Summary

This document ranks kernel optimization hypotheses for ML inference (attention, GEMM, decode)
on the Lambda H100 instance (ubuntu@192.222.55.210). Each hypothesis includes:
- Expected gain estimate (vs baseline fp16 PyTorch)
- Implementation complexity (Low/Medium/High/Very High)
- Known failure modes and benchmark artifact risks
- Current validation status

---

## Ranked Hypothesis Table

| Rank | ID   | Hypothesis                                | Expected Gain      | Complexity | Status     | Priority GPU |
|------|------|-------------------------------------------|--------------------|------------|------------|--------------|
| 1    | H-07 | KV-Cache Incremental Decode               | 18-69x (CPU proven)| Low        | CPU PASS   | HIGH         |
| 2    | H-01 | FlashAttention-2 (Triton tiled, causal)   | 2-5x latency       | Medium     | Unverified | HIGH         |
| 3    | H-02 | FlashAttention-3 (warp specialization)    | 1.5-2x vs FA2      | Very High  | Unverified | MEDIUM       |
| 4    | H-08 | Quantized KV-Cache (INT8/FP8 + incr)      | 2-4x vs H-07 alone | Medium     | Code ready | HIGH         |
| 5    | H-03 | Fused RoPE + Attention (single kernel)    | 1.3-1.8x           | Medium     | CPU PASS   | HIGH         |
| 6    | H-04 | Triton GEMM Tile Sweep (sm_90 tuned)      | 1.2-2x vs cuBLAS   | Medium     | Unverified | MEDIUM       |
| 7    | H-05 | Fused Softmax + LayerNorm (single pass)   | 1.2-1.5x           | Low        | Unverified | MEDIUM       |
| 8    | H-06 | Persistent Decode Kernel (streaming)      | 1.3-2x for decode  | High       | Unverified | HIGH         |
| 9    | H-09 | Warp-Level INT8 GEMM (per-channel scale)  | 1.5-3x vs fp16     | Medium     | CPU PARTIAL| HIGH         |
| 10   | H-10 | FP8 E4M3 GEMM (H100 native)               | 2-3x vs fp16       | Medium     | Code ready | HIGH         |
| 11   | H-11 | Paged KV-Cache Access Patterns            | 1.2-2x (decode)    | High       | Unverified | MEDIUM       |
| 12   | H-12 | Fused Attention + Residual + Norm         | 1.2-1.4x           | Medium     | Unverified | LOW          |
| 13   | H-18 | Block-Sparse Attention (CUDA kernel)      | 2-10x (sparse)     | Very High  | CPU FAIL   | MEDIUM       |
| 14   | H-22 | Optimal Einsum Reordering                 | 0.87-1.13x (mixed) | Low        | CPU MIXED  | LOW          |

---

## Detailed Hypotheses

### H-07: KV-Cache Incremental Decode
**Expected gain**: 18x (cache_len=512) to 69x (cache_len=2048) vs full recompute  
**Why**: During autoregressive decode, each new token only needs to attend to past tokens.
Full recompute is O(seqlen^2); incremental decode is O(seqlen * 1).  
**H100 implications**: Decode is HBM-bandwidth-bound. Incremental decode reduces memory traffic
by factor proportional to cache length. At cache_len=2048: touch 1 query + 2048 cached KV
instead of 2048 Q + 2048 K + 2048 V.  
**Implementation**: Cache K/V tensors; single-query SDPA per decode step. Already implemented
in quantized_kv_cache.py. No Triton needed; PyTorch SDPA suffices.  
**Failure modes**: 
- Cold cache: first token has no benefit (cache_len=0)
- KV cache management overhead if paged (separate hypothesis H-11)
- INT8 quantization of cached KV values introduces ~1e-3 error (acceptable for LLM inference)  
**Status**: CPU PASS (69x at cache_len=2048). GPU benchmark pending (SSH blocked).

---

### H-01: FlashAttention-2 (Triton Tiled, Causal)
**Expected gain**: 2-5x latency vs naive attention; 1.5-2x vs PyTorch SDPA (which may use FA2 internally)  
**Why**: Standard attention materializes the full O(N^2) attention matrix in HBM.
FlashAttention tiles the computation to fit in SRAM, reducing HBM reads/writes from O(N^2)
to O(N). On H100 with 80MB L2 + fast SRAM, this is the dominant optimization for prefill.  
**Implementation**: Triton kernel with BLOCK_M=128, BLOCK_N=64 tiling (optimal for sm_90).
The benchmark_harness.py includes a Triton FA2 kernel. Compare vs F.scaled_dot_product_attention
which on H100 routes to cuDNN FA3.  
**Failure modes**:
- PyTorch 2.x SDPA already uses FlashAttention on H100 -- comparison may show <10% gain vs SDPA
- Triton JIT compile time: first run ~2-5s; use warmup=10+ iters
- Head dim must be power of 2 and <=128 for this kernel; head_dim=256 needs separate tuning
- For seqlen=8192: BLOCK_M=128 grid has 64 blocks per head; verify no OOM on activation memory  
**Status**: Unverified on GPU.

---

### H-02: FlashAttention-3 (Warp Specialization + TMA)
**Expected gain**: 1.5-2x vs FA2 on H100  
**Why**: FA3 (Shah et al. 2024) adds:
1. Warp specialization: producer warps issue TMA loads while consumer warps compute GEMMs
2. Pingpong scheduling: GEMM1 overlaps with softmax, GEMM2 overlaps with GEMM1
3. Non-causal FA3 achieves ~740 TFLOPS vs 350 TFLOPS for FA2 on H100  
**Implementation complexity**: Very High. Requires PTX-level wgmma instructions or
use of the official FA3 library (https://github.com/Dao-AILab/flash-attention).
Not trivial to write from scratch in Triton.  
**Recommended approach**: pip install flash-attn==2.7+ which ships FA3 on H100;
benchmark against our Triton FA2 kernel.  
**Failure modes**:
- Requires CUDA 12.3+; check Lambda H100 driver version
- FA3 is bf16/fp16 only; fp32 paths fall back to FA2
- Installation may fail without NVCC matching CUDA version  
**Status**: Unverified. Needs Lambda GPU access.

---

### H-08: Quantized KV-Cache (INT8/FP8 + Incremental Decode)
**Expected gain**: 2-4x reduction in KV-cache memory footprint; compound speedup with H-07  
**Why**: On H100, autoregressive decode throughput scales with batch size; larger batch
needs more KV cache. INT8 KV quantization halves cache size vs fp16, enabling 2x batch
at same memory; FP8 enables 4x.  
**Combined effect**: H-07 gives 69x latency reduction; quantization gives 2-4x memory savings
enabling larger batches. For serving, this translates to 2-4x higher token throughput.  
**Implementation**: quantized_kv_cache.py (miniQuant, verified). Uses INT8 symmetric quantization
per channel for K and V; dequantize before attention to maintain accuracy.  
**Failure modes**:
- INT8 KV error: ~1e-3 max abs (acceptable; LLMs degrade <0.1 perplexity points at this level)
- FP8 only on sm_89+ (H100 is sm_90, OK)
- Requires careful scale factor management for long sequences (activation drift)  
**Status**: Code ready in miniQuant's sandbox. GPU benchmark pending.

---

### H-03: Fused RoPE + Attention (Single Kernel Pass)
**Expected gain**: 1.3-1.8x vs separate RoPE then SDPA  
**Why**: Standard LLaMA/Mistral implementation applies RoPE to Q and K separately,
then calls SDPA. Each operation is a full HBM read-modify-write. Fusing into a single
kernel (read Q/K once, apply RoPE, compute attention, write O) saves 2 HBM passes.  
**Implementation**: benchmark_harness.py includes fused_rope_attention using PyTorch ops.
For real fusion, a custom Triton kernel would apply the cos/sin in the GEMM epilogue.  
**Failure modes**:
- RoPE computation is elementwise and fast; the fusion benefit may be bandwidth-limited
  only at seqlen>=2048 where the full tensors don't fit in L2
- head_dim=128 with seqlen=8192 at batch=32: 32*32*8192*128*2bytes = 2.1GB per tensor (QKV)
  -- verify no OOM  
**Status**: CPU benchmark shows pass. GPU gain estimate is higher due to HBM bandwidth limits.

---

### H-04: Triton GEMM Tile Sweep (sm_90 Tuned)
**Expected gain**: 1.2-2x vs cuBLAS for specific shapes  
**Why**: cuBLAS has generic tile sizes optimized for large square matrices.
For transformer-specific shapes (e.g., M=4096, N=4096, K=128 for proj layers or
M=1, N=4096, K=4096 for decode), custom tile sizes can achieve 10-20% gains.  
**Implementation**: Triton `@triton.autotune` sweep over BLOCK_M in {16,32,64,128,256},
BLOCK_N in {16,32,64,128,256}, num_stages in {1,2,3,4}, num_warps in {4,8}.
Focus on shapes common in LLaMA-70B: (4096,4096), (4096,14336), (1,4096).  
**Failure modes**:
- cuBLAS on H100 is highly optimized; custom Triton GEMM is unlikely to beat it for
  large square matrices (benchmark may show parity or slight regression)
- Autotuning is expensive: 1000s of compilations; cache results to a JSON file
- For decode (M=1) shapes, cuBLAS uses GEMV not GEMM; Triton GEMV needs separate kernel  
**Status**: Unverified. High value to run on Lambda.

---

### H-05: Fused Softmax + LayerNorm (Single Pass)
**Expected gain**: 1.2-1.5x for fused vs separate ops  
**Why**: Both softmax and layernorm are bandwidth-bound reduction ops. Two separate
passes over the same tensor can be merged into one.  
**Triton reference**: `triton.ops.softmax` in Triton tutorials; `@triton.jit` layernorm.  
**Failure modes**:
- PyTorch 2.x already fuses these via torch.compile; standalone benefit may be small
- Test with torch.compile disabled to see raw benefit  
**Status**: Unverified.

---

### H-06: Persistent Decode Kernel (Streaming)
**Expected gain**: 1.3-2x for high-throughput token generation  
**Why**: Standard decode launches a new CUDA kernel per token. Persistent kernels keep
warps resident on SM, eliminating kernel launch overhead (~5-10us per token) and improving
L2 cache reuse across consecutive decode steps.  
**Implementation complexity**: High. Requires cooperative groups + persistent grid.
See Dao et al. "Flash-Decoding" and NVIDIA "Persistent Kernels for ML" (2024).  
**Failure modes**:
- Persistent kernels on H100 require careful occupancy tuning; wrong config → undersubscription
- Benefit most visible at batch=1 low-latency decode; at batch=32, kernel launch overhead amortized  
**Status**: Unverified. Needs GPU.

---

### H-09: Warp-Level INT8 GEMM (Per-Channel Scale, SmoothQuant)
**Expected gain**: 1.5-3x vs fp16 GEMM (2x from reduced arithmetic, offset by dequant overhead)  
**Why**: H100 Tensor Cores support INT8 GEMM at 2x fp16 peak TFLOPS. Per-channel quantization
(SmoothQuant, Xiao et al. 2022) achieves INT8 accuracy near fp16 for LLM weight matrices.  
**Implementation**: perchannel_quant.py (miniQuant). CPU results: 8.1e-5 error for small channels
(46x improvement over per-tensor); partial cert due to INT8 step size limits for large channels.  
**Failure modes**:
- Quantization calibration data required for real accuracy guarantee
- torch._scaled_mm requires sm_89+; check CUDA version on Lambda  
**Status**: CPU PARTIAL. GPU benchmark pending.

---

### H-10: FP8 E4M3 GEMM (H100 Native)
**Expected gain**: 2-3x vs fp16 GEMM; 4x peak TFLOPS (FP8 Tensor Cores)  
**Why**: H100 introduces FP8 (E4M3 and E5M2) support with 2x the TFLOPS vs FP16.
NVIDIA Transformer Engine uses FP8 in production for LLaMA inference.  
**Implementation**: quant_kernels.py (miniQuant) includes fp8_gemm and fp8_attention.
Requires PyTorch >= 2.1 with CUDA 12.1+ for torch.float8_e4m3fn support.  
**Failure modes**:
- FP8 E4M3 has only 3 exponent bits; overflow risk with large activations
- Scaling factor management critical (per-tensor amax + delay update pattern)
- First run will be slower due to FP8 fallback if sm_90 FP8 path not activated  
**Status**: Code ready. GPU benchmark pending.

---

### H-11: Paged KV-Cache Access Patterns
**Expected gain**: 1.2-2x for long-context serving (variable batch sizes)  
**Why**: vLLM's PagedAttention (Kwon et al. 2023) stores KV cache in non-contiguous pages,
enabling flexible memory allocation and eliminating fragmentation. For high-throughput
serving with variable sequence lengths, this 2x memory utilization gain translates to
2x throughput.  
**Implementation complexity**: High. Requires custom attention kernel that accepts
page table lookups. Not a standalone Python benchmark; needs integration with serving infra.  
**Failure modes**:
- Paged access adds indirection overhead (~5% per block lookup)
- For batch=1 latency benchmarks, paged is slower than contiguous
- Best measured via vLLM throughput benchmarks, not micro-benchmarks  
**Status**: Unverified. Lower priority for micro-benchmark sprint.

---

### H-18: Block-Sparse Attention (CUDA Kernel)
**Expected gain**: 2-10x for sparse patterns (e.g., 50% sparsity → 2-4x)  
**Why**: For long contexts, most token pairs have near-zero attention. Block-sparse
kernels skip computation for zero blocks.  
**CPU result**: FAIL -- Python loop overhead negates FLOP reduction.  
**H100 path**: Requires native CUDA or Triton kernel with block-sparse indexing.
See OpenAI Sparse Transformer and DeepSpeed Sparse Attention.  
**Failure modes**:
- Sparsity pattern must be known at compile time or use dynamic indexing (slow)
- Actual LLM attention is not block-sparse without explicit masking  
**Status**: CPU FAIL. Would need Triton/CUDA implementation to be viable on GPU.

---

### H-22: Optimal Einsum Reordering
**Expected gain**: 0.87-1.13x (mixed results from CPU)  
**Why**: Reordering einsum contractions can reduce intermediate tensor sizes.  
**CPU results**: B=1 FAIL (0.87x), B=8 marginal PASS (1.13x). F.sdp at 8.1x dominates.  
**Assessment**: Low-priority. PyTorch's SDPA already uses optimal contraction order.  
**Status**: CPU MIXED. Low priority.

---

## Sprint Benchmark Artifact Checklist

Before accepting any result as "confirmed":

1. **Cold cache effect**: Run warmup=10 iters before timing. First iter is always slow (kernel JIT + cache cold).
2. **Wrong batch size assumption**: Run all 3 batch sizes [1, 8, 32]. Batch=1 latency ≠ batch=32 throughput.
3. **Non-representative seqlen**: Test all 3 [512, 2048, 8192]. Short seqlen masks real attention O(N^2) behavior.
4. **Compiler auto-vectorization masking real gains**: Compare with `torch.compile` disabled AND enabled.
5. **Numerical accuracy**: Check max_abs_error vs fp32 baseline. Accept ≤1e-3 for fp16, ≤1e-4 for int8 (per-channel).
6. **Memory bound vs compute bound**: Measure both TFLOPS and memory bandwidth. If TFLOPS is low but BW is high, kernel is memory-bound.
7. **GPU utilization**: Use `nvidia-smi dmon` or NVTX ranges to confirm GPU is actually active.

---

## Ranked by (Confidence × Potential Gain)

| Rank | Hypothesis                          | Confidence | Gain   | Score | Next Action                          |
|------|-------------------------------------|------------|--------|-------|--------------------------------------|
| 1    | H-07 KV-Cache Incremental Decode    | 0.95       | 69x    | 65.6  | Run on H100 (same code)              |
| 2    | H-08 Quantized KV-Cache             | 0.85       | 4x     | 3.4   | Run quantized_kv_cache.py on H100    |
| 3    | H-10 FP8 GEMM (H100 native)         | 0.80       | 3x     | 2.4   | Run quant_kernels.py fp8 path        |
| 4    | H-09 INT8 GEMM (per-channel)        | 0.75       | 2.5x   | 1.9   | Run perchannel_quant.py on H100      |
| 5    | H-01 FlashAttention-2 Triton        | 0.70       | 3x     | 2.1   | Run benchmark_harness.py triton path |
| 6    | H-03 Fused RoPE+Attention           | 0.70       | 1.5x   | 1.1   | Run benchmark_harness.py rope path   |
| 7    | H-04 Triton GEMM Tile Sweep         | 0.55       | 1.5x   | 0.8   | New Triton autotune script needed    |
| 8    | H-06 Persistent Decode Kernel       | 0.40       | 1.5x   | 0.6   | Research + implementation needed     |
| 9    | H-02 FlashAttention-3               | 0.60       | 1.7x   | 1.0   | pip install flash-attn on Lambda     |
| 10   | H-11 Paged KV-Cache                 | 0.30       | 1.6x   | 0.5   | Needs serving infra integration      |

---

## Assignment to Agents

| Agent       | Current Assignment                          | Next After SSH Unblocked               |
|-------------|---------------------------------------------|----------------------------------------|
| miniQuant   | Quant suite (6 files, ready)                | Run deploy.sh on Lambda                |
| ooo         | Unknown (check sandbox)                     | Run benchmark_harness.py (H-01, H-03)  |
| coolstufs   | Sprint coordination                         | Synthesis update every 15 min          |
| serious-inf | Benchmark harness + hypothesis sweep        | Trigger all deploys once SSH works     |
