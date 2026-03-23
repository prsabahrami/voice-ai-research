# Kernel Optimization Hypotheses

Source: ooo:kernel_research_sprint_launch
Generated: 2026-03-23 22:07 UTC
Total: 20 hypotheses ranked by expected speedup

---

## TIER 1: Immediate Execution (H100 Native, High Confidence)

### H1 [ATTN, BW]: FlashAttention-3 FP8 MLA Decoding Kernel
- **hypothesis_id**: h001
- **kernel_type**: attention
- **method**: FlashAttention-3 FP8 MLA decoding kernel (FlashMLA reference implementation)
- **expected_speedup**: 1.3-2.0x over BF16
- **measurable_criteria**: speedup >= 1.10, p99/p50 < 1.05, abs_err < 1e-4, batch_sizes [1, 8, 32]
- **estimated_difficulty**: high
- **requirements**: CUDA >= 12.8, H100 SM90a FP8 e4m3
- **priority_rank**: 1

### H2 [FUSED, BW]: Fused RMSNorm+RoPE+QKV Triton Kernel
- **hypothesis_id**: h002
- **kernel_type**: fused-ops
- **method**: Fused RMSNorm+RoPE+QKV projection in single Triton kernel (Liger-Kernel reference: 2.3x RoPE)
- **expected_speedup**: 1.2-1.5x (2-4 fewer HBM round-trips)
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: medium
- **requirements**: Triton >= 2.1.0
- **priority_rank**: 2

### H3 [GEMM, QU]: W4A8 SplitK Decode GEMM
- **hypothesis_id**: h003
- **kernel_type**: GEMM
- **method**: W4A8 SplitK decode GEMM (SplitK +61% waves/SM on A100, +124% on H100)
- **expected_speedup**: 2.0-2.7x decode throughput
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: high
- **requirements**: H100 Tensor Cores, cuBLAS/Triton
- **priority_rank**: 3

### H4 [SCHED, BW]: CUDA Graphs Persistent Decode
- **hypothesis_id**: h004
- **kernel_type**: scheduling
- **method**: CUDA Graphs with persistent decode session to eliminate CPU kernel launch overhead
- **expected_speedup**: 1.15-1.30x (15-30% latency reduction)
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: medium
- **requirements**: CUDA >= 11.3, PyTorch >= 2.0
- **priority_rank**: 4

### H5 [QU, BW]: FP8 End-to-End Pipeline
- **hypothesis_id**: h005
- **kernel_type**: attention
- **method**: FP8 end-to-end pipeline using H100 native FP8 E4M3 operations throughout
- **expected_speedup**: 1.5x throughput, 2x KV cache capacity
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: high
- **requirements**: CUDA >= 12.3, H100 SM90a
- **priority_rank**: 5

---

## TIER 2: Active Kernel Development

### H6 [SCHED]: PagedAttention Block Size Tuning for H100 L2
- **hypothesis_id**: h006
- **kernel_type**: attention
- **method**: Tune PagedAttention block size to match H100 L2 cache size (50MB)
- **expected_speedup**: 1.05-1.15x throughput
- **measurable_criteria**: speedup >= 1.10, variance < 5%
- **estimated_difficulty**: low
- **priority_rank**: 6

### H7 [FUSED]: Fused Linear + CrossEntropy
- **hypothesis_id**: h007
- **kernel_type**: fused-ops
- **method**: Fused Linear + CrossEntropy kernel (Liger-Kernel reference: 84% peak memory reduction)
- **expected_speedup**: 1.15-1.40x (memory pressure reduction enables higher batch size)
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: medium
- **priority_rank**: 7

### H8 [ATTN]: MLA Absorbed-Weight Fused Kernel
- **hypothesis_id**: h008
- **kernel_type**: attention
- **method**: Multi-head Latent Attention (MLA) absorbed-weight fused kernel
- **expected_speedup**: 1.20-1.40x MLA attention
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: high
- **priority_rank**: 8

### H9 [ATTN]: Triton Paged Attention to FA3 Parity
- **hypothesis_id**: h009
- **kernel_type**: attention
- **method**: Triton paged attention kernel tuned to FlashAttention-3 parity (Ringlein et al. 2025: 19.7% -> 105.9% of theoretical peak)
- **expected_speedup**: 1.30-2.00x over naive paged attention
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: high
- **priority_rank**: 9

### H10 [FUSED]: GeLU/SwiGLU FFN Fusion
- **hypothesis_id**: h010
- **kernel_type**: fused-ops
- **method**: Fused GeLU/SwiGLU FFN gate+up activation in single Triton kernel
- **expected_speedup**: 1.05-1.15x throughput, 12-20% memory reduction
- **measurable_criteria**: speedup >= 1.10, variance < 5%, abs_err < 1e-4
- **estimated_difficulty**: medium
- **priority_rank**: 10

---

## TIER 3: Research Grade

| ID | Type | Method | Expected Speedup | Difficulty |
|----|------|--------|-----------------|------------|
| H11 | scheduling | Disaggregated prefill/decode KV transfer | 30-50% transfer latency | high |
| H12 | quantization | INT4 Marlin kernel for H100 | 2.5-2.7x decode throughput | high |
| H13 | attention | Roofline-guided attention tiling autotuner | 10-45% across config space | medium |
| H14 | attention | Persistent warp-specialized prefill | 840+ TFLOPS/s | high |
| H15 | fused-ops | Online softmax + attention output proj fusion | 10-20% decode | medium |
| H16 | quantization | FP8 block-quantized GEMM | match per-tensor FP8 with better accuracy | medium |
| H17 | attention | SplitK flash decode for large head dims | 30-50% decode latency at seqlen>16K | high |
| H18 | scheduling | torch.compile baseline | 20-30% over HF eager | low |
| H19 | scheduling | KV cache low-rank compression | 40-60% KV memory reduction | medium |
| H20 | scheduling | Speculative decoding draft-verify fusion | 20-35% verify overhead reduction | high |

---

## Success Criteria (All Hypotheses)

Per eval harness (serious-inference-engineer eval/validate_results.py):
- speedup >= 1.10 (10%+ wall-clock improvement)
- p99/p50 ratio < 1.05 (variance under 5%)
- correctness_max_abs_err < 1e-4 for FP16
- batch_sizes_tested includes [1, 8, 32]

Results schema (write to /home/ubuntu/kernel-research/results/results.jsonl):
```json
{"hypothesis_id": "h001", "kernel_type": "attention", "method": "...", "baseline_us": 123.4, "optimized_us": 110.2, "speedup": 1.12, "p99_latency_ms": 2.1, "p50_latency_ms": 2.05, "correctness_max_abs_err": 0.000045, "batch_sizes_tested": [1, 8, 32], "passed_criteria": true, "notes": "...", "timestamp": "2026-03-23T22:00:00Z", "agent": "ooo", "gpu": "H100 80GB"}
```
