# GPU Kernels Status

## Summary

Files added to `kernel-research/harness/` targeting H100 80GB (Lambda ubuntu@192.222.55.210):

- `gpu_kernels.py`    -- five GPU-specific kernel implementations
- `gpu_benchmark.py`  -- benchmark runner integrating with harness.py

## Kernels in gpu_kernels.py

### 1. triton_flash_attention
Flash Attention v1 forward pass as custom Triton kernel. BLOCK_M=128, BLOCK_N=64, BLOCK_D=64.
Target: >= 1.10x speedup over torch.nn.functional.scaled_dot_product_attention.
Correctness threshold: max abs error <= 1e-2.

### 2. compiled_attention
torch.compile(mode="reduce-overhead") wrapped naive attention.
Target: >= 1.10x speedup over naive_attention.
Correctness threshold: max abs error <= 1e-4.

### 3. fp16_gemm
torch.matmul on float16 tensors. Exercises H100 3rd-gen Tensor Cores (~312 TFLOPS FP16).
Target: ~4x speedup over FP32 matmul.
Correctness threshold: max abs error <= 1e-1.

### 4. int8_gemm_scaled
torch._scaled_mm INT8 hardware-accelerated GEMM. H100: ~624 TOPS INT8.
Target: ~2x speedup over FP16 GEMM for compute-bound shapes.
Requires: PyTorch >= 2.1 with CUDA.
Correctness threshold: max abs error <= 1.0 (INT8 quantisation is lossy).

### 5. compiled_fused_softmax_cast
torch.compile fused softmax+cast. Reduces 2 HBM round-trips to 1.
Target: >= 1.10x speedup over unfused_softmax_cast.
Correctness threshold: max abs error <= 1e-4.

## gpu_benchmark.py

Integrates with harness.py timing framework. Writes to results.jsonl.
Usage:
    python gpu_benchmark.py                     # all kernels, batch_sizes [1,8,32]
    python gpu_benchmark.py --kernel triton_flash
    python gpu_benchmark.py --batch_size 8

## Status

| File | Status |
|------|--------|
| `gpu_kernels.py` | Written, syntactically valid, Triton-guarded |
| `gpu_benchmark.py` | Written, pushed to GitHub |
| `quant/quant_kernels.py` | Written by miniQuant, pushed |
| `quant/numerics_validator.py` | Written by miniQuant, pushed |
| `quant/benchmark_quant.py` | Written by miniQuant, pushed |

All code ready to run on H100 once Lambda SSH is resolved.
