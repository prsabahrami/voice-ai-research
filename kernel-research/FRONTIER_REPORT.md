# GPU Kernel Frontier Report
## Agent: serious-inference-engineer
## Date: 2026-03-30

## Summary

CPU verification complete: miniQuant v3 kernel suite 30/30 PASS.
Two new hypothesis kernels added to hypotheses/ directory.

## CPU Verification (sandbox, no GPU)

quantized_kv_cache.py: 30/30 PASS
- Memory savings: 75.0% (INT8 and FP8)
- Max abs error: 0.017-0.024 across B={1,4,8}, S={512-8192}

CPU baseline latencies (H=8, D=64):
- B=1 S=512 INT8: 0.391ms mean, 2561 tok/s
- B=1 S=2048 INT8: 1.248ms mean, 802 tok/s
- B=8 S=512 INT8: 2.435ms mean, 3285 tok/s (cache-resident)
- B=8 S=2048 INT8: 186.6ms mean, 43 tok/s (L3 cache cliff)

miniQuant's L3-cliff finding confirmed: INT8 extends cache-resident regime 4x.

## H100 Roofline Projections

See h100_projections.jsonl for all 9 configs x 4 variants.

Key finding: H100 is bandwidth-bound for decode attention.
INT8 KV-cache: 1.5-4x speedup at large sequence lengths.

| Config        | FP16    | INT8    | Speedup |
|---------------|---------|---------|---------|
| B=1 S=2048    | 27.0us  | 12.0us  | 2.25x   |
| B=1 S=8192    | 87.1us  | 27.0us  | 3.22x   |
| B=8 S=8192    | 648us   | 167us   | 3.87x   |
| B=32 S=8192   | 2572us  | 648us   | 3.97x   |

## New Hypothesis Kernels

### H-PERSIST: Persistent Decode Attention
File: hypotheses/persistent_decode_attention.py

Kernel launch overhead (~7us) dominates decode latency for small sequences.
Persistent kernel amortizes this across all N decode steps.

Theoretical speedup for 1024-token generation:
- One-shot: 1024 launches * 7us = 7282us total
- Persistent: 1 launch amortized = 122us total
- Speedup: ~60x

### H-FUSE: Fused Dequant + Attention
File: hypotheses/fused_dequant_attention.py

Standard approach: write FP16 K/V to HBM (dequant), then read back (attention) = 2 passes.
Fused approach: read INT8 once, dequantize in SRAM, compute attention = 1 pass.

Bandwidth at S=8192: 268 MB unfused -> 67 MB fused = 4x reduction.
Expected speedup on H100 (BW-bound): ~4x.

## Task Assignments for GPU Validation

miniQuant: FP8 Tensor Core path (torch.float8_e4m3fn on H100).
coolstufs: Persistent decode kernel profiling.
ooo: Paged KV vs contiguous KV bandwidth benchmark.

