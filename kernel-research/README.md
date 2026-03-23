# Kernel Optimization Research Sprint

Automated scientific discovery of GPU kernel optimization methods for inference acceleration.

## Directory Structure

```
kernel-research/
├── hypotheses/        # Hypothesis lists (ooo's track)
│   └── hypotheses.md
├── harness/           # Benchmark harness (coolstufs' track)
│   ├── harness.py
│   ├── run_benchmark.py
│   ├── baseline_kernels.py
│   └── quant/         # Quantization kernels (miniQuant's track)
│       ├── quant_kernels.py
│       └── numerics_validator.py
├── results/           # Shared results database
│   └── results.jsonl
├── synthesis/         # Rolling synthesis document
│   └── SYNTHESIS.md
└── eval/              # Eval harness (serious-inference-engineer's track)
    ├── eval_harness.py
    ├── hypothesis_validator.py
    └── ablation_runner.py
```

## Results Schema (results.jsonl)

Each line is a JSON object:
```json
{
  "hypothesis_id": "h001",
  "kernel_type": "attention|GEMM|fused-ops|other",
  "method": "concrete description of optimization",
  "baseline_us": 123.4,
  "optimized_us": 110.2,
  "speedup": 1.12,
  "correctness_max_abs_err": 0.000045,
  "batch_sizes_tested": [1, 8, 32],
  "passed_criteria": true,
  "notes": "any additional context",
  "timestamp": "2026-03-23T21:00:00Z",
  "agent": "coolstufs",
  "gpu": "H100 80GB"
}
```

## Success Criteria (enforced by serious-inference-engineer eval harness)

- 10%+ wall-clock improvement (speedup >= 1.10)
- Variance under 5% (p99/p50 ratio < 1.05)
- Max absolute error below 1e-4 for FP16
- Must hold across batch sizes [1, 8, 32]

## Constraints

- No PRs, no email, no human approval
- All code committed directly to this directory
- serious-inference-engineer owns eval rigor -- route all final results through eval harness before declaring victory
- Use tmux sessions named kernel-research-<track> for persistent Lambda sessions
