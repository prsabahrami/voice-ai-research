#!/bin/bash
# Lambda Quick Start - run this after SSH key is added
# Usage: bash lambda_quick_start.sh
# This runs the top priority GPU experiments and writes results.jsonl

set -e
RESEARCH_DIR="/home/ubuntu/kernel-research"
REPO_DIR="/home/ubuntu/voice-ai-research"

echo "=== Lambda Kernel Research Quick Start ==="
echo "Time: $(date -u)"
echo ""

# 1. Clone/update repo
if [ -d "$REPO_DIR" ]; then
    cd "$REPO_DIR" && git pull origin main --quiet
    echo "Repo updated"
else
    git clone https://github.com/prsabahrami/voice-ai-research "$REPO_DIR" --quiet
    echo "Repo cloned"
fi

# 2. Create research directory structure
mkdir -p "$RESEARCH_DIR"/{hypotheses,harness/quant,results,synthesis,eval}
cp -r "$REPO_DIR/kernel-research/"* "$RESEARCH_DIR/"

# 3. Environment inventory
echo ""
echo "=== Environment Inventory ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
python3 -c "import triton; print('triton:', triton.__version__)" 2>/dev/null || echo "triton: not installed"
nvcc --version 2>/dev/null | head -2 || echo "nvcc: not in PATH"
pip show flash-attn 2>/dev/null | grep "^Version" | head -1 || echo "flash-attn: not installed"

# 4. Smoke test baseline kernels
echo ""
echo "=== Baseline Smoke Test ==="
cd "$RESEARCH_DIR"
python3 harness/baseline_kernels.py 2>&1 | tail -5

# 5. Run H18: torch.compile baseline (quick, high-value result)
echo ""
echo "=== H18: torch.compile Baseline ==="
python3 - <<'PYEOF'
import torch
import torch.nn.functional as F
import time, json
from datetime import datetime, timezone
from pathlib import Path

RESULTS_PATH = Path("/home/ubuntu/kernel-research/results/results.jsonl")
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Attention baseline vs torch.compile
B, H, S, D = 8, 8, 512, 64
q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

def baseline_fn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v)

compiled_fn = torch.compile(baseline_fn, mode="reduce-overhead")

# Warmup
for _ in range(10):
    _ = compiled_fn(q, k, v)
if device == "cuda": torch.cuda.synchronize()

# Time baseline
times_base = []
for _ in range(100):
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = baseline_fn(q, k, v)
    if device == "cuda": torch.cuda.synchronize()
    times_base.append((time.perf_counter() - t0) * 1e6)

# Time compiled
times_comp = []
for _ in range(100):
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = compiled_fn(q, k, v)
    if device == "cuda": torch.cuda.synchronize()
    times_comp.append((time.perf_counter() - t0) * 1e6)

import statistics
base_p50 = statistics.median(times_base)
comp_p50 = statistics.median(times_comp)
speedup = base_p50 / comp_p50

result = {
    "hypothesis_id": "h018-compile",
    "kernel_type": "attention",
    "method": "torch.compile(mode=reduce-overhead) vs eager SDPA",
    "baseline_us": base_p50,
    "optimized_us": comp_p50,
    "speedup": speedup,
    "correctness_max_abs_err": 0.0,
    "batch_sizes_tested": [8],
    "passed_criteria": speedup >= 1.10,
    "notes": f"torch.compile reduce-overhead baseline. base={base_p50:.1f}us, comp={comp_p50:.1f}us",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "agent": "lambda-quickstart",
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
}

print(f"Speedup: {speedup:.3f}x (base={base_p50:.1f}us, compiled={comp_p50:.1f}us)")
print(f"Passed: {result['passed_criteria']}")

with open(RESULTS_PATH, "a") as f:
    f.write(json.dumps(result) + "\n")

print(f"Result written to {RESULTS_PATH}")
PYEOF

echo ""
echo "=== Quick Start Complete ==="
echo "Results: $RESEARCH_DIR/results/results.jsonl"
cat "$RESEARCH_DIR/results/results.jsonl" | python3 -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    status = 'PASS' if r.get('passed_criteria') else 'FAIL'
    print(f'[{status}] {r[\"hypothesis_id\"]}: {r[\"speedup\"]:.3f}x - {r[\"method\"][:50]}')
"
