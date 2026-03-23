#!/usr/bin/env bash
# deploy.sh
# =========
# Copies the quant/ harness to Lambda H100 and runs the benchmark.
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# Assumes:
#   - SSH private key at ~/.ssh/id_ed25519
#   - Lambda instance reachable at ubuntu@192.222.55.210
#   - Python 3.10+ with PyTorch 2.1+, Triton installed on Lambda

set -euo pipefail

LAMBDA_HOST="ubuntu@192.222.55.210"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_BASE="/home/ubuntu/kernel-research/harness"
REMOTE_QUANT="${REMOTE_BASE}/quant"
REMOTE_RESULTS="${REMOTE_BASE}/../results"
LOCAL_QUANT="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="$(dirname "${LOCAL_QUANT}")/results"

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=15"

echo "========================================"
echo "miniQuant Deploy Script"
echo "Target : ${LAMBDA_HOST}"
echo "Key    : ${SSH_KEY}"
echo "Remote : ${REMOTE_QUANT}"
echo "========================================"

if [[ ! -f "${SSH_KEY}" ]]; then
    echo "ERROR: SSH key not found at ${SSH_KEY}"
    exit 1
fi

echo ""
echo "[1/5] Creating remote directories..."
ssh ${SSH_OPTS} "${LAMBDA_HOST}" "mkdir -p ${REMOTE_QUANT} ${REMOTE_RESULTS}"

echo "[2/5] Copying quant/ files..."
scp ${SSH_OPTS} \
    "${LOCAL_QUANT}/quant_kernels.py" \
    "${LOCAL_QUANT}/numerics_validator.py" \
    "${LOCAL_QUANT}/benchmark_quant.py" \
    "${LOCAL_QUANT}/deploy.sh" \
    "${LAMBDA_HOST}:${REMOTE_QUANT}/"

echo "Files copied."

echo "[3/5] Running benchmark on Lambda..."
ssh ${SSH_OPTS} "${LAMBDA_HOST}" bash <<'REMOTE_EOF'
set -euo pipefail
cd /home/ubuntu/kernel-research/harness/quant

if [[ -f ~/venv/bin/activate ]]; then
    source ~/venv/bin/activate
fi

echo "Python: $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"

python3 benchmark_quant.py
REMOTE_EOF

echo "Benchmark complete."

echo "[4/5] Retrieving results..."
mkdir -p "${LOCAL_RESULTS}"
scp ${SSH_OPTS} \
    "${LAMBDA_HOST}:${REMOTE_RESULTS}/quant_results.jsonl" \
    "${LOCAL_RESULTS}/quant_results.jsonl" || echo "Warning: could not retrieve results"

echo "[5/5] Done."
echo ""
echo "Results saved to: ${LOCAL_RESULTS}/quant_results.jsonl"
