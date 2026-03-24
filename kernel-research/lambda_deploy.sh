#!/usr/bin/env bash
# lambda_deploy.sh
# ================
# Deploys the kernel-research GPU benchmark suite to a Lambda H100 instance
# and runs the full benchmark suite.
#
# Usage:
#   chmod +x lambda_deploy.sh
#   ./lambda_deploy.sh
#
# Environment variables (override defaults):
#   LAMBDA_HOST  -- SSH target, default: ubuntu@192.222.55.210
#   SSH_KEY      -- Path to SSH private key, default: ~/.ssh/id_ed25519
#   REPO_URL     -- GitHub repo URL, default: https://github.com/prsabahrami/voice-ai-research
#
# What this script does:
#   1. Verifies SSH connectivity to Lambda
#   2. Clones or updates prsabahrami/voice-ai-research on Lambda
#   3. Runs the GPU benchmark runner (kernel-research/gpu_benchmark.py)
#   4. Retrieves results from Lambda to local results/

set -euo pipefail

LAMBDA_HOST="${LAMBDA_HOST:-ubuntu@192.222.55.210}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
REPO_URL="${REPO_URL:-https://github.com/prsabahrami/voice-ai-research}"
REMOTE_REPO="/home/ubuntu/voice-ai-research"
REMOTE_RESULTS="${REMOTE_REPO}/kernel-research/results"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_RESULTS="${LOCAL_DIR}/results"

SSH_OPTS="-i ${SSH_KEY} -o StrictHostKeyChecking=no -o ConnectTimeout=20"

echo "============================================================"
echo "Lambda GPU Benchmark Deployment"
echo "Target : ${LAMBDA_HOST}"
echo "Key    : ${SSH_KEY}"
echo "Repo   : ${REPO_URL}"
echo "============================================================"

# 1. Check SSH key
if [[ ! -f "${SSH_KEY}" ]]; then
    echo "ERROR: SSH key not found at ${SSH_KEY}"
    echo "Set SSH_KEY env var or place key at ${SSH_KEY}"
    exit 1
fi

# 2. Verify SSH connectivity
echo ""
echo "[1/5] Testing SSH connectivity..."
if ! ssh ${SSH_OPTS} "${LAMBDA_HOST}" "echo SSH_OK" 2>/dev/null; then
    echo "ERROR: Cannot connect to ${LAMBDA_HOST}. Check SSH key and network."
    exit 1
fi
echo "  SSH connection successful"

# 3. Clone or update repo on Lambda
echo ""
echo "[2/5] Syncing repo on Lambda..."
ssh ${SSH_OPTS} "${LAMBDA_HOST}" bash << REMOTE_EOF
set -euo pipefail
if [[ -d "${REMOTE_REPO}/.git" ]]; then
    echo "  Pulling latest changes..."
    git -C "${REMOTE_REPO}" pull --ff-only
else
    echo "  Cloning repo..."
    git clone "${REPO_URL}" "${REMOTE_REPO}"
fi
echo "  Repo synced at $(git -C ${REMOTE_REPO} log --oneline -1)"
REMOTE_EOF

# 4. Run environment inventory
echo ""
echo "[3/5] Lambda environment inventory..."
ssh ${SSH_OPTS} "${LAMBDA_HOST}" bash << 'REMOTE_EOF'
set -euo pipefail
echo "  OS: $(uname -r)"
python3 --version 2>&1 | sed 's/^/  /'
python3 -c "import torch; print('  PyTorch:', torch.__version__, '|', 'CUDA:', torch.version.cuda if torch.cuda.is_available() else 'N/A')" 2>/dev/null || echo "  PyTorch: not installed"
python3 -c "import torch; print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')" 2>/dev/null || true
python3 -c "import triton; print('  Triton:', triton.__version__)" 2>/dev/null || echo "  Triton: not installed"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/  GPU: /' || echo "  nvidia-smi: not available"
REMOTE_EOF

# 5. Run the GPU benchmarks
echo ""
echo "[4/5] Running GPU benchmarks on Lambda (this may take 5-15 minutes)..."
ssh ${SSH_OPTS} "${LAMBDA_HOST}" bash << REMOTE_EOF
set -euo pipefail
cd "${REMOTE_REPO}/kernel-research"

# Activate virtualenv if present
if [[ -f ~/venv/bin/activate ]]; then
    source ~/venv/bin/activate
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

mkdir -p results/

echo "  Starting gpu_benchmark.py..."
python3 gpu_benchmark.py 2>&1
echo "  Benchmark complete."
REMOTE_EOF

# 6. Retrieve results
echo ""
echo "[5/5] Retrieving results..."
mkdir -p "${LOCAL_RESULTS}"
scp ${SSH_OPTS} \
    "${LAMBDA_HOST}:${REMOTE_RESULTS}/results.jsonl" \
    "${LOCAL_RESULTS}/lambda_gpu_results.jsonl" \
    || echo "  Warning: could not retrieve results.jsonl (may not exist yet)"

echo ""
echo "Deployment complete."
echo "Results saved to: ${LOCAL_RESULTS}/lambda_gpu_results.jsonl"
echo ""

# 7. Quick summary if jq is available
if command -v jq &>/dev/null && [[ -f "${LOCAL_RESULTS}/lambda_gpu_results.jsonl" ]]; then
    echo "Quick results summary:"
    jq -r '"  [" + .kernel_name + "] speedup=" + (.speedup|tostring) + "x  passed=" + (.passed_criteria|tostring)' \
       "${LOCAL_RESULTS}/lambda_gpu_results.jsonl" 2>/dev/null || true
fi
