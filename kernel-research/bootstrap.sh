#!/bin/bash
# Bootstrap script for Lambda kernel-research environment
# Run once to set up the full environment on Lambda
# Usage: bash bootstrap.sh

set -e

RESEARCH_DIR="/home/ubuntu/kernel-research"
REPO_URL="https://github.com/prsabahrami/voice-ai-research"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"  # set from env or .env

echo "=== Kernel Research Bootstrap ==="
echo "Target: $RESEARCH_DIR"

# 1. Create directory structure
mkdir -p "$RESEARCH_DIR"/{hypotheses,harness/quant,results,synthesis,eval}

# 2. Clone or pull the repo to get the kernel-research code
if [ -d "/home/ubuntu/voice-ai-research" ]; then
    cd /home/ubuntu/voice-ai-research
    git pull origin main
else
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone "https://${GITHUB_TOKEN}@github.com/prsabahrami/voice-ai-research" /home/ubuntu/voice-ai-research
    else
        git clone "$REPO_URL" /home/ubuntu/voice-ai-research
    fi
fi

# 3. Copy kernel-research code to research dir
cp -r /home/ubuntu/voice-ai-research/kernel-research/* "$RESEARCH_DIR/"

# 4. Install Python dependencies
pip install torch --quiet 2>/dev/null || true
pip install triton --quiet 2>/dev/null || true
pip install numpy --quiet 2>/dev/null || true

# 5. Create results.jsonl if not exists
touch "$RESEARCH_DIR/results/results.jsonl"
touch "$RESEARCH_DIR/results/certified.jsonl"

# 6. Set up tmux sessions for each track
echo "Setting up persistent tmux sessions..."

# Synthesis monitor
tmux new-session -d -s kernel-synthesis -c "$RESEARCH_DIR" 2>/dev/null || true
tmux send-keys -t kernel-synthesis "python synthesis/synthesis_monitor.py 2>&1 | tee synthesis/monitor.log" Enter

# Research harness (coolstufs)
tmux new-session -d -s kernel-research-harness -c "$RESEARCH_DIR" 2>/dev/null || true

# Quantization track (miniQuant)
tmux new-session -d -s kernel-research-quant -c "$RESEARCH_DIR" 2>/dev/null || true

# Hypothesis track (ooo)
tmux new-session -d -s kernel-research-hypotheses -c "$RESEARCH_DIR" 2>/dev/null || true

# Eval harness (serious-inference-engineer)
tmux new-session -d -s kernel-research-eval -c "$RESEARCH_DIR" 2>/dev/null || true

echo ""
echo "=== Bootstrap Complete ==="
echo "Directory: $RESEARCH_DIR"
echo "Tmux sessions: kernel-synthesis, kernel-research-harness, kernel-research-quant, kernel-research-hypotheses, kernel-research-eval"
echo ""
echo "Quick start:"
echo "  # Test the harness"
echo "  cd $RESEARCH_DIR && python harness/baseline_kernels.py"
echo ""
echo "  # Run a benchmark"
echo "  cd $RESEARCH_DIR && python harness/run_benchmark.py --hypothesis_id h001 --kernel_type attention --batch_size 8"
echo ""
echo "  # Validate numerics"
echo "  cd $RESEARCH_DIR && python harness/quant/numerics_validator.py --kernel all --batch_size 8"
echo ""
echo "  # Run eval harness"
echo "  cd $RESEARCH_DIR && python eval/eval_harness.py --run_all"

tmux list-sessions
