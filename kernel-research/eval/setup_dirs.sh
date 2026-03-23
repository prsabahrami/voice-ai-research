#!/usr/bin/env bash
# setup_kernel_research_dirs.sh -- Create /home/ubuntu/kernel-research/ structure
# Run directly on Lambda as ubuntu user
set -euo pipefail
mkdir -p /home/ubuntu/kernel-research/hypotheses
mkdir -p /home/ubuntu/kernel-research/harness/quant
mkdir -p /home/ubuntu/kernel-research/results
mkdir -p /home/ubuntu/kernel-research/synthesis
mkdir -p /home/ubuntu/kernel-research/eval
echo "Directory structure created:"
find /home/ubuntu/kernel-research -type d | sort
