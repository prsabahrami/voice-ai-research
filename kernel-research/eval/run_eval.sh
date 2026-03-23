#!/usr/bin/env bash
# run_eval.sh -- Wrapper to run the kernel optimization result validator
# Usage: ./run_eval.sh [--results-path PATH]
#
# Runs validate_results.py and prints a summary of PASS/FAIL verdicts.
# Default results path: /home/ubuntu/kernel-research/results/results.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_PATH="${1:-/home/ubuntu/kernel-research/results/results.jsonl}"
VALIDATOR="${SCRIPT_DIR}/validate_results.py"

echo "========================================"
echo "  Kernel Optimization Eval Harness"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================"
echo ""
echo "Results file: ${RESULTS_PATH}"
echo "Validator:    ${VALIDATOR}"
echo ""

if [ ! -f "${VALIDATOR}" ]; then
    echo "ERROR: Validator script not found at ${VALIDATOR}"
    exit 1
fi

# Optionally override results path via env var
if [ -n "${KERNEL_RESULTS_PATH:-}" ]; then
    RESULTS_PATH="${KERNEL_RESULTS_PATH}"
fi

# Run the validator
RESULTS_PATH="${RESULTS_PATH}" python3 "${VALIDATOR}"
EXIT_CODE=$?

echo ""
echo "========================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  Eval completed. Check output above."
elif [ ${EXIT_CODE} -eq 2 ]; then
    echo "  WARNING: No entries passed all criteria."
else
    echo "  Eval completed with exit code ${EXIT_CODE}."
fi
echo "========================================"

exit ${EXIT_CODE}
