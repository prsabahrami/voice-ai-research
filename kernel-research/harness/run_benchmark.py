#!/usr/bin/env python3
"""
CLI entry point for running benchmarks.
Writes results to /home/ubuntu/kernel-research/results/results.jsonl

Usage:
    python run_benchmark.py --hypothesis_id h001 --kernel_type attention --batch_size 8
    python run_benchmark.py --list  # list available hypothesis implementations
"""

import argparse
import json
import sys
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from harness import benchmark_kernel, print_result
from baseline_kernels import (
    baseline_attention,
    baseline_gemm,
    make_attention_args,
    make_gemm_args,
)


# Registry: hypothesis_id -> implementation module
HYPOTHESIS_REGISTRY = {}


def register_hypothesis(hypothesis_id, fn, kernel_type, method, make_args_fn):
    HYPOTHESIS_REGISTRY[hypothesis_id] = {
        "fn": fn,
        "kernel_type": kernel_type,
        "method": method,
        "make_args_fn": make_args_fn,
    }


def load_hypothesis_implementations():
    """Load all hypothesis implementations from hypotheses/ directory."""
    hyp_dir = Path(__file__).parent.parent / "hypotheses"
    for py_file in hyp_dir.glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                mod.register(register_hypothesis)
        except Exception as e:
            print(f"Warning: could not load {py_file}: {e}")


def run_hypothesis(hypothesis_id: str, kernel_type: str, batch_size: int, agent: str = "unknown"):
    """Run benchmark for a specific hypothesis."""
    load_hypothesis_implementations()

    if hypothesis_id not in HYPOTHESIS_REGISTRY:
        print(f"Error: hypothesis_id {hypothesis_id!r} not found in registry.")
        print(f"Available: {list(HYPOTHESIS_REGISTRY.keys())}")
        sys.exit(1)

    entry = HYPOTHESIS_REGISTRY[hypothesis_id]
    fn_opt = entry["fn"]
    method = entry["method"]
    make_args_fn = entry["make_args_fn"]

    # Determine baseline
    if kernel_type == "attention":
        fn_base = baseline_attention
    elif kernel_type == "GEMM":
        fn_base = baseline_gemm
    else:
        fn_base = baseline_gemm  # default

    # Default args for smoke test
    if kernel_type == "attention":
        args = make_attention_args(batch_size=batch_size)
    else:
        args = make_gemm_args(batch_size=batch_size)

    if make_args_fn is not None:
        args = make_args_fn(batch_size)

    result = benchmark_kernel(
        fn_optimized=fn_opt,
        fn_baseline=fn_base,
        args=args,
        hypothesis_id=hypothesis_id,
        kernel_type=kernel_type,
        method=method,
        agent=agent,
    )

    print_result(result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Run kernel optimization benchmark")
    parser.add_argument("--hypothesis_id", "-H", type=str, help="Hypothesis ID (e.g. h001)")
    parser.add_argument("--kernel_type", "-k", type=str, default="attention",
                        choices=["attention", "GEMM", "fused-ops", "quant"],
                        help="Kernel type")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--agent", "-a", type=str, default="unknown", help="Agent name")
    parser.add_argument("--list", action="store_true", help="List available hypotheses")

    args = parser.parse_args()

    if args.list:
        load_hypothesis_implementations()
        print("Available hypotheses:")
        for hid, entry in HYPOTHESIS_REGISTRY.items():
            print(f"  {hid}: [{entry['kernel_type']}] {entry['method']}")
        sys.exit(0)

    if not args.hypothesis_id:
        parser.print_help()
        sys.exit(1)

    run_hypothesis(args.hypothesis_id, args.kernel_type, args.batch_size, args.agent)


if __name__ == "__main__":
    main()
