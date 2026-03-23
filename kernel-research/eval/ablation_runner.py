#!/usr/bin/env python3
"""
Ablation runner - runs all registered hypotheses in sequence and writes results.
Useful for automated batch evaluation.

Usage:
    python eval/ablation_runner.py --all                    # run all hypotheses
    python eval/ablation_runner.py --ids h001 h002 h006    # run specific hypotheses
    python eval/ablation_runner.py --category GEMM         # run by kernel type
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent / "harness"))
sys.path.insert(0, str(Path(__file__).parent.parent / "hypotheses"))

RESULTS_PATH = Path("/home/ubuntu/kernel-research/results/results.jsonl")


def run_ablation(hypothesis_ids=None, kernel_type=None, agent="unknown"):
    """Run benchmarks for specified hypotheses."""
    # Try to import harness
    try:
        from harness import benchmark_kernel, print_result
        from baseline_kernels import baseline_attention, baseline_gemm
        import torch
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PyTorch and harness/ are on the Python path.")
        sys.exit(1)

    # Load hypothesis implementations
    import importlib.util
    registry = {}
    
    hyp_dir = Path(__file__).parent.parent / "hypotheses"
    for py_file in sorted(hyp_dir.glob("*.py")):
        try:
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "register"):
                def make_register(reg):
                    def register_fn(hid, fn, kt, method, make_args):
                        reg[hid] = {"fn": fn, "kernel_type": kt, "method": method, "make_args": make_args}
                    return register_fn
                mod.register(make_register(registry))
        except Exception as e:
            print(f"Warning: could not load {py_file.name}: {e}")

    if not registry:
        print("No hypothesis implementations found.")
        sys.exit(0)

    # Filter
    to_run = {}
    for hid, entry in registry.items():
        if hypothesis_ids and hid not in hypothesis_ids:
            continue
        if kernel_type and entry["kernel_type"] != kernel_type:
            continue
        to_run[hid] = entry

    print(f"\nRunning {len(to_run)} hypotheses...")

    results = []
    for hid, entry in sorted(to_run.items()):
        fn_opt = entry["fn"]
        kt = entry["kernel_type"]
        method = entry["method"]
        make_args = entry["make_args"]
        
        if kt == "attention":
            from baseline_kernels import baseline_attention as fn_base
        else:
            from baseline_kernels import baseline_gemm as fn_base
        
        # Default args for batch=8
        args = make_args(8)
        
        print(f"\n[{hid}] {kt}: {method}")
        try:
            result = benchmark_kernel(
                fn_optimized=fn_opt,
                fn_baseline=fn_base,
                args=args,
                hypothesis_id=hid,
                kernel_type=kt,
                method=method,
                agent=agent,
                make_args_for_batch=make_args,
            )
            print_result(result)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")

    # Summary
    passed = sum(1 for r in results if r.get("passed_criteria"))
    print(f"\n{'='*60}")
    print(f"Ablation complete: {len(results)} hypotheses, {passed} passed criteria")
    print(f"Results written to: {RESULTS_PATH}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run ablation benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all registered hypotheses")
    parser.add_argument("--ids", nargs="+", help="Specific hypothesis IDs to run")
    parser.add_argument("--category", choices=["attention", "GEMM", "fused-ops", "quant", "other"])
    parser.add_argument("--agent", default="unknown")
    args = parser.parse_args()

    if not (args.all or args.ids or args.category):
        parser.print_help()
        sys.exit(1)

    run_ablation(
        hypothesis_ids=set(args.ids) if args.ids else None,
        kernel_type=args.category,
        agent=args.agent,
    )


if __name__ == "__main__":
    main()
