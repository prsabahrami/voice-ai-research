#!/usr/bin/env python3
"""
validate_results.py -- Kernel optimization eval result validator.

Reads /home/ubuntu/kernel-research/results/results.jsonl and enforces:
  - speedup >= 1.10 (10%+ improvement)
  - p99/p50 ratio < 1.05 (variance under 5%)
  - correctness_max_abs_err < 1e-4
  - batch_sizes_tested includes [1, 8, 32]

Outputs PASS/FAIL per entry, writes validated_results.jsonl, prints summary table.
"""

import json
import sys
import os
from datetime import datetime, timezone

RESULTS_PATH = os.environ.get(
    "RESULTS_PATH",
    "/home/ubuntu/kernel-research/results/results.jsonl"
)
VALIDATED_PATH = os.environ.get(
    "VALIDATED_PATH",
    "/home/ubuntu/kernel-research/results/validated_results.jsonl"
)

REQUIRED_BATCH_SIZES = {1, 8, 32}
MIN_SPEEDUP = 1.10
MAX_VARIANCE_RATIO = 1.05
MAX_ABS_ERR = 1e-4


def check_entry(entry):
    """Return (passed, reasons) for a single result entry."""
    reasons = []
    passed = True

    # Check speedup
    speedup = entry.get("speedup")
    if speedup is None:
        reasons.append("MISSING speedup field")
        passed = False
    elif speedup < MIN_SPEEDUP:
        reasons.append(
            f"speedup {speedup:.4f} < {MIN_SPEEDUP} (need >=10% improvement)"
        )
        passed = False
    else:
        reasons.append(f"speedup {speedup:.4f} OK")

    # Check p99/p50 variance ratio
    p99 = entry.get("p99_latency_ms")
    p50 = entry.get("p50_latency_ms")
    if p99 is None or p50 is None:
        # Accept flat variance_ratio or p99_p50_ratio field
        variance_ratio = entry.get("variance_ratio") or entry.get("p99_p50_ratio")
        if variance_ratio is None:
            reasons.append(
                "MISSING p99_latency_ms / p50_latency_ms (or variance_ratio)"
            )
            passed = False
        elif variance_ratio >= MAX_VARIANCE_RATIO:
            reasons.append(
                f"variance_ratio {variance_ratio:.4f} >= {MAX_VARIANCE_RATIO}"
                f" (need <5% variance)"
            )
            passed = False
        else:
            reasons.append(f"variance_ratio {variance_ratio:.4f} OK")
    else:
        if p50 == 0:
            reasons.append("p50_latency_ms is 0 (division by zero)")
            passed = False
        else:
            ratio = p99 / p50
            if ratio >= MAX_VARIANCE_RATIO:
                reasons.append(
                    f"p99/p50={ratio:.4f} >= {MAX_VARIANCE_RATIO}"
                    f" (need <5% variance)"
                )
                passed = False
            else:
                reasons.append(f"p99/p50={ratio:.4f} OK")

    # Check correctness
    max_abs_err = entry.get("correctness_max_abs_err")
    if max_abs_err is None:
        reasons.append("MISSING correctness_max_abs_err field")
        passed = False
    elif max_abs_err >= MAX_ABS_ERR:
        reasons.append(
            f"correctness_max_abs_err {max_abs_err:.2e} >= {MAX_ABS_ERR:.1e}"
        )
        passed = False
    else:
        reasons.append(f"correctness_max_abs_err {max_abs_err:.2e} OK")

    # Check batch sizes
    batch_sizes_tested = entry.get("batch_sizes_tested")
    if batch_sizes_tested is None:
        reasons.append("MISSING batch_sizes_tested field")
        passed = False
    else:
        tested_set = set(batch_sizes_tested)
        missing = REQUIRED_BATCH_SIZES - tested_set
        if missing:
            reasons.append(
                f"batch_sizes_tested missing {sorted(missing)}"
                f" (need {sorted(REQUIRED_BATCH_SIZES)})"
            )
            passed = False
        else:
            reasons.append(
                f"batch_sizes_tested {sorted(tested_set)}"
                f" covers required {sorted(REQUIRED_BATCH_SIZES)} OK"
            )

    return passed, reasons


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results file not found: {RESULTS_PATH}")
        print(
            "Create results.jsonl with at least one entry before running validation."
        )
        sys.exit(1)

    entries = []
    with open(RESULTS_PATH, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry["_lineno"] = lineno
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"WARNING: Line {lineno} is not valid JSON: {e}")

    if not entries:
        print("No valid entries found in results.jsonl")
        sys.exit(1)

    passing = []
    failing = []
    rows = []

    for entry in entries:
        passed, reasons = check_entry(entry)
        name = (
            entry.get("kernel_name")
            or entry.get("name")
            or f"entry_line_{entry['_lineno']}"
        )
        hypothesis = (
            entry.get("hypothesis_id")
            or entry.get("hypothesis")
            or "unknown"
        )
        verdict = "PASS" if passed else "FAIL"
        rows.append(
            {
                "name": name,
                "hypothesis": hypothesis,
                "verdict": verdict,
                "reasons": reasons,
            }
        )
        if passed:
            passing.append(entry)
        else:
            failing.append(entry)

    # Print summary table
    sep = "-" * 100
    print(sep)
    print(
        f"{'KERNEL / ENTRY':<40} {'HYPOTHESIS':<20} {'VERDICT':<8} DETAILS"
    )
    print(sep)
    for row in rows:
        verdict_str = row["verdict"]
        detail_lines = row["reasons"]
        first = detail_lines[0] if detail_lines else ""
        print(
            f"{row['name']:<40} {row['hypothesis']:<20} {verdict_str:<8} {first}"
        )
        for detail in detail_lines[1:]:
            print(f"{'':70} {detail}")

    print(sep)
    print(
        f"TOTAL: {len(entries)} entries | PASS: {len(passing)} | FAIL: {len(failing)}"
    )
    print(sep)

    # Write validated results
    with open(VALIDATED_PATH, "w") as f:
        for entry in passing:
            clean = {k: v for k, v in entry.items() if k != "_lineno"}
            f.write(json.dumps(clean) + "\n")

    ts = datetime.now(timezone.utc).isoformat()
    print(f"\nValidated results written to: {VALIDATED_PATH}")
    print(f"Timestamp: {ts}")

    if len(passing) == 0 and len(entries) > 0:
        print("\nWARNING: No entries passed all criteria.")
        sys.exit(2)
    elif len(failing) > 0:
        print(f"\nNOTE: {len(failing)} entries failed. See details above.")
        sys.exit(0)
    else:
        print("\nAll entries passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
