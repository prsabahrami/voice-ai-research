#!/usr/bin/env python3
"""
Synthesis monitor - polls results.jsonl every 10 minutes,
updates SYNTHESIS.md, and posts org chat summary every 30 minutes.

Run in persistent tmux: tmux new-session -d -s kernel-synthesis -c /home/ubuntu/kernel-research
python synthesis/synthesis_monitor.py

This is a long-running process.
"""

import json
import time
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

RESULTS_PATH = Path("/home/ubuntu/kernel-research/results/results.jsonl")
CERTIFIED_PATH = Path("/home/ubuntu/kernel-research/results/certified.jsonl")
SYNTHESIS_PATH = Path("/home/ubuntu/kernel-research/synthesis/SYNTHESIS.md")

POLL_INTERVAL_SECONDS = 600   # 10 minutes
CHAT_INTERVAL_SECONDS = 1800  # 30 minutes


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def generate_synthesis(results: list, certified: list) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    passed = [r for r in results if r.get("passed_criteria")]
    failed = [r for r in results if not r.get("passed_criteria")]

    # Group by kernel type
    by_type = defaultdict(list)
    for r in results:
        by_type[r.get("kernel_type", "unknown")].append(r)

    # Top speedups
    top = sorted(results, key=lambda r: r.get("speedup", 0), reverse=True)[:5]

    lines = [
        f"# Kernel Optimization Synthesis",
        f"",
        f"Last updated: {now}",
        f"",
        f"## Summary",
        f"",
        f"- Total experiments: {len(results)}",
        f"- Passed criteria: {len(passed)} ({100*len(passed)/max(len(results),1):.1f}%)",
        f"- Failed criteria: {len(failed)}",
        f"- Certified results: {len(certified)}",
        f"",
        f"## Success Criteria",
        f"",
        f"- Speedup >= 1.10x",
        f"- Variance ratio (p99/p50) < 1.05",
        f"- Max absolute error < 1e-4 (FP16)",
        f"- Tested on batch sizes [1, 8, 32]",
        f"",
        f"## Top Results by Speedup",
        f"",
    ]

    if top:
        lines.append("| Hypothesis | Kernel | Method | Speedup | Err | Passed |")
        lines.append("|---|---|---|---|---|---|")
        for r in top:
            hid = r.get("hypothesis_id", "?")
            kt = r.get("kernel_type", "?")
            method = r.get("method", "?")[:50]
            sp = r.get("speedup", 0)
            err = r.get("correctness_max_abs_err", 0)
            passed_str = "YES" if r.get("passed_criteria") else "NO"
            lines.append(f"| {hid} | {kt} | {method} | {sp:.3f}x | {err:.2e} | {passed_str} |")
    else:
        lines.append("No results yet.")

    lines.extend([
        f"",
        f"## Results by Kernel Type",
        f"",
    ])

    for kt, kt_results in by_type.items():
        kt_passed = sum(1 for r in kt_results if r.get("passed_criteria"))
        lines.append(f"### {kt}")
        lines.append(f"Experiments: {len(kt_results)} | Passed: {kt_passed}")
        lines.append("")

    lines.extend([
        f"## What Worked",
        f"",
    ])

    for r in sorted(passed, key=lambda r: r.get("speedup", 0), reverse=True)[:10]:
        lines.append(f"- **{r.get('hypothesis_id')}** [{r.get('kernel_type')}]: "
                     f"{r.get('method', 'unknown')} -- {r.get('speedup', 0):.3f}x speedup")

    if not passed:
        lines.append("No passing results yet.")

    lines.extend([
        f"",
        f"## What Failed",
        f"",
    ])

    for r in failed[:10]:
        sp = r.get("speedup", 0)
        err = r.get("correctness_max_abs_err", 0)
        fail_reason = []
        if sp < 1.10:
            fail_reason.append(f"speedup={sp:.3f}x < 1.10x")
        if err >= 1e-4:
            fail_reason.append(f"err={err:.2e} >= 1e-4")
        lines.append(f"- **{r.get('hypothesis_id')}**: {', '.join(fail_reason) or 'unknown'}")

    if not failed:
        lines.append("No failing results yet.")

    lines.extend([
        f"",
        f"## Next Hypotheses to Try",
        f"",
        f"See /home/ubuntu/kernel-research/hypotheses/hypotheses.md for the ranked list.",
        f"",
        f"## Certified Results",
        f"",
    ])

    for r in certified:
        lines.append(f"- **{r.get('hypothesis_id')}**: {r.get('method', '?')} -- "
                     f"{r.get('speedup', 0):.3f}x certified at {r.get('certified_at', '?')[:16]}")

    if not certified:
        lines.append("No certified results yet.")

    return "\n".join(lines) + "\n"


def main():
    SYNTHESIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    last_chat_time = 0

    print(f"Synthesis monitor started at {datetime.now(timezone.utc).isoformat()}")
    print(f"Polling {RESULTS_PATH} every {POLL_INTERVAL_SECONDS}s")
    print(f"Updating {SYNTHESIS_PATH}")

    while True:
        try:
            results = load_jsonl(RESULTS_PATH)
            certified = load_jsonl(CERTIFIED_PATH)

            synthesis = generate_synthesis(results, certified)
            SYNTHESIS_PATH.write_text(synthesis)

            now = time.time()
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Synthesis updated: {len(results)} results, {len(certified)} certified")

            # Post to org chat every 30 minutes
            if now - last_chat_time >= CHAT_INTERVAL_SECONDS:
                passed = [r for r in results if r.get("passed_criteria")]
                top = sorted(results, key=lambda r: r.get("speedup", 0), reverse=True)[:3]
                top_str = "\n".join(
                    f"  - {r.get('hypothesis_id')} [{r.get('kernel_type')}]: {r.get('method','?')[:40]} -- {r.get('speedup',0):.3f}x"
                    for r in top
                ) if top else "  No results yet"

                chat_msg = (
                    f"SYNTHESIS UPDATE ({datetime.now(timezone.utc).strftime('%H:%M UTC')}):\n"
                    f"Total experiments: {len(results)} | Passed: {len(passed)} | Certified: {len(certified)}\n"
                    f"Top speedups:\n{top_str}\n"
                    f"See /home/ubuntu/kernel-research/synthesis/SYNTHESIS.md for full report."
                )
                print(f"[ORG CHAT] {chat_msg}")
                # In production: call org chat API here
                last_chat_time = now

        except Exception as e:
            print(f"Error in synthesis loop: {e}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
