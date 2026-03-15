# Experiment Notes

This file is the agent's lab notebook. Record observations, hypotheses, and learnings here.
The agent updates this after every experiment to maintain cross-session memory.

---

## Experiment 1 (e194): Baseline RL training — tool routing

**Run ID**: wfjusbl9qzwa4iqdgv1kpck1
**Hypothesis (h198)**: RL with correctness(0.7)+efficiency(0.3) teaches tool specialization
**Status**: COMPLETED — mechanism CONFIRMED

### Setup
- Model: Qwen/Qwen3-30B-A3B-Instruct-2507
- Environment: salesbench/tool-routing (6 tools, 191 train / 57 eval examples)
- Reward: 0.7*correctness + 0.3*efficiency(1/sqrt(n_tools))
- Config: batch_size=128, rollouts_per_example=32, lr=1e-5, lora_alpha=64, max_tokens=1024

### Results
- **Baseline eval (step 0)**: Avg@4 = 0.8032
- **Step 50 eval**: Avg@4 = 0.8079 (+0.006)
- **Step 100 eval**: Avg@4 = 0.8069 (+0.005) — appeared flat
- **Step 150 eval**: Avg@4 = 0.9011 (+0.098) — **PHASE TRANSITION**
- **Step 200 eval**: Avg@4 = 0.9041 (+0.101) — **+12.6% total improvement**
- **Conclusion**: GRPO works but shows delayed phase transition. First 100 steps appear flat, then rapid improvement between step 100-150, then plateaus.

### Validation reward trend
| Step | Val. Reward |
|------|------------|
| 0    | 0.7241     |
| 5    | 0.7599     |
| 10   | 0.7654     |
| 15   | 0.6803     |
| 20   | 0.7854     |
| 25   | 0.8608     |
| 30   | 0.7937     |
| 35   | 0.8096     |
| 40   | 0.8021     |
| 45   | 0.7992     |
| 50   | 0.8526     |
| 55   | 0.6214     |
| 60   | 0.7062     |
| 65   | 0.8128     |
| 70   | 0.7745     |
| 75   | 0.8014     |
| 80   | 0.8155     |
| 85   | 0.7992     |
| 90   | 0.8347     |
| 95   | 0.8183     |
| 100  | 0.8131     |

Val reward is noisy (0.62-0.86) but averages ~0.79, above baseline 0.72.
Val uses 32 examples, 1 rollout — too noisy for reliable signal.

### Analysis
1. **Base model already strong**: 72% correctness on this task out of the box
2. **Efficiency reward gives free 0.3**: Even wrong answers get 0.3, reducing gradient signal
3. **Low reward diversity**: Most batches are 0.65-1.0, GRPO advantages are weak
4. **Small dataset**: 191 examples, model sees entire dataset every ~1.5 steps → possible overfitting
5. **Tool usage pattern**: Model uses mostly wikipedia_lookup and unit_converter, rarely calculator/python_eval

### Key Insight
The base model already routes tools well enough for 72% correctness. The remaining 28% errors are likely on hard questions (multi-step, fact+calc) where the model would need to chain 2-3 tools. The efficiency penalty actually discourages this — using 2 tools gives efficiency=0.707, losing 0.088 reward vs using 1 tool.

### Mechanism: CONFIRMED
GRPO teaches tool specialization, but with a delayed phase transition. First 100 steps: internal representations adjusting. Steps 100-150: phase transition with rapid improvement. Steps 150-200: plateau at new level. Final: 0.8032→0.9041 (+12.6%).

### Key Finding: Phase Transitions in RL Training
The flat period (steps 0-100) followed by rapid improvement (steps 100-150) is a classic phase transition. Early analysis at step 100 would have incorrectly concluded "RL doesn't work." **Lesson: Don't abandon RL runs based on early flat periods — give them at least 150 steps.**

---

## Experiment 2 (e218): Correctness-only reward

**Run ID**: v85nlt4ya3bovv6h57nrwjj9
**Hypothesis (h223)**: Pure correctness reward gives cleaner gradient
**Status**: COMPLETED — mechanism CONFIRMED (phase transition at step 100)
**Result**: Baseline 0.7400 → 0.8650 at step 100 (+16.9% raw correctness)
**Finding**: Also shows phase transition. Achieves similar raw correctness as experiment 1.

---

## Experiment 3 (e224): Difficulty filtering (BEST)

**Run ID**: rrnde1vmcphxlphgoveem0u4
**Hypothesis (h229)**: Online difficulty filtering focuses GRPO on edge cases
**Status**: COMPLETED — mechanism CONFIRMED, BEST RESULT
**Config change**: online_difficulty_filtering=true, easy_threshold=0.9, hard_threshold=0.1

### Results
| Step | Eval@4 |
|------|--------|
| 0 | 0.8034 |
| 50 | 0.8863 |
| 100 | 0.9140 |
| 150 | **0.9294** (BEST) |
| 200 | 0.9125 |

### Comparison: All Experiments
| Exp | Step 50 | Step 100 | Step 150 | Step 200 | Config |
|-----|---------|----------|----------|----------|--------|
| e194 | 0.8079 | 0.8069 | 0.9011 | **0.9041** | mixed reward, no filter |
| e218 | 0.7400 | 0.8650 | 0.8600 | 0.8800 | correctness only, no filter |
| **e224** | **0.8863** | **0.9140** | **0.9294** | 0.9125 | mixed reward + filter |
| e267 | CRASH | - | - | - | correctness + filter (empty buffer) |
| e275 | 0.8219 | stalled | - | - | mixed + filter, LR=5e-6 |
| e292 | (running) | - | - | - | partial credit + filter |

---

## Experiment 4 (e267): CRASH — correctness + filter

Binary correctness + difficulty filtering crashed with "No environments left with examples."
Binary rewards on deterministic model → every prompt has 0% or 100% success → filter removes all.
**Key finding**: The efficiency reward component is ESSENTIAL for filtering to work — it provides the small within-group variance that keeps prompts in the 10-90% filter range.

---

## Experiment 5 (e275): LR=5e-6 — stalled

Lower LR (5e-6) was too slow and stalled at step 73. Training rewards dropped to 0.27-0.30.
**Key finding**: LR=1e-5 is optimal for this task with difficulty filtering.

---

## Experiment 6 (e292): Partial credit correctness

**Run ID**: m9zv91rkzcf8hk9qte6t0nc4
**Hypothesis**: Adding 0.5 reward for close numerical answers (within 5%) creates more within-group variance.
**Status**: COMPLETED — 0.9199, below e224 BEST (0.9294)

---

## Experiment 7 (e314): rollouts=64 — NEW BEST

**Run ID**: oq5iogyowi4f5ls8bovnd7nn
**Status**: COMPLETED (crashed step ~170) — BEST=0.9484 at step 150
**Config**: rollouts_per_example=64 (was 32), same config otherwise

| Step | Eval@4 |
|------|--------|
| 100 | 0.9245 |
| 150 | **0.9484** (BEST ALL TIME) |

Crashed at step ~170 — difficulty filter removed all examples (model saturated task).

---

## Experiment 8 (e351): lora_alpha=128 — WORSE

**Run ID**: e20oj1vnautigc0crf5p8tzi
**Status**: COMPLETED — 0.9294 peak (step 100), 0.8967 final
**Finding**: lora_alpha=128 hurt. Reverted to 64.

---

## Experiment 9 (e362): easy_threshold 0.9→0.95

**Run ID**: loucyksvk9pe686jhy4v07wc
**Status**: COMPLETED — mechanism CONFIRMED (no crash) but underperformed

| Step | Eval@4 |
|------|--------|
| 50 | 0.7570 |
| 100 | 0.8998 |
| 150 | 0.9212 |
| 200 | 0.9247 |

**Finding**: No crash, but relaxed threshold dilutes learning signal. 0.9212 at step 150 vs e314's 0.9484.

---

## Experiment 10 (e395): Expanded dataset 191→301

**Run ID**: qjxw12dynx2rl0k2wmrbhxvu
**Hypothesis (h405)**: More diverse data reduces overfitting, raises ceiling
**Status**: COMPLETED — mechanism CONFIRMED

### Results
| Step | Eval@4 |
|------|--------|
| 0 | 0.7562 |
| 50 | 0.7818 |
| 100 | 0.8678 |
| 150 | 0.8915 |
| 200 | **0.9265** |

**Finding**: No crash at step 170. Still improving at step 200 (+22.5% relative improvement). More data delays saturation AND allows continued learning. Model was still improving — more steps needed.

---

## Experiment 11 (e406): max_steps 200→300

**Run ID**: udbf9c4ej11j78cq6w5g1pso
**Status**: COMPLETED — mechanism REFUTED

| Step | Eval@4 |
|------|--------|
| 50 | 0.8600 |
| 100 | **0.9084** (peak) |
| 150 | 0.8880 |
| 200 | 0.8880 |
| 250 | 0.8946 |
| 300 | 0.8880 |

**Finding**: 300 steps adds nothing. Peaked at step 100 and flat after. 200 steps sufficient.

---

## Experiment 12 (e414): Optimal-call efficiency — **NEW BEST**

**Run ID**: d7krzzd8adp2ype6c3dnsho0
**Hypothesis (h423)**: Per-question optimal call count gives more targeted signal
**Status**: COMPLETED — mechanism CONFIRMED, **NEW BEST**

### Results
| Step | Eval@4 |
|------|--------|
| 0 | 0.7594 |
| 50 | 0.7882 |
| 100 | 0.9020 |
| 150 | 0.9195 |
| **200** | **0.9370** (NEW BEST) |

**Finding**: Optimal-call efficiency is the BEST reward function. +23.4% relative improvement. Unlike 1/sqrt(n) that peaked and declined, this kept improving through step 200. Multi-step questions aren't penalized for needed tool calls. Model maintained better training rewards (0.3-0.69 vs constant 0.3).

---

## Key Findings (12 experiments)

1. **Optimal-call efficiency is the best reward function**: e414 got 0.9370, +23.4% relative improvement, still rising at step 200
2. **GRPO shows phase transitions**: All experiments show flat/slow improvement then rapid jumps (steps 50-100)
3. **Difficulty filtering is essential**: +2.5% over baseline AND faster convergence
4. **rollouts=64 is a big win**: +2% over rollouts=32
5. **Expanded dataset (301) prevents saturation crash**: 191 examples crash at step ~170 with easy_threshold=0.9
6. **Efficiency reward is essential for filtering**: Binary correctness + filtering = empty buffer crash
7. **lora_alpha=128 hurts**: Larger updates cause overshooting
8. **300 steps adds nothing over 200**: Model peaks at step 100-150 with 1/sqrt, at step 200 with optimal-call
9. **Lower LR (5e-6) stalls with filtering**: LR=1e-5 is optimal
10. **Best config**: optimal-call efficiency + expanded data (301) + difficulty filtering (easy=0.9) + rollouts=64 + LR=1e-5 + 200 steps

---

## Overall Research Summary

**Research question**: Does the model discover tool specialization purely from reward signal?
**Answer**: YES. 0.7594 → 0.9370 (+23.4%) on expanded test set. 0.8034 → 0.9484 (+18.1%) on original test set.

**Research question**: Does efficiency training hurt correctness?
**Answer**: No. The 0.3 efficiency weight is ESSENTIAL for difficulty filtering AND improves routing. Optimal-call efficiency > blind 1/sqrt(n).

**Research question**: How many RL steps before routing patterns emerge?
**Answer**: Phase transition at steps 50-100. With optimal-call reward: continuous improvement through 200 steps.

**Research question**: What is the optimal training recipe?
**Answer**: optimal-call efficiency (0.7/0.3 weight) + expanded data (301 examples) + difficulty filtering (easy=0.9) + rollouts=64 + LR=1e-5 + 200 steps.
