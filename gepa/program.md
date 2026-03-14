# gepa

Autonomous prompt optimization using [GEPA](https://github.com/gepa-ai/gepa) (Genetic-Pareto). GEPA evolves text artifacts (prompts, rubrics, agent architectures) using LLM reflection and Pareto-efficient search. It outperforms GRPO by 6% on average while using 35x fewer rollouts.

## 1. Task Description (USER: FILL THIS IN)

**What to optimize**: [e.g., "system prompt for math reasoning", "evaluation rubric for code quality", "agent architecture for tool use"]

**Task LM**: [e.g., `openai/gpt-4.1-mini`, `anthropic/claude-sonnet-4-20250514`]

**Evaluation data**: [describe your train/val data and what correct answers look like]

> Replace this section with your optimization target. Examples:
> - "Optimize a system prompt for GPT-4.1-mini to solve AIME competition math problems"
> - "Evolve an evaluation rubric that discriminates good vs bad code reviews"
> - "Optimize a multi-step agent prompt for tool-use tasks"

## 2. Setup

1. **Create a worktree for your experiment** — NEVER work on main/master directly:
   ```bash
   REPO=$(git rev-parse --show-toplevel)
   git worktree add "$REPO/../<experiment-name>" -b experiment/<experiment-name>
   cd "$REPO/../<experiment-name>/gepa"
   ```
   Each worktree is an isolated copy — multiple experiments can run in parallel in separate terminals. Main stays clean as the starter template.
2. **Read the files**:
   - `optimize.py` — the script you modify (models, data, seed prompt, budget)
3. **Set API key**: `export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`)
4. **Install GEPA**: `pip install gepa`

## 3. The Loop

Each iteration of this loop is ONE experiment. You run this loop **continuously and indefinitely** until the human manually stops you. Do not stop after 1, 2, or 5 experiments. The human may be asleep or away. You are expected to run dozens of experiments autonomously.

```
LOOP FOREVER:
  1. Reconstruct state: read results.tsv + notes.md (+ optionally ../lab context)
  2. Decide what to try next. State WHY it will improve val_score.
     Priority: evaluation data > seed prompt > model choice > GEPA config
  3. Modify optimize.py → git commit
  4. Run: python optimize.py > run.log 2>&1
  5. Extract: grep '^val_score:' run.log
  6. Read the best_prompt in run.log (verify it makes sense, not degenerate)
  7. Append results to results.tsv. Update notes.md with what you learned.
  8. If val_score improved → KEEP commit. If not → git reset --hard HEAD~1.
  9. IMMEDIATELY start the next iteration. Do not pause, summarize, or ask.
```

Optionally use the `../lab` CLI for structured tracking (hypothesis, experiment, result commands), but do NOT let it slow you down. The core loop is: edit → commit → run → grep → log → keep/revert → repeat.

**Logging results:** Append to `results.tsv` (tab-separated). Columns: `commit	val_score	status	description`. Example:
```
a1b2c3d	0.850000	keep	better seed prompt for math
b2c3d4e	0.720000	discard	switched to gpt-4.1-nano (too weak)
```

### Best Practices (from 550+ experiments)

- **One change at a time per experiment.** So you know what caused the effect.
- **Data quality >> model choice >> GEPA config.** Better evaluation data improves results more than switching models.
- **Start with a small budget (30), increase once stable.** Avoids wasting API calls on broken configs.

## 4. What You Can Modify

**optimize.py** — everything is fair game:
- `SEED` — the starting prompt/artifact to optimize
- `TRAINSET` / `VALSET` — evaluation data (more diverse = better GEPA performance)
- `TASK_LM` — which model is being prompted
- `REFLECTION_LM` — which model proposes improvements (stronger = better)
- `MAX_METRIC_CALLS` — budget per run (start at 30, increase to 100-500 once stable)
- GEPA config: `candidate_selection_strategy`, `use_merge`, `reflection_minibatch_size`, `frontier_type`

**What you CANNOT modify:**
- `program.md` — read-only
- GEPA library internals

## 5. GEPA Configuration Guide

Key parameters to experiment with:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `max_metric_calls` | 30 | Start small. 100-500 for serious runs. |
| `candidate_selection_strategy` | `'pareto'` | `'pareto'` (diverse) vs `'current_best'` (greedy) |
| `frontier_type` | `'instance'` | `'instance'` (per-example), `'objective'`, `'hybrid'` |
| `use_merge` | `False` | Combine complementary candidates. Try `True` for diverse tasks. |
| `reflection_minibatch_size` | `None` (auto) | Examples per reflection step. Smaller = faster, larger = better proposals. |
| `skip_perfect_score` | `True` | Skip examples already solved perfectly. |

## 6. Ideas for Advanced Use

- **Dynamic rubrics**: Use GEPA to optimize evaluation rubrics (inspired by DR-TULU's RLER). The rubric IS the text artifact being optimized.
- **Agent architecture**: Optimize multi-step agent prompts with tool-use instructions.
- **Cascade optimization**: Optimize prompt for cheap model to match expensive model quality.
- **Custom evaluators**: Return `{"score": float, "feedback": str}` from your metric for richer GEPA reflection.

---

## NEVER STOP

Once the experiment loop has begun (after setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". Do NOT summarize progress and wait for input. The human may be asleep or away from the computer and expects you to continue working **indefinitely** until you are manually stopped.

You are autonomous. After each experiment, **immediately** start the next one. If you run out of ideas, think harder — try different seed prompts, better evaluation data, different GEPA parameters, search for papers on prompt optimization. The loop runs until the human interrupts you, period.

As a reference: each GEPA run takes roughly 2-10 minutes. You should be able to run **6-20 experiments per hour**. If the human is asleep for 8 hours, that's 50-160 experiments. The human wakes up to a full results.tsv and optimized artifacts. That is the goal.
