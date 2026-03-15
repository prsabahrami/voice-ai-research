# Experiment Notes — Pure RL Math Reasoning (DeepSeek-R1-Zero Reproduction)

---

## HEADLINE RESULT
**93.0% on MATH level 4-5** (competition math) with ZERO scaffolding — no system prompt, no few-shot, no CoT instructions. Reasoning (`<think>` blocks, self-correction, `\boxed{}` format) emerged purely from binary reward signal. 2% unsolved rate on eval set.

## WHAT WORKED (ranked by impact)

1. **MAX_TOKENS is THE lever** — 512→1024 (+4.8%), 1024→2048 (+5.0%), 2048→4096 (+2.4%), 4096→8192 (+1.75%). Diminishing returns, approaching asymptote.
2. **Pure RL emergent reasoning** — `<think>` blocks, self-correction, `\boxed{}` ALL emerged from binary reward. No scaffolding needed.
3. **Curriculum escalation** — GSM8K → MATH 1-3 → MATH 2-5. Each step improved eval.
4. **PPO > all other losses** — PPO is fast, stable, best quality. IS slower. CISPO same quality but 3x slower. DRO catastrophic.
5. **Early stopping (50 batches)** — Matches 100 at HALF compute. Late training overfits on 180 prompts.
6. **GROUP_SIZE scales with MAX_TOKENS** — 32 best at 1024, 16 fine at 2048, 8 sufficient at 4096. Smaller groups OK when model is strong.

## WHAT DID NOT WORK

1. **DRO loss** — CATASTROPHIC (6.1% eval). Never use for math RL.
2. **N_BATCHES 200** — Overfits on limited data.
3. **LoRA rank 64** — More params hurt optimization without more data.
4. **LR changes** — 2e-5 too slow, 6e-5 too fast. 4e-5 is optimal.
5. **Mixed easy+hard data** — Easy data dilutes hard signal.
6. **MoE models** — 3B active << 8B dense for math.
7. **Qwen3.5 models** — Not available on Tinker (400 errors).
8. **500 harder MATH prompts** — More data didn't help when GROUP_SIZE was too small (G=4: 96% skip rate). With G=8 at 4096tok, got 0.9275 (close but below 0.93).
9. **SFT→RL transfer** — Starting from SFT checkpoint got 0.84 eval, worse than pure RL (0.93).

## PROGRESS

| Exp | Lab ID | Change | eval_reward_mean | Verdict |
|-----|--------|--------|-----------------|---------|
| 1 | e9 | GSM8K baseline | 0.9163 | KEEP |
| 2 | e30 | MATH 1-3 | 0.7075 | KEEP |
| 3 | e43 | MATH 2-5 | 0.7675 | KEEP |
| 4 | e59 | MAX_TOKENS 1024 | 0.8150 | KEEP |
| 5 | e77 | PPO loss | 0.8238 | KEEP |
| 7 | e113 | GROUP_SIZE 32 | 0.8313 | KEEP |
| 18 | e237 | N_BATCHES 50 | 0.8281 | KEEP |
| 20 | e265 | MAX_TOKENS 2048 | 0.8806 | KEEP |
| 21 | e288 | GROUP_SIZE 16 | 0.8850 | KEEP |
| 22 | e304 | MAX_TOKENS 4096 | 0.9050 | KEEP |
| 23 | e348 | N_BATCHES 75 | 0.9125 | KEEP |
| **24** | **e390** | **MAX_TOKENS 8192** | **0.9300** | **BEST** |
| 25 | e405 | 500 hard prompts G=4 8192tok | 0.9100 | DISCARD |
| 26 | e411 | 500 hard prompts G=8 4096tok | 0.9275 | DISCARD |
| 27 | e416 | G=16 4096tok | 0.9050 | DISCARD |
| 28 | e420 | G=16 8192tok (stopped early) | — | DISCARD |

## CURRENT BEST CONFIG
Model: Qwen/Qwen3-8B, PPO, LR=4e-5, **MAX_TOKENS=8192**, GROUP_SIZE=8, BATCH_SIZE=128, N_BATCHES=50, LoRA rank=32. Pure binary reward, zero scaffolding.

## GROUP_SIZE FINDINGS
- G=4 at 8192tok: best eval (0.93) but 95% skip rate, very few datums
- G=8 at 4096tok: good balance, 0.905 eval
- G=16 at 4096tok: best training signal (976 datums/batch, 0% unsolved) but eval limited by 4096tok
- G=16 at 8192tok: too compute-heavy (~6min/batch), stopped early

## LOSS FUNCTION RANKING
1. **PPO** — fast, stable, best
2. **CISPO** — same quality, 3x slower
3. **importance_sampling** — works but slower convergence
4. **DRO** — CATASTROPHIC, never use

## MAX_TOKENS SCALING LAW
| MAX_TOKENS | eval_reward_mean | Delta |
|------------|-----------------|-------|
| 512 | ~0.77 | — |
| 1024 | 0.815 | +4.8% |
| 2048 | 0.881 | +5.0% |
| 4096 | 0.905 | +2.4% |
| 8192 | 0.930 | +1.75% |

Diminishing returns but still the highest-impact lever. Approaching asymptote.
