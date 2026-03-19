## SDFT / SDPO / Haiku Distillation on GPT-OSS: Living PR

**Status:** In progress (running)
**Last updated:** 2026-03-19 13:53 UTC

---

### What This PR Does

This is a multi-agent, persistent training experiment PR. It implements and runs:

1. **SDFT** (Style-Distillation Fine-Tuning): SFT of GPT-OSS-20B on conversational transcript data with Haiku-generated responses as targets
2. **SDPO** (Style-Distillation Preference Optimization): DPO-style preference training on GPT-OSS-20B with Haiku-style chosen vs generic rejected
3. **Haiku Distillation (3 methods)**:
   - Method A: SFT on Haiku outputs (direct style imitation)
   - Method B: DPO with Haiku preferences (preference-based learning)
   - Method C: Style-transfer prompting baseline (zero-shot, no training) -- COMPLETE

All experiments target Claude Haiku's conversational style and use the Tinker API for distributed LoRA fine-tuning on GPT-OSS-20B (20B MoE model).

---

### Multi-Agent Coordination

This PR is maintained by multiple agents working in parallel:

| Agent | Responsibilities |
|-------|-----------------|
| `coolstufs` (this branch) | SDFT training + Haiku SFT distillation + PR management |
| `@serious-inference-engineer` | SDPO/DPO training + Haiku DPO distillation |
| `@ooo` | SDFT on larger transcript datasets + Haiku prompting baseline extension |

---

### Repository Structure

```
sdft_sdpo_haiku_distill/
├── scripts/
│   ├── sdft_train_v2.py          # SDFT training (correct Tinker v0.16.1 API)
│   ├── sdpo_train_v2.py          # SDPO/DPO training
│   ├── haiku_distill_sft.py      # Haiku Method A: SFT
│   ├── haiku_distill_dpo.py      # Haiku Method B: DPO
│   └── haiku_distill_prompting.py # Haiku Method C: Prompting baseline
├── configs/
│   ├── sdft_gpt_oss_20b.yaml     # SDFT base config
│   ├── sdft_gpt_oss_20b_large.yaml # SDFT large config
│   ├── sdpo_gpt_oss_20b.yaml     # SDPO config
│   ├── haiku_sft.yaml            # Haiku SFT config
│   └── haiku_dpo.yaml            # Haiku DPO config
├── data/
│   ├── haiku_sft_train_batch1.json         # 20 Haiku training pairs
│   └── haiku_dpo_preference_pairs_batch1.json # 2 DPO preference pairs
├── results/
│   ├── sdft/                     # SDFT metrics (updated live)
│   ├── sdpo/                     # SDPO metrics (updated live)
│   └── haiku_distill/            # Haiku distillation results
│       └── haiku_prompting_baseline_*.json # COMPLETE
└── RESULTS.md                    # Living results document
```

---

### Key Findings So Far

#### SDFT on GPT-OSS-20B (In Progress)
- Model: openai/gpt-oss-20b, LoRA rank 16
- Steps 1-11/200 complete
- Loss trajectory: 2.1529 -> 0.9821 (strong early convergence)
- Full run ~200 steps with checkpoints at every 25 steps

#### Haiku Prompting Baseline (Complete)
- 10 evaluation prompts, 3 response conditions per prompt
- styled vs gold length_ratio: 0.912
- generic vs gold length_ratio: 0.951
- Key insight: Surface metrics insufficient to capture Haiku style quality
- SFT/DPO training is needed to properly distill the style

---

### Tinker API Notes

This PR uses Tinker v0.16.1 (Thinking Machines Lab) for distributed LoRA fine-tuning.

Correct Datum format for cross-entropy loss (v0.16.1):
```python
types.Datum(
    model_input=types.ModelInput.from_ints(tokens=input_tokens),   # tokens[:-1]
    loss_fn_inputs={
        "target_tokens": target_token_list,   # tokens[1:]
        "weights": weight_list,               # 0.0=prompt, 1.0=response
    }
)
training_client.forward_backward(data, "cross_entropy")
training_client.optim_step(types.AdamParams(learning_rate=2e-4))
```

---

### How to Run

```bash
export TINKER_API_KEY=<key>
export ANTHROPIC_API_KEY=<key>
pip install tinker transformers torch datasets

# Run SDFT
python scripts/sdft_train_v2.py

# Run SDPO/DPO
python scripts/sdpo_train_v2.py

# Run Haiku distillation (Method C, no training)
python scripts/haiku_distill_prompting.py

# Run Haiku distillation (Method A: SFT)
python scripts/haiku_distill_sft.py
```

---

This PR will be updated as runs complete. Each checkpoint commit includes updated metrics files and RESULTS.md.
