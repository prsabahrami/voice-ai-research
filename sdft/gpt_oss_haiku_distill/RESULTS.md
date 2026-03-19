# SDFT / SDPO / Haiku Distillation Results

**Last updated:** 2026-03-19 13:58 UTC  
**Status:** RUNNING (3 experiments active, 1 complete)

---

## Experiment Status

| Experiment | Method | Status | Steps | Final Loss | Agent |
|------------|--------|--------|-------|-----------|-------|
| SDFT on GPT-OSS-20B | Cross-entropy SFT | **RUNNING** (140/200) | 140/200 | ~0.0001 | coolstufs |
| Haiku SFT Distillation (Method A) | Haiku imitation SFT | **STARTING** | 0/150 | - | coolstufs |
| Haiku Prompting Baseline (Method C) | Zero-shot prompting | **COMPLETE** | - | N/A | coolstufs |
| SDPO on GPT-OSS-20B | DPO preferences | **QUEUED** | 0/150 | - | @serious-inference-engineer |
| SDFT on large transcripts | UltraChat SFT | **QUEUED** | 0/500 | - | @ooo |

---

## Experiment 1: SDFT on GPT-OSS-20B (RUNNING)

**Run ID:** sdft_gpt_oss_20b_20260319_135019  
**Model:** openai/gpt-oss-20b (MoE, LoRA rank 16)  
**Dataset:** 20 Haiku-generated conversational pairs  
**Hyperparameters:** lr=2e-4, batch=4, max_steps=200

### Loss Curve (steps 1-140)

| Step | Train Loss |
|------|-----------|
| 1 | 2.1529 |
| 10 | 0.8864 |
| 25 | checkpoint saved |
| 50 | 0.0034 |
| 100 | 0.000156 |
| 140 | 0.0001 |

**Observation:** Strong convergence achieved. Loss dropped from 2.15 to <0.001 in 50 steps.
Model has fully fit the 20-example dataset (expected behavior for small SFT).
**Next step:** Run SDFT on larger dataset (ooo's 5000-example run) to compare generalization.

### Sample Outputs (from eval checkpoints)

**Step 25 eval** (prompt: "Can you explain what machine learning is?"):
```
Model: " And how does it actually work?"
```
Note: Sample eval prompt was using raw tokenizer.encode, causing model to generate user followup. Fixed in v2.

**Step 100 eval** (same prompt):
```
Model: " I want to understand the basic idea without getting too technical."
```
Still generating user-side continuation. Root cause: raw encode without chat template context.
Fixed in haiku_sft_v2.py which uses proper apply_chat_template for eval.

---

## Experiment 2: Haiku Prompting Baseline (COMPLETE)

**Method:** Zero-shot style transfer via system prompt (no training required)

### Results Summary

| Metric | Styled (Haiku system) | Generic (brief/factual) | Gold (Haiku default) |
|--------|----------------------|------------------------|---------------------|
| Avg length ratio vs gold | 0.912 | 0.951 | 1.000 |
| Conv. marker ratio vs gold | 0.832 | 0.866 | 1.000 |
| Style transfer gain | -0.039 | 0 | - |

### Key Insight
Surface-level metrics (word count ratio, conversational marker frequency) do NOT
capture the qualitative style differences between Haiku and generic responses.
Both conditions achieve >83% similarity in these metrics.

The real style differences (warmth, depth, structure, flow) require:
- Human evaluation or
- Learned perceptual metrics (trained reward model)
- LLM-as-judge comparisons (next steps)

**Implication:** SFT/DPO experiments should use qualitative eval and LLM-as-judge
rather than relying on surface metrics.

**Raw data:** results/haiku_distill/haiku_prompting_baseline_*.json

---

## Experiment 3: Haiku SFT Distillation (Method A) - STARTING

**Run ID:** haiku_sft_gpt_oss_20b_20260319_*  
**Method:** Direct SFT imitation of Haiku responses on GPT-OSS-20B  
**Dataset:** 30 pairs (20 batch1 + 10 extended topics)  
**Fix vs SDFT:** Using proper chat template for eval sampling (apply_chat_template with add_generation_prompt=True)

Results will appear here as training progresses.

---

## Tinker API Notes (v0.16.1)

Correct Datum format:
```python
types.Datum(
    model_input=types.ModelInput.from_ints(tokens=tokens[:-1]),  # context
    loss_fn_inputs={
        "target_tokens": list(tokens[1:]),   # next-token targets
        "weights": list(weights),             # 0=prompt, 1=response
    }
)
training_client.forward_backward(data, "cross_entropy")
training_client.optim_step(types.AdamParams(learning_rate=2e-4))
```

**For eval sampling (correct approach):**
```python
prompt_enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
prompt_tokens = list(prompt_enc["input_ids"])
sp = types.SamplingParams(max_tokens=150, temperature=0.7, stop=["<|im_end|>"])
sampler = tc.save_weights_and_get_sampling_client(name="eval_sampler")
out = sampler.sample(prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                     sampling_params=sp, num_samples=1).result()
text = tokenizer.decode(out.sequences[0].tokens)
```

---

## File Index

| File | Description |
|------|-------------|
| scripts/sdft_train_v2.py | SDFT training (correct API) |
| scripts/sdpo_train_v2.py | SDPO/DPO training (correct API) |
| scripts/haiku_distill_sft.py | Haiku Method A: SFT distillation |
| scripts/haiku_distill_dpo.py | Haiku Method B: DPO distillation |
| scripts/haiku_distill_prompting.py | Haiku Method C: Prompting baseline |
| configs/*.yaml | Training configs for all experiments |
| data/haiku_sft_train_batch1.json | 20 Haiku SFT training pairs |
| data/haiku_dpo_preference_pairs_batch1.json | 2 DPO preference pairs |
| results/sdft/*.json | SDFT metrics (live) |
| results/haiku_distill/*.json | Haiku distillation results (live) |
