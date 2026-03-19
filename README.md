# voice-ai-research

Multi-agent collaborative research repository for voice AI training methods.
Forked from [Hamza-Mos/praxlab](https://github.com/Hamza-Mos/praxlab).

## Project Goal

Comparative study of training paradigms for voice AI models (TTS / spoken dialogue / voice cloning):
- **SFT** (Supervised Fine-Tuning) -- baseline
- **SDFT** (Self-Distillation Fine-Tuning) -- continual learning approach
- **DPO** (Direct Preference Optimization) -- RL-based alignment

Evaluation metrics: RTF (real-time factor), MOS naturalness, WER intelligibility, speaker similarity.

## Repository Structure

```
voice-ai-research/
  sft/          <- SFT baseline experiments (owner: ooo, branch: sft-branch)
  sdft/         <- SDFT experiments (owner: coolstufs, branch: sdft-branch)
  dpo/          <- DPO / RL experiments (owner: serious-inference-engineer, branch: dpo-branch)
  data/         <- Shared datasets, data prep scripts, splits
  docs/         <- Research reports, literature notes, design docs
```

## Agent Workflow

Each agent works on a dedicated branch and opens PRs to main.

### Branch assignments

| Agent | Branch | Folder |
|---|---|---|
| ooo | sft-branch | sft/ |
| coolstufs | sdft-branch | sdft/ |
| serious-inference-engineer | dpo-branch | dpo/ |

### Getting started

1. Clone the repo:
   ```bash
   git clone https://github.com/prsabahrami/voice-ai-research.git
   cd voice-ai-research
   ```

2. Create your assigned branch:
   ```bash
   # ooo
   git checkout -b sft-branch

   # coolstufs
   git checkout -b sdft-branch

   # serious-inference-engineer
   git checkout -b dpo-branch
   ```

3. Work in your designated folder (`sft/`, `sdft/`, or `dpo/`).

4. Commit your changes and open a PR to main:
   ```bash
   git add sft/   # or sdft/ or dpo/
   git commit -m "sft: <description>"
   git push origin sft-branch  # or your branch name
   ```
   Then open a PR on GitHub: https://github.com/prsabahrami/voice-ai-research/compare

### Data

Shared dataset: LJSpeech-1.1 (located on Lambda H100 at `/home/ubuntu/datasets/LJSpeech-1.1/`)
- 13,100 utterances, ~24 hours, 22.05 kHz

Put data prep scripts and small metadata files in `data/`. Large files stay on the Lambda instance.

### Docs

Research reports and literature notes live in `docs/`. See:
- `docs/rl_latency_research.md` -- RL methods and inference latency analysis

## Infrastructure

Lambda H100 instance (shared):
- Host: 192.222.55.210
- User: ubuntu
- Venv: `/home/ubuntu/voice_ai_venv`
- DPO pipeline: `/home/ubuntu/voice_ai_dpo/train_dpo.py`

## References

- praxlab upstream: https://github.com/Hamza-Mos/praxlab
- LJSpeech dataset: https://keithito.com/LJ-Speech-Dataset/
