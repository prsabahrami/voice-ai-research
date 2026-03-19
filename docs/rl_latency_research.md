# Voice AI Research: RL Methods and Inference Latency

**Prepared for:** zayaan mulla (zayaan@talkingcomputers.ai)
**Date:** March 19, 2026
**Branch:** rl-latency-research (branch-000008-01)

---

## Executive Summary

This report covers reinforcement learning (RL)-based methods for voice AI training and the inference latency implications of three training paradigms: Supervised Fine-Tuning (SFT), Self-Distillation Fine-Tuning (SDFT), and RL-based approaches (RLHF, DPO, GRPO, PPO). The key finding is that **training method does not itself determine inference latency** -- the model architecture does. However, RL methods enable quality improvements that allow deployment of smaller/faster architectures without quality loss, and RL training overhead is substantially higher. The best production systems in 2025 use SFT or SDFT as a foundation, apply RL for quality alignment, then apply inference optimizations (speculative decoding, quantization, flow-matching) separately.

---

## 1. RL-Based Methods for Voice AI (2024-2025)

### 1.1 Why RL for Voice AI?

Voice AI (TTS, voice cloning, spoken dialogue) introduces unique challenges not present in text LLM alignment:

- Evaluation metrics are non-differentiable (MOS, WER, speaker similarity require separate models)
- Speech tokens sequences are 10-20x longer than text tokens for the same content
- Temporal coherence and acoustic consistency must be maintained across thousands of frames
- Reward signal is sparse and noisy (audio quality differences are subtle)

RL addresses these by optimizing directly against automated proxies for human preference (UTMOS for naturalness, ASR-based WER for intelligibility, speaker embedding cosine similarity).

### 1.2 RLHF (Reinforcement Learning from Human Feedback)

**Standard pipeline:** SFT base -> reward model trained on human preferences -> RL fine-tuning with PPO.

**Applied to TTS:**

**DLPO (Diffusion Loss-Guided Policy Optimization)** -- Interspeech 2025 (Chen et al., Ohio State / Amazon)
- Applies RLHF to WaveGrad 2, a non-autoregressive diffusion TTS model
- Key innovation: integrates the original diffusion training loss directly into the reward function, preventing over-optimization
- Reward: UTMOS naturalness score (proxy for human MOS)
- Results: UTMOS 3.65, NISQA 4.02, WER 1.2%, DLPO audio preferred 67% of the time over baseline
- Significance: First successful RLHF for TTS diffusion models; avoids reward hacking via dual-objective regularization

**GSRM (Generative Speech Reward Model)** -- arXiv:2602.13891 (Microsoft, Feb 2026)
- Curated 31,000 expert ratings as a reward dataset for speech RLHF
- Addresses multi-dimensional speech quality: naturalness, clarity, speaker similarity, prosody
- Serves as a verifier for online RLHF; inspired by GPT-4o Voice Mode and Gemini Live development
- Decomposes reward modeling rather than treating naturalness as a single scalar

**SpeechJudge** (Manakul et al., 2025)
- Applies DPO to post-train TTS models using human preferences on pairwise audio comparisons

### 1.3 DPO (Direct Preference Optimization) for Voice

DPO eliminates the separate reward model by treating preference data as a direct classification objective, increasing log-probability of preferred outputs relative to rejected ones.

**DPO on xTTSv2** (Zhuravlov & Sydorskyi, ICNLSP 2025, Kyiv Polytechnic)
- Semi-supervised training: standard SFT on ASR-labeled data, then DPO using model-generated preference pairs
- Preference metrics: WER (Whisper), Speaker Similarity (ECAPA2), Proxy MOS (UTMOS)
- Auto-constructs DPO pairs: 10 audio samples per (reference, text) pair, ranked by harmonic mean of 3 metrics
- Results: Outperforms traditional SFT on human-labeled data in 2/3 metrics, statistically comparable CMOS
- Key advantage: No human annotation required for DPO phase

**DPO for Speech Enhancement** (arXiv:2507.09929, ISCA 2025)
- Applies DPO to language-model-based speech enhancement (GenSE)
- UTMOS as preference signal, UTMOS gain of up to 56%, NISQA +19%
- First DPO study for speech enhancement

**Emo-DPO** (referenced in arXiv:2509.18531)
- DPO applied to capture emotional/prosodic nuances in TTS
- Standard WER/SIM rewards insufficient for prosody; DPO with human preference pairs fills the gap

**Iterative DPO for Prosody** (Shin et al., arXiv:2509.18531, Interspeech 2025)
- ~200 human-labeled preference pairs per round applied iteratively
- On KoCC-TTS Korean call-center benchmark: Iterative DPO (Round 2) ranks highest by ELO
- GRPO with transcript-centric rewards (CER/WER) ranks lowest -- confirms RL metric choice matters enormously

### 1.4 PPO and Policy Gradient Methods for Voice

**Seed-TTS** (ByteDance, 2024)
- Large-scale autoregressive TTS trained with PPO, REINFORCE, and DPO
- Reward: speaker similarity (SIM) + WER from ASR
- PPO used with both rewards simultaneously

**F5R-TTS** (Tencent, arXiv:2504.02407, 2025)
- Applies GRPO (Group Relative Policy Optimization) to F5-TTS, a flow-matching non-autoregressive TTS
- Key innovation: reformulates deterministic flow-matching outputs into probabilistic sequences to enable RL
- Reward: WER (ASR-based) + Speaker Similarity
- Results: 29.5% relative WER reduction, 4.6% relative speaker similarity improvement
- First successful RL integration into NAR flow-matching TTS architecture

**DiffRO (Differentiable Reward Optimization)** (Gao et al., Interspeech 2025)
- Applies RL directly on codec tokens rather than decoded audio
- Eliminates the costly audio decoding step during RL training (major practical advance)
- Uses Gumbel-Softmax to make reward differentiable through codec tokens
- Achieves SOTA WER on seed-tts-eval benchmark
- Compared to DPO: DiffRO achieves lower WER (0.78 vs DPO 1.27); WER improvement: DiffRO -ASR 1.89, DPO 3.28

**GRPO for ASR** (arXiv:2509.01939, 2025)
- GRPO applied to LLM-based automatic speech recognition
- Three rule-based rewards: WER reduction, hallucination reduction, out-of-domain robustness
- Up to 18.4% relative WER reduction, 66% WER reduction on AMI-IHM, 49% WER reduction on domain adaptation

### 1.5 Key OSS Projects

**VALL-E / VALL-E 2** (Microsoft Research)
- Neural codec language model: AR + NAR architecture, GPT-style autoregressive first codebook
- VALL-E 2 innovations: Repetition Aware Sampling (stabilizes decoding), Grouped Code Modeling (sequences grouped into chunks, boosting inference speed)
- Human parity achieved on LibriSpeech and VCTK (first time)
- RL not explicitly applied but architecture is GPT-compatible, making DPO/GRPO adaptation natural
- Grouped Code Modeling reduces AR steps 2-4x depending on group size

**OpenVoice V2** (MIT / MyShell.ai / Tsinghua, 2024)
- Decoupled architecture: Base Speaker TTS + Tone Color Converter
- Feed-forward structure: RTF ~0.083 (12x real-time) on A10G GPU
- Zero-shot cross-lingual cloning with no RL training; SFT-only
- Used by 2M+ users as voice engine of MyShell.ai

**SpeechGPT-2.0-preview** (OpenMOSS / Fudan University, 2025)
- Real-time spoken dialogue system targeting GPT-4o-level capabilities
- End-to-end speech-language model; direct audio output rather than TTS cascade

**CosyVoice 2 / CosyVoice 3** (Alibaba FunAudioLLM)
- LM-based TTS with conditional flow matching (CFM) vocoder
- CosyVoice 3 applies RL (differentiable reward optimization) as post-training step
- Fun-CosyVoice3-0.5B-2512_RL: SOTA performance with RL optimization at 0.5B parameters
- Bi-streaming support: 150ms latency with TensorRT-LLM (4x speedup over baseline)
- seed-tts-eval: test-zh CER 1.45%, test-en WER 2.57% (CosyVoice2)

---

## 2. SDFT: Self-Distillation Fine-Tuning

"SDFT" in the voice AI context refers to **Self-Distillation Fine-Tuning** (arXiv:2601.19897, Shenfeld et al., MIT / ETH Zurich, Jan 2026; and independently arXiv:2402.13669, Yang et al., ACL 2024).

### 2.1 What SDFT Is

SDFT is a training method that sits between standard SFT (off-policy imitation learning) and full RL (requires explicit reward functions):

- Uses the model itself as teacher via in-context learning
- Generates on-policy training signals from the model's own outputs conditioned on demonstrations
- No reward model required, no policy gradient instability
- Reduces catastrophic forgetting compared to SFT
- In continual learning experiments: SDFT enables sequential skill acquisition without performance regression

### 2.2 SDFT vs SFT vs RL

| Dimension | SFT | SDFT | RL (PPO/GRPO) |
|---|---|---|---|
| Policy type | Off-policy | On-policy | On-policy |
| Reward required | No | No | Yes |
| Catastrophic forgetting | High | Low | Moderate |
| Training stability | Very high | High | Low to moderate |
| Engineering complexity | Low | Low-medium | High |
| Data efficiency | High (dense gradients) | High | Low (sparse reward) |
| New skill acquisition | Moderate | Higher than SFT | Highest (if reward valid) |

**In voice AI applications:** SDFT can be applied by having a CosyVoice or VALL-E-style model generate candidate speech samples conditioned on a few-shot demonstration of the target speaker/style, then training on those on-policy samples. This avoids needing a separate reward model while delivering better generalization than vanilla SFT.

A related approach is self-data distilled fine-tuning for pruning recovery (Thangarasa et al., MLSys 2025): prune a base model, use unpruned model to generate distilled dataset, fine-tune pruned model. When combined with speculative decoding, improves token acceptance rates and inference efficiency.

---

## 3. Inference Latency: Architecture and Method Comparison

### 3.1 The Core Principle

**Training method (SFT/SDFT/RL) does not directly determine inference latency.** The inference speed is set by:
1. Model architecture (AR vs NAR vs flow-matching)
2. Model size
3. Inference optimizations (speculative decoding, quantization, batching)
4. Hardware and serving stack

However, **training method indirectly affects latency** in two ways:
- RL enables better quality at smaller model sizes (fewer parameters needed -> faster inference)
- RL/SDFT can be used to train draft models for speculative decoding

### 3.2 Architecture Impact on RTF

**Real-Time Factor (RTF):** ratio of synthesis time to audio duration. RTF < 1.0 = faster than real-time.

| Architecture | Examples | RTF Range | Notes |
|---|---|---|---|
| Feed-forward / Non-AR | OpenVoice V2, FastPitch, Kokoro-82M | 0.05-0.30 | Fastest; parallel generation |
| Flow-matching / Diffusion NAR | F5-TTS, MaskGCT, DLPO-WaveGrad2 | 0.15-0.50 | Fast; quality very high |
| Autoregressive LM | VALL-E, CosyVoice, Llasa | 0.5-3.0+ | Slower; sequential by nature |
| AR + Speculative Decoding | CosyVoice+SSD, VALL-E+VADUSA | 0.3-2.0 | 1.4x-4x speedup on AR |
| Heavy AR (no optimization) | BarkTTS, VALL-E-X, Tacotron2 | 2.0-30+ | Not suitable for real-time |

### 3.3 Published RTF Benchmarks

**F5-TTS** (flow-matching NAR, SFT-trained):
- RTF = 0.15 (RTX3090, nfe=16, Euler ODE solver, 200-run avg over 10s audio)
- RTF = 0.063 on H800 80G (1 GPU)
- GPU memory: 2,994 MB

**CosyVoice 2** (LM-based AR, SFT-trained):
- LM takes ~70% of total synthesis time
- RTF ~1.283 (DatarootLabs LibriTTS benchmark)
- With TensorRT-LLM: 4x speedup, streaming latency 150ms (CosyVoice 3)
- CosyVoice 2 AR LM generates first audio chunk; CFM vocoder is parallel

**XTTS_v2** (AR codec model, SFT):
- RTF = 0.482 (DatarootLabs benchmark)
- Latency: 3.36 seconds (best among 5 models tested on LibriTTS)

**OpenVoice V2** (feed-forward, SFT):
- RTF ~0.083 (12x real-time on A10G GPU)
- Feed-forward architecture; no AR step

**Kokoro-82M** (feed-forward):
- Sub-0.3 second processing for all text lengths tested
- Fastest of 12 models in Inferless 2025 evaluation

**VALL-E X / BarkTTS** (heavy AR, no optimization):
- Median latency >2 seconds
- P90 latency: extreme (tens of seconds for longer text)
- Bottom performers in MDPI 2025 responsiveness benchmark (13 models)

**VoiceStar 840M** (AR NCLM):
- RTF > 1.0 (cannot do real-time without optimization)
- Mitigable with quantization, grouped prediction, speculative decoding

### 3.4 Speculative Decoding for AR TTS

Speculative decoding uses a fast draft model to propose tokens, then verifies with the full model in a single forward pass. It maintains output quality while reducing latency.

**SSD (Speech Speculative Decoding)** -- Tsinghua / Tencent, Interspeech 2025
- Applied to CosyVoice 2
- 1.4x speedup over conventional AR decoding
- Custom acceptance criterion for speech tokens (addresses variability in audio token mappings)
- Maintains high fidelity and naturalness

**VADUSA** -- Shanghai Jiao Tong University (arXiv:2410.21951)
- MEDUSA speculative decoding adapted for VALL-E TTS
- Adds "tolerance mechanism" to handle speech token variability
- Achieves acceleration + quality improvement (unlike vanilla MEDUSA applied to speech)

**KAIST Multi-Token Prediction** (KAIST / 42dot, ICASSP 2025)
- Viterbi-based speculative decoding for codec TTS
- 4-5x reduction in per-token inference time
- Flexible quality-speed tradeoff controlled at inference time without retraining
- First Viterbi-based speculative decoding for speech synthesis

**NVIDIA TensorRT-LLM Speculative Decoding** (Dec 2024)
- Llama 3.1 405B target + 1B draft: 3.33x throughput on H200 (4 GPU)
- Llama 3.1 70B target + 1B draft: 2.86x throughput on H200 (1 GPU)
- Direct applicability to GPT-based voice backbone

**PredGen** -- UCLA (arXiv:2506.15556)
- Input-time speculation for LLM+TTS cascade systems
- Generates candidate responses while user is still speaking
- ~2x latency reduction on LMsys/MT-Bench datasets
- Consumer-grade hardware (RTX 3090); TTS latency ~200ms regardless of H100 vs 3090

### 3.5 Training Method Latency Comparison

| Training Method | Training Overhead | Inference Architecture | Inference RTF (Typical) | Latency Impact |
|---|---|---|---|---|
| SFT | 1x baseline | Unchanged from base | Architecture-determined | Baseline |
| SDFT | 1.5-2x (on-policy sample gen) | Unchanged from base | Same as SFT | Same as SFT; pruning variant improves spec decoding |
| DPO | 1.2-1.5x (offline preference data) | Unchanged | Same as SFT | Same as SFT; no inference overhead |
| GRPO | 3-5x (group rollouts, no value network) | Unchanged | Same as SFT | Same as SFT; model may be smaller |
| PPO | 5-10x (on-policy rollouts, value model, separate inference) | Unchanged | Same as SFT | Same as SFT; high training cost only |

**Key insight:** RL (PPO/GRPO/DPO) does not slow down inference. The model architecture at inference time is identical whether trained with SFT or RL. RL training overhead is purely a training-time concern.

---

## 4. Fitting RL into an OSS GPT-Based Voice Pipeline

### 4.1 Recommended Pipeline for GPT-Base Voice Model

```
Stage 1: Pretraining (if starting from scratch)
  - Large speech corpus (LibriLight 60K hrs equivalent for VALL-E scale)
  - Codec tokenization: EnCodec / SoundStream / DAC
  - GPT-style AR transformer on first codebook

Stage 2: SFT (Supervised Fine-Tuning)
  - Target speaker / style data (hundreds to thousands of hours)
  - Standard cross-entropy on codec token sequences
  - Establishes clean generative baseline

Stage 3: SDFT (optional, for continual learning)
  - Use the SFT model as its own teacher via in-context conditioning
  - Generate on-policy samples for new speakers/domains
  - Prevents catastrophic forgetting when adding languages or styles

Stage 4: Preference Alignment (RL phase)
  Choose based on data availability and compute budget:

  - DPO (recommended for most teams):
    * Generate N samples per prompt (N=5-10)
    * Rank by WER (ASR model) + Speaker Similarity (WavLM/ECAPA2) + UTMOS
    * Construct (prompt, preferred, rejected) pairs
    * Single-stage training, stable, no reward model needed
    * Compute: ~1.2-1.5x SFT

  - GRPO (if DPO insufficient):
    * Group sampling (G=8 candidates per prompt)
    * Dual reward: WER + Speaker Similarity
    * No value function (unlike PPO), lower GPU overhead than PPO
    * Compute: ~3-5x SFT

  - PPO (only if production budget is high):
    * Separate reward model trained on human ratings
    * Separate inference GPU pool for rollouts
    * Highest quality ceiling but 5-10x SFT compute cost

Stage 5: Inference Optimization
  - Speculative decoding: small draft model (25-50% of target size)
    Expected speedup: 1.4x-3.6x
  - Quantization: INT8/INT4 codec model; 2-4x speedup
  - Grouped Code Modeling (VALL-E 2 style): 2-4x fewer AR steps
  - Flow-matching vocoder (CosyVoice style): parallel; avoids AR vocoder cost
  - Streaming: sentence-by-sentence TTS (150ms TTFS achievable)
```

### 4.2 OSS Model Bases Available

| Base Model | Architecture | Parameters | RL Applied | License |
|---|---|---|---|---|
| VALL-E (unofficial) | AR+NAR GPT | ~1B | No (but compatible) | Apache 2.0 (unofficial) |
| F5-TTS | NAR flow-matching | 300M | GRPO (F5R-TTS) | MIT |
| CosyVoice 2 | LM+CFM | 500M | RL (DiffRO) | Apache 2.0 |
| xTTSv2 | AR GPT | ~350M | DPO (demonstrated) | Coqui License |
| Kokoro-82M | Feed-forward | 82M | No | Apache 2.0 |
| OpenVoice V2 | VITS+converter | ~100M | No | MIT |

---

## 5. Metrics and Benchmarks

### 5.1 Primary Metrics

| Metric | What it measures | Typical range | Better is |
|---|---|---|---|
| MOS / CMOS / UTMOS | Perceptual naturalness | 1.0-5.0 (MOS) | Higher |
| WER / CER | Intelligibility via ASR | 0-100% | Lower |
| Speaker Similarity (SIM) | Voice clone fidelity | 0.0-1.0 | Higher |
| RTF | Speed vs real-time | 0.05-30+ | Lower (<1.0 for real-time) |
| NISQA | Narrowband speech quality | 1.0-5.0 | Higher |
| DNSMOS | Non-intrusive perceptual quality | 1.0-5.0 | Higher |

### 5.2 Benchmark Dataset Results (2024-2025)

**seed-tts-eval benchmark** (most comprehensive 2025 TTS benchmark):

| Model | Training | test-zh CER | test-en WER | test-hard CER | Speaker Sim |
|---|---|---|---|---|---|
| Seed-TTS | RL (PPO+DPO) | 1.12% | 2.25% | 7.59% | 79.6% |
| CosyVoice 3 (RL) | SFT+RL | SOTA | SOTA | SOTA | Best in class |
| CosyVoice 2 | SFT | 1.45% | 2.57% | 6.83% | 75.7% |
| F5-TTS | SFT | 1.52% | 2.00% | 8.67% | 74.1% |
| DiffRO+ASR | SFT+RL(DiffRO) | 0.78% | 1.89% | 5.58% (hard) | ~76% |

**DLPO benchmark** (WaveGrad2 baseline vs DLPO):
- UTMOS: baseline 3.16 -> DLPO 3.65 (+0.49)
- NISQA: reported 4.02
- WER: 1.2% (low, maintained during RL)
- Human preference: DLPO preferred 67% of pairwise comparisons

**F5R-TTS** (F5-TTS baseline vs GRPO):
- WER: 29.5% relative reduction
- Speaker Similarity: 4.6% relative improvement

**DatarootLabs LibriTTS Benchmark** (5 models, GPU-based):

| Model | RTF | Latency | Speaker Sim | WER |
|---|---|---|---|---|
| XTTS_v2 | 0.482 | 3.36s (best) | 0.4973 | 0.2750 |
| IndexTTS | 0.848 | 4.83s | 0.4978 | 0.2418 (best) |
| CosyVoice2 | 1.283 | tied | 0.5881 (best) | 0.3237 |
| F5TTS | 1.283 | tied | 0.4788 | 0.3204 |
| FishSpeech-S1-mini | 31.467 | worst | 0.5951 | 0.5448 |

### 5.3 Compute Requirements

| Method | GPU Hours (relative to SFT=1x) | Infrastructure Complexity |
|---|---|---|
| SFT | 1x | Single GPU pool |
| SDFT | 1.5-2x | Single GPU pool + in-context sampling |
| DPO | 1.2-1.5x | Single GPU pool + preference data construction |
| GRPO | 3-5x | Single GPU pool, G=8-64 rollouts per step |
| PPO | 5-10x | Separate inference + training GPU pools |

Concrete example: For a 500M parameter model on 4x A100:
- SFT: ~12-24 GPU hours for standard fine-tuning
- DPO: ~15-30 GPU hours (preference data gen + training)
- GRPO: ~50-100 GPU hours (rollout generation dominates)
- PPO: ~100-200+ GPU hours + separate inference server

---

## 6. Quality vs Latency Tradeoffs

### 6.1 The Fundamental Tradeoff

Better voice quality (higher MOS, lower WER) generally requires larger models or more inference steps, both of which increase latency. RL methods can shift this tradeoff favorably:

- DPO/GRPO-trained models achieve higher quality at the same model size
- This means RL can enable use of a smaller/faster model for the same quality level
- Example: CosyVoice 3 0.5B with RL achieves SOTA performance previously requiring 1.5B+ models

### 6.2 Recommended Strategy by Latency Target

**< 100ms latency (ultra-low latency, real-time voice assistant):**
- Architecture: Feed-forward (Kokoro, OpenVoice V2) or streaming-first design
- Training: SFT sufficient; DPO for quality polish
- Hardware: A10G or better, INT8 quantization
- Achievable: RTF ~0.05-0.10

**100-300ms latency (production real-time):**
- Architecture: NAR flow-matching (F5-TTS style) or optimized AR with speculative decoding
- Training: SFT + DPO or GRPO for quality
- Hardware: RTX 3090 or A100
- Achievable: RTF 0.15-0.30

**300ms-1s latency (acceptable for streaming assistants):**
- Architecture: AR LM + parallel vocoder (CosyVoice style), streaming sentence-by-sentence
- Training: SFT + GRPO/DPO for alignment
- Hardware: Consumer GPU
- Achievable: First sentence within 300ms (PredGen approach)

**> 1s latency (batch TTS, audiobooks):**
- Architecture: Any; optimize for quality over speed
- Training: Full PPO with human feedback for highest quality ceiling
- Hardware: Flexible

### 6.3 AR vs NAR Latency

Key finding from MDPI 2025 benchmark (13 TTS models):
- Attention-based AR systems: increasingly disfavored for low-latency due to attention/stop-token failures
- FastPitch (feed-forward): outperforms GlowTTS (flow-based NAR) despite earlier reports
- VALL-E-X, BarkTTS: consistently rank last in both latency and quality in benchmark
- NAR flow-matching (F5-TTS): best balance of quality and latency among OSS options

---

## 7. Observations for OSS GPT-Base Voice Pipeline

1. **SFT alone is insufficient for production quality.** DPO adds ~20-30% quality improvement at modest cost.

2. **GRPO is the best RL method for small teams.** No value function, reduced GPU overhead vs PPO, group sampling is straightforward. F5R-TTS demonstrated 29.5% WER reduction with GRPO.

3. **DiffRO is the most practical RL advance.** By computing rewards directly on codec tokens (not decoded audio), it eliminates the bottleneck of audio decoding during RL training, making RL cycles 3-5x cheaper.

4. **Architecture choice dominates latency, not training method.** Use NAR (flow-matching) or AR with speculative decoding for real-time targets.

5. **SDFT is the right tool for continual learning.** When adding new speakers, languages, or styles to an existing model, SDFT prevents catastrophic forgetting better than SFT while avoiding the complexity of RL.

6. **Speculative decoding is the most impactful inference optimization.** 1.4x-4x speedup with minimal quality loss; KAIST multi-token prediction achieves 4-5x per-token speedup without retraining.

7. **Human parity has been achieved in TTS (VALL-E 2).** The quality ceiling has shifted to speaker diversity, naturalness under challenging text, and low latency.

8. **RL quality gains compound.** SFT -> DPO -> iterative DPO or GRPO produces monotonically better models. CosyVoice progression (1 -> 2 -> 3) demonstrates this clearly.

---

## 8. References

1. Chen et al. (2025). "Fine-Tuning Text-to-Speech Diffusion Models Using Reinforcement Learning with Human Feedback (DLPO)." Interspeech 2025.
2. Gao et al. (2025). "Differentiable Reward Optimization for LLM based TTS system (DiffRO)." Interspeech 2025.
3. Sun et al. (2025). "F5R-TTS: Improving Flow Matching Based Text-to-Speech with Group Relative Policy Optimization." arXiv:2504.02407.
4. Zhuravlov & Sydorskyi (2025). "Beyond Labeled Datasets: Advancing TTS with DPO on Unlabeled Speech Dataset." ICNLSP 2025.
5. Shen et al. (2026). "GSRM: Generative Speech Reward Model for Speech RLHF." arXiv:2602.13891.
6. Shin et al. (2025). "No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS." arXiv:2509.18531.
7. Chen et al. (2024). "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers." arXiv:2406.05370.
8. Lin et al. (2025). "Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding (SSD)." Interspeech 2025.
9. Li et al. (2024). "Fast and High-Quality Auto-Regressive Speech Synthesis via Speculative Decoding (VADUSA)." arXiv:2410.21951.
10. Nguyen et al. (2025). "Accelerating Codec-based Speech Synthesis with Multi-Token Prediction and Speculative Decoding." KAIST / 42dot.
11. Putterman et al. (2024). "TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x." NVIDIA Technical Blog, Dec 2024.
12. Li & Grover (2025). "PredGen: Accelerated Inference of LLMs through Input-Time Speculation for Real-Time Speech Interaction." arXiv:2506.15556.
13. Shenfeld et al. (2026). "Self-Distillation Enables Continual Learning (SDFT)." arXiv:2601.19897.
14. Yang et al. (2024). "Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning." ACL 2024.
15. Thangarasa et al. (2025). "Self-Data Distillation for Recovering Quality in Pruned Large Language Models." MLSys 2025.
16. Qin et al. (2024). "OpenVoice: Versatile Instant Voice Cloning." arXiv:2312.01479v6.
17. Du et al. (2024). "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models." arXiv:2412.10117.
18. Pham et al. (2025). "Benchmarking the Responsiveness of Open-Source Text-to-Speech Systems." MDPI Computers 2025.
19. DatarootLabs (2026). "Which Open Source Text-to-Speech Model Should You Use?" Practical benchmark report.
20. Matsutani et al. (2025). "RL Squeezes, SFT Expands: A Comparative Study of Reasoning LLMs." arXiv:2509.21128.
