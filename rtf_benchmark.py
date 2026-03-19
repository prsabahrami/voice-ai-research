#!/usr/bin/env python3
"""
RTF (Real-Time Factor) Latency Benchmark for Voice AI models.
Tests SFT, DPO, and SDFT checkpoints on Orpheus-3B TTS.
RTF = generation_time / audio_duration
RTF < 1.0 = faster than real-time (good)
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SNAC 24kHz codec: ~75 tokens/second
SNAC_TOKENS_PER_SECOND = 75
# Number of codec levels
SNAC_LEVELS = 3  # SNAC 24kHz has 3 levels

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Welcome to the future of text to speech synthesis.",
    "Scientists have discovered a new species of deep sea fish.",
    "The stock market closed higher today after strong earnings reports.",
    "Please leave a message after the tone and we will call you back.",
    "In a hole in the ground there lived a hobbit.",
    "It was the best of times, it was the worst of times.",
    "To be or not to be, that is the question.",
    "Four score and seven years ago our fathers brought forth on this continent a new nation.",
    "The weather forecast calls for partly cloudy skies with a chance of afternoon showers.",
]


def format_input(text: str, tokenizer) -> dict:
    """Format text as TTS input for Orpheus."""
    # Orpheus TTS format
    prompt = f"<|text_start|>{text}<|text_end|><|audio_start|>"
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def count_audio_tokens(output_ids: torch.Tensor, tokenizer) -> int:
    """Count the number of audio codec tokens in the output."""
    # Audio tokens are in a specific range for Orpheus
    # In the output, count tokens after <|audio_start|>
    audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_start|>")
    audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_end|>")
    
    ids = output_ids[0].tolist()
    try:
        start_idx = ids.index(audio_start_id) + 1
        if audio_end_id in ids:
            end_idx = ids.index(audio_end_id)
        else:
            end_idx = len(ids)
        audio_tokens = end_idx - start_idx
        return max(0, audio_tokens)
    except (ValueError, IndexError):
        return 0


def estimate_audio_duration(num_audio_tokens: int) -> float:
    """Estimate audio duration from number of codec tokens."""
    # SNAC: 75 tokens/second per level, 3 levels interleaved
    # Total tokens for N seconds = N * 75 * 3 = N * 225
    if num_audio_tokens == 0:
        return 0.0
    return num_audio_tokens / (SNAC_TOKENS_PER_SECOND * SNAC_LEVELS)


def benchmark_model(model_path: str, base_model: str, num_sentences: int = 10, max_new_tokens: int = 1200) -> Dict:
    """Run RTF benchmark on a model."""
    logger.info(f"Benchmarking: {model_path}")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if Path(model_path).exists() else base_model,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (detect LoRA adapter)
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        logger.info(f"Loading LoRA adapter from {model_path}")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = PeftModel.from_pretrained(base, model_path)
    else:
        logger.info(f"Loading full model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    
    model.eval()
    device = next(model.parameters()).device
    logger.info(f"Model loaded on {device}")
    
    results = []
    test_sents = TEST_SENTENCES[:num_sentences]
    
    # Warmup
    logger.info("Warming up...")
    with torch.no_grad():
        warmup_input = format_input(test_sents[0], tokenizer)
        warmup_input = {k: v.to(device) for k, v in warmup_input.items()}
        _ = model.generate(**warmup_input, max_new_tokens=50, do_sample=False)
    
    logger.info(f"Running benchmark on {len(test_sents)} sentences...")
    for i, text in enumerate(test_sents):
        inputs = format_input(text, tokenizer)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        
        t0 = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                repetition_penalty=1.1,
            )
        t1 = time.perf_counter()
        
        generation_time = t1 - t0
        output_len = output.shape[1] - input_len
        audio_tokens = count_audio_tokens(output, tokenizer)
        audio_duration = estimate_audio_duration(audio_tokens)
        rtf = generation_time / max(audio_duration, 0.001)
        
        result = {
            "sentence": text,
            "input_tokens": input_len,
            "output_tokens": output_len,
            "audio_tokens_estimated": audio_tokens,
            "audio_duration_seconds": round(audio_duration, 3),
            "generation_time_seconds": round(generation_time, 3),
            "rtf": round(rtf, 4),
            "tokens_per_second": round(output_len / generation_time, 1),
        }
        results.append(result)
        logger.info(f"  [{i+1}/{len(test_sents)}] RTF={rtf:.3f}, gen={generation_time:.2f}s, audio~{audio_duration:.2f}s")
    
    # Summary stats
    rtfs = [r["rtf"] for r in results if r["audio_tokens_estimated"] > 0]
    tps_list = [r["tokens_per_second"] for r in results]
    
    summary = {
        "model_path": str(model_path),
        "base_model": base_model,
        "num_sentences": len(test_sents),
        "mean_rtf": round(sum(rtfs) / len(rtfs), 4) if rtfs else None,
        "min_rtf": round(min(rtfs), 4) if rtfs else None,
        "max_rtf": round(max(rtfs), 4) if rtfs else None,
        "mean_tokens_per_second": round(sum(tps_list) / len(tps_list), 1),
        "results": results,
    }
    
    logger.info(f"Summary: mean_RTF={summary['mean_rtf']}, mean_TPS={summary['mean_tokens_per_second']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="Model paths to benchmark")
    parser.add_argument("--base_model", default="unsloth/orpheus-3b-0.1-ft")
    parser.add_argument("--num_sentences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=600)
    parser.add_argument("--output", default="/home/ubuntu/rtf_benchmark_results.json")
    args = parser.parse_args()
    
    all_results = {}
    for model_path in args.models:
        try:
            result = benchmark_model(
                model_path, args.base_model,
                num_sentences=args.num_sentences,
                max_new_tokens=args.max_new_tokens
            )
            name = Path(model_path).name
            all_results[name] = result
        except Exception as e:
            logger.error(f"Failed to benchmark {model_path}: {e}")
            all_results[Path(model_path).name] = {"error": str(e)}
    
    # Comparison table
    logger.info("
=== RTF COMPARISON TABLE ===")
    logger.info(f"{'Model':<30} {'Mean RTF':<12} {'Tokens/sec':<12} {'Real-time?'}")
    for name, res in all_results.items():
        if "error" not in res:
            rtf = res.get("mean_rtf", "N/A")
            tps = res.get("mean_tokens_per_second", "N/A")
            realtime = "YES" if isinstance(rtf, float) and rtf < 1.0 else "NO"
            logger.info(f"{name:<30} {str(rtf):<12} {str(tps):<12} {realtime}")
    
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
