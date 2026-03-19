#!/usr/bin/env python3
"""
Haiku Distillation Method C: Style-Transfer Prompting Baseline.
No training required. Use system prompts to transfer Haiku-style to GPT-OSS.
Evaluates how much of Haiku's style can be achieved via prompting alone.

This is the baseline: if prompting alone achieves high style similarity,
SFT/DPO methods need to beat this bar.
"""

import os
import sys
import json
import time
import random
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv("/workspace/.env")

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


HAIKU_STYLE_DESCRIPTOR = """You are a warm, conversational AI assistant in the style of Claude Haiku.

Key characteristics to embody:
- Use a natural, flowing conversational tone
- Be warm and engaging without being saccharine
- Structure your response for easy reading
- Use simple, clear language with natural transitions
- Acknowledge the human dimension of questions
- Be concise but don't sacrifice clarity
- End with something actionable or forward-looking when appropriate
"""

EVALUATION_PROMPTS = [
    "What's the best way to learn programming from scratch?",
    "I feel like I'm not making progress in my career. What should I do?",
    "Can you explain what machine learning actually does?",
    "How do I have a difficult conversation with my manager?",
    "What makes a good team culture?",
    "Help me understand compound interest.",
    "Why is it hard to change habits?",
    "What's the best approach to debugging a complex bug?",
    "How do you stay creative when you feel stuck?",
    "What should I focus on when learning a new codebase?",
]


def style_similarity_score(model_text: str, haiku_text: str) -> dict:
    """
    Compute style similarity between model output and Haiku reference.
    Uses heuristic features (not a learned metric).
    """
    import re
    
    def word_count(text):
        return len(text.split())
    
    def sentence_count(text):
        return len(re.split(r'[.!?]+', text))
    
    def avg_sentence_length(text):
        sents = re.split(r'[.!?]+', text)
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            return 0
        return sum(len(s.split()) for s in sents) / len(sents)
    
    def conversational_markers(text):
        markers = ["you", "your", "i", "we", "let's", "here's", "think", "feel", "actually"]
        text_lower = text.lower()
        return sum(text_lower.count(m) for m in markers)

    model_wc = word_count(model_text)
    haiku_wc = word_count(haiku_text)
    length_ratio = min(model_wc, haiku_wc) / max(model_wc, haiku_wc) if max(model_wc, haiku_wc) > 0 else 0

    model_asl = avg_sentence_length(model_text)
    haiku_asl = avg_sentence_length(haiku_text)
    asl_diff = abs(model_asl - haiku_asl)

    model_cm = conversational_markers(model_text)
    haiku_cm = conversational_markers(haiku_text)
    cm_ratio = min(model_cm, haiku_cm) / max(model_cm + 1, haiku_cm + 1)

    return {
        "model_word_count": model_wc,
        "haiku_word_count": haiku_wc,
        "length_ratio": round(length_ratio, 3),
        "avg_sentence_length_model": round(model_asl, 1),
        "avg_sentence_length_haiku": round(haiku_asl, 1),
        "avg_sentence_length_diff": round(asl_diff, 1),
        "conversational_markers_model": model_cm,
        "conversational_markers_haiku": haiku_cm,
        "conversational_marker_ratio": round(cm_ratio, 3),
    }


def run_prompting_baseline(results_dir: Path):
    """
    Run style-transfer prompting baseline.
    Generates responses with:
    1. Plain GPT-OSS-style system prompt (via Haiku with minimal context)
    2. Haiku-style system prompt (via Haiku with full Haiku descriptor)
    3. Direct Haiku call (gold reference)
    
    Since we don't have direct GPT-OSS inference without Tinker training,
    we use Haiku with different system prompts to approximate GPT-OSS behavior.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    
    results = {
        "method": "prompting_baseline",
        "description": "Style transfer via system prompt - no training required",
        "timestamp": datetime.now().isoformat(),
        "examples": [],
    }
    
    logger.info("Running Haiku style-transfer prompting baseline...")
    
    for i, prompt in enumerate(EVALUATION_PROMPTS):
        logger.info(f"Prompt {i+1}/{len(EVALUATION_PROMPTS)}: {prompt[:60]!r}")
        
        try:
            # Condition 1: No style guidance (baseline generic)
            generic_resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                system="Be brief and factual.",
                messages=[{"role": "user", "content": prompt}]
            )
            generic_text = generic_resp.content[0].text
            
            # Condition 2: Haiku style-transfer prompting
            styled_resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                system=HAIKU_STYLE_DESCRIPTOR,
                messages=[{"role": "user", "content": prompt}]
            )
            styled_text = styled_resp.content[0].text
            
            # Condition 3: Haiku default (gold)
            gold_resp = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            gold_text = gold_resp.content[0].text

            # Compute style similarity scores
            styled_vs_gold = style_similarity_score(styled_text, gold_text)
            generic_vs_gold = style_similarity_score(generic_text, gold_text)

            example = {
                "prompt": prompt,
                "generic_response": generic_text,
                "styled_response": styled_text,
                "gold_haiku_response": gold_text,
                "styled_vs_gold_similarity": styled_vs_gold,
                "generic_vs_gold_similarity": generic_vs_gold,
            }
            results["examples"].append(example)

            logger.info(f"  Generic length ratio vs gold: {generic_vs_gold['length_ratio']:.3f}")
            logger.info(f"  Styled length ratio vs gold:  {styled_vs_gold['length_ratio']:.3f}")

            time.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"  Failed: {e}")

    # Aggregate metrics
    if results["examples"]:
        styled_lr = [e["styled_vs_gold_similarity"]["length_ratio"] for e in results["examples"]]
        generic_lr = [e["generic_vs_gold_similarity"]["length_ratio"] for e in results["examples"]]
        styled_cm = [e["styled_vs_gold_similarity"]["conversational_marker_ratio"] for e in results["examples"]]
        generic_cm = [e["generic_vs_gold_similarity"]["conversational_marker_ratio"] for e in results["examples"]]

        results["aggregate"] = {
            "n_examples": len(results["examples"]),
            "styled_avg_length_ratio": round(sum(styled_lr) / len(styled_lr), 3),
            "generic_avg_length_ratio": round(sum(generic_lr) / len(generic_lr), 3),
            "styled_avg_conv_marker_ratio": round(sum(styled_cm) / len(styled_cm), 3),
            "generic_avg_conv_marker_ratio": round(sum(generic_cm) / len(generic_cm), 3),
            "style_transfer_gain": round(
                sum(styled_lr) / len(styled_lr) - sum(generic_lr) / len(generic_lr), 3
            ),
        }
        
        logger.info(f"\n=== Prompting Baseline Results ===")
        logger.info(f"Styled vs gold length ratio: {results['aggregate']['styled_avg_length_ratio']:.3f}")
        logger.info(f"Generic vs gold length ratio: {results['aggregate']['generic_avg_length_ratio']:.3f}")
        logger.info(f"Style transfer gain: {results['aggregate']['style_transfer_gain']:.3f}")

    out_path = results_dir / f"haiku_prompting_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved prompting baseline results to {out_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/haiku_distill")
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_prompting_baseline(results_dir)


if __name__ == "__main__":
    main()
