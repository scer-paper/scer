#!/usr/bin/env python3
"""
Step 3: Generate meaning-preserving paraphrases for HotpotQA questions.

Creates:
  - paraphrases.json: {question_id: [paraphrase1, ..., paraphrase5]}

Usage:
  PYTHONUNBUFFERED=1 python3 pipeline/generate_paraphrases.py
"""

import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
OUTPUT = os.path.join(DATA_DIR, "paraphrases.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, format_chat, unload_model


NUM_PARAPHRASES = 5
BATCH_SIZE = 256


def generate_paraphrases_batch(questions, llm, tokenizer):
    """Generate paraphrases for a batch of questions."""
    from vllm import SamplingParams

    prompts = []
    for q in questions:
        raw = f"""Generate exactly {NUM_PARAPHRASES} paraphrases of this question. Each paraphrase must:
- Ask exactly the same thing
- Use different words and sentence structure
- Be a complete, natural question

Question: {q}

Output ONLY {NUM_PARAPHRASES} paraphrases, one per line, numbered 1-{NUM_PARAPHRASES}. No other text."""
        prompts.append(format_chat(tokenizer, raw))

    params = SamplingParams(temperature=0.7, max_tokens=300)
    outputs = llm.generate(prompts, params)

    results = {}
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        paras = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                para = line.lstrip('0123456789.):- ').strip()
                if para and len(para) > 10 and para.endswith('?'):
                    paras.append(para)
        # Fallback: if parsing failed, try without question mark check
        if len(paras) < 3:
            paras = []
            for line in text.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    para = line.lstrip('0123456789.):- ').strip()
                    if para and len(para) > 10:
                        paras.append(para)
        results[questions[i]] = paras[:NUM_PARAPHRASES]

    return results


def main():
    # Load questions
    with open(os.path.join(DATA_DIR, "questions.json")) as f:
        questions = json.load(f)

    q_texts = [q["question"] for q in questions]
    q_ids = [q["id"] for q in questions]
    print(f"Loaded {len(q_texts)} questions", flush=True)

    # Check for existing results (resume)
    existing = {}
    if os.path.exists(OUTPUT):
        with open(OUTPUT) as f:
            existing = json.load(f)
        print(f"Resuming: {len(existing)} already done", flush=True)

    # Filter to remaining
    remaining_texts = []
    remaining_ids = []
    for qid, qtxt in zip(q_ids, q_texts):
        if qid not in existing:
            remaining_texts.append(qtxt)
            remaining_ids.append(qid)

    if not remaining_texts:
        print("All paraphrases already generated!", flush=True)
        return

    print(f"Need to generate paraphrases for {len(remaining_texts)} questions", flush=True)

    # Load model
    print("Loading Qwen3-32B...", flush=True)
    llm, tokenizer = load_model("Qwen3-32B", gpu_memory_utilization=0.90)

    # Process in batches
    all_paras = dict(existing)
    for start in range(0, len(remaining_texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(remaining_texts))
        batch_texts = remaining_texts[start:end]
        batch_ids = remaining_ids[start:end]

        print(f"\nBatch [{start+1}-{end}/{len(remaining_texts)}]...", flush=True)
        batch_results = generate_paraphrases_batch(batch_texts, llm, tokenizer)

        for qid, qtxt in zip(batch_ids, batch_texts):
            paras = batch_results.get(qtxt, [])
            all_paras[qid] = paras

        # Save incrementally
        with open(OUTPUT, 'w') as f:
            json.dump(all_paras, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(all_paras)} total paraphrases", flush=True)

        # Stats
        good = sum(1 for v in all_paras.values() if len(v) >= 3)
        avg = sum(len(v) for v in all_paras.values()) / max(len(all_paras), 1)
        print(f"  Quality: {good}/{len(all_paras)} have 3+ paraphrases, avg={avg:.1f}", flush=True)

    # Cleanup
    unload_model(llm)

    # Final stats
    total = len(all_paras)
    good = sum(1 for v in all_paras.values() if len(v) >= NUM_PARAPHRASES)
    avg = sum(len(v) for v in all_paras.values()) / total
    print(f"\n=== DONE ===", flush=True)
    print(f"Total: {total} questions", flush=True)
    print(f"Full paraphrases ({NUM_PARAPHRASES}): {good}/{total}", flush=True)
    print(f"Avg paraphrases: {avg:.1f}", flush=True)


if __name__ == '__main__':
    main()
