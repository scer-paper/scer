"""Qwen3-32B m=10 paraphrase extension.

For each query, generates 5 ADDITIONAL paraphrases using Qwen3-32B-FP8 (same
model as the original m=5 run) and appends them to the existing 5, yielding
paraphrases_m10.json with 10 paraphrases per query.

Purpose: enable the m ∈ {1, 3, 5, 10} scaling analysis.
For m=1,3,5 we subsample from the original paraphrases.json (CPU, no extra
work); m=10 needs this fresh generation.

Checkpointing:
  - Output: data/<bench>/paraphrases_m10.json
  - On resume, skips any query that already has 10 paraphrases in the output.
  - Writes incrementally after every BATCH_SIZE queries.
"""
import json, os, sys, time

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "qwen32b_m10_paraphrases"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, format_chat, unload_model

NUM_NEW = 5         # generate 5 more (added on top of existing 5 → total 10)
TARGET_M = 10
BATCH_SIZE = 128    # smaller batch since 32B is larger
BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODEL_LABEL = "Qwen3-32B"
OUTPUT_NAME = "paraphrases_m10.json"



def make_prompt(question, existing_paras):
    """Generate 5 more paraphrases, different from the existing 5."""
    if existing_paras:
        existing_str = "\n".join(f"- {p}" for p in existing_paras)
        return f"""Generate exactly {NUM_NEW} NEW paraphrases of this question, different from the ones below. Each paraphrase must:
- Ask exactly the same thing
- Use different words and sentence structure from BOTH the question and the existing paraphrases
- Be a complete, natural question

Question: {question}

Existing paraphrases to avoid duplicating:
{existing_str}

Output ONLY {NUM_NEW} NEW paraphrases, one per line, numbered 1-{NUM_NEW}. No other text."""
    else:
        # No existing paraphrases (edge case) - generate 5 fresh
        return f"""Generate exactly {NUM_NEW} paraphrases of this question. Each paraphrase must:
- Ask exactly the same thing
- Use different words and sentence structure
- Be a complete, natural question

Question: {question}

Output ONLY {NUM_NEW} paraphrases, one per line, numbered 1-{NUM_NEW}. No other text."""


def parse_paraphrases(text):
    paras = []
    for line in text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            para = line.lstrip('0123456789.):- ').strip()
            if para and len(para) > 10 and para.endswith('?'):
                paras.append(para)
    if len(paras) < 3:
        paras = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                para = line.lstrip('0123456789.):- ').strip()
                if para and len(para) > 10:
                    paras.append(para)
    return paras[:NUM_NEW]


def generate_extras_batch(question_ids, questions, existing_para_lists, llm, tokenizer):
    """Generate NUM_NEW additional paraphrases for a batch."""
    from vllm import SamplingParams
    prompts = [format_chat(tokenizer, make_prompt(q, p))
               for q, p in zip(questions, existing_para_lists)]
    params = SamplingParams(temperature=0.7, max_tokens=400)
    outputs = llm.generate(prompts, params)
    out = {}
    for qid, q, existing, output in zip(question_ids, questions, existing_para_lists, outputs):
        new_paras = parse_paraphrases(output.outputs[0].text)
        # Combine: original 5 + 5 new → 10 total (deduplicated)
        combined = list(existing)
        for p in new_paras:
            if p not in combined:
                combined.append(p)
            if len(combined) >= TARGET_M:
                break
        out[qid] = combined[:TARGET_M]
    return out


def process_benchmark(bench, llm, tokenizer, status):
    data_dir = os.path.join(BASE, "data", bench)
    in_path = os.path.join(data_dir, "paraphrases.json")
    out_path = os.path.join(data_dir, OUTPUT_NAME)

    with open(os.path.join(data_dir, "questions.json")) as f:
        questions = json.load(f)
    q_by_id = {q["id"]: q["question"] for q in questions}

    # Load existing m=5 paraphrases
    with open(in_path) as f:
        original_paras = json.load(f)

    # Resume from existing m=10 output
    existing_m10 = {}
    if os.path.exists(out_path):
        with open(out_path) as f: existing_m10 = json.load(f)
        n_complete = sum(1 for v in existing_m10.values() if len(v) >= TARGET_M - 2)
        print(f"  [{bench}] resume: {n_complete} / {len(questions)} have near-{TARGET_M} paras", flush=True)
    else:
        print(f"  [{bench}] fresh start: {len(questions)} queries", flush=True)

    # Determine which queries still need extending
    remaining_ids = []
    for qid in q_by_id:
        if qid in existing_m10 and len(existing_m10[qid]) >= TARGET_M - 2:
            continue   # near-complete; skip
        remaining_ids.append(qid)

    if not remaining_ids:
        print(f"  [{bench}] ✓ already complete", flush=True)
        status.setdefault(STAGE_KEY, {})[bench] = "complete"
        return

    all_paras = dict(existing_m10)
    n_remaining = len(remaining_ids)
    t0 = time.time()

    for start in range(0, n_remaining, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_remaining)
        batch_ids = remaining_ids[start:end]
        batch_qs = [q_by_id[qid] for qid in batch_ids]
        batch_existing = [original_paras.get(qid, []) for qid in batch_ids]

        batch_results = generate_extras_batch(batch_ids, batch_qs, batch_existing, llm, tokenizer)
        all_paras.update(batch_results)

        with open(out_path, 'w') as f:
            json.dump(all_paras, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        rate = (end) / elapsed if elapsed > 0 else 0
        eta = (n_remaining - end) / rate if rate > 0 else 0
        avg_len = sum(len(v) for v in all_paras.values()) / len(all_paras)
        print(f"  [{bench}] batch [{start+1}-{end}/{n_remaining}] "
              f"avg_m={avg_len:.2f} rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    n_full = sum(1 for v in all_paras.values() if len(v) == TARGET_M)
    print(f"  [{bench}] ✓ done: {n_full}/{len(all_paras)} have full m={TARGET_M}", flush=True)
    status.setdefault(STAGE_KEY, {})[bench] = "complete"


def main():
    print(f"=== : Qwen3-32B m={TARGET_M} paraphrase extension ===", flush=True)
    status = load_status()
    stage_status = status.get(STAGE_KEY, {})
    if all(stage_status.get(b) == "complete" for b in BENCHMARKS):
        print("All benchmarks already complete. Nothing to do.", flush=True)
        return

    print(f"Loading model: {MODEL_LABEL}...", flush=True)
    llm, tokenizer = load_model(MODEL_LABEL, gpu_memory_utilization=0.90)
    print("Model loaded.\n", flush=True)

    for bench in BENCHMARKS:
        if stage_status.get(bench) == "complete":
            print(f"[{bench}] already marked complete, skipping.", flush=True)
            continue
        print(f"\n--- Processing {bench} ---", flush=True)
        process_benchmark(bench, llm, tokenizer, status)

    unload_model(llm)
    status[STAGE_KEY + "_overall"] = "complete"
    print("\n===  complete ===", flush=True)


if __name__ == '__main__':
    main()
