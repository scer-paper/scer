"""Llama-3.1-8B alternative paraphraser regeneration.

Generates 5 meaning-preserving paraphrases per query with Llama-3.1-8B-Instruct
for HotpotQA, FEVER, SQuAD 2.0. Same prompt template as the original Qwen3-32B run.

Checkpointing:
  - Output: data/<bench>/paraphrases_llama8b.json
  - On resume, skips any query already in the output JSON.
  - Writes incrementally after every BATCH_SIZE queries.

Purpose: show the instability finding is robust to the
paraphraser model choice. Llama-3.1-8B is a different family from Qwen3-32B,
so any consistent instability across both rules out "Qwen-specific artifact."
"""
import json, os, sys, time

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "llama8b_paraphrases"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, format_chat, unload_model

NUM_PARAPHRASES = 5
BATCH_SIZE = 256
BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODEL_LABEL = "Llama-8B"   # resolves via pipeline/utils.py MODEL_REGISTRY
OUTPUT_NAME = "paraphrases_llama8b.json"



def make_prompt(question):
    """Same template as generate_paraphrases.py - ensures apples-to-apples
    comparison with the original Qwen3-32B paraphrases."""
    return f"""Generate exactly {NUM_PARAPHRASES} paraphrases of this question. Each paraphrase must:
- Ask exactly the same thing
- Use different words and sentence structure
- Be a complete, natural question

Question: {question}

Output ONLY {NUM_PARAPHRASES} paraphrases, one per line, numbered 1-{NUM_PARAPHRASES}. No other text."""


def parse_paraphrases(text):
    """Extract numbered paraphrases from LLM output. Matches generate_paraphrases.py logic."""
    paras = []
    for line in text.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            para = line.lstrip('0123456789.):- ').strip()
            if para and len(para) > 10 and para.endswith('?'):
                paras.append(para)
    # Fallback without ? check (some paraphrases drop the final ?)
    if len(paras) < 3:
        paras = []
        for line in text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                para = line.lstrip('0123456789.):- ').strip()
                if para and len(para) > 10:
                    paras.append(para)
    return paras[:NUM_PARAPHRASES]


def generate_paraphrases_batch(questions, llm, tokenizer):
    from vllm import SamplingParams
    prompts = [format_chat(tokenizer, make_prompt(q)) for q in questions]
    params = SamplingParams(temperature=0.7, max_tokens=300)
    outputs = llm.generate(prompts, params)
    return {q: parse_paraphrases(out.outputs[0].text) for q, out in zip(questions, outputs)}


def process_benchmark(bench, llm, tokenizer, status):
    data_dir = os.path.join(BASE, "data", bench)
    out_path = os.path.join(data_dir, OUTPUT_NAME)

    with open(os.path.join(data_dir, "questions.json")) as f:
        questions = json.load(f)
    q_by_id = {q["id"]: q["question"] for q in questions}

    # Resume from existing output
    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f: existing = json.load(f)
        print(f"  [{bench}] resume: {len(existing)} / {len(questions)} already done", flush=True)
    else:
        print(f"  [{bench}] fresh start: {len(questions)} queries", flush=True)

    remaining_ids = [qid for qid in q_by_id if qid not in existing]
    remaining_texts = [q_by_id[qid] for qid in remaining_ids]

    if not remaining_ids:
        print(f"  [{bench}] ✓ already complete", flush=True)
        status.setdefault(STAGE_KEY, {})[bench] = "complete"
        return

    all_paras = dict(existing)
    n_remaining = len(remaining_ids)
    t0 = time.time()

    for start in range(0, n_remaining, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_remaining)
        batch_texts = remaining_texts[start:end]
        batch_ids = remaining_ids[start:end]

        batch_results = generate_paraphrases_batch(batch_texts, llm, tokenizer)
        for qid, qtxt in zip(batch_ids, batch_texts):
            all_paras[qid] = batch_results.get(qtxt, [])

        # Checkpoint
        with open(out_path, 'w') as f:
            json.dump(all_paras, f, indent=2, ensure_ascii=False)
        elapsed = time.time() - t0
        rate = (end) / elapsed if elapsed > 0 else 0
        eta = (n_remaining - end) / rate if rate > 0 else 0
        avg_len = sum(len(v) for v in all_paras.values()) / len(all_paras)
        print(f"  [{bench}] batch [{start+1}-{end}/{n_remaining}] "
              f"saved={len(all_paras)} avg_paras={avg_len:.2f} "
              f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    # Final stats
    n_complete = sum(1 for v in all_paras.values() if len(v) == NUM_PARAPHRASES)
    print(f"  [{bench}] ✓ done: {n_complete}/{len(all_paras)} have full {NUM_PARAPHRASES} paraphrases",
          flush=True)
    status.setdefault(STAGE_KEY, {})[bench] = "complete"


def main():
    print(f"=== : Llama-3.1-8B paraphraser regen ({MODEL_LABEL}) ===", flush=True)
    status = load_status()

    # If all benchmarks already complete, skip model load entirely
    stage_status = status.get(STAGE_KEY, {})
    if all(stage_status.get(b) == "complete" for b in BENCHMARKS):
        print("All benchmarks already complete. Nothing to do.", flush=True)
        return

    print(f"Loading model: {MODEL_LABEL}...", flush=True)
    llm, tokenizer = load_model(MODEL_LABEL, gpu_memory_utilization=0.85)
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
