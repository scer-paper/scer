"""Build OpenAI Batch API input JSONL for paraphrase fidelity audit.

Samples 500 paraphrases per benchmark (1,500 total) and emits a JSONL file
ready for `openai.batches.create(...)` upload.

Output: outputs/paraphrase_audit_batch_input.jsonl + sample_index.json
        (the second file records which (qid, paraphrase_idx) each batch row
         corresponds to, so analyze_audit_results.py can join them back.)

Usage:
  python pipeline/gpt5.4_audit/build_paraphrase_batch.py
"""
import json, os, random

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "pipeline", "gpt5.4_audit", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "paraphrase_audit_batch_input.jsonl")
OUT_INDEX = os.path.join(OUT_DIR, "paraphrase_audit_sample_index.json")

MODEL = "gpt-5.4"          # change to "gpt-5.4-mini" for cheaper run
N_PER_BENCHMARK = 500
SEED = 42

PROMPT_TEMPLATE = """You are evaluating whether a paraphrase preserves the meaning of an original question.

ORIGINAL QUESTION: {original}
PARAPHRASE: {paraphrase}

Score the paraphrase on a 1-5 scale:
  5 = Fully equivalent. The paraphrase asks for exactly the same information; any correct answer to the original is a correct answer to the paraphrase and vice versa.
  4 = Minor wording difference that does not change the answer (e.g., "what is" vs "can you tell me what is"; switching tense).
  3 = Slightly narrower or broader scope; most correct answers still match.
  2 = Different focus (e.g., asks about a related entity, swaps a key attribute, or drops a constraint); some answers diverge.
  1 = Asks a substantively different question; correct answers would differ.

If the score is 1 or 2, classify the failure mode as one of:
  - "entity_substitution" (proper noun -> common noun, e.g., "Isles" -> "islands")
  - "scope_change"        (constraint dropped or added)
  - "subject_change"      (asks about something else)
  - "factual_error"       (introduces an incorrect assumption)
  - "other"

Output ONLY a single JSON object with no additional text:
{{"score": <1-5>, "failure_mode": <one of the above or null>, "rationale": "<one short sentence>"}}"""

def sample_paraphrases(bench, n, rng):
    data_dir = os.path.join(BASE, "data", bench)
    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)
    q_by_id = {q["id"]: q["question"] for q in questions}
    pool = []
    for qid, plist in paraphrases.items():
        if qid not in q_by_id: continue
        for pi, p in enumerate(plist):
            pool.append((qid, pi, q_by_id[qid], p))
    rng.shuffle(pool)
    return pool[:n]

def build_batch():
    rng = random.Random(SEED)
    index = {"model": MODEL, "n_per_benchmark": N_PER_BENCHMARK, "seed": SEED, "rows": []}
    with open(OUT_JSONL, "w") as f:
        row_n = 0
        for bench in ["hotpotqa_full", "fever_full", "squad_full"]:
            samples = sample_paraphrases(bench, N_PER_BENCHMARK, rng)
            print(f"  [{bench}] sampled {len(samples)} paraphrases")
            for qid, pi, original, paraphrase in samples:
                custom_id = f"paraphrase-{bench}-{qid}-{pi}"
                msg = PROMPT_TEMPLATE.format(original=original, paraphrase=paraphrase)
                req = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "messages": [{"role": "user", "content": msg}],
                        "response_format": {"type": "json_object"},
                        "max_completion_tokens": 1200,  # GPT-5.4 reasoning tokens count toward this; smoke test showed ~250 visible + reasoning overhead
                    },
                }
                f.write(json.dumps(req) + "\n")
                index["rows"].append({"custom_id": custom_id, "bench": bench, "qid": qid,
                                       "paraphrase_idx": pi, "original": original,
                                       "paraphrase": paraphrase})
                row_n += 1
    with open(OUT_INDEX, "w") as f: json.dump(index, f, indent=2)
    print(f"\n  Wrote {row_n} requests → {OUT_JSONL}")
    print(f"  Wrote sample index → {OUT_INDEX}")

if __name__ == "__main__":
    build_batch()
