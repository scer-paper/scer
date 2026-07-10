"""Build OpenAI Batch API input JSONL for hallucination-grounding audit.

Samples 200 (question, evidence, answer) triples per (benchmark, embedder) setting
(200 × 9 = 1,800 total). The evidence is the EXACT top-5 retrieved by SCER or
Hybrid that was shown to Qwen3-32B at answer-generation time (same seed, same
construction as pipeline/run_baselines.py).

For each sampled query we emit 2 judgments (Hybrid answer, SCER answer) - half
of the 200 per-setting are Hybrid, half SCER, so per-method support rates are
directly comparable.

Outputs:
  outputs/hallucination_audit_batch_input.jsonl
  outputs/hallucination_audit_sample_index.json
"""
import json, os, pickle, sys, random
import numpy as np
from collections import defaultdict

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAG_DIR = os.path.join(BASE, "results", "rag_generations")
OUT_DIR = os.path.join(BASE, "pipeline", "gpt5.4_audit", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "hallucination_audit_batch_input.jsonl")
OUT_INDEX = os.path.join(OUT_DIR, "hallucination_audit_sample_index.json")

MODEL = "gpt-5.4"
N_PER_SETTING = 200      # 100 SCER + 100 Hybrid → 1800 total over 9 settings
EVIDENCE_TRUNC = 500     # chars per evidence doc (matches run_baselines.py)
TOP_K = 20               # retrieval depth (used by SCER aggregation)
TOP_K_CTX = 5            # evidence depth sent to generator
SEED = 7
# SCER weights - must match run_baselines.py
OPT_A, OPT_B, OPT_C, OPT_D, OPT_WS, OPT_DISC = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4

PROMPT_TEMPLATE = """You are evaluating whether a generated answer is supported by retrieved evidence.

QUESTION: {question}

RETRIEVED EVIDENCE:
{evidence}

GENERATED ANSWER: {answer}

Judge whether the answer is supported by the evidence:
  - "SUPPORTED"           = All factual claims in the answer can be directly verified from the evidence.
  - "PARTIALLY_SUPPORTED" = Some claims are supported, but at least one specific claim is missing from the evidence (extra info, not necessarily wrong).
  - "NOT_SUPPORTED"       = The answer's core claim cannot be found in the evidence; the model is guessing or generating from parametric knowledge.
  - "CONTRADICTED"        = The evidence directly contradicts the answer.

Ignore stylistic differences. Judge factual content only.

Output ONLY a single JSON object:
{{"verdict": "<SUPPORTED|PARTIALLY_SUPPORTED|NOT_SUPPORTED|CONTRADICTED>", "key_claim": "<the single most important factual claim being judged>", "rationale": "<one short sentence citing which evidence span supports or doesn't>"}}"""


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def reconstruct_top5(bench, model_key):
    """Reproduce Hybrid + SCER top-5 per query exactly as run_baselines.py does it.
    Returns dict qid -> {"hybrid_top5": [doc_ids], "scer_top5": [doc_ids]}."""
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")
    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)
    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)
    bm25_f = os.path.join(cache_dir, "bm25_pyserini.pkl")
    if not os.path.exists(bm25_f): bm25_f = os.path.join(cache_dir, "bm25.pkl")
    with open(bm25_f, 'rb') as f: bm25_results = pickle.load(f)

    # Same test-split logic as run_baselines.py
    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]
    if len(test_indices) > 2000:
        np.random.seed(123)
        test_indices = sorted(np.random.choice(test_indices, 2000, replace=False).tolist())

    qid_to_top5 = {}
    for qi in test_indices:
        q = questions[qi]
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]
        dr = para_indices[oi].tolist()[:TOP_K]
        br = bm25_results.get(oi, [])[:TOP_K]
        sm = sent_as_para[oi][:TOP_K] if oi < len(sent_as_para) and sent_as_para[oi] else []
        psets = [set(para_indices[pi].tolist()[:TOP_K]) for pi in pis]
        st = float(np.mean([jaccard(set(dr), ps) for ps in psets])) if psets else 1.0
        # Hybrid (RRF over dense + bm25, k0=60)
        rrf = defaultdict(float)
        for r, i in enumerate(dr): rrf[i] += 1.0 / (60 + r + 1)
        for r, i in enumerate(br): rrf[i] += 1.0 / (60 + r + 1)
        hybrid = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:TOP_K]
        # SCER (rank-discounted weighted voting)
        w_d = OPT_A + OPT_B * st; w_b = OPT_C - OPT_D * st
        w_pd = w_d * OPT_DISC; w_pb = w_b * OPT_DISC; w_ps = OPT_WS * OPT_DISC
        votes = defaultdict(float)
        for r, i in enumerate(dr): votes[i] += w_d / (r + 1)
        for r, i in enumerate(br): votes[i] += w_b / (r + 1)
        for r, i in enumerate(sm): votes[i] += OPT_WS / (r + 1)
        for pi in pis:
            for r, i in enumerate(para_indices[pi].tolist()[:TOP_K]): votes[i] += w_pd / (r + 1)
            for r, i in enumerate(bm25_results.get(pi, [])[:TOP_K]): votes[i] += w_pb / (r + 1)
            ps = sent_as_para[pi][:TOP_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
            for r, i in enumerate(ps): votes[i] += w_ps / (r + 1)
        scer = [i for i, _ in sorted(votes.items(), key=lambda x: -x[1])][:TOP_K]

        qid_to_top5[q["id"]] = {
            "hybrid_top5": hybrid[:TOP_K_CTX],
            "scer_top5": scer[:TOP_K_CTX],
        }
    return qid_to_top5


def build_evidence(doc_ids, para_lookup):
    parts = []
    for i, did in enumerate(doc_ids):
        if 0 <= did < len(para_lookup):
            text = para_lookup[did].get("text", "")[:EVIDENCE_TRUNC]
        else:
            text = "<missing>"
        parts.append(f"[{i+1}] {text}")
    return "\n\n".join(parts)


def find_rag(bench_short, model):
    for name in (f"rag_{bench_short}_full_{model}_extractive.json",
                  f"rag_{bench_short}_full_{model}.json"):
        p = os.path.join(RAG_DIR, name)
        if os.path.exists(p): return p
    return None


def build_batch():
    rng = random.Random(SEED)
    index = {"model": MODEL, "n_per_setting": N_PER_SETTING, "seed": SEED, "rows": []}
    row_n = 0
    with open(OUT_JSONL, "w") as f:
        for bench_short in ["hotpotqa", "fever", "squad"]:
            bench = f"{bench_short}_full"
            # Load paragraph corpus as a list-indexed array (doc IDs are integer indices)
            with open(os.path.join(BASE, "data", bench, "paragraph_corpus.json")) as fp:
                para_corpus = json.load(fp)
            for model in ["minilm", "qwen3_0.6b", "qwen3_8b"]:
                rag_path = find_rag(bench_short, model)
                if not rag_path:
                    print(f"  [{bench}/{model}] no RAG output, skipping"); continue
                with open(rag_path) as fp: rag = json.load(fp)
                pq = [p for p in rag.get("per_query", []) if p.get("scer_pred") and p.get("hybrid_pred")]
                if not pq:
                    print(f"  [{bench}/{model}] no per-query predictions"); continue

                # Reconstruct top-5 retrieval lists
                top5_lookup = reconstruct_top5(bench, model)

                # Filter to queries we have top-5 for
                eligible = [p for p in pq if p["id"] in top5_lookup]
                rng.shuffle(eligible)
                pool = eligible[:N_PER_SETTING // 2]   # 100 queries → 200 judgments

                for p_item in pool:
                    qid = p_item["id"]; question = p_item["question"]; gold = p_item.get("gold_answer", "")
                    tops = top5_lookup[qid]
                    for method in ["scer", "hybrid"]:
                        answer = p_item.get(f"{method}_pred", "").strip()
                        ev_ids = tops[f"{method}_top5"]
                        evidence = build_evidence(ev_ids, para_corpus)
                        custom_id = f"halluc-{bench}-{model}-{qid}-{method}"
                        msg = PROMPT_TEMPLATE.format(question=question, evidence=evidence, answer=answer)
                        req = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": MODEL,
                                "messages": [{"role": "user", "content": msg}],
                                "response_format": {"type": "json_object"},
                                "max_completion_tokens": 1500,
                            },
                        }
                        f.write(json.dumps(req) + "\n")
                        index["rows"].append({"custom_id": custom_id, "bench": bench, "model": model,
                                               "qid": qid, "method": method, "question": question,
                                               "gold": gold, "answer": answer})
                        row_n += 1
                print(f"  [{bench}/{model}] emitted {len(pool) * 2} rows ({len(pool)} queries × 2 methods)")

    with open(OUT_INDEX, "w") as f: json.dump(index, f, indent=2)
    print(f"\n  Wrote {row_n} requests → {OUT_JSONL}")
    print(f"  Wrote sample index → {OUT_INDEX}")


if __name__ == "__main__":
    build_batch()
