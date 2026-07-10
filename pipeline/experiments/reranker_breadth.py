"""Cross-encoder reranker across all 9 settings.

Matches the existing HotpotQA/MiniLM reranker setup (Qwen3-Reranker-8B,
top-50 candidates → top-20 reranked) and extends to all 3 benchmarks × 3
embedding models.

Checkpointing:
  - Output: results/reranker_<bench>_<model>.json
  - On resume, skips any setting whose output JSON already exists.
  - Within a setting, writes incrementally every 200 queries.

Cross-encoder reranking on all 9 (benchmark, embedder) cells.
"""
import json, os, pickle, sys, gc, time
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "reranker_breadth"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")

TOP_K = 20
RERANK_K = 50
N_TEST_MAX = 1000   # subsample for tractability across 9 settings (was full ~6k)
MODEL_PATH = os.environ.get("QWEN3_RERANKER_PATH", "Qwen/Qwen3-Reranker-8B")

OPT_A, OPT_B, OPT_C, OPT_D, OPT_WS, OPT_DISC = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4
BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f: return json.load(f)
    return {}
def save_status(status):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, 'w') as f: json.dump(status, f, indent=2)


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)
def coverage(rset, gold):
    if not gold: return True
    return gold.issubset(rset)


def setup_setting(bench, model_key):
    """Load all per-setting data + compute SCER/Hybrid top-50 candidate pools."""
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")

    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paragraph_corpus.json")) as f: para_corpus = json.load(f)
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

    # Test split (same as run_baselines.py)
    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]
    if len(test_indices) > N_TEST_MAX:
        np.random.seed(123)
        test_indices = sorted(np.random.choice(test_indices, N_TEST_MAX, replace=False).tolist())

    test_data = []
    for qi in test_indices:
        q = questions[qi]
        gold = set(q.get("gold_para_ids", []))
        if not gold: continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]

        dr = para_indices[oi].tolist()[:RERANK_K]
        br = bm25_results.get(oi, [])[:RERANK_K]
        sm = sent_as_para[oi][:RERANK_K] if oi < len(sent_as_para) and sent_as_para[oi] else []
        psets = [set(para_indices[pi].tolist()[:TOP_K]) for pi in pis]
        st = float(np.mean([jaccard(set(dr[:TOP_K]), ps) for ps in psets])) if psets else 1.0

        # Hybrid top-50
        rrf = defaultdict(float)
        for r, i in enumerate(dr): rrf[i] += 1.0/(60+r+1)
        for r, i in enumerate(br): rrf[i] += 1.0/(60+r+1)
        hybrid_50 = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:RERANK_K]

        # SCER top-50
        w_d = OPT_A + OPT_B*st; w_b = OPT_C - OPT_D*st; w_s = OPT_WS
        w_pd = w_d*OPT_DISC; w_pb = w_b*OPT_DISC; w_ps = w_s*OPT_DISC
        votes = defaultdict(float)
        for r, i in enumerate(dr): votes[i] += w_d/(r+1)
        for r, i in enumerate(br): votes[i] += w_b/(r+1)
        for r, i in enumerate(sm): votes[i] += w_s/(r+1)
        for pi in pis:
            for r, i in enumerate(para_indices[pi].tolist()[:RERANK_K]): votes[i] += w_pd/(r+1)
            for r, i in enumerate(bm25_results.get(pi, [])[:RERANK_K]): votes[i] += w_pb/(r+1)
            ps = sent_as_para[pi][:RERANK_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
            for r, i in enumerate(ps): votes[i] += w_ps/(r+1)
        scer_50 = [i for i, _ in sorted(votes.items(), key=lambda x: -x[1])][:RERANK_K]

        test_data.append({
            "question": q["question"], "gold": gold,
            "dense_20": dr[:TOP_K], "hybrid_20": hybrid_50[:TOP_K],
            "scer_20": scer_50[:TOP_K],
            "hybrid_50": hybrid_50, "scer_50": scer_50,
        })
    return test_data, para_corpus


def load_reranker():
    print(f"Loading Qwen3-Reranker-8B from {MODEL_PATH}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16,
                                                  device_map="cuda:0", trust_remote_code=True)
    model.eval()
    yes_id = tok.convert_tokens_to_ids("yes")
    no_id = tok.convert_tokens_to_ids("no")
    return model, tok, yes_id, no_id


def rerank_batch(query, doc_ids, para_corpus, model, tok, yes_id, no_id, batch_size=16):
    pairs = []
    for did in doc_ids:
        if 0 <= did < len(para_corpus):
            pairs.append((did, para_corpus[did]["text"][:500]))
        else:
            pairs.append((did, ""))
    scores = []
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        prompts = [
            f"Is the following document relevant to the query?\nQuery: {query}\nDocument: {doc_text}\nAnswer:"
            for _, doc_text in pairs[start:end]
        ]
        inputs = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                     max_length=512).to("cuda:0")
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[:, -1, :]
            yes_s = logits[:, yes_id].float()
            no_s = logits[:, no_id].float()
            scores.extend((yes_s - no_s).cpu().numpy().tolist())
    ranked = sorted(zip(doc_ids, scores), key=lambda x: -x[1])
    return [did for did, _ in ranked[:TOP_K]]


def process_setting(bench, model_key, reranker_bundle, status):
    out_path = os.path.join(RESULTS_DIR, f"reranker_{bench}_{model_key}.json")
    tmp_path = out_path + ".tmp"

    if os.path.exists(out_path):
        with open(out_path) as f: existing = json.load(f)
        if existing.get("n_queries", 0) > 0 and "scer_rerank" in existing:
            print(f"  [{bench}/{model_key}] ✓ already complete (n={existing['n_queries']})", flush=True)
            status.setdefault(STAGE_KEY, {})[f"{bench}/{model_key}"] = "complete"
            return

    # Resume from partial if .tmp exists
    per_query = []
    if os.path.exists(tmp_path):
        with open(tmp_path) as f: per_query = json.load(f).get("per_query", [])
        print(f"  [{bench}/{model_key}] resume: {len(per_query)} queries already reranked", flush=True)

    print(f"  [{bench}/{model_key}] setup...", flush=True)
    test_data, para_corpus = setup_setting(bench, model_key)
    start_i = len(per_query)
    print(f"  [{bench}/{model_key}] test set: {len(test_data)} queries, starting at {start_i}", flush=True)

    model, tok, yes_id, no_id = reranker_bundle
    t0 = time.time()
    for i in range(start_i, len(test_data)):
        t = test_data[i]
        hr = rerank_batch(t["question"], t["hybrid_50"], para_corpus, model, tok, yes_id, no_id)
        sr = rerank_batch(t["question"], t["scer_50"], para_corpus, model, tok, yes_id, no_id)
        per_query.append({
            "question": t["question"],
            "dense_cov": float(coverage(set(t["dense_20"]), t["gold"])),
            "hybrid_cov": float(coverage(set(t["hybrid_20"]), t["gold"])),
            "scer_cov": float(coverage(set(t["scer_20"]), t["gold"])),
            "hybrid_rerank_cov": float(coverage(set(hr), t["gold"])),
            "scer_rerank_cov": float(coverage(set(sr), t["gold"])),
        })
        if (i + 1) % 50 == 0 or (i + 1) == len(test_data):
            with open(tmp_path, 'w') as f: json.dump({"per_query": per_query}, f)
            elapsed = time.time() - t0
            done = i + 1 - start_i
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            sr_avg = np.mean([p["scer_rerank_cov"] for p in per_query])
            hr_avg = np.mean([p["hybrid_rerank_cov"] for p in per_query])
            print(f"    [{i+1}/{len(test_data)}] H+R={hr_avg:.3f} S+R={sr_avg:.3f} "
                  f"rate={rate:.2f}q/s eta={eta/60:.1f}min", flush=True)

    # Aggregate + final save
    result = {
        "config": {"benchmark": bench, "model": model_key, "n_queries": len(per_query),
                   "rerank_pool_k": RERANK_K, "topk_out": TOP_K, "reranker": "Qwen3-Reranker-8B"},
        "dense": float(np.mean([p["dense_cov"] for p in per_query])),
        "hybrid": float(np.mean([p["hybrid_cov"] for p in per_query])),
        "scer_adaptive": float(np.mean([p["scer_cov"] for p in per_query])),
        "hybrid_rerank": float(np.mean([p["hybrid_rerank_cov"] for p in per_query])),
        "scer_rerank": float(np.mean([p["scer_rerank_cov"] for p in per_query])),
        "n_queries": len(per_query),
        "per_query": per_query,
    }
    with open(out_path, 'w') as f: json.dump(result, f, indent=2)
    if os.path.exists(tmp_path): os.remove(tmp_path)

    print(f"  [{bench}/{model_key}] ✓ done: dense={result['dense']:.3f} "
          f"hybrid={result['hybrid']:.3f} scer={result['scer_adaptive']:.3f} "
          f"H+R={result['hybrid_rerank']:.3f} S+R={result['scer_rerank']:.3f}", flush=True)
    status.setdefault(STAGE_KEY, {})[f"{bench}/{model_key}"] = "complete"


def main():
    print(f"=== : Qwen3-Reranker-8B on 9 settings (subsample to {N_TEST_MAX} per setting) ===", flush=True)
    status = load_status()
    stage_status = status.get(STAGE_KEY, {})

    settings = [(b, m) for b in BENCHMARKS for m in MODELS]
    pending = [(b, m) for b, m in settings if stage_status.get(f"{b}/{m}") != "complete"]
    if not pending:
        print("All 9 settings already complete. Nothing to do.", flush=True)
        return
    print(f"Pending: {len(pending)} of 9 settings.", flush=True)

    bundle = load_reranker()
    for bench, model_key in pending:
        print(f"\n--- {bench}/{model_key} ---", flush=True)
        process_setting(bench, model_key, bundle, status)

    # cleanup
    del bundle
    gc.collect(); torch.cuda.empty_cache()
    status[STAGE_KEY + "_overall"] = "complete"
    print("\n===  complete ===", flush=True)


if __name__ == '__main__':
    main()
