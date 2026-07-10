"""Main retrieval-baseline runner - computes Cov/MRR/nDCG/Rec for all baselines from cached retrievals.

Covers:
  (1) - All-source RRF baseline (vanilla RRF over all 18 sources)
  (2) - Uniform SCER (all weights = 1.0, no stability-guided weighting)
  (3) - Coverage@5, Recall@5, MRR, nDCG@20 for every method
  (4) - BM25-only Coverage@20 column
  (5) - Cost reporting (# retrievals per method)
  (6) - Stratified Coverage by st(q) for Qwen3-8B (extension of Table 6)

Methods scored (6):
  dense, bm25, hybrid (RRF dense+bm25), all_source_rrf (18-source vanilla RRF),
  uniform_scer (18-source SCER weights all = 1.0), scer (current paper config)

Output: results/main_baselines.json
"""
import json, os, pickle, sys, numpy as np
from collections import defaultdict
import math

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(OUT_DIR, "main_baselines.json")

TOP_K = 20
K_RRF = 60
# SCER grid-search-tuned weights
OPT_A, OPT_B, OPT_C, OPT_D, OPT_WS, OPT_DISC = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4

BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]

# ---------- metrics ----------
def coverage_at_k(top_k, gold):
    """All gold in top-k? (strict, paper's Coverage)"""
    return 1.0 if gold.issubset(set(top_k)) else 0.0

def recall_at_k(top_k, gold):
    """Fraction of gold paragraphs found in top-k."""
    if not gold: return 0.0
    return len(set(top_k) & gold) / len(gold)

def mrr(ranked, gold):
    """Reciprocal of rank of first gold paragraph; 0 if none."""
    for i, d in enumerate(ranked, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked, gold, k):
    """Standard nDCG@k with binary relevance."""
    dcg = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in gold:
            dcg += 1.0 / math.log2(i + 1)
    n_gold_in_topk = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_gold_in_topk + 1))
    return dcg / idcg if idcg > 0 else 0.0

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

# ---------- ranking constructors ----------
def rrf_aggregate(sources, top_n=TOP_K):
    """Vanilla RRF over a list of ranked lists."""
    scores = defaultdict(float)
    for src in sources:
        for r, d in enumerate(src):
            scores[d] += 1.0 / (K_RRF + r + 1)
    ranked = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]
    return ranked[:top_n]

def uniform_scer_aggregate(orig_sources, para_sources_per_para, discount=OPT_DISC, top_n=TOP_K):
    """SCER with all weights = 1.0 (uniform), paraphrase discount still applied (= OPT_DISC).
    Set discount=1.0 for fully uniform (no discount). Default mirrors current SCER discount."""
    votes = defaultdict(float)
    for src in orig_sources:
        for r, d in enumerate(src):
            votes[d] += 1.0 / (r + 1)
    for para_sources in para_sources_per_para:
        for src in para_sources:
            for r, d in enumerate(src):
                votes[d] += discount * 1.0 / (r + 1)
    return [d for d, _ in sorted(votes.items(), key=lambda x: -x[1])][:top_n]

def scer_aggregate(orig_sources, para_sources_per_para, st_value, top_n=TOP_K):
    """Current paper SCER: stability-weighted rank-discounted voting."""
    w_d = OPT_A + OPT_B * st_value
    w_b = OPT_C - OPT_D * st_value
    w_s = OPT_WS
    w_pd = w_d * OPT_DISC; w_pb = w_b * OPT_DISC; w_ps = w_s * OPT_DISC
    dr, br, sm = orig_sources  # dense, bm25, sent_mapped
    votes = defaultdict(float)
    for r, d in enumerate(dr): votes[d] += w_d / (r + 1)
    for r, d in enumerate(br): votes[d] += w_b / (r + 1)
    for r, d in enumerate(sm): votes[d] += w_s / (r + 1)
    for para_sources in para_sources_per_para:
        pdr, pbr, psm = para_sources
        for r, d in enumerate(pdr): votes[d] += w_pd / (r + 1)
        for r, d in enumerate(pbr): votes[d] += w_pb / (r + 1)
        for r, d in enumerate(psm): votes[d] += w_ps / (r + 1)
    return [d for d, _ in sorted(votes.items(), key=lambda x: -x[1])][:top_n]

# ---------- main loop ----------
def run_setting(bench, model_key):
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")

    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)

    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]

    bm25_f = os.path.join(cache_dir, "bm25_pyserini.pkl")
    if not os.path.exists(bm25_f): bm25_f = os.path.join(cache_dir, "bm25.pkl")
    with open(bm25_f, 'rb') as f: bm25_results = pickle.load(f)

    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)

    # Test split: 4/5ths, seed=42
    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]
    n_test = len(test_indices)

    methods = ["dense", "bm25", "hybrid", "all_source_rrf", "uniform_scer", "scer"]
    metrics_init = lambda: {"cov5": [], "cov20": [], "rec5": [], "rec20": [], "mrr": [], "ndcg20": []}
    per_method = {m: metrics_init() for m in methods}

    # Also collect per-query st for stratified analysis ((6))
    per_query_st = []
    per_query_scer_cov20 = []

    for qi in test_indices:
        q = questions[qi]
        gold = set(q.get("gold_para_ids", []))
        if not gold: continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]

        # original-query sources
        dr = para_indices[oi].tolist()[:TOP_K]
        br = bm25_results.get(oi, [])[:TOP_K]
        sm = sent_as_para[oi][:TOP_K] if oi < len(sent_as_para) and sent_as_para[oi] else []

        # paraphrase sources
        para_dr = [para_indices[pi].tolist()[:TOP_K] for pi in pis]
        para_br = [bm25_results.get(pi, [])[:TOP_K] for pi in pis]
        para_sm = [sent_as_para[pi][:TOP_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
                   for pi in pis]
        para_triples = [(d, b, s) for d, b, s in zip(para_dr, para_br, para_sm)]

        # stability (dense, paraphrases)
        st = float(np.mean([jaccard(set(dr), set(pd)) for pd in para_dr])) if para_dr else 1.0
        per_query_st.append(st)

        # method rankings
        m_rank = {}
        m_rank["dense"] = dr
        m_rank["bm25"] = br
        m_rank["hybrid"] = rrf_aggregate([dr, br])
        all_source_lists = [dr, br, sm] + [x for trip in para_triples for x in trip]
        m_rank["all_source_rrf"] = rrf_aggregate(all_source_lists)
        m_rank["uniform_scer"] = uniform_scer_aggregate(
            orig_sources=[dr, br, sm], para_sources_per_para=para_triples)
        m_rank["scer"] = scer_aggregate(
            orig_sources=(dr, br, sm), para_sources_per_para=para_triples, st_value=st)

        for method, ranked in m_rank.items():
            top5 = ranked[:5]; top20 = ranked[:TOP_K]
            per_method[method]["cov5"].append(coverage_at_k(top5, gold))
            per_method[method]["cov20"].append(coverage_at_k(top20, gold))
            per_method[method]["rec5"].append(recall_at_k(top5, gold))
            per_method[method]["rec20"].append(recall_at_k(top20, gold))
            per_method[method]["mrr"].append(mrr(ranked, gold))
            per_method[method]["ndcg20"].append(ndcg_at_k(ranked, gold, TOP_K))

        per_query_scer_cov20.append(per_method["scer"]["cov20"][-1])

    # Aggregate
    agg = {}
    for m in methods:
        agg[m] = {k: float(np.mean(v)) for k, v in per_method[m].items()}
        agg[m]["n_test"] = n_test

    # Stratified ((6)): split at st(q) = 0.5
    sts = np.array(per_query_st)
    covs = np.array(per_query_scer_cov20)
    unstable_mask = sts < 0.5
    stable_mask = ~unstable_mask
    strat = {
        "st_lt_0.5": {"n": int(unstable_mask.sum()),
                       "mean_st": float(sts[unstable_mask].mean()) if unstable_mask.any() else None,
                       "scer_cov20": float(covs[unstable_mask].mean()) if unstable_mask.any() else None},
        "st_ge_0.5": {"n": int(stable_mask.sum()),
                       "mean_st": float(sts[stable_mask].mean()) if stable_mask.any() else None,
                       "scer_cov20": float(covs[stable_mask].mean()) if stable_mask.any() else None},
        "overall": {"n": n_test, "mean_st": float(sts.mean()),
                    "scer_cov20": float(covs.mean())},
    }
    return {"methods": agg, "stratified_scer": strat}

# Run all 9 settings
results = {}
for bench in BENCHMARKS:
    for model in MODELS:
        key = f"{bench}/{model}"
        print(f"=== {key} ===", flush=True)
        results[key] = run_setting(bench, model)
        # quick log
        for m in ["dense", "bm25", "hybrid", "all_source_rrf", "uniform_scer", "scer"]:
            a = results[key]["methods"][m]
            print(f"  {m:<16s} cov5={a['cov5']:.3f} cov20={a['cov20']:.3f} "
                  f"rec20={a['rec20']:.3f} mrr={a['mrr']:.3f} ndcg20={a['ndcg20']:.3f}",
                  flush=True)
        strat = results[key]["stratified_scer"]
        print(f"  stratified: st<0.5 n={strat['st_lt_0.5']['n']} cov20={strat['st_lt_0.5']['scer_cov20']} "
              f"| st>=0.5 n={strat['st_ge_0.5']['n']} cov20={strat['st_ge_0.5']['scer_cov20']}",
              flush=True)

# Cost reporting ((5)): # retrievals per method per query
cost = {
    "dense": {"retrievals_per_query": 1, "uses_paraphrases": False},
    "bm25": {"retrievals_per_query": 1, "uses_paraphrases": False},
    "hybrid": {"retrievals_per_query": 2, "uses_paraphrases": False, "notes": "1 dense + 1 BM25"},
    "all_source_rrf": {"retrievals_per_query": 18, "uses_paraphrases": True,
                       "notes": "6 queries (original + 5 paraphrases) × 3 retrievers (dense, BM25, sent_mapped)"},
    "uniform_scer": {"retrievals_per_query": 18, "uses_paraphrases": True,
                     "notes": "same 18 sources; weights = 1.0 (paraphrase discount = 0.4)"},
    "scer": {"retrievals_per_query": 18, "uses_paraphrases": True,
             "notes": "same 18 sources; stability-guided weights (Eqs. 4-5)"},
}

output = {
    "metadata": {
        "weights": {"OPT_A": OPT_A, "OPT_B": OPT_B, "OPT_C": OPT_C, "OPT_D": OPT_D,
                    "OPT_WS": OPT_WS, "OPT_DISC": OPT_DISC},
        "top_k": TOP_K, "k_rrf": K_RRF,
        "split_seed": 42, "split_cal_frac": 0.2,
    },
    "cost_per_method": cost,
    "results_per_setting": results,
}
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {OUT_PATH}", flush=True)
