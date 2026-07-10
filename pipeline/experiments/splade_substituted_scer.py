"""SPLADE baseline integration (CPU-only post-processing).

Reads SPLADE top-K outputs from 
(data/<bench>/cache_splade/splade_topk.json) and computes:
  1. SPLADE-only Coverage@20 (single-source baseline)
  2. SPLADE + Dense Hybrid via RRF
  3. SCER variant that swaps BM25 for SPLADE in the 18-source pool

This isolates whether the SCER mitigation story depends on BM25 specifically or
generalizes to other sparse retrievers - the retriever-agnostic claim.

Output: results/splade_baseline.json
"""
import json, os, pickle, sys
import numpy as np
from collections import defaultdict

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "splade_substituted_scer"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "splade_baseline.json")

TOP_K = 20
OPT_A, OPT_B, OPT_C, OPT_D, OPT_WS, OPT_DISC = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4
BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f: return json.load(f)
    return {}
def save_status(s):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, 'w') as f: json.dump(s, f, indent=2)


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)
def coverage(rset, gold): return 1.0 if gold.issubset(rset) else 0.0


def process(bench, model_key, splade_dir):
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")
    splade_topk_path = os.path.join(splade_dir, "splade_topk.json")
    if not os.path.exists(splade_topk_path):
        print(f"  [{bench}] no SPLADE output yet, skipping", flush=True); return None

    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)
    with open(splade_topk_path) as f: splade = json.load(f)

    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)

    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)

    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]

    cov_splade, cov_splade_dense, cov_scer_splade = [], [], []
    for qi in test_indices:
        q = questions[qi]
        gold = set(q.get("gold_para_ids", []))
        if not gold: continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]
        if str(oi) not in splade: continue
        dr = para_indices[oi].tolist()[:TOP_K]
        sm = sent_as_para[oi][:TOP_K] if oi < len(sent_as_para) and sent_as_para[oi] else []
        spl = splade[str(oi)][:TOP_K]

        # 1. SPLADE-only top-20
        cov_splade.append(coverage(set(spl), gold))

        # 2. SPLADE+Dense Hybrid via RRF
        rrf = defaultdict(float)
        for r, i in enumerate(dr): rrf[i] += 1.0/(60+r+1)
        for r, i in enumerate(spl): rrf[i] += 1.0/(60+r+1)
        sd_hybrid = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:TOP_K]
        cov_splade_dense.append(coverage(set(sd_hybrid), gold))

        # 3. SCER with SPLADE replacing BM25 (uses same stability + weighting)
        psets = [set(para_indices[pi].tolist()[:TOP_K]) for pi in pis]
        st = float(np.mean([jaccard(set(dr), ps) for ps in psets])) if psets else 1.0
        w_d = OPT_A + OPT_B*st; w_b = OPT_C - OPT_D*st; w_s = OPT_WS
        w_pd = w_d*OPT_DISC; w_pb = w_b*OPT_DISC; w_ps = w_s*OPT_DISC
        votes = defaultdict(float)
        for r, i in enumerate(dr): votes[i] += w_d/(r+1)
        for r, i in enumerate(spl): votes[i] += w_b/(r+1)
        for r, i in enumerate(sm): votes[i] += w_s/(r+1)
        for pi in pis:
            for r, i in enumerate(para_indices[pi].tolist()[:TOP_K]): votes[i] += w_pd/(r+1)
            if str(pi) in splade:
                for r, i in enumerate(splade[str(pi)][:TOP_K]): votes[i] += w_pb/(r+1)
            ps = sent_as_para[pi][:TOP_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
            for r, i in enumerate(ps): votes[i] += w_ps/(r+1)
        scer_splade = [i for i, _ in sorted(votes.items(), key=lambda x: -x[1])][:TOP_K]
        cov_scer_splade.append(coverage(set(scer_splade), gold))

    return {
        "n": len(cov_splade),
        "splade_cov20": float(np.mean(cov_splade)),
        "splade_plus_dense_cov20": float(np.mean(cov_splade_dense)),
        "scer_splade_cov20": float(np.mean(cov_scer_splade)),
    }


def main():
    print("=== : SPLADE baseline integration ===", flush=True)
    results = {}
    for bench in BENCHMARKS:
        splade_dir = os.path.join(BASE, "data", bench, "cache_splade")
        if not os.path.exists(os.path.join(splade_dir, "splade_topk.json")):
            print(f"  [{bench}] SPLADE not ready", flush=True); continue
        for model in MODELS:
            print(f"  [{bench}/{model}] computing...", flush=True)
            r = process(bench, model, splade_dir)
            if r: results[f"{bench}/{model}"] = r
            print(f"    splade={r['splade_cov20']:.3f} splade+dense={r['splade_plus_dense_cov20']:.3f} "
                  f"scer_w_splade={r['scer_splade_cov20']:.3f}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)
    save_status({**load_status(), STAGE_KEY: "complete"})


if __name__ == '__main__':
    main()
