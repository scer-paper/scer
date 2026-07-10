"""Extend ablation table to Qwen3-0.6B + Qwen3-8B (CPU only).

Existing Table 3 (`tab:ablation`) shows 5 component ablations on MiniLM:
  full SCER, − paraphrases, − multi-view, − multi-retriever, − stability weights.

This script re-runs the same 5 variants on Qwen3-0.6B and Qwen3-8B using existing
cached retrievals. Plus 2 new diagnostic variants per the pivot:
  − all 18 sources but uniform RRF (== All-source RRF row from Table 4)
  − all 18 sources, uniform weights with paraphrase discount (== Uniform SCER)

Extended component-ablation table over all 9 (benchmark, embedder) cells.

Output: results/component_ablation.json
"""
import json, os, pickle, sys, numpy as np
from collections import defaultdict

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "ablation_extended"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "component_ablation.json")

TOP_K = 20
K_RRF = 60
A, B_, C_, D_, WS, GAMMA = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4

BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]


def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)
def cov(rset, gold): return 1.0 if gold.issubset(rset) else 0.0


def build_per_query(bench, model):
    data_dir = os.path.join(BASE, "data", bench); cache_dir = os.path.join(data_dir, f"cache_{model}")
    with open(os.path.join(data_dir, "questions.json")) as f: qs = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paras = json.load(f)
    all_qt = []; t2i = {}
    for q in qs:
        for t in [q["question"]] + paras.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)
    bm25_f = os.path.join(cache_dir, "bm25_pyserini.pkl")
    if not os.path.exists(bm25_f): bm25_f = os.path.join(cache_dir, "bm25.pkl")
    with open(bm25_f, 'rb') as f: bm25 = pickle.load(f)
    np.random.seed(42); n = len(qs); perm = np.random.permutation(n)
    cal = set(perm[:n//5].tolist())
    test = [i for i in range(n) if i not in cal]
    data = []
    for qi in test:
        q = qs[qi]
        gold = set(q.get("gold_para_ids", []))
        if not gold: continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paras.get(q["id"], []) if p in t2i]
        dr = para_indices[oi].tolist()[:TOP_K]
        br = bm25.get(oi, [])[:TOP_K]
        sm = sent_as_para[oi][:TOP_K] if oi < len(sent_as_para) and sent_as_para[oi] else []
        para_data = []
        for pi in pis:
            pdr = para_indices[pi].tolist()[:TOP_K]
            pbr = bm25.get(pi, [])[:TOP_K]
            psm = sent_as_para[pi][:TOP_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
            para_data.append((pdr, pbr, psm))
        st = float(np.mean([jaccard(set(dr), set(pd)) for pd, _, _ in para_data])) if para_data else 1.0
        data.append({"gold": gold, "dr": dr, "br": br, "sm": sm, "para": para_data, "st": st})
    return data


def scer_topk(q, use_para=True, use_sm=True, use_bm25=True, use_st=True, gamma=GAMMA):
    """SCER aggregation with selected components."""
    st = q["st"] if use_st else 0.5  # neutral st (0.5) when no st guidance
    w_d = A + B_ * st
    w_b = C_ - D_ * st if use_bm25 else 0.0
    w_s = WS if use_sm else 0.0
    votes = defaultdict(float)
    for r, d in enumerate(q["dr"]): votes[d] += w_d / (r + 1)
    for r, d in enumerate(q["br"]): votes[d] += w_b / (r + 1)
    for r, d in enumerate(q["sm"]): votes[d] += w_s / (r + 1)
    if use_para:
        for pdr, pbr, psm in q["para"]:
            for r, d in enumerate(pdr): votes[d] += w_d * gamma / (r + 1)
            for r, d in enumerate(pbr): votes[d] += w_b * gamma / (r + 1)
            for r, d in enumerate(psm): votes[d] += w_s * gamma / (r + 1)
    return set(sorted(votes, key=lambda x: -votes[x])[:TOP_K])


def uniform_rrf_topk(q):
    """Vanilla RRF over 18 sources."""
    scores = defaultdict(float)
    sources = [q["dr"], q["br"], q["sm"]]
    for pdr, pbr, psm in q["para"]: sources += [pdr, pbr, psm]
    for src in sources:
        for r, d in enumerate(src): scores[d] += 1.0 / (K_RRF + r + 1)
    return set(sorted(scores, key=lambda x: -scores[x])[:TOP_K])


def main():
    print("=== : extended ablation (all 3 models) ===", flush=True)
    results = {}
    for bench in BENCHMARKS:
        for model in MODELS:
            key = f"{bench}/{model}"
            print(f"  [{key}] loading data...", flush=True)
            data = build_per_query(bench, model)
            print(f"  [{key}] n_test={len(data)} computing variants...", flush=True)
            variants = {
                "scer_full":           lambda q: scer_topk(q, True, True, True, True),
                "minus_paraphrases":   lambda q: scer_topk(q, False, True, True, True),
                "minus_multi_view":    lambda q: scer_topk(q, True, False, True, True),
                "minus_multi_retr":    lambda q: scer_topk(q, True, True, False, True),
                "minus_stab_weights":  lambda q: scer_topk(q, True, True, True, False),
                "uniform_rrf_18src":   lambda q: uniform_rrf_topk(q),
                "uniform_scer_18src":  lambda q: scer_topk(q, True, True, True, False, gamma=GAMMA),  # uniform st => w_d=0.55, w_b=1.0, gamma=0.4
            }
            results[key] = {}
            for vname, fn in variants.items():
                covs = [cov(fn(q), q["gold"]) for q in data]
                results[key][vname] = {"cov20": float(np.mean(covs)), "n": len(covs)}
            row = results[key]
            base = row["scer_full"]["cov20"]
            print(f"  [{key}] scer_full={base:.3f} -para={row['minus_paraphrases']['cov20']:.3f} "
                  f"-mv={row['minus_multi_view']['cov20']:.3f} "
                  f"-mr={row['minus_multi_retr']['cov20']:.3f} "
                  f"-sw={row['minus_stab_weights']['cov20']:.3f} "
                  f"unif_rrf={row['uniform_rrf_18src']['cov20']:.3f}", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f: json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)
    # Update STATUS
    print("===  complete ===", flush=True)


if __name__ == '__main__':
    main()
