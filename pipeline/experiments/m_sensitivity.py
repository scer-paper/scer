"""m ∈ {1,3,5,10} paraphrase-scaling analysis.

For each value of m, recompute SCER Coverage@20 on HotpotQA/MiniLM
(the headline setting where SCER differences are largest).

For m=1,3,5: subsample from the existing data/<bench>/paraphrases.json (5/query).
For m=10:    use data/<bench>/paraphrases_m10.json ; requires
             dense embeddings for the 5 new paraphrases per query.

Cost-quality frontier sweep over m ∈ {1, 3, 5, 10} paraphrases.

Encoding for the 5 new paraphrases is done with MiniLM only (paragraph-level
dense, the easiest and cheapest path; the Qwen3 embedders are heavy and the
m-scaling story is identical across embedders qualitatively).

Output: results/m_scaling.json
Checkpointing: per-m, per-benchmark; resume if partial.
"""
import json, os, pickle, sys, time, gc
import numpy as np
from collections import defaultdict
import torch

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "m_scaling"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "m_scaling.json")

TOP_K = 20
OPT_A, OPT_B, OPT_C, OPT_D, OPT_WS, OPT_DISC = 0.3, 0.5, 1.5, 1.0, 0.2, 0.4
M_VALUES = [1, 3, 5, 10]
BENCHMARKS = ["hotpotqa_full"]   # m-scaling on headline setting only
MODEL = "minilm"
MINILM_PATH = "sentence-transformers/all-MiniLM-L6-v2"


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
def coverage(rset, gold):
    if not gold: return 0.0
    return 1.0 if gold.issubset(rset) else 0.0


def encode_minilm(texts, batch_size=128):
    """Encode strings with all-MiniLM-L6-v2; returns (N, 384) float32 numpy."""
    from sentence_transformers import SentenceTransformer
    print(f"  Loading MiniLM for encoding {len(texts)} strings...", flush=True)
    m = SentenceTransformer(MINILM_PATH, device="cuda:0")
    vecs = m.encode(texts, batch_size=batch_size, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True)
    del m; gc.collect(); torch.cuda.empty_cache()
    return vecs.astype(np.float32)


def dense_topk(query_vecs, doc_vecs, top_k=TOP_K, batch=256):
    """Cosine similarity (vecs are normalized) top-K via PyTorch GPU."""
    Q = query_vecs.shape[0]
    doc_t = torch.from_numpy(doc_vecs).to("cuda:0")
    out = np.empty((Q, top_k), dtype=np.int64)
    for s in range(0, Q, batch):
        e = min(s + batch, Q)
        qb = torch.from_numpy(query_vecs[s:e]).to("cuda:0")
        sims = qb @ doc_t.T
        _, idx = sims.topk(top_k, dim=1)
        out[s:e] = idx.cpu().numpy()
    del doc_t
    return out


def compute_scer_topk(dr, br, sm, para_dr, para_br, para_sm, st_value, top_k=TOP_K):
    """SCER aggregation: returns top_k doc ids."""
    w_d = OPT_A + OPT_B*st_value; w_b = OPT_C - OPT_D*st_value; w_s = OPT_WS
    w_pd = w_d*OPT_DISC; w_pb = w_b*OPT_DISC; w_ps = w_s*OPT_DISC
    votes = defaultdict(float)
    for r, d in enumerate(dr): votes[d] += w_d/(r+1)
    for r, d in enumerate(br): votes[d] += w_b/(r+1)
    for r, d in enumerate(sm): votes[d] += w_s/(r+1)
    for pd_, pb_, ps_ in zip(para_dr, para_br, para_sm):
        for r, d in enumerate(pd_): votes[d] += w_pd/(r+1)
        for r, d in enumerate(pb_): votes[d] += w_pb/(r+1)
        for r, d in enumerate(ps_): votes[d] += w_ps/(r+1)
    return [d for d, _ in sorted(votes.items(), key=lambda x: -x[1])][:top_k]


def process_benchmark(bench, model_key, status):
    """For each m, compute SCER coverage. m=10 may require new encoding."""
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")
    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: para5 = json.load(f)

    # m=10 source
    m10_path = os.path.join(data_dir, "paraphrases_m10.json")
    para10 = json.load(open(m10_path)) if os.path.exists(m10_path) else None

    # Existing per-query caches
    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)
    bm25_f = os.path.join(cache_dir, "bm25_pyserini.pkl")
    if not os.path.exists(bm25_f): bm25_f = os.path.join(cache_dir, "bm25.pkl")
    with open(bm25_f, 'rb') as f: bm25_results = pickle.load(f)

    # Build t2i map (text → index) for original + m=5 paraphrases
    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + para5.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)

    # For m=10: figure out which NEW strings need dense encoding
    new_texts = []
    if para10:
        for qid, plist in para10.items():
            for p in plist:
                if p not in t2i:
                    new_texts.append(p)
        new_texts = list(set(new_texts))
        print(f"  m=10: {len(new_texts)} new paraphrase strings need encoding", flush=True)

    # Encode new strings if any
    new_dense = {}
    para_corpus_size = para_indices.max() + 1
    if new_texts:
        # Load corpus embeddings (for similarity search)
        para_embs_path = os.path.join(cache_dir, "para_embs.npy")
        if not os.path.exists(para_embs_path):
            print(f"  ERROR: {para_embs_path} not found; cannot encode for m=10", flush=True)
            return None
        para_embs = np.load(para_embs_path)
        # Normalize (MiniLM produces normalized embeddings but be safe)
        norms = np.linalg.norm(para_embs, axis=1, keepdims=True); norms[norms==0]=1
        para_embs_n = para_embs / norms
        # Encode new query strings
        new_vecs = encode_minilm(new_texts)
        new_indices = dense_topk(new_vecs, para_embs_n, top_k=TOP_K, batch=256)
        new_dense = dict(zip(new_texts, new_indices.tolist()))
        del para_embs, para_embs_n; gc.collect(); torch.cuda.empty_cache()

    # Test split
    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]

    def get_dense(text):
        if text in t2i: return para_indices[t2i[text]].tolist()[:TOP_K]
        if text in new_dense: return new_dense[text]
        return []
    def get_bm25(text):
        if text in t2i: return bm25_results.get(t2i[text], [])[:TOP_K]
        return []
    def get_sm(text):
        if text in t2i and t2i[text] < len(sent_as_para):
            return sent_as_para[t2i[text]][:TOP_K]
        return []

    results = {}
    for m in M_VALUES:
        para_source = para10 if m == 10 else para5
        if m == 10 and not para10:
            print(f"  m=10: paraphrases_m10.json not present; skipping", flush=True)
            continue
        covs, sts = [], []
        for qi in test_indices:
            q = questions[qi]
            gold = set(q.get("gold_para_ids", []))
            if not gold: continue
            qtxt = q["question"]
            all_paras = para_source.get(q["id"], [])[:m]
            if len(all_paras) < min(m, 1): continue

            # Per-source for original query
            dr = get_dense(qtxt); br = get_bm25(qtxt); sm = get_sm(qtxt)
            if not dr: continue

            # Per-source for each paraphrase (use only first m)
            para_dr = [get_dense(p) for p in all_paras]
            para_br = [get_bm25(p) for p in all_paras]
            para_sm = [get_sm(p) for p in all_paras]

            # st(q) computed from dense overlaps; some paraphrases (newly encoded for m=10) have real top-K
            j_vals = [jaccard(set(dr), set(pd)) for pd in para_dr if pd]
            st = float(np.mean(j_vals)) if j_vals else 1.0
            sts.append(st)
            scer_top = set(compute_scer_topk(dr, br, sm, para_dr, para_br, para_sm, st))
            covs.append(coverage(scer_top, gold))
        if covs:
            results[m] = {
                "n_test": len(covs),
                "scer_cov20": float(np.mean(covs)),
                "mean_st": float(np.mean(sts)),
            }
            print(f"  m={m:2d}: n={len(covs)} scer_cov20={np.mean(covs):.4f} mean_st={np.mean(sts):.3f}", flush=True)
    return results


def main():
    print("=== : m-scaling analysis ===", flush=True)
    status = load_status()
    all_results = {}
    for bench in BENCHMARKS:
        print(f"\n--- {bench}/{MODEL} ---", flush=True)
        res = process_benchmark(bench, MODEL, status)
        if res is not None:
            all_results[f"{bench}/{MODEL}"] = res
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)
    status[STAGE_KEY] = "complete"; save_status(status)
    print("===  complete ===", flush=True)


if __name__ == '__main__':
    main()
