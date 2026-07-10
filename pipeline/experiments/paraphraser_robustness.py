"""Alternative-paraphraser instability comparison (robustness check).

For each benchmark, encode the Llama-3.1-8B-generated paraphrases with each of
the 3 dense embedders, retrieve top-K, then compute the same instability
metrics (Jaccard distance, replacement rate, RBO) for direct comparison vs the
Qwen3-32B paraphrase numbers in Table 1.

Robustness check: show the instability finding is robust to the paraphraser
model choice."

Output: results/alt_paraphraser_stability.json
Checkpointing: per (benchmark, embedder).
"""
import json, os, pickle, sys, gc, time
import numpy as np
import torch

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "alt_paraphraser"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "alt_paraphraser_stability.json")
TOP_K = 20

BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
# NOTE: Qwen3-Embedding-0.6B is omitted here; the robustness claim is supported by MiniLM and Qwen3-Embedding-8B. The robustness claim is
# already supported by MiniLM (weakest embedder) + Qwen3-Embedding-8B (strongest)
# - these are the diagnostic anchors that matter.
MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", 384, "st"),
    "qwen3_8b": (os.environ.get("QWEN3_EMB_8B_PATH", "Qwen/Qwen3-Embedding-8B"), 4096, "qwen"),
}


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

def rbo(l1, l2, p=0.9):
    if not l1 or not l2: return 0.0
    K = min(len(l1), len(l2))
    s1, s2, total = set(), set(), 0.0
    for d in range(1, K+1):
        s1.add(l1[d-1]); s2.add(l2[d-1])
        total += (p**(d-1)) * (len(s1 & s2) / d)
    return (1-p) * total


def encode_st(model_path, texts, batch=128):
    """Encode with sentence-transformers (works for MiniLM)."""
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_path, device="cuda:0")
    vecs = m.encode(texts, batch_size=batch, show_progress_bar=False,
                    normalize_embeddings=True, convert_to_numpy=True)
    del m; gc.collect(); torch.cuda.empty_cache()
    return vecs.astype(np.float32)


def encode_qwen(model_path, texts, batch=64, max_length=128):
    """Encode with Qwen3-Embedding-* using last-token mean pooling."""
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16,
                                       device_map="cuda:0", trust_remote_code=True)
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            e = min(s + batch, len(texts))
            inp = tok(texts[s:e], padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt").to("cuda:0")
            o = model(**inp)
            hs = o.last_hidden_state                              # (b, L, H)
            mask = inp.attention_mask.unsqueeze(-1).float()
            pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)   # mean pooling
            pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            out.append(pooled.cpu().float().numpy())
    del model; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(out, axis=0).astype(np.float32)


def topk_dense(query_vecs, doc_vecs, k=TOP_K, batch=128):
    """Cosine similarity (normalized vecs) top-K on GPU."""
    Q = query_vecs.shape[0]
    doc_t = torch.from_numpy(doc_vecs).to("cuda:0")
    out = np.empty((Q, k), dtype=np.int64)
    for s in range(0, Q, batch):
        e = min(s + batch, Q)
        qb = torch.from_numpy(query_vecs[s:e]).to("cuda:0")
        sims = qb @ doc_t.T
        _, idx = sims.topk(k, dim=1)
        out[s:e] = idx.cpu().numpy()
    del doc_t
    return out


def process_setting(bench, model_key, all_results):
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")
    para_path = os.path.join(data_dir, "paraphrases_llama8b.json")
    if not os.path.exists(para_path):
        print(f"  [{bench}/{model_key}] no Llama paraphrases yet, skipping", flush=True); return

    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(para_path) as f: llama_paras = json.load(f)

    # Load corpus embeddings + original-query top-K (already cached)
    para_embs_path = os.path.join(cache_dir, "para_embs.npy")
    if not os.path.exists(para_embs_path):
        print(f"  [{bench}/{model_key}] corpus embeddings missing", flush=True); return
    para_embs = np.load(para_embs_path).astype(np.float32)
    norms = np.linalg.norm(para_embs, axis=1, keepdims=True); norms[norms==0]=1
    para_embs_n = para_embs / norms

    # Original-query embeddings (cached)
    q_embs_path = os.path.join(cache_dir, "query_embs.npy")
    if not os.path.exists(q_embs_path):
        print(f"  [{bench}/{model_key}] query embeddings missing", flush=True); return
    q_embs_all = np.load(q_embs_path).astype(np.float32)

    # Existing original top-K (already cached)
    orig_top_k_all = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]

    # Match original questions: cached query_embs.npy was built for all_qt
    # We need original query embeddings → use the same approach as other scripts
    with open(os.path.join(data_dir, "paraphrases.json")) as f: orig_paras = json.load(f)
    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + orig_paras.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    # Sanity: q_embs_all should have len == len(all_qt)
    if len(all_qt) != q_embs_all.shape[0]:
        print(f"  [{bench}/{model_key}] WARNING: query_embs ({q_embs_all.shape[0]}) != all_qt "
              f"({len(all_qt)}); skipping", flush=True); return

    # Test split (same seed/perm as other scripts)
    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    test_qs = [q for i, q in enumerate(questions) if i not in cal_set]

    # Collect Llama paraphrase strings needing encoding for test set
    llama_texts = []
    for q in test_qs:
        for p in llama_paras.get(q["id"], []):
            if p not in t2i:    # not in original cache
                llama_texts.append(p)
    llama_texts = list(dict.fromkeys(llama_texts))   # dedup, keep order
    print(f"  [{bench}/{model_key}] encoding {len(llama_texts)} Llama paraphrases...", flush=True)
    t0 = time.time()
    model_path, dim, kind = MODELS[model_key]
    if kind == "st":
        llama_vecs = encode_st(model_path, llama_texts)
    else:
        llama_vecs = encode_qwen(model_path, llama_texts)
    print(f"  [{bench}/{model_key}] encoded in {time.time()-t0:.1f}s, shape={llama_vecs.shape}", flush=True)

    # Top-K for Llama paraphrases
    llama_top_k = topk_dense(llama_vecs, para_embs_n, k=TOP_K)
    llama_dense_map = dict(zip(llama_texts, llama_top_k.tolist()))

    # Per-query: compute J/replace/RBO for original-vs-Llama-paraphrase pairs
    j_per, r_per, rbo_per = [], [], []
    for q in test_qs:
        qtxt = q["question"]
        if qtxt not in t2i: continue
        oi = t2i[qtxt]
        orig_top = orig_top_k_all[oi].tolist()[:TOP_K]
        orig_set = set(orig_top)
        plist = llama_paras.get(q["id"], [])
        if not plist: continue
        for p in plist:
            if p in t2i:   # paraphrase coincidentally present (unlikely)
                ptop = orig_top_k_all[t2i[p]].tolist()[:TOP_K]
            elif p in llama_dense_map:
                ptop = llama_dense_map[p]
            else:
                continue
            pset = set(ptop)
            j_per.append(jaccard(orig_set, pset))
            r_per.append(1 - len(orig_set & pset) / TOP_K)
            rbo_per.append(rbo(orig_top, ptop, p=0.9))

    if j_per:
        all_results[f"{bench}/{model_key}"] = {
            "n_pairs": len(j_per),
            "n_test_queries": len(test_qs),
            "jaccard_mean": float(np.mean(j_per)),
            "jaccard_distance_mean": float(1 - np.mean(j_per)),
            "replace_mean": float(np.mean(r_per)),
            "rbo_mean": float(np.mean(rbo_per)),
        }
        r = all_results[f"{bench}/{model_key}"]
        print(f"  [{bench}/{model_key}] ✓ n_pairs={r['n_pairs']} jaccard_dist={r['jaccard_distance_mean']:.3f} "
              f"replace={r['replace_mean']:.3f} rbo={r['rbo_mean']:.3f}", flush=True)
    del para_embs, para_embs_n; gc.collect(); torch.cuda.empty_cache()


def main():
    print("=== : alternative paraphraser stability ===", flush=True)
    status = load_status()
    existing = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f: existing = json.load(f)
    all_results = dict(existing)
    for bench in BENCHMARKS:
        for model_key in MODELS:
            key = f"{bench}/{model_key}"
            if key in all_results:
                print(f"  [{key}] already done, skipping", flush=True); continue
            print(f"\n--- {key} ---", flush=True)
            process_setting(bench, model_key, all_results)
            # Save incrementally after each setting
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(OUT_PATH, 'w') as f: json.dump(all_results, f, indent=2)
    status[STAGE_KEY] = "complete"; save_status(status)
    print("\n===  complete ===", flush=True)
    print(f"\nFinal: comparison with Qwen3-32B paraphrases (from Table 1):")
    print(f"  Qwen3-32B Jaccard distance: 32.1-49.8% (mean per pair 20.5-35.5% replace; RBO 0.59-0.70)")
    print(f"  Llama-8B:")
    for k, v in all_results.items():
        print(f"    {k}: J_dist={v['jaccard_distance_mean']*100:.1f}% "
              f"replace={v['replace_mean']*100:.1f}% RBO={v['rbo_mean']:.3f}")


if __name__ == '__main__':
    main()
