"""SPLADE sparse retrieval baseline.

Indexes each corpus (HotpotQA 507K, FEVER 14K, SQuAD 2.0 20K) with a SPLADE model
and runs retrieval for all queries (originals + paraphrases) to produce a sparse-
retrieval alternative to BM25 + dense.

Uses naver/splade-cocondenser-ensembledistil (a strong open-source SPLADE)
via the SentenceTransformers SparseEncoder API. Downloads from HF on first run.

Checkpointing:
  - Per-benchmark output: data/<bench>/cache_splade/{para_embs.pkl, query_embs.pkl,
    splade_topk.json}
  - Skips any benchmark where splade_topk.json already exists.
  - Encoding is batched; on crash, partial pickles can be resumed manually but
    in practice we rebuild from scratch on a per-benchmark basis (encoding is fast).

SPLADE baseline (stronger sparse retriever than BM25).
"""
import json, os, pickle, sys, time, gc
import numpy as np
import torch

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "splade_baseline"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP_K = 20
BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"   # ~268MB SPLADE checkpoint
BATCH_SIZE_DOC = 64
BATCH_SIZE_QUERY = 128


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f: return json.load(f)
    return {}
def save_status(status):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, 'w') as f: json.dump(status, f, indent=2)


def load_splade():
    """Use raw transformers (SparseEncoder optional dep may not be installed).
    Returns model + tokenizer that expose a SPLADE-style sparse projection."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    print(f"Loading SPLADE: {MODEL_NAME}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda:0")
    model.eval()
    return model, tok


def encode_sparse(texts, model, tok, batch_size=32, max_length=256):
    """SPLADE encoding: log(1 + ReLU(MLM logits)) max-pooled over sequence.
    Returns dense float16 tensors (vocab-sized, but mostly zero)."""
    all_vecs = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        inputs = tok(batch, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model(**inputs).logits          # (B, L, V)
            # SPLADE: log(1+ReLU(x)) then max-pool over sequence (with attention mask)
            x = torch.log(1 + torch.relu(out))
            mask = inputs.attention_mask.unsqueeze(-1)
            x = x * mask                          # zero out padding
            sparse_vec = x.max(dim=1).values      # (B, V)
        all_vecs.append(sparse_vec.cpu())
    return torch.cat(all_vecs, dim=0)             # (N, V) float16


def topk_sparse_dot(query_vecs, doc_vecs, top_k=TOP_K, batch=128):
    """Sparse dot product: query_vecs (Q,V) @ doc_vecs (D,V).T → (Q, D) → top-k.
    Done in batches over queries to manage memory.

    Returns: (Q, top_k) int64 indices."""
    Q, V = query_vecs.shape
    D = doc_vecs.shape[0]
    doc_t = doc_vecs.t().to("cuda:0").to(torch.float16)   # (V, D)
    out = torch.zeros((Q, top_k), dtype=torch.int64)
    for start in range(0, Q, batch):
        end = min(start + batch, Q)
        qb = query_vecs[start:end].to("cuda:0").to(torch.float16)   # (b, V)
        scores = qb @ doc_t                                          # (b, D)
        _, idx = scores.topk(top_k, dim=1)
        out[start:end] = idx.cpu()
        del scores, qb, idx
    del doc_t
    torch.cuda.empty_cache()
    return out


def process_benchmark(bench, model, tok, status):
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, "cache_splade")
    os.makedirs(cache_dir, exist_ok=True)
    topk_path = os.path.join(cache_dir, "splade_topk.json")

    if os.path.exists(topk_path):
        print(f"  [{bench}] ✓ already complete", flush=True)
        status.setdefault(STAGE_KEY, {})[bench] = "complete"
        return

    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paragraph_corpus.json")) as f: para_corpus = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)

    # Build query text list (originals + paraphrases) using same id scheme as other caches
    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    n_queries = len(all_qt)

    para_texts = [p["text"] for p in para_corpus]
    n_paras = len(para_texts)
    print(f"  [{bench}] corpus={n_paras} paragraphs, {n_queries} query strings", flush=True)

    # Encode corpus
    t0 = time.time()
    para_vecs_path = os.path.join(cache_dir, "para_vecs.pt")
    if os.path.exists(para_vecs_path):
        print(f"  [{bench}] reload cached corpus encoding", flush=True)
        para_vecs = torch.load(para_vecs_path)
    else:
        print(f"  [{bench}] encoding corpus...", flush=True)
        para_vecs = encode_sparse(para_texts, model, tok, batch_size=BATCH_SIZE_DOC,
                                  max_length=256)
        torch.save(para_vecs, para_vecs_path)
        elapsed = time.time() - t0
        print(f"  [{bench}] encoded {n_paras} paras in {elapsed/60:.1f}min, "
              f"shape={tuple(para_vecs.shape)}", flush=True)

    # Encode queries
    t0 = time.time()
    query_vecs_path = os.path.join(cache_dir, "query_vecs.pt")
    if os.path.exists(query_vecs_path):
        print(f"  [{bench}] reload cached query encoding", flush=True)
        query_vecs = torch.load(query_vecs_path)
    else:
        print(f"  [{bench}] encoding queries...", flush=True)
        query_vecs = encode_sparse(all_qt, model, tok, batch_size=BATCH_SIZE_QUERY,
                                    max_length=128)
        torch.save(query_vecs, query_vecs_path)
        elapsed = time.time() - t0
        print(f"  [{bench}] encoded {n_queries} queries in {elapsed:.1f}s", flush=True)

    # Compute top-K via sparse dot product
    print(f"  [{bench}] computing top-{TOP_K} retrievals...", flush=True)
    t0 = time.time()
    topk_idx = topk_sparse_dot(query_vecs, para_vecs, top_k=TOP_K, batch=64)
    elapsed = time.time() - t0
    print(f"  [{bench}] retrieval done in {elapsed:.1f}s", flush=True)

    # Save as JSON keyed by query text idx for easy join with other pipeline outputs
    topk_dict = {str(i): topk_idx[i].tolist() for i in range(n_queries)}
    with open(topk_path, 'w') as f: json.dump(topk_dict, f)
    print(f"  [{bench}] ✓ saved {topk_path}", flush=True)

    status.setdefault(STAGE_KEY, {})[bench] = "complete"


def main():
    print("=== : SPLADE baseline ===", flush=True)
    status = load_status()
    stage_status = status.get(STAGE_KEY, {})
    if all(stage_status.get(b) == "complete" for b in BENCHMARKS):
        print("All benchmarks already complete.", flush=True); return

    model, tok = load_splade()
    for bench in BENCHMARKS:
        if stage_status.get(bench) == "complete":
            print(f"[{bench}] already complete, skipping.", flush=True); continue
        print(f"\n--- {bench} ---", flush=True)
        process_benchmark(bench, model, tok, status)

    del model
    gc.collect(); torch.cuda.empty_cache()
    status[STAGE_KEY + "_overall"] = "complete"
    print("\n===  complete ===", flush=True)


if __name__ == '__main__':
    main()
