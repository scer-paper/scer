#!/usr/bin/env python3
"""
Step 2: Build dense (Qwen3-Embedding-8B) + BM25 indices for paragraph and sentence corpora.

Creates:
  - FAISS indices for paragraph and sentence views
  - BM25 indices (pickled) for paragraph and sentence views
  - Embedding numpy arrays for reuse

Usage:
  PYTHONUNBUFFERED=1 python3 pipeline/build_indices.py [--para-only] [--sent-only] [--bm25-only] [--dense-only]
"""

import json
import os
import pickle
import sys
import time
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
MODEL_PATH = os.environ.get("QWEN3_EMB_8B_PATH", "Qwen/Qwen3-Embedding-8B")


def build_bm25_index(corpus_texts, save_path):
    """Build BM25 index from corpus texts."""
    from rank_bm25 import BM25Okapi

    print("  Tokenizing for BM25...", flush=True)
    tokenized = [doc.lower().split() for doc in corpus_texts]

    print("  Building BM25 index...", flush=True)
    bm25 = BM25Okapi(tokenized)

    with open(save_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"  Saved BM25 index to {save_path}", flush=True)
    return bm25


def embed_corpus(corpus_texts, save_emb_path, batch_size=1024):
    """Embed corpus with Qwen3-Embedding-8B and save embeddings."""
    from vllm import LLM
    import faiss

    print("  Loading Qwen3-Embedding-8B...", flush=True)
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        runner="pooling",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        dtype="float16",
    )

    # Truncate to ~1500 chars to stay within 2048 tokens
    corpus_texts = [t[:1500] for t in corpus_texts]
    print(f"  Embedding {len(corpus_texts)} texts (batch_size={batch_size})...", flush=True)
    t0 = time.time()

    all_embeddings = []
    for start in range(0, len(corpus_texts), batch_size):
        end = min(start + batch_size, len(corpus_texts))
        batch = corpus_texts[start:end]
        outputs = llm.embed(batch)
        batch_embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(batch_embs, axis=1, keepdims=True)
        batch_embs = batch_embs / (norms + 1e-8)
        all_embeddings.append(batch_embs)

        elapsed = time.time() - t0
        rate = end / elapsed
        print(f"    [{end}/{len(corpus_texts)}] {rate:.0f} texts/s", flush=True)

    embeddings = np.concatenate(all_embeddings, axis=0)
    elapsed = time.time() - t0
    print(f"  Done: {len(embeddings)} embeddings, dim={embeddings.shape[1]}, {elapsed:.1f}s", flush=True)

    # Save embeddings
    np.save(save_emb_path, embeddings)
    print(f"  Saved to {save_emb_path}", flush=True)

    # Cleanup GPU
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)

    return embeddings


def build_faiss_index(emb_path, index_path):
    """Build FAISS index from saved embeddings."""
    import faiss

    print(f"  Loading embeddings from {emb_path}...", flush=True)
    embeddings = np.load(emb_path)
    print(f"  Shape: {embeddings.shape}", flush=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"  Saved FAISS index to {index_path}", flush=True)


def main():
    args = set(sys.argv[1:])
    do_para = not args or "--para-only" in args or "--dense-only" not in args
    do_sent = not args or "--sent-only" in args or "--dense-only" not in args
    do_bm25 = "--dense-only" not in args
    do_dense = "--bm25-only" not in args

    # Load corpora
    print("Loading paragraph corpus...", flush=True)
    with open(os.path.join(DATA_DIR, "paragraph_corpus.json")) as f:
        para_corpus = json.load(f)
    para_texts = [p["text"] for p in para_corpus]
    print(f"  {len(para_texts)} paragraphs", flush=True)

    print("Loading sentence corpus...", flush=True)
    with open(os.path.join(DATA_DIR, "sentence_corpus.json")) as f:
        sent_corpus = json.load(f)
    sent_texts = [s["text"] for s in sent_corpus]
    print(f"  {len(sent_texts)} sentences", flush=True)

    # BM25 indices
    if do_bm25:
        bm25_para_path = os.path.join(DATA_DIR, "bm25_paragraph.pkl")
        bm25_sent_path = os.path.join(DATA_DIR, "bm25_sentence.pkl")

        if not os.path.exists(bm25_para_path):
            print("\n=== Building paragraph BM25 ===", flush=True)
            build_bm25_index(para_texts, bm25_para_path)
        else:
            print(f"Paragraph BM25 exists, skipping", flush=True)

        if not os.path.exists(bm25_sent_path):
            print("\n=== Building sentence BM25 ===", flush=True)
            build_bm25_index(sent_texts, bm25_sent_path)
        else:
            print(f"Sentence BM25 exists, skipping", flush=True)

    # Dense indices
    if do_dense:
        para_emb_path = os.path.join(DATA_DIR, "embeddings_paragraph.npy")
        para_idx_path = os.path.join(DATA_DIR, "faiss_paragraph.index")
        sent_emb_path = os.path.join(DATA_DIR, "embeddings_sentence.npy")
        sent_idx_path = os.path.join(DATA_DIR, "faiss_sentence.index")

        if not os.path.exists(para_emb_path):
            print("\n=== Embedding paragraphs ===", flush=True)
            embed_corpus(para_texts, para_emb_path, batch_size=1024)
        else:
            print(f"Paragraph embeddings exist, skipping", flush=True)

        if not os.path.exists(para_idx_path):
            print("\n=== Building paragraph FAISS index ===", flush=True)
            build_faiss_index(para_emb_path, para_idx_path)
        else:
            print(f"Paragraph FAISS exists, skipping", flush=True)

        if not os.path.exists(sent_emb_path):
            print("\n=== Embedding sentences ===", flush=True)
            embed_corpus(sent_texts, sent_emb_path, batch_size=2048)
        else:
            print(f"Sentence embeddings exist, skipping", flush=True)

        if not os.path.exists(sent_idx_path):
            print("\n=== Building sentence FAISS index ===", flush=True)
            build_faiss_index(sent_emb_path, sent_idx_path)
        else:
            print(f"Sentence FAISS exists, skipping", flush=True)

    print("\n=== All indices built ===", flush=True)


if __name__ == '__main__':
    main()
