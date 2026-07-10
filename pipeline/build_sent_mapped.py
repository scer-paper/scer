"""Build the sentence-mapped retrieval channel cache.
Rebuilds query_embs, faiss_para, bm25, sent_mapped for each model
by reusing old embeddings and computing only the 8 new ones.
Usage: CUDA_VISIBLE_DEVICES=X python build_sent_mapped.py <model_key>
  model_key: minilm, qwen3_0.6b, qwen3_8b"""
import json, os, sys, pickle, numpy as np, time

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(BASE, "data", "hotpotqa_full")

model_key = sys.argv[1]
cache_dir = os.path.join(DATA, f"cache_{model_key}")

print(f"=== Patching {model_key} ===", flush=True)

# Load data
with open(os.path.join(DATA, "questions.json")) as f: questions = json.load(f)
with open(os.path.join(DATA, "paragraph_corpus.json")) as f: para_corpus = json.load(f)
with open(os.path.join(DATA, "sentence_corpus.json")) as f: sent_corpus = json.load(f)
with open(os.path.join(DATA, "paraphrases.json")) as f: paraphrases = json.load(f)

# Build NEW text_to_idx
new_qt = []
new_t2i = {}
for q in questions:
    for t in [q["question"]] + paraphrases.get(q["id"], []):
        if t not in new_t2i:
            new_t2i[t] = len(new_qt)
            new_qt.append(t)

new_size = len(new_qt)
print(f"  New query list size: {new_size}", flush=True)

# Load old arrays
old_qe = np.load(os.path.join(cache_dir, "query_embs.npy"))
old_faiss = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
bm25_path = os.path.join(cache_dir, "bm25_pyserini.pkl")
if not os.path.exists(bm25_path):
    bm25_path = os.path.join(cache_dir, "bm25.pkl")
with open(bm25_path, "rb") as f:
    old_bm25 = pickle.load(f)
with open(os.path.join(cache_dir, "sent_mapped.pkl"), "rb") as f:
    old_sent = pickle.load(f)

old_size = old_qe.shape[0]
emb_dim = old_qe.shape[1]
print(f"  Old size: {old_size}, New size: {new_size}, Dim: {emb_dim}", flush=True)

# Build OLD text list to create old_text -> old_index mapping
# We need to figure out which text was at which old index.
# The old text list was built the same way but with old paraphrases.
# Since we only changed 6 queries' paraphrases, we can reconstruct.

# The old paraphrases had duplicates. The unique texts were assigned indices.
# We need the OLD paraphrases to rebuild. But we overwrote paraphrases.json.
# Instead, we can identify old texts by checking which NEW texts are missing from old.

# Strategy: iterate new_qt. For each text, check if it existed in the old run.
# If old_qt[i] == new_qt[i] for most i, then the mapping is simple where they match.
# Where they differ (due to insertions), we need to handle.

# Better approach: build old_qt from new_qt by removing the 8 new texts.
# The 8 new texts are the ones whose indices shifted everything.

# Actually, simplest: rebuild old text_to_idx using the old paraphrases.
# We know which 6 queries changed. We can reconstruct old paraphrases from
# the new ones by reverting the fix.

# Even simpler: the old embedding at old_index i corresponds to old_qt[i].
# If we can build old_qt, we can map: old_text -> old_embedding.
# Then for each new_qt[j], if it existed in old_qt, copy the embedding.
# If not, compute a new embedding.

# To build old_qt, we need old paraphrases. But we have a hint:
# the old had duplicates. The new has no duplicates. The fix script
# kept unique paras and replaced duplicates with new texts.
#
# Fix details (from earlier output):
# Query 5a7b3b4f: old=[A,B,C,A,B] -> new=[C,B,A,D,E] (kept C,B,A; added D,E)
# But actually the fix script did: new_paras = list(set(old_paras)) + generated
# The set() is unordered so we can't easily reconstruct.
#
# Alternative: Just check text membership. If a text in new_qt matches one in
# old embedding (same text, we just need to find its old index).

# Fastest approach: build a mapping from text -> old_index by scanning all
# questions with the old (pre-fix) paraphrases to get old_qt.

# BUT we don't have the old paraphrases anymore (overwritten).
# So we need a different approach.

# PRACTICAL APPROACH:
# 1. The old query_embs has embeddings for texts in some order.
# 2. We can't map old indices to texts without the old text list.
# 3. Instead: re-embed ALL new_qt texts that need new embeddings.
#    For the ones that already had embeddings, the positions may have shifted.
#
# Actually, the cleanest solution: re-embed just the 8 new texts,
# then rebuild the arrays in the new order.
#
# Since we can't map old text -> old index without the old paraphrases,
# let's use a different approach:
# - The old and new text lists differ only at positions where new paraphrases
#   were inserted. Before the first insertion point, indices are identical.
# - We can compute the mapping by comparing.

# Let me try: iterate through questions with NEW paraphrases, track which texts
# would have been in the OLD list too (i.e., texts that aren't newly generated).

# The 8 new texts (newly generated paraphrases):
NEW_TEXTS = set()
fix_ids = ['5a7b3b4f5542992d025e67ac', '5a8c4ebf554299585d9e365d', '5ae1fd995542997283cd2313',
           '5a84bda45542992a431d1a96', '5a7720d05542993569682cde', '5a7dbc175542995ed0d1666b']

# We know from the fix script output exactly which texts are new:
new_paras_by_query = {
    '5a7b3b4f5542992d025e67ac': [
        "Who was the elder sibling of the dethroned monarch depicted in the Henriad?",
        "Which senior brother belonged to the king who was deposed in the Henriad?"
    ],
    '5a8c4ebf554299585d9e365d': [
        "Which of the earliest religious doubters had the most devoted disciple?"
    ],
    '5ae1fd995542997283cd2313': [
        "Which city in Suffolk is the hometown of the English rock band that released the single \"Is It Just Me?\"?",
        "What Suffolk city is the origin of the English rock band that put out the single \"Is It Just Me?"
    ],
    '5a84bda45542992a431d1a96': [
        "Which professor has an August birthdate?"
    ],
    '5a7720d05542993569682cde': [
        "Which actor and director from the United States is the child of Debbie Reynolds and Eddie Fisher?"
    ],
    '5a7dbc175542995ed0d1666b': [
        "Who was the Russian Marxist revolutionary, born in 1875, who was connected with the Vpered organizat"
    ],
}

# Actually, let's just identify new texts by checking: any text in new_qt
# that doesn't appear in the old run's text list. Since we can't reconstruct
# the old list, let's use a trick: the old embeddings were built for texts
# that existed BEFORE the fix. Any text in paraphrases.json that was NOT
# in the old paraphrases.json is new.
#
# We can identify these by: the fix script replaced duplicates with new unique texts.
# The new texts are the ones that appear in paraphrases[qid] but would NOT have
# appeared in the old (pre-fix) paraphrases.
#
# Since we can't check old file, let's use the simplest possible approach:
# RE-EMBED EVERYTHING. It's only 44430 queries with a small model.

print("  Re-embedding all query texts (safest approach)...", flush=True)

if model_key == "minilm":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    model = model.to("cuda")

    BATCH = 512
    new_qe = np.zeros((new_size, emb_dim), dtype=np.float32)
    for start in range(0, new_size, BATCH):
        end = min(start + BATCH, new_size)
        embs = model.encode(new_qt[start:end], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        new_qe[start:end] = embs
        if (end % 5000 == 0) or end == new_size:
            print(f"    [{end}/{new_size}]", flush=True)
else:
    # Qwen3-Embedding models via vllm or sentence-transformers
    model_paths = {
        "qwen3_0.6b": os.environ.get("QWEN3_EMB_06B_PATH", "Qwen/Qwen3-Embedding-0.6B")
                       if os.path.exists(os.environ.get("QWEN3_EMB_06B_PATH", "Qwen/Qwen3-Embedding-0.6B"))
                       else "Qwen/Qwen3-Embedding-0.6B",
        "qwen3_8b": os.environ.get("QWEN3_EMB_8B_PATH", "Qwen/Qwen3-Embedding-8B"),
    }
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_paths[model_key], trust_remote_code=True)
    model = model.to("cuda")

    BATCH = 256 if model_key == "qwen3_0.6b" else 64
    new_qe = np.zeros((new_size, emb_dim), dtype=np.float32)
    for start in range(0, new_size, BATCH):
        end = min(start + BATCH, new_size)
        embs = model.encode(new_qt[start:end], convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        new_qe[start:end] = embs
        if (end % 5000 == 0) or end == new_size:
            print(f"    [{end}/{new_size}]", flush=True)

np.save(os.path.join(cache_dir, "query_embs.npy"), new_qe)
print(f"  Saved query_embs.npy ({new_qe.shape})", flush=True)

# Dense retrieval: inner product search against paragraph corpus
print("  Running dense retrieval...", flush=True)
para_embs = np.load(os.path.join(cache_dir, "para_embs.npy"))
TOP_K = 20

import torch
q_torch = torch.from_numpy(new_qe).cuda()
p_torch = torch.from_numpy(para_embs).cuda()

SEARCH_BATCH = 1000
all_indices = np.zeros((new_size, TOP_K), dtype=np.int32)
for start in range(0, new_size, SEARCH_BATCH):
    end = min(start + SEARCH_BATCH, new_size)
    scores = torch.mm(q_torch[start:end], p_torch.T)
    _, top_idx = scores.topk(TOP_K, dim=1)
    all_indices[start:end] = top_idx.cpu().numpy()
    if (end % 10000 == 0) or end == new_size:
        print(f"    [{end}/{new_size}]", flush=True)

np.savez(os.path.join(cache_dir, "faiss_para.npz"), indices=all_indices)
print(f"  Saved faiss_para.npz ({all_indices.shape})", flush=True)

del q_torch, p_torch, scores
torch.cuda.empty_cache()

# Sentence-mapped retrieval
print("  Running sentence-mapped retrieval...", flush=True)
sent_embs = np.load(os.path.join(cache_dir, "sent_embs.npy"))
# Build sentence -> parent paragraph mapping
sent_to_para = {}
for si, s in enumerate(sent_corpus):
    sent_to_para[si] = s.get("para_id", s.get("parent_id", -1))

q_torch = torch.from_numpy(new_qe).cuda()
s_torch = torch.from_numpy(sent_embs).cuda()

new_sent_mapped = [None] * new_size
for start in range(0, new_size, SEARCH_BATCH):
    end = min(start + SEARCH_BATCH, new_size)
    scores = torch.mm(q_torch[start:end], s_torch.T)
    _, top_idx = scores.topk(TOP_K * 3, dim=1)  # get more to account for dedup
    top_idx = top_idx.cpu().numpy()

    for i in range(end - start):
        seen_paras = set()
        mapped = []
        for si in top_idx[i]:
            pid = sent_to_para.get(int(si), -1)
            if pid >= 0 and pid not in seen_paras:
                seen_paras.add(pid)
                mapped.append(pid)
                if len(mapped) >= TOP_K:
                    break
        new_sent_mapped[start + i] = mapped
    if (end % 10000 == 0) or end == new_size:
        print(f"    [{end}/{new_size}]", flush=True)

with open(os.path.join(cache_dir, "sent_mapped.pkl"), "wb") as f:
    pickle.dump(new_sent_mapped, f)
print(f"  Saved sent_mapped.pkl ({len(new_sent_mapped)} entries)", flush=True)

del q_torch, s_torch
torch.cuda.empty_cache()

# BM25: re-query for ALL texts (BM25 is CPU, fast)
print("  Running BM25 retrieval...", flush=True)
from pyserini.search.lucene import LuceneSearcher
index_path = os.path.join(cache_dir, "lucene_index")
if not os.path.exists(index_path):
    index_path = os.path.join(DATA, "cache_minilm", "lucene_index")  # shared index

searcher = LuceneSearcher(index_path)
new_bm25 = {}
for i, text in enumerate(new_qt):
    try:
        hits = searcher.search(text, k=TOP_K)
        new_bm25[i] = [int(h.docid) for h in hits]
    except:
        new_bm25[i] = []
    if (i + 1) % 10000 == 0 or i + 1 == new_size:
        print(f"    [{i+1}/{new_size}]", flush=True)

with open(bm25_path, "wb") as f:
    pickle.dump(new_bm25, f)
print(f"  Saved {os.path.basename(bm25_path)} ({len(new_bm25)} entries)", flush=True)

print(f"\n=== Done patching {model_key} ===", flush=True)
