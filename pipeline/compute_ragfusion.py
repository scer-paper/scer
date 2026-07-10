"""RAG-Fusion baseline: paraphrase + dense RRF only.

Implements the RAG-Fusion baseline reported in Table 4 of the paper:
combine the original query with 5 LLM-generated paraphrases via
Reciprocal Rank Fusion (k0=60) over the dense retriever's top-20 lists.

This script reads from the same pre-computed dense retrieval caches
used by the main SCER pipeline and produces ragfusion_results.json.

Usage:
    python compute_ragfusion.py
"""

import json
import os
import pickle
import numpy as np
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE, "results")
TOP_K = 20
RRF_K = 60  # standard RRF constant


def evidence_coverage(rset, gold):
    if not gold:
        return True
    return gold.issubset(rset)


def compute_ragfusion(dense_orig, para_indices_for_paraphrases):
    """Fuse the original query's dense top-K with each paraphrase's dense top-K via RRF.

    Args:
        dense_orig: list of int doc ids, top-K dense for the original query
        para_indices_for_paraphrases: list of [top-K doc ids per paraphrase]

    Returns: ranked list of doc ids (length K), highest-RRF-score first.
    """
    rrf = defaultdict(float)
    for r, i in enumerate(dense_orig[:TOP_K]):
        rrf[i] += 1.0 / (RRF_K + r + 1)
    for para_topk in para_indices_for_paraphrases:
        for r, i in enumerate(para_topk[:TOP_K]):
            rrf[i] += 1.0 / (RRF_K + r + 1)
    return [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:TOP_K]


def run_one(bench, data_root):
    """Run RAG-Fusion on one benchmark (MiniLM encoder).

    Reads the dense top-K cache from data_root/<bench>/cache_minilm/.
    Returns Cov@20 on the test split (80% of questions, same split as the paper).
    """
    data_dir = os.path.join(data_root, bench)
    cache_dir = os.path.join(data_dir, "cache_minilm")

    with open(os.path.join(data_dir, "questions.json")) as f:
        questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f:
        paraphrases = json.load(f)

    # Build the same text-to-index map used by the SCER pipeline
    all_text, t2i = [], {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i:
                t2i[t] = len(all_text)
                all_text.append(t)

    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]

    # 20/80 cal/test split, seed 42 (matches the paper's evaluation split)
    np.random.seed(42)
    n = len(questions)
    perm = np.random.permutation(n)
    cal_set = set(perm[: n // 5].tolist())

    covs = []
    for qi, q in enumerate(questions):
        if qi in cal_set:
            continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]
        gold = set(q["gold_para_ids"])

        dense_orig = para_indices[oi].tolist()[:TOP_K]
        para_topks = [para_indices[pi].tolist()[:TOP_K] for pi in pis]

        ranked = compute_ragfusion(dense_orig, para_topks)
        covs.append(float(evidence_coverage(set(ranked[:TOP_K]), gold)))

    return {
        "n_test": len(covs),
        "n_paraphrases_per_query": 5,
        "rrf_k": RRF_K,
        "ragfusion_cov20": float(np.mean(covs)),
    }


def main():
    # Default data root: the experiments folder used to build the caches.
    # Override with --data-root if running from a different copy.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="./data",
        help="Path to the data directory containing per-benchmark cache_minilm/ folders.",
    )
    args = parser.parse_args()

    out = {
        "description": "RAG-Fusion baseline (paraphrase + dense RRF, k0=60), MiniLM encoder",
        "config": {"top_k": TOP_K, "rrf_k": RRF_K, "n_paraphrases": 5,
                   "encoder": "all-MiniLM-L6-v2"},
    }
    for bench in ["hotpotqa_full", "fever_full", "squad_full"]:
        print(f"Running {bench}...", flush=True)
        out[bench] = run_one(bench, args.data_root)
        print(f"  cov20 = {out[bench]['ragfusion_cov20']:.4f}", flush=True)

    out_path = os.path.join(RESULTS_DIR, "ragfusion_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
