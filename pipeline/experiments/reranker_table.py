"""Reranker breadth table (CPU-only post-processing).

Reads the 9 per-setting reranker outputs from 
(reranker_<bench>_<model>.json) and aggregates them into a single LaTeX-ready
table for the paper.

Resolves review feedback (reranker breadth) by producing a 3-benchmark × 3-embedder
× 5-method matrix of Cov@20: Dense, Hybrid, SCER, Hybrid+Reranker, SCER+Reranker.

Output: results/reranker_breadth.json
        and a printed LaTeX table snippet.
"""
import json, os, sys

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "reranker_table"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "reranker_breadth.json")

BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f: return json.load(f)
    return {}
def save_status(s):
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, 'w') as f: json.dump(s, f, indent=2)


def main():
    print("=== : reranker breadth table aggregation ===", flush=True)
    table = {}
    missing = []
    for bench in BENCHMARKS:
        for model in MODELS:
            key = f"{bench}/{model}"
            f = os.path.join(RESULTS_DIR, f"reranker_{bench}_{model}.json")
            if not os.path.exists(f):
                missing.append(key); continue
            with open(f) as h: d = json.load(h)
            table[key] = {
                "n_queries": d.get("n_queries"),
                "dense": d.get("dense"),
                "hybrid": d.get("hybrid"),
                "scer": d.get("scer_adaptive"),
                "hybrid_rerank": d.get("hybrid_rerank"),
                "scer_rerank": d.get("scer_rerank"),
                "delta_rerank": (d.get("scer_rerank") or 0) - (d.get("hybrid_rerank") or 0),
            }
    if missing:
        print(f"  Missing: {missing}", flush=True)
    if not table:
        print("  No reranker results yet;  must run first.", flush=True); return

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f: json.dump(table, f, indent=2)
    print(f"  Saved {OUT_PATH}", flush=True)

    # Print LaTeX-ready table snippet
    print("\n=== LaTeX-ready table 'tab:rerank-breadth' ===")
    print("% Reranker on all 9 settings (Qwen3-Reranker-8B, top-50 → top-20)")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(" & & \\textbf{Hybrid} & \\textbf{SCER} & \\textbf{H+Rerank} & \\textbf{S+Rerank} & $\\Delta$(S+R - H+R) \\\\")
    print("\\midrule")
    for bench in BENCHMARKS:
        bench_short = {"hotpotqa_full": "HotpotQA", "fever_full": "FEVER", "squad_full": "SQuAD 2.0"}[bench]
        for i, model in enumerate(MODELS):
            key = f"{bench}/{model}"
            if key not in table: continue
            r = table[key]
            bn = bench_short if i == 0 else ""
            mn = {"minilm": "MiniLM", "qwen3_0.6b": "Qwen3-0.6B", "qwen3_8b": "Qwen3-8B"}[model]
            print(f"{bn} & {mn} & .{int(r['hybrid']*1000):03d} & .{int(r['scer']*1000):03d} "
                  f"& .{int(r['hybrid_rerank']*1000):03d} & .{int(r['scer_rerank']*1000):03d} "
                  f"& {r['delta_rerank']*100:+.1f} \\\\")
        if bench != BENCHMARKS[-1]: print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Headline summary
    print("\n=== Headline numbers ===")
    avg_d = sum(t['delta_rerank'] for t in table.values()) / len(table)
    print(f"  Mean Δ(S+Rerank − H+Rerank) across {len(table)} settings: {avg_d*100:+.2f} pp")
    pos = sum(1 for t in table.values() if t['delta_rerank'] > 0)
    print(f"  S+Rerank > H+Rerank in {pos}/{len(table)} settings")
    save_status({**load_status(), STAGE_KEY: "complete"})


if __name__ == '__main__':
    main()
