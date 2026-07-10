"""Leave-one-setting-out weight tuning (CPU-only, ~15-30 min wall).

For each of the 9 (benchmark, model) settings, hold it out and tune weights
on the 8 remaining settings' calibration splits via grid search. Then evaluate
the LOO-tuned weights on the held-out setting's test split.

Leave-one-(benchmark, embedder)-out weight-tuning sensitivity analysis (LOO as a
robustness check).

Reports:
  - LOO-tuned vs joint-tuned vs uniform-weight Cov@20 per fold
  - Variance of the 6 selected weights across the 9 folds (do they cluster
    around the joint-tuned values, or vary wildly?)
  - Honest takeaway: are the paper's weights overfit?

Output: results/loo_weights.json

Subsampling: calibration splits subsampled to 200 queries per setting for
tractability; this leaves enough power to rank weight tuples reliably but
makes the 729-config × 9-fold grid run in ~10-15 min on CPU.
"""
import json, os, pickle, sys, time
import numpy as np
from collections import defaultdict

# Lightweight in-memory stub for the optional cross-script state dict;
# scripts in this release run standalone without inter-script state.
STAGE_KEY = "loo_weights"
def load_status(): return {}
def save_status(status): pass


BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "loo_weights.json")
TOP_K = 20

BENCHMARKS = ["hotpotqa_full", "fever_full", "squad_full"]
MODELS = ["minilm", "qwen3_0.6b", "qwen3_8b"]
SETTINGS = [(b, m) for b in BENCHMARKS for m in MODELS]

# Joint-tuned weights from the paper
PAPER_W = dict(a=0.3, b=0.5, c=1.5, d=1.0, ws=0.2, gamma=0.4)
UNIFORM_W = dict(a=1.0, b=0.0, c=1.0, d=0.0, ws=1.0, gamma=1.0)  # all 1.0; no st dependence

# Grid for LOO tuning
GRID_a = [0.1, 0.3, 0.5]
GRID_b = [0.3, 0.5, 0.7]
GRID_c = [1.0, 1.5, 2.0]
GRID_d = [0.5, 1.0, 1.5]
GRID_ws = [0.1, 0.2, 0.4]
GRID_gamma = [0.3, 0.4, 0.5]

N_CAL_SUB = 200   # subsample of calibration set per setting
RNG_SEED = 7


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


def load_setting(bench, model_key):
    data_dir = os.path.join(BASE, "data", bench)
    cache_dir = os.path.join(data_dir, f"cache_{model_key}")
    with open(os.path.join(data_dir, "questions.json")) as f: questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f: paraphrases = json.load(f)
    all_qt = []; t2i = {}
    for q in questions:
        for t in [q["question"]] + paraphrases.get(q["id"], []):
            if t not in t2i: t2i[t] = len(all_qt); all_qt.append(t)
    para_indices = np.load(os.path.join(cache_dir, "faiss_para.npz"))["indices"]
    with open(os.path.join(cache_dir, "sent_mapped.pkl"), 'rb') as f: sent_as_para = pickle.load(f)
    bm25_f = os.path.join(cache_dir, "bm25_pyserini.pkl")
    if not os.path.exists(bm25_f): bm25_f = os.path.join(cache_dir, "bm25.pkl")
    with open(bm25_f, 'rb') as f: bm25_results = pickle.load(f)

    np.random.seed(42); n = len(questions); perm = np.random.permutation(n)
    cal_set = set(perm[:n//5].tolist())
    cal_indices = sorted(list(cal_set))
    test_indices = [i for i in range(n) if i not in cal_set]

    # Build per-query data tuples ONCE: (gold, dr, br, sm, list of (pdr, pbr, psm), st)
    data = []
    for qi in cal_indices + test_indices:
        q = questions[qi]
        gold = set(q.get("gold_para_ids", []))
        if not gold: continue
        oi = t2i[q["question"]]
        pis = [t2i[p] for p in paraphrases.get(q["id"], []) if p in t2i]
        dr = para_indices[oi].tolist()[:TOP_K]
        br = bm25_results.get(oi, [])[:TOP_K]
        sm = sent_as_para[oi][:TOP_K] if oi < len(sent_as_para) and sent_as_para[oi] else []
        para_data = []
        for pi in pis:
            pdr = para_indices[pi].tolist()[:TOP_K]
            pbr = bm25_results.get(pi, [])[:TOP_K]
            psm = sent_as_para[pi][:TOP_K] if pi < len(sent_as_para) and sent_as_para[pi] else []
            para_data.append((pdr, pbr, psm))
        psets = [set(pdr) for pdr, _, _ in para_data]
        st = float(np.mean([jaccard(set(dr), ps) for ps in psets])) if psets else 1.0
        data.append({"is_cal": qi in cal_set, "gold": gold, "dr": dr, "br": br, "sm": sm,
                     "para": para_data, "st": st})
    return data


def coverage_for_weights(data_subset, w_params):
    """Compute SCER Coverage@20 over a list of pre-built query records."""
    a, b, c, d, ws, gamma = w_params["a"], w_params["b"], w_params["c"], w_params["d"], w_params["ws"], w_params["gamma"]
    n_hit = 0
    for q in data_subset:
        st = q["st"]
        w_d = a + b * st
        w_b = c - d * st
        w_s = ws
        votes = defaultdict(float)
        for r, doc in enumerate(q["dr"]): votes[doc] += w_d / (r + 1)
        for r, doc in enumerate(q["br"]): votes[doc] += w_b / (r + 1)
        for r, doc in enumerate(q["sm"]): votes[doc] += w_s / (r + 1)
        for pdr, pbr, psm in q["para"]:
            for r, doc in enumerate(pdr): votes[doc] += w_d * gamma / (r + 1)
            for r, doc in enumerate(pbr): votes[doc] += w_b * gamma / (r + 1)
            for r, doc in enumerate(psm): votes[doc] += w_s * gamma / (r + 1)
        top = set([doc for doc, _ in sorted(votes.items(), key=lambda x: -x[1])[:TOP_K]])
        if q["gold"].issubset(top): n_hit += 1
    return n_hit / len(data_subset) if data_subset else 0.0


def main():
    print("=== : leave-one-setting-out weight tuning ===", flush=True)
    t0 = time.time()

    # Load all 9 settings once
    print(f"Loading per-query data for all 9 settings...", flush=True)
    all_data = {}
    for bench, model in SETTINGS:
        key = f"{bench}/{model}"
        print(f"  [{key}]...", flush=True)
        all_data[key] = load_setting(bench, model)
    print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

    # Subsample cal indices
    rng = np.random.default_rng(RNG_SEED)
    cal_subs, test_full = {}, {}
    for key, data in all_data.items():
        cal_records = [q for q in data if q["is_cal"]]
        if len(cal_records) > N_CAL_SUB:
            cal_records = list(rng.choice(cal_records, N_CAL_SUB, replace=False))
        cal_subs[key] = cal_records
        test_full[key] = [q for q in data if not q["is_cal"]]
        print(f"  {key}: cal_sub={len(cal_subs[key])} test={len(test_full[key])}", flush=True)

    # Build the grid
    grid = [{"a": a, "b": b, "c": c, "d": d, "ws": ws, "gamma": gamma}
            for a in GRID_a for b in GRID_b for c in GRID_c
            for d in GRID_d for ws in GRID_ws for gamma in GRID_gamma]
    print(f"Grid size: {len(grid)} configs; {len(grid) * len(SETTINGS)} cal evaluations", flush=True)

    # LOO over 9 settings
    fold_results = {}
    for held_out_idx in range(len(SETTINGS)):
        held_out = SETTINGS[held_out_idx]; held_key = f"{held_out[0]}/{held_out[1]}"
        train_keys = [f"{b}/{m}" for i, (b, m) in enumerate(SETTINGS) if i != held_out_idx]
        print(f"\n--- Fold {held_out_idx+1}/9: hold-out {held_key} ---", flush=True)

        # Score each config on the 8 training cal subs
        best_cov, best_w = -1, None
        for w in grid:
            covs = [coverage_for_weights(cal_subs[k], w) for k in train_keys]
            mean_cov = float(np.mean(covs))
            if mean_cov > best_cov:
                best_cov = mean_cov; best_w = w

        # Eval on held-out test
        loo_test = coverage_for_weights(test_full[held_key], best_w)
        paper_test = coverage_for_weights(test_full[held_key], PAPER_W)
        # Uniform baseline (a=1, b=0, c=1, d=0, ws=1, gamma=1) → independent of st(q)
        uniform_test = coverage_for_weights(test_full[held_key], UNIFORM_W)

        fold_results[held_key] = {
            "best_w_loo": best_w,
            "best_cal_mean_cov": best_cov,
            "test_cov_loo": loo_test,
            "test_cov_paper_joint": paper_test,
            "test_cov_uniform": uniform_test,
            "delta_loo_vs_paper": loo_test - paper_test,
            "delta_loo_vs_uniform": loo_test - uniform_test,
        }
        print(f"  best LOO weights: {best_w}", flush=True)
        print(f"  test Cov@20: LOO={loo_test:.4f} paper={paper_test:.4f} uniform={uniform_test:.4f} "
              f"(LOO-paper {loo_test-paper_test:+.4f})", flush=True)

    # Summarize weight variance across folds
    weight_arrays = {k: [fold_results[s["b_key"] if False else key]["best_w_loo"][k] for key in fold_results]
                      for k in ["a", "b", "c", "d", "ws", "gamma"]}
    weight_arrays = {k: [fold_results[key]["best_w_loo"][k] for key in fold_results]
                      for k in ["a", "b", "c", "d", "ws", "gamma"]}
    summary = {
        "fold_results": fold_results,
        "weight_variance_across_folds": {k: {"mean": float(np.mean(v)),
                                              "sd": float(np.std(v)),
                                              "min": float(np.min(v)),
                                              "max": float(np.max(v))}
                                          for k, v in weight_arrays.items()},
        "paper_weights": PAPER_W,
        "mean_delta_loo_vs_paper": float(np.mean([r["delta_loo_vs_paper"] for r in fold_results.values()])),
        "mean_delta_loo_vs_uniform": float(np.mean([r["delta_loo_vs_uniform"] for r in fold_results.values()])),
        "n_folds_loo_beats_paper": int(sum(1 for r in fold_results.values() if r["delta_loo_vs_paper"] > 0)),
        "n_folds_loo_beats_uniform": int(sum(1 for r in fold_results.values() if r["delta_loo_vs_uniform"] > 0)),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'w') as f: json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT_PATH}", flush=True)
    print(f"\n=== Summary ===", flush=True)
    print(f"  Mean Δ(LOO - paper joint): {summary['mean_delta_loo_vs_paper']*100:+.2f} pp Cov@20", flush=True)
    print(f"  Mean Δ(LOO - uniform):     {summary['mean_delta_loo_vs_uniform']*100:+.2f} pp Cov@20", flush=True)
    print(f"  LOO beats paper in {summary['n_folds_loo_beats_paper']}/9 folds", flush=True)
    print(f"  LOO beats uniform in {summary['n_folds_loo_beats_uniform']}/9 folds", flush=True)
    print(f"  Weight variance across folds (a, b, c, d, ws, gamma):", flush=True)
    for k, v in summary["weight_variance_across_folds"].items():
        print(f"    {k}: mean={v['mean']:.3f} sd={v['sd']:.3f} range=[{v['min']:.2f},{v['max']:.2f}]"
              f"  (paper: {PAPER_W[k]})", flush=True)

    save_status({**load_status(), STAGE_KEY: "complete"})


if __name__ == '__main__':
    main()
