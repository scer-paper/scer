"""Parse GPT-5.4 audit results (temp=0) into LaTeX-ready statistics."""
import json, os
from collections import defaultdict, Counter
from statistics import mean

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "pipeline", "gpt5.4_audit", "outputs")

P_OUT = os.path.join(OUT_DIR, "paraphrase_audit_gpt54_temp0_output.jsonl")
P_IDX = os.path.join(OUT_DIR, "paraphrase_audit_sample_index.json")
H_OUT = os.path.join(OUT_DIR, "hallucination_audit_gpt54_temp0_output.jsonl")
H_IDX = os.path.join(OUT_DIR, "hallucination_audit_sample_index.json")


def parse_line(line):
    rec = json.loads(line)
    if rec.get("error"): return None, rec["error"]
    try:
        return json.loads(rec["response"]["body"]["choices"][0]["message"]["content"]), None
    except Exception as e:
        return None, str(e)


def analyze_paraphrases():
    idx = json.load(open(P_IDX))
    idx_by_id = {r["custom_id"]: r for r in idx["rows"]}
    by_bench = defaultdict(list)
    fail_modes = defaultdict(Counter)
    n_errors = 0
    with open(P_OUT) as f:
        for line in f:
            parsed, err = parse_line(line)
            if err: n_errors += 1; continue
            cid = json.loads(line)["custom_id"]
            rec = idx_by_id.get(cid)
            if not rec: continue
            bench = rec["bench"]
            try: score = int(parsed.get("score", 0))
            except Exception: n_errors += 1; continue
            by_bench[bench].append({**rec, "score": score, "failure_mode": parsed.get("failure_mode")})
            if score <= 2 and parsed.get("failure_mode"):
                fail_modes[bench][parsed["failure_mode"]] += 1

    summary = {"errors": n_errors, "by_benchmark": {}}
    for bench, rows in by_bench.items():
        scores = [r["score"] for r in rows]
        summary["by_benchmark"][bench] = {
            "n": len(scores),
            "mean_score": round(mean(scores), 3) if scores else None,
            "pct_5":  round(100 * sum(1 for s in scores if s == 5) / len(scores), 1),
            "pct_4":  round(100 * sum(1 for s in scores if s == 4) / len(scores), 1),
            "pct_3":  round(100 * sum(1 for s in scores if s == 3) / len(scores), 1),
            "pct_2":  round(100 * sum(1 for s in scores if s == 2) / len(scores), 1),
            "pct_1":  round(100 * sum(1 for s in scores if s == 1) / len(scores), 1),
            "pct_high_fidelity_4plus": round(100 * sum(1 for s in scores if s >= 4) / len(scores), 1),
            "pct_usable_3plus": round(100 * sum(1 for s in scores if s >= 3) / len(scores), 1),
            "pct_broken_2minus": round(100 * sum(1 for s in scores if s <= 2) / len(scores), 1),
            "failure_modes": dict(fail_modes[bench]),
        }
    return summary


def analyze_hallucination():
    idx = json.load(open(H_IDX))
    idx_by_id = {r["custom_id"]: r for r in idx["rows"]}
    by_key = defaultdict(Counter)
    n_errors = 0
    with open(H_OUT) as f:
        for line in f:
            parsed, err = parse_line(line)
            if err: n_errors += 1; continue
            cid = json.loads(line)["custom_id"]
            rec = idx_by_id.get(cid)
            if not rec: continue
            verdict = parsed.get("verdict", "?")
            key = (rec["bench"], rec["model"], rec["method"])
            by_key[key][verdict] += 1

    summary = {"errors": n_errors, "per_setting": {}}
    for (bench, model, method), counts in by_key.items():
        total = sum(counts.values())
        if total == 0: continue
        summary["per_setting"][f"{bench}/{model}/{method}"] = {
            "n": total,
            "supported_pct": round(100 * counts.get("SUPPORTED", 0) / total, 1),
            "partially_supported_pct": round(100 * counts.get("PARTIALLY_SUPPORTED", 0) / total, 1),
            "not_supported_pct": round(100 * counts.get("NOT_SUPPORTED", 0) / total, 1),
            "contradicted_pct": round(100 * counts.get("CONTRADICTED", 0) / total, 1),
            "any_unsupported_pct": round(100 * (counts.get("NOT_SUPPORTED", 0)
                                                + counts.get("CONTRADICTED", 0)
                                                + counts.get("PARTIALLY_SUPPORTED", 0)) / total, 1),
        }
    return summary


def main():
    print("=" * 70)
    print("GPT-5.4 (temp=0, reasoning=none) AUDIT RESULTS")
    print("=" * 70)

    p_sum = analyze_paraphrases()
    with open(os.path.join(OUT_DIR, "paraphrase_audit_gpt54_summary.json"), "w") as f:
        json.dump(p_sum, f, indent=2)
    print(f"\n=== PARAPHRASE FIDELITY (errors={p_sum['errors']}) ===")
    print(f"{'Benchmark':<18} {'n':>5} {'mean':>5} {'5':>5} {'4':>5} {'3':>5} {'2':>5} {'1':>5} {'>=4':>5} {'>=3':>5} {'<=2':>5}")
    for bench, v in p_sum["by_benchmark"].items():
        print(f"{bench:<18} {v['n']:>5} {v['mean_score']:>5} "
              f"{v['pct_5']:>5} {v['pct_4']:>5} {v['pct_3']:>5} {v['pct_2']:>5} {v['pct_1']:>5} "
              f"{v['pct_high_fidelity_4plus']:>5} {v['pct_usable_3plus']:>5} {v['pct_broken_2minus']:>5}")
        if v["failure_modes"]:
            print(f"   failure modes: {v['failure_modes']}")

    h_sum = analyze_hallucination()
    with open(os.path.join(OUT_DIR, "hallucination_audit_gpt54_summary.json"), "w") as f:
        json.dump(h_sum, f, indent=2)
    print(f"\n=== HALLUCINATION GROUNDING (errors={h_sum['errors']}) ===")
    print(f"{'Setting':<40} {'n':>4} {'SUP':>5} {'PART':>5} {'NOT':>5} {'CON':>5} {'unsup':>6}")
    for k, v in sorted(h_sum["per_setting"].items()):
        print(f"{k:<40} {v['n']:>4} {v['supported_pct']:>5} {v['partially_supported_pct']:>5} "
              f"{v['not_supported_pct']:>5} {v['contradicted_pct']:>5} {v['any_unsupported_pct']:>6}")


if __name__ == "__main__":
    main()
