#!/usr/bin/env python3
"""
End-to-end RAG evaluation.
Feed retrieved evidence (dense, hybrid, SCER) to Qwen3-32B, measure EM/F1.

Usage:
  PYTHONUNBUFFERED=1 python3 \
    pipeline/run_rag_eval.py \
    --benchmark hotpotqa --model qwen3 --top-k 5
"""

import argparse
import json
import os
import pickle
import re
import string
import sys
import time
import numpy as np
from collections import defaultdict, Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOP_K_CONTEXT = 5  # number of passages to use as context


# ── Answer evaluation ────────────────────────────────────────────

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(c for c in s if c not in string.punctuation)
    # Remove extra whitespace
    s = ' '.join(s.split())
    return s


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction, gold):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def fever_accuracy(prediction, gold):
    """Check if FEVER prediction matches gold label."""
    pred_lower = prediction.lower().strip()
    gold_lower = gold.lower().strip()
    # Map various phrasings
    if gold_lower == "supports":
        return float(any(w in pred_lower for w in ["supports", "true", "yes", "correct", "supported"]))
    elif gold_lower == "refutes":
        return float(any(w in pred_lower for w in ["refutes", "false", "no", "incorrect", "refuted"]))
    return 0.0


# ── Retrieval reconstruction ────────────────────────────────────

def jaccard(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)


def reconstruct_retrieval(benchmark, model_key):
    """Load cached retrieval results and reconstruct ranked lists per question."""
    if benchmark == "hotpotqa":
        data_dir = os.path.join(DATA_DIR)
        if model_key == "qwen3":
            cache_dir = os.path.join(data_dir, "retrieval_cache")
        else:
            cache_dir = os.path.join(data_dir, "retrieval_cache_MiniLM-L6-v2")
    else:
        data_dir = os.path.join(DATA_DIR, benchmark)
        cache_dir = os.path.join(data_dir, f"cache_{model_key}")

    # Load questions and paraphrases
    with open(os.path.join(data_dir, "questions.json")) as f:
        questions = json.load(f)
    with open(os.path.join(data_dir, "paraphrases.json")) as f:
        paraphrases = json.load(f)
    with open(os.path.join(data_dir, "paragraph_corpus.json")) as f:
        para_corpus = json.load(f)

    # Build text_to_idx
    all_query_texts = []
    text_to_idx = {}
    for q in questions:
        qtxt = q["question"]
        paras = paraphrases.get(q["id"], [])
        for t in [qtxt] + paras:
            if t not in text_to_idx:
                text_to_idx[t] = len(all_query_texts)
                all_query_texts.append(t)

    # Load cached retrieval — try multiple filename conventions
    for fname in ["faiss_para_results.npz", "faiss_para.npz"]:
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            para_faiss = np.load(fpath)
            para_indices = para_faiss["indices"]
            break
    else:
        raise FileNotFoundError(f"No FAISS para results in {cache_dir}")

    for fname in ["faiss_sent_para_mapped.pkl", "sent_para_mapped.pkl", "sent_mapped.pkl"]:
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                sent_as_para = pickle.load(f)
            break
    else:
        raise FileNotFoundError(f"No sent_para mapping in {cache_dir}")

    for fname in ["bm25_para_results.pkl", "bm25_results.pkl", "bm25.pkl"]:
        fpath = os.path.join(cache_dir, fname)
        if os.path.exists(fpath):
            with open(fpath, 'rb') as f:
                bm25_results = pickle.load(f)
            break
    else:
        raise FileNotFoundError(f"No BM25 results in {cache_dir}")

    return questions, paraphrases, para_corpus, text_to_idx, para_indices, sent_as_para, bm25_results


def get_ranked_lists(q, paraphrases_dict, text_to_idx, para_indices, sent_as_para, bm25_results, top_k=20):
    """Get dense, hybrid, SCER ranked lists for a single question."""
    qtxt = q["question"]
    qid = q["id"]
    paras = paraphrases_dict.get(qid, [])

    oi = text_to_idx[qtxt]
    pis = [text_to_idx[p] for p in paras if p in text_to_idx]

    dr = para_indices[oi].tolist()
    br = bm25_results[oi]
    sm = sent_as_para[oi]

    # Hybrid RRF
    rrf = defaultdict(float)
    for r, i in enumerate(dr[:top_k]): rrf[i] += 1.0 / (60 + r + 1)
    for r, i in enumerate(br[:top_k]): rrf[i] += 1.0 / (60 + r + 1)
    hr = [i for i, _ in sorted(rrf.items(), key=lambda x: -x[1])][:top_k]

    # SCER consensus
    sources = [dr[:top_k], br[:top_k]]
    if sm: sources.append(sm[:top_k])
    psets = []
    for pi in pis:
        pd = para_indices[pi].tolist()
        pb = bm25_results[pi]
        ps = sent_as_para[pi]
        sources.extend([pd[:top_k], pb[:top_k]])
        if ps: sources.append(ps[:top_k])
        psets.append(set(pd[:top_k]))

    votes = defaultdict(float)
    n = len(sources)
    for ranked in sources:
        for rank, idx in enumerate(ranked[:top_k]):
            votes[int(idx)] += 1.0 / (rank + 1)
    for idx in votes:
        votes[idx] /= n
    sr = [i for i, _ in sorted(votes.items(), key=lambda x: -x[1])]

    # Stability
    orig_set = set(dr[:top_k])
    para_stab = [jaccard(orig_set, ps) for ps in psets]
    stability = float(np.mean(para_stab)) if para_stab else 1.0

    return {
        "dense": dr[:top_k],
        "hybrid": hr[:top_k],
        "scer": sr[:top_k],
    }, stability


# ── RAG answer generation ───────────────────────────────────────

def build_rag_prompts(questions, ranked_lists_all, para_corpus, benchmark, top_k_ctx):
    """Build prompts for each method for each question."""
    prompts = {}  # method -> list of prompts
    for method in ["dense", "hybrid", "scer"]:
        prompts[method] = []

    para_text_map = {p["id"]: p["text"] for p in para_corpus}

    for qi, q in enumerate(questions):
        for method in ["dense", "hybrid", "scer"]:
            ranked = ranked_lists_all[qi][method]
            # Get passage texts
            passages = []
            for pid in ranked[:top_k_ctx]:
                if pid in para_text_map:
                    passages.append(para_text_map[pid][:500])  # cap length

            context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages))

            if benchmark == "fever":
                question_text = q["question"]
                # Extract claim from question format
                claim = q.get("claim", question_text)
                prompt = f"""Based on the following evidence, determine if the claim is SUPPORTS or REFUTES.

Evidence:
{context}

Claim: {claim}

Answer with exactly one word: SUPPORTS or REFUTES."""
            else:
                prompt = f"""Answer the question based on the provided evidence. Give a short, direct answer.

Evidence:
{context}

Question: {q['question']}

Answer:"""

            prompts[method].append(prompt)

    return prompts


def generate_answers(prompts_list, batch_size=128):
    """Generate answers using Qwen3-32B."""
    from utils import load_model, format_chat, unload_model
    from vllm import SamplingParams

    llm, tokenizer = load_model("Qwen3-32B", gpu_memory_utilization=0.90)
    params = SamplingParams(temperature=0.0, max_tokens=50)

    answers = []
    for start in range(0, len(prompts_list), batch_size):
        end = min(start + batch_size, len(prompts_list))
        batch = [format_chat(tokenizer, p) for p in prompts_list[start:end]]
        outputs = llm.generate(batch, params)
        for out in outputs:
            text = out.outputs[0].text.strip()
            # Clean up: take first line, remove quotes
            text = text.split('\n')[0].strip().strip('"').strip("'")
            answers.append(text)
        if (start // batch_size + 1) % 5 == 0:
            print(f"    [{end}/{len(prompts_list)}]", flush=True)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    return answers


def run_one_config(benchmark, model_key, top_k_ctx, llm, tokenizer):
    """Run RAG eval for one benchmark+model config using a pre-loaded LLM."""
    from vllm import SamplingParams

    out_path = os.path.join(RESULTS_DIR, f"rag_{benchmark}_{model_key}_top{top_k_ctx}.json")
    if os.path.exists(out_path):
        print(f"  SKIP (cached): {out_path}", flush=True)
        with open(out_path) as f:
            saved = json.load(f)
        for method in ["dense", "hybrid", "scer"]:
            m = saved["aggregate"][method]
            print(f"    {method}: EM={m['em']:.4f}, F1={m['f1']:.4f}", flush=True)
        return

    print(f"\n{'='*50}", flush=True)
    print(f"  {benchmark} + {model_key} (top-{top_k_ctx})", flush=True)
    print(f"{'='*50}", flush=True)

    # Load retrieval results
    questions, paraphrases, para_corpus, text_to_idx, para_indices, sent_as_para, bm25_results = \
        reconstruct_retrieval(benchmark, model_key)

    # Test set (same split as step4)
    np.random.seed(42)
    n = len(questions)
    perm = np.random.permutation(n)
    cal_set = set(perm[:n // 5].tolist())
    test_indices = [i for i in range(n) if i not in cal_set]

    # Subsample to 2000
    if len(test_indices) > 2000:
        np.random.seed(123)
        test_indices = sorted(np.random.choice(test_indices, 2000, replace=False).tolist())

    test_questions = [questions[i] for i in test_indices]
    print(f"  Test questions: {len(test_questions)}", flush=True)

    # Reconstruct ranked lists
    ranked_lists_all = []
    stabilities = []
    for q in test_questions:
        rl, stab = get_ranked_lists(q, paraphrases, text_to_idx, para_indices, sent_as_para, bm25_results)
        ranked_lists_all.append(rl)
        stabilities.append(stab)

    # Build prompts
    prompts = build_rag_prompts(test_questions, ranked_lists_all, para_corpus, benchmark, top_k_ctx)

    # Combine all prompts
    all_prompts = []
    method_boundaries = {}
    offset = 0
    for method in ["dense", "hybrid", "scer"]:
        method_boundaries[method] = (offset, offset + len(prompts[method]))
        all_prompts.extend(prompts[method])
        offset += len(prompts[method])

    print(f"  Generating {len(all_prompts)} answers...", flush=True)
    params = SamplingParams(temperature=0.0, max_tokens=50)

    all_answers = []
    batch_size = 256
    for start in range(0, len(all_prompts), batch_size):
        end = min(start + batch_size, len(all_prompts))
        batch = []
        for p in all_prompts[start:end]:
            msgs = [{"role": "user", "content": p}]
            try:
                batch.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                             add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                batch.append(tokenizer.apply_chat_template(msgs, tokenize=False,
                             add_generation_prompt=True))
        outputs = llm.generate(batch, params)
        for out in outputs:
            text = out.outputs[0].text.strip().split('\n')[0].strip().strip('"').strip("'")
            all_answers.append(text)
        print(f"    [{end}/{len(all_prompts)}]", flush=True)

    # Split by method
    method_answers = {}
    for method in ["dense", "hybrid", "scer"]:
        s, e = method_boundaries[method]
        method_answers[method] = all_answers[s:e]

    # Evaluate
    results = {"per_question": [], "aggregate": {}}
    for method in ["dense", "hybrid", "scer"]:
        ems, f1s = [], []
        for qi, q in enumerate(test_questions):
            pred = method_answers[method][qi]
            gold = q["answer"]
            if benchmark == "fever":
                em = fever_accuracy(pred, gold)
                f1 = em
            else:
                em = exact_match(pred, gold)
                f1 = f1_score(pred, gold)
            ems.append(em)
            f1s.append(f1)
        results["aggregate"][method] = {
            "em": float(np.mean(ems)), "f1": float(np.mean(f1s)),
            "em_std": float(np.std(ems)), "f1_std": float(np.std(f1s)),
        }

    # Stratified
    for method in ["dense", "scer"]:
        for sname, lo, hi in [("unstable", 0, 0.5), ("moderate", 0.5, 0.7), ("stable", 0.7, 2.0)]:
            group = [(qi, q) for qi, (q, s) in enumerate(zip(test_questions, stabilities)) if lo <= s < hi]
            if not group: continue
            if benchmark == "fever":
                vals = [fever_accuracy(method_answers[method][qi], q["answer"]) for qi, q in group]
            else:
                vals = [f1_score(method_answers[method][qi], q["answer"]) for qi, q in group]
            results.setdefault("stratified", {}).setdefault(method, {})[sname] = {
                "n": len(group), "f1": float(np.mean(vals)),
            }

    # Per-question
    for qi, q in enumerate(test_questions):
        entry = {"id": q["id"], "gold": q["answer"], "stability": stabilities[qi]}
        for method in ["dense", "hybrid", "scer"]:
            pred = method_answers[method][qi]
            entry[f"{method}_pred"] = pred
            if benchmark == "fever":
                entry[f"{method}_em"] = fever_accuracy(pred, q["answer"])
            else:
                entry[f"{method}_em"] = exact_match(pred, q["answer"])
                entry[f"{method}_f1"] = f1_score(pred, q["answer"])
        results["per_question"].append(entry)

    results["config"] = {"benchmark": benchmark, "model": model_key,
                         "top_k_context": top_k_ctx, "n_test": len(test_questions)}

    # Print
    metric = "Acc" if benchmark == "fever" else "EM"
    print(f"\n  {'Method':<10s} {metric:>8s} {'F1':>8s}", flush=True)
    print(f"  {'-'*30}", flush=True)
    for method in ["dense", "hybrid", "scer"]:
        m = results["aggregate"][method]
        print(f"  {method:<10s} {m['em']:>8.4f} {m['f1']:>8.4f}", flush=True)

    if "stratified" in results:
        print(f"\n  Stratified F1:", flush=True)
        for sname in ["unstable", "moderate", "stable"]:
            d = results["stratified"].get("dense", {}).get(sname, {})
            s = results["stratified"].get("scer", {}).get(sname, {})
            if d and s:
                print(f"    {sname} (n={d['n']}): dense={d['f1']:.4f}, scer={s['f1']:.4f}, "
                      f"Δ={s['f1']-d['f1']:+.4f}", flush=True)

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=== End-to-End RAG Evaluation (all configs) ===\n", flush=True)

    # Load Qwen3-32B ONCE with TP=2
    print("Loading Qwen3-32B-FP8 (tensor_parallel=2)...", flush=True)
    from vllm import LLM as VLLM_LLM
    from transformers import AutoTokenizer

    model_path = "Qwen/Qwen3-32B-FP8"  # HF hub identifier; override with local path if needed
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = VLLM_LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=2,
    )
    print("Model loaded!\n", flush=True)

    # Run all 4 configs
    for benchmark in ["hotpotqa", "fever"]:
        for model_key in ["qwen3", "minilm"]:
            run_one_config(benchmark, model_key, args.top_k, llm, tokenizer)

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    print("\n=== ALL DONE ===", flush=True)


if __name__ == '__main__':
    main()
