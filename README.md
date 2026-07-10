# Paraphrase-Induced Retrieval Instability

Code and result files for the paper *Paraphrase-Induced Retrieval Instability: Systematic Measurement and the Limits of Inference-Time Mitigation*.

## Setup

```bash
pip install -r requirements.txt
export SCER_HOME=$(pwd)
```

See `data/README.md` for benchmark setup and `models/README.md` for embedder/LLM env vars.

## Reproduce paper tables

| Paper | Command |
|---|---|
| Table 3 (main results) | `python pipeline/run_baselines.py` |
| Table 9 (stratified) | `python pipeline/run_baselines.py` (writes stratified split alongside main) |
| Table 13 (component ablation) | `python pipeline/experiments/component_ablation.py` |
| Table 16 (rank-discount) | computed from the same retrieval cache as Table 3 |
| Table 17 (per-cell significance) | derived from `results/significance_tests.json` + `results/bootstrap_cis.json` |
| Table 19 (reranker) | `python pipeline/experiments/reranker_breadth.py` |

## Result files for the additional tables

| Paper | File |
|---|---|
| Table 2 (encoder-family instability) | `results/instability_e5.json`, `results/instability_bge.json` |
| Table 4 (consensus decomposition) | `results/component_ablation.json` |
| Table 14 (split robustness) | `results/multiseed_robustness_results.json` |
| Table 15 (weighting direction) | `results/weighting_direction_results.json` |


## Layout

```
pipeline/                       core code
  run_baselines.py              all retrieval baselines from cached lists
  build_indices.py              dense embedding caches
  generate_paraphrases.py       Qwen3-32B paraphrase generation
  build_sent_mapped.py          sentence-mapped retrieval cache
  experiments/                  extended experiments (9 scripts)
  run_rag_eval.py               end-to-end RAG generation (Qwen3-32B)
  gpt5.4_audit/                   GPT-5.4 paraphrase + hallucination audit
results/                        result JSONs
data/README.md                  data setup pointers
models/README.md                model env-var setup
```

## License

MIT (code) - see `LICENSE`. Benchmark datasets retain their original licenses (CC BY-SA for HotpotQA, etc.).
