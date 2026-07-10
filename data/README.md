# Data setup

The generated paraphrases and the evaluation query sets (with gold evidence ids) are bundled here for direct use:

| File | Contents |
|---|---|
| `<bench>/questions.json` | evaluation queries with gold evidence ids |
| `<bench>/paraphrases.json` | 5 paraphrases per query (Qwen3-32B, temperature 0.7) |
| `<bench>/paraphrases_llama8b.json` | paraphraser-sensitivity set (Llama-3.1-8B-Instruct) |

`<bench>` is one of `hotpotqa_full`, `fever_full`, `squad_full`.

Corpora and retrieval caches are not bundled: the HotpotQA corpus is large and the embedding/retrieval caches total about 69 GB. Corpora build from the public benchmarks below, and the caches regenerate deterministically from the pipeline.

| Dataset | Source |
|---|---|
| HotpotQA (distractor full) | `hotpot_qa` on Hugging Face Hub |
| FEVER | `fever` on Hugging Face Hub |
| SQuAD 2.0 (answerable only) | `rajpurkar/squad_v2` on Hugging Face Hub |

`pipeline/generate_paraphrases.py` generates the paraphrases, `pipeline/build_indices.py` builds the dense (FAISS) and BM25 indices and embedding caches, and `pipeline/build_sent_mapped.py` builds the sentence-mapped cache; retrieval lists and caches are written under `data/<bench>/cache_<embedder>/`. The 20/80 calibration/test split is seeded with NumPy seed 42 (fixed across all settings). Exact test sizes: HotpotQA 5,924, FEVER 5,205, SQuAD 2.0 4,743.
