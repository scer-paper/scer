# Models

Model weights are not bundled. Set these env vars to local paths or leave unset to let the scripts fall back to Hugging Face Hub IDs.

| Env var | Default (HF Hub ID) |
|---|---|
| `QWEN3_EMB_06B_PATH` | `Qwen/Qwen3-Embedding-0.6B` |
| `QWEN3_EMB_8B_PATH` | `Qwen/Qwen3-Embedding-8B` |
| `QWEN3_RERANKER_PATH` | `Qwen/Qwen3-Reranker-8B` |
| (MiniLM, HF ID) | `sentence-transformers/all-MiniLM-L6-v2` |
| (paraphraser, HF ID) | `Qwen/Qwen3-32B` (FP8 quantized variant via vLLM) |

For the GPT-5.4 audit (`pipeline/gpt5.4_audit/`) set `OPENAI_API_KEY` in a `.env` file or environment.
