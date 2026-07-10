"""Shared utilities for vLLM model loading and chat formatting.

Self-contained version bundled with the release. Resolves model labels to
either Hugging Face Hub identifiers (default) or local paths via environment
variables, so reviewers can point to pre-downloaded checkpoints without
editing scripts.
"""

import os
import time

# Model registry: label -> (default HF identifier, env var for local override, max_model_len)
MODEL_REGISTRY = {
    "Qwen3-32B":  ("Qwen/Qwen3-32B-FP8",              "QWEN3_32B_PATH",  4096),
    "Llama-8B":   ("meta-llama/Llama-3.1-8B-Instruct", "LLAMA_8B_PATH",   4096),
    "Qwen3-Embedding-8B":   ("Qwen/Qwen3-Embedding-8B",   "QWEN3_EMB_8B_PATH",   2048),
    "Qwen3-Embedding-0.6B": ("Qwen/Qwen3-Embedding-0.6B", "QWEN3_EMB_0.6B_PATH", 2048),
}

DEFAULT_GPU_MEMORY_UTILIZATION = 0.90


def resolve_model_path(model_label):
    """Resolve a model label to a path: env var override > HF Hub default."""
    if model_label not in MODEL_REGISTRY:
        # Treat as a path or HF identifier directly
        return model_label
    hf_default, env_var, _ = MODEL_REGISTRY[model_label]
    return os.environ.get(env_var, hf_default)


def load_model(model_label, gpu_memory_utilization=None, max_model_len=None):
    """Load a vLLM model and its tokenizer.

    Args:
        model_label: registry key (e.g. "Qwen3-32B") or direct HF id / local path.
        gpu_memory_utilization: vLLM memory fraction (default 0.90).
        max_model_len: override max context length (default per-model in registry).

    Returns:
        (llm, tokenizer) tuple.
    """
    from transformers import AutoTokenizer
    from vllm import LLM

    path = resolve_model_path(model_label)
    if max_model_len is None:
        _, _, max_model_len = MODEL_REGISTRY.get(model_label, (None, None, 2048))
    gpu_util = gpu_memory_utilization or DEFAULT_GPU_MEMORY_UTILIZATION

    print(f"Loading tokenizer for {model_label} ({path})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    print(f"Loading {model_label} via vLLM (max_model_len={max_model_len})...", flush=True)
    llm = LLM(
        model=path,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_util,
    )
    return llm, tokenizer


def unload_model(llm):
    """Delete model and free GPU memory."""
    import gc
    import torch
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(3)


def format_chat(tokenizer, user_message):
    """Apply the tokenizer's chat template to a single user message.

    For Qwen3 models, disables thinking mode to get a direct answer.
    Falls back gracefully if the tokenizer does not support the kwarg.
    """
    msgs = [{"role": "user", "content": user_message}]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            return user_message
