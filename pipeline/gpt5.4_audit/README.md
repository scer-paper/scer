# GPT-5.4 audit

Two LLM-judge audits referenced in the paper (Appendix A):

- **Paraphrase fidelity** (1,500 paraphrases, 500 per benchmark) - 1–5 rubric, mean fidelity 4.54–4.57, ≥4 rate 89.2–90.4%.
- **Hallucination grounding** (1,800 triples - 100 queries × 9 settings × {Hybrid, SCER}) - SCER 74.2% SUPPORTED vs Hybrid 72.8%.

## Scripts

| File | Purpose |
|---|---|
| `build_paraphrase_batch.py` | Build OpenAI Batch API input JSONL from the paraphrase sample |
| `run_audit.py` | Run the audit synchronously (1500 + 1800 judgments, `temperature=0`, deterministic) |
| `analyze_results.py` | Parse audit JSONL → summary JSON (the numbers in §A line 470) |

Requires `OPENAI_API_KEY`. Summary outputs are released at `results/audit_outputs/`.
