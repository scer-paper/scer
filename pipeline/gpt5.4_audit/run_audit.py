"""Re-run paraphrase (1500) + hallucination (1800) audits with GPT-5.4 + temp=0.

Uses Responses API (/v1/responses) with:
  - model = "gpt-5.4"
  - reasoning.effort = "none"      (required for temp control)
  - temperature = 0.0              (deterministic)
  - text.format.type = "json_object"
  - text.verbosity = "low"

Reads the EXISTING batch JSONL inputs (same prompts as GPT-5 ran), so this is
a pure model+temperature swap. Outputs to NEW files so GPT-5 results are kept
intact for comparison.

Resumable: skips custom_ids already present in the output file.

Usage:
  python pipeline/gpt5.4_audit/run_audit.py paraphrase
  python pipeline/gpt5.4_audit/run_audit.py hallucination
  python pipeline/gpt5.4_audit/run_audit.py both
"""
import asyncio, json, os, sys, time
from pathlib import Path

BASE = os.environ.get("SCER_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load OPENAI_API_KEY from a .env file next to the release if present;
# otherwise rely on the environment variable already being set.
env_path = os.path.join(BASE, ".env")
if os.path.exists(env_path):
    for line in Path(env_path).read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set; export it or add it to .env next to the release root.")

from openai import AsyncOpenAI
from openai import RateLimitError, APITimeoutError, APIError
OUT_DIR = os.path.join(BASE, "pipeline", "gpt5.4_audit", "outputs")
CONCURRENCY = 20
MAX_RETRIES = 3
MODEL = "gpt-5.4"

PATHS = {
    "paraphrase":    ("paraphrase_audit_batch_input.jsonl",
                       "paraphrase_audit_gpt54_temp0_output.jsonl", 400),
    "hallucination": ("hallucination_audit_batch_input.jsonl",
                       "hallucination_audit_gpt54_temp0_output.jsonl", 600),
}


async def call_one(client, req, sem, stats, max_out):
    body = req["body"]; cid = req["custom_id"]
    # Reconstruct prompt: take the user message content (single message)
    prompt_text = body["messages"][0]["content"]
    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.responses.create(
                    model=MODEL,
                    input=prompt_text,
                    reasoning={"effort": "none"},
                    temperature=0.0,
                    text={"format": {"type": "json_object"}, "verbosity": "low"},
                    max_output_tokens=max_out,
                )
                content = (getattr(resp, "output_text", None) or "").strip()
                if not content:
                    try:
                        for o in resp.output:
                            for c in o.content:
                                if hasattr(c, "text") and c.text:
                                    content = c.text; break
                            if content: break
                    except Exception: pass
                in_tok = getattr(resp.usage, "input_tokens", 0)
                out_tok = getattr(resp.usage, "output_tokens", 0)
                stats["ok"] += 1
                stats["in"] += in_tok
                stats["out"] += out_tok
                return {
                    "custom_id": cid,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "id": getattr(resp, "id", ""),
                            "model": getattr(resp, "model", MODEL),
                            "choices": [{
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }],
                            "usage": {"prompt_tokens": in_tok,
                                      "completion_tokens": out_tok,
                                      "total_tokens": in_tok + out_tok},
                        },
                    },
                    "error": None,
                }
            except (RateLimitError, APITimeoutError):
                await asyncio.sleep(2 ** attempt); continue
            except APIError as e:
                stats["err"] += 1
                return {"custom_id": cid, "response": None,
                         "error": {"code": getattr(e, "code", "api_error"),
                                   "message": str(e)[:200]}}
            except Exception as e:
                stats["err"] += 1
                return {"custom_id": cid, "response": None,
                         "error": {"code": "exception", "message": str(e)[:200]}}
        stats["err"] += 1
        return {"custom_id": cid, "response": None,
                 "error": {"code": "max_retries", "message": "exhausted retries"}}


async def run_file(input_path, output_path, max_out):
    requests = []
    with open(input_path) as f:
        for line in f: requests.append(json.loads(line))
    done = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try: done.add(json.loads(line)["custom_id"])
                except Exception: pass
    pending = [r for r in requests if r["custom_id"] not in done]
    print(f"  {os.path.basename(input_path)}: total={len(requests)} done={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        print("  All done already!"); return

    client = AsyncOpenAI(timeout=60.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {"ok": 0, "err": 0, "in": 0, "out": 0}
    t0 = time.time()

    out_f = open(output_path, "a")
    out_lock = asyncio.Lock()

    async def call_and_write(req):
        result = await call_one(client, req, sem, stats, max_out)
        async with out_lock:
            out_f.write(json.dumps(result) + "\n"); out_f.flush()
        n_done = stats["ok"] + stats["err"]
        if n_done % 50 == 0 or n_done == len(pending):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            eta = (len(pending) - n_done) / rate if rate > 0 else 0
            cost = (stats["in"] * 2.50 + stats["out"] * 15.0) / 1e6
            print(f"    [{n_done}/{len(pending)}] ok={stats['ok']} err={stats['err']} "
                  f"rate={rate:.1f}/s eta={eta/60:.1f}min  "
                  f"tokens={stats['in']/1000:.0f}K in / {stats['out']/1000:.0f}K out  cost=${cost:.2f}",
                  flush=True)

    await asyncio.gather(*[call_and_write(r) for r in pending])
    out_f.close()
    await client.close()
    cost = (stats["in"] * 2.50 + stats["out"] * 15.0) / 1e6
    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f}min: ok={stats['ok']} err={stats['err']} cost=${cost:.2f}")


async def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["paraphrase", "hallucination"]
    if "both" in targets: targets = ["paraphrase", "hallucination"]
    for t in targets:
        if t not in PATHS:
            print(f"  Unknown target: {t} (valid: {list(PATHS)})"); continue
        inp, out, max_out = PATHS[t]
        inp_path = os.path.join(OUT_DIR, inp)
        out_path = os.path.join(OUT_DIR, out)
        print(f"\n=== {t.upper()} (model={MODEL}, temp=0, reasoning=none) ===")
        await run_file(inp_path, out_path, max_out)


if __name__ == "__main__":
    asyncio.run(main())
