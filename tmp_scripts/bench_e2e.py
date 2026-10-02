"""Prefill-heavy e2e load generator for the fused w13 epilogue A/B.

The fused epilogue only engages on the gmm/prefill LoRA path
(_can_fuse_w13_epilogue requires moe_gmm_enabled()), so the metric that should
move is TTFT. TPOT is reported as a control: it should NOT change, and if it
does, something other than this kernel is drifting between runs.

Requests go to the LoRA adapter, since the epilogue only exists on the LoRA
path -- an unadapted request never reaches the kernel.

Deterministic: fixed seed, fixed prompt lengths, fixed arrival order, so the
two runs differ only in whether the fused kernel is enabled.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


def build_prompts(n, min_tok, max_tok, seed, tokenizer_path):
    """Token-accurate prompts so prefill cost is controlled, not approximated."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    rng = random.Random(seed)
    # a fixed pool of ordinary tokens; sampling ids directly keeps length exact
    vocab = tok.vocab_size
    prompts = []
    for _ in range(n):
        want = rng.randint(min_tok, max_tok)
        ids = [rng.randrange(1000, min(vocab, 100000)) for _ in range(want)]
        prompts.append(tok.decode(ids, skip_special_tokens=True))
    return prompts


async def one(session, url, model, prompt, max_tokens, results):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft = None
    ntok = 0
    prompt_tokens = None
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"error": f"HTTP {resp.status}: {await resp.text()}"})
                return
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    prompt_tokens = chunk["usage"].get("prompt_tokens")
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    ntok += 1
    except Exception as e:  # noqa: BLE001
        results.append({"error": repr(e)})
        return
    total = time.perf_counter() - t0
    results.append({
        "ttft": ttft,
        "total": total,
        "ntok": ntok,
        "prompt_tokens": prompt_tokens,
        "tpot": (total - ttft) / max(ntok - 1, 1) if ttft is not None else None,
    })


async def run(args, prompts):
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    results = []
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)

    async def guarded(session, p):
        async with sem:
            await one(session, url, args.model, p, args.max_tokens, results)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        t0 = time.perf_counter()
        await asyncio.gather(*(guarded(session, p) for p in prompts))
        wall = time.perf_counter() - t0
    return results, wall


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(int(len(xs) * p / 100), len(xs) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=1995)
    ap.add_argument("--model", default="lora-adapter")
    ap.add_argument("--tokenizer", default="/home/russia_mmo/models/dsv4-w8a8-mtp")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--min-prompt", type=int, default=4096)
    ap.add_argument("--max-prompt", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prompts = build_prompts(args.n + args.warmup, args.min_prompt, args.max_prompt,
                            args.seed, args.tokenizer)
    warm, measured = prompts[:args.warmup], prompts[args.warmup:]

    if warm:
        asyncio.run(run(args, warm))

    results, wall = asyncio.run(run(args, measured))

    errs = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r and r.get("ttft") is not None]
    if errs:
        print(f"{len(errs)} errors, first: {errs[0]['error'][:300]}")
    if not ok:
        raise SystemExit("no successful requests")

    ttfts = [r["ttft"] for r in ok]
    tpots = [r["tpot"] for r in ok if r["tpot"] is not None]
    ptoks = [r["prompt_tokens"] for r in ok if r["prompt_tokens"]]
    otoks = sum(r["ntok"] for r in ok)

    summary = {
        "label": args.label,
        "n_ok": len(ok),
        "n_err": len(errs),
        "wall_s": round(wall, 3),
        "prompt_tokens_total": sum(ptoks),
        "output_tokens_total": otoks,
        "prefill_tok_per_s": round(sum(ptoks) / wall, 1) if ptoks else None,
        "output_tok_per_s": round(otoks / wall, 1),
        "ttft_mean_s": round(statistics.mean(ttfts), 4),
        "ttft_median_s": round(statistics.median(ttfts), 4),
        "ttft_p90_s": round(pct(ttfts, 90), 4),
        "tpot_mean_ms": round(statistics.mean(tpots) * 1000, 3) if tpots else None,
        "tpot_median_ms": round(statistics.median(tpots) * 1000, 3) if tpots else None,
    }
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
