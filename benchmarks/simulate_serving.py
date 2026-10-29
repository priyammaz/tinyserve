"""
simulate_serving.py: load-test an OpenAI-compatible completion endpoint
with Poisson-distributed traffic.

To run:
python serve.py serve --model Qwen/Qwen3-8B -tp 2 --max-model-len 32768
python benchmarks/simulate_serving.py --model Qwen/Qwen3-8B --port 8000 --rate 3 --duration 20 --dashboard

    --dashboard will give an interactive view, created by Claude!

Results look like:

time to first token (s)      p50 0.11  p90 0.23  p99 0.31  max 0.31
total latency (s)            p50 11.98  p90 28.06  p99 36.11  max 37.80
tokens/request/s             p50 45.0  p10 41.0  p1 39.5  min 39.3

"""
import argparse
import asyncio
import random
import sys
import time
from pathlib import Path
import httpx

import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from openai import AsyncOpenAI
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

DASHBOARD_HTML_PATH = Path(__file__).resolve().parent.parent / "dashboards" / "dashboard.html"

SYSTEM_PROMPT_OPENINGS = [
    "You are a helpful assistant. Answer the user's request directly and "
    "concisely. Here are some examples of good answers:\n",
    "You are a friendly, thorough assistant. Explain your reasoning before "
    "giving the final answer. Here are some examples of good answers:\n",
    "You are a terse technical assistant. Prefer short, precise answers "
    "with no pleasantries. Here are some examples of good answers:\n",
]

def poisson_arrivals(rate: float, duration: float, rng: np.random.Generator) -> list[float]:
    arrivals = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            break
        arrivals.append(t)
    return arrivals

def build_shared_prefix(tok, alpaca, num_tokens: int, seed: int, opening: str) -> str:
    rng = random.Random(seed + 1)
    text = opening
    while len(tok(text)["input_ids"]) < num_tokens:
        x = alpaca[rng.randrange(len(alpaca))]
        text += (f"\n### Request\n{x['instruction']} {x['input'] or ''}".rstrip()
                 + f"\n### Answer\n{x['output']}\n")
    return text

def build_system_prompts(tok, alpaca, num_tokens: int, seed: int) -> list[str]:
    """One shared prefix per persona in SYSTEM_PROMPT_OPENINGS."""
    return [build_shared_prefix(tok, alpaca, num_tokens, seed + 1000 * i, opening)
            for i, opening in enumerate(SYSTEM_PROMPT_OPENINGS)]

def load_prompts(tok, n: int, edges: list[int], seed: int) -> list[str]:
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    texts = ([x["instruction"] + (x["input"] or "") for x in alpaca]
             + [x["instruction"] + " " + (x["context"] or "") for x in dolly])

    lens = [len(ids) for ids in tok(texts, add_special_tokens=True)["input_ids"]]
    names = ["short", "medium", "long"][:len(edges) - 1]
    by_bin = {name: [i for i, ln in enumerate(lens) if lo <= ln < hi]
              for name, lo, hi in zip(names, edges, edges[1:])}

    rng = random.Random(seed)
    per_bin = [n // len(names) + (i < n % len(names)) for i in range(len(names))]
    picked = []
    for name, k in zip(names, per_bin):
        avail = by_bin[name]
        if not avail:
            raise ValueError(f"bin {name!r} ({edges[names.index(name)]}-"
                             f"{edges[names.index(name) + 1]} tokens) has no prompts at all")
        picked += (rng.sample(avail, k) if len(avail) >= k
                  else [rng.choice(avail) for _ in range(k)])
    n_guaranteed = min(3, len(picked))
    candidates = [i for i in range(len(texts)) if lens[i] <= edges[-1]] or list(range(len(texts)))
    longest = sorted(candidates, key=lambda i: lens[i], reverse=True)[:n_guaranteed]
    picked[:n_guaranteed] = longest
    rng.shuffle(picked)
    return [texts[i] for i in picked], alpaca

class Dashboard:
    def __init__(self):
        self.clients: set = set()

    async def send(self, msg: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

def build_dashboard_app(dashboard: Dashboard):
    app = FastAPI()
    html = DASHBOARD_HTML_PATH.read_text()

    @app.get("/")
    async def index():
        return HTMLResponse(html)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        dashboard.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()  # not expecting client messages, just detects disconnect
        except WebSocketDisconnect:
            pass
        finally:
            dashboard.clients.discard(websocket)

    return app


async def _poll_server_stats(base_url: str, dashboard: Dashboard, interval: float = 0.5):
    async with httpx.AsyncClient() as client:
        while True:
            try:
                r = await client.get(f"{base_url}/metrics", timeout=2.0)
                r.raise_for_status()
                data = r.json()
            except Exception:
                data = {}
            await dashboard.send({"type": "stats", "data": data})
            await asyncio.sleep(interval)

async def _fire_one(client, model: str, system: str, text: str, max_tokens: int,
                     send_at: float, t0: float, idx: int, results: list,
                     dashboard: "Dashboard | None" = None, tok=None):
    delay = send_at - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    sent = time.perf_counter() - t0

    rid = None
    try:
        messages = ([{"role": "system", "content": system}] if system else []) \
                   + [{"role": "user", "content": text}]
       
        prompt_tokens_est = len(tok(tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False))["input_ids"]) \
            if tok is not None else None
        stream = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens,
            temperature=0.0, stream=True,
            stream_options={"include_usage": True},
        )
        first_token, n_chunks, cached, prompt_tokens, completion_tokens, finish_reason = \
            None, 0, 0, 0, None, None
        preempted, chunked_prefill = False, False
        async for chunk in stream:
            if rid is None:
                rid = chunk.id
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
        
            preempted = preempted or bool(getattr(chunk, "tinyserve_preempted", False))
            chunked_prefill = chunked_prefill or bool(getattr(chunk, "tinyserve_chunked_prefill", False))
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is not None and delta.content:
                if first_token is None:
                    first_token = time.perf_counter() - t0
                    if dashboard is not None:
                        await dashboard.send({"type": "start", "idx": idx, "rid": rid, "prompt": text[:60],
                                              "prompt_tokens_est": prompt_tokens_est})
                n_chunks += 1
                if dashboard is not None:
                
                    await dashboard.send({
                        "type": "token", "rid": rid, "delta": delta.content,
                        "n_tokens": n_chunks,
                        "elapsed": (time.perf_counter() - t0) - first_token,
                    })
            if chunk.usage is not None:
        
                details = getattr(chunk.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", None) or 0
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens


        n_tokens = completion_tokens if completion_tokens is not None else n_chunks
        ttft = (first_token - sent) if first_token is not None else None
        latency = (time.perf_counter() - t0) - sent
        if dashboard is not None:
            if first_token is None:
            
                await dashboard.send({"type": "start", "idx": idx, "rid": rid, "prompt": text[:60],
                                      "prompt_tokens_est": prompt_tokens_est})
            await dashboard.send({"type": "done", "rid": rid, "ttft": ttft, "latency": latency,
                                  "n_tokens": n_tokens, "cached_tokens": cached,
                                  "prompt_tokens": prompt_tokens, "finish_reason": finish_reason,
                                  "preempted": preempted, "chunked_prefill": chunked_prefill})

        results[idx] = dict(sent=sent, ttft=ttft, latency=latency,
                            n_tokens=n_tokens, cached_tokens=cached)
    except Exception as e:
        print(f"[simulate] request {idx} failed: {e}", file=sys.stderr)
        if dashboard is not None and rid is not None:
        
            await dashboard.send({"type": "done", "rid": rid, "ttft": None,
                                  "latency": (time.perf_counter() - t0) - sent,
                                  "n_tokens": 0, "cached_tokens": 0, "prompt_tokens": 0,
                                  "finish_reason": "error"})

async def _run(a):
    rng = np.random.default_rng(a.seed)
    arrivals = poisson_arrivals(a.rate, a.duration, rng)
    n = len(arrivals)
    print(f"[simulate] {n} requests over {a.duration:.0f}s "
          f"(poisson arrivals, mean rate {a.rate:.2f} req/s)")

    tok = AutoTokenizer.from_pretrained(a.model)
    edges = [int(x) for x in a.bins.split(",")]
    texts, alpaca = load_prompts(tok, n, edges, a.seed)

    if a.prefix_tokens:
        systems = build_system_prompts(tok, alpaca, a.prefix_tokens, a.seed)
        print(f"[simulate] {len(systems)} shared system prompts (~{a.prefix_tokens} tokens each, "
              f"randomly assigned per request, plus a chance of none at all):")
    else:
        systems = []
        print("[simulate] no shared system prompt (--prefix-tokens 0)")

    options = systems + [""]
    prompt_rng = random.Random(a.seed + 777)
    chosen_systems = [prompt_rng.choice(options) for _ in range(n)]

    client = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="unused")
    dashboard = server = server_task = stats_task = None
    if a.dashboard:
        import uvicorn
        dashboard = Dashboard()
        config = uvicorn.Config(build_dashboard_app(dashboard), host="0.0.0.0",
                                port=a.dashboard_port, log_level="warning")
        server = uvicorn.Server(config)
        server_task = asyncio.create_task(server.serve())
        stats_task = asyncio.create_task(_poll_server_stats(f"http://localhost:{a.port}", dashboard))
        await asyncio.sleep(0.3)  # give it a moment to bind before printing the url
        print(f"[simulate] live dashboard: http://localhost:{a.dashboard_port}/?panels={a.panels}")

    t0 = time.perf_counter()
    results = [None] * n
    await asyncio.gather(*[
        _fire_one(client, a.model, chosen_systems[i], texts[i], a.max_tokens,
                  arrivals[i], t0, i, results, dashboard, tok)
        for i in range(n)
    ])

    if server is not None:
        await asyncio.sleep(2.0)
        stats_task.cancel()
        server.should_exit = True
        await server_task

    return results

def print_stats(results: list):
    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        print("\nno requests completed")
        return
    wall = (df["sent"] + df["latency"]).max()
    gen_time = (df["latency"] - df["ttft"].fillna(0)).clip(lower=0.001)
    df["tok_per_s"] = df["n_tokens"] / gen_time

    print(f"\ncompleted {len(df)} requests over {wall:.1f}s wall")
    print(f"cached prompt tokens (prefix hits)  mean: {df['cached_tokens'].mean():.0f}")

    for col, label in (
        ("ttft", "time to first token (s)"),
        ("latency", "total latency (s)"),
    ):
        stats = df[col].quantile([0.5, 0.9, 0.99])
        print(
            f"{label:28s} "
            f"p50 {stats[0.5]:.2f}  "
            f"p90 {stats[0.9]:.2f}  "
            f"p99 {stats[0.99]:.2f}  "
            f"max {df[col].max():.2f}"
        )

    stats = df["tok_per_s"].quantile([0.5, 0.1, 0.01])
    print(
        f"{'tokens/request/s':28s} "
        f"p50 {stats[0.5]:.1f}  "
        f"p10 {stats[0.1]:.1f}  "
        f"p1 {stats[0.01]:.1f}  "
        f"min {df['tok_per_s'].min():.1f}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--rate", type=float, default=1.0, help="mean requests/sec")
    ap.add_argument("--duration", type=float, default=60.0, help="simulated window, seconds")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--bins", default="1,64,512,2048")
    ap.add_argument("--prefix-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--dashboard-port", type=int, default=8089)
    ap.add_argument("--panels", type=int, default=16)
    a = ap.parse_args()

    results = asyncio.run(_run(a))
    print_stats(results)


if __name__ == "__main__":
    main()
