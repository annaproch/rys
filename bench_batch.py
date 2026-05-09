"""Benchmark batched /v1/completions throughput on the running server.

Sends 8 FinQA prompts in chunks of size B (1, 2, 4, 8) and reports
end-to-end wall and aggregate decode throughput. Must be run AFTER restarting
serve.py with the batched-generate patch applied.

Run:  uv run python bench_batch.py
"""

from __future__ import annotations

import time

import requests
from datasets import load_dataset

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "dnhkng/RYS-Qwen3.5-27B-FP8-M"
N_PROMPTS = 8
MAX_TOKENS = 200  # tighter than the eval to keep the bench short
STOP = ["\nQuestion:", "\n\nQuestion:"]


def fetch_prompts(n: int) -> list[str]:
    ds = load_dataset("ChanceFocus/flare-finqa", split="test")
    return [ds[i]["query"] for i in range(n)]


def run_chunked(prompts: list[str], batch: int) -> tuple[float, int]:
    """Send len(prompts) prompts split into requests of size `batch`."""
    total_gen = 0
    t0 = time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i : i + batch]
        r = requests.post(
            URL,
            json={
                "model": MODEL,
                "prompt": chunk,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stop": STOP,
            },
            timeout=900,
        )
        r.raise_for_status()
        total_gen += r.json()["usage"]["completion_tokens"]
    return time.time() - t0, total_gen


def main() -> None:
    prompts = fetch_prompts(N_PROMPTS)
    print(f"warming up server with 1 small request…")
    requests.post(
        URL,
        json={"model": MODEL, "prompt": "Hello", "max_tokens": 4, "temperature": 0.0},
        timeout=120,
    ).raise_for_status()

    print(f"\nbenchmarking {N_PROMPTS} FinQA prompts × max_tokens={MAX_TOKENS}\n")
    print(f"{'batch':>6} {'wall(s)':>10} {'gen tok':>10} {'tok/s':>10} {'q/s':>8} {'speedup':>10}")
    baseline: float | None = None
    for B in (1, 2, 4, 8):
        wall, gen = run_chunked(prompts, batch=B)
        tps = gen / wall
        qps = N_PROMPTS / wall
        speedup = wall / baseline if baseline else 1.0
        speedup_str = f"{1/speedup:.2f}x" if baseline else "—"
        if baseline is None:
            baseline = wall
        print(f"{B:>6d} {wall:>10.1f} {gen:>10d} {tps:>10.2f} {qps:>8.3f} {speedup_str:>10}")


if __name__ == "__main__":
    main()
