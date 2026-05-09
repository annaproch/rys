"""Offline re-scoring + paired comparison for RYS vs base FinQA runs.

Loads the saved lm-eval-harness samples (--log_samples) and re-applies the
current `tasks/finqa/utils.py` extractor + scorer. No re-generation needed —
all model outputs are already on disk.

Usage:  uv run python rescore.py
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from statistics import mean

sys.path.insert(0, "tasks/finqa")
from utils import _to_float, extract_final_number, extract_yes_no, process_results

RYS_GLOB = "runs/rys-finqa-sub100/dnhkng__RYS-Qwen3.5-27B-FP8-M/samples_finqa_*.jsonl"
BASE_GLOB = "runs/base-finqa-sub100/Qwen__Qwen3.5-27B-FP8/samples_finqa_*.jsonl"


def load_samples(pattern: str) -> list[dict]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no samples matched {pattern}")
    # Use the latest if multiple
    return [json.loads(l) for l in open(paths[-1])]


def rescore(samples: list[dict]) -> list[tuple[int, float, str, float | str | None, str]]:
    """Returns list of (doc_id, acc, gold, pred, response)."""
    out = []
    for s in samples:
        gold = str(s["target"]).strip()
        resp = s["resps"][0][0] if s["resps"] else ""
        result = process_results({"answer": gold}, [resp])
        gold_low = gold.lower()
        pred = (
            extract_yes_no(resp)
            if gold_low in {"yes", "no"}
            else extract_final_number(resp)
        )
        out.append((s["doc_id"], result["acc"], gold, pred, resp))
    return out


def fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} = {100.0 * n / total:.1f}%"


def stderr(p: float, n: int) -> float:
    return (p * (1 - p) / n) ** 0.5


def mcnemar_pvalue(b: int, c: int) -> float:
    """Exact two-sided binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    # P(X <= min(b,c)) under Binomial(n, 0.5), two-sided
    from math import comb
    k = min(b, c)
    one_tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0**n)
    return min(1.0, 2 * one_tail)


def main() -> None:
    rys_samples = load_samples(RYS_GLOB)
    base_samples = load_samples(BASE_GLOB)

    rys = rescore(rys_samples)
    base = rescore(base_samples)
    assert len(rys) == len(base) == 100, f"expected 100/100, got {len(rys)}/{len(base)}"

    # Sort by doc_id for paired comparison
    rys.sort(key=lambda x: x[0])
    base.sort(key=lambda x: x[0])
    assert all(r[0] == b[0] for r, b in zip(rys, base)), "doc_id mismatch"

    rys_acc = [r[1] for r in rys]
    base_acc = [b[1] for b in base]

    print("=" * 72)
    print("FinQA — 100-question paired comparison (re-scored with patched extractor)")
    print("=" * 72)
    print(f"RYS  acc: {fmt_pct(int(sum(rys_acc)), 100)}  (stderr ±{stderr(mean(rys_acc), 100):.3f})")
    print(f"Base acc: {fmt_pct(int(sum(base_acc)), 100)}  (stderr ±{stderr(mean(base_acc), 100):.3f})")
    print(f"Δ (RYS − Base): {sum(rys_acc) - sum(base_acc):+.1f} points")
    print()

    # Paired McNemar
    both_right = sum(1 for r, b in zip(rys_acc, base_acc) if r == 1.0 and b == 1.0)
    both_wrong = sum(1 for r, b in zip(rys_acc, base_acc) if r == 0.0 and b == 0.0)
    rys_only = sum(1 for r, b in zip(rys_acc, base_acc) if r == 1.0 and b == 0.0)
    base_only = sum(1 for r, b in zip(rys_acc, base_acc) if r == 0.0 and b == 1.0)
    print("=== paired contingency ===")
    print(f"  both correct:   {both_right}")
    print(f"  both wrong:     {both_wrong}")
    print(f"  RYS-only right: {rys_only}")
    print(f"  Base-only right:{base_only}")
    p = mcnemar_pvalue(rys_only, base_only)
    print(f"  McNemar exact two-sided p-value: {p:.4f}")
    if p < 0.05:
        print("  → discordance is unlikely under H0 of equal accuracy (p<0.05)")
    else:
        print("  → cannot reject H0 of equal accuracy at α=0.05")
    print()

    # By-bucket accuracy
    def bucket(g: str) -> str:
        gl = g.lower()
        if gl in {"yes", "no"}:
            return "yes/no"
        f = _to_float(g)
        if f is None:
            return "unparseable"
        if abs(f) < 1:
            return "small (<1)"
        if abs(f) < 100:
            return "mid (1..100)"
        return "large (≥100)"

    print("=== accuracy by gold bucket ===")
    print(f"{'bucket':<22}{'n':>5}{'RYS':>10}{'Base':>10}{'Δ':>8}")
    bucket_to_indices: dict[str, list[int]] = {}
    for i, (_, _, g, _, _) in enumerate(rys):
        bucket_to_indices.setdefault(bucket(g), []).append(i)
    for bk, idxs in bucket_to_indices.items():
        r = sum(rys_acc[i] for i in idxs)
        b = sum(base_acc[i] for i in idxs)
        print(f"{bk:<22}{len(idxs):>5}{int(r):>10}{int(b):>10}{int(r - b):>+8d}")
    print()

    # Disagreement examples — what RYS got right that base didn't (and vice versa)
    print("=== sample disagreements (first 5 each) ===")
    for label, predicate in [
        ("RYS-only correct", lambda i: rys_acc[i] == 1.0 and base_acc[i] == 0.0),
        ("Base-only correct", lambda i: rys_acc[i] == 0.0 and base_acc[i] == 1.0),
    ]:
        print(f"\n--- {label} ---")
        printed = 0
        for i in range(100):
            if predicate(i) and printed < 5:
                _, _, gold, rpred, _ = rys[i]
                _, _, _, bpred, _ = base[i]
                print(f"  doc_id={rys[i][0]:>3}  gold={gold!r:>12}  RYS->{rpred!r:>12}  Base->{bpred!r:>12}")
                printed += 1


if __name__ == "__main__":
    main()
