"""Aggregate the 4 LLM-as-judge batch outputs and compute paired metrics."""

import glob
import json
from math import comb

# Load and validate
batches = sorted(glob.glob("judge_outputs/batch_*.jsonl"))
assert len(batches) == 4, f"expected 4 batch files, got {len(batches)}"

rows = []
for path in batches:
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

# Sanity: 100 unique doc_ids 0..99
ids = sorted(r["doc_id"] for r in rows)
assert ids == list(range(100)), f"doc_ids missing/dup: {ids}"

rows.sort(key=lambda r: r["doc_id"])

# Save consolidated file with the inputs joined (full evidence for audit)
inputs = {}
for path in sorted(glob.glob("judge_inputs/batch_*.jsonl")):
    for line in open(path):
        c = json.loads(line)
        inputs[c["doc_id"]] = c

with open("judge_results.jsonl", "w") as f:
    for r in rows:
        c = inputs[r["doc_id"]]
        merged = {
            "doc_id": r["doc_id"],
            "question": c["question"],
            "gold": c["gold"],
            "rys": {
                "correct": r["rys_correct"],
                "reasoning": r["rys_reasoning"],
                "full_response": c["rys_response"],
            },
            "base": {
                "correct": r["base_correct"],
                "reasoning": r["base_reasoning"],
                "full_response": c["base_response"],
            },
            "outcome": (
                "both-correct" if r["rys_correct"] and r["base_correct"]
                else "both-wrong" if not r["rys_correct"] and not r["base_correct"]
                else "rys-only" if r["rys_correct"]
                else "base-only"
            ),
        }
        f.write(json.dumps(merged) + "\n")

# Metrics
rys_correct = sum(1 for r in rows if r["rys_correct"])
base_correct = sum(1 for r in rows if r["base_correct"])
both_right = sum(1 for r in rows if r["rys_correct"] and r["base_correct"])
both_wrong = sum(1 for r in rows if not r["rys_correct"] and not r["base_correct"])
rys_only = sum(1 for r in rows if r["rys_correct"] and not r["base_correct"])
base_only = sum(1 for r in rows if r["base_correct"] and not r["rys_correct"])


def stderr(p, n=100):
    return (p * (1 - p) / n) ** 0.5


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


print("=" * 72)
print("FinQA — LLM-as-judge paired comparison (n=100)")
print("=" * 72)
print(f"RYS  acc: {rys_correct}/100 = {rys_correct}%   (stderr ±{stderr(rys_correct/100):.3f})")
print(f"Base acc: {base_correct}/100 = {base_correct}%   (stderr ±{stderr(base_correct/100):.3f})")
print(f"Δ (RYS − Base): {rys_correct - base_correct:+d} points")
print()
print("=== paired contingency ===")
print(f"  both correct:    {both_right}")
print(f"  both wrong:      {both_wrong}")
print(f"  RYS-only right:  {rys_only}")
print(f"  Base-only right: {base_only}")
print(f"  McNemar exact two-sided p = {mcnemar_p(rys_only, base_only):.4f}")
print()

# Bucket-wise
def bucket(g):
    gl = g.strip().lower()
    if gl in {"yes", "no"}:
        return "yes/no"
    try:
        f = float(g.strip())
    except ValueError:
        return "unparseable"
    if abs(f) < 1: return "small (<1)"
    if abs(f) < 100: return "mid (1..100)"
    return "large (≥100)"


print("=== accuracy by gold bucket ===")
print(f"{'bucket':<22}{'n':>5}{'RYS':>10}{'Base':>10}{'Δ':>8}")
buckets = {}
for r in rows:
    bk = bucket(inputs[r["doc_id"]]["gold"])
    buckets.setdefault(bk, []).append(r)
for bk, items in sorted(buckets.items()):
    rs = sum(1 for x in items if x["rys_correct"])
    bs = sum(1 for x in items if x["base_correct"])
    print(f"{bk:<22}{len(items):>5}{rs:>10}{bs:>10}{rs - bs:>+8d}")

print()
print(f"→ full audit file: judge_results.jsonl")
print(f"  ({len(rows)} rows, includes question, gold, both responses, judge reasoning)")
