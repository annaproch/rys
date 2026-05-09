"""Hand-graded-style scoring of RYS vs Base on the 100-q FinQA subsample.

For each of the 100 questions we:
  1. Strip <think> blocks and LaTeX noise.
  2. Locate the model's "answer block" — the chunk after the last
     Answer:/Final answer:/Conclusion:/Therefore: marker, OR the last
     non-empty paragraph if no marker is present.
  3. Look for a yes/no token (when gold is yes/no).
  4. Otherwise extract the numeric the model intends:
     - bolded number with % (e.g. **11.69%**) wins
     - any % token in the block
     - bolded number without %
     - first number-shaped token
  5. Detect prose-sign cues ("decrease/decline/drop/loss/fall by N",
     "negative N", "(N)") to negate magnitudes when the model expressed
     negativity in words rather than with an explicit minus sign.
  6. Compare against gold with 1% relative + 1e-3 absolute tolerance.

Output: `manual_analysis.jsonl` — one row per question, with the gold,
the question, the FULL response from each model, our extracted answer,
our judgment, and which heuristic fired. Inspect by hand for any case.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass

# --- shared regex / normalisation -------------------------------------------

_NUM_RE = re.compile(
    r"""
    [-+]?            # optional sign
    \$?              # optional leading $
    \d[\d,]*         # main digits with commas
    (?:\.\d+)?       # optional decimal
    (?:e[-+]?\d+)?   # optional exponent
    \s*%?            # optional trailing %
    """,
    re.VERBOSE | re.IGNORECASE,
)

_PROSE_NEG_RE = re.compile(
    r"\b(?:decrease|decreased|decline|declined|drop|dropped|fell|fall|reduced|"
    r"reduction|loss|negative|down)\b",
    re.IGNORECASE,
)


def _normalise(text: str) -> str:
    if not text:
        return ""
    text = text.replace(r"\%", "%").replace(r"\$", "$")
    text = re.sub(r"\$\$([^$]*)\$\$", r"\1", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    t = s.strip().lower()
    if not t:
        return None
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]
    mult = 1.0
    for suffix, factor in (
        ("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3),
        (" tn", 1e12), (" bn", 1e9), (" mn", 1e6), (" m", 1e6), (" b", 1e9), (" k", 1e3),
    ):
        if t.endswith(suffix):
            mult = factor
            t = t[: -len(suffix)].strip()
            break
    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()
    t = t.replace("$", "").replace(",", "").replace(" ", "")
    if not t or t in {"-", "+", "."}:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    v *= mult
    if pct:
        v /= 100.0
    return v


# --- answer-block location --------------------------------------------------

_MARKERS = (
    r"final\s+answer",
    r"\bconclusion\b",
    r"\bthe\s+final\s+answer\b",
    r"\banswer\b",
    r"\btherefore\b",
)


def find_answer_block(text: str) -> tuple[str, str]:
    """Return (block_text, source_label).

    Searches for the LAST occurrence of any answer marker, then returns the
    text from that marker to either: the next answer marker further on, the
    end of text. Falls back to the last non-empty paragraph.
    """
    norm = _normalise(text)
    if not norm.strip():
        return "", "empty"

    last_match = None
    last_marker = None
    for marker in _MARKERS:
        for m in re.finditer(rf"(?i){marker}\s*[:=]?\s*", norm):
            if last_match is None or m.start() > last_match.start():
                last_match = m
                last_marker = marker
    if last_match is not None:
        block = norm[last_match.end():].strip()
        # cut at the next double-blank if any
        nb = block.find("\n\n\n")
        if nb > 0:
            block = block[:nb]
        return block, f"marker:{last_marker}"

    # No marker — use the final non-empty paragraph
    paragraphs = [p for p in norm.split("\n\n") if p.strip()]
    if paragraphs:
        return paragraphs[-1].strip(), "last-paragraph"
    return norm.strip(), "raw-tail"


# --- value extraction inside the block --------------------------------------

def _bolded_chunks(s: str) -> list[str]:
    return re.findall(r"\*\*([^*\n]+?)\*\*", s)


def extract_value(block: str) -> tuple[float | None, str]:
    """Pick the model's intended numeric answer from `block`.
    Returns (value, method-label).
    """
    if not block:
        return None, "no-block"

    # 1. Bolded with %
    for chunk in _bolded_chunks(block):
        for tok in _NUM_RE.findall(chunk):
            if "%" in tok:
                v = _to_float(tok)
                if v is not None:
                    return v, "bolded-pct"

    # 2. Any % token in block
    for tok in _NUM_RE.findall(block):
        if "%" in tok:
            v = _to_float(tok)
            if v is not None:
                return v, "any-pct"

    # 3. Bolded number with explicit unit (million/billion/k)
    for chunk in _bolded_chunks(block):
        for unit in ("million", "billion", "thousand"):
            if unit in chunk.lower():
                for tok in _NUM_RE.findall(chunk):
                    v = _to_float(tok + " " + unit)
                    if v is not None:
                        return v, f"bolded-{unit}"

    # 4. Bolded plain number
    for chunk in _bolded_chunks(block):
        for tok in _NUM_RE.findall(chunk):
            v = _to_float(tok)
            if v is not None:
                return v, "bolded-plain"

    # 5. Number after "= " or "≈"
    for m in re.finditer(r"(?:=|≈|approx(?:imately)?\.?|\\approx)\s*(-?\$?[\d,]+(?:\.\d+)?\s*%?)", block, re.IGNORECASE):
        v = _to_float(m.group(1))
        if v is not None:
            return v, "after-equals"

    # 6. First number-shaped token in the block
    for tok in _NUM_RE.findall(block):
        v = _to_float(tok)
        if v is not None:
            return v, "first-number"
    return None, "no-number-in-block"


def extract_yes_no(text: str) -> tuple[str | None, str]:
    text = _normalise(text)
    if not text:
        return None, "empty"
    block, _ = find_answer_block(text)
    chunk = (block or text).lower()
    yes = re.search(r"\byes\b", chunk)
    no_match = re.search(r"\bno\b", chunk)
    if yes and not no_match:
        return "yes", "yes-direct"
    if no_match and not yes:
        return "no", "no-direct"
    if yes and no_match:
        # take whichever appears first
        if yes.start() < no_match.start():
            return "yes", "yes-first"
        return "no", "no-first"
    return None, "no-yn"


# --- detect prose-negation around the chosen number -------------------------

def detect_prose_negation(block: str) -> bool:
    """Heuristic: did the model express the answer as a decrease/decline/drop?"""
    if not block:
        return False
    # Look in the answer block for negation cues.
    if _PROSE_NEG_RE.search(block):
        return True
    return False


# --- per-question grading ---------------------------------------------------

@dataclass
class Grade:
    extracted: float | str | None
    correct: bool
    method: str
    block: str


def grade_one(response: str, gold: str) -> Grade:
    gold_clean = gold.strip()
    gold_low = gold_clean.lower()
    norm = _normalise(response)
    block, source = find_answer_block(norm)

    if gold_low in {"yes", "no"}:
        pred_yn, yn_method = extract_yes_no(response)
        return Grade(
            extracted=pred_yn,
            correct=(pred_yn == gold_low),
            method=f"{source}|{yn_method}",
            block=block,
        )

    gold_num = _to_float(gold_clean)
    pred_num, val_method = extract_value(block or norm)

    # Apply prose-sign if the model wrote about a decrease/decline/drop
    # but the extracted value is positive AND the gold is negative.
    sign_applied = False
    if pred_num is not None and pred_num > 0 and detect_prose_negation(block):
        # Only flip if gold is also negative — we don't want to wrongly
        # negate cases where the model was right with a positive value and
        # "decrease" appears tangentially.
        if gold_num is not None and gold_num < 0:
            pred_num = -pred_num
            sign_applied = True

    if gold_num is None or pred_num is None:
        correct = False
    else:
        correct = abs(gold_num - pred_num) <= max(1e-3, 0.01 * abs(gold_num))

    method = f"{source}|{val_method}" + ("|prose-neg-applied" if sign_applied else "")
    return Grade(extracted=pred_num, correct=correct, method=method, block=block)


# --- main -------------------------------------------------------------------

RYS_GLOB = "runs/rys-finqa-sub100/dnhkng__RYS-Qwen3.5-27B-FP8-M/samples_finqa_*.jsonl"
BASE_GLOB = "runs/base-finqa-sub100/Qwen__Qwen3.5-27B-FP8/samples_finqa_*.jsonl"


def load_latest(glob_pat: str) -> list[dict]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise FileNotFoundError(glob_pat)
    return [json.loads(l) for l in open(paths[-1])]


def get_question(prompt: str) -> str:
    qi = prompt.rfind("Question:")
    if qi < 0:
        return ""
    return prompt[qi:].split("\nAnswer:")[0].replace("Question:", "").strip()


def main() -> None:
    rys = {s["doc_id"]: s for s in load_latest(RYS_GLOB)}
    base = {s["doc_id"]: s for s in load_latest(BASE_GLOB)}
    assert sorted(rys.keys()) == sorted(base.keys()) == list(range(100))

    out_path = "manual_analysis.jsonl"
    with open(out_path, "w") as f:
        for did in range(100):
            r = rys[did]
            b = base[did]
            gold = str(r["target"]).strip()
            question = get_question(r["arguments"]["gen_args_0"]["arg_0"])
            r_resp = r["resps"][0][0]
            b_resp = b["resps"][0][0]

            r_grade = grade_one(r_resp, gold)
            b_grade = grade_one(b_resp, gold)

            row = {
                "doc_id": did,
                "question": question,
                "gold": gold,
                "rys": {
                    "extracted": r_grade.extracted,
                    "correct": r_grade.correct,
                    "method": r_grade.method,
                    "answer_block": r_grade.block[:400],
                    "full_response": r_resp,
                },
                "base": {
                    "extracted": b_grade.extracted,
                    "correct": b_grade.correct,
                    "method": b_grade.method,
                    "answer_block": b_grade.block[:400],
                    "full_response": b_resp,
                },
                "outcome": (
                    "both-correct" if r_grade.correct and b_grade.correct
                    else "both-wrong" if not r_grade.correct and not b_grade.correct
                    else "rys-only" if r_grade.correct
                    else "base-only"
                ),
            }
            f.write(json.dumps(row) + "\n")

    # ---------- metrics ----------
    rows = [json.loads(l) for l in open(out_path)]
    rys_correct = sum(1 for r in rows if r["rys"]["correct"])
    base_correct = sum(1 for r in rows if r["base"]["correct"])
    rys_only = sum(1 for r in rows if r["outcome"] == "rys-only")
    base_only = sum(1 for r in rows if r["outcome"] == "base-only")
    both_right = sum(1 for r in rows if r["outcome"] == "both-correct")
    both_wrong = sum(1 for r in rows if r["outcome"] == "both-wrong")

    def stderr(p, n=100):
        return (p * (1 - p) / n) ** 0.5

    def mcnemar_p(b, c):
        from math import comb
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))

    print("=" * 72)
    print("FinQA — manual-grade paired comparison (n=100)")
    print("=" * 72)
    print(f"RYS  acc: {rys_correct}/100 = {rys_correct}%  (stderr ±{stderr(rys_correct/100):.3f})")
    print(f"Base acc: {base_correct}/100 = {base_correct}%  (stderr ±{stderr(base_correct/100):.3f})")
    print(f"Δ (RYS − Base): {rys_correct - base_correct:+d} points")
    print()
    print("=== paired contingency ===")
    print(f"  both correct:   {both_right}")
    print(f"  both wrong:     {both_wrong}")
    print(f"  RYS-only right: {rys_only}")
    print(f"  Base-only right:{base_only}")
    print(f"  McNemar exact two-sided p = {mcnemar_p(rys_only, base_only):.4f}")
    print()
    print("=== extraction-method usage ===")
    from collections import Counter
    rys_methods = Counter(r["rys"]["method"].split("|")[1] if "|" in r["rys"]["method"] else r["rys"]["method"] for r in rows)
    base_methods = Counter(b["base"]["method"].split("|")[1] if "|" in b["base"]["method"] else b["base"]["method"] for b in rows)
    print("RYS method usage:")
    for k, v in rys_methods.most_common():
        print(f"  {k:25s} {v}")
    print("Base method usage:")
    for k, v in base_methods.most_common():
        print(f"  {k:25s} {v}")

    print(f"\n→ full evidence written to {out_path}")
    print("  Each row has the question, gold, full response from both models,")
    print("  the extracted value, the heuristic that fired, and the answer block we used.")
    print("  Inspect any row by hand to verify our judgment.")


if __name__ == "__main__":
    main()
