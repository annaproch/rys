"""FinQA helpers for lm-evaluation-harness.

We use the FLARE mirror (`ChanceFocus/flare-finqa`) which ships the prompt
already preformatted in `query` (context + table + question + "Answer:") and
gold answers as canonical decimal strings (e.g. "94.0", "0.10039").

So:
  * `doc_to_text` returns `query` verbatim.
  * `doc_to_target` returns the gold answer.
  * `process_results` extracts the model's final number, compares to gold with
    relative tolerance.
"""

from __future__ import annotations

import re

# strip a few common modes the model might emit:
#  - thousands separators ("1,234,567" -> "1234567")
#  - currency / percent / unit suffixes
#  - parenthesised negatives "(94)" -> "-94"
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


def _to_float(s: str) -> float | None:
    """Parse a candidate numeric token to a float, or None if not parsable.

    Normalises:
      - drops commas, $, whitespace
      - converts "X%" -> X / 100
      - converts "(N)" -> -N (only at the outer level, handled before this fn)
      - "million"/"billion"/"thousand" multipliers
    """
    if s is None:
        return None
    t = s.strip().lower()
    if not t:
        return None

    # parenthesised negatives "(94)" -> "-94"
    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]

    # unit multipliers — match BEFORE stripping non-digits
    mult = 1.0
    for suffix, factor in (
        ("trillion", 1e12),
        ("billion", 1e9),
        ("million", 1e6),
        ("thousand", 1e3),
        (" tn", 1e12),
        (" bn", 1e9),
        (" mn", 1e6),
        (" m", 1e6),
        (" b", 1e9),
        (" k", 1e3),
    ):
        if t.endswith(suffix):
            mult = factor
            t = t[: -len(suffix)].strip()
            break

    pct = t.endswith("%")
    if pct:
        t = t[:-1].strip()

    # strip $, commas, spaces
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


def extract_final_number(text: str) -> float | None:
    """Pull the most likely 'final answer' number from a model response.

    Heuristics, in order:
      1. Strip any <think>...</think> reasoning blocks.
      2. If a line starts with 'Answer:' / 'Final answer:' / 'A:', use the
         number on that line.
      3. Otherwise use the LAST number-shaped token in the response.
    """
    if not text:
        return None

    # remove reasoning blocks if the model emits them
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # try "Answer:" / "Final answer:" / "A:" lines (case-insensitive)
    for marker in (r"final answer", r"answer", r"\ba\b"):
        m = re.search(rf"(?i){marker}\s*[:=]\s*(.+?)(?:\n|$)", text)
        if m:
            for tok in reversed(_NUM_RE.findall(m.group(1))):
                v = _to_float(tok)
                if v is not None:
                    return v

    # fallback: last number-shaped token in the whole text
    for tok in reversed(_NUM_RE.findall(text)):
        v = _to_float(tok)
        if v is not None:
            return v
    return None


def is_close(gold: float, pred: float, rel_tol: float = 0.01, abs_tol: float = 1e-3) -> bool:
    """Numeric equality with both relative and absolute tolerance."""
    return abs(gold - pred) <= max(abs_tol, rel_tol * abs(gold))


# --- lm-eval-harness hooks --------------------------------------------------


def doc_to_text(doc: dict) -> str:
    return doc["query"]


def doc_to_target(doc: dict) -> str:
    return " " + str(doc["answer"])


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    pred_text = results[0] if results else ""
    pred_num = extract_final_number(pred_text)
    gold_num = _to_float(str(doc["answer"]))

    if gold_num is None or pred_num is None:
        strict = 0.0
        loose = 0.0
    else:
        strict = 1.0 if is_close(gold_num, pred_num, rel_tol=0.01) else 0.0
        loose = 1.0 if is_close(gold_num, pred_num, rel_tol=0.05) else 0.0

    return {"acc": strict, "acc_loose": loose}
