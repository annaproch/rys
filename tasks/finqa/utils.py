"""FinQA helpers for lm-evaluation-harness.

We use the FLARE mirror (`ChanceFocus/flare-finqa`) which ships the prompt
already preformatted in `query` (context + table + question + "Answer:") and
gold answers as canonical decimal strings (e.g. "94.0", "0.10039") plus a
small minority of yes/no answers.

Scoring path:
  * `doc_to_text` returns `query` verbatim.
  * `doc_to_target` returns the gold answer.
  * `process_results` extracts the model's final answer (numeric or yes/no)
    and compares to gold (numeric: relative tolerance; yes/no: string match).
"""

from __future__ import annotations

import re

# Number-shaped token: optional sign, optional $, digits with commas, optional
# decimal, optional exponent, optional trailing % (handled after match).
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
    """Parse a candidate numeric token to a float, or None if not parsable."""
    if s is None:
        return None
    t = s.strip().lower()
    if not t:
        return None

    if t.startswith("(") and t.endswith(")"):
        t = "-" + t[1:-1]

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


def _normalise(text: str) -> str:
    """Pre-substitute LaTeX/markdown noise that confuses tokenisation."""
    if not text:
        return text
    # LaTeX percent: 111.97\% -> 111.97%
    text = text.replace(r"\%", "%")
    # LaTeX dollar: \$ -> $
    text = text.replace(r"\$", "$")
    # Remove dollar-math delimiters that wrap inline math: $...$ -> ...
    # (only the delimiters, not the content)
    text = re.sub(r"\$\$([^$]*)\$\$", r"\1", text)
    return text


def _strip_thinking(text: str) -> str:
    """Strip any closed <think>...</think> blocks. Unclosed blocks are kept
    as-is so we still see the model's reasoning when generation was truncated.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _pick_best_number(line: str) -> float | None:
    """From a single answer-line, pick the model's intended numeric answer.

    Priority order (highest first):
      1. A bolded percentage like **11.69%**.
      2. Any percentage anywhere on the line (model usually states the answer
         in percent form when the gold is a fraction).
      3. A bolded plain number like **94 million**.
      4. The first numeric token on the line.

    The earlier "last number" rule was wrong: in lines like
    "**11.69%** of payments are due in 2018." it picked 2018.
    """
    if not line:
        return None

    bolded = re.findall(r"\*\*([^*\n]+?)\*\*", line)

    # 1. Bolded with %
    for chunk in bolded:
        for tok in _NUM_RE.findall(chunk):
            if "%" in tok:
                v = _to_float(tok)
                if v is not None:
                    return v

    # 2. Any % token on the line
    for tok in _NUM_RE.findall(line):
        if "%" in tok:
            v = _to_float(tok)
            if v is not None:
                return v

    # 3. Bolded numeric without %
    for chunk in bolded:
        for tok in _NUM_RE.findall(chunk):
            v = _to_float(tok)
            if v is not None:
                return v

    # 4. First numeric token on the line
    for tok in _NUM_RE.findall(line):
        v = _to_float(tok)
        if v is not None:
            return v

    return None


def extract_final_number(text: str) -> float | None:
    """Pull the most likely 'final answer' number from a model response.

    1. Normalise LaTeX `\\%`/`\\$` and strip <think>…</think> blocks.
    2. If an "Answer:" / "Final answer:" line exists, prefer-bolded-%-then-first
       number on that line.
    3. Otherwise, scan for the line with the strongest "answer signal" (bolded
       % anywhere) and apply the same priority.
    4. Fallback: last numeric-with-% token in the response, then last numeric
       token overall.
    """
    if not text:
        return None

    text = _normalise(text)
    text = _strip_thinking(text)

    # 2. Answer / Final answer line
    for marker in (r"final\s+answer", r"\banswer\b", r"\b[aA]\b"):
        m = re.search(rf"(?i){marker}\s*[:=]\s*(.+?)(?:\n|$)", text)
        if m:
            v = _pick_best_number(m.group(1))
            if v is not None:
                return v

    # 3. No explicit Answer marker — find any bolded line with a %
    for line in reversed(text.splitlines()):
        if "**" in line and "%" in line:
            v = _pick_best_number(line)
            if v is not None:
                return v

    # 4. Fallbacks: prefer % tokens, then any number, scanned from the end.
    all_toks = _NUM_RE.findall(text)
    for tok in reversed(all_toks):
        if "%" in tok:
            v = _to_float(tok)
            if v is not None:
                return v
    for tok in reversed(all_toks):
        v = _to_float(tok)
        if v is not None:
            return v
    return None


def extract_yes_no(text: str) -> str | None:
    """Pull a yes/no answer. Used when gold is exactly 'yes' or 'no'."""
    if not text:
        return None
    text = _strip_thinking(text)

    # Prefer the answer line if present
    for marker in (r"final\s+answer", r"\banswer\b"):
        m = re.search(rf"(?i){marker}\s*[:=]\s*(.+?)(?:\n|$)", text)
        if m:
            chunk = m.group(1).lower()
            # Word-boundary match to avoid matching "no" inside "not", etc.
            if re.search(r"\byes\b", chunk):
                return "yes"
            if re.search(r"\bno\b", chunk):
                return "no"

    # Fallback: last non-empty line of the response
    for line in reversed([ln for ln in text.splitlines() if ln.strip()]):
        low = line.lower()
        if re.search(r"\byes\b", low):
            return "yes"
        if re.search(r"\bno\b", low):
            return "no"
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
    gold = str(doc["answer"]).strip()
    gold_low = gold.lower()

    # Yes/no path — small but real subset of FinQA
    if gold_low in {"yes", "no"}:
        pred_yn = extract_yes_no(pred_text)
        ok = 1.0 if pred_yn == gold_low else 0.0
        return {"acc": ok, "acc_loose": ok}

    # Numeric path
    pred_num = extract_final_number(pred_text)
    gold_num = _to_float(gold)
    if gold_num is None or pred_num is None:
        strict = 0.0
        loose = 0.0
    else:
        strict = 1.0 if is_close(gold_num, pred_num, rel_tol=0.01) else 0.0
        loose = 1.0 if is_close(gold_num, pred_num, rel_tol=0.05) else 0.0

    return {"acc": strict, "acc_loose": loose}
