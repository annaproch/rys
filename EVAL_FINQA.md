# FinQA evaluation: RYS-Qwen3.5-27B-FP8-M vs Qwen3.5-27B-FP8

Goal: measure whether the RYS layer-duplication trick (67 vs 64 layers) lifts
numerical-reasoning accuracy over financial-report excerpts on the canonical
FinQA test split.

## Why FinQA (vs alternatives)

- Closed, manageable context (pre_text + table + post_text, ~2–4k tokens).
  Tests reasoning, not retrieval — the right hypothesis test for RYS.
- Numerically scorable with normalization, no LLM-judge dependency.
- 1,147 test rows — paired RYS-vs-base run is ~30–42 hours wall on M4 Max MPS
  (revised after dry-run: ~52s/question observed, longer if the model uses more
  of the higher `max_gen_toks` budget).
- Standard benchmark (Chen et al., EMNLP 2021), comparable to literature.

FinanceQA (arxiv 2501.18062) is more discriminating but free-form scored.
Deferred to a follow-up.

## Setup decisions (locked)

- Runner: `lm-evaluation-harness` with `local-completions` model pointed at our `serve.py`.
- Endpoint: `/v1/completions` (raw prompts, no chat-template wrapping). Will add to `serve.py`.
- Decoding: `temperature=0`, `do_sample=False`, fixed seed.
- Metric: numerical exact-match with 1% relative tolerance + unit/percentage/currency normalization.
- Prompt: pre_text → markdown table → post_text → question, with explicit
  "Final answer:" hint and unit-normalization instructions.

## Risks / known issues

- HF dataset (`dreamerdeo/finqa`) drops the `program` column → can only do
  numeric-match scoring, not the paper's "execution accuracy". Acceptable for
  paired comparison.
- Answer formats are messy ($, %, "million", etc.). Normalizer must be tested
  against the dataset before trusting numbers.
- Mac MPS: dry-run measured **~52s/question** (prefill on long FinQA prompts is
  the dominant cost). Sub-sampling to 200–300 questions for the paired
  comparison is the pragmatic move; full 1,147 only if the sub-sample shows a
  real delta.
- `max_gen_toks=256` truncated the model mid-CoT in 2/5 dry-run questions
  (model emits empty `<think></think>` then ~200 tokens of natural-language
  reasoning before the final number). Bumped to **768** in `finqa.yaml`.
- lm-eval-harness's `local-completions` defaults to `max_length=2048`. None of
  the 5 dry-run prompts hit that, but to be safe across the full split we pass
  `max_length=8192` in `--model_args`.
- Models can't co-reside in 128 GB (each ~55 GB resident in bf16). Run
  serially, swap `MODEL_ID`, restart server.

## Plan / checklist

### 0. Foundations
- [x] Plan file in repo
- [x] Add `lm-eval` to project deps

### 1. Server changes (`serve.py`)
- [x] Add `/v1/completions` endpoint (raw prompt, no chat template)
- [x] Accept and apply `seed` field (forwarded to `torch.manual_seed`)
- [x] Configured `timeout_keep_alive=600` on uvicorn

### 2. FinQA task files (`tasks/finqa/`)
- [x] `finqa.yaml` — lm-eval-harness task config (uses `ChanceFocus/flare-finqa`)
- [x] `utils.py` — doc_to_text, doc_to_target, answer normalizer, process_results
- [x] Switched dataset source: `dreamerdeo/finqa` is a script-only repo (datasets v3+ rejects it). FLARE mirror has parquet + preformatted prompts + canonicalised gold answers.
- [x] Unit-tested normalizer + extractor on currency/%/parens/units cases
- [x] Lm-eval-harness can discover the task via `--include_path tasks`
      (`lm-eval validate --tasks finqa` → "All tasks found and valid")
- [x] Bumped `max_gen_toks` 256 → 768 after dry-run showed mid-CoT truncation

### 3. Validation
- [x] Dataset loads (1,147 test rows; gold answers are clean decimals)
- [x] Dry-run #1 (5 q, max_gen_toks=256): acc 0.6, wall ~52s/q. Two failures
      caused by mid-CoT truncation, not bad reasoning.
- [x] Dry-run #2 (5 q, max_gen_toks=768, max_length=8192): acc 0.8, wall ~6 min.
      Truncation fix worked.
- [x] Sub-sample 100 RYS run: ~95 min wall, ~48s/q steady-state.
- [x] Sub-sample 100 base run: ~120 min wall, ~60s/q steady-state.
- [x] Comparison done across 5 scoring methods. **See `FINDINGS_FINQA.md`** for
      the full write-up.

### 4. Full run — DEFERRED
- [ ] Skipped pending follow-up experiments (max_gen_toks=1500 re-run,
      blinded judge pass, MuSR/IFEval cross-checks). The 100-q paired result
      with LLM-as-judge already shows RYS +12 at p=0.0042; full 1,147-q run
      would tighten the CI but is unlikely to change the qualitative finding.

## Status log

(append-only; newest at bottom)

- 2026-05-09 12:50 — `lm-eval` added; `serve.py` extended with `/v1/completions`,
  `seed`, and 600s keep-alive. Dataset path switched to `ChanceFocus/flare-finqa`
  (since `dreamerdeo/finqa` is script-only and rejected by datasets v3+).
- 2026-05-09 12:55 — `tasks/finqa/{finqa.yaml, utils.py}` written.
  `lm-eval validate --tasks finqa --include_path tasks` → "All tasks found and
  valid". Normalizer + extractor unit tests pass on currency, %, parens,
  unit-suffix, and CoT-wrapped responses.
- 2026-05-09 12:57 — Currently-running server is the pre-patch build; needs
  restart before `--model local-completions` will work. `/v1/chat/completions`
  is still up so we could fall back to `--model local-chat-completions` for the
  dry-run if you don't want to restart yet.
- 2026-05-09 14:46 — Dry-run #1 complete (5 q, RYS, max_gen_toks=256, no
  max_length override). 5/5 requests succeeded; mean **51.8s/q**, range 26–91s.
  Score: **acc 0.6, acc_loose 0.6**. Inspecting samples: prompts 415–1294
  tokens, generations always near or at the 256-token cap. **Two of three
  failures were caused by truncation mid-CoT**, not by wrong reasoning
  (sample 2 cut off at `Difference: 153.7 - 139.9 =`; sample 4 at
  `Return = $\\`). Acc score under-counts true reasoning ability.
- 2026-05-09 14:55 — Mitigations applied: `finqa.yaml` `max_gen_toks` bumped
  to 768; run snippets updated to pass `max_length=8192` in `--model_args`
  (lm-eval-harness's client-side default of 2048 is too low for FinQA
  prompts + 768 generation budget). `max_length` is harness-side only; not a
  YAML field.
- 2026-05-09 15:05 — `/v1/completions` rewritten to do real batched
  generation: left-padded tokenization + single `model.generate(...)` call,
  amortising the per-token weight read across the batch. Tokenizer set to
  `padding_side="left"` once at load. Added `bench_batch.py` to measure actual
  MPS scaling on 8 FinQA prompts at batch sizes 1/2/4/8. Must restart server
  to pick up the change.
- 2026-05-09 15:30 — Batching benchmark result: **batching does not help on
  MPS for this hybrid model.**
  ```
   batch    wall(s)    gen tok      tok/s    speedup
       1      315.1       1539       4.88        —
       2      348.8       1600       4.59     0.90x
  ```
  B=2 is 10% **slower** than B=1. Apple Silicon serialises the batched
  attention path internally for the GDN+full-attention hybrid; we pay padding
  + housekeeping overhead with zero decode amortisation. Aborted the bench
  before B=4/8 (would only get worse). Keeping the batched code in `serve.py`
  (no-op at batch=1, harmless) but **all real eval runs will use batch=1**.
  Decision: skip the speculative-decoding experiment for now; run 100-q
  subsample at batch=1 to get a directional signal. ~2h per model.
- 2026-05-09 17:09 — RYS 100-q sub-sample complete. Naive scoring: 50%.
  Patched scoring (LaTeX %, yes/no, prefer-first-pct-on-Answer-line): 63%.
- 2026-05-09 17:18 — Hit `NameError: c is not defined` on first base-run
  attempt — typo in `_batched_generate` call after a linter edit. Fixed
  (`c(...)` → `_batched_generate(...)`); restarted server.
- 2026-05-09 19:18 — Base 100-q sub-sample complete. Patched scoring: 65%.
  Initial reading: RYS slightly behind base (Δ=−2, p=0.79). But 5 cases of
  RYS-only-correct and 7 cases of base-only-correct on closer inspection
  showed the patched extractor was systematically penalising both models'
  verbose answer styles asymmetrically.
- 2026-05-09 ~19:40 — Built `manual_grade.py` (rule-based grader handling
  multi-line answer blocks, prose-sign detection, broader marker set).
  Result: RYS 64, Base 58, Δ=+6, p=0.24. Closer inspection showed the
  manual grader had its own anti-base bias (3 unit-double-counting bugs,
  2 CoT-intermediate captures, 1 yes/no out-of-block). The headline number
  was swinging by 14 points (50→63→64) depending on extractor heuristics —
  extractor design had become the dominant variable.
- 2026-05-09 ~20:30 — Pivoted to **LLM-as-judge** (4 parallel `general-purpose`
  subagents, 25 questions each, identical strict rubric). Result:
  **RYS 80, Base 68, Δ=+12, McNemar exact p=0.0042**. First scoring method
  to cross statistical significance. Lift concentrated entirely in
  percentage questions (small <1 gold: RYS 47/52 vs Base 36/52, Δ=+11).
- 2026-05-09 ~21:00 — Behavioural analysis: **the +12 lift is largely an
  output-discipline effect, not raw reasoning**. RYS opens-and-immediately-
  closes its `<think>` block 94/100 times (skipping silent CoT); base does
  the same 80/100 times and **18/100 times engages real silent thinking
  and gets cut off inside an unclosed `<think>` block** before producing a
  visible answer. Of 14 RYS-wins, 13 follow the same pattern (base gets
  stuck in silent CoT, never emerges); 1 base-win is the symmetric
  case (RYS got stuck the same way); 1 base-win is a real RYS reasoning
  miss (doc 78, didn't sum two values). Full write-up in
  `FINDINGS_FINQA.md`.

## Run commands

All commands assume `serve.py` is running with the matching `MODEL_ID` (RYS
is the default). `max_length=8192` caps prompt+gen tokens that the harness
will send (default 2048 is too tight given `max_gen_toks=768`).

### Dry-run #2 (5 questions, RYS, fixed truncation)
After restarting the server (`caffeinate -is uv run python serve.py`):

```bash
mkdir -p runs
caffeinate -is uv run lm-eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=dnhkng/RYS-Qwen3.5-27B-FP8-M,num_concurrent=1,max_retries=2,tokenized_requests=False,max_length=8192 \
  --tasks finqa \
  --include_path tasks \
  --limit 5 \
  --output_path runs/rys-finqa-dryrun2 \
  --log_samples
```

### Sub-sample (100 questions, paired) — ~2h per model
```bash
# RYS (server already running with MODEL_ID=dnhkng/RYS-Qwen3.5-27B-FP8-M)
caffeinate -is uv run lm-eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=dnhkng/RYS-Qwen3.5-27B-FP8-M,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks finqa --include_path tasks --limit 100 \
  --output_path runs/rys-finqa-sub100 --log_samples

# Base — stop the RYS server first, then:
MODEL_ID=Qwen/Qwen3.5-27B-FP8 caffeinate -is uv run python serve.py
# (separate shell)
caffeinate -is uv run lm-eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=Qwen/Qwen3.5-27B-FP8,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks finqa --include_path tasks --limit 100 \
  --output_path runs/base-finqa-sub100 --log_samples
```

### Full RYS run (1,147 questions, ~16 h)
```bash
caffeinate -is uv run lm-eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=dnhkng/RYS-Qwen3.5-27B-FP8-M,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks finqa --include_path tasks \
  --output_path runs/rys-finqa --log_samples
```

### Full base run (~16 h, after server swap)
```bash
caffeinate -is uv run lm-eval \
  --model local-completions \
  --model_args base_url=http://127.0.0.1:8000/v1/completions,model=Qwen/Qwen3.5-27B-FP8,num_concurrent=1,max_retries=3,tokenized_requests=False,max_length=8192 \
  --tasks finqa --include_path tasks \
  --output_path runs/base-finqa --log_samples
```
