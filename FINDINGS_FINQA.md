# FinQA findings — RYS-Qwen3.5-27B-FP8-M vs Qwen3.5-27B-FP8

Paired evaluation, n=100 (deterministic first-100 of FinQA test split via
`ChanceFocus/flare-finqa`). Greedy decoding (`temperature=0`,
`do_sample=False`), `max_gen_toks=768`, raw-prompt path through
`/v1/completions`. Both models served from the same `serve.py` on M4 Max MPS,
swapped via `MODEL_ID`.

## Headline

| Scoring method | RYS | Base | Δ | McNemar p |
|---|---|---|---|---|
| Naive extractor (last-number-on-line) | 50% | 43% | +7 | — |
| Patched extractor (first-%-on-Answer-line) | 63% | 65% | −2 | 0.79 |
| Rule-based "manual" grader | 64% | 58% | +6 | 0.24 |
| **LLM-as-judge (Claude subagents, semantic)** | **80%** | **68%** | **+12** | **0.0042** |

The LLM-as-judge result is the only one to cross statistical significance.
Earlier extractor-based results swung over 14 points (50→63→64→80 for RYS;
43→65→58→68 for base) depending on heuristic choices, which made the
extractor itself the dominant variable. The semantic judge removes that
confound.

## Where the lift sits

Bucket breakdown under the LLM-as-judge:

| Gold-answer bucket | n | RYS | Base | Δ |
|---|---|---|---|---|
| **small (<1, mostly percentages)** | 52 | **47** | **36** | **+11** |
| mid (1..100) | 22 | 16 | 14 | +2 |
| large (≥100) | 23 | 14 | 15 | −1 |
| yes/no | 3 | 3 | 3 | 0 |

**Almost the entire +12 lead is on percentage questions** — the
multi-step-arithmetic class (extract two values, subtract, divide, ×100)
where the original RYS-72B paper's largest gains were also concentrated
(MATH Lvl 5 +8.16%, MuSR +17.72%). Mid-range and large-number questions
are tied; yes/no is tied.

## What's actually happening — behavioural analysis

Both models receive *identical* raw prompts. Neither is in an explicit
"thinking mode" via the chat template — there's no `enable_thinking` flag
because we're using `/v1/completions`, not `/v1/chat/completions`. Yet both
models emit `<think>` tokens on their own. The split:

| | RYS | Base |
|---|---|---|
| Starts with `<think>` | 100/100 | 100/100 |
| Closes `</think>` cleanly | 95/100 | **82/100** |
| Truncated mid-thought (unclosed `<think>`) | 5/100 | **18/100** |
| Empty `<think></think>` block (no silent CoT) | 94/100 | 80/100 |

**RYS opens `<think>` and immediately closes it 94% of the time** — i.e. it
effectively skips silent thinking and reasons in the visible output. **Base
skips silent thinking 80% of the time, and 18% of the time engages real
silent CoT and gets cut off inside the unclosed block** before producing a
visible answer.

This explains the +12 lift almost entirely.

## Discordant cases — full categorisation

| Failure mode | Cases base lost (RYS-only correct) | Cases RYS lost (base-only correct) |
|---|---|---|
| Stuck inside `<think>`, never emerges to answer | **13 / 14** | **1 / 2** |
| Real reasoning miss (skipped a required step) | 0 | 1 (doc 78) |
| Hedged / refused without committing | 1 (doc 37) | 0 |

The single "real reasoning miss" by RYS is doc_id 78 (gold 160,035, the
*total* of two values). RYS listed `$78,227 + $81,808` and stopped without
summing. Base summed.

The 13 base losses follow one pattern: base computes the right answer
deep inside its `<think>` block, then runs out of tokens before closing
the block and writing the answer. Examples:
- doc 50: base reaches `−2.918%` mid-thought, truncated.
- doc 60: base reaches `27.27`, truncated before answer block.
- doc 73: base reaches `60.8%`, then spends remaining budget self-checking,
  truncated.
- doc 98: base reaches `62.55%`, truncated mid double-check.

In *every* one of these the model arithmetic was correct. The model just
never emerged.

## What the +12 means and doesn't mean

It **does** mean:
- At our token budget (`max_gen_toks=768`), RYS produces correct, fully-formed
  answers ~12 percentage points more often than base.
- Of the 16 paired disagreements, 14 favour RYS — far enough from 50/50 to
  reject the "equally accurate" null hypothesis at p<0.01.

It **probably doesn't** mean:
- That RYS is doing better arithmetic than base. Both compute the right
  numbers when they reach them. The gap is entirely about **whether the
  model emerges from `<think>` mode in time to write the answer.**
- That the lift is robust to a larger token budget. If `max_gen_toks` were
  bumped to 1500, most of base's silent-CoT runs would likely finish, and
  the gap should shrink — possibly to zero. Untested as of this writing.

The honest interpretation aligns with the original RYS-I paper's IFEval
regression: the layer-duplication trick changes the model's **output
discipline**. On numerical-reasoning tasks where compactness wins, this
helps. On instruction-following tasks where the model needs to comply
strictly with prompt-specified formats, it hurts.

## Methodological caveats

1. **Judge wasn't blinded.** The input JSON labels each response as
   `rys_response` and `base_response`. The four parallel judge agents could
   in principle have been biased by the labels. The rubric is concrete
   enough that this is unlikely, but a clean replication should randomise
   labels per case and reveal them only at aggregation.
2. **n=100 with `--limit`.** Marginal stderr is ±5pp; we got significance
   from the lopsided discordance (14 vs 2), not from a tight marginal CI.
   A 200-q run would tighten things; a 1,147-q full run would settle it.
3. **Token budget interaction.** As noted above, the +12 lift is plausibly
   token-budget-bounded. The headline number applies *at the max_gen_toks
   we used*; it's not necessarily a property of the architectures alone.
4. **Four parallel judges may have stringency variance.** Per-batch correct
   counts: RYS 19/20/22/19, Base 15/18/16/19. Variance is small and
   symmetric; no obvious per-batch bias against either model.
5. **Single benchmark.** FinQA is one task class. Generalisation requires
   running benchmarks where RYS-I showed bigger gains (MuSR +17.7%) or
   regressions (IFEval −2.05%) on the 27B variant.

## Files in this experiment

- `tasks/finqa/finqa.yaml` — lm-eval-harness task config
- `tasks/finqa/utils.py` — extractor + scorer (patched version)
- `EVAL_FINQA.md` — running plan and status log
- `manual_grade.py` — rule-based "manual" grader (used for the +6 / p=0.24
  result above)
- `rescore.py` — re-scores saved JSONL with the patched extractor
- `judge_prep.py` — splits 100 cases into 4 batches for parallel judging
- `judge_aggregate.py` — aggregates judge outputs and computes paired
  metrics
- `judge_inputs/batch_{0..3}.jsonl` — per-batch input cases
- `judge_outputs/batch_{0..3}.jsonl` — per-batch judge verdicts
- `judge_results.jsonl` — consolidated 100-row audit file: question, gold,
  full responses from both models, judge reasoning, outcome
- `manual_analysis.jsonl` — same shape as judge_results but with the
  rule-based grader's output, kept for cross-reference

## Recommended follow-ups (in priority order)

1. **`max_gen_toks=1500` re-run.** Cheap (~2h per model), settles whether
   the +12 is robust or a token-budget artifact.
2. **Blinded judge pass.** Randomise `rys/base` labels per case before
   sending to subagents; reveal at aggregation. Removes the unblinded-judge
   caveat. ~5 min.
3. **Different benchmark.** MuSR is the highest-signal test from RYS-I:
   `--tasks musr` is built into lm-eval-harness, no custom task work
   needed. ~2h per model.
4. **IFEval.** RYS-I's only regression in the paper. Tests the
   "output-discipline" interpretation: if RYS regresses on instruction
   following too, that's strong evidence for the discipline-not-reasoning
   reading.
