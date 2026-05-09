"""Split the 100 saved FinQA cases into 4 batches of 25 for parallel judging.

Each batch file is JSONL with one row per question:
  doc_id, question, gold, rys_response, base_response.

The subagents will read these and produce a corresponding judgments file.
"""

import glob
import json
import os

OUT_DIR = "judge_inputs"
os.makedirs(OUT_DIR, exist_ok=True)

rys = {s["doc_id"]: s for s in (json.loads(l) for l in open(sorted(glob.glob(
    "runs/rys-finqa-sub100/dnhkng__RYS-Qwen3.5-27B-FP8-M/samples_finqa_*.jsonl"
))[-1]))}
base = {s["doc_id"]: s for s in (json.loads(l) for l in open(sorted(glob.glob(
    "runs/base-finqa-sub100/Qwen__Qwen3.5-27B-FP8/samples_finqa_*.jsonl"
))[-1]))}


def get_question(prompt: str) -> str:
    qi = prompt.rfind("Question:")
    return prompt[qi:].split("\nAnswer:")[0].replace("Question:", "").strip() if qi >= 0 else ""


cases = []
for did in range(100):
    r = rys[did]
    b = base[did]
    cases.append({
        "doc_id": did,
        "gold": str(r["target"]).strip(),
        "question": get_question(r["arguments"]["gen_args_0"]["arg_0"]),
        "rys_response": r["resps"][0][0],
        "base_response": b["resps"][0][0],
    })

# Split into 4 batches of 25 (slightly uneven if 100 doesn't divide cleanly — here it does)
N_BATCHES = 4
batch_size = (len(cases) + N_BATCHES - 1) // N_BATCHES
for i in range(N_BATCHES):
    chunk = cases[i * batch_size : (i + 1) * batch_size]
    path = f"{OUT_DIR}/batch_{i}.jsonl"
    with open(path, "w") as f:
        for c in chunk:
            f.write(json.dumps(c) + "\n")
    print(f"wrote {path}: {len(chunk)} cases  doc_ids {chunk[0]['doc_id']}..{chunk[-1]['doc_id']}")
