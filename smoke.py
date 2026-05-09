"""Smoke test for dnhkng/RYS-Qwen3.5-27B-FP8-M on Apple Silicon (M4 Max).

Loads the FP8 checkpoint, which Transformers' FineGrainedFP8 quantizer
auto-dequantizes to bf16 because no FP8-capable GPU is present. Runs one
short text-only generation and reports throughput.

Run:   uv run python smoke.py
       (prefix with `caffeinate -is` to keep the machine awake)
"""

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "dnhkng/RYS-Qwen3.5-27B-FP8-M"
DEVICE = "mps"  # flip to "cpu" if MPS errors or OOMs


def build_inputs(processor, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    # In transformers v5, apply_chat_template forwards extra kwargs straight to
    # the Jinja template. Qwen's template reads `enable_thinking` directly.
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,           # multimodal processors otherwise return a bare Tensor
        enable_thinking=False,
    )


def main() -> None:
    print(f"torch {torch.__version__}  mps={torch.backends.mps.is_available()}  device={DEVICE}")

    print("Loading processor…")
    processor = AutoProcessor.from_pretrained(MODEL)

    print("Loading model (dequantizing ~32 GB FP8 → bf16; first load: 5–15 min)…")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.0f}s")

    inputs = build_inputs(processor, "In one sentence: what is RYS layer duplication?").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[-1]

    print("Generating…")
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.time() - t0

    gen = out[0, prompt_len:]
    text = processor.decode(gen, skip_special_tokens=True)
    n = gen.numel()
    print("\n--- generation ---")
    print(text)
    print("------------------")
    print(f"{n} tokens in {elapsed:.1f}s = {n / elapsed:.2f} tok/s")


if __name__ == "__main__":
    main()
