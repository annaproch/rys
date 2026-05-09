"""Minimal OpenAI-compatible server for dnhkng/RYS-Qwen3.5-27B-FP8-M on Apple Silicon.

Sidesteps the `transformers serve` CLI, which (in transformers 5.8.0.dev0) strips
`config.quantization_config` before calling from_pretrained and crashes on FP8
pre-quantized checkpoints. This wrapper reuses the load path that smoke.py
validated, then exposes the endpoints we actually need.

Run:   uv run python serve.py
       caffeinate -is uv run python serve.py     # keep the Mac awake
       PORT=8000 DEVICE=mps uv run python serve.py
"""

from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = os.environ.get("MODEL_ID", "dnhkng/RYS-Qwen3.5-27B-FP8-M")
DEVICE = os.environ.get("DEVICE", "mps")
DTYPE = torch.bfloat16
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    enable_thinking: bool = False
    stream: bool = False
    seed: int | None = None
    stop: list[str] | str | None = None


class CompletionRequest(BaseModel):
    """OpenAI legacy /v1/completions — raw prompt, no chat-template wrapping.

    Used by `lm-evaluation-harness --model local-completions`.
    """

    model: str | None = None
    prompt: str | list[str]
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    stop: list[str] | str | None = None
    echo: bool = False
    n: int = 1


print(f"[serve] loading processor for {MODEL_ID}")
processor = AutoProcessor.from_pretrained(MODEL_ID)
print(f"[serve] loading model on {DEVICE} (FP8 → bf16 dequant on load)")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=DTYPE,
    device_map={"": DEVICE},
    low_cpu_mem_usage=True,
)
model.eval()
print(f"[serve] loaded in {time.time() - t0:.1f}s")

# Tokenizer used by /v1/completions for raw-prompt batching.
_tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
# For autoregressive generation, padding must be on the LEFT so every sequence
# ends at the same position; generation continues uniformly from there.
_tokenizer.padding_side = "left"
if _tokenizer.pad_token is None:
    _tokenizer.pad_token = _tokenizer.eos_token

# Generation is not thread-safe (single GPU/CPU stream + KV cache reuse), so we
# serialize requests. For interactive single-user use this is the right default.
_gen_lock = Lock()


def _build_inputs(messages: list[ChatMessage], enable_thinking: bool):
    return processor.apply_chat_template(
        [m.model_dump() for m in messages],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=enable_thinking,
    ).to(DEVICE)


def _normalize_stop(stop: list[str] | str | None) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _truncate_at_stop(text: str, stops: list[str] | None) -> str:
    if not stops:
        return text
    cuts = [text.find(s) for s in stops]
    cuts = [c for c in cuts if c >= 0]
    return text[: min(cuts)] if cuts else text


def _generate_from_inputs(inputs, max_tokens: int, temperature: float, top_p: float,
                          top_k: int, seed: int | None,
                          repetition_penalty: float | None) -> tuple[str, int, int]:
    if seed is not None:
        torch.manual_seed(seed)
    prompt_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    if repetition_penalty:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    with _gen_lock, torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, prompt_len:]
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return text, prompt_len, int(gen_ids.numel())


def _generate(req: ChatCompletionRequest) -> tuple[str, int, int]:
    inputs = _build_inputs(req.messages, req.enable_thinking)
    rep_pen = (1.0 + req.presence_penalty * 0.1) if req.presence_penalty else None
    text, p, c = _generate_from_inputs(
        inputs, req.max_tokens, req.temperature, req.top_p, req.top_k,
        req.seed, rep_pen,
    )
    return _truncate_at_stop(text, _normalize_stop(req.stop)), p, c


app = FastAPI(title="rys-local")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    if req.stream:
        # Streaming would need a TextIteratorStreamer + SSE; skipped for v1.
        raise HTTPException(status_code=501, detail="streaming not implemented yet")
    t0 = time.time()
    try:
        text, prompt_tokens, completion_tokens = _generate(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    elapsed = time.time() - t0
    print(
        f"[serve] {completion_tokens} tok in {elapsed:.1f}s "
        f"= {completion_tokens / max(elapsed, 1e-6):.2f} tok/s"
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _batched_generate(
    prompts: list[str], max_tokens: int, temperature: float, top_p: float,
    top_k: int, seed: int | None, stops: list[str] | None,
) -> tuple[list[str], list[int], list[int]]:
    """Batched generate via left-padded tokenization + single model.generate call.

    Returns (texts, prompt_token_counts, completion_token_counts) — completion
    counts are pre-truncation (the cost the model actually paid).
    """
    if seed is not None:
        torch.manual_seed(seed)

    encoded = _tokenizer(
        prompts, padding=True, return_tensors="pt", truncation=False,
    ).to(DEVICE)
    _batch, padded_len = encoded["input_ids"].shape
    do_sample = temperature > 0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_tokens, do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)

    with _gen_lock, torch.inference_mode():
        out = model.generate(**encoded, **gen_kwargs)

    # Real (un-padded) prompt token counts per item
    pad_id = _tokenizer.pad_token_id
    prompt_counts = [
        int((encoded["input_ids"][i] != pad_id).sum().item())
        for i in range(len(prompts))
    ]
    # Decode only the newly-generated tokens (everything past the padded prompt)
    texts: list[str] = []
    completion_counts: list[int] = []
    for i in range(len(prompts)):
        gen_ids = out[i, padded_len:]
        text = _tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(_truncate_at_stop(text, stops))
        completion_counts.append(int(gen_ids.numel()))
    return texts, prompt_counts, completion_counts


@app.post("/v1/completions")
def completions(req: CompletionRequest) -> dict:
    """Raw-prompt completion endpoint for lm-evaluation-harness `local-completions`.

    Accepts a single prompt or a batch (list of prompts). Batched prompts are
    tokenised with left-padding and run through a single `model.generate` call
    — on bandwidth-bound autoregressive decode, this amortises the per-token
    weight read across the batch.
    """
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    if req.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 supported")
    stops = _normalize_stop(req.stop)

    t0 = time.time()
    try:
        texts, prompt_counts, completion_counts = _batched_generate(
            prompts, req.max_tokens, req.temperature, req.top_p, req.top_k,
            req.seed, stops,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")
    elapsed = time.time() - t0

    total_completion = sum(completion_counts)
    print(
        f"[serve][/v1/completions] batch={len(prompts)} "
        f"{total_completion} tok in {elapsed:.1f}s "
        f"= {total_completion / max(elapsed, 1e-6):.2f} tok/s aggregate"
    )

    choices = [
        {
            "index": i,
            "text": (prompts[i] + texts[i]) if req.echo else texts[i],
            "finish_reason": "stop",
            "logprobs": None,
        }
        for i in range(len(prompts))
    ]
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model or MODEL_ID,
        "choices": choices,
        "usage": {
            "prompt_tokens": sum(prompt_counts),
            "completion_tokens": total_completion,
            "total_tokens": sum(prompt_counts) + total_completion,
        },
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        timeout_keep_alive=600,  # long generations on slow MPS path
    )
