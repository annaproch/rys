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


def _generate(req: ChatCompletionRequest) -> tuple[str, int, int]:
    inputs = _build_inputs(req.messages, req.enable_thinking)
    prompt_len = inputs["input_ids"].shape[-1]
    do_sample = req.temperature > 0
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=req.max_tokens,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs.update(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
        )
    if req.presence_penalty:
        gen_kwargs["repetition_penalty"] = 1.0 + req.presence_penalty * 0.1

    with _gen_lock, torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, prompt_len:]
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return text, prompt_len, int(gen_ids.numel())


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


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
