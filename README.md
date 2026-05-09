# rys — local serving of RYS-Qwen3.5-27B on Apple Silicon

Minimal setup for running [`dnhkng/RYS-Qwen3.5-27B-FP8-M`](https://huggingface.co/dnhkng/RYS-Qwen3.5-27B-FP8-M)
locally on a Mac with an OpenAI-compatible HTTP endpoint.

The model uses David Ng's [RYS layer-duplication](https://dnhkng.github.io/posts/rys-ii/)
technique on top of `Qwen/Qwen3.5-27B-FP8`: layers 31, 32, 33 of the base 64-layer
checkpoint are physically duplicated, producing a 67-layer model. Those extra layers
are baked into the safetensors, so stock Hugging Face Transformers loads it with no
custom modeling code.

## Why this works on a Mac (and why vLLM doesn't)

- **No FP8 hardware on Apple Silicon.** When Transformers' `FineGrainedFP8` quantizer
  detects no CUDA/XPU device with FP8 support, it auto-dequantizes the weights to
  bf16 during load. We then run inference in bf16 on Metal/MPS.
- **`vllm` is not viable here.** vLLM has no Metal backend (the `vllm-metal` plugin
  is text-only and doesn't cover the `qwen3_5` hybrid + FP8 + RYS stack), and the
  CUDA path needs Hopper-class FP8 tensor cores plus 8× tensor parallelism.
- **`transformers serve` is broken** for this checkpoint in transformers 5.8.0.dev0:
  its `ModelManager` strips `config.quantization_config` before `from_pretrained`,
  so the FP8 quantizer dispatch crashes on `None.get(...)`. `serve.py` here
  sidesteps it with a small FastAPI wrapper around the validated load path.

## Requirements

- **Apple Silicon Mac** (tested on M4 Max, 40-core GPU, 128 GB unified memory).
  Should also work on M1/M2/M3 Max/Ultra with **≥ 96 GB unified memory**.
- **macOS ≥ 14** (Sonoma) for current PyTorch MPS.
- **~32 GB free disk** for the model download, **~65–75 GB free RAM** for the
  bf16-dequantized working set (weights ~55–60 GB + KV cache + activations).
- **Python 3.13** (managed by `uv`) and **`uv`** itself (`brew install uv`).
- **No CUDA libraries.** Do **not** install `mamba-ssm`, `causal-conv1d`,
  `flash-attn`, `flash-linear-attention`, `triton`, `bitsandbytes`, or `vllm` —
  they are CUDA-only and will either fail to build on arm64 or break things
  silently. The pure-PyTorch fallback path is what runs on Mac.

Working set fits comfortably on 128 GB; on 96 GB you should close other apps and
keep `max_position_embeddings` modest.

## Install

```bash
git clone <this repo> rys && cd rys
uv sync                       # creates .venv and installs from uv.lock
```

Verify the install:

```bash
uv run python -c "import torch, transformers as t; \
  print(torch.__version__, torch.backends.mps.is_available(), t.__version__)"
# expected: 2.11.0 True 5.8.0.dev0   (or newer)
```

## Download the model (~32 GB, one-time)

Add `~/.cache/huggingface` to **System Settings → Spotlight → Privacy** *before*
downloading, so Spotlight doesn't reindex 32 GB of safetensors.

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
caffeinate -is uv run hf download dnhkng/RYS-Qwen3.5-27B-FP8-M
```

## Smoke test

```bash
caffeinate -is uv run python smoke.py 2>&1 | tee smoke.log
```

What to expect:

- Two warnings — both are correct and harmless:
  - `Using FP8 quantized models requires a GPU or XPU, we will default to dequantizing the model to bf16…`
  - `The fast path is not available because one of the required library is not installed.` (Gated DeltaNet falls back to a pure-PyTorch implementation; do not try to install the suggested CUDA libraries.)
- Load time: ~7 s on a warm filesystem cache, up to ~15 min cold.
- Throughput: **~5 tok/s on MPS** for short single-stream generations on M4 Max.

If MPS errors out under load, set `DEVICE = "cpu"` at the top of `smoke.py` and
retry — CPU is often only ~1.5–2× slower than MPS-with-fallback for this hybrid
architecture and avoids GPU↔CPU round-trips for the unsupported ops.

## Run the server

```bash
caffeinate -is uv run python serve.py
```

Environment overrides (all optional):

```bash
DEVICE=cpu PORT=9000 HOST=0.0.0.0 uv run python serve.py
```

Endpoints:

| Method | Path                      | Notes                                  |
|-------:|---------------------------|----------------------------------------|
| GET    | `/health`                 | `{ "status": "ok", ... }`              |
| GET    | `/v1/models`              | OpenAI-shaped models list              |
| POST   | `/v1/chat/completions`    | Non-streaming. Streaming returns 501.  |

Once it logs `Application startup complete`:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

### Querying examples

**curl, minimal:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hi in three words."}],
    "max_tokens": 32
  }' | python -m json.tool
```

**curl, with the model card's recommended sampling:**

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are concise."},
      {"role": "user",   "content": "Explain RYS layer duplication in one sentence."}
    ],
    "max_tokens": 128,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "enable_thinking": false
  }' | python -m json.tool
```

**Python with the OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="dnhkng/RYS-Qwen3.5-27B-FP8-M",
    messages=[{"role": "user", "content": "Two-sentence summary of bf16 vs fp8."}],
    max_tokens=160,
    temperature=1.0,
    top_p=0.95,
)
print(resp.choices[0].message.content)
print(resp.usage)
```

**Pointing third-party clients:**

- Open WebUI, Cursor, Aider, etc.: set base URL to `http://127.0.0.1:8000/v1`,
  any non-empty API key, model id `dnhkng/RYS-Qwen3.5-27B-FP8-M`.

### Request fields

`serve.py` accepts the OpenAI subset plus a few extras:

| Field              | Default | Notes                                                            |
|--------------------|---------|------------------------------------------------------------------|
| `messages`         | —       | OpenAI shape (`role`, `content`).                                |
| `max_tokens`       | 256     | Capped at 8192 in the wrapper. At ~5 tok/s, keep this small.     |
| `temperature`      | 1.0     | `0` switches to greedy decoding.                                 |
| `top_p`            | 0.95    |                                                                  |
| `top_k`            | 20      |                                                                  |
| `presence_penalty` | 1.5     | Mapped to HF `repetition_penalty = 1 + presence_penalty * 0.1`.  |
| `enable_thinking`  | `false` | When `true`, model emits `<think>…</think>` before the answer.   |
| `stream`           | `false` | Streaming not implemented yet — returns HTTP 501.                |

## Performance expectations

- **MPS, M4 Max, single user, short prompts:** ~5 tok/s decode.
- **Time-to-first-token:** dominated by Gated DeltaNet's pure-PyTorch path
  (~1–3 s for short prompts).
- **Long contexts** (32k+) are technically possible (262k native, 1M with YaRN)
  but practically capped by patience — at 5 tok/s, 32k output is roughly
  1.7 hours.
- Concurrent requests are serialized inside `serve.py` (single device stream,
  shared KV cache); this is intentional.

## File map

```
.
├── pyproject.toml      # uv-managed deps (incl. transformers from git)
├── uv.lock             # pinned versions
├── smoke.py            # one-shot generation, used to validate the load path
├── serve.py            # FastAPI OpenAI-compatible server
└── README.md           # this file
```

## License

Code in this repo: choose your own (none specified by default).
The model itself is Apache 2.0; see the [HF model card](https://huggingface.co/dnhkng/RYS-Qwen3.5-27B-FP8-M).
