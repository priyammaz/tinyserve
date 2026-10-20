# tinyserve
<img src="banner.png" alt="img" width="700"/>

LLM inference is all the rage these days! Whether is user side inference or for RL rollout generation during post-training, being able to serve LLMs is extremely important. We create here a single file `serve.py` that implements all the core parts of popular serving platforms like vLLM!

It implements, from scratch, the core pieces that make modern LLM serving
fast:

- **Paged KV cache** with **prefix caching** (shared prompt prefixes reuse
  cached blocks instead of recomputing them)
- **Continuous batching** with **chunked prefill**
- **FlashAttention**, with a pure-PyTorch reference backend as a fallback
- **CUDA graphs** for the decode step
- An **OpenAI-compatible HTTP server** (`/v1/completions`, `/v1/chat/completions`)

Only the Qwen3 model has been tested so far, I'm sure others will work but may require some small edits mainly in the model definition and forward pass. 

## Install

```bash
pip install torch transformers datasets huggingface_hub safetensors \
            fastapi uvicorn pandas numpy openai
pip install flash-attn --no-build-isolation   # optional, for the flash_attn backend
```

## Usage

Every command is `python serve.py <command> [flags]`:

```
--model Qwen/Qwen3-8B             any HF model of a supported architecture (not all will work though...)
--attention-backend flash_attn    or "torch" (reference path, no CUDA graphs)
--max-model-len 8192              max prompt + output tokens per request
--gpu-memory-utilization 0.85     fraction of the card the engine may use
--no-cuda-graphs / --no-prefix-caching
```

### One-off generation

```bash
python serve.py generate --prompt "The meaning of life is" --max-tokens 64
```

### Benchmark: tinyserve vs HF generate with static batches

Samples instructions from Alpaca and Dolly equally across prompt lengths, and
puts the same ~256 token system prompt in front of each of them to test
prefix caching.

```bash
python serve.py bench
```

Or you can try different options like:
```bash
python serve.py bench --bench-num-prompts 30 --bench-max-tokens 32   # quick check
python serve.py bench --bench-bins 1,64,512,2048    # bin edges in tokens
python serve.py bench --bench-hf-batch-size 64      # HF static batch size
python serve.py bench --ignore-eos                  # fixed output length
python serve.py bench --bench-raw                   # no chat template
python serve.py bench --bench-prefix-tokens 4096    # longer shared system prompt
python serve.py bench --bench-prefix-tokens 0       # no shared prefix at all
CUDA_VISIBLE_DEVICES=1 python serve.py bench        # pick a GPU
```

Example result (`Qwen/Qwen3-8B`, default settings):

| Setup                       | Wall time (s) | Requests/s |
| --------------------------- | ------------- | ---------- |
| HF static                   | 191.72        | 0.52       |
| tinyserve (no prefix cache) | 29.21         | 3.42       |
| tinyserve (prefix cache)    | 23.93         | 4.18       |

### Inference server

```bash
python serve.py serve --port 8000
```

Endpoints:

| Method | Path                   | Description                             |
|--------|------------------------|------------------------------------------|
| GET    | `/health`              | liveness check                          |
| GET    | `/v1/models`           | the loaded model                        |
| GET    | `/metrics`             | running / waiting requests, free KV blocks |
| POST   | `/v1/completions`      | raw prompt in, text out                 |
| POST   | `/v1/chat/completions` | messages in, chat template applied      |

Request fields: `temperature`, `top_p`, `top_k` (default: the model's own
`generation_config.json`, e.g. 0.6 / 0.95 / 20 for Qwen3), `max_tokens` or
`max_completion_tokens` (default: the rest of the context window),
`ignore_eos`, `stream`. Chat requests also accept `chat_template_kwargs`,
which is passed to the tokenizer's chat template.

```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="unused",
)
stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Describe the transformer model"}],
    max_tokens=1024,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content is not None:
        print(delta.content, end="", flush=True)
```

### Demos (no model weights needed)

```bash
python serve.py kv-demo      # KV cache bookkeeping step by step, CPU only
python serve.py attn-demo    # paged attention metadata, torch vs flash_attn (GPU)
```
