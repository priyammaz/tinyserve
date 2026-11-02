# tinyserve
<img src="imgs/banner.png" alt="img" width="700"/>

LLM inference is all the rage these days! Whether it is user-side inference or rollout generation for RL post-training, being able to serve LLMs fast matters. tinyserve is a small, readable, vLLM-style inference engine: the whole scheduler, KV cache, model, tensor parallelism and HTTP server live in one file, `serve.py`, and the GPU kernels live in a second one, `tinykernels.py`.

It is meant to be read. `serve.py` is written as a plain PyTorch reference with the Triton Kernels being the default fast path. We implement:

- **Paged KV Cache** with **Prefix Caching**: Shared prompt prefixes reuse cache blocks instead of recomputing them. 
- **Continuous Batching** with **chunked prefill**, request **preemption** and **async scheduling** (the cpu schedules step n+1 while the GPU runs step n)
- **Triton Kernels Everywhere** (`tinykernels.py`): Paged Attention w/ Split-KV Decoding, fused RMSNorm, q-k-norm + RoPE, SwiGLU, Fused Mixture-of-Experts, and a Fused Sampler. 
- **Full CUDA-Graphs** for decode steps and mixed-prefill
- **Tensor Parallelism** (`-tp`) and **expert parallelism** (`--expert-parallel`) across GPUs
- **Custom All-Reduce Sum** to minimize decode time latency (requries `torch >= 2.9` )
- An **OpenAI-Compatible HTTP Server** (`/v1/completions`, `/v1/chat/completions`) with streaming

Models: Llama-style architectures (`llama`, `qwen2`, `qwen3`) and `qwen3_moe`. Tested on Qwen3-0.6B, 8B, 32B and Qwen3-30B-A3B; other Llama-class models should work but have not been tried yet.

### Let's Build It Together

| Project | Summary | Source Code | Video |
|---|---|---|---|
| `The Annotated LLM Server` | LLM serving fundamentals, KV caching, batching, scheduling | — | [Video](https://youtu.be/SzArjwEkvKI) |
| `tinyserve 1.0` | Paged KV Cache, Prefix Caching, Continuous Batching, Chunked Prefill, CUDA Graphs | [Code](https://github.com/priyammaz/tinyserve/tree/tinyserve-1.0) | [Video](https://youtu.be/qIVzHM2FfFY) |
| `tinyserve 2.0` | Triton Kernels, Split-KV, Tensor/Expert Parallelism, MoE, Async Scheduling, CUDA Graphs | [Code](https://github.com/priyammaz/tinyserve/tree/tinyserve-2.0) | [Video](https://youtu.be/1demEePpFXU) |
| `tinyserve 2.1` | Faster All-Reduce, MoE Optimization, Fused Comms | [Code](https://github.com/priyammaz/tinyserve/tree/tinyserve-2.1) | — |

## Install

```bash
pip install torch transformers datasets huggingface_hub safetensors \
            fastapi uvicorn pandas numpy openai httpx
pip install flash-attn --no-build-isolation   # optional
```
`flash_attn` install is needed if you want to test that backend, otherwise we default to the built Triton kernels


## Usage

Every command is `python serve.py <serve|generate> [flags]`:

```
--model Qwen/Qwen3-8B             any HF model of a supported architecture
--attention-backend triton        triton | flash_attn | torch
--no-triton-kernels               run nonfused ops
--no-cuda-graphs                  run every step eagerly
--no-prefix-caching               don't build prefix hash table
--tensor-parallel-size 2          split the model across this many GPUs
--no-p2p-all-reduce               Use NCCL for communications
--expert-parallel                 MoE with -tp > 1 
--max-model-len 8192              max prompt + output tokens per request
--max-num-seqs 256                max requests in one batch
--max-num-batched-tokens 2048     token budget per step
--gpu-memory-utilization 0.85     fraction of the card the engine may use
--log-stats-interval 10           verbosity frequency
```

## Serve a Model

```bash
python serve.py serve --model Qwen/Qwen3-8B --port 8000

# two GPUs
python serve.py serve --model Qwen/Qwen3-32B -tp 2 --max-model-len 32768

# a mixture-of-experts model, each GPU holding whole experts
python serve.py serve --model Qwen/Qwen3-30B-A3B -tp 2 --expert-parallel
```

## One-off Generation
```bash
python serve.py generate --prompt "The meaning of life is" --max-tokens 64
```
`--temperature`, `--repetition-penalty` and `--repeat-last-n` can be applied here as well

## Python Usage
```python
# assumes already running at localhost:8000
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
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

## Load Testing

To a running endpoint you can send **Poisson-Distributed** arrivals as if it was real traffic with random gaps. Prompts are instructions sampled from [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) and [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) sampled equally across length bins, and optionally behind one of three shared ~256-token system prompts to exercise prefix caching. You can control the rate (`--rate`) at which the requests will hit your endpoint.


```bash
python serve.py serve --model Qwen/Qwen3-8B --port 8000        # in one terminal
python benchmarks/simulate_serving.py --model Qwen/Qwen3-8B --rate 4 --duration 60 --max-tokens 256
```
```
--rate 1.0           mean requests per second
--duration 60        seconds of arrivals
--max-tokens 4096    output limit per request
--bins 1,64,512,2048 prompt-length bin edges, in tokens
--prefix-tokens 256  length of shared system prompt 
--dashboard          also serve a live page
```

## Performance

Everything below is run on my local **2 x A6000** system connected with NVLink with **4096-token context**, **up to 256 sequences** and at **0.85 GPU memory**. We test three different workloads:

- **Single User**: One request at a time with nothing else running, so there's no batching. After a warmup, each request generates 512 tokens with `ignore_eos` and temperature at 0. 
- **200-request Burst**: All 200 requests are submitted at once (132,590 input tokens). Each has its own output length of 64-512 tokens with `ignore_eos`, so every engine generates exactly 60,523 tokens
- **Random Arrivals**: Requests arrive at Poisson-Distributed intervals for 60 seconds at 40 req/s on 0.6B, 8 on 8B and 2 on 32B and 30B-A3B

**OOM** indicates that the weights don't fit on a single 48GB card, **n/a** is because the old engine didn't support Tensor Parallelism. 


We use `vLLM 0.22.1` with the `TRITON_ATTN` attention backend and `FULL` cudagraph mode as our comparison


### Single-User Decode Speeds

|                       | Qwen3-0.6B |        | Qwen3-8B |        | Qwen3-32B |       | Qwen3-30B-A3B |       |
|-----------------------|-----------:|-------:|---------:|-------:|----------:|------:|--------------:|------:|
|                       | 1 GPU      | 2 GPU  | 1 GPU    | 2 GPU  | 1 GPU     | 2 GPU | 1 GPU         | 2 GPU |
| tinyserve 1.0, tok/s  | 189.8      | n/a    | 36.5     | n/a    | OOM       | n/a   | OOM           | n/a   |
| tinyserve 2.0, tok/s  | 379.3      | 420.4  | 42.0     | 70.2   | OOM       | 19.4  | OOM           | 90.6  |
| tinyserve 2.1, tok/s  | **398.5**  | **569.7** | 42.2  | **79.1** | OOM     | 20.3  | OOM           | **149.0** |
| vLLM, tok/s           | 393.9      | 545.2  | **42.4** | 79.0   | OOM       | 20.3  | OOM           | 139.1 |

<details>
  <summary>Additional Performance Numbers</summary>

### 200 Request Burst
|               | Qwen3-0.6B |          | Qwen3-8B |          | Qwen3-32B |          | Qwen3-30B-A3B |          |
|---------------|-----------:|---------:|---------:|---------:|----------:|---------:|--------------:|---------:|
|               | 1 GPU      | 2 GPU    | 1 GPU    | 2 GPU    | 1 GPU     | 2 GPU    | 1 GPU         | 2 GPU    |
| tinyserve 1.0 | 3,237      | n/a      | 1,252    | n/a      | OOM       | n/a      | OOM           | n/a      |
| tinyserve 2.0 | 6,901      | 9,822    | 1,900    | 3,013    | OOM       | 686      | OOM           | 1,956    |
| tinyserve 2.1 | **6,908**  | **10,327** | **1,903** | **3,180** | OOM     | 785      | OOM           | **2,224** |
| vLLM          | 6,210      | 9,080    | 1,853    | 3,048    | OOM       | **787**  | OOM           | 2,060    |

### Random Arrivals
| Model | Config | Burst tok/s | Latency p50 / p99 (s) | TTFT p50 (s) | tok/s per request |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B (40 req/s) | tinyserve 1.0, 1 GPU | 3,237 | 100.9 / 152.0 | 79.96 | 14.7 |
| | tinyserve 2.0, 1 GPU | 6,901 | 12.3 / 19.8 | 4.56 | 31.5 |
| | tinyserve 2.1, 1 GPU | **6,908** | 12.5 / 20.3 | 4.59 | 31.1 |
| | vLLM, 1 GPU | 6,210 | **10.8** / **17.3** | **3.34** | **32.0** |
| | tinyserve 2.0, 2 GPU | 9,822 | 2.9 / 4.5 | 0.11 | 90.4 |
| | tinyserve 2.1, 2 GPU | **10,327** | 2.4 / 3.9 | **0.08** | **108.5** |
| | vLLM, 2 GPU | 9,080 | 2.4 / **3.6** | 0.09 | 105.5 |
| Qwen3-8B (8 req/s) | tinyserve 1.0, 1 GPU | 1,252 | 47.5 / 56.5 | 0.72 | 5.5 |
| | tinyserve 2.0, 1 GPU | 1,900 | 18.2 / 19.9 | 0.29 | 14.2 |
| | tinyserve 2.1, 1 GPU | **1,903** | **18.2** / **20.1** | 0.29 | **14.2** |
| | vLLM, 1 GPU | 1,853 | 19.0 / 21.7 | **0.25** | 13.4 |
| | tinyserve 2.0, 2 GPU | 3,013 | 7.0 / 8.1 | 0.12 | 36.8 |
| | tinyserve 2.1, 2 GPU | **3,180** | **6.0** / **7.0** | 0.11 | **43.4** |
| | vLLM, 2 GPU | 3,048 | 6.3 / 7.3 | **0.09** | 41.1 |
| Qwen3-32B (2 req/s) | tinyserve 2.0, 2 GPU | 686 | 22.5 / 25.1 | 0.42 | 11.5 |
| | tinyserve 2.1, 2 GPU | 785 | **20.9** / **23.6** | 0.41 | **12.6** |
| | vLLM, 2 GPU | **787** | 21.7 / 24.6 | **0.34** | 11.9 |
| Qwen3-30B-A3B (2 req/s) | tinyserve 2.0, 2 GPU | 1,956 | 8.5 / 9.4 | 0.14 | 30.5 |
| | tinyserve 2.1, 2 GPU | **2,224** | **6.1** / **7.3** | 0.11 | **40.1** |
| | vLLM, 2 GPU | 2,060 | 8.2 / 9.1 | 0.11 | 31.2 |

</details>


## Changelog

### 2.1 (2026-11-02)

The focus of this release is 2-GPU speeds. Compared to 2.0, one-user decode on 2 GPUs is shown to be faster by +35% on Qwen3-0.6B, +13% on 8B, +5% on 32B and +64% on 30B-A3B. Tinyserve now matches vLLM on our 4 target models for burst and one user decodes on 2 GPUs!

#### Added

- **Custom All-Reduce Sum** in `tinycomm.py`. Triton kernels that sum across GPUs by reading and writing each other's memory over NVLink or PCIe P2P, using Torch Symmetric Memory
  - Flag-in-data kernel for messages up to 64K values, and a one-shot kernel for anything above that. 
  - Support bf16/fp16 and picks its block size, warps and grid at startup
  - Fallback to NCCL for any unsupported dtypes and non-contiguous tensors
  - 8KB all-reduce from 10.8 to 2.3µs against torch + NCCL
- Triton `moe_route` and `moe_align_block_size` fusing the routing decisions as well as the indexing preparation for the `fused_moe` kernel leading to ~35% improvement in decode throughput
- **MoE topk sum fused into all-reduce** saving an extra `.sum()` op

#### Changed 
- **`lm_head` split across GPUs** by vocabulary instead of rank 0 computing all of it
- **Embedding table is split across GPUs** with an extra all-reduce, making more space for KV-Cache
- **KV Cache write fused into RoPE** instead of a separate write
- **More accurate KV-Cache sizing**, excluding memory used by other processes

### 2.0 (2026-10-26)

Moves from flash_attn reference engine to our own Triton kernels and Multi-GPU serving. Burst
throughput on 1 GPU roughly doubled on Qwen3-0.6B (3237 -> 6901 toks/s) and rose by 50% on 8B (2 -> 1900). 

#### Added
- `tinykernels.py` with paged attention w/ split-kv decoding, fused RMSNorm, q/k norm + rope, fused MoE layer and a fused sampler
- Tensor Parallelism and Expert Parallelism
- Async Scheduling
- CUDA Graphs for Decoding and Mixed Prefills
- MoE Support
- Repetition Penalty
- Better server side logging


### 1.0 (2026-10-19)

First release: a single-file serving engine in `serve.py`

- Paged-KV cache with Prefix Caching
- Continuous Batching with chunked prefill
- FlashAttention backend with a pure PyTorch reference
- CUDA Graphs for decode steps
- OpenAI-Compatible HTTP server
- Tested on Qwen3

## Plan

### **TinyAgents**
  - **Core**
    - [ ] Code-writing Agent Loop
    - [ ] Sandboxed Execution
  - **Tools and MultiAgents**
    - [ ] Custom Tool Use Support
    - [ ] Agents as Tools (Manager / Worker)
    - [ ] Prebuilt Tools (Websearch, Read, File Operations)

### TinyRL

- [ ] Generate rollouts with per-token logprobs
- [ ] Update server weights in-place
- [ ] GRPO Trainer on GSM8K 

### Increase Supported Models
- [ ] Muse Glimmer
- [ ] Qwen 3-Next Family
- [ ] Qwen 3.6 Family
- [ ] Gemma4 Family

### New Infra Features
- [ ] Speculative Decoding
- [ ] Quantized Models
  
### Things I Need To Explore
- [x] vLLM performs better on multiGPU on smaller models (could be an NCCL Bottleneck?)
- [x] MoE is slow on decoding compared to vLLM for 1 user (Fuse MoE routing as well)
- [x] KVCache Allocation for the same `gpu_memory_utilization` is less here compared to vLLM
- [x] KVCache write can be fused into `qk_norm_rope` (the values are just copied in, the keys, after RoPE are copied in)
- [x] In TP>1, the `lm_head` is only run by the main GPU, we can split vocab under TP and then all-reduce after
- [x] In TP>1 the `embed_tokens` is repeated on every GPU, instead we can split the vocab under TP and then all-reduce after



