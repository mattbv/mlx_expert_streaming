# MLX Expert Streaming Lab

Local experiment harness for testing TurboQuant-MLX Mixture-of-Experts expert streaming on Apple Silicon.

This repository is intentionally small. It wraps the public `turboquant-mlx-full` package with reproducible `make` targets, conservative defaults, a minimal OpenAI-compatible HTTP shim, and a benchmark client.

## Origin

This experiment was motivated by Manjunath Janardhan's Medium article:

[A Qwen 3.5 122B LLM on a 16 GB Mac mini: MoE Expert Streaming with TurboQuant-MLX](https://medium.com/data-science-collective/a-qwen-3-6-122b-llm-on-a-16-gb-mac-mini-moe-expert-streaming-with-turboquant-mlx-4f77f0b48518)

The article demonstrates that very large sparse MoE models can run on small Apple Silicon machines by streaming only the active experts from disk per token. The key idea is that MoE inference does not need every expert resident at once: the router selects a small subset of experts per token, and those expert slices can be read from disk into a bounded cache.

Important context from the article:

| Model | Disk size | Machine | Cache | Peak memory | Reported speed |
| --- | ---: | --- | ---: | ---: | ---: |
| `manjunathshiva/Qwen3.6-35B-A3B-tq3-g32` | ~16 GB | 16 GB Mac mini | 2-8 GB | ~4-9 GB | ~3-4.5 tok/s |
| `manjunathshiva/qwen3.5-122b-tq3` | ~54 GB | 16 GB Mac mini | 1-4 GB | ~6-9 GB | ~0.65-1.08 tok/s |

This repo starts even more conservatively because the initial local test machine is an 8 GB M1 Mac.

## What This Repo Provides

| File | Purpose |
| --- | --- |
| `Makefile` | Stable entry points for setup, generation, serving, and benchmarking. |
| `scripts/setup-turboquant.sh` | Creates `~/tq-env` and installs dependencies. |
| `scripts/tq_openai_server.py` | Minimal OpenAI-compatible chat-completions shim around `load_streaming`. |
| `scripts/benchmark_openai.py` | Sends a small prompt suite to the local shim and prints timing. |

## Hardware Expectations

This is experimental and can stress memory, Metal, SSD bandwidth, and disk capacity.

Recommended minimums:

| Test | Minimum suggested machine | Disk needed | Notes |
| --- | --- | ---: | --- |
| 35B streaming smoke test | Apple Silicon, 8-16 GB RAM | ~20 GB free | Starts with `CACHE_GB=0.5` here. |
| 35B larger-cache test | Apple Silicon, 16 GB+ RAM | ~20 GB free | Higher cache improves speed but increases wired memory. |
| 122B smoke test | Apple Silicon, 16 GB+ RAM | ~60 GB free | May still Metal OOM; slow even when successful. |

On an 8 GB Mac, prefer `CACHE_GB=0.25` or `CACHE_GB=0.5`. Avoid `CACHE_GB=4` unless you know the machine can handle it.

## Prerequisites

Install Homebrew Python 3.12 if needed:

```bash
brew install python@3.12
```

Optional but recommended for faster Hugging Face downloads:

```bash
export HF_TOKEN="your_hugging_face_token"
```

The models are downloaded into the Hugging Face cache, typically:

```text
~/.cache/huggingface/hub
```

## Setup

From the repo root:

```bash
make setup
```

This creates:

```text
~/tq-env
```

And installs:

```text
turboquant-mlx-full>=0.4.1
requests
```

You do not need to activate the virtualenv for the Makefile targets. They use `~/tq-env/bin/python` directly.

## Quick Start

Run a conservative one-shot 35B generation:

```bash
make generate
```

Default values:

```text
MODEL=manjunathshiva/Qwen3.6-35B-A3B-tq3-g32
CACHE_GB=0.5
MAX_TOKENS=64
PROMPT="Explain why the sky is blue in one concise paragraph."
```

On first run, expect a large download:

```text
~16.9 GB for the 35B model
```

Use a smaller cache and output length if the machine feels unstable:

```bash
make generate CACHE_GB=0.25 MAX_TOKENS=32
```

Use a custom prompt:

```bash
make generate PROMPT="Explain expert streaming for MoE models in five bullet points." MAX_TOKENS=128
```

## Make Targets

| Target | Description |
| --- | --- |
| `make setup` | Create/update the Python virtualenv and install dependencies. |
| `make generate` | Run direct `turboquant_mlx.stream.stream_generate` generation. |
| `make serve` | Start the local OpenAI-compatible HTTP shim. |
| `make bench` | Benchmark the HTTP shim with three prompts. |
| `make bench-122b` | Run a very constrained 122B direct-generation smoke test. |
| `make clean-cache` | Purge pip's package cache only. Does not remove downloaded models. |

## Configurable Variables

Pass variables to `make` to change runtime behavior.

| Variable | Default | Used by | Meaning |
| --- | --- | --- | --- |
| `PYTHON` | `$(HOME)/tq-env/bin/python` | all Python targets | Python interpreter. |
| `MODEL` | `manjunathshiva/Qwen3.6-35B-A3B-tq3-g32` | generate, serve, bench | Hugging Face model repo. |
| `CACHE_GB` | `0.5` | generate, serve | Expert cache budget in GB. |
| `MAX_TOKENS` | `64` | generate, serve, bench | Max generation length. |
| `PROMPT` | sky-blue prompt | generate | Prompt for direct generation. |
| `HOST` | `127.0.0.1` | serve, bench | Local bind/connect host. |
| `PORT` | `8090` | serve, bench | Local HTTP port. |

Examples:

```bash
make generate CACHE_GB=1 MAX_TOKENS=128
```

```bash
make serve PORT=8091 CACHE_GB=0.25 MAX_TOKENS=32
```

```bash
make bench PORT=8091 MAX_TOKENS=32
```

## OpenAI-Compatible Shim

Start the local server:

```bash
make serve
```

The server loads the streaming model once, then exposes a small subset of the OpenAI API:

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
```

Health check:

```bash
curl http://127.0.0.1:8090/health
```

List models:

```bash
curl http://127.0.0.1:8090/v1/models
```

Chat completion:

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "manjunathshiva/Qwen3.6-35B-A3B-tq3-g32",
    "messages": [
      {
        "role": "user",
        "content": "Explain expert streaming for MoE models in one paragraph."
      }
    ],
    "max_tokens": 64,
    "temperature": 0
  }'
```

Limitations of the shim:

| Capability | Status |
| --- | --- |
| Non-streaming chat completions | Supported. |
| `GET /v1/models` | Supported. |
| OpenAI streaming/SSE | Not implemented. |
| Tool calls/function calling | Not implemented. |
| Embeddings/completions/responses API | Not implemented. |
| Multi-model routing | Not implemented. |

This shim is for benchmarks and integration experiments, not production serving.

## Benchmarking

Start the server first:

```bash
make serve
```

Then run the benchmark from another terminal:

```bash
make bench
```

The benchmark sends three prompts:

```text
1. JSON validation function
2. mmap vs pread explanation
3. two-largest-values function
```

It reports elapsed time, approximate word count, and an aggregate words-per-second summary. The HTTP shim also includes a `benchmark` object in each response with elapsed seconds and cache stats when available.

## 122B Smoke Test

The 122B target is intentionally constrained:

```bash
make bench-122b
```

This expands to:

```text
MODEL=manjunathshiva/qwen3.5-122b-tq3
CACHE_GB=0.25
MAX_TOKENS=32
```

Expect:

| Resource | Approximate expectation |
| --- | --- |
| Download size | ~54 GB |
| Speed | Slow; potentially around reading pace or worse |
| Risk on 8 GB M1 | High: possible Metal OOM or system pressure |
| Better fit | 16 GB+ Apple Silicon with enough free SSD space |

If the machine becomes unresponsive, stop the process with `Ctrl+C`. If the OS is already under heavy pressure, a reboot may be required.

## Disk Cache Management

Model downloads are not stored in this repository. They are stored in the Hugging Face cache.

Inspect cache size:

```bash
du -sh ~/.cache/huggingface/hub ~/.cache/huggingface/hub/* 2>/dev/null
```

Remove the 35B streaming model:

```bash
rm -rf ~/.cache/huggingface/hub/models--manjunathshiva--Qwen3.6-35B-A3B-tq3-g32
```

Remove the 122B streaming model:

```bash
rm -rf ~/.cache/huggingface/hub/models--manjunathshiva--qwen3.5-122b-tq3
```

Remove Xet transfer cache if needed:

```bash
rm -rf ~/.cache/huggingface/xet
```

`make clean-cache` only runs `pip cache purge`; it does not remove Hugging Face model files.

## Troubleshooting

### `Python 3.12 not found`

Install Homebrew Python:

```bash
brew install python@3.12
```

Or override the Python path:

```bash
PYTHON_BIN=/path/to/python3.12 make setup
```

### Hugging Face warns about unauthenticated requests

Set a token for better rate limits:

```bash
export HF_TOKEN="your_hugging_face_token"
```

### Metal OOM or command buffer failure

Lower the cache and output length:

```bash
make generate CACHE_GB=0.25 MAX_TOKENS=32
```

Close memory-heavy apps before retrying.

### The model downloads but generation is extremely slow

That is expected for expert streaming on small machines. Once memory is bounded, SSD bandwidth becomes the bottleneck.

### The benchmark cannot connect

Start the server first:

```bash
make serve
```

Then in another terminal:

```bash
make bench
```

If you changed `PORT`, pass the same value to both commands.

## Implementation Notes

The core streaming implementation lives in `turboquant-mlx-full`, not in this repository. This repo simply calls:

```python
from turboquant_mlx.stream.loader import load_streaming
```

And, for generation:

```python
from mlx_lm import generate
```

The Medium article describes the key low-level ideas:

| Concept | Why it matters |
| --- | --- |
| Per-expert slicing | Only selected MoE experts are read for each token. |
| Bounded LRU cache | Recently used experts stay available without loading the full model. |
| `pread` | Reads explicit byte ranges from model shards. |
| `F_NOCACHE` | Avoids macOS page-cache bloat that would otherwise consume RAM. |
| Metal wired-memory cap | Cache size can fail due to GPU-wired memory before total RAM is exhausted. |

## License

This repository is Apache-2.0 licensed. See `LICENSE`.

The underlying models and packages have their own licenses. Check the relevant Hugging Face model cards and package metadata before redistribution or production use.
