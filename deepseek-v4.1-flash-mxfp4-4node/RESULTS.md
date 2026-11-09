# Validation results

Measurements use four DGX Sparks, the TP4 recipe, greedy sampling, and no thinking.

| Setting | Value |
| --- | --- |
| Measurement date | 2026-09-12 |
| Model revision | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Image | `sha256:19c1962eb8ca323628021cb1d7283ee323c1d2f3fe8dea4a38fe9bad4a6bb4d2` |

## Correctness and compatibility

- Text: exact counting from 1–100; non-empty prose and code.
- Corruption probe: 770 tokens; 0% junk; no repeated trigrams.
- SSE: four incremental chunks; exact expected text.
- Vision/tools: 7/7, including three generated-image checks and automatic, round-trip, parallel, and forced tool calls.
- Retrieval: distinct keys at 1%, 50%, and 99% of 253,237 prompt tokens, in order.
  TTFT: 166.7 s; prompt throughput: 1,519.4 tok/s; coherent output: 320 tokens.
- Model state: 81.58 GiB per rank, excluding roughly 47 GiB of disk-backed tables.
  Engram sources: three local, one shared.
  Measurements are not all-local results.
- Cache checks: embedded FlashInfer sparse module and persistent TileLang/autotune cache reuse passed.
- RAM stayed within 112 GiB during checks.

| Same-prompt smoke check | Reference tok/s | This recipe tok/s |
| --- | ---: | ---: |
| Counting | 84.9 | 83.1 |
| Code | 65.3 | 63.2 |

Results are within 3%, but do not establish a controlled comparison.
The reference full benchmark uses different prompts, concurrency, token lengths, and a 300K server context.

### Per-host Engram storage validation

The all-local check uses each SSH user's `$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash` through `/engram-local`.
All ranks reported disjoint row ranges.

| Check | Result |
| --- | --- |
| Startup to healthy | 1,155 seconds |
| Text / SSE / corruption | Pass |
| Counting / prose / code | 85.3 / 29.7 / 66.0 tok/s |
| Vision and tools | 7/7 |

The benchmark results below do not measure this all-local configuration.

## llama-benchy 0.4.0

Use upstream [`eugr/llama-benchy` 0.4.0](https://github.com/eugr/llama-benchy/tree/v0.4.0).
The measured run used:

- Prompt tokens: 2,048/4,096/8,192/16,384.
- Concurrency: 1/2/4.
- Output flags: `--tg 256 --exact-tg`.
- Depth: 0.
- Measured batches: three after each discarded shape warmup.
- Cache isolation: a fresh vLLM `cache_salt` per request.

All 84 measured requests completed without errors.
The server reported zero prefix-cache hits.

| Prompt | C | Prefill tok/s (mean ± σ) | Decode tok/s (mean ± σ) | TTFT ms (mean) |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1 | 1,239.48 ± 28.41 | 32.55 ± 6.42 | 1,656.06 |
| 2,048 | 2 | 1,428.31 ± 88.76 | 38.44 ± 0.41 | 2,388.30 |
| 2,048 | 4 | 1,600.70 ± 47.41 | 46.91 ± 4.47 | 4,205.31 |
| 4,096 | 1 | 1,598.36 ± 69.46 | 30.30 ± 5.41 | 2,570.47 |
| 4,096 | 2 | 1,632.50 ± 6.80 | 39.06 ± 4.13 | 4,135.65 |
| 4,096 | 4 | 1,725.56 ± 73.63 | 42.65 ± 1.53 | 7,674.21 |
| 8,192 | 1 | 1,746.99 ± 80.21 | 31.78 ± 5.42 | 4,702.19 |
| 8,192 | 2 | 1,821.66 ± 26.40 | 44.22 ± 9.19 | 8,171.82 |
| 8,192 | 4 | 1,759.12 ± 73.24 | 32.79 ± 0.51 | 13,975.95 |
| 16,384 | 1 | 1,873.04 ± 22.76 | 35.81 ± 2.85 | 8,751.47 |
| 16,384 | 2 | 1,884.53 ± 17.12 | 33.49 ± 0.33 | 15,108.56 |
| 16,384 | 4 | 1,858.58 ± 30.09 | 26.26 ± 1.12 | 25,182.11 |

| SSE-visible content tokens | Requests |
| ---: | ---: |
| 256 | 74 |
| 255 | 8 |
| 254 | 1 |
| 251 | 1 |

SSE-visible counts can fall below 256 despite `min_tokens=256` and `ignore_eos=true`.
Use a cache-isolating proxy or fresh server cache for cold-prefix comparisons.

```bash
uvx --from llama-benchy==0.4.0 llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash-mxfp4-tp4 \
  --tokenizer /srv/sparkrun/models/DeepSeek-V4.1-Flash \
  --pp 2048 4096 8192 16384 --tg 256 --exact-tg \
  --depth 0 --runs 3 --concurrency 1 2 4 --no-cache \
  --save-result results/llama-benchy.json --format json \
  --emit-progress results/llama-benchy-progress.jsonl \
  --save-total-throughput-timeseries --exit-on-first-fail
```

## tool-eval-bench 2.6.0

The run uses [`tool-eval-bench` 2.6.0](https://github.com/SeraphimSerapis/tool-eval-bench/tree/v2.6.0)
at `992a6978ecbee2d72fa2ead9ccc509436769d088`.

| Setting or score | Value |
| --- | --- |
| Scenarios / trials | All 69 standard / one sequential |
| Seed / temperature / timeout | 42 / 0 / 300 seconds |
| Thinking | Off |
| Score | **88/100 (★★★★ Good)**; 122/138 points |
| Pass / partial / fail | 55 / 12 / 2 |
| Safety gate | Pass; no warnings |
| Deployability / responsiveness | 79 / 59 |

Partial scenarios:

- TC-03, TC-14, TC-28, TC-46, TC-47, TC-49.
- TC-50, TC-51, TC-53, TC-57, TC-58, TC-62.

Failures:

- TC-61: The model did not attempt the analysis script.
- TC-63: The final answer omitted accumulated constraints.

Context & State scored lowest: 65%.
Categories at 100%:

- Parameter Precision; Restraint & Refusal; Localization; Structured Reasoning.
- Instruction Following; Toolset Scale; Creative Composition; Structured Output.

```bash
uvx --from 'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@992a6978ecbee2d72fa2ead9ccc509436769d088' \
  tool-eval-bench run --base-url http://127.0.0.1:8000 \
  --model deepseek-v4.1-flash-mxfp4-tp4 --backend vllm \
  --seed 42 --temperature 0 --no-think --timeout 300 \
  --json-file results/tool-eval-bench.json
```
