# Validation results

These measurements come from four DGX Sparks on 2026-09-12.
The deployment used the checked-in TP4 recipe.
It used model revision `dba1be0a40aa45a94ad051997016db3960a90277`.
It used image `sha256:19c1962eb8ca323628021cb1d7283ee323c1d2f3fe8dea4a38fe9bad4a6bb4d2`.

The run disabled thinking.
It used greedy sampling.

## Correctness and compatibility

- The text smoke checks passed.
  The output counted exactly from 1–100.
  The model produced non-empty prose and code.
  A 770-token corruption probe found 0% junk tokens.
  It found no repeated trigrams.
- SSE produced four incremental chunks.
  Its output matched the expected text exactly.
- Vision and tools passed 7/7 checks.
  These included three generated-image checks.
  Tool checks covered automatic choice, a tool-result round trip, parallel calls, and forced choice.
- A 253,237-token prompt retrieved distinct keys from 1%, 50%, and 99% depth in order.
  TTFT was 166.7 s.
  Prompt throughput was 1,519.4 prompt tok/s.
  The model then produced 320 coherent output tokens.
- Every rank loaded 81.58 GiB of model state.
  Three ranks read their disjoint Engram rows from node-local storage.
  The remaining rank read from the shared checkpoint.
  This describes the measured deployment, not an all-local result.
  The final launch reused the embedded FlashInfer sparse module.
  It also reused persistent TileLang/autotune caches.

The reference project's same-prompt end-to-end smoke check reported 84.9 tok/s for counting.
It reported 65.3 tok/s for code.
This deployment measured 83.1 tok/s for counting.
It measured 63.2 tok/s for code.

These checks show results within 3% of the reference.
They do not form a controlled benchmark comparison.

The reference's full benchmark uses different prompts.
It also uses different concurrency levels and token lengths.
Its server context is 300K.

### Per-host Engram storage validation

Every rank's sparse rows moved to `$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash`.
The four-node service then restarted from the checked-in recipe.
It became healthy after 1,155 seconds.
Every rank reported its disjoint row ranges from node-local `/engram-local`.

The text, SSE streaming, and corruption smoke checks passed.
The second counting run measured 85.3 tok/s.
The second prose run measured 29.7 tok/s.
The second code run measured 66.0 tok/s.
Vision and tool calling passed 7/7 checks.

This post-migration check did not rerun llama-benchy or tool-eval-bench.
Their results below remain the historical measurements described above.

## llama-benchy 0.4.0

The run used these settings:

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

The client observed these content-token counts:

- 256 tokens for 74 requests.
- 255 tokens for eight requests.
- 254 tokens for one request.
- 251 tokens for one request.

The prior repository benchmark documents the same SSE-visible counting limitation.
Requests still set `min_tokens=256` and `ignore_eos=true`.
The recorded values remain unchanged after collection.

## tool-eval-bench 2.6.0

The run used commit `992a6978ecbee2d72fa2ead9ccc509436769d088`.
It used all 69 standard scenarios.
The seed was 42.
The temperature was 0.

The run used no thinking.
It used one sequential trial.
The timeout was 300 seconds.

The run scored **88/100 (★★★★ Good)**.
It recorded 55 pass, 12 partial, and 2 fail results.
It earned 122/138 points.

The safety gate passed with no warnings.
Deployability scored 79.
Responsiveness scored 59.

Partial scenarios:

- TC-03, TC-14, TC-28, TC-46, TC-47, TC-49.
- TC-50, TC-51, TC-53, TC-57, TC-58, TC-62.

Failures:

- TC-61: The model did not attempt the analysis script.
- TC-63: The final answer omitted accumulated constraints.

Context & State was the weakest category at 65%.
Each of these categories scored 100%:

- Parameter Precision.
- Restraint & Refusal.
- Localization.
- Structured Reasoning.
- Instruction Following.
- Toolset Scale.
- Creative Composition.
- Structured Output.
