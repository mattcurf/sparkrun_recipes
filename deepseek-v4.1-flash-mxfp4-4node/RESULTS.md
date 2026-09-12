# Validation results

Measured 2026-09-12 on four DGX Sparks with the checked-in TP4 recipe, model
revision `dba1be0a40aa45a94ad051997016db3960a90277`, and image
`sha256:19c1962eb8ca323628021cb1d7283ee323c1d2f3fe8dea4a38fe9bad4a6bb4d2`.
Thinking was disabled and sampling was greedy.

## Correctness and compatibility

- Text smoke passed: exact count 1–100, non-empty prose and code, and a
  770-token corruption probe with 0% junk tokens and no repeated trigrams.
- SSE passed with four incremental chunks and the exact expected text.
- Vision and tools passed 7/7: three generated-image checks, automatic tool
  choice, a tool-result round trip, parallel calls, and forced tool choice.
- A 253,237-token prompt retrieved distinct keys placed at 1%, 50%, and 99%
  depth in order. TTFT was 166.7 s (1,519.4 prompt tok/s), followed by 320
  coherent output tokens.
- Every rank loaded 81.58 GiB of model state. Three ranks read their disjoint
  Engram rows from node-local storage; the remaining rank read from the shared
  checkpoint. This describes the measured deployment, not an all-local result.
  The final launch reused the embedded FlashInfer sparse module and persistent
  TileLang/autotune caches.

The reference project's same-prompt end-to-end smoke reported 84.9 tok/s for
counting and 65.3 tok/s for code. This deployment measured 83.1 and 63.2 tok/s,
respectively. These are useful parity checks (within 3%), not a controlled
benchmark comparison; the reference's full benchmark uses different prompts,
concurrency levels, token lengths, and a 300K server context.

### Per-host Engram storage validation

After moving every rank's sparse rows to
`$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash`, the four-node service
was restarted from the checked-in recipe. It became healthy after 1,155 seconds,
and every rank reported its disjoint row ranges as read from node-local
`/engram-local`. The text, SSE streaming, and corruption smoke checks passed;
the measured second runs were 85.3 tok/s for counting, 29.7 tok/s for prose, and
66.0 tok/s for code. Vision and tool calling passed 7/7. This post-migration
check did not rerun llama-benchy or tool-eval-bench, whose results below remain
the historical measurements described above.

## llama-benchy 0.4.0

The run used 2,048/4,096/8,192/16,384 prompt tokens, concurrency 1/2/4,
`--tg 256 --exact-tg`, depth 0, three measured batches after each discarded
shape warmup, and a fresh vLLM `cache_salt` per request. All 84 measured requests
completed without errors and the server reported zero prefix-cache hits.

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

The client observed 256 content tokens for 74 requests, 255 for eight, 254 for
one, and 251 for one. This is the same SSE-visible counting caveat documented by
the prior repository benchmark: requests still set `min_tokens=256` and
`ignore_eos=true`; values were not rewritten after collection.

## tool-eval-bench 2.6.0

Commit `992a6978ecbee2d72fa2ead9ccc509436769d088`, all 69 standard scenarios,
seed 42, temperature 0, no thinking, one sequential trial, and a 300-second
timeout. The run completed with **88/100 (★★★★ Good)**: 55 pass, 12 partial,
and 2 fail (122/138 points). Safety gate passed with no warnings; deployability
was 79 and responsiveness 59.

Partial scenarios: TC-03, TC-14, TC-28, TC-46, TC-47, TC-49, TC-50, TC-51,
TC-53, TC-57, TC-58, and TC-62. Failures were TC-61 (did not attempt the
analysis script) and TC-63 (final answer omitted accumulated constraints). The
weakest category was Context & State at 65%; Parameter Precision, Restraint &
Refusal, Localization, Structured Reasoning, Instruction Following, Toolset
Scale, Creative Composition, and Structured Output each scored 100%.

## Runtime caveat

SparkRun 0.3.6 applies the recipe's 112 GiB RAM limit at container creation but
cannot express the reference's matching 112 GiB combined memory-and-swap limit;
Docker therefore reports 224 GiB combined. Post-launch `docker update` is not a
workaround: it reproducibly removed GPU access on the two nodes running NVIDIA
driver 580.159.03. Across final startup and all tests, the highest cgroup RAM
peak was 34.9 GiB and the highest swap peak was 14.0 GiB, with no OOM events.
