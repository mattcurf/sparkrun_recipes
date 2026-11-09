# Two-Spark evaluation

DFlash2 is the recommended default for these workloads.
These functional probes do not establish general quality, complex vision/video support, or an EXL3 speed comparison.

## Benchmark setup

Measurements used two DGX Sparks (GB10/SM121) on 2026-09-11.
Code probes used identical prompts, greedy sampling, and a 384-output-token cap on an idle service.
Request-wall rates include overhead, reasoning, and final content.
Only throughput probes permit deliberate `length` termination.

Run from this directory against each idle, healthy profile:

```bash
python3 benchmark.py --base http://<head-node>:8000 --output results.json
# Final regression: single and four concurrent decodes beyond 24K context.
python3 benchmark.py --base http://<head-node>:8000 \
  --long-context --quality-stress --output long-context-results.json
# Near the 192K limit: approximately 195K input tokens, one long request.
python3 benchmark.py --base http://<head-node>:8000 --rounds 0 \
  --long-context --context-lines 13000 --long-concurrency 0 \
  --request-timeout 1200 --output 192k-results.json
```

Export `VLLM_API_KEY` for authenticated services.
The harness records replies, finish reasons, API token counts, latency, TTFT, and Prometheus metrics.
Inspect speculative acceptance alongside throughput.
Long probes require over 24K prompt tokens and at least 100 output tokens.
Distinct first blocks prevent shared prefixes from hiding cache pressure.
Quality stress checks Korean text, replacement characters, repetition, and completion.

## 192K profile

| Measurement | Result |
| --- | --- |
| Common API checks | 10/10 passed |
| Short code probes, output tok/s | 59.06 / 56.69 |
| Long decode | 195,045 prompt + 384 output tokens; 121.46 s total; 115.21 s TTFT; capped output |
| Retrieval | 195,029 prompt tokens; three correct beginning/middle/end audit codes; 112.23 s; normal stop; no reasoning tokens |
| Post-decode checks | Health, model registration, streaming, JSON tool calls, 7,532-token prompt, four short concurrent requests passed |
| KV allocator | 155 blocks; 1.78 concurrency units; displayed estimate 350,278 tokens |

The allocator estimate does not increase the 196,608-token per-request limit.
Budget for one full-context request.
Additional long requests queue.
No corruption, repetition lock, or engine failure occurred in these probes.
Repetitive retrieval probes do not establish general long-context reasoning quality.

## 32K comparison

Launch comparison profiles with `-o max_model_len=32768`; the recipe default is 192K.
Image labels identify measurement conditions, not supported deployment alternatives.
The v2 checks use structured JSON, unlike the v1 plain-JSON probe.
Keep backend and speculation changes separate when comparing quality.

| Measured profile | Output tok/s, runs 1 / 2 | Functional checks |
| --- | --- | --- |
| v1, CUTLASS, no speculation, explicit low effort | 14.41 / 14.45 | Plain JSON failed: Markdown fence |
| v2, CUTLASS, no speculation | 14.57 / 14.56 | 10/10 passed |
| v2, Marlin W4A16, no speculation | 14.86 / 14.87 | 10/10 passed |
| v2, CUTLASS + MTP-1 | 24.98 / 25.03 | 10/10 passed |
| v2, CUTLASS + MTP-2 | 30.80 / 30.43 | 10/10 passed |
| v2, CUTLASS + MTP-4 | 38.17 / 38.00 | 10/10 passed |
| v2, CUTLASS + DFlash2-7 | 59.26 / 58.65 | 10/10 passed |

| Additional measurement | Result |
| --- | --- |
| DFlash2 speed ratio | ~4.0× non-speculative control; 1.55× MTP-4 |
| Accepted/drafted ratio | DFlash2 80.4%; MTP-1/2/4: 97.0% / 93.8% / 85.6% |
| Debugging-answer task latency | DFlash2 5.88 s; control 12.93 s; MTP-4 7.66 s |
| DFlash2 stress suite | 16/16 passed; three longer Korean generations; five long decodes |
| DFlash2 single long decode | 30,045 prompt + 384 output tokens; 28.14 s total; 21.74 s TTFT |
| DFlash2 four distinct long decodes | Each 30,050 prompt + 384 output tokens; 53.12–82.44 s including queue/prefill; up to three admitted together |
| MTP-4 single long decode | 30,045 prompt + 384 output tokens; 32.36 s total; 22.36 s TTFT |
| Four-request batch compilation / warm repeat | ~5.3 s batch / ~0.56 s per request |
| Marlin speed ratio | ~2% above CUTLASS control |

Marlin's small speed gain does not justify replacing calibrated W4A4 as default.
More draft tokens can reduce acceptance and throughput.
No replacement characters, repetition locks, or engine failures occurred in the long-decode probes.
Marlin also passed Korean stress, JSON tool calls, streaming, and the 7,532-token smoke test.

## Functional and resource checks

| Non-speculative v1 check | Result |
| --- | --- |
| Health / model registration | Passed |
| Greedy / streaming arithmetic | Final answer `42`; separate reasoning; SSE completion marker |
| Automatic tool choice | `get_weather({"city":"Paris"})` |
| Long-prompt smoke tests | Correct at 7,532 and 30,032 prompt tokens |
| Four short concurrent requests | Correct arithmetic answers |
| Simple image input | Solid red image identified |
| Checkpoint gate/up global scales | All 12,096 routed-expert pairs match |

| Resource measurement | Result |
| --- | --- |
| Loaded weights per rank | Non-speculative 88.67 GiB; DFlash2 90.22 GiB; native MTP 95.63 GiB |
| Model loading | ~13–14 min per rank |
| Container start to ready, profiling/warmup included | ~25 min custom image; ~17–20 min cached MTP v2 |

The drafter accepts text-only inputs.
Image-heavy requests can reduce acceptance.
Small functional checks do not prove the absence of checkpoint corruption.
The [reference deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3)
does not provide directly comparable NVIDIA measurements.
