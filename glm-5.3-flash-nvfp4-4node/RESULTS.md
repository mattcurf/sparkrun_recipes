# Four-Spark evaluation

Use NVIDIA NVFP4 for the measured workloads: it runs faster and leaves more memory available.
Both profiles pass functional and near-256K probes.
These results establish neither general model quality nor optimal FP8 tuning.

## Serving-profile measurements

NVFP4 measurements dated 2026-09-30 use the same cluster, checkpoint, API key, and idle-service benchmark harness.
Additional probes measure concurrency and interference.
The 2,048-token profile uses image `v4`, persistent JIT caches, and `GLM_FAST_LOAD=1`.
Its draft schedule is `[[1,1,7],[2,2,5],[3,8,3]]`.

| Measurement | 8,192-token chunks, 4 slots | 2,048-token chunks, 8 slots |
| --- | --- | --- |
| Weight load (rank 0) | 402.7 s | 22.8 s |
| Launch to `/health` (image already distributed) | ~11.5 min | ~4.9 min |
| First 4-request batch after boot, TTFT | 5.2 s | 0.4 s |
| Single stream, 384 tokens, greedy / temperature 1 | 56.5 / 53.7 tok/s | 59.5 / 60.8 tok/s |
| 4 streams aggregate, greedy / temperature 1 | 94.0 / 93.6 tok/s | 100.0 / 92.6 tok/s |
| 8 streams aggregate, temperature 1 | queued at 4 | 132.6 tok/s |
| Cold prefill alone, 25K–50K tokens | ~2,880 tok/s | ~2,220 tok/s |
| Decode steps/s while a 50K prefill runs | 0.64 (max gap 2.8 s) | 2.02 (max gap 0.71 s) |

Drafts per verification step measure 7.00, 3.28, and 3.04 at one, four, and eight streams.
Smaller chunks trade ~23% cold-prefill throughput for a 4× shorter worst-case decode stall during concurrent prefill.

## Weight-comparison setup

Measurements dated 2026-09-11 use four DGX Sparks with switched dual-rail RoCE, one rank per node.

| Shared measurement setting | Value |
| --- | --- |
| Parallelism / context | TP4 + EP; 262,144 tokens |
| KV / draft | FP8 target; BF16 DFlash2, seven drafts |
| Scheduling | Four sequences; 8,192-token prefill batches; prefix caching; CUDA graphs |
| Sampling | Low reasoning effort; greedy |

All ranks use this measurement image:

```text
glm53-flash-nvfp4-sparkrun-v2
sha256:c3f6d5d357da4d08fbfd0965802a242cb544f7c8c9852b00d937d90537bcad91
vLLM 0.28.1rc1.dev580+g385dce36b
Torch 2.13.0+cu130
CUDA 13.0
PyNCCL serving log: 2.30.7 (Torch version report: NCCL 2.29.7)
```

NVFP4 uses calibrated CUTLASS W4A4; FP8 uses Triton block-scaled W8A8 MoE and DeepGEMM FP8 linears.
Neither uses Marlin weight-only conversion.
FP8 lacks an exact `E=72,N=2048` GB10 tuning file and uses vLLM defaults.
Tests include no Triton kernel-parameter sweep.
See [current launch and evaluation commands](README.md#matched-evaluation); current scheduling differs from these measurement settings.

## Warm short-context results

Code/prose rates are medians of three normally completed outputs.
Request-wall rates include prefill, reasoning, and final content.
Approximate decode rates exclude TTFT and subtract one output token; speculative chunks limit precision.
C4 aggregate covers the complete concurrent batch, including prefill; it does not sum stream rates.
The evaluator records replies, API counts, TTFT, rates, and Prometheus metrics.
It does not execute generated code.

| Metric | NVFP4, graphs, 12 GiB KV | FP8, graphs, 8 GiB KV |
| --- | ---: | ---: |
| Completed code, request-wall tok/s | **66.36** | 49.58 |
| Completed code, approximate decode tok/s | **74.65** | 55.31 |
| Completed prose, request-wall tok/s | **30.68** | 22.32 |
| Completed prose, approximate decode tok/s | **31.81** | 23.14 |
| Warm C4 code aggregate tok/s | **159.06** | 104.91 |

Receipts: [NVFP4 warm](results/nvfp4-graphs-warm.json), [FP8 warm](results/fp8-graphs-warm.json).
Prompts and harness match; outputs differ.

| NVFP4 eager/OMP4 control | Request-wall tok/s |
| --- | ---: |
| Code | 70.09 |
| Prose | 29.27 |
| C4 aggregate | 149.56 |

Use `-o execution_flags=--enforce-eager` for eager execution.
The [measured eager control](results/nvfp4-eager-warm.json) also uses four OpenMP threads; graphs use one.
This comparison does not isolate graph effects.
Graphs do not improve every prompt uniformly.

## Actual near-limit validation

| Check | NVFP4 | FP8 |
| --- | ---: | ---: |
| Retrieval prompt tokens | 256,572 | 256,572 |
| Three audit codes at beginning/middle/end | All correct | All correct |
| Retrieval completion | Normal stop, 39 tokens | Normal stop, 36 tokens |
| Retrieval total seconds | **91.23** | 131.15 |
| Long decode prompt tokens | 256,545 | 256,545 |
| Long decode output tokens | 384 | 384 |
| Long decode TTFT seconds | **90.67** | 119.15 |
| Long decode total seconds | **94.69** | 124.62 |

Capped throughput probes end with `length`; they do not count as completed-answer quality passes.
Retrieval requires at least 250,000 actual prompt tokens, exact expected JSON, and normal termination.
Both profiles pass arithmetic, JSON, Korean, streaming, tool calling, four short concurrent requests, and four distinct concurrent ~30K decodes.
These probes show no corruption, repetition lock, or engine failure.
Complex vision/video and four simultaneous full-256K requests remain unvalidated.
Repetitive-filler retrieval does not establish general long-context reasoning quality.

Receipts: [NVFP4 retrieval](results/nvfp4-graphs-evaluation.json), [FP8 retrieval](results/fp8-graphs-evaluation.json),
[NVFP4 long decode](results/nvfp4-graphs-256k.json), [FP8 long decode](results/fp8-graphs-256k.json),
[NVFP4 quality/concurrency](results/nvfp4-graphs-quality.json), [FP8 quality/concurrency](results/fp8-graphs-quality.json).

## Memory

Each node has approximately 121.69 GiB physical unified memory.

| Measurement | NVFP4 | FP8 |
| --- | ---: | ---: |
| Loaded target + draft, reported by rank 0 | 45.81 GiB | 76.67 GiB |
| Reserved KV, per rank | 12 GiB | 8 GiB |
| Allocator's full-256K concurrency estimate | 5.55× | 3.70× |
| Linux MemAvailable after probes, across nodes | ~33.5–36.3 GiB | ~11.8–12.2 GiB |

KV estimates describe pool accounting, not larger per-request limits or extra OS memory.
These measurements cap active sequences at four; current recipes use eight.
Budget for three fully occupied FP8 contexts, not four.
FP8 head-node swap occupancy measures ~6.9 GiB; an idle sample shows zero swap I/O.
This sample does not establish swap-free operation throughout testing.
