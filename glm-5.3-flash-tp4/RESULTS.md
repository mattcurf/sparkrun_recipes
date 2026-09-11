# Four-Spark evaluation — 2026-09-11

**Recommended profile: NVIDIA NVFP4.** Both weight variants passed the recorded
functional and near-256K checks. NVFP4 was faster on the tested workloads and
left substantially more memory available. This is not a general model-quality
comparison or a claim that every FP8 backend has been optimized.

## Controlled setup

Four DGX Sparks on the saved `gb` cluster, switched dual-rail RoCE, one rank per
node. Both profiles use TP4 + EP, 262,144 maximum tokens, FP8 target KV, BF16
DFlash2 KV, k7, four sequences, 8,192-token prefill batches, prefix caching,
low reasoning effort, greedy sampling, and CUDA graphs. All four ranks used
the identical image:

```text
glm53-flash-nvfp4-sparkrun-v2
sha256:c3f6d5d357da4d08fbfd0965802a242cb544f7c8c9852b00d937d90537bcad91
vLLM 0.28.1rc1.dev580+g385dce36b
Torch 2.13.0+cu130
CUDA 13.0
PyNCCL serving log: 2.30.7 (Torch version report: NCCL 2.29.7)
```

Target and draft revisions are pinned in the YAMLs. NVFP4 uses NVIDIA's
calibrated CUTLASS W4A4 path; native FP8 uses Triton block-scaled W8A8 MoE
and DeepGEMM FP8 linear kernels. There is no Marlin weight-only conversion.
The FP8 runtime warns that its exact `E=72,N=2048` GB10 MoE tuning file is
absent; it uses vLLM defaults. No Triton kernel-parameter sweep was performed.
Consequently, the results describe these tested serving profiles, not the
best possible performance of each numerical format.

## Warm short-context results

Medians of three completed outputs for code/prose. Request-wall rates include
prefill, reasoning, and final content. Approximate decode rates exclude TTFT
and subtract one output token; speculative chunks make these approximate.
The C4 aggregate covers the whole four-request batch, including its prefill,
and is **not** a single conversation's generation speed.

| Metric | NVFP4, graphs, 12 GiB KV | FP8, graphs, 8 GiB KV |
| --- | ---: | ---: |
| Completed code, request-wall tok/s | **66.36** | 49.58 |
| Completed code, approximate decode tok/s | **74.65** | 55.31 |
| Completed prose, request-wall tok/s | **30.68** | 22.32 |
| Completed prose, approximate decode tok/s | **31.81** | 23.14 |
| Warm C4 code aggregate tok/s | **159.06** | 104.91 |

Receipts: [NVFP4 warm](results/nvfp4-graphs-warm.json),
[FP8 warm](results/fp8-graphs-warm.json). Both used the same harness and prompts;
the outputs themselves differ. Near-limit results and these short-prompt rates
must not be combined into a claim of 66 tok/s over a full cold 256K request.

The original NVFP4 eager/OMP4 control measured **70.09 code**, **29.27 prose**,
and **149.56 C4 aggregate** request-wall tok/s. Graphs/OMP1 did not uniformly
improve every prompt: code was slightly slower in the warm repeat, while prose
and aggregate improved. Graphs remain enabled and passed long-context checks.
This is a serving-profile comparison, not an isolated graph-only experiment.
See [eager warm receipt](results/nvfp4-eager-warm.json).

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

The 384-token decode is deliberately capped and ends with `length`; this is
not an unfinished-answer pass in the quality tests. Retrieval must stop
normally and return exactly the expected JSON. Both also passed arithmetic,
structured JSON, Korean generation, SSE streaming, automatic tool calling,
four short concurrent requests, and four distinct concurrent ~30K decodes.
No corruption, repetition lock, or engine failure occurred in the successful
test runs. Complex vision/video and four simultaneous full-256K requests were
not validated. Retrieval over repetitive filler is a smoke test, not a full
long-context reasoning benchmark.

Receipts: [NVFP4 retrieval](results/nvfp4-graphs-evaluation.json),
[FP8 retrieval](results/fp8-graphs-evaluation.json),
[NVFP4 long decode](results/nvfp4-graphs-256k.json),
[FP8 long decode](results/fp8-graphs-256k.json),
[NVFP4 quality/concurrency](results/nvfp4-graphs-quality.json),
[FP8 quality/concurrency](results/fp8-graphs-quality.json).

## Memory and failed configurations

Each node has approximately 121.69 GiB physical unified memory.

| Measurement | NVFP4 | FP8 |
| --- | ---: | ---: |
| Loaded target + draft, reported by rank 0 | 45.81 GiB | 76.67 GiB |
| Reserved KV, per rank | 12 GiB | 8 GiB |
| Allocator's full-256K concurrency estimate | 5.55× | 3.70× |
| Linux MemAvailable after probes, across nodes | ~33.5–36.3 GiB | ~11.8–12.2 GiB |

These KV estimates are pool/concurrency accounting, **not larger per-request
context windows**. `max_num_seqs=4` still caps active sequences. Budget for
three fully occupied FP8 contexts, not four. Unused KV blocks are already
reserved and are not additional OS memory. Head-node swap occupancy after
FP8 startup/testing was approximately 6.9 GiB; a short idle `vmstat` sample
showed zero swap I/O, which does not establish that the entire run was swap-free.

Failed configurations were not silently treated as successes:

- Parallel Docker image exports exhausted the head's root filesystem. The
  identical image was transferred sequentially from node 2, with image IDs
  verified on every node. No existing image or model cache was deleted.
- The 85% startup utilization threshold rejected ~101.94 GiB free against
  103.44 GiB requested. Both recipes now use 80%, with explicit KV sizing.
- **FP8 weights with 12 GiB KV failed at allocation**, with kernel
  `NV_ERR_NO_MEMORY` / `_memdescAllocInternal` errors and stalled initialization.
  All ranks were stopped, page cache reclaimed, and the successful FP8 retry
  used 8 GiB. It retained 256K context, DFlash2, and CUDA graphs. No checkpoint
  change, reboot, or runtime patch was needed for the retry.

NVIDIA NVFP4 is the selected four-node deployment; native FP8 remains a tested
alternative, not a claim of better output quality. The existing two-node
recipes are unchanged and can still be launched separately after stopping TP4.
