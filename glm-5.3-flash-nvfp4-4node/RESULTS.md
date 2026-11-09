# Four-Spark evaluation — 2026-09-11

**Recommended profile: NVIDIA NVFP4.**
Both weight variants passed the recorded functional and near-256K checks.
NVFP4 ran faster on the tested workloads.
It left substantially more memory available.
This is not a general model-quality comparison.
These results do not establish that every FP8 backend has optimal settings.

## 2026-09-30 serving update (NVFP4, image `v4`)

This update changes these settings from the profile below:

- `max_num_batched_tokens`: 8,192 → 2,048.
- `max_num_seqs`: 4 → 8.
- Batch-size draft schedule: `[[1,1,7],[2,2,5],[3,8,3]]`.
- Persistent JIT caches.
- `GLM_FAST_LOAD=1`.

Tests used the same cluster, checkpoint, and `VLLM_API_KEY`.
Measurements used an idle service with the scripts in `../glm-5.3-flash-nvfp4-2node/benchmark.py`.
They also used ad-hoc concurrency and interference probes.

| Measurement | Before | After |
| --- | --- | --- |
| Weight load (rank 0) | 402.7 s | 22.8 s |
| Launch to `/health` (image already distributed) | ~11.5 min | ~4.9 min |
| First 4-request batch after boot, TTFT | 5.2 s | 0.4 s |
| Single stream, 384 tokens, greedy / temperature 1 | 56.5 / 53.7 tok/s | 59.5 / 60.8 tok/s |
| 4 streams aggregate, greedy / temperature 1 | 94.0 / 93.6 tok/s | 100.0 / 92.6 tok/s |
| 8 streams aggregate, temperature 1 | queued at 4 | 132.6 tok/s |
| Cold prefill alone, 25K–50K tokens | ~2,880 tok/s | ~2,220 tok/s |
| Decode steps/s while a 50K prefill runs | 0.64 (max gap 2.8 s) | 2.02 (max gap 0.71 s) |

Drafted tokens per verification step were 7.00, 3.28, and 3.04 at one, four, and eight streams respectively.
The 2,048-token batches reduced cold-prefill throughput by about 23%.
They provided a 4× shorter worst-case decode stall during concurrent prefill.

## Controlled setup

Tests used four DGX Sparks with switched dual-rail RoCE.
Each node ran one rank.
Both profiles use TP4 + EP with 262,144 maximum tokens.
Both use FP8 target KV, BF16 DFlash2 KV, and k7.
Both use four sequences, 8,192-token prefill batches, and prefix caching.
Both use low reasoning effort, greedy sampling, and CUDA graphs.
All four ranks used the identical image:

```text
glm53-flash-nvfp4-sparkrun-v2
sha256:c3f6d5d357da4d08fbfd0965802a242cb544f7c8c9852b00d937d90537bcad91
vLLM 0.28.1rc1.dev580+g385dce36b
Torch 2.13.0+cu130
CUDA 13.0
PyNCCL serving log: 2.30.7 (Torch version report: NCCL 2.29.7)
```

The YAMLs pin target and draft revisions.
NVFP4 uses NVIDIA's calibrated CUTLASS W4A4 path.
Native FP8 uses Triton block-scaled W8A8 MoE and DeepGEMM FP8 linear kernels.
Neither profile uses Marlin weight-only conversion.

The FP8 runtime warns that its exact `E=72,N=2048` GB10 MoE tuning file is absent.
It uses vLLM defaults.
Tests did not include a Triton kernel-parameter sweep.
Thus, the results describe these tested serving profiles, not the best possible performance of each numerical format.

## Warm short-context results

Code/prose results show medians of three completed outputs.
Request-wall rates include prefill, reasoning, and final content.
Approximate decode rates exclude TTFT and subtract one output token.
Speculative chunks make these rates approximate.
The C4 aggregate covers the whole four-request batch, including its prefill.
It is **not** a single conversation's generation speed.

| Metric | NVFP4, graphs, 12 GiB KV | FP8, graphs, 8 GiB KV |
| --- | ---: | ---: |
| Completed code, request-wall tok/s | **66.36** | 49.58 |
| Completed code, approximate decode tok/s | **74.65** | 55.31 |
| Completed prose, request-wall tok/s | **30.68** | 22.32 |
| Completed prose, approximate decode tok/s | **31.81** | 23.14 |
| Warm C4 code aggregate tok/s | **159.06** | 104.91 |

Receipts: [NVFP4 warm](results/nvfp4-graphs-warm.json),
[FP8 warm](results/fp8-graphs-warm.json).
Both used the same harness and prompts.
The outputs themselves differ.
Do not combine near-limit results and these short-prompt rates to claim 66 tok/s over a full cold 256K request.

The original NVFP4 eager/OMP4 control measured **70.09 code**, **29.27 prose**, and **149.56 C4 aggregate** request-wall tok/s.
Graphs/OMP1 did not improve every prompt uniformly.
Code ran slightly slower in the warm repeat.
Prose and aggregate rates improved.

Graphs remain enabled.
They passed long-context checks.
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

The 384-token decode deliberately caps output and ends with `length`.
This is not an unfinished-answer pass in the quality tests.
Retrieval must stop normally and return exactly the expected JSON.

Both also passed arithmetic, structured JSON, Korean generation, SSE streaming, and automatic tool calling.
Both passed four short concurrent requests and four distinct concurrent ~30K decodes.
No corruption, repetition lock, or engine failure occurred in the successful test runs.

Tests did not validate complex vision/video or four simultaneous full-256K requests.
Retrieval over repetitive filler is a smoke test, not a full long-context reasoning benchmark.

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

These KV estimates describe pool/concurrency accounting, **not larger per-request context windows**.
`max_num_seqs=4` still caps active sequences.
Budget for three fully occupied FP8 contexts, not four.
The runtime already reserves unused KV blocks.
They are not additional OS memory.

Head-node swap occupancy after FP8 startup/testing was approximately 6.9 GiB.
A short idle `vmstat` sample showed zero swap I/O.
This does not establish that the entire run was swap-free.

The evaluation did not silently treat failed configurations as successes:

- Parallel Docker image exports exhausted the head's root filesystem.
  The operator transferred the identical image sequentially from node 2.
  The operator verified image IDs on every node.
  The operator did not delete any existing image or model cache.
- The 85% startup utilization threshold rejected ~101.94 GiB free against 103.44 GiB requested.
  Both recipes now use 80%, with explicit KV sizing.
- **FP8 weights with 12 GiB KV failed at allocation**.
  The kernel reported `NV_ERR_NO_MEMORY` / `_memdescAllocInternal` errors.
  Initialization stalled.
  The operator stopped all ranks and reclaimed page cache.
  The successful FP8 retry used 8 GiB.
  It retained 256K context, DFlash2, and CUDA graphs.
  The retry needed no checkpoint change, reboot, or runtime patch.

NVIDIA NVFP4 is the selected four-node deployment.
Native FP8 remains a tested alternative.
These results do not establish better FP8 output quality.
The existing two-node recipes remain unchanged.
You can still launch them separately after stopping TP4.
