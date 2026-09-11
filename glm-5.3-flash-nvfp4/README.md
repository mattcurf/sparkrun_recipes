# GLM-5.3-Flash NVFP4

A separate recipe for
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
It does not replace the EXL3/DFlash2 recipe or any Qwen configuration.

**Tested on two DGX Sparks (GB10/SM121), 2026-09-11.** NVIDIA documents this
checkpoint on GB200 with TP4. This recipe uses TP2 with expert parallelism and
a native NoPE attention plugin to run on two Sparks.

## Configuration

- Pinned NVIDIA checkpoint and a custom image built from the digest of NVIDIA's
  recommended `vllm/vllm-openai:glm53-flash-arm64-cu130` image.
- The base image includes Transformers 5.16.1. The build adds a pinned,
  checksum-verified LibertAI native NoPE sparse-attention plugin for SM121.
- ModelOpt NVFP4 quantization is detected from the checkpoint metadata.
- TP2, expert parallelism, allgather/reduce-scatter communication, FP8 KV cache,
  and **DFlash2 with seven speculative tokens by default**. Native MTP at
  depths 1, 2, and 4 was also tested at 32K and is selectable.
- Settings: **196,608-token (192K) context**, four sequences, 2,048-token prefill
  batches, an explicit **4 GiB KV-cache budget per GPU**, and eager execution.
  Explicit cache sizing overrides automatic `gpu_memory_utilization` sizing.
  MTP's BF16 draft needs additional memory, and automatic profiling after
  unified-memory loading produced inconsistent budgets across ranks. Four short
  requests can run together; the cache budget allows **one full-192K request**
  at a time, with additional long requests queued. Context includes both prompt
  and output tokens. Context above 192K and concurrency above four are not validated.
- Uses the checkpoint's chat template and GLM reasoning/tool parsers, with
  **low reasoning effort by default**. Clients can explicitly request high/max
  effort. Set a bounded `max_tokens` and check `finish_reason`: a `length` stop
  without final content is an incomplete answer, not a successful task.

## Why a custom image?

The stock image loads this checkpoint but fails during warmup on SM121 with
`pe_dim must be 64 for fp8_ds_mla`. GLM-5.3 uses NoPE attention: its positional
component has zero dimensions. The plugin uses a native 512-wide cache and
plain FP8 KV instead of the incompatible packed `fp8_ds_mla` format.

The build adapts the plugin to read `hf_text_config.index_topk` and preserve
GLM's complete 2,176-wide index buffer, including up to three always-selected
recent tokens. It neither truncates the buffer to 2,048 nor drops a selected
pool. The CUDA kernel supports this width directly; unused columns remain
masked by -1. Patch application fails if the pinned source anchors change.

The plugin's optional weight-only MoE input-scale override is **not enabled**:
NVIDIA already supplies calibrated activation scales. No checkpoint tensors,
on-disk model configuration, or chat template are modified.

The image also includes bounded compatibility patches:

- Initialize sparse-index buffers to -1, bounds-check pool expansion, avoid the
  persistent top-k kernel on SM12x, and disable unsupported PDL on SM121.
- Preserve vLLM's native DFlash2 implementation. Add the GLM target's auxiliary
  hidden-state interface, reconstruct/contract mHC streams at the requested
  layer boundaries, and account for the draft's sliding-window KV allocations.
- NVIDIA's native MTP layer is **BF16**, not packed NVFP4: the pinned checkpoint
  has 888 BF16 tensors and one F32 tensor in layer 45, without quantization
  scales. The draft-only in-memory config and constructor must use its own
  quantization configuration rather than inheriting the NVFP4 target's.
  `patch_nvidia_mtp.py` pins this adaptation to the verified NVIDIA revision.
  The target's calibrated quantization configuration remains unchanged.

## Run

Install [SparkRun](https://sparkrun.dev/) 0.3.6 or newer and configure a two-node
cluster. Ensure both nodes have enough free disk for the checkpoint and image,
and stop conflicting inference workloads only with the operator's approval.

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-nvfp4.yaml
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> --dry-run
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster>
```

SparkRun distributes the image and checkpoint to the worker. The API defaults
to port 8000 and model ID `GLM-5.3-Flash-NVFP4`. On the development system the
saved two-node cluster is named `first_pair`, not `first-pair`.

The checkpoint occupies approximately 204 GB on disk per node. Model loading
took about 13–14 minutes per rank, with 88.67 GiB allocated for the loaded model
on each rank. The first custom-image launch took about 25 minutes after the
containers started, including multimodal profiling and kernel warmup; allow
additional time for the initial download and distribution.

Native MTP raises loaded weight allocation to **95.63 GiB per rank**. Its v2
launches took approximately 17–20 minutes with the checkpoint already cached.

When replacing another model, stop it first. If unified memory remains retained,
the repository's `../tools/cleanup-memory.sh` can reclaim it on explicitly named
nodes before launch. Do not run memory cleanup against active inference jobs.

## Decoding and backend comparisons

The target remains NVIDIA's pinned checkpoint in every mode. Stop the previous
job before switching profiles; these commands are alternatives, not concurrent
deployments:

```bash
# Control: calibrated NVFP4, without speculative decoding.
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -o speculative_config=null

# Native MTP alternative. Depth 4 was fastest of the tested MTP depths 1/2/4.
# The BF16 draft needs its own backend selection, not forced NVFP4 CUTLASS.
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -o max_model_len=32768 \
  -o 'speculative_config={"method":"mtp","num_speculative_tokens":4,"moe_backend":"auto"}'

# Default: DFlash2's trained eight-token block, seven drafts plus one bonus.
# Its separate sliding-window KV cache stays BF16; the target stays FP8 KV.
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -o 'speculative_config={"method":"dflash","model":"incoai/GLM-5.3-Flash-DFlash2","revision":"dc77ff1c99eeb2df044ee3d4f0094eb033fee410","num_speculative_tokens":7,"kv_cache_dtype":"auto"}'

# MoE backend comparison, isolating it from speculative decoding.
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -o speculative_config=null -o moe_backend=marlin
```

Native MTP depths above one repeatedly apply the **same** prediction layer;
more speculative tokens can lower acceptance and throughput. DFlash2 uses a
separate trained draft, distributed to both nodes by this recipe.

Marlin uses NVIDIA's packed weights but **weight-only W4A16 computation**, not
the calibrated W4A4 activation path used by `flashinfer_cutlass`. It is a
numerically different backend, not a performance-only flag. Keep backend and
speculation changes separate when investigating output quality. Passing small
Korean/JSON/tool tests is not proof that a checkpoint is corruption-free.

Compared with
[tonyd2wild's reference](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3),
this build keeps the NVIDIA ModelOpt target, TP2 **with expert parallelism**,
native 512-wide NoPE attention, and the pinned image's own DFlash2 implementation.
It adapts the target/KV integration and SM121 guards instead of copying older
model implementations or changing to RedHatAI/LibertAI weights. The reference's
reported throughput is not an apples-to-apples NVIDIA benchmark.

### Reproduce the comparison

Run against an otherwise idle, healthy service after each profile starts:

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

The harness records complete replies, finish reasons, API token counts, latency,
time to first token, and a Prometheus metrics snapshot. It uses greedy sampling
and bounded output, checks final answers/structured JSON/Korean/repetition,
and exercises four concurrent requests. Performance prompts have a
384-output-token cap; their deliberate `length` termination is allowed only
for throughput probes. With `--long-context`, each long request must exceed
24K prompt tokens and produce at least 100 output tokens. Distinct first blocks
prevent prefix sharing from hiding concurrent cache pressure. `--quality-stress`
adds three longer Korean-generation probes, checking for replacement characters,
repetition, missing Korean text, and unfinished answers.

Output tokens/second includes reasoning and final content and is measured over
total request wall time, not by counting SSE chunks. Inspect speculative
acceptance metrics alongside throughput; this is a functional comparison,
not a general quality benchmark.

### Current 192K deployment on `first_pair`

The API advertises **196,608 tokens**, matching the EXL3 recipe. The target,
DFlash2-7, image, 4 GiB per-GPU KV pool, and 2,048-token prefill batches are
unchanged from the earlier 32K deployment.

- All ten common API checks passed; the short-context code probes measured
  **59.06 / 56.69 output tokens/s**.
- **195,045 prompt tokens + 384 output tokens** completed in **121.46 seconds**,
  including **115.21 seconds to first token**. This throughput probe deliberately
  reached its output cap; no corruption, repetition lock, or engine failure occurred.
- A separate **195,029-token retrieval prompt** correctly returned all three
  audit codes placed near the beginning, middle, and end. It completed normally
  in **112.23 seconds**, with no reasoning tokens.
- Health/model registration, streaming, automatic JSON tool calling, a
  7,532-token prompt, and four short concurrent requests passed after the
  near-limit decode.

The cache allocator reports 155 pool blocks and 1.78 maximum-concurrency units
at 192K: budget for **one fully occupied 192K request**, not two. Its displayed
350,278-token cache-size estimate is derived from concurrency and is not the
per-request context limit. Long-prompt prefill takes substantially longer than
short-prompt generation. These smoke tests do not establish general long-context
reasoning/retrieval quality, complex vision support, or performance above 192K.

### Original 32K-profile measurements on `first_pair`

Two 384-output-token code probes, identical prompts and greedy sampling, on
an otherwise idle two-Spark service. Rates include request overhead and count
both reasoning and final-answer tokens; these are **short-context** results
from the earlier 32K deployment. To reproduce that comparison, launch profiles
with `-o max_model_len=32768`; the recipe now defaults to 192K.

| Profile | Output tokens/s, runs 1 / 2 | Functional checks |
| --- | --- | --- |
| Original v1, CUTLASS, no speculation, explicit low effort | 14.41 / 14.45 | Plain JSON returned a Markdown fence |
| v2 control, CUTLASS, no speculation | 14.57 / 14.56 | 10/10 passed |
| v2 control, Marlin W4A16, no speculation | 14.86 / 14.87 | 10/10 passed |
| v2, CUTLASS + native MTP-1 | 24.98 / 25.03 | 10/10 passed |
| v2, CUTLASS + native MTP-2 | 30.80 / 30.43 | 10/10 passed |
| v2, CUTLASS + native MTP-4 | 38.17 / 38.00 | 10/10 passed |
| **v2, CUTLASS + DFlash2-7 (default)** | **59.26 / 58.65** | **10/10 passed** |

DFlash2 was about **4.0×** the same-image non-speculative control and **1.55×**
MTP-4 on these code probes, with 80.4% accepted/drafted tokens across the suite.
The debugging answer completed in 5.88 seconds, compared with 12.93 seconds
without speculation and 7.66 seconds with MTP-4. This is the selected default,
not a claim that every workload will improve by the code-probe ratio.

The DFlash2 follow-up passed all 16 checks, including three longer Korean
generations and five long-context decodes. A single 30,045-token prompt plus
384 output tokens completed in 28.14 seconds (21.74 seconds to first token).
Four distinct 30,050-token prompts, each generating 384 tokens, completed in
53.12–82.44 seconds including queuing and prefill; the cache admitted up to
three long requests at once. No replacement characters, repetition locks,
or engine failures were observed. DFlash2 loaded 90.22 GiB of weights per rank.

The selected deployment also passed health/model registration, exact arithmetic,
streaming, automatic `get_weather({"city":"Paris"})` tool calling, a 7,532-token
prompt, four short concurrent requests, and a solid-red image-input check.
The target handles images; the built-in drafter uses text-only draft inputs
because it does not support external multimodal embeddings. Complex vision and
video are not validated, and image-heavy requests may have lower acceptance.

Marlin improved these probes by only about 2% over the same-image CUTLASS
control. It also passed three longer Korean-generation probes, automatic
JSON tool calling, streaming, and the 7,532-token-prompt smoke test. This small
speed difference does not justify switching away from NVIDIA's calibrated
W4A4 computation as the default; Marlin remains an explicit comparison option.

The v2 checks use structured JSON output, unlike the original plain-JSON
probe. Native MTP's accepted/drafted token ratios across each complete suite
were 97.0%, 93.8%, and 85.6% for depths 1, 2, and 4 respectively. Lower
acceptance at depth 4 still produced higher throughput on these code prompts.
Do not extrapolate these rates to arbitrary reasoning or agent workloads.

MTP-4 also completed a **30,045-token prompt plus 384 output tokens** in
32.36 seconds, with 22.36 seconds to the first token. No replacement characters,
repetition lock, or engine failure was observed. First use of a new batch shape
can incur compilation: the initial four-request batch took about 5.3 seconds;
the warmed repeat took about 0.56 seconds per request. These small functional
checks do not establish general model quality or an EXL3 speed comparison.

## Verify after deployment

```bash
curl -f http://<head-node>:8000/health
curl -fsS http://<head-node>:8000/v1/models
curl --fail-with-body http://<head-node>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"GLM-5.3-Flash-NVFP4",
    "messages":[{"role":"user","content":"What is 17 + 25? Answer with only the number."}],
    "chat_template_kwargs":{"reasoning_effort":"low"},
    "temperature":0,
    "max_tokens":256
  }'
```

The pinned checkpoint always starts an assistant turn with `<think>` and
supports `reasoning_effort` (`low`, `high`, or default `max`), not
`enable_thinking=false`. Passing the latter can cause reasoning text to appear
in the final content because the parser and template disagree. Leave thinking
enabled and use the reasoning-effort parameter instead.

Expect `42` in the final answer. Check both ranks' logs for an EP-compatible
NVFP4 backend, missing weights/scales, gate/up scale mismatches, NoPE
sparse-attention errors, and OOMs.

### Recorded deployment checks

Initial non-speculative deployment on the `first_pair` two-Spark cluster
(v1 image, before the optimization changes):

| Check | Result |
| --- | --- |
| `/health` and `/v1/models` | Healthy; expected model registered |
| Greedy arithmetic | Exact final answer `42`, reasoning separated |
| Streaming arithmetic | Exact final answer `42` and SSE completion marker |
| Automatic tool choice | `get_weather` with valid JSON `{"city":"Paris"}` |
| Long prompts | Correct answers with 7,532 and 30,032 prompt tokens |
| Four concurrent requests | All four arithmetic answers correct |
| Image input | Solid red image identified correctly |
| Checkpoint gate/up global scales | All 12,096 routed-expert pairs match |

These are functional smoke/regression tests, not a model-quality benchmark.
The deployment uses `FLASHINFER_CUTLASS` NVFP4 MoE, 144/288 experts per rank,
and the plugin's plain FP8 KV path. Its registration overrides vLLM's
`FLASHINFER_MLA_SPARSE_SM120` slot, so that name still appears in the logs.

## Attention regression tests

Run on an idle Spark after building the image:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/recipe:ro" glm53-flash-nvfp4-sparkrun-v2 -lc '
    python3 /recipe/test_kv_layout.py &&
    python3 /recipe/test_sparse_mla.py &&
    python3 /opt/glm53-sparse-mla/tests/run_installed.py &&
    python3 /opt/glm53-sparse-mla/tests/run_fp8.py
  '
```

The local regression exercises the installed backend and metadata builder,
including the full kpool tail and fully masked rows, against an FP32 reference.
The upstream suites additionally cover masking, duplicate indices, extreme
logits, and different token counts for BF16 and FP8 KV.

With the pinned checkpoints already cached, test draft configuration and load
the real MTP weights without allocating the full target:

```bash
docker run --rm --gpus all --shm-size 1g --entrypoint bash \
  -e HF_HUB_OFFLINE=1 -e VLLM_GLM53_CUDA_SPARSE_MLA=0 \
  -v "$PWD:/recipe:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  glm53-flash-nvfp4-sparkrun-v2 -lc '
    python3 /recipe/test_nvidia_mtp.py && python3 /recipe/test_load_mtp.py
  '
```

This isolated TP1 test disables the NoPE plugin only because no attention is
executed and its 64-head TP1 kernel exceeds SM121 shared-memory limits. The
actual TP2 service **must** retain `VLLM_GLM53_CUDA_SPARSE_MLA=1`. The test covers
both Hub IDs and vLLM's resolved snapshot paths, draft/target quantization
separation, picklable config transforms, and real BF16 expert-weight loading.
It is not a substitute for a healthy distributed service and API tests.

## License and attribution

Original recipe, build files, test, and documentation are covered by the
recipe's [Unlicense](LICENSE). Runtime adaptation scripts and source-overlay
tests contain Apache-2.0 source excerpts and are distributed under
[Apache-2.0](LICENSE-APACHE), as identified by their SPDX headers.
The configuration follows the deployment guidance in NVIDIA's model card. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream licenses. Model
weights and container images are not distributed here.
