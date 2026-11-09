# GLM-5.3-Flash NVFP4

This directory contains a separate recipe for
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
It does not replace the EXL3/DFlash2 recipe or any Qwen configuration.

**Tests used two DGX Sparks (GB10/SM121) on 2026-09-11.**
NVIDIA documents this checkpoint on GB200 with TP4.
This recipe uses TP2 with expert parallelism to run on two Sparks.
It also uses a native NoPE attention plugin.

## Configuration

- The recipe pins the NVIDIA checkpoint.
  It builds a custom image from the digest of NVIDIA's recommended `vllm/vllm-openai:glm53-flash-arm64-cu130` image.
- The base image includes Transformers 5.16.1.
  The build adds a pinned LibertAI native NoPE sparse-attention plugin for SM121.
  It verifies the plugin's checksum.
- The runtime detects ModelOpt NVFP4 quantization from the checkpoint metadata.
- The recipe uses TP2, expert parallelism, allgather/reduce-scatter communication, and FP8 KV cache.
  It uses **DFlash2 with seven speculative tokens by default**.
  Tests also covered native MTP at depths 1, 2, and 4 at 32K.
  You can select native MTP.
- The recipe provides **196,608-token (192K) context**.
  It uses four sequences, 2,048-token prefill batches, and eager execution.
  It sets an explicit **4 GiB KV-cache budget per GPU**.
  Explicit cache sizing overrides automatic `gpu_memory_utilization` sizing.

  MTP's BF16 draft needs additional memory.
  Automatic profiling after unified-memory loading produced inconsistent budgets across ranks.

  Four short requests can run together.
  The cache budget allows **one full-192K request** at a time.
  Additional long requests enter a queue.
  Context includes both prompt and output tokens.
  Tests do not validate context above 192K or concurrency above four.
- The recipe uses the checkpoint's chat template and GLM reasoning/tool parsers.
  It uses **low reasoning effort by default**.
  Clients can explicitly request high/max effort.
  Set a bounded `max_tokens`.
  Check `finish_reason`.
  A `length` stop without final content indicates an incomplete answer, not a successful task.

## Why a custom image?

The stock image loads this checkpoint.
However, warmup on SM121 fails with `pe_dim must be 64 for fp8_ds_mla`.
GLM-5.3 uses NoPE attention.
Its positional component has zero dimensions.
The plugin uses a native 512-wide cache and plain FP8 KV.
These replace the incompatible packed `fp8_ds_mla` format.

The build adapts the plugin to read `hf_text_config.index_topk`.
The adapted plugin preserves GLM's complete 2,176-wide index buffer.
This buffer includes up to three always-selected recent tokens.
The plugin does not truncate the buffer to 2,048.
It does not drop a selected pool.

The CUDA kernel supports this width directly.
The -1 value masks unused columns.
Patch application fails if the pinned source anchors change.

The recipe does **not enable** the plugin's optional weight-only MoE input-scale override.
NVIDIA already supplies calibrated activation scales.
The build does not modify checkpoint tensors, on-disk model configuration, or the chat template.

The image also includes bounded compatibility patches:

- The patches initialize sparse-index buffers to -1.
  They check bounds during pool expansion.
  They avoid the persistent top-k kernel on SM12x.
  They disable unsupported PDL on SM121.
- The patches preserve vLLM's native DFlash2 implementation.
  They add the GLM target's auxiliary hidden-state interface.
  They reconstruct/contract mHC streams at the requested layer boundaries.
  They account for the draft's sliding-window KV allocations.
- NVIDIA's native MTP layer is **BF16**, not packed NVFP4.
  The pinned checkpoint has 888 BF16 tensors and one F32 tensor in layer 45.
  This layer has no quantization scales.

  The draft-only in-memory config and constructor must use the draft's own quantization configuration.
  They must not inherit the NVFP4 target's configuration.
  `patch_nvidia_mtp.py` pins this adaptation to the verified NVIDIA revision.
  The target's calibrated quantization configuration remains unchanged.
- `patch_openai_compat.py` makes chat completions treat `"tools": []` as omitted tools.
  It clamps `max_tokens`/`max_completion_tokens` to the context remaining after the prompt instead of rejecting oversized requests.

  `GLM_MAX_REASONING_EFFORT` accepts `low`, `high`, or `max`.
  The patch lowers a request's `reasoning_effort` to this cap if necessary.
  This applies to top-level values and values in `chat_template_kwargs`.
  The template renders values other than `low` and `high` as `max`.
  The GLM recipes set `low`.
- `glm_fast_load.py` (MIT, knapcio) replaces vLLM's lazy safetensors iterator when `GLM_FAST_LOAD=1`.
  It uses parallel `preadv` reads into reused pinned slabs.
  It yields the same tensors in the same order.
  `glm_fast_load.pth` enables it per process.
  The four-node NVIDIA recipe sets it.
  This recipe does not.

## Run

1. Install [SparkRun](https://sparkrun.dev/) 0.3.6 or newer.
2. Configure a two-node cluster.
3. Make sure both nodes have enough free disk for the checkpoint and image.
4. Stop conflicting inference workloads only with the operator's approval.
5. Run these commands:

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-nvfp4.yaml
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> --dry-run
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster>
```

SparkRun distributes the image and checkpoint to the worker.
The API defaults to port 8000.
Its default model ID is `GLM-5.3-Flash-NVFP4`.
Use your configured two-node cluster name.

The checkpoint occupies approximately 204 GB on disk per node.
Model loading took about 13–14 minutes per rank.
Each rank allocated 88.67 GiB for the loaded model.

The first custom-image launch took about 25 minutes after the containers started.
This time included multimodal profiling and kernel warmup.
Allow additional time for the initial download and distribution.

Native MTP raises loaded weight allocation to **95.63 GiB per rank**.
Its v2 launches took approximately 17–20 minutes with the checkpoint already cached.

When replacing another model, stop it first.
The repository's `../tools/cleanup-memory.sh` can reclaim retained unified memory on explicitly named nodes before launch.
Do not run memory cleanup against active inference jobs.

### API key

Pass a key at launch to require `Authorization: Bearer <key>` on `/v1/*`:

```bash
umask 077; openssl rand -hex 32 > ~/.config/vllm-api-key   # once
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -e VLLM_API_KEY="$(cat ~/.config/vllm-api-key)"
```

`/health` and `/metrics` remain unauthenticated.
`docker inspect` shows the key on each node.
SparkRun's owner-only job record under `~/.cache/sparkrun/jobs/` also shows the key.
`benchmark.py` and `../glm-5.3-flash-nvfp4-4node/evaluate.py` send the key when their environment sets `VLLM_API_KEY`.
Without `-e VLLM_API_KEY`, the API requires no authentication.

## Decoding and backend comparisons

The target remains NVIDIA's pinned checkpoint in every mode.
Stop the previous job before switching profiles.
These commands are alternatives, not concurrent deployments:

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

Native MTP depths above one repeatedly apply the **same** prediction layer.
More speculative tokens can lower acceptance and throughput.
DFlash2 uses a separate trained draft.
This recipe distributes that draft to both nodes.

Marlin uses NVIDIA's packed weights with **weight-only W4A16 computation**.
It does not use the calibrated W4A4 activation path of `flashinfer_cutlass`.
It is a numerically different backend, not a performance-only flag.
Keep backend and speculation changes separate when investigating output quality.
Passing small Korean/JSON/tool tests does not prove that a checkpoint has no corruption.

This build differs from
[tonyd2wild's reference](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3).
It keeps the NVIDIA ModelOpt target and TP2 **with expert parallelism**.
It retains native 512-wide NoPE attention and the pinned image's own DFlash2 implementation.
It adapts the target/KV integration and SM121 guards.
It does not copy older model implementations or change to RedHatAI/LibertAI weights.
The reference's reported throughput does not provide a directly comparable NVIDIA benchmark.

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

The harness records complete replies, finish reasons, API token counts, latency, and time to first token.
It also records a Prometheus metrics snapshot.
It uses greedy sampling and bounded output.
It checks final answers, structured JSON, Korean, and repetition.
It exercises four concurrent requests.

Performance prompts have a 384-output-token cap.
Only throughput probes permit deliberate `length` termination.

With `--long-context`, each long request must exceed 24K prompt tokens.
It must also produce at least 100 output tokens.
Distinct first blocks prevent prefix sharing from hiding concurrent cache pressure.

`--quality-stress` adds three longer Korean-generation probes.
These check for replacement characters, repetition, missing Korean text, and unfinished answers.

Output tokens/second includes reasoning and final content.
The harness measures this rate over total request wall time, not by counting SSE chunks.
Inspect speculative acceptance metrics alongside throughput.
This is a functional comparison, not a general quality benchmark.

### Current 192K deployment

The API advertises **196,608 tokens**, matching the EXL3 recipe.
The target, DFlash2-7, and image remain unchanged from the earlier 32K deployment.
The 4 GiB per-GPU KV pool and 2,048-token prefill batches also remain unchanged.

- All ten common API checks passed.
  The short-context code probes measured **59.06 / 56.69 output tokens/s**.
- **195,045 prompt tokens + 384 output tokens** completed in **121.46 seconds**.
  This included **115.21 seconds to first token**.
  This throughput probe deliberately reached its output cap.
  No corruption, repetition lock, or engine failure occurred.
- A separate **195,029-token retrieval prompt** correctly returned all three audit codes.
  The prompt placed the codes near the beginning, middle, and end.
  It completed normally in **112.23 seconds**.
  It produced no reasoning tokens.
- Health/model registration, streaming, and automatic JSON tool calling passed after the near-limit decode.
  A 7,532-token prompt and four short concurrent requests also passed.

The cache allocator reports 155 pool blocks and 1.78 maximum-concurrency units at 192K.
Budget for **one fully occupied 192K request**, not two.
The allocator derives its displayed 350,278-token cache-size estimate from concurrency.
That estimate is not the per-request context limit.

Long-prompt prefill takes substantially longer than short-prompt generation.
These smoke tests do not establish general long-context reasoning/retrieval quality.
They do not establish complex vision support or performance above 192K.

### Original 32K-profile measurements

Tests used two 384-output-token code probes on an otherwise idle two-Spark service.
The probes used identical prompts and greedy sampling.
Rates include request overhead.
They count both reasoning and final-answer tokens.
These are **short-context** results from the earlier 32K deployment.

To reproduce that comparison, launch profiles with `-o max_model_len=32768`.
The recipe now defaults to 192K.

| Profile | Output tokens/s, runs 1 / 2 | Functional checks |
| --- | --- | --- |
| Original v1, CUTLASS, no speculation, explicit low effort | 14.41 / 14.45 | Plain JSON returned a Markdown fence |
| v2 control, CUTLASS, no speculation | 14.57 / 14.56 | 10/10 passed |
| v2 control, Marlin W4A16, no speculation | 14.86 / 14.87 | 10/10 passed |
| v2, CUTLASS + native MTP-1 | 24.98 / 25.03 | 10/10 passed |
| v2, CUTLASS + native MTP-2 | 30.80 / 30.43 | 10/10 passed |
| v2, CUTLASS + native MTP-4 | 38.17 / 38.00 | 10/10 passed |
| **v2, CUTLASS + DFlash2-7 (default)** | **59.26 / 58.65** | **10/10 passed** |

DFlash2 throughput reached about **4.0×** the same-image non-speculative control on these code probes.
It reached **1.55×** MTP-4 throughput.
The accepted/drafted token ratio was 80.4% across the suite.

The debugging answer completed in 5.88 seconds with DFlash2.
It took 12.93 seconds without speculation and 7.66 seconds with MTP-4.

DFlash2 is the selected default.
These results do not establish that every workload will improve by the code-probe ratio.

The DFlash2 follow-up passed all 16 checks.
These included three longer Korean generations and five long-context decodes.

A single 30,045-token prompt plus 384 output tokens completed in 28.14 seconds.
Time to first token was 21.74 seconds.

Four distinct 30,050-token prompts each generated 384 tokens.
They completed in 53.12–82.44 seconds, including queuing and prefill.
The cache admitted up to three long requests at once.
The tests found no replacement characters, repetition locks, or engine failures.
DFlash2 loaded 90.22 GiB of weights per rank.

The selected deployment also passed health/model registration, exact arithmetic, streaming, and automatic `get_weather({"city":"Paris"})` tool calling.
It passed a 7,532-token prompt, four short concurrent requests, and a solid-red image-input check.

The target handles images.
The built-in drafter uses text-only draft inputs because it does not support external multimodal embeddings.
Tests do not validate complex vision or video.
Image-heavy requests may have lower acceptance.

Marlin improved these probes by only about 2% over the same-image CUTLASS control.
It also passed three longer Korean-generation probes, automatic JSON tool calling, streaming, and the 7,532-token-prompt smoke test.
This small speed difference does not justify replacing NVIDIA's calibrated W4A4 computation as the default.
Marlin remains an explicit comparison option.

The v2 checks use structured JSON output, unlike the original plain-JSON probe.
Native MTP's accepted/drafted token ratios across each complete suite were 97.0%, 93.8%, and 85.6%.
These correspond to depths 1, 2, and 4 respectively.
Lower acceptance at depth 4 still produced higher throughput on these code prompts.
Do not extrapolate these rates to arbitrary reasoning or agent workloads.

MTP-4 also completed a **30,045-token prompt plus 384 output tokens** in 32.36 seconds.
Time to the first token was 22.36 seconds.
The test found no replacement characters, repetition lock, or engine failure.

First use of a new batch shape can require compilation.
The initial four-request batch took about 5.3 seconds.
The warmed repeat took about 0.56 seconds per request.
These small functional checks do not establish general model quality or an EXL3 speed comparison.

## Verify after deployment

```bash
# With an API key, add: -H "Authorization: Bearer $VLLM_API_KEY" to /v1 requests.
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

The pinned checkpoint always starts an assistant turn with `<think>`.
It supports `reasoning_effort` (`low`, `high`, or default `max`), not `enable_thinking=false`.
Passing `enable_thinking=false` can put reasoning text in the final content because the parser and template disagree.
Leave thinking enabled.
Use the reasoning-effort parameter instead.

1. Expect `42` in the final answer.
2. Check both ranks' logs for an EP-compatible NVFP4 backend.
3. Check for missing weights/scales, gate/up scale mismatches, NoPE sparse-attention errors, and OOMs.

### Recorded deployment checks

Initial non-speculative deployment on a two-Spark cluster
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
The deployment uses `FLASHINFER_CUTLASS` NVFP4 MoE with 144/288 experts per rank.
It uses the plugin's plain FP8 KV path.
Its registration overrides vLLM's `FLASHINFER_MLA_SPARSE_SM120` slot.
Thus, that name still appears in the logs.

## Attention regression tests

Run on an idle Spark after building the image:

```bash
docker run --rm --gpus all --entrypoint bash \
  -v "$PWD:/recipe:ro" glm53-flash-nvfp4-sparkrun-v5 -lc '
    python3 /recipe/test_openai_compat.py &&
    (cd /recipe && python3 test_fast_load.py) &&
    python3 /recipe/test_kv_layout.py &&
    python3 /recipe/test_sparse_mla.py &&
    python3 /opt/glm53-sparse-mla/tests/run_installed.py &&
    python3 /opt/glm53-sparse-mla/tests/run_fp8.py
  '
```

`test_openai_compat.py` checks the installed chat-completion patch.
It covers empty tool arrays, clamped output limits, and the reasoning-effort cap.
It also checks idempotent reapplication and fail-closed drift.

The local regression compares the installed backend and metadata builder against an FP32 reference.
It includes the full kpool tail and fully masked rows.

The upstream suites additionally cover masking, duplicate indices, and extreme logits.
They also cover different token counts for BF16 and FP8 KV.

1. Make sure the cache contains the pinned checkpoints.
2. Test draft configuration and load the real MTP weights without allocating the full target:

```bash
docker run --rm --gpus all --shm-size 1g --entrypoint bash \
  -e HF_HUB_OFFLINE=1 -e VLLM_GLM53_CUDA_SPARSE_MLA=0 \
  -v "$PWD:/recipe:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  glm53-flash-nvfp4-sparkrun-v5 -lc '
    python3 /recipe/test_nvidia_mtp.py && python3 /recipe/test_load_mtp.py
  '
```

This isolated TP1 test disables the NoPE plugin.
The test executes no attention.
The plugin's 64-head TP1 kernel exceeds SM121 shared-memory limits.
These are the only reasons for disabling the plugin here.
The actual TP2 service **must** retain `VLLM_GLM53_CUDA_SPARSE_MLA=1`.

The test covers both Hub IDs and vLLM's resolved snapshot paths.
It checks draft/target quantization separation, picklable config transforms, and real BF16 expert-weight loading.
It is not a substitute for a healthy distributed service and API tests.

## License and attribution

Original recipe, build files, test, and documentation use the recipe's [Unlicense](LICENSE).
Runtime adaptation scripts and source-overlay tests contain Apache-2.0 source excerpts.
We distribute those scripts and tests under [Apache-2.0](LICENSE-APACHE).
Their SPDX headers identify that license.
The configuration follows the deployment guidance in NVIDIA's model card.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream licenses.
This directory does not distribute model weights or container images.
