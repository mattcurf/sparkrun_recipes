# GLM-5.3-Flash NVFP4

A separate recipe for
[`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4).
It does not replace the EXL3/DFlash2 recipe or any Qwen configuration.

**Tested on two DGX Sparks (GB10/SM121), 2026-09-10.** NVIDIA documents this
checkpoint on GB200 with TP4. This recipe uses TP2 with expert parallelism and
a native NoPE attention plugin to run on two Sparks.

## Configuration

- Pinned NVIDIA checkpoint and a custom image built from the digest of NVIDIA's
  recommended `vllm/vllm-openai:glm53-flash-arm64-cu130` image.
- The base image includes Transformers 5.16.1. The build adds a pinned,
  checksum-verified LibertAI native NoPE sparse-attention plugin for SM121.
- ModelOpt NVFP4 quantization is detected from the checkpoint metadata.
- TP2, expert parallelism, allgather/reduce-scatter communication, FP8 KV cache,
  and no speculative decoding.
- Tested settings: 32,768-token context, four sequences, 2,048-token prefill
  batches, 85% GPU memory utilization, and eager execution. Larger context
  limits and higher concurrency are not validated by this recipe.
- Uses the checkpoint's chat template and GLM reasoning/tool parsers. A basic
  image-input smoke test passed; video and complex vision tasks remain untested.

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
model configuration, or chat template are modified.

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

When replacing another model, stop it first. If unified memory remains retained,
the repository's `../tools/cleanup-memory.sh` can reclaim it on explicitly named
nodes before launch. Do not run memory cleanup against active inference jobs.

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

On the `first_pair` two-Spark cluster, with the recipe defaults:

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
  -v "$PWD:/recipe:ro" glm53-flash-nvfp4-sparkrun-v1 -lc '
    python3 /recipe/test_sparse_mla.py &&
    python3 /opt/glm53-sparse-mla/tests/run_installed.py &&
    python3 /opt/glm53-sparse-mla/tests/run_fp8.py
  '
```

The local regression exercises the installed backend and metadata builder,
including the full kpool tail and fully masked rows, against an FP32 reference.
The upstream suites additionally cover masking, duplicate indices, extreme
logits, and different token counts for BF16 and FP8 KV.

## License and attribution

Original recipe, build files, test, and documentation are covered by the
recipe's [Unlicense](LICENSE). The backend adaptation script contains
Apache-2.0 source excerpts and is distributed under [Apache-2.0](LICENSE-APACHE).
The configuration follows the deployment guidance in NVIDIA's model card. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream licenses. Model
weights and container images are not distributed here.
