# GLM-5.3-Flash NVFP4

Use two DGX Sparks with [SparkRun](https://sparkrun.dev/) 0.3.6 or newer.
Use the custom image for native NoPE attention on SM121.

| Setting | Value |
| --- | --- |
| Target | [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4), pinned revision |
| Image | `glm53-flash-nvfp4-sparkrun-v5` |
| Parallelism / MoE | TP2 + expert parallelism; allgather/reduce-scatter; calibrated CUTLASS W4A4 |
| MoE input-scale override | Disabled; retain NVIDIA calibrated activation scales |
| Context, prompt plus output | 196,608 tokens; one full-context request at a time |
| Scheduling | Four sequences; 2,048-token prefill batches; eager execution |
| KV | 4 GiB per GPU, explicit allocation; FP8 target, BF16 DFlash2 |
| Startup utilization | 85%; explicit KV budget overrides automatic cache sizing |
| Draft | DFlash2, seven speculative tokens |
| Reasoning | `low`; `GLM_MAX_REASONING_EFFORT=low` caps higher requests |
| API | Port 8000; `GLM-5.3-Flash-NVFP4` |
| API compatibility | Empty tools treated as omitted; output limits clamp to remaining context |
| Checkpoint disk space | Approximately 204 GB per node, plus image and temporary space |

## Run

Run from this directory with a two-node SparkRun cluster:

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-nvfp4.yaml
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> --dry-run
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster>
```

### API key

Without a key, the API requires no authentication.
Run from this directory:

```bash
umask 077; openssl rand -hex 32 > ~/.config/vllm-api-key   # once
sparkrun run glm-5.3-flash-nvfp4.yaml --cluster <two-node-cluster> \
  -e VLLM_API_KEY="$(cat ~/.config/vllm-api-key)"
```

Use `Authorization: Bearer <key>` on `/v1/*`.
`/health` and `/metrics` remain unauthenticated.
`docker inspect` and owner-only `~/.cache/sparkrun/jobs/` records expose the key.
Export `VLLM_API_KEY` for benchmark and evaluation scripts.

## Alternative profiles

Native MTP requires additional BF16 weight memory.
Marlin uses numerically different W4A16 computation, not calibrated W4A4.

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

## Verify after deployment

Use `reasoning_effort`, not `enable_thinking=false`.
The latter can expose reasoning as final content.

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

Expect final answer `42`.
Check both ranks for an EP-compatible NVFP4 backend, missing weights/scales, scale mismatches, attention errors, and OOMs.
The native plugin uses the log name `FLASHINFER_MLA_SPARSE_SM120`.

## Attention regression tests

Run from this directory on an idle Spark after building:

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

Cache the pinned checkpoints before the isolated MTP weight test:

```bash
docker run --rm --gpus all --shm-size 1g --entrypoint bash \
  -e HF_HUB_OFFLINE=1 -e VLLM_GLM53_CUDA_SPARSE_MLA=0 \
  -v "$PWD:/recipe:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  glm53-flash-nvfp4-sparkrun-v5 -lc '
    python3 /recipe/test_nvidia_mtp.py && python3 /recipe/test_load_mtp.py
  '
```

This TP1 test executes no attention.
The plugin's TP1 kernel exceeds SM121 shared-memory limits.
The TP2 service requires `VLLM_GLM53_CUDA_SPARSE_MLA=1`.

## Results and license

[Results and benchmark commands](RESULTS.md) support DFlash2 as default, not general quality claims.
Context above 192K, concurrency above four, and complex vision/video remain unvalidated.
Image-heavy requests can reduce draft acceptance.

Original recipe, build files, test, and documentation use the [Unlicense](LICENSE).
Runtime adaptations and source-overlay tests retain [Apache-2.0](LICENSE-APACHE) SPDX headers.
See [upstream attribution and licenses](THIRD_PARTY_NOTICES.md), including NVIDIA, LibertAI, knapcio, and tonyd2wild.
This directory distributes no weights or images.
