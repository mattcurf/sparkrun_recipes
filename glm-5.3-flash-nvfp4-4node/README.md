# GLM-5.3-Flash NVFP4 on four DGX Sparks

This directory contains the NVIDIA NVFP4 recipe and shared TP4 evaluation tools and receipts.
The [native FP8 recipe](../glm-5.3-flash-fp8-4node/) has its own directory.
The comparison concerns **model-weight quantization**, not KV-cache formats.
Both profiles have a **262,144-token (256K) total context window**, including prompt and output.

**Tests covered both profiles on four Sparks on 2026-09-11.**
**Use NVIDIA NVFP4 as the recommended profile.**
See [measured results, raw receipts, and limitations](RESULTS.md).
These include successful ~256K-input retrieval/decode.
They also describe the FP8 cache-allocation failure that required its smaller pool.

| Profile | Weights | MoE computation |
| --- | --- | --- |
| [NVFP4](glm-5.3-flash-nvfp4-tp4.yaml) | `nvidia/GLM-5.3-Flash-NVFP4` | Calibrated CUTLASS W4A4 |
| [FP8](../glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml) | `zai-org/GLM-5.3-Flash` | Native block-scaled Triton W8A8 |

Both retain TP4 plus expert parallelism and allgather/reduce-scatter.
Both use native NoPE sparse attention, FP8 target KV, and BF16 draft KV.
Both retain DFlash2, chunked prefill, GLM tool/reasoning parsers, and low reasoning effort.
`GLM_MAX_REASONING_EFFORT=low` also lowers higher per-request efforts to `low`.
This includes an agent's `high` effort.

`num_speculative_tokens_per_batch_size` controls the DFlash2 draft schedule:

- DFlash2 drafts seven tokens for one running request.
- It drafts five tokens for two running requests.
- It drafts three tokens for three or more running requests.

Both use eight sequences and 2,048-token prefill batches.
Both use CUDA graphs for the target and drafter.
Both use one OpenMP thread to avoid CPU spin-wait contention.
SparkRun's per-model `/cache/runtime` mount stores Triton, TileLang, and FlashInfer JIT caches.
NVFP4 also sets `GLM_FAST_LOAD=1` for the image's fast safetensors loader.

NVFP4 reserves **12 GiB KV per GPU**.
FP8 uses **8 GiB** because its larger weights leave less allocation headroom.
These are different cache capacities at the same per-request context limit, not different KV precision.
Additional long requests may queue when the pool is full.

The explicit KV budget overrides automatic utilization-based sizing.
The 80% utilization setting still controls the startup free-memory check.
The 85% setting rejected the four-node setup after distributed initialization.

For the eager control, override `-o execution_flags=--enforce-eager`.
The recorded initial NVFP4 eager baseline also used four OpenMP threads.
Thus, its comparison with the tuned profile is not a graph-only A/B.

## Build and run

The recipes reuse the pinned runtime integration from the sibling NVIDIA directory instead of duplicating its patches.

1. Stop conflicting workloads only with the operator's approval.
2. Run these commands from the repository root:

```bash
./glm-5.3-flash-nvfp4-2node/build-image.sh
sparkrun recipe validate glm-5.3-flash-nvfp4-4node/glm-5.3-flash-nvfp4-tp4.yaml
sparkrun run glm-5.3-flash-nvfp4-4node/glm-5.3-flash-nvfp4-tp4.yaml --cluster <four-node-cluster>

# Alternative weights; stop the previous job before starting this profile.
sparkrun recipe validate glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

The validated four-node setup uses switched RoCE networking.
Use SparkRun's network detection rather than copying machine-specific interface names.
All ranks must use the same Docker image ID, not just the same tag.
Allow substantial time for first-time image/model distribution and warmup.

NVIDIA's pinned checkpoint occupies approximately 191 GiB on disk per node.
Z.ai's native FP8 checkpoint occupies approximately 306 GiB.
Both can coexist in the Hugging Face cache.
Budget separately for images, download temporary space, and Docker's image-export temporary files.
A full root filesystem can prevent `docker save` even when the model-cache filesystem has free space.

To require an API key on `/v1/*`, add `-e VLLM_API_KEY="$(cat ~/.config/vllm-api-key)"` to `sparkrun run`.
See the [sibling API key section](../glm-5.3-flash-nvfp4-2node/README.md#api-key).
Export the same `VLLM_API_KEY` before running the evaluation scripts.

The API listens on port 8000.
NVFP4 uses served name `GLM-5.3-Flash-NVFP4-TP4`.
FP8 uses served name `GLM-5.3-Flash-FP8-TP4`.
Always bound generation with `max_tokens`.
Inspect `finish_reason`.
A length stop is not a completed answer.

## Matched evaluation

1. Start each profile separately.
2. Run against an idle endpoint.
3. Change only `MODEL` and the output filenames between weight variants:

```bash
MODEL=GLM-5.3-Flash-NVFP4-TP4
python3 glm-5.3-flash-nvfp4-2node/benchmark.py --model "$MODEL" \
  --reasoning low --long-context --quality-stress --output /tmp/quality.json
python3 glm-5.3-flash-nvfp4-4node/evaluate.py --model "$MODEL" \
  --near-limit --output /tmp/evaluation.json
# A long decode near the 256K limit, in addition to retrieval:
python3 glm-5.3-flash-nvfp4-2node/benchmark.py --model "$MODEL" \
  --reasoning low --rounds 0 --long-context --context-lines 17100 \
  --long-concurrency 0 --request-timeout 1800 --output /tmp/256k-decode.json
# Repeat the short evaluation after shapes are warm; use this for warm rates.
python3 glm-5.3-flash-nvfp4-4node/evaluate.py --model "$MODEL" --output /tmp/warm.json
```

The evaluation records full replies, API token counts, TTFT, and total request rates.
It also records approximate post-first-token decode rates and Prometheus metrics.
Completed code/prose probes must stop normally.
The sibling harness's capped throughput probes deliberately allow length stops.

The evaluation measures four-stream aggregate throughput over the entire concurrent batch.
It does not sum per-stream rates.

Near-limit retrieval checks three audit codes at beginning/middle/end.
It requires at least 250,000 actual prompt tokens.

These are functional probes and workload-specific measurements, not a general model-quality benchmark.
The evaluation does not execute generated code.
More weight precision does not automatically imply better quality or speed on these workloads.

CPU-only checks (PyYAML required):

```bash
python3 -m unittest discover -s glm-5.3-flash-nvfp4-4node -v
uvx ruff check glm-5.3-flash-nvfp4-4node/*.py
uvx ruff format --check glm-5.3-flash-nvfp4-4node/*.py
```

## License and references

Original configurations, evaluation code, and documentation use the
[Unlicense](LICENSE). Downloaded weights and container components keep their
own licenses. The shared runtime's Apache-2.0 adaptations and upstream notices
are in [the sibling directory](../glm-5.3-flash-nvfp4-2node/THIRD_PARTY_NOTICES.md).
This repository does not distribute weights or images.

Reference deployments informing the experiment (not measurements of our rig):

- [Tony's TP4 NVFP4 deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
- [Alex Ellis's TP4 NVFP4 deployment](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
- [Jacopo Nardiello's native FP8 TP4 deployment](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
