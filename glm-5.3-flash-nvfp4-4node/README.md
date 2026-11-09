# GLM-5.3-Flash NVFP4 on four DGX Sparks

Use NVIDIA NVFP4 for these workloads; [measurements and limits](RESULTS.md) support this recommendation.
The [FP8 alternative](../glm-5.3-flash-fp8-4node/) compares weight quantization, not KV precision.

| Shared TP4 setting | Value |
| --- | --- |
| Image | `glm53-flash-nvfp4-sparkrun-v5` |
| Context, prompt plus output | 262,144 tokens (256K) |
| Parallelism | TP4 + expert parallelism; allgather/reduce-scatter |
| Attention / KV | Native NoPE sparse attention; FP8 target KV; BF16 draft KV |
| DFlash2 drafts | Seven for one request; five for two; three for three or more |
| Scheduling | Eight sequences; 2,048-token prefill batches; chunked prefill; target/draft CUDA graphs |
| Reasoning / tools | GLM parsers; `GLM_MAX_REASONING_EFFORT=low` caps higher requests |
| CPU / caches | One OpenMP thread; JIT caches under `/cache/runtime` |
| Startup utilization | 80%; explicit KV budgets override automatic cache sizing |
| API | Port 8000 |

| Profile | NVFP4 | FP8 |
| --- | --- | --- |
| Weights | `nvidia/GLM-5.3-Flash-NVFP4` | `zai-org/GLM-5.3-Flash` |
| MoE | Calibrated CUTLASS W4A4 | Triton block-scaled W8A8 |
| KV per GPU | 12 GiB | 8 GiB |
| Checkpoint disk per node | ~191 GiB | ~306 GiB |
| Served name | `GLM-5.3-Flash-NVFP4-TP4` | `GLM-5.3-Flash-FP8-TP4` |
| Fast loader | `GLM_FAST_LOAD=1` | Disabled |

## Build and run

SparkRun detects the switched RoCE interfaces.
Docker exports require root-filesystem space in addition to model-cache storage.
Run from the repository root:

```bash
./glm-5.3-flash-nvfp4-2node/build-image.sh
sparkrun recipe validate glm-5.3-flash-nvfp4-4node/glm-5.3-flash-nvfp4-tp4.yaml
sparkrun run glm-5.3-flash-nvfp4-4node/glm-5.3-flash-nvfp4-tp4.yaml --cluster <four-node-cluster>

# Alternative weights:
sparkrun recipe validate glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

Additional long requests can queue when KV fills.
Budget for three fully occupied FP8 contexts, not four.
Four simultaneous full-256K requests remain unvalidated.

For authentication, add `-e VLLM_API_KEY="$(cat ~/.config/vllm-api-key)"` to launch.
See [authentication and key visibility](../glm-5.3-flash-nvfp4-2node/README.md#api-key).
Export the same `VLLM_API_KEY` for evaluation.

## Matched evaluation

1. Start each profile separately on an idle endpoint.
2. Run from the repository root.
3. Change only `MODEL` and output filenames between variants:

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

Use `MODEL=GLM-5.3-Flash-FP8-TP4` for FP8.
These functional probes do not execute generated code or establish general quality.
See [measurement definitions](RESULTS.md#warm-short-context-results).

CPU-only checks require PyYAML:

```bash
python3 -m unittest discover -s glm-5.3-flash-nvfp4-4node -v
uvx ruff check glm-5.3-flash-nvfp4-4node/*.py
uvx ruff format --check glm-5.3-flash-nvfp4-4node/*.py
```

## License and references

Original configurations, evaluation code, and documentation use the [Unlicense](LICENSE).
Weights and container components retain upstream licenses.
See [shared runtime notices](../glm-5.3-flash-nvfp4-2node/THIRD_PARTY_NOTICES.md) for Apache-2.0 adaptations.
This repository distributes no weights or images.

Deployment references, not measurements of this cluster:

- [Tony's TP4 NVFP4 deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
- [Alex Ellis's TP4 NVFP4 deployment](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
- [Jacopo Nardiello's native FP8 TP4 deployment](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
