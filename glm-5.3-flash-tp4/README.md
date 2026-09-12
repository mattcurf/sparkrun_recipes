# GLM-5.3-Flash on four DGX Sparks

Separate TP4 recipes comparing **model-weight quantization**, not KV-cache
formats. Existing two-node recipes are unchanged. Both profiles have a
**262,144-token (256K) total context window**, including prompt and output.

**Both profiles tested on four Sparks, 2026-09-11. NVIDIA NVFP4 is recommended.**
See [measured results, raw receipts, and limitations](RESULTS.md), including
successful ~256K-input retrieval/decode and the FP8 cache-allocation failure
that required its smaller pool.

| Profile | Weights | MoE computation |
| --- | --- | --- |
| `glm-5.3-flash-nvfp4-tp4.yaml` | `nvidia/GLM-5.3-Flash-NVFP4` | Calibrated CUTLASS W4A4 |
| `glm-5.3-flash-fp8-tp4.yaml` | `zai-org/GLM-5.3-Flash` | Native block-scaled Triton W8A8 |

Both retain TP4 plus expert parallelism, allgather/reduce-scatter, native NoPE
sparse attention, FP8 target KV, BF16 draft KV, DFlash2 with seven speculative
tokens, chunked prefill, GLM tool/reasoning parsers, and low reasoning effort.
Both use four sequences, 8,192-token prefill batches, CUDA graphs for the target
and drafter, and one OpenMP thread to avoid CPU spin-wait contention. NVFP4
reserves **12 GiB KV per GPU**; FP8 uses **8 GiB** because its larger weights
leave less allocation headroom. These are different cache capacities at the
same per-request context limit, not different KV precision. Additional long
requests may queue when the pool is full.

The explicit KV budget overrides automatic utilization-based sizing. The 80%
utilization setting still controls the startup free-memory check; 85% rejected
the four-node setup after distributed initialization. For the eager control,
override `-o execution_flags=--enforce-eager`; the recorded initial NVFP4 eager
baseline also used four OpenMP threads, so its comparison with the tuned
profile is not a graph-only A/B.

## Build and run

The recipes reuse the pinned runtime integration from the sibling NVIDIA
directory instead of duplicating its patches. Run from the repository root:

```bash
./glm-5.3-flash-nvfp4/build-image.sh
sparkrun recipe validate glm-5.3-flash-tp4/glm-5.3-flash-nvfp4-tp4.yaml
sparkrun run glm-5.3-flash-tp4/glm-5.3-flash-nvfp4-tp4.yaml --cluster <four-node-cluster>

# Alternative weights; stop the previous job before starting this profile.
sparkrun recipe validate glm-5.3-flash-tp4/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-tp4/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

Stop conflicting workloads only with the operator's approval. The development
cluster is named `gb`; its four nodes share switched RoCE networking. Use
SparkRun's network detection rather than copying machine-specific interface
names. All ranks must use the same Docker image ID, not just the same tag.
Allow substantial time for first-time image/model distribution and warmup.

NVIDIA's pinned checkpoint occupies approximately 191 GiB on disk per node;
Z.ai's native FP8 checkpoint occupies approximately 306 GiB. Both can coexist
in the Hugging Face cache, but budget for images, download temporary space,
and Docker's image-export temporary files separately. A full root filesystem
can prevent `docker save` even when the model-cache filesystem has free space.

The API listens on port 8000. Served names are `GLM-5.3-Flash-NVFP4-TP4` and
`GLM-5.3-Flash-FP8-TP4` respectively. Always bound generation with `max_tokens`
and inspect `finish_reason`; a length stop is not a completed answer.

## Matched evaluation

Run against an idle endpoint after each profile starts, changing only `MODEL`
and the output filenames between weight variants:

```bash
MODEL=GLM-5.3-Flash-NVFP4-TP4
python3 glm-5.3-flash-nvfp4/benchmark.py --model "$MODEL" \
  --reasoning low --long-context --quality-stress --output /tmp/quality.json
python3 glm-5.3-flash-tp4/evaluate.py --model "$MODEL" \
  --near-limit --output /tmp/evaluation.json
# A long decode near the 256K limit, in addition to retrieval:
python3 glm-5.3-flash-nvfp4/benchmark.py --model "$MODEL" \
  --reasoning low --rounds 0 --long-context --context-lines 17100 \
  --long-concurrency 0 --request-timeout 1800 --output /tmp/256k-decode.json
# Repeat the short evaluation after shapes are warm; use this for warm rates.
python3 glm-5.3-flash-tp4/evaluate.py --model "$MODEL" --output /tmp/warm.json
```

The evaluation records full replies, API token counts, TTFT, total request
rates, approximate post-first-token decode rates, and Prometheus metrics.
Completed code/prose probes must stop normally. The sibling harness's capped
throughput probes deliberately allow length stops. Four-stream aggregate
throughput is measured over the entire concurrent batch, not summed per-stream
rates. Near-limit retrieval checks three audit codes at beginning/middle/end
and requires at least 250,000 actual prompt tokens.

These are functional probes and workload-specific measurements, not a general
model-quality benchmark. Generated code is not executed. More weight precision
does not automatically imply better quality or speed on these workloads.

CPU-only checks (PyYAML required):

```bash
python3 -m unittest discover -s glm-5.3-flash-tp4 -v
uvx ruff check glm-5.3-flash-tp4/*.py
uvx ruff format --check glm-5.3-flash-tp4/*.py
```

## License and references

Original configurations, evaluation code, and documentation use the
[Unlicense](LICENSE). Downloaded weights and container components keep their
own licenses. The shared runtime's Apache-2.0 adaptations and upstream notices
are in [the sibling directory](../glm-5.3-flash-nvfp4/THIRD_PARTY_NOTICES.md).
No weights or images are distributed in this repository.

Reference deployments informing the experiment (not measurements of our rig):

- [Tony's TP4 NVFP4 deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark)
- [Alex Ellis's TP4 NVFP4 deployment](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless)
- [Jacopo Nardiello's native FP8 TP4 deployment](https://github.com/jnardiello/GLM-5.3-Flash-FP8-4-DGX-Spark-Switchless)
