# GLM-5.3-Flash FP8 on four DGX Sparks

Native `zai-org/GLM-5.3-Flash` FP8 weights, TP4 + expert parallelism, DFlash2
with seven speculative tokens, and **262,144 tokens (256K)** of total context.
The recipe uses Triton FP8 MoE, FP8 target KV, BF16 draft KV, CUDA graphs,
one OpenMP thread, four sequences, and an 8,192-token prefill batch.

The fixed KV budget is **8 GiB per rank**. A 12 GiB budget failed allocation
during the recorded FP8 evaluation; the successful 8 GiB retry retained 256K.
See the [shared results and limitations](../glm-5.3-flash-nvfp4-4node/RESULTS.md).
NVIDIA NVFP4 was faster and left more memory available in those tests; FP8 is
a tested alternative, not a claim of better quality or optimal FP8 tuning.

## Build and run

The pinned runtime is shared with the NVIDIA recipes. From the repository root:

```bash
./glm-5.3-flash-nvfp4-2node/build-image.sh
sparkrun recipe validate glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

Stop conflicting workloads with the operator's approval first. The API listens
on port 8000 with served name `GLM-5.3-Flash-FP8-TP4`. Allow time for distribution,
loading and warmup; the pinned FP8 checkpoint is approximately 306 GiB on disk
per node. All ranks must use the same container image.

## Evaluation and licensing

The [shared TP4 evaluation instructions](../glm-5.3-flash-nvfp4-4node/README.md#matched-evaluation)
apply with `MODEL=GLM-5.3-Flash-FP8-TP4`. The shared evaluator, cross-profile
tests and original result receipts remain in the NVFP4 four-node directory
to avoid duplicating code or historical evidence.

Original recipe and documentation use the [Unlicense](LICENSE). Weights and
container components retain their own licenses; see the
[shared runtime notices](../glm-5.3-flash-nvfp4-2node/THIRD_PARTY_NOTICES.md).
