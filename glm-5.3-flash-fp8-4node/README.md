# GLM-5.3-Flash FP8 on four DGX Sparks

This recipe uses native `zai-org/GLM-5.3-Flash` FP8 weights with TP4 + expert parallelism.
It uses DFlash2 with seven speculative tokens.
It provides **262,144 tokens (256K)** of total context.
The recipe uses Triton FP8 MoE, FP8 target KV, and BF16 draft KV.
It uses CUDA graphs and one OpenMP thread.
It supports four sequences with an 8,192-token prefill batch.

The fixed KV budget is **8 GiB per rank**.
A 12 GiB budget caused an allocation failure during the recorded FP8 evaluation.
The successful 8 GiB retry retained 256K.
See the [shared results and limitations](../glm-5.3-flash-nvfp4-4node/RESULTS.md).
NVIDIA NVFP4 ran faster in those tests.
It also left more memory available.
FP8 is a tested alternative.
These results do not establish better quality or optimal FP8 tuning.

## Build and run

The NVIDIA recipes use the same pinned runtime.

1. Get the operator's approval before you stop conflicting workloads.
2. Run these commands from the repository root:

```bash
./glm-5.3-flash-nvfp4-2node/build-image.sh
sparkrun recipe validate glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

The API listens on port 8000.
It uses served name `GLM-5.3-Flash-FP8-TP4`.
Allow time for distribution, loading, and warmup.
The pinned FP8 checkpoint occupies approximately 306 GiB on disk per node.
All ranks must use the same container image.

## Evaluation and licensing

The [shared TP4 evaluation instructions](../glm-5.3-flash-nvfp4-4node/README.md#matched-evaluation)
apply with `MODEL=GLM-5.3-Flash-FP8-TP4`.
The NVFP4 four-node directory contains the shared evaluator, cross-profile tests, and original result receipts.
This arrangement avoids duplicate code and historical evidence.

Original recipe and documentation use the [Unlicense](LICENSE).
Weights and container components retain their own licenses.
See the
[shared runtime notices](../glm-5.3-flash-nvfp4-2node/THIRD_PARTY_NOTICES.md).
