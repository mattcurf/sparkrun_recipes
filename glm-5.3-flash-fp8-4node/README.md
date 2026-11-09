# GLM-5.3-Flash FP8 on four DGX Sparks

Use NVFP4 for the [measured workloads](../glm-5.3-flash-nvfp4-4node/RESULTS.md).
FP8 remains a tested alternative.
These tests do not establish better FP8 quality or optimal tuning.

| Setting | Value |
| --- | --- |
| Weights | Native `zai-org/GLM-5.3-Flash` FP8 |
| Context, prompt plus output | 262,144 tokens (256K) |
| MoE / KV | Triton W8A8; FP8 target KV; BF16 draft KV |
| KV budget | 8 GiB per rank; budget for three fully occupied contexts |
| Scheduling | Eight sequences; 2,048-token prefill batches |
| DFlash2 drafts | Seven for one request; five for two; three for three or more |
| API | Port 8000; `GLM-5.3-Flash-FP8-TP4` |
| Checkpoint disk | Approximately 306 GiB per node |

See [shared TP4 settings](../glm-5.3-flash-nvfp4-4node/README.md).

## Build and run

Run from the repository root:

```bash
./glm-5.3-flash-nvfp4-2node/build-image.sh
sparkrun recipe validate glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml
sparkrun run glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml --cluster <four-node-cluster>
```

## Evaluation and licensing

The [shared TP4 evaluation instructions](../glm-5.3-flash-nvfp4-4node/README.md#matched-evaluation)
apply with `MODEL=GLM-5.3-Flash-FP8-TP4`.
That directory contains the shared evaluator, cross-profile tests, and result receipts.

Original recipe and documentation use the [Unlicense](LICENSE).
Weights and container components retain their own licenses.
See the
[shared runtime notices](../glm-5.3-flash-nvfp4-2node/THIRD_PARTY_NOTICES.md).
