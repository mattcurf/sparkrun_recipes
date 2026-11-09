# Qwen3.8-Flash-Next FP8

Serve the Qwen team's [`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) on two DGX Sparks.

## Run

Complete [SparkRun setup](../README.md#install-sparkrun) before launch.

```bash
sparkrun recipe validate qwen3.8-flash-next.yaml
sparkrun run qwen3.8-flash-next.yaml --cluster <two-node-cluster>
```

SparkRun distributes the pinned image and model.
The API listens on port 8000 by default.

## Settings

| Setting | Value |
| --- | --- |
| Tensor parallelism | 2 |
| MTP draft tokens | 3 |
| Maximum context | 262,144 |
| Maximum batched tokens | 8,192 |
| Maximum sequences | 8 |
| KV cache | auto |
| Load format | InstantTensor |
| Execution | eager mode |

## Attribution and licensing

[vLLM](https://github.com/vllm-project/vllm) and [SparkRun](https://github.com/spark-arena/sparkrun) use Apache-2.0.
The recipe and docs use the [Unlicense](LICENSE): public domain, without warranty.
See [third-party notices](THIRD_PARTY_NOTICES.md) for model terms and distribution exclusions.
