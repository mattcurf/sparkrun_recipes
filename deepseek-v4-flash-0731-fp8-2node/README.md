# DeepSeek V4 Flash 0731 with DSpark

Run the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release on two DGX Sparks with these settings:

| Setting | Value |
| --- | --- |
| Tensor parallelism | TP2 |
| KV cache | FP8 |
| DSpark speculative decoding | Three tokens |
| Maximum context | 1,048,576 tokens |

## Run

1. Install [SparkRun](https://sparkrun.dev/).
2. Configure SparkRun.
3. Run these commands:

```bash
sparkrun recipe validate deepseek-v4-flash-0731.yaml
sparkrun run deepseek-v4-flash-0731.yaml --cluster <two-node-cluster>
```

Both nodes need network access for the pinned DSpark overlay download on first launch.
SparkRun distributes the model and exposes an OpenAI-compatible API on port 8000 by default.

## Attribution and licensing

This SparkRun port adapts Tony Deangelo's MIT-licensed
[`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark).
That project includes Apache-2.0 vLLM/DSpark-derived runtime work.
It credits Keys/drowzeys, Roady001, Fable, Wpnx330, Rafael Caricio, Fraser Price, and MiaAI-Lab.
See its
[`CREDITS.md`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/CREDITS.md)
for full attribution.

The recipe itself uses the Unlicense public-domain dedication.
The runtime overlay, weights, container image, vLLM, and SparkRun retain their upstream terms.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
You receive everything **as is**, without warranty.
