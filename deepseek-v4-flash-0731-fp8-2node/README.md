# DeepSeek V4 Flash 0731 with DSpark

This recipe serves the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release on two DGX Sparks.
It uses TP2.
It uses an FP8 KV cache.
It uses three-token DSpark speculative decoding.
The default maximum context is 1,048,576 tokens.

## Run

1. Install [SparkRun](https://sparkrun.dev/).
2. Configure SparkRun.
3. Run these commands:

```bash
sparkrun recipe validate deepseek-v4-flash-0731.yaml
sparkrun run deepseek-v4-flash-0731.yaml --cluster <two-node-cluster>
```

The recipe uses a pinned container image.
During container preparation, it downloads a pinned revision of the DSpark runtime overlay.
Both cluster nodes need network access on first launch.
SparkRun distributes the model.
By default, SparkRun exposes an OpenAI-compatible API on port 8000.

## Attribution and licensing

This SparkRun port adapts Tony Deangelo's MIT-licensed
[`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark).
That project includes Apache-2.0 vLLM/DSpark-derived runtime work.
It credits Keys/drowzeys, Roady001, Fable, Wpnx330, Rafael Caricio, Fraser Price, and MiaAI-Lab.
See its
[`CREDITS.md`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/CREDITS.md)
for the complete project history.

The recipe itself uses the Unlicense public-domain dedication.
The runtime overlay retains its upstream terms.
The model weights retain their upstream terms.
The container image retains its upstream terms.

vLLM retains its upstream terms.
SparkRun retains its upstream terms.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
You receive everything **as is**, without warranty.
