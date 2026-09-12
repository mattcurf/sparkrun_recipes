# DeepSeek V4 Flash 0731 with DSpark

Serves the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
release on two DGX Sparks with TP2, FP8 KV cache, and three-token DSpark
speculative decoding. The default maximum context is 1,048,576 tokens.

## Run

Install and configure [SparkRun](https://sparkrun.dev/), then run:

```bash
sparkrun recipe validate deepseek-v4-flash-0731.yaml
sparkrun run deepseek-v4-flash-0731.yaml --cluster <two-node-cluster>
```

The recipe uses a pinned container image and, during container preparation,
downloads a pinned revision of the DSpark runtime overlay. Both cluster nodes
therefore need network access on first launch. SparkRun distributes the model
and exposes an OpenAI-compatible API on port 8000 by default.

## Attribution and licensing

This SparkRun port is based on Tony Deangelo's MIT-licensed
[`tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark).
That project incorporates Apache-2.0 vLLM/DSpark-derived runtime work and
credits Keys/drowzeys, Roady001, Fable, Wpnx330, Rafael Caricio, Fraser Price,
and MiaAI-Lab. See its
[`CREDITS.md`](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark/blob/main/CREDITS.md)
for the complete lineage.

The recipe itself is dedicated to the public domain under the Unlicense. The
overlay downloaded at runtime, the model weights, container image, vLLM, and
SparkRun retain their respective upstream terms. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`. Everything is provided **as is**, without warranty.
