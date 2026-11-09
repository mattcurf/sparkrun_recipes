# Qwen3.8-Flash-Next FP8

This recipe serves the original
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) quantization on two DGX Sparks.
It uses vLLM tensor parallelism.
It uses three-token MTP speculative decoding.
The recipe uses the model's native 262,144-token context.

## Run

Install [SparkRun](https://sparkrun.dev/).
Configure SparkRun.
Run these commands:

```bash
sparkrun recipe validate qwen3.8-flash-next.yaml
sparkrun run qwen3.8-flash-next.yaml --cluster <two-node-cluster>
```

The recipe pins the vLLM image by digest.
The image requires no local build.
The image requires no runtime patch.
SparkRun distributes the image to the worker.
SparkRun distributes the model to the worker.
The OpenAI-compatible API listens on port 8000 by default.
You may need a Hugging Face login to download the model.

## Verified configuration

- Tensor parallelism: 2
- MTP draft tokens: 3
- Maximum context: 262,144
- Maximum batched tokens: 8,192
- Maximum sequences: 8
- KV cache: auto
- Load format: InstantTensor
- Execution: eager mode

## Attribution and licensing

- The Qwen team publishes
  [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8).
  The model and weights retain the terms in the model card.
  This repository does not distribute the model or weights.
- [vLLM](https://github.com/vllm-project/vllm) and
  [SparkRun](https://github.com/spark-arena/sparkrun) are dependencies.
  These dependencies use the Apache-2.0 license.
  This repository does not distribute these dependencies.

We dedicate the recipe and documentation to the public domain under the Unlicense.
See `LICENSE`.
See `THIRD_PARTY_NOTICES.md`.
We provide everything **as is**, without warranty.
