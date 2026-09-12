# Qwen3.8-Flash-Next FP8

Serves the original
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
quant on two DGX Sparks with vLLM tensor parallelism and three-token MTP
speculative decoding. The recipe uses the model's native 262,144-token context.

## Run

Install and configure [SparkRun](https://sparkrun.dev/), then run:

```bash
sparkrun recipe validate qwen3.8-flash-next.yaml
sparkrun run qwen3.8-flash-next.yaml --cluster <two-node-cluster>
```

The vLLM image is pinned by digest and requires no local image build or runtime
patch. SparkRun distributes the image and model to the worker. The
OpenAI-compatible API listens on port 8000 by default. A Hugging Face login may
be required to download the model.

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

- [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
  is published by the Qwen team. Its model and weights retain the terms stated
  in the model card and are not distributed here.
- [vLLM](https://github.com/vllm-project/vllm) and
  [SparkRun](https://github.com/spark-arena/sparkrun) are Apache-2.0 licensed
  dependencies and are not distributed here.

The recipe and documentation are dedicated to the public domain under the
Unlicense. See `LICENSE` and `THIRD_PARTY_NOTICES.md`. Everything is provided
**as is**, without warranty.
