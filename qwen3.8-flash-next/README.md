# Qwen3.8-Flash-Next NVFP4

Serves [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
on two DGX Sparks with vLLM tensor parallelism and expert parallelism. The
recipe uses the model's native 262,144-token context and does not enable MTP.

## Build and run

Install and configure [SparkRun](https://sparkrun.dev/), then run:

```bash
./build-image.sh
sparkrun recipe validate qwen3.8-flash-next.yaml
sparkrun run qwen3.8-flash-next.yaml --cluster <two-node-cluster>
```

SparkRun distributes the locally built image and model to the worker. The
OpenAI-compatible API listens on port 8000 by default. A Hugging Face login may
be required to download the model.

## Why the custom image?

The NVIDIA checkpoint stores its PLE embeddings in FP8 while the model body is
a mixed ModelOpt checkpoint. The small, fail-closed build patch lets vLLM
select its existing FP8 PLE implementation from `ple_embedding_dtype`.

## Attribution and licensing

- [vLLM](https://github.com/vllm-project/vllm) is Apache-2.0 licensed. The PLE
  patch targets and includes small excerpts from its Qwen3.8 integration and is
  distributed under Apache-2.0.
- PLE handling was informed by [MiaAI-Lab's Qwen3.8 dual-DGX-Spark work](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
  and the upstream vLLM work it credits. No Mia infrastructure scripts are
  included here.
- The NVIDIA model, downloaded weights, base image, and SparkRun are not part
  of this repository and retain their respective upstream terms.

The original recipe, Dockerfile, build script, and documentation in this
directory are dedicated to the public domain under the Unlicense. See
`LICENSE` and `THIRD_PARTY_NOTICES.md`. Everything is provided **as is**, without
warranty.
