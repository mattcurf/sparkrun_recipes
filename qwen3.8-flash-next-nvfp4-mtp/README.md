# Qwen3.8-Flash-Next NVFP4 with MTP3

Serves [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
on two DGX Sparks with vLLM tensor parallelism, expert parallelism, and three
multi-token-prediction draft tokens. The recipe uses the native 262,144-token
context.

## Build and run

Install and configure [SparkRun](https://sparkrun.dev/), then run:

```bash
./build-image.sh
sparkrun recipe validate qwen3.8-flash-next-nvfp4-mtp.yaml
sparkrun run qwen3.8-flash-next-nvfp4-mtp.yaml --cluster <two-node-cluster>
```

SparkRun distributes the locally built image and model to the worker. The
OpenAI-compatible API listens on port 8000 by default. A Hugging Face login may
be required to download the model.

## Why the custom image?

The pinned NVIDIA checkpoint needs three compatibility adaptations in the
pinned vLLM base image:

1. Select FP8 for its PLE embeddings from `ple_embedding_dtype`.
2. Resolve vLLM's absolute MTP layer 48 against checkpoint-relative layer 0.
3. Dispatch the MTP routed experts' `FP8_BLOCK_SCALES` metadata to vLLM's
   existing block-quantized `Fp8MoEMethod`, using the checkpoint's group size.

Patch application is fail-closed: the image build stops if the expected pinned
vLLM source anchors change.

## Attribution and licensing

- [vLLM](https://github.com/vllm-project/vllm) is Apache-2.0 licensed. The
  patches target and include small excerpts from its quantization integration
  and are distributed under Apache-2.0.
- The checkpoint diagnosis and ModelOpt integration approach were informed by
  [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
  by Mia's AI Lab under AGPL-3.0-or-later. This standalone implementation uses
  an image-time ModelOpt alias instead of Mia's runtime checkpoint overlays;
  none of Mia's infrastructure scripts are included.
- The NVIDIA model, downloaded weights, base image, and SparkRun are not part
  of this repository and retain their respective upstream terms.

The original recipe, Dockerfile, build script, and documentation in this
directory are dedicated to the public domain under the Unlicense. See
`LICENSE` and `THIRD_PARTY_NOTICES.md`. Everything is provided **as is**, without
warranty.
