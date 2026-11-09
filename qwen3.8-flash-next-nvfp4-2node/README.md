# Qwen3.8-Flash-Next NVFP4 with MTP3

This recipe serves [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) on two DGX Sparks.
It uses vLLM tensor parallelism.
It uses expert parallelism.
It uses three multi-token-prediction draft tokens.
The recipe uses the native 262,144-token context.

## Build and run

Install [SparkRun](https://sparkrun.dev/).
Configure SparkRun.
Run these commands:

```bash
./build-image.sh
sparkrun recipe validate qwen3.8-flash-next-nvfp4-mtp.yaml
sparkrun run qwen3.8-flash-next-nvfp4-mtp.yaml --cluster <two-node-cluster>
```

SparkRun distributes the locally built image to the worker.
SparkRun distributes the model to the worker.
The OpenAI-compatible API listens on port 8000 by default.
You may need a Hugging Face login to download the model.

## Why the custom image?

The pinned NVIDIA checkpoint needs three compatibility changes in the pinned vLLM base image:

1. Select FP8 for its PLE embeddings from `ple_embedding_dtype`.
2. Resolve vLLM's absolute MTP layer 48 against checkpoint-relative layer 0.
3. Dispatch the MTP routed experts' `FP8_BLOCK_SCALES` metadata to vLLM's existing block-quantized `Fp8MoEMethod`.
   Use the checkpoint's group size.

The image build applies patches in fail-closed mode.
The build stops if the expected pinned vLLM source anchors change.

## Attribution and licensing

- [vLLM](https://github.com/vllm-project/vllm) uses the Apache-2.0 license.
  The patches target its quantization integration.
  They include small excerpts from that integration.
  We distribute the patches under Apache-2.0.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
  informed the checkpoint diagnosis.
  It also informed the ModelOpt integration approach.
  Mia's AI Lab publishes that project under AGPL-3.0-or-later.
  This standalone implementation uses an image-time ModelOpt alias instead of Mia's runtime checkpoint overlays.
  This repository does not include Mia's infrastructure scripts.
- This repository does not include the NVIDIA model, downloaded weights, base image, or SparkRun.
  Each retains its upstream terms.

We dedicate this directory's original recipe, Dockerfile, build script, and documentation to the public domain under the Unlicense.
See `LICENSE`.
See `THIRD_PARTY_NOTICES.md`.
We provide everything **as is**, without warranty.
