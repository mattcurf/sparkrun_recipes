# Qwen3.8-Flash-Next NVFP4 with MTP3

Serve [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) on two DGX Sparks.

| Setting | Value |
| --- | --- |
| Parallelism | TP2 with expert parallelism |
| MTP draft tokens | 3 |
| Maximum context | 262,144 |

## Build and run

Complete [SparkRun setup](../README.md#install-sparkrun) before launch.

```bash
./build-image.sh
sparkrun recipe validate qwen3.8-flash-next-nvfp4-mtp.yaml
sparkrun run qwen3.8-flash-next-nvfp4-mtp.yaml --cluster <two-node-cluster>
```

Use the custom image for checkpoint-compatible PLE and MTP support.
The build stops if pinned vLLM source anchors differ.
SparkRun distributes the image and model.
The API listens on port 8000 by default.

## Attribution and licensing

The [vLLM](https://github.com/vllm-project/vllm)-derived patches use Apache-2.0.
Integration reference: [MiaAI-Lab Qwen](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks), AGPL-3.0-or-later.
This implementation uses an image-time ModelOpt alias.
The original recipe, Dockerfile, build script, and docs use the [Unlicense](LICENSE): public domain, without warranty.
See [third-party notices](THIRD_PARTY_NOTICES.md) for component terms and distribution exclusions.
