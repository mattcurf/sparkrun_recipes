# SparkRun recipes for DGX Spark clusters

Use these recipes to serve large models on two or four NVIDIA DGX Sparks.
Each model folder contains setup instructions.
Each folder states its test results and license.
Some folders also contain build files.

## Choose a recipe

Folder names use `<model>-<weight-quant>-<N>node`.
The quantization name describes the model weights.
It does not describe the KV cache or runtime kernels.
Each folder contains one recipe.
Folder names do not include speculative-decoding methods.
Recipe filenames remain unchanged.
Served API names remain unchanged.
Container tags remain unchanged.
Checkpoint revisions remain unchanged.

| Model | Configuration | Directory |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 | FP8 weights, TP2, DSpark, 1M context | [`deepseek-v4-flash-0731-fp8-2node/`](deepseek-v4-flash-0731-fp8-2node/) |
| DeepSeek V4.1 Flash | Official MXFP4/MXFP8 weights, TP4, disk Engram, DSpark, 256K context | [`deepseek-v4.1-flash-mxfp4-4node/`](deepseek-v4.1-flash-mxfp4-4node/) |
| DeepSeek V4.1 Flash 1017 | Official MXFP4/MXFP8 weights, TP4, patched disk Engram, maximum thinking, DSpark, 256K context | [`deepseek-v4.1-flash-1017-mxfp4-4node/`](deepseek-v4.1-flash-1017-mxfp4-4node/) |
| DeepSeek V4.1 Flash 1017 — TensorFold | Official MXFP4/MXFP8 weights, native Zig TP4, local-SSD Engram, DSpark, 1M context | [`deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node/`](deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node/) |
| DeepSeek V4.1 Flash 1017 — spark-ds41f | spark-ds41f-1107 TileLang/vLLM, switched TP4, NVMe Engram, DSpark, 6-GiB KV, 1M context | [`deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/`](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/) |
| GLM-5.3-Flash EXL3 | EXL3/TR3 4 bpw, TP2, DFlash2, FP8 KV, 192K context | [`glm-5.3-flash-exl3-2node/`](glm-5.3-flash-exl3-2node/) |
| GLM-5.3-Flash NVFP4 | NVIDIA NVFP4, TP2, expert parallelism, DFlash2, FP8 KV, 192K context | [`glm-5.3-flash-nvfp4-2node/`](glm-5.3-flash-nvfp4-2node/) |
| GLM-5.3-Flash NVFP4 | NVIDIA NVFP4, TP4, expert parallelism, DFlash2, FP8 KV, 256K context | [`glm-5.3-flash-nvfp4-4node/`](glm-5.3-flash-nvfp4-4node/) |
| GLM-5.3-Flash FP8 | Native Z.ai FP8, TP4, expert parallelism, DFlash2, FP8 KV, 256K context | [`glm-5.3-flash-fp8-4node/`](glm-5.3-flash-fp8-4node/) |
| Qwen3.8-Flash-Next FP8 | Original Qwen FP8, TP2, MTP3, 256K context | [`qwen3.8-flash-next-fp8-2node/`](qwen3.8-flash-next-fp8-2node/) |
| Qwen3.8-Flash-Next NVFP4 | NVIDIA NVFP4, TP2, expert parallelism, MTP3, 256K context | [`qwen3.8-flash-next-nvfp4-2node/`](qwen3.8-flash-next-nvfp4-2node/) |

The TensorFold recipe records settings from a working service.
Its SparkRun launcher has only passed offline checks and a dry run.
Its Engram storage does not use NFS.

The spark-ds41f recipe comes from Christopher Owen's project.
Its configured context limit is 1M tokens.
Service checks covered approximately 119K tokens.
A matched benchmark also passed.
These checks do not prove full 1M capacity.

Update scripts and bookmarks to use the current folder names.
Old folder names have no aliases.
Folder changes do not restart workloads.
Use `sparkrun status` to find jobs that use old paths.
Manage those jobs with their reported job IDs.

## Install SparkRun

These recipes require [SparkRun](https://github.com/spark-arena/sparkrun).
They also require Docker.
Each node must permit passwordless SSH access.

1. Install `uv` if necessary.
2. Start SparkRun setup.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx sparkrun setup
```

The setup tool installs SparkRun.
It configures the cluster and SSH access.
It detects ConnectX-7 networking.
It applies the recommended host settings.
See the [SparkRun quick start](https://sparkrun.dev/getting-started/quick-start/) for current requirements.

## Run a recipe

1. Open the recipe folder.
2. Read its README.
3. Check the recipe.
4. Start the workload only after the required preparation.

The usual commands follow:

```bash
sparkrun recipe validate <recipe.yaml>
sparkrun run <recipe.yaml> --cluster <cluster-name>
sparkrun status
```

## Clear unused memory

[`tools/cleanup-memory.sh`](tools/cleanup-memory.sh) releases page cache and unified memory after a model stops.
**Stop inference before you use this script.**
List every node explicitly:

```bash
tools/cleanup-memory.sh local <worker-host>
```

The script runs a privileged container on each selected host.
Remote hosts require passwordless SSH access.
Remote hosts also require Docker.

## License and attribution

Original material uses the [Unlicense](LICENSE), unless a file or folder states a different license.
The Unlicense releases this material into the public domain.
The Unlicense provides no warranty.
Third-party software keeps its own license.
Model weights keep their own license.
Container images keep their own licenses.
Each recipe folder gives its own attribution.

This repository uses work from these projects:

- [SparkRun](https://github.com/spark-arena/sparkrun): Apache-2.0.
- [vLLM](https://github.com/vllm-project/vllm): Apache-2.0.
- [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f): Apache-2.0.
  It supplies the 1107 serving integration for the [spark-ds41f recipe](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/).
  This integration includes TileLang kernels.
  It also includes DSpark and the sparknet networking profile.
  The recipe includes a separate [LICENSE](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/LICENSE).
  It also includes a [NOTICE](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/NOTICE).
  Upstream components keep their own licenses.
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks): MIT.
  This project supplies the GLM model conversion.
  It also supplies the patched runtime image.
- [SparkRun recipe registry](https://github.com/spark-arena/recipe-registry): MIT.
  Its experimental GLM recipe provides the starting point for this collection's tuned 196K-context variant.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks): AGPL-3.0-or-later.
  Its NVIDIA-checkpoint investigations helped this collection's Qwen integration.
  Its MTP investigations also helped that integration.
- [Tony Deangelo's DeepSeek V4 Flash DSpark project](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark): MIT.
  Its Apache-2.0-derived runtime components keep that license.
- [Tony Deangelo's DeepSeek V4.1 Flash TP4 project](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark): MIT.
  Its Apache-2.0-derived runtime components keep that license.

This repository does not distribute model weights.
It does not distribute third-party container images.
