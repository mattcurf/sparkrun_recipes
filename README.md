# SparkRun recipes for DGX Spark clusters

Recipes for serving large models across two or four NVIDIA DGX
Sparks. Each model directory documents its validation, build files when needed,
usage instructions, license, and upstream notices.

## Recipes

Folders follow `<model>-<weight-quant>-<N>node`, where quantization describes
the model checkpoint, not the KV cache or runtime kernel format. Each folder
contains one recipe. Speculative decoding methods remain in the configuration
description rather than the folder name. Recipe filenames, served API names,
container tags, and checkpoint revisions are unchanged.

| Model | Configuration | Directory |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 | FP8 weights, TP2, DSpark speculative decoding, 1M context | [`deepseek-v4-flash-0731-fp8-2node/`](deepseek-v4-flash-0731-fp8-2node/) |
| DeepSeek V4.1 Flash | Official MXFP4/MXFP8 weights, TP4, disk-backed Engram, DSpark, 256K context | [`deepseek-v4.1-flash-mxfp4-4node/`](deepseek-v4.1-flash-mxfp4-4node/) |
| GLM-5.3-Flash EXL3 | EXL3/TR3 4 bpw, TP2, DFlash2, FP8 KV, 192K context | [`glm-5.3-flash-exl3-2node/`](glm-5.3-flash-exl3-2node/) |
| GLM-5.3-Flash NVFP4 | NVIDIA NVFP4, TP2 + expert parallelism, DFlash2, FP8 KV, 192K context | [`glm-5.3-flash-nvfp4-2node/`](glm-5.3-flash-nvfp4-2node/) |
| GLM-5.3-Flash NVFP4 | NVIDIA NVFP4, TP4 + expert parallelism, DFlash2, FP8 KV, 256K context | [`glm-5.3-flash-nvfp4-4node/`](glm-5.3-flash-nvfp4-4node/) |
| GLM-5.3-Flash FP8 | Native Z.ai FP8, TP4 + expert parallelism, DFlash2, FP8 KV, 256K context | [`glm-5.3-flash-fp8-4node/`](glm-5.3-flash-fp8-4node/) |
| Qwen3.8-Flash-Next FP8 | Original Qwen FP8, TP2, MTP3, 256K context | [`qwen3.8-flash-next-fp8-2node/`](qwen3.8-flash-next-fp8-2node/) |
| Qwen3.8-Flash-Next NVFP4 | NVIDIA NVFP4, TP2 + expert parallelism, MTP3, 256K context | [`qwen3.8-flash-next-nvfp4-2node/`](qwen3.8-flash-next-nvfp4-2node/) |

Update local scripts/bookmarks to these paths; old folders are not retained as
aliases. Renaming files does not restart an existing workload. For a job launched
under an old path, use `sparkrun status` and the reported job ID to manage it.

## Install SparkRun

These recipes depend on the [`sparkrun`](https://github.com/spark-arena/sparkrun)
command-line tool, Docker, and passwordless SSH access to the DGX Spark nodes.
Install `uv` if needed, then use SparkRun's guided setup:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx sparkrun setup
```

The wizard installs SparkRun, configures a cluster and SSH, detects ConnectX-7
networking, and applies recommended host setup. See the
[SparkRun quick start](https://sparkrun.dev/getting-started/quick-start/) for
current prerequisites and options.

After setup, enter a recipe directory and follow its README. In general:

```bash
sparkrun recipe validate <recipe.yaml>
sparkrun run <recipe.yaml> --cluster <cluster-name>
sparkrun status
```

## Tools

[`tools/cleanup-memory.sh`](tools/cleanup-memory.sh) reclaims page cache and
unified memory retained after a model is stopped. Pass every node explicitly:

```bash
tools/cleanup-memory.sh local <worker-host>
```

The script runs a privileged container on each selected host. Stop the
inference workload first; remote hosts require passwordless SSH and Docker.

## License and attribution

Except where a file or directory says otherwise, original material in this
repository is released into the public domain under the [Unlicense](LICENSE).
It is offered **as is**, without warranty of any kind.

Third-party projects, downloaded model weights, container images, and portions
derived from upstream software retain their own licenses. Each recipe directory
has standalone attribution. Major upstream work includes:

- [SparkRun](https://github.com/spark-arena/sparkrun), Apache-2.0.
- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0.
- [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks),
  MIT, which provides the GLM model conversion and patched runtime image.
- [SparkRun recipe registry](https://github.com/spark-arena/recipe-registry),
  MIT, whose experimental GLM recipe is the baseline for this collection's
  tuned 196K-context variant.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
  AGPL-3.0-or-later, whose NVIDIA-checkpoint and MTP investigations informed
  the Qwen integration.
- [Tony Deangelo's DeepSeek V4 Flash DSpark project](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark),
  MIT with Apache-2.0-derived runtime components.
- [Tony Deangelo's DeepSeek V4.1 Flash TP4 project](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark),
  MIT with Apache-2.0-derived runtime components.

No model weights or third-party container images are distributed here.
