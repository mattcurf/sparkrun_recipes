# SparkRun recipes for dual DGX Sparks

Curated, tested recipes for serving large models across a pair of NVIDIA DGX
Sparks. Each model directory contains its own recipe, build files when needed,
usage instructions, license, and upstream notices.

## Recipes

| Model | Configuration | Directory |
| --- | --- | --- |
| DeepSeek V4 Flash 0731 | TP2, FP8 KV, DSpark speculative decoding, 1M context | [`deepseek-v4-flash-0731/`](deepseek-v4-flash-0731/) |
| Qwen3.8-Flash-Next | NVIDIA NVFP4, TP2 + expert parallelism, 262K context | [`qwen3.8-flash-next/`](qwen3.8-flash-next/) |
| Qwen3.8-Flash-Next NVFP4 + MTP | NVIDIA NVFP4, TP2 + expert parallelism, MTP3, 262K context | [`qwen3.8-flash-next-nvfp4-mtp/`](qwen3.8-flash-next-nvfp4-mtp/) |

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

## License and attribution

Except where a file or directory says otherwise, original material in this
repository is released into the public domain under the [Unlicense](LICENSE).
It is offered **as is**, without warranty of any kind.

Third-party projects, downloaded model weights, container images, and portions
derived from upstream software retain their own licenses. Each recipe directory
has standalone attribution. Major upstream work includes:

- [SparkRun](https://github.com/spark-arena/sparkrun), Apache-2.0.
- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0.
- [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks),
  AGPL-3.0-or-later, whose NVIDIA-checkpoint and MTP investigations informed
  the Qwen integration.
- [Tony Deangelo's DeepSeek V4 Flash DSpark project](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark),
  MIT with Apache-2.0-derived runtime components.

No model weights or third-party container images are distributed here.
