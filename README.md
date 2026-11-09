# SparkRun recipes for DGX Spark clusters

Serve large models on two or four NVIDIA DGX Sparks.

## Choose a recipe

Folder names identify the model, weight quantization, and node count.

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

TensorFold's SparkRun launcher has offline validation only.
The spark-ds41f recipe has serving checks through approximately 119K tokens, not full 1M qualification.

## Install SparkRun

Install Docker and configure passwordless SSH on every node.
Follow the [SparkRun quick start](https://sparkrun.dev/getting-started/quick-start/) for host requirements.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uvx sparkrun setup
```

## Run a recipe

Complete the recipe folder's preparation steps before using these example commands.

```bash
sparkrun recipe validate <recipe.yaml>
sparkrun run <recipe.yaml> --cluster <cluster-name>
sparkrun status
```

## Clear unused memory

[`tools/cleanup-memory.sh`](tools/cleanup-memory.sh) releases page cache and unified memory after a model stops.
Name every target node explicitly:

```bash
tools/cleanup-memory.sh local <worker-host>
```

The script runs a privileged container on each selected host.

## License and attribution

Original material uses the [Unlicense](LICENSE), unless a file or folder states a different license.
The Unlicense dedicates material to the public domain without warranty.
Third-party code, model weights, and images retain their own licenses.

| Project | Contribution | License |
| --- | --- | --- |
| [SparkRun](https://github.com/spark-arena/sparkrun) | Cluster launcher | Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | Inference runtime | Apache-2.0 |
| [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f) | 1107 TileLang, DSpark, and sparknet integration | Apache-2.0 |
| [MiaAI-Lab GLM](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | GLM conversion and runtime | MIT |
| [SparkRun recipe registry](https://github.com/spark-arena/recipe-registry) | GLM EXL3 recipe | MIT |
| [MiaAI-Lab Qwen](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | Qwen integration reference | AGPL-3.0-or-later |
| [Tony Deangelo's DeepSeek V4](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark) | V4 DSpark integration | MIT; Apache-2.0 runtime components |
| [Tony Deangelo's DeepSeek V4.1](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark) | V4.1 TP4 integration | MIT; Apache-2.0 runtime components |

The spark-ds41f recipe includes its own [LICENSE](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/LICENSE)
and [NOTICE](deepseek-v4.1-flash-1017-spark-ds41f-mxfp4-4node/NOTICE).
See each recipe's attribution for component terms.
This repository does not distribute model weights or third-party container images.
