# Third-party notices

- **vLLM** — Copyright the vLLM contributors; Apache License 2.0.
  The patch generators modify small source anchors from vLLM's Qwen3.8 and ModelOpt integrations.
  They also quote these anchors.
  <https://github.com/vllm-project/vllm>
- **MiaAI-Lab Qwen3.8-Flash-Next Dual DGX Sparks** — Mia's AI Lab;
  AGPL-3.0-or-later.
  Its diagnosis of checkpoint-relative MTP metadata informed this implementation.
  Its diagnosis of the missing `FP8_BLOCK_SCALES` ModelOpt dispatch also informed this implementation.
  This recipe does not redistribute Mia's infrastructure scripts.
  <https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks>
- **NVIDIA Qwen3.8-Flash-Next-NVFP4** — The model and weights retain the terms in the model card.
  The recipe references the model and weights.
  It downloads them at runtime.
  This repository does not include them.
  <https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4>
- **SparkRun** — Copyright the SparkRun contributors; Apache License 2.0.
  <https://github.com/spark-arena/sparkrun>
