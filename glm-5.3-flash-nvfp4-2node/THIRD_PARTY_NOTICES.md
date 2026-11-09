# Third-party notices

- [NVIDIA GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  is a ModelOpt-quantized variant of ZAI's GLM-5.3-Flash.
  Its model card declares the MIT license.
  The recipe follows that model card's deployment guidance.
- [vLLM](https://github.com/vllm-project/vllm) is Apache-2.0 licensed. This recipe
  references its published ARM64 container.
  Bundled components retain their respective licenses.
  `patch_dflash2.py`, `patch_sm121.py`,
  `patch_nvidia_mtp.py`, `patch_openai_compat.py`, and the source-overlay tests
  contain/adapt vLLM source excerpts and retain Apache-2.0 licensing.
  Copyright contributors to the vLLM project.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) is
  Apache-2.0 licensed.
  The upstream project used it to quantize the checkpoint.
- [SparkRun](https://github.com/spark-arena/sparkrun) is Apache-2.0 licensed.
- [knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4/tree/770d115)
  is MIT licensed.
  This recipe copies `glm_fast_load.py` from its `overlay/` directory with an attribution header.
  See [LICENSE-MIT-knapcio](LICENSE-MIT-knapcio).
- [LibertAI's sparse-MLA plugin](https://github.com/Libertai/vllm-sparse-mla-blackwell/tree/3701f85be073ff75578bdffb22469161223b09fc)
  is Apache-2.0 licensed.
  The Docker build fetches this pinned source.
  It verifies the archive checksum.
  It compiles the native SM121 attention kernel.
  It adapts the backend to GLM's nested text config.
  It retains the complete kpool tail instead of truncating the index buffer.
  `patch_sparse_mla.py` records these changes.
  It includes upstream source excerpts and is also Apache-2.0 licensed.
  See [LICENSE-APACHE](LICENSE-APACHE).
- [tonyd2wild's dual-Spark deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3)
  supplies reference failure analysis and DFlash2 integration design.
  This build retains the newer image's upstream DFlash2 implementation.
  It does not download that deployment's overlay or substitute its target checkpoint.
- [IncoAI's GLM-5.3-Flash DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  is an optional draft checkpoint.
  Its weights retain their model-card license.
  The runtime downloads its weights separately.
  This repository does not distribute them.

This directory does not include model weights or container images.
The original recipe, build files, benchmark, runtime regression tests, and documentation use the recipe's [Unlicense](LICENSE).
We dedicate this original material to the public domain without warranty.
This dedication excludes the Apache-2.0 files that their SPDX headers identify.
