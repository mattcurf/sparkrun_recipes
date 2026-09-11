# Third-party notices

- [NVIDIA GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  is a ModelOpt-quantized variant of ZAI's GLM-5.3-Flash. Its model card declares
  the MIT license and provides the deployment guidance informing this recipe.
- [vLLM](https://github.com/vllm-project/vllm) is Apache-2.0 licensed. This recipe
  references its published ARM64 container; bundled components retain their
  respective licenses. `patch_dflash2.py`, `patch_sm121.py`,
  `patch_nvidia_mtp.py`, and the source-overlay tests contain/adapt vLLM source
  excerpts and retain Apache-2.0 licensing.
  Copyright contributors to the vLLM project.
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) is
  Apache-2.0 licensed and was used upstream to quantize the checkpoint.
- [SparkRun](https://github.com/spark-arena/sparkrun) is Apache-2.0 licensed.
- [LibertAI's sparse-MLA plugin](https://github.com/Libertai/vllm-sparse-mla-blackwell/tree/3701f85be073ff75578bdffb22469161223b09fc)
  is Apache-2.0 licensed. The Docker build fetches this pinned source, verifies
  its archive checksum, and compiles its native SM121 attention kernel. It
  adapts the backend to GLM's nested text config and retains the complete kpool
  tail instead of truncating the index buffer. These changes are recorded in
  `patch_sparse_mla.py`, which includes upstream source excerpts and is also
  Apache-2.0 licensed. See [LICENSE-APACHE](LICENSE-APACHE).
- [tonyd2wild's dual-Spark deployment](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3)
  supplies reference failure analysis and DFlash2 integration design. This
  build retains the newer image's upstream DFlash2 implementation rather than
  downloading that deployment's overlay or substituting its target checkpoint.
- [IncoAI's GLM-5.3-Flash DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
  is an optional draft checkpoint; its weights retain their model-card license
  and are downloaded separately, not distributed in this repository.

No model weights or container images are included in this directory. Apart
from the Apache-2.0 files identified by their SPDX headers, original recipe,
build files, benchmark, runtime regression tests, and
documentation are dedicated to the public domain under the recipe's
[Unlicense](LICENSE), without warranty.
