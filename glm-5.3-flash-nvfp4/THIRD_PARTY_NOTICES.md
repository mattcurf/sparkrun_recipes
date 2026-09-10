# Third-party notices

- [NVIDIA GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
  is a ModelOpt-quantized variant of ZAI's GLM-5.3-Flash. Its model card declares
  the MIT license and provides the deployment guidance informing this recipe.
- [vLLM](https://github.com/vllm-project/vllm) is Apache-2.0 licensed. This recipe
  references its published ARM64 container; bundled components retain their
  respective licenses.
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

No model weights or container images are included in this directory. Apart
from the Apache-2.0 adaptation script, original recipe, build files, test, and
documentation are dedicated to the public domain under the repository's
[Unlicense](../LICENSE), without warranty.
