# SPDX-License-Identifier: Apache-2.0
"""Load NVIDIA's unquantized native MTP layer without quantizing the target.

The pinned checkpoint's layer 45 contains 888 BF16 tensors and one F32 tensor,
with no quantization scales. Its top-level ModelOpt metadata describes the
target, not this draft. Apply a draft-only Hugging Face config transform before
vLLM derives its quantization configuration. No checkpoint files are edited.
"""

from pathlib import Path


def patch(root):
    path = root / "config/speculative.py"
    source = path.read_text()
    anchor = "    @staticmethod\n    def _is_custom_proposer_path("
    helper = """    @staticmethod
    def _nvidia_bf16_mtp_override(target_hf_overrides, hf_config):
        # vLLM probes override callables with a config containing only model_type.
        if hf_config.model_type == "dummy_glm5_next":
            return hf_config
        transform = SpeculativeConfig.compose_draft_hf_overrides(target_hf_overrides)
        hf_config = transform(hf_config)
        if hf_config.model_type != "glm5_next_mtp":
            raise ValueError("NVIDIA BF16 MTP override requires glm5_next_mtp")
        hf_config.quantization_config = None
        return hf_config

"""
    insertion = "                self.draft_model_config = ModelConfig(\n"
    replacement = """                # NVIDIA's pinned native MTP layer is BF16, unlike its target.
                if (
                    self.method == "mtp"
                    and (
                        self.model == "nvidia/GLM-5.3-Flash-NVFP4"
                        or self.model.rstrip("/").endswith(
                            "/models--nvidia--GLM-5.3-Flash-NVFP4/snapshots/"
                            "423acf37583782c51c142d145aef733d72943d93"
                        )
                    )
                    and self.target_model_config.revision
                    == "423acf37583782c51c142d145aef733d72943d93"
                ):
                    if self.revision not in (None, self.target_model_config.revision):
                        raise ValueError("BF16 MTP requires the pinned NVIDIA revision")
                    self.revision = self.target_model_config.revision
                    self.quantization = None
                    draft_hf_overrides = functools.partial(
                        SpeculativeConfig._nvidia_bf16_mtp_override,
                        self.target_model_config.hf_overrides,
                    )
"""
    for old, new in ((anchor, helper + anchor), (insertion, replacement + insertion)):
        if source.count(old) != 1:
            raise RuntimeError(f"Pinned vLLM source drift: {old!r}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    mtp = root / "models/glm5next/nvidia/mtp.py"
    mtp_source = mtp.read_text()
    old = """        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
"""
    new = """        super().__init__()
        # Use vLLM's standard draft quantization helper, never the target's.
        # Shallow copying preserves shared layer registration and cache state.
        from copy import copy
        from vllm.model_executor.models.utils import get_draft_quant_config

        draft_config = copy(vllm_config)
        draft_config.quant_config = get_draft_quant_config(vllm_config)
        draft_config.model_config = vllm_config.speculative_config.draft_model_config
        vllm_config = draft_config
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
"""
    if mtp_source.count(old) != 1:
        raise RuntimeError("Pinned GLM MTP constructor drift")
    mtp_source = mtp_source.replace(old, new)
    compile(mtp_source, str(mtp), "exec")
    path.write_text(source)
    mtp.write_text(mtp_source)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
