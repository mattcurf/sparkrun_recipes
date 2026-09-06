# SPDX-License-Identifier: Apache-2.0
# ModelOpt integration approach informed by MiaAI-Lab's Qwen3.8 work.

"""Add NVIDIA Qwen3.8-Flash-Next-NVFP4 MTP support to vLLM ModelOpt.

The checkpoint records its one MTP layer relative to the MTP stack as layer 0,
while vLLM creates it at absolute layer 48. Its routed experts use 128x128
FP8_BLOCK_SCALES, for which ModelOpt's mixed-precision dispatcher has no branch.
Both gaps otherwise leave those experts unquantized and make weight loading fail.
"""

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/"
    "layers/quantization/modelopt.py"
)
source = path.read_text()

candidate_anchor = '''        candidates = [prefix]

        if prefix.endswith(".lm_head"):
'''
candidate_replacement = '''        candidates = [prefix]

        # Qwen3.8-Flash-Next checkpoints number MTP metadata relative to the MTP
        # stack, while vLLM gives the layer an absolute index after the main
        # stack. This model has one MTP layer, so every runtime MTP layer maps to
        # checkpoint-relative layer 0.
        mtp_parts = prefix.split(".", 3)
        if (
            len(mtp_parts) == 4
            and mtp_parts[:2] == ["mtp", "layers"]
            and mtp_parts[2].isdigit()
        ):
            candidates.append(f"mtp.layers.0.{mtp_parts[3]}")

        if prefix.endswith(".lm_head"):
'''

helper_anchor = '''    @staticmethod
    def _quantized_layer_prefix_candidates(prefix: str) -> tuple[str, ...]:
'''
helper = '''    def _fp8_block_scales_config(self, prefix: str):
        """Build FP8 config from checkpoint metadata; never guess block shape."""
        from vllm.model_executor.layers.quantization.fp8 import Fp8Config

        info = None
        for candidate in self._quantized_layer_prefix_candidates(prefix):
            info = self.quantized_layers.get(candidate)
            if info is None:
                candidate_dot = candidate + "."
                for key, value in self.quantized_layers.items():
                    if key.startswith(candidate_dot):
                        info = value
                        break
            if info is not None:
                break

        group_size = (info or {}).get("group_size")
        if not isinstance(group_size, int) or group_size <= 0:
            raise ValueError(
                f"FP8_BLOCK_SCALES layer {prefix} has unusable group_size "
                f"{group_size!r}; refusing to guess its block shape."
            )
        return Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[group_size, group_size],
        )

'''

dispatch_anchor = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            return None
'''
dispatch_replacement = '''            if quant_algo == "MXFP8":
                return ModelOptMxFp8FusedMoE(
                    quant_config=self.mxfp8_config,
                    moe_config=layer.moe_config,
                )
            if quant_algo == "FP8_BLOCK_SCALES":
                from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

                logger.info_once(
                    "Routed experts %s use FP8_BLOCK_SCALES; building them as "
                    "block-quantized FP8.",
                    prefix,
                )
                return Fp8MoEMethod(
                    quant_config=self._fp8_block_scales_config(prefix),
                    layer=layer,
                )
            return None
'''

for anchor, replacement, name in (
    (candidate_anchor, candidate_replacement, "MTP relative-layer candidate"),
    (helper_anchor, helper + helper_anchor, "FP8 block-scale config helper"),
    (dispatch_anchor, dispatch_replacement, "FP8 block-scale MoE dispatch"),
):
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {name} anchor, found {count}")
    source = source.replace(anchor, replacement)

path.write_text(source)
