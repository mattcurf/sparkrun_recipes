# SPDX-License-Identifier: Apache-2.0
# Derived from vLLM's Qwen3.8 Flash Next PLE integration interfaces.

from pathlib import Path


path = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/ple_layer.py"
)
source = path.read_text()

function_anchor = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if not isinstance(quant_config, Fp8Config):
        return None
'''
function_replacement = '''def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    ple_embedding_dtype: object = None,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    # Mixed-precision ModelOpt checkpoints describe PLE independently from
    # the body quantization. NVIDIA's checkpoint supplies this via hf_overrides.
    if str(ple_embedding_dtype).lower() in {
        "float8_e4m3fn",
        "torch.float8_e4m3fn",
        "fp8",
    }:
        return Qwen3_8FlashNextPLEFp8EmbeddingMethod()

    if not isinstance(quant_config, Fp8Config):
        return None
'''

call_anchor = '''            quant_method=_get_ple_embedding_quant_method(
                quant_config, f"{prefix}.ngram_embedding"
            ),
'''
call_replacement = '''            quant_method=_get_ple_embedding_quant_method(
                quant_config,
                f"{prefix}.ngram_embedding",
                getattr(config, "ple_embedding_dtype", None),
            ),
'''

for anchor, replacement, name in (
    (function_anchor, function_replacement, "PLE quantization dispatcher"),
    (call_anchor, call_replacement, "PLE dispatcher call site"),
):
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {name} anchor, found {count}")
    source = source.replace(anchor, replacement)

path.write_text(source)
