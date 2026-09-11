# SPDX-License-Identifier: Apache-2.0
"""Fail-closed SM121 indexer/PDL fixes for the pinned vLLM image.

Source anchors are Apache-2.0 vLLM excerpts. These changes follow the failure
analysis in tonyd2wild's GLM-5.3-Flash dual-Spark deployment, without replacing
our native512 attention backend or NVIDIA's calibrated activation scales.
"""

from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"Expected one source anchor: {old!r}")
    return source.replace(old, new)


def patch(root):
    indexer = root / "model_executor/layers/sparse_attn_indexer_kpool.py"
    source = indexer.read_text()
    for indent in (16, 12):
        space = " " * indent
        old = f"{space}pool_topk = torch.empty(\n{space}    (num_rows, select_k), dtype=torch.int32, device=logits.device\n{space})"
        new = f"{space}pool_topk = torch.full(\n{space}    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n{space})"
        source = replace_once(source, old, new)
    source = replace_once(
        source,
        "if current_platform.is_cuda() and select_k in (512, 1024, 2048):",
        "if (current_platform.is_cuda()\n"
        "            and not current_platform.is_device_capability_family(120)\n"
        "            and select_k in (512, 1024, 2048)):",
    )
    compile(source, str(indexer), "exec")
    indexer.write_text(source)
    compress = root / "models/glm5next/nvidia/ops/kpool_compress.py"
    source = replace_once(
        compress.read_text(),
        "hist_out = tl.where(pid >= 0, hist_val, -1)",
        "hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)",
    )
    compile(source, str(compress), "exec")
    compress.write_text(source)
    platform = root / "platforms/cuda.py"
    source = replace_once(
        platform.read_text(), "return major >= 9", "return major in (9, 10)"
    )
    compile(source, str(platform), "exec")
    platform.write_text(source)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
