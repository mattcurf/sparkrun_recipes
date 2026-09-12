# SPDX-License-Identifier: Apache-2.0
"""Adapt the pinned LibertAI backend to GLM's nested config and full kpool tail.

The CUDA kernel accepts any top-k width divisible by 32. Preserve the complete
2176-wide index buffer, including up to three always-selected recent tokens;
the remaining padding is masked by -1. Do not truncate to 2048 or drop a pool.
Source excerpts are from Libertai/vllm-sparse-mla-blackwell (Apache-2.0).
"""

import sys
from pathlib import Path


def patch(path: Path) -> None:
    source = path.read_text()
    replacements = [
        (
            "vllm_config.model_config.hf_config.index_topk",
            "vllm_config.model_config.hf_text_config.index_topk",
        ),
        (
            (
                "        # The kpool indexer emits topk + (kpool - 1) columns, which vLLM rounds up\n"
                "        # to a multiple of 128 (2048 -> 2176). Narrow at the call site rather than\n"
                "        # resizing the indexer's buffer: touching what the indexer writes risks it\n"
                "        # selecting nothing, and an all--1 row makes MLA return zeros, which\n"
                "        # presents as the model copying its prompt verbatim.\n"
                "        cap = int(attn_metadata.topk_tokens)\n"
                "        if topk_indices.shape[1] > cap:\n"
                "            topk_indices = topk_indices[:, :cap].contiguous()\n"
            ),
            (
                "        # Keep every selected pool and always-selected recent tail token.\n"
                "        # The CUDA kernel accepts the full 2176-wide, -1-padded buffer.\n"
            ),
        ),
    ]
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected exactly one patch anchor: {old!r}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
