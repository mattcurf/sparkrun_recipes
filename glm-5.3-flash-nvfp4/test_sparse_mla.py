# SPDX-License-Identifier: Unlicense
"""GPU regression: the installed backend must retain GLM's kpool tail tokens.

Run inside the built image on an idle Spark, mounting this directory at /recipe:
    python3 /recipe/test_sparse_mla.py
"""

from types import SimpleNamespace

import torch
from glm53_sparse_mla.backend import (
    Glm53SparseMLAImpl,
    Glm53SparseMLAMetadataBuilder,
)


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")
    tokens, heads, dim, rows, width = 4, 32, 512, 4096, 2176
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(index_topk=2048)),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=tokens),
    )
    builder = Glm53SparseMLAMetadataBuilder(
        SimpleNamespace(block_size=64), [], config, device
    )
    starts = torch.tensor([0, tokens], dtype=torch.int32)
    metadata = builder.build(
        0,
        SimpleNamespace(
            num_actual_tokens=tokens,
            query_start_loc_cpu=starts,
            query_start_loc=starts.to(device),
            num_reqs=1,
            max_query_len=tokens,
            max_seq_len=rows,
            slot_mapping=torch.arange(tokens, device=device),
            block_table_tensor=torch.arange(
                rows // 64, dtype=torch.int32, device=device
            ).view(1, -1),
        ),
    )
    indices = torch.full((tokens, width), -1, dtype=torch.int32, device=device)
    # The second row has ONLY tail entries: truncating to 2048 yields zeros.
    indices[1, 2048:2051] = torch.tensor([4093, 4094, 4095], device=device)
    indices[2, :2051] = torch.arange(2051, device=device)
    indices[3, :128] = torch.randperm(rows, device=device)[:128]
    q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16, device=device)
    original_kv = torch.randn(rows, dim, device=device)
    scale = dim**-0.5

    for dtype, kv_scale in [(torch.bfloat16, 1.0), (torch.float8_e4m3fn, 0.125)]:
        kv = (original_kv / kv_scale).to(dtype)
        impl = Glm53SparseMLAImpl(
            num_heads=heads,
            head_size=dim,
            scale=scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8" if dtype == torch.float8_e4m3fn else "auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=indices,
            kv_lora_rank=dim,
        )
        out, _ = impl.forward_mqa(
            q, kv.view(-1, 64, dim), metadata, SimpleNamespace(_k_scale_float=kv_scale)
        )
        dequantized = (kv.float() * kv_scale).bfloat16().float()
        reference = torch.zeros_like(out, dtype=torch.float32)
        for token in range(tokens):
            valid = indices[token][indices[token] >= 0].long()
            if valid.numel():
                selected = dequantized[valid]
                probs = (q[token].float() @ selected.T * scale).softmax(-1)
                reference[token] = probs @ selected
        assert torch.isfinite(out).all()
        assert (out[0] == 0).all()
        relative = (out.float() - reference).abs().max() / reference.abs().max()
        cosine = torch.nn.functional.cosine_similarity(
            out[1:].float(), reference[1:], dim=-1
        ).min()
        assert relative < 0.006, relative
        assert cosine > 0.9999, cosine
        print(
            f"PASS {dtype}: full 2176-wide backend, tail, masked row; "
            f"relative={relative.item():.6f}, cosine={cosine.item():.6f}"
        )


if __name__ == "__main__":
    main()
