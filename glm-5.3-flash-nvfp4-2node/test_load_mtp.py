# SPDX-License-Identifier: Unlicense
"""Load the real native MTP weights on one idle GPU, without loading the target.

This catches quantization/shape errors cheaply before a full TP2 launch. It is
not an inference test and does not replace the distributed API regression.
Set VLLM_GLM53_CUDA_SPARSE_MLA=0 for this TP1 weight-only test: the native
attention kernel requires at least TP2, and no attention is executed here.
"""

import torch
from huggingface_hub import snapshot_download
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model


def main():
    revision = "423acf37583782c51c142d145aef733d72943d93"
    snapshot = snapshot_download(
        "nvidia/GLM-5.3-Flash-NVFP4", revision=revision, local_files_only=True
    )
    config = EngineArgs(
        model=snapshot,
        revision=revision,
        enforce_eager=True,
        max_model_len=32768,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        kv_cache_dtype="fp8",
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        enable_expert_parallel=True,
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "moe_backend": "auto",
        },
    ).create_engine_config()
    assert config.speculative_config.draft_model_config.quantization is None
    assert config.model_config.quantization == "modelopt_fp4"
    torch.cuda.set_device(0)
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="tcp://127.0.0.1:29691",
        )
        initialize_model_parallel()
        try:
            model = get_model(
                vllm_config=config,
                model_config=config.speculative_config.draft_model_config,
            )
            assert model.quant_config is None
            layer = model.model.layers["45"]
            experts = layer.mtp_block.mlp.experts.routed_experts
            assert experts.w13_weight.dtype == torch.bfloat16
            assert experts.w2_weight.dtype == torch.bfloat16
            print("PASS: real NVIDIA BF16 MTP weights loaded; target remains NVFP4")
        finally:
            cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
