# SPDX-License-Identifier: Unlicense
"""Exercise the installed draft configuration without loading model weights."""

import json
import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from huggingface_hub import snapshot_download
from vllm.config import LoadConfig, ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.models.glm5next.nvidia import mtp


class NvidiaMTPConfigTests(unittest.TestCase):
    def test_bf16_draft_preserves_nvfp4_target(self):
        self.check_mtp("nvidia/GLM-5.3-Flash-NVFP4")

    def test_resolved_snapshot_preserves_nvfp4_target(self):
        self.check_mtp(
            snapshot_download(
                "nvidia/GLM-5.3-Flash-NVFP4",
                revision="423acf37583782c51c142d145aef733d72943d93",
                local_files_only=True,
            )
        )

    def check_mtp(self, model):
        revision = "423acf37583782c51c142d145aef733d72943d93"
        target = ModelConfig(
            model=model,
            revision=revision,
            max_model_len=32768,
            enforce_eager=True,
        )
        before = json.dumps(target.hf_config.to_dict(), sort_keys=True)
        self.assertEqual(target.quantization, "modelopt_fp4")
        for depth in (1, 2, 4):
            with self.subTest(depth=depth):
                spec = SpeculativeConfig(
                    method="mtp",
                    num_speculative_tokens=depth,
                    moe_backend="auto",
                    target_model_config=target,
                    target_parallel_config=ParallelConfig(
                        tensor_parallel_size=2, nnodes=2
                    ),
                )
                self.assertIsNone(spec.draft_model_config.quantization)
                self.assertIsNone(spec.draft_model_config.hf_config.quantization_config)
                self.assertEqual(spec.draft_model_config.revision, revision)
                self.assertEqual(
                    spec.draft_model_config.hf_config.model_type, "glm5_next_mtp"
                )
                pickle.loads(pickle.dumps(spec.draft_model_config.hf_overrides))
                self.assertEqual(
                    before, json.dumps(target.hf_config.to_dict(), sort_keys=True)
                )
                self.assertEqual(target.quantization, "modelopt_fp4")
                target_quant = object()
                config = SimpleNamespace(
                    model_config=target,
                    quant_config=target_quant,
                    speculative_config=spec,
                    load_config=LoadConfig(),
                )
                with (
                    patch.object(
                        mtp,
                        "Glm5NextMultiTokenPredictor",
                        return_value=torch.nn.Module(),
                    ) as predictor,
                    patch.object(mtp.Glm5NextMTP, "set_moe_parameters"),
                ):
                    draft = mtp.Glm5NextMTP(vllm_config=config)
                passed_config = predictor.call_args.kwargs["vllm_config"]
                self.assertIsNone(draft.quant_config)
                self.assertIsNone(passed_config.quant_config)
                self.assertIs(passed_config.model_config, spec.draft_model_config)
                self.assertIs(config.quant_config, target_quant)
                self.assertIs(config.model_config, target)

    def test_dflash2_draft_keeps_independent_configuration(self):
        target = ModelConfig(
            model="nvidia/GLM-5.3-Flash-NVFP4",
            revision="423acf37583782c51c142d145aef733d72943d93",
            max_model_len=32768,
            enforce_eager=True,
        )
        spec = SpeculativeConfig(
            method="dflash",
            model="incoai/GLM-5.3-Flash-DFlash2",
            revision="dc77ff1c99eeb2df044ee3d4f0094eb033fee410",
            num_speculative_tokens=7,
            kv_cache_dtype="auto",
            target_model_config=target,
            target_parallel_config=ParallelConfig(tensor_parallel_size=2, nnodes=2),
        )
        self.assertIsNone(spec.draft_model_config.quantization)
        self.assertEqual(
            spec.draft_model_config.hf_config.architectures, ["DFlash2DraftModel"]
        )
        self.assertEqual(spec.kv_cache_dtype, "auto")
        self.assertEqual(target.quantization, "modelopt_fp4")
        self.assertEqual(
            spec.draft_model_config.hf_config.dflash_config["target_layer_ids"],
            [5, 14, 24, 33, 42],
        )


if __name__ == "__main__":
    unittest.main()
