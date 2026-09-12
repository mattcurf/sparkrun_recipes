# SPDX-License-Identifier: Unlicense
"""CPU tests of the installed patched vLLM cache accounting (run in v2 image)."""

import unittest
from types import SimpleNamespace

import torch
from vllm.v1.core import kv_cache_utils as kv
from vllm.v1.kv_cache_interface import (
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


class DrafterLayoutTests(unittest.TestCase):
    def check_layout(self, draft_heads):
        config = SimpleNamespace(
            parallel_config=SimpleNamespace(
                pipeline_parallel_size=1, decode_context_parallel_size=1
            ),
            cache_config=SimpleNamespace(
                num_gpu_blocks_override=None,
                prefix_cache_retention_interval=None,
                mamba_cache_mode="none",
            ),
            model_config=SimpleNamespace(max_model_len=32768),
            max_in_flight_tokens=2048,
        )
        specs = {
            "mla": MLAAttentionSpec(
                block_size=256, num_kv_heads=1, head_size=512, dtype=torch.uint8
            ),
            "indexer": MLAAttentionSpec(
                block_size=256,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.uint8,
                tokens_per_state=4,
            ),
            "mamba": MambaSpec(
                block_size=256, shapes=((16, 16),), dtypes=(torch.float32,)
            ),
            "tail": KpoolTailSpec(
                block_size=4,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                sliding_window=4,
            ),
        }
        if draft_heads:
            specs["draft"] = SlidingWindowSpec(
                block_size=256,
                num_kv_heads=draft_heads,
                head_size=128,
                dtype=torch.bfloat16,
                sliding_window=2048,
                page_size_padded=256 * draft_heads * 512 + 4096,
            )
        groups = kv._get_kv_cache_groups_glm5_next(config, specs)
        self.assertIsNotNone(groups)
        layout = kv._glm5_next_tensor_layout(groups)
        self.assertIsNotNone(layout)
        self.assertEqual(len(layout), 9)
        base = specs["mla"].page_size_bytes + specs["indexer"].page_size_bytes
        expected = base
        if draft_heads:
            draft = groups[-1].kv_cache_spec.kv_cache_specs["draft"]
            self.assertIsNone(draft.page_size_padded)
            if draft_heads == 3:
                expected += draft.page_size_bytes
            else:
                self.assertEqual(draft.page_size_bytes, specs["mla"].page_size_bytes)
        self.assertEqual(kv._get_kv_cache_bytes_per_block(groups), expected)
        self.assertEqual(kv._pool_bytes_per_block(groups), expected)
        # Full MLA context + one Mamba state + one tail block + draft window.
        blocks = 128 + 1 + 1
        if draft_heads:
            blocks += draft.max_admission_blocks_per_request(2048, 32768)
        self.assertEqual(
            kv._max_memory_usage_bytes_from_groups(config, groups), blocks * expected
        )
        result = kv.get_kv_cache_config_from_groups(config, groups, expected * 10)
        self.assertEqual(result.num_blocks, 10)
        tensors = {t.layers[0]: t for t in result.kv_cache_tensors}
        self.assertEqual(set(tensors), set(specs))
        self.assertEqual(tensors["tail"].offset, tensors["indexer"].offset)
        for tensor in tensors.values():
            self.assertEqual(tensor.size, expected * 10)
            self.assertLessEqual(tensor.offset + tensor.layer_stride, tensor.size)
        if draft_heads:
            self.assertEqual(
                tensors["draft"].offset, base * 10 if draft_heads == 3 else 0
            )

    def test_without_drafter(self):
        self.check_layout(0)

    def test_exact_fit_drafter(self):
        self.check_layout(1)

    def test_tp2_dflash2_resized_exact_fit(self):
        self.check_layout(4)

    def test_standalone_drafter(self):
        self.check_layout(3)


if __name__ == "__main__":
    unittest.main()
