# SPDX-License-Identifier: Unlicense
import hashlib
import json
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


class DeepSeekV41RecipeTest(unittest.TestCase):
    def setUp(self):
        self.recipe = yaml.safe_load(
            (ROOT / "deepseek-v4.1-flash-mxfp4-tp4.yaml").read_text()
        )

    def test_serving_contract(self):
        defaults = self.recipe["defaults"]
        self.assertEqual(self.recipe["min_nodes"], 4)
        self.assertEqual(self.recipe["max_nodes"], 4)
        self.assertEqual(defaults["tensor_parallel"], 4)
        self.assertEqual(defaults["max_model_len"], 262144)
        self.assertEqual(self.recipe["env"]["DSV41_ENGRAM_DISK"], "1")
        self.assertFalse(self.recipe["distribution_config"]["models"]["enabled"])
        speculation = json.loads(defaults["speculative_config"])
        self.assertEqual(speculation["method"], "dspark")
        self.assertEqual(speculation["num_speculative_tokens"], 5)
        self.assertFalse(speculation["enable_adaptive_verification"])
        self.assertIn("--tool-call-parser deepseek_v41", self.recipe["command"])
        self.assertIn("--reasoning-parser deepseek_v41", self.recipe["command"])
        self.assertIn("--block-size 128", self.recipe["command"])

    def test_pinned_patch_set(self):
        expected = {
            "attention.py": "da9ef19608848b6686c17300108610af",
            "engram.py": "c0329107bf338f6736063461e32ed539",
            "flashinfer_sparse.py": "af0f84473af44164f7050a7300937cb0",
            "model_state.py": "0a14bee67f103f1d616c6044134c0bab",
            "sparse_attn_indexer.py": "a9b7375619feb97b7299caa71346e8ad",
            "sparse_swa.py": "cc4193539ac4409ef1e2c99f8f909640",
            "weight_utils.py": "7e1027f15bc1f649bc3d2635e8556ee2",
        }
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    hashlib.md5((ROOT / "patches" / name).read_bytes()).hexdigest(),
                    digest,
                )

    def test_build_stages_python_package_not_checkout_root(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        build_script = (ROOT / "build-image.sh").read_text()
        self.assertIn("COPY vllm-package/", dockerfile)
        self.assertIn('cp -R "$VLLM_DIR/vllm"', build_script)
        self.assertNotIn('ln -s "$VLLM_DIR"', build_script)


if __name__ == "__main__":
    unittest.main()
