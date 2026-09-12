# SPDX-License-Identifier: Unlicense
import hashlib
import json
import os
import subprocess
import tempfile
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

    def test_storage_paths_are_portable(self):
        model_path = self.recipe["cluster_config"]["resolved_model_path"]
        volumes = self.recipe["executor_config"]["volumes"]
        tracked_text = "\n".join(
            path.read_text()
            for path in (
                ROOT / "deepseek-v4.1-flash-mxfp4-tp4.yaml",
                ROOT / "setup-model.sh",
                ROOT / "prepare-engram-home.sh",
                ROOT / "engram_local.py",
                ROOT / "README.md",
                ROOT / "RESULTS.md",
            )
        )
        self.assertEqual(model_path, "/srv/sparkrun/models/DeepSeek-V4.1-Flash")
        self.assertIn("{resolved_model_path}", self.recipe["command"])
        self.assertIn(
            "/srv/sparkrun/engram/DeepSeek-V4.1-Flash:/engram-local:ro",
            volumes,
        )
        self.assertNotIn("/home/", tracked_text)
        self.assertNotIn("NFS", tracked_text)

    def test_setup_stages_every_tensor_parallel_rank(self):
        setup = (ROOT / "setup-model.sh").read_text()
        self.assertIn("'1:0:96000564 14:0:96003054'", setup)
        self.assertIn("for rank in 0 1 2 3", setup)
        self.assertNotIn("if (( rank == 0 ))", setup)
        self.assertNotIn("LOCAL_ENGRAM_DIR", setup)

    def test_engram_path_resolves_each_remote_home(self):
        helper = ROOT / "prepare-engram-home.sh"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for hostname in ("host-a", "host-b"):
                home = root / hostname / "home"
                mount_source = root / hostname / "mount-source"
                home.mkdir(parents=True)
                env = {**os.environ, "HOME": str(home)}
                result = subprocess.run(
                    ["bash", helper, mount_source],
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                expected = (
                    home
                    / ".local/share/sparkrun/engram/DeepSeek-V4.1-Flash"
                )
                self.assertEqual(result.stdout.strip(), str(expected))
                self.assertTrue(expected.is_dir())
                self.assertTrue(mount_source.is_symlink())
                self.assertEqual(mount_source.resolve(), expected.resolve())

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
