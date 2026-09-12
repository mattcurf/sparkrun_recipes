# SPDX-License-Identifier: Unlicense
"""CPU-only contract checks; deployment probes remain required."""

import json
import unittest
from pathlib import Path

import yaml
from evaluate import retrieval_prompt

ROOT = Path(__file__).resolve().parent


class ProfilesTest(unittest.TestCase):
    def test_matched_profiles(self):
        paths = [
            ROOT / "glm-5.3-flash-nvfp4-tp4.yaml",
            ROOT.parent / "glm-5.3-flash-fp8-4node/glm-5.3-flash-fp8-tp4.yaml",
        ]
        profiles = [yaml.safe_load(path.read_text()) for path in paths]
        self.assertEqual(len(profiles), 2)
        for profile in profiles:
            with self.subTest(model=profile["model"]):
                defaults = profile["defaults"]
                self.assertEqual(profile["min_nodes"], 4)
                self.assertEqual(profile["max_nodes"], 4)
                self.assertEqual(defaults["tensor_parallel"], 4)
                self.assertEqual(defaults["max_model_len"], 262144)
                self.assertEqual(defaults["kv_cache_dtype"], "fp8")
                self.assertEqual(defaults["revision"], profile["model_revision"])
                self.assertRegex(defaults["revision"], r"^[0-9a-f]{40}$")
                target, draft = profile["distribution_config"]["models"]["entries"]
                self.assertEqual(target["name"], profile["model"])
                self.assertEqual(target["revision"], defaults["revision"])
                speculation = json.loads(defaults["speculative_config"])
                self.assertEqual(speculation["method"], "dflash")
                self.assertEqual(speculation["num_speculative_tokens"], 7)
                self.assertEqual(speculation["kv_cache_dtype"], "auto")
                self.assertEqual(speculation["model"], draft["name"])
                self.assertEqual(speculation["revision"], draft["revision"])
                self.assertIn("--enable-expert-parallel", profile["command"])
                self.assertIn("--enable-chunked-prefill", profile["command"])
                self.assertEqual(defaults["execution_flags"], "--no-enforce-eager")
                self.assertEqual(profile["env"]["OMP_NUM_THREADS"], "1")
                expected_kv_gib = 12 if profile["model"].startswith("nvidia/") else 8
                self.assertEqual(
                    defaults["kv_cache_memory_bytes"], expected_kv_gib * 2**30
                )
        for key in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "speculative_config",
            "execution_flags",
        ):
            self.assertEqual(profiles[0]["defaults"][key], profiles[1]["defaults"][key])

    def test_retrieval_positions(self):
        prompt = retrieval_prompt()
        size = len(prompt)
        self.assertLess(prompt.index("AMBER-7319") / size, 0.01)
        self.assertAlmostEqual(prompt.index("COBALT-4826") / size, 0.5, places=2)
        self.assertGreater(prompt.index("JADE-9053") / size, 0.99)
        for code in ("AMBER-7319", "COBALT-4826", "JADE-9053"):
            self.assertEqual(prompt.count(code), 1)


if __name__ == "__main__":
    unittest.main()
