# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the bounded DFlash2 source overlay."""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import patch_dflash2

AUDIT = Path(
    os.environ.get("VLLM_SOURCE_ROOT", "/usr/local/lib/python3.12/dist-packages")
)


class DFlash2PatchTests(unittest.TestCase):
    def setUp(self) -> None:
        if not AUDIT.is_dir():
            self.skipTest(f"reference checkout unavailable: {AUDIT}")
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for rel in (patch_dflash2.MODEL_REL, patch_dflash2.KV_REL):
            dst = self.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(AUDIT / rel, dst)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_patch_current_sources_and_repeat_is_noop(self) -> None:
        patch_dflash2.patch(self.root)
        first = {
            rel: (self.root / rel).read_bytes()
            for rel in (patch_dflash2.MODEL_REL, patch_dflash2.KV_REL)
        }
        patch_dflash2.patch(self.root)
        for rel, content in first.items():
            self.assertEqual(content, (self.root / rel).read_bytes())
            ast.parse(content)

    def test_model_contract(self) -> None:
        patch_dflash2.patch(self.root)
        text = (self.root / patch_dflash2.MODEL_REL).read_text()
        self.assertIn("nn.Module, EagleModelMixin", text)
        self.assertIn("if idx + 1 in self.aux_hidden_state_layers", text)
        self.assertIn("hc_contract(\n                        layer.hc_post(", text)
        self.assertIn("aux = sp_all_gather(aux)[:full_num_tokens]", text)
        conditional = text[text.index("class Glm5NextForConditionalGeneration") :]
        self.assertIn("MixtureOfExperts", conditional.split("):", 1)[0])
        self.assertIn("SupportsEagle3", conditional.split("):", 1)[0])

    def test_kv_contract_and_consumers(self) -> None:
        patch_dflash2.patch(self.root)
        text = (self.root / patch_dflash2.KV_REL).read_text()
        self.assertIn("if type(spec) is SlidingWindowSpec", text)
        self.assertIn("replace(spec, page_size_padded=None)", text)
        self.assertLess(text.index("page_size_padded=None"), text.index("draft_bpt ="))
        self.assertIn("+ ([draft_group] if draft_group is not None else [])", text)
        self.assertGreaterEqual(text.count("draft_group"), 20)
        self.assertIn("bytes_per_block += len(draft_names) * draft_page", text)
        self.assertIn("total_blocks += draft_uniform.max_memory_usage_pages", text)

    def test_drift_rejected_without_partial_write(self) -> None:
        model = self.root / patch_dflash2.MODEL_REL
        kv = self.root / patch_dflash2.KV_REL
        original_model = model.read_bytes()
        kv.write_text(
            kv.read_text().replace("    attn_specs = {", "    attention_specs = {", 1)
        )
        with self.assertRaisesRegex(RuntimeError, "drift"):
            patch_dflash2.patch(self.root)
        self.assertEqual(original_model, model.read_bytes())

    def test_cli(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(Path(patch_dflash2.__file__)),
                "--site-packages",
                str(self.root),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
