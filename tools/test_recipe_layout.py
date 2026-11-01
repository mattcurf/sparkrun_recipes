# SPDX-License-Identifier: Unlicense
"""CPU-only checks for recipe naming, threading policy, and documentation links."""

import importlib.util
import re
import shlex
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

try:
    from sparkrun.core.recipe import Recipe
except ImportError:
    Recipe = None

ROOT = Path(__file__).resolve().parents[1]


class RecipeLayoutTest(unittest.TestCase):
    def test_recipe_names_threads_and_index(self):
        paths = sorted(ROOT.glob("*/*.yaml")) + sorted(ROOT.glob("*/*.yml"))
        self.assertTrue(paths)
        index = (ROOT / "README.md").read_text()
        folders = set()
        for path in paths:
            with self.subTest(recipe=path.relative_to(ROOT)):
                folder = path.parent.name
                self.assertNotIn(folder, folders, "one recipe per directory")
                folders.add(folder)
                match = re.fullmatch(r"(.+)-(fp8|mxfp4|nvfp4|exl3)-(\d+)node", folder)
                self.assertIsNotNone(match)
                recipe = yaml.safe_load(path.read_text())
                nodes = int(match.group(3))
                self.assertEqual(recipe["min_nodes"], nodes)
                self.assertEqual(recipe["max_nodes"], nodes)
                self.assertEqual(recipe["env"]["OMP_NUM_THREADS"], "1")
                self.assertTrue((path.parent / "README.md").is_file())
                self.assertTrue((path.parent / "LICENSE").is_file())
                rows = [
                    line
                    for line in index.splitlines()
                    if line.startswith("|") and f"]({folder}/)" in line
                ]
                self.assertEqual(len(rows), 1, "exactly one index entry per recipe")
                context = re.search(r"(\d+)([KM]) context", rows[0])
                self.assertIsNotNone(context)
                tokens = int(context.group(1)) * 1024 ** (
                    1 if context.group(2) == "K" else 2
                )
                self.assertEqual(recipe["defaults"]["max_model_len"], tokens)

    def test_relative_markdown_links(self):
        for path in ROOT.rglob("*.md"):
            if ".git" in path.parts:
                continue
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                with self.subTest(document=path.relative_to(ROOT), target=target):
                    self.assertTrue((path.parent / unquote(url.path)).exists())


@unittest.skipIf(Recipe is None, "run with SparkRun's Python environment")
class TensorFoldRecipeTest(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node"
        self.recipe = Recipe.load(
            folder / "deepseek-v4.1-flash-1017-tensorfold-tp4.yaml"
        )
        spec = importlib.util.spec_from_file_location(
            "tensorfold_dsv41", folder / "runtime/tensorfold_dsv41.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.runtime = module.TensorFoldDSV41Runtime()

    def command(self, rank, overrides=None, nodes=4):
        return shlex.split(
            self.runtime.generate_node_command(
                self.recipe, overrides or {}, "192.0.2.10", nodes, rank
            ).replace("\\\n", "")
        )

    def test_defaults_preserve_live_serving_contract(self):
        head = self.command(0)
        for flag, expected in [
            ("--port", "8000"),
            ("--parallel", "16"),
            ("--context", "1048576"),
            ("--name", "deepseek-v4.1-flash-1017-mxfp4-tp4"),
            ("--alias", "DeepSeek-V4.1-Flash-1017"),
            ("--api-key-file", "/run/secrets/model-api-key"),
        ]:
            self.assertEqual(head[head.index(flag) + 1], expected)
        for rank in range(1, 4):
            with self.subTest(rank=rank):
                worker = self.command(rank)
                start = worker.index("tf-dsv41-lanes")
                self.assertEqual(
                    worker[start + 3 : start + 7],
                    [str(rank), "4", "10.10.1.228", "29760"],
                )
                self.assertEqual(worker[worker.index("--graphs") + 1], "0")
                self.assertEqual(worker[worker.index("--drafts") + 1], "1")
                self.assertIn("TF_DS_GRAPHS=0", worker)
                self.assertNotIn("--api-key-file", worker)

    def test_overrides_agree_on_head_and_every_worker(self):
        overrides = {
            "max_model_len": 131072,
            "pool": 262144,
            "max_num_seqs": 7,
            "graphs": 1,
            "drafts": 0,
        }
        for rank in range(4):
            with self.subTest(rank=rank):
                cmd = self.command(rank, overrides)
                self.assertEqual(cmd[cmd.index("--context") + 1], "131072")
                parallel = "--parallel" if rank == 0 else "--streams"
                self.assertEqual(cmd[cmd.index(parallel) + 1], "7")
                for expected in [
                    "TF_DS_WINDOW=262144",
                    "TF_DS_STREAMS=7",
                    "TF_DS_GRAPHS=1",
                ]:
                    self.assertIn(expected, cmd)
                if rank == 0:
                    self.assertIn("--no-drafts", cmd)
                else:
                    self.assertEqual(cmd[cmd.index("--pool") + 1], "262144")
                    self.assertEqual(cmd[cmd.index("--graphs") + 1], "1")
                    self.assertEqual(cmd[cmd.index("--drafts") + 1], "0")

    def test_invalid_layout_and_pool_fail_before_launch(self):
        for nodes in (1, 3, 5):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                self.command(0, nodes=nodes)
        with self.assertRaises(ValueError):
            self.command(0, {"tensor_parallel": 2})
        with self.assertRaises(ValueError):
            self.command(0, {"max_model_len": 131072, "pool": 131071})
        self.command(0, {"max_model_len": 131072, "pool": 131072})
        with self.assertRaises(ValueError):
            self.runtime.prepare(
                self.recipe,
                list(reversed(self.recipe.runtime_config["hosts"])),
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
