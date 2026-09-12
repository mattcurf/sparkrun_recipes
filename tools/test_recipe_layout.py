# SPDX-License-Identifier: Unlicense
"""CPU-only checks for recipe naming, threading policy, and documentation links."""

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

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


if __name__ == "__main__":
    unittest.main()
