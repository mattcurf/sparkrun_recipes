# SPDX-License-Identifier: Unlicense
"""Check the installed fast loader against vLLM's stock lazy safetensors path."""

import os
import random
import tempfile
import unittest

os.environ.update(
    GLM_FAST_LOAD_SLAB_MB="1",  # many slab boundaries and pass-through tensors
    GLM_FAST_LOAD_AHEAD_MB="3",
    GLM_FAST_LOAD_THREADS="4",
    GLM_FAST_LOAD_VERIFY="0",
)

import glm_fast_load
import torch
from safetensors.torch import safe_open, save_file

glm_fast_load.register()
from vllm.model_executor.model_loader import weight_utils

DTYPES = [torch.bfloat16, torch.float32, torch.uint8, torch.float8_e4m3fn, torch.int32]


def make_shards(directory, count=3):
    rng = random.Random(0)
    paths = []
    for shard in range(count):
        tensors = {}
        for i in range(80):
            dtype = rng.choice(DTYPES)
            shape = rng.choice([[], [0, 4], [700, 1024], [rng.randint(1, 64), 33]])
            data = torch.randn(shape) * 8 if shape else torch.tensor(rng.random())
            name = (
                f"model.layers.{shard}.mlp.experts.{i % 8}.w{i % 3}.weight"
                if i % 4 == 0
                else f"model.layers.{shard}.dense.{i}.weight"
            )
            tensors[name] = data.to(dtype)
        path = os.path.join(
            directory, f"model-{shard + 1:05d}-of-{count:05d}.safetensors"
        )
        save_file(tensors, path)
        paths.append(path)
    return paths


def stock(paths, local_expert_ids):
    for path in sorted(paths):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118
                if not weight_utils.should_skip_weight(name, local_expert_ids):
                    yield name, f.get_tensor(name)


def as_records(items):
    return [
        (
            name,
            t.dtype,
            tuple(t.shape),
            t.contiguous().reshape(-1).view(torch.uint8).clone(),
        )
        for name, t in items
    ]


class FastLoadTests(unittest.TestCase):
    def test_iterator_is_wrapped(self):
        self.assertTrue(
            getattr(weight_utils.safetensors_weights_iterator, "_glm_fast_load", False)
        )

    def test_matches_stock_lazy_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_shards(tmp)
            for experts in (None, {0, 1, 2, 3}):
                with self.subTest(local_expert_ids=experts):
                    expected = as_records(stock(paths, experts))
                    actual = as_records(
                        weight_utils.safetensors_weights_iterator(
                            paths, False, "lazy", local_expert_ids=experts
                        )
                    )
                    self.assertEqual([r[:3] for r in expected], [r[:3] for r in actual])
                    for e, a in zip(expected, actual):
                        self.assertTrue(torch.equal(e[3], a[3]), e[0])
                    if experts is not None:
                        self.assertLess(len(actual), sum(1 for _ in stock(paths, None)))


if __name__ == "__main__":
    unittest.main()
