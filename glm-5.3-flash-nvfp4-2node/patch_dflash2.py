# SPDX-License-Identifier: Apache-2.0
"""Fail-closed build-time compatibility patch for vLLM's built-in DFlash2.

This deliberately patches only the current GLM-5.3 target model and KV-cache
grouping.  It does not replace or register a speculator.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

DEFAULT_ROOT = Path("/usr/local/lib/python3.12/dist-packages")
MODEL_REL = Path("vllm/models/glm5next/nvidia/model.py")
KV_REL = Path("vllm/v1/core/kv_cache_utils.py")
MARKER = "DFLASH2-RUNTIME-COMPAT"


MODEL_EDITS = [
    (
        "interfaces",
        """from vllm.model_executor.models.interfaces import (\n    HasInnerState,\n    IsHybrid,\n    MixtureOfExperts,\n    SupportsPP,\n)\n""",
        """from vllm.model_executor.models.interfaces import (\n    EagleModelMixin,\n    HasInnerState,\n    IsHybrid,\n    MixtureOfExperts,\n    SupportsEagle3,\n    SupportsPP,\n)\n""",
    ),
    (
        "target mixin",
        "class Glm5NextModel(nn.Module):\n",
        "class Glm5NextModel(nn.Module, EagleModelMixin):\n",
    ),
    (
        "target forward return type",
        """        **kwargs,\n    ) -> torch.Tensor:\n        if get_pp_group().is_first_rank:\n""",
        """        **kwargs,\n    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:\n        if get_pp_group().is_first_rank:\n""",
    ),
    (
        "causal forward return type",
        """    ) -> torch.Tensor | IntermediateTensors:\n        hidden_states = self.model(\n""",
        """    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:\n        hidden_states = self.model(\n""",
    ),
    (
        "aux capture",
        """        for layer in self._active_layers:\n            hidden_states, residual, post, comb = layer(\n                positions, hidden_states, residual, post, comb\n            )\n""",
        """        # DFLASH2-RUNTIME-COMPAT: EAGLE-3 target auxiliaries.\n        aux_hidden_states: list[torch.Tensor] = []\n        for idx, layer in enumerate(self._active_layers, start=self.start_layer):\n            hidden_states, residual, post, comb = layer(\n                positions, hidden_states, residual, post, comb\n            )\n            if idx + 1 in self.aux_hidden_state_layers:\n                # Materialize deferred mHC reconstruction, then contract its\n                # streams.  Do not mutate the state consumed by the next layer.\n                aux = (\n                    hc_contract(\n                        layer.hc_post(hidden_states, residual, post, comb), layer.n\n                    )\n                    if post is not None\n                    else hidden_states\n                )\n                if self.is_sequence_parallel:\n                    aux = sp_all_gather(aux)[:full_num_tokens]\n                aux_hidden_states.append(aux)\n""",
    ),
    (
        "aux return",
        """        hidden_states = self.norm(hidden_states)\n        return hidden_states\n""",
        """        hidden_states = self.norm(hidden_states)\n        if aux_hidden_states:\n            return hidden_states, aux_hidden_states\n        return hidden_states\n""",
    ),
    (
        "causal interface",
        """class Glm5NextForCausalLM(\n    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid\n):\n""",
        """class Glm5NextForCausalLM(\n    nn.Module, HasInnerState, SupportsPP, SupportsEagle3, MixtureOfExperts, IsHybrid\n):\n""",
    ),
    (
        "conditional interface",
        """class Glm5NextForConditionalGeneration(\n    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts\n):\n""",
        """class Glm5NextForConditionalGeneration(\n    Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts,\n    SupportsEagle3\n):\n""",
    ),
]


KV_EDITS = [
    (
        "partition exact sliding-window specs",
        """    attn_specs = {\n        name: spec\n        for name, spec in kv_cache_spec.items()\n        if not isinstance(spec, (MambaSpec, KpoolTailSpec))\n    }\n""",
        """    # DFLASH2-RUNTIME-COMPAT: KpoolTailSpec is an SWA subclass; only\n    # exact SlidingWindowSpec instances belong to the built-in drafter.\n    draft_specs = {\n        name: replace(spec, page_size_padded=None)\n        for name, spec in kv_cache_spec.items()\n        if type(spec) is SlidingWindowSpec\n    }\n    attn_specs = {\n        name: spec\n        for name, spec in kv_cache_spec.items()\n        if not isinstance(spec, (MambaSpec, KpoolTailSpec))\n        and type(spec) is not SlidingWindowSpec\n    }\n""",
    ),
    (
        "append drafter group",
        """    return (\n        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]\n        + ([tail_group] if tail_group is not None else [])\n        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n    )\n""",
        """    draft_group = None\n    if draft_specs:\n        any_draft = next(iter(draft_specs.values()))\n        assert all(spec == any_draft for spec in draft_specs.values())\n        # Padding was cleared while partitioning, before deriving geometry.\n        draft_bpt = any_draft.page_size_bytes // any_draft.block_size\n        mla_block = mla_specs[mla_names[0]].block_size\n        fit_block = mla_page // draft_bpt if mla_page % draft_bpt == 0 else 0\n        exact_fit = (\n            fit_block > 0\n            and fit_block % 64 == 0\n            and (fit_block % mla_block == 0 or mla_block % fit_block == 0)\n            and len(draft_specs) <= len(mla_names)\n        )\n        if exact_fit:\n            draft_specs = {\n                name: replace(spec, block_size=fit_block, page_size_padded=None)\n                for name, spec in draft_specs.items()\n            }\n        draft_uniform = UniformTypeKVCacheSpecs.from_specs(draft_specs)\n        assert draft_uniform is not None\n        draft_group = KVCacheGroupSpec(list(draft_specs), draft_uniform)\n\n    return (\n        [KVCacheGroupSpec(list(attn_specs), uniform_spec)]\n        + ([tail_group] if tail_group is not None else [])\n        + create_kv_cache_group_specs(padded_specs, mamba_grouped_names)\n        + ([draft_group] if draft_group is not None else [])\n    )\n""",
    ),
    (
        "layout tuple annotation",
        """        list[str],\n        int,\n    ]\n""",
        """        list[str],\n        int,\n        KVCacheGroupSpec | None,\n    ]\n""",
    ),
    (
        "layout detect",
        """    attn_group: KVCacheGroupSpec | None = None\n    tail_group: KVCacheGroupSpec | None = None\n    for group in uniform_groups:\n        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs\n        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):\n            attn_group = group\n        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):\n            tail_group = group\n""",
        """    attn_group: KVCacheGroupSpec | None = None\n    tail_group: KVCacheGroupSpec | None = None\n    draft_group: KVCacheGroupSpec | None = None\n    for group in uniform_groups:\n        inner = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).kv_cache_specs\n        if all(type(spec) is MLAAttentionSpec for spec in inner.values()):\n            attn_group = group\n        elif all(isinstance(spec, KpoolTailSpec) for spec in inner.values()):\n            tail_group = group\n        elif inner and all(type(spec) is SlidingWindowSpec for spec in inner.values()):\n            draft_group = group\n""",
    ),
    (
        "layout validate",
        """    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):\n        return None\n\n    tail_names: list[str] = []\n""",
        """    if any(group.kv_cache_spec.page_size_bytes != mla_page for group in mamba_groups):\n        return None\n    if draft_group is not None:\n        draft_inner = cast(\n            UniformTypeKVCacheSpecs, draft_group.kv_cache_spec\n        ).kv_cache_specs\n        draft_pages = {spec.page_size_bytes for spec in draft_inner.values()}\n        if len(draft_pages) != 1 or any(\n            spec.page_size_padded is not None for spec in draft_inner.values()\n        ):\n            return None\n        if draft_pages.pop() == mla_page and len(draft_group.layer_names) > len(mla_names):\n            return None\n\n    tail_names: list[str] = []\n""",
    ),
    (
        "layout return",
        """        tail_names,\n        tail_page,\n    )\n""",
        """        tail_names,\n        tail_page,\n        draft_group,\n    )\n""",
    ),
    (
        "pool accounting",
        """        _, _, mla_names, idx_names, mla_page, idx_page, _, _ = glm5_layout\n        return len(mla_names) * mla_page + len(idx_names) * idx_page\n""",
        """        _, _, mla_names, idx_names, mla_page, idx_page, _, _, draft_group = glm5_layout\n        result = len(mla_names) * mla_page + len(idx_names) * idx_page\n        if draft_group is not None:\n            draft_specs = cast(\n                UniformTypeKVCacheSpecs, draft_group.kv_cache_spec\n            ).kv_cache_specs\n            draft_spec = next(iter(draft_specs.values()))\n            if draft_spec.page_size_bytes != mla_page:\n                result += len(draft_group.layer_names) * draft_spec.page_size_bytes\n        return result\n""",
    ),
    (
        "config destructure and accounting",
        """            tail_names,\n            _,\n        ) = glm5_layout\n        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n""",
        """            tail_names,\n            _,\n            draft_group,\n        ) = glm5_layout\n        draft_names = [] if draft_group is None else list(draft_group.layer_names)\n        draft_specs = (\n            {}\n            if draft_group is None\n            else cast(UniformTypeKVCacheSpecs, draft_group.kv_cache_spec).kv_cache_specs\n        )\n        draft_page = (\n            0 if not draft_specs else next(iter(draft_specs.values())).page_size_bytes\n        )\n        draft_shared = bool(draft_names) and draft_page == mla_page\n        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n        if draft_names and not draft_shared:\n            bytes_per_block += len(draft_names) * draft_page\n""",
    ),
    (
        "config exact-fit consumers",
        """            for group in mamba_groups:\n                if index < len(group.layer_names):\n                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)\n\n        idx_base = len(mla_names) * mla_page * num_blocks\n""",
        """            for group in mamba_groups:\n                if index < len(group.layer_names):\n                    add_tensor(group.layer_names[index], group.kv_cache_spec, offset)\n            if draft_shared and index < len(draft_names):\n                name = draft_names[index]\n                add_tensor(name, draft_specs[name], offset)\n\n        idx_base = len(mla_names) * mla_page * num_blocks\n""",
    ),
    (
        "config standalone consumers",
        """                add_tensor(tail_name, tail_specs[tail_name], offset)\n\n        return KVCacheConfig(\n""",
        """                add_tensor(tail_name, tail_specs[tail_name], offset)\n\n        if draft_names and not draft_shared:\n            draft_base = (\n                len(mla_names) * mla_page + len(idx_names) * idx_page\n            ) * num_blocks\n            for index, name in enumerate(draft_names):\n                add_tensor(name, draft_specs[name], draft_base + index * draft_page * num_blocks)\n\n        return KVCacheConfig(\n""",
    ),
    (
        "max-memory destructure",
        """            tail_names,\n            _,\n        ) = glm5_layout\n        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n""",
        """            tail_names,\n            _,\n            draft_group,\n        ) = glm5_layout\n        uniform_spec = cast(UniformTypeKVCacheSpecs, attn_group.kv_cache_spec)\n""",
    ),
    (
        "max-memory drafter consumer",
        """        if tail_names:\n            total_blocks += 1\n        return total_blocks * (len(mla_names) * mla_page + len(idx_names) * idx_page)\n""",
        """        if tail_names:\n            total_blocks += 1\n        bytes_per_block = len(mla_names) * mla_page + len(idx_names) * idx_page\n        if draft_group is not None:\n            draft_uniform = cast(UniformTypeKVCacheSpecs, draft_group.kv_cache_spec)\n            total_blocks += draft_uniform.max_memory_usage_pages(vllm_config)\n            draft_page = next(iter(draft_uniform.kv_cache_specs.values())).page_size_bytes\n            if draft_page != mla_page:\n                bytes_per_block += len(draft_group.layer_names) * draft_page\n        return total_blocks * bytes_per_block\n""",
    ),
]


def _apply(path: Path, edits: list[tuple[str, str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"{path}: already patched")
        return
    updated = text
    for name, anchor, replacement in edits:
        count = updated.count(anchor)
        if count != 1:
            raise RuntimeError(
                f"{path}: anchor {name!r} matched {count} times (expected 1); "
                "refusing to patch a drifted image"
            )
        updated = updated.replace(anchor, replacement, 1)
    ast.parse(updated, filename=str(path))
    path.write_text(updated, encoding="utf-8")
    print(f"{path}: applied {len(edits)} bounded DFlash2 edits")


def patch(root: Path) -> None:
    model, kv = root / MODEL_REL, root / KV_REL
    if not model.is_file() or not kv.is_file():
        raise RuntimeError(f"expected vLLM files below {root}")
    # Validate both files before writing either one (build-time fail closed).
    for path, edits in ((model, MODEL_EDITS), (kv, KV_EDITS)):
        text = path.read_text(encoding="utf-8")
        if MARKER not in text:
            for name, anchor, _ in edits:
                if text.count(anchor) != 1:
                    raise RuntimeError(
                        f"{path}: anchor {name!r} drifted; refusing patch"
                    )
    _apply(model, MODEL_EDITS)
    _apply(kv, KV_EDITS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", type=Path, default=DEFAULT_ROOT)
    patch(parser.parse_args().site_packages)


if __name__ == "__main__":
    main()
