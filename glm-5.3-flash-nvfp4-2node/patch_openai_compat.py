# SPDX-License-Identifier: Apache-2.0
"""Chat-completion request compatibility for agent clients.

- Treat `"tools": []` as omitted tools, dropping a `tool_choice` of `none` or
  `auto` with it. Other `tool_choice` values keep vLLM's normal validation.
- Reserve one token, not the full `max_tokens`/`max_completion_tokens`, when
  checking prompt length. `get_max_tokens` still limits output to the context
  remaining after the prompt, so requests are clamped instead of rejected.

Source anchors are Apache-2.0 vLLM excerpts. Patching fails closed if an anchor
changes, and repeated application is a no-op.
"""

from pathlib import Path

PROTOCOL_REL = Path("entrypoints/openai/chat_completion/protocol.py")
MARKER = "# glm53: openai compat"

EMPTY_TOOLS_OLD = """        # Reject empty tools array, matching OpenAI API behavior
        if data.get("tools") == []:
            raise VLLMValidationError(
                "`tools` must not be an empty array. "
                "Either provide at least one tool or omit the field entirely.",
                parameter="tools",
            )
"""
EMPTY_TOOLS_NEW = f"""        {MARKER}: treat an empty tools array as omitted tools.
        if data.get("tools") == []:
            data = dict(data)
            del data["tools"]
            if data.get("tool_choice") in ("none", "auto"):
                del data["tool_choice"]
"""

OUTPUT_RESERVE_OLD = """            max_output_tokens=max_output_tokens or 0,
            truncate_prompt_tokens=self.truncate_prompt_tokens,"""
OUTPUT_RESERVE_NEW = f"""            {MARKER}: reserve one output token; get_max_tokens clamps output.
            max_output_tokens=min(max_output_tokens or 0, 1),
            truncate_prompt_tokens=self.truncate_prompt_tokens,"""


def patch(root: Path) -> None:
    path = root / PROTOCOL_REL
    source = path.read_text()
    if MARKER in source:
        return
    for old, new in (
        (EMPTY_TOOLS_OLD, EMPTY_TOOLS_NEW),
        (OUTPUT_RESERVE_OLD, OUTPUT_RESERVE_NEW),
    ):
        if source.count(old) != 1:
            raise RuntimeError(f"Pinned vLLM source drift: {old!r}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)


if __name__ == "__main__":
    import vllm

    patch(Path(vllm.__file__).parent)
