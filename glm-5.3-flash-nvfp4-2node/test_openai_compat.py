# SPDX-License-Identifier: Unlicense
"""Exercise the installed chat-completion compatibility patch."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import patch_openai_compat
import vllm
from vllm.entrypoints.openai.chat_completion import protocol
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens

ROOT = Path(vllm.__file__).parent
MESSAGES = [{"role": "user", "content": "hi"}]
TOOL = {
    "type": "function",
    "function": {"name": "read_file", "parameters": {"type": "object"}},
}


class OpenAICompatTests(unittest.TestCase):
    def test_installed_patch_is_present(self):
        text = (ROOT / patch_openai_compat.PROTOCOL_REL).read_text()
        self.assertIn(patch_openai_compat.MARKER, text)

    def test_repeat_is_noop_and_drift_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / patch_openai_compat.PROTOCOL_REL
            copy.parent.mkdir(parents=True)
            shutil.copy2(ROOT / patch_openai_compat.PROTOCOL_REL, copy)
            before = copy.read_bytes()
            patch_openai_compat.patch(Path(tmp))
            self.assertEqual(before, copy.read_bytes())
            copy.write_text("unrelated source\n")
            with self.assertRaisesRegex(RuntimeError, "drift"):
                patch_openai_compat.patch(Path(tmp))

    def test_empty_tools_are_omitted(self):
        for choice in (None, "none", "auto"):
            with self.subTest(tool_choice=choice):
                extra = {} if choice is None else {"tool_choice": choice}
                req = ChatCompletionRequest(
                    model="m", messages=MESSAGES, tools=[], **extra
                )
                self.assertIsNone(req.tools)
                self.assertIn(req.tool_choice, (None, "none"))

    def test_real_tools_still_default_to_auto(self):
        req = ChatCompletionRequest(model="m", messages=MESSAGES, tools=[TOOL])
        self.assertEqual(req.tool_choice, "auto")

    def test_large_output_request_is_clamped_not_rejected(self):
        max_model_len = 262144
        req = ChatCompletionRequest(
            model="m", messages=MESSAGES, max_completion_tokens=128000
        )
        tok = req.build_tok_params(SimpleNamespace(max_model_len=max_model_len))
        self.assertEqual(tok.max_input_tokens, max_model_len - 1)
        prompt = 200000
        self.assertEqual(
            get_max_tokens(max_model_len, 128000, prompt, {}),
            max_model_len - prompt,
        )
        self.assertEqual(get_max_tokens(max_model_len, 128000, 1000, {}), 128000)

    def rendered_effort(self, cap, **request):
        with mock.patch.object(protocol, "_GLM53_MAX_EFFORT", cap):
            req = ChatCompletionRequest(model="m", messages=MESSAGES, **request)
            params = req.build_chat_params(None, "auto")
        return params.chat_template_kwargs.get("reasoning_effort")

    def test_reasoning_effort_is_capped(self):
        cases = [
            ("low", "high", "low"),
            ("low", "medium", "low"),
            ("low", "minimal", "low"),
            ("low", "low", "low"),
            ("low", "none", "none"),
            ("low", None, None),
            ("high", "max", "high"),
            ("high", "xhigh", "high"),
            ("high", "low", "low"),
            ("max", "high", "high"),
            (None, "high", "high"),
        ]
        for cap, effort, expected in cases:
            with self.subTest(cap=cap, effort=effort):
                self.assertEqual(
                    self.rendered_effort(cap, reasoning_effort=effort), expected
                )

    def test_template_kwargs_effort_is_capped(self):
        effort = self.rendered_effort(
            "low", chat_template_kwargs={"reasoning_effort": "high"}
        )
        self.assertEqual(effort, "low")

    def test_thinking_stays_enabled_when_capped(self):
        with mock.patch.object(protocol, "_GLM53_MAX_EFFORT", "low"):
            req = ChatCompletionRequest(
                model="m", messages=MESSAGES, reasoning_effort="high"
            )
            kwargs = req.build_chat_params(None, "auto").chat_template_kwargs
        self.assertIs(kwargs["enable_thinking"], True)

    def test_env_cap_is_read_and_validated(self):
        probe = (
            "import vllm.entrypoints.openai.chat_completion.protocol as p;"
            "print(p._GLM53_MAX_EFFORT)"
        )
        for value, ok in (("low", True), ("medium", False)):
            with self.subTest(value=value):
                env = {**os.environ, "GLM_MAX_REASONING_EFFORT": value}
                run = subprocess.run(
                    [sys.executable, "-c", probe],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                if ok:
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertEqual(run.stdout.strip().splitlines()[-1], "low")
                else:
                    self.assertNotEqual(run.returncode, 0)
                    self.assertIn("GLM_MAX_REASONING_EFFORT", run.stderr)


if __name__ == "__main__":
    unittest.main()
