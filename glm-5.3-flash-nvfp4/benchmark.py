# SPDX-License-Identifier: Unlicense
"""Bounded API quality/throughput comparison; records full replies and metrics.

No model-generated code is executed. Throughput uses API token usage, not SSE
chunk counts. Run on an otherwise idle server for meaningful comparisons.
"""

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from pathlib import Path


def request(base, payload):
    started = time.monotonic()
    first = None
    content, reasoning, calls = [], [], []
    usage, finish = None, None
    payload = dict(payload, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    done = False
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if time.monotonic() - started > 240:
                raise TimeoutError("request exceeded four-minute wall-clock budget")
            if not line.startswith(b"data: "):
                continue
            if line.strip() == b"data: [DONE]":
                done = True
                break
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if (
                    delta.get("content")
                    or delta.get("reasoning")
                    or delta.get("tool_calls")
                ):
                    first = first or time.monotonic()
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning") or "")
                calls.extend(delta.get("tool_calls") or [])
                finish = choice.get("finish_reason") or finish
    elapsed = time.monotonic() - started
    if not done or usage is None:
        raise RuntimeError("missing SSE completion marker or token usage")
    output = "".join(content)
    thought = "".join(reasoning)
    return {
        "content": output,
        "reasoning": thought,
        "tool_deltas": calls,
        "finish_reason": finish,
        "usage": usage,
        "seconds": elapsed,
        "ttft_seconds": None if first is None else first - started,
        "output_tokens_per_second": usage["completion_tokens"] / elapsed,
        "corruption": "\ufffd" in output + thought,
        "repetition": bool(re.search(r"(.{32,}?)\1{5,}", output + thought, re.DOTALL)),
    }


CASES = [
    ("arithmetic", "What is 173 * 29? Answer with only the integer.", "5017"),
    (
        "korean",
        "Repeat exactly this Korean text, without explanation: 안녕하세요 세계",
        "안녕하세요 세계",
    ),
    (
        "json",
        'Return only JSON: sort [9,2,9,-1,4] ascending, remove duplicates, and put the result in key "values".',
        {"values": [-1, 2, 4, 9]},
    ),
    (
        "debug",
        "A Python function uses `items=[]` as a default and appends each request to it. Explain the bug and give a corrected function `collect(item, items=None)`. Keep the answer under 200 words.",
        None,
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="GLM-5.3-Flash-NVFP4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reasoning", choices=["low", "default"], default="default")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--quality-stress", action="store_true")
    args = parser.parse_args()
    rows = []

    def run(name, prompt, expected=None, max_tokens=1024):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": max_tokens,
        }
        if isinstance(expected, dict):
            payload["response_format"] = {"type": "json_object"}
        if args.reasoning != "default":
            payload["chat_template_kwargs"] = {"reasoning_effort": args.reasoning}
        result = request(args.base, payload)
        result["name"] = name
        answer = result["content"].strip()
        passed = result["finish_reason"] == "stop" and bool(answer)
        if isinstance(expected, dict):
            try:
                passed = passed and json.loads(answer) == expected
            except ValueError:
                passed = False
        elif expected is not None:
            passed = passed and answer == expected
        elif name == "debug":
            passed = passed and all(
                s in answer for s in ["items=None", "is None", "append"]
            )
        elif name.startswith("korean-stress"):
            passed = passed and len(re.findall(r"[가-힣]", answer)) >= 30
        # Performance prompts deliberately request more than the output cap.
        if name.startswith("perf"):
            passed = result["finish_reason"] in ["stop", "length"] and bool(answer)
        if name.startswith("perf-long"):
            passed = (
                passed
                and result["usage"]["prompt_tokens"] > 24000
                and result["usage"]["completion_tokens"] >= 100
            )
        result["passed"] = (
            passed and not result["corruption"] and not result["repetition"]
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ["content", "reasoning", "tool_deltas"]
                }
            ),
            flush=True,
        )
        return result

    for case in CASES:
        rows.append(run(*case))
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    for round_id in range(args.rounds):
        rows.append(
            run(
                f"perf-{round_id}",
                f"Example set {round_id}: write 40 short Python functions clamp_00 through clamp_39, each returning min(max(x, low), high). Output code only.",
                max_tokens=384,
            )
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows.extend(
            pool.map(
                lambda n: run(
                    f"concurrent-{n}",
                    f"What is {n} + 10? Answer only the number.",
                    str(n + 10),
                ),
                [11, 22, 33, 44],
            )
        )
    # Preserve complete short-context responses even if a long-context run fails.
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    if args.quality_stress:
        for round_id in range(3):
            rows.append(
                run(
                    f"korean-stress-{round_id}",
                    "한국어로만 답하세요. 대규모 언어 모델의 양자화가 무엇인지 네 문장으로 설명하세요. 코드나 목록을 쓰지 마세요.",
                    max_tokens=512,
                )
            )
            args.output.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
            )
    if args.long_context:
        long_prompt = (
            "The orchard has apple trees. The river runs beside the orchard.\n" * 2000
            + "\nIgnore the orchard. Write 40 short Python functions clamp_00 through clamp_39, each returning min(max(x, low), high). Output code only."
        )
        rows.append(run("perf-long", long_prompt, max_tokens=384))
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        # Exercise the fixed KV budget with long requests and concurrent decode,
        # not just a single long prompt and four nearly empty arithmetic replies.
        # Distinct first blocks prevent prefix sharing from hiding cache pressure.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(
                lambda n: run(
                    f"perf-long-concurrent-{n}",
                    f"Example set {n}.\n" + long_prompt,
                    max_tokens=384,
                ),
                range(4),
            ):
                rows.append(result)
                args.output.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
                )
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    metrics = urllib.request.urlopen(args.base + "/metrics", timeout=10).read()
    args.output.with_suffix(".metrics.txt").write_bytes(metrics)
    if not all(row["passed"] for row in rows):
        raise SystemExit("Quality checks failed; inspect recorded responses")


if __name__ == "__main__":
    main()
