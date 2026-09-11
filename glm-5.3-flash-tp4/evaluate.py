# SPDX-License-Identifier: Unlicense
"""Matched TP4 probes; generated code is recorded, never executed.

Use the sibling benchmark for arithmetic, Korean, JSON, and long concurrent
decode. This adds completed prose/code, aggregate throughput, and near-256K
retrieval. Run against an idle endpoint; results include reasoning tokens.
"""

import argparse
import concurrent.futures
import json
import runpy
import time
import urllib.request
from pathlib import Path

REQUEST = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "glm-5.3-flash-nvfp4/benchmark.py")
)["request"]


def retrieval_prompt():
    filler = "The orchard has apple trees. The river runs beside the orchard.\n"
    return (
        "Read the record and retrieve its three audit codes.\n"
        "BEGIN audit code: AMBER-7319\n"
        + filler * 8550
        + "MIDDLE audit code: COBALT-4826\n"
        + filler * 8550
        + "END audit code: JADE-9053\n"
        + "\nReturn only JSON with keys begin, middle, end and their audit codes."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near-limit", action="store_true")
    args = parser.parse_args()
    rows = []

    def probe(name, prompt, tokens=1024, structured=False):
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": tokens,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        if structured:
            payload["response_format"] = {"type": "json_object"}
        result = REQUEST(args.base, payload, timeout=1800)
        result["name"] = name
        result["passed"] = (
            result["finish_reason"] == "stop"
            and bool(result["content"].strip())
            and not result["corruption"]
            and not result["repetition"]
        )
        usage = result["usage"]
        # Excludes prefill, but includes decoding of reasoning and final content.
        decode_seconds = result["seconds"] - result["ttft_seconds"]
        result["decode_tokens_per_second"] = (
            (usage["completion_tokens"] - 1) / decode_seconds
            if decode_seconds > 0 and usage["completion_tokens"] > 1
            else None
        )
        return result

    def record(result):
        rows.append(result)
        args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ("content", "reasoning", "tool_deltas")
                }
            ),
            flush=True,
        )

    for repeat in range(3):
        for kind, prompt in (
            (
                "code",
                "Write a Python function merge_intervals(intervals) that merges overlapping integer intervals. Include three short assert examples. Output only code, under 300 words.",
            ),
            (
                "prose",
                "Explain why a distributed database can return stale reads and how a client can get read-your-writes consistency. Use three paragraphs, under 220 words.",
            ),
        ):
            record(probe(f"{kind}-{repeat}", prompt))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(
                lambda n: probe(
                    f"concurrent-code-{n}",
                    f"Example {n}: write a Python function clamp_{n}(x, low, high) and 10 short assert examples. Output only code.",
                ),
                range(4),
            )
        )
    elapsed = time.monotonic() - started
    for result in results:
        record(result)
    record(
        {
            "name": "aggregate-c4",
            "seconds": elapsed,
            "output_tokens_per_second": sum(
                r["usage"]["completion_tokens"] for r in results
            )
            / elapsed,
            "passed": all(r["passed"] for r in results),
        }
    )
    if args.near_limit:
        result = probe("near-256k-retrieval", retrieval_prompt(), structured=True)
        try:
            correct = json.loads(result["content"]) == {
                "begin": "AMBER-7319",
                "middle": "COBALT-4826",
                "end": "JADE-9053",
            }
        except ValueError:
            correct = False
        result["passed"] = (
            result["passed"]
            and correct
            and 250000 <= result["usage"]["prompt_tokens"] < 262144
        )
        record(result)
    with urllib.request.urlopen(args.base + "/metrics", timeout=10) as response:
        args.output.with_suffix(".metrics.txt").write_bytes(response.read())
    if not all(r["passed"] for r in rows):
        raise SystemExit("Probe failed; inspect recorded replies")


if __name__ == "__main__":
    main()
