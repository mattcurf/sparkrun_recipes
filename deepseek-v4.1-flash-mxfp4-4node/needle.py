#!/usr/bin/env python3
"""Long-context needle test for DeepSeek-V4.1-Flash.

Distinct facts are buried near the beginning, middle, and end of deterministic
filler. The model must retrieve all three and continue coherently for at least
256 output tokens. Measures actual server prompt tokens and TTFT. Temperature 0,
thinking off.

usage: needle.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash-mxfp4-tp4 \
         --targets 131072,255000 [--out results.json]
"""

import argparse
import json
import random
import re
import time
import urllib.request

WORDS = [
    "amber",
    "basin",
    "cedar",
    "delta",
    "ember",
    "fjord",
    "garnet",
    "harbor",
    "iris",
    "juniper",
    "kestrel",
    "lumen",
    "meadow",
    "nimbus",
    "orchid",
    "pylon",
    "quartz",
    "raven",
    "sierra",
    "tundra",
    "umber",
    "vessel",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
]
KEYS = ("COPPER-LANTERN-8315", "JADE-HARBOR-2046", "SILVER-ORCHID-7923")
NEEDLES = tuple(
    f"Important record {i + 1}: the verification key is {key}."
    for i, key in enumerate(KEYS)
)
QUESTION = (
    "\n\nReturn the three verification keys in document order on the first line. "
    "Then write at least 280 tokens of coherent plain prose explaining why accurate "
    "retrieval from long records matters. Do not use headings or repeat phrases."
)


def filler(n_words, seed):
    rnd = random.Random(seed)
    out = []
    for i in range(n_words):
        out.append(f"{rnd.choice(WORDS)}{rnd.randint(0, 999)}")
        if i % 16 == 15:
            out.append(".")
    return " ".join(out)


def chat(base, model, prompt, max_tokens, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = None
    text = ""
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices") or []:
                piece = (ch.get("delta") or {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                text += piece
    return {"ttft_s": ttft, "total_s": time.time() - t0, "usage": usage, "text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="deepseek-v4.1-flash-mxfp4-tp4")
    ap.add_argument("--targets", default="131072")
    ap.add_argument("--out")
    args = ap.parse_args()
    probe = chat(args.base, args.model, filler(4000, 1), 1, 600)
    per_word = probe["usage"]["prompt_tokens"] / 4000.0
    print(f"calibration: {per_word:.3f} prompt tokens per filler word", flush=True)
    rows = []
    for tgt in [int(x) for x in args.targets.split(",")]:
        n = int((tgt - 200) / per_word)
        cuts = (int(n * 0.01), int(n * 0.50), int(n * 0.99))
        sections = []
        previous = 0
        for i, cut in enumerate(cuts):
            sections.extend((filler(cut - previous, tgt + i), NEEDLES[i]))
            previous = cut
        sections.append(filler(n - previous, tgt + len(cuts)))
        doc = " ".join(sections)
        r = chat(
            args.base,
            args.model,
            f"[needle {tgt}] " + doc + QUESTION,
            320,
            3600,
        )
        pt = (r["usage"] or {}).get("prompt_tokens")
        answer = r["text"].upper().replace(" ", "")
        positions = [answer.find(key) for key in KEYS]
        completion_tokens = (r["usage"] or {}).get("completion_tokens", 0)
        junk = len(re.findall(r"[^\x09\x0a\x0d\x20-\x7e]", r["text"]))
        ok = (
            all(pos >= 0 for pos in positions)
            and positions == sorted(positions)
            and completion_tokens >= 256
            and junk / max(len(r["text"]), 1) < 0.01
        )
        row = {
            "target": tgt,
            "prompt_tokens": pt,
            "depths": [0.01, 0.50, 0.99],
            "ttft_s": round(r["ttft_s"] or 0, 1),
            "prefill_tok_s": round(pt / r["ttft_s"], 1) if pt and r["ttft_s"] else None,
            "completion_tokens": completion_tokens,
            "answer": r["text"].strip()[:160],
            "pass": ok,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
    if not all(row["pass"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
