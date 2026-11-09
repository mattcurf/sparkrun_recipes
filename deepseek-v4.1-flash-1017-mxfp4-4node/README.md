# DeepSeek V4.1 Flash 1017: TP4 with maximum thinking

This recipe serves the official mixed MXFP4/MXFP8
[1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on four DGX Sparks.
Thinking defaults to **on**.
Reasoning effort defaults to **100/100**.
The recipe retains prior reasoning for multi-step coding/tool conversations.

It enables V4.1 reasoning/tool parsers.
It enables automatic tool choice.
It enables vision.
It enables DSpark k5.

Sampling defaults to the model author's temperature 1.0.
The default top-p is 1.0.
Clients can override these defaults.

**Validation:** These checks passed against these weights:

- Default thinking.
- Automatic and forced tool calls.
- A tool-result round trip that retained reasoning.
- Incremental SSE.

A 97-scenario tool-use evaluation scored 94/100.
See [results and limitations](RESULTS.md).
The runtime and 256K context settings come from the [original recipe](../deepseek-v4.1-flash-mxfp4-4node/).

Validation does not cover full-length context, concurrency limits, vision, or repository-level coding quality for 1017.
The original recipe's scores do not measure this checkpoint.

## Why this still uses the pinned disk-Engram runtime

Checked on 2026-10-18:

- [vLLM #58560](https://github.com/vllm-project/vllm/pull/58560) merged a fix
  that prevents compressor-ring writes into the null KV block.
  It does **not** implement disk-backed Engram storage.
  The released v0.31.0 source does not contain that fix.
  Newer upstream source contains it.
- The four-GB10 report in
  [issue #13](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/13)
  used the `gitbisector/vllm` **fork**, not unmodified upstream.
  That fork included disk-Engram and SM12x changes.
  Its follow-up [#59689](https://github.com/vllm-project/vllm/pull/59689) closed without a merge.
- [Upstream Engram storage](https://github.com/vllm-project/vllm/blob/da9134bc89364465f68aa679564435936d157a14/vllm/config/engram.py)
  offers pinned CPU memory or GPU memory, not disk.
  A Spark's CPU and GPU share the same memory.
  The original runtime measured 81.58 GiB per rank.
  This excludes roughly 47 GiB of disk-backed tables.
  Moving those tables to pinned RAM is not safe on four Sparks because of memory limits.
- SM12x attention also needs compatible FlashInfer page-size support; see
  [#57292](https://github.com/vllm-project/vllm/pull/57292).
  A merged ring fix alone does not prove that an arbitrary released image works.

Thus, this recipe reuses `deepseek-v4.1-flash-sparkrun-v1`.
It uses the existing Tony Deangelo-derived patches.
It keeps the exact vLLM pin.
It does not claim patch-free upstream support.

**Do not replace it with a stock image.**
The `DSV41_ENGRAM_DISK` environment variables are patch-specific.

The SparkRun mod named `use-official-vllm` is a launcher setting.
Its name does not describe the image contents.

## Prepare and run

1. Open this recipe directory.
2. Build the shared runtime if you do not already have it locally:

```bash
../deepseek-v4.1-flash-mxfp4-4node/build-image.sh
```

The build needs substantial disk space and memory.
Follow the
[runtime build instructions](../deepseek-v4.1-flash-mxfp4-4node/README.md).
**Do not run the original recipe's `setup-model.sh`.**
That script downloads the old weights.

3. Prepare a writable directory for the **new** checkpoint.
   Use the same absolute path on every host.
   Allow at least 511 GB for the checkpoint.
   Allow additional space for download overhead.
   Use the example path below or your own storage path.
   If you change the path, update both the download command and `cluster_config.resolved_model_path`.
4. Download the new checkpoint once into this dedicated directory.
   Never overlay it onto the original checkpoint.
   Never reuse the original recipe's Engram row copies.
5. Free all four GPUs from other inference workloads before launch.
6. Run the validation, dry-run, and launch commands after the download:

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --local-dir /srv/sparkrun/models/DeepSeek-V4.1-Flash-1017 --max-workers 8

sparkrun recipe validate deepseek-v4.1-flash-1017-mxfp4-tp4.yaml
sparkrun run deepseek-v4.1-flash-1017-mxfp4-tp4.yaml \
  --cluster <four-node-cluster> --dry-run
sparkrun run deepseek-v4.1-flash-1017-mxfp4-tp4.yaml \
  --cluster <four-node-cluster>
```

Change `cluster_config.resolved_model_path` if your storage uses another path.
The recipe disables model distribution.
SparkRun does not download or verify an externally provisioned checkpoint.
Thus, use the pinned download above.

The API serves `deepseek-v4.1-flash-1017-mxfp4-tp4` on the head node's port 8000.

Fragmentation may stall model allocation despite sufficient free memory.
Stop inference before you try these recovery steps:

1. Use the repository's `tools/cleanup-memory.sh` to reclaim caches on the selected hosts.
2. If necessary, compact memory before startup with `echo 1 > /proc/sys/vm/compact_memory` as root.

**Do not reclaim caches or compact memory during serving.**
Allow substantial time for weight loading, draft-model loading, compilation, and warmup.
Use `/health` and an inference smoke test to judge readiness.
Do not rely on container uptime alone.

Engram lookups read directly from this checkpoint's files.
The recipe does not use an `/engram-local` mount.
It sets `DSV41_ENGRAM_DIR` explicitly empty.
This prevents use of old rows with compatible shapes but different weights.

Shared-storage latency can limit throughput.
The recipe keeps the existing 112 GiB container RAM limit.
See the original README's SparkRun 0.3.6 memory/swap limitation.

TileLang and Triton compile into SparkRun's persistent, node-local `/cache/runtime` mount.
They do not compile into model storage.
Model storage may be read-only or permission-restricted.
FlashInfer continues to use the kernels already built into the pinned image.

## Coding-agent requests

1. Point your agent's OpenAI-compatible client at `http://<head-node>:8000/v1`.
2. Select `deepseek-v4.1-flash-1017-mxfp4-tp4`.
3. Supply tools on every applicable request.
4. Preserve assistant `reasoning_content` and `tool_calls`.
5. Return tool results with the matching `tool_call_id`.

Do not remove reasoning from the conversation history.
The server cannot execute tools for the client.

The recipe sets maximum effort through `chat_template_kwargs`.
This tokenizer accepts the numeric 1–100 budget.
Avoid client overrides such as `reasoning_effort: "none"` or `thinking: false`.
For clients that send explicit template settings, use:

```json
{
  "model": "deepseek-v4.1-flash-1017-mxfp4-tp4",
  "messages": [{"role": "user", "content": "Implement and test a bounded LRU cache in Python."}],
  "temperature": 1.0,
  "top_p": 1.0,
  "max_tokens": 65536,
  "chat_template_kwargs": {
    "thinking": true,
    "reasoning_effort": 100,
    "drop_thinking": false
  }
}
```

The example output budget includes reasoning and final output.
Allow long request timeouts.
Prefer streaming.

Effort 100 is the model's maximum reasoning setting.
It does not guarantee the best accuracy for every task.

The **256K total context** is a conservative deployment starting point.
It is not the checkpoint's 1M architectural limit.
Prompt, retained reasoning, tool results, and new output must all fit.

DeepSeek recommends a 1M context.
It also recommends at least 256K output capacity.
This starting configuration cannot provide both.

To attempt the full context:

1. Use an idle cluster.
2. Override `--max-model-len 1048576 -o max_num_seqs=1`.
3. Validate memory and long-context retrieval before you rely on this override.

This recipe has no hardware validation for this override.

## Runtime acceptance checks

Before unattended coding, complete these checks:

1. Check for coherent, non-repeating text.
2. Check for non-empty reasoning followed by a final answer.
3. Check incremental SSE.
4. Check automatic and forced tool calls.
5. Check a tool-result round trip that retains reasoning.
6. Check retrieval near the configured context limit.
7. Confirm that all four ranks report disk-backed Engram from the **1017** directory.
8. Confirm that no rank reports the old `/engram-local` source.

Do not reuse the old recipe's benchmark/quality scores as 1017 results.

## License

Original contributions use the [Unlicense](LICENSE).
Adapted material retains its upstream terms, including the [Tech2wild MIT notice](LICENSE-MIT).
Download the model separately under its [MIT license](LICENSE-MODEL-MIT).
This directory does not distribute model weights, images, or additional runtime patches.
See [third-party notices and provenance](THIRD_PARTY_NOTICES.md).
