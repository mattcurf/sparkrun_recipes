# DeepSeek V4.1 Flash 1017: TP4 with maximum thinking

Serves the official mixed MXFP4/MXFP8
[1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on four DGX Sparks. Thinking defaults to **on**, reasoning effort to **100/100**,
and prior reasoning is retained for multi-step coding/tool conversations.
V4.1 reasoning/tool parsers, automatic tool choice, vision, and DSpark k5 are
enabled. Sampling defaults to the model author's temperature 1.0 and top-p 1.0;
clients can override these defaults.

**Validation:** default thinking, automatic and forced tool calls, a tool-result
round trip retaining reasoning, and incremental SSE passed against these
weights. A 97-scenario tool-use evaluation scored 94/100; see
[results and limitations](RESULTS.md). The runtime and 256K context settings
come from the [original recipe](../deepseek-v4.1-flash-mxfp4-4node/).
Full-length context, concurrency limits, vision, and repository-level coding
quality have not been validated for 1017. The original recipe's scores are
not measurements of this checkpoint.

## Why this still uses the pinned disk-Engram runtime

Checked on 2026-10-18:

- [vLLM #58560](https://github.com/vllm-project/vllm/pull/58560) merged a fix
  preventing compressor-ring writes into the null KV block. It does **not**
  implement disk-backed Engram storage. The released v0.31.0 source does not
  contain that fix; newer upstream source does.
- The four-GB10 report in
  [issue #13](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/13)
  used the `gitbisector/vllm` **fork**, including disk-Engram and SM12x changes,
  not unmodified upstream. Its follow-up
  [#59689](https://github.com/vllm-project/vllm/pull/59689) was closed unmerged.
- [Upstream Engram storage](https://github.com/vllm-project/vllm/blob/da9134bc89364465f68aa679564435936d157a14/vllm/config/engram.py)
  offers pinned CPU memory or GPU memory, not disk. A Spark's CPU and GPU share
  the same memory. The original runtime measured 81.58 GiB per rank excluding
  roughly 47 GiB of disk-backed tables: moving those tables to pinned RAM is
  not a safe fit on four Sparks.
- SM12x attention also needs compatible FlashInfer page-size support; see
  [#57292](https://github.com/vllm-project/vllm/pull/57292). A merged ring fix
  alone is not evidence that an arbitrary released image works.

Consequently this recipe reuses `deepseek-v4.1-flash-sparkrun-v1`, with the
existing Tony Deangelo-derived patches and exact vLLM pin, rather than claiming
patch-free upstream support. **Do not replace it with a stock image:** the
`DSV41_ENGRAM_DISK` environment variables are patch-specific. The SparkRun mod
named `use-official-vllm` is a launcher setting, not a claim about image contents.

## Prepare and run

Run the following commands from this recipe directory. Build the shared
runtime if it is not already available locally:

```bash
../deepseek-v4.1-flash-mxfp4-4node/build-image.sh
```

The build has substantial disk/memory requirements; follow the
[runtime build instructions](../deepseek-v4.1-flash-mxfp4-4node/README.md).
Do not run the original recipe's `setup-model.sh`: it downloads the old weights.

Download the **new** checkpoint once into a dedicated directory visible at the
same absolute path on every host. Allow at least 511 GB for the checkpoint,
plus download overhead. Never overlay it onto the original checkpoint or reuse
the original recipe's Engram row copies. Provision a writable directory at
the example path below, or replace it with your own storage path in both the
download command and `cluster_config.resolved_model_path`.

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
Model distribution is disabled: SparkRun does not download or verify an
externally provisioned checkpoint, so use the pinned download above. The API
serves `deepseek-v4.1-flash-1017-mxfp4-tp4` on the head node's port 8000.
Launch only after freeing all four GPUs from other inference workloads.

If model allocation stalls despite sufficient free memory, fragmentation may
be a factor. With inference stopped, the repository's `tools/cleanup-memory.sh`
can reclaim caches on the selected hosts; pre-start memory compaction with
`echo 1 > /proc/sys/vm/compact_memory` as root may also help. Do not reclaim
caches or compact memory during serving. Allow substantial time for weight
loading, draft-model loading, compilation, and warmup. Use `/health` and an
inference smoke test, not container uptime alone, to judge readiness.

Engram lookups read directly from this checkpoint's files. No `/engram-local`
mount is used, and `DSV41_ENGRAM_DIR` is explicitly empty to prevent using old
rows with compatible shapes but different weights. Shared-storage latency can
limit throughput. The existing 112 GiB container RAM limit is retained; see the
original README's SparkRun 0.3.6 memory/swap caveat.

TileLang and Triton compile into SparkRun's persistent, node-local
`/cache/runtime` mount rather than model storage, which may be read-only or
permission-restricted. FlashInfer continues to use the kernels already built
into the pinned image.

## Coding-agent requests

Point your agent's OpenAI-compatible client at `http://<head-node>:8000/v1` and
select `deepseek-v4.1-flash-1017-mxfp4-tp4`. Supply tools on every applicable
request, preserve assistant `reasoning_content` and `tool_calls`, and return
tool results with the matching `tool_call_id`. Do not strip reasoning from
the conversation history. The server cannot execute tools for the client.

The recipe sets maximum effort through `chat_template_kwargs`, where this
tokenizer accepts the numeric 1–100 budget. Avoid client overrides such as
`reasoning_effort: "none"` or `thinking: false`. For clients that send explicit
template settings, use:

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

The example output budget includes reasoning and final output. Allow long
request timeouts and prefer streaming. Effort 100 is the model's maximum
reasoning setting, not a guarantee of best accuracy for every task.

The **256K total context** is a conservative deployment starting point, not
the checkpoint's 1M architectural limit. Prompt, retained reasoning, tool
results, and new output must all fit. DeepSeek recommends a 1M context and at
least 256K output capacity; this starting configuration cannot provide both.
To attempt the full context on an idle cluster, override
`--max-model-len 1048576 -o max_num_seqs=1`, then validate memory and long-context
retrieval before relying on it. This override is not hardware-validated here.

## Runtime acceptance checks

Before using this for unattended coding, check coherent non-repeating text,
non-empty reasoning followed by a final answer, incremental SSE, automatic and
forced tool calls, a tool-result round trip retaining reasoning, and retrieval
near the configured context limit. Confirm all four ranks report disk-backed
Engram from the **1017** directory and no old `/engram-local` source. Do not
reuse the old recipe's benchmark/quality scores as 1017 results.

## License

Original contributions: [Unlicense](LICENSE). Adapted material retains its
upstream terms, including the [Tech2wild MIT notice](LICENSE-MIT). The model
is downloaded separately under its [MIT license](LICENSE-MODEL-MIT).
No model weights, images, or additional runtime patches are distributed in
this directory. See [third-party notices and provenance](THIRD_PARTY_NOTICES.md).
