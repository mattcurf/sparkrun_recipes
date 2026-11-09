# DeepSeek V4.1 Flash 1017: TP4 with maximum thinking

Run the official mixed MXFP4/MXFP8
[1017 checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
on four DGX Sparks.
**The tool-evaluation safety gate failed.**
See [results and limits](RESULTS.md) before unattended use.

| Default | Value |
| --- | --- |
| Thinking / effort | On / **100/100**; retain prior reasoning |
| Temperature / top-p | Model author's 1.0 / 1.0; client-overridable |
| Runtime | TP4, DSpark k5, V4.1 reasoning/tool parsers, automatic tool choice, vision enabled |
| API name / head-node port | `deepseek-v4.1-flash-1017-mxfp4-tp4` / 8000 |
| Total context | 256K tokens |
| RAM limit | 112 GiB; see [swap warning](../deepseek-v4.1-flash-mxfp4-4node/README.md#build-and-run) |

<a id="why-this-still-uses-the-pinned-disk-engram-runtime"></a>

## Required runtime

Use `deepseek-v4.1-flash-sparkrun-v1` from the [shared recipe](../deepseek-v4.1-flash-mxfp4-4node/).
It requires Tony Deangelo-derived patches and vLLM `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
This runtime requires disk-backed Engram rather than shared CPU/GPU RAM.
The launcher mod `use-official-vllm` does not indicate stock-image compatibility.

Compatibility references:
[ring fix #58560](https://github.com/vllm-project/vllm/pull/58560),
[fork report #13](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/issues/13),
[#59689](https://github.com/vllm-project/vllm/pull/59689),
[memory-only upstream Engram storage](https://github.com/vllm-project/vllm/blob/da9134bc89364465f68aa679564435936d157a14/vllm/config/engram.py),
and [SM12x page-size support #57292](https://github.com/vllm-project/vllm/pull/57292).

## Prepare and run

1. Open this directory and build the shared runtime if absent:

```bash
../deepseek-v4.1-flash-mxfp4-4node/build-image.sh
```

   Follow the [build requirements](../deepseek-v4.1-flash-mxfp4-4node/README.md).
   The shared `setup-model.sh` downloads the older checkpoint; use the 1017 download below.
2. Prepare a dedicated writable checkpoint path, identical on all hosts, with at least 511 GB plus download overhead.
   For another path, change the download command and `cluster_config.resolved_model_path`.
   Engram rows are checkpoint-specific.
3. Download once, validate, dry-run, and launch:

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

Use the pinned download: SparkRun neither distributes nor verifies this external checkpoint.
For memory fragmentation, use this offline cleanup sequence:

1. Use the repository's `tools/cleanup-memory.sh` to reclaim caches on the selected hosts.
2. If necessary, run `echo 1 > /proc/sys/vm/compact_memory` as root before startup.

Readiness uses `/health` and an inference request.
Keep `DSV41_ENGRAM_DIR` empty: Engram must read this checkpoint directly, without `/engram-local`.
Shared-storage latency can limit throughput.
Keep node-local `/cache/runtime` writable; model storage can be read-only during serving.

## Coding-agent requests

1. Connect an OpenAI-compatible client to `http://<head-node>:8000/v1` with the API name above.
2. Supply tools on each applicable request; the client must execute them.
3. Preserve assistant `reasoning_content` and `tool_calls`; match tool results with `tool_call_id`.

Use numeric effort 1–100, not `reasoning_effort: "none"` or `thinking: false`.
For explicit template settings, use:

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

The output budget includes reasoning and final output.
Allow long timeouts; prefer streaming.
This configuration cannot provide DeepSeek's recommended 1M context with at least 256K output capacity.

For an **unvalidated** full-context attempt:

1. Override `--max-model-len 1048576 -o max_num_seqs=1`.
2. Validate memory and long-context retrieval.

## Runtime acceptance checks

Before unattended coding:

1. Check coherent, non-repeating text, non-empty reasoning, a final answer, and incremental SSE.
2. Check automatic/forced tool calls and a round trip with retained reasoning.
3. Check retrieval near the configured context limit.
4. Confirm all four ranks report disk-backed Engram from **1017**.

## License

Original contributions use the [Unlicense](LICENSE).
Adapted material retains its upstream terms, including the [Tech2wild MIT notice](LICENSE-MIT).
Download the model separately under its [MIT license](LICENSE-MODEL-MIT).
This directory does not distribute model weights, images, or additional runtime patches.
See [third-party notices and provenance](THIRD_PARTY_NOTICES.md).
