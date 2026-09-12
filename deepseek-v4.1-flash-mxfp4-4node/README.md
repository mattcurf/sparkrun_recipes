# DeepSeek V4.1 Flash on four DGX Sparks

Serves the official mixed MXFP4/MXFP8
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
checkpoint on four DGX Sparks with TP4, DSpark k5, CUDA graphs, and a 262,144-token
context window. The 203 GB Engram tables remain on disk; each rank can use a
rank-local sparse row copy while retaining the shared checkpoint as a safe fallback.
See [measured correctness, throughput, and tool-evaluation results](RESULTS.md).

This is a SparkRun adaptation of
[`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
at `fc725ecf10869c184f4347dd73336536d395753c`. It deliberately pins vLLM to
`e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`: later trees are not compatible
with these whole-file patches and have produced silently corrupted output.

## Storage setup

The checkpoint is about 510 GB and is not copied to every rank. All four hosts
must see it at the same absolute, read-only-capable path. The recipe declares
that path through SparkRun's supported `cluster_config.resolved_model_path`
schema. Its example location is `/srv/sparkrun/models/DeepSeek-V4.1-Flash`;
change that field when your shared model storage uses another path.

Engram rows always live under the SSH user's home directory on each host at
`$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash`. SparkRun 0.3.6 quotes
Docker volume sources, so putting `~/...` directly in `executor_config.volumes`
would pass a literal tilde instead of expanding the remote user's home. The setup
script therefore creates a stable per-host symlink at
`/srv/sparkrun/engram/DeepSeek-V4.1-Flash`, which the recipe mounts read-only at
`/engram-local`. The helper creates this root-owned alias through the already
required recipe image; Docker rejects symlinked mount sources under world-writable
directories such as `/var/tmp`. This avoids both control-host expansion and the
container's root `$HOME` while keeping setup and serving paths fixed.

Prepare the shared checkpoint and local Engram directories from a host with
`hf` after building the required recipe image, then name the four nodes in
SparkRun rank order:

```bash
./build-image.sh
./setup-model.sh /srv/sparkrun/models/DeepSeek-V4.1-Flash \
  <rank-0> <rank-1> <rank-2> <rank-3>
```

The script pins model revision `dba1be0a40aa45a94ad051997016db3960a90277`,
checks the final shard, and creates approximately 48 GB of rank-local Engram
rows on each node. It verifies 8,008 sampled weight and scale rows per rank
byte-for-byte. If a local copy is absent or does not cover the rank's rows, the
runtime safely reads those Engram rows from the shared checkpoint instead.

## Build and run

Build on a node with at least 80 GB free before running the storage setup above,
then distribute the resulting image by running SparkRun. The build compiles the
exact vLLM stable extension and FlashInfer 0.7.0rc1 SM 12.1a MXFP8 and sparse-MLA
kernels with low parallelism.

```bash
sparkrun recipe validate deepseek-v4.1-flash-mxfp4-tp4.yaml
sparkrun run deepseek-v4.1-flash-mxfp4-tp4.yaml --cluster <four-node-cluster>
```

The recipe uses host bind mounts and device access, so current SparkRun asks you
to trust the audited local recipe. The API name is
`deepseek-v4.1-flash-mxfp4-tp4` on port 8000. Thinking is off by default; enable
it per request with `chat_template_kwargs={"thinking":true}`.

The 112 GiB container memory limit matches the reference deployment. SparkRun
0.3.6 cannot set Docker's `memory-swap`, however, so Docker reports a 224 GiB
combined memory-and-swap ceiling instead of the reference's 112 GiB. Do not use
`docker update` to correct this after launch: on NVIDIA driver 580.159.03 it
removes GPU access from the running container. This recipe remains within the
112 GiB RAM limit during the checks below; use a SparkRun release with
create-time `memory-swap` support if an exact no-swap boundary is required.

## Correctness gates

Before throughput testing, verify all of the following against an idle endpoint:

1. Non-empty coherent greedy chat output in eager and graph modes.
2. SSE streaming emits incremental, non-repeating text.
3. Automatic, forced, parallel, and tool-result-round-trip calls produce valid
   V4.1 tool objects and arguments.
4. A prompt above 250,000 actual tokens retrieves distinct values from its
   beginning, middle, and end, then emits at least 256 coherent tokens.
5. Every rank log reports its own Engram row range. Ranks with a prepared sparse
   copy report the local `/engram-local` source; any fallback rank reports the
   shared checkpoint. FlashInfer's sparse module must load from the image; first
   launch may populate SparkRun's persistent TileLang shape cache, while a
   subsequent launch must reuse it.

The V4.1 protocol settings are not interchangeable with DeepSeek V4: the recipe
uses the model-selected V4.1 tokenizer plus `deepseek_v41` reasoning and tool
parsers. Vision remains enabled, with up to four images per prompt.

## Matched throughput benchmark

The checked-in results use upstream
[`eugr/llama-benchy` 0.4.0](https://github.com/eugr/llama-benchy/tree/v0.4.0),
matching the previous TP4 comparison: 2,048 / 4,096 / 8,192 / 16,384 input
tokens, concurrency 1 / 2 / 4, three measured batches after the tool's discarded
shape warmup, and exactly 256 requested output tokens. Use a cache-isolating
proxy or a fresh server cache for a cold-prefix comparison.

```bash
uvx --from llama-benchy==0.4.0 llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model deepseek-ai/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash-mxfp4-tp4 \
  --tokenizer /srv/sparkrun/models/DeepSeek-V4.1-Flash \
  --pp 2048 4096 8192 16384 --tg 256 --exact-tg \
  --depth 0 --runs 3 --concurrency 1 2 4 --no-cache \
  --save-result results/llama-benchy.json --format json \
  --emit-progress results/llama-benchy-progress.jsonl \
  --save-total-throughput-timeseries --exit-on-first-fail
```

## Tool evaluation

The recorded quality run uses
[`tool-eval-bench` 2.6.0](https://github.com/SeraphimSerapis/tool-eval-bench/tree/v2.6.0)
at commit `992a6978ecbee2d72fa2ead9ccc509436769d088`, all 69 standard scenarios,
greedy sampling, no thinking, seed 42, and a 300-second request timeout:

```bash
uvx --from 'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@992a6978ecbee2d72fa2ead9ccc509436769d088' \
  tool-eval-bench run --base-url http://127.0.0.1:8000 \
  --model deepseek-v4.1-flash-mxfp4-tp4 --backend vllm \
  --seed 42 --temperature 0 --no-think --timeout 300 \
  --json-file results/tool-eval-bench.json
```

## License

Original recipe material is dedicated to the public domain under the Unlicense.
Adapted upstream orchestration is MIT-licensed; vendored vLLM files are
Apache-2.0. Model weights and built images are not distributed. See
`LICENSE`, `LICENSE-MIT`, `LICENSE-APACHE`, and `THIRD_PARTY_NOTICES.md`.
