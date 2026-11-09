# DeepSeek V4.1 Flash on four DGX Sparks

This recipe serves the official mixed MXFP4/MXFP8
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
checkpoint on four DGX Sparks.
It uses TP4.
It uses DSpark k5.
It uses CUDA graphs.
The context window contains 262,144 tokens.

The 203 GB Engram tables remain on disk.
Each rank can use a local sparse row copy.
The shared checkpoint remains a safe fallback.
See [measured correctness, throughput, and tool-evaluation results](RESULTS.md).

This SparkRun recipe adapts
[`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
at `fc725ecf10869c184f4347dd73336536d395753c`.
It deliberately pins vLLM to `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
Later source versions are not compatible with these whole-file patches.
They have produced corrupted output without warnings.

## Storage setup

The checkpoint is about 510 GB.
The recipe does not copy it to every rank.
All four hosts must see it at the same absolute path.
The path must support read-only access.

The recipe declares this path through SparkRun's supported `cluster_config.resolved_model_path` schema.
The example location is `/srv/sparkrun/models/DeepSeek-V4.1-Flash`.
Change that field if your shared model storage uses another path.

Engram rows always use `$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash` under the SSH user's home directory on each host.
SparkRun 0.3.6 quotes Docker volume sources.
Thus, `~/...` in `executor_config.volumes` passes a literal tilde.
It does not expand the remote user's home directory.

The setup script creates a stable symlink on each host at `/srv/sparkrun/engram/DeepSeek-V4.1-Flash`.
The recipe mounts this path read-only at `/engram-local`.
The helper uses the required recipe image to create this root-owned alias.
Docker rejects symlinked mount sources under world-writable directories such as `/var/tmp`.

This arrangement avoids control-host expansion.
It also avoids the container's root `$HOME`.
The setup and serving paths stay fixed.

1. Select a host with `hf`.
2. Build the required recipe image.
3. Specify the four nodes in SparkRun rank order.
4. Prepare the shared checkpoint and local Engram directories.

Use these commands:

```bash
./build-image.sh
./setup-model.sh /srv/sparkrun/models/DeepSeek-V4.1-Flash \
  <rank-0> <rank-1> <rank-2> <rank-3>
```

The script pins model revision `dba1be0a40aa45a94ad051997016db3960a90277`.
It checks the final shard.
It creates approximately 48 GB of rank-local Engram rows on each node.
It verifies 8,008 sampled weight and scale rows per rank byte-for-byte.

The runtime safely reads Engram rows from the shared checkpoint if a local copy is absent.
It also uses this fallback if the local copy does not cover the rank's rows.

### Download locations and cleanup

`setup-model.sh` downloads the approximately 510 GB checkpoint into the directory from its first argument.
The example uses `/srv/sparkrun/models/DeepSeek-V4.1-Flash`.
The script creates sparse Engram data at `$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash` on each rank host.
This data allocates approximately 48 GB per host.
Its apparent size is approximately 190 GB per host.

Each host's `/srv/sparkrun/engram/DeepSeek-V4.1-Flash` path is only a symlink to that user's Engram directory.
It is not another copy.

1. Stop the recipe before you delete its files.
2. Sign in to **each** rank host as the SSH user that SparkRun uses.
3. Run these commands on each host to remove the rank-local Engram data:

```bash
docker run --rm --entrypoint sh \
  -v /srv:/host-srv deepseek-v4.1-flash-sparkrun-v1 \
  -c 'rm -f /host-srv/sparkrun/engram/DeepSeek-V4.1-Flash'
rm -rf -- "$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash"
```

4. Confirm that no other recipe uses the shared checkpoint before you delete it.
5. Run these commands once against the shared path that you passed to `setup-model.sh`:

```bash
SHARED_MODEL_DIR=/srv/sparkrun/models/DeepSeek-V4.1-Flash
rm -rf -- "$SHARED_MODEL_DIR"
```

The build checkout defaults to `${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun-build/deepseek-v4.1-flash`.
You can remove it after you build the image.
Manage Docker images separately on each host.
Use `docker image rm deepseek-v4.1-flash-sparkrun-v1` only where you no longer need the image.

## Build and run

1. Build on a node with at least 80 GB free.
2. Complete the storage setup above after the build.
3. Run SparkRun to distribute the resulting image.

The build compiles the exact vLLM stable extension with low parallelism.
It also compiles FlashInfer 0.7.0rc1 SM 12.1a MXFP8 and sparse-MLA kernels with low parallelism.

```bash
sparkrun recipe validate deepseek-v4.1-flash-mxfp4-tp4.yaml
sparkrun run deepseek-v4.1-flash-mxfp4-tp4.yaml --cluster <four-node-cluster>
```

The recipe uses host bind mounts.
It also uses device access.
Thus, current SparkRun asks you to trust the audited local recipe.

The API name is `deepseek-v4.1-flash-mxfp4-tp4`.
The API uses port 8000.
Thinking is off by default.
Enable thinking per request with `chat_template_kwargs={"thinking":true}`.

The 112 GiB container memory limit matches the reference deployment.
However, SparkRun 0.3.6 cannot set Docker's `memory-swap`.
Docker therefore reports a 224 GiB combined memory-and-swap ceiling.
The reference ceiling is 112 GiB.

**Do not use `docker update` to correct this after launch.**
On NVIDIA driver 580.159.03, it removes GPU access from the running container.

This recipe stays within the 112 GiB RAM limit during the checks below.
For an exact no-swap boundary, use a SparkRun release with create-time `memory-swap` support.

## Correctness gates

Before throughput testing, complete these checks against an idle endpoint:

1. Verify non-empty, coherent greedy chat output in eager and graph modes.
2. Verify that SSE streaming emits incremental, non-repeating text.
3. Verify valid V4.1 tool objects and arguments for automatic, forced, parallel, and tool-result-round-trip calls.
4. Use a prompt above 250,000 actual tokens.
   Verify retrieval of distinct values from its beginning, middle, and end.
   Then verify at least 256 coherent output tokens.
5. Verify that every rank log reports its own Engram row range.
   Ranks with a prepared sparse copy must report the local `/engram-local` source.
   Fallback ranks must report the shared checkpoint.
6. Verify that FlashInfer loads its sparse module from the image.
7. Check SparkRun's persistent TileLang shape cache.
   The first launch may populate this cache.
   A subsequent launch must reuse it.

Do not interchange V4.1 protocol settings with DeepSeek V4 settings.
The recipe uses the model-selected V4.1 tokenizer.
It uses `deepseek_v41` reasoning and tool parsers.

The recipe enables vision.
Each prompt supports up to four images.

## Matched throughput benchmark

The checked-in results use upstream
[`eugr/llama-benchy` 0.4.0](https://github.com/eugr/llama-benchy/tree/v0.4.0).
The settings match the previous TP4 comparison:

- Input tokens: 2,048 / 4,096 / 8,192 / 16,384.
- Concurrency: 1 / 2 / 4.
- Measured batches: three after the tool's discarded shape warmup.
- Requested output tokens: exactly 256.

Use a cache-isolating proxy or a fresh server cache for a cold-prefix comparison.

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
at commit `992a6978ecbee2d72fa2ead9ccc509436769d088`.
It uses all 69 standard scenarios.
It uses greedy sampling.
It uses no thinking.

The seed is 42.
The request timeout is 300 seconds.

Run this command:

```bash
uvx --from 'git+https://github.com/SeraphimSerapis/tool-eval-bench.git@992a6978ecbee2d72fa2ead9ccc509436769d088' \
  tool-eval-bench run --base-url http://127.0.0.1:8000 \
  --model deepseek-v4.1-flash-mxfp4-tp4 --backend vllm \
  --seed 42 --temperature 0 --no-think --timeout 300 \
  --json-file results/tool-eval-bench.json
```

## License

Original recipe material uses the Unlicense public-domain dedication.
Adapted upstream orchestration is MIT-licensed.
Vendored vLLM files are Apache-2.0.
This directory does not distribute model weights or built images.
See
`LICENSE`, `LICENSE-MIT`, `LICENSE-APACHE`, and `THIRD_PARTY_NOTICES.md`.
