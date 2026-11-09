# DeepSeek V4.1 Flash on four DGX Sparks

Run the official mixed MXFP4/MXFP8
[`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
checkpoint on four DGX Sparks with TP4, DSpark k5, and CUDA graphs.
See [correctness, throughput, and tool-evaluation results](RESULTS.md).

This recipe adapts
[`tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark`](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
at `fc725ecf10869c184f4347dd73336536d395753c`.
Use image `deepseek-v4.1-flash-sparkrun-v1` with vLLM `e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba`.
Other source versions are incompatible with these patches and can silently corrupt output.

## Storage setup

1. Select a build host with `hf` and at least 80 GB free.
2. Make the shared checkpoint path accessible at the same absolute path on all four hosts.
   Change `cluster_config.resolved_model_path` if necessary.
3. Build and prepare storage with nodes in SparkRun rank order:

```bash
./build-image.sh
./setup-model.sh /srv/sparkrun/models/DeepSeek-V4.1-Flash \
  <rank-0> <rank-1> <rank-2> <rank-3>
```

The script pins revision `dba1be0a40aa45a94ad051997016db3960a90277` and checks the final shard.
It verifies 8,008 sampled weight and scale rows per rank byte-for-byte.
The runtime uses shared Engram rows when local rows are absent or incomplete.

| Storage | Requirement |
| --- | --- |
| Shared checkpoint | Approximately 510 GB; read-only access during serving; no per-rank copy |
| Disk-backed Engram tables | 203 GB |
| Local sparse rows | Approximately 48 GB allocated, 190 GB apparent per host |
| Local row directory | SSH user's `$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash` |
| Root-owned symlink | `/srv/sparkrun/engram/DeepSeek-V4.1-Flash`, mounted read-only at `/engram-local` |

Keep the generated mount paths.
SparkRun 0.3.6 does not expand `~` in volume sources.
Docker rejects symlinked mount sources under world-writable directories such as `/var/tmp`.

### Download locations and cleanup

Remove local rows on each rank host as SparkRun's SSH user:

```bash
docker run --rm --entrypoint sh \
  -v /srv:/host-srv deepseek-v4.1-flash-sparkrun-v1 \
  -c 'rm -f /host-srv/sparkrun/engram/DeepSeek-V4.1-Flash'
rm -rf -- "$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash"
```

Remove the shared checkpoint at the path passed to `setup-model.sh`:

```bash
SHARED_MODEL_DIR=/srv/sparkrun/models/DeepSeek-V4.1-Flash
rm -rf -- "$SHARED_MODEL_DIR"
```

After building, you can remove `${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun-build/deepseek-v4.1-flash`.
Remove the image with `docker image rm deepseek-v4.1-flash-sparkrun-v1`.

## Build and run

After storage setup, validate and launch.
SparkRun distributes the image.

```bash
sparkrun recipe validate deepseek-v4.1-flash-mxfp4-tp4.yaml
sparkrun run deepseek-v4.1-flash-mxfp4-tp4.yaml --cluster <four-node-cluster>
```

| Setting | Value |
| --- | --- |
| API name / port | `deepseek-v4.1-flash-mxfp4-tp4` / 8000 |
| Context | 262,144 tokens |
| Thinking | Off; enable with `chat_template_kwargs={"thinking":true}` |
| Protocol | Model-selected V4.1 tokenizer; `deepseek_v41` reasoning and tool parsers |
| Vision | Up to four images per prompt |
| RAM limit | 112 GiB |
| SparkRun 0.3.6 RAM-plus-swap ceiling | 224 GiB; no create-time `memory-swap` support |

On NVIDIA driver 580.159.03, `docker update` removes GPU access from the running container.
For a 112 GiB no-swap boundary, use SparkRun with create-time `memory-swap` support.

## Correctness gates

Use an idle endpoint before throughput testing:

1. Check coherent greedy chat in eager and graph modes, plus incremental, non-repeating SSE.
2. Check automatic, forced, parallel, and round-trip tool objects and arguments.
3. Check beginning/middle/end retrieval above 250,000 actual prompt tokens, followed by at least 256 coherent output tokens.
4. Check each rank's row range and `/engram-local` source, or shared-checkpoint fallback.
5. Check the image's FlashInfer sparse module and persistent TileLang cache reuse on a subsequent launch.

## Matched throughput benchmark

See [setup, command, and measurements](RESULTS.md#llama-benchy-040).

## Tool evaluation

See [setup, command, scores, and failures](RESULTS.md#tool-eval-bench-260).

## License

Original recipe material uses the Unlicense public-domain dedication.
Adapted upstream orchestration is MIT-licensed.
Vendored vLLM files are Apache-2.0.
This directory does not distribute model weights or built images.
See
`LICENSE`, `LICENSE-MIT`, `LICENSE-APACHE`, and `THIRD_PARTY_NOTICES.md`.
