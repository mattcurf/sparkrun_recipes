# DeepSeek V4.1 Flash 1017: spark-ds41f, four Sparks

Serve the original checkpoint with [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f) 1107 TileLang/vLLM release.

| Item | Pinned value |
| --- | --- |
| Source revision | `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Checkpoint revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Release file | `releases/spark-ds41f-1107/cluster.json` |
| Release profile | `tp4-switched` |

## Runtime settings

| Setting | Value |
| --- | --- |
| Tensor parallelism | Four ranks |
| Engram storage | Disk |
| Context limit | 1M tokens |
| Concurrent sequences | 16 |
| Prefill budget | 16K tokens |
| KV memory per rank | 6 GiB |
| DSpark drafts | Five |
| Kernels | TileLang |
| Collectives | sparknet one-shot switched RoCE |

The KV pool holds about 5.90 full 1M contexts, not sixteen.

## Build without downloading the checkpoint

```bash
./build-image.sh --prepare-only
./build-image.sh --build
```

Set `SPARK_DS41F_SOURCE` to reuse a checkout at the exact source revision.
The helper rejects changed source checkouts and existing image tags.
Its generated node map contains example addresses.
Build and runtime data require node-local NVMe rather than NFS.

## Prepare a private deployment configuration

The public recipe requires a private fabric and storage overlay:

1. Mount the existing snapshot and its entire hub cache read-only.
2. Mount persistent node-local NVMe storage at `/cache`.
3. Render sparknet's `oneshot-switched` settings for your fabric.
   Set the HCA names, GIDs, subnets, control interfaces and rank addresses.
4. Read the origin authentication key from an existing private file.

`/cache` stores generated Engram rows and JIT caches.

Each request permits 16 images across its whole message history.
Raising `--limit-mm-per-prompt` does not increase the encoder-token budget.

## Check host requirements

Upstream qualifies TP4 with 64-KiB pages and memory saver.
Stock 4-KiB pages remain unqualified.
Upstream specifies 5 GiB free during startup and 3 GiB while serving on each node.

## Validate and launch

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-spark-ds41f-tp4.yaml
sparkrun run /path/to/rendered.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

Readiness uses authenticated `/v1/models` access.
See [results and qualification limits](RESULTS.md) for measured coverage.

## Attribution and licensing

This recipe and build helper adapt Christopher Owen's frozen
[1107 cluster configuration](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/releases/spark-ds41f-1107/cluster.json).

The adaptations use **Apache-2.0**, not the repository's default Unlicense.
See [LICENSE](LICENSE) and [NOTICE](NOTICE).
The [source map](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/docs/upstreams.md) credits vLLM/Local Inference Lab, TileLang, TileKernels, sparknet, NCCL, and CUTLASS.
Components and model weights retain their own licenses.
This repository does not distribute weights or images.
