# DeepSeek V4.1 Flash 1017 on TensorFold: four Sparks

Serve the original DeepSeek checkpoint with native Zig TensorFold on four DGX Sparks.
For vLLM, use the [separate recipe](../deepseek-v4.1-flash-1017-mxfp4-4node/).
**The SparkRun launcher has offline and dry-run validation only.**

## Settings

| Setting | Value |
| --- | --- |
| API | `http://<head-host>:8000/v1` |
| Model / alias | `deepseek-v4.1-flash-1017-mxfp4-tp4` / `DeepSeek-V4.1-Flash-1017` |
| Context limit | 1,048,576 |
| Shared position pool | 2,097,152 |
| Concurrent requests | 16 |
| Round rows | 48 |
| Drafting | Checkpoint's DSpark blocks |
| Thinking | On |
| Reasoning effort | Maximum |
| Sampling | Temperature 1.0, top-p 1.0, top-k 20 |
| CUDA graphs | Off on every rank |
| RDMA, side stream, prefetch, Engram AIO | All on |
| Engram | Original FP8 rows on each node's local SSD |
| Engram packing | Off |
| Authentication | Read-only key file on the head only |
| Fabric | Switched ConnectX-7 RoCE with two PCIe rails |
| Node order | `0,1,2,3` |
| OpenMP threads | 1 |
| Streaming heartbeat | 15 seconds |

## Pinned software

| Component | Revision |
| --- | --- |
| MiaAI-Lab deployment | `476bbc77c4e1d1abe45b1cd095526a9ea0d36da4` |
| TensorFold | `59402086381a42c74d53053443224eec2790b4c5` |
| Kernel kit | `6033ea48a27004b2715db82190e2a61f33411d92` |
| Checkpoint | `1bae0e1088993b9b82de08f59de5be5a34e42170` |

## Prepare assets

Require four 128-GB Sparks with Docker, NVIDIA runtime, key-based SSH, `git`, `rsync`, and `hf`.
Keep model data on node-local NVMe, not NFS.

| Storage | Required space |
| --- | --- |
| Checkpoint | About 510 GB per node |
| Head rank caches | About 320 GB |
| Worker rank cache | About 80 GB per worker |

Allow extra space for images, the kernel kit, and temporary files.
Skip preparation if the image, checkpoint, served template, token map, kit, and rank files already exist.
Otherwise, read the pinned [upstream requirements](https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/476bbc77c4e1d1abe45b1cd095526a9ea0d36da4).

```bash
cp local.env.example local.env
# Replace all example addresses, users, paths and interfaces before preparation.
./setup-model.sh
```

Preparation and startup require 100 GiB available on every node.
The helper preserves rank caches and rejects source or settings conflicts.

## Register the external runtime

Register this recipe's [runtime/](runtime/) with SparkRun **0.3.6**.
TensorFold uses this plugin rather than `runtime: vllm-distributed`.

Merge these settings into `~/.config/sparkrun/config.yaml` using your checkout path.

```yaml
features:
  core.external_plugins: true
plugins:
  paths:
    - /path/to/sparkrun_recipes/deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node/runtime
```

## Create a private recipe

The public example requires a private inventory for switched fabric, not ring topology.

1. Copy the recipe privately.
2. Replace `runtime_config.hosts` example addresses with your management addresses in rank order.
3. Set the master to the head's private fabric address.
   Workers use ports 29760 through 29762.
4. Replace NIC, HCA, cache, and key placeholders.
5. Match SparkRun's cluster host order to the recipe.

## Check the configuration

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-tensorfold-tp4.yaml
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

## Launch

```bash
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000
```

Keep `--port 8000` to prevent automatic endpoint changes.
Use HTTP readiness checks only.
TCP probes consume follower rendezvous sockets.
Wait for `/health` and authenticated `/v1/models` access.

## Manage the SparkRun job

Use SparkRun's job ID, not standalone `stop.sh`:

- `sparkrun status`.
- `sparkrun logs <job-id>`.
- `sparkrun stop <job-id>`.

The recipe provides no automatic restart or boot service.

## Check the recipe code

Run with SparkRun and PyYAML from the repository root:

```bash
python -m unittest discover -s tools
```

## License

This recipe uses Apache-2.0.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).
This repository does not include weights or images.
