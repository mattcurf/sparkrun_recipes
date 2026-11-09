# DeepSeek V4.1 Flash 1017 on TensorFold: four Sparks

This recipe records a TensorFold native Zig configuration tested on four DGX Sparks.
It provides a separate alternative to the [vLLM recipe](../deepseek-v4.1-flash-1017-mxfp4-4node/).
It uses the original DeepSeek checkpoint.
It does not use NVIDIA's newer NVFP4 export.
It does not re-quantize the weights.
It does not use NFS, DERISKED or an abliteration overlay.

## Test status

The image, commands and settings come from a working four-node service.
The SparkRun launcher has only passed offline checks and a dry run.
It has not replaced or managed that service.
**Do not launch either recipe while another server uses these GPUs.**

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
| Authentication | Existing key file on the head |
| Fabric | Switched ConnectX-7 RoCE with two PCIe rails |
| Node order | `0,1,2,3` |

The head mounts the key file read-only.
Never put the key in YAML.
Disabled CUDA graphs avoid observed lazy graph-instantiation errors.

## Pinned software

| Component | Revision |
| --- | --- |
| MiaAI-Lab deployment | `476bbc77c4e1d1abe45b1cd095526a9ea0d36da4` |
| TensorFold | `59402086381a42c74d53053443224eec2790b4c5` |
| Kernel kit | `6033ea48a27004b2715db82190e2a61f33411d92` |
| Checkpoint | `1bae0e1088993b9b82de08f59de5be5a34e42170` |

The image includes 36 upstream patches.
These patches include streaming keepalives, structured tools and idle RDMA parking.
The YAML metadata records the image ID and patch fingerprint.

The recipe sets `OMP_NUM_THREADS=1` to follow this repository's policy.
The standalone launcher leaves that setting unset.
Streaming heartbeats use the image's 15-second default.
SparkRun's memory estimate excludes parts of the native shared pool and runtime.
The preflight therefore requires 100 GiB free on every node.

## Prepare assets during maintenance only

You need four 128-GB DGX Sparks.
Install Docker and the NVIDIA runtime before preparation.
You also need key-based SSH, `git`, `rsync` and the `hf` CLI.
Ask before you install packages.
Ask before you use sudo.
Ask before you stop workloads.

Allow approximately 510 GB for the checkpoint on each SSD.
Allow approximately 320 GB for all four rank caches on the head.
Allow approximately 80 GB for each worker's rank cache.
Leave additional space for the image, kernel kit and temporary files.
Nothing uses NFS.

Read the pinned upstream README and CHANGELOG before preparation.
Read the upstream warnings:
<https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/476bbc77c4e1d1abe45b1cd095526a9ea0d36da4>.
Our local-SSD patch replaces upstream's NFS instructions.

```bash
cp local.env.example local.env
# Replace all example addresses, users, paths and interfaces before preparation.
./setup-model.sh
```

`setup-model.sh` creates a separate source checkout under `~/.cache/sparkrun-build/`.
It uses the pinned source revision.
It applies the local-SSD, authentication and memory-preflight changes.
It then runs upstream's `scripts/prepare.sh`.
It does not call `start.sh` or `stop.sh`.
It preserves existing rank-cache files.
It refuses preparation if any node has less than 100 GiB available.
It does not change the existing deployment directory.
It stops on source or settings conflicts instead of overwriting changes.

**Skip `setup-model.sh` if the required assets already exist.**
These assets include the image, checkpoint, served template, token map, kit and rank files.
Do not rebuild or copy model data during inference.
Stop inference before approved cache reclamation or memory compaction.
Neither action belongs in a recipe pre-exec hook.
SparkRun may attempt its own page-cache cleanup during a real launch.

## Register the external runtime

SparkRun **0.3.6** does not include a TensorFold runtime.
This recipe supplies one in [runtime/](runtime/).
Do not replace it with `runtime: vllm-distributed`.

Add the following settings to `~/.config/sparkrun/config.yaml`.
Keep existing feature settings and plugin paths.
Adjust the repository path for your checkout.

```yaml
features:
  core.external_plugins: true
plugins:
  paths:
    - /path/to/sparkrun_recipes/deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node/runtime
```

You do not need to install another package.
This repository does not change your SparkRun configuration automatically.
Keep the runtime registered for later `run`, `logs` and `stop` commands.

## Create a private recipe

The `192.0.2.10`–`192.0.2.13` addresses are RFC 5737 documentation examples.
They do not identify real nodes.
**Do not launch the public example directly.**
It describes a switched fabric, not a ring.

1. Make a private copy of the recipe.
2. Set `runtime_config.hosts` to your four management addresses in rank order.
3. Set the master address to the head's private fabric address.
   Workers use ports 29760 through 29762.
4. Replace the NIC and HCA placeholders.
5. Set the cache and key paths.
6. Match the SparkRun cluster's host order to the private recipe.

Keep the real inventory out of the public repository.
Keep `local.env` out of the public repository.

## Check the configuration

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-tensorfold-tp4.yaml
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

## Launch during an approved maintenance window

**Do not run the following command while the existing service uses the GPUs.**

1. Stop all existing standalone ranks with that deployment's own `stop.sh`.
2. Confirm at least 100 GiB available on each node.
3. Start the SparkRun workload.

```bash
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000
```

The explicit `--port 8000` prevents automatic port changes.
SparkRun gives the new containers its own labels and names.
It never takes over the old `dsv41-1017-tf` containers.
The memory preflight fails instead of clearing memory during inference.
Do not use the old standalone `stop.sh` to manage the new SparkRun job.

The runtime starts followers in order 3, 2, 1.
It then starts the head.
These commands match the working deployment.

Do not use SparkRun's generic native TCP readiness probe.
Each TensorFold rendezvous socket accepts only its real follower.
A probe connection consumes that follower's socket.
Use HTTP readiness checks only.
Wait for `/health` and authenticated `/v1/models` access.
Container uptime alone does not prove readiness.
Initial loading and warmup can take minutes.
Keep the LAN-only firewall and routing rules.
This recipe does not add internet exposure.

## Manage the SparkRun job

Use SparkRun's reported job ID with these commands:

- `sparkrun status`.
- `sparkrun logs <job-id>`.
- `sparkrun stop <job-id>`.

Stop all ranks before a restart.
Check memory before you repeat the `sparkrun run` command.
SparkRun keeps stopped containers for diagnosis with `auto_remove: false`.
This recipe installs no restart policy.
It installs no reboot-time system service.

The existing 0–2200 MHz GPU clock cap belongs to the host configuration.
This recipe does not change it.
It does not preserve that cap across reboots.
Review clock caps after a reboot.
The recipe does not enable optional `TF_DS_*` tuning.

## Check the recipe code

Activate a Python environment with SparkRun and PyYAML.
Run the following command from the repository root:

```bash
python -m unittest discover -s tools
```

Offline checks compare head and follower settings.
They also reject invalid rank layouts.
They do not prove a successful SparkRun migration.
A real launch still needs startup, streaming, tool and benchmark checks.
The original service passed chat, streaming, tools and concurrent-request checks.
Those results do not prove that this launcher starts successfully.

## License

This recipe uses Apache-2.0.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and [third-party notices](THIRD_PARTY_NOTICES.md).
This repository does not include weights or images.
