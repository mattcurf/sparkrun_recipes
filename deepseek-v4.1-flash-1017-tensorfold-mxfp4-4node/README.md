# DeepSeek V4.1 Flash 1017 on TensorFold: four Sparks

Captures a TensorFold native Zig configuration validated on four DGX Sparks.
This is a **separate alternative** to the retained
[vLLM recipe](../deepseek-v4.1-flash-1017-mxfp4-4node/), not an update to it.
It serves the original DeepSeek checkpoint as published, not NVIDIA's newer
NVFP4 export. No re-quantization, NFS, DERISKED or abliteration overlay.

**Status:** image/commands/settings taken from the running four-node service;
the new SparkRun launcher is validated offline and by dry-run only. It has
not replaced, adopted, stopped or restarted that service. Do not launch
either recipe while the existing server is using these GPUs.

## The captured configuration

| Setting | Value |
| --- | --- |
| API | `http://<head-host>:8000/v1` |
| Model / alias | `deepseek-v4.1-flash-1017-mxfp4-tp4` / `DeepSeek-V4.1-Flash-1017` |
| Context / shared position pool | 1,048,576 / 2,097,152 |
| Concurrent requests / round rows | 16 / 48 |
| Drafting / thinking / effort | checkpoint's DSpark blocks / on / max |
| Sampling defaults | temperature 1.0, top-p 1.0, top-k 20 |
| CUDA graphs | **off**, on every rank; avoids observed lazy graph-instantiation errors |
| RDMA / side stream / prefetch / Engram AIO | on / on / on / on |
| Engram | original FP8 rows on each node's local SSD; packing off |
| Authentication | existing key file, read-only on the head only; never embedded in YAML |
| Fabric | switched ConnectX-7 RoCE, two PCIe rails; node order `0,1,2,3` |

The recipe pins MiaAI-Lab's deployment at
`476bbc77c4e1d1abe45b1cd095526a9ea0d36da4`, TensorFold at
`59402086381a42c74d53053443224eec2790b4c5`, the kernel kit at
`6033ea48a27004b2715db82190e2a61f33411d92`, and checkpoint at
`1bae0e1088993b9b82de08f59de5be5a34e42170`. The image includes the upstream
36 patches (streaming keepalives, structured tools, idle RDMA parking, etc.).
The recorded image ID and patch fingerprint are in the YAML metadata.

`OMP_NUM_THREADS=1` follows this repository's CPU-threading policy; the
standalone launcher leaves it unset. Streaming heartbeats retain the image's
15-second default. SparkRun's generic memory estimate does not include the
full native shared pool/runtime; the 100-GiB-per-node preflight is required.

## Prepare assets only during maintenance

Requires four 128-GB DGX Sparks, Docker/NVIDIA runtime, key-based SSH,
`git`, `rsync`, and the `hf` CLI already installed. Ask before installing
packages, using sudo, or stopping any workload. Allow about 510 GB for the
checkpoint on **each** SSD, plus about 320 GB for all four rank caches on
the head and about 80 GB for the appropriate cache on each worker, plus
the image/kit and spare space. Nothing uses NFS.

Read the pinned upstream README, CHANGELOG and warnings before preparing:
<https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold/tree/476bbc77c4e1d1abe45b1cd095526a9ea0d36da4>.
Its NFS instructions are replaced by our local-SSD patch, not followed.

```bash
cp local.env.example local.env
# Replace all example addresses, users, paths and interfaces before preparation.
./setup-model.sh
```

`setup-model.sh` checks out a separate pinned source tree under
`~/.cache/sparkrun-build/`, applies the captured local-SSD/authentication and
memory-preflight changes, then runs upstream `scripts/prepare.sh`. It does
not call `start.sh` or `stop.sh`. It preserves existing rank-cache files.
It refuses preparation when any node has less than 100 GiB available.
The existing deployment directory is not modified. A source or settings
conflict fails rather than overwriting edits.

On this already prepared cluster, **skip setup-model.sh**: the captured image,
checkpoint, served template, token map, kit and rank files already exist on
the nodes. Do not rebuild or copy model data while serving. Cache reclamation
and compaction are separate, approved maintenance actions after inference
stops; neither is a recipe pre-exec hook. SparkRun itself may attempt its
usual best-effort page-cache clearing during a real launch.

## Register the external SparkRun runtime

SparkRun **0.3.6** does not ship a TensorFold runtime. This recipe includes
one in [runtime/](runtime/); do not substitute `runtime: vllm-distributed`.
Merge the following into `~/.config/sparkrun/config.yaml`, preserving existing
features and plugin paths (adjust the repository path if needed):

```yaml
features:
  core.external_plugins: true
plugins:
  paths:
    - /path/to/sparkrun_recipes/deepseek-v4.1-flash-1017-tensorfold-mxfp4-4node/runtime
```

No package installation is necessary. This repository does not automatically
change your SparkRun configuration. Keep the runtime registered for future
`run`, `logs` and `stop` commands.

## Validate now; launch only after the current server is stopped

The checked-in `192.0.2.10`–`192.0.2.13` addresses are RFC 5737 documentation
examples, not real hosts. Before launch, create a private copy of the recipe
and replace `runtime_config.hosts` with your four management addresses in rank
order. Set the master address to the head's private fabric address; workers use
ports 29760 through 29762. Replace the NIC/HCA placeholders, cache and key paths
together, and make the SparkRun cluster's host order match the private recipe.
Keep that inventory and `local.env` out of the public repository.
This captures a **switch**, not a ring; the bare public example is not deployable.

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-tensorfold-tp4.yaml
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

**Do not run the following yet.** In an approved maintenance window, stop the
existing standalone service coherently with its own `stop.sh`, ensure at least
100 GiB available on each node, then launch:

```bash
sparkrun run /path/to/private-recipe.yaml --cluster YOUR_CLUSTER --port 8000
```

The explicit `--port 8000` prevents SparkRun's automatic port increment from
silently changing the endpoint. The new job uses SparkRun's labelled container
names; it never takes over the old `dsv41-1017-tf` containers. Memory preflight
fails rather than trying to make room while the old server runs. The original
standalone `stop.sh` must not be used to manage the new SparkRun job.

The runtime starts followers 3,2,1 and then the head, with the same commands
as the live deployment. It deliberately avoids SparkRun's generic native
TCP readiness probe: TensorFold's rendezvous sockets accept one actual
follower each. Only HTTP readiness checks are safe. Wait for `/health` and
the authenticated `/v1/models` response, not just container uptime; first
loading/warmup can take minutes. Preserve the existing LAN-only firewall
and routing; the recipe adds no internet exposure.

After migration, use SparkRun's reported job ID with `sparkrun status`,
`sparkrun logs <job-id>` and `sparkrun stop <job-id>`. Restart with the same
`sparkrun run` command after a coherent stop and memory checks. Containers
are retained for diagnosis (`auto_remove: false`); no restart policy or
reboot-time system service is installed.

The existing 0–2200 MHz GPU clock cap is host state, not model configuration.
This recipe neither changes it nor makes it reboot-persistent. Review caps
separately after a reboot. It also does not enable any opt-in `TF_DS_*` tuning.

## Verification and licensing

From the repository root, activate a Python environment with SparkRun and
PyYAML, then run the CPU-only recipe tests:

```bash
python -m unittest discover -s tools
```

Offline command checks cover matching head/follower settings and invalid
rank layouts; a real SparkRun migration, startup, streaming/tool checks and
rebenchmark remain necessary before claiming this launcher is live-validated.
The original service has already passed chat, streaming, tools and concurrent
request checks; those results do not prove a new launcher boots successfully.

Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE) and
[third-party notices](THIRD_PARTY_NOTICES.md). No weights or images are included.
