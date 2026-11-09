# DeepSeek V4.1 Flash 1017: spark-ds41f, four Sparks

Alternative to TensorFold adapted from
[Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f)
**1107 TileLang/vLLM release**, pinned at
`cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5`.
Uses the original `deepseek-ai/DeepSeek-V4.1-Flash-1017` checkpoint at
`1bae0e1088993b9b82de08f59de5be5a34e42170`; no conversion or new weights.

The recipe is based on the frozen `releases/spark-ds41f-1107/cluster.json`
**tp4-switched** profile: TP4, disk-backed Engram, 1M context, 16 sequences,
16K prefill budget, **6 GiB KV per rank**, DSpark five drafts, TileLang kernels
and sparknet one-shot switched RoCE. It does not use Ray, TensorFold rank
files, the old disk-Engram patches, or a stock vLLM image. SparkRun's native
distributed runtime supplies exactly one set of rank/rendezvous flags.

Differences from upstream: KV is reduced from 10.5 to 6 GiB to reserve more
allocation headroom after the first local startup failed. This reduces total
resident context capacity, not the per-request limit or model arithmetic;
startup reports 6,182,120 tokens (5.90 full 1M contexts), not sixteen full
1M contexts at once. The existing served model name is retained,
`OMP_NUM_THREADS=1` follows this repository's threading policy, and GLP
steering/the second model name are omitted. The stock checkpoint is unchanged.
The 264-byte packed Engram row layout is explicitly generated in a
checkpoint/release-specific directory under `/cache`, not alongside read-only
checkpoint blobs. This is a derived disk layout, not re-quantized weights.
No kernel, firewall, endpoint, host recovery, or routing change is installed.

## Build without downloading the checkpoint

```bash
./build-image.sh --prepare-only
# Only after inference is stopped in an authorized maintenance window:
./build-image.sh --build
```

The helper uses upstream's pinned source locks, patch stacks, Dockerfile,
build checks, guarded build, and GPU import smoke. It refuses a changed source
checkout and does not overwrite an existing image tag. Set `SPARK_DS41F_SOURCE`
to reuse an existing checkout at the exact pin. Preparation fetches code and
build wheels, **not model weights**, and does not stop or start inference.
The ignored example node map it creates is only a build prerequisite, not a
deployment configuration. Copy the built image to every node and verify the
same image content before launching. All build and runtime storage must be
on node-local NVMe, not NFS.

The helper disables inherited Git signing for build subprocesses and uses the
recorded committer email for vLLM's final 0761 patch. This reproduces upstream's
exact locked commit/tree; it does not change the patch or relax verification.

## Site configuration is required

Do not launch this bare recipe. Supply a private deployment overlay with:

- the existing snapshot path and the **entire enclosing hub cache** mounted
  read-only, so the snapshot's relative blob symlinks resolve;
- a persistent writable node-local NVMe mount at `/cache`, including Engram's
  generated shards and the JIT caches;
- sparknet's `oneshot-switched` environment rendered for the site's HCA names,
  GIDs/subnets, control-plane interfaces, and rank addresses;
- origin authentication read from an existing private key file, never YAML.

The public recipe admits 16 images per request, counting the whole message
history. A private deployment may raise `--limit-mm-per-prompt` for image-heavy
histories; that changes admission, not the scheduler's encoder-token budget.
Keep actual host inventories, usernames, storage paths, endpoint mappings and
credentials in the private overlay, not this directory.

The qualified upstream TP4 profile requires 64-KiB pages plus memory saver.
Stock 4-KiB TP4 is an **unqualified site trial**, not a promise of equivalent
memory capacity. Do not install a kernel or reboot to satisfy this recipe.
Check all four nodes' startup/steady memory headroom and stop the trial if
unsafe. Upstream's documented floors are 5 GiB startup and 3 GiB steady.

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-spark-ds41f-tp4.yaml
# Use the private rendered recipe, not this incomplete site-independent one:
sparkrun run /path/to/rendered.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

Never run two TP4 engines on the same nodes. Stop the existing runtime
coherently before building/launching. No pre-exec hook clears live caches.
Keep context, sequence count, speculative depth, graph ladder and prefill
threshold consistent when tuning. Wait for authenticated `/v1/models`, inspect
all ranks for errors/fallbacks, and test streaming/tools before manual use.

## Validation status

Recipe parsing, native rank-command generation, and deployment dry-run are
offline checks only. A build, boot and matched benchmark are separate gates;
this recipe does not inherit upstream's measurements as local results.

The initial 2026-11-08 four-node, 4-KiB site trial with 10.5-GiB KV
**did not reach API readiness**.
The pinned image built and passed GPU, TileLang and sparknet import checks;
every rank read all 48 existing checkpoint shards through the read-only mount.
Startup loaded the weights, generated the NVMe Engram layout and reached
DSpark graph/tail warmup, then failed with a RoCE wait for rank 0 and NVIDIA
driver allocation errors. Recorded host memory stayed above 21 GiB, so the
host-memory floor alone did not catch this failure. The precise causal chain
is unresolved; do not attribute it to the page size without further evidence.
That trial was stopped and the previous runtime restored at the time.

The subsequent **6-GiB configuration passed on all four nodes**, with unchanged image,
checkpoint, DSpark, graphs, sequence count and 1M per-request limit. Startup
guards remained active through readiness; minimum available memory was
25.51 GiB, with no driver allocation errors or Xids. Authorized/denied access,
streaming with usage, thinking, automatic tools and tool-result round trips,
eight concurrent requests, and 118,896-token retrieval passed. All 166 generated
tokens reconciled with the vLLM counter. Weights and derived caches were
rechecked on NVMe on every node.

A subsequent matched benchmark completed all 86 requests and 20,102 output
tokens against the saved prior-runtime arm, with matching prompt hashes and
token counts, zero cached/reasoning tokens and no overlapping traffic. The
candidate had no request failures, new driver errors, swap growth or vLLM
preemptions; it remained running afterward. Measurements and operational
evidence are maintained in the private deployment repository.

The benchmarks cover one boot per engine with retained JIT caches, not a causal
proof of the first failure, a 1M capacity qualification or a stability soak.
The synthetic comparison has three repeats per decode point but no independent
boot repeats or matched thermal monitoring of the prior runtime. A later restart
with a private 999-image admission override passed image-history retrieval,
authentication, streaming, thinking and tool-result checks; that higher limit
is not the public recipe's default or a guarantee that 999 images fit.

## Attribution and licensing

Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f)
provides the serving integration, patched TileLang kernel stack, DSpark
configuration and sparknet collective integration used here. This recipe and
build helper adapt its frozen
[1107 cluster configuration](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/releases/spark-ds41f-1107/cluster.json)
under **Apache-2.0**; see the standalone [LICENSE](LICENSE) and [NOTICE](NOTICE).
The repository's default Unlicense does not replace that license.

The serving stack also relies on vLLM/Local Inference Lab, tile-ai/TileLang,
DeepSeek's TileKernels, Christopher Owen's sparknet, NVIDIA NCCL/CUTLASS and
other inputs listed in the pinned project's
[upstream source map](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/docs/upstreams.md).
Those projects retain their own copyright notices and licenses in the upstream
source and built image. DeepSeek supplies the unchanged model checkpoint;
its model license remains separate. No weights or images are distributed here.
