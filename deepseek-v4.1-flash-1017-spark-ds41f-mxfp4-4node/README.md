# DeepSeek V4.1 Flash 1017: spark-ds41f, four Sparks

This recipe provides an alternative to TensorFold.
It adapts [Christopher Owen's spark-ds41f](https://github.com/christopherowen/spark-ds41f) 1107 TileLang/vLLM release.

| Item | Pinned value |
| --- | --- |
| Source revision | `cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5` |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| Checkpoint revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Release file | `releases/spark-ds41f-1107/cluster.json` |
| Release profile | `tp4-switched` |

The recipe uses the original checkpoint.
It does not convert the weights.
It does not require new weights.

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

SparkRun's native distributed runtime supplies the rank and rendezvous flags once.
This recipe does not use Ray.
It does not use TensorFold rank files.
It does not use the older disk-Engram patches.
It does not use a stock vLLM image.

## Changes from upstream

The recipe reduces KV memory from 10.5 GiB to 6 GiB per rank.
This change leaves more memory for allocations.
The first 10.5-GiB trial failed during startup.
The smaller KV pool holds fewer simultaneous contexts.
It does not change the per-request context limit.
It does not change model arithmetic.
The engine reports 6,182,120 KV tokens.
This capacity equals 5.90 full 1M contexts, not sixteen full contexts.

The recipe keeps the existing served model name.
It sets `OMP_NUM_THREADS=1` to follow this repository's policy.
It omits GLP steering.
It also omits the second model name.

The runtime generates 264-byte packed Engram rows under `/cache`.
It uses a separate directory for this checkpoint and release.
It does not write beside the read-only checkpoint blobs.
This disk layout does not re-quantize the weights.

The recipe does not change the kernel, firewall, endpoint, host recovery or routing.

## Build without downloading the checkpoint

**Stop inference before you build the image.**
Use an approved maintenance window for the build.

```bash
./build-image.sh --prepare-only
# Only after inference is stopped in an authorized maintenance window:
./build-image.sh --build
```

The helper uses upstream's source locks, patches and Dockerfile.
It also uses upstream's build checks and GPU import test.
It refuses a changed source checkout.
It refuses to overwrite an existing image tag.

Set `SPARK_DS41F_SOURCE` to reuse a checkout at the exact source revision.
Preparation downloads source code and build wheels.
It does not download model weights.
It does not start or stop inference.

The helper creates an ignored example node map for the build tools.
Do not use that map to deploy the model.
Copy the finished image to every node.
Check that every node has identical image contents.
Keep all build and runtime data on node-local NVMe.
Do not use NFS.

The helper disables inherited Git signing only for child build processes.
It uses the recorded committer email for vLLM's final 0761 patch.
This reproduces upstream's exact commit and source tree.
It does not change the patch.
It does not weaken source checks.

## Prepare a private deployment configuration

**Do not launch the public recipe directly.**
Create a private deployment overlay first.
Include these settings:

1. Mount the existing snapshot and its entire hub cache read-only.
   The full cache lets relative blob links resolve.
2. Mount persistent node-local NVMe storage at `/cache`.
   Engram files and JIT caches use this mount.
3. Render sparknet's `oneshot-switched` settings for your fabric.
   Set the HCA names, GIDs, subnets, control interfaces and rank addresses.
4. Read the origin authentication key from an existing private file.
   Never put the key in YAML.

Keep host inventories, usernames, storage paths and endpoint mappings in the private overlay.
Keep credentials out of this directory.

The public recipe permits 16 images in each request.
This count includes images in earlier messages.
A private deployment can raise `--limit-mm-per-prompt` for longer image histories.
That change raises admission limits only.
It does not increase the scheduler's encoder-token budget.

## Check host requirements

Upstream qualifies TP4 with 64-KiB pages and memory saver.
The stock 4-KiB configuration remains an unqualified site trial.
Do not assume equal memory capacity.
Do not install a kernel or reboot to meet this recipe's requirements.

Check free memory on all four nodes during startup.
Continue these checks while the model serves requests.
Stop the trial if memory becomes unsafe.
Upstream specifies a 5-GiB startup minimum.
It specifies a 3-GiB steady-state minimum.

## Check the recipe and start safely

```bash
sparkrun recipe validate deepseek-v4.1-flash-1017-spark-ds41f-tp4.yaml
# Use the private rendered recipe, not this incomplete site-independent one:
sparkrun run /path/to/rendered.yaml --cluster YOUR_CLUSTER --port 8000 --dry-run
```

Never run two TP4 engines on the same nodes.
Stop the existing runtime as one group before you build or launch.
No pre-exec hook clears live caches.

Keep related settings consistent when you tune the recipe:

- Context limit.
- Sequence count.
- Speculative depth.
- Graph ladder.
- Prefill threshold.

Wait for authenticated `/v1/models` access.
Check every rank for errors and fallback paths.
Test streaming and tools before manual use.

## Test results

Recipe checks and dry runs test offline configuration only.
They do not prove a successful build or startup.
This recipe does not claim upstream measurements as local results.

### First trial: 10.5-GiB KV

The four-node trial on 2026-11-08 used 4-KiB pages.
It **did not reach API readiness**.
The image passed GPU, TileLang and sparknet import checks.
Each rank read all 48 checkpoint shards through the read-only mount.
Startup generated the NVMe Engram layout.
It then reached DSpark graph and tail warmup.
Workers reported a RoCE wait for rank 0.
The driver reported allocation errors.
Host memory stayed above 21 GiB.
The host-memory minimum did not catch this failure.
The cause remains unresolved.
Do not blame page size without more evidence.
Operators stopped the trial and restored the previous runtime at that time.

### Later trial: 6-GiB KV

The 6-GiB configuration passed startup on all four nodes.
The image, checkpoint, DSpark, graphs, sequence count and 1M request limit stayed unchanged.
Startup guards remained active through readiness.
Available memory stayed above 25.51 GiB.
The monitors found no new driver allocation errors or Xids.

The following checks passed:

- Allowed and denied authentication.
- Streaming with usage counts.
- Thinking.
- Automatic tools and tool-result round trips.
- Eight concurrent requests.
- Retrieval from 118,896 prompt tokens.

The vLLM counter matched all 166 generated tokens.
Storage checks confirmed NVMe for weights and derived caches on every node.

The matched benchmark completed 86 requests.
It generated 20,102 output tokens.
Prompt hashes and token counts matched the saved prior-runtime run.
Neither run reported cached or reasoning tokens.
No other traffic overlapped the comparison.
The candidate reported no request failures or new driver errors.
The monitors found no swap growth.
vLLM reported no preemptions.
The candidate stayed running after the test.
The private deployment repository holds the measurements and operational evidence.

### Limits of these tests

The benchmark covers one boot per engine with retained JIT caches.
Each decode point has three repeats.
The comparison has no independent boot repeats.
The prior-runtime run has no matching thermal measurements.
These tests do not prove the first failure's cause.
They do not qualify full 1M capacity.
They do not establish long-term stability.

A later restart used a private 999-image admission override.
Image-history retrieval, authentication, streaming, thinking and tool-result checks passed after that restart.
The public recipe still defaults to 16 images.
The private limit does not guarantee that 999 images fit.

## Attribution and licensing

Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f) supplies the serving integration.
It also supplies the patched TileLang kernel stack.
Its configuration supplies DSpark and sparknet collective integration.
This recipe and build helper adapt its frozen
[1107 cluster configuration](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/releases/spark-ds41f-1107/cluster.json).

These adaptations use **Apache-2.0**.
See the separate [LICENSE](LICENSE) and [NOTICE](NOTICE).
The repository's default Unlicense does not replace Apache-2.0.

The serving stack also uses these projects:

- vLLM and Local Inference Lab.
- tile-ai/TileLang.
- DeepSeek's TileKernels.
- Christopher Owen's sparknet.
- NVIDIA NCCL and CUTLASS.

The pinned project's [upstream source map](https://github.com/christopherowen/spark-ds41f/blob/cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5/docs/upstreams.md) lists the other inputs.
These projects keep their own copyright notices and licenses.
The upstream source and image contain those notices.
DeepSeek supplies the unchanged model checkpoint.
Its model license remains separate.
This repository does not distribute weights or images.
