# GLM-5.3-Flash EXL3 with DFlash2

This recipe serves
[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
on two DGX Sparks.
It uses EXL3/TR3 4 bpw weights, TP2, and FP8 KV cache.
It uses seven-token DFlash2 speculative decoding.
This variant uses a conservative 196,608-token maximum context on a 2×128 GB Spark pair.

## Run

1. Install [SparkRun](https://sparkrun.dev/).
2. Configure SparkRun.
3. Run these commands:

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-exl3.yaml
sparkrun run glm-5.3-flash-exl3.yaml --cluster <two-node-cluster>
```

The first launch downloads and distributes approximately 164 GB of target weights.
It also downloads and distributes the DFlash2 draft model and the pinned container image.
The OpenAI-compatible API listens on port 8000.
It uses model ID `GLM-5.3-Flash-EXL3`.

The recipe requires SparkRun 0.3.6 or newer for the native `vllm-distributed` runtime.
It pins the target and draft model revisions.
The custom image extends Spark-Arena's arm64 image.
Spark-Arena built that image from MiaAI-Lab commit `94f5d38bfa8792e3ac2fe5dc75075440a21db7c7`.
The custom image adds MiaAI-Lab's later K-pool tail mapping fixes from commit `c707598ebcf02fd827d079a7c47e785069425efe`.
It also adds that commit's sparse-indexer fixes to size the workspace correctly.
The build script downloads only those three runtime patch files.
It verifies the pinned source archive before building.

## Verify

```bash
curl -f http://<head-node>:8000/health
curl -s http://<head-node>:8000/v1/models
curl -s http://<head-node>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"GLM-5.3-Flash-EXL3",
    "messages":[{"role":"user","content":"Reply with exactly: server ready"}],
    "chat_template_kwargs":{"enable_thinking":false},
    "temperature":0,
    "max_tokens":32
  }'
```

## Attribution and licensing

This tuned recipe adapts the MIT-licensed experimental recipe in
[`spark-arena/recipe-registry`](https://github.com/spark-arena/recipe-registry).
That recipe uses MiaAI-Lab's MIT-licensed
[`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
The derived YAML and Docker build retain the upstream MIT License.
We dedicate original documentation in this directory to the public domain under the Unlicense.

This directory does not distribute model weights, the DFlash2 draft model, the container image, vLLM, or SparkRun.
These components retain their upstream terms.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
We provide everything **as is**, without warranty.
