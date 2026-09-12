# GLM-5.3-Flash EXL3 with DFlash2

Serves
[`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw)
on two DGX Sparks with EXL3/TR3 4 bpw weights, TP2, FP8 KV cache, and
seven-token DFlash2 speculative decoding. This variant uses a conservatively
sized 196,608-token maximum context on a 2×128 GB Spark pair.

## Run

Install and configure [SparkRun](https://sparkrun.dev/), then run:

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-exl3.yaml
sparkrun run glm-5.3-flash-exl3.yaml --cluster <two-node-cluster>
```

The first launch downloads and distributes approximately 164 GB of target
weights, the DFlash2 draft model, and the pinned container image. The
OpenAI-compatible API listens on port 8000 with model ID
`GLM-5.3-Flash-EXL3`.

The recipe requires SparkRun 0.3.6 or newer for the native
`vllm-distributed` runtime. It pins the target and draft model revisions. The
custom image extends Spark-Arena's arm64 image built from MiaAI-Lab commit
`94f5d38bfa8792e3ac2fe5dc75075440a21db7c7`. It adds MiaAI-Lab's later K-pool
tail mapping and sparse-indexer workspace right-sizing fixes from commit
`c707598ebcf02fd827d079a7c47e785069425efe`. The build script downloads only
those three runtime patch files and verifies the pinned source archive before
building.

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

This tuned recipe is derived from the MIT-licensed experimental recipe in the
[`spark-arena/recipe-registry`](https://github.com/spark-arena/recipe-registry),
which is based on MiaAI-Lab's MIT-licensed
[`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
The derived YAML and Docker build retain the upstream MIT License. Original
documentation in this directory is dedicated to the public domain under the
Unlicense.

Model weights, the DFlash2 draft model, container image, vLLM, and SparkRun are
not distributed here and retain their upstream terms. See `LICENSE` and
`THIRD_PARTY_NOTICES.md`. Everything is provided **as is**, without warranty.
