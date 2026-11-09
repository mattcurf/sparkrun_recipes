# GLM-5.3-Flash EXL3 with DFlash2

Use two 128 GB DGX Sparks with SparkRun 0.3.6 or newer.

| Setting | Value |
| --- | --- |
| Target | [`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw), pinned revision |
| Runtime | Native `vllm-distributed`; TP2; FP8 KV |
| Context | 196,608 tokens maximum |
| Draft | Pinned DFlash2; seven speculative tokens |
| API | Port 8000; `GLM-5.3-Flash-EXL3` |
| Distribution | Approximately 164 GB target weights, plus draft and pinned image |

## Run

1. Install and configure [SparkRun](https://sparkrun.dev/).
2. Run from this directory:

```bash
./build-image.sh
sparkrun recipe validate glm-5.3-flash-exl3.yaml
sparkrun run glm-5.3-flash-exl3.yaml --cluster <two-node-cluster>
```

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

This recipe adapts the MIT-licensed experimental recipe in
[`spark-arena/recipe-registry`](https://github.com/spark-arena/recipe-registry).
That recipe uses MiaAI-Lab's MIT-licensed
[`GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).
The derived YAML and Docker build retain the upstream MIT License.
We dedicate original documentation in this directory to the public domain under the Unlicense.

Weights, images, vLLM, and SparkRun retain upstream terms; this directory does not distribute them.
See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for pinned runtime sources and terms.
We provide everything **as is**, without warranty.
