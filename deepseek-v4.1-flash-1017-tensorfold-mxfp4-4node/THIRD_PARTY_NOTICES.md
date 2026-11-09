# Third-party notices

- **MiaAI-Lab DeepSeek-V4.1-Flash-1017 four-Spark TensorFold recipe** —
  Apache-2.0, Copyright 2026 MiaAI-Lab.
  <https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold>
  This recipe pins revision `476bbc77c4e1d1abe45b1cd095526a9ea0d36da4`.
  It keeps the original [NOTICE](NOTICE) and [LICENSE](LICENSE).
  The local patch replaces worker NFS storage with local SSD storage and SSH copies.
  It preserves old weight files.
  It adds file-based API authentication.
  It adds a model alias.
  It enforces memory checks.
  This repository includes its regression-test changes.
  Preparation fetches runtime patches and the chat template from the pinned revision.
  This repository does not store those fetched files.
- **TensorFold / BertholomusAI DeepSeek-V4.1 integration** — Apache-2.0.
  <https://github.com/ashhart/TensorFold>
  <https://github.com/bertholomus/TensorFold>
- **BertholomusAI kernel kit** — Apache-2.0 with additional component licenses
  listed in its `kit/KIT-LICENSES.md`.
  <https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10>
- **DeepSeek-V4.1-Flash-1017** — DeepSeek, MIT-licensed weights.
  <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>
  Preparation downloads the weights separately at their pinned revision.
- **SparkRun** — Apache-2.0.
  <https://github.com/spark-arena/sparkrun>

The new SparkRun YAML, runtime and preparation wrapper use Apache-2.0.
The fetched runtime's NVIDIA base image keeps its own license.
Bundled third-party components keep their own licenses.
See the upstream NOTICE and image license files.
