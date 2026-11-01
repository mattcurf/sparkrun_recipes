# Third-party notices

- **MiaAI-Lab DeepSeek-V4.1-Flash-1017 four-Spark TensorFold recipe** —
  Apache-2.0, Copyright 2026 MiaAI-Lab.
  <https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold>
  Pinned at `476bbc77c4e1d1abe45b1cd095526a9ea0d36da4`; the original
  [NOTICE](NOTICE) and [LICENSE](LICENSE) are retained. The local patch changes
  worker storage from NFS to local SSD/SSH copies, preserves old weight files,
  adds file-based API authentication/model alias, and enforces memory checks.
  Its regression test changes are included. Runtime patches and the chat
  template are fetched from that pin, not vendored here.
- **TensorFold / BertholomusAI DeepSeek-V4.1 integration** — Apache-2.0.
  <https://github.com/ashhart/TensorFold>
  <https://github.com/bertholomus/TensorFold>
- **BertholomusAI kernel kit** — Apache-2.0 with additional component licenses
  listed in its `kit/KIT-LICENSES.md`.
  <https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10>
- **DeepSeek-V4.1-Flash-1017** — DeepSeek, MIT-licensed weights, separately
  downloaded at their pinned revision.
  <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>
- **SparkRun** — Apache-2.0.
  <https://github.com/spark-arena/sparkrun>

The new SparkRun YAML/runtime and preparation wrapper are Apache-2.0.
The fetched runtime's NVIDIA base image and bundled third-party components
retain their own licenses; see the upstream NOTICE and image license files.
