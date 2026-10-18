# Third-party notices

## Recipe provenance and changes

The serving configuration is adapted from the sibling
[DeepSeek V4.1 recipe](../deepseek-v4.1-flash-mxfp4-4node/) and Tony Deangelo's
[DGX Spark integration](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/fc725ecf10869c184f4347dd73336536d395753c).
The latter is Copyright (c) 2026 Tech2wild, MIT License; its complete notice
is retained in [LICENSE-MIT](LICENSE-MIT).

This adaptation pins the 1017 model revision, changes the served model name
and checkpoint location, enables maximum thinking with retained reasoning,
reads Engram tables from the new checkpoint instead of older staged rows,
and keeps writable compiler caches in node-local runtime storage. The shared
runtime build and its vendored patches are reused without modification.
Original additions are released under the [Unlicense](LICENSE); this does
not replace the licenses or copyright notices of upstream material.

## Model and runtime dependencies

- **DeepSeek-V4.1-Flash-1017** — Copyright (c) 2023 DeepSeek, MIT License.
  Model and reference code are downloaded separately at revision
  `1bae0e1088993b9b82de08f59de5be5a34e42170` from
  [the model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
  Its [upstream license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE)
  text is retained in [LICENSE-MODEL-MIT](LICENSE-MODEL-MIT), with line
  endings normalized to LF.
- **vLLM** — Copyright the vLLM contributors, Apache License 2.0.
  The [shared build](../deepseek-v4.1-flash-mxfp4-4node/build-image.sh) pins
  [vLLM source](https://github.com/vllm-project/vllm/tree/e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba)
  and reuses the original recipe's Apache-2.0-derived replacement files with
  their SPDX and copyright headers intact.
- **FlashInfer** — Copyright the FlashInfer contributors, Apache License 2.0,
  pinned by the shared build to
  [07869c61](https://github.com/flashinfer-ai/flashinfer/tree/07869c61ba581e6d6b8ad8d142f4a6c89b707cc1).
- **SparkRun** — Copyright the SparkRun contributors, Apache License 2.0,
  <https://github.com/spark-arena/sparkrun>.

The shared runtime's [third-party notices](../deepseek-v4.1-flash-mxfp4-4node/THIRD_PARTY_NOTICES.md)
and complete [Apache-2.0 license text](../deepseek-v4.1-flash-mxfp4-4node/LICENSE-APACHE)
remain alongside its build and patch files. No model weights, container
images, or additional copies of runtime patches are distributed here.

## Evaluation attribution

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench/tree/42a100b18ff7b959b22304ed7a36ec5f652dd88a)
by SeraphimSerapis and contributors, MIT License, was used without source or
scoring changes for the results reported in [RESULTS.md](RESULTS.md).
No benchmark source, scenario fixtures, or conversation traces are copied here.
