# Third-party notices

<a id="recipe-provenance-and-changes"></a>

## Recipe provenance

This serving configuration adapts the sibling
[DeepSeek V4.1 recipe](../deepseek-v4.1-flash-mxfp4-4node/).
It also adapts Tony Deangelo's
[DGX Spark integration](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark/tree/fc725ecf10869c184f4347dd73336536d395753c).
The latter is Copyright (c) 2026 Tech2wild, MIT License;
[LICENSE-MIT](LICENSE-MIT) retains its complete notice.

This recipe uses the shared runtime build and vendored patches without modification.

Original additions use the [Unlicense](LICENSE).
This does not replace upstream licenses or copyright notices.

## Model and runtime dependencies

- **DeepSeek-V4.1-Flash-1017** — Copyright (c) 2023 DeepSeek, MIT License.
  Download the model and reference code separately at revision `1bae0e1088993b9b82de08f59de5be5a34e42170` from
  [the model repository](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
  [LICENSE-MODEL-MIT](LICENSE-MODEL-MIT) retains its
  [upstream license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/LICENSE) text.
- **vLLM** — Copyright the vLLM contributors, Apache License 2.0.
  The [shared build](../deepseek-v4.1-flash-mxfp4-4node/build-image.sh) pins
  [vLLM source](https://github.com/vllm-project/vllm/tree/e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba).
  The shared Apache-2.0-derived replacement files retain their SPDX and copyright headers.
- **FlashInfer** — Copyright the FlashInfer contributors, Apache License 2.0,
  pinned by the shared build to
  [07869c61](https://github.com/flashinfer-ai/flashinfer/tree/07869c61ba581e6d6b8ad8d142f4a6c89b707cc1).
- **SparkRun** — Copyright the SparkRun contributors, Apache License 2.0,
  <https://github.com/spark-arena/sparkrun>.

The shared runtime's [third-party notices](../deepseek-v4.1-flash-mxfp4-4node/THIRD_PARTY_NOTICES.md)
and complete [Apache-2.0 license text](../deepseek-v4.1-flash-mxfp4-4node/LICENSE-APACHE)
apply to its build and patch files.
This directory does not distribute model weights, container images, or additional copies of runtime patches.

## Evaluation attribution

[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench/tree/42a100b18ff7b959b22304ed7a36ec5f652dd88a)
by SeraphimSerapis and contributors, MIT License, supplies the evaluation in [RESULTS.md](RESULTS.md).
The evaluation uses unchanged benchmark source and scoring.
This directory contains no copies of benchmark source, scenario fixtures, or conversation traces.
