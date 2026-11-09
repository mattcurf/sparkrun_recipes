# Serving checks and benchmark limits

Measurements cover the recipe's 6-GiB KV configuration on four Sparks with 4-KiB pages.
Upstream qualification requires 64-KiB pages and memory saver.

## Serving checks

| Check | Measured result |
| --- | --- |
| Authentication | Allowed and denied requests pass |
| Streaming | Usage counts pass |
| Thinking | Pass |
| Tools | Automatic calls and tool-result round trips pass |
| Concurrency | Eight requests pass |
| Retrieval | 118,896 prompt tokens pass |
| Token accounting | All 166 generated tokens match the vLLM counter |
| Storage | Weights and derived caches use NVMe on every node |
| Available host memory | Minimum 25.51 GiB |
| Driver allocation errors / Xids | None |

## Matched benchmark

| Item | Value |
| --- | --- |
| Completed requests | 86 |
| Output tokens | 20,102 |
| Prompt hashes and token counts | Match the saved baseline |
| Cached / reasoning tokens | Zero in both runs |
| Other traffic | None |
| Request failures / new driver errors | None |
| Swap growth / vLLM preemptions | None |
| Boots per engine | One, with retained JIT caches |
| Repeats per decode point | Three |
| Matched baseline thermal measurements | None |

The private deployment repository holds the measurements and operational evidence.
These results do not qualify full 1M capacity or long-term stability.

## Image admission

The public recipe permits 16 images per request, including message history.
A private 999-image admission setting passes image-history retrieval, authentication, streaming, thinking, and tool-result checks.
That setting does not establish capacity for 999 images.
