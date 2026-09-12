# SPDX-License-Identifier: MIT
FROM vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c

# The patch set is tied to this exact vLLM tree. build-image.sh verifies the
# checkout and stages its Python package before Docker can copy it.
COPY vllm-package/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY patches/sparse_swa.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py
COPY patches/attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/attention.py
COPY patches/flashinfer_sparse.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/flashinfer_sparse.py
COPY patches/engram.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py
COPY patches/weight_utils.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py
COPY patches/model_state.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py
COPY patches/sparse_attn_indexer.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer.py

RUN find /usr/local/lib/python3.12/dist-packages/vllm -name __pycache__ -type d -prune -exec rm -rf {} + \
 && python3 -m py_compile \
      /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
      /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py \
 && python3 -c 'import vllm; print(vllm.__version__)'

ENV VLLM_HAS_FLASHINFER_CUBIN=1
