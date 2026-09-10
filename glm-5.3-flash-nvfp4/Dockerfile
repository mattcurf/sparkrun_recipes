# SPDX-License-Identifier: Unlicense
FROM vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5

# Native 512-wide NoPE sparse attention for SM121; no EXL3/model-weight patches.
RUN mkdir -p /tmp/glm53-sparse-mla \
    && curl -fsSL https://codeload.github.com/Libertai/vllm-sparse-mla-blackwell/tar.gz/3701f85be073ff75578bdffb22469161223b09fc -o /tmp/glm53-sparse-mla.tar.gz \
    && echo 'ad22cce654ccfee2a6fbb7f9075c6e4b6ac713589efaf694b7501c19cdee1852  /tmp/glm53-sparse-mla.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/glm53-sparse-mla.tar.gz --strip-components=1 -C /tmp/glm53-sparse-mla

COPY patch_sparse_mla.py /tmp/patch_sparse_mla.py
RUN python3 /tmp/patch_sparse_mla.py /tmp/glm53-sparse-mla/glm53_sparse_mla/backend.py \
    && GLM53_ARCHS=121a MAX_JOBS=4 pip install --no-deps --no-build-isolation /tmp/glm53-sparse-mla \
    && mkdir -p /opt/glm53-sparse-mla \
    && cp -r /tmp/glm53-sparse-mla/tests /opt/glm53-sparse-mla/tests \
    && cp /tmp/glm53-sparse-mla/LICENSE /opt/glm53-sparse-mla/LICENSE \
    && rm -rf /tmp/glm53-sparse-mla /tmp/glm53-sparse-mla.tar.gz /tmp/patch_sparse_mla.py
