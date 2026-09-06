# SPDX-License-Identifier: MIT

FROM ghcr.io/spark-arena/glm-5.3-flash-2x-dgx-sparks:exl3-94f5d38@sha256:56e5c29b32b7cd651e2378ad7caf5395c736dab6bb013b0d209d4b689c33ad21

COPY .build/patch_kpool_tail_slotmap.py /opt/glm53/patch_kpool_tail_slotmap.py
COPY .build/patch_indexer_workspace.py /opt/glm53/patch_indexer_workspace.py
COPY .build/patch_spinwait.py /opt/glm53/patch_spinwait.py

RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py \
    && python3 /opt/glm53/patch_indexer_workspace.py \
    && GLM53_SPINWAIT_MS=stock python3 /opt/glm53/patch_spinwait.py --preflight
