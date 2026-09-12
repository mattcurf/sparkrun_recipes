#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VLLM_SHA=e47aa780bccf59f59dfa2cbb18e17a10b4fe69ba
BASE=vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c
IMAGE=${IMAGE:-deepseek-v4.1-flash-sparkrun-v1}
VLLM_DIR=${VLLM_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun-build/deepseek-v4.1-flash/vllm}

if [[ ! -d "$VLLM_DIR/.git" ]]; then
  git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
fi
git -C "$VLLM_DIR" fetch origin "$VLLM_SHA"
git -C "$VLLM_DIR" checkout --detach --force "$VLLM_SHA"
[[ $(git -C "$VLLM_DIR" rev-parse HEAD) == "$VLLM_SHA" ]]

docker pull "$BASE"
docker rm -f dsv41-extension-build >/dev/null 2>&1 || true
docker run -d --name dsv41-extension-build \
  -v "$VLLM_DIR:/src" --entrypoint sleep "$BASE" infinity >/dev/null
cleanup() { docker rm -f dsv41-extension-build >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker exec dsv41-extension-build python3 -m pip install --no-cache-dir cmake ninja

# V4.1's stable extension must include Blackwell SM 12.1a kernels. Keep compile
# parallelism low because compiler memory and GPU unified memory share DRAM.
docker exec dsv41-extension-build bash -ceu '
  export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=2 NVCC_THREADS=1
  cd /src
  mkdir -p build/_deps/cutlass-src
  if [[ ! -f build/_deps/cutlass-src/include/cutlass/cutlass.h ]]; then
    rm -rf build/_deps/cutlass-src
    mkdir -p build/_deps/cutlass-src
    curl -fsSL --retry 3 https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.7.1 \
      | tar xz -C build/_deps/cutlass-src --strip-components=1
  fi
  # checkout --force above restores this tracked file before every build.
  sed -i -E "s|^(\\s*)include\\(cmake/external_projects/|\\1# stable-only: include(cmake/external_projects/|" CMakeLists.txt
  rm -rf build/CMakeCache.txt build/CMakeFiles
  pypath=$(python3 -c "import sys; print(\":\".join(p for p in sys.path if p))")
  torch_prefix=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
  nvrtc=$(find /usr/local/cuda /usr/local/lib/python3.12/dist-packages/nvidia /usr/lib/aarch64-linux-gnu \
    -name "libnvrtc.so*" -print 2>/dev/null | head -1)
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVLLM_TARGET_DEVICE=cuda \
    -DVLLM_PYTHON_EXECUTABLE="$(command -v python3)" -DVLLM_PYTHON_PATH="$pypath" \
    -DFETCHCONTENT_BASE_DIR=/src/build/_deps \
    -DFETCHCONTENT_SOURCE_DIR_CUTLASS=/src/build/_deps/cutlass-src \
    -DCMAKE_PREFIX_PATH="$torch_prefix" -DNVCC_THREADS=1 -DCUDA_nvrtc_LIBRARY="$nvrtc"
  cmake --build build --target _C_stable_libtorch -j 2
  find build -name "_C_stable_libtorch*.so" -exec ls -lh {} +
  cp build/_C_stable_libtorch*.so vllm/
'

overlay_context=$(mktemp -d)
trap 'rm -rf "$overlay_context"; cleanup' EXIT
cp "$ROOT/Dockerfile" "$overlay_context/"
cp -R "$ROOT/patches" "$overlay_context/patches"
# Stage the Python package, not the vLLM repository root. Copying the checkout
# itself under site-packages leaves the base image's control-plane in place.
cp -R "$VLLM_DIR/vllm" "$overlay_context/vllm-package"
docker build -t dsv41-overlay1 "$overlay_context"

tmp=$(mktemp -d)
trap 'rm -rf "$overlay_context" "$tmp"; cleanup' EXIT
cat >"$tmp/Dockerfile" <<'EOF'
FROM dsv41-overlay1
ARG FI_SHA=07869c61ba581e6d6b8ad8d142f4a6c89b707cc1
ARG CUTLASS_SHA=b46b16d003484063bca4ed365e44095c4c6ed633
ARG CCCL_SHA=16bd510c9b712e82b0ab6cbb630d8e29ba1f7116
ARG SPDLOG_SHA=c3aed4b68373955e1cc94307683d44dca1515d2b
RUN pip uninstall -y flashinfer-jit-cache flashinfer-cubin flashinfer-python || true
RUN mkdir -p /opt/fi-src/3rdparty/{cutlass,cccl,spdlog} \
 && curl -fsSL --retry 3 "https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/${FI_SHA}" | tar xz -C /opt/fi-src --strip-components=1 \
 && curl -fsSL --retry 3 "https://codeload.github.com/NVIDIA/cutlass/tar.gz/${CUTLASS_SHA}" | tar xz -C /opt/fi-src/3rdparty/cutlass --strip-components=1 \
 && curl -fsSL --retry 3 "https://codeload.github.com/NVIDIA/cccl/tar.gz/${CCCL_SHA}" | tar xz -C /opt/fi-src/3rdparty/cccl --strip-components=1 \
 && curl -fsSL --retry 3 "https://codeload.github.com/gabime/spdlog/tar.gz/${SPDLOG_SHA}" | tar xz -C /opt/fi-src/3rdparty/spdlog --strip-components=1 \
 && cd /opt/fi-src \
 && BUILD_NVEP=0 FLASHINFER_BUILD_NO_PIP=1 pip install --no-deps --no-build-isolation .
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a MAX_JOBS=2 FLASHINFER_NVCC_THREADS=1 \
  python3 -c "from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as g; s=g(); b=getattr(s,'build',None); b(verbose=True) if b else s.build_and_load()"
RUN FLASHINFER_CUDA_ARCH_LIST=12.1a TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 VLLM_HAS_FLASHINFER_CUBIN=1 \
    FLASHINFER_NVCC_THREADS=1 MAX_JOBS=2 \
  python3 -c "from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module as g; g()"
RUN python3 -c "from flashinfer.mla import supported_sparse_mla_sm120_configs as f; assert f()['dsv4'].supports_decode(num_heads=16, topk=1152)"
EOF
docker build -t "$IMAGE" "$tmp"
docker image inspect "$IMAGE" --format '{{.Id}}'
