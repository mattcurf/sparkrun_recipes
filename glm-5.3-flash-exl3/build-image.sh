#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

SOURCE_REVISION=c707598ebcf02fd827d079a7c47e785069425efe
SOURCE_SHA256=5c845d2e2f75e43c565f27a53f7fafd5eef77658f81b9434b0ed4bf0e6d783a8
IMAGE=${IMAGE:-glm53-flash-exl3-sparkrun-v1}
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="$ROOT/.build"
ARCHIVE="$BUILD_DIR/source.tar.gz"
SOURCE_DIR="GLM-5.3-Flash-EXL3-2x-DGX-Sparks-$SOURCE_REVISION"

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR"
curl -L --fail --retry 3 \
  "https://codeload.github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tar.gz/$SOURCE_REVISION" \
  -o "$ARCHIVE"
printf '%s  %s\n' "$SOURCE_SHA256" "$ARCHIVE" | sha256sum --check

for patch in patch_kpool_tail_slotmap.py patch_indexer_workspace.py patch_spinwait.py; do
  tar -xzf "$ARCHIVE" -C "$BUILD_DIR" --strip-components=2 \
    "$SOURCE_DIR/overlay/$patch"
done

docker build --pull=false -t "$IMAGE" "$ROOT"
