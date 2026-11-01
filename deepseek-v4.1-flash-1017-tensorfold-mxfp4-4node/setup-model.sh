#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds/copies assets only; never starts or stops the inference ranks.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RECIPE_REF=476bbc77c4e1d1abe45b1cd095526a9ea0d36da4
SOURCE_DIR=${SOURCE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/sparkrun-build/deepseek-v4.1-flash-1017-tensorfold}
[[ -f "$ROOT/local.env" ]] || { echo 'Copy local.env.example to local.env and review the README first.' >&2; exit 1; }
if [[ ! -e "$SOURCE_DIR" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone --no-checkout https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold.git "$SOURCE_DIR"
  git -C "$SOURCE_DIR" checkout --detach "$RECIPE_REF"
fi
[[ $(git -C "$SOURCE_DIR" rev-parse HEAD) == "$RECIPE_REF" ]] || { echo "Wrong revision in $SOURCE_DIR; left unchanged." >&2; exit 1; }
if [[ -z $(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no) ]]; then
  git -C "$SOURCE_DIR" apply --check "$ROOT/patches/local-ssd.patch"
  git -C "$SOURCE_DIR" apply "$ROOT/patches/local-ssd.patch"
else
  cmp "$ROOT/patches/local-ssd.patch" <(git -C "$SOURCE_DIR" diff --binary) || {
    echo "Unexpected edits in $SOURCE_DIR; left unchanged." >&2; exit 1;
  }
fi
if [[ -e "$SOURCE_DIR/.env" ]]; then
  cmp "$ROOT/local.env" "$SOURCE_DIR/.env" || { echo "Existing .env differs in $SOURCE_DIR; review it manually." >&2; exit 1; }
else
  cp "$ROOT/local.env" "$SOURCE_DIR/.env"
fi
(
  cd "$SOURCE_DIR"
  source scripts/config.sh
  source scripts/nodes.sh
  [[ "$DERISKED" == 0 && "$ABLIT" == 0 && "$ENGRAM_PACKED" == 0 ]] || die "This recipe captures the stock, unpacked checkpoint only"
  # Even the CPU-only preparation competes for unified memory. Do not build,
  # convert or copy hundreds of GB while the cluster is serving requests.
  (( $(awk '/MemAvailable:/ {print $2}' /proc/meminfo) >= 104857600 )) || die "Need 100 GiB available on the head; leave the live server alone until maintenance"
  for i in 1 2 3; do
    free_kib=$(worker "$i" "awk '/MemAvailable:/ {print \$2}' /proc/meminfo")
    (( free_kib >= 104857600 )) || die "Need 100 GiB available on $(worker_host "$i"); do not reclaim memory during serving"
  done
  ./scripts/prepare.sh
)
echo "Assets prepared. No inference containers started. Read the README before the separate SparkRun launch."
