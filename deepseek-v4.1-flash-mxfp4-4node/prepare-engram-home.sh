#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ENGRAM_DIR="$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash"
MOUNT_SOURCE=${1:-/var/tmp/sparkrun-deepseek-v4.1-flash-engram}

mkdir -p "$ENGRAM_DIR"
if [[ -L "$MOUNT_SOURCE" ]]; then
  current_target=$(readlink -- "$MOUNT_SOURCE")
  if [[ "$current_target" != "$ENGRAM_DIR" ]]; then
    echo "Refusing to replace $MOUNT_SOURCE; it points to $current_target" >&2
    exit 1
  fi
elif [[ -e "$MOUNT_SOURCE" ]]; then
  echo "Refusing to replace existing non-symlink $MOUNT_SOURCE" >&2
  exit 1
else
  ln -s -- "$ENGRAM_DIR" "$MOUNT_SOURCE"
fi

printf '%s\n' "$ENGRAM_DIR"
