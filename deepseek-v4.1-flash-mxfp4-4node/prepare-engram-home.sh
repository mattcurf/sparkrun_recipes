#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

ENGRAM_DIR="$HOME/.local/share/sparkrun/engram/DeepSeek-V4.1-Flash"
MOUNT_SOURCE=${1:-/srv/sparkrun/engram/DeepSeek-V4.1-Flash}

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
  if [[ "$MOUNT_SOURCE" == /srv/sparkrun/engram/DeepSeek-V4.1-Flash ]]; then
    docker run --rm --entrypoint sh \
      -v /srv:/host-srv deepseek-v4.1-flash-sparkrun-v1 \
      -c 'mkdir -p /host-srv/sparkrun/engram && ln -s -- "$1" /host-srv/sparkrun/engram/DeepSeek-V4.1-Flash' \
      sh "$ENGRAM_DIR"
  else
    # An alternate mount source is accepted only for isolated tests.
    mkdir -p "$(dirname -- "$MOUNT_SOURCE")"
    ln -s -- "$ENGRAM_DIR" "$MOUNT_SOURCE"
  fi
fi

printf '%s\n' "$ENGRAM_DIR"
