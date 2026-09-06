#!/usr/bin/env bash
set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE="${1:-qwen38-flash-next-nvfp4-mtp-v1}"

docker build --tag "$IMAGE" "$SCRIPT_DIR"
docker image inspect --format 'Built {{.RepoTags}} ({{.Id}})' "$IMAGE"
