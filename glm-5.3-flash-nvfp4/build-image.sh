#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
docker build --tag glm53-flash-nvfp4-sparkrun-v1 "$ROOT"
