#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 SHARED_MODEL_DIR RANK0_HOST RANK1_HOST RANK2_HOST RANK3_HOST" >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=$1
shift
HOSTS=("$@")
REVISION=dba1be0a40aa45a94ad051997016db3960a90277

hf download deepseek-ai/DeepSeek-V4.1-Flash --revision "$REVISION" \
  --local-dir "$MODEL_DIR" --max-workers 8
[[ -f "$MODEL_DIR/model-00048-of-00048.safetensors" ]]
printf '%s\n' "$REVISION" >"$MODEL_DIR/.sparkrun-model-revision"

ranges=(
  ''
  '1:96000564:192001740 14:96003054:192007016'
  '1:192001740:288003654 14:192007016:288011564'
  '1:288003654:384006168 14:288011564:384016682'
)

for rank in 0 1 2 3; do
  host=${HOSTS[$rank]}
  ssh "$host" "test -r '$MODEL_DIR/config.json' && mkdir -p /var/tmp/engram-local/DeepSeek-V4.1-Flash"
  if (( rank == 0 )); then
    # Rank 0 may use the shared checkpoint directly when local NVMe is tight.
    continue
  fi
  scp "$ROOT/engram_local.py" "$host:/tmp/dsv41-engram-local.py"
  # shellcheck disable=SC2086
  ssh "$host" "python3 /tmp/dsv41-engram-local.py '$MODEL_DIR' /var/tmp/engram-local/DeepSeek-V4.1-Flash ${ranges[$rank]} --mbps=600"
done
