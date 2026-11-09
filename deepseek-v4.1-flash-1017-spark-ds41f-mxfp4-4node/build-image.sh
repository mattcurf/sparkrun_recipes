#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
revision=cc20ed18be1f7cbf5c6c53cdee945f7cc77d5fc5
source_dir=${SPARK_DS41F_SOURCE:-$HOME/.cache/sparkrun-build/spark-ds41f-1107}
mode=${1:---prepare-only}
case "$mode" in
  --prepare-only|--build) ;;
  *) echo "Usage: $0 [--prepare-only|--build]" >&2; exit 2 ;;
esac
if [[ ! -d "$source_dir/.git" ]]; then
  git clone https://github.com/christopherowen/spark-ds41f.git "$source_dir"
  git -C "$source_dir" checkout --detach "$revision"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$revision" ]] || {
  echo "Source revision differs from $revision; refusing to overwrite it" >&2; exit 1;
}
[[ -z $(git -C "$source_dir" status --porcelain) ]] || {
  echo "Source checkout is dirty; refusing to build" >&2; exit 1;
}
# Build tooling requires a map even though prepare/build do not deploy nodes.
# This ignored example is ONLY for image building, never for cluster start/sync.
if [[ ! -e "$source_dir/config/nodes-switched.local.json" ]]; then
  cp "$source_dir/config/examples/nodes-switched.json" "$source_dir/config/nodes-switched.local.json"
fi
# Signed git-am commits would change the exact patch heads upstream verifies.
# Scope this to child Git processes; leave the user's signing settings intact.
index=${GIT_CONFIG_COUNT:-0}
export "GIT_CONFIG_KEY_$index=commit.gpgsign" "GIT_CONFIG_VALUE_$index=false"
export GIT_CONFIG_COUNT=$((index + 1))
cd "$source_dir"
# The frozen 1107 head's final patch used the author's newer committer email,
# unlike build.py's fixed older email. Reproduce that exact commit (320021e3),
# not a new source tree or a regenerated lock. All upstream checks still run.
python3 - <<'PY'
import sys
from pathlib import Path
import spark_ds41f as spark

checked = spark.checked

def apply_with_recorded_committer(command, cwd=None):
    if (command[:1] == ['git'] and 'am' in command and
            Path(command[-1]).name == '0761-perf-worker-flush-allocator-and-host-heap-across-sta.patch'):
        at = command.index('am')
        command = command[:at] + ['-c', 'user.email=christopherowen@users.noreply.github.com'] + command[at:]
    return checked(command, cwd=cwd)

spark.checked = apply_with_recorded_committer
sys.exit(spark.main(['--cluster-config', 'releases/spark-ds41f-1107/cluster.json',
                     '--shape', 'tp4-switched', 'build', 'prepare']))
PY
bin/spark --cluster-config releases/spark-ds41f-1107/cluster.json --shape tp4-switched build check
if [[ "$mode" == --build ]]; then
  # Upstream refuses live inference and enforces its memory guard; never bypass it.
  bin/spark --cluster-config releases/spark-ds41f-1107/cluster.json --shape tp4-switched \
    build image --tag spark-ds41f:1107 --max-compile-jobs 2 --apply
fi
