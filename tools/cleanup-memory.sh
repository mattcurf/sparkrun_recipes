#!/usr/bin/env bash
# SPDX-License-Identifier: Unlicense
set -euo pipefail

IMAGE="${MEMORY_CLEANUP_IMAGE:-alpine:3.22}"

usage() {
    cat <<'EOF'
Usage: cleanup-memory.sh local|HOST [HOST ...]

Reclaim Linux page cache, including memory retained after unified-memory model
workloads. Pass "local" for this machine and hostnames or IP addresses for
remote nodes reached through passwordless SSH.

Set MEMORY_CLEANUP_IMAGE to override the container image used to obtain
privileged access to /proc/sys/vm/drop_caches (default: alpine:3.22).

Stop inference workloads before running this command. It invokes a privileged
container and drops page cache on every selected host.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi

cleanup_command=$(cat <<EOF
before=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
docker run --rm --privileged --entrypoint sh '$IMAGE' -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
after=\$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
awk -v b="\$before" -v a="\$after" 'BEGIN {
    printf "MemAvailable: %.1f GiB -> %.1f GiB (reclaimed %.1f GiB)\\n",
        b / 1048576, a / 1048576, (a - b) / 1048576
}'
EOF
)

for target in "$@"; do
    echo "== $target =="
    if [[ "$target" == "local" || "$target" == "localhost" ]]; then
        bash -c "$cleanup_command"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" bash -c "$(printf '%q' "$cleanup_command")"
    fi
done
