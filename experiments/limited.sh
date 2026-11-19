#!/bin/bash
# Run a command inside a memory-capped cgroup so that an experiment that needs
# too much memory is killed by the kernel instead of taking the machine down.
#
#   bash experiments/limited.sh <command> [args...]
#
# MEM_MAX (default 5G) is the hard limit for everything the command starts.
# Memory use is written to experiments/results/memory.log once a minute.
set -euo pipefail
MEM_MAX="${MEM_MAX:-5G}"
MEM_HIGH="${MEM_HIGH:-4500M}"
SWAP_MAX="${SWAP_MAX:-2G}"
LOG="$(cd "$(dirname "$0")" && pwd)/results/memory.log"
mkdir -p "$(dirname "$LOG")"
exec systemd-run --user --scope -q -p MemoryMax="$MEM_MAX" -p MemoryHigh="$MEM_HIGH" -p MemorySwapMax="$SWAP_MAX" \
    bash -c '
        CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
        ( while sleep 60; do
              echo "$(date +%H:%M:%S) cgroup_MB=$(( $(cat $CG/memory.current) / 1048576 )) peak_MB=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 )) available_MB=$(awk "/MemAvailable/ {print int(\$2/1024)}" /proc/meminfo)" >> "$0"
          done ) &
        LOGGER=$!
        "$@"; STATUS=$?
        kill $LOGGER 2>/dev/null
        echo "$(date +%H:%M:%S) finished status=$STATUS peak_MB=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 )) oom_kills=$(awk "/oom_kill / {print \$2}" $CG/memory.events) : $*" >> "$0"
        exit $STATUS
    ' "$LOG" "$@"
