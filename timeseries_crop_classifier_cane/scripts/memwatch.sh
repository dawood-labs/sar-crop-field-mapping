#!/bin/bash
# Log this pod's memory every 15 s to a file on the persistent disk, so after a crash/restart we can see what the memory
# looked like just before it. Kubernetes pod: MEM_GUARANTEE (request) is what we can safely use; above it the pod can be
# evicted when the node is under memory pressure, up to MEM_LIMIT where the kernel OOM-kills.
LOG=logs/memwatch.log
echo "$(date -u +%FT%TZ) START guarantee=$((MEM_GUARANTEE/1024/1024)) MB limit=$((MEM_LIMIT/1024/1024)) MB" >> "$LOG"
while true; do
  cur=$(( $(cat /sys/fs/cgroup/memory.current) / 1048576 ))
  anon=$(( $(awk '/^anon /{print $2}' /sys/fs/cgroup/memory.stat) / 1048576 ))
  act=$(( $(awk '/^active_file /{print $2}' /sys/fs/cgroup/memory.stat) / 1048576 ))
  top=$(ps -eo rss,comm --sort=-rss | awk 'NR>1 && NR<=4 {printf "%s:%dMB ", $2, $1/1024}')
  echo "$(date -u +%FT%TZ) cur=${cur}MB anon=${anon}MB active_file=${act}MB | $top" >> "$LOG"
  [ $(stat -c %s "$LOG") -gt 20000000 ] && mv "$LOG" "$LOG.1"
  sleep 15
done
