#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
GPU_ID=${GPU_ID:?Set GPU_ID}
read -r -a jobs <<< "${EXPERIMENTS:?Set EXPERIMENTS}"
STATE="$ROOT/work_dirs/runtime/mccl_gpu${GPU_ID}"
mkdir -p "$STATE"
exec 9>"$STATE/queue.lock"
flock -n 9 || { echo 'MCCL queue already active'; exit 1; }
if [[ -n "${AFTER_FILE:-}" ]]; then
 while [[ ! -f "$AFTER_FILE" ]]; do
  printf '%s\twaiting_predecessor\t%s\n' "$(date -u +%FT%TZ)" "${jobs[0]}" > "$STATE/status.tsv"
  sleep 60
 done
fi
for job in "${jobs[@]}"; do
 [[ ! -f "$STATE/${job}.completed" ]] || continue
 idle=0
 while [[ "$idle" -lt 2 ]]; do
  memory=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)
  processes=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ "$memory" -lt 500 && -z "$processes" ]]; then idle=$((idle+1)); else idle=0; fi
  printf '%s\twaiting_gpu\t%s\n' "$(date -u +%FT%TZ)" "$job" > "$STATE/status.tsv"
  [[ "$idle" -ge 2 ]] || sleep 60
 done
 printf '%s\trunning\t%s\n' "$(date -u +%FT%TZ)" "$job" > "$STATE/status.tsv"
 git rev-parse HEAD > "$STATE/${job}.code_commit"
 if TRANSFER="$job" CUDA_DEVICES="$GPU_ID" SPLIT_SOURCE=random bash scripts/train_semi_source_hrda.sh > "$STATE/${job}.log" 2>&1; then
  touch "$STATE/${job}.completed"
 else
  printf '%s\tfailed\t%s\n' "$(date -u +%FT%TZ)" "$job" > "$STATE/status.tsv"
  exit 1
 fi
done
printf '%s\tcompleted\tall\n' "$(date -u +%FT%TZ)" > "$STATE/status.tsv"
