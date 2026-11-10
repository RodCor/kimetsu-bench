#!/bin/bash
# Usage: gpu_watchdog.sh PARENT_PID CSV_LOG [SUITE_DIR]
# Every 10 s logs GPU temp/util/mem/power; every 60 s checks the Windows System log for
# new nvlddmkm events (via WSL interop, read-only). Stops everything (and writes STOP)
# if temp > 85 C, nvidia-smi fails, or a new nvlddmkm event appears.
B=/mnt/d/kimetsu-bench; parent=$1; L=$2; S=${3:-}
since=$(date '+%Y-%m-%dT%H:%M:%S')
echo "time,temp_c,util_pct,mem_mib,power_w" >> "$L"
stop_all() {
 echo "WATCHDOG_STOP $(date +%F_%T) $1" | tee -a "$L" >&2
 [[ -n "$S" ]] && echo "$(date +%F_%T) $1" > "$S/STOP"
 pkill -f "suite.py" ; pkill -f llama-server; pkill -f "$B/venv/bin/python"; kill "$parent"; exit 1
}
i=0
while kill -0 "$parent" 2>/dev/null; do
 line=$(/usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>&1)
 echo "$(date +%F_%T),$line" >> "$L"
 t=$(echo "$line" | cut -d, -f1 | cut -d. -f1 | tr -d ' ')
 if ! [[ "$t" =~ ^[0-9]+$ ]]; then stop_all "nvidia-smi failed: $line"; fi
 if (( t > 85 )); then stop_all "temp=$t > 85C"; fi
 if (( i % 6 == 0 )); then
  n=$(timeout 60 powershell.exe -NoProfile -Command "@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';StartTime=[datetime]'$since'} -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -dc '0-9')
  if [[ -z "$n" ]]; then echo "$(date +%F_%T),EVENTLOG_CHECK_FAILED" >> "$L"
  elif (( n > 0 )); then stop_all "nvlddmkm System event(s) since $since: $n"; fi
 fi
 i=$((i+1)); sleep 10
done
