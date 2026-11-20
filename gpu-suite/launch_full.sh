#!/bin/bash
# Full Kimetsu suite: RAM-scratch preparation (shared, model-independent) + 0.5B and 7B
# inference reading the same saved contexts + hybrid phase. Resumable: rerun the same command.
# Usage: bash /mnt/d/kimetsu-bench/launch_full.sh        (env: SUITE_DIR, SUITE_LIMIT, PREP_WORKERS, SKIP_HYBRID, MODELS)
set -uo pipefail
B=/mnt/d/kimetsu-bench; export SUITE_DIR=${SUITE_DIR:-$B/suite}; S=$SUITE_DIR; PY=$B/venv/bin/python
mkdir -p "$S/logs"
exec 9>"$S/run.lock"; flock -n 9 || { echo "Suite already running ($S/run.lock)"; exit 1; }
if [[ -e "$S/STOP" ]]; then echo "STOP present: $(cat "$S/STOP"). Inspect, delete $S/STOP, then relaunch."; exit 1; fi
"$PY" -c "import shutil; assert shutil.disk_usage('$B').free>20*1024**3, 'D: below 20 GiB safety floor'" || exit 1
export PATH="/usr/local/cuda-12.8/bin:$PATH" KIMETSU_USER_BRAIN=0 KIMETSU_BRAIN_AMBIENT=off TMPDIR="$B/cache/tmp"
exec >> "$S/logs/launch.log" 2>&1
echo "=== launch $(date '+%F %T %z') pid $$ suite=$S"
bash "$B/gpu_watchdog.sh" $$ "$S/logs/gpu.csv" "$S" & echo "watchdog $!"
pids=()
"$PY" -u "$B/suite.py" prep >> "$S/logs/prep.log" 2>&1 & pids+=($!); echo "prep $!"
for m in ${MODELS:-0.5b 7b}; do "$PY" -u "$B/suite.py" infer "$m" >> "$S/logs/infer-$m.log" 2>&1 & pids+=($!); echo "infer-$m $!"; done
if [[ -e "$S/SKIP_HYBRID" ]]; then export SKIP_HYBRID=1; echo "SKIP_HYBRID marker present; hybrid stage will no-op (pending #68)"; fi
"$PY" -u "$B/suite.py" hybrid >> "$S/logs/hybrid.log" 2>&1 & pids+=($!); echo "hybrid $!"
trap 'kill ${pids[@]} 2>/dev/null; pkill -P $$; exit 1' INT TERM
wait "${pids[@]}"
echo "=== all stages exited $(date '+%F %T %z')"
