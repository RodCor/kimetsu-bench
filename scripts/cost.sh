#!/usr/bin/env bash
# Token usage -> Sonnet-4 cost, per trial and aggregated by family/arm.
#
# Reads the per-trial agent event streams that Harbor leaves under
# /tmp/kbench-<task>-<agent>/.../agent/sessions/*.jsonl and prices the token
# counts. The task -> family mapping is loaded from the families manifest
# (datasets/prog-families-v1.json) so there is a single source of truth — no
# hardcoded task lists to drift out of sync with `--family` runs.
#
# Usage: scripts/cost.sh        (or: make cost)
set -u

# Sonnet-4 per-token prices (USD): input, output, cache-write, cache-read.
IN=0.000003; OUT=0.000015; CW=0.00000375; CR=0.0000003

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MANIFEST="$SCRIPT_DIR/../datasets/prog-families-v1.json"

if [ ! -f "$MANIFEST" ]; then
  echo "cost.sh: families manifest not found at $MANIFEST" >&2
  exit 1
fi

# task -> family map, emitted as "task<TAB>family" lines from the manifest.
MAPFILE=$(mktemp)
python3 - "$MANIFEST" >"$MAPFILE" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    fams = json.load(f).get("families", {})
for fam, entry in fams.items():
    for task in entry.get("tasks", []):
        print(f"{task}\t{fam}")
PY

declare -A FAMOF
while IFS=$'\t' read -r task fam; do
  [ -n "$task" ] && FAMOF["$task"]="$fam"
done <"$MAPFILE"
rm -f "$MAPFILE"

shopt -s nullglob
DIRS=(/tmp/kbench-*/)
if [ ${#DIRS[@]} -eq 0 ]; then
  echo "cost.sh: no /tmp/kbench-* trial dirs found (run a benchmark first)."
  exit 0
fi

echo "model(s) seen:"
for d in "${DIRS[@]}"; do
  J=$(find "$d" -path "*/agent/sessions/*.jsonl" 2>/dev/null | head -1)
  [ -z "$J" ] && continue
  grep -aoE '"model":"[^"]+"' "$J" 2>/dev/null
done | sort | uniq -c
echo

TSV=$(mktemp)
printf "%-32s %-10s %11s %10s %12s %12s %8s\n" TASK AGENT IN OUT CACHE_W CACHE_R 'COST$'
for d in "${DIRS[@]}"; do
  base=$(basename "$d"); rest=${base#kbench-}
  if   [[ "$rest" == *-claude+km ]]; then agent="claude+km"; task=${rest%-claude+km}
  elif [[ "$rest" == *-claude   ]]; then agent="claude";    task=${rest%-claude}
  elif [[ "$rest" == *-codex+km ]]; then agent="codex+km";  task=${rest%-codex+km}
  elif [[ "$rest" == *-codex    ]]; then agent="codex";     task=${rest%-codex}
  else continue; fi
  fam=${FAMOF[$task]:-other}
  J=$(find "$d" -path "*/agent/sessions/*.jsonl" 2>/dev/null | head -1)
  [ -z "$J" ] && continue
  read i o cw cr <<<"$(grep -aoE '"(input_tokens|output_tokens|cache_creation_input_tokens|cache_read_input_tokens)":[0-9]+' "$J" \
    | awk -F: '/cache_creation/{cw+=$2;next} /cache_read/{cr+=$2;next} /output_tokens/{o+=$2;next} /input_tokens/{i+=$2} END{print i+0, o+0, cw+0, cr+0}')"
  cost=$(awk -v i=$i -v o=$o -v cw=$cw -v cr=$cr -v IN=$IN -v OUT=$OUT -v CW=$CW -v CR=$CR 'BEGIN{printf "%.2f", i*IN+o*OUT+cw*CW+cr*CR}')
  printf "%-32s %-10s %11d %10d %12d %12d %8s\n" "$task" "$agent" "$i" "$o" "$cw" "$cr" "$cost"
  printf "%s\t%s\t%s\n" "$fam" "$agent" "$cost" >> "$TSV"
done

echo
echo "=== aggregate by family / arm ==="
printf "%-12s %-10s %8s %7s\n" FAMILY AGENT 'COST$' trials
# Dynamic aggregation: sum by (family|agent) over whatever appeared, then an
# OVERALL row per agent. Sorted for stable output.
awk -F'\t' '{t[$1"|"$2]+=$3; n[$1"|"$2]++; all[$2]+=$3; aln[$2]++}
END{
  for(k in t) print "ROW\t"k"\t"t[k]"\t"n[k];
  for(a in all) print "ALL\t"a"\t"all[a]"\t"aln[a];
}' "$TSV" | sort | while IFS=$'\t' read -r kind key cost trials; do
  if [ "$kind" = "ROW" ]; then
    fam=${key%%|*}; agent=${key#*|}
    printf "%-12s %-10s %8.2f %7d\n" "$fam" "$agent" "$cost" "$trials"
  fi
done
echo
awk -F'\t' '{all[$2]+=$3; aln[$2]++} END{for(a in all) printf "%-12s %-10s %8.2f %7d\n","OVERALL",a,all[a],aln[a]}' "$TSV" | sort
rm -f "$TSV"
