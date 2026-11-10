#!/bin/bash
# Progress, GPU, disk and ETA for the suite. Usage: bash /mnt/d/kimetsu-bench/status.sh [SUITE_DIR]
B=/mnt/d/kimetsu-bench; S=${1:-${SUITE_DIR:-$B/suite}}
/mnt/d/kimetsu-bench/venv/bin/python - "$S" <<'PY'
import json,sys,pathlib,time,collections,statistics,datetime
S=pathlib.Path(sys.argv[1]);SH=S/'shared'
def rows(p):
 out=[]
 if p.exists():
  for l in p.read_text().splitlines():
   try:out.append(json.loads(l))
   except Exception:pass
 return out
ts=lambda t:datetime.datetime.fromtimestamp(t).strftime('%a %d %b %H:%M ART')
now=time.time()
if (S/'STOP').exists():print('!!! STOP:',(S/'STOP').read_text().strip())
plan=json.loads((SH/'plan.json').read_text())['units'] if (SH/'plan.json').exists() else []
if not plan:print('no plan yet');sys.exit()
prep={(x['dataset'],x['unit']):x for x in rows(SH/'logs/prepared.jsonl') if (SH/'contexts'/x['dataset']/f"{x['unit']}.json").exists()}
print('PREPARATION (RAM scratch, saved to D:)')
tot_c=sum(u['chunks'] for u in plan);done_c=0
for ds in ['locomo','oracle','s-cleaned']:
 us=[u for u in plan if u['dataset']==ds];d=[prep[(ds,u['unit'])] for u in us if (ds,u['unit']) in prep];done_c+=sum(x['chunks'] for x in d)
 rc=sum(x['record_calls'] for x in d);rs=sum(x['record_seconds'] for x in d)
 print(f"  {ds:10s} units {len(d)}/{len(us)}  saves {rc}  s/save {rs/rc:.4f}" if rc else f"  {ds:10s} units 0/{len(us)}")
starts=rows(SH/'logs/prep-starts.jsonl');st=starts[-1]['at'] if starts else now
recent=[x for x in prep.values() if x['at']>=st]
if (SH/'prep-complete.json').exists():print('  prep complete',ts(json.loads((SH/'prep-complete.json').read_text())['at']))
elif recent and now-st>60:
 rate=sum(x['chunks'] for x in recent)/(now-st);eta=(tot_c-done_c)/rate if rate else 0
 print(f"  chunks {done_c}/{tot_c}; this launch {rate:.1f} chunks/s wall -> prep ETA {ts(now+eta)} ({eta/3600:.1f} h)")
errs=rows(SH/'logs/prep-errors.jsonl');print('  prep errors/exclusions:',dict(collections.Counter((e['dataset'],e['condition']) for e in errs)))
hyb=list((SH/'hybrid-contexts').rglob('*.json')) if (SH/'hybrid-contexts').exists() else []
print('HYBRID units',len(hyb),'/',len(plan),'| errors',len(rows(SH/'logs/hybrid-errors.jsonl')),'| complete' if (SH/'hybrid-complete.json').exists() else '')
nq={ds:sum(u['questions'] for u in plan if u['dataset']==ds) for ds in ['locomo','oracle','s-cleaned']}
for m in ['0.5b','7b']:
 rs=rows(S/'runs'/m/'predictions.jsonl')
 if not rs and not (S/'runs'/m).exists():continue
 keys=collections.Counter((x['dataset'],x['condition']) for x in rs);dup=len(rs)-len({(x['dataset'],x['id'],x['condition']) for x in rs})
 print(f'MODEL {m}: rows {len(rs)}/{8*sum(nq.values())}  duplicates {dup}  errors {len(rows(S/"runs"/m/"inference-errors.jsonl"))}' + ('  COMPLETE' if (S/'runs'/m/'infer-complete.json').exists() else ''))
 for (ds,c),n in sorted(keys.items()):
  sc=[x['score'] for x in rs if x['dataset']==ds and x['condition']==c]
  print(f'   {ds:10s} {c:28s} {n:5d}/{nq[ds]}  mean score {100*statistics.mean(sc):.2f}%  (partial)' if n<nq[ds] else f'   {ds:10s} {c:28s} {n:5d}/{nq[ds]}  mean score {100*statistics.mean(sc):.2f}%')
 if len(rs)>20:
  last=rs[-200:];print(f'   recent mean llm {statistics.mean(x["llm_ms"] for x in last)/1000:.3f} s/row; last row {ts(rs[-1]["at"])}')
PY
echo "GPU: $(/usr/lib/wsl/lib/nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv,noheader)"
tail -1 "$S/logs/gpu.csv" 2>/dev/null | sed 's/^/last watchdog row: /'
grep -c EVENTLOG_CHECK_FAILED "$S/logs/gpu.csv" 2>/dev/null | sed 's/^/event-log check failures: /'
df -h /mnt/c /mnt/d /dev/shm | sed 's/^/  /'
free -g | head -2
ps -eo pid,etime,cmd | grep -E 'suite.py|llama-server|launch_full|gpu_watchdog' | grep -v grep
