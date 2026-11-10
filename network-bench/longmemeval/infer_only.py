import run_full as core
import time,json,pathlib,hashlib,collections
R=core.R
# No-memory requires no ingestion; predeclare all 500 real questions in both datasets.
cache={};seen=set()
for path in [R/'logs/predictions.jsonl',R/'logs/predictions-quic.jsonl']:
 if path.exists():
  for line in path.read_text().splitlines():
   x=json.loads(line);seen.add((x['dataset'],x['id'],x['condition']));cache[x['prompt_key']]={k:x[k] for k in ('prediction','llm_ms','usage','finish_reason')}
tasks={};tail=open(R/'logs/tasks.jsonl');n=len(seen)
def drain():
 while True:
  pos=tail.tell();line=tail.readline()
  if not line:break
  try:t=json.loads(line)
  except json.JSONDecodeError:tail.seek(pos);break
  tasks[t['dataset'],t['id'],t['condition']]=t

def run(t):
 global n
 key=hashlib.sha256(json.dumps([t['question'],t['context'],t['question_date']]).encode()).hexdigest();cached=key in cache
 try:
  x=cache[key] if cached else core.llm(t)
  if not cached:cache[key]=x
  row=dict(t,**{k:v for k,v in x.items() if k!='response'},prompt_key=key,inference_cached_identical_prompt=cached)
  core.pred.write(json.dumps(row)+'\n');seen.add((t['dataset'],t['id'],t['condition']));n+=1
  print('ANSWER',n,t['dataset'],t['id'],t['condition'],x['prediction'][:70].replace('\n',' '),'CACHE' if cached else '',flush=True)
 except Exception:
  core.event(op='inference_error',dataset=t['dataset'],id=t['id'],condition=t['condition'],traceback=core.traceback.format_exc());print(core.traceback.format_exc(),flush=True);seen.add((t['dataset'],t['id'],t['condition']))
priority=['no-memory','single-mcp-lexical','directory-A-to-B','split-B-alone','split-B-merged-directory']
for ds,path,conditions in [('oracle',core.ROOT/'data/longmemeval_oracle.json',priority),('s-cleaned',R/'longmemeval_s_cleaned.json',priority),('oracle',core.ROOT/'data/longmemeval_oracle.json',['truncated-history']),('s-cleaned',R/'longmemeval_s_cleaned.json',['truncated-history'])]:
 data=json.loads(path.read_text())
 for cond in conditions:
  for i,q in enumerate(data):
   key=(ds,q['question_id'],cond)
   if key in seen:continue
   if cond=='no-memory':
    t=dict(dataset=ds,id=q['question_id'],question=q['question'],answer=q['answer'],question_type=q['question_type'],category='abstention' if q['question_id'].endswith('_abs') else q['question_type'],question_date=q['question_date'],condition=cond,context='',retrieval_ms=0,capsules=0)
   else:
    drain()
    while key not in tasks:
     ev=core.R/'logs/events.jsonl'
     if any(json.loads(l)['op']=='producer_error' for l in ev.read_text().splitlines()):raise RuntimeError('Producer failed; inspect events.jsonl')
     time.sleep(2);drain()
    t=tasks[key]
   run(t)
  core.event(op='condition_inference_complete',dataset=ds,condition=cond,n=500)
 core.event(op='dataset_inference_complete',dataset=ds,n=500)
core.event(op='run_complete',predictions=n)
