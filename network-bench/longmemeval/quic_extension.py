"""Uses sibling's actual kimetsu-sdk signed request_pack helper, not emulation."""
import run_full as c
import json,time,hashlib,pathlib,subprocess,traceback
R=c.R;helper=c.ROOT/'full-locomo/target/release/kimetsu-locomo-transport';out=open(R/'logs/predictions-quic.jsonl','a',buffering=1);audit=open(R/'logs/quic.jsonl','a',buffering=1)
def readrows():
 rows=[]
 for p in [R/'logs/predictions.jsonl',R/'logs/predictions-quic.jsonl']:
  if p.exists():
   for l in p.read_text().splitlines():
    try:rows.append(json.loads(l))
    except json.JSONDecodeError:pass
 return rows
seen={(r['dataset'],r['id']) for r in readrows() if r['condition']=='signed-pack-QUIC-loopback'}
for ds,path in [('oracle',c.ROOT/'data/longmemeval_oracle.json'),('s-cleaned',R/'longmemeval_s_cleaned.json')]:
 for q in json.loads(path.read_text()):
  if (ds,q['question_id']) in seen:continue
  # Main priority arms go first: wait for this question's real directory inference.
  while True:
   rs=readrows();matches=[x for x in rs if x['dataset']==ds and x['id']==q['question_id'] and x['condition']=='directory-A-to-B']
   if matches:break
   time.sleep(10)
  ref=matches[-1];base=R/'brains'/ds/q['question_id'];src=base/'full-a';dst=base/'quic-b'
  try:
   assert helper.exists(),'sibling helper binary missing'
   if dst.exists():
    i=1
    while dst.with_name(dst.name+f'-attempt{i}').exists():i+=1
    dst.rename(dst.with_name(dst.name+f'-attempt{i}'))
   c.pilot.setup(dst,'quic-b',base/'quic-unused-directory');assert c.pilot.rows(dst)==[]
   start=time.perf_counter();p=subprocess.run([str(helper),str(src),str(dst)],capture_output=True,text=True,timeout=360,env=c.pilot.ENV);ms=(time.perf_counter()-start)*1000
   audit.write(json.dumps(dict(dataset=ds,id=q['question_id'],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,ms=ms))+'\n');assert p.returncode==0,p.stderr
   assert c.pilot.rows(src)==c.pilot.rows(dst),'QUIC active text mismatch'
   m=c.pilot.MCP(dst);ctx,ret,n=c.context(m,q['question']);m.close()
   t=dict(ref,condition='signed-pack-QUIC-loopback',context=ctx,retrieval_ms=ret,capsules=n,quic_export_transfer_import_ms=ms,transport='iroh QUIC loopback; BusRelay::Disabled; one physical machine')
   for k in ['prediction','llm_ms','usage','finish_reason','prompt_key','inference_cached_identical_prompt']:t.pop(k,None)
   key=hashlib.sha256(json.dumps([t['question'],ctx,t['question_date']]).encode()).hexdigest();cached=[x for x in readrows() if x['prompt_key']==key]
   if cached:x={k:cached[-1][k] for k in ['prediction','llm_ms','usage','finish_reason']}
   else:x=c.llm(t)
   out.write(json.dumps(dict(t,**{k:v for k,v in x.items() if k!='response'},prompt_key=key,inference_cached_identical_prompt=bool(cached)))+'\n');print(ds,q['question_id'],'SIGNED_QUIC_OK',n,'CACHE' if cached else '',flush=True)
  except Exception:
   c.event(op='quic_error',dataset=ds,id=q['question_id'],traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
c.event(op='quic_extension_complete')
