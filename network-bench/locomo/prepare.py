from common import *
import pilot_harness as H,subprocess,traceback
H.LOG.close();H.LOG=open(R/'logs/operations.jsonl','a',buffering=1)
W=R/'workspaces';CTX=R/'contexts';W.mkdir(exist_ok=True);CTX.mkdir(exist_ok=True)

def state(c,s):
 x=json.loads((R/'conditions.json').read_text());x[c]=s;write(R/'conditions.json',x)
def memory_texts(w):return H.rows(w)
def new(w,label,shared):
 if not (w/'.kimetsu/brain.db').exists():H.setup(w,label,shared)
def ingest(w,turns):
 m=H.MCP(w)
 try:
  for text in turns:
   for offset in range(0,len(text),1200):
    out,_=m.tool('kimetsu_brain_record',dict(lesson=text[offset:offset+1200],tags=['benchmark','conversation'],confidence=1.0));assert out.get('ok'),out
 finally:m.close()
def contexts(g,cond,w):
 p=CTX/f'{cond}-{g["id"]}.json'
 if p.exists():return
 m=H.MCP(w);out={}
 try:
  for i,q in enumerate(g['questions']):
   payload,ms=H.retrieve(m,q['question']);caps=payload.get('capsules',[]);ctx='\n\n'.join(c.get('text',c.get('content',c.get('summary',''))) for c in caps)
   if caps:assert ctx,payload
   out[q['id']]=dict(context=ctx,payload=payload,ms=ms)
   append(R/'logs/retrieval-timings.jsonl',dict(condition=cond,group=g['id'],id=q['id'],ms=ms,capsules=len(caps)))
   if i%25==0:print('RETRIEVAL',cond,g['id'],i,'/',len(g['questions']),flush=True)
  write(p,out)
 finally:m.close()
def main():
 state('baseline','inference in progress');state('full_context','queued after priority conditions')
 for cond in ['single','directory','split_isolated','split_merged']:state(cond,'preparing; per-conversation checkpoints')
 for g in groups():
  base=W/g['id'];shared=base/'transport';a=base/'a';b=base/'directory-b';sa=base/'split-a';sb=base/'split-b';ss=base/'split-transport'
  for w,label,t in [(a,'a',shared),(b,'b',shared),(sa,'split-a',ss),(sb,'split-b',ss)]:new(w,label,t)
  if not (base/'ingested.json').exists():
   assert not memory_texts(b),'replica must start empty';start=time.perf_counter();ingest(a,g['history'])
   ah=[t for i,s in enumerate(g['sessions']) if i%2==0 for t in s['turns']];bh=[t for i,s in enumerate(g['sessions']) if i%2==1 for t in s['turns']];ingest(sa,ah);ingest(sb,bh)
   write(base/'ingested.json',dict(a_rows=len(memory_texts(a)),split_a_rows=len(memory_texts(sa)),split_b_rows=len(memory_texts(sb)),sessions_a=[s['id'] for i,s in enumerate(g['sessions']) if i%2==0],sessions_b=[s['id'] for i,s in enumerate(g['sessions']) if i%2==1],ms=(time.perf_counter()-start)*1000))
  if not (base/'directory-synced.json').exists():
   assert not memory_texts(b);t=time.perf_counter();H.cli(['brain','sync'],a);H.cli(['brain','sync'],b);ar=memory_texts(a);br=memory_texts(b);assert ar==br,(len(ar),len(br));write(base/'directory-synced.json',dict(empty_before=True,equal_after=True,rows=len(br),sync_ms=(time.perf_counter()-t)*1000));append(R/'logs/preparation.jsonl',dict(op='directory-sync',group=g['id'],rows=len(br),equality=True))
  contexts(g,'single',a);contexts(g,'directory',b)
  if not (CTX/f'split_isolated-{g["id"]}.json').exists():
   assert not (base/'split-synced.json').exists(),'cannot recompute isolated half after merged';contexts(g,'split_isolated',sb)
  if not (base/'split-synced.json').exists():
   before=memory_texts(sb);expected=sorted(set(memory_texts(sa)+before));t=time.perf_counter();H.cli(['brain','sync'],sa);H.cli(['brain','sync'],sb);after=memory_texts(sb);assert sorted(set(after))==expected,(len(after),len(expected));assert sorted(set(after))==sorted(set(memory_texts(a))),'merged halves must equal original full history';write(base/'split-synced.json',dict(b_rows_before=len(before),b_rows_after=len(after),union_verified=True,sync_ms=(time.perf_counter()-t)*1000))
  contexts(g,'split_merged',sb)
  print('PRIMARY GROUP PREPARED',g['id'],flush=True)
 for c in ['single','directory','split_isolated','split_merged']:state(c,'all contexts ready; inference queued/running')
 helper=R/'target/release/kimetsu-locomo-transport'
 if helper.exists():
  try:
   for g in groups():
    base=W/g['id'];dest=base/'quic-b';new(dest,'quic-b',base/'unused-quic-directory')
    if not (base/'quic-synced.json').exists():
     assert not memory_texts(dest);env=H.ENV.copy();p=subprocess.run([str(helper),str(base/'a'),str(dest)],text=True,capture_output=True,env=env,timeout=600)
     append(R/'logs/transport.jsonl',dict(group=g['id'],returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0,p.stderr
     assert memory_texts(base/'a')==memory_texts(dest),'signed-pack text mismatch';write(base/'quic-synced.json',dict(empty_before=True,equal_after=True,rows=len(memory_texts(dest)),output=p.stdout))
    contexts(g,'quic',dest)
   state('quic','all signed QUIC contexts ready')
  except Exception as e:state('quic','failed: '+repr(e));append(R/'logs/errors.jsonl',dict(phase='quic',error=repr(e),traceback=traceback.format_exc()))
 else:state('quic','unavailable: helper build did not produce binary; see logs/transport-build.log')
 # Build subprocess writes an exit code marker, so absence never silently passes as success.
 while not (R/'logs/embeddings-final.exit').exists():time.sleep(30)
 embed=R/'target/release/kimetsu'
 if (R/'logs/embeddings-final.exit').read_text().strip()=='0' and embed.exists():
  try:
   H.BIN=str(embed);H.ENV['KIMETSU_BRAIN_EMBEDDER']='bge-small-en-v1.5'
   for g in groups():
    w=W/g['id']/'hybrid';new(w,'hybrid',w.parent/'hybrid-unused-transport');ingest(w,g['history']);H.cli(['brain','reindex'],w)
    import sqlite3
    db=sqlite3.connect('file:'+str(w/'.kimetsu/brain.db')+'?mode=ro',uri=True);total,embedded=db.execute('select count(*),sum(embedding is not null and length(embedding)>0) from memories where invalidated_at is null and superseded_by is null').fetchone();models=db.execute('select distinct embedding_model from memories where embedding is not null').fetchall();db.close()
    assert total>0 and embedded==total,('no verified semantic coverage',total,embedded)
    write(w/'embedding-verification.json',dict(total=total,embedded=embedded,models=models));contexts(g,'hybrid',w)
   state('hybrid','all embedding contexts ready; verify embedder activation in stderr')
  except Exception as e:state('hybrid','failed: '+repr(e));append(R/'logs/errors.jsonl',dict(phase='hybrid',error=repr(e),traceback=traceback.format_exc()))
  finally:H.BIN='/home/box/.local/bin/kimetsu';H.ENV['KIMETSU_BRAIN_EMBEDDER']='off'
 else:state('hybrid','unavailable: embeddings feature build failed; exact compiler/dependency error in logs/embeddings-build-final.log')
 # Feedback must wait for complete first-pass single predictions. Ground-truth supervised; exploratory only.
 try:
  for g in groups():
   ids={q['id'] for q in g['questions']}
   while True:
    first=[x for x in predictions() if x['condition']=='single' and x['id'] in ids]
    if len(first)==len(ids):break
    time.sleep(30)
   m=H.MCP(W/g['id']/'a')
   try:
    tools,_=m.call('tools/list',{});schema=next(t for t in tools['tools'] if t['name']=='kimetsu_brain_cite');write(R/'cite-schema.json',schema)
    for row in first:
     if row['f1']!=1.0 or row['category']==5:continue
     payload=row.get('payload',{});exposure=payload.get('exposure_id')
     if not exposure:continue
     for cap in payload.get('capsules',[]):
      mid=cap.get('expansion_handle','').removeprefix('memory:')
      if mid:
       result,ms=m.tool('kimetsu_brain_cite',dict(exposure_id=exposure,memory_id=mid,outcome='helped',note='Exploratory benchmark supervised signal: first-pass official F1 equals 1; labels only used in this optional arm'))
       append(R/'logs/cite-feedback.jsonl',dict(id=row['id'],memory_id=mid,result=result,ms=ms))
   finally:m.close()
   contexts(g,'cite_exploratory',W/g['id']/'a')
  state('cite_exploratory','contexts ready; supervised test-label feedback, not generalization')
 except Exception as e:state('cite_exploratory','failed: '+repr(e));append(R/'logs/errors.jsonl',dict(phase='cite',error=repr(e),traceback=traceback.format_exc()))
 print('ALL_PREPARATION_FINISHED',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  append(R/'logs/errors.jsonl',dict(phase='preparation',error=repr(e),traceback=traceback.format_exc()))
  for c in CONDITIONS:
   if c not in ['baseline','full_context']:state(c,'failed: preparation stopped: '+repr(e))
  raise
