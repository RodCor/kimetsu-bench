from common import *
import traceback

def run():
 done={(x['id'],x['condition']) for x in predictions()}
 for cond in CONDITIONS:
  for g in groups():
   path=R/'contexts'/f'{cond}-{g["id"]}.json';context_map={}
   if cond not in ['baseline','full_context']:
    while not path.exists():
     states=json.loads((R/'conditions.json').read_text())
     if states.get(cond,'').startswith(('unavailable','failed','blocked')):break
     time.sleep(10)
    if not path.exists():break
    context_map=json.loads(path.read_text())
   history=''
   if cond=='full_context':
    tokens=post('/tokenize',dict(content='\n'.join(g['history']),add_special=False))['tokens'];history=post('/detokenize',dict(tokens=tokens[-7000:]))['content']
   for q in g['questions']:
    if (q['id'],cond) in done:continue
    data=context_map.get(q['id'],{});context=history if cond=='full_context' else data.get('context','')
    try:
     ans,ms,x,request=generate(q,context)
     append(R/'logs/inference-operations.jsonl',dict(condition=cond,id=q['id'],request=request,response=x,ms=ms))
     row=dict(q,group=g['id'],condition=cond,prediction=ans,f1=score(q,ans),llm_ms=ms,usage=x.get('usage'),finish_reason=x['choices'][0].get('finish_reason'),context=context,retrieval_ms=data.get('ms'),payload=data.get('payload',{}))
     append(R/'logs/predictions.jsonl',row);done.add((q['id'],cond));print(cond,q['id'],'score',row['f1'],'answer',ans[:100].replace('\n',' '),flush=True)
     if len(done)%25==0:report()
    except Exception as e:
     append(R/'logs/errors.jsonl',dict(phase='inference',condition=cond,id=q['id'],error=repr(e),traceback=traceback.format_exc()));report();raise
   report()
  print('CONDITION FINISHED OR SKIPPED',cond,flush=True)
 report();print('ALL_APPLICABLE_CONDITIONS_FINISHED',flush=True)
if __name__=='__main__':run()
