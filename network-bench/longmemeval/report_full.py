#!/usr/bin/env python3
import pathlib,json,collections,statistics,re,string,datetime,hashlib
R=pathlib.Path(__file__).resolve().parent
CONDITIONS=['no-memory','single-mcp-lexical','directory-A-to-B','split-B-alone','split-B-merged-directory','supersede-knowledge-update','signed-pack-QUIC-loopback','truncated-history','hybrid-embeddings']
def normalize(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in str(s).lower() if c not in string.punctuation)).split())
def scores(x):
 p=normalize(x['prediction']);g=normalize(x['answer']);pt=p.split();gt=g.split();same=sum((collections.Counter(pt)&collections.Counter(gt)).values())
 return dict(f1=2*same/(len(pt)+len(gt)) if same else 0.,exact_match=float(p==g),contains=float(bool(g) and g in p),refusal=float('no information available' in p))
def load(p):
 out=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:out.append(json.loads(line))
   except json.JSONDecodeError:pass
 return out

def summarize(rs):
 out=dict(n=len(rs))
 for k in ['f1','exact_match','contains','refusal']:out[k+'_pct']=100*statistics.mean([x[k] for x in rs]) if rs else None
 out['retrieval_median_ms']=statistics.median([x['retrieval_ms'] for x in rs]) if rs else None
 out['llm_actual_median_ms']=statistics.median([x['llm_ms'] for x in rs if not x.get('inference_cached_identical_prompt')]) if any(not x.get('inference_cached_identical_prompt') for x in rs) else None
 out['inference_reused_count']=sum(x.get('inference_cached_identical_prompt',False) for x in rs)
 return out

def main():
 raw=load(R/'logs/predictions.jsonl')+load(R/'logs/predictions-quic.jsonl');rows=list({(x['dataset'],x['id'],x['condition']):dict(x,**scores(x)) for x in raw}.values());events=load(R/'logs/events.jsonl');errors=[e for e in events if 'error' in e['op']];summ=[];diag=[]
 blockers=json.loads((R/'blockers.json').read_text()) if (R/'blockers.json').exists() else {}
 typelist=['single-session-user','single-session-assistant','multi-session','temporal-reasoning','knowledge-update','single-session-preference','abstention']
 for dataset in ['oracle','s-cleaned']:
  for cond in CONDITIONS:
   rs=[x for x in rows if x['dataset']==dataset and x['condition']==cond];s=dict(dataset=dataset,condition=cond,expected=78 if cond=='supersede-knowledge-update' else 500,**summarize(rs));s['complete']=s['n']==s['expected'];s['status']='complete' if s['complete'] else blockers.get(cond,'in progress / queued');s['per_type']={t:summarize([x for x in rs if x['category']==t]) for t in typelist};s['per_original_type']={t:summarize([x for x in rs if x['question_type']==t]) for t in typelist if t!='abstention'};s['answerable']=summarize([x for x in rs if x['category']!='abstention']);summ.append(s)
  by={(x['id'],x['condition']):x for x in rows if x['dataset']==dataset}
  for a,b in [('no-memory','single-mcp-lexical'),('single-mcp-lexical','directory-A-to-B'),('split-B-alone','split-B-merged-directory')]:
   ids=sorted({i for i,c in by if c==a}&{i for i,c in by if c==b})
   if ids:diag.append(dict(dataset=dataset,a=a,b=b,n=len(ids),f1_delta_pp=100*statistics.mean(by[i,b]['f1']-by[i,a]['f1'] for i in ids),identical_context_count=sum(by[i,a]['context']==by[i,b]['context'] for i in ids),identical_answer_count=sum(by[i,a]['prediction']==by[i,b]['prediction'] for i in ids)))
 now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat();out=dict(updated_at=now,timezone='America/Buenos_Aires UTC-3',official=False,metric='normalized-token-overlap F1 proxy; exact-match and substring-contains proxies',rows=len(rows),raw_rows=len(raw),summary=summ,paired_diagnostics=diag,errors=errors,finish_reasons=dict(collections.Counter(x['finish_reason'] for x in rows)),noncached_prediction_rows=sum(not x.get('inference_cached_identical_prompt') for x in rows),prepared_questions={d:len({e['id'] for e in events if e['op']=='prepared' and e.get('dataset')==d}) for d in ['oracle','s-cleaned']},raw_prepared_events=dict(collections.Counter(e.get('dataset') for e in events if e['op']=='prepared')),transport_assertions_failed=sum(not e.get('full_replication_equal',True) or not e.get('split_union_equal',True) for e in events),blockers=blockers)
 server_log=(R/'logs/llama-server.log').read_text()
 out['server_slot_launch_count']=server_log.count('launch_slot_:')
 out['server_completed_decode_count']=server_log.count('total time =')
 (R/'summary.json').write_text(json.dumps(out,indent=2))
 def f(v):return '—' if v is None else f'{v:.2f}'
 text=f'# LongMemEval × Kimetsu — live full-run checkpoint\n\nUpdated {now} ART (America/Buenos_Aires, UTC−3). **Not complete unless n equals expected.** All numbers are proxy metrics, NOT official LongMemEval accuracy. Oracle has 500 questions (470 answerable, 30 abstention). The s-cleaned variant is the real long-haystack retrieval test; oracle is evidence-session-only.\n'
 for ds in ['oracle','s-cleaned']:
  text+=f'\n## {ds}\n\n| Condition | n / expected | Proxy F1 % | Exact % | Contains % | Retrieval median ms | Actual inference median ms | Status |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
  for s in summ:
   if s['dataset']!=ds:continue
   text+=f"| {s['condition']} | {s['n']}/{s['expected']} | {f(s['f1_pct'])} | {f(s['exact_match_pct'])} | {f(s['contains_pct'])} | {f(s['retrieval_median_ms'])} | {f(s['llm_actual_median_ms'])} | {s['status']} |\n"
  text+='\n### By type — proxy F1 % (n)\n\n| Condition | '+' | '.join(typelist)+' |\n|---|'+'---:|'*len(typelist)+'\n'
  for s in summ:
   if s['dataset']==ds and s['n']:text+='| '+s['condition']+' | '+' | '.join(f"{f(s['per_type'][t]['f1_pct'])} ({s['per_type'][t]['n']})" for t in typelist)+' |\n'
  text+='\nAbstention F1 compares the response with an explanatory reference and is not a semantic abstention score; fixed-phrase refusal rate is provided separately in summary.json. Original type stratification (including abstention IDs) is also retained.\n'
 text+='\n## Matched-question comparisons\n'
 for d in diag:text+=f"- {d['dataset']}: {d['b']} − {d['a']}: {d['f1_delta_pp']:+.2f} F1 points on {d['n']} matched questions; identical contexts {d['identical_context_count']}, identical answers {d['identical_answer_count']}.\n"
 text+='\n## Design, interpretation, and blockers\n\n'+(R/'PLAN.md').read_text()+f'\n\nObserved inference rows: {len(rows)}; non-cached scored prediction rows: {out["noncached_prediction_rows"]}; errors: {len(errors)}. Summary includes individual error tracebacks. Directory replay verifies active text equality, but context order can differ due to distinct event metadata/ranking; never assume equal QA. Matched comparisons, not unmatched partial prefixes, support diagnostic deltas. One greedy pass, no significance or latency speedup claims. Retrieval times include the first context lookup in each brain: A and isolated B follow writes; synced/QUIC B use fresh MCP processes. These are not uniform warm-cache latency measurements. Resumption duplicates are deduplicated by dataset/question/condition; duplicate answer/prompt identities were audited.\n\n'
 for k,v in blockers.items():text+=f'- {k}: {v}\n'
 text+='\nArtifacts: summary.json; scores.png; logs/predictions.jsonl (all contexts and raw answers); logs/tasks.jsonl; logs/operations.jsonl (MCP requests/responses and CLI calls); logs/events.jsonl (ingestion/sync assertions/errors); logs/llama-server.log; logs/run*.log; brains/; model/scorer checksums in ../manifest.json. Resume commands: `python3 prepare_only.py`, `python3 infer_only.py`, `python3 quic_extension.py`; it skips completed prediction keys and preserves abandoned incomplete brains under -attemptN.\n'
 (R/'RESULTS.md').write_text(text)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(15,6))
 for ax,ds in zip(axes,['oracle','s-cleaned']):
  ss=[s for s in summ if s['dataset']==ds and s['n']]
  if ss:
   bars=ax.barh([s['condition']+f" (n={s['n']})" for s in ss],[s['f1_pct'] for s in ss],color=['#0284c7' if s['complete'] else '#94a3b8' for s in ss]);ax.invert_yaxis()
   for bar,s in zip(bars,ss):ax.text(bar.get_width()+.3,bar.get_y()+bar.get_height()/2,f"{s['f1_pct']:.1f}%",va='center')
  ax.set_xlim(0,100);ax.set_title(ds+' — '+('complete' if all(s['complete'] for s in ss) and ss else 'IN PROGRESS'));ax.set_xlabel('Proxy token F1 %, NOT official accuracy')
 fig.suptitle('LongMemEval · Qwen2.5-0.5B Q4 · one physical host, directory replication');fig.tight_layout();fig.savefig(R/'scores.png',dpi=150);plt.close(fig)
 print(json.dumps(dict(rows=len(rows),errors=len(errors),updated_at=now)))
if __name__=='__main__':main()
