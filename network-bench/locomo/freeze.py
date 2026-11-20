from common import *
import re,hashlib
out=[]
for conv in json.loads((P/'data/locomo10.json').read_text()):
 history=[];sessions=[];c=conv['conversation']
 for k in sorted((k for k in c if re.fullmatch(r'session_\d+',k)),key=lambda x:int(x.split('_')[1])):
  turns=[]
  for turn in c[k]:
   text=f"[{turn['dia_id']}] {c[k+'_date_time']} | {turn['speaker']}: {turn['text']}"+(f" [image caption: {turn['blip_caption']}]" if turn.get('blip_caption') else '')
   turns.append(text);history.append(text)
  sessions.append(dict(id=k,turns=turns))
 qs=[dict(id=f"{conv['sample_id']}-q{i}",question=q['question'],answer=q.get('answer',''),category=q['category'],evidence=q.get('evidence',[])) for i,q in enumerate(conv['qa'])]
 out.append(dict(id=conv['sample_id'],history=history,sessions=sessions,questions=qs))
write(R/'full-dataset.json',out)
write(R/'conditions.json',dict.fromkeys(CONDITIONS,'queued'))
write(R/'manifest.json',dict(data_sha256=hashlib.sha256((P/'data/locomo10.json').read_bytes()).hexdigest(),scorer_sha256=hashlib.sha256((P/'data/locomo_evaluation.py').read_bytes()).hexdigest(),model_sha256='74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db',expected=sum(len(g['questions']) for g in out),category_counts=dict(collections.Counter(q['category'] for g in out for q in g['questions'])),claim='Measure full-dataset memory QA benefit and whether complementary networked memories beat one isolated half-brain; not transport speedup or LAN.' ))
print(report()['expected_per_condition'])
