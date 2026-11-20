from common import *
import sys
rows=predictions();gs=groups();truth={q['id']:q for g in gs for q in g['questions']};expected=len(truth)
assert expected==1986 and len(gs)==10
assert len(rows)==len({(x['id'],x['condition']) for x in rows}),'duplicate predictions'
for row in rows:
 q=truth[row['id']];assert row['question']==q['question'] and row['category']==q['category'] and row['answer']==q['answer']
 assert abs(score(q,row['prediction'])-row['f1'])<1e-12
 assert isinstance(row['prediction'],str) and row['prediction'].strip()
 if row['condition']=='baseline':assert row['context']==''
 elif row['condition']!='full_context':
  caps=row['payload'].get('capsules',[]);ctx='\n\n'.join(c.get('text',c.get('content',c.get('summary',''))) for c in caps);assert ctx==row['context']
 assert row['usage'].get('prompt_tokens',0)+row['usage'].get('completion_tokens',0)<=8192
print('Checked',len(rows),'real predictions, official scores, question identity, contexts, token limits; no duplicates')
summary=report();mandatory=['baseline','single','directory','split_isolated','split_merged','quic'];incomplete=[s['condition'] for s in summary['summary'] if s['condition'] in mandatory and not s['complete']]
if '--final' in sys.argv and incomplete:raise SystemExit('INCOMPLETE mandatory arms: '+', '.join(incomplete))
if '--final' in sys.argv:
 for g in gs:
  for name in ['directory-synced.json','split-synced.json','quic-synced.json']:
   assert (R/'workspaces'/g['id']/name).exists(),(g['id'],name)
 print('FULL_MANDATORY_ARMS_VERIFIED')
