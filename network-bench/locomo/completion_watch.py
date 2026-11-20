from common import *
import subprocess
pid=41390
# A local pipeline supervisor, not a scheduler or an external action.
# Wait for the already-running worker, then verify final artifacts; no duplicate inference.
while True:
 try:os.kill(pid,0)
 except ProcessLookupError:break
 time.sleep(60)
p=subprocess.run([str(P/'.venv/bin/python'),str(R/'verify.py'),'--final'],text=True,capture_output=True)
write(R/'completion.json',dict(finished_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat(),verification_exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(p.stdout,p.stderr,flush=True)
raise SystemExit(p.returncode)
