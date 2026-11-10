import time,pathlib,json,traceback
import report_full,importlib
r=pathlib.Path(__file__).resolve().parent
while True:
 try:importlib.reload(report_full);report_full.main()
 except Exception:traceback.print_exc()
 events=r/'logs/events.jsonl'
 if events.exists() and any(json.loads(l).get('op')=='run_complete' for l in events.read_text().splitlines()):
  # Keep enough time for the signed-QUIC tail to finish independently.
  if any(json.loads(l).get('op')=='quic_extension_complete' for l in events.read_text().splitlines()):break
 time.sleep(60)
