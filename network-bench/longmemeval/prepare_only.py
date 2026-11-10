import run_full as core
class Sink:
 def put(self,item):pass
core.Q=Sink()
core.producer()
