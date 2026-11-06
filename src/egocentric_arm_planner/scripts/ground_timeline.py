"""Bounded in-memory event timeline; disk writes occur only at shutdown."""
from collections import deque
from contextlib import contextmanager
import json,os,sys,threading,time
from pathlib import Path
import numpy as np

class GroundTimeline:
    def __init__(self,root):
        self.root=Path(root);self.events=deque();self.guard=threading.Lock()
        self.limit=100000;self.dropped=0;self.errors=0;self.closed=False
        self.environment=dict(executable=sys.executable,numpy_version=np.__version__,numpy_file=np.__file__,
                              affinity=sorted(os.sched_getaffinity(0)))

    def mark(self,kind,**fields):
        try:
            event=dict(kind=kind,wall=time.monotonic(),cpu=time.thread_time(),thread=threading.get_native_id(),**fields)
            with self.guard:
                if self.closed:return
                if len(self.events)>=self.limit:self.dropped+=1
                else:self.events.append(event)
        except Exception:self.errors+=1

    @contextmanager
    def hold(self,lock,scope,node):
        request=time.monotonic();request_cpu=time.thread_time()
        lock.acquire();acquired=time.monotonic();acquired_cpu=time.thread_time()
        try:yield
        finally:
            released=time.monotonic();released_cpu=time.thread_time()
            try:
                fields=dict(scope=scope,request_wall=request,acquired_wall=acquired,released_wall=released,
                    wait_s=acquired-request,wait_cpu_s=acquired_cpu-request_cpu,
                    held_s=released-acquired,held_cpu_s=released_cpu-acquired_cpu,
                    measured_stamp_ns=getattr(node,'_latest_measured_stamp_ns',0),
                    actuator_stamp_ns=getattr(node,'_latest_actuator_stamp_ns',0))
            finally:lock.release()
            self.mark('state_lock',**fields)

    def close(self):
        with self.guard:
            if self.closed:return
            self.closed=True;events=list(self.events)
        with (self.root/'ground_timeline.jsonl').open('x') as f:
            for event in events:f.write(json.dumps(event,separators=(',',':'),allow_nan=False)+'\n')
        (self.root/'ground_timeline_status.json').write_text(json.dumps(dict(events=len(events),
            dropped=self.dropped,errors=self.errors,environment=self.environment),indent=2)+'\n')
