"""Finite diagnostic capture, serialized outside the continuity callback lock."""
import hashlib
import json
from pathlib import Path
import queue
import threading
import time


class GroundCapture:
    def __init__(self, root, geometry, warn):
        self.root=Path(root);self.geometry=str(geometry);self.warn=warn
        self.geometry_sha256=hashlib.sha256(Path(geometry).read_bytes()).hexdigest()
        self.counts={};self.accepted=0;self.dropped=0;self.written=0;self.bytes=0;self.errors=[]
        self.queue=queue.Queue(maxsize=8);self.closed=False
        self.worker=threading.Thread(target=self._run,name='ground-capture',daemon=True)
        self.worker.start()

    def submit(self, record):
        # Caller freezes only bounded numerical payloads; JSON and disk I/O are on the worker.
        reason=record['result']['reason'];limit=3 if reason=='ground_wall_budget' else 1
        if self.closed or self.accepted>=8 or self.counts.get(reason,0)>=limit:return False
        record.update(schema=2,geometry_file=self.geometry,geometry_sha256=self.geometry_sha256)
        try:self.queue.put_nowait(record)
        except queue.Full:self.dropped+=1;return False
        self.counts[reason]=self.counts.get(reason,0)+1;self.accepted+=1
        return True

    def _run(self):
        while True:
            record=self.queue.get()
            try:
                if record is None:return
                started=time.monotonic();data=json.dumps(record,allow_nan=False,separators=(',',':')).encode()
                if len(data)>1024*1024 or self.bytes+len(data)>8*1024*1024:
                    raise ValueError('ground_capture_byte_limit')
                path=self.root/('known_ground_rejection_'+record['raw_candidate_stamp_ns']+'.json')
                with path.open('xb') as f:f.write(data)
                self.written+=1;self.bytes+=len(data)
                timing=dict(raw_candidate_stamp_ns=record['raw_candidate_stamp_ns'],bytes=len(data),
                            serialize_and_write_wall_s=time.monotonic()-started)
                with (self.root/'ground_capture_writer.jsonl').open('a') as f:f.write(json.dumps(timing)+'\n')
            except Exception as exc:
                self.errors.append(repr(exc));self.warn('[KNOWN_GROUND] snapshot failed: %s',exc)
            finally:self.queue.task_done()

    def close(self):
        if self.closed:return
        self.closed=True
        # At most eight records; shutdown only, never an admission or timer callback.
        self.queue.put(None,timeout=2.);self.worker.join(timeout=2.)
        report=dict(accepted=self.accepted,written=self.written,dropped=self.dropped,
                    bytes=self.bytes,errors=self.errors,worker_alive=self.worker.is_alive(),counts=self.counts)
        (self.root/'ground_capture_status.json').write_text(json.dumps(report,indent=2)+'\n')
