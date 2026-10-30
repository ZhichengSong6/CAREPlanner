#!/usr/bin/env python3
"""Real-record diagnostic overhead, not whole-planner performance acceptance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/egocentric_arm_planner/scripts'))
from rejection_snapshot import RejectionSnapshotSink
from run_phase_e_vbc_geometry_pair import distribution
from audit_phase_e_rejection_snapshots import validate_record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--repetitions',type=int,default=100)
    a=ap.parse_args();assert a.repetitions>=20
    sources=sorted((a.run/'rejection_snapshots').glob('*.json'));assert sources
    a.output_dir.mkdir(parents=True,exist_ok=False);results=[]
    for index,path in enumerate(sources):
        record=json.loads(path.read_text());validate_record(record)
        context=record.pop('local_context');samples=[]
        sink=RejectionSnapshotSink(a.output_dir/('sink_'+str(index)))
        try:
            for rep in range(a.repetitions+5):
                start=time.perf_counter_ns();encoded=json.dumps(context,allow_nan=False)
                t1=time.perf_counter_ns();sink.add_context(encoded);t2=time.perf_counter_ns()
                status=sink.submit(record);t3=time.perf_counter_ns()
                assert status=='queued';sink.jobs.join();t4=time.perf_counter_ns()
                if rep>=5:samples.append(dict(context_encode_ms=(t1-start)/1e6,
                    context_decode_ms=(t2-t1)/1e6,copy_enqueue_ms=(t3-t2)/1e6,
                    producer_total_ms=(t3-start)/1e6,writer_completion_ms=(t4-t3)/1e6))
            assert sink.written==1 and sink.errors==0
            # Non-ROS repeated JSON trajectory/GCDF copies are a separate lower
            # level measurement, not mislabeled as the actual admission callback.
            results.append(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                record_bytes=path.stat().st_size,context_bytes=len(encoded),
                statistics={k:distribution([s[k] for s in samples]) for k in samples[0]},
                coalesced=sink.coalesced,saved=sink.written))
        finally:sink.close()
    report=dict(status='PASS',meaning='Functional bounded sink benchmark completed',results=results,
        whole_pipeline_performance='NOT_RUN',
        exclusions=['C++ context serialization','ROS message conversion/transmission',
                    'admission callback trajectory conversion/copies','confidence map/visibility/clustering',
                    'GCDF model and tracker execution'],
        note='Absolute component costs only, not proof that full planner is no slower.')
    with (a.output_dir/'result.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print('PASS',[(r['record_bytes'],r['statistics']['producer_total_ms']) for r in results])


if __name__=='__main__':main()
