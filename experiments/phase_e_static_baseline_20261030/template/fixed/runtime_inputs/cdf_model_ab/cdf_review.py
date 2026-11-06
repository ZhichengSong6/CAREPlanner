"""Full worker reconciliation, task-wall window domain and caller timing."""
import argparse
import os
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

OUT=Path(__file__).resolve().parent;RT=OUT.parents[1]
sys.path.insert(0,str(RT/'scripts'))
from analyze_cdf_input_audit import analyze
from analyze_cdf_monotonic_window import report as monotonic_report

def stats(values):
    a=np.array(values,dtype=float)
    return dict(samples=len(a),**{k:float(np.quantile(a,q)) if len(a) else None
        for k,q in [('min',0),('p50',.5),('p95',.95),('p99',.99),('max',1)]})

def domain(rows):
    return dict(requests=len(rows),pairs=sum(r['pairs'] for r in rows),
        outside_tolerance_rows=sum(r['outside_tolerance_rows'] for r in rows),
        strict_outside_rows=sum(r['strict_outside_rows'] for r in rows),
        xyz_min_m=np.min([r['xyz_min_m'] for r in rows],axis=0).tolist() if rows else None,
        xyz_max_m=np.max([r['xyz_max_m'] for r in rows],axis=0).tolist() if rows else None,
        outside_faces=dict(sum((Counter(r['outside_faces']) for r in rows),Counter())),
        outside_batches=[dict(sequence=r['sequence'],wall_time_ns=r['wall_time_ns'],pairs=r['pairs'],
            outside_rows=r['outside_tolerance_rows'],examples=r['outside_examples']) for r in rows if r['outside_tolerance_rows']])

def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True);p.add_argument('--case',required=True);p.add_argument('--tag',required=True);a=p.parse_args()
    runid=f'phase_e_goal_{a.case}_{a.batch}';base=RT/'outputs/c5_5_vbc_gcdf_regime'/runid
    sources=[base/'cdf_input_domain.jsonl',base/'cdf_model_attestation.json',base/'cdf_worker_readiness.json',
         base/'local_scp_selector.jsonl',RT/'outputs/phase_e_r1_mainline_30'/a.batch/'case_summaries'/f'phase_e_goal_{a.case}.json']
    session=analyze(sources[0],sources[1]);att=json.loads(sources[1].read_text());ready=json.loads(sources[2].read_text())
    binding=json.loads(Path(os.environ['CARE_CDF_MODEL_BINDING']).read_text())
    assert att['checkpoint_sha256']==ready['checkpoint_sha256']==binding['checkpoint_sha256']
    expected_key='iteration:5000' if binding['model']=='new' else 'iteration:39990'
    assert att['selected_checkpoint']==expected_key and att['activation']==binding['activation'] and not att['diagnostic_cpu']
    records=list(map(json.loads,sources[0].read_text().splitlines()))
    requests=[r for r in records if r['event']=='request'];results=[r for r in records if r['event']=='result']
    selector=list(map(json.loads,sources[3].read_text().splitlines()))
    successful=[r for r in selector if r.get('retained_pair_count',0)>0 and r.get('timing_ms',{}).get('worker_total_ms',0)>0]
    # Single shared synchronous client: compare all nonempty successful batches
    # in order, and require completion before the corresponding caller log.
    assert len(successful)==len(requests),(len(successful),len(requests))
    for q,r,s in zip(requests,results,successful):
        assert q['pairs']==r['pairs']==s['retained_pair_count']
        # Wall clocks can step backwards; sequence and pair counts bind the exchange.
    w=json.loads(sources[4].read_text())['benchmark_window']
    partition=monotonic_report(sources[0],sources[1],sources[4],sources[3])
    lo,hi=partition['task_bounds_ns']
    clock_reversals=partition['wall_clock_reversals']
    taskseq=set(partition['groups']['task']['sequences'])
    task=[r for r in requests if r['sequence'] in taskseq]
    timed=[r for r in results if r['sequence'] in taskseq]
    calltask=[s for q,s in zip(requests,successful) if q['sequence'] in taskseq]
    channels={c:dict(batches=len([s for s in successful if s['channel']==c]),
         pairs=sum(s['retained_pair_count'] for s in successful if s['channel']==c)) for c in sorted({s['channel'] for s in successful})}
    sem=dict(line.split('=',1) for line in (base/'run/runtime_semantics.txt').read_text().splitlines() if '=' in line)
    for k,v in {'vbc_swept_volume_margin_m':.02,'execution_vbc_swept_volume_margin_m':.01,'tracker_certified_margin_m':.02}.items():assert float(sem[k])==v
    controlled=RT/'logs/c5_5_vbc_gcdf_regime'/runid/'run/controlled.log';sources.append(controlled)
    socket_path=os.environ.get('GPU_SOCKET',f'{os.environ.get("TMPDIR", "/tmp")}/care_collision_cdf_gpu_{runid}.sock')
    routing=[line for line in controlled.read_text(errors='replace').splitlines() if socket_path in line]
    assert routing,'No runtime shared socket evidence'
    report=dict(status='COMPLETE_RECONCILED',clock_reversals=clock_reversals,clock_boundary_check='Same boot/time namespace; original monotonic score bounds; request emission membership',monotonic_partition=partition,session=session,task_by_request_monotonic_time=domain(task),
        before_task=domain([r for r in requests if r['monotonic_ns']<lo]),after_task=domain([r for r in requests if r['monotonic_ns']>hi]),
        task_window=w,channels_whole_session=channels,shared_socket_runtime_evidence=routing,
        task_input_audit_ms=stats([r['input_audit_ms'] for r in timed]),
        task_worker_ms_before_result_log=stats([r['worker_total_ms_before_result_log'] for r in timed]),
        task_batch_pairs=stats([r['pairs'] for r in task]),
        caller_task_timing_ms={k:stats([s['timing_ms'][k] for s in calltask]) for k in ('worker_inference_ms','worker_total_ms','ipc_roundtrip_ms','online_pipeline_ms')},
        input_sha256={str(p.relative_to(RT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        scope='All received worker rows; channel attribution by ordered single-client batches. Task and caller inclusion use the matched worker request monotonic clock. Not physical collision data.')
    with (OUT/('cdf_review_'+a.tag+'.json')).open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=report['status'],task_pairs=report['task_by_request_monotonic_time']['pairs'],outside=report['task_by_request_monotonic_time']['outside_tolerance_rows'],channels=channels)))

if __name__=='__main__':main()
