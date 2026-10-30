#!/usr/bin/env python3
"""Sensor-dependent scenario for the private, real-map serial-ledger harness."""
import copy
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import xmlrpc.client
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/care_visibility_cdf/scripts'))
from stopping_observation_demands import query_prefix_observation_demands,ObservationDemandQueue
from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
from phase_e_stopping_contract import bounded_update_travel
from phase_e_owned_processes import identity


def benchmark_transports(old_query,new_query,points,count=40):
    """Paired native service inputs while the controller is still WAITING."""
    import numpy as np
    rows=[]
    for pair in range(count):
        for name,query in ([('old',old_query),('persistent',new_query)] if pair%2==0 else
                           [('persistent',new_query),('old',old_query)]):
            start=time.monotonic();query([],0.,None);e=query(points,.3,None)
            version=tuple(e[k] for k in ('instance','epoch','revision'))
            checked=query([],0.,version);committed=query([],0.,version)
            elapsed=time.monotonic()-start
            assert e['valid'] and len(e['inside_map'])==len(points) and all(e['model_free_at_horizon'])
            rows.append(dict(pair=pair,channel=name,elapsed_ms=elapsed*1000,point_count=len(points),
                snapshot_ns=e['snapshot_ns'],revision=e['revision'],
                both_rechecks_stable=bool(checked['valid'] and committed['valid'])))
    return dict(scope='40 paired four-RPC native transactions at initial rest, alternating order; not complete stopping-check timings or bounds',
        rows=rows,summary={name:dict(zip(['p50_ms','p95_ms','p99_ms','max_ms'],map(float,np.percentile(
            [r['elapsed_ms'] for r in rows if r['channel']==name],[50,95,99,100])))) for name in ['old','persistent']})


def run_sensor_prefix_scenario(*,plan_path,ns,last,issue,current,ledger,result,check,event,
        set_pose,query_factory,planning_geometry,owned,remember,protocol_stats):
    plan=json.loads(Path(plan_path).read_text())
    task=plan['task_q'];view=plan['observation_q'];speed=plan['speed_limits']
    queue=ObservationDemandQueue()
    for pose in (task,view):
        if len(pose)!=7 or any(not math.isfinite(q) or not lo<=q<=hi for q,(lo,hi) in zip(pose,planning_geometry.configuration_limits)):
            raise ValueError('observation_pose_outside_joint_limits')
    planning_query=query_factory();paused=[]
    result['sensor_prefix']=dict(plan=plan,requests=[],moves=[],sensor_dependent_commits=0,
        source_model='actual ideal_fov updates from actual robot TF; no occlusion qualification',
        tf_pauses=[],ack_waits=[],runtime_admission=False)
    waiting_sequences=set()
    def maintain(cap):
        if last()['mode']!=1:
            raise AssertionError('controller stopped while completing sensor prefix')
        grant=current[0]
        # A normal renewal starts from the receipt for the last emission. This
        # avoids repeatedly spending a check against an older, shorter applied
        # deadline while its replacement is still travelling to the controller.
        # The ledger independently retains ALL unexpired rights.
        with ledger.lock:applied_sequence=ledger.state.get('lease_sequence')
        if applied_sequence!=grant['lease_sequence']:
            if grant['lease_sequence'] not in waiting_sequences:
                waiting_sequences.add(grant['lease_sequence'])
                result['sensor_prefix']['ack_waits'].append(dict(t_ns=ns(),waiting_sequence=grant['lease_sequence'],
                    observed_sequence=applied_sequence))
            time.sleep(.002)
            return
        if grant['valid_until_ns']-ns()<=95000000 or any(c>g+1e-10 for c,g in zip(cap,grant['target_abs_velocity'])):
            issue(cap)
        time.sleep(.002)
    def hold(seconds):
        end=ns()+int(seconds*1e9)
        set_pose(None)
        while ns()<end:maintain([0.]*7)
    def move(goal,label):
        start=ns();origin=last()['q'];sequence=ledger.sequence
        set_pose(goal)
        deadline=time.monotonic()+90.
        while time.monotonic()<deadline:
            state=last();error=[g-q for g,q in zip(goal,state['q'])]
            cap=[min(v,2.5*abs(e)+.001) if abs(e)>.0002 else 0. for v,e in zip(speed,error)]
            maintain(cap)
            if max(map(abs,error))<.0005 and max(map(abs,state['measured']))<.003:
                break
        else:raise AssertionError('pose acquisition timeout: '+label)
        hold(.15)
        row=dict(label=label,start_ns=start,end_ns=ns(),origin=origin,target=goal,actual=last()['q'],commits=ledger.sequence-sequence)
        result['sensor_prefix']['moves'].append(row);event('sensor_prefix_move',**row)
        print(json.dumps(dict(stage=label,t_s=ns()*1e-9,q=last()['q'])),flush=True)
    travel,stop=bounded_update_travel([0.,-.03,0.,0.,0.,0.,0.],[3.,3.,4.,4.,6.,6.,6.],[2.]*7,.12,.001,
        authorized_target_abs=[0.,.03,0.,0.,0.,0.,0.])
    def scan(label,first_use_delay,required_delay):
        started=ns();answers=[];failures=[]
        def work():
            try:
                segment=dict(q_low=task,q_high=task,stop_travel_rad=travel.tolist(),
                    first_use_s=started*1e-9+first_use_delay,required_until_s=started*1e-9+required_delay)
                answers.append(query_prefix_observation_demands(segments=[segment],geometry=planning_geometry,
                    query=planning_query,now_ns=ns,path_id='sensor_prefix_task',trajectory_stamp_ns=plan['trajectory_stamp_ns'],
                    observation_timing=plan['observation_timing'],wall_budget_s=.5))
            except Exception as exc:failures.append(repr(exc))
        thread=threading.Thread(target=work,daemon=True);thread.start()
        while thread.is_alive():maintain([0.]*7)
        if failures:raise AssertionError('prefix query failed: '+str(failures))
        report=answers[0];pending=queue.update(report,ns()*1e-9)
        row=dict(label=label,report=report,pending=pending)
        result['sensor_prefix']['requests'].append(row);event('stopping_prefix_report',**row)
        print(json.dumps(dict(stage=label,t_s=ns()*1e-9,demands=len(pending),actions=sorted(set(d['action'] for d in pending)))),flush=True)
        return report,pending
    try:
        report,pending=scan('lookahead_before_first_use',15.,15.+stop)
        check('future space outside initial prior requests observation before it is used',
              pending and any(d['action']=='first_observation' and d['first_use_s']>ns()*1e-9 for d in pending))
        move(view,'first_observation_pose')
        move(task,'first_sensor_dependent_task_pose')
        first_task=ns()
        report,pending=scan('actual_observation_resolves_first_use',0.,stop+.3)
        check('all task cells covered by actual map with sensor-only evidence',
              not pending and any(c['provenance']==1 and c['action']=='covered' for c in report['cells']))
        first_sensor_times={c['grid_index']:c['last_seen_s'] for c in report['cells'] if c['provenance']==1}
        report,pending=scan('proactive_refresh_before_expiry',0.,30.)
        refresh=[d for d in pending if d['action']=='refresh_observation']
        check('longer prefix requests refresh while current stopping evidence remains valid',
              refresh and all(d['confidence_now']>=.8 for d in refresh))
        latest=min(d['latest_start_s'] for d in refresh)
        while ns()*1e-9<latest:maintain([0.]*7)
        move(view,'proactive_reobservation_pose')
        move(task,'task_after_reobservation')
        report,pending=scan('new_source_observations_resolve_refresh',0.,stop+.3)
        newer=[c for c in report['cells'] if c['grid_index'] in first_sensor_times and c['provenance']==1 and
               c['last_seen_s']>first_sensor_times[c['grid_index']]]
        check('refresh resolves through genuinely newer source observations',not pending and newer)
        result['sensor_prefix']['sensor_dependent_commits']=protocol_stats['sensor_commits']
        result['sensor_prefix']['bounded_sensor_commit_samples']=copy.deepcopy(protocol_stats['sensor_samples'])
        check('real checked diagnostic commits depend on sensor-only cells',protocol_stats['sensor_commits']>0)
        hold(max(0.,(-100*math.log(.8)+1.)-(ns()-first_task)*1e-9))
        # Pause only this private master's owned robot_state_publisher. Actual
        # controller receipts, physics and fresh old commands continue.
        import rosgraph
        uri=rosgraph.Master('/sensor_prefix_tf_control').lookupNode('/robot_state_publisher')
        pid=int(xmlrpc.client.ServerProxy(uri).getPid('/sensor_prefix_tf_control')[2]);remember()
        assert pid in owned and pid not in (4177434,4177473,4177525)
        assert b'robot_state_publisher' in (Path('/proc')/str(pid)/'cmdline').read_bytes()
        assert identity(pid)==owned[pid]
        os.kill(pid,signal.SIGSTOP);paused.append(pid)
        pause=ns();result['sensor_prefix']['tf_pauses'].append(dict(pid=pid,pause_ns=pause))
        event('sensor_tf_source_paused',pid=pid,pause_ns=pause)
        print(json.dumps(dict(stage='actual_tf_source_paused',t_s=pause*1e-9,pid=pid)),flush=True)
        last_good=current[0]['valid_until_ns'];rejection=None;deadline=time.monotonic()+60.
        while time.monotonic()<deadline and last()['mode']==1:
            if current[0]['valid_until_ns']-ns()<=95000000:
                grant=issue([0.]*7)
                if grant:last_good=grant['valid_until_ns']
                elif result['rejections'][-1]['reason']=='uncovered_reference_pose_margin':
                    rejection=copy.deepcopy(result['rejections'][-1]);break
            time.sleep(.002)
        check('lost real TF eventually rejects renewal of sensor-dependent evidence',rejection is not None)
        while last()['mode']!=3 and time.monotonic()<deadline:time.sleep(.002)
        check('sensor evidence expiry reaches latch at unchanged last accepted deadline',
              last()['mode']==3 and current[0]['valid_until_ns']==last_good and last()['t_ns']>=last_good)
        quiet=ns()+200000000
        while ns()<quiet:
            check_mode=last()['mode'];assert check_mode==3,check_mode;time.sleep(.002)
        check('fresh old commands cannot revive the controller after observation expiry',last()['mode']==3)
        result['sensor_prefix']['failure_stop']=dict(tf_paused_ns=pause,last_accepted_until_ns=last_good,
            first_missing_evidence_rejection=rejection,latched_ns=last()['t_ns'],mode=last()['mode'])
        result['sensor_prefix']['sensor_dependent_commits_before_fault']=result['sensor_prefix']['sensor_dependent_commits']
        result['sensor_prefix']['sensor_dependent_commits']=protocol_stats['sensor_commits']
    finally:
        for pid in paused:
            if identity(pid)==owned.get(pid):os.kill(pid,signal.SIGCONT)
