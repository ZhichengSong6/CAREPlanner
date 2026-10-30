#!/usr/bin/env python3
"""Read-only map feasibility for receipt-anchored command enclosures.

No qualified physical response is supplied. Full point queries here are
command-domain diagnostics, explicitly rejected by the physical qualifier.
"""
import argparse,gzip,json,threading,time
from pathlib import Path
import sys
sys.path.append('/usr/lib/python3/dist-packages')
import rospy
from care_velocity_controller.msg import CommandStatus,LeasedVelocityCommand,ExecutionLease
from phase_e_leased_stop_evidence import (status_record,command_record,IDENTITY,
    query_complete_stopping_snapshot,query_adaptive_stopping_snapshot)
from phase_e_stopping_contract import qualify
from phase_e_stop_shadow import PrimitiveEnclosure
from phase_e_stop_shadow_node import snapshot_dict,BoundedQuery,PersistentServiceTransport
from care_confidence_map.srv import QueryConfidenceSnapshot,QueryConfidenceSnapshotRequest
from geometry_msgs.msg import Point

ROOT=Path(__file__).resolve().parents[1]
BASE='/care_arm/arm_group_velocity_controller'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--local-geometry',action='store_true')
    parser.add_argument('--subdivisions',type=int,default=8)
    parser.add_argument('--adaptive-geometry',action='store_true')
    parser.add_argument('--wall-budget-s',type=float,default=.05)
    args=parser.parse_args(rospy.myargv()[1:])
    if args.adaptive_geometry and not args.local_geometry:parser.error('adaptive geometry requires local geometry')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    rospy.init_node('leased_command_shadow',disable_signals=True)
    state=[None];commands={};grants=[];lock=threading.Lock()
    def on_state(msg):
        with lock:state[0]=status_record(msg)
    def on_command(msg):
        value=command_record(msg)
        with lock:
            commands[tuple(value[k] for k in IDENTITY)]=value
            while len(commands)>4096:commands.pop(next(iter(commands)))
    def on_grant(msg):
        if not msg.revoke:
            with lock:
                grants.append(dict(session=msg.controller_session,sequence=msg.lease_sequence,
                                   token=msg.execution_stamp_ns,until_ns=msg.valid_until.to_nsec(),
                                   target_range_enforced=msg.target_range_enforced,
                                   target_abs_velocity=list(msg.target_abs_velocity)))
                del grants[:-512]
    rospy.Subscriber(BASE+'/state',CommandStatus,on_state,queue_size=100,tcp_nodelay=True)
    rospy.Subscriber(BASE+'/command_leased',LeasedVelocityCommand,on_command,queue_size=100,tcp_nodelay=True)
    rospy.Subscriber(BASE+'/execution_lease',ExecutionLease,on_grant,queue_size=100,tcp_nodelay=True)
    if args.local_geometry:
        from phase_e_local_stop_geometry import LocalPrimitiveEnclosure
        geometry=LocalPrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf',subdivisions=args.subdivisions)
    else:geometry=PrimitiveEnclosure(ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf')
    def encode(points,horizon,expected):
        request=QueryConfidenceSnapshotRequest(points=[Point(*p) for p in points],horizon_s=horizon,minimum_confidence=.8)
        if expected:request.expected_instance,request.expected_epoch,request.expected_revision=expected
        return request
    transport=PersistentServiceTransport(lambda:rospy.ServiceProxy('/shadow_map/query_snapshot',
        QueryConfidenceSnapshot,persistent=True),encode,snapshot_dict)
    query=BoundedQuery(transport,.3)
    with (out/'rows.jsonl').open('x') as stream,gzip.open(out/'point_evidence.jsonl.gz','xt',compresslevel=1) as point_stream:
        while not rospy.is_shutdown():
            start=time.monotonic();now=rospy.Time.now().to_nsec()
            row=dict(runtime_admission=False,scope='conditional_command_only_map_feasibility',
                     physical_response_qualified=False,assumed_update_gap_s=.001,reasons=[],attempt_ns=now)
            with lock:
                s=state[0];command=commands.get(tuple(s[k] for k in IDENTITY)) if s else None
                outstanding=[g for g in grants if s and g['session']==s['controller_session'] and g['until_ns']>now]
            try:
                if not s or not command:raise ValueError('missing_exact_applied_command')
                row['state']=s
                row['outstanding_authorizations']=outstanding
                if s['mode']!=1:raise ValueError('controller_not_tracking')
                if s['clock_fault'] or not s['integral_valid']:raise ValueError('invalid_controller_time')
                if s['max_update_gap']>.001000001:raise ValueError('observed_gap_exceeds_assumption')
                end_ns=max([s['valid_until_ns']]+[g['until_ns'] for g in outstanding])
                row['enclosed_authorization_until_ns']=end_ns
                ranges=[s]+outstanding
                cap=[max(r['target_abs_velocity'][j] if r.get('target_range_enforced') else s['velocity_limits'][j]
                         for r in ranges) for j in range(7)]
                row['enclosed_target_abs_velocity']=cap
                row['range_scope']='union_of_active_and_observed_unexpired_grants'
                result=qualify(now_s=now*1e-9,measured_s=s['t_ns']*1e-9,applied=s['applied'],
                    acceleration=s['acceleration_limits'],velocity_limits=s['velocity_limits'],
                    next_decision_s=.01,transport_bound_s=0.,update_gap_s=.001,
                    lease_end_s=end_ns*1e-9,lease_enforced=True,
                    physical_error_rad=None,physical_settle_s=None,snapshot=None,
                    max_braking_step_s=s['max_braking_step_s'],authorized_target_abs=cap)
                row['budgets']=result['budgets']
                if not result['budgets']:raise ValueError(','.join(result['reasons']))
                operation=query_adaptive_stopping_snapshot if args.adaptive_geometry else query_complete_stopping_snapshot
                options=(dict(measured_ns=s['t_ns'],authorization_until_ns=end_ns,
                              wall_budget_s=args.wall_budget_s) if args.adaptive_geometry else {})
                row['map']=operation(budgets=result['budgets'],q=s['q'],
                    geometry=geometry,query=query,now_ns=lambda:rospy.Time.now().to_nsec(),
                    enclosure_field='command_travel_rad',
                    evidence_sink=lambda value:point_stream.write(json.dumps(value)+'\n'),**options)
                if args.adaptive_geometry and not row['map']['diagnostic_accepted']:
                    row['reasons'].append(row['map']['adaptive']['reason'])
                end=rospy.Time.now().to_nsec();row['end_state_age_s']=(end-s['t_ns'])*1e-9
                if row['end_state_age_s']>.05:row['reasons'].append('state_aged_out_during_check')
                with lock:
                    if any(g['session']==s['controller_session'] and g['until_ns']>end_ns for g in grants):
                        row['reasons'].append('authorization_advanced_during_check')
                    if any(g['session']==s['controller_session'] and g['until_ns']>end and
                           any((g['target_abs_velocity'][j] if g['target_range_enforced'] else s['velocity_limits'][j])>cap[j]
                               for j in range(7)) for g in grants):
                        row['reasons'].append('authorized_target_range_expanded_during_check')
                    latest=state[0]
                if latest and latest['controller_session']!=s['controller_session']:
                    row['reasons'].append('controller_session_changed_during_check')
            except (ValueError,RuntimeError,rospy.ROSException) as exc:row['reasons'].append(str(exc))
            row['wall_elapsed_s']=time.monotonic()-start
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            time.sleep(max(0.,.1-(time.monotonic()-start)))

if __name__=='__main__':
    try:main()
    except (KeyboardInterrupt,rospy.ROSInterruptException):pass
