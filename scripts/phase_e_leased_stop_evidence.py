"""Bind fixed-lease controller receipts to conditional stopping evidence.

Read-only qualification/diagnostics. This module has no ROS publisher and never
issues an ExecutionLease. Model bounds supplied by a caller remain externally
qualified assumptions; replay maxima do not establish those bounds.
"""
import hashlib
import json
import math
import time
import numpy as np
from phase_e_stopping_contract import qualify

JOINTS=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
IDENTITY=('controller_session','execution_stamp_ns','lease_sequence','source_sequence',
          'valid_until_ns','certificate_sha256','generated_ns')

def command_record(msg):
    row={k:getattr(msg,k) for k in IDENTITY if k not in ('valid_until_ns','generated_ns')}
    row.update(valid_until_ns=msg.valid_until.to_nsec(),generated_ns=msg.header.stamp.to_nsec(),
               joint_names=list(msg.joint_names),velocity=list(msg.velocity),stop=msg.stop)
    row.update(target_range_enforced=msg.target_range_enforced,target_abs_velocity=list(msg.target_abs_velocity))
    return row

def status_record(msg):
    row={k:getattr(msg,k) for k in IDENTITY if k not in ('valid_until_ns','generated_ns')}
    row.update(valid_until_ns=msg.lease_valid_until.to_nsec(),generated_ns=msg.command_stamp.to_nsec(),
               t_ns=msg.header.stamp.to_nsec(),mode=msg.mode,clock_fault=msg.clock_fault,
               integral_valid=msg.integral_valid,lease_required=msg.execution_lease_required,
               q=list(msg.measured_position),measured=list(msg.measured_velocity),
               applied=list(msg.applied_velocity),max_update_gap=msg.max_update_gap,
               rejected_commands=msg.rejected_commands,rejected_leases=msg.rejected_leases,
               rearm_successes=msg.rearm_successes,rearm_rejections=msg.rearm_rejections)
    row.update(velocity_limits=list(msg.velocity_limits),acceleration_limits=list(msg.acceleration_limits),
               max_braking_step_s=msg.max_braking_step_s,command_timeout_s=msg.command_timeout_s,
               max_execution_lease_s=msg.max_execution_lease_s)
    row.update(target_range_enforced=msg.target_range_enforced,target_abs_velocity=list(msg.target_abs_velocity))
    return row

def finite7(values):
    return isinstance(values,(list,tuple)) and len(values)==7 and all(math.isfinite(v) for v in values)

def evaluate(*,now_ns,state,command,velocity_limits,acceleration_limits,
             timing=None,response=None,outstanding_leases=(),snapshot=None,runtime_sha256=None):
    row=dict(runtime_admission=False,qualified=False,binding_consistent=False,reasons=[],budgets={})
    def reject(reason):
        if reason not in row['reasons']:row['reasons'].append(reason)
    if not state or not command:
        reject('missing_exact_controller_command');return row
    row['identity']={k:state.get(k) for k in IDENTITY}
    if any(state.get(k)!=command.get(k) for k in IDENTITY):reject('controller_command_identity_mismatch')
    bounded=state.get('target_range_enforced',False)
    cap=state.get('target_abs_velocity',[0.]*7)
    if (bounded!=command.get('target_range_enforced',False) or
            cap!=command.get('target_abs_velocity',[0.]*7)):
        reject('controller_command_target_range_mismatch')
    if (not finite7(cap) or any(v<0 for v in cap) or
            (finite7(list(velocity_limits)) and any(v>limit for v,limit in zip(cap,velocity_limits)))):
        reject('invalid_authorized_target_range')
    if not state.get('lease_required') or not state.get('controller_session'):reject('controller_lease_not_enforced')
    if state.get('clock_fault') or not state.get('integral_valid'):reject('invalid_controller_time')
    if state.get('mode')!=1 or command.get('stop'):reject('controller_not_authorized_to_continue')
    if command.get('joint_names')!=JOINTS or not finite7(command.get('velocity')):reject('invalid_source_command')
    elif bounded and finite7(cap) and any(abs(v)>limit for v,limit in zip(command['velocity'],cap)):
        reject('source_command_exceeds_authorized_target_range')
    if not all(finite7(state.get(k)) for k in ('q','measured','applied')):reject('invalid_controller_state')
    for field,expected in [('velocity_limits',velocity_limits),('acceleration_limits',acceleration_limits)]:
        if (not finite7(state.get(field)) or not finite7(list(expected)) or
                not np.allclose(state[field],expected,rtol=0.,atol=1e-12)):
            reject('controller_limits_mismatch')
    step=state.get('max_braking_step_s')
    if step is None or not math.isfinite(step) or step<=0:reject('invalid_controller_braking_step')
    if any(not isinstance(state.get(k),int) or state[k]<=0 for k in ('t_ns','generated_ns','valid_until_ns')):
        reject('invalid_controller_timestamps')
    if row['reasons']:return row
    if not state['generated_ns']<=state['t_ns']<=now_ns:reject('future_or_reversed_controller_state')
    if now_ns-state['t_ns']>50000000:reject('stale_controller_state')
    if state['valid_until_ns']<=now_ns:reject('expired_execution_lease')
    for lease in outstanding_leases:
        if lease['controller_session']==state['controller_session'] and lease['valid_until_ns']>state['valid_until_ns']:
            reject('outstanding_authorization_not_enclosed')
        if (bounded and lease['controller_session']==state['controller_session'] and lease['valid_until_ns']>now_ns):
            other=lease.get('target_abs_velocity') if lease.get('target_range_enforced') else list(velocity_limits)
            if not finite7(other) or any(v>limit or v<0 for v,limit in zip(other,cap)):
                reject('outstanding_target_range_not_enclosed')
    if row['reasons']:return row
    row['binding_consistent']=True
    limits=dict(velocity=list(velocity_limits),acceleration=list(acceleration_limits))
    limits_sha=hashlib.sha256(json.dumps(limits,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    row['limits_sha256']=limits_sha
    def qualified_contract(contract,kind):
        return (isinstance(contract,dict) and contract.get('kind')==kind and
                contract.get('limits_sha256')==limits_sha and runtime_sha256 is not None and
                contract.get('runtime_sha256')==runtime_sha256 and
                isinstance(contract.get('evidence_sha256'),str) and len(contract['evidence_sha256'])==64)
    timing_ok=qualified_contract(timing,'enforced_ros_sim_time_bounds')
    response_ok=qualified_contract(response,'qualified_full_horizon_response_bound')
    if not timing_ok:reject('missing_qualified_time_contract')
    if not response_ok:reject('missing_physical_response_bound')
    t=timing if timing_ok else {}
    r=response if response_ok else {}
    if timing_ok and state['max_update_gap']>t['update_gap_s']+1e-12:
        reject('observed_update_gap_exceeds_contract')
    result=qualify(now_s=now_ns*1e-9,measured_s=state['t_ns']*1e-9,applied=state['applied'],
        acceleration=acceleration_limits,velocity_limits=velocity_limits,
        next_decision_s=t.get('next_decision_s'),transport_bound_s=t.get('transport_bound_s'),
        update_gap_s=t.get('update_gap_s'),lease_end_s=state['valid_until_ns']*1e-9,
        lease_enforced=True,physical_error_rad=r.get('error_rad'),physical_settle_s=r.get('settle_s'),snapshot=snapshot,
        max_braking_step_s=state['max_braking_step_s'],authorized_target_abs=cap if bounded else None)
    row['budgets']=result['budgets']
    for reason in result['reasons']:reject(reason)
    if snapshot and snapshot.get('future_environment_qualified') is not True:
        reject('missing_future_environment_contract')
    row['qualified']=not row['reasons']
    return row

def query_complete_stopping_snapshot(*,budgets,q,geometry,query,now_ns,margin=.020,
                                     enclosure_field='physical_travel_rad',evidence_sink=None,
                                     geometry_subdivisions=None,check_budget=None,
                                     skip_outside_map=False,geometry_workspace=None,metadata_cache=None):
    """Query the complete union of BOTH enclosures, with one content recheck.

    geometry is the production PrimitiveEnclosure; query is the existing coherent
    QueryConfidenceSnapshot adapter. Grid-center clearance semantics are retained.
    Map updates after this check still require a separate environment contract.
    """
    required=['withdraw_now','continue_then_stop']
    wall_start=time.monotonic()
    check=check_budget or (lambda:None)
    check()
    if sorted(budgets)!=sorted(required):raise ValueError('missing_stopping_mode')
    if enclosure_field not in ('physical_travel_rad','command_travel_rad'):
        raise ValueError('invalid_enclosure_kind')
    if any(enclosure_field not in budgets[k] for k in required):
        raise ValueError('missing_physical_enclosure')
    # Geometry metadata may be reused only within this one adaptive check.
    # Every level still queries its entire point set with fresh content and
    # checks instance/epoch/geometry against this metadata before its recheck.
    meta=(metadata_cache.get('metadata') if metadata_cache is not None else None)
    if meta is None:
        meta=query([],0.,None)
        if metadata_cache is not None:metadata_cache['metadata']=meta
    check()
    wall_meta=time.monotonic()
    if not meta['valid']:raise ValueError('invalid_map_metadata')
    points={};inside=True;reference_ids=set();reference_available=True
    until=max(budgets[k]['required_until_s'] for k in required)
    geometry_modes=required
    # A proven enclosure of the larger joint box covers BOTH motion sets. This
    # need not include every excess conservative point of a separately built
    # smaller-box approximation. Arbitrary geometry adapters keep full union.
    if getattr(geometry,'nested_joint_box_coverage',False):
        a,b=[np.asarray(budgets[k][enclosure_field]) for k in required]
        if np.all(b>=a):geometry_modes=[required[1]]
        elif np.all(a>=b):geometry_modes=[required[0]]
    for mode in geometry_modes:
        check()
        options={} if geometry_subdivisions is None else dict(subdivisions=geometry_subdivisions,check_budget=check)
        if geometry_workspace is not None and getattr(geometry,'supports_refinement_workspace',False):
            options['workspace']=geometry_workspace
        box=geometry.raster(q,np.asarray(budgets[mode][enclosure_field]),meta,margin,**options)
        check()
        inside=inside and box['inside']
        if skip_outside_map and not inside:raise ValueError('enclosure_may_leave_map')
        if 'reference_pose_grid_indices' in box:reference_ids.update(box['reference_pose_grid_indices'])
        else:reference_available=False
        if len(box['grid_indices'])!=len(box['points']):raise ValueError('incomplete_raster')
        for index,point in zip(box['grid_indices'],box['points']):
            value=np.asarray(point,dtype=float).tolist()
            if index in points and points[index]!=value:raise ValueError('grid_identity_conflict')
            points[index]=value
    ids=sorted(points);values=[points[k] for k in ids]
    if not values:raise ValueError('empty_stopping_enclosure')
    # Time consumed by construction is included by the caller-provided clock.
    now=now_ns()*1e-9;horizon=max(0.,until-now)
    wall_geometry=time.monotonic()
    evidence=query(values,horizon,None)
    check()
    wall_query=time.monotonic()
    version=lambda s:(s['instance'],s['epoch'],s['revision'])
    geom=lambda s:(s['frame'],s['resolution'],s['map_min'],s['map_max'],s['grid_size'])
    if (not evidence['valid'] or evidence.get('clock_fault') or geom(evidence)!=geom(meta) or
            version(evidence)[:2]!=version(meta)[:2] or evidence['revision']<meta['revision']):
        raise ValueError('map_changed_or_invalid')
    arrays=('inside_map','confidence_at_horizon','occupancy','provenance','model_free_at_horizon')
    if any(len(evidence[k])!=len(ids) for k in arrays):raise ValueError('incomplete_stopping_snapshot')
    if any(not all(math.isfinite(x) for x in evidence[k]) for k in arrays):raise ValueError('nonfinite_map_evidence')
    free=all(i==1 and 0<=o<=.5 and .8<=c<=1 and f==1 and p in (1,2,3)
        for i,o,c,f,p in zip(evidence['inside_map'],evidence['occupancy'],evidence['confidence_at_horizon'],
                           evidence['model_free_at_horizon'],evidence['provenance']))
    reference_uncovered=([index for index,i,o,c,f,p in zip(ids,evidence['inside_map'],evidence['occupancy'],
        evidence['confidence_at_horizon'],evidence['model_free_at_horizon'],evidence['provenance'])
        if index in reference_ids and not (i==1 and 0<=o<=.5 and .8<=c<=1 and f==1 and p in (1,2,3))]
        if reference_available else [])
    recheck=query([],0.,version(evidence))
    check()
    wall_recheck=time.monotonic()
    stable=bool(recheck['valid'] and not recheck.get('clock_fault') and version(recheck)==version(evidence) and geom(recheck)==geom(evidence))
    observed_end=now_ns()*1e-9
    if (evidence['snapshot_ns']>observed_end*1e9 or recheck['snapshot_ns']>observed_end*1e9 or
            evidence['snapshot_ns']<meta['snapshot_ns'] or recheck['snapshot_ns']<evidence['snapshot_ns']):
        raise ValueError('invalid_snapshot_time')
    if abs(evidence['horizon_s']-horizon)>1e-9:raise ValueError('snapshot_horizon_mismatch')
    if observed_end>until:raise ValueError('stopping_evidence_already_expired')
    digest=hashlib.sha256(json.dumps(dict(ids=ids,points=values,budgets=budgets,margin=margin,
        enclosure_field=enclosure_field),
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if evidence_sink is not None:
        evidence_sink(dict(enclosure_sha256=digest,enclosure_field=enclosure_field,budgets=budgets,
                           q=list(q),margin=margin,grid_indices=ids,points=values,
                           metadata=meta,evidence=evidence,version_recheck=recheck,
                           reference_pose_grid_indices=sorted(reference_ids) if reference_available else None,
                           geometry_subdivisions=geometry_subdivisions))
    check()
    from phase_e_observation_evidence import observation_window
    observation=observation_window(evidence,until)
    check()
    return dict(complete=True,stable_version=stable,enclosure_sha256=digest,bound_modes=required,
        inside_map=inside and all(x==1 for x in evidence['inside_map']),all_centers_free=free,
        valid_until_s=evidence['snapshot_ns']*1e-9+evidence['horizon_s'],
        future_environment_qualified=False,map_identity=list(version(evidence)),
        snapshot_ns=evidence['snapshot_ns'],grid_indices=ids,points_queried=len(ids),
        map_geometry=list(geom(evidence)),
        reference_pose_uncovered_grid_indices=reference_uncovered,
        observation_window=observation,
        enclosure_kind='physical' if enclosure_field=='physical_travel_rad' else 'command_only',
        geometry_modes_evaluated=geometry_modes,
        wall_timing_s=dict(metadata=wall_meta-wall_start,geometry=wall_geometry-wall_meta,
            full_query=wall_query-wall_geometry,validation_and_recheck=wall_recheck-wall_query,
            total=time.monotonic()-wall_start),
        evidence_counts=dict(unknown_or_decayed=sum(c<.8 for c in evidence['confidence_at_horizon']),
            occupied=sum(o>.5 for o in evidence['occupancy']),
            oracle=sum(p not in (1,2,3) for p in evidence['provenance'])),
        runtime_admission=False)


def query_adaptive_stopping_snapshot(*,budgets,q,geometry,query,now_ns,measured_ns,
        authorization_until_ns,margin=.020,enclosure_field='physical_travel_rad',
        evidence_sink=None,levels=(1,2,4,8),wall_budget_s=.05,
        max_state_age_ns=50000000,wall_clock=time.monotonic):
    """Refine complete enclosures within two separately measured time budgets.

    Every successful level queries its OWN complete point set at a coherent
    content version. No finer-level saved answers certify a larger coarse set.
    Checkpoints reject late work; bounded RPC adapters additionally cap waits.
    Python/OS scheduling is not certified by this engineering timeout.
    """
    if (not levels or any(not isinstance(n,int) or n<1 or n>128 for n in levels) or
            list(levels)!=sorted(set(levels)) or not math.isfinite(wall_budget_s) or wall_budget_s<=0 or
            not isinstance(measured_ns,int) or measured_ns<=0 or max_state_age_ns<=0):
        raise ValueError('invalid_adaptive_query_contract')
    start=wall_clock();previous=[now_ns()];attempts=[];last={}
    def check():
        now=now_ns()
        if now<previous[0]:raise ValueError('query_ros_time_regressed')
        previous[0]=now
        if now<measured_ns:raise ValueError('future_query_measurement')
        if now-measured_ns>max_state_age_ns:raise ValueError('query_state_age_budget_exhausted')
        if now>=authorization_until_ns:raise ValueError('query_authorization_expired')
        if wall_clock()-start>=wall_budget_s:raise ValueError('query_wall_budget_exhausted')
    def bounded_query(*args):
        check()
        if hasattr(query,'call_with_timeout'):
            value=query.call_with_timeout(wall_budget_s-(wall_clock()-start),*args)
        else:value=query(*args)
        check()
        return value
    accepted=False;reason='refinement_exhausted'
    geometry_workspace={} if getattr(geometry,'supports_refinement_workspace',False) else None
    metadata_cache={}
    for n in levels:
        attempt_start=wall_clock();attempt=dict(subdivisions=n)
        try:
            check()
            last=query_complete_stopping_snapshot(budgets=budgets,q=q,geometry=geometry,
                query=bounded_query,now_ns=now_ns,margin=margin,enclosure_field=enclosure_field,
                evidence_sink=evidence_sink,geometry_subdivisions=n,check_budget=check,
                skip_outside_map=True,geometry_workspace=geometry_workspace,metadata_cache=metadata_cache)
            attempt.update(points_queried=last['points_queried'],inside_map=last['inside_map'],
                all_centers_free=last['all_centers_free'],stable_version=last['stable_version'],
                evidence_counts=last['evidence_counts'],wall_timing_s=last['wall_timing_s'])
            attempt['reference_pose_uncovered_count']=len(last['reference_pose_uncovered_grid_indices'])
            if not last['stable_version']:reason='map_version_changed_during_check'
            elif last['inside_map'] and last['all_centers_free']:
                accepted=True;reason='complete_free_stable_within_budget'
            elif last['reference_pose_uncovered_grid_indices']:reason='uncovered_reference_pose_margin'
            elif last['evidence_counts']['occupied']:reason='occupied_stopping_enclosure'
            else:reason='unknown_or_decayed_stopping_enclosure'
        except (ValueError,RuntimeError) as exc:
            reason=str(exc);attempt['error']=reason
        attempt['wall_elapsed_s']=wall_clock()-attempt_start;attempts.append(attempt)
        if accepted or reason not in ('enclosure_may_leave_map','unknown_or_decayed_stopping_enclosure',
                                       'occupied_stopping_enclosure'):
            break
    # A completed earlier level must never survive a later timeout as accepted.
    last=dict(last)
    last.update(runtime_admission=False,diagnostic_accepted=accepted,
        adaptive=dict(accepted=accepted,reason=reason,attempts=attempts,
            selected_subdivisions=attempts[-1]['subdivisions'] if accepted else None,
            wall_elapsed_s=wall_clock()-start,end_state_age_s=(now_ns()-measured_ns)*1e-9,
            wall_budget_s=wall_budget_s,max_state_age_ns=max_state_age_ns))
    if not accepted:last['complete']=False
    return last
