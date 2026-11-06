"""Explicit, content-bound recovery intent; only native geometry can authorize it."""
import math
import re

def validate(value):
    fields = {'schema', 'raw_candidate_stamp_ns', 'query_stamp_ns', 'mode_epoch',
              'observation_token', 'points'}
    if isinstance(value,dict) and value.get('schema')==2:fields.add('query_ros_ns')
    if isinstance(value,dict) and 'margin_recovery' in value:
        if value.get('schema')!=2 or value['margin_recovery'] is not True:
            raise ValueError('Invalid margin recovery intent')
        fields.add('margin_recovery')
    if isinstance(value,dict) and 'fixed_start_recovery_points' in value:
        if value.get('schema')!=2 or value.get('margin_recovery',False):
            raise ValueError('Invalid fixed-start recovery intent')
        fields.add('fixed_start_recovery_points')
    if not isinstance(value, dict) or set(value) != fields or type(value['schema']) is not int or value['schema'] not in (1,2):
        raise ValueError('Invalid geometric restoration intent')
    for name in ('raw_candidate_stamp_ns', 'query_stamp_ns', 'mode_epoch')+ (('query_ros_ns',) if value['schema']==2 else ()):
        v = value[name]
        if not isinstance(v, str) or re.fullmatch('[0-9]{1,20}', v) is None or not 0<int(v)<2**64:
            raise ValueError('Invalid geometric restoration identity')
    # The query identity is monotonically uniquified and may be a few ns
    # ahead of ROS time. Physical freshness uses the separately bound clock.
    query_time=value['query_ros_ns'] if value['schema']==2 else value['query_stamp_ns']
    if (int(query_time) > int(value['raw_candidate_stamp_ns']) or
            not isinstance(value['observation_token'], str) or
            re.fullmatch('care_obs_v1_[A-Za-z0-9_]{1,100}', value['observation_token']) is None):
        raise ValueError('Invalid geometric restoration observation')
    points=value['points']
    if not isinstance(points,list) or not 1<=len(points)<=32:
        raise ValueError('Invalid geometric restoration point count')
    seen=set()
    for p in points:
        if not isinstance(p,list) or len(p)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in p):
            raise ValueError('Invalid geometric restoration point')
        key=tuple(p)
        if key in seen: raise ValueError('Duplicate geometric restoration point')
        seen.add(key)
    if 'fixed_start_recovery_points' in value:
        subset=value['fixed_start_recovery_points']
        if (not isinstance(subset,list) or not 1<=len(subset)<=32 or
                any(not isinstance(p,list) or len(p)!=3 or
                    any(type(x) not in (int,float) or not math.isfinite(x) for x in p) for p in subset)):
            raise ValueError('Invalid fixed-start recovery points')
        if len({tuple(p) for p in subset})!=len(subset) or any(tuple(p) not in seen for p in subset):
            raise ValueError('Unbound fixed-start recovery point')
    return value

def permits_unknown_conflicts(intent, batch, distances, margin):
    """Only explicit UNKNOWN point identities may request alternate geometry.

    This returns permission to request certification, never permission to move.
    Every other row and all occupied rows retain the original learned gate.
    """
    validate(intent)
    n=int(batch.num_pairs)
    if (len(distances)!=n or len(batch.source_type)!=n or len(batch.point_flat)!=3*n or
            any(not math.isfinite(d) for d in distances)):
        return False
    unknown=int(getattr(batch,'SOURCE_UNKNOWN',0))
    for i,d in enumerate(distances):
        if d>=margin: continue
        # Consuming a geometric planning reserve does not waive learned rows.
        if intent.get('margin_recovery',False): return False
        if int(batch.source_type[i]) != unknown: return False
        p=list(batch.point_flat[3*i:3*i+3])
        if not any(max(abs(p[j]-q[j]) for j in range(3))<=1e-5 for q in intent['points']):
            return False
    return True


def validate_fixed_start_reaudit(intent, evidence, planning_floor, enabled):
    """Check the native classification without changing the immutable intent.

    This checks proof metadata, not geometry. Native continuous proofs and the
    live residual/identity/freshness checks remain mandatory before commit.
    """
    keys={'fixed_start_reaudit_points','fixed_start_reaudit_floor_m','fixed_start_reaudit_contract'}
    present=keys.intersection(evidence)
    fixed=[] if intent is None else intent.get('fixed_start_recovery_points',[])
    if not fixed:
        if present:raise ValueError('Unbound fixed-start re-audit')
        return
    if not math.isfinite(planning_floor) or planning_floor<=0:
        raise ValueError('Invalid planning floor')
    rows={}
    for row in evidence['points']:
        p=row['point']
        if (not isinstance(p,list) or len(p)!=3 or
                any(type(x) not in (int,float) or not math.isfinite(x) for x in p) or tuple(p) in rows):
            raise ValueError('Invalid point proof identity')
        rows[tuple(p)]=row
    required=set()
    for p in fixed:
        row=rows[tuple(p)]
        if type(row['start_m']) not in (int,float) or not math.isfinite(row['start_m']):
            raise ValueError('Invalid fixed-start distance')
        if row['start_m']>=planning_floor-1e-12:required.add(tuple(p))
    if not required:
        if present:raise ValueError('Unexpected fixed-start re-audit')
        return
    if (not enabled or present!=keys or
            evidence['fixed_start_reaudit_contract']!='planning_floor_path_onset_stop_v1' or
            type(evidence['fixed_start_reaudit_floor_m']) not in (float,int) or
            evidence['fixed_start_reaudit_floor_m']!=planning_floor):
        raise ValueError('Missing or incompatible planning-floor proof')
    declared=evidence['fixed_start_reaudit_points']
    if (not isinstance(declared,list) or len(declared)!=len(required) or
            any(not isinstance(p,list) or len(p)!=3 or
                any(type(x) not in (int,float) or not math.isfinite(x) for x in p) for p in declared) or
            {tuple(p) for p in declared}!=required):
        raise ValueError('Re-audit classification mismatch')
    for point in required:
        row=rows[point]
        if (any(type(row[k]) not in (int,float) or not math.isfinite(row[k])
                for k in ('floor_m','start_m','actual_m','end_m')) or
                row['floor_m']!=planning_floor or
                min(row['start_m'],row['actual_m'],row['end_m'])<=planning_floor):
            raise ValueError('Re-audit did not establish the planning floor')
