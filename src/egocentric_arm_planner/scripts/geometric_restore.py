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
