"""Actual native-query hints and finite observation-request lifetime.

Hints are advisory, not complete swept-volume evidence or execution authority.
Only existing final GCDF/VBC audits decide motion. Cancelling an old request
never changes the confidence map, counts it seen, or drops a safety witness.
"""
import json
import math
import threading
from collections import OrderedDict


def native_unknown_hints(context, threshold=.010):
    if not context.get('rolling_task') or context.get('repair'):
        return None
    batch=context['local_gcdf_batch']; trajectory=context['query_trajectory']
    points=trajectory['points']; rows=int(batch['num_pairs'])
    # Batch and query must describe the same native CDF query. Candidate is a
    # separately optimized curve; its later execution audit is authoritative.
    if str(batch['stamp_ns']) != str(trajectory['stamp_ns']):
        raise ValueError('query/batch identity mismatch')
    by_cell={}
    for i in range(rows):
        if int(batch['source_type'][i]) != 0 or not math.isfinite(batch['distance'][i]) or batch['distance'][i]>=threshold:
            continue
        k=int(batch['original_timestep'][i]);p=batch['point_flat'][3*i:3*i+3]
        if len(p)!=3 or not all(math.isfinite(v) for v in p):raise ValueError('invalid point')
        # Grid identity, rather than float32/double representation, deduplicates.
        key=tuple(int(round(float(v)/.05)) for v in p)
        if not 0<=k<len(points):continue
        q=points[k]['positions'];linear=batch['q_linearization_flat'][7*i:7*i+7]
        if len(q)!=7 or len(linear)!=7 or max(abs(a-b) for a,b in zip(q,linear))>1e-6:
            raise ValueError('row does not belong to native query pose')
        t=int(points[k]['time_ns'])*1e-9
        value=dict(point=[.05*v for v in key],native_query_time_s=t,source='native_query_gcdf_unknown',distance_joint_space=float(batch['distance'][i]))
        if key not in by_cell or t<by_cell[key]['native_query_time_s']:by_cell[key]=value
    return dict(goal_revision=context['goal_revision'],plan_seq=context['plan_seq'],mode_epoch=context['mode_epoch'],
        raw_candidate_stamp_ns=context['raw_candidate_stamp_ns'],query_stamp_ns=batch['stamp_ns'],
        points=sorted(by_cell.values(),key=lambda v:v['native_query_time_s']),motion_authorized=False,
        qualification='Original native CDF query constraints; advisory and non-exhaustive, not optimized-curve certification')


def request_matches(ob, event, region_token):
    return (event.get('consumed_after_actual_task_prefix') is True and
            str(event.get('execution_stamp_ns','')).isdigit() and int(event['execution_stamp_ns'])>0 and
            event.get('region_token')==region_token(ob) and
            event.get('observation_token')==event.get('point_gain_event',{}).get('observation_token') and
            event.get('region_token')==event.get('point_gain_event',{}).get('region_token'))


class RollingTaskObservation:
    def __init__(self,node):
        import rospy
        from std_msgs.msg import String
        self.node=node;self.lock=threading.Lock();self.latest=None;self.consumed=OrderedDict();self.last_plan=None
        self.sub=rospy.Subscriber('/care_planner/local_planner/candidate_trajectory/audit_context',String,self.context,queue_size=2)
        self.done=rospy.Subscriber('/care_planner/rolling_task/consumed_request',String,self.consume,queue_size=8)

    def context(self,msg):
        try:
            context=json.loads(msg.data);hints=native_unknown_hints(context)
            if hints is None:return
            with self.lock:
                if self.latest is not None and int(hints['raw_candidate_stamp_ns'])<=int(self.latest['raw_candidate_stamp_ns']):return
                self.latest=hints
            # No global completion/takeover is emitted for future unknowns.
            self.node._trace_observation('rolling_native_task_hints',**hints)
            if hints['points']:
                points=[v['point'] for v in hints['points']]
                import rospy
                from care_confidence_map.srv import QueryConfidence, QueryConfidenceRequest
                from geometry_msgs.msg import Point
                req=QueryConfidenceRequest(points=[Point(*p) for p in points])
                reply_message=rospy.ServiceProxy('/care_planner/confidence_map/query',QueryConfidence,persistent=False)(req)
                inside=reply_message.inside_map
                inside=list(inside) if not isinstance(inside,bytes) else list(bytes(inside))
                reply=dict(confidence=list(reply_message.confidence),inside_map=inside,
                    current_visibility=list(reply_message.current_visibility),query_ros_s=rospy.Time.now().to_sec(),
                    map_version='service_does_not_expose_version',head_attribution='unknown')
                self.node._trace_observation('rolling_hint_actual_query',raw_candidate_stamp_ns=hints['raw_candidate_stamp_ns'],
                    query_stamp_ns=hints['query_stamp_ns'],points=points,actual_query=reply,
                    changes_map=False,motion_authorized=False)
        except Exception as exc:
            self.node._trace_observation('rolling_native_hint_error',error=repr(exc),motion_authorized=False)

    def consume(self,msg):
        from observation_identity import observation_region_token
        try:
            event=json.loads(msg.data);key=(event['observation_token'],event['execution_stamp_ns'])
            with self.lock:
                if key in self.consumed:return
                # Malformed/ambiguous receipts cannot cancel any request.
                if not request_matches(dict(id=0,points=[],q_vis=[0.]*7),event,lambda ob:event.get('region_token')):return
                self.consumed[key]=True
                while len(self.consumed)>64:self.consumed.popitem(last=False)
            n=self.node
            with n._coherent_bundle_lock:
                with n._obligation_lock:
                    removed=[ob for ob in n._obligations if request_matches(ob,event,observation_region_token)]
                    ids={int(ob['id']) for ob in removed}
                    n._obligations=[ob for ob in n._obligations if int(ob['id']) not in ids]
                old_points={tuple(round(float(v),5) for v in p) for ob in removed for p in ob['points']}
                retired=[]
                for record in list(n._pending_vbc_blockers.records.values()):
                    cells={tuple(round(float(v),5) for v in p) for p in record['points']}
                    if record['source']=='candidate' and cells and cells<=old_points:
                        if n._pending_vbc_blockers.retire_obsolete(record['generation']):retired.append(record['generation'])
                n._coherent_bundle_pending_nonempty=n._pending_vbc_blockers.pending
            n._trace_observation('rolling_request_consumed',receipt=event,retired_obligation_ids=sorted(ids),
                retired_pending_generations=retired,actual_observation_completed=False,map_changed=False,
                remaining_obligation_count=len(n._obligations))
            if ids:
                n._prune_or_initialize_stack();n._publish_schedule()
        except Exception as exc:
            self.node._trace_observation('rolling_request_consume_error',error=repr(exc))
