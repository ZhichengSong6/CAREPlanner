"""Read-only startup proof of receipt on actual observation topic connections.

The q topic carries one empty startup sample, which the planner rejects as a
target. Persistent points may be latched; live active/summary/frontier streams
must advance. Unused C4.9 multi-deadline schedule topics are deliberately absent.
"""
import time
from phase_e_actuator_readiness import rpc, received_count

PREFIX = '/care_planner/active_sensing/'
GENERATOR = '/vbc_deadline_waypoint_rolling'
PLANNER = '/local_sparse_scp_planner_node'
EDGES = [
    (PLANNER, 'visibility_waypoint_active', True),
    (PLANNER, 'visibility_waypoint_q', False),
    (PLANNER, 'current_visibility_obligation_points', False),
    (PLANNER, 'visibility_frontier_target', True),
    (PLANNER, 'visibility_acquisition_summary', True),
    ('/vbc_execution_reference_gate', 'visibility_waypoint_summary', True),
    ('/c4_4_verified_regime_manager', 'visibility_waypoint_active', True),
    ('/c4_4_verified_regime_manager', 'visibility_acquisition_complete', True),
    ('/predicted_vbc_auditor_node', 'visibility_waypoint_active', True),
]
RECORDED = ('visibility_waypoint_summary', 'visibility_acquisition_summary',
            'visibility_acquisition_complete', 'blocker_stack_summary',
            'visibility_frontier_target', 'visibility_frontier_summary')


class ObservationReadiness:
    def __init__(self, required_recorders, rospy, master_uri):
        self.recorders = set(required_recorders)
        self.rospy, self.master_uri = rospy, master_uri
        self.previous = None
        self.last = dict(ready=False, reason='waiting_observation_receivers')

    def check(self, deadline, subscribers):
        try:
            edges = [(node, PREFIX+topic, live) for node, topic, live in EDGES]
            by_topic = dict(subscribers)
            for topic in RECORDED:
                selected = self.recorders.intersection(by_topic.get(PREFIX+topic, []))
                if not selected:
                    raise RuntimeError('missing_observation_recorder:' + PREFIX+topic)
                edges.extend((node, PREFIX+topic, True) for node in sorted(selected))
            caller = self.rospy.get_name()
            nodes = {GENERATOR} | {e[0] for e in edges}
            uris = {node: rpc(self.master_uri, 'lookupNode', deadline, caller, node)
                    for node in sorted(nodes)}
            buses = {node: (rpc(uris[node], 'getBusInfo', deadline, caller),
                            rpc(uris[node], 'getBusStats', deadline, caller))
                     for node in sorted(nodes - {GENERATOR})}
            counts = []
            for receiver, topic, live in edges:
                count = received_count(*buses[receiver], topic, GENERATOR, uris[GENERATOR])
                if count is None or count[1] < 1:
                    raise RuntimeError('observation_not_received:' + receiver + ':' + topic)
                counts.append((receiver, uris[receiver], uris[GENERATOR], topic, live, *count))
            now = time.monotonic()
            prior = self.previous
            self.previous = (now, counts)
            advancing = (prior is not None and 0 < now-prior[0] <= 2.0 and
                         len(counts) == len(prior[1]) and all(
                             cur[:-1] == old[:-1] and
                             (cur[-1] > old[-1] if cur[4] else cur[-1] >= old[-1])
                             for cur, old in zip(counts, prior[1])))
            self.last = dict(ready=bool(advancing and now < deadline),
                reason='observation_consumers_receiving' if advancing else 'waiting_live_observation_counts',
                checked_monotonic_s=now, deadline_exhausted=now >= deadline,
                poll_span_s=None if prior is None else now-prior[0],
                nonadvancing=[] if prior is None else [cur[0]+':'+cur[3]
                    for cur, old in zip(counts, prior[1])
                    if cur[:-1] != old[:-1] or (cur[4] and cur[-1] <= old[-1])],
                receiver_counts=[dict(receiver=r[0], receiver_uri=r[1], sender_uri=r[2], topic=r[3],
                                      periodic=r[4], connection_id=r[5], received_messages=r[6]) for r in counts],
                qualification='Actual named receivers on the generator connections; neutral q startup sample is not an observation target or motion authorization.')
        except Exception as exc:
            self.previous = None
            exhausted = time.monotonic() >= deadline
            self.last = dict(ready=False, reason=str(exc), deadline_exhausted=exhausted,
                             last_observation_evidence=self.last if exhausted else None)
        return self.last['ready']
