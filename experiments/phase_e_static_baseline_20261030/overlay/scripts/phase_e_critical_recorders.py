"""Benchmark-only connection barrier for one-shot audit and commit records."""
import time
from phase_e_actuator_readiness import rpc


def connection(info, topic, sender, sender_uri):
    matches = [r for r in info if len(r) >= 6 and r[2] == 'i'
               and r[3] == 'TCPROS' and r[4] == topic and r[5] is True
               and r[1] in (sender, sender_uri)]
    return matches[0][0] if len(matches) == 1 else None


class CriticalRecorderReadiness:
    def __init__(self, recorders, topics, rospy, master_uri):
        if not topics or len(set(topics)) != len(topics):
            raise ValueError('critical recorder topics must be nonempty and unique')
        self.recorders, self.topics = set(recorders), list(topics)
        self.rospy, self.master_uri = rospy, master_uri
        self.last = dict(ready=False, reason='waiting_critical_recorder_connections')

    def check(self, deadline, publishers, subscribers):
        edges = []
        try:
            pubs, subs = dict(publishers), dict(subscribers)
            caller = self.rospy.get_name()
            uris, buses = {}, {}
            def uri(node):
                if node not in uris:
                    uris[node] = rpc(self.master_uri, 'lookupNode', deadline, caller, node)
                return uris[node]
            for topic in self.topics:
                receivers = sorted(self.recorders.intersection(subs.get(topic, [])))
                senders = sorted(pubs.get(topic, []))
                if not receivers or not senders:
                    raise RuntimeError('critical_topic_not_registered:' + topic)
                for receiver in receivers:
                    if receiver not in buses:
                        buses[receiver] = rpc(uri(receiver), 'getBusInfo', deadline, caller)
                    for sender in senders:
                        cid = connection(buses[receiver], topic, sender, uri(sender))
                        if cid is None:
                            raise RuntimeError('critical_recorder_not_connected:' + receiver + ':' + topic)
                        edges.append(dict(topic=topic, receiver=receiver, sender=sender,
                            receiver_uri=uri(receiver), sender_uri=uri(sender), connection_id=cid))
            self.last = dict(ready=time.monotonic() < deadline,
                reason='critical_recorders_connected', checked_monotonic_s=time.monotonic(),
                topics=self.topics, edges=edges,
                qualification='Actual incoming TCPROS connections before goal arm; no synthetic audit or commit messages are published.')
        except Exception as exc:
            self.last = dict(ready=False, reason=str(exc), topics=self.topics, edges=edges)
        return self.last['ready']
