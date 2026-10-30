"""Read-only transport evidence from the actual command receivers before arm."""
import math
import threading
import time
import xmlrpc.client


class TimedTransport(xmlrpc.client.Transport):
    def __init__(self, timeout):
        super().__init__(); self.timeout = timeout

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = self.timeout
        return connection


def bus_stats_payload(value):
    """Validate the subscriber table used for receipt evidence."""
    array = lambda x: isinstance(x, (list, tuple))
    integer = lambda x: type(x) is int and x >= 0
    if not array(value) or len(value) not in (2, 3):
        return False
    if not array(value[0]) or not array(value[1]):
        return False
    if len(value) == 3 and value[2] is not None and not array(value[2]):
        return False
    for row in value[0]:
        # roscpp: [topic, connections]; rospy: [topic, bytes, connections].
        if not array(row) or len(row) not in (2, 3) or not isinstance(row[0], str):
            return False
        if not array(row[-1]) or (len(row) == 3 and not integer(row[1])):
            return False
    for row in value[1]:
        if not array(row) or len(row) != 2 or not isinstance(row[0], str) or not array(row[1]):
            return False
        for connection in row[1]:
            if not array(connection) or len(connection) != 5:
                return False
            if not all(integer(x) for x in connection[:3]):
                return False
            if type(connection[3]) not in (int, float) or not math.isfinite(connection[3]):
                return False
            if type(connection[4]) is not bool and not (type(connection[4]) is int and connection[4] == 0):
                return False
    return True


def rpc(uri, method, deadline, *args):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('actuator readiness deadline')
    with xmlrpc.client.ServerProxy(uri, transport=TimedTransport(min(.15, remaining))) as proxy:
        result = getattr(proxy, method)(*args)
    wrapped = (isinstance(result, (list, tuple)) and len(result) == 3 and
               type(result[0]) is int and result[0] == 1 and isinstance(result[1], str))
    if wrapped and (method != 'getBusStats' or bus_stats_payload(result[2])):
        return result[2]
    # Noetic roscpp getBusStatsCallback writes the tables into result itself,
    # overwriting its status envelope. The unset service entry can be omitted
    # by XMLRPC, yielding [publish_stats, subscribe_stats]. Accept that shape
    # only for this method and only after validating the receipt table.
    # https://github.com/ros/ros_comm/blob/noetic-devel/clients/roscpp/src/libros/topic_manager.cpp
    if method == 'getBusStats' and bus_stats_payload(result):
        return result
    raise RuntimeError('actuator readiness RPC failed: ' + method)


def received_count(info, stats, topic, sender, sender_uri):
    """Match receiver-side incoming connection ID to its received-message count."""
    incoming = [row for row in info if len(row) >= 6 and row[2] == 'i' and
                row[3] == 'TCPROS' and row[4] == topic and row[5] and
                row[1] in (sender, sender_uri)]
    if len(incoming) != 1:
        return None
    connection = incoming[0][0]
    # rospy provides a boolean connected flag. roscpp Subscription::getStats
    # writes integer 0 as an unimplemented placeholder in that slot:
    # https://github.com/ros/ros_comm/blob/noetic-devel/clients/roscpp/src/libros/subscription.cpp
    # Connection identity comes from getBusInfo and readiness still requires
    # advancing receiver counters; the placeholder cannot act as a gate.
    matches = [c for row in stats[1] if len(row) == 2 and row[0] == topic
               for c in row[1] if len(c) >= 5 and c[0] == connection and
               (c[4] is True or (type(c[4]) is int and c[4] == 0))]
    if len(matches) != 1:
        return None
    count = matches[0][2]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return None
    return connection, count


class ActuatorReadiness:
    def __init__(self, config, rospy, master_uri):
        from std_msgs.msg import Float64MultiArray
        from sensor_msgs.msg import JointState
        self.config = dict(config)
        for key in ('tracker_node', 'limiter_node', 'controller_node', 'desired_topic', 'command_topic'):
            if not isinstance(config.get(key), str) or not config[key].startswith('/'):
                raise ValueError('missing actuator readiness configuration: ' + key)
        self.rospy = rospy; self.master_uri = master_uri
        self.lock = threading.Lock(); self.zero_time = -math.inf
        self.previous = None; self.last = {'ready': False, 'reason': 'waiting_receivers'}
        self.command_sub = rospy.Subscriber(config['command_topic'],
                                            JointState if config.get('command_stamped', False) else Float64MultiArray,
                                            self.command, queue_size=1, tcp_nodelay=True)

    def command(self, msg):
        values = msg.velocity if self.config.get('command_stamped', False) else msg.data
        valid = len(values) == 7 and all(math.isfinite(x) and abs(x) <= 1e-6 for x in values)
        if self.config.get('command_stamped', False):
            age = (self.rospy.Time.now()-msg.header.stamp).to_sec()
            valid = valid and 0 <= age <= .1
        with self.lock:
            self.zero_time = time.monotonic() if valid else -math.inf

    def check(self, deadline):
        c = self.config; caller = self.rospy.get_name()
        try:
            uris = {node: rpc(self.master_uri, 'lookupNode', deadline, caller, node)
                    for node in [c['tracker_node'], c['limiter_node'], c['controller_node']]}
            counts = []
            for receiver, sender, topic in [
                    (c['limiter_node'], c['tracker_node'], c['desired_topic']),
                    (c['controller_node'], c['limiter_node'], c['command_topic'])]:
                info = rpc(uris[receiver], 'getBusInfo', deadline, caller)
                stats = rpc(uris[receiver], 'getBusStats', deadline, caller)
                count = received_count(info, stats, topic, sender, uris[sender])
                if count is None:
                    raise RuntimeError('receiver_not_connected:' + receiver)
                counts.append((receiver, uris[receiver], sender, uris[sender], topic, *count))
            now = time.monotonic()
            with self.lock:
                zero_fresh = 0 <= now-self.zero_time <= .2
            prior = self.previous; self.previous = (now, counts)
            # An old counter or a restarted receiver cannot acknowledge new traffic.
            advancing = prior is not None and 0 < now-prior[0] <= .5 and all(
                current[:-1] == old[:-1] and current[-1] > old[-1]
                for current, old in zip(counts, prior[1]))
            self.last = dict(ready=bool(advancing and zero_fresh),
                reason='received_zero_stream' if advancing and zero_fresh else 'waiting_fresh_zero_and_receiver_counts',
                checked_monotonic_s=now, zero_stream_fresh=zero_fresh,
                receiver_counts=[dict(receiver=row[0], receiver_uri=row[1], sender=row[2],
                    sender_uri=row[3], topic=row[4], connection_id=row[5], received_messages=row[6]) for row in counts],
                qualification='Receiver transport receipt of the pre-goal zero stream; not controller application or mechanical response.')
        except Exception as exc:
            self.previous = None
            if time.monotonic()>=deadline:
                self.last.update(ready=False, deadline_exhausted=True)
            else:
                self.last = dict(ready=False, reason=str(exc))
        return self.last['ready']

    def close(self):
        self.command_sub.unregister()
