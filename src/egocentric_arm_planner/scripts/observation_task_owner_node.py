#!/usr/bin/env python3
"""Explicit task/context publisher. It publishes planning state only."""
import json
import sys
import threading
from pathlib import Path
import rospkg
import rospy
from std_msgs.msg import String

sys.path.insert(0, str(Path(rospkg.RosPack().get_path('care_visibility_cdf'))/'scripts'))
from task_observation_binding import ObservationTaskOwner, canonical


class TaskOwnerNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.owner = ObservationTaskOwner(heartbeat_s=rospy.get_param('~heartbeat_valid_s', .8))
        self.last_map_scope = self.invalidated_map_scope = None
        self.snapshot_worker = None
        self.pub = rospy.Publisher('/care_planner/active_sensing/observation_context', String,
                                   queue_size=32, latch=True)
        self.audit = rospy.Publisher('/care_planner/task_owner/events', String, queue_size=64)
        rospy.Subscriber('/care_planner/task_request', String, self.command, queue_size=32)
        rospy.Subscriber(rospy.get_param('~demands_topic',
                         '/care_planner/active_sensing/stopping_observation_demands'), String,
                         self.invalidated, queue_size=32)
        self.timer = rospy.Timer(rospy.Duration(.1), self.tick)

    def emit(self, events):
        # Call under the owner lock so cancellation cannot be overtaken by a heartbeat.
        for event in events:
            message = String(data=canonical(event))
            self.pub.publish(message)
            self.audit.publish(message)

    def command(self, msg):
        try:
            command = json.loads(msg.data)
            with self.lock:
                now = rospy.Time.now().to_nsec()
                if command.get('action') == 'resume_map':
                    command['fresh_map_scope'] = self.fresh_map_scope()
                events = (self.owner.resume_map(command, now) if command.get('action') == 'resume_map'
                          else self.owner.command(command, now))
                if command.get('action') == 'resume_map':
                    self.last_map_scope = command['fresh_map_scope']
                self.emit(events)
        except (ValueError, KeyError, TypeError) as exc:
            self.audit.publish(String(data=canonical(dict(kind='task_request_rejected', reason=str(exc)))))

    def invalidated(self, msg):
        try:
            with self.lock:
                message = json.loads(msg.data)
                if message.get('kind') == 'observation_context_invalidated':
                    events = self.owner.map_invalidated(message, rospy.Time.now().to_nsec())
                    if events: self.invalidated_map_scope = self.last_map_scope
                    self.emit(events)
                elif message.get('task_context') == (self.owner.current or {}).get('candidate_binding'):
                    identity, geometry = message.get('map_identity'), message.get('map_geometry')
                    if isinstance(identity, list) and len(identity) == 3 and geometry:
                        self.last_map_scope = dict(instance=identity[0], epoch=identity[1], geometry=geometry)
        except (ValueError, KeyError, TypeError) as exc:
            rospy.logwarn('Invalid task map event: %s', exc)

    def fresh_map_scope(self):
        """A caller's claimed map identity is never used to reopen work."""
        if self.snapshot_worker is not None and self.snapshot_worker.is_alive():
            raise ValueError('fresh_map_snapshot_still_inflight')
        from care_confidence_map.srv import QueryConfidenceSnapshot, QueryConfidenceSnapshotRequest
        result = []
        ready = threading.Event()
        def query():
            try:
                proxy = rospy.ServiceProxy(rospy.get_param('~snapshot_service',
                    '/care_planner/confidence_map/query_snapshot'), QueryConfidenceSnapshot)
                result.append(proxy(QueryConfidenceSnapshotRequest(horizon_s=0.,minimum_confidence=.8)))
            except Exception as exc:
                result.append(exc)
            finally: ready.set()
        self.snapshot_worker = threading.Thread(target=query,daemon=True)
        self.snapshot_worker.start()
        if not ready.wait(.15) or not result or isinstance(result[0],Exception):
            raise ValueError('fresh_map_snapshot_unavailable')
        value = result[0]
        age = (rospy.Time.now()-value.snapshot_stamp).to_sec()
        geometry = [value.frame,value.resolution,
                    [value.map_min.x,value.map_min.y,value.map_min.z],
                    [value.map_max.x,value.map_max.y,value.map_max.z],list(value.grid_size)]
        scope = dict(instance=value.instance,epoch=value.epoch,geometry=geometry)
        if (not value.valid or value.clock_fault or not 0 <= age <= .05 or
                value.frame != (self.owner.current or {}).get('frame_id') or
                self.invalidated_map_scope is None or scope == self.invalidated_map_scope):
            raise ValueError('fresh_map_scope_not_verified')
        return scope

    def tick(self, _event):
        with self.lock:
            self.emit(self.owner.tick(rospy.Time.now().to_nsec()))


if __name__ == '__main__':
    rospy.init_node('observation_task_owner')
    TaskOwnerNode()
    rospy.spin()
