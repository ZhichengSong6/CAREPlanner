"""Generate task detours from measured free space; R1 owns observation directions."""

import numpy as np
import rospy
from std_msgs.msg import String
from local_observation_actions import LocalObservationActions

from live_tof_route import TaskRouteComparison as LiveToFRoute


class TaskRouteComparison(LiveToFRoute):
    local_observation_directions = True
    choose_observation_only = True

    def __init__(self, node):
        self.direction_history = LocalObservationActions()
        super().__init__(node)
        self.handoff_sub = rospy.Subscriber('/care_planner/local_planner/summary/observation_handoff',
                                           String, self._handoff, queue_size=20)

    def _handoff(self, msg):
        info = dict(field.split('=', 1) for field in msg.data.split() if '=' in field)
        token = info.get('observation_token')
        if not token or info.get('event') not in ('refused', 'received'):
            return
        event = info.pop('event')
        self.node._trace_observation('local_r1_' + event, **info)
        if event == 'refused':
            published = getattr(self.node, '_trace_published_target', None) or {}
            if published.get('observation_token') == token:
                with self.selection_lock:
                    self.last_request = -1.
                self.node._stalled_view_cb(String(data=token))

    def choose(self, blocked, reference, identity, previous_view=None, excluded=()):
        # The acquisition node handles this blocker with its local R1 generator.
        # Task detours use the independent measured-map callback in LiveToFRoute.
        return None

    def _goal(self, message):
        with self.selection_lock:
            previous = self.goal_pose
            super()._goal(message)
            if previous is None or self.goal_pose is previous:
                return
            if all(
                np.allclose(a, b, rtol=0.0, atol=1.0e-8)
                for a, b in zip(previous, self.goal_pose)
            ):
                return
            # A result already queued for the previous goal cannot be published.
            work = getattr(self, "route_future", None)
            if work is not None:
                work.cancel()
                self.route_future = None
            self.last_ready = None
            self.direction_history.clear()
