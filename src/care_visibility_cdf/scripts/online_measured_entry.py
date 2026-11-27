"""Give measured body entry its first scheduling opportunity after a ToF update."""
import time

import rospy
from care_confidence_map.srv import QueryConfidenceSnapshot

from measured_body_entry import TaskRouteComparison as MeasuredEntry
from observation_corridor_cost import BoundedCorridorQuery


class TaskRouteComparison(MeasuredEntry):
    def __init__(self, node):
        self.entry_attempts = 0
        super().__init__(node)
        # This worker is asynchronous. A complete snapshot may wait behind
        # map integration while the existing local planning loop keeps running.
        self.query = BoundedCorridorQuery(rospy.ServiceProxy(
            node.confidence_query_service + '_snapshot', QueryConfidenceSnapshot), timeout_s=.18)

    def _entry_support(self, grid, ids):
        support = super()._entry_support(grid, ids)
        self.node._trace_observation('task_route_body_snapshot', cells=len(ids),
                                     valid=support is not None,
                                     sensor_supported=int(support[1].sum()) if support is not None else 0)
        return support

    def _entry(self, runtime, packets, start, stopped, goal, accel):
        resumed = self._resume(runtime, start, stopped, goal, accel)
        if resumed is not None:
            return resumed
        stamp = packets[-1].header.stamp.to_sec()
        if stamp - self.last_body_search >= .55:
            self.entry_attempts += 1
            self.last_body_search = stamp
            began = time.monotonic()
            entry = self._enter_free_space(runtime, start, stopped, goal, accel)
            self.node._trace_observation('task_route_body_entry_attempt', attempt=self.entry_attempts,
                                         measured_start=start.tolist(), stopped_state=stopped.tolist(),
                                         has_proposal=entry is not None,
                                         wall_ms=1000. * (time.monotonic() - began))
            if entry is not None:
                return entry
        if self.entry_attempts < 4:
            # The unchanged R1/REPAIR fallback continues in the main planner.
            return None
        return self._new_view(runtime, packets, start, stopped, goal, accel)
