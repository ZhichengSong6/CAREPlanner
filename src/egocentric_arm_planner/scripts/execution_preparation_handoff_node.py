#!/usr/bin/env python3
"""Shadow stopping-admission checks and validated staging; no lease publisher."""
import json
import copy
import threading
import time
import rospy
from std_msgs.msg import String
from care_velocity_controller.msg import CommandStatus
from candidate_admission import CandidateTransfer, WallWorker
from execution_handoff import ExecutionHandoffShadow


class ExecutionPreparationHandoffNode:
    def __init__(self):
        self.lock = threading.RLock()
        self.native_acknowledged = False
        self.last_forward = -float('inf')
        self.forward_attempts = 0
        names = rospy.get_param('~joint_names')
        def limits(key, configured):
            value = rospy.get_param('~'+key, None)
            return value if value is not None else [rospy.get_param('~mpc/'+configured+'/'+n) for n in names]
        self.shadow = ExecutionHandoffShadow(names,
            limits('velocity_limits', 'joint_velocity_limits'),
            limits('acceleration_limits', 'joint_acceleration_limits'),
            rospy.get_param('~runtime_sha256', ''))
        self.staging = rospy.Publisher(rospy.get_param('~staged_topic',
            '/care_planner/execution/staged_trajectory'), CandidateTransfer, queue_size=1)
        self.status = rospy.Publisher(rospy.get_param('~status_topic',
            '/care_planner/execution/handoff_status'), String, queue_size=10)
        self.subscribers = [
            rospy.Subscriber(rospy.get_param('~task_context_topic', '/care_planner/active_sensing/observation_context'),
                String, lambda m: self.call(self.shadow.update_context, json.loads(m.data))),
            rospy.Subscriber(rospy.get_param('~task_control_topic', '/care_planner/c4_4/task_control'),
                String, lambda m: self.call(self.shadow.update_control, m.data)),
            rospy.Subscriber(rospy.get_param('~controller_state_topic', '/care_arm/arm_group_velocity_controller/state'),
                CommandStatus, lambda m: self.call(self.shadow.observe_controller, m), queue_size=20),
            rospy.Subscriber(rospy.get_param('~prepared_topic', '/care_planner/execution/prepared_trajectory'),
                CandidateTransfer, self.prepared, queue_size=1),
            rospy.Subscriber(rospy.get_param('~tracker_status_topic',
                '/care_planner/execution/tracker_preparation_status'), String, self.tracker_status, queue_size=10)]
        self.worker = WallWorker(self.tick, lambda e: rospy.logwarn_throttle(1., str(e)), .01)
        rospy.on_shutdown(self.worker.stop.set)

    def call(self, method, value):
        with self.lock:
            try:
                return method(value, rospy.Time.now().to_nsec())
            except (ValueError, KeyError, TypeError) as exc:
                rospy.logwarn_throttle(.2, 'Handoff input rejected: '+str(exc))

    def prepared(self, message):
        with self.lock:
            try:
                forwarded = self.shadow.accept(message, rospy.Time.now().to_nsec())
                if forwarded is not None:
                    self.native_acknowledged = False
                    self.last_forward = -float('inf')
            except (ValueError, KeyError, TypeError) as exc:
                rospy.logwarn_throttle(.2, 'Preparation rejected: '+str(exc))

    def tracker_status(self, message):
        with self.lock:
            try:
                if len(message.data) > 16384:
                    return
                value = json.loads(message.data)
                if not isinstance(value, dict):
                    return
                pending = self.shadow.pending
                if pending:
                    expected = pending['message']
                    if (value.get('pending') is True and
                            value.get('preparation_session') == expected.session and
                            value.get('preparation_sequence') == expected.sequence and
                            value.get('preparation_sha256') == expected.payload_sha256):
                        self.native_acknowledged = True
            except (ValueError, TypeError):
                pass

    def tick(self):
        with self.lock:
            now = rospy.Time.now().to_nsec()
            if now:
                report = self.shadow.report(now)
                wall = time.monotonic()
                if self.shadow.pending and not self.native_acknowledged and wall-self.last_forward >= .02:
                    forwarded = copy.deepcopy(self.shadow.pending['message'])
                    forwarded.remaining_budget_s = (self.shadow.pending['expires_ns']-now)*1e-9
                    self.staging.publish(forwarded)
                    self.last_forward = wall
                    self.forward_attempts += 1
                report.update(native_staging_acknowledged=bool(self.shadow.pending and self.native_acknowledged),
                              staging_forward_attempts=self.forward_attempts)
                self.status.publish(String(data=json.dumps(report, sort_keys=True)))


if __name__ == '__main__':
    rospy.init_node('execution_preparation_handoff')
    ExecutionPreparationHandoffNode()
    rospy.spin()
