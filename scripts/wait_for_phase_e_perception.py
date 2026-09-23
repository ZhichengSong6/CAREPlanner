#!/usr/bin/env python3
"""Benchmark prerequisite only: observe real inputs; never publish a goal/command."""
import argparse
import json
import math
from pathlib import Path
import re
import threading
import time


class PerceptionReadiness:
    def __init__(self, topics, fresh_s=1.0):
        if not topics or len(set(topics)) != len(topics):
            raise ValueError('Expected distinct input topics')
        self.topics = list(topics)
        self.fresh_s = fresh_s
        self.clouds = {t: dict(messages=0, advances=0, stamp_ns=0, last_valid_wall=None)
                       for t in topics}
        self.summaries = {}

    def cloud(self, topic, stamp_ns, organized_data_valid, now):
        state = self.clouds[topic]
        state['messages'] += 1
        if organized_data_valid and stamp_ns > state['stamp_ns']:
            state['advances'] += 1
            state['stamp_ns'] = stamp_ns
            state['last_valid_wall'] = now

    def summary(self, kind, text, now):
        tokens = dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)', text))
        state = self.summaries.setdefault(kind, dict(first=tokens, latest={}, received=0))
        state.update(latest=tokens, last_wall=now, received=state['received']+1)

    def snapshot(self, now, raw_only=False):
        missing = []
        for topic, state in self.clouds.items():
            if (state['advances'] < 2 or state['last_valid_wall'] is None or
                    not 0 <= now-state['last_valid_wall'] <= self.fresh_s):
                missing.append('fresh_advancing_cloud:'+topic)
        if not raw_only:
            requirements = {
                'fusion': (('callbacks_received', 'callbacks_processed'),
                           {'configured_sensors': len(self.topics), 'active_sensors': len(self.topics),
                            'ray_observations': 1}),
                'map': (('ray_packet_count',), {'last_ray_count': 1, 'last_free_cell_count': 1})}
            for kind, (counters, minima) in requirements.items():
                state = self.summaries.get(kind)
                if not state or now-state['last_wall'] > self.fresh_s:
                    missing.append('fresh_summary:'+kind)
                    continue
                try:
                    if any(int(state['latest'][k]) <= int(state['first'][k]) for k in counters):
                        missing.append('advancing_counters:'+kind)
                    if any(int(state['latest'][k]) < minimum for k, minimum in minima.items()):
                        missing.append('nonempty_sensor_data:'+kind)
                except (KeyError, ValueError):
                    missing.append('invalid_summary:'+kind)
        return dict(ready=not missing, missing=missing, raw_only=raw_only,
                    expected_topics=self.topics, freshness_s=self.fresh_s,
                    clouds=self.clouds, summaries=self.summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=20.)
    parser.add_argument('--raw-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 60:
        parser.error('timeout must be finite and in (0,60] seconds')
    if args.output.exists():
        parser.error('output already exists; no readiness evidence overwrite')
    import yaml
    import rospy
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    cfg = yaml.safe_load(args.config.read_text())
    topics = cfg['input_topics']
    if len(topics) != 8 or len(cfg['sensor_frames']) != 8:
        parser.error('Phase-E expects the original eight ToF sensors')
    rospy.init_node('phase_e_perception_readiness', anonymous=True, disable_signals=True)
    start = time.monotonic()
    health = PerceptionReadiness(topics)
    lock = threading.Lock()

    def cloud_cb(msg, topic):
        # Organized no-hit pixels may all be NaN; they still carry real rays.
        names = {field.name for field in msg.fields}
        valid = (msg.width > 0 and msg.height > 1 and msg.point_step > 0 and
                 msg.row_step >= msg.width*msg.point_step and
                 len(msg.data) >= msg.row_step*msg.height and {'x','y','z'} <= names)
        with lock:
            health.cloud(topic, msg.header.stamp.to_nsec(), valid, time.monotonic())

    def summary_cb(msg, kind):
        with lock:
            health.summary(kind, msg.data, time.monotonic())

    subscribers = [rospy.Subscriber(t, PointCloud2, cloud_cb, callback_args=t,
                                   queue_size=1, buff_size=4*1024*1024) for t in topics]
    if not args.raw_only:
        subscribers += [rospy.Subscriber(cfg['summary_topic'], String, summary_cb,
                                        callback_args='fusion', queue_size=1),
                        rospy.Subscriber('/care_planner/confidence_map/e3_summary', String,
                                         summary_cb, callback_args='map', queue_size=1)]
    while True:
        now = time.monotonic()
        with lock:
            result = health.snapshot(now, args.raw_only)
        if result['ready'] or now-start >= args.timeout or rospy.is_shutdown():
            break
        time.sleep(.05)
    for subscriber in subscribers:
        subscriber.unregister()
    result.update(protocol='real_tof_readiness_v1', elapsed_wall_s=time.monotonic()-start,
                  timeout_s=args.timeout, config=str(args.config.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:result[k] for k in ('ready','missing','elapsed_wall_s','raw_only')}))
    return 0 if result['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
