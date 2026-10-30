"""Bounded discovery work; entries never authorize motion or retire obligations."""
from collections import OrderedDict


class PendingVbcBlockers:
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.records = OrderedDict()
        self.enqueued = self.coalesced = self.materialized = 0
        self.overflow_count = 0
        # Losing responsibility at capacity must not report acquisition complete.
        # This fault is explicit and remains latched for this node lifetime.
        self.overflow = False

    @property
    def pending(self):
        return bool(self.records) or self.overflow

    def add(self, record):
        identity = record['identity'] or {}
        key = (record['source'], identity.get('trajectory_stamp_ns'),
               tuple(tuple(float(x) for x in p) for p in record['points']))
        if key in self.records:
            self.coalesced += 1
            return 'coalesced'
        if len(self.records) >= self.capacity:
            self.overflow = True
            self.overflow_count += 1
            return 'capacity_hold'
        self.records[key] = record
        self.enqueued += 1
        return 'enqueued'

    def peek(self):
        return next(iter(self.records.values()), None)

    def acknowledge(self, generation):
        for key, record in self.records.items():
            if record['generation'] == generation:
                del self.records[key]
                self.materialized += 1
                return True
        return False

    def contains(self, audit_stamp, points):
        requested = {tuple(float(x) for x in p) for p in points}
        return any(record['source'] == 'candidate' and
                   (record['identity'] or {}).get('trajectory_stamp_ns') == str(audit_stamp) and
                   requested.intersection(tuple(float(x) for x in p) for p in record['points'])
                   for record in self.records.values())
