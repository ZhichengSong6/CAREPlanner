#!/usr/bin/env python3
"""Bind a partial-observation task return to its actual published target.

The caller owns the regime lock. This ledger grants no motion, completes no
obligation and changes no planning or execution budget.
"""
import math
import re
import json
from collections import deque

TOKEN = re.compile(r'care_obs_v1_(\d+)_[0-9a-f]{24}')


class ObservationReleaseIdentity:
    def __init__(self, max_age=.5, capacity=16, allow_completed_gain=False):
        self.max_age=float(max_age);self.capacity=int(capacity);self.allow_completed_gain=allow_completed_gain
        self.high_water=0;self.pending={};self.last_reason='startup'
        self.observed_gains=0;self.unbound_gains=0;self.expired_gains=0;self.retired_gains=0;self.released_gains=0
        self.seen_events=deque(maxlen=256)

    def observe(self, fields, now):
        try:
            count=int(fields.get('acquired_point_count','0'))
            if count<0:raise ValueError('negative count')
        except (TypeError,ValueError):
            self.last_reason='invalid_counter';return False
        self.high_water=max(self.high_water,count)
        if fields.get('complete')=='1' and not self.allow_completed_gain:
            self.last_reason='full_completion_uses_existing_path';return False
        try:
            events=json.loads(fields.get('point_gain_events','[]'))
            if not isinstance(events,list) or len(events)>64:raise ValueError('invalid events')
        except (TypeError,ValueError):
            self.last_reason='invalid_point_gain_events';return False
        accepted=False
        for event in events:
            if not isinstance(event,dict):continue
            event_id=event.get('event_id')
            if not isinstance(event_id,str) or not event_id or event_id in self.seen_events:continue
            self.seen_events.append(event_id);self.observed_gains+=1
            token=event.get('observation_token','')
            try:
                query=float(event['query_ros_s']);identity=TOKEN.fullmatch(token)
                region=event.get('region_token','')
                points=event.get('points',[])
                valid=(identity is not None and identity.group(1)==str(event.get('obligation_id')) and
                    re.fullmatch(r'care_region_v1_'+identity.group(1)+r'_[0-9a-f]{24}',region) is not None and
                    isinstance(points,list) and 0<len(points)<=1024 and
                    all(len(p)==3 and all(math.isfinite(float(x)) for x in p) for p in points) and
                    math.isfinite(now) and math.isfinite(query) and 0<=now-query<=self.max_age)
            except (KeyError,TypeError,ValueError):valid=False
            if not valid:
                self.unbound_gains+=1;self.last_reason='gain_not_bound_to_fresh_target';continue
            self.pending[token]=dict(token=token,query_ros_s=query,point_gain_event=event,
                acquisition_snapshot_id=fields.get('acquisition_snapshot_id','none'),counter=count)
            while len(self.pending)>self.capacity:del self.pending[next(iter(self.pending))]
            accepted=True
        if accepted:self.last_reason='gain_pending_target_receipt'
        elif not events:self.last_reason='no_point_gain_event'
        return accepted

    def take(self, waypoint_token, retired, now):
        for token,event in list(self.pending.items()):
            if token in retired or not 0<=now-event['query_ros_s']<=self.max_age:
                if token in retired:self.retired_gains+=1
                else:self.expired_gains+=1
                del self.pending[token]
        event=self.pending.pop(waypoint_token,None)
        if event:self.released_gains+=1
        self.last_reason='matched_partial_gain' if event else 'no_fresh_gain_for_target'
        return event


class ObservationHandoffIdentity:
    """Preserve the owned observation identity when scheduling clears a pose."""
    def __init__(self): self.target=None
    def clear(self): self.target=None
    def bind(self, token, region, scope):
        target=TOKEN.fullmatch(token)
        if target is None or re.fullmatch('care_region_v1_'+target.group(1)+r'_[0-9a-f]{24}',region) is None:return False
        self.target=(token,region,scope);return True
    def take(self, ledger, scope, retired, now):
        if self.target is None:return None
        token,region,owned_scope=self.target
        if owned_scope!=scope or token in retired:
            self.clear();return None
        event=ledger.take(token,retired,now)
        if event is None:return None
        if event['point_gain_event']['region_token']!=region:return None
        self.clear();return event
