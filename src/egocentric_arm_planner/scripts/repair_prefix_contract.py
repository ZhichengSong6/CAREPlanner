#!/usr/bin/env python3
"""Selected construction intent; never a safety certificate."""
import json
import math
import re
from geometric_restore import validate as validate_geometric_restore

def unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate prefix contract key')
        result[key] = value
    return result

def parse(raw):
    if not isinstance(raw, str) or not raw or len(raw)>8192:
        raise ValueError('Missing/oversized prefix contract')
    c = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(c, dict): raise ValueError('Invalid prefix contract')
    if 'geometric_restore' in c: validate_geometric_restore(c['geometric_restore'])
    if (set(c)-{'geometric_restore'} != {
            'schema', 'session', 'revision', 'prefix_s', 'acceleration',
            'brake_dt_s', 'brake_steps'} or type(c['schema']) is not int or c['schema'] != 1 or
            not isinstance(c['session'], str) or
            re.fullmatch('[a-f0-9]{32}', c['session']) is None or
            type(c['revision']) is not int or not 0<c['revision']<2**64 or
            type(c['brake_steps']) is not int or not 0<c['brake_steps']<=20 or
            not isinstance(c['acceleration'], list) or len(c['acceleration']) != 7):
        raise ValueError('Invalid prefix contract identity')
    numbers = [c['prefix_s'], c['brake_dt_s']]+c['acceleration']
    if any(type(v) not in (float,int) or not math.isfinite(v) or v<=0 for v in numbers):
        raise ValueError('Invalid prefix construction parameters')
    return c

def construction(c):
    return {k:v for k,v in c.items() if k!='geometric_restore'}

def make(session, revision, prefix, acceleration, dt, steps):
    raw = json.dumps(dict(schema=1,session=session,revision=revision,prefix_s=prefix,
        acceleration=list(acceleration),brake_dt_s=dt,brake_steps=steps),
        separators=(',', ':'),allow_nan=False)
    parse(raw)
    return raw
