"""Read the separate wall-clock guard without introducing another ROS wait."""
from pathlib import Path
import json,os,time

def read():
    prefix=os.environ.get('CARE_SCENE_GUARD_PREFIX')
    if not prefix:return dict(ready=True,disabled=True)
    trip=Path(prefix+'_trip.json')
    if trip.exists():return dict(ready=False,abort=json.loads(trip.read_text()))
    try:state=json.loads(Path(prefix+'_state.json').read_text())
    except FileNotFoundError:return dict(ready=False)
    if time.monotonic()-state['heartbeat_monotonic_s']>1:
        return dict(ready=False,abort=dict(reason='contact_guard_heartbeat_stale'))
    if state.get('tripped') or state.get('finished'):return dict(ready=False,abort=state)
    return state
