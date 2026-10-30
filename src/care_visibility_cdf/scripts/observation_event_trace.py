"""Diagnostic-only identities; never an execution or visibility certificate."""
import hashlib
import json
import threading
import time
from pathlib import Path

_WRITE_LOCK = threading.Lock()


def vbc_bundle_identity(msg, expected_seq):
    """Optional metadata; old numeric bundles still work, without guessed IDs."""
    try:
        prefix, stamp, seq = msg.layout.dim[0].label.rsplit('_', 2)
        if prefix == 'care_vbc_v1' and int(stamp) > 0 and int(seq) == expected_seq:
            return dict(trajectory_stamp_ns=str(int(stamp)), bundle_seq=int(seq))
    except (AttributeError, IndexError, TypeError, ValueError):
        pass
    return None


def observation_region_token(ob):
    """Exact map-query identity, independent of the chosen steering pose."""
    payload = {"generation": ob.get("generation_event_id", "legacy"),
               "id": int(ob["id"]),
               "points": sorted(tuple(float(x) for x in p) for p in ob["points"])}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return "care_region_v1_" + str(ob["id"]) + "_" + digest


def observation_token(ob):
    # Distinguish geometry refreshes and q_vis replacements, even under the
    # same persistent obligation id. No rounding or nearest-time matching.
    payload = {"generation": ob.get("generation_event_id", "legacy"),
               "id": int(ob["id"]),
               "points": sorted(tuple(float(x) for x in p) for p in ob["points"]),
               "q_vis": [float(x) for x in ob["q_vis"]]}
    if "q_vis_joint_mask" in ob:
        payload["joint_mask"] = [
            float(x) for x in ob["q_vis_joint_mask"]]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return "care_obs_v1_" + str(ob["id"]) + "_" + digest


def append_event(output_root, stage, ros_s, **fields):
    """Callers catch I/O failures: trace loss must not change planner behavior."""
    event = dict(fields, schema_version=1, stage=stage, ros_s=float(ros_s),
                 wall_unix_s=time.time(), monotonic_s=time.monotonic())
    line = json.dumps(event, sort_keys=True, allow_nan=True) + "\n"
    with _WRITE_LOCK:
        with (Path(output_root) / "observation_events.jsonl").open("a") as stream:
            stream.write(line)
    return event
