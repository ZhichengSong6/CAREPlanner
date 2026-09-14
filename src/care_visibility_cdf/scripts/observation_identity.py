"""Stable runtime target identities, not visibility/execution certificates.

Kept separate from local diagnostic trace writers so the mainline has no
dependency on debug scripts or filesystem logging.
"""
import hashlib
import json


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
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return "care_obs_v1_" + str(ob["id"]) + "_" + digest
