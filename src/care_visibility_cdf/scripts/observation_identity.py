"""Stable runtime target identities, not visibility/execution certificates.

Kept separate from local diagnostic trace writers so the mainline has no
dependency on debug scripts or filesystem logging.
"""
import copy
import hashlib
import json


def observation_target_state(source):
    """Copy a pose and its objective mask as one target, including legacy absence.

    Callers retain their existing validity/identity checks. This is a snapshot,
    not a new target, observation completion, or execution authorization.
    """
    return {key: copy.deepcopy(source[key])
            for key in ('q_vis', 'q_vis_joint_mask', 'per_sensor_selected_sensor_id',
                        'per_sensor_hybrid_used') if key in source and
            (key != 'q_vis_joint_mask' or source.get('q_vis_joint_mask_explicit', True))}


def restore_observation_target_state(destination, source):
    """Restore the mask belonging to the pinned pose, never a different pose's mask."""
    target = observation_target_state(source)
    destination.update(target)
    if 'q_vis_joint_mask' not in target:
        destination.pop('q_vis_joint_mask', None)
    for key in ('per_sensor_selected_sensor_id', 'per_sensor_hybrid_used'):
        if key not in target:
            destination.pop(key, None)


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
    # Preserve historical token identity for legacy obligations. Newly
    # generated masked targets include the mask because it changes the
    # optimizer semantics even when all seven q values happen to match.
    if "q_vis_joint_mask" in ob:
        payload["joint_mask"] = [
            float(x) for x in ob["q_vis_joint_mask"]]
    sensor = int(ob.get('per_sensor_selected_sensor_id', -1))
    if sensor >= 0:
        payload['sensor_id'] = sensor
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return "care_obs_v1_" + str(ob["id"]) + "_" + digest
