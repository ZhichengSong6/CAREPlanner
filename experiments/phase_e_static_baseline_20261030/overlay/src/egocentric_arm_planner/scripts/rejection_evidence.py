"""Passive rejection evidence. Never grants or changes a safety decision."""
import json,hashlib,os,time
def capture(candidate,batch,diagnostic,indices,minimum,malformed,threshold):
    path=os.environ.get('CARE_REJECTION_EVIDENCE_FILE')
    if not path:return
    curve=dict(frame=candidate.header.frame_id,stamp_ns=str(candidate.header.stamp.to_nsec()),
        sequence=int(candidate.header.seq),joint_names=list(candidate.joint_names),
        q=[list(p.positions) for p in candidate.points],v=[list(p.velocities) for p in candidate.points],
        t=[p.time_from_start.to_sec() for p in candidate.points])
    payload=dict(curve=curve,curve_sha256=hashlib.sha256(json.dumps(curve,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        batch_stamp_ns=str(batch.header.stamp.to_nsec()),batch_frame=batch.header.frame_id,
        num_pairs=int(batch.num_pairs),dof=int(batch.dof),original_timestep=list(batch.original_timestep),
        source_type=list(batch.source_type),point_flat=list(batch.point_flat),q_linearization_flat=list(batch.q_linearization_flat),
        distance=list(batch.distance),gradient_flat=list(batch.gradient_flat),approx_body_clearance_m=list(batch.approx_body_clearance_m),
        occupied_violations=list(indices),minimum_occupied_volume_m=minimum,malformed=bool(malformed),
        learned_threshold=threshold,diagnostic=diagnostic,monotonic_s=time.monotonic())
    with open(path,'a') as f:f.write(json.dumps(payload,separators=(',',':'))+'\n')
