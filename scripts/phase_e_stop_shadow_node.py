#!/usr/bin/env python3
"""Optional observer: JSONL evidence and diagnostics only, no control publishers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
# viscdf supplies Pinocchio; system ROS Python packages supply rospkg/genpy.
sys.path.append('/usr/lib/python3/dist-packages')
import rospy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from care_velocity_controller.msg import CommandStatus
from care_confidence_map.srv import QueryConfidenceSnapshot,QueryConfidenceSnapshotRequest
from phase_e_stop_shadow import EvidenceBuffer,PrimitiveEnclosure,ShadowAudit
from phase_e_stop_envelope import JOINTS

ROOT=Path(__file__).resolve().parents[1]
BASE='/care_arm/arm_group_velocity_controller'


def snapshot_dict(r):
    d={k:getattr(r,k) for k in ('valid','reason','instance','epoch','revision','frame','resolution',
        'observation_mode','clock_fault','future_free_space_guaranteed','horizon_s','temporal_decay_time')}
    d.update(snapshot_ns=r.snapshot_stamp.to_nsec(),map_update_ns=r.map_update_stamp.to_nsec(),
        map_min=[r.map_min.x,r.map_min.y,r.map_min.z],map_max=[r.map_max.x,r.map_max.y,r.map_max.z],grid_size=list(r.grid_size))
    for k in ('inside_map','confidence_now','confidence_at_horizon','occupancy','last_seen_s','provenance','model_free_at_horizon'):
        d[k]=list(getattr(r,k))
    return d


class BoundedQuery:
    """At most one daemon RPC. A late response is discarded, never reused."""
    def __init__(self,call,timeout):
        self.call=call; self.timeout=timeout; self.thread=None

    def __call__(self,*args):
        if self.thread and self.thread.is_alive(): raise RuntimeError('snapshot_service_busy_after_timeout')
        result=[]; done=threading.Event()
        def work():
            try: result.append((True,self.call(*args)))
            except Exception as exc: result.append((False,str(exc)))
            finally: done.set()
        self.thread=threading.Thread(target=work,daemon=True); self.thread.start()
        if not done.wait(self.timeout): raise RuntimeError('snapshot_service_timeout')
        ok,value=result[0]
        if not ok: raise RuntimeError('snapshot_service_error: '+value)
        return value


def clean(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean(v) for v in value]
    return value


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--state-topic',default=BASE+'/state')
    ap.add_argument('--command-topic',default=BASE+'/command_stamped')
    ap.add_argument('--snapshot-service',default='/care_planner/confidence_map/query_snapshot')
    ap.add_argument('--rate',type=float,default=5.)
    ap.add_argument('--query-timeout',type=float,default=.3)
    ap.add_argument('--max-state-age',type=float,default=.1)
    ap.add_argument('--assumed-update-gap',type=float,default=.001)
    args=ap.parse_args(rospy.myargv()[1:])
    for v in (args.rate,args.query_timeout,args.max_state_age,args.assumed_update_gap):
        if not math.isfinite(v) or v<=0: ap.error('timing values must be finite and positive')
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    rospy.init_node('phase_e_stop_shadow')
    config={k:rospy.get_param(BASE+'/'+k) for k in ('joints','velocity_limits','acceleration_limits','command_timeout','max_braking_step')}
    if config['joints']!=JOINTS: raise ValueError('controller joint order mismatch')
    urdf=ROOT/'src/arm_description/urdf/Arm_with_self_filter_collision.urdf'
    dependencies=[Path(__file__).resolve(),ROOT/'scripts/phase_e_stop_shadow.py',ROOT/'scripts/phase_e_stop_envelope.py',
        ROOT/'scripts/audit_case001_motion_prior.py',urdf,ROOT/'src/care_confidence_map/srv/QueryConfidenceSnapshot.srv',
        ROOT/'src/care_velocity_controller/msg/CommandStatus.msg']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    manifest=dict(input_sha256=hashes,controller_params=config,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        protocol='content_snapshot_then_recheck_v2',runtime_admission=False,control_publishers=[],scope='read-only shadow observer, no physical or environment qualification')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    buffer=EvidenceBuffer()
    def state(m):
        buffer.on_state(dict(t_ns=m.header.stamp.to_nsec(),stamp_ns=m.command_stamp.to_nsec(),sequence=m.command_sequence,
            mode=m.mode,clock_fault=m.clock_fault,integral_valid=m.integral_valid,max_update_gap=m.max_update_gap,
            q=list(m.measured_position),applied=list(m.applied_velocity),measured=list(m.measured_velocity),
            rearms=m.rearm_successes,rejected=m.rejected_commands))
    def command(m):
        buffer.on_command(dict(stamp_ns=m.header.stamp.to_nsec(),sequence=m.header.seq,names=list(m.name),velocity=list(m.velocity)))
    rospy.Subscriber(args.state_topic,CommandStatus,state,queue_size=200,tcp_nodelay=True)
    rospy.Subscriber(args.command_topic,JointState,command,queue_size=500,tcp_nodelay=True)
    pub=rospy.Publisher('/care_planner/stop_shadow/report',String,queue_size=10)
    def call(points,horizon,expected):
        req=QueryConfidenceSnapshotRequest(points=[Point(*p) for p in points],horizon_s=horizon,minimum_confidence=.8)
        if expected: req.expected_instance,req.expected_epoch,req.expected_revision=expected
        return snapshot_dict(rospy.ServiceProxy(args.snapshot_service,QueryConfidenceSnapshot)(req))
    query=BoundedQuery(call,args.query_timeout)
    audit=ShadowAudit(buffer,PrimitiveEnclosure(urdf),query,lambda:rospy.Time.now().to_nsec(),config,
                      args.max_state_age,args.assumed_update_gap)
    count=0; histogram={}
    try:
        with (out/'rows.jsonl').open('x') as stream:
            while not rospy.is_shutdown():
                start=time.monotonic()
                attempt_start=rospy.Time.now().to_nsec()
                try: row=audit.run()
                except Exception as exc:
                    row=dict(runtime_admission=False,binding_consistent=False,conditional_map_coverage=False,reasons=['observer_exception: '+repr(exc)])
                row['row_id']=count; row['wall_elapsed_s']=time.monotonic()-start
                row['attempt_start_ros_ns']=attempt_start
                row['attempt_end_ros_ns']=rospy.Time.now().to_nsec()
                for reason in row['reasons']: histogram[reason]=histogram.get(reason,0)+1
                stream.write(json.dumps(clean(row),allow_nan=False,separators=(',',':'))+'\n'); stream.flush()
                summary={k:row.get(k) for k in ('row_id','runtime_admission','binding_consistent','conditional_map_coverage','reasons','evidence_counts')}
                count+=1
                if row.get('state') and row.get('metadata',{}).get('valid') and not (out/'ready.json').exists():
                    ready=out/'ready.tmp'
                    ready.write_text(json.dumps(dict(ready=True,row_id=row['row_id'],protocol=manifest['protocol']))+'\n')
                    ready.replace(out/'ready.json')
                try: pub.publish(String(data=json.dumps(summary)))
                except rospy.ROSException:
                    if rospy.is_shutdown(): break
                    raise
                time.sleep(max(0.,1./args.rate-(time.monotonic()-start)))
    except (KeyboardInterrupt,rospy.ROSInterruptException): pass
    finally:
        manifest.update(rows=count,reason_counts=histogram,input_hashes_unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in dependencies))
        (out/'result.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()
