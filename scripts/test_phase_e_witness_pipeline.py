#!/usr/bin/env python3
"""Actual C++ selector + unchanged CUDA GCDF worker on a private ROS master.

Synthetic map/anchors are deliberate fixtures, never robot commands.
--verify-json uses the viscdf interpreter to independently check raw-q gradients.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import xmlrpc.client

REPO=Path(__file__).resolve().parents[1]
PYTHON='/home/zhicheng/miniconda3/envs/viscdf/bin/python'
CHECKPOINT=REPO/'src/care_collision_cdf/checkpoints/yiming_cdf/model_dict_signed.pt'


def verify(path):
    import torch
    sys.path.insert(0,str(REPO/'src/care_collision_cdf/scripts'))
    from collision_cdf_model import CollisionCDF
    net=CollisionCDF(str(CHECKPOINT),device='cuda')
    rows=json.loads(Path(path).read_text())
    max_d=max_g=max_fd=0.
    for row in rows:
        p=torch.tensor(row['point'],device='cuda',dtype=torch.float32).reshape(1,3)
        q=torch.tensor(row['q'],device='cuda',dtype=torch.float32).reshape(1,7).requires_grad_(True)
        f=net.model(torch.cat((p,q),dim=-1)).reshape(())
        g=torch.autograd.grad(f,q)[0].flatten()
        max_d=max(max_d,abs(f.item()-row['distance']))
        max_g=max(max_g,max(abs(a-b) for a,b in zip(g.tolist(),row['gradient'])))
        for j in range(7):
            plus=q.detach().clone(); minus=q.detach().clone(); plus[0,j]+=.001; minus[0,j]-=.001
            with torch.no_grad():
                fd=((net.model(torch.cat((p,plus),-1))-net.model(torch.cat((p,minus),-1)))/.002).item()
            max_fd=max(max_fd,abs(fd-g[j].item()))
            assert abs(fd-g[j].item())<.005+.02*abs(g[j].item()),(j,fd,g.tolist())
    assert max_d<1e-5 and max_g<2e-5,(max_d,max_g)
    print(json.dumps(dict(status='PASS',rows=len(rows),max_distance_error=max_d,
        max_gradient_error=max_g,max_finite_difference_error=max_fd)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir'); ap.add_argument('--verify-json')
    ap.add_argument('--geometry-backend',choices=['samples','primitive'],default='samples')
    args=ap.parse_args()
    if args.verify_json: return verify(args.verify_json)
    if not args.output_dir: ap.error('--output-dir required')
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=False)
    uri='http://127.0.0.1:11349'
    try: xmlrpc.client.ServerProxy(uri).getPid('/witness_probe')
    except OSError: pass
    else: raise RuntimeError('private ROS port already occupied')
    os.environ.update(ROS_MASTER_URI=uri,ROS_HOSTNAME='localhost',ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[]; handles=[]; refs=[]
    report=dict(status='FAIL',checks=[],checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
    assert report['checkpoint_sha256']=='41a4228eee797971d56b46849e0f52c663eae79dc35667e8704c1bc3c8d72c70'
    def start(argv,name):
        f=(out/name).open('x'); handles.append(f)
        p=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); children.append(p); return p
    def stop(p):
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGINT)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM); p.wait(timeout=5)
    def wait(pred,label,timeout=10):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if pred():return
            time.sleep(.01)
        raise AssertionError('timeout: '+label)
    tmp=tempfile.TemporaryDirectory(prefix='care_witness_')
    try:
        start(['roscore','-p','11349'],'roscore.log')
        def ready():
            try:return xmlrpc.client.ServerProxy(uri).getPid('/witness_probe')[0]==1
            except OSError:return False
        wait(ready,'roscore')
        import rospy
        from std_msgs.msg import Header
        from sensor_msgs.msg import PointCloud2,PointField
        from sensor_msgs import point_cloud2
        from care_collision_cdf.msg import CollisionCDFConstraintBatch as Batch,CollisionCDFWitnessRequest as Req,CollisionCDFWitnessResponse as Resp
        rospy.init_node('witness_pipeline_test',disable_signals=True)
        socket=str(Path(tmp.name)/'gpu.sock')
        worker=start([PYTHON,str(REPO/'src/care_collision_cdf/scripts/collision_cdf_gpu_worker.py'),
            '--checkpoint',str(CHECKPOINT),'--device','cuda','--socket',socket,'--warmup-pairs','8'],'worker.log')
        wait(lambda:Path(socket).exists(),'CUDA worker',timeout=30)
        point=[.1,.1,.3]
        base_q=[-.24,.75,-.3,-1.63,.19,-.18,-.38]
        def fields(names,integer=()):
            return [PointField(name=n,offset=i*4,datatype=PointField.INT32 if n in integer else PointField.FLOAT32,count=1) for i,n in enumerate(names)]
        def map_cloud(conf=0.,occ=0.,points=None):
            points=points if points is not None else [point]
            return point_cloud2.create_cloud(Header(stamp=rospy.Time.now(),frame_id='base_link'),
                fields(['x','y','z','confidence','current_visibility','occupancy']),[p+[conf,0.,occ] for p in points])
        def anchors(stamp,q=None,near=False,steps=(1,)):
            q=base_q if q is None else q
            center=point if near else [-.7,-.7,.8]
            if args.geometry_backend=='primitive':
                names=['primitive_schema','primitive_kind','eval_timestep','original_timestep','shape_count','shape_index','knot_count']
                fs=[PointField(name=n,offset=i*4,datatype=PointField.INT32,count=1) for i,n in enumerate(names)]
                fs += [PointField(name='q'+str(i),offset=28+i*4,datatype=PointField.FLOAT32,count=1) for i in range(7)]
                fs += [PointField(name='shape'+str(i),offset=56+i*8,datatype=PointField.FLOAT64,count=1) for i in range(18)]
                # A sphere primitive deliberately matches the old fixture;
                # production URDF boxes/cylinders are checked separately.
                shape=center+[1.,0.,0.,0.,1.,0.,0.,0.,1.]+[0.,0.,0.,.005,0.,0.]
                return point_cloud2.create_cloud(Header(stamp=stamp,frame_id='base_link'),fs,
                    [[1,2,i,k,1,0,len(steps)]+q+shape for i,k in enumerate(steps)])
            return point_cloud2.create_cloud(Header(stamp=stamp,frame_id='base_link'),
                fields(['x','y','z']+['q'+str(i) for i in range(7)]+['radius','eval_timestep','original_timestep'],('eval_timestep','original_timestep')),
                [center+q+[.005,k,k] for k in steps])
        def request(stamp,q=None,points=None,steps=None):
            points=points if points is not None else [point]; steps=steps if steps is not None else [1]*len(points)
            return Req(header=Header(stamp=stamp,frame_id='base_link'),plan_sequence=1,mode_epoch=2,target_revision=3,progress_epoch=4,dof=7,
                original_timestep=steps,point_flat=[x for p in points for x in p],q_flat=(base_q if q is None else q)*len(points))
        class Selector:
            def __init__(self,name,required=True,cap=250):
                prefix='/witness/'+name
                cfg=dict(anchor_topic=prefix+'/anchor',final_anchor_topic=prefix+'/final_anchor',execution_anchor_topic=prefix+'/execution_anchor',
                    map_topic=prefix+'/map',constraint_batch_topic=prefix+'/batch',final_constraint_batch_topic=prefix+'/final_batch',
                    execution_constraint_batch_topic=prefix+'/execution_batch',summary_topic=prefix+'/summary',
                    gpu_socket=socket,output_jsonl=str(out/(name+'.jsonl')),process_on_input_callback=True,
                    local_witness_requests_required=required,proximity_margin=.025,max_pairs_per_step=cap)
                cfg['geometry_backend']=args.geometry_backend
                rospy.set_param('/witness_'+name,cfg)
                self.responses=[]; self.batches=[]; self.final=[]
                self.subs=[rospy.Subscriber(prefix+'/batch/witness_response',Resp,self.responses.append),
                    rospy.Subscriber(prefix+'/batch',Batch,self.batches.append),rospy.Subscriber(prefix+'/final_batch',Batch,self.final.append)]
                self.map=rospy.Publisher(prefix+'/map',PointCloud2,queue_size=10)
                self.anchor=rospy.Publisher(prefix+'/anchor',PointCloud2,queue_size=10)
                self.req=rospy.Publisher(prefix+'/batch/witness_request',Req,queue_size=10)
                self.final_anchor=rospy.Publisher(prefix+'/final_anchor',PointCloud2,queue_size=10)
                self.proc=start([str(REPO/'devel/lib/care_collision_cdf/cpp_forbidden_voxel_gpu_shadow_node'),'__name:=witness_'+name],name+'.log')
                wait(lambda:all(p.get_num_connections() for p in (self.map,self.anchor,self.req,self.final_anchor)),name+' subscriptions')
            def run(self,req,cloud,conf=0.,occ=0.,points=None,anchor_first=False):
                self.map.publish(map_cloud(conf,occ,points)); time.sleep(.03)
                if anchor_first:
                    self.anchor.publish(cloud); time.sleep(.05)
                    assert not any(b.header.stamp==req.header.stamp for b in self.batches),'anchor escaped before request'
                    self.req.publish(req)
                else:self.req.publish(req); time.sleep(.02); self.anchor.publish(cloud)
                wait(lambda:any(r.request.header.stamp==req.header.stamp for r in self.responses),'query-scoped response')
                return next(r for r in self.responses if r.request.header.stamp==req.header.stamp)
        def remember(response):
            b=response.batch
            for i in range(b.num_pairs):
                refs.append(dict(point=list(b.point_flat[3*i:3*i+3]),q=list(b.q_linearization_flat[7*i:7*i+7]),
                    distance=b.distance[i],gradient=list(b.gradient_flat[7*i:7*i+7])))
        s=Selector('primary')
        stamp=rospy.Time.now(); r=s.run(request(stamp),anchors(stamp))
        assert list(r.status)==['evaluated_unknown'] and r.batch.num_pairs==1
        assert r.batch.approx_body_clearance_m[0]>.025
        remember(r); first=r
        report['checks'].append('far UNKNOWN outside unchanged proximity band explicitly re-evaluated by real CUDA worker')
        stamp=rospy.Time.now(); q=[v+.03*(i+1) for i,v in enumerate(base_q)]
        r=s.run(request(stamp,q=q),anchors(stamp,q=q),anchor_first=True)
        assert list(r.status)==['evaluated_unknown'] and r.batch.num_pairs==1
        assert max(abs(a-b) for a,b in zip(r.batch.q_linearization_flat,q))<1e-6
        assert abs(r.batch.distance[0]-first.batch.distance[0])>1e-5
        remember(r); report['checks'].append('anchor-first rendezvous; changed current 7D q recomputed, no stale distance reuse')
        stamp=rospy.Time.now(); r=s.run(request(stamp),anchors(stamp,near=True))
        assert r.batch.num_pairs==1; remember(r)
        report['checks'].append('natural and requested identical pair deduplicated')
        stamp=rospy.Time.now(); r=s.run(request(stamp,points=[point,point],steps=[1,2]),anchors(stamp,steps=(1,2)))
        assert r.batch.num_pairs==2 and sorted(r.batch.original_timestep)==[1,2]; remember(r)
        report['checks'].append('same voxel at two timesteps preserved independently')
        stamp=rospy.Time.now(); r=s.run(request(stamp),anchors(stamp),conf=1.,occ=1.)
        assert list(r.status)==['evaluated_occupied'] and list(r.batch.source_type)==[1]; remember(r)
        report['checks'].append('fresh OCCUPIED classification overrides high confidence')
        stamp=rospy.Time.now(); r=s.run(request(stamp),anchors(stamp),conf=1.,occ=0.)
        assert list(r.status)==['resolved_free'] and r.batch.num_pairs==0
        report['checks'].append('current high-confidence unoccupied map voxel explicitly resolved free')
        for name in ('off_grid','map_missing','wrong_q','wrong_step','bad_dimensions'):
            stamp=rospy.Time.now(); req=request(stamp); ac=anchors(stamp); pts=None
            if name=='off_grid':req.point_flat[0]+=.001
            if name=='map_missing':pts=[]
            if name=='wrong_q':req.q_flat[6]+=.1
            if name=='wrong_step':req.original_timestep=[2]
            if name=='bad_dimensions':req.q_flat=[]
            r=s.run(req,ac,points=pts)
            assert len(r.status)==1 and r.status[0].startswith('error_'),(name,r.status)
            assert not any(b.header.stamp==stamp for b in s.batches)
            report['checks'].append(name+': '+r.status[0]+', no partial ordinary batch')
        # Pending LOCAL rendezvous must not block FINAL certification.
        stamp=rospy.Time.now(); s.map.publish(map_cloud()); time.sleep(.03); s.anchor.publish(anchors(stamp))
        final_stamp=rospy.Time.now(); s.final_anchor.publish(anchors(final_stamp,near=True))
        wait(lambda:any(b.header.stamp==final_stamp for b in s.final),'FINAL independent of LOCAL request')
        assert not any(b.header.stamp==stamp for b in s.batches)
        s.req.publish(request(stamp)); wait(lambda:any(r.request.header.stamp==stamp for r in s.responses),'delayed LOCAL request')
        report['checks'].append('FINAL unaffected by LOCAL waiting for explicit request')
        stop(s.proc)
        s=Selector('budget',cap=1)
        stamp=rospy.Time.now(); second=[.15,.1,.3]
        r=s.run(request(stamp,points=[point,second]),anchors(stamp),points=[point,second])
        assert 'error_pair_budget_exceeded' in r.status and not s.batches
        report['checks'].append('isolated capacity fixture rejects overload without dropping natural constraints or raising budget')
        stop(s.proc)
        s=Selector('legacy',required=False)
        s.map.publish(map_cloud()); time.sleep(.03); stamp=rospy.Time.now(); s.anchor.publish(anchors(stamp,near=True))
        wait(lambda:any(b.header.stamp==stamp for b in s.batches),'legacy batch without request')
        assert s.batches[-1].num_pairs==1
        report['checks'].append('legacy selector interface works with no witness request')
        stop(s.proc)
        (out/'gpu_rows.json').write_text(json.dumps(refs,indent=2)+'\n')
        result=subprocess.run([PYTHON,str(Path(__file__).resolve()),'--verify-json',str(out/'gpu_rows.json')],capture_output=True,text=True)
        (out/'gradient_check.log').write_text(result.stdout+result.stderr)
        assert result.returncode==0,(result.stdout,result.stderr)
        report['gradient_check']=json.loads(result.stdout.strip().splitlines()[-1])
        report['checks'].append('all raw 7D gradients match independent autograd and per-joint finite differences')
        report['status']='PASS'
        print(json.dumps(report,indent=2))
    finally:
        for p in reversed(children):stop(p)
        for f in handles:f.close()
        tmp.cleanup()
        (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
