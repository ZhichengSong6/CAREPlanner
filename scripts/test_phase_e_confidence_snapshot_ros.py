#!/usr/bin/env python3
"""Real confidence-map services with deterministic clock/rays and version invalidation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import traceback
import xmlrpc.client
from phase_e_owned_processes import identity, remember_descendants, reap_owned

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    out=ap.parse_args().output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
    os.environ.update(ROS_MASTER_URI='http://127.0.0.1:'+str(port),ROS_HOSTNAME='localhost',
                      ROS_HOME=str(out/'ros_home'),ROS_LOG_DIR=str(out/'ros_logs'))
    children=[]; logs=[]; owned={}; done=threading.Event(); clock_thread=None
    result={'status':'RUNNING','checks':[],'responses':[]}
    paths=[ROOT/'src/care_confidence_map/src/confidence_map_node.cpp',
           ROOT/'src/care_confidence_map/srv/QueryConfidence.srv',
           ROOT/'src/care_confidence_map/srv/QueryConfidenceSnapshot.srv',
           ROOT/'devel/lib/care_confidence_map/confidence_map_node',Path(__file__).resolve()]
    result['sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    def start(cmd,label):
        f=(out/(label+'.log')).open('w'); logs.append(f)
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p); owned[p.pid]=identity(p.pid); return p
    def wait(fn,label,timeout=15):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            remember_descendants({p.pid for p in children if p.poll() is None},owned)
            if fn(): return
            time.sleep(.02)
        raise AssertionError('timeout: '+label)
    def check(label,condition):
        assert condition,label
        result['checks'].append(label)
    def stop(p):
        if p.poll() is None:
            os.kill(p.pid,signal.SIGINT); p.wait(timeout=8)
    try:
        start(['roscore','-p',str(port)],'master')
        def ready():
            try: return xmlrpc.client.ServerProxy(os.environ['ROS_MASTER_URI']).getPid('/snapshot_test')[0]==1
            except OSError: return False
        wait(ready,'master')
        import rospy
        import yaml
        from geometry_msgs.msg import Point
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2,PointField
        from sensor_msgs.point_cloud2 import create_cloud
        from std_msgs.msg import Header
        from std_srvs.srv import Trigger
        from care_confidence_map.srv import QueryConfidence,QueryConfidenceSnapshot,QueryConfidenceSnapshotRequest
        rospy.set_param('/use_sim_time',True)
        rospy.init_node('snapshot_test',anonymous=True,disable_signals=True)
        clock_pub=rospy.Publisher('/clock',Clock,queue_size=1,latch=True)
        sim=[10.]
        def send_clock():
            while not done.is_set():
                clock_pub.publish(Clock(clock=rospy.Time.from_sec(sim[0]))); time.sleep(.01)
        clock_thread=threading.Thread(target=send_clock,daemon=True); clock_thread.start()
        wait(lambda:rospy.Time.now().to_sec()==10.,'clock')
        urdf=out/'prior.urdf'
        urdf.write_text('<robot name="snapshot_prior"><link name="base_link"><collision><origin xyz="-.5 .5 .3"/><geometry><box size=".1 .1 .1"/></geometry></collision></link></robot>')
        config=yaml.safe_load((ROOT/'src/care_confidence_map/config/confidence_map_phase_e_ray.yaml').read_text())
        config['confidence_map'].update(query_service='/snapshot/query',ray_observation_topic='/snapshot/rays',
            update_rate=.1,publish_rate=.1,temporal_decay_time=2.,
            test_known_free={'enabled':True,'points':[[-.5,-.5,.3]],'inflation_radius':0.})
        config['sensor_model']['sensors']=[dict(id=0,name='test',frame='base_link')]
        config['current_body_prior'].update(enabled=True,geometry_backend='primitive',
            primitive_urdf_file=str(urdf),link_inflation_radius={},inflation_radius=0.,
            refresh_on_startup=False,refresh_service='/snapshot/refresh',deactivate_service='/snapshot/deactivate')
        rospy.set_param('/snapshot_map',config)
        points=[Point(.15,0.,.3),Point(.4,0.,.3),Point(.7,.7,.3),Point(-.5,-.5,.3),Point(-.5,.5,.3),
                Point(2.,0.,.3),Point(float('nan'),0.,.3),Point(1e300,0.,.3),Point(.951,0.,.3)]
        pub=rospy.Publisher('/snapshot/rays',PointCloud2,queue_size=1)
        fields=[PointField(n,i*4,PointField.FLOAT32,1) for i,n in enumerate(['x','y','z','origin_x','origin_y','origin_z'])]
        fields += [PointField('sensor_id',24,PointField.INT32,1),PointField('hit',28,PointField.UINT8,1)]
        def ray(stamp,hit=1):
            pub.publish(create_cloud(Header(stamp=rospy.Time.from_sec(stamp),frame_id='base_link'),fields,
                        [(.4,0.,.3,0.,0.,.3,0,hit)]))
        def launch(label):
            proc=start([str(ROOT/'devel/lib/care_confidence_map/confidence_map_node'),'__name:=snapshot_map'],label)
            rospy.wait_for_service('/snapshot/query_snapshot',timeout=15)
            wait(lambda:pub.get_num_connections()>0,'ray subscriber')
            return proc
        proc=launch('map_initial')
        query=rospy.ServiceProxy('/snapshot/query_snapshot',QueryConfidenceSnapshot)
        legacy=rospy.ServiceProxy('/snapshot/query',QueryConfidence)
        result['legacy_md5']=QueryConfidence._md5sum
        def snap(label='',expected=None,horizon=.2,threshold=.8):
            req=QueryConfidenceSnapshotRequest(points=points,horizon_s=horizon,minimum_confidence=threshold)
            if expected:
                req.expected_instance=expected.instance; req.expected_epoch=expected.epoch; req.expected_revision=expected.revision
            r=query(req)
            if label:
                result['responses'].append(dict(label=label,valid=r.valid,reason=r.reason,instance=r.instance,
                    epoch=r.epoch,revision=r.revision,t=r.snapshot_stamp.to_sec(),update=r.map_update_stamp.to_sec(),
                    confidence=list(r.confidence_now),future=list(r.confidence_at_horizon),occupancy=list(r.occupancy),
                    provenance=list(r.provenance),free=list(r.model_free_at_horizon),inside=list(r.inside_map)))
            check_lease=not r.future_free_space_guaranteed
            assert check_lease
            return r
        initial=snap('initial')
        check('grid geometry available',initial.resolution==.05 and initial.map_min.x==-.95 and
              initial.map_max.x==.95 and all(n>0 for n in initial.grid_size))
        check('unknown stays unknown',initial.valid and initial.confidence_now[0]==0 and initial.model_free_at_horizon[0]==0)
        check('test oracle provenance',initial.provenance[3]==4 and initial.model_free_at_horizon[3]==1)
        check('nonfinite huge and outside configured bounds rejected',list(initial.inside_map)[5:]==[0,0,0,0])
        check('same version recheck',snap(expected=initial).valid)
        check('invalid horizon rejected',not snap(horizon=float('nan')).valid and not snap(horizon=-1).valid)
        check('invalid confidence threshold rejected',not snap(threshold=0).valid and not snap(threshold=1.1).valid)
        old=legacy(points[:5]); check('legacy query shape unchanged',len(old.confidence)==5 and list(old.inside_map)==[1]*5)
        check('refresh prior service',rospy.ServiceProxy('/snapshot/refresh',Trigger)().success)
        prior=snap('prior')
        check('bootstrap separate from sensor evidence',prior.provenance[4]==2 and prior.confidence_at_horizon[4]==1 and prior.last_seen_s[4]==-1)
        check('refresh invalidates old version',not snap(expected=initial).valid)
        ray(10.)
        wait(lambda:snap().provenance[0]==1,'ray integrated')
        observed=snap('ray')
        check('ray free and occupied distinguished',observed.model_free_at_horizon[0]==1 and observed.confidence_now[1]==1 and
              observed.occupancy[1]==1 and observed.model_free_at_horizon[1]==0)
        check('future decay computed',abs(observed.confidence_at_horizon[0]-math.exp(-.1))<1e-6)
        check('ray write invalidates version',not snap(expected=prior).valid)
        legacy_at_ray=legacy(points[:2]); check('legacy values preserved',list(legacy_at_ray.confidence)==[1.,1.])
        sim[0]=11.; wait(lambda:rospy.Time.now().to_sec()==11.,'clock advance'); time.sleep(.05)
        decayed=snap('query_time_decay',expected=observed)
        check('query decay without timer or revision change',decayed.valid and decayed.revision==observed.revision and
              abs(decayed.confidence_now[0]-math.exp(-.5))<1e-6 and decayed.model_free_at_horizon[0]==0)
        check('snapshot time advances despite same map version',decayed.snapshot_stamp>observed.snapshot_stamp)
        check('cached legacy visibility does not freeze projection',legacy(points[:1]).confidence[0]==1.)
        check('deactivate prior service',rospy.ServiceProxy('/snapshot/deactivate',Trigger)().success)
        cleared=snap('prior_removed')
        check('deactivate removes only bootstrap',cleared.confidence_now[4]==0 and cleared.provenance[3]==4 and cleared.provenance[0]==1)
        check('deactivate invalidates version',not snap(expected=decayed).valid)
        ray(11.,hit=0)
        wait(lambda:snap().last_seen_s[1]==11.,'new ray')
        persistent=snap('persistent_occupied')
        check('persistent occupancy wins over fresh free ray',persistent.occupancy[1]==1 and persistent.model_free_at_horizon[1]==0)
        # Advance past the periodic update to verify timer mutations version.
        sim[0]=21.; wait(lambda:rospy.Time.now().to_sec()==21.,'timer clock')
        wait(lambda:snap().revision>persistent.revision,'timer revision')
        timer=snap('timer_decay')
        check('timer write advances revision',timer.map_update_stamp.to_sec()==21.)
        sim[0]=20.5; wait(lambda:rospy.Time.now().to_sec()==20.5,'clock rollback')
        rollback=snap('rollback')
        check('clock rollback invalidates epoch',not rollback.valid and rollback.clock_fault and rollback.epoch>timer.epoch)
        sim[0]=22.; wait(lambda:rospy.Time.now().to_sec()==22.,'clock catchup')
        check('clock fault sticky after catchup',not snap().valid)
        stop(proc)
        proc=launch('map_restart')
        restarted=snap('restarted')
        check('restart changes instance and resets fault',restarted.valid and restarted.instance!=initial.instance and restarted.epoch==0)
        check('old instance rejected',not snap(expected=initial).valid)
        ray(23.); wait(lambda:not snap().valid,'future observation invalidation')
        check('future observation rejected as evidence',snap('future_ray').reason=='invalid_evidence_time_restart_required')
        stop(proc); proc=launch('map_reordered')
        ray(22.); wait(lambda:snap().provenance[0]==1,'first ordered ray')
        ray(21.); wait(lambda:not snap().valid,'reordered observation invalidation')
        check('out of order observation invalidates evidence',not snap('reordered_ray').valid)
        stop(proc)
        config['confidence_map']['observation_mode']='ideal_fov'
        rospy.set_param('/snapshot_map',config)
        # ideal_fov intentionally has no ray subscriber; use a separate launch.
        proc=start([str(ROOT/'devel/lib/care_confidence_map/confidence_map_node'),'__name:=snapshot_map'],'map_ideal')
        rospy.wait_for_service('/snapshot/query_snapshot',timeout=15)
        sim[0]=33.; wait(lambda:rospy.Time.now().to_sec()==33.,'ideal clock')
        wait(lambda:snap().provenance[0]==1,'ideal sensor update')
        ideal=snap('ideal_fov')
        check('ideal visibility explicitly labelled',ideal.valid and ideal.observation_mode=='ideal_fov' and
              ideal.confidence_now[0]==1. and not ideal.future_free_space_guaranteed)
        result['status']='PASS'
    except BaseException:
        result['status']='FAIL'; result['error']=traceback.format_exc()
    finally:
        done.set()
        if clock_thread: clock_thread.join(2)
        remember_descendants({p.pid for p in children if p.poll() is None},owned)
        result['cleanup']=reap_owned(owned)
        for p in children:
            try: p.wait(timeout=2)
            except subprocess.TimeoutExpired: pass
        for f in logs: f.close()
        result['input_hashes_unchanged']=all(hashlib.sha256(p.read_bytes()).hexdigest()==result['sha256'][str(p.relative_to(ROOT))] for p in paths)
        if result['cleanup']['remaining'] or not result['input_hashes_unchanged']: result['status']='FAIL'
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('responses','sha256')},indent=2))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
