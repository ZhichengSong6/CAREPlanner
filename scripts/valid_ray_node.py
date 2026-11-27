#!/usr/bin/env python3
"""ROS adapter for a NEW research run; historical topics/files stay unchanged.

Input is the existing post-arm ray stream. Output origins move to the optical
near plane; every retained endpoint is an actual finite return. A JSONL record
links each output byte digest to its input and source-stamped sensor TF.
"""
import copy
import hashlib
import json
import threading
import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2
from ray_evidence import array

FRAMES=['link2_sensor1_tof_link','link2_sensor2_tof_link',
        'link3_sensor1_tof_link','link3_sensor2_tof_link',
        'link4_sensor1_tof_link','link4_sensor2_tof_link',
        'EE_sensor1_tof_link','EE_sensor2_tof_link']

def forward_axis(q):
    return np.array([2*(q.x*q.z+q.w*q.y),2*(q.y*q.z-q.w*q.x),1-2*(q.x*q.x+q.y*q.y)])

def convert(msg, axes):
    a=array(msg)
    origin=np.column_stack([a['origin_'+s] for s in 'xyz'])
    endpoint=np.column_stack([a[s] for s in 'xyz'])
    # axes are resolved for each individual source timestamp, not fused time.
    delta=endpoint-origin;length=np.linalg.norm(delta,axis=1)
    depth=np.einsum('ij,ij->i',delta,axes)
    valid=((a['hit']==1)|(a['hit']==2))&(a['sensor_id']>=0)&(a['sensor_id']<8)
    valid &= np.isfinite(origin).all(axis=1)&np.isfinite(endpoint).all(axis=1)
    valid &= (depth>=.15)&(depth<=.75)&(length>0)
    result=a[valid].copy();shifted=origin[valid]+delta[valid]*(.15/depth[valid])[:,None]
    self_return=result['hit']==2
    # Stop25mm before the measured body surface; never create self OCCUPIED.
    clipped=endpoint[valid]-delta[valid]/length[valid,None]*.025
    for j,s in enumerate('xyz'):result[s][self_return]=clipped[self_return,j]
    result['hit'][self_return]=0
    keep=np.linalg.norm(np.column_stack([result[s] for s in 'xyz'])-shifted,axis=1)>=.05
    # Reject segments where the clipping guard crossed the near plane.
    keep &= length[valid]-.025 > .15*length[valid]/depth[valid]
    result=result[keep];shifted=shifted[keep]
    for j,s in enumerate('xyz'):result['origin_'+s]=shifted[:,j]
    output=copy.deepcopy(msg);output.height=1;output.width=len(result)
    output.row_step=output.width*output.point_step;output.data=result.tobytes()
    return output

def main():
    rospy.init_node('valid_interval_ray_adapter')
    trace=open(rospy.get_param('~trace'),'x',buffering=1); lock=threading.Lock()
    buffer=tf2_ros.Buffer(cache_time=rospy.Duration(5)); listener=tf2_ros.TransformListener(buffer)
    pub=rospy.Publisher('/care_planner/perception/valid_interval_tof_rays',PointCloud2,queue_size=1)
    last_published_ns = 0

    def callback(msg):
        nonlocal last_published_ns
        with lock:
            stamp = msg.header.stamp.to_nsec()
            entry = dict(input_stamp_ns=stamp, input_sha256=hashlib.sha256(msg.data).hexdigest())
            if not last_published_ns <= stamp <= rospy.Time.now().to_nsec() or not stamp:
                entry.update(status='rejected', reason='nonmonotone_or_invalid_ray_stamp',
                             previous_published_ray_stamp_ns=last_published_ns)
                trace.write(json.dumps(entry)+'\n')
                rospy.logwarn('[TOF_RAY_TIME] drop packet stamp=%d previous=%d', stamp, last_published_ns)
                return
            try:
                a=array(msg);axes=np.zeros((len(a),3));cache={}
                assert msg.header.frame_id=='base_link'
                assert {'source_stamp_sec','source_stamp_nsec'}<=set(a.dtype.names)
                for i in np.flatnonzero(a['hit']!=0):
                    sensor=int(a['sensor_id'][i]);sec=int(a['source_stamp_sec'][i]);nsec=int(a['source_stamp_nsec'][i])
                    assert 0<=sensor<8 and 0<=nsec<1000000000 and (sec or nsec)
                    key=(sensor,sec,nsec)
                    if key not in cache:
                        transform=buffer.lookup_transform('base_link',FRAMES[sensor],rospy.Time(sec,nsec),rospy.Duration(.03))
                        cache[key]=forward_axis(transform.transform.rotation)
                    axes[i]=cache[key]
                output=convert(msg,axes)
                entry.update(status='published',input_points=len(a),output_points=output.width,
                    source_transforms=[dict(sensor=k[0],sec=k[1],nsec=k[2],forward_axis=v.tolist()) for k,v in cache.items()],
                    finite_self_returns=int(np.sum(a['hit']==2)), output_sha256=hashlib.sha256(output.data).hexdigest())
                pub.publish(output)
                last_published_ns = stamp
            except Exception as e:
                entry.update(status='rejected',reason=type(e).__name__+': '+str(e))
            trace.write(json.dumps(entry)+'\n')
    rospy.Subscriber('/care_planner/perception/task_tof_ray_observations',PointCloud2,callback,queue_size=1,buff_size=2**23)
    try:rospy.spin()
    finally:trace.close()

if __name__=='__main__':main()
