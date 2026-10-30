"""Passive scene evidence. Does not publish a goal, command, or map update."""
from pathlib import Path
import json,os,signal,time,threading
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import GetModelState

def main():
    output=Path(os.environ['CARE_STATIC_SCENE_TRACE']);assert not output.exists()
    scene=json.loads(Path(os.environ['CARE_STATIC_SCENE_JSON']).read_text())['configurations'][os.environ['CARE_STATIC_SCENE_ID']]
    box=scene['box'];center=np.array(box['center_m']) if box else None;half=np.array(box['size_m'])/2 if box else None
    rospy.init_node('phase_e_record_static_scene',disable_signals=True)
    lock=threading.Lock();stop=threading.Event();start=time.monotonic();last={};counts=dict(contact_messages=0,nonempty_contact_messages=0,ray_samples=0,box_surface_hits=0,map_samples=0)
    with output.open('x',buffering=1) as f:
        def emit(event,**fields):
            with lock:f.write(json.dumps(dict(event=event,ros_ns=rospy.Time.now().to_nsec(),monotonic_ns=time.monotonic_ns(),**fields))+'\n')
        def contacts(msg):
            counts['contact_messages']+=1
            if msg.states:
                counts['nonempty_contact_messages']+=1
                emit('contact',stamp_ns=msg.header.stamp.to_nsec(),states=[dict(collision1=s.collision1_name,collision2=s.collision2_name,depths=list(s.depths)) for s in msg.states])
            elif time.monotonic()-last.get('contact',0)>.5:
                last['contact']=time.monotonic();emit('contact_heartbeat',stamp_ns=msg.header.stamp.to_nsec(),nonempty=False)
        def select(x,tol):
            d=np.abs(x-center)-half;sd=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(np.max(d,axis=1),0)
            return np.abs(sd)<=tol
        def rays(msg):
            if time.monotonic()-last.get('ray',0)<1:return
            last['ray']=time.monotonic();counts['ray_samples']+=1
            if not box:return
            rows=np.array(list(pc2.read_points(msg,field_names=('x','y','z','sensor_id','hit'),skip_nans=True)))
            hit=rows[rows[:,4]>.5] if rows.size else np.empty((0,5));selected=hit[select(hit[:,:3],.004)] if len(hit) else np.empty((0,5))
            counts['box_surface_hits']+=len(selected);emit('sampled_rays',stamp_ns=msg.header.stamp.to_nsec(),box_surface_hits=len(selected),examples=selected[:8].tolist())
        def cloud(msg):
            if time.monotonic()-last.get('map',0)<1:return
            last['map']=time.monotonic();counts['map_samples']+=1
            if not box:return
            rows=np.array(list(pc2.read_points(msg,field_names=('x','y','z','confidence','occupancy'),skip_nans=True)))
            selected=rows[(rows[:,4]>.5)&select(rows[:,:3],.045)] if rows.size else np.empty((0,5))
            emit('sampled_map',stamp_ns=msg.header.stamp.to_nsec(),occupied_near_box=len(selected),examples=selected[:16].tolist())
        signal.signal(signal.SIGINT,lambda *a:stop.set());signal.signal(signal.SIGTERM,lambda *a:stop.set())
        emit('started',configuration=os.environ['CARE_STATIC_SCENE_ID'],box=box,world=scene['world'],world_sha256=scene['world_sha256'],sampling_wall_s=1)
        subs=[rospy.Subscriber('/care_planner/static_box/contacts',ContactsState,contacts,queue_size=100),
            rospy.Subscriber('/care_planner/perception/tof_ray_observations',PointCloud2,rays,queue_size=1,buff_size=2**23),
            rospy.Subscriber('/care_planner/confidence_map/points',PointCloud2,cloud,queue_size=1,buff_size=2**23)]
        if box:
            rospy.wait_for_service('/gazebo/get_model_state',timeout=5);r=rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)('care_static_box','world')
            emit('gazebo_box',success=r.success,position=[r.pose.position.x,r.pose.position.y,r.pose.position.z])
        while not stop.is_set() and not rospy.is_shutdown():stop.wait(.1)
        for s in subs:s.unregister()
        emit('finished',counts=counts,elapsed_wall_s=time.monotonic()-start)

if __name__=='__main__':main()
