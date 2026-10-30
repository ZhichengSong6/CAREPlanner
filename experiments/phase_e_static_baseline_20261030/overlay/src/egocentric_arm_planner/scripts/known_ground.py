"""Bounded exclusion of a complete cubic joint curve from the existing floor."""
import json
import math
import os
import time
import numpy as np
from urdf_parser_py.urdf import URDF
from compute_ee_workspace_bounds import find_chain_joints, get_joint_origin_transform as joint_origin_transform, axis_angle_to_rot


class KnownGround:
    def __init__(self, urdf, geometry, optimized=True, backend='python'):
        self.optimized=optimized
        if backend not in ('python','native'):raise ValueError('invalid_ground_backend')
        self.backend=backend;self._native=None;self._native_fallbacks=0
        from ground_support_policy import load_ground_policy
        self.contract,self.body_floors=load_ground_policy(urdf,geometry)
        plane=self.contract.get('plane',dict(normal=[0.,0.,1.],offset_m=-self.contract['plane_z_m']))
        self.plane_normal=np.asarray(plane['normal'],dtype=float)
        self.plane_offset=float(plane['offset_m'])
        self.robot=URDF.from_xml_file(urdf)
        self.names=['joint1','joint2','joint3','joint4','wrist_joint1','wrist_joint2','wrist_joint3']
        self.frames={j.child:(j.parent,joint_origin_transform(j),j.type,
            np.asarray(j.axis,dtype=float) if j.axis is not None else np.zeros(3),
            self.names.index(j.name) if j.name in self.names else -1) for j in self.robot.joints}
        self.rotations={}
        for name,(_,_,kind,axis,_) in self.frames.items():
            if kind in ('revolute','continuous'):
                unit=axis/np.linalg.norm(axis);x,y,z=unit
                self.rotations[name]=(np.outer(unit,unit),np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]]))
        self.identity3=np.eye(3)
        self.bodies=[]
        for body in self.contract['bodies']:
            chain=find_chain_joints(self.robot,'base_link',body['link'])
            corners=np.asarray(body['vertices'],dtype=float)
            if corners.ndim!=2 or corners.shape[1]!=3 or not 4<=len(corners)<=16384 or not np.isfinite(corners).all():
                raise ValueError('invalid_ground_vertices')
            offsets=[float(np.linalg.norm(joint_origin_transform(j)[:3,3])) for j in chain]
            radii=np.zeros(7)
            for i,j in enumerate(chain):
                if j.type in ('revolute','continuous'):
                    radii[self.names.index(j.name)]=sum(offsets[i+1:])+float(np.linalg.norm(corners,axis=1).max())
                elif j.type=='prismatic':radii[self.names.index(j.name)]=1.
            self.bodies.append((body['link'],chain,corners,radii))
        if backend=='native':
            from known_ground_native import NativeGround
            self._native=NativeGround(self.frames,self.bodies,self.rotations,self.contract['plane_z_m'],
                self.body_floors,self.contract['margin_m'],self.plane_normal,self.plane_offset)

    def clearance(self,q,radius=None):
        q=np.asarray(q,dtype=float);radius=np.zeros(7) if radius is None else radius
        if self._native is not None:
            value=self._native.clearance(q,radius)
            # Preserve the original evaluation at the numerical decision boundary.
            if abs(value[0]-self.contract['margin_m'])>1e-9:return value
            self._native_fallbacks+=1
        minimum=math.inf;link=None;cache={'base_link':np.eye(4)}
        def pose(name):
            if name not in cache:
                parent,origin,kind,axis,index=self.frames[name]
                tf=pose(parent)@origin
                if kind in ('revolute','continuous'):
                    move=np.eye(4)
                    if self.optimized:
                        outer,skew=self.rotations[name];c=math.cos(q[index]);s=math.sin(q[index])
                        move[:3,:3]=c*self.identity3+(1.-c)*outer+s*skew
                    else:move[:3,:3]=axis_angle_to_rot(axis,q[index])
                    tf=tf@move
                elif kind=='prismatic':
                    move=np.eye(4);move[:3,3]=axis*q[index];tf=tf@move
                cache[name]=tf
            return cache[name]
        for (name,chain,corners,radii),required in zip(self.bodies,self.body_floors):
            tf=pose(name)
            low=float(np.min(corners@(self.plane_normal@tf[:3,:3])+self.plane_normal@tf[:3,3])+self.plane_offset-radii@radius
                -required+self.contract['margin_m'])
            if low<minimum:minimum,link=low,name
        return minimum,link

    @staticmethod
    def interval_box(coeff,lo,hi):
        a,b,c,d=coeff;lower=np.empty(7);upper=np.empty(7)
        for j in range(7):
            points=[lo,hi]
            if abs(a[j])<1e-15:
                if abs(b[j])>1e-15:points.append(-c[j]/(2*b[j]))
            else:
                disc=4*b[j]*b[j]-12*a[j]*c[j]
                if disc>=0:
                    points.extend([(-2*b[j]+math.sqrt(disc))/(6*a[j]),(-2*b[j]-math.sqrt(disc))/(6*a[j])])
            values=[((a[j]*s+b[j])*s+c[j])*s+d[j] for s in points if lo<=s<=hi]
            lower[j],upper[j]=min(values),max(values)
        return (lower+upper)/2,(upper-lower)/2

    @staticmethod
    def roots(coeff):
        a,b,c,_=coeff;result=[]
        for j in range(7):
            if abs(a[j])<1e-15:
                result.append([-c[j]/(2*b[j])] if abs(b[j])>1e-15 else [])
            else:
                disc=4*b[j]*b[j]-12*a[j]*c[j]
                result.append([(-2*b[j]+math.sqrt(disc))/(6*a[j]),(-2*b[j]-math.sqrt(disc))/(6*a[j])] if disc>=0 else [])
        return result

    @staticmethod
    def box_with_roots(coeff,roots,lo,hi):
        a,b,c,d=coeff;lower=np.empty(7);upper=np.empty(7)
        for j in range(7):
            points=[lo,hi]+[x for x in roots[j] if lo<=x<=hi]
            values=[((a[j]*x+b[j])*x+c[j])*x+d[j] for x in points]
            lower[j],upper[j]=min(values),max(values)
        return (lower+upper)/2,(upper-lower)/2

    def check_arrays(self,q,v,t,max_fk=256,wall_budget_s=.03,deadline=None,time_ns=None):
        # Explicit research numerical allowance, not centimetres of extra reserve.
        # Knot and midpoint penetration checks retain the original threshold.
        numerical_tolerance = 1e-5 if os.environ.get('CARE_TASK_FIRST_RESEARCH') == 'true' else 0.
        started=time.monotonic();cpu_started=time.thread_time();limit=min(started+wall_budget_s,deadline) if deadline is not None else started+wall_budget_s
        used=0;actual_fk=0;hits=0;intervals=0;depth_seen=0;fk_wall=0.;interval_wall=0.;minimum=math.inf;cache={}
        last_interval=None;fk_cpu=0.
        native_start=self._native.calls if self._native is not None else 0;fallback_start=self._native_fallbacks
        def check_time():
            if time.monotonic()>=limit:raise TimeoutError('ground_wall_budget')
        def finish(valid,reason,**extra):
            if valid:check_time() # no over-budget success, including the last numerical operation
            return dict(valid=valid,reason=reason,fk=actual_fk,evaluations=used,cache_hits=hits,
                intervals=intervals,max_depth=depth_seen,elapsed_s=time.monotonic()-started,
                fk_wall_s=fk_wall,fk_cpu_s=fk_cpu,cpu_s=time.thread_time()-cpu_started,
                clearance_semantics='minimum per-body reference slack plus common reference margin',
                reference_margin_m=self.contract['margin_m'],tracking_tube_assumed_m=self.contract.get('assumed_tracking_tube_m',0.),
                interval_numerical_tolerance_m=numerical_tolerance,
                backend=self.backend,native_calls=(self._native.calls-native_start) if self._native is not None else 0,
                native_threshold_fallbacks=self._native_fallbacks-fallback_start,
                last_interval=last_interval,interval_wall_s=interval_wall,budget_s=max(0.,limit-started),**extra)
        def evaluate(point,radius=None):
            nonlocal used,actual_fk,hits,fk_wall,fk_cpu
            check_time()
            if used>=max_fk:raise TimeoutError('ground_fk_budget')
            used+=1
            key=(np.asarray(point).tobytes(),None if radius is None else np.asarray(radius).tobytes())
            if self.optimized and key in cache:hits+=1;return cache[key]
            tic=time.monotonic();cpu=time.thread_time();value=self.clearance(point,radius);fk_wall+=time.monotonic()-tic;fk_cpu+=time.thread_time()-cpu
            actual_fk+=1;check_time()
            if self.optimized:cache[key]=value
            return value
        try:
            check_time()
            q=np.asarray(q,dtype=float);v=np.asarray(v,dtype=float);t=np.asarray(t,dtype=float)
            if (q.ndim!=2 or q.shape[1]!=7 or v.shape!=q.shape or len(t)!=len(q) or len(q)<2 or
                    not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(t).all() or
                    t[0]!=0 or np.any(np.diff(t)<=0)):
                return finish(False,'malformed_ground_curve')
            from tracker_curve import curve_data
            bends,durations,canonical_time_ns=curve_data(q,v,t,time_ns)
            for point in q:
                height,link=evaluate(point)
                if height<self.contract['margin_m']-1e-10:
                    return finish(False,'ground_knot_margin',minimum_m=height,link=link)
            for k,dt in enumerate(durations):
                check_time()
                # Match trajectoryCurvature() and the actuator tracker:
                # quadratic velocity ramp when all joints are trapezoid
                # consistent, otherwise the complete segment is linear.
                bend=bends[k]
                a=np.zeros(7);b=-4.*bend;linear=q[k+1]-q[k]+4.*bend
                coeff=(a,b,linear,q[k]);roots=self.roots(coeff) if self.optimized else None
                stack=[(0.,1.,0)]
                while stack:
                    check_time();lo,hi,depth=stack.pop();intervals+=1;depth_seen=max(depth_seen,depth)
                    tic=time.monotonic()
                    center,radius=self.box_with_roots(coeff,roots,lo,hi) if roots is not None else self.interval_box(coeff,lo,hi)
                    last_interval=dict(segment=k,lo=lo,hi=hi,depth=depth,pending=len(stack),center=center.tolist(),radius=radius.tolist())
                    interval_wall+=time.monotonic()-tic
                    height,link=evaluate(center,radius)
                    interval_bound=height
                    if height>=self.contract['margin_m']:
                        minimum=min(minimum,height);continue
                    mid=(lo+hi)/2;actual=((a*mid+b)*mid+linear)*mid+q[k]
                    height,link=evaluate(actual)
                    if height<self.contract['margin_m']-1e-10:
                        return finish(False,'ground_curve_margin',minimum_m=height,link=link)
                    if numerical_tolerance and interval_bound>=self.contract['margin_m']-numerical_tolerance:
                        minimum=min(minimum,interval_bound);continue
                    if depth>=12:return finish(False,'ground_interval_unresolved')
                    stack.extend([(lo,mid,depth+1),(mid,hi,depth+1)])
            return finish(True,'ground_curve_excluded',minimum_m=minimum)
        except TimeoutError as exc:
            return finish(False,str(exc))
        except (ValueError,TypeError,IndexError):
            return finish(False,'malformed_ground_curve')

    def check(self,msg,deadline=None):
        entered=time.monotonic()
        deadline=min(entered+.03,deadline) if deadline is not None else entered+.03
        if list(msg.joint_names)!=self.names or msg.header.frame_id!='base_link':
            return dict(valid=False,reason='ground_curve_frame_or_order',fk=0)
        return self.check_arrays([p.positions for p in msg.points],[p.velocities for p in msg.points],
                                 [p.time_from_start.to_sec() for p in msg.points],deadline=deadline,
                                 time_ns=[p.time_from_start.to_nsec() for p in msg.points])
