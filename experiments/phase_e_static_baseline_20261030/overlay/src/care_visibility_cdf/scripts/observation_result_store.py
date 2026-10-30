"""Task-scoped observation proposals; never map evidence or motion grants."""
import copy
from collections import OrderedDict
import numpy as np


def point_set(points):
    return frozenset(tuple(float(x) for x in p) for p in np.asarray(points).reshape(-1,3))


class ObservationResultStore:
    def __init__(self,capacity=64):
        self.capacity=capacity
        self.revision=0
        self.records=OrderedDict()

    def clear(self,revision):
        self.revision=revision
        self.records.clear()

    def save(self,ob,revision,reason):
        if revision!=self.revision:return False
        points=np.asarray(ob.get('points',[]),dtype=float).reshape(-1,3)
        q=np.asarray(ob.get('q_vis',[]),dtype=float)
        if not len(points) or q.shape!=(7,) or not np.isfinite(points).all() or not np.isfinite(q).all():return False
        key=point_set(points)
        self.records[key]=dict(obligation=copy.deepcopy(ob),reason=reason)
        self.records.move_to_end(key)
        while len(self.records)>self.capacity:self.records.popitem(last=False)
        return True

    def candidates(self,region,revision,distance):
        if revision!=self.revision:return []
        requested=point_set(region['points']);centroid=np.mean(region['points'],axis=0)
        ranked=[]
        for key,record in self.records.items():
            ob=record['obligation']
            shift=float(np.linalg.norm(np.mean(ob['points'],axis=0)-centroid))
            overlap=len(requested&key)
            if not overlap and shift>distance:continue
            score=(requested.issubset(key),overlap/max(1,len(requested|key)),-shift)
            ranked.append((score,record))
        return [copy.deepcopy(record) for _,record in sorted(ranked,key=lambda x:x[0],reverse=True)]
