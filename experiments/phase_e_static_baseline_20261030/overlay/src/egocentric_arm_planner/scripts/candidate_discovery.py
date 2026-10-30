"""Publish observation work for terminal rejected NORMAL candidates only.

This never changes a map value, safety verdict, or execution authorization.
Unowned/bootstrap and REPAIR/PROBE discoveries pass through unchanged.
"""
from collections import OrderedDict

class CandidateDiscovery:
    def __init__(self, capacity=512):
        self.capacity=capacity
        self.audits=OrderedDict()
        self.owners={}
        self.retired_floor=0
        self.ready=[]
        self.counts=dict(held=0,released=0,suppressed=0,unowned=0)

    def register(self, key, stamp):
        if stamp in self.audits:
            assert self.audits[stamp]['key']==key
            return
        self.audits[stamp]=dict(key=key,bundles=OrderedDict())
        self.owners.setdefault(key,dict(audits=[],closed=False,selected=None,delivered=False))['audits'].append(stamp)
        while len(self.audits)>self.capacity:
            old,record=next(iter(self.audits.items()))
            if not self.owners[record['key']]['closed']:break
            self.audits.pop(old);self.retired_floor=max(self.retired_floor,old)
            owner=self.owners[record['key']];owner['audits'].remove(old)
            if not owner['audits']:del self.owners[record['key']]

    def push(self, stamp, sequence, message, nonempty):
        if stamp<=self.retired_floor:
            self.counts['suppressed']+=1
            return
        record=self.audits.get(stamp)
        if record is None:
            self.counts['unowned']+=1;self.ready.append(message)
            return
        owner=self.owners[record['key']]
        if not owner['closed']:
            record['bundles'][sequence]=(message,nonempty)
            while len(record['bundles'])>16:record['bundles'].popitem(last=False)
            self.counts['held']+=1
        elif (not owner['delivered'] and owner['selected'] is not None and
              owner['selected'][0]==stamp and owner['selected'][1] in (None,sequence)):
            owner['delivered']=True;self.ready.append(message);self.counts['released']+=1
        else:self.counts['suppressed']+=1

    def finish(self, key, result, audit=0, sequence=None, vbc_failure=False):
        owner=self.owners.get(key)
        if owner is None or owner['closed']:return
        owner['closed']=True
        if result=='unsafe' and vbc_failure and audit in owner['audits']:
            owner['selected']=(audit,sequence)
        elif result in ('timeout','expired'):
            # A timeout grants no movement. Its latest already observed VBC
            # blocker may still guide an observation instead of losing the
            # only useful failure evidence at the admission deadline.
            for stamp in reversed(owner['audits']):
                choices=[(seq,msg) for seq,(msg,nonempty) in self.audits[stamp]['bundles'].items() if nonempty]
                if choices:
                    owner['selected']=(stamp,choices[-1][0]);break
        for stamp in owner['audits']:
            record=self.audits[stamp]
            selected=owner['selected']
            if selected and selected[0]==stamp:
                candidates=[(seq,msg) for seq,(msg,_) in record['bundles'].items() if selected[1] in (None,seq)]
                if candidates:
                    owner['delivered']=True;self.ready.append(candidates[-1][1]);self.counts['released']+=1
            self.counts['suppressed']+=len(record['bundles'])-int(owner['delivered'] and selected is not None and selected[0]==stamp)
            record['bundles'].clear()

    def drain(self):
        result=self.ready;self.ready=[];return result
