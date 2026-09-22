"""Strict paired-cache reader; never exposes unverified candidates as target gradients.

Not a trainer. The original single-mask objective is NOT compatible without an
explicit value/gradient/union masking and normalization adapter.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


class PairedLabels(Dataset):
    def __init__(self,root,label_set='new',split='train',cache_shards=2):
        if label_set not in ('old','new') or split not in ('train','val'):
            raise ValueError('Choose label_set old/new, split train/val')
        self.root=Path(root).resolve();self.mode=label_set;self.cache=OrderedDict();self.cap=max(1,int(cache_shards))
        idx=json.loads((self.root/'dataset_index.json').read_text())
        if idx.get('format')!='careplanner_paired_offline_labels_v3' or not idx.get('complete') or not idx.get('audit_complete'):
            raise ValueError('Use a complete paired_cache after the declared audit; base_cache is not an audited training cache')
        if idx.get('gradient_policy')!='FD_PASS_ONLY':raise ValueError('Unexpected gradient policy')
        if digest(self.root/'summary.json')!=idx['summary_sha256']:raise ValueError('Corrupt summary')
        self.files=[];self.rows=[];split_id=0 if split=='train' else 1
        for e in idx['shards']:
            p=(self.root/e['path']).resolve()
            if not p.is_relative_to(self.root) or digest(p)!=e['sha256']:raise ValueError('Corrupt/unsafe shard')
            k=len(self.files);self.files.append(p)
            with np.load(p,allow_pickle=False) as z:
                if np.any(z['paired_grad_mask'] & (z['gradient_status']!='PASS')):
                    raise ValueError('Unverified gradient has a training mask')
                self.rows.extend((k,int(i)) for i in np.flatnonzero(z['split']==split_id))

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        k,i=self.rows[index]
        if k not in self.cache:
            with np.load(self.files[k],allow_pickle=False) as z:self.cache[k]={key:z[key] for key in z.files}
            while len(self.cache)>self.cap:self.cache.popitem(last=False)
        self.cache.move_to_end(k);a=self.cache[k];m=self.mode
        vm=a['paired_value_mask'][i].copy();gm=a['paired_grad_mask'][i].copy()
        uv=bool(a['paired_union_value_mask'][i]);ug=bool(a['paired_union_grad_mask'][i])
        return dict(query_id=a['query_id'][i],x_index=a['x_index'][i],
            inputs=np.concatenate((a['x'][i],a['q_query'][i])).astype(np.float32),
            sensor_value=np.where(vm,a[m+'_value'][i],0).astype(np.float32),
            sensor_grad=np.where(gm[:,None],a[m+'_grad'][i],0).astype(np.float32),
            sensor_value_mask=vm,sensor_grad_mask=gm,sensor_support=a['support'][i].copy(),
            union_value=np.float32(a['union_'+m+'_value'][i] if uv else 0),
            union_grad=a['union_'+m+'_grad'][i].copy() if ug else np.zeros(7,np.float32),
            union_value_mask=np.bool_(uv),union_grad_mask=np.bool_(ug))
