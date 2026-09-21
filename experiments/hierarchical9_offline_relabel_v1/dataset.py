"""Read paired cached labels. This is a reader, NOT a replacement training script.

The old objective accepts one mask and recomputes union from masked sensors.
DO NOT blindly feed this cache into that function: it would erase missing heads
and conflate value/gradient supervision. A future cache-aware objective must
use the explicit union masks and separate normalization counts below.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from cache_io import FORMAT,read_json
from repo_oracle import sha256_file


class OfflineLabelDataset(Dataset):
    def __init__(self,root,label_set='new',split='train',verify_hashes=True,cache_shards=2):
        self.root=Path(root).resolve()
        if label_set not in ('old','new') or split not in ('train','val'):
            raise ValueError('label_set=old/new; split=train/val')
        self.label_set=label_set;self.split=split;self.cache_shards=max(1,int(cache_shards));self.cache=OrderedDict()
        index=read_json(self.root/'dataset_index.json')
        if index.get('format')!=FORMAT or not index.get('complete'):
            raise ValueError('A fully merged cache is required')
        if sha256_file(self.root/'run_spec.json')!=index['run_spec_sha256']:
            raise ValueError('Run spec changed')
        self.files=[];self.rows=[]
        split_id=0 if split=='train' else 1
        for entry in index['shards']:
            path=self.root/entry['path']
            if verify_hashes and sha256_file(path)!=entry['sha256']:
                raise ValueError(f'Corrupt labels: {path}')
            file_id=len(self.files);self.files.append(path)
            with np.load(path,allow_pickle=False) as z:
                self.rows.extend((file_id,int(i)) for i in np.flatnonzero(z['split']==split_id))

    def __len__(self): return len(self.rows)

    def _shard(self,k):
        if k not in self.cache:
            with np.load(self.files[k],allow_pickle=False) as z:
                self.cache[k]={n:z[n] for n in z.files}
            while len(self.cache)>self.cache_shards: self.cache.popitem(last=False)
        self.cache.move_to_end(k)
        return self.cache[k]

    def __getitem__(self,i):
        file_id,row=self.rows[i];a=self._shard(file_id);mode=self.label_set
        vm=a['paired_value_mask'][row].copy();gm=a['paired_grad_mask'][row].copy()
        uv=bool(a['paired_union_value_mask'][row]);ug=bool(a['paired_union_grad_mask'][row])
        # Safe placeholders ONLY with the explicit masks retained. Never multiply a NaN by zero in a loss.
        return dict(query_id=np.int64(a['query_id'][row]),x_index=np.int64(a['x_index'][row]),
            inputs=np.concatenate((a['x'][row],a['q_query'][row])).astype(np.float32),
            sensor_value=np.where(vm,a[mode+'_value'][row],0).astype(np.float32),
            sensor_grad=np.where(gm[:,None],a[mode+'_grad'][row],0).astype(np.float32),
            sensor_value_mask=vm,sensor_grad_mask=gm,sensor_support=a['support'][row].copy(),
            union_value=np.float32(a['union_'+mode+'_value'][row] if uv else 0.),
            union_grad=(a['union_'+mode+'_grad'][row].copy() if ug else np.zeros(7,np.float32)),
            union_value_mask=np.bool_(uv),union_grad_mask=np.bool_(ug))
