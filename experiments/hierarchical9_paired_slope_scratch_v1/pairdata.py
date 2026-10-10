"""Derive VERIFIED same-anchor +/- V4 tube pairs. Original production/cache are read-only.

Pairs are only formed from original, signed, geometrically checked tube records.
Never synthesize joint configurations or reuse an old label for a newly generated q.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
from data import Cache, OFFSETS, offset_codes, sha, write_json

FORMAT = "care_h9_v4_verified_pairs_v1"
RADII = (0.005, 0.01, 0.02)
FIELDS = {"x_index": (np.int32, ()), "sensor": (np.uint8, ()),
          "radius_id": (np.uint8, ()), "q_plus": (np.float32, (7,)),
          "q_minus": (np.float32, (7,))}


def _pairs(z) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Return (plus indices, minus indices, zero indices, radius id) per shard."""
    mandatory = ("x_index", "split", "sensor", "source_slot", "offset", "q", "normal", "g_m")
    if any(k not in z for k in mandatory):
        raise ValueError("Original V4 stage lacks anchor identity or geometry")
    n = len(z["offset"])
    if not n:
        return []
    if any(len(z[k]) != n for k in mandatory):
        raise ValueError("V4 arrays have different lengths")
    q = np.asarray(z["q"], dtype=np.float32)
    normal = np.asarray(z["normal"], dtype=np.float32)
    margin = np.asarray(z["g_m"], dtype=np.float32)
    if q.shape != (n, 7) or normal.shape != (n, 7) or not np.isfinite(q).all() or not np.isfinite(normal).all() or not np.isfinite(margin).all():
        raise ValueError("Invalid V4 q/normal/margin")
    if np.any(np.abs(np.linalg.norm(normal, axis=1)-1) > 5e-3):
        raise ValueError("V4 normal is not unit length")
    split = np.asarray(z["split"], np.uint8)
    sensor = np.asarray(z["sensor"], np.uint8)
    if not np.isin(split, [0, 1]).all() or np.any(sensor >= 8):
        raise ValueError("Invalid split or sensor")
    codes = offset_codes(np.asarray(z["offset"], np.float32)).astype(np.int64)
    ids = np.stack((np.asarray(z["x_index"], np.int64), sensor.astype(np.int64),
                    np.asarray(z["source_slot"], np.int64)), axis=1)
    unique, group = np.unique(ids, axis=0, return_inverse=True)
    address = 7*group+codes
    if len(np.unique(address)) != n:
        raise ValueError("Duplicate V4 source anchor/offset in shard")
    positions = np.full((len(unique), 7), -1, np.int64)
    positions.reshape(-1)[address] = np.arange(n)
    if (positions[:, 3] < 0).any():
        raise ValueError("Anchor with tube offsets but missing exact zero-offset record")
    out = []
    zero = positions[:, 3]
    for rid, radius in enumerate(RADII):
        minus = positions[:, 2-rid]
        plus = positions[:, 4+rid]
        valid = (plus >= 0) & (minus >= 0)
        if not valid.any():
            continue
        pi, mi, zi = plus[valid], minus[valid], zero[valid]
        if not (np.array_equal(split[pi], split[mi]) and np.array_equal(split[pi], split[zi])):
            raise ValueError("Paired samples cross original train/validation splits")
        if not (np.all(margin[pi] > 0) and np.all(margin[mi] < 0) and np.all(np.abs(margin[zi]) <= 2e-6)):
            raise ValueError("Paired samples fail original analytic FOV signs")
        delta_p = q[pi] - q[zi] - radius*normal[zi]
        delta_m = q[mi] - q[zi] + radius*normal[zi]
        if (np.max(np.abs(delta_p)) > 3e-5 or np.max(np.abs(delta_m)) > 3e-5 or
            np.max(np.abs(normal[pi]-normal[zi])) > 1e-5 or
            np.max(np.abs(normal[mi]-normal[zi])) > 1e-5):
            raise ValueError("Pair does not match the SAME anchor and signed normal offsets")
        out.append((pi, mi, zi, rid))
    return out


def production_shards(cache: Cache) -> tuple[Path, list[Path], str]:
    p = Path(cache.manifest["source"]).resolve()
    m = p / "manifest.json"
    if not m.is_file() or sha(m) != cache.manifest["source_manifest_sha256"]:
        raise ValueError("Original V4 production source/manifest mismatch")
    status = json.loads((p/"v4_summary.json").read_text())
    if status.get("status") != "COMPLETE":
        raise ValueError("Original V4 tube stage incomplete")
    paths = sorted((p/"v4_tubes").glob("shard_*.npz"))
    if not paths:
        raise ValueError("Original V4 tube shards unavailable")
    names = [x.stem for x in paths]
    if names != [f"shard_{i:04d}" for i in range(len(paths))]:
        raise ValueError("Missing V4 production shard")
    return p, paths, sha(m)


def build_pairs(cache: Cache, out: Path):
    """Atomic two-pass conversion into mmap files, no modifications to original artifacts."""
    out = Path(out)
    if out.exists():
        PairCache(out, cache, verify=True)
        return
    production, shards, prod_hash = production_shards(cache)
    split_space = {i: np.unique(cache.a[s]["global"]["x_index"]) for i,s in ((0,"train"),(1,"val"))}
    counts = {s: np.zeros(24, np.int64) for s in ("train","val")}
    sidecars = {}
    for pass_id in (0, 1):
        if pass_id == 1:
            out.parent.mkdir(parents=True,exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix=".paired-tubes.",dir=out.parent))
            buffers = {}
            for split in ("train", "val"):
                total = int(counts[split].sum())
                d = tmp/split;d.mkdir(parents=True,exist_ok=True)
                for field, (dtype, tail) in FIELDS.items():
                    buffers[split,field] = np.lib.format.open_memmap(d/f"{field}.npy",mode="w+",dtype=dtype,shape=(total,)+tail)
            cursors = {split: 0 for split in ("train", "val")}
        try:
            for si, path in enumerate(shards):
                sidecar = path.with_suffix(".json")
                if pass_id == 0:
                    meta = json.loads(sidecar.read_text())
                    digest = sha(path)
                    if meta.get("sha256") != digest:
                        raise ValueError(f"Corrupt original V4 tube shard: {path}")
                    sidecars[path.name] = digest
                else:
                    if sha(path) != sidecars[path.name]:
                        raise ValueError(f"Original V4 shard changed during pair derivation: {path}")
                with np.load(path,allow_pickle=False) as z:
                    pairs = _pairs(z)
                    xx, sens, splits, qq = (z[k] for k in ("x_index","sensor","split","q"))
                    for pi, mi, zi, rid in pairs:
                        for code, sp in ((0,"train"),(1,"val")):
                            mask = splits[zi] == code
                            if not mask.any(): continue
                            pids, mids, zids = pi[mask], mi[mask], zi[mask]
                            x = np.asarray(xx[zids],np.int64)
                            if not np.isin(x, split_space[code]).all():
                                raise ValueError("Paired x not in the frozen source train/val split")
                            c = np.asarray(sens[zids],np.int64)*3+rid
                            if pass_id == 0:
                                counts[sp] += np.bincount(c,minlength=24)
                            else:
                                a = cursors[sp];b = a+len(pids)
                                for field, values in (("x_index",x),("sensor",sens[zids]),
                                     ("radius_id",np.full(len(pids),rid,np.uint8)),
                                     ("q_plus",qq[pids]),("q_minus",qq[mids])):
                                    buffers[sp,field][a:b] = values
                                cursors[sp] = b
                if (si+1)%30==0 or si+1==len(shards):
                    print(f"[pairs] pass={pass_id+1}/2 shards={si+1}/{len(shards)}",flush=True)
            if pass_id == 0:
                for sp in counts:
                    if np.any(counts[sp] == 0):
                        raise ValueError(f"Missing complete +/- pairs for split/radius/sensor: {sp} {counts[sp].tolist()}")
            else:
                for sp in counts:
                    if cursors[sp] != int(counts[sp].sum()):
                        raise RuntimeError("Pair count changed across passes")
                for v in buffers.values():v.flush()
                del buffers
                pair_manifest = {"format":FORMAT,"status":"COMPLETE", "source":str(production),
                     "source_manifest_sha256":prod_hash, "cache_identity":cache.identity,
                     "source_shards":sidecars,"stratum_counts":{k:v.tolist() for k,v in counts.items()},
                     "fields":{}, "total_by_split":{k:int(v.sum()) for k,v in counts.items()},
                     "semantics":"Strict same-anchor +/- signed V4 tubes, original q only; 8 sensors x 3 radii"}
                for f in sorted(tmp.rglob("*.npy")):
                    pair_manifest["fields"][str(f.relative_to(tmp))] = sha(f)
                # Deterministic 24-stratum balanced sampler; no data duplication in cache.
                for sp in counts:
                    sensors = np.load(tmp/sp/"sensor.npy", mmap_mode="r")
                    radius = np.load(tmp/sp/"radius_id.npy", mmap_mode="r")
                    cell = sensors.astype(np.int32)*3+radius.astype(np.int32)
                    order = np.argsort(cell,kind="stable").astype(np.uint32)
                    ptr = np.r_[0,np.cumsum(np.bincount(cell,minlength=24))].astype(np.int64)
                    np.save(tmp/sp/"order.npy",order,allow_pickle=False)
                    np.save(tmp/sp/"ptr.npy",ptr,allow_pickle=False)
                    pair_manifest["fields"][f"{sp}/order.npy"] = sha(tmp/sp/"order.npy")
                    pair_manifest["fields"][f"{sp}/ptr.npy"] = sha(tmp/sp/"ptr.npy")
                write_json(tmp/"manifest.json",pair_manifest)
                os.rename(tmp,out)
                print("[pairs] complete",pair_manifest["total_by_split"],flush=True)
        except BaseException:
            if pass_id == 1 and tmp.exists():shutil.rmtree(tmp,ignore_errors=True)
            raise
    PairCache(out,cache,verify=True)


class PairCache:
    def __init__(self, root: Path, base: Cache, *, verify=False):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        if (self.manifest.get("format")!=FORMAT or self.manifest.get("status")!="COMPLETE" or
            self.manifest.get("cache_identity") != base.identity):
            raise ValueError("Invalid pair cache identity/format")
        if verify:
            for name,digest in self.manifest["fields"].items():
                if sha(self.root/name)!=digest:raise ValueError(f"Pair cache file corrupted: {name}")
        self.identity = sha(self.root/"manifest.json")
        self.x = base.x
        self.data = {}
        self.index = {}
        for sp in ("train","val"):
            self.data[sp] = {k:np.load(self.root/sp/f"{k}.npy",mmap_mode="r",allow_pickle=False) for k in FIELDS}
            self.index[sp] = tuple(np.load(self.root/sp/f"{k}.npy",mmap_mode="r",allow_pickle=False) for k in ("order","ptr"))
            if len(self.index[sp][0])!=self.manifest["total_by_split"][sp]:
                raise ValueError("Bad pair indices")

    def ids(self, split:str, n:int, seed:int, step:int, rank:int, world:int):
        if n%world:raise ValueError("Pair batch must divide world")
        rng = np.random.default_rng(np.random.SeedSequence([seed,step,rank,world,73,split=="val"]))
        order,ptr=self.index[split]
        local=n//world
        code=(np.arange(local)+rank*local+step)%24
        pos=ptr[code]+(rng.random(local)*(ptr[code+1]-ptr[code])).astype(np.int64)
        return np.asarray(order[pos],np.int64)

    def batch(self, split:str, indices:np.ndarray, device:torch.device):
        a=self.data[split]
        idx=np.asarray(indices,np.int64)
        x=torch.as_tensor(np.asarray(self.x[a["x_index"][idx]]).copy(),device=device).float()
        qp=torch.as_tensor(np.asarray(a["q_plus"][idx]).copy(),device=device).float()
        qm=torch.as_tensor(np.asarray(a["q_minus"][idx]).copy(),device=device).float()
        return {"plus_inputs":torch.cat((x,qp),dim=1),
                "minus_inputs":torch.cat((x,qm),dim=1),
                "sensor":torch.as_tensor(np.asarray(a["sensor"][idx]).copy(),device=device).long(),
                "radius_id":torch.as_tensor(np.asarray(a["radius_id"][idx]).copy(),device=device).long()}
