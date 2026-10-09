"""Read the existing frozen cache; write ONLY stratified indices in the new run."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import torch

EXPECTED_CACHE = "060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc"
OFFSETS = np.asarray([-0.02, -0.01, -0.005, 0., 0.005, 0.01, 0.02], np.float32)
FIELDS = {
    "global": ("x_index", "q", "sensor_sign", "union_sign"),
    "v3": ("x_index", "q", "sensor_value", "sensor_value_mask"),
    "v4": ("x_index", "q", "sensor", "value"),
    "boundary": ("x_index", "q", "sensor", "grad"),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def source_identity(root: Path) -> str:
    files = {p.name: sha(p) for p in sorted(Path(root).iterdir())
             if p.is_file() and p.suffix in (".py", ".sh", ".json")}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def offset_codes(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    nearest = np.abs(values[:, None] - OFFSETS).argmin(1)
    if not np.isfinite(values).all() or np.any(np.abs(values - OFFSETS[nearest]) > 2e-7):
        raise ValueError("V4 offsets do not match the frozen seven-level labels")
    return nearest


class Cache:
    def __init__(self, root: Path, *, verify: bool = False,
                 expected: str | None = EXPECTED_CACHE):
        self.root = Path(root).resolve()
        self.identity = sha(self.root / "manifest.json")
        if expected is not None and self.identity != expected:
            raise ValueError(f"Wrong cache identity: {self.identity}")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if (self.manifest.get("format") != "care_h9_v4_training_cache_v1" or
                self.manifest.get("status") != "COMPLETE" or
                self.manifest.get("training_ready") is not True):
            raise ValueError("Incomplete/wrong cache")
        if verify:
            for rel, spec in self.manifest["files"].items():
                p = (self.root / rel).resolve()
                if not p.is_relative_to(self.root):
                    raise ValueError("Cache manifest path escapes root")
                if p.stat().st_size != spec["bytes"] or sha(p) != spec["sha256"]:
                    raise ValueError(f"Corrupt cache file: {rel}")
        self.x = np.load(self.root / "x.npy", mmap_mode="r", allow_pickle=False)
        if self.x.ndim != 2 or self.x.shape[1] != 3 or not np.isfinite(self.x).all():
            raise ValueError("Bad workspace array")
        self.a = {}
        for split in ("train", "val"):
            self.a[split] = {}
            for kind, fields in FIELDS.items():
                arrays = {k: np.load(self.root / split / f"{kind}_{k}.npy",
                                    mmap_mode="r", allow_pickle=False) for k in fields}
                n = int(self.manifest["counts"][split][kind])
                if n <= 0 or any(len(v) != n for v in arrays.values()):
                    raise ValueError(f"Count mismatch: {split}/{kind}")
                if arrays["q"].shape != (n, 7):
                    raise ValueError("Bad joint dimensions")
                self.a[split][kind] = arrays

    def build_indices(self, target: Path) -> None:
        """One atomic, read-only-cache preparation, within the formal job."""
        target = Path(target)
        if target.exists():
            self.load_indices(target, verify=True)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".indices.", dir=target.parent))
        report = {"format": "boundary_first_indices_v1", "cache_identity": self.identity,
                  "strata": {}, "files": {}}
        try:
            spaces = {s: np.unique(self.a[s]["global"]["x_index"]) for s in ("train", "val")}
            if np.intersect1d(spaces["train"], spaces["val"]).size:
                raise ValueError("Train/validation x overlap")
            for split in ("train", "val"):
                for kind, arrays in self.a[split].items():
                    n = len(arrays["q"])
                    codes = np.empty(n, np.int16) if kind in ("v4", "boundary") else None
                    for lo in range(0, n, 250000):
                        hi = min(n, lo + 250000)
                        xi = arrays["x_index"][lo:hi]
                        if (np.any(xi < 0) or np.any(xi >= len(self.x)) or
                                not np.isin(xi, spaces[split]).all()):
                            raise ValueError(f"Wrong x/split in {split}/{kind}")
                        if not np.isfinite(arrays["q"][lo:hi]).all():
                            raise ValueError("Nonfinite query")
                        if kind == "global":
                            ss = arrays["sensor_sign"][lo:hi]
                            us = arrays["union_sign"][lo:hi]
                            if (ss.shape != (hi-lo, 8) or not np.isin(ss, [-1, 1]).all() or
                                    not np.array_equal(us, np.where((ss > 0).any(1), 1, -1))):
                                raise ValueError("Bad analytic signs/union sign")
                        elif kind == "v3":
                            mask = arrays["sensor_value_mask"][lo:hi]
                            val = arrays["sensor_value"][lo:hi]
                            if mask.shape != (hi-lo, 8) or mask.dtype != np.bool_ or val.shape != mask.shape:
                                raise ValueError("Bad V3 masks")
                            if not np.isfinite(val[mask]).all():
                                raise ValueError("Nonfinite VALID V3 distance")
                        else:
                            sensor = arrays["sensor"][lo:hi].astype(np.int64)
                            if np.any((sensor < 0) | (sensor >= 8)):
                                raise ValueError("Bad sensor ID")
                            if kind == "v4":
                                codes[lo:hi] = 7*sensor + offset_codes(arrays["value"][lo:hi])
                            else:
                                normal = arrays["grad"][lo:hi]
                                if (normal.shape != (hi-lo, 7) or not np.isfinite(normal).all() or
                                        np.max(np.abs(np.linalg.norm(normal, axis=1)-1)) > 5e-3):
                                    raise ValueError("Bad boundary normal")
                                codes[lo:hi] = sensor
                    if codes is not None:
                        groups = 56 if kind == "v4" else 8
                        count = np.bincount(codes, minlength=groups)
                        if np.any(count == 0):
                            raise ValueError(f"Missing declared stratum in {split}/{kind}: {count.tolist()}")
                        order = np.argsort(codes, kind="stable").astype(np.uint32)
                        ptr = np.r_[0, np.cumsum(count)].astype(np.int64)
                        np.save(tmp / f"{split}_{kind}_order.npy", order, allow_pickle=False)
                        np.save(tmp / f"{split}_{kind}_ptr.npy", ptr, allow_pickle=False)
                        report["strata"][f"{split}/{kind}"] = count.tolist()
                    print(f"[index] verified {split}/{kind}: {n} rows", flush=True)
            for p in sorted(tmp.glob("*.npy")):
                report["files"][p.name] = sha(p)
            write_json(tmp / "manifest.json", report)
            os.rename(tmp, target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        self.load_indices(target, verify=True)

    def load_indices(self, path: Path, *, verify: bool = False) -> None:
        path = Path(path)
        report = json.loads((path / "manifest.json").read_text())
        if report.get("cache_identity") != self.identity:
            raise ValueError("Index/cache mismatch")
        if verify:
            for name, digest in report["files"].items():
                if sha(path / name) != digest:
                    raise ValueError("Corrupt sampling index")
        self.index_identity = sha(path / "manifest.json")
        self.index = {(s, k): tuple(np.load(path / f"{s}_{k}_{f}.npy", mmap_mode="r")
                                    for f in ("order", "ptr"))
                      for s in ("train", "val") for k in ("v4", "boundary")}

    def ids(self, split: str, kind: str, n: int, seed: int, step: int,
            rank: int, world: int) -> np.ndarray:
        if split == "val" and kind == "v3":
            return np.arange(rank, len(self.a[split][kind]["q"]), world)
        if n % world:
            raise ValueError("Global batch must divide world size")
        local = n // world
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, step, rank, world, list(FIELDS).index(kind), split == "val"]))
        if kind in ("global", "v3"):
            return rng.integers(len(self.a[split][kind]["q"]), size=local)
        order, ptr = self.index[(split, kind)]
        # Global positions keep stratum counts balanced across ranks (difference <= 1).
        codes = (np.arange(local) + rank*local + step) % (len(ptr)-1)
        pos = ptr[codes] + (rng.random(local) * (ptr[codes+1]-ptr[codes])).astype(np.int64)
        return np.asarray(order[pos], np.int64)

    def batch(self, split: str, kind: str, ids: np.ndarray, device: torch.device) -> dict:
        a = self.a[split][kind]
        out = {k: torch.as_tensor(np.asarray(v[ids]).copy(), device=device) for k, v in a.items()}
        x = torch.as_tensor(np.asarray(self.x[a["x_index"][ids]]).copy(), device=device)
        out["inputs"] = torch.cat([x.float(), out["q"].float()], 1)
        if kind == "v4":
            out["cell"] = out["sensor"].long()*7 + torch.as_tensor(offset_codes(a["value"][ids]), device=device)
        return out
