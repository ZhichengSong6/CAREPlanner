"""Freeze stratified samples from the REAL V4 shards, linking all three caches.

Selection uses only original label eligibility and a seeded random priority;
never model predictions, optimizer success or solver outcomes.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from common import Dependencies, read_json, sha, write_json, write_npz


class Reservoir:
    """Streaming uniform-priority sampling without replacement in each cell."""
    def __init__(self, k: int, seed: int):
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.rows = None

    def add(self, arrays: dict[str, np.ndarray]):
        n = len(next(iter(arrays.values())))
        if any(len(v) != n for v in arrays.values()):
            raise ValueError("Reservoir column lengths differ")
        self.count += n
        new = dict(arrays, priority=self.rng.random(n))
        if self.rows is not None:
            new = {k: np.concatenate((self.rows[k], v), axis=0) for k, v in new.items()}
        keep = np.argsort(new["priority"], kind="stable")[:self.k]
        self.rows = {k: v[keep] for k, v in new.items()}


def source_checks(cache, pairs):
    source = Path(cache.manifest["source"]).resolve()
    if sha(source / "manifest.json") != cache.manifest["source_manifest_sha256"]:
        raise ValueError("V4 production manifest changed")
    if pairs.manifest["source_manifest_sha256"] != cache.manifest["source_manifest_sha256"]:
        raise ValueError("Pair cache and training cache have different production roots")
    if Path(pairs.manifest["source"]).resolve() != source:
        raise ValueError("Wrong pair-cache production path")
    if read_json(source / "v4_summary.json").get("status") != "COMPLETE":
        raise ValueError("V4 stage incomplete")
    for name, digest in cache.manifest.get("source_summaries_sha256", {}).items():
        if sha(source / name) != digest:
            raise ValueError(f"Production summary changed: {name}")
    entries = pairs.manifest["source_shards"]
    names = sorted(entries)
    if names != [f"shard_{i:04d}.npz" for i in range(len(names))] or not names:
        raise ValueError("Pair-source shards missing/noncontiguous")
    paths = [source / "v4_tubes" / n for n in names]
    if sorted(p.name for p in (source / "v4_tubes").glob("shard_*.npz")) != names:
        raise ValueError("Source shard list changed")
    return source, paths


def compare_array(actual, expected, label: str):
    if not np.array_equal(actual, expected):
        raise ValueError(f"Original/cache mapping mismatch: {label}")


def verify_selected_links(a: dict, cache, pairs):
    """Exact equality to stored training inputs; no new q or label is fabricated."""
    for code, split in ((0, "train"), (1, "val")):
        m = a["split"] == code
        if not m.any():
            continue
        ii = a["pair_cache_index"][m]
        pa = pairs.data[split]
        for source_key, cached_key in (("q_plus", "q_plus"), ("q_minus", "q_minus"),
                                        ("x_index", "x_index"), ("sensor", "sensor"),
                                        ("radius_id", "radius_id")):
            compare_array(a[source_key][m], pa[cached_key][ii], "pair/" + cached_key)
        ba = cache.a[split]["boundary"]
        bi = a["boundary_cache_index"][m]
        for source_key, cached_key in (("q_zero", "q"), ("normal", "grad"),
                                        ("x_index", "x_index"), ("sensor", "sensor")):
            compare_array(a[source_key][m], ba[cached_key][bi], "boundary/" + cached_key)
        ta = cache.a[split]["v4"]
        radii = np.asarray([.005, .01, .02], np.float32)[a["radius_id"][m]]
        for side, sign in (("plus", 1), ("minus", -1)):
            vi = a[f"v4_{side}_index"][m]
            for source_key, cached_key in (("q_" + side, "q"), ("x_index", "x_index"), ("sensor", "sensor")):
                compare_array(a[source_key][m], ta[cached_key][vi], "v4/" + side + "/" + cached_key)
            compare_array(sign * radii, ta["value"][vi], "v4/" + side + "/value")


def make_plan(out: Path, cfg: dict, deps: Dependencies, cache, pairs) -> dict:
    if (out / "selected.npz").exists():
        raise FileExistsError("Refusing to replace an existing frozen sample plan")
    source, paths = source_checks(cache, pairs)
    k = cfg["pairs_per_split_sensor_radius"]
    bins = {(sp, s, r): Reservoir(k, cfg["seed"] + 1000*sp + 10*s + r)
            for sp in (0, 1) for s in range(8) for r in range(3)}
    cursors = {sp: {"pair": 0, "v4": 0, "boundary": 0} for sp in (0, 1)}
    spaces = {sp: np.unique(cache.a[name]["global"]["x_index"])
              for sp, name in ((0, "train"), (1, "val"))}
    if np.intersect1d(spaces[0], spaces[1]).size:
        raise ValueError("Source train/validation x overlap")
    seen_x = set()
    for shard_id, path in enumerate(paths):
        digest = sha(path)
        if digest != pairs.manifest["source_shards"][path.name] or read_json(path.with_suffix(".json"))["sha256"] != digest:
            raise ValueError(f"Corrupt production V4 shard: {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            # NpzFile re-decompresses on each key access; load each immutable column once.
            z = {k: archive[k] for k in archive.files}
            current_x = set(np.unique(z["x_index"]).tolist())
            if current_x & seen_x:
                raise ValueError("Spatial x index appears in multiple original source shards")
            seen_x.update(current_x)
            grouped = deps.pairdata._pairs(z)  # verified production value/grad schema
            split = z["split"]
            n = len(split)
            vpos, bpos = {}, {}
            for sp in (0, 1):
                vm = split == sp
                bm = vm & (z["value"] == 0)
                vpos[sp] = np.full(n, -1, np.int64)
                bpos[sp] = np.full(n, -1, np.int64)
                vpos[sp][vm] = np.arange(vm.sum()) + cursors[sp]["v4"]
                bpos[sp][bm] = np.arange(bm.sum()) + cursors[sp]["boundary"]
                cursors[sp]["v4"] += int(vm.sum())
                cursors[sp]["boundary"] += int(bm.sum())
            for pi, mi, zi, rid in grouped:
                for sp in (0, 1):
                    chosen = split[zi] == sp
                    p, m, zero = pi[chosen], mi[chosen], zi[chosen]
                    if not len(zero):
                        continue
                    ids = np.arange(len(zero)) + cursors[sp]["pair"]
                    cursors[sp]["pair"] += len(zero)
                    xx = np.asarray(z["x_index"][zero], np.int64)
                    if not np.isin(xx, spaces[sp]).all():
                        raise ValueError("Raw V4 rows disagree with training/validation x split")
                    if np.any(bpos[sp][zero] < 0):
                        raise ValueError("Missing corresponding boundary cache row")
                    block = {
                        "split": np.full(len(zero), sp, np.uint8),
                        "sensor": z["sensor"][zero], "radius_id": np.full(len(zero), rid, np.uint8),
                        "x_index": xx, "source_slot": z["source_slot"][zero],
                        "shard": np.full(len(zero), shard_id, np.int32),
                        "plus_row": p, "minus_row": m, "zero_row": zero,
                        "q_plus": z["q"][p], "q_minus": z["q"][m], "q_zero": z["q"][zero],
                        "normal": z["grad"][zero],
                        "stored_g_m": np.stack((z["g_m"][zero], z["g_m"][p], z["g_m"][m]), 1),
                        "pair_cache_index": ids, "boundary_cache_index": bpos[sp][zero],
                        "v4_plus_index": vpos[sp][p], "v4_minus_index": vpos[sp][m],
                    }
                    for s in range(8):
                        sm = block["sensor"] == s
                        if sm.any():
                            bins[sp, s, rid].add({key: v[sm] for key, v in block.items()})
        if (shard_id+1) % 20 == 0 or shard_id+1 == len(paths):
            print(f"[prepare] original V4 shards verified {shard_id+1}/{len(paths)}", flush=True)
    for sp, name in ((0, "train"), (1, "val")):
        expected = {"pair": pairs.manifest["total_by_split"][name],
                    "v4": cache.manifest["counts"][name]["v4"],
                    "boundary": cache.manifest["counts"][name]["boundary"]}
        if cursors[sp] != expected:
            raise ValueError(f"Whole-source/cache counts differ: {name} {cursors[sp]} != {expected}")
    if any(b.rows is None for b in bins.values()):
        raise ValueError("An entire prespecified diagnostic stratum is empty")
    first = next(iter(bins.values())).rows
    selected = {key: np.concatenate([b.rows[key] for b in bins.values()]) for key in first}
    selected["case_id"] = np.arange(len(selected["split"]), dtype=np.int64)
    selected["x"] = np.asarray(cache.x[selected["x_index"]]).copy()
    verify_selected_links(selected, cache, pairs)
    write_npz(out / "selected.npz", selected)
    report = {"status": "COMPLETE", "selection": "seeded random priority, original eligible pairs only; no learned selection",
              "selected_pairs": len(selected["split"]), "full_source_shards": len(paths),
              "whole_source_cache_counts": cursors, "source": str(source),
              "selected_sha256": sha(out / "selected.npz"), "cache_links": "ALL_SELECTED_EXACT_MATCH",
              "strata": [{"split": sp, "sensor": s, "radius_id": r, "eligible": b.count,
                          "selected": len(b.rows["split"]), "shortfall": max(0, k-b.count)}
                         for (sp, s, r), b in bins.items()],
              "limitation": "Geometry audit and model diagnostics are on this prespecified subset, not the whole V4 population."}
    write_json(out / "selection.json", report)
    return report
