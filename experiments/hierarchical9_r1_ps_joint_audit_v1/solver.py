"""Full fixed-solver regression; reuse the repository runtime, no solver fork."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from common import json_safe


def run_solver(out, starts, models, deps, oracle, masks, lo, hi, cfg, rank):
    probes = {name: deps.probe.make_probe(deps.sensor_view(model), masks, lo, hi)
              for name, model in models.items()}
    for name, probe in probes.items():
        actual = {key: getattr(probe, key) for key in cfg["solver_settings"]}
        if actual != cfg["solver_settings"]:
            raise ValueError(f"Frozen runtime parameters changed for {name}: {actual}")
    names = tuple(models)
    ids = range(rank, len(starts), cfg["world"])
    with (Path(out)/f"solves.rank{rank}.jsonl").open("x") as handle:
        for count, i in enumerate(ids):
            spec = starts[i]
            x = torch.tensor(spec["x"], device=lo.device, dtype=torch.float32)
            q = torch.tensor(spec["q_init"], device=lo.device, dtype=torch.float32)
            row = dict(spec, case_id=i, models={})
            order = names[i % len(names):] + names[:i % len(names)]
            for name in order:
                row["models"][name] = deps.probe.run_probe(probes[name], oracle, x, q, int(spec["sensor"]))
            handle.write(json.dumps(json_safe(row), allow_nan=False)+"\n")
            if (count+1) % 50 == 0:
                handle.flush()
                print(f"[solver rank{rank}] {count+1}/{len(ids)} cases", flush=True)
    print(f"[solver rank{rank}] DONE {len(ids)} cases", flush=True)
