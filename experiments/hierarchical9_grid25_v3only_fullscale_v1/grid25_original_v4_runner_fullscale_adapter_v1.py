#!/usr/bin/env python3
"""Pickle-compatible spawn shim for the frozen original Grid25 stage worker.

The fullscale adapter loads the scientific production runner under the exact
module name of this file using importlib. Spawned ProcessPool workers need
a real importable module of that name to unpickle stage_worker. This tiny
shim preserves the original stage_worker callable signature and delegates
into the fullscale runner process-locally, without changing any frozen source,
manifest, or algorithm implementation.

Do not import the fullscale runner at module level: doing so would overwrite
sys.modules[name] while this shim itself is being imported by unpickling.
"""
from pathlib import Path


def stage_worker(stage, out, rank, world, urdf):
    # Deferred import: spawn first resolves this module and its public
    # stage_worker function, then the already-unpickled callable executes.
    import runner as fullscale_runner
    if stage not in ("global_pool", "v3_labels"):
        raise RuntimeError("Only fullscale V3 stages allowed")
    return fullscale_runner.base.stage_worker(stage, out, rank, world, urdf)
