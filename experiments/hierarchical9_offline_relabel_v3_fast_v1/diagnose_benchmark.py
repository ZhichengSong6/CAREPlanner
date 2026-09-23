#!/usr/bin/env python3
"""Read-only diagnosis of an existing V3-fast benchmark output."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    a=ap.parse_args(); root=a.input.resolve()
    rows=[]
    for p in sorted(root.glob("rank*.jsonl")):
        rows.extend(json.loads(x) for x in p.read_text().splitlines() if x.strip())
    rows.sort(key=lambda r:r["bench_id"])
    if not rows: raise SystemExit("no rank*.jsonl rows")
    mismatch=[r for r in rows if r["old_value_valid"]!=r["fast_value_valid"]]
    fb=[r for r in rows if r["fallback_planes"]]
    nf=[r for r in rows if not r["fallback_planes"]]
    def block(rr):
        if not rr:return {}
        fast=np.asarray([r["fast_elapsed_ms"] for r in rr],float)/1000
        old=np.asarray([r["old_elapsed_ms"] for r in rr],float)/1000
        return dict(n=len(rr),fast_median_s=float(np.median(fast)),fast_p95_s=float(np.quantile(fast,.95)),
                    speedup=float(old.sum()/fast.sum()),old_attempts=sum(r["old_attempts"] for r in rr),
                    fast_attempts=sum(r["fast_attempts"] for r in rr),
                    memo_calls=sum(r["memo_calls"] for r in rr),memo_hits=sum(r["memo_hits"] for r in rr))
    report={"tasks":len(rows),"mismatch_count":len(mismatch),"fallback":block(fb),"no_fallback":block(nf),
            "fallback_plane_count_hist":dict(Counter(len(r["fallback_planes"]) for r in rows)),
            "fast_attempt_hist":dict(Counter(r["fast_attempts"] for r in rows)),
            "mismatches":mismatch}
    print(json.dumps(report,indent=2,allow_nan=False))
    if mismatch:
        print("\n=== VALIDITY MISMATCHES ===")
        for r in mismatch:
            print(f"bench={r['bench_id']} query={r['query_id']} S{r['sensor']} "
                  f"old_valid={r['old_value_valid']} fast_valid={r['fast_value_valid']} "
                  f"old_value={r['old_value']} fast_value={r['fast_value']} "
                  f"old_s={r['old_elapsed_ms']/1000:.3f} fast_s={r['fast_elapsed_ms']/1000:.3f} "
                  f"attempts={r['old_attempts']}->{r['fast_attempts']} fallback={r['fallback_planes']} "
                  f"qstar_l2={r['qstar_l2']} grad_cos={r['grad_cosine']}")
if __name__=="__main__":main()
