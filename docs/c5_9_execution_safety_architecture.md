# CAREPlanner C5.9 execution-safety architecture

C5.9 consolidates the planner/controller split around one rule:

> A trajectory is executable only after the exact trajectory that may reach the
> actuator has passed the current-map learned-GCDF gate and exact VBC, and one
> committed trajectory has one execution owner until completion or safety
> preemption.

## 1. Module responsibilities

### Local sparse SCP planner

Produces a candidate trajectory only. It never owns the actuator.

- NORMAL objective: task tracking.
- PROBE_NORMAL objective: task tracking.
- REPAIR objective: visibility/q_vis.
- Hard low-confidence GCDF is enforced in the optimizer over the configured
  executable planning horizon in PROBE/REPAIR and full horizon in NORMAL.
- A solved SCP candidate is **not** an execution certificate.

### PROBE single-flight gate

`probe_single_flight_gate_node.py`

PROBE_NORMAL is serialized as:

```
IDLE -> VERIFYING -> EXECUTING -> IDLE
```

Only one PROBE candidate may be in verification/execution at a time. Extra
PROBE candidates are dropped while busy. Leaving PROBE clears the gate
immediately so REPAIR can preempt for safety.

NORMAL and REPAIR are transparent passthrough through this gate.

### Verified trajectory commit gate

Implemented by `optimized_trajectory_continuity_node.py` for compatibility with
older launches. Its C5.9 role is a commit gate, not a trajectory controller.

For PROBE/REPAIR it first constructs the exact executable view:

```
short optimizer prefix + actual braking tail + hold
```

Then:

```
executable view
    -> final current-map learned-GCDF audit
    -> exact VBC audit
    -> single committed publication
```

Both safety gates must pass. Timeout or malformed safety data fails closed.

### Shared GCDF selector / GPU worker

One selector owns one persistent GPU-worker connection and two independent
request channels:

- local-SCP channel
- final-executable channel

The final-executable channel has priority and its request/batch topic is
separate from the local-SCP channel. The two channels therefore cannot overwrite
one another while still sharing a single GPU client.

### Trajectory execution manager

Owns a committed trajectory after a **single** publication. It advances through
that trajectory from elapsed execution time and must not require continuation
republishing.

`Header.seq` is publisher-local diagnostic data only.

The stable execution token is the committed trajectory `header.stamp`, exported
as `execution_stamp_ns` in tracker summaries.

### Execution VBC audit

Execution monitoring is a separate audit stream. It may publish trajectory
suffixes for observation by the VBC auditor, but those suffixes are never sent
back to the execution manager and therefore cannot reset execution time.

### Regime manager

Owns only regime transitions and completion semantics.

- GCDF/VBC unsafe candidate: no commit.
- REPAIR completes only on real visibility/confidence acquisition.
- PROBE success is counted only when tracker reports
  `trajectory_complete_hold` for the exact pending `execution_stamp_ns`.
- Three verified **and completed** PROBE prefixes may return the system to
  NORMAL (current experimental setting).

## 2. Safety invariants

### Invariant A — VBC never certifies unknown space as collision-safe

VBC answers whether information arrives before future sweep/contact. It never
removes the current-map GCDF requirement from the trajectory that may actually
be executed.

### Invariant B — final safety checks the real braking trajectory

The final GCDF audit runs on the actual generated braking tail, not on a longer
nominal task continuation used as a surrogate.

### Invariant C — no unverified actuator trajectory

Only the exact trajectory that passed final GCDF and exact VBC may be committed.

### Invariant D — single owner of execution time

The execution manager receives a committed trajectory once and owns its time
base. Diagnostic/audit republishing must use isolated topics.

### Invariant E — PROBE is single-flight

A second PROBE candidate cannot replace the trajectory whose real completion is
currently being used as evidence of progress.

### Invariant F — safety preemption remains possible

Single-flight serialization applies only while the regime remains
PROBE_NORMAL. A PROBE->REPAIR transition clears the gate immediately, allowing a
new safety/active-sensing trajectory to preempt.

## 3. C5.9 message flow

```
local sparse SCP candidate
        |
        v
PROBE single-flight gate
        |
        v
build exact executable view
(prefix + brake + hold)
        |
        v
final executable GCDF
        |
        v
exact candidate VBC
        |
        v
commit ONCE
        |
        +----------------------+
        |                      |
        v                      v
trajectory execution      isolated execution
manager                   VBC audit stream
        |
        v
tracker summary
execution_stamp_ns
complete=1
        |
        v
regime manager
        |
        +-- PROBE: count completed prefix, request next fresh plan
        +-- REPAIR: acquisition gate determines exit
```

## 4. What must not be reintroduced

Do not reintroduce any of the following into the C5.4+ path:

1. periodic committed-trajectory continuation messages to the execution manager;
2. `Header.seq` as a cross-publisher execution identity;
3. one shared latest-anchor slot for local-SCP and final-executable GCDF queries;
4. PROBE success counted at candidate commit time;
5. a second PROBE commit while the previous PROBE execution is still in flight;
6. VBC-based removal of low-confidence space from the hard current-map safety
   definition.
