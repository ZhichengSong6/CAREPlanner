# Expiring simulation velocity controller

`care_velocity_controller/ExpiringJointVelocityController` claims seven ordered
`VelocityJointInterface` joints. Command expiry and braking run inside the
controller-manager update, independently of the Python command producer.
This package targets the existing Gazebo direct-velocity simulation interface.

## Interfaces

All names below are relative to the controller namespace.

| Interface | Type | Meaning |
| --- | --- | --- |
| `command_stamped` input | `sensor_msgs/JointState` | Exact ordered `name`, seven finite bounded `velocity`, and generation time in `header.stamp`. Position/effort are unused. |
| `command` output | `std_msgs/Float64MultiArray` | Applied joint-handle commands sampled at 100 Hz, compatible with existing planner feedback/recorders. This is **not** a command input. |
| `state` output | `care_velocity_controller/CommandStatus` | Application time, source stamp/sequence, applied/measured velocity, mode, expiry/stop times and rejection/rearm counters. |
| `rearm` service | `std_srvs/Trigger` | Queues a request. Service success means queued; observe the state counters for the actual decision. |

The controller does not subscribe to an unstamped velocity topic. The limiter's
legacy output goes to `requested_command` when the experiment is enabled.

## Time and state transitions

All command times, deadlines and rate limits use **ROS simulation time**.
`command_timeout` is 0.100 s measured from generation, not from receipt. A valid
replacement must have a strictly newer stamp, not be in the future, and be
younger than the timeout. Malformed, nonfinite and out-of-limit messages do not
renew the deadline. Expiry is checked before processing a replacement, so a
late fresh message cannot conceal an elapsed deadline.

```
WAITING -- valid post-start command --> TRACKING
TRACKING -- expiry --> BRAKING -- commanded zero --> LATCHED
LATCHED -- explicit successful rearm --> TRACKING (zero only)
```

Braking is latched through producer recovery and incoming nonzero commands.
Rearm requires all of:

- Already in `LATCHED`, without a clock fault.
- The latest accepted message is zero, generated after the commanded stop and
  still fresh at the rearm update.
- All measured joint speeds are finite and at most 0.001 rad/s in magnitude.
- An explicit service request. Early/invalid requests are consumed and rejected.

Rearm applies zero. Further motion requires a command generated after that
rearm update. Clock rollback latches a fault that requires controller restart.
Before the first valid command the controller waits at zero indefinitely.

Applied commands respect per-joint velocity and acceleration settings in both
tracking and braking. Time credit per update is the minimum of positive elapsed
time, the supplied period and `max_braking_step` (0.020 s). Typical Gazebo
updates are 0.001 s; that observed/configured value is not a wall-time guarantee.

## Enable and build

```bash
source /opt/ros/noetic/setup.bash
catkin build care_velocity_controller -j2 -p1 --no-status
source devel/setup.bash
catkin build care_velocity_controller --no-deps --make-args run_tests
```

Standalone Gazebo uses `gazebo_velocity_control.launch use_command_watchdog:=true`.
The full R1 entry uses `ACTUATOR_COMMAND_WATCHDOG=true`; it checks the loaded
controller type, wires the stamped limiter output, and records applied state.
The switch defaults to false while task qualification remains incomplete.

## Contract limits

This covers a stopped/late **producer while the Gazebo/controller update keeps
running**. It does not cover a stalled controller manager, simulator process or
real actuator. Physics also pauses when Gazebo simulation time pauses; no
wall-clock stopping bound is claimed. Lifecycle controller stop writes zero
directly because no more update cycles are available; switching/shutdown is
outside the bounded-rate active-controller braking contract.

The acknowledgement reports the joint-handle command, not measured mechanical
compliance. Physical stopping distance, state freshness, confidence validity and
the entire swept volume still require separate certification. The limiter
timestamps its own output; its upstream unstamped desired input still does not
carry an end-to-end execution token or original generation age.

Run `scripts/verify_phase_e_actuator_watchdog_gazebo.py` for diagnostics;
its generated reports remain local.

## Response diagnostics

`state` also reports measured position and the applied-velocity integral at the
same control update. The integral accumulates the previous output over elapsed
time before writing the next output, so dropped 100 Hz status samples do not
drop command integration. `integral_valid` becomes false on time/period mismatch
or rollback until controller restart. `max_update_gap` is an observation, not
a guaranteed scheduler bound. The new status fields change its ROS message MD5;
rebuild subscribers using this experimental message.

Use `scripts/verify_phase_e_response_domain_gazebo.py` for response diagnostics.
Sampled response errors must not be promoted to physical bounds automatically.

## Fixed execution leases (candidate, 2026-10-19)

`require_execution_lease=true` selects a separate protocol. The controller then
subscribes only to `command_leased` (`LeasedVelocityCommand`) and
`execution_lease` (`ExecutionLease`), and does not accept `command_stamped`.
The normal switch remains false pending complete stopping qualification.

An external stopping-envelope authorizer supplies a lease containing the
controller's fresh random session, execution token, monotonically increasing
lease sequence, issue time, fixed expiry, and certificate SHA. The SHA identifies
the certificate; this transport does not verify its geometry or qualify it.
No production authorizer is enabled by this change. Component tests use clearly
identified synthetic leases on a private ROS master.

The tracker preserves the lease and adds its original generation time and
monotonic source sequence. The limiter preserves all identity/deadline fields;
repeated timer ticks reuse the same source time/sequence. Only velocity is
limited. Legacy unstamped inputs are not subscribed to in this mode, and no
legacy velocity output is emitted by the tracker or limiter. Expiry and the old
source timeout are checked in the controller before replacements. A pending
lease changes the applied identity only when its matching command is accepted.
Renewal requires a new externally authorized lease, never merely a new command.

Stopping retires the execution token for this controller session. A newer lease
for that token cannot resume motion. Recovery requires a greater token, a fresh
post-stop zero under a new lease, measured rest, and explicit rearm; new motion
must be generated after rearm. Sticky authority revocations and matching fresh
terminal stops survive overwrites by subsequent ordinary command callbacks.
Execution tokens must increase within a session; the mainline uses ROS-nanosecond
execution identities. Delayed prior-session receipts cannot reset consumers.

`state` now binds the actual applied identity and original generation time,
source sequence, lease expiry, retirement floor, and actual configured velocity,
acceleration and braking-step limits. Tracker/limiter reject inconsistent limits;
the offline stopping check uses the reported braking step. The experimental
watchdog YAML now uses 2.0 rad/s on all seven joints, matching the current planner
candidate. Changing `CommandStatus` requires rebuilding all subscribers.

Gazebo launch accepts `use_command_watchdog:=true require_execution_lease:=true`;
the tracker launch accepts `require_execution_lease:=true` with
`use_acceleration_limiter:=true`. These switches alone do not authorize motion:
without a qualified issuer there are no production leases. Build and freeze a
consistent isolated runtime before using either switch; the current canonical
ELF/message outputs remain the previous version.

`scripts/phase_e_leased_stop_evidence.py` binds real application receipts to both
stopping modes and checks a complete grid-center enclosure union with coherent
snapshot/version/horizon evidence. It has no control publisher. Qualified
simulation timing, full-horizon mechanical response and future-environment
contracts remain required; neither old response samples nor the ideal-hardware
protocol fixture supplies them. `runtime_admission` remains false.

Stopping budgets are anchored at the controller measurement time. Receipt age
is part of the allowed acceleration interval before expiry and braking; adding
only a past-displacement allowance would underestimate the later braking speed.
Reported remaining stop times start at evaluation time, while the enclosure
remains centered on the measured pose. Independent production-policy replays
cover delayed receipts, reversals, expiry boundaries and braking-step caps.
The map query also supports an explicitly marked `command_only` diagnostic;
that result cannot qualify a physical stopping envelope.

An optional lease target range uses `target_range_enforced` and seven
`target_abs_velocity` values. Commands echo both fields; they are part of the
exact lease identity. The tracker bounds its targets, the limiter preserves the
range and bounds its emitted targets, and the controller rejects out-of-range
targets or forged range fields. The applied receipt reports the applied grant's
range. Invalid, negative, nonfinite or above-controller-limit ranges are rejected.
The default false flag retains the full configured velocity domain for budgeting.
Zero target range does not erase existing output: actual output still changes
at the controller acceleration limit and must be enclosed through full braking.
Unexpired outstanding grants with larger ranges cannot be ignored, even if their
deadline is earlier than the newest grant.

`phase_e_local_stop_geometry.py` provides a separate offline geometry backend.
It encloses the complete joint box using interval FK and intersects that
enclosure with the existing global solid dilation. Subdivision covers the whole
box; there is no nominal-path sampling or point truncation. The current backend
supports fixed and local +Z revolute joints in the audited URDF. All other joint
types/axes fail closed. A larger joint-box enclosure may cover both stopping
modes with one rasterization. Grid-center clearance semantics are unchanged.
These command/geometry improvements do not supply the missing timing, physical
response or future-environment qualifications and do not enable an issuer.
