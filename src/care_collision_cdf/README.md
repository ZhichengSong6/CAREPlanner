# care_collision_cdf

ROS package for Yiming-style configuration-space distance field inference in
CAREPlanner.

## Responsibility

This package intentionally does **not** decide why a workspace point is
forbidden. Upstream modules will build

    P_forbidden = P_occupied union P_low_confidence_frontier

and this package evaluates the scene-level configuration-space distance

    d(q) = min_{p in P_forbidden} f_cdf(p, q)

plus its gradient with respect to q.

The downstream MPC can linearize this quantity into a QP inequality. Exact VBC
remains the final visibility-safety commit authority.

## Checkpoint

Default:

    checkpoints/yiming_cdf/model_dict.pt

See the README in that directory for supported checkpoint layouts.
