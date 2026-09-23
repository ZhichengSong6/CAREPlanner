# V3-fast v1

Goal: accelerate the existing V3 continuous-boundary label while preserving its screening semantics.

Changes:
- run geometry/SLSQP on CPU and use 16 independent workers on one node;
- classify existing q0 bank points by active FOV face in one batched pass;
- first solve each face only from nearest face-matched boundary seeds;
- if a face has no qualified solution, is competitive with insufficient independent evidence, or the screened result is invalid/ambiguous/uncertain, add the original V3 seeds as fallback;
- exact task-local geometry memoization across faces/starts.

The original V3 files and caches are immutable. The first job is a parity/speed benchmark against the completed 2004-task V3 pilot; it does not create training labels.
