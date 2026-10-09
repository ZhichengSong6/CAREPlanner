# H9 V4 scratch training v2

v2 is an isolated objective revision after the completed v1 two-step smoke exposed severe class imbalance in the global analytic sign pool.

Unchanged from v1:
- the verified mmap training cache (identity 060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc);
- random-init R1/private-tail H9 architecture;
- V3 continuous values and strict all-8 union-value policy;
- V3 gradients remain disabled;
- all V4 tube values;
- zero-offset regular V4 boundary normal, Eikonal and tension supervision;
- one combined FP32 forward/backward per update;
- batch sizes and 50k formal update budget.

Only scientific change:
- global sign loss is class-balanced. Each available (sensor, negative/positive) cell has equal weight under exact global DDP normalization. Union negative/positive classes are balanced the same way.
- raw accuracy is still logged, but balanced accuracy and positive fraction are now first-class metrics.

Workflow: smoke (2 updates) -> pilot (500 updates) -> formal train (50k). All use identical formal batch composition.
