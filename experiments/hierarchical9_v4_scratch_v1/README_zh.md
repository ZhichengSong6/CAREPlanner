# H9 V4 scratch training v1

Random-init R1/private-tail H9 trained on the completed full-scale V4 dataset.

Supervision semantics:
- Global pool: all eight analytic FOV signs and exact union sign. Sign loss is a smooth zero-margin logistic; it does not invent distance magnitudes.
- V3: per-sensor continuous value only where value_valid. V3 gradients are deliberately unused. Union continuous value is used only if all eight sensor values for that (x,q) are valid.
- V4 tubes: sensor-specific signed local value.
- V4 boundary: only zero-offset rows receive analytic-normal gradient, Eikonal, and tension supervision.
- Union consistency uses union ~= max(sensor outputs) on the global-sign batch.

Architecture is exactly R1 private-tail H9 and starts from random initialization. Default formal training is 50k optimizer updates on four GPUs.

Numerics: v1 intentionally uses one combined FP32 forward/backward per update. This keeps the zero-offset boundary input-gradient/Hessian supervision in FP32 and avoids multi-forward DDP unused-parameter ambiguity. The smoke uses the exact formal batch sizes before code is frozen for 50k.
