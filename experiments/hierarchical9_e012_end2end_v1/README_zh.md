# H9 E0/E1/E2 End-to-End V1

目标：训练一个**离线保存后本身就比 Mainline-A 当前 V1 更强**的完整网络，而不是长期依赖 frozen P0/V1 feature extractor。

## 共同结构

三条 arm 都从同一个完成的 P0 checkpoint 初始化，step 0 对 P0 的 9 个输出值和 raw-q 梯度严格等价：

```text
[x,q] -> NeRF encode 30D -> shared early 30->1024->512
                                  |-> union private 512->256 -> union decoder
                                  |-> S0 private 512->256 -> S0 decoder
                                  |-> ...
                                  `-> S7 private 512->256 -> S7 decoder
```

所有参数都 trainable。学习率分组：
- shared early: `1e-5`
- union tail/head: `5e-5`
- sensor private tails/heads: `1e-4`

正式训练 2000 successful updates；原 hierarchical9 global objective 完整保留：SDF + q-gradient + absolute Eikonal + directional-Hessian tension + union + consistency。

## E0

普通 end-to-end control。所有 head loss 正常一起反传到 shared early。

## E1

冲突感知 shared routing。private sensor paths 每步仍吃完整 original global objective；但 sensor loss 对 shared early 的参数梯度不在同一步全部相加。

实现方式：
1. 主 loss 中 sensor path 使用**同一 early 权重的 detached-parameter functional call**。因此 sensor 的 q-gradient/Eikonal/tension 仍看到真实 early Jacobian，但不直接更新 early 参数。
2. union + consistency 的 shared gradient 正常更新 early。
3. 每个 optimizer update 按 `S0,S1,...,S7` round-robin 选一个 sensor，把该 sensor 的完整原 objective 作为 shared-early 的无偏随机估计加入梯度。

因此 shared backbone 仍持续学习，不是 frozen；只是避免同一步叠加 8 个已确认存在冲突的 sensor gradient。

## E2

E1 + runtime-aligned projection replay。

每 20 updates（smoke 中每步）从训练 x 和随机 q 产生 actual-outside starts，用现有 `_projection` 生成当前网络 candidate，再用 analytic conservative FOV oracle 标注 candidate。候选按 sensor 存入 4096 容量 replay buffer。

Replay 使用 0.02 model-field margin，并除以 `margin^2` 做归一化，因此 projection candidate 即使落在 `f≈0` 仍有有效梯度：
- actual outside: `relu(0.02 + f_s)^2 / 0.02^2`
- actual inside: `0.25 * relu(0.02 - f_s)^2 / 0.02^2`

Replay 总权重前 500 updates 从 0 线性 warmup 到 `0.02`。private branch 每步使用所有 replay sensor；shared early 仍只接受当步 round-robin sensor 的 replay gradient，保持 E1 的冲突控制原则。

注意：这只是 projection-aligned hard-example mining，不声称复现完整 Sparse-SCP/VBC/GCDF runtime。

## Promotion 观察项

最终评估同时比较 `V1 / P0 / ABC-B / ABC-C / E0 / E1 / E2`：
- offline sensor-max MAE / sign / grad cosine / ranking
- union field（不能显著退化）
- fixed radial neighborhood
- boundary + two-sided actual-FOV sign
- matched local/uniform/all solver
- failure stage / candidate-FOV mismatch
- projection / ascent1 / ascent10
- `weight_drift_from_initial`，确认 early、union、sensor tails、sensor heads 都真正离开 P0 初始化

不会自动 promotion 到 Mainline-A。

## Smoke

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
export E012_REFERENCE_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
bash experiments/hierarchical9_e012_end2end_v1/submit.sh smoke
```

三个 arm 在 GPU0/1/2 并行，完成后 GPU3 统一 eval。

## Formal

smoke COMPLETE 后：

```bash
bash experiments/hierarchical9_e012_end2end_v1/submit.sh pilot
```

## Pack

```bash
bash experiments/hierarchical9_e012_end2end_v1/submit.sh pack
```

生成 `$E012_REFERENCE_ROOT/e012_end2end_reports.zip`。
