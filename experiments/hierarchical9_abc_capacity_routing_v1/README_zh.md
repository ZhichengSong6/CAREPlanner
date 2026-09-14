# 主线 B：H9 A/B/C capacity-routing 诊断

本实验不是 P4 loss ablation。它直接回答三个结构问题：

1. **A：现有 P0 shared feature 是否已经足够？** 只训练现有 S0-S7 decoder；shared backbone 与 union 路径冻结。
2. **B：如果 A 不够，是否只是 decoder capacity 不够？** 在 A 基础上，每个 sensor 增加一个 `256→64→256` residual adapter，最后线性层零初始化，所以开始时严格等价 P0；adapter+原 decoder 可训练。
3. **C：如果 A/B 都不够，是否需要 sensor-specific representation？** 只共享到 `30→1024→512`；把 P0 原来的 shared `512→256+ReLU` 复制八份作为 private tail，并接原 sensor decoder。early shared 和 union 路径冻结。

三个模型全部从**同一个已完成的 P0 checkpoint**开始。这里的 P0 是 hierarchical9 的 original-global-objective 2k-update control，不是更早的 scalar 网络。

## 固定训练条件

- parent：`P0/final.pt`，52000 total updates 时的模型。
- A/B/C 初始化时所有 9 个输出值和 raw-q gradient 都必须与 P0 在数值容差内一致。
- fresh Adam，lr=1e-4，formal 2000 updates；smoke 2 updates。
- 每个 arm 每 update 都使用完整 `4000 x 100 = 400000` global pairs。
- A/B/C 使用独立进程但相同 RNG 定义；formal/smoke 最终 stream SHA 必须三者一致，否则 eval 拒绝。
- 训练目标只保留原 hierarchical9 的 **per-sensor** SDF + q-gradient + global Eikonal + tension。
- **不训练 union，不训练 consistency，不加入 P1/P2/P3 boundary/normal/side losses。**
- union path 全程冻结，用于验证“sensor adaptation 是否能在不破坏 P0 global path 的情况下增加能力”。
- A/B/C 各占一张 3090 并行训练；第 4 张 GPU 在三者完成后运行统一 matched eval。

这不是 equal-parameter 实验：A/B/C 的 trainable capacity 故意不同。它是 capacity/routing diagnostic。

## 为什么 B 使用 residual adapter

B 的 adapter：

```text
z256 ────────────────┐
  └→ Linear 256→64 → ReLU → Linear 64→256 (zero init) ─┘
```

输出为 `z + adapter(z)`。最后一层零初始化，因此 update 0 时 B 与 P0 完全等价，同时允许后续学习更深的 sensor-specific 非线性。

## 为什么 C 不是简单“再加一层”

P0：

```text
30 → 1024 → 512 → 256 → sensor decoder
```

C：

```text
30 → 1024 → 512   [frozen shared]
                  ├→ copied private 512→256 → S0 decoder
                  ├→ copied private 512→256 → S1 decoder
                  ...
                  └→ copied private 512→256 → S7 decoder
```

每个 private tail 从 P0 原 shared tail 精确复制，所以初始输出不变；训练时每个 sensor 可修改自己的高层 representation，而不会回写 early shared backbone。

## 一次 formal eval 输出什么

统一 eval 比较 `V1 / P0 / A / B / C`，并一次性输出：

- true analytic boundary 值、梯度范数、法向 cosine、线性化零点位移；
- 固定两侧真实 FOV sign profiles；
- 固定 continuous-radius neighborhood TP/TN/FP/FN；
- matched local/outside 与 uniform-outside runtime solver；
- root source × failure stage 与 P0-vs-A/B/C paired outcomes；
- original field/ranking sentinel；
- union/sensor-max projection + ascent；
- A/B/C trainable/total parameter count、train/val curves、初始化等价误差；
- **P0 union/S0…S7 对 shared backbone 的 9×9 gradient cosine matrix**，同时报告 full original head objective 与 SDF-only，并分别看全 shared trunk 与最后的 512→256 shared layer。

所以这轮不再只回答“多了几个 solve”，而是区分：

- frozen feature 已经够不够；
- decoder capacity 是否瓶颈；
- 是否需要 private representation；
- shared task gradient 是否存在明显冲突；
- train/val gap 是否随 capacity 增长而恶化。

## 服务器同步

在服务器 Mainline-B worktree：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b

git status --short
BRANCH=mainline-b/h9-abc-capacity-routing-v1
git fetch origin "$BRANCH"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH"
  git merge --ff-only "origin/$BRANCH"
else
  git switch -c "$BRANCH" --track "origin/$BRANCH"
fi

git rev-parse HEAD
```

不要切本地 Mainline-A worktree，不修改旧 P0/P1/P2/P3 输出。

默认 reference root：

```bash
export ABC_REFERENCE_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
```

## 1. Smoke

```bash
bash experiments/hierarchical9_abc_capacity_routing_v1/submit.sh smoke
```

一个 4×3090 allocation：CPU unit tests → A/B/C 各 2 个完整 update 并行 → GPU3 matched smoke eval。

查看：

```bash
REF="$ABC_REFERENCE_ROOT"
source "$REF/abc_jobs.env"
tail -n 100 -F "$REF/logs/abc_smoke_${ABC_SMOKE_JOB}.out"
sacct -j "$ABC_SMOKE_JOB" --format=JobID,State,ExitCode,Elapsed
```

必须看到：

```text
[done] abc_arm_complete arm=A mode=smoke ...
[done] abc_arm_complete arm=B mode=smoke ...
[done] abc_arm_complete arm=C mode=smoke ...
[done] abc_evaluation_complete mode=smoke ...
[done] abc_workflow_complete mode=smoke
```

并且 Slurm `COMPLETED / 0:0`。

## 2. Formal pilot

Smoke 完整通过且任务退出后：

```bash
bash experiments/hierarchical9_abc_capacity_routing_v1/submit.sh pilot
source "$REF/abc_jobs.env"
tail -n 100 -F "$REF/logs/abc_pilot_${ABC_PILOT_JOB}.out"
```

formal job 内部自动完成 A/B/C 2000 updates 和统一 eval，不需要再提交单独 eval job。

每个 arm 的独立训练日志位于：

```text
$REF/logs/abc_pilot_<JOBID>/A.out
$REF/logs/abc_pilot_<JOBID>/B.out
$REF/logs/abc_pilot_<JOBID>/C.out
```

正式输出：

```text
$REF/abc_capacity_routing/{A,B,C}/
$REF/evaluation_abc_capacity_routing/
```

## 3. 打包

formal job 完成后：

```bash
bash experiments/hierarchical9_abc_capacity_routing_v1/submit.sh pack
```

生成：

```text
$REF/abc_capacity_routing_reports.zip
```

只包含 summary/report/manifest 和 A/B/C run/validation/metrics，不包含 checkpoint 或数据集。

## 解释规则

- A ≈/优于 P0，B/C 无明显增益：优先解释为当前 shared feature 已足够，过去的 sensor→backbone coupling 可能不必要。
- A 差、B 明显好：shared feature 有信息，但现有 decoder capacity 不足。
- A/B 差、C 明显好：需要 sensor-specific 高层 representation，完全 frozen 256D feature 上的 decoder 不够。
- A/B/C 都差：需要重新检查更早 shared representation、输入/标签覆盖或数据泛化；此时才更有依据考虑扩大 shared backbone。
- capacity 增加若只改善 train 而不改善 held-out/solver，则不要把更深网络自动解释为成功。

本实验仍然是 FOV-only development validation，不是 LOS/GCDF/VBC/轨迹/真实 seen 或安全证书，也不会自动切换 Mainline-A checkpoint。
