# H9 Scratch-50k R0/R1/R2

目标：在完全相同的 50,000-update、4×RTX3090、400k pairs/update 条件下，公平比较 V1 原结构、private-tail 结构、以及 conflict-aware shared routing。

## 三条 arm

### R0 — V1 architecture scratch 50k

```text
[x,q] -> encode 30D -> 1024 -> 512 -> 256
                                  |-> union 256->128->128->1
                                  |-> S0    256->128->128->1
                                  ...
                                  `-> S7    256->128->128->1
```

所有 head 的 loss 正常同时更新 shared trunk。

### R1 — private-tail architecture scratch 50k

```text
[x,q] -> encode 30D -> 1024 -> 512
                              |-> union 512->256 -> 128->128->1
                              |-> S0    512->256 -> 128->128->1
                              ...
                              `-> S7    512->256 -> 128->128->1
```

所有 head 的 loss 正常同时更新 shared early trunk。

### R2 — R1 architecture + conflict-aware shared routing scratch 50k

网络参数结构与 R1 完全一致，seed=0 时初始化逐参数一致。

每个 step：
- union、8 个 sensor private tail/head 都使用完整 original H9 objective；
- sensor path 的 q-gradient / Eikonal / tension 不被 detach；
- sensor loss 对 shared early 的参数梯度在主 pass 中阻断；
- shared early 接收 union/consistency 的正常梯度 + 一个 round-robin sensor 的完整原 head objective；
- selected sensor 按 S0,S1,...,S7 循环。

## 公平性冻结项

三条 arm：
- random scratch，无 V1/P0/E checkpoint 初始化；
- model seed = 0；
- architecture 构造完成后将 sampling RNG 重置到 stream seed = 190915；
- global x = 4000，shared q = 100，即 400,000 pairs / successful update；
- Adam lr=1e-3；
- 与原 scratch V1 相同 ReduceLROnPlateau；
- original H9 loss：5*SDF + 0.1*q-grad + 0.01*abs-Eikonal + 0.01*tension；union + mean sensor + 0.1 consistency；
- fp16 AMP，overflow retry 不计 successful update；
- val every 1000，checkpoint every 5000；
- 4-GPU DDP，固定 3090node3。

每个训练 update 会把 global x indices 与 shared q 写入 rolling SHA256。最终 R0/R1/R2 的 `training_stream_sha256` 必须完全相同。

## 为什么不是一个 job 连跑三条

原 V1 scratch-50k 本身就申请 3-day walltime。把三条 50k 硬塞一个 Slurm job 容易超过 walltime。因此：
- smoke：一个 4-GPU job 顺序跑 R0/R1/R2 各 2 steps；
- formal：R0、R1、R2 各自一个 4-GPU / 3-day job，按顺序提交；
- 用户 association 同时只能有一个 job，正好依次运行。

## 使用

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
BRANCH=mainline-b/h9-scratch50k-r012-v1
git fetch origin "$BRANCH"
git switch "$BRANCH" 2>/dev/null || git switch -c "$BRANCH" --track "origin/$BRANCH"
git merge --ff-only "origin/$BRANCH"
```

### 1. Smoke

```bash
export R012_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh smoke
```

日志：

```bash
source "$R012_ROOT/r012_jobs.env"
tail -n 160 -F "$R012_ROOT/logs/r012_smoke_${R012_SMOKE_JOB}.out"
```

必须看到：
- CPU tests OK；
- R0/R1/R2 都 COMPLETE @2；
- 三条 `training_stream_sha256` 完全一致；
- `[done] r012_smoke_complete`。

### 2. Formal R0

smoke 成功后不要再修改该 branch：

```bash
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh R0
```

### 3. Formal R1

R0 `COMPLETED 0:0` 且 run.json COMPLETE 后：

```bash
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh R1
```

### 4. Formal R2

R1 完成后：

```bash
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh R2
```

### 5. 验证三条 stream 完全一致

```bash
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh verify
```

## 3-day walltime 中断恢复

每 5000 steps 写 `latest.pt`。只有确认原 job 已退出后，执行：

```bash
bash experiments/hierarchical9_scratch50k_r012_v1/submit.sh resume-R0
# 或 resume-R1 / resume-R2
```

恢复会校验：arm、训练源码 SHA、world size、训练配置，并恢复 optimizer/scheduler/scaler/RNG/stream-chain 状态。

## 输出

```text
$R012_ROOT/smoke/{R0,R1,R2}/
$R012_ROOT/formal/R0/
$R012_ROOT/formal/R1/
$R012_ROOT/formal/R2/
```

每个 formal arm 最终包含至少：
- `final.pt`
- `run.json`
- `metrics.jsonl`
- `train_args.json`
- `latest.pt`（若中途 checkpoint 已写）

Mainline-A、runtime、URDF safety、GCDF/VBC/tracker 均不修改。
