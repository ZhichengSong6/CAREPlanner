# 主线 B：H9 真实边界校准 P0/P1 短程实验

这是固定 V1 之后的网络训练实验。只新增本目录，不修改主线 A 的 runtime、planner、
URDF、阈值、控制器或全局开关。基于 b558b54 的边界审计代码。
审计发现：困难 sensor 的边界方向可以较好，但零值和梯度尺度失准；非原库边界处的
离散最近点标签不能当作连续 signed distance 真值。这轮检验校准是否有实际求解收益。
不会因为 Case026 没有无遮挡候选而把它强行作为训练必须成功的标签。

## 固定实验定义

网络不变：编码 [u,sin(u),cos(u)]，shared 30→1024→512→256，九个独立
256→128→128→1 decoder，ReLU，输出 [union,S0,...,S7]，1,133,705 个可训练参数。

两组都加载**同一份 V1 final.pt 权重**：
`979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199`。
加载前验证哈希；不加载旧 optimizer、scheduler 或 RNG。不冻结任何参数。
这是**微调筛选实验**，不是重新从头训练，也不是恢复 V1 的 Adam 状态。

| 设置 | P0 | P1 |
|---|---|---|
| 初始权重 | V1 @ 50000 | 同一份 V1 @ 50000 |
| optimizer | fresh Adam，固定 lr=1e-4 | 相同 |
| 成功 update | 2000 | 2000 |
| 每步全局样本 | 4000 x × 100 shared q = 400000 | 完全相同 |
| 附加边界查询 | 4096，计算 loss 但乘零反传 | 同一批 4096，加入校准监督 |
| AMP | 全局 FP16 前向，原 FP32 loss | 相同 |
| 边界 | FP32 前向、输入梯度、loss | 相同 |
| 梯度裁剪/自动 LR scheduler | 不用 | 不用 |
| 校准权重 ramp | 0 | min(update/500,1) |

P0 保留原 uniform 数据/标签/loss；P1 只新增边界损失。为减少同时改变的变量，
**这轮没有换成 70/30 查询混合**。两组计算相同边界图，P0 的校准梯度严格为零。
P1 获得额外几何监督，不能把它称为“相同标签信息预算”的实验。计算耗时也实测记录，
不宣称每一步 GPU 时间完全相同。每个成功 update 的 uniform 输入和边界索引累计哈希，
最终 evaluator 强制验证 P0/P1 两条采样流相同；AMP retry 重用原样本，不算 update。

原 V1 uniform objective 原样复用（绝对值 Eikonal、directional Hessian tension、
union 监督和 consistency 均保留）。新增项：

```
L_P0 = L_V1_uniform
L_P1 = L_V1_uniform + min(update/500,1) * L_boundary

L_boundary = average over 8 sensors and 2 origins of:
    5.0 * mean(f_s(x,q0)^2)
  + 0.1 * mean(1 - cosine(d f_s / d q, analytic_unit_normal_s))
  + 0.1 * mean(abs(norm(d f_s / d q) - 1))
```

这些系数是本轮预先固定的候选，不是已调优的最佳值。局部范数系数 .1 不改变原全局 .01。
所有导数相对于原始 7D q；不用 feature-space 梯度，也不把 raw gradient 的无关关节分量隐藏。
每个 sensor 内 bank/offbank 两种来源等权，各自使用全局计数归一化，再跨 sensor 平均。
单个 S7 锚点不监督其它 head 为零，尤其不把 union 当成零。
新增边界锚点上没有 nearest-bank value/gradient、union 或 consistency 监督。
边界两侧当前仅做评估，不强行把法向偏移 t 当作精确 signed distance 训练。
全局抽样原监督继续保留，可能仍存在与局部校准的张力；P0/P1 正是在检验这种校准方案，
不预先保证成功、不因损失下降而自动晋升 checkpoint。

## 1. 数据准备与隔离

prepare.py 读取原 7.4GB q0 数据，不修改、不复制原数据。
严格复用 seed=0、1000 validation spatial points 的 split。
train cache 只从原训练空间点生成；val cache 只作监测/后评估，绝不回流训练。
缓存加载时检查 canonical split、(x_index,x)、sensor mask、q limits、单位法向、
几何残差以及 train/val 文件 SHA256。原数据的身份记录 size/mtime，**不是完整内容哈希**。

- bank_refined：对原 sensor q0 做解析精修和规则边界检查。
- offbank_refined：对合格 bank root 做一次 0.10 rad 切向扰动，重新精修，并检查距原库至少 .001 rad。
- 条件与上一轮审计相同：root residual≤1e-5 m，排除平面并列、退化、靠近限位和 FD 法向不一致。
- 每次生成尝试/拒绝都记录；不无限补采掩盖失败。
- 全部 16 个 sensor/origin strata 必须非空，否则任务失败，不能静默改成只训练部分 sensor。
- train 默认每 sensor 512 个有支持的训练点、每点最多4个库锚点。
- val 默认每 sensor 64 个有支持的验证点、每点最多2个库锚点。
- 同一点可能多次用于采样；缓存中的实际不同点数、接受数和重复利用由 manifest 记录。
- 这是局部切向生成，不是均匀独立边界样本，也不是最近边界优化。

## 2. 服务器同步（不要在本地 Codex 的 A 线目录执行）

继续使用已有的 B worktree，确认无需要保存的 tracked 修改，切到新分支：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git status --short
git fetch origin mainline-b/h9-boundary-calibration-pilot-v1
git switch -c mainline-b/h9-boundary-calibration-pilot-v1 --track origin/mainline-b/h9-boundary-calibration-pilot-v1
git rev-parse HEAD
sha256sum -c experiments/hierarchical9_calibration_pilot_v1/SHA256SUMS
```

创建分支只执行一次；之后在该分支用 `git pull --ff-only origin mainline-b/h9-boundary-calibration-pilot-v1`。
不 reset/clean、不切换主线 A 的本地目录、不 merge master 到正在跑的实验。

## 3. 一条命令提交整个 smoke 工作流

```bash
bash experiments/hierarchical9_calibration_pilot_v1/submit.sh smoke
```

提交器依次排队三个阶段，**不是在登录节点运行训练**：

1. prepare：1张3090，缓存生成 + 强制真实仓库依赖/几何测试。
2. train array：P0/P1 各4张3090，array 并发限制为1，最多同时4卡，执行先后顺序不重要。
   两组各跑2次真正400000-pair update；边界批512。每组先跑4-rank NCCL数学等价测试。
3. evaluate：1张3090，只有两组都成功才运行；比较V1/P0/P1。

使用 `afterok` 依赖。上游失败时下游可能保持 DependencyNeverSatisfied，不要盲目重复提交，
先看上游日志并处理/取消已失效的等待任务。没有任务会自动切换 runtime。

默认3090node3，GPU partition；prepare/evaluate 4 CPU，train 16 CPU。
**不添加 --mem 或 --mem-per-***。Conda 默认 `$HOME/miniforge3` 的 `viscdf`。
支持 `CONDA_SH/CONDA_ENV` 指向已有环境；不重装环境。

绝对日志路径在 sbatch 命令行传入并在提交前创建目录。避免相对日志路径和引用通配符问题。
读取最近一次同模式提交（不是任意历史训练任务）：

```bash
ARTIFACT_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
RUN_ROOT=$(cat "$ARTIFACT_ROOT/outputs/mainline_b/last_h9_cal_smoke.path")
source "$RUN_ROOT/jobs.env"
squeue -r -j "$PREP_JOB,$TRAIN_JOB,$EVAL_JOB"
tail -n 100 -F "$RUN_ROOT/logs/prepare_${PREP_JOB}.out"
```

按Ctrl+C只退出tail，不会取消任务。准备结束后可看：

```bash
tail -n 100 -F "$RUN_ROOT/logs/train_${TRAIN_JOB}_0.out"
tail -n 100 -F "$RUN_ROOT/logs/train_${TRAIN_JOB}_1.out"
tail -n 100 -F "$RUN_ROOT/logs/evaluate_${EVAL_JOB}.out"
```

排队时文件还不存在是可能的；也可以 `scontrol show job "$EVAL_JOB"` 查看实际 StdOut。
不要把带`*`的路径加双引号后当作通配符使用。脚本最后标记：

```
[done] calibration_cache_complete ...
[done] calibration_pilot_complete arm=P0 successful_updates=2 ...
[done] calibration_pilot_complete arm=P1 successful_updates=2 ...
[done] calibration_pair_evaluation_complete ...
```

smoke只有2步，不能用指标判断效果；它的boundary ramp仍按500步，目标是测试实际训练链。

## 4. Smoke 完成后，单独提交正式2000-step pilot

```bash
bash experiments/hierarchical9_calibration_pilot_v1/submit.sh pilot
```

它生成新的训练/验证cache，P0/P1分别重新加载V1，不继承smoke的权重或optimizer。
将上面的 `last_h9_cal_smoke.path` 换成 `last_h9_cal_pilot.path` 即可读取此工作流的真实路径。
训练和输出目录独立，已存在目录拒绝覆盖；原V1 final.pt不变。

可用环境选项（两组同时采用，保持对照）：

```bash
CAL_MICROBATCH_X=125 CAL_DECODE_CHUNK=32 bash experiments/hierarchical9_calibration_pilot_v1/submit.sh smoke
```

这不改变400000全局pairs。`CAL_AMP=bf16`或`off`是精度诊断，改变配置后需要重新跑两组。
不要给正常运行的任务临时改代码；worker会检查提交时HEAD及tracked工作区是否改变。

## 5. 输出与比较

全部输出在 `ARTIFACT_ROOT/outputs/mainline_b/h9_cal_{smoke|pilot}_.../`：

```
cache/{manifest.json,train.npz,val.npz,train_generation.jsonl,val_generation.jsonl}
P0/{run.json,initial_validation.json,validation.jsonl,metrics.jsonl,latest.pt,final.pt}
P1/{...同上...}
evaluation/{manifest.json,report.json,summary.md,solves.jsonl,planning_samples.npz}
logs/{prepare_*.out,train_*_0.out,train_*_1.out,evaluate_*.out}
jobs.env
```

pilot checkpoint 单独格式：parent_updates=50000，pilot_updates=2000，total_updates=52000。
它不是新的“从头训练50000步”的模型。旧evaluator的V1固定身份加载器不会接受它；
本目录evaluate_pair.py专门处理P0/P1身份与采样流检查，不需要伪装checkpoint或改原loader。
latest.pt只是中间诊断快照；本版本**没有自动恢复训练功能**，中断不算完成。
短程实验失败后保留目录，根据原因重新提交新工作流，不在原目录覆盖假装续训。

每500步及首步监测固定uniform validation和边界train/val差异。最终比较：
- 同一验证cache中V1/P0/P1的abs(f)、raw normal cosine、范数分布及逐样本线性化零点偏移。
- ±.005/.01/.02/.05 rad两侧实际解析符号；不强制内侧一定仍在同一FOV plane。
- 逐sensor局部见证起点和随机外侧起点的原runtime solver、配对FOV通过、root、耗时、失败阶段。
  默认每sensor最多32个cache支持验证点；smoke为2。选定head自身g通过，不用union代替。
- 64000个uniform held-out field/ranking sentinel（smoke6400）。离散标签误差不等同几何误差。
- 每种模型的独立union和sensor-max，1024个匹配起点的原projection/ascent sentinel（无root refinement）。
  mask是离线库availability，不是运行时可见性认证。

首先发 evaluation/summary.md、report.json、manifest.json，以及两组validation.jsonl。
最终预期是更好的边界校准且无明显求解/总体场回归，不是只看新增loss下降。
**没有自动PASS/晋升规则**。不以Case026必须换sensor成功作为门槛；FOV不等于LOS或执行。
禁止把checkpoint、cache、原数据、outputs日志git-add到仓库。

## 本地验证范围

见TEST_RESULTS.md。这里的本地CPU/Gloo测试不替代服务器真实数据/CUDA/NCCL smoke。
本轮代码只服务网络实验；主线A模型适配、执行预算与liveness由本地Codex继续推进。
