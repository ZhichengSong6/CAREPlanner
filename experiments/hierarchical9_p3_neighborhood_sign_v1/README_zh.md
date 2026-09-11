# 主线 B：P3，真实 FOV 符号约束的边界邻域实验

基于 `a84ef5329a919c44663c8cf995aef8218906d241`，只新增本目录。
不修改 P0/P1/P2、原网络、原全局目标、runtime、solver、URDF 或安全阈值。
主线 A 继续 V1；P3 是否有收益未知，不自动晋升，也不要求 Case026 必须存在替代解。

## 本轮只检验一个新增监督

P2 在部分困难分支方向上恢复，但实际求解仍退化。P3 保留 P2 的全部设置，
只增加匹配 point-sensor 的、具有真实 FOV 符号标签的有限邻域查询。

```
L_P3 = L_global_original + alpha * (L_P2_anchor + L_side)
alpha = min(successful_update / 500, 1)
L_P2_anchor = Avg_16_sensor_origin(5*f_s(x,q0)^2 + .1*(1-cos(grad_q f_s,n_s)))
L_side = 5 * Avg_32_sensor_origin_intended_side(mean(relu(m_i - y_i*f_s(x,q_i))^2))
```

每个原始锚点仍来自完全相同的 P2 anchor stream；新增：
- 每次更新用独立、可复现 RNG 抽 `r ~ Uniform(.005,.05)` rad，同一锚点生成 `q0-r*n` 和 `q0+r*n`。
- `y = sign(g_s(x,q))` 来自对应 sensor 的**实际解析 conservative FOV**，不是按正负偏移猜标签。
- 新 PairwiseFOV 复用原 FK，逐查询匹配 x；服务器强制比较原 oracle/平面索引，delta 只减一次。
- 出 joint limits 的样本不 clamp、不训练、不补采；`|g_s|<=1e-5 m` 的不确定符号不训练。
  每个尝试和有效数量、限位排除、模糊排除、偏移方向与真实符号相反的数量均记录。
- `m_i=.25*r_i`，数值范围 [.00125,.0125]，是**模型输出域的弱不等式间隔**，
  不是精确 C-space distance 标签，不要求 `f=+/-r`，也不是运行时安全余量。
  正确且满足间隔的样本不再受这一项推动，避免无限增大分类置信度。
- 5、.25 和半径范围是预先固定的试验值，不是已证明最优的权重。
- 32 组按 sensor / bank-offbank / intended offset side 等权，不按真实类别强制补平。
  各组用实际 GLOBAL 有效数归一化，空组贡献零且显式报告；不对 microbatch 均值再平均。
- 邻域样本只监督所选 sensor 输出 `1+s`；没有 union-zero、最近库距离/梯度、
  单位范数、teacher loss 或其它 sensor 的标签。

这轮刷新**径向位置**，没有增加新的空间点或重新生成根，不声称消除了空间泛化差距。
训练只用 cache/train.npz；cache/val.npz 仅监测与评估，不回流。
不修改 P2 原锚点 loss；原全局 Eikonal/tension/consistency/union supervision 都保留。

## 训练与来源

- 原 hierarchical9：shared 30→1024→512→256；9个256→128→128→1 decoder，ReLU。
- 全部1,133,705参数可训练；从同一原 V1 final.pt 独立初始化，不读取P2或smoke权重续训。
- fresh Adam，lr=1e-4，2000次成功更新，500步预热，无scheduler/裁剪。
- 4×3090；每update仍400000 uniform + 4096原锚点，另外8192个邻域尝试。
- 原全局AMP/FP32 loss不变；原锚点和新邻域都FP32；只做一次合并后的optimizer update。
- 旧全局/锚点损失函数原样导入。DDP把最终同步移至最后一个邻域microbatch，
  新的累积驱动经过完整batch梯度和Adam等价测试；没有第二次optimizer.step。
- 成功更新时累计原uniform/anchor哈希，正式final必须与P0/P1/P2完全匹配。
  新邻域另记 proposal 和实际标签/排除哈希，**不宣称新增监督量或GPU耗时与P2相同**。
- AMP重试复用同一批uniform/anchor/neighbor，不重新生成，不计为成功update。
- 每500步及首步记录原全局/边界train/val指标，另记录固定半径流的邻域train/val混淆统计。
  正式比较只用2000步final；latest是中间快照，此版本没有自动resume。
- 旧checkpoints、缓存及旧源文件都验证哈希与来源，不伪装P3为P2，不绕开原split与身份检查。
  pickle只加载自己信任的项目权重；内容哈希不是任意pickle的安全证明。

## 服务器使用

原B任务结束后，在服务器B worktree同步，不切换本地Codex的A worktree：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git status --short
git fetch origin mainline-b/h9-p3-neighborhood-sign-v1
git switch -c mainline-b/h9-p3-neighborhood-sign-v1 --track origin/mainline-b/h9-p3-neighborhood-sign-v1
git rev-parse HEAD
sha256sum -c experiments/hierarchical9_p3_neighborhood_sign_v1/SHA256SUMS
```

创建分支只执行一次，已有分支/修改时不要强制覆盖。运行中不要改当前worktree。
默认复用：
```
/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
```
可显式 `export P3_REFERENCE_ROOT=...`。不重新生成cache、不重训P0/P1/P2。
需要原P2的正式final及run.json，原引用完整性/设置必须通过。

### 1. Smoke

```bash
bash experiments/hierarchical9_p3_neighborhood_sign_v1/submit.sh smoke
```

只提交一个四卡任务：测试→NCCL数学验证→2次完整规模P3更新→同allocation内cuda:0小规模评估。
无array、无依赖、无自动后续sbatch，不添加--mem；仍需等待账户其它任务释放提交额度。
边界缓存不重建，smoke也使用8192邻域尝试/update。
评估比较V1/P3_smoke，不把2步与2000步当作公平效果比较。
完成标记：
```
[done] p3_training_complete mode=smoke successful_updates=2 ...
[done] p3_evaluation_complete mode=smoke ...
[done] p3_smoke_workflow_complete
```
查看日志：
```bash
REF=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
source "$REF/p3_jobs.env"
tail -n 100 -F "$REF/logs/p3_smoke_${P3_SMOKE_JOB}.out"
sacct -j "$P3_SMOKE_JOB" --format=JobID,State,ExitCode,Elapsed
```
Ctrl+C只结束tail，不取消任务。COMPLETED、0:0和workflow标记全部满足再继续。

### 2. 正式P3

```bash
bash experiments/hierarchical9_p3_neighborhood_sign_v1/submit.sh pilot
source "$REF/p3_jobs.env"
tail -n 100 -F "$REF/logs/p3_pilot_${P3_TRAIN_JOB}.out"
```

独立从V1开始，4卡2000更新，不继承smoke，不提前提交eval。
完成标记 `[done] p3_training_complete mode=pilot successful_updates=2000 ...`。
输出 `P3/final.pt`，格式为 `care_h9_p3_neighborhood_sign_v1`，总更新50000+2000。

### 3. 正式评估

训练成功且allocation退出后：
```bash
bash experiments/hierarchical9_p3_neighborhood_sign_v1/submit.sh eval
source "$REF/p3_jobs.env"
tail -n 100 -F "$REF/logs/p3_eval_${P3_EVAL_JOB}.out"
```

一个单卡任务，匹配比较V1/P0/P1/P2/P3；输出evaluation_p3，不覆盖旧报告。
沿用原fixed-offset profiles、局部/随机分支起点生成顺序和同一solver，
64000 field查询、每条union/max路径1024个planning起点不变。
另加固定连续半径的neighborhood_sentinel，不替代原benchmark、不改变接受条件。
报告包含TP/TN/FP/FN、正/负召回、root_source×failure_stage、逐起点配对、
绝对/条件筛选线性化位移、场/ranking/projection/ascent与耗时。
主线A继续V1；不执行LOS、GCDF/VBC认证、机器人、Gazebo、actual seen或自动晋升。

### 4. 一条命令打包报告

```bash
bash experiments/hierarchical9_p3_neighborhood_sign_v1/submit.sh pack
```

不申请GPU，标准库生成 `$REF/p3_pilot_reports.zip`，包含：
```
evaluation_p3/{summary.md,report.json,manifest.json}
P3/{validation.jsonl,run.json,metrics.jsonl}
REPORT_SHA256SUMS
```
没有权重、数据或逐次大日志。校验ZIP，不覆盖已有包，下载后直接上传即可。

## 故障与验收

所有输出用新目录P3/P3_smoke/evaluation_p3/evaluation_p3_smoke；旧实验不变。
每个阶段有提交锁，重复调用拒绝；明确sbatch拒绝释放锁。运行失败保留现场，无自动重试覆盖。
不要重跑旧submit.sh，不删除旧cache或锁来伪造完成。
主要验收：相比P2减少外侧假阳性/候选FOV失败，相比P0/V1保住实际求解与sensor-max；
边界|f|或sign loss降低不是成功保证，8192额外标签也不保证泛化。
测评仍为反复开发用held-out集合，不宣称独立最终泛化测试或安全证书。
