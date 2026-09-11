# 主线 B：P2，仅删除新增边界 Eikonal

基于已完成 P0/P1 的 commit `0c50a7998a32c9484f78681731da4239f5c10f6c`。
只新增本目录。旧实验、model、global objective、runtime、URDF、安全模块一律不改。
V1 继续用于主线 A。P1 是负面对照，不续训、不覆盖；P2 是否改善由配对评估决定。

## 唯一训练干预

```
P1: L_global + alpha * Avg_sensor,origin(5*f(q0)^2 + .1*(1-cos(grad_q f,n)) + .1*abs(norm(grad_q f)-1))
P2: L_global + alpha * Avg_sensor,origin(5*f(q0)^2 + .1*(1-cos(grad_q f,n)) + 0.*abs(norm(grad_q f)-1))
```

这里删除的只是**新增边界** Eikonal。原全局目标内的 Eikonal、tension、union supervision、
consistency 均原样保留。边界梯度范数仍计算、记录，但不产生该项训练梯度。
代码直接调用旧 `calibration.boundary_loss`，传入 `BoundaryWeights(5,.1,0)`，
以及旧 `train_pilot.perform_update` / `validate`，不复制重写数学更新函数、不伪装为 P1。

结构不变：30→1024→512→256，一个主干；9 个 256→128→128→1 decoder；
ReLU，1,133,705 个参数全部可训练。输出 [union,S0,...,S7]；边界样本只监督对应 sensor。

从相同 V1 `final.pt` 权重独立开始，不从 P0/P1 继续，不恢复旧 Adam。
固定 2000 次成功更新，fresh Adam lr=1e-4，warmup=500，400000 global queries/update，
4096 boundary queries/update，4 卡，microbatch_x=250，FP16 global / FP32 boundary。
不加层、不冻结、不换采样、不加双侧训练标签、不调 solver。

## 对照与来源检查

默认复用服务器目录：
```
/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
```
保留其中 cache、P0、P1、evaluation 和原 jobs.env 不变。新增：
```
P2_smoke/                 # 2-update plumbing test, also uses full 4096 boundary batch
P2/                       # independently starts V1 for 2000 updates
evaluation_p2_smoke/
evaluation_p2/
p2_jobs.env
logs/p2_{smoke|pilot|eval}_{JOBID}.out
```

`protocol.py` 拒绝没有完成的旧 checkpoint、错误缓存、改变的原有源文件、不同训练参数。
全部原 P0/P1 源文件 SHA256 必须与 checkpoint 一致；新目录拥有独立 fingerprint。
新训练参数由 P0 记录复制，只改变 arm/output；smoke 额外把 steps 改为2。
每步 RNG 和样本累计哈希复用原实现。正式 P2 完成时必须与 P0/P1 四卡的 global / boundary
样本累计哈希完全一致，否则不写成功 final.pt。

新 checkpoint 使用单独 format `care_h9_p2_no_boundary_eikonal_v1` 和 arm=P2。
它是 50000+2000 updates 的微调，不是新的从头训练。
模型格式、缓存、原 optimizer 规则、模型输出与源文件检查均保留；不能把 P2 改名当 P1。
旧 checkpoint 使用 pickle，仅加载你自己生成并信任的项目文件；哈希是完整性检查，不是任意 pickle 的安全证明。

## 服务器更新（本地 A 线不要切分支）

先确认原 B 线任务已结束，保存已有改动，再在服务器 B worktree 执行：
```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner-mainline-b
git status --short
git fetch origin mainline-b/h9-p2-no-boundary-eikonal-v1
git switch -c mainline-b/h9-p2-no-boundary-eikonal-v1 --track origin/mainline-b/h9-p2-no-boundary-eikonal-v1
sha256sum -c experiments/hierarchical9_p2_no_boundary_eikonal_v1/SHA256SUMS
```
创建分支只执行一次；已存在时不要强制覆盖。运行中不修改此 worktree。
没有数据准备阶段，不复制或重新生成27,656个训练锚点和1,708个验证锚点。

## 三次单任务提交

每次调用**只有一个 sbatch**，无数组、无提前排队的下游依赖。遵守 AssocMaxSubmitJobLimit。
脚本不主动取消其它任务；若额度被其它任务占用，等待它们退出后再提交。

### 1. P2 smoke
```bash
bash experiments/hierarchical9_p2_no_boundary_eikonal_v1/submit.sh smoke
```
1个4×3090任务：单元测试、4-rank NCCL 数学检查、2次完整global+4096-boundary更新，
训练进程结束后，在同一allocation的cuda:0运行小规模V1/P2_smoke评估。
这是端到端代码检查，不是与2000步P0/P1的效果比较，不宣称smoke流哈希匹配2000步。
最后必须出现：
```
[done] p2_training_complete mode=smoke successful_updates=2 ...
[done] p2_evaluation_complete mode=smoke ...
[done] p2_smoke_workflow_complete
```

### 2. 正式 P2（smoke 完整成功且任务退出后）
```bash
bash experiments/hierarchical9_p2_no_boundary_eikonal_v1/submit.sh pilot
```
4×3090，2000 updates，从V1重新开始，不读取smoke权重。保存 `P2/final.pt`。
完成日志：`[done] p2_training_complete mode=pilot successful_updates=2000 ...`。
没有自动启动评估，等待任务退出再提交下一项。

### 3. 正式配对评估
```bash
bash experiments/hierarchical9_p2_no_boundary_eikonal_v1/submit.sh eval
```
1×3090，重新在同样的held-out样本上评估V1/P0/P1/P2。
延续原来的field/planning helper与逐sensor solver；固定solver预算与接受条件。
报告在 `evaluation_p2/`，不覆盖原 `evaluation/`。
默认64,000 field queries、每条union/max路径1024个planning起点、每sensor最多32个cache支持点用于局部/随机分支测试。

## 查队列和日志

每次提交打印绝对日志路径。也可使用：
```bash
REF=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_cal_pilot_20260910_183735_099nu5
source "$REF/p2_jobs.env"
tail -n 100 -F "$REF/logs/p2_smoke_${P2_SMOKE_JOB}.out"
# pilot提交后：tail -n 100 -F "$REF/logs/p2_pilot_${P2_TRAIN_JOB}.out"
# eval提交后：tail -n 100 -F "$REF/logs/p2_eval_${P2_EVAL_JOB}.out"
```
`tail`的Ctrl+C不取消任务。输出目录/提交锁存在会拒绝重复提交。
sbatch被明确拒绝时释放提交锁，不损坏cache/P0/P1；后续不要调用旧submit.sh。
运行失败或中断保留现场，本版本没有自动resume或覆盖重试。

## 新增解释性指标（不改变验收阈值）

`report.json` 在旧指标外增加：
- 真实解析g的正负类人数、TP/TN/FP/FN、正类/负类召回；不把正法向offset自动当正标签。
- 有符号和绝对线性化零点位移分布、定义覆盖率、条件数筛选覆盖率。
  `abs(slope)>1e-6,norm>1e-8,abs(cos)>=.1`只是解释性筛选，不证明线性化在该距离有效。
- root_sources、root_source×failure_stage、P0/P1/V1对P2的逐起点配对结果。
- boundary_samples.npz 保存原值/梯度范数/cosine/法向斜率和样本身份，便于不靠均值推断。

不放宽FOV、安全、joint limits或root阈值，不运行LOS/轨迹/Gazebo/actual seen，不自动晋升模型。
请返回正式 `evaluation_p2/{summary.md,report.json,manifest.json}` 和 `P2/validation.jsonl`。

## 验证范围

见 TEST_RESULTS.md。合成CPU/Gloo测试不能替代服务器真实V1/cache/URDF/CUDA AMP测试。
P2是否有效未知；本轮只检验去掉新增边界范数项的作用，不预先保证改善。
