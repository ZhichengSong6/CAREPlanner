# 主线 B：真实 FOV 边界标签审计 + 逐 sensor 求解测试

本分支只新增本目录，不修改训练器、V1 模型、主线 A runtime、URDF、GCDF/VBC 或 tracker。
不提交权重/7.4GB 数据、不运行机器人、不训练 V2、不改变 PER_SENSOR_HYBRID_ENABLED 默认值。
代码基线：dcde98b2dd75b1f590d34b4c2e7e1bee97b66bc6。

## 这次回答什么问题

1. 真正的单 sensor FOV 边界在哪里？原库距离标签与 V1/旧8-head 在那里是否正确？
2. 同样的 x/sensor/起点和原 runtime 预算，新旧分支能否进入该 sensor 的真实 FOV？

只输出证据，不根据几个平均数自动决定重训/替换模型。
本诊断不包含 self-occlusion、Case026 replay、路径碰撞认证、deadline 或 actual seen。
这些字段是 NOT_RUN，FOV_PASS 不等于可执行或安全。主线 A 的本地 Codex 独立继续。

## 文件

- core.py：离散标签、边界精修、法向合格条件、分位数、逐分支场/梯度。
- oracle.py：沿用已有 Torch FK，构造六个归一化 FOV 平面；与现有 visibility_g_batch 逐值/active-plane 校验。
- runtime_probe.py：继承原 PerSensorVisibilityRuntime 的求解方法，仅注入固定模型/限位/mask/计数。
- audit.py：数据/模型身份检查、固定样本、两项测试、完整 JSON/JSONL 结果。
- test_audit.py：数值单元测试、合成端到端流程、服务器真实 FK/runtime import 测试。
- submit_3090node3.sh：1 x RTX3090 Slurm；不申请 --mem，不需要四卡，不启动训练。

需要原 viscdf 环境（Python >=3.10、PyTorch 2.x、已有 URDF/几何依赖），不重装 ROS/CUDA。
只接受 eval16036 已验证的两份 50000-step final.pt（先 SHA256 再反序列化）：
H9=979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199
old8=43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb
不需要旧 scalar；本轮测指定 sensor，不改总体 projector。只加载自己信任的 checkpoint。

## 使用独立 worktree，不打扰主线 A 或服务器现有目录

以下命令在服务器执行。主线 A 的本地工作目录不用切分支。

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
git status --short
git fetch origin mainline-b/h9-boundary-audit-v1:refs/remotes/origin/mainline-b/h9-boundary-audit-v1
git worktree add -b mainline-b/h9-boundary-audit-v1 \
  ../CAREPlanner-mainline-b origin/mainline-b/h9-boundary-audit-v1
cd ../CAREPlanner-mainline-b
mkdir -p outputs/mainline_b_logs
sha256sum -c experiments/hierarchical9_boundary_audit_v1/SHA256SUMS
```

上面 worktree add 只执行一次。若分支/目录已存在，先 git worktree list 确认位置；
不要 -B/reset --hard/clean -fd，不覆盖本地 Codex 的修改。后续在这个独立目录 ff-only pull 本分支。

新 worktree 不会自动有数据/权重。脚本使用 --artifact-root 指向原 CAREPlanner 的文件：
默认 /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner。
DATA/H9_CHECKPOINT/EIGHT_CHECKPOINT 可显式覆盖路径，但模型身份仍校验，不自动猜 checkpoint。
URDF 与源码来自 B worktree，URDF 内容哈希必须与 V1 训练记录一致。

## 先 smoke，再 full

```bash
sbatch --export=ALL,AUDIT_MODE=smoke --time=00:45:00 \
  experiments/hierarchical9_boundary_audit_v1/submit_3090node3.sh
squeue -u zsong142
# 将真实的任务号放入 JOBID，不要原样输入尖括号占位符：
JOBID=你的任务号
tail -f outputs/mainline_b_logs/h9_boundary_${JOBID}.out
```

Smoke：每 sensor 两个 held-out x、每点一个库锚点/一次切向 off-bank 尝试、一个随机起点；
使用真实 final.pt/数据/几何。先跑 --require-repo 测试，再执行 margin/label/solver parity。
检查 [preflight] 各项 PASS 和最后 [done] mainline_b_boundary_audit_complete。
root/normal/offbank 样本可能被拒绝：这是诊断结果，不是应当放宽阈值的理由。
如果某 sensor 的有效边界为0，相关指标是 null/N/A，不能宣布它已通过。

通过后（Ctrl+C 仅退出 tail）：

```bash
sbatch --export=ALL,AUDIT_MODE=full \
  experiments/hierarchical9_boundary_audit_v1/submit_3090node3.sh
```

Full：每 sensor 最多64个唯一 held-out x（有该sensor库支持），每点两个不同库锚点，
每锚点一次切向新边界尝试，每点两个独立 uniform q。
局部 solver 只用每点第一个锚点的 bank_refined/offbank_refined 两组、0.02/0.05 rad 外侧起点。
拒绝不合格样本后不补采来凑成功数。日志与 generation_counts 记录分母/原因。
默认附带18步解析 g_s normalized ascent对照（ANALYTIC_CONTROL=0可关）。
完整运行资源时限8小时仅是 Slurm 上限，不是运行时间预测；先看 smoke 的真实耗时。

只跑边界部分可加 AUDIT_STAGE=boundary；仅预检使用 AUDIT_MODE=preflight。
已有 MODE/OUT/STEPS 等训练变量不控制本脚本。AUDIT_OUT 可指定新的输出目录，非空目录拒绝覆盖。
CONDA_SH/CONDA_ENV 可指定已有环境路径（默认 $HOME/miniforge3，viscdf）。

## 测试定义：不要把标签误差和网络误差混在一起

### 三组边界

bank_raw：原 valid_fov 库样本，先如实检查 g_s 残差。
bank_refined：对同一个样本用解析 g_s 的 Newton+回溯精修，记录每次尝试/残差。
offbank_refined：沿规则边界的有效关节切空间扰动总范数0.10 rad，再用解析 g_s 精修；
距原库小于0.001 rad的点不作为 offbank 接受。它们是切向局部生成，不是独立均匀样本。
精修仅找零面，不求证最近 C-space 边界；没有用网络选取/过滤对其有利的测试样本。

边界残差 <=1e-5 m、最小两个平面差 >=1e-4 m、法向范数 >=1e-6、
active关节距限位 >0.002 rad，且 h=0.001 rad 中心差分平面不切换/相对误差<=0.05，
才进入 regular normal 子集；其它样本仍记录 g、预测值、拒绝原因。
这些是诊断筛选参数，不是主线 A 安全阈值变更。

在 regular 点处检查原库距离、网络 f 和 raw/masked 梯度与解析单位法向的对齐。
沿 +/-[.005,.01,.02,.05] rad 法向采样，重新检查真实符号、plane switch、线性化误差；
超限就排除，不 clamp 后仍称原法向样本。offset不是全局精确 signed distance。
局部求解只使用负侧确实 outside 且对应正侧确实 inside 的成对样本，属于条件化子集；
正侧 witness 不是 LOS、路径可达或碰撞安全证明。

标签检查沿用 legacy eps=1e-8 的平方距离 floor：原库精确点的 legacy幅值约1e-4 rad、梯度为0。
另报不带 floor 的 bank_distance；精确库点/近邻并列不强行计算库梯度法向 cosine。
正常数值先与仓库实际 decode_per_sensor_distance_and_grad 校验，禁止把 g_s(m) 直接当 distance(rad)。

### 求解

只比较 old8 与 H9 的 sensor 输出1:9，不比较 union，也不平均各sensor梯度。
直接复用 _optimize_branch/_projection_step/_refine_branch_root/_ascent_step/_clamp，
不构造 runtime 的 ROS/LOS对象、不改方法实现；预检对原基类作数值等价检查并记录方法源码哈希。
预算：projection10/damping.5/epsilon_f.03/max_step.25；root12/tol_f.002；
ascent1/step.05/max_step.25；root-not-found 时原有8步best-effort ascent也保留。
和原runtime一样使用chain-masked梯度；边界审计另外报告raw梯度及inactive-joint leakage。

每个方法从相同 measured-free 诊断起点独立开始；随机起点不限于边界样本成功的点。
只把实际 g_s< -1e-5 m 的起点计入 outside 子集，并记录被排除数量。
返回 q_zero 不等于找到根：initial_positive、tolerance、sign-crossing、best-effort 分开报告。
最终按该sensor自己的真实g_s判定，不使用union FOV替代选定sensor，也不限制必须先找到预测root才能报FOV通过。
记录值/梯度调用数、完整迭代q、限位clamp次数、阶段和耗时。
解析对照目标/阈值/成本不同，不宣称同目标同时间公平竞赛；有限预算找不到不是不可行性证明。

## 输出和验收

默认输出在原artifact目录下：
outputs/mainline_b/h9_boundary_full_JOBID/
- manifest.json：配置、两份checkpoint SHA/step、代码/URDF哈希、设备、preflight、状态。
- report.json、summary.md：按sensor/边界来源/offset/起点类别分组的统计与有效分母。
- generation.jsonl：每个库精修/offbank尝试/失败和起点排除，无成功后补采。
- boundary.jsonl：raw/refined/offbank逐样本位置、原库距离、网络值、raw/masked法向指标。
- profiles.jsonl：边界两侧实际g/标签/f/梯度/符号，包含限位排除。
- solves.jsonl：新旧配对求解完整轨迹、真实FOV、调用数、耗时、可选解析对照。
- partial_report.json：逐sensor进度或失败时保留，不可冒充完整结果。

先发 summary.md + report.json + manifest.json；需要定位特定失败再筛 JSONL，不必粘贴全部日志。
空样本=null/N/A，相关计数仍显示；主表网络边界指标仅对regular样本计算，bank gap列对该组所有尝试计算。
多个样本共享x和父锚点，不能作为独立伯努利试验直接宣称显著性；不根据单次测试微调阈值制造成功。
V1固定不变；本轮不自动开始V2训练。主线A也不等待这里全部通过才继续受控诊断。

后续合并：先保持本地 Codex 的主线A修改已提交/妥善保存，再fetch本分支、合并或cherry-pick。
本分支只新增独立目录，合并风险较小但不保证零冲突；如果主线A改了被调用的solver，必须重跑preflight并比较方法hash。
