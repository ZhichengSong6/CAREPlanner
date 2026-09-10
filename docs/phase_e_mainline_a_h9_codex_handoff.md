# CAREPlanner 主线 A：Hierarchical9 V1 本地集成 / Codex handoff

## 本轮任务与边界

这是已完成训练、已完成离线配对评估之后的工程接入任务。当前只推进主线 A：
固定 V1 checkpoint，完善候选生成、分支诊断与 sensing/acquisition liveness。
不要启动主线 B，不续训、不改网络层数或 loss、不重新采集训练集。
不要因为上下文里曾讨论 Frozen Scalar + Kinematic Experts，就实施另一套网络。

本轮默认先完成 A0–A2（本地预检、兼容适配、离线 Case026），给出真实结果后停止。
A3 是后续受控仿真 regression，不在本轮自动启动完整 30-case 或真实机器人。
“停止并报告”不是要求 Case026 必须成功；NO_CLEAR_SENSOR_BRANCH 也是有效诊断结果。
本 handoff 不代表任何代码已在用户本机执行，下载与运行状态要实际检查。

## 已有代码与模型身份

本 handoff 核对代码基线：bf085d57b2b04944f6bc930f6907d8ac070431ee。
训练代码提交：052feda570f310edbc3af2b22e9310c0cc98c2d0。
实际本地 HEAD 可能更新；记录 HEAD/diff，不能强制 reset 到基线。

模型（默认相对仓库根目录）：

| 用途 | 路径 | SHA256 |
|---|---|---|
| 新 hierarchical9 V1 | src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt | 979552db20bc7e20775758b273613532921c5dbf11c480b13597127683c4c199 |
| 原 scalar 基线 | src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt | fea15cb71b278b9d003337d200f3796ccbbe8a59106287a8ea85abb2c89cb0df |
| 原 8-head 基线 | src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt | 43f962729adcd17aa114edb9fc410facbbb97ebe7343f0ad3309fe50d273acdb |

以上三份都是用户 eval 16036 日志中加载的 step=50000 final.pt。新模型约 14MB，
从头训练、没有冻结参数；现在作为运行时固定 V1 使用，不是继续更新其权重。
完整训练 checkpoint 含 optimizer/RNG 等对象；先验 hash，只加载自己信任的文件。
PyTorch/NumPy 兼容问题先使用或适配已有 checkpoint helper，不重装整个 ROS 环境。

架构固定：

```text
[x(3),q(7)] -> [u,sin(u),cos(u)](30)
-> shared 30->1024->512->256
-> 9 independent decoders, each 256->128->128->1
-> [f_union, f0, ..., f7]
```

隐藏层 ReLU；共 1,133,705 参数。输出0是 union，输出1:9才是八个 sensor。
新模型没有变成九个 sensors，不能把 out_dim=9 当成旧 Linear(128,8) 来加载。
联合训练时使用独立 union supervision；当前 V1 保留 baseline consistency/tension。
本轮不要修改这些科学配置。

## 离线证据与不能据此声称的结论

本地下载后优先读 outputs/hierarchical9_eval_full_16036/{manifest.json,comparison.json,summary.md}。
数据来源是用户提供的 eval 16036 输出，不是推测的新实验；field 163840，planning 5120，
四条路径相同样本/起点，FP32，held-out validation（不是独立最终 test）。

| 路径 | MAE | 梯度 cosine | Projection abs(g)<.03 | Asc10 g>=.03 |
|---|---:|---:|---:|---:|
| old scalar | .10082 | .68734 | .63730 | .40977 |
| hierarchical union | .10649 | .67730 | .60234 | .43770 |
| hierarchical sensor max | .09949 | .69990 | .64375 | .40273 |
| old8 max | .13125 | .65880 | .44844 | .41348 |

新 S6/S7 cosine=.59048/.58847；GT-winner-removed ranking=.83278，不是可执行 fallback 成功率。
近离散标签零面子集 abs(target)<.1：S4/S5各37例，S6/S7为0例，null不是0分也不是通过；
不能声称 S6/S7 真实 FOV 边界已验证，不能推断训练从未采到边界或替代解不存在。
离线成功率不含 self-occlusion/轨迹/碰撞认证，也没有 runtime sign-crossing root refinement。
因此 V1 值得受控集成，但没有获准自动替换生产默认路径。

## 不允许回退修改的系统约束

优先级：physical feasibility > collision safety > task tracking > visibility/active sensing。
GCDF 是 hard collision safety；VBC 是未来 swept-body visibility-before-contact；
confidence map 只表达真实 seen/unknown/occupied；VisCDF/NCDF 仅负责 visibility steering；
tracker 是唯一执行 owner。UNKNOWN 可以约束未来规划，不能直接作为 emergency stop。

不修改 self-filter URDF、startup body prior、moving trusted-free bootstrap，
不扩大 ignore_links、不改变 ray padding、不关闭已有碰撞/VBC/几何认证以获得成功。
不把网络正值当成 seen，不创建另一个控制命令发布 owner。
历史 ghost occupied 原因已定位为 Gazebo render/TF pose 不一致，不重新归咎于URDF覆盖。
非零静态姿态诊断不使用 spawn_model -J / SetModelConfiguration，使用 joint controller。

本轮保留全局默认 PER_SENSOR_HYBRID_ENABLED=false；只在明确命名的诊断/仿真配置中启用。
不改默认安全阈值，不执行真实机器人运动，不自动 push/merge，不提交权重、大数据、输出日志。

## 第一轮接法：保留旧 scalar，只替换 sensor 分支实现

```text
obligation + measured q
  -> 原 scalar projection/root/ascent 与 q_zero
  -> 新 hierarchical9 的八个 sensor 输出作 ranking
  -> 选定 sensor，按该分支求值和 7D q 梯度
  -> 原 projection / sign-crossing refinement / ascent
  -> 原 conservative analytic FOV 与 primitive self-occlusion
  -> 合格 q_vis -> 原 Sparse-SCP/VBC/GCDF -> tracker
```

这是变量隔离的工程对照，不是最终必须运行两个网络。
新 union head保留，但本轮不替换旧 scalar、不使用sensor-max作为默认projector。
ranking 继续在原 scalar q_zero 计算；每个 sensor branch 从同一个 measured q 独立开始，
不从 q_zero 或上一失败分支的终点开始。多 obligation 点的原 min 聚合语义保持不变。

可用性 mask：离线 q0 library coverage、运行时 enabled sensors、真实 FOV/LOS 是三回事。
不能通过“预测负值就不尝试”屏蔽需要 acquisition 的分支；也不能假装任意新 x 都有离线mask。
本轮保留已有运行时可用性规则，仅适配模型；遇到不明确规则先记录，不悄悄改变策略。

所有候选失败时返回结构化原因，保留已有回退/重新规划策略。原 scalar fallback 也不能跳过
既有认证。加载失败/NaN不得伪装成可见候选，不得静默随机初始化新模型。诊断应明确失败。

## A0：本地预检（先做）

1. 读取现有 AGENTS.md/局部规则、git status、HEAD、diff，保留用户未提交修改。
   使用独立本地分支（建议 codex/phase-e-h9-v1-mainline-a）；需要更改 .git 时遵守审批。
   不自动 stash/reset/clean/切换掉用户正在工作的分支。与现有改动冲突时报告。
2. python3 scripts/pull_hierarchical9_v1_artifacts.py --repo . --verify-only。
   文件未下载时请用户先运行下载步骤，不读取 SSH 私钥、不索要明文密码。
3. 记录本地 Python/PyTorch/CUDA/ROS/URDF parser 等版本，沿用本地已有运行环境；
   不把服务器 miniforge 路径直接当成本地路径，不全局 pip upgrade，不用 sudo 重装环境。
4. CPU 可做模型/梯度/离线几何测试；CUDA可用再测试CUDA。没有ROS/Gazebo不能宣称做过完整框架测试。
   A0–A2 不需要 7.4GB 训练集；不要因此开始下载/生成数据。

优先阅读的现有文件（先看真实代码再修改）：

- experiments/hierarchical9_scratch_v1/model.py
- experiments/hierarchical9_scratch_v1/evaluate.py
- experiments/hierarchical9_scratch_v1/EVAL_zh.md
- src/care_visibility_cdf/scripts/hierarchical_visibility_cdf_model.py
- src/care_visibility_cdf/scripts/per_sensor_visibility_runtime.py
- src/care_visibility_cdf/scripts/evaluate_direct_vs_projection_ascent.py
- src/care_visibility_cdf/scripts/check_visibility_self_occlusion.py
- src/care_visibility_cdf/scripts/vbc_deadline_waypoint_rolling_impl.py
- src/care_visibility_cdf/scripts/vbc_deadline_waypoint_online_node.py
- scripts/test_phase_e_case026_targeted_per_sensor_fallback.py
- scripts/run_phase_e_case026_targeted_per_sensor_fallback.sh
- scripts/run_phase_e_case026_per_sensor_hybrid.sh

## A1：最小兼容适配与测试

当前 build_per_sensor_model 仅接受旧8-head语义/out_dim=8。扩展为明确识别旧8-head与新H9，
不破坏旧checkpoint和默认行为。通过metadata/format/architecture严格分发，不仅按文件名猜。
new format=careplanner_hierarchical9_scratch_v1；new output_semantics=
hierarchical_union_plus_per_sensor_signed_visibility_cdf；step=50000；out_dim=9。

可优先复用 src 中的 hierarchical_visibility_cdf_model，或用小型无状态wrapper给runtime提供
8-output视图。先验证它与训练模型 state_dict/前向/输入梯度等价，不复制第三份独立架构。
完整新模型始终是9输出；兼容视图只把完整输出1:9暴露给原sensor接口，不删除权重。
保持 q 顺序和 sensor frame 顺序与checkpoint一致，strict=True加载，记录checkpoint SHA。

参数 eval/frozen 仍必须允许原始q求导；禁止在梯度路径用no_grad/inference_mode/detach(h)。
每次 q 更新都重算特征，不能复用旧q特征。FP32为对照；不得用输出梯度之和代替某个fs梯度，
不得将八个sensor梯度平均，也不得把f_s(x,q)>=0全部同时塞进QP改变原问题。

需要实际执行并记录：
- 旧8-head加载和调用回归测试；新checkpoint shape/semantics/step/hash guard。
- H9完整输出1+s与sensor视图输出s逐值一致；s=0和s=7避免off-by-one。
- 各分支对原始7D q梯度与参考模型一致；单点与原多点min聚合都要验证。
- 有限差分只在避开ReLU/分支并列的点对照；不把不可微点误报为加载错误。
- q变化后值/梯度重算；参数冻结时仍能求q梯度；NaN/无效输入/模型缺失明确失败。
- 保留超时/尝试次数/关节限位/失败回退/几何检查语义，测试全branch失败不会误报成功。
- CPU以及可用时CUDA的值/梯度、ranking和实际多点branch耗时；报告warmup后p50/p95及测试配置。
  不能把服务器离线批量吞吐量当成本地在线planner频率。

## A2：原 Case026 targeted 对照（本轮完成后报告）

复用脚本中的完整精度 TARGET / MEASURED_SEED / KNOWN_BLOCKED_QVIS，不另造接近样本。
手动核对用的近似值：

```text
x=[0.10,0.05,0.15]
measured q=[-0.23994,0.75771,-0.30183,-1.62670,0.19068,-0.18052,-0.38080]
known blocked q_vis=[-0.26948,0.75081,-0.26677,-1.87081,0.19022,-0.17289,-0.37924]
```

先确认历史 S4 候选 conservative g约+0.018168，primitive LOS被link3遮挡。
若不复现，检查坐标系/URDF版本/参数/依赖，不修改几何规则强行复现。
然后比较“相同旧scalar + 旧8-head”和“相同旧scalar + 新H9 sensor视图”。

保持 targeted 脚本既有配置：projection_iters=10，damping=.5，epsilon_f=.03，max_step=.25；
root_refine_iters=12，root_tolerance_f=.002；branch_ascent_steps=1，step=.05，max_step=.25；
max_branch_attempts=8，force_first_sensor=4。其它既有fallback选项先读代码并保持一致，记录完整配置。
这只是targeted诊断的预算，不偷偷把在线默认branch attempts同步改成8。
FOV conservative=50/66度，z=.20/.70，delta=.01；self-filter及ray参数不改。

逐分支记录 measured seed、rank/order、f/g、梯度范数、projection/root/ascent迭代、最终q、
FOV检查、LOS结果/命中link、拒绝原因、时间。raw trace存JSON，摘要存Markdown，模型/代码/URDF标识齐全。
结果按实际返回NO_CLEAR_SENSOR_BRANCH或合格候选；不得保证S6/S7必然成功，不能把有限搜索失败说成不可行证明。
若成功，只能声称“找到通过本次FOV/primitive LOS检查的候选”，不是轨迹/碰撞/执行安全已认证。
本脚本不需要Gazebo、不运行planner、GCDF、VBC或confidence map动态。

本轮输出：变更文件与理由、可复现命令、所有测试PASS/FAIL/NOT_RUN、两模型逐sensor对照表、
输出目录、性能数据、未解决问题，以及是否建议进入A3。不自动commit/push权重或改动默认开关。

## A3：下一轮受控仿真框架 regression（需下一条明确指令）

A1接口/回退测试通过后，再按已有case列表选择少量empty-world及sensing/liveness cases；
从仓库/已有日志找13-case清单，找不到就说明，不凭记忆编造case编号。
Case026未成功不阻止其它受控case诊断，但必须保留失败记录，不能宣称该模式问题关闭。
小范围baseline vs H9对照后，才考虑扩到13-case，再到完整30-case。
记录 proposed -> FOV pass -> LOS pass -> trajectory certified -> deadline met -> actual seen。
记录REPAIR/NORMAL转换、planner周期/超时、tracking和safety计数，并确保tracker仍是唯一owner。
离线不能获得的trajectory/seen等字段用NOT_RUN/N/A，不能填0假装测过。
不同时换scalar projector、branch solver、loss和安全阈值；每轮只改有证据支持的最小一处。
框架固定V1；未来V2必须有独立目录/hash和同协议验收后再晋升，不热更新运行中的权重。

## 本机下载及 Codex 启动

请先在本机打开现有CAREPlanner根目录（不要在SSH服务器终端执行）：

```bash
git status --short
git branch --show-current
# 确认当前为master且没有需要保护的冲突，再执行：
git pull --ff-only origin master
python3 scripts/pull_hierarchical9_v1_artifacts.py --repo . --dry-run
python3 scripts/pull_hierarchical9_v1_artifacts.py --repo .
```

下载脚本只用Python标准库+git/OpenSSH，不需要PyTorch。它读取远端文件哈希，核对日志固定的
三份模型SHA256，临时文件下载后再校验并原子发布，跳过内容相同文件，拒绝覆盖内容不同文件。
还下载manifest/comparison/summary/train_args，不下载7.4GB数据、latest/best，不改变runtime。
需要能连10.120.17.131的校园网/VPN/既有SSH路由；凭证仅在本机SSH提示中输入。
支持--host SSH_ALIAS；跳板机/端口用现有~/.ssh/config，不关闭host-key验证。
末尾出现[done] artifacts_ready_for_mainline_a才表示本机下载完整通过。
--verify-only仅离线检查模型hash、报告结构/关联，不宣称重新验证了服务器报告内容。

在激活本地已有运行环境后启动：

```bash
codex --version
codex --sandbox workspace-write --ask-for-approval on-request \
  "请读取现有AGENTS.md和docs/phase_e_mainline_a_h9_codex_handoff.md。先给出简短计划，然后只完成A0-A2：预检、最小H9 sensor兼容适配、单元测试和原Case026离线对照。保留旧scalar和所有安全规则，默认hybrid关闭，不重训、不运行真实机器人、不自动启动A3或push。按handoff输出实际测试结果和阻塞项。"
```

Codex本地新会话不应被假定自动拥有这段Chat对话；以仓库文件及真实下载报告为上下文。
遵守现有AGENTS.md，不创建覆盖它的全局规则。若CLI版本不认识参数，先检查codex --help，
不要改用绕过所有权限的选项。SSH/ROS端口等沙箱阻塞按需批准具体命令，不禁用全部保护。
恢复同一仓库最近的本地会话可用codex resume --last；仍读取最新状态报告。

本次交付仅包含下载辅助脚本、其8项本地合成测试和本handoff。
测试不连接服务器，不读取真实checkpoint；没有在用户本机完成传输或执行A0-A3。
