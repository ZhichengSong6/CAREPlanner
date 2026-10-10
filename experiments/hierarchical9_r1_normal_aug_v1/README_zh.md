# R1 原训练 + 单项解析边界法向方向增广 v1

## 对比回答的问题

**保留 R1 原来的全空间旧离散距离与梯度方向监督，只加入很小的解析边界法向方向增广，能否比冻结原 R1 更好地引导 FOV solver？**

训练设置与原 R1 保持一致：
- 同一个 private-tail H9 网络，随机初始化 seed=0，不加载旧 checkpoint。
- 50,000 成功 Adam updates；每步 4,000 个 x × 100 个共享随机 q，总共 400,000 组合。
- **运行原 R1 的 R012/main 和 upstream base/objective**；保留原 5×SDF +0.1×梯度方向 +0.01×绝对 Eikonal +0.01×tension、union loss 与 consistency。
- LR=0.001、原 ReduceLROnPlateau、FP16 AMP 与 overflow 同批重试、4×3090 DDP。
- 原训练 x/q StreamChain 的 50k SHA256 必须与旧 R1 完全一致；调度仍只使用旧 R1 原 loss。

**唯一新增监督**：每个 GPU rank、每个 sensor 采 32 个已有解析 V4 边界锚点（合计 4×8×32=1,024/step），仅监督 (1-cosine(∇q f_s(x,q*), n_s))，新增权重 0.01；不要求新边界值等于零，不监督新的距离、符号 margin、单位法向斜率或成对差分。

这些数据全部来自既有的 V4 training cache boundary zero-offset 记录；不生成新数据，不修改 cache。新边界训练样本与原 R1 训练 x-index 相交、排除 R1 的 validation x-index。V4 验证边界仅作为额外诊断，不影响原 R1 val 或学习率调度。

原 R1 主干仍为每步 400k 输入，50k 共 200 亿；边界方向增加 5,120 万个锚点监督，且包含高阶导数及一次额外参数梯度同步，**不称为严格等算力，只是固定 R1 原始训练预算后的受控增广**。

## 为什么不直接微调 R1

同点评估已发现旧 R1 对新局部单位距离标签的值尺度与零值目标不一致。直接附加强制零值/单位斜率可能破坏原引导能力。本实验仅增加一个小权重的解析方向目标，是待验证假设，不保证获益。

## 三项严格验收

1. 验证旧 R1 final.pt SHA256 为 4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002，旧 R1 完成 50k，V4 cache manifest SHA256 为 060eef93a0030800c8ef2265258c7dff850944ab9c5df52382adb98cb506f7cc。
2. 新 worker 读取原 R1 的 x 数组与原 V4 x-index，明确要求数值匹配和不使用旧 R1 val x 训练。
3. 完成后新 arm 的原 R1 全空间 x/q stream SHA256 必须等于旧 R1 50k；否则标记 INVALID，不接受等采样流声明。

## 命令

更新原 GitHub 分支后，在仓库根目录执行：

    export VIS_PYTHON="$HOME/miniforge3/envs/viscdf/bin/python"
    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh train

这是**单个完整四卡正式训练**，默认 Slurm 3 天时限、排除 3090node1，不含独立 GPU pilot。如果 Slurm 超时/中断但已经保存本实验 latest.pt，只能在旧 job 退出后执行 resume：

    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh status
    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh log
    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh resume

仅恢复本新实验 checkpoint 的 optimizer、scheduler、GradScaler、全部 rank RNG、边界样本 update 序号与原 training-stream SHA。不得从 R1/V2/PS checkpoint resume。

训练正式完成后：

    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh verify
    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh summary
    bash experiments/hierarchical9_r1_normal_aug_v1/run.sh eval

eval 是**单独一次四卡评估作业**：原 R1 对新增 arm 的 final.pt，在既有、冻结的 1963 starts 上，用完全相同的 runtime_probe（尤其 epsilon_f=0.03）评估，不扫阈值。原 R1 成功数必须精确为 local 618/683、uniform 673/1280、total 1291/1963。报告逐 sensor/场景和 paired gains/losses、失败阶段；这份复用多次的 starts 是回归集，不是新的最终泛化集。

## 训练监控

原 R1 监督 metrics 保持，另在每100步训练以及每次验证的 boundary_direction_diagnostic 中分别记录八个 sensor：
- 解析边界 normal cosine（理想1）；
- q 梯度 norm；
- 边界值绝对大小（只观察，不监督为零）；
- sensor-head 第一层 ReLU 平均激活率和至少活跃一次的神经元数；
- 增广项的参数梯度范数（训练 monitor）。

不在验证期间用该指标更改模型权重，不用它替代原 R1 val objective。ReLU 激活和梯度监控仅为诊断，并不单独证明因果。

原运行时 FOV/LOS、GCDF/VBC、安全流程均不改变。学习器只是 proposer；解析 FOV 仍作独立判断。CPU 测试不替代 CUDA/NCCL 真实训练确认。
