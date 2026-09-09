# Hierarchical9：不加深，全部从头训练

## 固定结构与训练目标

```text
(x,q) 10D -> [u,sin(u),cos(u)] 30D
-> shared trunk: 30 -> 1024 -> 512 -> 256
-> 9 independent decoders, each: 256 -> 128 -> 128 -> 1
-> [f_union, f0, ..., f7]
```

隐藏层 ReLU，输出层 Linear；1,133,705 个参数全部训练。不加载旧 scalar，
不冻结主干，不加深，不新增 Softplus、kinematic experts 或近边界采样。
Union 是独立 decoder，直接使用 union 标签和 GT winning-sensor 梯度监督。

本轮保留 handoff 中 hierarchical9 baseline loss，不同时改结构和监督：

```text
L_h = 5 * mean((f_h-target_h)^2)
    + 0.1 * mean(1-cosine(grad_q f_h,target_grad_h))
    + 0.01 * mean(abs(norm(grad_q f_h)-1))
    + 0.01 * mean(norm(grad_q(sum_j grad_q_j f_h))^2)
L = L_union + mean_active_sensors(L_s)
    + 0.1 * mean((f_union-max_available_sensors(f_s))^2)
```

每个 head 在自身有效样本内平均。Unavailable sensors 在 max 前 mask。
Eikonal 为绝对值；tension 为原来的 `||H^T 1||^2`，不是完整 Hessian 范数。
本轮没有新增零边界/法向 loss；consistency/tension 的开关留作独立消融。

## 文件及依赖

`model.py` 定义网络和逐行值/7D 输入梯度接口；`objective.py` 定义 loss；
`train.py` 是四卡 DDP 训练器；`test_synthetic.py` 是无数据集合成测试；
`submit_3090node3.sh` 提交 Slurm；`TEST_RESULTS.md` 记录本地验证范围。

复用仓库已有的数据、FOV oracle、距离/梯度标签及 Cartesian sampling：

```text
src/care_visibility_cdf/scripts/train_signed_visibility_cdf_pairwise_replace.py
src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf.py
src/care_visibility_cdf/scripts/train_per_sensor_visibility_cdf_ddp.py
```

接口核对版本：`841a4a992386388aa0cdb00c885f3c607c8e997c`。
需要 Python >=3.10、支持 CUDA 的 PyTorch 2.x 和已有 Pinocchio 等依赖。
不包含大型数据集，不采新 Gazebo 数据，不重装 CUDA。

## 数值及可复现性说明

修复 AMP 下 400,000 分母转 FP16 变成 inf 的问题：loss 运算、求和、分母用 FP32。
按整个 global batch 的实际有效标签数归一化，乘 world_size 抵消 DDP 平均；
累积 microbatch 时不再除以分块数量。非有限 loss 停止训练；FP16 参数梯度溢出
缩小 GradScaler scale，并重试同一批数据，不把跳过更新算作 optimizer update。

保留 Adam 与原 ReduceLROnPlateau 设置，监控 train loss，不新增 gradient clipping。
Validation 使用固定采样并保存/恢复训练 RNG；不宣称与旧 trainer 随机序列逐位相同。
Checkpoint 记录 repo SHA、源码/URDF 哈希、数据路径/大小/mtime；后两者不是数据内容哈希。

## 服务器同步

```bash
ssh zsong142@10.120.17.131
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
git status --short
git pull --ff-only origin master
git rev-parse HEAD
sha256sum -c experiments/hierarchical9_scratch_v1/SHA256SUMS
mkdir -p outputs/per_sensor_training_logs
unset RESUME PRETRAINED RESUME_TRAINING
```

在已有 master 分支执行；若有本地修改或未跟踪同名文件，先保留这些文件，不要强制覆盖。
Slurm 在运行脚本前打开日志，必须先创建 log 目录。
默认 conda：`$HOME/miniforge3/etc/profile.d/conda.sh`，环境 `viscdf`；
路径不同可传 `CONDA_SH` 和 `CONDA_ENV`，不必创建新环境。

## 先冒烟，再正式训练

```bash
sbatch --export=ALL,MODE=smoke --time=00:30:00 \
  experiments/hierarchical9_scratch_v1/submit_3090node3.sh
squeue -u zsong142
# 将 <JOBID> 替换为 sbatch 返回的编号：
tail -f outputs/per_sensor_training_logs/hierarchical9_scratch_<JOBID>.out
```

Smoke 先运行 CUDA 合成测试和四卡 NCCL 等价测试，再运行真实数据两次完整
400,000-pair optimizer update 及 validation。它使用独立输出目录。
检查合成测试 PASS、随机初始化、全部参数可训练、train/val loss 有限，以及两步完成。
不要用 step1 cosine 判断最终模型质量。

通过后，重新从 seed0 随机初始化提交正式训练，不读取 smoke checkpoint：

```bash
sbatch --export=ALL,MODE=train \
  experiments/hierarchical9_scratch_v1/submit_3090node3.sh
```

固定默认配置：3090node3、4 x RTX 3090、16 CPU、3 天，不申请 `--mem`；
50,000 次成功 update、seed0、Adam lr=1e-3、FP16 AMP；
global x=4000、所有 rank 共用 q=100，即 400,000 pairs/update；
local x/GPU=1000、microbatch x/GPU=250；validation global x=512、q=100。
GPU 显存和速度尚未实测。OOM 时仅缩小分块，保持 global batch 不变：

```bash
sbatch --export=ALL,MODE=smoke,MICROBATCH_X=125,VAL_MICROBATCH_X=64,DECODE_X_CHUNK=32 \
  --time=00:30:00 experiments/hierarchical9_scratch_v1/submit_3090node3.sh
```

通过后正式任务使用相同分块变量，并改为 MODE=train。

## Checkpoint 与恢复

正式输出：`src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/`，
包含 `train_args.json`、`metrics.jsonl`、`latest.pt`、`final.pt`。
非空目录拒绝覆盖；新实验显式指定新的 OUT。正式比较只用 50,000-step `final.pt`，
不使用 smoke final，不选择 best.pt。latest.pt 在 step1/每5000步/最后一步原子写入。

只有集群中断后恢复同一次训练时，才显式设置：

```bash
sbatch --export=ALL,MODE=train,RESUME_TRAINING=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/latest.pt \
  experiments/hierarchical9_scratch_v1/submit_3090node3.sh
```

只接受本训练器的 checkpoint；恢复模型、Adam、scheduler、scaler、各 rank RNG。
保持相同核心配置、AMP、数据路径、microbatch 和四卡；STEPS 是总步数，不是追加步数。
仅加载自己信任的 checkpoint，完整恢复使用 Python pickle。

## Runtime 不自动切换

本实验不修改 planner、GCDF/VBC、URDF/self-filter、tracker 或旧 checkpoint；
继续保持 `PER_SENSOR_HYBRID_ENABLED=false`。输出索引为 union=0、sensor s=1+s。
旧 8-head runtime loader 不能直接读取新模型，验证后再适配。
`forward_union` / `forward_sensor` 保留到原始 q 的梯度，不增加 QP 决策变量；
点集 min 聚合、sensor 可用性、branch solver、FOV/self-occlusion 与安全认证留在 runtime。
训练数据 availability mask 不是运行时可见性或安全证书。

先比较 union field/projection，再检查 sensor 梯度/排序，再做 Case026 targeted fallback；
通过后才切换 runtime 和进行后续 Phase-E qualification。
