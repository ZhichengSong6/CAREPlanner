# Hierarchical9 final.pt 的离线评估

新增 evaluator 不改训练、runtime、URDF 或 checkpoint。仅加载自己信任的训练 checkpoint。
先比较同一批样本上的旧 scalar、新独立 union、新 sensor-max，以及旧 8-head max。
新模型必须为本实验完整 50,000-update 的 final.pt；拒绝 smoke、latest、best。
旧 checkpoint 是独立复跑的基线，不把历史打印值直接当作本次配对结果。

## 同步与提交

在服务器 master 分支、保留本地修改的前提下执行：

```bash
cd /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner
git status --short
git pull --ff-only origin master
mkdir -p outputs/per_sensor_training_logs
sha256sum -c experiments/hierarchical9_scratch_v1/EVAL_SHA256SUMS
# 确認是正式训练完成，而不是 smoke updates=2：
grep -F '[done] successful_optimizer_updates=50000' \
  outputs/per_sensor_training_logs/hierarchical9_scratch_15999.out
```

先用最终 checkpoint 跑小规模 evaluator smoke（不训练）：

```bash
sbatch --export=ALL,EVAL_MODE=smoke --time=00:30:00 \
  experiments/hierarchical9_scratch_v1/submit_eval_3090node3.sh
# 把 <JOBID> 换成返回的编号，不用 *.out 通配符：
tail -f outputs/per_sensor_training_logs/hierarchical9_eval_<JOBID>.out
```

出现 `[done] offline_evaluation_complete` 后提交正式离线评估：

```bash
sbatch --export=ALL,EVAL_MODE=full \
  experiments/hierarchical9_scratch_v1/submit_eval_3090node3.sh
```

本评估只申请 3090node3 上 1 张 RTX 3090，不是 4 卡 DDP；不设置 --mem。
原 viscdf conda 环境、原数据、原 URDF。CONDA_SH/CONDA_ENV 可覆盖激活路径。
默认 checkpoint：

```text
src/care_visibility_cdf/checkpoints/hierarchical9_scratch_seed0/final.pt
src/care_visibility_cdf/checkpoints/exp1_yiming_k500_fov_signed/final.pt
src/care_visibility_cdf/checkpoints/per_sensor_e2e_fullbatch_seed0/final.pt
```

路径不同用 H9_CHECKPOINT/SCALAR_CHECKPOINT/EIGHT_CHECKPOINT 覆盖；不自动搜索/猜测模型。
旧 8-head 确实不存在时，显式传 SKIP_OLD_EIGHT=1；仍比较旧 scalar 和新模型的两种 union。
这会缺少旧8-head的配对 sensor 改进对照，不能宣称四方法评估齐全。
默认输出有 Job ID，EVAL_OUT 可覆盖，非空目录拒绝覆盖。训练的 OUT/MODE/AMP 变量不用于本脚本。

## 输出与协议

```text
outputs/hierarchical9_eval_full_<JOBID>/
    manifest.json
    comparison.json
    summary.md
    samples.npz
```

manifest 记录 checkpoint SHA256、step、配置、源码哈希、URDF 哈希与设备；
还包括 checkpoint 保存的 final validation。数据大小/mtime 不是数据内容哈希。
comparison 包括所有 field、各 sensor 梯度、近目标零面子集、ranking 和 planning 统计。
summary 是便于阅读的对比表；samples 保存每批 x 索引、实际 x、q 及部分规划候选，可复查配对性。

Full：field=20*128*64=163,840 queries；planning=10*8*64=5,120 starts。
保留训练时 seed0、1000 个空间点的 validation split；在其中有放回抽样。
独立 NumPy RNG 固定 seed123，planning 用124，模型数/加载顺序不改变采样；
因此各模型在同一轮严格同样本，但不宣称复现旧 evaluator 的逐位随机序列。
这仍是 held-out validation 评估，不是从未使用过的独立最终 test 集。
FP32，关闭 AMP/TF32；参数不求梯度，但保留对原始7D q的梯度。

复用 compare_scalar_vs_per_sensor_apples_to_apples.py 的既有求解与成功率定义：
projection=10步、damping=0.5、max_step=0.25；ascent=0.05，记录1/3/5/10步。
本 benchmark 不做 runtime 的 sign-crossing root refinement，不得混称完整 branch solver。
成功率依据解析 conservative FOV g，不能用预测 f 过零代替真实 g 检查。
新 union 取输出0，新 sensor-max 取输出1:9并mask；旧8-head同样mask。
字段的 near-boundary 子集为 abs(离散标签)<0.1 rad，不是真实连续FOV边界标签。
梯度范数误差统一为 abs(norm(predicted)-norm(target))。

ranking 的 fallback 指删除 GT winner 后是否选中 GT 次优分支；没有运行遮挡检查，
不是可执行替代分支的成功率。top-k 与 fallback 都屏蔽 unavailable sensor。
mask 来自离线 q0 库覆盖，不是运行时感知可见性或安全证书。

## 决策顺序

先看 summary 的 union projection/ascent，再看 S6/S7 梯度和 ranking。
这些结果回答 union head 相比同模型 sensor-max 是否有实际收益；不能回答
“去掉 union 监督后重训八个 heads 是否更好”，后者需要另一个训练消融。
第一轮结果之后再适配并跑原 Case026 targeted self-occlusion/fallback 检查。
当前没有修改旧8-head runtime loader，不能直接把新checkpoint路径塞进去。
继续保持 PER_SENSOR_HYBRID_ENABLED=false；本次不跑完整Phase-E，不产生安全合格结论。

## 本地测试范围

已通过 Python编译、Shell语法及 test_evaluate.py 的6项CPU合成测试：
head输出/7D梯度等价、masked-max梯度等价、ranking/fallback掩码、
field统计分块一致性、样本与模型初始化解耦、final checkpoint guard/序列化。
没有在此环境读取服务器真实final.pt或7.4GB数据，也没有执行CUDA评估。
服务器 evaluator smoke 是真实环境验证，不是已完成的测量。
