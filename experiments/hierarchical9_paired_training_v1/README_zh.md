# R1 paired-label micro-training v1

目的：在不扩大标签缓存、不修改 runtime/URDF/FOV/LOS/Sparse-SCP/VBC/GCDF 的情况下，测试新连续边界标签是否能在固定 held-out paired cache 上带来可重复的学习收益。

四个 arm 同时从同一个 R1 scratch-50k checkpoint 开始，强制 SHA256：
4f395926fa79c29474be8748cef4733ec400d155cd8fadb76c632c2838864002

- old_value：旧值，仅 value loss
- new_value：新值，仅 value loss
- old_value_grad：旧值 + 同一 FD-PASS mask 上的旧梯度
- new_value_grad：新值 + 同一 FD-PASS mask 上的新梯度

公平性：相同 R1 初始权重、相同 320-query audited cache、相同 train/val split、相同 paired masks、相同 Adam/lr/updates、相同确定性 row stream。梯度监督只允许 paired_grad_mask；v3 保证该 mask 只来自实际 gradient_status=PASS。

默认：400 updates，batch 64，lr=2e-5；4 GPU 各跑一个 arm。这个实验只用于 paired held-out 学习筛选，不证明 solver/runtime 已改善。

先测试：
    $HOME/miniforge3/envs/viscdf/bin/python -m unittest discover \
      -s experiments/hierarchical9_paired_training_v1 -p test_paired_train.py -v

提交一次：
    NODE=3090node1 \
    VIS_PYTHON=$HOME/miniforge3/envs/viscdf/bin/python \
    R012_ROOT=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/h9_scratch50k_r012_v1 \
    PAIRED_CACHE=/mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_offline_labels_v3/pilot_20260922_115134/paired_cache \
    bash experiments/hierarchical9_paired_training_v1/submit.sh

只读查看：
    bash experiments/hierarchical9_paired_training_v1/watch.sh
    bash experiments/hierarchical9_paired_training_v1/watch.sh \
      /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_paired_training_v1/latest_train.env logs
    bash experiments/hierarchical9_paired_training_v1/watch.sh \
      /mnt/slurmfs-3090node3/user_data/zsong142/CAREPlanner/outputs/mainline_b/r1_paired_training_v1/latest_train.env summary

完成后看 comparison.md/json。负的 NEW-minus-OLD MAE delta 表示新标签 arm 在 held-out 新标签上 MAE 更低；还要同时看 analytic sensor sign 和 parent drift。
