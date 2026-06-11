# 项目状态

## 当前活跃版本

- FDPI Reachability Dreamer clean standalone 版本。
- 当前主训练入口：`scripts/train.sh` -> `fdpi_reachability_dreamer_isaaclab22/train.py` -> `fdpi_reachability_dreamer_isaaclab22/trainer.py`。
- 当前默认配置：`configs/reachability_gp_isaaclab22.yaml`。
- 当前默认任务：`SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1`，对应 entry point `surgical_robot5.env:SurgicalRobot5HeadPipeGraspGoalDreamerForceV1Env`。
- `scripts/train_isaaclab22.sh` 仅作为兼容转发脚本调用 `scripts/train.sh`；旧 IsaacLab 1.4 / UR3 Lite 启动接口已不再保留为默认入口。
- IsaacLab 2.2 迁移版已同步到 `fdpi-1`，远端项目目录为 `/root/gpufree-data/fdpi_reachability_dreamer_clean`，SurgicalRobot5 目录为 `/root/gpufree-data/surgical_robot5`，远端 IsaacLab 为 `/root/IsaacLab` 2.2.1。

## 当前主要目标

- 在 IsaacLab 2.2 / SurgicalRobot5 任务中训练带安全可达性风险估计的 Dreamer 系统。
- 使用 `GpReachabilityCritic` 的 n-step reachability TD 目标，把未来危险触达信号传播到当前 latent-action。
- 通过 `GdRiskCritic` 和 dual policy 生成安全关键数据，提高高风险、边界风险片段在 replay 和训练中的覆盖。
- 通过 FDPI regime 将主策略 imagined rollout 分为 feasible / critical / infeasible，并在 reward 优化和风险约束之间动态权衡。

## 当前最佳实验

- 实验名称：暂无收敛意义上的最佳训练结果。
- 最近工程验证：`2026-06-11_4090训练加速参数对比`。
- 实验目录：`experiments/2026-06-11_4090训练加速参数对比/`。
- 结果简述：在 RTX 4090 上完成 warmup 后带更新 profiling；baseline `213.49 env steps/s`，4090 balanced `520.89 env steps/s`，4090 fast `659.81 env steps/s`。fast 相对 baseline 总吞吐 `3.09x`，采样+更新相关耗时 `3.40x`，10M env steps 估算约 `4.21` 小时。该结果证明加速配置有效，但不代表最终收敛质量。

## 当前主要问题

- 世界模型预测精度仍缺少系统验证。需要分别评估主策略数据分布与对偶策略数据分布下的动态、reward、continuous cost / binary cost 预测误差，明确主策略下的预测精度是否足以支撑 imagined rollout，以及对偶策略采样进入高风险区域后是否会显著放大模型误差。
- 对偶策略的数据采集收益尚未被实验证明。需要验证 dual policy 生成的危险数据是少量极端样本，还是覆盖了多样化的安全边界状态；同时需要与随机噪声或随机探索数据对比，确认引入对偶数据是否真的改善世界模型在高风险/对偶数据分布上的学习效果。
- 安全约束与任务奖励之间的平衡尚未稳定。当前 FDPI regime 中 reward 项、安全风险项和不同风险区间权重的协调关系还需要实验校准，重点观察主策略是否会因安全项过强而保守失效，或因 reward 项过强而忽视 reachability 风险。


## 已废弃方案或版本

- 暂无明确废弃方案记录。README 表示共享组件已从早期实验折叠进当前 clean package，但没有保留可引用的旧版本说明。

## 下一步计划

- 首先是检测代码运行的速度，确定训练速度最快能到什么程度，争取在最大限度利用gpu下加快训练
- 优化策略熵的问题，思考这部分如何解决|
- 针对距离预测大的问题，思考如何解决
- IsaacLab 2.2 迁移版已在 fdpi-1 完成非零步短训练烟测；下一步可运行更长的云端训练速度测试和稳定性测试。
- 4090 fast/balanced 训练配置已完成 warmup 后带更新 profiling；fast 总吞吐约为 baseline `3.09x`，但 W&B 观察显示 fast 接近 `4M env steps` 仍未学会夹取，policy recovery 到约 `2.45M env steps` 仍未恢复成功率。下一步优先运行 `experiments/2026-06-12_baseline工程优化长训/`，保持 baseline 更新密度，仅启用工程优化，验证学习能力是否恢复。
- `2026-06-11_危险区域世界模型50比50采样` 的 4090 fast+safety50 长训已停止，停止原因是策略学习明显偏弱；随后启动的 `2026-06-11_4090策略更新强度恢复验证` / W&B run `2hceqcms` 在约 `105k env steps` 异常停止，日志无 traceback。下一步应先排查停止原因，再重启 policy recovery 长训。
- `surgical_robot5` 已补充最小 `pyproject.toml` 以支持 pip 25 editable 安装；IsaacLab 2.2 中的 `quat_rotate` / `quat_rotate_inverse` deprecation warning 已通过替换为 `quat_apply` / `quat_apply_inverse` 清理。
- 新增实际训练实验时，在 `experiments/YYYY-MM-DD_中文实验名称/` 下归档配置、命令、日志、指标和实验记录。
- 对比 `ReachabilityH=3` 与 H=5、`binary_cost` 与 `continuous_cost` 作为 `Gp` target 信号的效果。
