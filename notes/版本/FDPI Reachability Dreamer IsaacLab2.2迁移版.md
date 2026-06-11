# FDPI Reachability Dreamer IsaacLab2.2迁移版

上级索引：
- [[版本索引]]

相关版本：
- [[FDPI Reachability Dreamer clean版]]

相关概念：
- [[Gp可达性风险估计]]
- [[dual policy高风险采样]]
- [[FDPI regime分区]]

## 版本定位

本版本是从 clean standalone 版本迁移出的 IsaacLab 2.2 / IsaacSim 5.0 独立训练主链路，目标是在 SurgicalRobot5 任务中继续训练带安全可达性风险估计的 FDPI Reachability Dreamer。

## 入口与任务

- 训练入口：`scripts/train_isaaclab22.sh`
- 代码包：`fdpi_reachability_dreamer_isaaclab22/`
- 默认配置：`configs/reachability_gp_isaaclab22.yaml`
- 输出目录：`ckpt_isaaclab22/`
- IsaacLab 路径：`/home/yhy/IsaacLab5`
- conda 环境：`isaaclab`
- 任务 gym id：`SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1`
- 任务 entry point：`surgical_robot5.env:SurgicalRobot5HeadPipeGraspGoalDreamerForceV1Env`

## 与 clean 版的关系

该版本复制核心 Dreamer/FDPI 训练代码并改为自引用导入，不直接复用旧 `fdpi_reachability_dreamer/` 包。第一阶段只迁移训练主链路，不迁移 `tools/` 下评估、normalizer 收集和 benchmark 工具。

## 关键差异

- 使用 IsaacLab 2.2 的 `isaaclab.app.AppLauncher` 与 `isaaclab_tasks.utils.parse_env_cfg`。
- 启动脚本显式激活 `isaaclab` conda 环境，并加入 IsaacLab5 source 与 `surgical_robot5` extension 到 `PYTHONPATH`。
- 适配 IsaacLab 2.2 `DirectRLEnv.step()` 自动 reset 行为，在 wrapper 中缓存终止步观测，避免 Dreamer replay 把 reset 后观测误记为 episode 最后一帧。
- 默认关闭旧 UR3 Lite normalizer，避免跨任务观测统计混用。

## 当前待验证

- `isaaclab` 环境已补齐 `yacs`，并已通过 0 步小规模烟测：
  `WANDB_MODE=disabled NUM_ENVS=2 SAMPLE_MAX_STEPS=0 bash scripts/train_isaaclab22.sh`
- 最近 0 步烟测运行目录：`ckpt_isaaclab22/fdpi-reachability-dreamer-isaaclab22/20260611_110508/`
- 非零步烟测应归档到 `experiments/2026-06-11_IsaacLab22迁移烟测/`。
