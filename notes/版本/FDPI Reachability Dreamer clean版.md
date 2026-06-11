# FDPI Reachability Dreamer clean版

上级入口：
- [[版本索引]]
- [[项目地图]]

相关概念：
- [[Gp可达性风险估计]]
- [[FDPI regime分区]]
- [[dual policy高风险采样]]

## 项目定位

- 当前仓库是 FDPI Reachability Dreamer 的 clean standalone 版本，服务 IsaacLab / UR3 Lite 任务。
- 项目目标不是单纯做 reward-cost 加权，而是构建“可达性风险估计 + 安全关键采样 + 分区策略更新”的 Dreamer 训练闭环。
- 默认启动链路是 `scripts/train.sh` 调用 `fdpi_reachability_dreamer/train.py`，再进入 `fdpi_reachability_dreamer/trainer.py` 的联合训练循环。

## 核心组件

- `ContinuousCostWorldModel`：在 Dreamer world model 上增加 force head 和 continuous cost head，用 latent 表征预测动态、奖励、接触力和安全代价。
- `GpReachabilityCritic`：估计主策略下未来触达危险区域的风险，是主策略 FDPI regime 更新使用的主要安全信号。
- `GdRiskCritic`：评估 dual policy 延续风险，主要服务 dual policy 的高风险数据生成。
- `DualPolicy`：从主策略初始化，通过 imagined risk return 或 max risk 目标学习产生更高风险、更有信息量的动作。
- `FDPIReplayBuffer`：除常规 Dreamer 序列外，额外保存连续代价、二值代价、极端代价、底部力、force excess 和数据来源。

## 当前算法主线

- `Gp.TargetType` 当前为 `n_step_reachability_td`。
- `Gp.CostKey` 当前为 `binary_cost`。
- reachability target 是 horizon 内 discounted binary cost 的最大值，再与 horizon 后 bootstrap Gp 风险取最大；`done` 边界会截断未来 cost 和 bootstrap。
- `configs/reachability_gp.yaml` 中当前 `ReachabilityH` 为 3；README 中提到的 H=5 更像历史说明或测试覆盖，不应直接当作当前实验配置。
- 主策略更新在 `MainFDPIRegime.StartStep` 之后启用，按照 `Gp` 风险把 imagined latent-action 分为 feasible、critical、infeasible，分别调整 reward loss 与 risk loss 权重。

## 训练闭环

- warmup 阶段使用主策略加噪声采样。
- buffer ready 后，world model 做在线 latent 推理，主策略采样主动作。
- dual sampling 根据近期 feasible / critical / infeasible 比例、主策略真实 cost rate 和 dual KL 动态决定混入比例。
- 环境返回后，从 info 或 obs force 中提取 cost force，再计算 continuous/binary/extreme cost。
- replay 采样支持 safety-critical ratio，以提升高 cost、边界 cost 和 dual 来源片段的训练密度。
- 周期性更新 world model、Gp、Gd、dual policy 和主策略，并保存 component checkpoint 与 full state。

## 当前边界

- `docs/research/PROJECT_STATE.md` 和 `docs/research/EXPERIMENT_INDEX.md` 目前还没有真实最佳实验沉淀。
- 外部任务语义依赖 `ur3_lite` IsaacLab extension；没有 IsaacLab 运行环境时，应优先执行静态阅读和 CPU 单测。
- 当前项目理解基于仓库静态阅读，没有默认代表某次完整训练结果。
