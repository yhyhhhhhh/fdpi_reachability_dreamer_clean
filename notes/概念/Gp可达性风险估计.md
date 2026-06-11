# Gp可达性风险估计

上级入口：
- [[概念索引]]
- [[项目地图]]

相关版本：
- [[FDPI Reachability Dreamer clean版]]

相关概念：
- [[FDPI regime分区]]
- [[dual policy高风险采样]]

## 核心含义

- `GpReachabilityCritic` 用于估计主策略动作之后未来触达危险状态或高代价区域的风险。
- 在当前 clean 版本中，`Gp` 的主要训练目标是 n-step reachability TD，而不是普通的一步 cost TD。

## 当前实现要点

- 入口文件：`fdpi_reachability_dreamer/risk_critics.py`。
- 当前配置：`configs/reachability_gp.yaml` 中 `Gp.TargetType: n_step_reachability_td`，`Gp.CostKey: binary_cost`。
- target 计算：在 horizon 窗口内对 discounted binary cost 取最大值，再与 bootstrap 的 target Gp 风险取最大值。
- `done` 边界会阻断边界之后的 future cost 和 bootstrap risk。
- `Gp` 使用 double critic，但 reduce 方向是 `maximum`，符合保守风险上界的直觉。

## 与主策略的关系

- 主策略 FDPI 更新读取 `Gp` 风险，把样本划分为 feasible、critical、infeasible。
- feasible 区域主要追求 reward。
- critical 区域保留 reward，但逐步增强风险约束。
- infeasible 区域使用更强的风险惩罚，避免策略继续推向高风险动作。

## 仍需实验验证

- 当前 `ReachabilityH=3` 是否足够覆盖任务中的危险传播延迟。
- `binary_cost` 与 `continuous_cost` 哪个更适合作为 reachability target 的基础信号。
- reachability positive weighting 是否会改善边界区域学习，或导致风险过度保守。
