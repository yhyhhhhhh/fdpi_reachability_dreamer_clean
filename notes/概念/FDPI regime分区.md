# FDPI regime分区

上级入口：
- [[概念索引]]
- [[项目地图]]

相关版本：
- [[FDPI Reachability Dreamer clean版]]

相关概念：
- [[Gp可达性风险估计]]
- [[dual policy高风险采样]]

## 核心含义

- FDPI regime 分区是主策略更新阶段使用的风险分段机制。
- 当前实现根据 `Gp` 风险值 `g`、失败概率阈值 `Pf` 和缓冲区间 `Cg`，把 imagined latent-action 分为 feasible、critical、infeasible。

## 当前分区

- feasible：`g < Pf - Cg`，主要保留普通 Dreamer reward 优化。
- critical：`Pf - Cg <= g < Pf`，保留 reward，同时逐步加入风险项。
- infeasible：`g >= Pf`，使用更强的风险惩罚，降低策略进入高风险区域的倾向。

## 代码位置

- 分区损失实现：`fdpi_reachability_dreamer/agent.py`。
- 训练调用位置：`fdpi_reachability_dreamer/trainer.py` 的 `train_agent_step`。
- 配置位置：`configs/reachability_gp.yaml` 的 `FDPIRegimeDreamer.MainFDPIRegime` 与 `RiskCritic`。
