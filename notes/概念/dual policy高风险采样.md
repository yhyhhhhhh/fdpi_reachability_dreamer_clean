# dual policy高风险采样

上级入口：
- [[概念索引]]
- [[项目地图]]

相关版本：
- [[FDPI Reachability Dreamer clean版]]

相关概念：
- [[Gp可达性风险估计]]
- [[FDPI regime分区]]

## 核心含义

- dual policy 是从主策略初始化的辅助策略，用于生成更高风险、更有信息量的样本。
- 它不是最终部署策略，而是服务 replay 数据覆盖和风险 critic 学习的采样机制。

## 当前训练方式

- dual policy 可使用 `imagined_risk_return` 目标：在 world model 中 rollout，最大化预测 cost 与 terminal `Gd` 风险回报。
- 也支持 `max_risk` 目标：直接寻找 `Gd` 评分更高的 latent-action。
- loss 中包含 KL 到主策略的项，避免 dual policy 与主策略分布偏离过大。

## 当前采样方式

- 训练循环根据近期 feasible / critical / infeasible 比例、主策略真实 cost rate 和 dual KL 动态计算 dual sampling ratio。
- dual 采样得到的数据会在 replay 中标记 `source=dual`。
- replay 和 critic loss 可对 dual 来源、高 cost、边界 cost 样本加权。

## 代码位置

- dual policy 网络：`fdpi_reachability_dreamer/dual_policy.py`。
- dual 更新：`fdpi_reachability_dreamer/dual_update.py`。
- dual 采样比例：`fdpi_reachability_dreamer/sampling.py`。
- 训练循环调用：`fdpi_reachability_dreamer/trainer.py`。
