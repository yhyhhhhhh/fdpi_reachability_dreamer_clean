# 修改记录

## 2026-06-12

### LOGIC_CHANGE：尊重 Gp.Enable 关闭态的主策略训练路径

修改内容：
- 训练循环新增 `gp_enabled` gating。
- 当 `FDPIRegimeDreamer.Gp.Enable=false` 时，rollout 不再用 Gp critic 计算 `g_main` 和 feasible window，主策略更新也不再把 Gp critic 传入 `train_agent_step(...)`。

修改原因：
- 新增无 FDPI / 无对偶采样 baseline 时，需要确保关闭 Gp 后不会隐式使用 reachability critic 参与 rollout 诊断或主策略更新路径。

影响：
- 默认配置中 `Gp.Enable=true`，旧实验行为不变。
- 仅影响显式设置 `Gp.Enable=false` 的配置；这类配置会更接近普通 Dreamer actor-critic baseline。

验证方式：
- 已通过：配置加载检查确认新 baseline 中 `Gp.Enable=false`、`MainFDPIRegime.Enable=false`、`DualSampling.Enable=false`。
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/trainer.py`。

相关实验：
- `experiments/2026-06-12_128环境无FDPI无对偶采样基线/`

---

### CONFIG_CHANGE：新增 128 环境无 FDPI 无对偶采样 baseline 配置

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_baseline_engineering_no_fdpi_no_dual_128env.yaml`。
- 该配置继承 `baseline_engineering_dual_kl01_128env` 的 128 env、batch 128 和 baseline-engineering 训练密度。
- 显式关闭 `MainFDPIRegime.Enable`、`Gp.Enable`、`Gd.Enable`、`DualSampling.Enable` 和 `DualUpdate.Enable`。
- 保留 `WorldModelEval.Enable=true`，便于和 FDPI+dual run 直接比较 world model 预测质量。

修改原因：
- 需要一个真实可跑的对照训练入口，用于判断 FDPI 与对偶采样是否降低主任务区域建模质量和主策略学习速度。

影响：
- 默认训练配置不变。
- 新配置会训练普通 Dreamer 主策略和 world model，不更新安全 critic / dual policy，不向 replay 混入 dual 来源样本。

验证方式：
- 已通过：配置加载检查确认 128 env 与关键关闭项生效。
- 已通过：`bash -n experiments/2026-06-12_128环境无FDPI无对偶采样基线/运行命令.sh experiments/2026-06-12_128环境无FDPI无对偶采样基线/启动tmux长训.sh`。

相关实验：
- `experiments/2026-06-12_128环境无FDPI无对偶采样基线/`

---

### EXPERIMENT_ONLY：新增 128 环境无 FDPI 无对偶采样 baseline 训练入口

修改内容：
- 新增实验目录 `experiments/2026-06-12_128环境无FDPI无对偶采样基线/`。
- 新增 `运行命令.sh`、`启动tmux长训.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json`、`配置快照/` 和 `代码变更.patch`。

修改原因：
- 将无 FDPI / 无对偶采样 baseline 沉淀为可复现短命令，便于和当前 KL003 对偶探索诊断 run 做直接对比。

影响：
- 可通过 `bash experiments/2026-06-12_128环境无FDPI无对偶采样基线/启动tmux长训.sh` 启动独立 tmux 长训。
- 输出、日志和 checkpoint 独立保存到该实验目录。

验证方式：
- 已通过：实验目录结构和配置快照已创建。
- 已通过：启动脚本 shell 语法检查。

相关实验：
- `experiments/2026-06-12_128环境无FDPI无对偶采样基线/`

---

### LOGIC_CHANGE：新增低频 WorldModelEval 主/对偶分布评估

修改内容：
- 新增 `FDPIRegimeDreamer.WorldModelEval` 配置块，默认 `Enable=false`，支持一键开启/关闭训练中低频 world model 评估。
- 新增 `run_world_model_eval(...)`，每跨过 `StartStep + k * EverySteps` 阈值后在 replay 上评估一次，避免 `env_steps` 与 `1M` 不整除导致漏评估。
- 新增 `WMEval/{split}/...` 指标，按 `all/main/dual` 以及可选 cost 区间统计 posterior、one-step prior、open-loop 多 horizon 的 reward/cost/extreme/force/done 预测误差。
- 新增 `WMEvalGap/dual_minus_main/*`，用于直接观察 dual 分布相对 main 分布的 world model 误差差距。
- 在 `configs/reachability_gp_isaaclab22.yaml` 中显式加入关闭态配置；在 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl003_128env.yaml` 中开启 `WorldModelEval.Enable=true`。

修改原因：
- 当前需要判断对偶策略采样是否让 world model 对 main 任务区域或 dual 高风险区域的预测质量发生变化，仅看训练 loss 和 replay 全局统计不够直接。
- 低频评估可以在不明显影响长训吞吐的情况下，持续观察 world model 对 main/dual 分布的拟合、one-step dynamics 和 open-loop imagination 质量。

影响：
- 默认关闭，不改变旧配置训练行为。
- 开启后每 `1M env steps` 左右额外采样 replay 并执行一次 `torch.no_grad()` 评估，只记录 W&B 指标，不更新模型、不修改 replay、不保存额外文件。
- 当前 KL003 诊断配置的新 run 会自动记录 `WMEval/*` 指标；已在运行中的 run 不会自动生效，需要重启。

验证方式：
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/trainer.py fdpi_reachability_dreamer_isaaclab22/train.py tests/test_reachability_gp.py`。
- 已通过：`python -m unittest tests.test_reachability_gp`。
- 已通过：轻量 fake replay / fake world model 调用 `run_world_model_eval(...)`，确认会产出 main/dual posterior 与 open-loop 指标。

相关实验：
- `experiments/2026-06-12_128环境KL003对偶探索诊断/`

---

### LOGIC_CHANGE：新增最近窗口 source-aware 真实 cost 诊断

修改内容：
- 新增 `SourceCostStatsWindow`，按最近窗口同时统计 main / dual 来源样本的 `binary_cost`、`continuous_cost`、`extreme_cost` 和来源占比。
- 训练循环每个环境步将当前 `source`、`continuous_cost`、`binary_cost`、`extreme_cost` 写入最近窗口。
- 按 `LogEverySteps` 新增 W&B 指标 `RecentCost/*`，包括 `main_binary_cost_rate`、`dual_binary_cost_rate`、`main_continuous_cost_mean`、`dual_continuous_cost_mean`、`main_extreme_cost_rate`、`dual_extreme_cost_rate`、`main_ratio`、`dual_ratio`、`main_count`、`dual_count` 和 `window_count`。
- 窗口长度复用 `DualSampling.FeasibleRatioWindow`，当前主线配置为最近 `10000` 个环境样本。

修改原因：
- `Replay/main_cost_rate` / `Replay/dual_cost_rate` 是全 replay buffer 历史统计，main 来源会包含 `1M env steps` 前未启用安全优化和 dual 对照的数据。
- 直接用全 replay 的 main/dual cost rate 判断当前 dual 是否更危险不够公平，需要同一最近窗口下的 source-aware 真实 cost 对比。

影响：
- 不改变训练采样、replay 内容、loss 或优化步骤。
- 新指标用于诊断当前 rollout 中 main 与 dual 的真实危险率，避免被早期 replay 历史污染。
- 已在运行中的训练不会自动出现该指标，需要重启新 run 后生效。

验证方式：
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/sampling.py fdpi_reachability_dreamer_isaaclab22/trainer.py`。
- 已通过：`python -m unittest tests.test_reachability_gp`。

相关实验：
- 后续重启 `experiments/2026-06-12_128环境低KL对偶探索长训/` 或启动 `experiments/2026-06-12_128环境KL003对偶探索诊断/` 后生效。

---

### CONFIG_CHANGE：新增 128 环境 KL 0.03 对偶探索诊断配置

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl003_128env.yaml`。
- 该配置继承 `baseline_engineering_dual_kl01_128env`，仅将 `FDPIRegimeDreamer.DualUpdate.KLCoeff` 从 `0.1` 降到 `0.03`。
- 保持 `DualUpdate.Type=imagined_risk_return`、128 env、batch 128、更新触发频率和 Gd/Dual 延后到 `1M env steps` 等设置不变。

修改原因：
- 当前 W&B run `riz27lm5` 中 `Dual/kl_loss` 仍显著大于 `Dual/policy_loss`，且 `Replay/dual_cost_rate` 低于 `Replay/main_cost_rate`。
- 用户希望先采用最小改动继续降低 KL，而不是切换到 `max_risk` 目标。

影响：
- 默认配置不变。
- 新实验可用于判断仅降低 KL 是否能提升 dual 真实危险采样率。

验证方式：
- 已通过：配置文件静态检查确认新配置仅覆盖 `DualUpdate.KLCoeff=0.03`。
- 已通过：`bash -n experiments/2026-06-12_128环境KL003对偶探索诊断/运行命令.sh experiments/2026-06-12_128环境KL003对偶探索诊断/启动tmux长训.sh`。

相关实验：
- `experiments/2026-06-12_128环境KL003对偶探索诊断/`

---

### EXPERIMENT_ONLY：新增 128 环境 KL 0.03 对偶探索诊断入口

修改内容：
- 新增实验目录 `experiments/2026-06-12_128环境KL003对偶探索诊断/`。
- 新增 `运行命令.sh`、`启动tmux长训.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json` 和配置快照。

修改原因：
- 将 KL 0.03 对偶探索诊断沉淀为可复现短命令，便于和当前 KL 0.1 的 128 env run 做直接对照。

影响：
- 可通过 `bash experiments/2026-06-12_128环境KL003对偶探索诊断/启动tmux长训.sh` 启动独立 tmux 长训。
- 输出、日志和 checkpoint 独立保存到该实验目录，不影响当前 KL 0.1 长训。

验证方式：
- 已通过：实验目录结构和配置快照已创建。
- 已通过：启动脚本 shell 语法检查。

相关实验：
- `experiments/2026-06-12_128环境KL003对偶探索诊断/`

---

### LOGIC_CHANGE：降低正式训练中的非关键指标同步开销

修改内容：
- 为 world model、Gp/Gd critic、主策略 FDPI update、dual update 增加 `return_metrics` 开关。
- 在非日志步不再构造大量 Python float 指标，避免每个 update 内部反复 `.item()` 触发 GPU 同步。
- 主训练循环仅在 `LogEverySteps` 记录主策略/对偶策略关键优化指标，在 `DetailedLogEverySteps` 才记录 world model、Gp/Gd、batch composition、replay stats 等细项。
- `Train/*` 更新计数和 ratio 改为走低频 `step_logger`，减少 W&B 高频 flush。
- 对偶更新在非日志步仍保留最小 `kl_to_main` 返回值，保证 dual sampling ratio 控制逻辑不变。

修改原因：
- 当前瓶颈主要在 warmup 后更新部分，且用户主要关注主策略与对偶策略优化情况，不需要每个 update 都记录世界模型和 critic 的大量细粒度指标。
- `.item()` 会造成 GPU/CPU 同步；减少非关键指标同步可以提升正式长训吞吐，同时尽量不改变算法更新比例。

影响：
- 算法 loss、采样比例、更新步数和 replay 内容不变。
- W&B 中 world model、Gp/Gd、replay 细项变为低频记录；主策略与对偶策略的核心指标仍按 `LogEverySteps` 保留。
- 已在运行中的训练不会自动应用该代码，需要重启新 run 才生效。

验证方式：
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/trainer.py fdpi_reachability_dreamer_isaaclab22/world_model.py fdpi_reachability_dreamer_isaaclab22/risk_critics.py fdpi_reachability_dreamer_isaaclab22/agent.py fdpi_reachability_dreamer_isaaclab22/dual_update.py`。
- 已通过：`python -m unittest tests.test_reachability_gp`。

相关实验：
- 后续重启 `experiments/2026-06-12_128环境低KL对偶探索长训/` 或低 KL 对偶探索长训后生效。

---

### CONFIG_CHANGE：正式 baseline engineering 配置关闭 timing 同步

修改内容：
- 将 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml` 中 `FDPIRegimeDreamer.Performance.TimingLogEverySteps` 从 `8192` 改为 `0`。
- 将 128 env 派生配置 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl01_128env.yaml` 的 `TimingLogEverySteps` 同步改为 `0`。

修改原因：
- `_TrainTimer` 需要 `torch.cuda.synchronize()` 才能准确计时，适合 profiling，但不适合正式长训常开。

影响：
- 正式 baseline-engineering 系列新 run 默认不再记录 `Timing/*`。
- 需要 profiling 时仍可通过配置或命令行单独打开 timing。

验证方式：
- 已通过：配置文件静态检查确认 `TimingLogEverySteps=0`。

相关实验：
- `experiments/2026-06-12_128环境低KL对偶探索长训/`

---

### CONFIG_CHANGE：新增 128 环境低 KL 对偶探索配置

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl01_128env.yaml`。
- 该配置继承 `baseline_engineering_dual_kl01`，将 `NumEnvs`、`BatchSize` 和 `ImagineBatchSize` 提高到 `128`。
- 保留 `TrainModelEverySteps=256`、`TrainAgentEverySteps=256`、`ModelUpdate=4`、`AgentUpdate=4` 和 `Gp/Gd/Dual UpdateSteps=2`，避免重复 fast 配置削弱策略学习的问题。
- 保留 `DualUpdate.KLCoeff=0.1`，以及 `Gd/DualSampling/DualUpdate StartStep=1000000`。

修改原因：
- 64 env 版本在 RTX 4090 上仍有 GPU 利用率空间。
- 用户希望启动环境数更多、硬件利用率更高的训练，同时避免 fast 模式因更新密度下降导致夹取学习变慢。

影响：
- 默认配置不改变。
- 新配置会提高单次 rollout 和更新 batch，对显存和更新耗时要求更高；若稳定，可后续再尝试 `BatchSize=256`。

验证方式：
- 已通过：加载 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl01_128env.yaml`，确认 `NumEnvs=128`、`BatchSize=128`、`ImagineBatchSize=128`，且 replay batch 约束满足。
- 已通过：启动 128 env 长训后确认 IsaacLab 环境数为 `128`，W&B run 为 `riz27lm5`，训练进入 rollout / update 循环，日志未出现 traceback、replay assertion 或 OOM。

相关实验：
- `experiments/2026-06-12_128环境低KL对偶探索长训/`

---

### EXPERIMENT_ONLY：新增 128 环境低 KL 对偶探索长训入口

修改内容：
- 新增实验目录 `experiments/2026-06-12_128环境低KL对偶探索长训/`。
- 新增 `运行命令.sh`、`启动tmux长训.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json` 和配置快照。

修改原因：
- 将 128 env 高利用率训练沉淀为可复现短命令，避免用临时环境变量启动后无法追踪参数。

影响：
- 可通过 `bash experiments/2026-06-12_128环境低KL对偶探索长训/启动tmux长训.sh` 启动独立 tmux 长训。
- 输出、日志和 checkpoint 独立保存到该实验目录。

验证方式：
- 已通过：`bash -n experiments/2026-06-12_128环境低KL对偶探索长训/运行命令.sh experiments/2026-06-12_128环境低KL对偶探索长训/启动tmux长训.sh`。
- 已通过：tmux 会话 `dual_kl01_128env_long` 已启动并进入训练循环；实验状态和长期索引已更新。

相关实验：
- `experiments/2026-06-12_128环境低KL对偶探索长训/`

---

### LOGIC_CHANGE：将 Gd 更新延后到对偶策略启用后

修改内容：
- 新增 `FDPIRegimeDreamer.Gd.StartStep` 配置项，默认补全值为 `0`，保持旧配置兼容。
- 在 `configs/reachability_gp_isaaclab22.yaml` 和 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml` 中设置 `Gd.StartStep=1000000`。
- 在 `fdpi_reachability_dreamer_isaaclab22/trainer.py` 中增加 Gd 更新门控：`env_steps < Gd.StartStep` 时不采样 Gd batch，也不执行 `gd_critic.update()`。

修改原因：
- Gd 主要服务 dual policy 的高风险数据采集和 dual update；当 `DualSampling` 与 `DualUpdate` 已延后到 `1M env steps` 时，1M 前持续更新 Gd 对主策略学习收益有限。
- 在不降低 world model、Gp 和主策略更新密度的前提下，减少前 `1M env steps` 的更新开销。

影响：
- 后续 IsaacLab22 主线及其派生配置在 `1M env steps` 前会跳过 Gd 更新。
- `sample_many` 和 batched latent encoding 保持启用，不作为本次疑点处理。
- 已在运行中的训练不会自动应用该改动，需要重启新 run 才生效。

验证方式：
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/train.py fdpi_reachability_dreamer_isaaclab22/trainer.py`。
- 已通过：使用项目 `load_config()` 加载默认、baseline-engineering、4090 fast、4090 balanced 和 policy-recovery 配置，确认 `Gd.StartStep=1000000`，且 `UseSampleMany=True`、`UseBatchedCriticLatentEncoding=True` 保持不变。

相关实验：
- 建议后续基于 baseline-engineering 配置重启一组 “dual/Gd 延后到 1M” 对照长训。

---

### LOGIC_CHANGE：新增 dual-main 动作距离诊断日志

修改内容：
- 在 `fdpi_reachability_dreamer_isaaclab22/trainer.py` 的真实 rollout dual 替换路径中新增 `DualAction/rollout_*` 指标。
- 在 `fdpi_reachability_dreamer_isaaclab22/dual_update.py` 的 `imagined_risk_return` 和 `max_risk` dual 更新中新增 dual/main 策略分离指标。
- 新增指标包括 sampled action L2、策略均值 L2、绝对差均值和 logprob gap。
- 更新 `tests/test_reachability_gp.py`，确认 dual update 返回新增诊断字段。

修改原因：
- W&B run `yygyodz1` 显示 dual 数据占比不低，但真实危险率没有高于 main，需要直接判断 dual policy 是否真的偏离主策略。
- 仅看 `Dual/kl_to_main` 难以区分“动作分布均值接近”“采样动作接近”和“logprob gap 较大但真实动作未带来危险”。

影响：
- 不改变训练 loss、采样比例、优化步数或 replay 内容。
- W&B 新增 `DualAction/rollout_sample_l2`、`DualAction/rollout_mean_l2`、`DualAction/rollout_logprob_gap`、`Dual/mean_l2_to_main`、`Dual/sample_l2_to_main_mean` 等诊断曲线。

验证方式：
- 已通过：`python -m pytest tests/test_reachability_gp.py -q`。
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/trainer.py fdpi_reachability_dreamer_isaaclab22/dual_update.py`。

相关实验：
- `experiments/2026-06-12_低KL对偶探索诊断/`

---

### CONFIG_CHANGE：新增 dual KL 0.1 对偶探索诊断配置

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl01.yaml`。
- 该配置继承 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml`，只将 `FDPIRegimeDreamer.DualUpdate.KLCoeff` 调整为 `0.1`，并更新 W&B name。

修改原因：
- 当前 run `yygyodz1` 中 `Dual/kl_loss` 量级明显大于 `Dual/policy_loss`，怀疑 KL 项压制了 Gd 风险探索项。
- 需要第一档保守 ablation，验证降低 KL 后 dual 是否能产生更高真实危险率。

影响：
- 默认配置不因本条新增配置而改变。
- 新实验仍保留 baseline 工程优化训练密度、`DualSampling.StartStep=1000000` 与 `DualUpdate.StartStep=1000000`。

验证方式：
- 已通过：加载 `configs/reachability_gp_isaaclab22_baseline_engineering_dual_kl01.yaml`，确认 `DualUpdate.KLCoeff=0.1` 且 baseline 训练密度保持不变。

相关实验：
- `experiments/2026-06-12_低KL对偶探索诊断/`

---

### EXPERIMENT_ONLY：新增低KL对偶探索诊断实验目录

修改内容：
- 新增实验目录 `experiments/2026-06-12_低KL对偶探索诊断/`。
- 新增 `运行命令.sh`、`启动tmux长训.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json`、`配置快照/` 和 `代码变更.patch`。
- 更新 `docs/research/EXPERIMENT_INDEX.md` 与 `docs/research/PROJECT_STATE.md`。

修改原因：
- 用户要求执行“降低 KL 到 0.1”和“记录 dual-main 动作距离”两个验证项。
- 实验命令涉及配置、W&B、checkpoint 和日志路径，按实验命令管理规则沉淀为短脚本入口。

影响：
- 新实验可通过短命令 `bash experiments/2026-06-12_低KL对偶探索诊断/运行命令.sh` 运行。
- 当前正在运行的 baseline 工程优化长训不会自动切换到该配置。

验证方式：
- 已通过：`bash -n experiments/2026-06-12_低KL对偶探索诊断/运行命令.sh experiments/2026-06-12_低KL对偶探索诊断/启动tmux长训.sh`。

相关实验：
- `experiments/2026-06-12_低KL对偶探索诊断/`

---

### CONFIG_CHANGE：将对偶策略采样与更新延后到 1M

修改内容：
- 将 `configs/reachability_gp_isaaclab22.yaml` 中 `FDPIRegimeDreamer.DualSampling.StartStep` 从 `100000` 调整为 `1000000`。
- 将 `configs/reachability_gp_isaaclab22.yaml` 中 `FDPIRegimeDreamer.DualUpdate.StartStep` 从 `100000` 调整为 `1000000`。
- 在 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml` 中显式写入相同的 `DualSampling.StartStep=1000000` 与 `DualUpdate.StartStep=1000000`，避免派生配置启动时隐式继承早期对偶策略时机。

修改原因：
- W&B 对比显示前 `1M env steps` 的 reward 学习速度差异明显，而当前默认配置从 `100k env steps` 开始引入 dual sampling 与 dual update。
- 需要先让主策略和 world model 在前 `1M env steps` 内更接近纯主策略数据分布，再观察夹取/奖励学习是否恢复。

影响：
- 后续新启动训练在 `1M env steps` 前不会使用 dual policy 采集环境动作，也不会更新 dual policy。
- `Gp` 仍按原配置训练；`Gd` 后续已单独增加 `StartStep` 门控并与 dual 启用时机对齐；`MainFDPIRegime` 仍在 `1M env steps` 后启用。
- 已经运行中的 tmux 训练不会自动应用此配置，需要重启新 run 才生效。

验证方式：
- 已通过：使用项目 `load_config()` 加载 `configs/reachability_gp_isaaclab22.yaml` 和 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml`，确认 `DualSampling.StartStep=1000000`、`DualUpdate.StartStep=1000000`。

相关实验：
- 后续建议基于 `experiments/2026-06-12_baseline工程优化长训/` 新开延后 dual 的对照长训。

---

### EXPERIMENT_ONLY：停止 policy recovery 并启动 baseline 工程优化长训

修改内容：
- 向 tmux 会话 `policy_recovery_long` 发送 `Ctrl-C`，停止 W&B run `c6400qld` 对应的 policy recovery 长训。
- 使用 `experiments/2026-06-12_baseline工程优化长训/启动tmux长训.sh` 启动 baseline 工程优化长训。
- 更新两个实验目录的 `实验记录.md`、`指标结果.json` 和 `实验清单.yaml`，并更新 `docs/research/EXPERIMENT_INDEX.md` 与 `docs/research/PROJECT_STATE.md`。

修改原因：
- policy recovery 到约 `3.06M env steps` 仍未恢复旧 baseline 的夹取学习表现，需要回到 baseline 更新密度，只保留工程优化做对照。

影响：
- 当前活跃训练切换为 tmux 会话 `baseline_engineering_long`。
- 新 W&B run 为 `yygyodz1`，checkpoint 目录为 `experiments/2026-06-12_baseline工程优化长训/检查点/fdpi-baseline-engineering/baseline_engineering_20260612_012750`。

验证方式：
- 已确认旧训练 Python / IsaacLab 进程退出，GPU 显存回落到约 `58 MiB`。
- 已确认新训练 IsaacLab 环境创建成功，环境数为 `64`，训练进度条已开始，step 0 checkpoint 已保存。

相关实验：
- `experiments/2026-06-11_4090策略更新强度恢复验证/`
- `experiments/2026-06-12_baseline工程优化长训/`

---

### CONFIG_CHANGE：完善 Git 忽略规则排除训练大文件

修改内容：
- 扩展 `.gitignore`，忽略训练 checkpoint 目录、模型权重文件、W&B/运行缓存、数组数据、归档包、视频和临时日志。
- 保留实验目录中的 Markdown 记录、配置快照、运行脚本、指标 JSON、图片和 patch 等轻量可追溯文件进入 Git。

修改原因：
- 当前实验目录中已有大量 `.pth` 训练模型和 checkpoint 产物，容易让仓库体积快速膨胀。
- 需要让实验记录可长期保存，同时避免把训练权重和大文件纳入版本管理。

影响：
- 新生成的训练模型、checkpoint 和常见大文件默认不会再进入 Git。
- 已经进入 Git 索引的 checkpoint 产物已从索引移除，本地文件保留。

验证方式：
- 已通过：`git check-ignore --no-index -v experiments/2026-06-11_云端IsaacLab22短训练烟测/检查点/fdpi-cloud-isaaclab22-smoke/cloud_smoke/agent_0.pth`，确认 checkpoint 模型文件被忽略。
- 已通过：`git check-ignore --no-index -v experiments/2026-06-12_baseline工程优化长训/实验记录.md` 无输出，确认实验记录文件未被忽略。
- 已通过：`git ls-files -ci --exclude-standard` 无输出，确认 Git 索引中没有残留的被忽略文件。
- 已执行：`git ls-files -ci --exclude-standard -z | xargs -0 -r git rm --cached --ignore-unmatch --`，从索引移除已暂存的 checkpoint 产物，本地文件未删除。

相关实验：
- `experiments/`

---

### EXPERIMENT_ONLY：新增 baseline 工程优化长训入口

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml`。
- 新增实验目录 `experiments/2026-06-12_baseline工程优化长训/`，包含 `运行命令.sh`、`启动tmux长训.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json` 和配置快照目录。
- 更新 `docs/research/EXPERIMENT_INDEX.md`，登记该实验入口。

修改原因：
- 4090 fast 与 policy recovery 虽提升吞吐，但 W&B 观察显示夹取学习明显弱于旧 baseline。
- 需要一个只保留工程优化、不降低 baseline 更新密度的长训入口，验证学习能力是否恢复。

影响：
- 不改变默认配置和训练逻辑。
- 新入口保持 `NumEnvs=64`、`BatchSize=64`、`Train*EverySteps=256`、`ModelUpdate=4`、`AgentUpdate=4`、`Gp/Gd/Dual UpdateSteps=2`。
- 仅启用 TF32、低频日志、timing、replay starts cache、`sample_many`、批量 critic latent encoding 和低频 FDPI 梯度诊断。

验证方式：
- 已通过：`bash -n experiments/2026-06-12_baseline工程优化长训/运行命令.sh experiments/2026-06-12_baseline工程优化长训/启动tmux长训.sh`。
- 已通过：加载 `configs/reachability_gp_isaaclab22_baseline_engineering.yaml`，确认 baseline 训练密度与工程优化参数生效。
- 待执行：运行 `bash experiments/2026-06-12_baseline工程优化长训/运行命令.sh` 或 tmux 入口进行长训。

相关实验：
- `experiments/2026-06-12_baseline工程优化长训/`

---

### LOGIC_CHANGE：优化训练更新阶段的采样、编码和诊断开销

修改内容：
- 新增 replay `sample_many` 批量采样路径，trainer 中 world model、Gp、Gd、dual 和 agent 连续采样默认使用该路径，保留旧采样回退。
- 新增 frozen world model posterior 的批量编码路径，Gp/Gd/dual 仍分别使用各自 replay batch，但在同一 update 触发周期内合并执行 latent 编码并拆回各自更新。
- FDPI 主策略梯度诊断改为低频计算，默认每 `65536 env steps` 计算一次，避免每次策略更新都额外执行多组 `autograd.grad`。
- world model 详细 cost/force 指标改为懒计算，默认只在 detailed log step 计算，普通更新只保留训练所需 loss 与核心指标。
- 全局性能默认值启用 `UseSampleMany`、`UseBatchedCriticLatentEncoding`，并将 `DetailedLogEverySteps` 默认设为 `8192`。

修改原因：
- 4090 policy recovery timing 显示 warmup 后 update 触发周期主要耗时来自多路 replay sampling、重复 posterior 编码、Gp/Gd/dual/agent 串行更新和诊断反传。
- 需要在不降低 `AgentUpdate`、不改变 Gp/Gd/dual replay 分布、不改变 loss 的前提下降低 wall-clock 训练时间。

影响：
- 不改变训练 loss、update 次数、更新触发间隔、batch size 或 Gp/Gd/dual 的 replay 采样比例。
- 旧配置若未显式覆盖 `Performance`，详细诊断日志会默认低频化；核心训练指标仍保留。
- 当前已运行的训练进程不会自动获得新代码优化，需要重启训练后生效。

验证方式：
- 已补充单测覆盖 `sample_many`、posterior state/feature 一致性、Gp 预编码更新路径和 dual 预编码更新路径。
- 已通过：`python -m compileall fdpi_reachability_dreamer_isaaclab22 tests`。
- 已通过：`python -m unittest tests.test_isaaclab22_package tests.test_reachability_gp`。
- 已通过：加载 `configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml`，确认 `UseSampleMany=True`、`UseBatchedCriticLatentEncoding=True`、`BatchedLatentEncodeMaxBatch=1024`、`FDPIGradDiagnosticsEverySteps=65536` 生效。
- 待执行：使用 policy recovery 配置做短 profiling，对比 `sample_*_batch_seconds`、`critic_dual_latent_encode_seconds`、`gp_update_seconds`、`gd_update_seconds`、`dual_update_seconds` 和整体 update 周期。

相关实验：
- `experiments/2026-06-11_4090策略更新强度恢复验证/`

---

## 2026-06-11

### BUGFIX：增强训练异常退出诊断并准备重跑 policy recovery

问题表现：
- `2026-06-11_4090策略更新强度恢复验证` 首次 tmux run `policy_recovery_20260611_213921` 在约 `105k env steps` 后停止。
- tmux 会话、训练 Python 进程和 W&B 进程均消失，日志没有 `Traceback`、`TRAIN_EXIT_STATUS` 或 NaN/OOM 明确信息。

修复方式：
- 在 `fdpi_reachability_dreamer_isaaclab22/train.py` 启用 `faulthandler`。
- 在最外层 `main()` 外捕获 `BaseException`，确保在进程退出前打印完整 traceback。
- 对 `SIGTERM` / `SIGINT` / `SIGHUP` 安装诊断处理器，打印所有 Python 线程栈、PID、RUN_ID、checkpoint 目录和 argv。
- 更新 `scripts/train.sh` 和实验 `运行命令.sh`，设置 `PYTHONUNBUFFERED=1`、`PYTHONFAULTHANDLER=1`。
- 更新实验 `运行命令.sh`，退出时强制打印 `[FDPI_EXPERIMENT_EXIT] status=...`，收到 shell signal 时打印 `[FDPI_EXPERIMENT_SIGNAL] ...`。

影响：
- 不改变训练算法、模型结构或配置参数。
- 如果后续仍发生 Python 异常、信号终止或非零退出，日志尾部应保留更明确的诊断信息。

验证方式：
- 已通过：`python -m py_compile fdpi_reachability_dreamer_isaaclab22/train.py`。
- 已通过：`bash -n scripts/train.sh experiments/2026-06-11_4090策略更新强度恢复验证/运行命令.sh`。
- 已通过：使用 `experiments/2026-06-11_4090策略更新强度恢复验证/启动tmux长训.sh` 重启 policy recovery 长训，RUN_ID `policy_recovery_diag_20260611_215744`，W&B run `c6400qld`；IsaacLab 环境创建成功，环境数为 128，训练进入 rollout/update 循环。

相关实验：
- `experiments/2026-06-11_4090策略更新强度恢复验证/`

---

### EXPERIMENT_ONLY：停止 fast+safety50 长训并准备 policy recovery 长训

修改内容：
- 停止 `experiments/2026-06-11_危险区域世界模型50比50采样/` 中的 4090 fast+safety50 长训 `wm_safety50_fast_long` / W&B run `n60y55jj`。
- 新增实验目录 `experiments/2026-06-11_4090策略更新强度恢复验证/`。
- 新增 `运行命令.sh`、`配置.yaml`、`参数说明.md`、`实验清单.yaml`、`实验记录.md`、`指标结果.json` 和配置快照。
- 更新旧实验状态、实验索引和项目状态，记录 fast 模式策略学习偏弱后切换到 policy recovery。

修改原因：
- 用户观察原训练约 `1M env steps` 已学会夹取，但 fast 模式接近 `4M env steps` 仍未学会夹取，说明 fast 过多牺牲了策略学习能力。
- 需要保留 4090 吞吐优化，同时恢复 agent/Gp/Gd/dual 更新强度进行长训验证。

影响：
- 不改变默认训练配置。
- 新实验默认使用 `configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml`，总步数为 `10M env steps`。

验证方式：
- 已通过：旧训练 tmux 与训练进程均停止。
- 已通过：`bash -n experiments/2026-06-11_4090策略更新强度恢复验证/运行命令.sh`。
- 已通过：启动新 tmux 长训 `policy_recovery_long`，IsaacLab 环境创建成功，环境数为 128，W&B run 为 `2hceqcms`，checkpoint 路径为 `experiments/2026-06-11_4090策略更新强度恢复验证/检查点/fdpi-4090-policy-recovery/policy_recovery_20260611_213921`。

相关实验：
- `experiments/2026-06-11_4090策略更新强度恢复验证/`
- `experiments/2026-06-11_危险区域世界模型50比50采样/`

---

### CONFIG_CHANGE：新增 4090 policy recovery 配置恢复策略更新强度

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml`。
- 保留 4090 友好的 `NumEnvs=128`、`BatchSize=256`、`ImagineBatchSize=256`、AMP 和 TF32。
- 将 `TrainModelEverySteps` / `TrainAgentEverySteps` 从 fast 的 `1024` 调为 `512`。
- 将 `AgentUpdate` 从 fast 的 `2` 调为 `4`。
- 将 `Gp/Gd/Dual UpdateSteps` 从 fast 的 `1` 恢复为 `2`。
- 更新 `notes/问题/策略熵使用情况与潜在问题.md`，记录 W&B 中 fast 接近 4M 仍未学会夹取、原训练约 1M 已学会夹取的观察。

修改原因：
- 4090 fast 配置虽有 `3.09x` 吞吐提升，但每 `1024 env steps` 的 agent update 从 baseline 约 `16` 次降到 `2` 次，策略 optimizer step 密度约为 baseline 的 `1/8`。
- W&B 结果显示 fast 过多牺牲策略学习能力，需要一个保留吞吐优势但恢复更新强度的中间配置。

影响：
- 不修改默认配置和 fast 配置。
- 新配置每 `1024 env steps` 约有 `8` 次 agent update，介于 baseline 的 `16` 次和 fast 的 `2` 次之间。
- 预期吞吐会低于 fast，但应比 baseline 快，并可能改善夹取学习速度。

验证方式：
- 已通过：加载 `configs/reachability_gp_isaaclab22_4090_policy_recovery.yaml`，确认 `NumEnvs=128`、`BatchSize=256`、`ImagineBatchSize=256`、`Train*EverySteps=512`、`AgentUpdate=4`、`Gp/Gd/Dual UpdateSteps=2`、TF32 生效，并通过 batch/NumEnvs 约束检查。
- 已通过：`python -m compileall scripts/profiling/benchmark_training_speed.py fdpi_reachability_dreamer_isaaclab22/train.py tests/test_isaaclab22_package.py`。
- 待执行：用 policy recovery 配置跑 `1M env steps`，对比 W&B 中 reward、success/夹取行为、cost、Gp/Gd loss。

相关实验：
- 后续建议新建 `experiments/2026-06-11_4090策略更新强度恢复验证/`。

---

### REFACTOR：新增 W&B 访问流程 skill 与脱敏检查脚本

修改内容：
- 新增 repo-local skill `wandb-access-manager`，沉淀 IsaacLab 环境中访问 W&B 的标准流程。
- 新增 `.agents/skills/wandb-access-manager/scripts/verify_wandb_access.py`，用于脱敏检查 wandb 登录态、viewer、project、run 列表、history keys 和 sampled history。
- 明确记录当前环境中 `wandb login --verify` 可通过，但 `wandb.Api()` 可能报 `relogin required` 时应改用窄字段 GraphQL 查询。

修改原因：
- W&B CLI 登录态、Public API wrapper 和 GraphQL API 的表现可能不一致，需要一个可重复、不会泄露 token 的访问流程。
- 后续整理训练曲线和实验记录时，需要稳定读取 run 指标，避免每次重新排查认证和查询字段问题。

影响：
- 不改变训练逻辑、配置或实验输出路径。
- 后续查询 W&B 数据时可直接使用 `wandb-access-manager` 和检查脚本。

验证方式：
- 已通过：`/opt/conda/envs/isaaclab/bin/python -m py_compile .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py`。
- 已通过：检查脚本读取 viewer `2332133796`、entity `2332133796-yhyper`，并列出项目 `IsaacLab22-SurgicalRobot5-FDPI-Reachability`。
- 已通过：检查脚本列出该项目 4 个 run，并对 `n60y55jj` 过滤 `entropy` 指标、抽样读取 `MainFDPI/entropy` 历史点。

相关实验：
- 暂无。

---

### CONFIG_CHANGE：将危险区域 50 比 50 实验切换到 4090 fast 基座

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_4090_fast_wm_safety50.yaml`，继承 `configs/reachability_gp_isaaclab22_4090_fast.yaml`，仅覆盖 world model 50% uniform + 50% safety-critical 采样。
- 更新 `experiments/2026-06-11_危险区域世界模型50比50采样/运行命令.sh`，默认改用 fast+safety50 配置，并将 `RUN_NAME` / `RUN_ID` / tags 标记为 fast。
- 记录并停止误用 baseline 基座的旧 tmux run `wm_safety50_long` / W&B run `f93hle70`。
- 更新实验记录、实验清单、参数说明和实验索引。

修改原因：
- 旧 `configs/reachability_gp_isaaclab22_wm_safety50.yaml` 继承默认 baseline 配置，实际使用 `NumEnvs=64`、`TrainModelEverySteps=256`、`ModelUpdate=4`，预计长训接近 12 小时。
- 本实验需要在 4090 fast 训练参数下验证 safety50 采样，避免 wall-clock 过长。

影响：
- 后续运行 `bash experiments/2026-06-11_危险区域世界模型50比50采样/运行命令.sh` 将使用 4090 fast 基座。
- 保留原 baseline safety50 配置文件，便于追溯，但不再作为该实验默认入口。

验证方式：
- 已通过：加载 `configs/reachability_gp_isaaclab22_4090_fast_wm_safety50.yaml`，确认 `NumEnvs=128`、`BatchSize=256`、`TrainModelEverySteps=1024`、`ModelUpdate=2` 和 `WorldModelSampling.SafetyCriticalRatio=0.5` 生效。
- 已通过：`bash -n experiments/2026-06-11_危险区域世界模型50比50采样/运行命令.sh scripts/train.sh scripts/train_isaaclab22.sh`。
- 已通过：tmux 启动 `wm_safety50_fast_long`，IsaacLab 环境创建成功，环境数为 128，W&B run 为 `n60y55jj`。

相关实验：
- `experiments/2026-06-11_危险区域世界模型50比50采样/`

---

### LOGIC_CHANGE：修复世界模型 safety-critical 采样配额并新增 50 比 50 危险区域实验

修改内容：
- 将 `FDPIReplayBuffer.sample()` 的 safety-critical 采样配额从按单个 env 的 `per_env_batch` 取整，改为按全 batch 统一分配。
- 在 `BatchSize == NumEnvs`、`SafetyCriticalRatio=0.5` 时，world model batch 现在可以真实抽到约一半 safety-critical 窗口。
- 新增 CPU 单测覆盖 `per_env_batch=1` 时的全 batch safety 配额。
- 新增配置 `configs/reachability_gp_isaaclab22_wm_safety50.yaml`，用于 50% uniform + 50% safety-critical world model 采样实验。
- 新增实验目录 `experiments/2026-06-11_危险区域世界模型50比50采样/`，包含运行命令、配置、参数说明、实验清单和初始记录。

修改原因：
- 旧实现中 `BatchSize=64`、`NumEnvs=64` 时 `per_env_batch=1`，`round(1 * 0.45)` 使 world model safety-critical 采样实际退化为 0，dual/high-risk 数据主要只能自然混入 replay。
- 为验证危险区域建模是否改善，需要在不完全破坏主策略分布的前提下，让 world model 更新 batch 明确保留约一半 safety-critical 数据。

影响：
- 默认配置的 `SafetyCriticalRatio=0.45` 现在会在全 batch 维度真实生效，不再受 `per_env_batch=1` 的取整问题影响。
- 新实验配置只改变 world model replay 采样比例和日志/性能设置，不额外提高 dual 采集比例，也不增加 source-aware loss 权重。

验证方式：
- 已通过：`python -m unittest tests.test_isaaclab22_package`。
- 已通过：加载 `configs/reachability_gp_isaaclab22_wm_safety50.yaml`，确认 `BatchSize=64`、`NumEnvs=64`、`WorldModelSampling.SafetyCriticalRatio=0.5`、`UniformRatio=0.5` 生效。
- 已通过：`bash -n experiments/2026-06-11_危险区域世界模型50比50采样/运行命令.sh scripts/train.sh scripts/train_isaaclab22.sh`。
- 已通过：`python -m compileall fdpi_reachability_dreamer_isaaclab22/replay_buffer.py tests/test_isaaclab22_package.py`。

相关实验：
- `experiments/2026-06-11_危险区域世界模型50比50采样/`

---

### BUGFIX：修复 4090 profiling 入口静默退出并完成 warmup 后速度测试

问题表现：
- `scripts/profiling/benchmark_training_speed.py` 在 IsaacLab 下只输出到 `building env/models/replay`，没有进入 prefill / rollout / update，也没有生成 `指标结果.json`，但进程返回 0。

问题原因：
- profiling 内部 `_build_everything()` 构造的 `train_args` 缺少 `num_envs` 字段，调用 IsaacLab22 `build_env()` 时触发 `AttributeError`。
- `simulation_app.close()` 会干扰异常返回码，导致错误看起来像正常退出。

修复方式：
- 给 profiling 的 `train_args` 补充 `num_envs=None`。
- profiling 异常时打印 traceback，并避免在异常路径调用 `simulation_app.close()` 吞掉错误。
- 捕获无 git 仓库时 `git diff` 的 stderr，避免污染 profiling 日志。

影响：
- `experiments/2026-06-11_4090训练加速参数对比/运行命令.sh` 和 `运行全部对比.sh` 能正常生成指标。
- 不改变训练算法逻辑，只修复 profiling 可靠性。

验证方式：
- 已完成三组 warmup 后带更新 profiling：baseline、balanced、fast 均生成 `指标结果.json`、阶段耗时、GPU 采样和图片。
- baseline：`213.49 env steps/s`，采样+更新耗时 `516.46s`。
- balanced：`520.89 env steps/s`，采样+更新耗时 `204.34s`。
- fast：`659.81 env steps/s`，采样+更新耗时 `151.95s`。

相关实验：
- `experiments/2026-06-11_4090训练加速参数对比/`

---

### LOGIC_CHANGE：新增 4090 训练加速配置与深度优化

修改内容：
- 新增 `configs/reachability_gp_isaaclab22_4090_fast.yaml` 和 `configs/reachability_gp_isaaclab22_4090_balanced.yaml`，通过 `BaseConfig` 继承当前 IsaacLab22 默认配置，只覆盖 4090 加速参数。
- 为 IsaacLab22 配置加载增加 `BaseConfig` / `_BASE_` overlay 支持，减少配置复制。
- 新增 `FDPIRegimeDreamer.Performance` 配置段，支持 TF32、matmul precision、低频日志、timing 日志和 replay starts 缓存开关。
- 在训练入口增加 `BatchSize` / `ImagineBatchSize` 与 `NumEnvs` 的约束检查，提前发现 replay sampling 参数错误。
- 在 `scripts/train.sh` 与 `fdpi_reachability_dreamer_isaaclab22/train.py` 增加 batch、imagine、train_every、update 次数的可选命令行覆盖。
- 在训练循环中加入轻量 timing 日志，分项记录 policy inference、env step、replay append、各类 batch sample、world model / Gp / Gd / dual / agent update、logging 和 checkpoint。
- 在 replay buffer 中缓存 valid starts 与 safety-critical starts，`can_sample()` 也复用缓存，并在 append/load 后失效，减少连续更新时重复扫描窗口。
- 将 batch composition、force 细项和多数训练细节日志改为可配置低频记录；默认配置保持 `LogEverySteps=1`，4090 配置使用低频日志。
- 新增实验目录 `experiments/2026-06-11_4090训练加速参数对比/`，包含 fast 单组入口、三组对比入口、参数说明、实验清单和初始实验记录；三组对比输出分别写入 `对比结果/baseline`、`对比结果/balanced`、`对比结果/fast`，避免覆盖。
- 新增 IsaacLab22 replay starts 缓存 CPU 单测。
- 更新 README、PROJECT_STATE 和 EXPERIMENT_INDEX。

修改原因：
- 当前训练耗时主要集中在更新阶段；通过降低更新/采样比例、增大 batch、减少重复 replay 采样开销和启用 4090 友好的 TF32，提高 wall-clock 吞吐和 GPU 利用率。

影响：
- 默认 `configs/reachability_gp_isaaclab22.yaml` 行为保持不变。
- 4090 fast 配置优先吞吐，可能牺牲一定样本效率；balanced 配置用于保守对照。
- 配置参数若违反 `BatchSize >= NumEnvs 且可整除` 会在训练入口提前报错。

验证方式：
- 已通过：`python -m compileall fdpi_reachability_dreamer_isaaclab22 scripts/profiling/benchmark_training_speed.py tests`。
- 已通过：加载 `configs/reachability_gp_isaaclab22.yaml`、`configs/reachability_gp_isaaclab22_4090_fast.yaml` 与 `configs/reachability_gp_isaaclab22_4090_balanced.yaml`，确认默认/fast/balanced 的 NumEnvs、BatchSize、ImagineBatchSize、UseAmp、TF32 和 LogEverySteps 生效。
- 已通过：加载 `experiments/2026-06-11_4090训练加速参数对比/配置.yaml`、`配置_baseline.yaml`、`配置_balanced.yaml`、`配置_fast.yaml`，确认 profiling overlay 与输出目录生效。
- 已通过：`bash -n scripts/train.sh scripts/train_isaaclab22.sh experiments/2026-06-11_4090训练加速参数对比/运行命令.sh experiments/2026-06-11_4090训练加速参数对比/运行全部对比.sh`。
- 已通过：`python -m unittest discover -s tests`，共 9 个测试通过；仅保留旧包 `torch.cuda.amp.GradScaler` FutureWarning。
- 已通过：`WANDB_MODE=disabled CONFIG_PATH=configs/reachability_gp_isaaclab22_4090_fast.yaml NUM_ENVS=2 BATCH_SIZE=2 IMAGINE_BATCH_SIZE=2 SAMPLE_MAX_STEPS=0 RUN_NAME=fdpi-4090-fast-smoke RUN_ID=zero_step_after_speedup RUN_ROOT=/tmp/fdpi_4090_fast_smoke bash scripts/train.sh`，0-step IsaacLab22 启动烟测通过。

相关实验：
- `experiments/2026-06-11_4090训练加速参数对比/`

---

### BUGFIX：清理 SurgicalRobot5 quat_rotate 弃用 warning

问题表现：
- IsaacLab22 训练启动和采样过程中反复输出 `quat_rotate` / `quat_rotate_inverse` 将弃用的 warning，日志噪声很大。

问题原因：
- 外部任务扩展 `/root/gpufree-data/surgical_robot5/exts/surgical_robot5/surgical_robot5/env.py` 仍在多个坐标变换和速度变换位置调用 IsaacLab 2.2 已弃用的 `math_utils.quat_rotate` 与 `math_utils.quat_rotate_inverse`。

修复方式：
- 将上述调用等价替换为 IsaacLab 推荐的 `math_utils.quat_apply` 与 `math_utils.quat_apply_inverse`。
- 更新 `docs/research/PROJECT_STATE.md`，记录该 warning 已清理。

影响：
- 不改变任务几何语义；IsaacLab 的弃用函数内部本身也转调对应 `quat_apply` / `quat_apply_inverse`。
- 后续 `bash scripts/train.sh` 启动时不再输出该类弃用 warning。

验证方式：
- `rg -n "quat_rotate|quat_rotate_inverse" /root/gpufree-data/surgical_robot5/exts/surgical_robot5/surgical_robot5 /root/gpufree-data/fdpi_reachability_dreamer_clean/fdpi_reachability_dreamer_isaaclab22`，确认任务扩展和训练包无弃用调用残留。
- `conda activate isaaclab && python -m compileall /root/gpufree-data/surgical_robot5/exts/surgical_robot5/surgical_robot5/env.py`。
- `conda activate isaaclab && PYTHONPATH=... python -c "import surgical_robot5; print(surgical_robot5.TASK_ID)"`。
- `WANDB_MODE=disabled NUM_ENVS=2 SAMPLE_MAX_STEPS=0 RUN_NAME=fdpi-warning-check RUN_ID=quat_apply_check RUN_ROOT=/tmp/fdpi_quat_warning_check bash scripts/train.sh`，启动成功且 `/tmp/fdpi_quat_warning_check.log` 中无 `quat_rotate` warning。

相关实验：
- 暂无。

---

### LOGIC_CHANGE：覆盖旧训练入口为 IsaacLab22 主入口

修改内容：
- 将 `scripts/train.sh` 从旧 IsaacLab 1.4 / UR3 Lite 启动器改为 IsaacLab 2.2 / SurgicalRobot5 主训练入口。
- `scripts/train.sh` 现在默认调用 `fdpi_reachability_dreamer_isaaclab22/train.py`，使用 `configs/reachability_gp_isaaclab22.yaml`、`SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1` 和 `ckpt_isaaclab22/`。
- 将默认依赖路径改为云端环境：`/opt/conda/etc/profile.d/conda.sh`、`/root/IsaacLab`、`/root/gpufree-data/surgical_robot5/exts/surgical_robot5`。
- 将 `scripts/train_isaaclab22.sh` 收敛为兼容转发脚本，统一转到 `scripts/train.sh`，避免两套 IsaacLab22 启动逻辑分叉。
- 将 `scripts/profiling/benchmark_training_speed.py` 的默认 launcher、PYTHONPATH、训练包、任务名和配置切到 IsaacLab22 / SurgicalRobot5，避免 profiling 默认回落到旧 1.4 接口。
- 更新 `README.md`、`AGENTS.md` 和 `docs/research/PROJECT_STATE.md` 中的训练入口、依赖和默认配置说明。

修改原因：
- 项目当前已切到 IsaacLab 2.2 / SurgicalRobot5 迁移版，旧 IsaacLab 1.4 / UR3 Lite 启动接口不再作为主入口保留。

影响：
- 以后直接运行 `bash scripts/train.sh` 即启动 IsaacLab22 训练链路。
- 旧的 `scripts/train.sh` 默认 1.4/UR3 Lite 行为被覆盖；需要旧版时应显式从历史版本恢复或另行创建专用脚本。
- 不改变训练算法实现和配置内容，只改变默认启动接口与说明。

验证方式：
- `bash -n scripts/train.sh scripts/train_isaaclab22.sh`。
- `conda activate isaaclab && python -m compileall scripts/profiling/benchmark_training_speed.py`。
- 静态搜索确认主入口相关文件中无冲突标记，且无旧 IsaacLab 1.4 / UR3 Lite 默认入口残留。
- 静态检查确认 `/opt/conda/etc/profile.d/conda.sh`、`/root/IsaacLab/isaaclab.sh`、`/root/gpufree-data/surgical_robot5/exts/surgical_robot5` 存在。

相关实验：
- 暂无。

---

### EXPERIMENT_ONLY：完成 fdpi-1 云端 IsaacLab22 短训练烟测

修改内容：
- 新增并补全实验目录 `experiments/2026-06-11_云端IsaacLab22短训练烟测/`，包含远端短命令、配置说明、配置快照、运行日志、指标结果和 checkpoint。
- 将本地 SurgicalRobot5 与 FDPI Reachability Dreamer IsaacLab22 迁移版同步到 `fdpi-1` 的 `/root/gpufree-data/` 下，并在远端使用 `/root/IsaacLab` 2.2.1 与 `isaaclab` conda 环境运行。
- 为 `/home/yhy/surgical_robot5/exts/surgical_robot5/` 补充最小 `pyproject.toml`，使 `pip install -e` 在 pip 25 构建隔离下可以找到 `toml` 并完成 editable 安装。
- 更新 `docs/research/EXPERIMENT_INDEX.md` 与 `docs/research/PROJECT_STATE.md`，记录云端迁移版本已经完成非零步训练闭环验证。

修改原因：
- 验证云端 RTX 4090 环境、SurgicalRobot5 任务扩展和 IsaacLab22 迁移版训练主链路能够真实运行并保存 checkpoint。
- 修正 SurgicalRobot5 扩展在新版 pip editable 安装时缺少 build requirement 的部署问题。

影响：
- 不改变训练算法逻辑。
- SurgicalRobot5 包装元数据新增 build requirements 后，远端可直接 `python -m pip install -e /root/gpufree-data/surgical_robot5/exts/surgical_robot5`。

验证方式：
- 本地：`bash -n scripts/train_isaaclab22.sh`。
- 本地：`conda activate isaaclab && python -m unittest tests.test_isaaclab22_package`。
- 远端：`python -m compileall fdpi_reachability_dreamer_isaaclab22`。
- 远端：`python -m unittest tests.test_isaaclab22_package`。
- 远端：`python -m pip install -e /root/gpufree-data/surgical_robot5/exts/surgical_robot5`。
- 远端：`python -c "import surgical_robot5; print(surgical_robot5.TASK_ID)"`。
- 远端：0 步 IsaacLab 环境检查通过。
- 远端：`bash experiments/2026-06-11_云端IsaacLab22短训练烟测/运行命令.sh`，完成 `512/512` 训练进度并保存 step 0 与 step 512 checkpoint。

相关实验：
- `experiments/2026-06-11_云端IsaacLab22短训练烟测/`

---

### LOGIC_CHANGE：新增可回退训练加速预修改版 speedexp

修改内容：
- 新增 `fdpi_reachability_dreamer_speedexp/`，作为 clean 版的独立预修改包，用于训练速度优化试验。
- 新增 `scripts/train_speedexp.sh`、`configs/reachability_gp_speedexp.yaml` 和 `configs/reachability_gp_speedexp_update_half.yaml`，默认输出到 `ckpt_speedexp/`，不覆盖原 `ckpt/`。
- 在 speedexp replay buffer 中加入采样候选缓存，用于减少 repeated valid-start / safety mask 构造；不改变 replay batch 字段和采样语义。
- 在 speedexp trainer 中增加细分计时日志：world model、Gp、Gd、dual、main agent 更新及对应采样耗时。
- 新增 `tests/test_speedexp_replay_buffer.py`，检查 speedexp replay sample 合同与 clean 版一致。
- 新增实验目录 `experiments/2026-06-11_训练加速预修改验证/`，用于后续 warmup 后 profiling 对比。

修改原因：
- 在不污染主线 clean 训练代码的前提下，提供可试验、可对比、可删除回退的训练加速预修改版本。

影响：
- 不修改 `fdpi_reachability_dreamer/`、`configs/reachability_gp.yaml` 和 `scripts/train.sh`。
- 默认 speedexp 配置保持训练调度不变；`reachability_gp_speedexp_update_half.yaml` 属于可选行为变化配置，只用于对照实验。

验证方式：
- `python3 -m compileall fdpi_reachability_dreamer_speedexp scripts/profiling tests/test_speedexp_replay_buffer.py`
- `bash -n scripts/train_speedexp.sh`
- `bash -n experiments/2026-06-11_训练加速预修改验证/运行命令.sh`
- 使用 IsaacLab Python + conda site-packages 运行 `python -m unittest tests.test_speedexp_replay_buffer`，已通过。
- 已执行 0 步 speedexp 启动烟测，确认输出目录为 `ckpt_speedexp/`。

相关实验：
- `experiments/2026-06-11_训练加速预修改验证/`

---

### EXPERIMENT_ONLY：新增训练速度与 GPU 利用率 profiling 实验

修改内容：
- 新增 `scripts/profiling/benchmark_training_speed.py`、`collect_gpu_stats.py`、`plot_training_speed.py`，用于可复现地记录训练阶段耗时、吞吐和 GPU 使用情况。
- 新增实验目录 `experiments/2026-06-11_训练速度与GPU利用率测试/`，包含 `运行命令.sh`、`配置.yaml`、`参数说明.md`、`指标结果.json`、`实验记录.md`、图片、日志和检查点。
- 更新 `docs/research/EXPERIMENT_INDEX.md`，登记本次 64 env 平衡短测结果。

修改原因：
- 在优化训练速度前，先建立当前 clean 训练代码的速度基线，并定位环境 step、采样、模型更新和 GPU 利用率瓶颈。

影响：
- 不修改算法逻辑，不重构核心训练流程。
- 新增 profiling 脚本会在运行时通过 IsaacLab 1.4 launcher 启动，并默认禁用 wandb，只写本地实验目录。

验证方式：
- `python3 -m compileall scripts/profiling fdpi_reachability_dreamer`
- `bash -n experiments/2026-06-11_训练速度与GPU利用率测试/运行命令.sh`
- 已运行 2 env 冒烟配置，确认 IsaacLab 启动、GPU 采样、指标和图片生成链路可用。
- 已修正统计口径：先 warmup `51200` env steps，不计入正式统计；warmup 后 64 env 平衡短测平均 `205.57 env steps/s`，GPU 平均利用率 `60.00%`，单项最大耗时为 `actor_critic_update_time`。

相关实验：
- `experiments/2026-06-11_训练速度与GPU利用率测试/`

---

### LOGIC_CHANGE：新增 IsaacLab2.2 / IsaacSim5.0 独立迁移版

修改内容：
- 新增 `fdpi_reachability_dreamer_isaaclab22/`，作为 IsaacLab 2.2 训练主链路的独立代码包。
- 新增 `scripts/train_isaaclab22.sh` 和 `configs/reachability_gp_isaaclab22.yaml`，默认使用 `/home/yhy/IsaacLab5/isaaclab.sh`、conda 环境 `isaaclab`、`surgical_robot5` 任务扩展与独立输出目录 `ckpt_isaaclab22/`。
- 在 IsaacLab22 版本中适配 `isaaclab.app.AppLauncher`、`isaaclab_tasks.utils.parse_env_cfg` 和 IsaacLab 2.2 自动 reset 后的终止观测缓存逻辑。
- 新增轻量单测，确认 IsaacLab22 代码包运行时导入不会反向依赖旧包。

修改原因：
- 将当前基于 IsaacLab 1.4 / UR3 Lite 的 clean 版本迁移一份到 IsaacLab 2.2 / IsaacSim 5.0，并对接 `SurgicalRobot5-HeadPipe-GraspGoalDreamerForce-OSC-RL-Direct-v1` 任务。

影响：
- 不改变原 `fdpi_reachability_dreamer/` 和 `scripts/train.sh` 的 1.4 训练行为。
- 新版本需要 `isaaclab` conda 环境中具备 `yacs`、`wandb`、`torch`、`gymnasium`、`colorama` 等依赖。

验证方式：
- `bash -n scripts/train_isaaclab22.sh`
- `conda activate isaaclab && python -m compileall fdpi_reachability_dreamer_isaaclab22`
- `conda activate isaaclab && python -m unittest tests.test_isaaclab22_package`
- 已补齐 `isaaclab` 环境 `yacs`，并执行 0 步 IsaacLab22 烟测：`WANDB_MODE=disabled NUM_ENVS=2 SAMPLE_MAX_STEPS=0 bash scripts/train_isaaclab22.sh`。

相关实验：
- 暂无。后续非零步烟测应归档到 `experiments/2026-06-11_IsaacLab22迁移烟测/`。

---

## 2026-06-09

### EXPERIMENT_ONLY：固化项目理解与 agent 上下文

修改内容：
- 在 `AGENTS.md` 增加项目上下文速记，说明 clean 版本定位、训练入口、核心模块、Gp reachability 目标和外部依赖边界。
- 更新 `docs/research/PROJECT_STATE.md`，记录当前活跃版本、主要目标、当前问题和下一步计划。
- 新增并链接 `notes/版本/FDPI Reachability Dreamer clean版.md`、`notes/概念/Gp可达性风险估计.md`、`notes/概念/FDPI regime分区.md`、`notes/概念/dual policy高风险采样.md`。
- 更新 `notes/主页.md`、`notes/项目地图.md`、`notes/概念/概念索引.md`、`notes/版本/版本索引.md`。

修改原因：
- 将当前对项目结构、算法主线和训练闭环的理解持久化，方便后续 Codex/agent 和人工阅读者快速接续。

影响：
- 不改变训练代码、配置或实验结果，仅影响长期记录和知识库入口。

验证方式：
- 静态检查新增记录与索引链接，确认关键概念均有对应笔记。

相关实验：
- 暂无。

---

### LOGIC_CHANGE：示例标题

修改内容：
- 

修改原因：
- 

影响：
- 

验证方式：
- 

相关实验：
- 

---

### BUGFIX：示例标题

问题表现：
- 

问题原因：
- 

修复方式：
- 

验证方式：
- 

相关实验：
- 
