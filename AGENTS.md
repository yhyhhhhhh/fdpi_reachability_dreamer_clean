# AGENTS.md

## 项目上下文速记

- 本仓库是 FDPI Reachability Dreamer 的 clean standalone 版本，当前主线是在 IsaacLab 2.2 / IsaacSim 5.0 的 SurgicalRobot5 任务中训练带安全可达性风险估计的 Dreamer 系统。
- 核心训练入口是 `scripts/train.sh` -> `fdpi_reachability_dreamer_isaaclab22/train.py` -> `fdpi_reachability_dreamer_isaaclab22/trainer.py`，默认配置是 `configs/reachability_gp_isaaclab22.yaml`。
- 当前算法主线是：world model 学动态、reward、force/continuous cost；`GpReachabilityCritic` 学主策略下未来危险可达性；`GdRiskCritic` 服务 dual policy 的高风险数据采集；主策略通过 FDPI regime 在 feasible / critical / infeasible 区间内调整 reward 与安全项权重。
- 当前 `Gp` 重点是 `TargetType: n_step_reachability_td`，使用 `binary_cost`，目标为 horizon 内 discounted binary cost 的最大值与 bootstrap risk 的最大值，并尊重 `done` 边界。当前 `configs/reachability_gp_isaaclab22.yaml` 实际为 `ReachabilityH: 3`。
- replay buffer 除 Dreamer 常规字段外，还维护 `continuous_cost`、`binary_cost`、`extreme_cost`、`bottom_force`、`force_excess`、`source`，其中 `source` 区分 main / dual / random，用于安全关键采样和 source-aware loss 权重。
- 外部运行依赖包括 IsaacLab 2.2 / IsaacSim 5.0、`surgical_robot5` 任务扩展、`isaaclab` Python 环境；没有这些依赖时应优先做静态检查和 CPU 单测。
- 项目理解的长期笔记见 `notes/版本/FDPI Reachability Dreamer clean版.md`，研究状态见 `docs/research/PROJECT_STATE.md`。

## 项目开发规则

本仓库的科研开发流程以“实验留痕、修改留痕、版本隔离、中文记录、知识链接”为核心。

### 1. 实验记录规则

- 每次新增实验或运行实验时，都应在 `experiments/` 下建立实验目录。
- 实验目录命名默认使用：`YYYY-MM-DD_中文实验名称`。
- 实验目录内部文件默认优先使用中文命名。
- 每个实验目录应尽量包含以下内容：
  - `实验清单.yaml`
  - `运行命令.sh`
  - `配置.yaml`
  - `参数说明.md`
  - `配置快照/`
  - `代码变更.patch`
  - `指标结果.json`
  - `实验记录.md`
  - `图片/`
  - `日志/`
  - `检查点/`

### 2. 实验命令管理规则

- Codex 不应默认生成难以维护的超长终端命令。
- 当实验命令超过 120 个字符，或包含超过 5 个显式参数时，应优先创建或更新实验目录中的 `运行命令.sh`、`配置.yaml` 和 `参数说明.md`。
- 最终回复中应优先给出短命令，例如：`bash experiments/YYYY-MM-DD_中文实验名称/运行命令.sh`。
- 实验参数应尽量写入 `配置.yaml`，不要长期依赖一整行命令行参数。
- 重要参数的含义、当前值和修改建议应写入 `参数说明.md`。

### 3. 实验记录语言规则

- 实验记录默认使用中文撰写。
- 图片命名默认优先采用中文命名，推荐格式：`图1_中文描述.png`。
- 如果实验目录中存在关键结果图，撰写 `实验记录.md` 时应自动插入合适图片，并给出简短中文说明。

### 4. 长期记录规则

- 长期研究状态保存在 `docs/research/PROJECT_STATE.md`。
- 代码与逻辑修改记录保存在 `docs/research/CHANGELOG.md`。
- 实验索引保存在 `docs/research/EXPERIMENT_INDEX.md`。
- 当代码有修改时，应更新 `CHANGELOG.md`。
- 当新增或完成实验时，应更新 `EXPERIMENT_INDEX.md`。
- 当项目阶段、当前主版本、当前最佳结果或主要问题发生变化时，应更新 `PROJECT_STATE.md`。

### 5. 修改记录分类

`CHANGELOG.md` 中的修改记录应区分以下类型：

- `BUGFIX`：修正错误，不改变预期设计。
- `LOGIC_CHANGE`：改变实现逻辑、训练流程、实验逻辑或系统行为。
- `REFACTOR`：不改变行为，仅优化结构与可读性。
- `CONFIG_CHANGE`：仅修改配置。
- `EXPERIMENT_ONLY`：仅增加或调整实验脚本、实验记录、实验目录。

### 6. 版本隔离规则

- 当进行较大修改或创建新版本时，应建立独立版本目录、独立配置、独立实验输出目录。
- 新版本不应隐式依赖旧版本的核心实现。
- 公共复用代码应放在 `common/`、`utils/` 或明确共享的模块中。

### 7. 知识笔记规则

- `notes/` 用于保存理论、知识、实验分析、问题复盘、文献理解和方法比较。
- `notes/` 中的笔记默认使用中文。
- `notes/` 中优先使用 Foam/Obsidian 风格的 `[[双链]]`。
- 实验原始结果保存在 `experiments/`，不要把完整实验工件复制进 `notes/`。
- 实验分析笔记应链接到对应的 `experiments/.../实验记录.md`。
- 新建知识笔记时，应尽量补充相关概念、问题、版本或实验链接。
- 避免孤立笔记。每个重要笔记至少应有 1 个上级索引链接和 1 个相关内容链接。

### 8. Skill 创建规则

- 当用户要求把重复工作流沉淀为 skill 时，使用 `skill-creation-manager`。
- 新 skill 默认创建在 `.agents/skills/<skill-name>/SKILL.md`。
- 创建或更新 skill 时，必须写入具体技术细节。
- 新 skill 不应只写原则性描述，应明确路径、文件名、模板、命令、检查项、完成标准和禁止事项。
- 如果某个检查可以脚本化，应优先建议创建脚本，而不是只写自然语言规则。
- 不要为一次性任务、临时 debug 或尚未稳定的算法判断创建长期 skill。

### 9. 行为要求

- 使用 Codex 原生计划模式即可，不需要重复输出冗长计划。
- 不要主动引入与当前任务无关的算法审查。
- 优先保证实验归档、命令管理、修改记录、版本隔离、知识链接、结果整理的一致性。
