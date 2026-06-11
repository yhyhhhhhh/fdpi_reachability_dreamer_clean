---
name: wandb-access-manager
description: 当需要在 IsaacLab 环境中验证 W&B 登录态、查询 wandb project/run/metrics、或规避 wandb.Api() 认证异常时使用。不用于训练算法分析或修改实验配置。
---

# wandb-access-manager

## 作用

在本仓库的 IsaacLab 训练环境中，稳定、脱敏地访问 Weights & Biases（wandb / W&B）数据。

本 skill 重点处理三类问题：

- 确认当前 shell 是否真的使用 IsaacLab Python 环境。
- 确认 `/root/.netrc` 中的 W&B 登录凭据是否可用，但不暴露 API key。
- 当 `wandb.Api()` / Public API wrapper 报 `relogin required` 时，改用窄字段 GraphQL 请求读取 viewer、project、run 和 history 数据。

## 适用场景

- 用户问“现在能不能访问 wandb”。
- 用户已经在 IsaacLab 环境中 `wandb login`，需要 Codex 读取 W&B 数据。
- 需要列出当前账号、entity、project、run 列表或 run 的指标曲线。
- 需要从 W&B 拉取指标，用于更新 `experiments/<实验目录>/实验记录.md`、生成曲线或整理实验结论。
- `wandb login --verify` 成功，但 Python 中 `wandb.Api()` 报 `relogin required`。

## 不适用场景

- 不用于修改训练算法、reward、risk critic 或 replay 逻辑。
- 不用于创建实验目录；新增实验目录应使用 `experiment-archive-manager`。
- 不用于撰写完整实验报告；报告整理应使用 `experiment-report-writer`。
- 不用于把完整 W&B 历史数据长期复制进 `notes/`。

## 输入

常见输入包括：

- entity：默认优先从 W&B viewer 查询得到，本项目当前常见值为 `2332133796-yhyper`。
- project：默认从配置读取，本项目当前主项目为 `IsaacLab22-SurgicalRobot5-FDPI-Reachability`。
- run id / run name：W&B run 的短 ID，例如 `n60y55jj`，或 run 的 `name` 字段。
- metrics keys：需要拉取的指标名，例如 `_step`、`eval/return`、`train/reward`、`cost/*` 等。
- samples：历史曲线抽样点数，常用 `500` 或 `2000`。

## 固定路径

读取：

- `/opt/conda/envs/isaaclab/bin/python`：IsaacLab 环境 Python。
- `/root/.netrc`：wandb CLI 登录后保存的凭据，禁止打印其中 password。
- `configs/reachability_gp_isaaclab22.yaml`：默认 W&B project/group/name 来源。
- `configs/reachability_gp_isaaclab22_*.yaml`：派生实验配置中的 W&B 覆盖项。
- `experiments/<实验目录>/实验记录.md`：需要结合 W&B 指标整理实验时读取或更新。

写入：

- 正常访问检查不写入文件，只输出脱敏状态。
- 若用户要求沉淀结果，优先写入 `experiments/<实验目录>/指标结果.json`、`experiments/<实验目录>/图片/` 和 `experiments/<实验目录>/实验记录.md`。

本 skill 自带脚本：

- `.agents/skills/wandb-access-manager/scripts/verify_wandb_access.py`

## 必须创建或更新的文件

普通 W&B 访问任务不必须创建文件。

如果用户要求“把 W&B 结果整理进实验”，应按实验记录规则更新：

- `experiments/<实验目录>/指标结果.json`
- `experiments/<实验目录>/图片/图N_中文描述.png`
- `experiments/<实验目录>/实验记录.md`
- 必要时更新 `docs/research/EXPERIMENT_INDEX.md`

## 命令模板

确认当前 Python 环境：

```bash
which python
python - <<'PY'
import wandb
print(wandb.__version__)
PY
```

验证 wandb CLI 登录态，输出可显示用户名和 entity，但不得显示 API key：

```bash
wandb login --verify
```

运行本 skill 的脱敏访问检查：

```bash
/opt/conda/envs/isaaclab/bin/python .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py \
  --project IsaacLab22-SurgicalRobot5-FDPI-Reachability \
  --list-projects \
  --list-runs
```

只检查 viewer 和当前 project 是否可访问：

```bash
/opt/conda/envs/isaaclab/bin/python .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py \
  --project IsaacLab22-SurgicalRobot5-FDPI-Reachability
```

抽样读取某个 run 的历史指标：

```bash
/opt/conda/envs/isaaclab/bin/python .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py \
  --project IsaacLab22-SurgicalRobot5-FDPI-Reachability \
  --run n60y55jj \
  --show-history-keys \
  --history-key-filter entropy
```

```bash
/opt/conda/envs/isaaclab/bin/python .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py \
  --project IsaacLab22-SurgicalRobot5-FDPI-Reachability \
  --run n60y55jj \
  --history-keys _step,eval/return,train/reward \
  --samples 500
```

## 操作流程

1. 先确认环境：运行 `which python`，应优先看到 `/opt/conda/envs/isaaclab/bin/python`。如果不是 IsaacLab 环境，显式使用 `/opt/conda/envs/isaaclab/bin/python` 执行检查脚本。
2. 检查 `wandb` 是否安装：导入 `wandb` 并打印版本。本仓库已验证过 `wandb==0.27.2` 可用。
3. 检查 `/root/.netrc` 是否存在，且包含 `api.wandb.ai`。只允许输出 host、login、password 长度，不允许输出 password 内容。
4. 运行 `wandb login --verify`。如果失败，告知用户在 IsaacLab 环境中执行 `wandb login --relogin`，不要让用户把 API key 发到对话里。
5. 尝试 `wandb.Api()` 时，如果出现 `WandbApiFailedError: relogin required`，不要直接判定 W&B 不可访问。本仓库环境中曾出现 CLI 登录成功但 Public API wrapper 失败的情况。
6. 使用窄字段 GraphQL 请求验证 viewer：只请求 `viewer { id username entity }`，避免请求 `apiKeys`、`email` 等敏感或易触发权限错误的字段。
7. 列 project 时使用窄字段 GraphQL：只请求 `models { edges { node { id name entityName createdAt } } }`。不要使用 W&B 生成的 `GET_PROJECTS_GQL` 原样查询，因为其中的 `UserFragment` 会请求 `apiKeys` 字段，可能触发 401。
8. 读取 run 列表时优先请求轻量字段：`name`、`displayName`、`state`、`createdAt`、`heartbeatAt`、`group`、`tags`、`historyLineCount`。只有确实需要结论时再请求 `summaryMetrics` 或 history。
9. 读取曲线前先用 `--show-history-keys` 或 `historyKeys` 查询确认实际指标名，再用 `sampledHistory` 抽样，并限制 `samples`。只在用户明确需要完整数据时使用 full history。
   - 当前 W&B 返回的 `historyKeys` 可能是 `{"keys": {...}, "lastStep": ..., "sets": ...}` 结构；指标名应从 `historyKeys["keys"]` 展开。
10. 如果要写入实验记录，把原始指标摘要写到 `experiments/<实验目录>/指标结果.json`，把图写到 `experiments/<实验目录>/图片/`，再用中文更新 `实验记录.md`。

## GraphQL 请求模板

所有请求都应从 `/root/.netrc` 读取 key，并通过 Basic Auth 发送；代码不得打印 key。

Viewer：

```graphql
query Viewer {
  viewer { id username entity }
}
```

Projects：

```graphql
query NarrowProjects($entity: String!, $perPage: Int = 10) {
  models(entityName: $entity, first: $perPage) {
    edges { node { id name entityName createdAt } }
  }
}
```

Runs：

```graphql
query Runs($project: String!, $entity: String!, $perPage: Int = 10, $order: String) {
  project(name: $project, entityName: $entity) {
    name
    entityName
    runCount
    runs(first: $perPage, order: $order) {
      edges {
        node { name displayName state createdAt heartbeatAt group tags historyLineCount }
      }
    }
  }
}
```

Sampled history：

```graphql
query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
  project(name: $project, entityName: $entity) {
    run(name: $name) { sampledHistory(specs: $specs) }
  }
}
```

其中 `specs` 示例：

```json
[{"keys": ["_step", "eval/return", "train/reward"], "samples": 500}]
```

## 检查项

- 当前 Python 是否为 IsaacLab 环境，或命令是否显式使用 `/opt/conda/envs/isaaclab/bin/python`。
- `wandb` 是否可导入，版本是否输出成功。
- `/root/.netrc` 是否包含 `api.wandb.ai`，且没有泄露 password。
- `wandb login --verify` 是否成功。
- 窄字段 GraphQL viewer 是否返回 username/entity。
- 目标 project 是否存在于当前 entity 下。
- 如需 run 数据，目标 run 是否存在且 state/historyLineCount 可见。
- 如需写入实验记录，是否把结果放在 `experiments/<实验目录>/`，而不是复制到 `notes/`。

## 完成标准

一次 W&B 访问任务完成时，应至少明确以下结论之一：

- 可访问：给出当前 username、entity、目标 project、可见 run 或指标摘要。
- 登录缺失：说明 `wandb login --verify` 失败，并请用户在 IsaacLab 环境中运行 `wandb login --relogin`。
- 权限不足：说明 viewer 可访问但目标 project/run 不可访问，列出已验证的 entity/project。
- SDK wrapper 异常：说明 `wandb.Api()` 失败但 GraphQL 可用，并继续使用 GraphQL 路径完成读取。

## 输出要求

对用户输出时应包含：

- 当前环境：Python 路径和 wandb 版本。
- 登录状态：是否通过 verify，username/entity 可显示，API key 不可显示。
- 访问路径：`wandb.Api()` 是否可用；如不可用，说明已改用窄字段 GraphQL。
- 查询结果：project/run/指标摘要，中文简洁说明。
- 后续落盘：如果写了实验文件，列出具体路径。

## 禁止事项

- 禁止打印 `/root/.netrc` 中的 password。
- 禁止打印 `WANDB_API_KEY`、Authorization header 或任何可复用 token。
- 禁止让用户把 API key 粘贴到聊天里。
- 禁止仅凭 `wandb.Api()` 报 `relogin required` 就断言 W&B 不可访问。
- 禁止原样使用会请求 `apiKeys` 字段的宽 GraphQL fragment 来做普通 project 检查。
- 禁止把大量原始 history 数据复制进 `notes/`。
- 禁止执行删除、覆盖、停止远端 W&B run 的操作，除非用户明确要求并确认目标 run。

## 示例

用户说：“你现在可以访问到 wandb 么？”

推荐执行：

```bash
/opt/conda/envs/isaaclab/bin/python .agents/skills/wandb-access-manager/scripts/verify_wandb_access.py \
  --project IsaacLab22-SurgicalRobot5-FDPI-Reachability \
  --list-projects
```

推荐回复要点：

```text
当前 shell 使用 IsaacLab Python，wandb 已安装。
W&B 登录态有效，viewer 为 <username>，entity 为 <entity>。
已通过窄字段 GraphQL 看到项目 IsaacLab22-SurgicalRobot5-FDPI-Reachability。
wandb.Api() 如果仍报 relogin required，后续查询将继续使用 GraphQL 路径。
```
