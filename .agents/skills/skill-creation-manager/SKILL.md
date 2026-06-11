---
name: skill-creation-manager
description: 当用户明确要求创建、更新、细化、沉淀 Codex skill，或希望把重复工作流转化为 repo-local skill 时使用。创建 skill 时必须写入具体技术细节、路径、命令、模板和检查项，避免泛泛描述。
---

# skill-creation-manager

## 作用

将重复出现的科研开发流程沉淀为 repo-local Codex skill。

本 skill 的核心要求是：创建出来的 skill 必须像“可执行工作手册”，而不是抽象原则说明。

不要只写：

- “整理实验结果”。
- “更新相关文档”。
- “检查版本依赖”。
- “维护知识链接”。

必须写清楚：

- 整理到哪个目录。
- 创建哪些文件。
- 文件内容包含哪些字段。
- 用什么命令检查。
- 什么情况算完成。
- 什么情况禁止。
- 给出示例结构或示例片段。

## 适用场景

- “帮我创建一个 skill”。
- “把这个流程整理成 skill”。
- “这个以后经常用，做成 skill”。
- “更新某个 skill”。
- “优化现有 skill”。
- “这个 skill 写得太泛了”。
- “把技术细节写具体”。
- “以后 Codex 不要每次重新推导怎么做”。

## 不适用场景

- 不用于普通代码实现。
- 不用于直接运行实验。
- 不用于直接分析实验结果。
- 不用于理论审查。
- 不用于把一次性任务沉淀为长期 skill。

## 创建前判断

### 适合做 skill

- 这是一个会重复出现的工作流。
- 包含多个明确步骤。
- 有稳定输入和输出。
- 有固定目录或文件位置。
- Codex 以后应该按同一流程执行。
- 流程中存在容易忘记的细节。

### 适合写入 AGENTS.md

- 是全项目长期规则。
- 内容很短。
- 对多数任务都适用。
- 不需要复杂步骤。

### 适合写成脚本

- 可以确定性检查。
- 可以自动运行。
- 不需要语言模型判断。
- 例如检查目录结构、版本依赖、缺失文件、命名规范。

### 不应沉淀

- 一次性实验想法。
- 临时 debug 过程。
- 尚未稳定的算法判断。
- 只适用于当前一次对话的偏好。
- 无法明确触发的宽泛规则。

## 标准输出位置

repo-local skill 默认创建在：

```text
.agents/skills/<skill-name>/SKILL.md
```

如果需要附加模板、示例或脚本，可以创建：

```text
.agents/skills/<skill-name>/references/
.agents/skills/<skill-name>/scripts/
.agents/skills/<skill-name>/assets/
```

## 命名规则

skill 名称使用英文小写短横线：

```text
experiment-archive-manager
experiment-report-writer
knowledge-link-maintainer
version-dependency-checker
```

禁止使用过宽名称：

```text
general-helper
research-agent
do-everything
code-assistant
```

## 创建 skill 时必须包含的技术细节

每个新 skill 的 `SKILL.md` 必须尽量包含以下部分。

### 1. 固定路径

明确该 skill 会读取或修改哪些路径。

示例：

```text
读取：
- experiments/<实验目录>/
- experiments/<实验目录>/图片/
- experiments/<实验目录>/指标结果.json

写入：
- experiments/<实验目录>/实验记录.md
- docs/research/EXPERIMENT_INDEX.md
```

### 2. 固定文件名

明确必须创建或更新哪些文件。

示例：

```text
必须创建：
- 实验清单.yaml
- 运行命令.sh
- 实验记录.md
- 图片/
- 日志/

可选创建：
- 指标结果.json
- 代码变更.patch
- 配置快照/
```

### 3. 文件模板

如果 skill 会创建 Markdown、YAML、JSON、shell 脚本，必须给出模板。

### 4. 命令模板

如果 skill 涉及 shell、git、python、pytest、脚本检查，必须写出命令模板。

示例：

```bash
git status --short
git diff > experiments/<实验目录>/代码变更.patch
python scripts/checks/check_version_dependency.py --target <版本目录>
```

### 5. 操作步骤必须具体

不要只写：

```text
1. 分析需求
2. 更新文件
3. 输出结果
```

应写成：

```text
1. 根据当前日期生成实验目录名：experiments/YYYY-MM-DD_中文实验名称/
2. 创建 图片/、日志/、配置快照/、检查点/ 子目录
3. 将用户给出的运行命令写入 运行命令.sh
4. 将实验目的、日期、命令、配置路径写入 实验清单.yaml
5. 如果存在图片，则按 图1_中文描述.png 的方式整理到 图片/
6. 初始化 实验记录.md，并插入已有关键图片
7. 更新 docs/research/EXPERIMENT_INDEX.md
```

### 6. 完成标准

每个 skill 必须写清楚“什么叫完成”。

### 7. 检查项

每个 skill 必须包含 checklist。

### 8. 禁止事项

必须写清楚容易犯的具体错误。

### 9. 示例输入输出

如果 workflow 复杂，必须给出至少一个示例。

## SKILL.md 标准结构

每个新 skill 推荐使用以下结构：

```md
---
name: <skill-name>
description: <明确触发条件 + 不适用场景>
---

# <skill-name>

## 作用

## 适用场景

## 不适用场景

## 输入

## 固定路径

## 必须创建或更新的文件

## 文件模板

## 操作流程

## 检查项

## 完成标准

## 输出要求

## 禁止事项

## 示例
```

## 创建流程

1. 确认用户想沉淀的是哪类重复工作流。
2. 判断应创建 skill、写入 AGENTS.md、写脚本，还是不沉淀。
3. 检查 `.agents/skills/` 下是否已有相似 skill。
4. 如果已有相似 skill，优先更新旧 skill，不要重复创建。
5. 确定新 skill 名称。
6. 创建或更新 `.agents/skills/<skill-name>/SKILL.md`。
7. 写入具体技术细节，包括路径、文件名、模板、命令、检查项、完成标准。
8. 如果需要脚本或模板，创建 `scripts/` 或 `references/`。
9. 如果该 skill 改变全局规则，建议同步更新 `AGENTS.md`。
10. 输出新增或修改的文件路径。

## 输出要求

完成后输出：

```text
创建/更新的 skill：
- <skill-name>

路径：
- .agents/skills/<skill-name>/SKILL.md

主要触发场景：
- ...

写入的具体技术细节：
- 固定路径：
- 文件模板：
- 命令模板：
- 检查项：
- 完成标准：

是否建议更新 AGENTS.md：
- 是/否

是否建议新增脚本：
- 是/否
```

## 质量标准

一个合格 skill 应满足：

- Codex 不需要重新推导主要步骤。
- 路径明确。
- 文件明确。
- 模板明确。
- 检查项明确。
- 输出格式明确。
- 不适用场景明确。
- 禁止事项具体。
- 不依赖当前对话上下文也能独立使用。

## 禁止事项

- 不要创建只有原则没有操作细节的 skill。
- 不要用“适当”“相关”“必要时”等模糊表达替代具体路径和文件。
- 不要把所有需求都塞进一个大 skill。
- 不要为一次性任务创建长期 skill。
- 不要自动创建算法专属 skill，除非用户明确要求。
- 不要把确定性检查只写成自然语言，应建议脚本化。
