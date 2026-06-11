# 科研开发 Skill 框架使用说明

将本目录内容复制到仓库根目录后，Codex 可以基于以下结构工作：

- `AGENTS.md`：项目级规则。
- `.agents/skills/`：可复用工作流 skill。
- `experiments/`：实验工件。
- `docs/research/`：长期项目记录。
- `notes/`：Foam/Obsidian 风格知识笔记。
- `templates/`：实验、参数、笔记模板。
- `scripts/checks/`：确定性检查脚本。

## 推荐 VS Code 插件

见 `.vscode/extensions.json`。

建议安装：

- Foam
- Markdown All in One
- markdownlint
- Todo Tree
- GitLens

## 典型用法

### 新建实验

让 Codex 使用：

- `experiment-archive-manager`
- `experiment-command-manager`

目标：创建实验目录、短运行脚本、配置文件、参数说明。

### 整理实验结果

让 Codex 使用：

- `experiment-report-writer`
- `research-log-maintainer`

目标：更新 `实验记录.md`，插入关键图片，更新实验索引。

### 做知识笔记

让 Codex 使用：

- `knowledge-note-manager`
- `knowledge-link-maintainer`

目标：在 `notes/` 中创建中文双链笔记，链接理论、问题、版本和实验分析。

### 创建新 skill

让 Codex 使用：

- `skill-creation-manager`

目标：创建具体、可执行、低歧义的 repo-local skill。
