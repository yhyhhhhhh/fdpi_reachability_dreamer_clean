---
name: paper-note-manager
description: 当已有 papers/parsed/<slug>/<slug>.md，用户要求基于论文内容创建中文知识笔记、关联理论/实验/问题并建立 Foam/Obsidian 风格双链时使用。不用于 PDF 转换。
---

# paper-note-manager

## 作用

基于解析后的论文 Markdown，生成中文论文知识笔记，并连接到项目知识库。

论文笔记不是普通摘要。它应回答：

- 这篇论文解决什么问题？
- 方法核心是什么？
- 和我的研究、实验或代码有什么关系？
- 哪些点可以借鉴，哪些点不适用？
- 后续可以产生什么实验假设？

## 固定输入

```text
papers/parsed/<slug>/<slug>.md
papers/metadata/<slug>.yaml
```

## 固定输出

```text
notes/文献/<论文标题或短名>.md
```

可选更新：

```text
notes/项目地图.md
notes/概念/*.md
notes/理论/*.md
notes/方法比较/*.md
notes/实验分析/*.md
```

## 操作流程

1. 读取 `papers/metadata/<slug>.yaml`
2. 读取 `papers/parsed/<slug>/<slug>.md`
3. 先提取论文基本信息：标题、年份、作者、任务、方法、实验、结论
4. 生成 `notes/文献/<论文名>.md`
5. 使用中文撰写
6. 加入 PDF 和 parsed Markdown 的相对链接
7. 补充 Foam/Obsidian 风格 `[[双链]]`
8. 如果和已有实验相关，链接到 `notes/实验分析/` 或 `experiments/.../实验记录.md`
9. 如果涉及稳定概念，链接到 `notes/概念/`
10. 如果涉及理论机制，链接到 `notes/理论/`

## 标准模板

```md
# <论文标题>

文献条目：[@citekey]

PDF：
[papers/pdf/<文件名>.pdf](../../papers/pdf/<文件名>.pdf)

解析文本：
[papers/parsed/<slug>/<slug>.md](../../papers/parsed/<slug>/<slug>.md)

相关概念：
- [[概念1]]
- [[概念2]]

相关理论：
- [[理论1]]

相关实验：
- [[实验分析笔记]]

## 1. 一句话总结

## 2. 研究问题

## 3. 方法核心

## 4. 关键机制

## 5. 实验设计

## 6. 主要结论

## 7. 对我当前项目的启发

## 8. 局限与不适用点

## 9. 后续可验证想法

## 10. 原文摘录与页码
```

## 双链规则

每篇论文笔记至少包含：

- 2 个相关概念链接
- 1 个相关理论或方法链接
- 如果有对应实验，至少 1 个实验分析链接
- 如果有开放问题，至少 1 个问题链接

## 完成标准

- `notes/文献/<论文名>.md` 已创建或更新
- 笔记为中文
- 包含 PDF 与 parsed Markdown 链接
- 包含相关双链
- 至少写出“对我当前项目的启发”和“局限与不适用点”

## 禁止事项

- 不要只做普通摘要
- 不要把 parsed Markdown 全文复制进 notes
- 不要无根据扩展论文没有说的结论
- 不要省略和当前项目的关系
- 不要把实验原始日志复制进论文笔记
