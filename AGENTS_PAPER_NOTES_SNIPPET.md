# AGENTS.md 可追加片段：论文入库与知识笔记规则

## 论文入库规则

- 原始 PDF 放入 `papers/pdf/`。
- PDF 不应直接作为长期笔记输入；应先转换为 `papers/parsed/<slug>/<slug>.md`。
- 论文元信息放入 `papers/metadata/<slug>.yaml`。
- 论文知识笔记放入 `notes/文献/`，默认中文。
- 论文笔记应使用 Foam/Obsidian 风格 `[[双链]]`，连接概念、理论、方法比较、实验分析和问题。
- 如果用户关注公式、图片、表格和排版，优先使用 Marker 解析 PDF，而不是普通文本抽取。
- PyMuPDF fallback 只用于快速文本预览，不作为高质量论文解析主输入。
