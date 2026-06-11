---
name: paper-pdf-ingest-manager
description: 当用户将 PDF 论文放入仓库，要求转换为 Markdown、保留公式/图片/排版，或希望为后续 Codex 论文笔记生成准备输入时使用。不用于直接写论文综述。
---

# paper-pdf-ingest-manager

## 作用

将论文 PDF 标准化入库，并转换成 Codex 更容易读取的 Markdown。

核心目标：

- 保留公式、图片、表格和阅读顺序
- 不直接依赖一次性 PDF 解析
- 为 `paper-note-manager` 提供稳定输入

## 固定路径

读取：

```text
papers/pdf/<论文文件>.pdf
```

写入：

```text
papers/parsed/<slug>/<slug>.md
papers/parsed/<slug>/解析检查.md
papers/metadata/<slug>.yaml
```

## 推荐工具

主工具：Marker

原因：更适合学术 PDF，可输出 Markdown、保存图片、处理表格和公式。

备用工具：PyMuPDF

仅用于快速文本预览，不作为公式、图片、表格密集论文的主方案。

## 操作流程

1. 确认 PDF 已放入 `papers/pdf/`
2. 为论文确定英文 slug，例如 `safe_dreamer`
3. 使用 Marker 解析 PDF：

```bash
python scripts/papers/pdf_to_markdown_marker.py \
  --pdf papers/pdf/<文件名>.pdf \
  --slug <slug> \
  --title "<论文标题>" \
  --quality math
```

4. 如果论文公式、图表、双栏排版复杂，或第一次结果较差，改用：

```bash
python scripts/papers/pdf_to_markdown_marker.py \
  --pdf papers/pdf/<文件名>.pdf \
  --slug <slug> \
  --title "<论文标题>" \
  --quality llm
```

5. 运行解析检查：

```bash
python scripts/papers/check_paper_parse.py --slug <slug>
```

6. 检查以下文件是否存在：

```text
papers/parsed/<slug>/<slug>.md
papers/parsed/<slug>/解析检查.md
papers/metadata/<slug>.yaml
```

7. 如果解析质量可接受，再调用 `paper-note-manager` 生成中文论文笔记。

## 质量模式

### standard

适合普通数字 PDF，速度较快。

### math

默认推荐。更关注 inline math 和公式质量。

### llm

适合复杂版面、跨页表格、公式和阅读顺序要求高的论文。需要按 Marker 要求配置 LLM 后端。

## 检查项

- Markdown 是否生成
- 图片是否被提取并能被 Markdown 正确引用
- 公式是否尽量转换为 LaTeX
- 表格是否仍保持可读结构
- 双栏论文是否存在阅读顺序错乱
- `papers/metadata/<slug>.yaml` 是否记录 PDF、Markdown、解析质量模式

## 完成标准

- `papers/parsed/<slug>/<slug>.md` 存在
- `papers/metadata/<slug>.yaml` 存在
- `解析检查.md` 已生成
- Codex 可以基于 parsed Markdown 继续写论文笔记

## 禁止事项

- 不要只把 PDF 路径丢给 Codex 后直接要求总结
- 不要用 PyMuPDF 备用文本作为高质量论文解析主输入
- 不要忽略图片链接缺失
- 不要把 parsed Markdown 写进 `notes/文献/`
- 不要把论文笔记和 PDF 解析结果混在一个文件里
