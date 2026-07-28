# Paper Trans

从科学论文中提取结构化数据，输出 Markdown + JSON。**数据优先**：表格、图表、数值数据全面提取，实验方法概括性总结。

## 功能

- **PDF 提取** — 纯文本（DeepSeek）+ 多模态（Gemini 看图）两种模式
- **结构化表格检测** — PDF 坐标聚类恢复二维表格，空单元格精确定位
- **自审修正** — 纯文本模式支持二次审核纠错（审查同样使用结构化表格）
- **网页爬取** — 浏览器加载页面，支持登录态保持，自动降级下载 PDF
- **批量处理** — 一键处理目录下所有 PDF
- **Web UI** — 浏览器上传 PDF，在线查看结果
- **双重输出** — Markdown（人读）+ JSON（机读）

## 安装

```bash
uv sync
cp .env.example .env                  # 填入 API key
uv run playwright install chromium      # 网页爬取需要
```

## 使用

### CLI

```bash
# 纯文本（DeepSeek）
uv run paper-trans extract paper.pdf
uv run paper-trans extract paper.pdf --no-review

# 多模态（Gemini 看图）
uv run paper-trans extract paper.pdf --multimodal -m gemini-3.6-flash

# 批量处理
uv run paper-trans batch PDFs/
uv run paper-trans batch PDFs/ --multimodal -m gemini-3.6-flash

# 网页版
uv run paper-trans-web
# → 打开 http://127.0.0.1:5000
```

### 网页爬取

```bash
uv run paper-trans login https://publisher.com/login
uv run paper-trans scrape https://arxiv.org/abs/2102.00554
```

### Python API

```python
from paper_trans.parser import parse_pdf
from paper_trans.pdf_parser import detect_tables
from paper_trans.extractor import extract_with_review
from paper_trans.renderer import render_markdown, save_markdown, save_json

text, pages, images = parse_pdf("paper.pdf")
tables = detect_tables("paper.pdf")
data = extract_with_review(text, tables=tables)
md = render_markdown(data, "paper.pdf")
save_markdown(md, data, "./output")
save_json(data, "./output")
```

## 项目结构

```
paper_trans/
├── cli.py                # Typer 命令行
├── config.py             # 配置管理
├── client.py             # LLM 客户端（DeepSeek + Gemini）
├── extractor.py          # 数据提取 + 自审 + 多模态
├── parser.py             # PDF 解析 + 图片提取
├── prompts.py            # 系统提示词
├── renderer.py           # Markdown / JSON 渲染
├── schemas.py            # Pydantic 数据模型
├── web.py                # Web UI（Flask）
├── pdf_parser/           # PDF 结构化解析
│   └── table_detector.py # 表格检测 + 坐标聚类
└── scraper/              # 网页爬取
    ├── browser.py        # Playwright 会话管理
    └── html_extractor.py # HTML → 文本 → 数据
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `GEMINI_API_KEY` | - | Gemini API 密钥（多模态） |
| `PAPER_TRANS_MODEL` | `deepseek-v4-flash` | 默认模型 |
| `PAPER_TRANS_MULTIMODAL_MODEL` | `gemini-3.6-flash` | 多模态模型 |
| `PAPER_TRANS_REVIEW_LIMIT` | `15000` | 审查时原文截断长度 |
| `PAPER_TRANS_OUTPUT_DIR` | `./output` | 输出目录 |
| `PAPER_TRANS_IMAGE_DIR` | `./output/images` | 图片目录 |

## 路线图

- [x] PDF 解析 + 图片提取
- [x] LLM 结构化提取 + 自审
- [x] 通用数据模型（不绑定特定学科）
- [x] 网页爬取 + 登录态管理
- [x] PDF 自动降级（摘要页 → 自动下载 PDF）
- [x] 页眉/装饰图过滤（hash 去重）
- [x] 多模态图表提取（Gemini）
- [x] 结构化表格检测（坐标聚类，空单元格精确定位）
- [x] 批量处理
- [x] Web UI
- [ ] 多模型支持（Claude, GPT 等）
