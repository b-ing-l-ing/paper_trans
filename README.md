# Paper Trans

论文 PDF → 结构化 Markdown + JSON

## 使用

```bash
uv sync
cp .env.example .env        # 填入 DeepSeek API key
# 改 paper_trans.py 末尾的 pdf_path
uv run python paper_trans.py
```

输出：`output/` 目录下生成 `.md` + `.json` + 图片。
