# paper2data — 论文 PDF → 机器学习数据集

把科学论文 PDF 变成可喂 ML 的结构化数据集：从**正文、表格、图表**里提取数据，
按字段库（schema）整理成「一行一条试验」的 CSV。

**方法原则**：能从结构里确定性拿到的，绝不让模型猜——数值由代码测量 / 结构取值，
模型只负责识别与兜底。这也是本项目和「纯 LLM 读图」的根本区别。

## 数据流

```
论文 PDF
  → parser            pdfplumber 读文本（布局感知）+ PyMuPDF 提线框表 + 提取图片
  → table_detector    坐标聚类还原无边框表
  → figure_extract    图 → 文本：**代码切块** + vision 识别/读数
  → extract           调 LLM 抽取（records + new_fields）
  → verify            溯源（页码 + 证据句）+ 规则校验
  → compute           数值归一化（单位换算、空值统一）+ computed 字段硬算
  → export            DataFrame → CSV
```

## 快速开始

```bash
# 一篇论文 → CSV（字段库默认 paper2data/schema.json）
.venv/Scripts/python.exe -m paper2data 论文.pdf

# 跳过图提取（不调 vision，快很多；溯源仍对正文/表格生效）
.venv/Scripts/python.exe -m paper2data 论文.pdf --no-figures

# 调试：落盘中间产物（每张图的提取文本 + 表格检测结构）
.venv/Scripts/python.exe -m paper2data 论文.pdf --debug

# 单测
.venv/Scripts/python.exe -X utf8 -m unittest discover -s paper2data/tests -v
```

> 用 venv 的 python（项目跑在 3.11）；裸 `python` 是 3.12。
> Windows cmd 编码出错加 `-X utf8`。

## 数据集与评测

**真值**：`data/多元固废数据搜集.xlsx`（材料专业学生逐篇读论文整理）。
旧 `paper2data/benchmarks/*.csv` 是早期 LLM 生成、未审核的，**已弃用**。

| 文件 / 目录 | 内容 |
|---|---|
| `data/dev_papers.json` | 开发集 14 篇 / 327 行（可反复跑、反复调） |
| `data/test_papers.json` | 测试集 15 篇 / 257 行（**已冻结**，只评分和找问题） |
| `data/开发集/`、`data/测试集/` | 对应 PDF（复制，原 210 篇库不动） |
| `data/benchmarks/*.csv` | 每篇真值，59 字段 + `_xlsx_row` |
| `data/benchmarks/CORRECTIONS.md` | 真值修正记录（只改有原文证据的） |

```bash
# 生成 benchmark（可重建，别手改 CSV）
.venv/Scripts/python.exe scripts/build_benchmarks.py

# 评分：工具输出 vs 真值
.venv/Scripts/python.exe scripts/evaluate.py data/benchmarks/dev02.csv output/dev02.csv
```

**评分口径**（详见 `PAPER2DATA_METHODS.md` §6）：
- 行匹配：真值定基准 → 22 个原料算相似度 → 候选图上跑匈牙利（**允许不匹配**）；
  多候选时才用工艺字段辅助；性能字段**完全不参与匹配**（防循环论证）。
- 空值：真值空+工具空跳过；真值有+工具空=漏抽；真值空+工具有=「工具多填」（单列）。
- 三种指标分开报：行级 / 字段级 / 覆盖率。

**纪律**：测试集只能用来评分和找问题——发现某类失败记成未来研究方向，
**不许针对它加规则再用同一批证明有效**。

## 输出

- **CSV**：纯数值表，缺失 = 空（pandas 读为 NaN），可直接喂 pandas / CTGAN。
- **`.provenance.csv`**：每个值的页码 + 证据句 + 是否找到（零 LLM 溯源）。
- 列 = `schema.output` 的 40 列（与 `输出样式例.xlsx` 逐字一致）在前，
  字段库其余字段追加，动态新字段排在最后。

## 核心概念

### 字段库（`schema.json`）

```json
{
  "name": "MgO", "label": "氧化镁总含量", "category": "成分",
  "unit": "wt%", "type": "continuous", "computed": true,
  "desc": "混合原料 MgO 总含量",
  "compute": {"kind": "weighted_oxide", "oxide": "MgO"}
}
```

- `output`：最终数据集的列顺序（40 列，对应 `输出样式例.xlsx`）。
- 字段库可以比输出宽——多出来的字段帮 LLM 识别，不会丢数据。

### computed 字段

LLM 不读、代码硬算：

- `{"kind": "weighted_oxide", "oxide": "MgO"}` — 混合氧化物 = Σ(配比 × 成分)
- `{"kind": "molar_ratio", "num": "CaO", "den": "SiO2"}` — 摩尔比

⚠️ 计算依赖 `paper2data/composition.json`，**目前只有 3 种原料**（GGBS/PG/SS），
且不区分论文来源。默认**不传 `-c` 就不计算**；传了也要看覆盖率告警。

### 读图：默认 vision 读数

- **切块永远走代码**（`detect_plot_frames` 找长直线轴框）——vision 分块会按格子均分。
- **数值默认交给 vision**；设 `PAPER2DATA_FIGURE_VALUES=code` 才走代码标定 + 反算（对照实验用）。
  代码读柱状图还不完善（斜纹/粘连/刻度相位），待重设计。

### 空值约定

论文没提的字段显式填 `null`；`N/A` / `未提及` / `-` 等字符串在归一化时统一成空。
**0 是合法值**（原料不掺、性能为 0），不会被当空。

## 目录结构

```
paper2data/
├── cli.py                 命令行入口（PDF → CSV）
├── parser.py              读 PDF（文本/线框表/图片）
├── table_detector.py      无边框表坐标聚类
├── figure_extract.py      图 → 文本（切块 + vision）
├── chart_extract.py       柱状图代码测量（标定/检柱/反算）
├── schema.py / schema.json 字段库
├── records.py / prompts.py / extract.py / client.py   LLM 抽取
├── compute.py             归一化 + computed 硬算
├── export.py              DataFrame → CSV
├── verify/                溯源（provenance）+ 规则校验（rules）
├── tests/                 单元测试
└── composition.json       原料成分库（待扩充）

scripts/
├── build_benchmarks.py    从 xlsx 生成 benchmark
└── evaluate.py            评分（工具输出 vs 真值）

data/                      数据集（PDF 不入库）
PAPER2DATA_METHODS.md      独特方法档案（写论文用）
PAPER2DATA_STATUS.md       当前进度 / 卡点 / 待办
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | — | LLM 抽取 |
| `PAPER2DATA_MODEL` | `deepseek-v4-flash` | 文本模型 |
| `PAPER2DATA_VISION_API_KEY` | 回退 `DEEPSEEK_API_KEY` | 读图模型 key（可接 SiliconFlow / OpenRouter / 智谱） |
| `PAPER2DATA_VISION_BASE_URL` | 回退 `DEEPSEEK_BASE_URL` | 读图模型 base_url |
| `PAPER2DATA_VISION_MODEL` | `deepseek-v4-flash-vision-exp` | 读图模型名 |
| `PAPER2DATA_FIGURE_VALUES` | `vision` | 柱状图读数：`vision` / `code` |
| `PAPER2DATA_GEOMETRIC_FALLBACK` | `off` | 逐色法失败后是否试几何法（仅 `code` 模式） |

## 文档

- `CLAUDE.md` — 项目常量与硬规则（AI 协作入口）
- `PAPER2DATA_STATUS.md` — 当前状态、卡点、待办
- `PAPER2DATA_METHODS.md` — 独特方法档案（问题/方法/差异/实测/边界）
