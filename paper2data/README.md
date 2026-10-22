# paper2data

从科学论文 PDF 抽取**机器学习训练用的表格数据集**。每行一条试验,列由字段库(schema)决定,支持 LLM 动态补充模板外字段。

与主包 `paper_trans`(文档式总结抽取)不同,本包的产物是**表格**——最终导出 CSV,可直接喂给 pandas / CTGAN。

## 数据流

```
论文 PDF
  → parser          读文本 + 提取图片
  → figure_extract  多模态读图(子图分割 → OCR/代码反算 → 转文本)
  → extract         调 LLM 抽取(records + new_fields)
  → compute         归一化 + computed 字段硬算
  → verify          规则校验(比例和/值域/摩尔比)
  → export          DataFrame → CSV
```

## 安装

依赖已在根 `pyproject.toml` 里。额外需要 `pandas`(导出 CSV 用)。

```bash
uv sync
```

## 快速开始

```bash
# 最简:一篇论文一键出 CSV(字段库默认 paper2data/schema.json)
python -m paper2data 论文.pdf

# 指定输出路径
python -m paper2data 论文.pdf -o out.csv

# 带原料成分库(计算 Ca/Si、MgO 等 computed 字段)
python -m paper2data 论文.pdf -c paper2data/composition.json
```

## 命令行参数

| 参数 | 说明 | 默认 |
|---|---|---|
| `pdf` | 论文 PDF 路径(位置参数) | 必填 |
| `-s, --schema` | 字段库 JSON 路径 | `paper2data/schema.json` |
| `-o, --output` | 输出 CSV 路径 | 与 PDF 同名 |
| `-c, --composition` | 原料成分库 JSON 路径(可选) | 不计算 computed |

## 输出

- **CSV**:归一化后的纯数值表,缺失值 = 空(pandas 读为 NaN)。这是交付物,喂 CTGAN。
- 控制台打印:抽到几行、几个新字段。

## 核心概念

### 字段库(schema.json)

定义要抽取的列。每个字段:

```json
{
  "name": "MgO",              // 列名
  "label": "氧化镁总含量",      // 中文名
  "category": "成分",          // 原料/成分/工艺/性能
  "unit": "wt%",              // 单位(归一化目标)
  "type": "continuous",       // continuous/discrete/categorical(给 CTGAN)
  "computed": true,           // 是否计算字段(LLM 不填,代码硬算)
  "desc": "混合原料 MgO 总含量",
  "compute": {"kind": "weighted_oxide", "oxide": "MgO"}  // computed 的计算声明
}
```

### 计算字段(computed)

LLM 不读、代码硬算。算法由 `compute` 声明决定(通用):

- `{"kind": "weighted_oxide", "oxide": "MgO"}` — 混合氧化物含量 = Σ(配比 × 成分)
- `{"kind": "molar_ratio", "num": "CaO", "den": "SiO2"}` — 摩尔比

换领域只改 schema 声明 + 成分库,代码不动。

### 原料成分库(composition.json)

computed 字段的原料数据,结构 `{原料名: {氧化物: 百分比}}`:

```json
{"GGBS": {"SiO2": 31.48, "CaO": 34.0, "MgO": 5.82, "Na2O": null}}
```

缺某种氧化物就填 null,对应的 computed 字段自动留空。

### 动态字段

LLM 发现模板外字段时,自动补进 `new_fields`,带完整元信息(中文名/单位/类型/描述),并当场填值。新字段经人工确认后可入库,下次抽取复用固定名字。

### 空值约定

论文没提的字段,**显式填 `null`**(不填 0、不省略)。JSON 的 null 由 pandas 自动转 NaN。

## 多模态(图数据提取)

从论文图表里提取数据(柱状图/谱图/显微图),转成文本融入全文,让 LLM 统一抽取:

- **子图分割**:vision 定位一张图里的多个子图,逐个裁剪处理;
- **柱顶印数字** → vision OCR 读数值(精确);
- **柱顶没印数** → 代码标定 + 反算(检测刻度线 → 标定轴 → 基线检测柱子 → 反算);
- **谱图/显微图** → vision 定性描述(物相、形貌)。

多模态用的模型是 `deepseek-v4-flash-vision-exp`(配环境变量 `PAPER2DATA_VISION_MODEL`)。

## 验证管线(verify/)

零 LLM 的物理检查(取论文综述「physical checks」精华),目前实现规则校验,**已接进 cli,抽取后自动跑**:

```python
from paper2data.verify import check_records
issues = check_records(records, schema)
```

三条硬约束:
- 原料比例和超 100 → 抄错;
- 值域(性能负值、wt% 超 0~100);
- 摩尔比超 0~10。

后续补:provenance(溯源)、crosscheck(交叉验证)。

## 评估(eval.py)

对照人工测试集(benchmark)和 LLM 抽取结果,算字段级准确率,像 ML 的 accuracy 一样直观:

```bash
python -m paper2data.eval paper2data/benchmarks/固化剂_benchmark.csv output.csv
```

## 目录结构

```
paper2data/
├── cli.py / __main__.py   命令行入口
├── config.py / client.py  配置 + LLM 客户端(DeepSeek/Gemini)
├── parser.py              读 PDF(文本 + 图片)
├── table_detector.py      表格坐标定位
├── figure_extract.py      多模态图提取(子图分割/OCR/代码反算)
├── chart_extract.py       柱状图数值提取(标定/柱子检测/反算)
├── schema.py / schema.json 字段库
├── records.py             抽取输出结构(records + new_fields)
├── prompts.py             抽取 prompt
├── extract.py             调 LLM 抽取
├── compute.py             归一化 + computed 硬算
├── export.py              DataFrame → CSV
├── field_merge.py         字段别名归并
├── verify/                验证管线(规则校验/溯源/交叉验证)
├── eval.py                评估脚本(对照 benchmark 算准确率)
├── benchmarks/            人工测试集(ground truth)
└── composition.json       原料成分库
```

## Python API

```python
from paper2data.schema import Schema
from paper2data.parser import parse_pdf
from paper2data.extract import extract_dataset
from paper2data.compute import normalize_record, compute_computed_fields
from paper2data.export import to_dataframe, save_csv

schema = Schema.load("schema.json")
text, _, _ = parse_pdf("paper.pdf")
result = extract_dataset(text, schema)          # LLM 抽取
records = [normalize_record(r, schema, result.new_fields) for r in result.records]
df = to_dataframe(records, schema, result.new_fields)
save_csv(df, "out.csv")
```
