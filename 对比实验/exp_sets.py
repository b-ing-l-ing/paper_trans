"""对比实验的评测集定义 —— 让三个脚本（code_read / llm_read / score）统一选数据集。

- `dev`（默认）：开发集 `benchmark.json` + `images/`（4 张图，调参用的那批）
- `blindset`：盲测集 `benchmark_blindset.json`（输入是 `blindset/erased/`，无印值图）
- `heldout`：新留出集 `benchmark_heldout.json`（输入是 `heldout/erased/`，2 张新图）
  🚨 **留出集纪律**：跑完即冻结（存档带日期）；结果禁止用于调参；不满意 → 不许改代码
  回头再跑同一批，要改另找新图。**一调，它就变回开发集。**

盲测集的输入**故意用 erased（擦掉柱顶印值）**——那才逼两边都走「标定 + 反算」，
否则代码遇到印值会直接跳过反算、退化成和 LLM 一样的读数字。
"""
from pathlib import Path

BASE = Path(__file__).resolve().parent

SETS: dict[str, dict] = {
    "dev": {"bench": BASE / "benchmark.json", "suffix": ""},
    "blindset": {"bench": BASE / "benchmark_blindset.json", "suffix": "_blindset"},
    "heldout": {"bench": BASE / "benchmark_heldout.json", "suffix": "_heldout"},
    # 合成图：难度梯度可控、真值精确（生成器见 synthetic/make_charts.py）
    "synthetic": {"bench": BASE / "benchmark_synthetic.json", "suffix": "_synthetic"},
    # keke 合集：开发集 4 张 + 盲测集的 24_fig3（⚠ 4/5 是调参用过的图，非干净留出集）
    "keke": {"bench": BASE / "benchmark_keke.json", "suffix": "_keke"},
}


def bench_path(set_name: str) -> Path:
    return SETS[set_name]["bench"]


def out_path(stem: str, set_name: str, ext: str = "json") -> Path:
    """输出路径：开发集沿用原文件名，盲测集带 `_blindset` 后缀（不覆盖开发集结果）。"""
    return BASE / f"{stem}{SETS[set_name]['suffix']}.{ext}"


def add_set_arg(ap) -> None:
    """给 argparse 加统一的 --set 参数。"""
    ap.add_argument(
        "--set", default="dev", choices=list(SETS), dest="set_name",
        help="评测集：dev=开发集（默认），blindset=已降级，heldout=新留出集",
    )
