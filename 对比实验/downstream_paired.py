"""下游配对实验：同一份图文本、同一 prompt、同一模型参数，只换 **OFF / ON** 两种图文本。

问的问题只有一个：**几何兜底让下游 LLM 真的把"蒸发的值"找回来了，还是只是格式更整齐？**

- 图文本 OFF = `figure_to_text` 走现状（24_fig3 上=vision 落回；子图2/4 的值被渲染成
  嵌套 list，行尾不是数字）
- 图文本 ON  = 打开 `PAPER2DATA_GEOMETRIC_FALLBACK`（子图2/3/4 变成代码反算，逐柱一行）
- 其余全固定：模型 `settings.model_name`、temperature、prompt、解析方式
- 每条件抽 N 次（temperature=0 也应重复几次看方差）
- 评：对 benchmark 做一对一匹配 → 配对数 / ±5% / 精确 / 漏检（= 丢了几个值）

运行：python 对比实验/downstream_paired.py
产出：对比实验/downstream_paired.json（含每次的原始回复）
"""
import json
import os
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

from openai import OpenAI

from vision_cache import install

install()

from paper2data import figure_extract as fe
from paper2data.config import settings
from score import align

IMAGE = BASE / "blindset/erased/24_fig3.png"
BENCH = BASE / "benchmark_blindset.json"
N_RUNS = 3

PROMPT = (
    "下面是从一张柱状图里提取出来的文字描述。请读出图中**所有柱子的数值**，"
    '只输出 JSON：{"values": [数值, ...]}，按文字里出现的顺序全部列出，'
    "不要解释，不要遗漏。\n\n"
)


def figure_text_on(on: bool) -> str:
    os.environ["PAPER2DATA_GEOMETRIC_FALLBACK"] = "on" if on else "off"
    return fe.figure_to_text(str(IMAGE), "")


def truth_values() -> list[float]:
    t = json.loads(BENCH.read_text(encoding="utf-8"))["24_fig3"]
    return [v for p in t["panels"] for v in p["values"]]


def ask(text: str) -> tuple[list[float], str]:
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    resp = client.chat.completions.create(
        model=settings.model_name,
        temperature=settings.temperature,
        messages=[{"role": "user", "content": PROMPT + text}],
    )
    raw = resp.choices[0].message.content or ""
    vals: list[float] = []
    try:
        start, end = raw.find("["), raw.rfind("]")
        if start >= 0 and end > start:
            vals = [float(v) for v in json.loads(raw[start:end + 1])]
    except (ValueError, json.JSONDecodeError):
        vals = []
    if not vals:  # 兜底：正则抠数字
        vals = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    return vals, raw


def grade(vals: list[float], gt: list[float]) -> dict:
    if not vals:
        return {"读出": 0, "配对": 0, "±5%": 0, "精确": 0, "漏检": len(gt), "多检": 0}
    pred = [{"label": "", "value": v} for v in vals]
    gt_items = [("p", "", v) for v in gt]
    pairs, extra = align(gt_items, pred)
    rels = [abs(pred[pi]["value"] - gt_items[gi][2]) / abs(gt_items[gi][2]) * 100
            for gi, pi in pairs.items()]
    return {"读出": len(vals), "配对": len(pairs), "±5%": sum(1 for e in rels if e <= 5),
            "精确": sum(1 for e in rels if e <= 0.5),
            "漏检": len(gt) - len(pairs), "多检": len(extra)}


def main() -> None:
    gt = truth_values()
    print(f"真值 {len(gt)} 个值；模型={settings.model_name} temperature={settings.temperature}；"
          f"每条件 {N_RUNS} 次\n")
    out = {}
    for label, on in (("OFF", False), ("ON", True)):
        text = figure_text_on(on)
        out[label] = {"文本字数": len(text), "文本首行": text.splitlines()[0] if text else "",
                      "runs": []}
        print(f"===== {label} =====  图文本 {len(text)} 字")
        for i in range(N_RUNS):
            vals, raw = ask(text)
            g = grade(vals, gt)
            out[label]["runs"].append({"评价": g, "读出值": vals, "原始回复": raw[:800]})
            print(f"  第{i+1}次: 读出 {g['读出']:>2}  配对 {g['配对']:>2}/{len(gt)}  "
                  f"±5% {g['±5%']:>2}  精确 {g['精确']:>2}  漏检 {g['漏检']:>2}  多检 {g['多检']}")
    print()
    # 汇总
    for label in ("OFF", "ON"):
        rs = [r["评价"] for r in out[label]["runs"]]
        avg = lambda k: sum(r[k] for r in rs) / len(rs)
        print(f"  {label:<4} 均值：配对 {avg('配对'):.1f}/{len(gt)}  ±5% {avg('±5%'):.1f}  "
              f"精确 {avg('精确'):.1f}  漏检 {avg('漏检'):.1f}  多检 {avg('多检'):.1f}")
    (BASE / "downstream_paired.json").write_text(
        json.dumps({"真值数": len(gt), "结果": out}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"\ntrace 已写 downstream_paired.json")
    os.environ["PAPER2DATA_GEOMETRIC_FALLBACK"] = "off"


if __name__ == "__main__":
    main()
