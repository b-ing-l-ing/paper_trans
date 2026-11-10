"""检查「图文本的读数」与「最终输出的字段值」是否对齐。

**精检模式（推荐）**：用 `<输出名>.sample_ids.json`（每行的样品编号）+
`.mapping.json`（编号 → 配方）+ `.figures.txt`（编号 → 图值），
逐行核对。这是唯一能定位「同配方多行」错位的模式。

**粗检模式（无 sample_ids）**：值多重集合对比，只能发现「图里有、输出没有」，
分不清实验组。

用法：
    .venv/Scripts/python.exe scripts/check_alignment.py \
        output/dev02_sid.figures.txt output/dev02_sid.csv \
        --sample-ids output/dev02_sid.sample_ids.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SERIES_FIELD = {
    "3d": ("C-3d", "Z-3d"),
    "7d": ("C-7d", "Z-7d"),
    "28d": ("C-28d", "Z-28d"),
    "initial settingtime": ("IT", "IT"),
    "initial setting time": ("IT", "IT"),
    "final settingtime": ("TT", "TT"),
    "final setting time": ("TT", "TT"),
    "fluidity": ("FL", "FL"),
}

_ROW_RE = re.compile(r"^\s*-\s*([^:：]+)[:：]\s*([-+]?\d+(?:\.\d+)?)")
_SERIES_RE = re.compile(r"^\[(.+?)\]")


_SUBFIG_RE = re.compile(r"子图(\d+)")
# 系统名必须是「全大写 + 连字符」的形式（PG-GGBS-OPC / PG-GGBS-OPC-CS）；
# 原来的 [A-Za-z0-9-]+ 会把 "(a) 28-day"、"(b) Bar graph" 也当成系统名。
_SYS_RE = re.compile(r"\(([a-z])\)\s*([A-Z]{2,}(?:-[A-Z]{2,})+)")
_SYS_NAME_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z]{2,})+\b")


def parse_figures(path: str) -> list[dict]:
    """figures.txt → [{page, y_label, series, caption, sub_idx, items: [(编号, 值)]}]"""
    blocks: list[dict] = []
    cur = None
    fig_caption = ""
    sub_idx = 0

    def flush():
        if cur and cur["items"]:
            blocks.append(dict(cur))

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            flush()
            cur = {"page": line[4:].split()[0] if line[4:] else "",
                   "y_label": "", "series": "", "caption": "", "sub_idx": 0, "items": []}
            fig_caption = ""
            sub_idx = 0
            continue
        if cur is None:
            continue
        s = line.strip()
        if not s:
            continue
        if "图标题:" in s:
            fig_caption = s.split("图标题:", 1)[-1].strip()
            continue
        ms = _SUBFIG_RE.search(s)
        if ms:
            sub_idx = int(ms.group(1))
        if s.startswith("图") and "纵轴" in s:
            cur["y_label"] = s.split("纵轴:", 1)[-1].strip()
            cur["caption"] = fig_caption
            cur["sub_idx"] = sub_idx
        m = _SERIES_RE.match(s)
        if m:
            flush()
            cur = {"page": cur["page"], "y_label": cur["y_label"],
                   "series": m.group(1).strip(), "caption": cur["caption"],
                   "sub_idx": cur["sub_idx"], "items": []}
            continue
        rm = _ROW_RE.match(s)
        if rm:
            cur["items"].append((rm.group(1).strip(), float(rm.group(2))))
    flush()
    return blocks


def _system_for(block: dict) -> str:
    """从图标题的 (a)/(b) 推断这个子图属于哪个实验体系。

    `Fig. 5 ... (a) PG-GGBS-OPC and (b) PG-GGBS-CS` + 子图2 → `PG-GGBS-CS`。
    这是解决「同编号跨体系」（两个 P10）的唯一线索。
    """
    caption = block.get("caption", "")
    pairs = _SYS_RE.findall(caption)
    idx = block.get("sub_idx", 0)
    if pairs:
        # 多面板图：子图1 → (a)，子图2 → (b)
        if 1 <= idx <= len(pairs):
            return pairs[idx - 1][1]
        return ""
    # 单系统图（Fig. 4/6 这种没有 (a)/(b) 的）：caption 里只出现一个系统名就用它
    systems = set(_SYS_NAME_RE.findall(caption))
    if len(systems) == 1:
        return systems.pop()
    return ""


def _pick_field(block: dict) -> str | None:
    pair = _SERIES_FIELD.get(block["series"].lower())
    if pair is None:
        return None
    y = block["y_label"].lower()
    return pair[1] if ("抗折" in y or "flexural" in y) else pair[0]


def _close(a: float, b: float, tol: float = 0.05) -> bool:
    return a == b or (a != 0 and abs(a - b) / abs(a) <= tol)


def _short_id(sid: str) -> str:
    """'PG-GGBS-OPC-P10' → 'P10'（图文本里只有短编号）。"""
    return sid.rsplit("-", 1)[-1] if sid else ""


def main() -> None:
    ap = argparse.ArgumentParser(description="图读数 vs 输出字段值 的一致性检查")
    ap.add_argument("figures", help="--debug 生成的 .figures.txt")
    ap.add_argument("output", help="工具输出 CSV")
    ap.add_argument("--sample-ids", default=None,
                    help="<输出名>.sample_ids.json（精检模式，每行一个样品编号）")
    args = ap.parse_args()

    blocks = parse_figures(args.figures)
    out = pd.read_csv(args.output)

    # 两级索引：
    #   by_full  = 「系统-编号」→ [(字段, 值)]，来自 caption 的 (a)/(b) 推断
    #   by_short = 短编号 → [(字段, 值)]，跨体系会混在一起（只作回退）
    by_full: dict[str, list[tuple[str, float]]] = {}
    by_short: dict[str, list[tuple[str, float]]] = {}
    for b in blocks:
        field = _pick_field(b)
        if not field:
            continue
        sysname = _system_for(b)
        for label, v in b["items"]:
            by_short.setdefault(label, []).append((field, v))
            if sysname:
                by_full.setdefault(f"{sysname}-{label}", []).append((field, v))

    if args.sample_ids and Path(args.sample_ids).exists():
        sids = json.loads(Path(args.sample_ids).read_text(encoding="utf-8"))
        print(f"精检模式：{len(sids)} 行 / {len(by_short)} 个短编号 / "
              f"{len(by_full)} 个「系统-编号」\n")
        bad = amb = missing_id = 0
        used_short: set[str] = set()
        for i, sid in enumerate(sids):
            if not sid:
                continue
            short = _short_id(sid)
            # 先用「系统-编号」精确查（caption 推断出的体系）；查不到再退回短编号
            pairs = by_full.get(sid)
            if pairs:
                used_short.add(short)
            else:
                pairs = by_short.get(short, [])
                if pairs:
                    used_short.add(short)
            if not pairs:
                missing_id += 1
                print(f"  行{i+1} {sid}: ⚠ 输出有这个编号，但图文本里找不到它的读数")
                continue
            # 同一编号在同一字段有多个不同值 → 图本身有歧义（不同实验组同名）
            per_field: dict[str, list[float]] = {}
            for f, v in pairs:
                per_field.setdefault(f, []).append(v)
            for f, vs in per_field.items():
                uniq = {round(v, 6) for v in vs}
                if len(uniq) > 1:
                    amb += 1
                    print(f"  行{i+1} {sid} 的 {f}: 图里有多个值 {sorted(uniq)} → 编号同名，无法判断")
                    continue
                fig_v = vs[0]
                ov = out.iloc[i].get(f)
                if pd.isna(ov):
                    print(f"  行{i+1} {sid} 的 {f}: 图 {fig_v} / 输出为空")
                    bad += 1
                elif not _close(fig_v, float(ov)):
                    print(f"  行{i+1} {sid} 的 {f}: 图 {fig_v} / 输出 {float(ov)}  ← 不一致")
                    bad += 1
        unref = set(by_short) - used_short
        if unref:
            print(f"  图里有读数、但输出的行没有引用这些编号：{sorted(unref)}")
        parts = []
        if bad:
            parts.append(f"{bad} 处不一致")
        if amb:
            parts.append(f"{amb} 处编号歧义")
        if missing_id:
            parts.append(f"{missing_id} 个编号在图里找不到")
        print(f"\n{'、'.join(parts) if parts else '全部一致 ✅'}")
        return

    # 粗检
    print(f"粗检模式：{len(blocks)} 块（加 --sample-ids 才精检）\n")
    by_field: dict[str, list[float]] = {}
    for vals in by_id.values():
        seen = set()
        for f, v in vals:
            if (f, round(v, 6)) in seen:
                continue
            seen.add((f, round(v, 6)))
            by_field.setdefault(f, []).append(v)
    for field, fv in sorted(by_field.items()):
        if field not in out.columns:
            continue
        ov = [float(v) for v in out[field].dropna() if float(v) != 0]
        used = [False] * len(ov)
        missing = []
        for v in fv:
            hit = next((k for k, o in enumerate(ov) if not used[k] and _close(v, o)), None)
            if hit is None:
                missing.append(v)
            else:
                used[hit] = True
        extra = [o for k, o in enumerate(ov) if not used[k]]
        status = "⚠" if (missing or extra) else "✅"
        print(f"  {status} {field}: 图 {len(fv)} 个 / 输出 {len(ov)} 个", end="")
        if missing:
            print(f"  图里有、输出没有 {sorted(missing)}", end="")
        if extra:
            print(f"  输出有、图里没有 {sorted(extra)}", end="")
        print()


if __name__ == "__main__":
    main()
