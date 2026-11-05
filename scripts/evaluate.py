"""评分：工具输出 vs benchmark —— 按 2026-10-28 定的口径。

用法：
    .venv/Scripts/python.exe scripts/evaluate.py data/benchmarks/dev02.csv output/dev02.csv

口径（详见 PAPER2DATA_METHODS.md §6）：

**行匹配（两层，允许不匹配）**
  第 1 层  22 个原料字段算相似度；相似度达标 → 候选。
           候选唯一 → 直接配；候选为空 → 漏抽。
  第 2 层  **只对第一层有多个候选的行**，用 T/WB/SB/AA/WRA/MNa2SiO3 辅助排序。
           工艺缺失不进分母、不扣分；工艺冲突只降分，不否决。
           仍平手 → 标「不可区分」。
  最后在候选图上跑匈牙利（允许不匹配），求全局最优。
  ⚠️ **性能字段完全不参与匹配**——否则「用被评估的值帮助匹配」是循环论证。

**空值（四格）**
  真值空 + 工具空  → 跳过（报告规模）
  真值空 + 工具有  → 「工具多填」，单列，不判对错
  真值有 + 工具空  → 漏抽
  真值有 + 工具有  → 比数值

  ⚠️ 原料字段的 0 有两层语义，**不能混用**：
    · **行匹配**：真值 0（明确不掺）与工具缺失视为**弱等价**——
      否则真值里十几个 0 会把相似度稀释到 0.09，一对都配不上。
    · **字段评价**：**不抵消**。真值 0 + 工具空 = 漏抽；
      工具空值**绝不转成 0** 参与准确率，否则「什么都没抽」会被算成「正确识别不掺」。
  ⚠️ 这条只适用于**原料含量字段**；工艺字段（T/WB/…）的 0 不做等价。

**报告（三层分开）**
  行级：匹配率 / 多抽 / 漏抽 / 不可区分
  字段级：精确 / 容差 / 漏抽 / 多填
  覆盖率：行覆盖率 + 字段覆盖率（分母都是真值行总数）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper2data.schema import Schema  # noqa: E402

# 第二层使用的工艺字段（用户指定）
PROC_FIELDS = ["T", "WB", "SB", "AA", "WRA", "MNa2SiO3"]

# 相似度门槛与可比较字段下限
SIM_THRESHOLD = 0.6
MIN_COMMON = 3
_BIG = 1e9


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def _effective(v, is_mat: bool) -> float | None:
    """用于匹配/评分的「有效值」。

    原料字段的 **0 是「不掺」的结构性默认值**，和工具没写等价：
    工具通常只抽非零原料，若把真值里 18 个 0 都算进分母，
    相似度会被稀释到接近 0（dev02 实测：2/22 = 0.09，一对都配不上）。
    → 原料的 0 视为「无有效值」。
    其他字段（工艺/性能）的 0 是真实观测，保留。
    """
    if _is_missing(v):
        return None
    fv = float(v)
    if is_mat and fv == 0:
        return None
    return fv


def _eq(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def similarity(gt: pd.Series, pred: pd.Series, fields: list[str]) -> tuple[float, int]:
    """原料相似度。

    分母 = 两边**有效值**的并集（原料的 0 不算有效值）；
    分子 = 两边都有有效值且相同的个数。
    返回 (相似度, 可比较字段数)；可比较字段不足由 MIN_COMMON 卡掉。
    """
    g = {f: _effective(gt[f], True) for f in fields}
    p = {f: _effective(pred[f], True) for f in fields}
    union = [f for f in fields if g[f] is not None or p[f] is not None]
    if not union:
        return 0.0, 0
    common = [f for f in union if g[f] is not None and p[f] is not None]
    same = sum(1 for f in common if _eq(g[f], p[f]))
    return same / len(union), len(common)


def process_similarity(gt: pd.Series, pred: pd.Series) -> float | None:
    """工艺匹配度。缺失字段**不进分母**；无可比较字段 → None。"""
    common = [f for f in PROC_FIELDS
              if not _is_missing(gt[f]) and not _is_missing(pred[f])]
    if not common:
        return None
    return sum(1 for f in common if _eq(gt[f], pred[f])) / len(common)


def match_rows(gt: pd.DataFrame, pred: pd.DataFrame, mat_fields: list[str]) -> dict:
    """两层匹配，返回匹配结果与证据。"""
    n_gt, n_pred = len(gt), len(pred)
    sim = np.zeros((n_gt, n_pred))
    n_common = np.zeros((n_gt, n_pred), dtype=int)
    for i in range(n_gt):
        for j in range(n_pred):
            sim[i, j], n_common[i, j] = similarity(gt.iloc[i], pred.iloc[j], mat_fields)

    candidates = [
        [j for j in range(n_pred)
         if sim[i, j] >= SIM_THRESHOLD and n_common[i, j] >= MIN_COMMON]
        for i in range(n_gt)
    ]

    # 第二层触发条件：一条真值行有多个候选 **或** 多个真值行竞争同一个工具行
    # （后者也是身份歧义：两条真值行都只有同一个候选时，配给谁说不清）
    need_layer2 = [False] * n_gt
    for i in range(n_gt):
        if len(candidates[i]) > 1:
            need_layer2[i] = True
            continue
        for j in candidates[i]:
            rivals = sum(1 for k in range(n_gt) if k != i and j in candidates[k])
            if rivals:
                need_layer2[i] = True
                break

    # 工艺分**只用于打破原料分的平局**，绝不与原料分线性混合：
    # 否则「原料 1.0 + 工艺 0.8 = 0.94」会输给「原料 1.0 + 工艺缺失 = 1.0」，
    # 让缺证据的候选反而占优（审查指出的真 bug）。
    # 规则：候选间原料分相同、且**双方都有工艺证据**时，才用工艺分裁决。
    weight = sim.copy()
    proc_score = np.full((n_gt, n_pred), np.nan)
    for i in range(n_gt):
        if not need_layer2[i]:
            continue
        for j in candidates[i]:
            ps = process_similarity(gt.iloc[i], pred.iloc[j])
            proc_score[i, j] = np.nan if ps is None else ps
        # 按原料分分组，组内 ≥2 个候选且全部有工艺分 → 用工艺分微调（只破平局）
        groups: dict[float, list[int]] = {}
        for j in candidates[i]:
            groups.setdefault(round(float(sim[i, j]), 9), []).append(j)
        for _, js in groups.items():
            if len(js) < 2:
                continue
            if any(np.isnan(proc_score[i, j]) for j in js):
                continue  # 有候选缺工艺证据 → 不比较，保持原料分
            for j in js:
                weight[i, j] = sim[i, j] + 0.001 * float(proc_score[i, j])

    # 匈牙利：不可配的边给大代价
    cost = np.full((n_gt, n_pred), _BIG)
    for i in range(n_gt):
        for j in candidates[i]:
            cost[i, j] = 1.0 - weight[i, j]
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind) if cost[i, j] < _BIG]

    # 不可区分：判断两对匹配能否互换而不改变总权重。
    # 例如真值两行配方+工艺完全相同、工具也抽出两行——配给谁在数学上等价，
    # 不代表真实对应关系（性能又不参与匹配）。这类行**不能参与字段评价**。
    ambiguous = set()
    # ① 单行多候选、权重并列：配哪一个都是数学上的任意选择
    for i, j in pairs:
        rivals = [k for k in candidates[i]
                  if k != j and abs(weight[i, k] - weight[i, j]) <= 1e-9]
        if rivals:
            ambiguous.add((i, j))
    # ② 两对匹配可互换而不改变总权重：对应关系本身不确定
    for a in range(len(pairs)):
        for b in range(a + 1, len(pairs)):
            i, j = pairs[a]
            i2, j2 = pairs[b]
            if j2 not in candidates[i] or j not in candidates[i2]:
                continue
            cur = weight[i, j] + weight[i2, j2]
            cross = weight[i, j2] + weight[i2, j]
            if abs(cur - cross) <= 1e-9:
                ambiguous.add((i, j))
                ambiguous.add((i2, j2))

    evidence = []
    for i, j in pairs:
        evidence.append({
            "gt_row": int(gt.index[i]) + 1,
            "pred_row": int(j) + 1,
            "mat_sim": round(float(sim[i, j]), 4),
            "mat_common": int(n_common[i, j]),
            "proc_score": None if np.isnan(proc_score[i, j]) else round(float(proc_score[i, j]), 4),
            "n_candidates": len(candidates[i]),
            "ambiguous": (i, j) in ambiguous,
        })
    return {
        "pairs": pairs,
        "candidates": candidates,
        "evidence": evidence,
        "ambiguous": ambiguous,
        "matched_gt": {i for i, _ in pairs},
        "matched_pred": {j for _, j in pairs},
    }


def compare_value(gv, pv, field: str, kind: str) -> dict:
    """单个单元格比较。返回 {exact, tol, rel_err}。"""
    if kind == "perf":
        # 性能：相对 5%（dev02 实测误差中位 1.5~3.6%、P90 5~12%；
        # ±1% 只有 29~35%，±5% 有 76~88%）。真值 0 时要求工具也是 0。
        if float(gv) == 0:
            ok = float(pv) == 0
            return {"exact": ok, "tol": ok, "rel_err": None}
        rel = abs(float(pv) - float(gv)) / abs(float(gv))
        return {"exact": rel <= 1e-9, "tol": rel <= 0.05, "rel_err": rel}
    # 配比/工艺：绝对 0.1（wt% 的 0.1 个百分点；无量纲字段同样用 0.1）
    diff = abs(float(pv) - float(gv))
    return {"exact": diff <= 1e-9, "tol": diff <= 0.1, "rel_err": None}


def evaluate(bench_path: str, output_path: str, schema_path: str) -> dict:
    schema = Schema.load(schema_path)
    mat_fields = [f.name for f in schema.fields if f.category == "原料"]
    perf_fields = [f.name for f in schema.fields if f.category == "性能"]
    all_fields = schema.field_names()

    gt = pd.read_csv(bench_path)
    pred = pd.read_csv(output_path)
    # 列对齐：只**补**缺的列，不删多余的列——
    # 第二层匹配用的工艺字段可能不在 schema 里（如自定义 schema），
    # reindex 会把它们静默丢掉，随后 process_similarity 直接 KeyError。
    for f in list(all_fields) + PROC_FIELDS:
        if f not in gt.columns:
            gt[f] = np.nan
        if f not in pred.columns:
            pred[f] = np.nan

    m = match_rows(gt, pred, mat_fields)
    # 不可区分的配对**不参与字段评价**（否则随机分配会扭曲准确率），
    # 但仍计入行级「匹配」数。
    evaluable_pairs = [p for p in m["pairs"] if p not in m["ambiguous"]]

    # 行级
    n_gt, n_pred = len(gt), len(pred)
    row_stats = {
        "真值行": n_gt,
        "工具行": n_pred,
        "匹配": len(m["pairs"]),
        "参与字段评价": len(evaluable_pairs),
        "行覆盖率": len(m["pairs"]) / n_gt if n_gt else 0.0,
        "漏抽行": n_gt - len(m["pairs"]),
        "多抽行": n_pred - len(m["pairs"]),
        "不可区分对": len(m["ambiguous"]),
    }

    # 字段级
    field_stats = {}
    for f in all_fields:
        kind = "perf" if f in perf_fields else ("mat" if f in mat_fields else "proc")
        is_mat = f in mat_fields
        st = {"匹配": 0, "精确": 0, "容差": 0,
              "漏抽非零": 0, "漏抽零": 0, "多填": 0, "两边空": 0}
        rels = []
        for i, j in evaluable_pairs:
            # ⚠️ 评价阶段**不用** _effective：真值的 0 是「明确不掺」，
            # 工具的缺失是「不知道」，两者不能互相抵消——
            # 否则工具什么都没抽，也会被算成「正确识别了不掺」。
            # 真值 0 + 工具空 → 记「漏抽」（见 CORRECTIONS/口径说明）。
            gv, pv = gt.iloc[i][f], pred.iloc[j][f]
            gm, pm = _is_missing(gv), _is_missing(pv)
            if gm and pm:
                st["两边空"] += 1
            elif gm:
                st["多填"] += 1
            elif pm:
                # 真值 0（明确不掺）+ 工具空 → 形式性缺失，信息没丢；
                # 其余情况 → 实质性漏抽。判定都算「漏抽」，只是分开展示。
                if is_mat and float(gv) == 0:
                    st["漏抽零"] += 1
                else:
                    st["漏抽非零"] += 1
            else:
                st["匹配"] += 1
                r = compare_value(gv, pv, f, kind)
                st["精确"] += int(r["exact"])
                st["容差"] += int(r["tol"])
                if r["rel_err"] is not None:
                    rels.append(r["rel_err"])
        st["精确率"] = st["精确"] / st["匹配"] if st["匹配"] else None
        st["容差率"] = st["容差"] / st["匹配"] if st["匹配"] else None
        # 覆盖率：真值行中该字段**拿到了工具输出**的比例（含真值=0 的行）
        st["字段覆盖率"] = st["匹配"] / n_gt if n_gt else 0.0
        # 召回率：**真值非空**的单元格里，被工具抽到的比例
        denom = st["匹配"] + st["漏抽非零"] + st["漏抽零"]
        st["字段召回率"] = st["匹配"] / denom if denom else None
        if rels:
            st["相对误差中位数"] = float(np.median(rels))
            st["相对误差P90"] = float(np.percentile(rels, 90))
        field_stats[f] = st

    return {"row": row_stats, "field": field_stats, "evidence": m["evidence"]}


def print_report(r: dict) -> None:
    row = r["row"]
    print("=" * 68)
    print("行级")
    print(f"  真值 {row['真值行']} 行 / 工具 {row['工具行']} 行 → 匹配 {row['匹配']} 对"
          f"（行覆盖率 {row['行覆盖率']:.1%}）")
    print(f"  漏抽行 {row['漏抽行']} / 多抽行 {row['多抽行']}"
          f" / 不可区分对 {row['不可区分对']}（不参与字段评价）")
    print(f"  参与字段评价的行对：{row['参与字段评价']}")

    print("\n字段级（只统计可评价的匹配对；原料的 0 与工具空都记漏抽，但分开显示）")
    print(f"  {'字段':<12}{'匹配':>5}{'精确率':>8}{'容差率':>8}"
          f"{'漏抽≠0':>8}{'漏抽=0':>8}{'多填':>6}{'覆盖率':>8}{'召回率':>8}")
    for f, st in r["field"].items():
        if st["匹配"] + st["漏抽非零"] + st["漏抽零"] + st["多填"] == 0:
            continue
        exact = f"{st['精确率']:.0%}" if st["精确率"] is not None else "  —"
        tol = f"{st['容差率']:.0%}" if st["容差率"] is not None else "  —"
        rec = f"{st['字段召回率']:.0%}" if st["字段召回率"] is not None else "  —"
        print(f"  {f:<14}{st['匹配']:>5}{exact:>9}{tol:>9}"
              f"{st['漏抽非零']:>8}{st['漏抽零']:>8}{st['多填']:>6}"
              f"{st['字段覆盖率']:>8.0%}{rec:>9}")

    print("\n匹配证据（前 20 对）")
    for e in r["evidence"][:20]:
        flag = " ⚠不可区分" if e["ambiguous"] else ""
        print(f"  真值{e['gt_row']:>3} → 工具{e['pred_row']:>3}  "
              f"原料相似度 {e['mat_sim']:.2f}（{e['mat_common']} 字段）"
              f" 工艺 {e['proc_score'] if e['proc_score'] is not None else '—'}"
              f" 候选 {e['n_candidates']}{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description="工具输出 vs benchmark 评分")
    ap.add_argument("benchmark", help="benchmark CSV（如 data/benchmarks/dev02.csv）")
    ap.add_argument("output", help="工具输出 CSV（如 output/dev02.csv）")
    ap.add_argument("-s", "--schema", default="paper2data/schema.json")
    ap.add_argument("-j", "--json", default=None, help="把完整结果写成 JSON")
    args = ap.parse_args()

    r = evaluate(args.benchmark, args.output, args.schema)
    print_report(r)
    if args.json:
        Path(args.json).write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已写 {args.json}")


if __name__ == "__main__":
    main()
