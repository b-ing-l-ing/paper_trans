"""评分：代码读图 vs LLM 读图 vs benchmark ground truth，算误差 + 产出 CSV。

对齐策略（v3，修掉「落回被当反算」+「归属率名不副实」两个坑）：
1. 从代码文本抽 (label, value, approx)——行必须以数字结尾，排掉表格行/推理行
   噪声；vision 估读落回值**不计入代码成绩**（那是 LLM 干的活），单列「落回数」，
   否则等于拿 vision 的输出给代码加分。
   **判落回**看「块里有没有 `系列N（…）:` 头」（代码反算必带、vision 落回从不带），
   **不能只看「约」字**——vision 经常不写「约」，那样落回值会被记成代码反算。
2. 配对：**纯按值**做全局最优 1-1 配对（DP）。两侧同一把尺子——不看标签、不看
   输出顺序。LLM 侧压根没有标签，按标签配会让代码单方面占便宜。
3. 指标：配对数 / 覆盖率 / 平均相对误差 / 准确率(±5%，分母=真值总数，漏检算不命中)。

⚠️ **主指标只测数值、不测归属**：代码把「图1 的值」标成「图2(a)」时，按值配对
照样能配上（值对、面板归属错）。所以另报一列「**代码归属正确率**」——但它只在
**面板检查真生效**（两边都抠得出「图N」标记）的配对上算，没生效就**留空**。

⚠️ **那列现在多半是空的，这是如实反映，不是漏算**：代码切框把面板命名成「子图N」，
`_PANEL_RE` 又特意排除中文后的「图」→ 两边都抠不出标记，面板压根没被检查过。
旧版在这种时候让它恒判「对」，于是「归属正确率」退化成纯标签匹配率、却摆出一副
查过面板的样子（实测 dev 四张图全部如此）。现在拆成两列：**代码标签正确率%**（总能算）
和 **代码归属正确率%**（没生效就空）+ **面板生效对数**（>0 才代表归属那列有意义）。

读这一列时三个坑：
- **分母只算「配上的对」**：漏读的真值不进分母，所以单看 100% 不代表整图都对
  （覆盖率是单独一列，要一起看）。
- **配对错会记成 ×**：它是在「值配对结果」上算的。两个真值挨太近时 DP 会交叉配对，
  于是标签跟着对不上——那是配对的锅，不是代码标错了（论文1_图2 就有一例）。
- **盲测集上这列几乎失效**：它的面板名是 `a-A1`/`a-4:6:1`（抠不出「图N」），代码侧标题是
  「子图N」（代码切框的编号，不是真值的 a/b/c/d），两边都对不上 → 面板检查不生效；
  而它的标签是 `3d/7d/28d`——**跨面板重复**，光比标签没有区分度。**盲测集请看主指标 +
  覆盖率 + 多检。**

另有 **代码未解析行**：有 `- ` 头、带冒号却匹配不上的行（vision 把 values 报成嵌套 list
时 `_to_text` 会打出 `- B1: ['约7.5', …]` 这种 Python repr）。旧版让它**静默消失**
（不计读出数、也不计落回），覆盖率和落回数一起被低估；现在单列计数。

运行：python 对比实验/score.py [--set dev|blindset]
      （需先跑 code_read.py / llm_read.py 生成结果；盲测集要带 --set blindset）
产出：score_detail<set>.csv（明细）+ score_summary<set>.csv（汇总）
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))  # 让 `python 对比实验/score.py` 和「被 code_read import」两种情形都能找到 exp_sets

import exp_sets


# ---------------------------------------------------------------------------
# 真值
# ---------------------------------------------------------------------------

def flatten_gt(fig: dict) -> list[tuple[str, str, float]]:
    """一张图所有 panel 展平成 [(panel, label, value)]。panel 形如「图3」，label「组1」。"""
    flat = []
    for p in fig["panels"]:
        for label, v in zip(p["labels"], p["values"]):
            flat.append((p["name"], label, v))
    return flat


# ---------------------------------------------------------------------------
# 代码侧解析
# ---------------------------------------------------------------------------

# 行形如 "- 组1: 25.55" / "- 组1: 约148"；行必须以数字结尾（排掉表格/推理行）。
# **标签允许含冒号**：配比类标签写作「4 : 6 : 0.5 (3 d)」——早先写成 `[^:]+?`（不许有
# 冒号），这类行整条被吞，覆盖率被严重低估（实测盲测集 yansuan_fig4：24 条只解析出 9 条，
# 吞掉的 15 条全是这种标签）。改成 `.+?` 后由回溯找「最后一个能对上结尾数字的冒号」。
# 注：dev 的标签（组1 / 7d / A1）从不含冒号，所以这个 bug 在 dev 上从没暴露。
_ITEM_RE = re.compile(
    r"^-\s*(?P<label>.+?):\s*(?P<approx>约)?\s*(?P<val>-?\d+(?:\.\d+)?)\s*$"
)


# 前面不许是汉字：代码切块没真标签时用占位名「子图1」，「子」后面那个「图1」会被
# 误抠成面板 token「1」，跟真值面板全判错（实测图3-4 归属被压到 40%）。
# 真标题形如「图3 …」「图2(c) …」，「图」前面是行首/空白，不受影响。
_PANEL_RE = re.compile(r"(?<![一-鿿])图\s*\d+\s*[（(]?\s*([a-zA-Z]?)\s*[)）]?")

# 行尾注解「（估）」「(约)」等：vision 落回时常写。不剥掉的话，行尾数字的约束会把
# 整条挡掉——既不算代码、也不算落回，覆盖率和落回数两个指标一起低估。
_TAIL_NOTE_RE = re.compile(r"[（(][^（()）]{0,8}[）)]\s*$")


def extract_code_items(text: str, stats: dict | None = None) -> list[dict]:
    """从 figure_to_text 文本抽 [{"panel","label","value","approx"}]。

    panel = **最近一个含「图N」的标题行**。不能只取每个空行块的首行：
    `_format_bar_text_multi` 用空行分隔系列，series 2+ 的块首行是
    "系列2（#F2C08A）:"，面板从第 2 个系列起就丢了 → 多系列图的面板错位查不出来。

    **approx（落回标记）按「块」判定，不看「约」字**：
    代码反算的输出一律由 `_format_bar_text_multi` 产出，**必带 `系列N（…）:` 头**；
    vision 落回走 `_to_text`，**从不带这个头**。而「约」字是 vision 自己写的，
    **它经常不写**——实测 blindset `24_fig3` 五个 variance 轮次里落回值一个「约」
    都没有，只看「约」字会把整轮的落回值全记成代码反算（记录里「代码准 70%」
    实际是 vision 的成绩）。所以块级判据是主力，「约」只作兜底。

    stats 传字典时回填 {"unparsed": N}：有 `- ` 头、带冒号却匹配不上的行数
    （vision 把 values 报成嵌套 list 时 `_to_text` 会打出 `- B1: ['约7.5', …]`
    这种 Python repr）。这些行**不匹配就是静默消失**，计数是为了让它露面。
    """
    items = []
    panel = ""
    for block in text.split("\n\n"):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # 块首行不是数据行时它才是标题；直接以「- 」开头说明这个块没有标题行
        body = lines if lines[0].startswith("-") else lines[1:]
        is_code = any(ln.startswith("系列") for ln in lines)  # 代码反算块的指纹
        if _PANEL_RE.search(lines[0]) or not panel:
            panel = lines[0]  # 只被含「图N」的标题更新；后续系列块沿用
        for ln in body:
            m = _ITEM_RE.match(_TAIL_NOTE_RE.sub("", ln))
            if m:
                items.append({
                    "panel": panel,        # 沿用型：最近一个含「图N」的标题（归属检查用）
                    "heading": lines[0],   # 本块自己的首行（人看的 CSV 用，如「子图2 SEM…」）
                    "label": m.group("label").strip(),
                    "value": float(m.group("val")),
                    # 不是代码反算块 → 整块都是 vision 落回；再兜底认「约」字
                    "approx": (not is_code) or bool(m.group("approx")),
                    "from_code": is_code,
                })
            elif stats is not None and ln.startswith("-") and ":" in ln:
                stats["unparsed"] = stats.get("unparsed", 0) + 1
    return items


def _panel_token(s: str) -> str:
    """从标题文本 / 真值 label 里抠出「图N(x)」标记，归一成 "2c" 这种形式。"""
    m = _PANEL_RE.search(str(s) or "")
    if not m:
        return ""
    num = re.search(r"\d+", m.group(0))
    return (num.group() if num else "") + (m.group(1) or "").lower()


# ---------------------------------------------------------------------------
# 配对
# ---------------------------------------------------------------------------

# 跳过代价。注意 DP 里「跳过真值」和「跳过预测」**各花一份**，所以拆开一对的
# 总代价是 2×_SKIP_COST。取 0.5 → 相对误差 >100% 的配对被拆成「漏检+多检」，
# 不会拿垃圾值硬充命中。所以**有效阈值是 100%，不是 _SKIP_COST 本身**。
_SKIP_COST = 0.5


def _norm_label(s: str) -> str:
    return re.sub(r"[\s（）()\[\]：:，,、]", "", str(s))


def _label_match(code_label: str, gt_label: str) -> bool:
    """代码 label 与真值 label 语义一致。空值不可比。

    先试精确相等；再试包含，但**要求剥掉公共部分后剩下的不含数字**——否则
    「组1」会跟真值「组12/组17」全判对（论文2 图4 的 label 清一色「组1x」，
    只要代码有一条被截断成「组1」，归属正确率就被刷满）。而「7」vs「7d」
    剥完剩「d」不含数字 → 仍判对（论文1_图4 全是这种，不能误杀）。
    """
    a, b = _norm_label(code_label), _norm_label(gt_label)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b:
        return not any(ch.isdigit() for ch in b.replace(a, "", 1))
    if b in a:
        return not any(ch.isdigit() for ch in a.replace(b, "", 1))
    return False


_PLACEHOLDER_RE = re.compile(r"^柱\d+$")  # 代码没认出标签时的占位名


def align(
    gt_items: list[tuple[str, str, float]], pred_items: list[dict]
) -> tuple[dict[int, int], list[int]]:
    """全局最优 1-1 配对（最小化总相对误差）。gt_items 是 (panel, label, value)。

    1 维距离下最优配对不交叉，故把两侧按值排序后做 DP：每步可选「跳过真值
    (漏检)」「跳过预测(多检)」或「配对」，跳过代价 _SKIP_COST。两侧一视同仁
    （不看标签、不看输出顺序），避免「代码有标签、LLM 没标签」的不对等，也
    避免贪心先到先得被相近值带偏。返回 ({gt_idx: pred_idx}, 多检 pred 索引)。
    """
    gs = sorted(range(len(gt_items)), key=lambda i: gt_items[i][2])
    ps = sorted(range(len(pred_items)), key=lambda i: pred_items[i]["value"])
    n, m = len(gs), len(ps)

    def cost(i: int, j: int) -> float:
        g, p = gt_items[gs[i]][2], pred_items[ps[j]]["value"]
        return abs(g - p) / abs(g) if g != 0 else _SKIP_COST

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    ch = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0], ch[i][0] = dp[i - 1][0] + _SKIP_COST, "gt"
    for j in range(1, m + 1):
        dp[0][j], ch[0][j] = dp[0][j - 1] + _SKIP_COST, "pred"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j], ch[i][j] = min(
                (dp[i - 1][j] + _SKIP_COST, "gt"),
                (dp[i][j - 1] + _SKIP_COST, "pred"),
                (dp[i - 1][j - 1] + cost(i - 1, j - 1), "pair"),
            )

    pairs: dict[int, int] = {}
    i, j = n, m
    while i > 0 and j > 0:
        c = ch[i][j]
        if c == "pair":
            pairs[gs[i - 1]] = ps[j - 1]
            i, j = i - 1, j - 1
        elif c == "gt":
            i -= 1
        else:
            j -= 1
    used = set(pairs.values())
    extra = [pi for pi in range(len(pred_items)) if pi not in used]
    return pairs, extra


def rel_err(pred: float, gt: float) -> float:
    """相对误差（%）。"""
    return abs(pred - gt) / abs(gt) * 100 if gt != 0 else float("nan")


def _disp_width(s) -> int:
    """字符串的**显示宽度**：中日韩全角字符占 2 列，其余占 1 列。

    终端里 `ljust`/`rjust` 按**字符数**补齐，而「图」「未解析」这类字显示占两列
    → 表头（大量中文）会比数据行宽出一截，整张表错位。所以自己按显示宽度补。
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _cell(v, width: int, align: str = ">") -> str:
    """按显示宽度把 v 补到 width 列。"""
    s = str(v)
    pad = max(0, width - _disp_width(s))
    return (" " * pad + s) if align == ">" else (s + " " * pad)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="评分：代码读图 vs LLM 读图 vs benchmark")
    exp_sets.add_set_arg(ap)
    args = ap.parse_args()
    set_name = args.set_name

    bench = json.loads(exp_sets.bench_path(set_name).read_text(encoding="utf-8"))
    code = json.loads(exp_sets.out_path("code_result", set_name).read_text(encoding="utf-8"))
    llm = json.loads(exp_sets.out_path("llm_result", set_name).read_text(encoding="utf-8"))

    detail_rows = []
    summary_rows = []

    for fig_name, fig in bench.items():
        if fig_name == "说明":
            continue
        # **零高度柱**（原图只印标签「0」、没有柱体，如 heldout/rm_fig3 的 a-5/3d）：
        # 没有东西可测 → **不进配对、不进分母**，单列报数。旧行为把它留在分母里，
        # 而 `align` 里 g==0 的配对代价取 `_SKIP_COST`(0.5) < 「跳过真值+跳过预测」(1.0)
        # → DP 必然给它配上一个预测（不管对不对）→ **覆盖率虚高 1、还白吃一个真预测**。
        # dev / blindset 没有零值，所以这条改动**不影响它们已有的数字**。
        gt_all = flatten_gt(fig)
        gt_items = [g for g in gt_all if g[2] != 0]
        n_zero = len(gt_all) - len(gt_items)
        stats: dict = {}
        code_all = extract_code_items(code.get(fig_name, ""), stats)
        # 只拿**反算值**算代码成绩；落回值是 vision 估读、属 LLM 的活，单列报数
        code_items = [it for it in code_all if not it["approx"]]
        code_approx = len(code_all) - len(code_items)
        llm_items = [{"label": "", "value": float(v)} for v in llm.get(fig_name, [])]

        code_pairs, code_extra = align(gt_items, code_items)
        # LLM 侧的多检**必须记**，否则单边放水：准确率分母固定（=真值总数），
        # 值喷得越多、每个真值越容易撞上一个近的，却没任何代价——多检是唯一的杠杆。
        llm_pairs, llm_extra = align(gt_items, llm_items)

        code_rels, llm_rels = [], []
        label_ok = label_tot = 0   # 标签正确率：分母 = 有真标签的配对上
        panel_ok = panel_tot = 0   # 归属正确率：分母 = **面板检查真生效**的配对上
        for gi, (gpanel, glabel, g) in enumerate(gt_items):
            ci = code_pairs.get(gi)
            li = llm_pairs.get(gi)
            cpv = code_items[ci]["value"] if ci is not None else None
            clabel = (code_items[ci]["label"] or "") if ci is not None else ""
            cpanel = (code_items[ci]["panel"] or "") if ci is not None else ""
            # 归属正确 = 逐条标签对 **且** 面板对。面板名只在标题行里，不查它，
            # 「值对、面板错」会显示成 100% 假安心（实测图1-2 就撞上了）
            paired_label = bool(clabel) and not _PLACEHOLDER_RE.match(_norm_label(clabel))
            label_same = _label_match(clabel, glabel)
            ct, gp = _panel_token(cpanel), _panel_token(gpanel)
            # 面板只在两边都抠得出「图N」标记时才较真——论文1 的面板叫 7day/A1，
            # 抠不出标记，硬比会全判错（实测第一版就是这么全 0 的）。
            # ⚠️ 但代码切框把面板命名成「子图N」，而 `_PANEL_RE` 又特意排除中文后的
            # 「图」→ 两边**都抠不出标记**，面板检查**从未真正生效**。旧版让它在没生效
            # 时恒判 True，「归属正确率」于是退化成纯标签匹配率、却摆出一副查过面板的
            # 样子（实测 dev 四张图全部如此）。现在：没生效就不进分母，整图没生效就留空。
            panel_engaged = bool(ct and gp)
            panel_same = ct == gp
            ok = paired_label and label_same and (panel_same if panel_engaged else True)
            if paired_label:
                label_tot += 1
                if label_same:
                    label_ok += 1
                if panel_engaged:
                    panel_tot += 1
                    if panel_same:
                        panel_ok += 1
            lpv = llm_items[li]["value"] if li is not None else None
            detail_rows.append({
                "图": fig_name,
                "面板": gpanel,
                "柱子": glabel,
                "真值": g,
                "代码读值": cpv,
                "代码面板": cpanel,
                "代码标签": clabel,
                "归属对?": ("√" if ok else "×") if paired_label else "",
                "代码相对误差%": round(rel_err(cpv, g), 2) if cpv is not None and g != 0 else "",
                "LLM读值": lpv,
                "LLM相对误差%": round(rel_err(lpv, g), 2) if lpv is not None and g != 0 else "",
            })
            if cpv is not None and g != 0:
                code_rels.append(rel_err(cpv, g))
            if lpv is not None and g != 0:
                llm_rels.append(rel_err(lpv, g))

        n_gt = len(gt_items)

        def cov(pairs: int):
            return round(pairs / n_gt * 100, 1) if n_gt else ""

        def acc(rels: list[float]):
            # 分母 = 真值总数：漏检（没配上的真值）算不命中，杜绝「读得少反而分高」
            # 阈值是「±5%」→ 含边界（旧版写 `e < 5`，恰好 5.00% 会被判不命中）
            return round(sum(1 for e in rels if e <= 5) / n_gt * 100, 1) if n_gt else ""

        summary_rows.append({
            "图": fig_name,
            "来源": fig["source"],
            "柱子数": n_gt,
            "零高度柱": n_zero,
            "代码读出数": len(code_all),
            "代码反算数": len(code_items),
            "代码落回数": code_approx,
            "代码配对数": len(code_pairs),
            "代码覆盖率%": cov(len(code_pairs)),
            "代码多检": len(code_extra),
            "代码未解析行": stats.get("unparsed", 0),
            "代码标签正确率%": round(label_ok / label_tot * 100, 1) if label_tot else "",
            # 留空 = 这图的面板检查没生效（两边抠不出「图N」），该列**没有含义**，
            # 不是「100% 都对」。别把空当满分读。
            "代码归属正确率%": round(panel_ok / panel_tot * 100, 1) if panel_tot else "",
            "面板生效对数": panel_tot,
            "代码平均相对误差%": round(sum(code_rels) / len(code_rels), 2) if code_rels else "",
            "代码准确率±5%": acc(code_rels),
            "LLM读出数": len(llm_items),
            "LLM配对数": len(llm_pairs),
            "LLM覆盖率%": cov(len(llm_pairs)),
            "LLM多检": len(llm_extra),
            "LLM平均相对误差%": round(sum(llm_rels) / len(llm_rels), 2) if llm_rels else "",
            "LLM准确率±5%": acc(llm_rels),
        })

    detail_fields = ["图", "面板", "柱子", "真值", "代码读值", "代码面板", "代码标签", "归属对?",
                     "代码相对误差%", "LLM读值", "LLM相对误差%"]
    with open(exp_sets.out_path("score_detail", set_name, "csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
        w.writerows(detail_rows)

    summary_fields = ["图", "来源", "柱子数", "零高度柱", "代码读出数", "代码反算数", "代码落回数",
                      "代码未解析行", "代码配对数", "代码覆盖率%", "代码多检",
                      "代码标签正确率%", "代码归属正确率%", "面板生效对数",
                      "代码平均相对误差%", "代码准确率±5%",
                      "LLM读出数", "LLM配对数", "LLM覆盖率%", "LLM多检",
                      "LLM平均相对误差%", "LLM准确率±5%"]
    with open(exp_sets.out_path("score_summary", set_name, "csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    # 列宽按**显示宽度**给（中文占 2 列，见 _cell）
    COLS = [("图", 16, "<"), ("真值", 5, ">"), ("读", 5, ">"), ("反算", 5, ">"),
            ("落回", 5, ">"), ("未解析", 7, ">"), ("覆盖%", 7, ">"), ("多检", 5, ">"),
            ("标签%", 7, ">"), ("归属%", 7, ">"), ("均误", 8, ">"), ("准%", 7, ">")]
    LCOLS = [("LLM读", 6, ">"), ("覆盖%", 7, ">"), ("多检", 5, ">"),
             ("均误", 8, ">"), ("准%", 7, ">")]
    head = (" ".join(_cell(t, w, a) for t, w, a in COLS) + "   |  "
            + " ".join(_cell(t, w, a) for t, w, a in LCOLS))
    rule = "=" * _disp_width(head)
    print(rule)
    print(head)
    print(rule)
    for s in summary_rows:
        ce = f"{s['代码平均相对误差%']}%" if s["代码配对数"] else "—"
        le = f"{s['LLM平均相对误差%']}%" if s["LLM配对数"] else "—"
        # 归属% 留空 = 面板检查没生效，打「—」而不是打空白，免得被当成 0 或者满分
        attr = str(s["代码归属正确率%"]) if s["代码归属正确率%"] != "" else "—"
        vals = [s["图"], s["柱子数"], s["代码读出数"], s["代码反算数"], s["代码落回数"],
                s["代码未解析行"], s["代码覆盖率%"], s["代码多检"],
                s["代码标签正确率%"], attr, ce, s["代码准确率±5%"]]
        lvals = [s["LLM读出数"], s["LLM覆盖率%"], s["LLM多检"], le, s["LLM准确率±5%"]]
        print(" ".join(_cell(v, w, a) for v, (_, w, a) in zip(vals, COLS))
              + "   |  "
              + " ".join(_cell(v, w, a) for v, (_, w, a) in zip(lvals, LCOLS)))
    print(rule)
    # 把两个「会静默骗人」的地方显式喊出来，别让读表的人自己发现
    no_panel = [s["图"] for s in summary_rows if not s["面板生效对数"]]
    if no_panel:
        print(f"⚠ 面板检查未生效（两边都抠不出「图N」）：{', '.join(no_panel)}"
              f" → 这几行的「归属%」已置空，**不要**按标签% 去读它")
    dropped = [f"{s['图']}({s['代码未解析行']})" for s in summary_rows if s["代码未解析行"]]
    if dropped:
        print(f"⚠ 有 `- ` 行匹配不上被丢弃（多半是嵌套 list 被打成了 Python repr）：{', '.join(dropped)}")
    zeros = [f"{s['图']}({s['零高度柱']})" for s in summary_rows if s["零高度柱"]]
    if zeros:
        print(f"⚠ 有零高度柱（原图只印标签「0」、无柱体）：{', '.join(zeros)}"
              f" → 已**排除出配对与分母**，不进覆盖率/准确率")
    print(f"[{set_name}] 已产出 {exp_sets.out_path('score_detail', set_name, 'csv').name}"
          f"（明细）+ {exp_sets.out_path('score_summary', set_name, 'csv').name}（汇总）")


if __name__ == "__main__":
    main()
