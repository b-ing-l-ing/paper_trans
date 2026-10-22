"""回归测试：主刻度网格标定（**只调生产实现**，不再自带副本）。

被测对象在 `paper2data`：
  `chart_extract.detect_axis_ticks(..., with_len=True)`  —— 检出刻度并报线长
  `figure_extract._anchors_by_major_grid(...)`          —— 线长排候选次序 + 间距定主刻度
  `figure_extract._calibrate_axis(..., trace=...)`      —— 网格优先、老路径回退

冻结的成绩（离线验证得出，现已由生产代码复现）：
  子图2 ±5% 15/15 精确7；子图3 ±5% 15/15 精确2；子图4 ±5% 15/15 精确3
  子图1 = KNOWN-ISSUE（刻度数少、间距不规律），**只记录不判分**

为什么必须同时报「配对数」和「±5%」：DP 配对的**有效阈值是 100% 相对误差**
（`score._SKIP_COST=0.5` × 2），所以「配对数」会把差 50% 的配对照样算进去。
实测：修复前四块配对数都是 14~15/15，看着完好，±5% 却只有 0/0/1/1。
**只报配对数/覆盖率 = 把坏的说成好的。**

运行：python 对比实验/regression_axis_grid.py
产出：对比实验/regression_axis_grid.json（逐块全链路 trace，供 diff）
      控制台 PASS/FAIL；已验证的块退化 → 退出码 1
"""
import json
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

from PIL import Image

from vision_cache import install

install()

from paper2data import chart_extract as ce
from paper2data import figure_extract as fe
from score import align

IMAGE = BASE / "blindset/erased/24_fig3.png"
BENCH = BASE / "benchmark_blindset.json"
PANELS = ["a", "b", "c", "d"]

EV_KEY = {"配对": "配对数", "±5%": "±5%", "精确": "精确(≤0.5%)", "推断": "推断条数"}

BASELINE = {
    "子图2": {"配对": 15, "±5%": 15, "精确": 7, "推断": 0},
    "子图3": {"配对": 15, "±5%": 15, "精确": 2, "推断": 1},
    "子图4": {"配对": 15, "±5%": 15, "精确": 3, "推断": 1},
    "子图1": None,   # 未解决：只记录，不判分
}

truth = json.loads(BENCH.read_text(encoding="utf-8"))["24_fig3"]


def gt_for(letter):
    return [(p["name"], lb, v) for p in truth["panels"]
            if p["name"].startswith(letter + "-")
            for lb, v in zip(p["labels"], p["values"])]


def _block(bi, tmpdir):
    """取第 bi 块的（裁剪图, 临时路径, 轴框, tick_color）。"""
    img = Image.open(IMAGE)
    w, h = img.size
    fx0, fy0, fx1, fy1 = ce.detect_plot_frames(img)[bi]
    nx0, ny0, nx1, ny1 = fx0 / w, fy0 / h, fx1 / w, fy1 / h
    dx, dy = (nx1 - nx0) * 0.3, (ny1 - ny0) * 0.3
    cb = (int(max(0, nx0 - dx) * w), int(max(0, ny0 - dy) * h),
          int(min(1, nx1 + dx) * w), int(min(1, ny1 + dy) * h))
    crop = img.crop(cb)
    p = str(tmpdir / f"b{bi}.png")
    crop.save(p)
    return crop, p, fe._frame_to_plot_bounds((fx0, fy0, fx1, fy1), cb), \
        fe._detect_tick_color(crop)


def run_block(bi, tmpdir):
    crop, p, bounds, tc = _block(bi, tmpdir)
    base = float(bounds[3])

    # ↓↓↓ 全部走生产实现 ↓↓↓
    rows = ce.detect_axis_ticks(crop, tick_color=tc, tol=20, min_run_len=3, with_len=True)
    trace: dict = {}
    cal, _ = fe._calibrate_axis(crop, p, tc, base, bounds, trace=trace)
    bars = ce.detect_gray_bars(crop, bounds)
    # ↑↑↑ 全部走生产实现 ↑↑↑

    rec = {
        "块": f"子图{bi + 1}", "面板": PANELS[bi],
        "轴框": list(bounds),
        "原始刻度候选": [{"y": int(y), "线长": int(L)} for y, L in rows],
        "vision读的刻度值": fe._vision_read_axis_values(p),
        "柱数": len(bars),
        "主刻度候选": trace.get("主刻度候选"), "步长": trace.get("步长"),
        "网格": trace.get("网格"), "推断位置": trace.get("推断位置", []),
        "最终锚点": [[int(y), v] for y, v in zip(
            [int(base)] + [g[0] for g in (trace.get("网格") or [])],
            fe._vision_read_axis_values(p))] if trace.get("网格") else None,
    }
    if cal is None:
        rec["评价"] = {"状态": "标定失败（未解决）"}
        return rec

    pred = [{"label": "", "value": cal.value_at_pixel(float(b[2]))} for b in bars]
    gt = gt_for(PANELS[bi])
    pairs, extra = align(gt, pred)
    rels = [abs(pred[pi]["value"] - gt[gi][2]) / abs(gt[gi][2]) * 100 for gi, pi in pairs.items()]
    rec["评价"] = {
        "配对数": len(pairs), "真值数": len(gt),
        "±5%": sum(1 for e in rels if e <= 5),
        "精确(≤0.5%)": sum(1 for e in rels if e <= 0.5),
        "漏检": len(gt) - len(pairs), "多检": len(extra),
        "推断条数": len(rec["推断位置"]),
    }
    return rec


# ---- 负向回归：几何兜底守卫 ----
# 判据是「锚点来源」这个**可观测信号**（标定证据强度），**不是**子图1 的身份或文件名。
# 子图1 之所以被拒，是因为它的标定走的是回退路径（刻度只检出 3 条、网格造不出来），
# 而不是因为「它是子图1」。以后哪块走回退路径，哪块就该被拒。
GUARD_BASELINE = {
    "子图1": ("回退等距过滤", False),   # 标定"成功"但证据不足 → 拒收
    "子图2": ("主刻度网格", True),
    "子图3": ("主刻度网格", True),
    "子图4": ("主刻度网格", True),
}


def check_fallback_guard(tmpdir):
    """守门检查三件事：
       ① 各块的「锚点来源」（= 拒收/接管的理由来源）
       ② 开关 ON 时几何兜底到底接管没有
       ③ 子图1 的**最终输出**在 ON 下与 OFF 完全一致（没被悄悄改写）
    """
    import os
    os.environ["PAPER2DATA_GEOMETRIC_FALLBACK"] = "on"
    taken = {}
    try:
        for bi in range(4):
            crop, p, bounds, tc = _block(bi, tmpdir)
            tr: dict = {}
            fe._extract_gray_series(crop, p, tc, bounds, tr)
            taken[f"子图{bi+1}"] = (
                tr.get("锚点来源"),
                bool(fe._maybe_geometric_fallback(crop, p, tc, bounds, 3, "回归测试")),
            )
        on_text = fe.figure_to_text(str(IMAGE), "")
    finally:
        os.environ["PAPER2DATA_GEOMETRIC_FALLBACK"] = "off"
    off_text = fe.figure_to_text(str(IMAGE), "")
    _seg = lambda t: t.split("子图2")[0]   # 子图1 那一段
    return taken, _seg(on_text) == _seg(off_text)


def main() -> int:
    tmpdir = Path(tempfile.mkdtemp())
    traces = [run_block(bi, tmpdir) for bi in range(4)]

    out = BASE / "regression_axis_grid.json"
    out.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 100)
    print(f"{'块':<7}{'推断':>5}{'配对':>8}{'±5%':>7}{'精确':>6}{'漏检':>6}{'多检':>6}"
          f"   步长 / 网格来源(实=实测 推=推断)")
    print("=" * 100)
    failed, unresolved = [], []
    for t in traces:
        ev = t["评价"]
        if "状态" in ev:
            print(f"{t['块']:<7}{'—':>5}{'—':>8}{'—':>7}{'—':>6}{'—':>6}{'—':>6}   {ev['状态']}")
            unresolved.append(t["块"])
            continue
        grid = t.get("网格") or []
        src = "".join("实" if g[1] == "实测" else "推" for g in grid)
        print(f"{t['块']:<7}{ev['推断条数']:>5}{ev['配对数']:>5}/{ev['真值数']:<2}"
              f"{ev['±5%']:>7}{ev['精确(≤0.5%)']:>6}{ev['漏检']:>6}{ev['多检']:>6}   "
              f"步长={t.get('步长')} [{src}]")
        exp = BASELINE.get(t["块"])
        if exp and any(ev[EV_KEY[k]] != exp[k] for k in exp):
            failed.append((t["块"], f"退化：实际 {ev} vs 基线 {exp}"))
    # ---- 负向回归：几何兜底守卫 ----
    guard, seg_same = check_fallback_guard(tmpdir)
    print("=" * 100)
    print("几何兜底守卫（判据 = 锚点来源，不是块的编号/文件名）：")
    print(f"  {'块':<7}{'锚点来源':<16}{'兜底接管':<10}期望")
    for blk, (src, took) in guard.items():
        exp_src, exp_took = GUARD_BASELINE[blk]
        ok = (src == exp_src) and (took == exp_took)
        print(f"  {blk:<7}{str(src):<16}{'是' if took else '否':<10}"
              f"{exp_src} / {'是' if exp_took else '否'}   {'✓' if ok else '✗'}")
        if not ok:
            failed.append((blk, f"守卫判决变了：实际 ({src}, 接管={took}) vs 期望 {GUARD_BASELINE[blk]}"))
    print(f"  子图1 最终输出在 ON/OFF 下一致：{'✓' if seg_same else '✗'}")
    if not seg_same:
        failed.append(("子图1", "开关 ON 竟然改写了它的最终输出"))
    print("=" * 100)
    print("⚠ 只报「配对数」会骗人：修复前四块配对数都是 14~15/15，±5% 却只有 0/0/1/1。")
    print("⚠ 四块结论**不构成泛化证据**；「取线长前一半」在**线长分布相近**的图上会失效"
          "（实测 论文2_图1-2 子图2 值被算成一半）——生产代码已加「线长须真分成两组」的守卫。")
    if failed:
        print("\n❌ FAIL（退化）：")
        for name, why in failed:
            print(f"   {name}: {why}")
        print(f"\ntrace 已写 {out.name}")
        return 1
    tail = f"；未解决（KNOWN-ISSUE，不判分）：{', '.join(unresolved)}" if unresolved else ""
    print(f"\n✅ PASS（已验证的块全部与基线一致）{tail}。trace 已写 {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
