"""量抖动：跑 N 轮，**每轮都关掉 vision 缓存 → 重新抽**，统计代码侧/LLM 侧的波动。

用法：python 对比实验/variance.py                  # 开发集，默认 5 轮
      python 对比实验/variance.py --set blindset   # 盲测集
      N_RUNS=10 python 对比实验/variance.py
产出：<set>/variance_detail.csv（每轮 × 每图一行，含所有指标）
      每轮原始输出备份到 %TEMP%/p2d_variance/<set>_run{i}_*（便于回查哪一轮崩在哪）
"""
import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
PY = sys.executable
N = int(os.getenv("N_RUNS", "5"))
OUT_DIR = Path(os.getenv("TEMP", "/tmp")) / "p2d_variance"

import exp_sets


def _run(script: str, set_name: str, extra_env: dict) -> str | None:
    """跑一个脚本，返回错误信息（None = 成功）。

    **不抛异常**——一轮里某个脚本挂了（网络抖/API 报错）不该把整轮实验打挂，
    那会白跑前面 N-1 轮。失败的那轮记账跳过。
    """
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        **extra_env,
    }
    p = subprocess.run(
        [PY, str(BASE / script), "--set", set_name], env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if p.returncode != 0:
        return f"{script} 退出码 {p.returncode}：{p.stderr[-400:]}"
    return None


def _stat(rs: list[dict], key: str, dec: int = 0) -> str:
    """N 轮的「均值±跨度」。dec 是小数位——均误得给 2，否则 1.99% 会被打成「2%」。"""
    vs = [float(r[key]) for r in rs if r.get(key) not in ("", None)]
    if not vs:
        return "—"
    return f"{sum(vs) / len(vs):.{dec}f}%±{max(vs) - min(vs):.{dec}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="量抖动：N 轮重抽")
    exp_sets.add_set_arg(ap)
    args = ap.parse_args()
    set_name = args.set_name

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i in range(1, N + 1):
        print(f"=== [{set_name}] 第 {i}/{N} 轮（重抽 vision）===", flush=True)
        off = {"COMPARE_VISION_CACHE": "off"}
        errs = [e for e in (
            _run("code_read.py", set_name, off),
            _run("llm_read.py", set_name, off),
            _run("score.py", set_name, {}),
        ) if e]
        if errs:
            print(f"  ⚠ 本轮失败，跳过：{errs}", flush=True)
            (OUT_DIR / f"{set_name}_run{i}_error.txt").write_text("\n".join(errs), encoding="utf-8")
            continue
        for stem in ("code_result", "llm_result"):
            src = exp_sets.out_path(stem, set_name)
            (OUT_DIR / f"{set_name}_run{i}_{src.name}").write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8"
            )
        summary = exp_sets.out_path("score_summary", set_name, "csv")
        for r in csv.DictReader(open(summary, encoding="utf-8-sig")):
            rows.append({"轮": i, **r})

    out = exp_sets.out_path("variance_detail", set_name, "csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    figs = list(dict.fromkeys(r["图"] for r in rows))
    # 两边**同样四列**（准/均误），别只报代码的误差、LLM 只报准 —— 那样读起来不对等
    from score import _cell, _disp_width
    COLS = [("图", 14, "<"), ("代码准(均±跨)", 17, ">"), ("代码均误(均±跨)", 18, ">"),
            ("LLM准(均±跨)", 17, ">"), ("LLM均误(均±跨)", 18, ">")]
    head = " ".join(_cell(t, w, a) for t, w, a in COLS)
    rule = "=" * _disp_width(head)
    print()
    print(rule)
    print(head)
    print(rule)
    for fig in figs:
        rs = [r for r in rows if r["图"] == fig]
        vals = [fig, _stat(rs, "代码准确率±5%"), _stat(rs, "代码平均相对误差%", 2),
                _stat(rs, "LLM准确率±5%"), _stat(rs, "LLM平均相对误差%", 2)]
        print(" ".join(_cell(v, w, a) for v, (_, w, a) in zip(vals, COLS)))
    print(rule)
    print(f"[{set_name}] 共 {N} 轮（实际成功 {len({r['轮'] for r in rows})} 轮）。"
          f"明细 {out.name}；每轮原始输出 {OUT_DIR}")


if __name__ == "__main__":
    main()
