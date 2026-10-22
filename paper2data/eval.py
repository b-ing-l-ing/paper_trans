"""评估 —— 对照 benchmark（人工测试集）和 output（LLM 抽取），算字段级准确率。

通用：只评估 schema 的直接字段（computed=false），按列名对齐。
换论文只需换 benchmark CSV，本脚本不改。

用法：
    python -m paper2data.eval benchmark.csv output.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .schema import Schema


def evaluate(
    benchmark_path: str | Path,
    output_path: str | Path,
    schema_path: str | Path,
    tolerance: float = 0.01,
) -> tuple[dict, float]:
    """对照 benchmark 和 output，返回 (字段级结果, 总体准确率)。

    判定规则：
    - 两边都空 → 跳过，不计入；
    - 一边空一边有值 → 算错（漏抽 / 多抽）；
    - 数值差 ≤ tolerance → 对。
    """
    schema = Schema.load(schema_path)
    direct = [f.name for f in schema.direct_fields()]

    bench = pd.read_csv(benchmark_path)
    out = pd.read_csv(output_path)

    common = [c for c in direct if c in bench.columns and c in out.columns]

    # 组匹配：benchmark 每行找 output 中最匹配的行（不依赖行号，抗多抽/漏抽/乱序）
    pairs = _greedy_match(bench, out, common)

    results: dict[str, tuple[int, int, float | None]] = {}
    for col in common:
        correct = total = 0
        for bi, oi, _score in pairs:
            b = bench.iloc[bi][col]
            o = out.iloc[oi][col]
            if pd.isna(b) and pd.isna(o):
                continue
            total += 1
            if pd.isna(b) or pd.isna(o):
                continue
            try:
                if abs(float(b) - float(o)) <= tolerance:
                    correct += 1
            except (ValueError, TypeError):
                if str(b).strip() == str(o).strip():
                    correct += 1
        results[col] = (correct, total, correct / total if total else None)

    all_correct = sum(c for c, _, _ in results.values())
    all_total = sum(t for _, t, _ in results.values())
    overall = all_correct / all_total if all_total else 0.0
    return results, overall


def _greedy_match(bench: pd.DataFrame, out: pd.DataFrame, common: list[str]):
    """贪心匹配：benchmark 每行找 output 中最匹配的未用行。返回 [(bi, oi, score)]。"""
    pairs: list[tuple[int, int, int]] = []
    used_out: set[int] = set()
    for bi in range(len(bench)):
        best_oi, best_score = None, -1
        for oi in range(len(out)):
            if oi in used_out:
                continue
            score = _match_score(bench.iloc[bi], out.iloc[oi], common)
            if score > best_score:
                best_score, best_oi = score, oi
        if best_oi is not None:
            pairs.append((bi, best_oi, best_score))
            used_out.add(best_oi)
    return pairs


def _match_score(b_row, o_row, common: list[str]) -> int:
    """匹配分数 = 共同非空字段中值相等（容差 0.01）的个数。"""
    score = 0
    for c in common:
        b, o = b_row[c], o_row[c]
        if pd.isna(b) or pd.isna(o):
            continue
        try:
            if abs(float(b) - float(o)) <= 0.01:
                score += 1
        except (ValueError, TypeError):
            if str(b).strip() == str(o).strip():
                score += 1
    return score


def print_report(
    results: dict, overall: float, bench_rows: int, out_rows: int
) -> None:
    """打印评估报告。"""
    print(f"总体准确率：{overall:.1%}")
    print(f"行数：benchmark {bench_rows} 行 vs output {out_rows} 行")
    print("-" * 52)
    for col, (c, t, acc) in results.items():
        acc_s = f"{acc:6.0%}" if acc is not None else "  (无数据)"
        print(f"  {col:14s} {c:3d}/{t:3d}  {acc_s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="paper2data.eval", description="对照 benchmark 和 output 算准确率"
    )
    parser.add_argument("benchmark", help="人工测试集 CSV 路径")
    parser.add_argument("output", help="LLM 抽取结果 CSV 路径")
    parser.add_argument(
        "-s", "--schema", default="paper2data/schema.json", help="字段库 JSON 路径"
    )
    parser.add_argument(
        "-t", "--tolerance", type=float, default=0.01,
        help="数值容差(估读的性能字段建议放宽到 2.5)",
    )
    args = parser.parse_args()

    bench = pd.read_csv(args.benchmark)
    out = pd.read_csv(args.output)
    results, overall = evaluate(
        args.benchmark, args.output, args.schema, args.tolerance
    )
    print_report(results, overall, len(bench), len(out))


if __name__ == "__main__":
    main()
