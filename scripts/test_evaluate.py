"""evaluate.py 的逻辑单测 —— 合成数据，不读真实文件、不调 LLM。

跑：.venv/Scripts/python.exe -X utf8 -m unittest scripts.test_evaluate -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

MAT = ["CG", "GGBS", "FA", "OSA"]


def _df(rows):
    full = [{c: r.get(c, np.nan) for c in MAT + ev.PROC_FIELDS + ["C-28d"]}
            for r in rows]
    return pd.DataFrame(full)


class TestProcessTiebreak(unittest.TestCase):
    """工艺分只能破平局，不能因为缺失而占优（审查指出的真 bug）。"""

    def test_missing_process_does_not_win(self):
        gt = _df([{"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "WB": 0.4, "AA": 18}])
        pred = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "WB": 0.4, "AA": 18},   # 工艺全对
            {"CG": 40, "GGBS": 40, "FA": 20},                        # 工艺全缺
        ])
        m = ev.match_rows(gt, pred, MAT)
        # 组内有候选缺工艺证据 → 不做比较 → 权重并列 → 标不可区分
        self.assertEqual(len(m["pairs"]), 1)
        self.assertIn(m["pairs"][0], m["ambiguous"])

    def test_process_breaks_tie_when_both_have_evidence(self):
        gt = _df([{"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "WB": 0.4, "AA": 20}])
        pred = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "WB": 0.4, "AA": 20},   # 工艺全对
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "WB": 0.4, "AA": 99},   # 工艺冲突
        ])
        m = ev.match_rows(gt, pred, MAT)
        self.assertEqual(m["pairs"], [(0, 0)], "工艺匹配的候选应该胜出")
        self.assertNotIn((0, 0), m["ambiguous"])


class TestAmbiguous(unittest.TestCase):
    """同配方两行 → 无法确定对应关系。"""

    def test_swappable_pairs_are_ambiguous(self):
        gt = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18},
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18},
        ])
        pred = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18},
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18},
        ])
        m = ev.match_rows(gt, pred, MAT)
        self.assertEqual(len(m["pairs"]), 2)
        self.assertEqual(len(m["ambiguous"]), 2, "两对都可互换 → 都应标不可区分")


class TestFieldEvaluation(unittest.TestCase):
    """不可区分的对不参与字段评价。"""

    def _write(self, obj, suffix):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w",
                                        encoding="utf-8-sig")
        obj.to_csv(f.name, index=False)
        return f.name

    def test_ambiguous_excluded_from_field_stats(self):
        gt = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18, "C-28d": 20},
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18, "C-28d": 30},
        ])
        pred = _df([
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18, "C-28d": 30},
            {"CG": 40, "GGBS": 40, "FA": 20, "T": 25, "AA": 18, "C-28d": 20},
        ])
        schema = {
            "domain": "t",
            "fields": [
                {"name": "CG", "label": "煤矸石", "category": "原料", "unit": "wt%",
                 "type": "continuous"},
                {"name": "GGBS", "label": "矿渣", "category": "原料", "unit": "wt%",
                 "type": "continuous"},
                {"name": "FA", "label": "粉煤灰", "category": "原料", "unit": "wt%",
                 "type": "continuous"},
                {"name": "T", "label": "温度", "category": "工艺", "unit": "℃",
                 "type": "continuous"},
                {"name": "AA", "label": "碱激发剂", "category": "工艺", "unit": "wt%",
                 "type": "continuous"},
                {"name": "C-28d", "label": "28天强度", "category": "性能", "unit": "MPa",
                 "type": "continuous"},
            ],
        }
        import json
        sp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w",
                                         encoding="utf-8")
        json.dump(schema, sp, ensure_ascii=False)
        sp.close()

        r = ev.evaluate(self._write(gt, ".csv"), self._write(pred, ".csv"), sp.name)
        self.assertEqual(r["row"]["不可区分对"], 2)
        self.assertEqual(r["row"]["参与字段评价"], 0)
        # 字段统计为空（没有可评价的对）
        self.assertEqual(r["field"]["C-28d"]["匹配"], 0)


class TestRates(unittest.TestCase):
    def test_recall_excludes_zero_fill(self):
        """真值 0 + 工具空 → 进召回率分母，但覆盖率只按真值行数算。"""
        gt = _df([{"CG": 40, "GGBS": 40, "FA": 20}])
        pred = _df([{"CG": 40, "FA": 20}])          # GGBS 漏抽
        # 直接验证公式含义
        self.assertEqual(ev._effective(0, True), None, "原料 0 在匹配时视为无有效值")
        self.assertEqual(ev._effective(0, False), 0.0, "工艺的 0 是真实值")
        self.assertEqual(ev._effective(np.nan, True), None)


if __name__ == "__main__":
    unittest.main()


class TestMinCommonAdaptive(unittest.TestCase):
    """只有 2 种原料的论文不能被 MIN_COMMON=3 误杀（dev03 实测）。"""

    def test_two_material_row_can_match(self):
        gt = _df([{"CG": 30, "GGBS": 70}])          # 只有 2 个有效原料
        pred = _df([{"CG": 30, "GGBS": 70}])
        m = ev.match_rows(gt, pred, MAT)
        self.assertEqual(len(m["pairs"]), 1, "2 个原料全相同，应该能配上")
        self.assertEqual(m["pairs"][0], (0, 0))

    def test_one_material_row_still_guarded(self):
        """只有 1 个有效原料时信息太少，仍然不该直接配。"""
        gt = _df([{"CG": 50}])
        pred = _df([{"CG": 50}])
        m = ev.match_rows(gt, pred, MAT)
        # 1 个字段相同不足以确认同一行（可能是巧合）
        self.assertEqual(len(m["pairs"]), 0, "只有 1 个原料不该判为匹配")
