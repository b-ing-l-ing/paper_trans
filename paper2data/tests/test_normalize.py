"""数值归一化的单元测试 —— 重点盯**空值**，全部内存数据，不调 LLM。

跑：python -m unittest paper2data.tests.test_normalize -v
"""

import unittest

from paper2data.compute import is_empty_value, normalize_record
from paper2data.schema import FieldSpec, Schema


def _schema():
    return Schema(
        domain="test",
        fields=[
            FieldSpec(name="FL", label="流动性", category="性能", unit="mm",
                      type="continuous"),
            FieldSpec(name="C-28d", label="28天抗压强度", category="性能",
                      unit="MPa", type="continuous"),
        ],
    )


class TestEmptyValues(unittest.TestCase):
    """LLM 吐的空值字符串必须统一成 None → CSV 空格子。"""

    def test_null_token_strings_become_none(self):
        for token in ["", "-", "—", "N/A", "n/a", "null", "None", "nan",
                      "未提及", "未报道", "无", "   "]:
            with self.subTest(token=token):
                self.assertIsNone(
                    normalize_record({"FL": token}, _schema())["FL"],
                    f"{token!r} 应该被当成空",
                )

    def test_real_zero_is_not_empty(self):
        """0 是合法配比 —— 绝不能当空值。"""
        self.assertFalse(is_empty_value(0))
        self.assertFalse(is_empty_value("0"))
        # "0" 能被解析成数值 0，输出必须是 0 而不是 None
        self.assertEqual(normalize_record({"FL": "0"}, _schema())["FL"], 0.0)

    def test_none_stays_none(self):
        self.assertIsNone(normalize_record({"FL": None}, _schema())["FL"])

    def test_values_still_normalize(self):
        """加了空值判定后，正常数值不能被误伤。"""
        out = normalize_record({"C-28d": "4000 kPa"}, _schema())
        self.assertAlmostEqual(out["C-28d"], 4.0)

    def test_empty_does_not_break_units(self):
        """同一 record 里既有空值又有正常值，互不影响。"""
        out = normalize_record({"FL": "未提及", "C-28d": "42.5 MPa"}, _schema())
        self.assertIsNone(out["FL"])
        self.assertAlmostEqual(out["C-28d"], 42.5)


if __name__ == "__main__":
    unittest.main()
