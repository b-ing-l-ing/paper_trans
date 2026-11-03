"""schema 输出列顺序的单元测试 —— 内存数据，不读真实 schema。

跑：python -m unittest paper2data.tests.test_schema_output -v
"""

import unittest

from paper2data.schema import FieldSpec, Schema


def _f(name, **kw):
    return FieldSpec(name=name, label=name, category="原料", unit="wt%",
                     type="continuous", **kw)


class TestOutputNames(unittest.TestCase):
    def test_no_output_returns_all_fields(self):
        """没定义 output → 老行为：全部字段。"""
        s = Schema(domain="t", fields=[_f("A"), _f("B")])
        self.assertEqual(s.output_names(), ["A", "B"])

    def test_output_first_then_rest_in_field_order(self):
        """output 定义的列排最前（按定义顺序），其余按字段库顺序追加。"""
        s = Schema(domain="t", output=["C", "A"],
                   fields=[_f("A"), _f("B"), _f("C")])
        self.assertEqual(s.output_names(), ["C", "A", "B"])

    def test_unknown_output_field_raises(self):
        """output 里有字段库里没有的名字 → 报错，不静默输出空列。"""
        s = Schema(domain="t", output=["A", "NOPE"], fields=[_f("A")])
        with self.assertRaises(ValueError):
            s.output_names()

    def test_output_does_not_drop_fields(self):
        """方案 B：output 之外的字段仍然输出，一个不丢。"""
        s = Schema(domain="t", output=["A"], fields=[_f("A"), _f("B"), _f("C")])
        self.assertEqual(len(s.output_names()), len(s.field_names()))


if __name__ == "__main__":
    unittest.main()
