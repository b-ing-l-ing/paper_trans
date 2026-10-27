"""溯源模块的单元测试 —— 全部用内存数据，**不调 LLM、不读 PDF**。

跑：python -m unittest paper2data.tests.test_provenance -v
"""

import unittest

from paper2data.verify import build_provenance


class _F:
    """假的 FieldSpec（只用 build_provenance 会读的那几个属性）。"""

    def __init__(self, name, label="", computed=False, aliases=None):
        self.name = name
        self.label = label or name
        self.computed = computed
        self.aliases = aliases or []


class _S:
    """假的 Schema。"""

    def __init__(self, *fields):
        self.fields = list(fields)


def _run(records, pages_text, *, tables=None, figure_texts=None, fields=None):
    schema = _S(*(fields or [_F("strength", "抗压强度")]))
    return build_provenance(
        records, schema, pages_text,
        tables=tables, figure_texts=figure_texts,
    )


class TestProvenance(unittest.TestCase):
    def test_exact_hit_in_text(self):
        """原串（含单位）在正文里，能定位到页 + 证据句。"""
        r = _run([{"strength": "4000 kPa"}], {1: "试样抗压强度为 4000 kPa，符合要求。"})
        e = r.entries[0]
        self.assertEqual(e.status, "located_text")
        self.assertEqual(e.page, 1)
        self.assertEqual(e.match_kind, "exact")
        self.assertIn("4000", e.evidence)
        self.assertTrue(e.found)

    def test_compact_match_ignores_spaces(self):
        """中文里多一个空格也要能命中（parser 会删中文字间空格）。"""
        r = _run([{"strength": "矿渣 45"}], {2: "掺入矿渣45的质量分数较高。"},
                 fields=[_F("strength", "掺量")])
        e = r.entries[0]
        self.assertEqual(e.status, "located_text")
        self.assertEqual(e.page, 2)

    def test_number_boundary(self):
        """裸数字 `45` 不能命中 `450`。"""
        r = _run([{"strength": "45 g"}], {1: "总量为 450 g。"})
        e = r.entries[0]
        self.assertEqual(e.status, "absent")
        self.assertFalse(e.found)

    def test_number_without_anchor_is_ambiguous(self):
        """数字在正文里、但附近没有字段名 → 有歧义，不能算「找到」。"""
        r = _run([{"strength": "45 g"}], {1: "本次试验取样 45 份。"})
        e = r.entries[0]
        self.assertEqual(e.status, "ambiguous")
        self.assertEqual(e.page, 1)
        self.assertFalse(e.found)   # ambiguous 不计入「可溯源」

    def test_number_with_anchor_is_located(self):
        """同样的数字，旁边有字段名 → 算找到。"""
        r = _run([{"strength": "45 g"}], {1: "抗压强度取 45 作为代表值。"})
        e = r.entries[0]
        self.assertEqual(e.status, "located_text")
        self.assertTrue(e.found)

    def test_table_hit_carries_page(self):
        """结构化表格自带页码，命中记 located_table 并写出行证据。"""
        tables = [{"page": 3, "header": ["编号", "强度"], "rows": [["S1", "4000 kPa"]]}]
        r = _run([{"strength": "4000 kPa"}], {}, tables=tables)
        e = r.entries[0]
        self.assertEqual(e.status, "located_table")
        self.assertEqual(e.source, "table")
        self.assertEqual(e.page, 3)
        self.assertIn("表1", e.evidence)

    def test_figure_only_value_is_located(self):
        """只在图文本里出现的值，必须是 located_figure —— 不能误报「查无出处」。"""
        figs = [{"page": 7, "filename": "p7_img1.png",
                 "text": "图(第7页) 子图1 纵轴: 抗压强度(MPa)\n- 7d: 45.6"}]
        r = _run([{"strength": "45.6 MPa"}], {}, figure_texts=figs)
        e = r.entries[0]
        self.assertEqual(e.status, "located_figure")
        self.assertEqual(e.page, 7)
        self.assertIn("45.6", e.evidence)

    def test_noise_header_is_skipped(self):
        """页眉/卷期里的命中要跳过，继续往后找真正的正文位置。"""
        pages = {
            1: "第12卷 第3期 抗压强度 4000 kPa",
            2: "试样抗压强度为 4000 kPa，满足要求。",
        }
        r = _run([{"strength": "4000 kPa"}], pages)
        self.assertEqual(r.entries[0].page, 2)

    def test_empty_and_computed_are_excluded(self):
        """空值与 computed 字段不进分母。"""
        schema = _S(
            _F("strength", "抗压强度"),
            _F("mgo", "MgO", computed=True),
        )
        r = build_provenance(
            [{"strength": None, "mgo": "5.2"}],
            schema, {1: "任意文本"},
        )
        statuses = [e.status for e in r.entries]
        self.assertEqual(statuses, ["empty"])      # computed 直接不生成条目
        self.assertEqual(r.summary["分母"], 0)

    def test_summary_identity(self):
        """统计恒等式：分母 = 各 status 之和（排除 empty）。"""
        records = [
            {"strength": "4000 kPa"},   # 找得到
            {"strength": "9999 kPa"},   # 找不到
            {"strength": None},         # 空
        ]
        r = _run(records, {1: "抗压强度为 4000 kPa。"})
        s = r.summary
        counted = sum(
            v for k, v in s["按状态"].items() if k not in ("empty", "computed")
        )
        self.assertEqual(s["分母"], counted)
        self.assertEqual(s["分母"], 2)
        self.assertEqual(s["查无出处"], 1)
        self.assertEqual(s["可溯源"], 1)


if __name__ == "__main__":
    unittest.main()
