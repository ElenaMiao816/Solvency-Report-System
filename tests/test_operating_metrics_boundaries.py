from __future__ import annotations

import unittest

from services.solvency_ai_table_extractor import (
    ExtractionQualityError,
    PageGrid,
    _slice_grid_for_target,
    _source_completeness_profile,
    _source_item_labels,
    _source_section_titles,
)
from services.solvency_hybrid_pipeline import (
    _merge_locator_pages,
    _radar_content_anchor,
    _validate_single_page,
)
from services.solvency_pdf_locator import _expand_to_item_boundaries
from services.solvency_pdf_locator import (
    _anchor_profile,
    _continuation_profile,
    _excluded_detail_page,
)
from services.solvency_table_boundaries import (
    TableBoundaryError,
    boundary_items,
    enforce_output_boundaries,
)
from services.solvency_table_extractor import TABLE_SIGNATURES


class OperatingMetricsBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.header = [
            "\u6307\u6807\u540d\u79f0",
            "\u672c\u5b63\u5ea6\u6570",
            "\u672c\u5e74\u5ea6\u7d2f\u8ba1\u6570",
        ]

    def test_limited_disclosure_can_end_at_comprehensive_return(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u51c0\u5229\u6da6", "10", "18"],
            ["\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", "3.2%", "3.1%"],
            ["\u524d\u4e94\u5927\u4ea7\u54c1\u7684\u4fe1\u606f", "", ""],
        ]

        bounded, note = enforce_output_boundaries("OPERATING_METRICS", rows)

        self.assertEqual(
            [row[0] for row in bounded],
            [
                "\u6307\u6807\u540d\u79f0",
                "\u4fdd\u9669\u4e1a\u52a1\u6536\u5165",
                "\u51c0\u5229\u6da6",
                "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387",
            ],
        )
        self.assertIn("\u5b9e\u9645\u7ec8\u6b62\u9879\u76ee\uff1a\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", note)
        self.assertIn("\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", boundary_items("OPERATING_METRICS"))

    def test_full_disclosure_prefers_attrition_over_fallback(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", "3.2%", "3.1%"],
            ["\u6548\u76ca\u7c7b\u6307\u6807", "", ""],
            ["\u7efc\u5408\u9000\u4fdd\u7387", "1.1%", "1.0%"],
            ["\u54c1\u8d28\u7c7b\u6307\u6807", "", ""],
            ["\u8425\u9500\u5458\u8131\u843d\u7387", "8%", "9%"],
            ["\u4e0b\u4e00\u5f20\u8868", "", ""],
        ]

        bounded, note = enforce_output_boundaries("OPERATING_METRICS", rows)

        labels = [row[0] for row in bounded]
        self.assertIn("\u7efc\u5408\u9000\u4fdd\u7387", labels)
        self.assertEqual(labels[-1], "\u8425\u9500\u5458\u8131\u843d\u7387")
        self.assertIn("\u5b9e\u9645\u7ec8\u6b62\u9879\u76ee\uff1a\u8425\u9500\u5458\u8131\u843d\u7387", note)

    def test_numbered_billion_yuan_start_and_product_placeholder_are_normalized(self):
        rows = [
            self.header,
            ["（一）保险业务收入（亿元）", "152", "152"],
            ["（十）综合投资收益率（合并口径）", "0.33%", "0.33%"],
            ["5.前五大产品的信息", "--", "--"],
            ["6.分渠道的签单保费（亿元）", "160", "160"],
            ["（十三）品质类指标", "--", "--"],
            ["5.营销员脱落率", "7.79%", "7.79%"],
        ]

        bounded, _ = enforce_output_boundaries(
            "OPERATING_METRICS",
            rows,
            require_complete=True,
        )

        labels = [row[0] for row in bounded]
        self.assertEqual(labels[1], "（一）保险业务收入（亿元）")
        self.assertNotIn("5.前五大产品的信息", labels)
        self.assertIn("6.分渠道的签单保费（亿元）", labels)
        self.assertEqual(labels[-1], "5.营销员脱落率")

    def test_missing_all_supported_end_items_still_fails(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u51c0\u5229\u6da6", "10", "18"],
        ]

        with self.assertRaises(TableBoundaryError):
            enforce_output_boundaries("OPERATING_METRICS", rows)

    def test_candidate_signature_requires_only_disclosure_start(self):
        self.assertEqual(TABLE_SIGNATURES["OPERATING_METRICS"], ("\u4fdd\u9669\u4e1a\u52a1\u6536\u5165",))

    def test_quality_continuation_after_product_details_starts_at_quality_section(self):
        grid_text = "\n".join([
            "000|前五大产品的信息：",
            "001|指标名称 产品类型 签单保费本年度累计数（元）",
            "002|第一大产品的信息 产品甲 分红寿险 158,339,293.67",
            "003|第二大产品的信息 产品乙 分红寿险 119,365,357.70",
            "004|注：前五大产品为报告期内签单保费占前五位的产品",
            "005|品质类指标：",
            "006|指标名称 本季度数 本年累计数",
            "007|13个月续保率（%） 96.11% 96.11%",
            "008|综合退保率（%） 0.39% 0.39%",
            "009|个人营销渠道的件均保费（元） 10,637.93 10,637.93",
            "010|人均保费（元） 13,888.14 13,888.14",
            "011|营销员脱落率（%） 9.56% 9.56%",
            "012|（五）近三年（综合）投资收益率",
        ])

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        grid = PageGrid(17, grid_text, 64)
        expected_rows, required_terms = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertTrue(sliced.startswith("005|品质类指标："))
        self.assertNotIn("第一大产品的信息", sliced)
        self.assertNotIn("前五大产品为报告期内", sliced)
        self.assertNotIn("近三年（综合）投资收益率", sliced)
        self.assertEqual(expected_rows, 5)
        self.assertEqual(required_terms, ("营销员脱落率",))
        self.assertTrue(any(label.startswith("营销员脱落率") for label in source_labels))
        self.assertFalse(any("第一大产品的信息" in label for label in source_labels))

        score, expected, actual = _validate_single_page(
            "OPERATING_METRICS",
            [
                ["指标名称", "本季度数", "本年累计数"],
                ["品质类指标", "--", "--"],
                ["13个月续保率（%）", "96.11%", "96.11%"],
                ["综合退保率（%）", "0.39%", "0.39%"],
                ["个人营销渠道的件均保费（元）", "10,637.93", "10,637.93"],
                ["人均保费（元）", "13,888.14", "13,888.14"],
                ["营销员脱落率（%）", "9.56%", "9.56%"],
            ],
            grid,
        )
        self.assertEqual((expected, actual), (5, 5))
        self.assertGreater(score, 70)

    def test_quality_tail_before_product_details_stops_at_attrition(self):
        grid_text = "\n".join([
            "000|（十三）品质类指标 -- --",
            "001|1.13个月续保率 97.70% 97.70%",
            "002|2.综合退保率 0.16% 0.16%",
            "003|3.个人营销渠道的件均保费 0.00 0.00",
            "004|4.人均保费 0.00 0.00",
            "005|5.营销员脱落率 0.00% 0.00%",
            "006|注：上述指标为非年化结果",
            "007|人身保险公司主要经营指标",
            "008|指标名称 产品名称 产品类型 签单保费本年度累计数",
            "009|前五大产品的信息 -- -- 235,518,361.10",
            "010|1 大家鑫享至尊养老年金保险 普通年金 69,138,830.10",
        ])

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)

        self.assertTrue(sliced.startswith("000|（十三）品质类指标"))
        self.assertIn("005|5.营销员脱落率 0.00% 0.00%", sliced)
        self.assertNotIn("前五大产品的信息", sliced)
        self.assertNotIn("产品名称", sliced)
        self.assertNotIn("上述指标为非年化结果", sliced)

    def test_top_five_product_block_before_quality_is_removed(self):
        grid_text = "\n".join([
            "000|报告期内签单保费占前五位的产品",
            "001|产品名称 产品类型 签单保费（万元） 本季度数 本年度累计数",
            "002|横琴琴童尊享增额终身寿险 普通型 36,524.58 36,524.58",
            "003|横琴传世恒富增额终身寿险 普通型 31,560.94 31,560.94",
            "004|横琴人寿福寿年年年金保险 分红型 21,302.00 21,302.00",
            "005|4．品质类指标",
            "006|指标名称 本季度数 本年度累计数",
            "007|13个月续保率（%） 92.55 92.55",
            "008|综合退保率（%） 0.43 0.43",
            "009|个人营销渠道的件均保费（元） 35,640.17 35,640.17",
            "010|人均保费（元） 44,157.78 44,157.78",
            "011|营销员脱落率（%） 12.22 12.22",
            "012|（四）近三年（综合）投资收益率",
        ])
        grid = PageGrid(15, grid_text, 60)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, required_terms = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertTrue(sliced.startswith("005|4．品质类指标"))
        self.assertNotIn("签单保费占前五位的产品", sliced)
        self.assertNotIn("横琴琴童尊享增额终身寿险", sliced)
        self.assertNotIn("近三年（综合）投资收益率", sliced)
        self.assertEqual(expected_rows, 5)
        self.assertEqual(required_terms, ("营销员脱落率",))
        self.assertFalse(any("横琴" in label for label in source_labels))

        score, expected, actual = _validate_single_page(
            "OPERATING_METRICS",
            [
                ["指标名称", "本季度数", "本年度累计数"],
                ["品质类指标", "--", "--"],
                ["13个月续保率（%）", "92.55", "92.55"],
                ["综合退保率（%）", "0.43", "0.43"],
                ["个人营销渠道的件均保费（元）", "35,640.17", "35,640.17"],
                ["人均保费（元）", "44,157.78", "44,157.78"],
                ["营销员脱落率（%）", "12.22", "12.22"],
            ],
            grid,
        )
        self.assertEqual((expected, actual), (5, 5))
        self.assertGreater(score, 70)

    def test_ungrouped_rows_after_fallback_are_not_dropped(self):
        grid_text = "\n".join([
            "000|（三）主要经营指标",
            "001|指标名称 本季度数 本年度累计数",
            "002|保险业务收入（万元） 677,828.63 677,828.63",
            "003|综合投资收益率（%） -0.05% -0.05%",
            "004|剩余边际（万元） 561,830.78 561,830.78",
            "005|新业务利润率（%） 7.45% 7.45%",
            "006|新业务价值（万元） 25,533.27 25,533.27",
            "007|签单保费（万元） 703,182.68 703,182.68",
            "008|9",
        ])
        grid = PageGrid(12, grid_text, 36)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )

        self.assertIn("004|剩余边际", sliced)
        self.assertIn("007|签单保费", sliced)
        self.assertNotIn("008|9", sliced)
        self.assertEqual(expected_rows, 6)

    def test_continuation_rows_before_section_title_are_kept(self):
        grid_text = "\n".join([
            "000|净资产（万元） 651,190.33 651,190.33",
            "001|保险合同负债（万元） 20,970,609.33 20,970,609.33",
            "002|基本每股收益（元） 0.26 0.26",
            "003|净资产收益率（%） 12.69% 12.69%",
            "004|总资产收益率（%） 0.32% 0.32%",
            "005|投资收益率（%） 0.56% 0.56%",
            "006|综合投资收益率（%） 0.66% 0.66%",
            "007|效益类指标",
            "008|剩余边际（万元） 2,620,868.73 2,620,868.73",
            "009|品质类指标",
            "010|营销员脱落率（%） 10.63% 10.63%",
            "011|注：上表根据会计准则编制",
            "012|（五）前五大产品的信息",
            "013|产品名称 产品类型 签单保费",
        ])
        grid = PageGrid(14, grid_text, 48)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )

        self.assertTrue(sliced.startswith("000|净资产"))
        self.assertIn("006|综合投资收益率", sliced)
        self.assertIn("010|营销员脱落率", sliced)
        self.assertNotIn("前五大产品的信息", sliced)
        self.assertEqual(expected_rows, 9)

    def test_benefit_tail_before_scale_section_is_kept(self):
        grid_text = "\n".join([
            "000|新业务价值（万元） 4,939.82 4,939.82",
            "001|十二、规模类指标",
            "002|签单保费（万元） 360,071.88 360,071.88",
            "003|期末个人营销员数量（人） 354 354",
            "004|十三、品质类指标",
            "005|综合退保率（%） 0.60 0.60",
            "006|营销员脱落率（%） 15.11 15.11",
            "007|前五大产品信息",
            "008|指标名称 产品名称 产品类型 签单保费",
        ])
        grid = PageGrid(14, grid_text, 36)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertTrue(sliced.startswith("000|新业务价值"))
        self.assertIn("001|十二、规模类指标", sliced)
        self.assertIn("006|营销员脱落率", sliced)
        self.assertNotIn("前五大产品信息", sliced)
        self.assertTrue(
            any(label.startswith("新业务价值") for label in source_labels)
        )

    def test_sections_after_embedded_three_year_table_are_reconnected(self):
        grid_text = "\n".join([
            "000|总资产收益率 0.07% 0.07%",
            "001|投资收益率 0.12% 0.12%",
            "002|综合投资收益率 -0.28% -0.28%",
            "003|注：以上指标根据公司财务报表数据列报",
            "004|其他指标按照偿付能力监管规则计算",
            "005|（五）近三年（综合）投资收益率",
            "006|近三年平均投资收益率 1.84%",
            "007|近三年平均综合投资收益率 2.21%",
            "008|（六）效益类指标",
            "009|指标名称 本季度数 本年度累计数",
            "010|剩余边际（万元） 84,382.38 84,382.38",
            "011|新业务价值（万元） 3,390.94 3,390.94",
            "012|（七）规模类指标",
            "013|签单保费（万元） 81,272.33 81,272.33",
            "014|（八）品质类指标",
            "015|营销员脱落率 0.00% 0.00%",
            "016|（九）前五大签单保费产品信息",
            "017|产品名称 产品类型 本年度累计数",
        ])
        grid = PageGrid(10, grid_text, 72)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])
        source_sections = _source_section_titles("OPERATING_METRICS", [grid])

        self.assertTrue(sliced.startswith("000|总资产收益率"))
        self.assertNotIn("注：", sliced)
        self.assertNotIn("近三年平均投资收益率", sliced)
        self.assertIn("008|（六）效益类指标", sliced)
        self.assertIn("012|（七）规模类指标", sliced)
        self.assertIn("014|（八）品质类指标", sliced)
        self.assertNotIn("前五大签单保费产品信息", sliced)
        self.assertEqual(expected_rows, 7)
        self.assertEqual(
            source_sections,
            ("效益类指标", "规模类指标", "品质类指标"),
        )
        self.assertTrue(any("剩余边际" in label for label in source_labels))
        self.assertTrue(any("签单保费" in label for label in source_labels))
        self.assertTrue(any("营销员脱落率" in label for label in source_labels))

    def test_not_applicable_terminal_row_closes_continuation_page(self):
        page_texts = [
            "人身保险公司主要经营指标\n"
            "指标名称 本季度数 本年度累计数\n"
            "保险业务收入 627,484,661.16 627,484,661.16\n"
            "综合投资收益率 0.18% 0.18%\n"
            "效益类指标 -- --\n规模类指标 -- --\n品质类指标 -- --\n"
            "个人营销渠道的件均保费 <不适用> <不适用>",
            "4.人均保费 <不适用> <不适用>\n"
            "5.营销员脱落率 <不适用> <不适用>\n"
            "备注：以下为指标编制说明",
        ]
        selected = [(1, 30.0, ["主要经营指标"])]
        expanded = _expand_to_item_boundaries(
            page_texts,
            {
                "table_id": "OPERATING_METRICS",
                "max_pages": 5,
            },
            selected,
        )
        grid_text = "\n".join([
            "000|4.人均保费 <不适用> <不适用>",
            "001|5.营销员脱落率 <不适用> <不适用>",
            "002|备注：",
            "003|1.保险业务收入为会计准则口径数据",
        ])
        grid = PageGrid(13, grid_text, 18)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, required_terms = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertEqual([page for page, _, _ in expanded], [1, 2])
        self.assertIn("终止项目：营销员脱落率", expanded[-1][2])
        self.assertTrue(sliced.startswith("000|4.人均保费"))
        self.assertIn("001|5.营销员脱落率", sliced)
        self.assertNotIn("备注", sliced)
        self.assertEqual(expected_rows, 2)
        self.assertIn("营销员脱落率", required_terms)
        self.assertTrue(any("人均保费" in label for label in source_labels))
        self.assertTrue(any("营销员脱落率" in label for label in source_labels))

        score, expected, actual = _validate_single_page(
            "OPERATING_METRICS",
            [
                ["指标名称", "本季度数", "本年度累计数"],
                ["4.人均保费", "<不适用>", "<不适用>"],
                ["5.营销员脱落率", "<不适用>", "<不适用>"],
            ],
            grid,
        )
        self.assertEqual((expected, actual), (2, 2))
        self.assertGreater(score, 70)

    def test_fallback_end_keeps_later_sections_and_validates_source_order(self):
        grid_text = "\n".join([
            "000|人身保险公司主要经营指标",
            "001|指标名称 本季度数 本年度累计数",
            "002|（一）保险业务收入 100 100",
            "003|（十）综合投资收益率 0.65% 0.65%",
            "004|（十一）效益类指标 -- --",
            "005|1.剩余边际 50 50",
            "006|（十二）规模类指标 -- --",
            "007|2.新单首年期交签单",
            "008|657,867,229.89 657,867,229.89",
            "009|保费",
            "010|（十三）品质类指标 -- --",
            "011|2.综合退保率 0.92% 0.92%",
        ])
        page = PageGrid(18, grid_text, 48)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            [page],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [page])
        ordered_rows = [
            ["指标名称", "本季度数", "本年度累计数"],
            ["（一）保险业务收入", "100", "100"],
            ["（十）综合投资收益率", "0.65%", "0.65%"],
            ["（十一）效益类指标", "--", "--"],
            ["1.剩余边际", "50", "50"],
            ["（十二）规模类指标", "--", "--"],
            ["2.新单首年期交签单保费", "657,867,229.89", "657,867,229.89"],
            ["（十三）品质类指标", "--", "--"],
            ["2.综合退保率", "0.92%", "0.92%"],
        ]

        self.assertIn("（十一）效益类指标", sliced)
        self.assertIn("（十二）规模类指标", sliced)
        self.assertIn("（十三）品质类指标", sliced)
        self.assertEqual(expected_rows, 5)
        self.assertTrue(
            any("新单首年期交签单保费" in label for label in source_labels)
        )
        score, expected, actual = _validate_single_page(
            "OPERATING_METRICS",
            ordered_rows,
            page,
        )
        self.assertEqual((expected, actual), (5, 5))
        self.assertGreater(score, 70)

        disordered_rows = [
            *ordered_rows[:3],
            ordered_rows[5],
            ordered_rows[6],
            ordered_rows[3],
            ordered_rows[4],
            *ordered_rows[7:],
        ]
        with self.assertRaisesRegex(ExtractionQualityError, "类别顺序"):
            _validate_single_page(
                "OPERATING_METRICS",
                disordered_rows,
                page,
            )

    def test_limited_disclosure_stops_before_numbered_filling_instructions(self):
        grid_text = "\n".join([
            "000|（三）流动性风险监测指标",
            "001|综合退保率（%） 0.98% 2.59%",
            "002|（四）主要经营指标",
            "003|指标名称 本季度数 本年度累计数",
            "004|主要经营指标 -- --",
            "005|（一）保险业务收入（元） 1,644,536,823.46 1,644,536,823.46",
            "006|（二）净利润（元） 19,892,921.68 19,892,921.68",
            "007|（三）总资产（元） 33,571,800,778.34 33,571,800,778.34",
            "008|（四）净资产（元） -508,502,524.91 -508,502,524.91",
            "009|（五）保险合同负债（元） 29,321,585,974.91 29,321,585,974.91",
            "010|（六）基本每股收益（元） -- --",
            "011|（七）净资产收益率（%） -3.73% -3.73%",
            "012|（八）总资产收益率（%） 0.06% 0.06%",
            "013|（九）投资收益率（%） 0.85% 0.85%",
            "014|（十）综合投资收益率（%） 0.66% 0.66%",
            "015|填表说明：财务报告根据2017年修订的企业会计准则第22号编制",
            "016|并根据2020年修订的企业会计准则第25号编制",
            "017|收益率按照偿付能力监管规则第18号计算",
            "018|（五）近三年（综合）投资收益率",
        ])
        grid = PageGrid(16, grid_text, 99)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, required_terms = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertTrue(sliced.startswith("002|（四）主要经营指标"))
        self.assertIn("003|指标名称 本季度数 本年度累计数", sliced)
        self.assertIn("014|（十）综合投资收益率", sliced)
        self.assertNotIn("填表说明", sliced)
        self.assertNotIn("企业会计准则", sliced)
        self.assertNotIn("近三年（综合）投资收益率", sliced)
        self.assertEqual(expected_rows, 10)
        self.assertEqual(
            required_terms,
            ("保险业务收入", "综合投资收益率"),
        )
        self.assertEqual(len(source_labels), 10)
        self.assertFalse(any("财政部" in label for label in source_labels))

    def test_numbered_page_bottom_footnote_is_not_counted_as_business_rows(self):
        grid_text = "\n".join([
            "000|（四）主要经营指标2",
            "001|行次 指标名称 本季度数 本年累计数",
            "002|1 主要经营指标 -- --",
            "003|2 （一）保险业务收入(万元) 2,624,329 2,624,329",
            "004|11 （十）综合投资收益率 0.31% 0.31%",
            "005|12 （十一）效益类指标 -- --",
            "006|13 1.剩余边际(万元) 123,438 2,603,487",
            "007|14 2.新业务利润率 5.85% 5.85%",
            "008|2 表中的净利润、总资产指标根据公司财务报告数据披露",
            "009|报告数据披露根据财政部于2017年修订的第22号准则",
            "010|并依据2020年修订的第25号准则编制",
        ])
        grid = PageGrid(12, grid_text, 48)

        sliced = _slice_grid_for_target("OPERATING_METRICS", grid_text)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            [grid],
        )
        source_labels = _source_item_labels("OPERATING_METRICS", [grid])

        self.assertIn("007|14 2.新业务利润率", sliced)
        self.assertNotIn("表中的净利润", sliced)
        self.assertNotIn("2017年", sliced)
        self.assertEqual(expected_rows, 6)
        self.assertFalse(any("报告数据披露" in label for label in source_labels))


class OperatingMetricsLocatorTests(unittest.TestCase):
    table = {"table_id": "OPERATING_METRICS", "max_pages": 5}

    def test_limited_disclosure_closes_at_fallback_page(self):
        page_texts = [
            "\u4e3b\u8981\u7ecf\u8425\u6307\u6807\n\u4fdd\u9669\u4e1a\u52a1\u6536\u5165 100 90",
            "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387 3.2% 3.1%",
            "\u524d\u4e94\u5927\u4ea7\u54c1\u7684\u4fe1\u606f",
        ]

        expanded = _expand_to_item_boundaries(page_texts, self.table, [(1, 8.0, ["anchor"])])

        self.assertEqual([page for page, _, _ in expanded], [1, 2])
        self.assertIn("\u7ec8\u6b62\u9879\u76ee\uff1a\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", expanded[-1][2])

    def test_full_disclosure_closes_at_preferred_later_page(self):
        page_texts = [
            "\u4e3b\u8981\u7ecf\u8425\u6307\u6807\n\u4fdd\u9669\u4e1a\u52a1\u6536\u5165 100 90",
            "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387 3.2% 3.1%",
            "\u6548\u76ca\u7c7b\u6307\u6807\n\u7efc\u5408\u9000\u4fdd\u7387 1.1% 1.0%",
            "\u54c1\u8d28\u7c7b\u6307\u6807\n\u8425\u9500\u5458\u8131\u843d\u7387 8% 9%",
        ]

        expanded = _expand_to_item_boundaries(page_texts, self.table, [(1, 8.0, ["anchor"])])

        self.assertEqual([page for page, _, _ in expanded], [1, 2, 3, 4])
        self.assertIn("\u7ec8\u6b62\u9879\u76ee\uff1a\u8425\u9500\u5458\u8131\u843d\u7387", expanded[-1][2])

    def test_product_placeholder_stays_but_dedicated_detail_page_is_excluded(self):
        table = {
            "table_id": "OPERATING_METRICS",
            "title_terms": ["主要经营指标"],
            "content_terms": ["保险业务收入", "签单保费", "营销员脱落率"],
            "header_terms": ["指标名称", "本季度数", "本年度累计数"],
            "continuation_terms": ["主要经营指标", "签单保费"],
            "start_item_terms": ["保险业务收入"],
            "end_item_terms": ["营销员脱落率"],
            "terminal_terms": ["营销员脱落率"],
            "exclude_item_terms": ["前五大产品的信息"],
            "exclude_page_terms": [
                "产品名称",
                "产品类型",
                "第一大产品的信息",
                "第二大产品的信息",
                "第三大产品的信息",
                "第四大产品的信息",
                "第五大产品的信息",
            ],
            "exclude_page_min_hits": 2,
            "anchor_min_structure_hits": 2,
        }
        main_page = (
            "人身保险公司主要经营指标\n指标名称 本季度数 本年度累计数\n"
            "保险业务收入 152 152\n签单保费 160 160\n"
            "前五大产品的信息 -- --\n营销员脱落率 7.79% 7.79%"
        )
        product_page = (
            "人身保险公司主要经营指标\n"
            "指标名称 产品名称 产品类型 签单保费本年度累计数\n"
            "前五大产品的信息 -- -- 70\n"
            "第一大产品的信息 产品甲 年金险 27\n"
            "第二大产品的信息 产品乙 终身寿险 17\n"
            "第三大产品的信息 产品丙 年金险 15"
        )

        self.assertFalse(_excluded_detail_page(main_page, table))
        self.assertTrue(_excluded_detail_page(product_page, table))
        self.assertGreater(_anchor_profile(main_page, table)[0], 0)
        self.assertEqual(_anchor_profile(product_page, table)[0], 0)
        self.assertEqual(
            _continuation_profile(main_page, product_page, table, [])[0],
            0,
        )


class LocatorEvidenceFusionTests(unittest.TestCase):
    def test_operating_radar_rejects_generic_narrative_terms(self):
        self.assertFalse(
            _radar_content_anchor(
                "OPERATING_METRICS",
                content_hits=["签单保费", "总资产"],
                header_hits=[],
                terminal_hits=[],
                table_like=True,
            )
        )

    def test_operating_radar_accepts_terminal_continuation_row(self):
        self.assertTrue(
            _radar_content_anchor(
                "OPERATING_METRICS",
                content_hits=["营销员脱落率", "净资产"],
                header_hits=[],
                terminal_hits=["营销员脱落率"],
                table_like=True,
            )
        )

    def test_radar_cannot_append_false_adjacent_page(self):
        pages = _merge_locator_pages(
            local_pages=[16, 17],
            ai_pages=[16, 17],
            radar_pages=[16, 17, 18],
            max_pages=5,
        )

        self.assertEqual(pages, [16, 17])

    def test_radar_can_fill_internal_page_gap(self):
        pages = _merge_locator_pages(
            local_pages=[16, 18],
            ai_pages=[],
            radar_pages=[17],
            max_pages=5,
        )

        self.assertEqual(pages, [16, 17, 18])

    def test_radar_remains_fallback_when_primary_sources_find_nothing(self):
        pages = _merge_locator_pages(
            local_pages=[],
            ai_pages=[],
            radar_pages=[16, 17],
            max_pages=5,
        )

        self.assertEqual(pages, [16, 17])


if __name__ == "__main__":
    unittest.main()
