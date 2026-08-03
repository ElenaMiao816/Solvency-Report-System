from __future__ import annotations

import unittest

from services.solvency_ai_table_extractor import (
    PageGrid,
    _slice_grid_for_target,
    _source_completeness_profile,
    recover_minimum_capital_terminal_rows,
)
from services.solvency_pdf_locator import _expand_to_item_boundaries
from services.solvency_table_boundaries import (
    boundary_items,
    enforce_output_boundaries,
)


class MinimumCapitalBoundaryTests(unittest.TestCase):
    header = ["行次", "指标名称（单位：万元）", "本季度末数", "上季度末数"]

    def test_summary_first_layout_keeps_risk_detail_tables(self):
        rows = [
            self.header,
            ["1", "最低资本（1=2+3+4）", "369,992.25", "342,378.32"],
            ["2", "量化风险最低资本（考虑特征系数后）", "367,817.52", "340,365.90"],
            ["", "（二）保险风险最低资本汇总", "", ""],
            ["8", "非寿险业务保险风险间的相关性效应", "-", "-"],
            ["", "（三）市场风险最低资本汇总", "", ""],
            ["8", "市场风险间的相关性效应", "173,485.63", "146,445.47"],
            ["", "（四）信用风险最低资本汇总", "", ""],
            ["4", "信用风险间的相关性效应", "20,766.36", "21,666.07"],
            ["", "S06-风险综合评级", "", ""],
        ]

        bounded, note = enforce_output_boundaries("MINIMUM_CAPITAL", rows)

        labels = [row[1] for row in bounded[1:]]
        self.assertEqual(labels[0], "最低资本（1=2+3+4）")
        self.assertIn("（二）保险风险最低资本汇总", labels)
        self.assertIn("（三）市场风险最低资本汇总", labels)
        self.assertIn("（四）信用风险最低资本汇总", labels)
        self.assertEqual(labels[-1], "信用风险间的相关性效应")
        self.assertNotIn("S06-风险综合评级", labels)
        self.assertIn("最低资本在前并展开风险汇总", note)

    def test_original_quantity_first_layout_remains_supported(self):
        rows = [
            self.header,
            ["1", "量化风险最低资本", "367,817.52", "340,365.90"],
            ["2", "市场风险最低资本", "232,004.10", "190,725.91"],
            ["3", "控制风险最低资本", "2,174.73", "2,012.43"],
            ["4", "最低资本", "369,992.25", "342,378.32"],
        ]

        bounded, note = enforce_output_boundaries("MINIMUM_CAPITAL", rows)

        self.assertEqual([row[1] for row in bounded[1:]], [
            "量化风险最低资本",
            "市场风险最低资本",
            "控制风险最低资本",
            "最低资本",
        ])
        self.assertIn("量化风险在前的标准汇总", note)

    def test_capitalizable_risk_first_layout_keeps_opening_row(self):
        rows = [
            self.header,
            ["1", "可资本化风险最低资本", "366,965.42", "347,555.26"],
            ["1*", "量化风险最低资本（未考虑特征系数前）", "407,739.36", "386,172.51"],
            ["2", "控制风险最低资本", "18,579.23", "17,596.51"],
            ["3", "附加资本", "0.00", "0.00"],
            ["4", "最低资本", "385,544.65", "365,151.77"],
        ]

        bounded, note = enforce_output_boundaries("MINIMUM_CAPITAL", rows)

        self.assertEqual(bounded[1], rows[1])
        self.assertEqual(bounded[-1], rows[-1])
        self.assertIn("可资本化风险在前的完整汇总", note)

    def test_all_summary_first_terminal_aliases_are_semantic_items(self):
        items = boundary_items("MINIMUM_CAPITAL")

        self.assertIn("信用风险间的相关性效应", items)
        self.assertIn("市场风险间的相关性效应", items)
        self.assertIn("非寿险业务保险风险间的相关性效应", items)

    def test_row_four_total_alias_closes_standard_layout(self):
        rows = [
            self.header,
            ["1", "量化风险最低资本", "213.44", "196.15"],
            ["2", "控制风险最低资本", "-1.84", "-1.69"],
            ["3", "附加资本", "-", "-"],
            ["4", "合计（万元）", "211.61", "194.47"],
        ]

        bounded, _ = enforce_output_boundaries("MINIMUM_CAPITAL", rows)

        self.assertEqual(bounded[-1], rows[-1])


class MinimumCapitalLocatorTests(unittest.TestCase):
    table = {"table_id": "MINIMUM_CAPITAL", "max_pages": 4}

    def test_summary_first_layout_closes_on_credit_risk_page(self):
        page_texts = [
            "偿付能力充足率\n4 量化风险最低资本 367 340\n7 最低资本 369 342",
            (
                "十、最低资本\n"
                "1 最低资本 369,992.25 342,378.32\n"
                "2 量化风险最低资本 367,817.52 340,365.90\n"
                "8 非寿险业务保险风险间的相关性效应 - -"
            ),
            (
                "1 市场风险最低资本 232,004.10 190,725.91\n"
                "8 市场风险间的相关性效应 173,485.63 146,445.47\n"
                "4 信用风险间的相关性效应 20,766.36 21,666.07"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(2, 12.0, ["anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [2, 3])
        self.assertIn("起始项目：最低资本", expanded[0][2])
        self.assertIn("终止项目：信用风险间的相关性效应", expanded[-1][2])

    def test_original_layout_requires_end_after_start_on_same_page(self):
        page_texts = [
            (
                "最低资本表\n"
                "1 量化风险最低资本 367,817.52 340,365.90\n"
                "2 市场风险最低资本 232,004.10 190,725.91\n"
                "4 最低资本 369,992.25 342,378.32"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 12.0, ["anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [1])
        self.assertIn("边界版式：量化风险在前的标准汇总", expanded[0][2])

    def test_title_anchor_beats_shorter_solvency_summary_false_closure(self):
        page_texts = [
            (
                "三、主要指标表\n"
                "量化风险最低资本 1,051,101.01 1,006,616.55\n"
                "控制风险最低资本 7,243.80 6,937.23\n"
                "最低资本 1,058,344.82 1,013,553.79"
            ),
            "其他披露",
            (
                "十、最低资本\nS05-最低资本表\n"
                "1 量化风险最低资本 1,051,101.01 1,006,616.55\n"
                "2 控制风险最低资本 7,243.80 6,937.23"
            ),
            (
                "3 附加资本 0.00 0.00\n"
                "4 最低资本 1,058,344.82 1,013,553.79"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(3, 35.0, ["S05-最低资本表"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [3, 4])
        self.assertIn("起始项目：量化风险最低资本", expanded[0][2])
        self.assertIn("终止项目：最低资本", expanded[-1][2])

    def test_formal_standard_layout_beats_capitalizable_summary_variant(self):
        page_texts = [
            (
                "四、主要指标表\n"
                "可资本化风险最低资本 3,766,872,368.16 3,670,965,609.49\n"
                "控制风险最低资本 12,498,186.09 12,179,975.02\n"
                "最低资本 3,779,370,554.25 3,683,145,584.51"
            ),
            "其他披露",
            (
                "十一、最低资本\nS05-最低资本表\n"
                "行次 项目 期末数 期初数\n"
                "1 量化风险最低资本 3,766,872,368.16 3,670,965,609.49\n"
                "1* 量化风险最低资本（未考虑特征系数前） "
                "3,965,128,808.58 3,864,174,325.78"
            ),
            (
                "3 附加资本 - -\n"
                "4 最低资本 3,779,370,554.25 3,683,145,584.51"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(3, 35.0, ["S05-最低资本表", "行次", "期末数", "期初数"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [3, 4])
        self.assertIn("正式章节标题", expanded[0][2])
        self.assertIn("边界版式：量化风险在前的标准汇总", expanded[0][2])

    def test_formal_section_beats_denser_solvency_status_summary(self):
        page_texts = [
            (
                "（六）偿付能力状况表\n"
                "4 最低资本 385,544.65 365,151.77\n"
                "4.1 量化风险最低资本 366,965.42 347,555.26\n"
                "4.1.1 保险风险最低资本 101,000.00 99,000.00\n"
                "4.1.2 市场风险最低资本 220,000.00 210,000.00\n"
                "4.1.3 信用风险最低资本 80,000.00 75,000.00\n"
                "4.2 控制风险最低资本 18,579.23 17,596.51\n"
                "4.3 附加资本 0.00 0.00"
            ),
            "其他披露",
            (
                "十一、最低资本\n"
                "行次 项目 期末数 期初数\n"
                "1 量化风险最低资本 366,965.42 347,555.26\n"
                "1.1 保险风险最低资本 101,000.00 99,000.00\n"
                "1.2 市场风险最低资本 220,000.00 210,000.00\n"
                "1.3 信用风险最低资本 80,000.00 75,000.00"
            ),
            (
                "2 控制风险最低资本 18,579.23 17,596.51\n"
                "3 附加资本 0.00 0.00\n"
                "4 最低资本 385,544.65 365,151.77"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 24.3, ["偿付能力状况表字段密集"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [3, 4])
        self.assertIn("正式章节标题", expanded[0][2])
        self.assertIn("起始项目：量化风险最低资本", expanded[0][2])
        self.assertIn("终止项目：最低资本", expanded[-1][2])

    def test_capitalizable_risk_layout_beats_solvency_summary(self):
        page_texts = [
            (
                "四、主要指标\n"
                "最低资本（万元） 385,544.65 365,151.77\n"
                "附加资本（万元） 0.00 0.00"
            ),
            "其他披露",
            (
                "十一、最低资本\n行次 项目 期末数 期初数\n"
                "1 可资本化风险最低资本 366,965.42 347,555.26\n"
                "1* 量化风险最低资本（未考虑特征系数前） 407,739.36 386,172.51\n"
                "3 附加资本 0.00 0.00\n"
                "4 最低资本 385,544.65 365,151.77"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(3, 21.8, ["行次", "期末数", "期初数"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [3])
        self.assertIn("起始项目：可资本化风险最低资本", expanded[0][2])
        self.assertIn("终止项目：最低资本", expanded[0][2])

    def test_standard_layout_can_end_with_row_four_total_alias(self):
        page_texts = [
            (
                "十一、最低资本\n最低资本表\n行次 项目 本季度 上季度\n"
                "1 量化风险最低资本 2,134,449 1,961,546\n"
                "3 附加资本 - -\n3.2 D-SII附加资本 - -"
            ),
            (
                "行次 项目 本季度 上季度\n"
                "3.3 G-SII附加资本 - -\n3.4 其他附加资本 - -\n"
                "4 合计（万元） 2,116,093 1,944,677"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 26.8, ["最低资本表", "量化风险最低资本"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [1, 2])
        self.assertIn("终止项目：最低资本", expanded[-1][2])


class MinimumCapitalGridSliceTests(unittest.TestCase):
    def test_numbered_section_title_is_not_mistaken_for_terminal_data_row(self):
        grid = "\n".join([
            "000|行次  项目              认可价值期末数  认可价值期初数",
            "001|6     资本性负债        0.00            0.00",
            "002|7     其他认可负债      0.00            0.00",
            "003|8     认可负债合计      3,057,597.17    2,839,312.40",
            "004|十一、最低资本",
            "005|行次  项目              期末数          期初数",
            "006|1     量化风险最低资本  126,663.16      118,554.55",
            "007|1.1   保险风险最低资本  39,636.68       39,084.89",
        ])

        sliced = _slice_grid_for_target("MINIMUM_CAPITAL", grid)

        self.assertNotIn("认可负债合计", sliced)
        self.assertIn("005|行次", sliced)
        self.assertIn("量化风险最低资本", sliced)

    def test_capitalizable_risk_layout_slice_keeps_full_page(self):
        grid = "\n".join([
            "000|十一、最低资本",
            "001|行次  项目                     期末数      期初数",
            "002|1     可资本化风险最低资本     366,965.42  347,555.26",
            "003|1*    量化风险最低资本（未考虑特征系数前） 407,739.36 386,172.51",
            "004|2     控制风险最低资本         18,579.23   17,596.51",
            "005|3     附加资本                 0.00        0.00",
            "006|4     最低资本                 385,544.65  365,151.77",
            "007|29",
        ])

        sliced = _slice_grid_for_target("MINIMUM_CAPITAL", grid)

        self.assertIn("可资本化风险最低资本", sliced)
        self.assertIn("量化风险最低资本（未考虑特征系数前）", sliced)
        self.assertIn("4     最低资本", sliced)

    def test_first_page_does_not_stop_at_main_table_or_insurance_fallback(self):
        grid = "\n".join([
            "000|十、最低资本",
            "001|（一）最低资本",
            "002|行次  指标名称             本季度末数    上季度末数",
            "003|1     最低资本             369,992.25  342,378.32",
            "004|4     附加资本             -           -",
            "005|（二）保险风险最低资本汇总",
            "006|8     非寿险业务保险风险间的相关性效应  -  -",
            "007|（三）市场风险最低资本汇总",
        ])

        sliced = _slice_grid_for_target("MINIMUM_CAPITAL", grid)

        self.assertIn("最低资本", sliced)
        self.assertIn("369,992.25", sliced)
        self.assertIn("非寿险业务保险风险间的相关性效应", sliced)
        self.assertIn("（三）市场风险最低资本汇总", sliced)

    def test_final_page_stops_after_credit_risk_terminal_row(self):
        grid = "\n".join([
            "000|行次  指标名称             本季度末数    上季度末数",
            "001|8     市场风险间的相关性效应  173,485.63  146,445.47",
            "002|（四）信用风险最低资本汇总",
            "003|4     信用风险间的相关性效应  20,766.36  21,666.07",
            "004|S06-风险综合评级",
        ])

        sliced = _slice_grid_for_target("MINIMUM_CAPITAL", grid)

        self.assertIn("信用风险间的相关性效应", sliced)
        self.assertNotIn("S06-风险综合评级", sliced)

    def test_standard_tail_page_keeps_additional_capital_rows_before_total(self):
        grid = "\n".join([
            "000|3     附加资本        0.00  0.00",
            "001|3.1   逆周期附加资本  0.00  0.00",
            "002|3.2   D-SII附加资本   0.00  0.00",
            "003|3.3   G-SII附加资本   0.00  0.00",
            "004|3.4   其他附加资本    0.00  0.00",
            "005|4     最低资本        1,058,344.82  1,013,553.79",
            "006|28",
        ])

        sliced = _slice_grid_for_target("MINIMUM_CAPITAL", grid)

        self.assertIn("3     附加资本", sliced)
        self.assertIn("逆周期附加资本", sliced)
        self.assertIn("D-SII附加资本", sliced)
        self.assertIn("G-SII附加资本", sliced)
        self.assertIn("其他附加资本", sliced)
        self.assertIn("最低资本", sliced)

    def test_missing_standard_terminal_row_is_recovered_from_source_grid(self):
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["1.6.1", "损失吸收调整-不考虑上限", "474,596.03", "342,036.34"],
            ["1.6.2", "损失吸收效应调整上限", "2,033,042.90", "1,452,776.27"],
            ["2", "控制风险最低资本", "29,479.19", "25,226.22"],
            ["3", "附加资本", "0.00", "0.00"],
        ]
        grid = PageGrid(
            29,
            "\n".join([
                "009|2 控制风险最低资本 29,479.19 25,226.22",
                "010|3 附加资本 0.00 0.00",
                "011|3.1 逆周期附加资本 0.00 0.00",
                "012|3.2 D-SII附加资本 0.00 0.00",
                "013|3.3 G-SII附加资本 0.00 0.00",
                "014|3.4 其他附加资本 0.00 0.00",
                "015|4 最低资本 4,779,387.95 4,089,864.52",
            ]),
            30,
        )

        recovered, note = recover_minimum_capital_terminal_rows(
            "MINIMUM_CAPITAL",
            rows,
            [grid],
        )

        self.assertEqual(recovered[-1], [
            "4",
            "最低资本",
            "4,779,387.95",
            "4,089,864.52",
        ])
        self.assertEqual(
            [row[0] for row in recovered[-5:]],
            ["3.1", "3.2", "3.3", "3.4", "4"],
        )
        self.assertEqual(recovered[-2][1], "其他附加资本")
        self.assertIn("补回", note)

    def test_row_four_total_alias_is_recovered_as_minimum_capital(self):
        rows = [
            ["行次", "项目", "本季度", "上季度"],
            ["2", "控制风险最低资本（万元）", "-18,356", "-16,869"],
            ["3", "附加资本（万元）", "-", "-"],
        ]
        grids = [
            PageGrid(
                25,
                "031|2 控制风险最低资本（万元） -18,356 -16,869\n"
                "032|3 附加资本（万元） - -\n"
                "033|3.1 逆周期附加资本（万元） - -\n"
                "034|3.2 D-SII附加资本（万元） - -",
                24,
            ),
            PageGrid(
                26,
                "001|3.3 G-SII附加资本（万元） - -\n"
                "002|3.4 其他附加资本（万元） - -\n"
                "003|4 合计（万元） 2,116,093 1,944,677",
                18,
            ),
        ]

        recovered, note = recover_minimum_capital_terminal_rows(
            "MINIMUM_CAPITAL",
            rows,
            grids,
        )

        self.assertEqual(recovered[-1], [
            "4",
            "最低资本（万元）",
            "2,116,093",
            "1,944,677",
        ])
        self.assertEqual(
            [row[0] for row in recovered[-3:]],
            ["3.3", "3.4", "4"],
        )
        self.assertIn("补回", note)

        _, required_terms = _source_completeness_profile(
            "MINIMUM_CAPITAL",
            grids,
        )
        self.assertIn("最低资本", required_terms)
        self.assertNotIn("合计", required_terms)

    def test_existing_terminal_row_is_not_duplicated(self):
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["3", "附加资本", "0.00", "0.00"],
            ["4", "最低资本", "4,779,387.95", "4,089,864.52"],
        ]
        grid = PageGrid(
            29,
            "001|3 附加资本 0.00 0.00\n"
            "002|4 最低资本 4,779,387.95 4,089,864.52",
            12,
        )

        recovered, note = recover_minimum_capital_terminal_rows(
            "MINIMUM_CAPITAL",
            rows,
            [grid],
        )

        self.assertEqual(recovered, rows)
        self.assertEqual(note, "")

    def test_summary_first_opening_total_is_not_moved_to_tail(self):
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["2", "量化风险最低资本", "367,817.52", "340,365.90"],
            ["4", "附加资本", "-", "-"],
            ["8", "信用风险间的相关性效应", "20,766.36", "21,666.07"],
        ]
        grid = PageGrid(
            20,
            "\n".join([
                "001|1 最低资本 369,992.25 342,378.32",
                "002|2 量化风险最低资本 367,817.52 340,365.90",
                "003|4 附加资本 - -",
                "004|8 信用风险间的相关性效应 20,766.36 21,666.07",
            ]),
            20,
        )

        recovered, note = recover_minimum_capital_terminal_rows(
            "MINIMUM_CAPITAL",
            rows,
            [grid],
        )

        self.assertEqual(recovered, rows)
        self.assertEqual(note, "")


if __name__ == "__main__":
    unittest.main()
