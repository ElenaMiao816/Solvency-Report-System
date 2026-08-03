from __future__ import annotations

import unittest

from services.solvency_ai_table_extractor import (
    PageGrid,
    _slice_grid_for_target,
    _source_completeness_profile,
    _source_item_labels,
    _validate,
)
from services.solvency_pdf_locator import _expand_to_item_boundaries
from services.table_strategy_handlers import PostprocessRequest
from services.table_strategy_registry import resolve_table_strategy
from services.solvency_table_boundaries import (
    boundary_instruction,
    boundary_items,
    enforce_output_boundaries,
    row_has_item,
)


class ActualCapitalBoundaryTests(unittest.TestCase):
    header = ["行次", "指标名称（单位：万元）", "本季度末", "上季度末"]

    def test_summary_first_layout_keeps_summary_and_core_details(self):
        rows = [
            self.header,
            ["1", "认可资产", "9,308,253.75", "8,851,412.82"],
            ["2", "认可负债", "8,711,831.69", "8,189,769.92"],
            ["3", "实际资本", "596,422.05", "661,642.90"],
            ["3.1", "核心一级资本", "404,902.10", "456,833.29"],
            ["3.2", "核心二级资本", "8,660.56", "9,393.09"],
            ["3.3", "附属一级资本", "182,859.40", "195,416.52"],
            ["3.4", "附属二级资本", "-", "-"],
            ["", "（二）核心资本", "", ""],
            ["1.1", "净资产", "305,390.79", "355,175.80"],
            ["5", "实际资本合计", "596,422.05", "661,642.90"],
            ["", "（三）认可资产", "", ""],
        ]

        bounded, note = enforce_output_boundaries("ACTUAL_CAPITAL", rows)

        labels = [row[1] for row in bounded[1:]]
        self.assertEqual(labels[0], "认可资产")
        self.assertIn("认可负债", labels)
        self.assertIn("实际资本", labels)
        self.assertIn("（二）核心资本", labels)
        self.assertEqual(labels[-1], "实际资本合计")
        self.assertNotIn("（三）认可资产", labels)
        self.assertIn("实际资本汇总在前并展开核心资本明细", note)

    def test_financial_statement_summary_keeps_all_rows_before_capital_details(self):
        rows = [
            self.header,
            ["1", "财务报表资产总额", "17,677,527.30", "17,275,442.21"],
            ["2", "认可资产总额", "17,582,518.95", "17,321,539.94"],
            ["3", "财务报表负债总额", "16,845,577.89", "16,266,724.65"],
            ["4", "认可负债总额", "16,192,480.60", "15,691,371.32"],
            ["5", "财务报表净资产总额（=1-3）", "831,949.41", "1,008,717.56"],
            ["6", "实际资本（=2-4）", "1,390,038.35", "1,630,168.62"],
            ["", "2. 实际资本表", "", ""],
            ["1", "核心一级资本", "688,098.58", "1,093,292.93"],
            ["2", "核心二级资本", "177,992.38", "99,891.04"],
            ["3", "附属一级资本", "430,772.53", "345,749.58"],
            ["4", "附属二级资本", "93,174.86", "91,235.07"],
            ["5", "实际资本合计", "1,390,038.35", "1,630,168.62"],
        ]

        bounded, note = enforce_output_boundaries(
            "ACTUAL_CAPITAL",
            rows,
            require_complete=True,
        )

        labels = [row[1] for row in bounded[1:]]
        self.assertEqual(labels[0], "财务报表资产总额")
        self.assertIn("认可资产总额", labels)
        self.assertIn("财务报表负债总额", labels)
        self.assertIn("认可负债总额", labels)
        self.assertTrue(any(label.startswith("财务报表净资产总额") for label in labels))
        self.assertEqual(labels[-1], "实际资本合计")
        self.assertIn("财务报表口径汇总在前并展开实际资本明细", note)
        quality, _ = _validate(
            "ACTUAL_CAPITAL",
            bounded,
            0.0,
            expected_source_rows=11,
            source_required_terms=(
                "财务报表资产总额",
                "认可资产",
                "财务报表负债总额",
                "认可负债",
                "财务报表净资产总额",
                "实际资本",
                "核心一级资本",
                "核心二级资本",
                "附属一级资本",
                "附属二级资本",
                "实际资本合计",
            ),
        )
        self.assertGreater(quality, 0)

    def test_original_core_detail_layout_remains_supported(self):
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["1", "核心一级资本", "404,902.10", "456,833.29"],
            ["2", "核心二级资本", "8,660.56", "9,393.09"],
            ["3", "附属一级资本", "182,859.40", "195,416.52"],
            ["4", "附属二级资本", "-", "-"],
            ["5", "实际资本合计", "596,422.05", "661,642.90"],
        ]

        bounded, note = enforce_output_boundaries("ACTUAL_CAPITAL", rows)

        self.assertEqual([row[1] for row in bounded[1:]], [
            "核心一级资本",
            "核心二级资本",
            "附属一级资本",
            "附属二级资本",
            "实际资本合计",
        ])
        self.assertIn("核心资本明细标准版", note)

    def test_standard_core_detail_accepts_and_normalizes_row_five_total_alias(self):
        rows = [
            ["行次", "项目", "本季度", "上季度"],
            ["1", "核心一级资本（万元）", "1,362,355", "1,562,420"],
            ["2", "核心二级资本（万元）", "0", "0"],
            ["3", "附属一级资本（万元）", "1,290,430", "1,869,740"],
            ["4", "附属二级资本（万元）", "0", "0"],
            ["5", "合计（万元）", "2,652,785", "3,432,160"],
        ]

        bounded, boundary_note = enforce_output_boundaries("ACTUAL_CAPITAL", rows)
        normalized, postprocess_notes = resolve_table_strategy(
            "ACTUAL_CAPITAL"
        ).postprocess(
            PostprocessRequest(table_id="ACTUAL_CAPITAL", rows=bounded)
        )

        self.assertEqual(bounded[-1][1], "合计（万元）")
        self.assertIn("核心资本明细标准版", boundary_note)
        self.assertEqual(normalized[-1][1], "实际资本合计（万元）")
        self.assertTrue(any("标准化" in note for note in postprocess_notes))

    def test_actual_capital_followed_by_tier_one_adjustment_is_complete(self):
        rows = [
            ["项目", "本季度数", "上季度数"],
            ["实际资本", "4,867,047,381", "4,988,209,124"],
            ["核心一级资本", "2,451,379,279", "2,585,757,433"],
            ["核心二级资本", "-", "-"],
            ["附属一级资本", "2,415,668,102", "2,402,451,691"],
            ["附属二级资本", "-", "-"],
            ["核心一级资本", "2,451,379,279", "2,585,757,433"],
            ["净资产", "1,524,870,105", "1,565,268,541"],
            ["对净资产的调整额", "926,509,175", "1,020,488,891"],
            ["各项非认可资产的账面价值", "-115,048,504", "-120,902,391"],
            ["长期股权投资的认可价值和账面价值的差额", "2,516,667", "2,494,918"],
            [
                "递延所得税资产（由经营性亏损引起的递延所得税资产除外）",
                "-91,060,213",
                "-91,060,213",
            ],
            ["计入核心一级资本的保单未来盈余", "486,794,731", "780,047,429"],
            ["银保监会规定的其他调整项目", "643,306,494", "449,909,149"],
            ["现金及流动性管理工具", "2,680,009,299", "1,500,634,187"],
        ]

        bounded, note = enforce_output_boundaries("ACTUAL_CAPITAL", rows)

        labels = [row[0] for row in bounded[1:]]
        self.assertEqual(labels[0], "实际资本")
        self.assertIn("净资产", labels)
        self.assertEqual(labels[-1], "银保监会规定的其他调整项目")
        self.assertNotIn("现金及流动性管理工具", labels)
        self.assertIn("实际资本表后接核心一级资本调整表", note)

    def test_recognized_summary_before_tier_one_adjustment_is_kept(self):
        rows = [
            ["项目", "本季度（末）数", "上季度（末）数"],
            ["认可资产", "2,349,375.01", "2,253,429.83"],
            ["认可负债", "2,143,520.99", "2,049,234.46"],
            ["实际资本", "205,854.02", "204,195.37"],
            ["核心一级资本", "167,543.95", "175,195.06"],
            ["核心二级资本", "-", "-"],
            ["附属一级资本", "37,895.81", "28,565.17"],
            ["附属二级资本", "414.26", "435.15"],
            ["核心一级资本调整表", "", ""],
            ["核心一级资本", "167,543.95", "175,195.06"],
            ["净资产", "128,622.00", "142,524.09"],
            ["对净资产的调整额", "38,921.95", "32,670.97"],
            ["各项非认可资产的账面价值", "-1,761.29", "-1,715.33"],
            ["计入核心一级资本的保单未来盈余", "24,258.05", "17,982.98"],
            ["银保监会规定的其他调整项目", "16,425.19", "16,403.33"],
            ["现金及流动性管理工具", "431,461.96", "415,140.33"],
        ]

        bounded, note = enforce_output_boundaries("ACTUAL_CAPITAL", rows)

        labels = [row[0] for row in bounded[1:]]
        self.assertEqual(labels[0], "认可资产")
        self.assertEqual(labels[1], "认可负债")
        self.assertIn("实际资本", labels)
        self.assertIn("净资产", labels)
        self.assertEqual(labels[-1], "银保监会规定的其他调整项目")
        self.assertNotIn("现金及流动性管理工具", labels)
        self.assertIn("认可资产汇总后接核心一级资本调整表", note)

        quality, _ = _validate(
            "ACTUAL_CAPITAL",
            bounded,
            0.0,
            expected_source_rows=10,
            source_required_terms=(
                "认可资产",
                "认可负债",
                "实际资本",
                "核心一级资本",
                "附属一级资本",
                "附属二级资本",
                "净资产",
                "对净资产的调整额",
                "银保监会规定的其他调整项目",
            ),
        )
        self.assertGreater(quality, 0)

    def test_core_detail_source_does_not_infer_recognized_assets_from_nonrecognized_asset(self):
        grid = PageGrid(
            26,
            "\n".join([
                "000|十、实际资本",
                "001|（一）实际资本",
                "002|行次 项目 本季度数 上季度可比数",
                "003|1 核心一级资本 31,769 41,124",
                "004|1.2.1 各项非认可资产的账面价值 76,929 65,062",
                "005|2 核心二级资本 1,216 1,498",
                "006|3 附属一级资本 6,313 8,634",
                "007|4 附属二级资本 1,005 1,205",
                "008|5 实际资本合计 40,302 52,461",
            ]),
            42,
        )

        expected_rows, required_terms = _source_completeness_profile(
            "ACTUAL_CAPITAL",
            [grid],
        )

        self.assertNotIn("认可资产", required_terms)
        self.assertIn("核心一级资本", required_terms)
        self.assertIn("实际资本合计", required_terms)
        _, summary_terms = _source_completeness_profile(
            "ACTUAL_CAPITAL",
            [
                PageGrid(
                    27,
                    "\n".join([
                        "000|行次 项目 本季度数 上季度可比数",
                        "001|1 认可资产 930,825 885,141",
                        "002|2 认可负债 871,183 818,976",
                        "003|3 实际资本 59,642 66,164",
                    ]),
                    18,
                )
            ],
        )
        self.assertIn("认可资产", summary_terms)
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["1", "核心一级资本", "31,769", "41,124"],
            ["1.2.1", "各项非认可资产的账面价值", "76,929", "65,062"],
            ["2", "核心二级资本", "1,216", "1,498"],
            ["3", "附属一级资本", "6,313", "8,634"],
            ["4", "附属二级资本", "1,005", "1,205"],
            ["5", "实际资本合计", "40,302", "52,461"],
        ]
        quality, _ = _validate(
            "ACTUAL_CAPITAL",
            rows,
            0.0,
            expected_source_rows=expected_rows,
            source_required_terms=required_terms,
        )
        self.assertGreater(quality, 0)

    def test_summary_rows_are_semantic_completeness_items(self):
        items = boundary_items("ACTUAL_CAPITAL")

        self.assertIn("认可资产", items)
        self.assertIn("财务报表资产总额", items)
        self.assertIn("认可负债", items)
        self.assertIn("实际资本", items)
        self.assertIn("实际资本合计", items)
        self.assertIn("实际资本披露支持多种原文顺序", boundary_instruction("ACTUAL_CAPITAL"))

    def test_total_and_formula_suffixes_match_without_arbitrary_fuzzy_matching(self):
        self.assertTrue(row_has_item(["2", "认可资产总额"], "认可资产"))
        self.assertTrue(row_has_item(["6", "实际资本（=2-4）"], "实际资本"))
        self.assertFalse(row_has_item(["", "实际资本表"], "实际资本"))


class ActualCapitalLocatorTests(unittest.TestCase):
    table = {"table_id": "ACTUAL_CAPITAL", "max_pages": 4}

    def test_summary_first_layout_ignores_solvency_main_false_candidate(self):
        page_texts = [
            (
                "偿付能力充足率指标\n"
                "1 认可资产 9,308,253.75 8,851,412.82\n"
                "3 实际资本 596,422.05 661,642.90\n"
                "3.1 核心一级资本 404,902.10 456,833.29"
            ),
            (
                "九、实际资本\n"
                "1 认可资产 9,308,253.75 8,851,412.82\n"
                "2 认可负债 8,711,831.69 8,189,769.92\n"
                "3 实际资本 596,422.05 661,642.90\n"
                "1 核心一级资本 404,902.10 456,833.29\n"
                "5 实际资本合计 596,422.05 661,642.90"
            ),
            "（三）认可资产\n1 现金及流动性管理工具 100 90",
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(2, 18.0, ["actual capital title anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [2])
        self.assertIn("起始项目：认可资产", expanded[0][2])
        self.assertIn("终止项目：实际资本合计", expanded[0][2])

    def test_original_layout_still_closes_from_core_to_total(self):
        page_texts = [
            (
                "实际资本表\n"
                "1 核心一级资本 404,902.10 456,833.29\n"
                "2 核心二级资本 8,660.56 9,393.09\n"
                "5 实际资本合计 596,422.05 661,642.90"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 12.0, ["anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [1])
        self.assertIn("边界版式：核心资本明细标准版", expanded[0][2])

    def test_original_layout_closes_when_row_five_is_named_total_only(self):
        page_texts = [
            (
                "十、实际资本\n实际资本表\n"
                "1 核心一级资本（万元） 1,362,355 1,562,420\n"
                "2 核心二级资本（万元） 0 0\n"
                "3 附属一级资本（万元） 1,290,430 1,869,740\n"
                "4 附属二级资本（万元） 0 0\n"
                "5 合计（万元） 2,652,785 3,432,160"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 12.0, ["anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [1])
        self.assertIn("边界版式：核心资本明细标准版", expanded[0][2])

    def test_financial_statement_summary_closes_at_actual_capital_total(self):
        page_texts = [
            (
                "九、实际资本\n1.实际资本汇总表\n"
                "1 财务报表资产总额 17,677,527.30 17,275,442.21\n"
                "2 认可资产总额 17,582,518.95 17,321,539.94\n"
                "3 财务报表负债总额 16,845,577.89 16,266,724.65\n"
                "4 认可负债总额 16,192,480.60 15,691,371.32\n"
                "5 财务报表净资产总额 831,949.41 1,008,717.56\n"
                "6 实际资本 1,390,038.35 1,630,168.62\n"
                "2.实际资本表\n"
                "1 核心一级资本 688,098.58 1,093,292.93\n"
                "2 核心二级资本 177,992.38 99,891.04\n"
                "3 附属一级资本 430,772.53 345,749.58\n"
                "4 附属二级资本 93,174.86 91,235.07\n"
                "5 实际资本合计 1,390,038.35 1,630,168.62"
            ),
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(1, 18.0, ["actual capital title anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [1])
        self.assertIn("起始项目：财务报表资产总额", expanded[0][2])
        self.assertIn("边界版式：财务报表口径汇总在前并展开实际资本明细", expanded[0][2])

    def test_tier_one_adjustment_layout_closes_before_recognized_assets(self):
        page_texts = [
            "偿付能力充足率指标\n实际资本 4,867,047,381 4,988,209,124",
            (
                "十、实际资本\n（一）实际资本表\n"
                "实际资本 4,867,047,381 4,988,209,124\n"
                "核心一级资本 2,451,379,279 2,585,757,433\n"
                "（二）核心一级资本调整表\n"
                "净资产 1,524,870,105 1,565,268,541\n"
                "银保监会规定的其他调整项目 643,306,494 449,909,149\n"
                "（三）认可资产表"
            ),
            "（四）认可负债表\n准备金负债 28,637,813,296 27,439,136,140",
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            self.table,
            [(2, 18.0, ["actual capital title anchor"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [2])
        self.assertIn("起始项目：实际资本", expanded[0][2])
        self.assertIn("终止项目：银保监会规定的其他调整项目", expanded[0][2])
        self.assertIn("边界版式：实际资本表后接核心一级资本调整表", expanded[0][2])


class ActualCapitalGridSliceTests(unittest.TestCase):
    def test_row_five_total_alias_is_kept_and_canonicalized_for_completeness(self):
        grid_text = "\n".join([
            "000|十、实际资本",
            "001|实际资本表",
            "002|行次 项目 本季度 上季度",
            "003|1 核心一级资本（万元） 1,362,355 1,562,420",
            "004|1.1 净资产（万元） 1,000,000 1,100,000",
            "005|2 核心二级资本（万元） 0 0",
            "006|3 附属一级资本（万元） 1,290,430 1,869,740",
            "007|4 附属二级资本（万元） 0 0",
            "008|5 合计（万元） 2,652,785 3,432,160",
            "009|20",
        ])
        page = PageGrid(22, grid_text, 90)

        sliced = _slice_grid_for_target("ACTUAL_CAPITAL", grid_text)
        _, required_terms = _source_completeness_profile("ACTUAL_CAPITAL", [page])

        self.assertIn("5 合计（万元）", sliced)
        self.assertIn("实际资本合计", required_terms)
        self.assertNotIn("合计", required_terms)

    def test_standard_s02_keeps_total_after_other_adjustment_row(self):
        grid = "\n".join([
            "000|九、实际资本",
            "001|S02-实际资本表",
            "002|行次 项目 期末数 期初数",
            "003|1 核心一级资本 1,641,334,342.78 2,050,181,427.34",
            "004|1.1 净资产 1,704,823,187.32 1,759,134,636.38",
            "005|1.2 对净资产的调整额 -63,488,844.54 291,046,790.96",
            "006|1.2.8 银保监会规定的其他调整项目 -396,089,619.74 0.00",
            "007|2 核心二级资本 54,315,981.02 44,930,371.42",
            "008|3 附属一级资本 597,233,712.38 377,328,591.43",
            "009|4 附属二级资本 4,791,721.35 36,400,907.30",
            "010|5 实际资本合计 2,297,675,757.53 2,508,841,297.49",
        ])

        sliced = _slice_grid_for_target("ACTUAL_CAPITAL", grid)

        self.assertIn("银保监会规定的其他调整项目", sliced)
        self.assertIn("核心二级资本", sliced)
        self.assertIn("附属一级资本", sliced)
        self.assertIn("附属二级资本", sliced)
        self.assertIn("实际资本合计", sliced)

    def test_grid_slice_keeps_both_subtables_and_stops_before_recognized_assets(self):
        grid = "\n".join([
            "000|九、实际资本",
            "001|（一）实际资本汇总",
            "002|行次  指标名称（单位：万元）  本季度末  上季度末",
            "003|1     认可资产              9,308,253.75  8,851,412.82",
            "004|2     认可负债              8,711,831.69  8,189,769.92",
            "005|3     实际资本              596,422.05    661,642.90",
            "006|（二）核心资本",
            "007|行次  项目                  本季度数（万元）  上季度数（万元）",
            "008|1     核心一级资本          404,902.10  456,833.29",
            "009|1.1   净资产                305,390.79  355,175.80",
            "010|5     实际资本合计          596,422.05  661,642.90",
            "011|（三）认可资产",
            "012|1     现金及流动性管理工具  100  90",
        ])

        sliced = _slice_grid_for_target("ACTUAL_CAPITAL", grid)

        self.assertIn("认可资产", sliced)
        self.assertIn("认可负债", sliced)
        self.assertIn("（二）核心资本", sliced)
        self.assertIn("净资产", sliced)
        self.assertIn("实际资本合计", sliced)
        self.assertNotIn("（三）认可资产", sliced)
        self.assertNotIn("现金及流动性管理工具", sliced)

    def test_grid_slice_starts_at_financial_statement_assets(self):
        grid = "\n".join([
            "000|九、实际资本",
            "001|1.实际资本汇总表",
            "002|1 财务报表资产总额 17,677,527.30 17,275,442.21",
            "003|2 认可资产总额 17,582,518.95 17,321,539.94",
            "004|3 财务报表负债总额 16,845,577.89 16,266,724.65",
            "005|4 认可负债总额 16,192,480.60 15,691,371.32",
            "006|5 财务报表净资产总额 831,949.41 1,008,717.56",
            "007|6 实际资本 1,390,038.35 1,630,168.62",
            "008|2.实际资本表",
            "009|1 核心一级资本 688,098.58 1,093,292.93",
            "010|2 核心二级资本 177,992.38 99,891.04",
            "011|3 附属一级资本 430,772.53 345,749.58",
            "012|4 附属二级资本 93,174.86 91,235.07",
            "013|5 实际资本合计 1,390,038.35 1,630,168.62",
            "014|十、最低资本",
        ])

        sliced = _slice_grid_for_target("ACTUAL_CAPITAL", grid)

        self.assertIn("财务报表资产总额", sliced)
        self.assertIn("财务报表负债总额", sliced)
        self.assertIn("财务报表净资产总额", sliced)
        self.assertIn("实际资本合计", sliced)
        self.assertNotIn("十、最低资本", sliced)

    def test_grid_slice_keeps_tier_one_adjustment_after_internal_footnote(self):
        grid_text = "\n".join([
            "000|十、实际资本",
            "001|（一）实际资本表",
            "002|项目 本季度数 上季度数",
            "003|实际资本 4,867,047,381 4,988,209,124",
            "004|核心一级资本 2,451,379,279 2,585,757,433",
            "005|核心二级资本 - -",
            "006|附属一级资本 2,415,668,102 2,402,451,691",
            "007|附属二级资本 - -",
            "008|注：上季度数据为审计后数据",
            "009|（二）核心一级资本调整表",
            "010|项目 本季度数 上季度数",
            "011|核心一级资本 2,451,379,279 2,585,757,433",
            "012|净资产 1,524,870,105 1,565,268,541",
            "013|对净资产的调整额 926,509,175 1,020,488,891",
            "014|各项非认可资产的账面价值 -115,048,504 -120,902,391",
            "015|长期股权投资的认可价值和账面价值的差额 2,516,667 2,494,918",
            "016|递延所得税资产（由经营性亏损引起的递延所得税资产除外） -91,060,213 -91,060,213",
            "017|计入核心一级资本的保单未来盈余 486,794,731 780,047,429",
            "018|银保监会规定的其他调整项目 643,306,494 449,909,149",
            "019|（三）认可资产表",
            "020|1 现金及流动性管理工具 2,680,009,299 1,500,634,187",
        ])
        page = PageGrid(24, grid_text, 90)

        sliced = _slice_grid_for_target("ACTUAL_CAPITAL", grid_text)
        expected_rows, required_terms = _source_completeness_profile(
            "ACTUAL_CAPITAL",
            [page],
        )
        source_labels = _source_item_labels("ACTUAL_CAPITAL", [page])

        self.assertIn("实际资本 4,867,047,381", sliced)
        self.assertIn("（二）核心一级资本调整表", sliced)
        self.assertIn("银保监会规定的其他调整项目", sliced)
        self.assertNotIn("（三）认可资产表", sliced)
        self.assertNotIn("现金及流动性管理工具", sliced)
        self.assertEqual(expected_rows, 11)
        self.assertIn("实际资本", required_terms)
        self.assertIn("净资产", required_terms)
        self.assertIn("银保监会规定的其他调整项目", required_terms)
        self.assertIn("银保监会规定的其他调整项目", source_labels)
        self.assertNotIn("现金及流动性管理工具", source_labels)


if __name__ == "__main__":
    unittest.main()
