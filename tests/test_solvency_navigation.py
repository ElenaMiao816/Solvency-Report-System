from __future__ import annotations

import unittest

import pandas as pd

from services.solvency_navigation import (
    ACTUAL_CAPITAL_LEVEL,
    COMPANY_FIRST_LEVELS,
    COMPANY_OVERVIEW_LEVEL,
    INDUSTRY_NAVIGATION,
    INDUSTRY_QUANT_CHART,
    INDUSTRY_FIRST_LEVELS,
    MINIMUM_CAPITAL_LEVEL,
    OVERVIEW_LEVEL,
    PRINT_ALL_LABEL,
    apply_navigation_labels,
    chart_names,
    first_levels_for_codes,
    metric_codes_for_chart,
    resolve_chart_selection,
    second_levels,
)
from services.solvency_normalizer import standardize_uploaded_frame


class SolvencyNavigationTests(unittest.TestCase):
    def test_company_and_industry_navigation_match_approved_guide(self):
        self.assertEqual(
            COMPANY_FIRST_LEVELS,
            (
                COMPANY_OVERVIEW_LEVEL,
                ACTUAL_CAPITAL_LEVEL,
                MINIMUM_CAPITAL_LEVEL,
                PRINT_ALL_LABEL,
            ),
        )
        self.assertEqual(INDUSTRY_FIRST_LEVELS, (OVERVIEW_LEVEL, PRINT_ALL_LABEL))
        self.assertEqual(second_levels(ACTUAL_CAPITAL_LEVEL)[0], "资本规模与结构")
        self.assertIn("四级资本规模与结构", chart_names(ACTUAL_CAPITAL_LEVEL, "全部"))
        self.assertNotIn("资本结构占比", chart_names(ACTUAL_CAPITAL_LEVEL, "全部"))
        self.assertIn("风险分散效应", chart_names(MINIMUM_CAPITAL_LEVEL, "全部"))
        self.assertIn("损失吸收效应", chart_names(MINIMUM_CAPITAL_LEVEL, "全部"))
        self.assertNotIn("风险分散效应和损失吸收", chart_names(MINIMUM_CAPITAL_LEVEL, "全部"))
        self.assertNotIn(
            INDUSTRY_QUANT_CHART,
            chart_names(MINIMUM_CAPITAL_LEVEL, "全部"),
        )
        self.assertIn(
            INDUSTRY_QUANT_CHART,
            chart_names(OVERVIEW_LEVEL, "全部", industry=True),
        )
        self.assertTrue(any(entry.chart_name == INDUSTRY_QUANT_CHART for entry in INDUSTRY_NAVIGATION))
        self.assertEqual(
            metric_codes_for_chart("综合偿付能力充足率"),
            ("COMBINED_SOLVENCY_RATIO",),
        )

    def test_step5_standard_frame_receives_navigation_labels(self):
        source = pd.DataFrame([
            {
                "公司": "测试人寿",
                "指标编码": "COMBINED_SOLVENCY_RATIO",
                "指标名称": "综合偿付能力充足率",
                "一级模块": "主要指标",
                "二级模块": "偿付能力",
                "数值": 180.0,
            },
            {
                "公司": "测试人寿",
                "指标编码": "NET_PROFIT",
                "指标名称": "净利润",
                "一级模块": "经营指标",
                "二级模块": "经营成果",
                "数值": 10.0,
            },
        ])
        result = standardize_uploaded_frame(source)
        ratio = result[result["指标编码"] == "COMBINED_SOLVENCY_RATIO"].iloc[0]
        profit = result[result["指标编码"] == "NET_PROFIT"].iloc[0]
        self.assertEqual(ratio["一级模块"], COMPANY_OVERVIEW_LEVEL)
        self.assertEqual(ratio["二级模块"], "偿付能力充足率整体分布")
        self.assertEqual(profit["一级模块"], "经营指标")
        self.assertEqual(profit["二级模块"], "经营成果")

    def test_apply_navigation_labels_preserves_unmapped_metrics(self):
        frame = pd.DataFrame([
            {"指标编码": "CORE_T1_CAPITAL", "一级模块": "旧一级", "二级模块": "旧二级"},
            {"指标编码": "UNMAPPED", "一级模块": "保留一级", "二级模块": "保留二级"},
        ])
        result = apply_navigation_labels(frame)
        self.assertEqual(result.iloc[0]["一级模块"], ACTUAL_CAPITAL_LEVEL)
        self.assertEqual(result.iloc[0]["二级模块"], "资本规模与结构")
        self.assertEqual(result.iloc[1]["一级模块"], "保留一级")

    def test_removed_capital_share_chart_is_not_offered(self):
        available = {
            "CORE_T1_TO_ACTUAL_CAPITAL",
            "CORE_T2_TO_ACTUAL_CAPITAL",
            "ANC_T1_TO_ACTUAL_CAPITAL",
            "ANC_T2_TO_ACTUAL_CAPITAL",
        }
        self.assertEqual(
            first_levels_for_codes(available),
            (PRINT_ALL_LABEL,),
        )
        self.assertEqual(
            second_levels(ACTUAL_CAPITAL_LEVEL, available_codes=available),
            [],
        )
        self.assertEqual(
            chart_names(
                ACTUAL_CAPITAL_LEVEL,
                "资本结构占比分布",
                available_codes=available,
            ),
            [],
        )

    def test_matrix_requires_both_metric_codes(self):
        only_core = {"CORE_SOLVENCY_RATIO"}
        both = {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}
        self.assertNotIn(
            "偿付能力矩阵",
            chart_names(COMPANY_OVERVIEW_LEVEL, "全部", available_codes=only_core),
        )
        self.assertIn(
            "偿付能力矩阵",
            chart_names(COMPANY_OVERVIEW_LEVEL, "全部", available_codes=both),
        )

    def test_all_second_level_resolves_every_available_chart(self):
        expected = chart_names(ACTUAL_CAPITAL_LEVEL, "全部")
        resolved = resolve_chart_selection(
            ACTUAL_CAPITAL_LEVEL,
            "全部",
            selected_chart=expected[0],
        )
        self.assertGreater(len(expected), 1)
        self.assertEqual(resolved, expected)

    def test_specific_second_level_keeps_single_chart_selection(self):
        resolved = resolve_chart_selection(
            ACTUAL_CAPITAL_LEVEL,
            "资本规模与结构",
            selected_chart="四级资本规模与结构",
        )
        self.assertEqual(resolved, ["四级资本规模与结构"])
