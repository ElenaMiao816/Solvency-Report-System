from __future__ import annotations

import unittest

import pandas as pd

from services.solvency_navigation import (
    ACTUAL_CAPITAL_LEVEL,
    APPENDIX_LEVEL,
    COMPANY_FIRST_LEVELS,
    COMPANY_NAVIGATION,
    COMPANY_OVERVIEW_LEVEL,
    CREDIT_RISK_ASSET_SCATTER,
    INSURANCE_RISK_LIABILITY_SCATTER,
    INDUSTRY_NAVIGATION,
    INDUSTRY_QUANT_CHART,
    INDUSTRY_FIRST_LEVELS,
    KPMG_BRIGHT_CHART_COLORS,
    KPMG_BRIGHT_SERIES_COLORS,
    KPMG_CAPITAL_COMBO_COLORS,
    KPMG_CAPITAL_TIER_COLORS,
    KPMG_CHART_COLORS,
    KPMG_DEFAULT_COLORS,
    KPMG_LIGHT_CHART_COLORS,
    KPMG_PERIOD_CHART_COLORS,
    KPMG_QUANT_RISK_COLORS,
    MINIMUM_CAPITAL_LEVEL,
    MARKET_RISK_ASSET_SCATTER,
    OVERVIEW_LEVEL,
    PRINT_ALL_LABEL,
    RECOGNIZED_ASSET_LIABILITY_LEVEL,
    RECOGNIZED_ASSETS_LEVEL,
    OPERATING_QUALITY_LEVEL,
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
                RECOGNIZED_ASSET_LIABILITY_LEVEL,
                OPERATING_QUALITY_LEVEL,
                APPENDIX_LEVEL,
                PRINT_ALL_LABEL,
            ),
        )
        self.assertEqual(INDUSTRY_FIRST_LEVELS, (OVERVIEW_LEVEL, PRINT_ALL_LABEL))
        self.assertEqual(second_levels(ACTUAL_CAPITAL_LEVEL)[0], "行业资本分级")
        self.assertIn("资本规模与结构", chart_names(ACTUAL_CAPITAL_LEVEL, "全部"))
        self.assertIn("核心一级资本明细", chart_names(ACTUAL_CAPITAL_LEVEL, "全部"))
        policy_surplus_charts = chart_names(ACTUAL_CAPITAL_LEVEL, "保单未来盈余")
        self.assertIn("计入各级资本的保单未来盈余构成占比", policy_surplus_charts)
        self.assertNotIn(
            "计入核心资本的保单未来盈余/核心资本的比例",
            policy_surplus_charts,
        )
        self.assertNotIn("各级资本中的保单未来盈余占比", policy_surplus_charts)
        self.assertIn("量化风险最低资本构成", chart_names(MINIMUM_CAPITAL_LEVEL, "全部"))
        quant_codes = metric_codes_for_chart("量化风险最低资本构成")
        self.assertNotIn("FEATURE_FACTOR_IMPACT", quant_codes)
        self.assertNotIn("CONTROL_RISK_CAPITAL", quant_codes)
        self.assertIn("各类市场风险占比", chart_names(MINIMUM_CAPITAL_LEVEL, "全部"))
        market_codes = metric_codes_for_chart("各类市场风险占比")
        self.assertIn("OVERSEAS_EQUITY_RISK_CAPITAL", market_codes)
        self.assertIn("MARKET_RISK_CAPITAL", market_codes)
        self.assertIn("各类信用风险占比", chart_names(MINIMUM_CAPITAL_LEVEL, "信用风险"))
        self.assertNotIn("各类信用风险", chart_names(MINIMUM_CAPITAL_LEVEL, "信用风险"))
        credit_codes = metric_codes_for_chart("各类信用风险占比")
        self.assertIn("CREDIT_RISK_CAPITAL", credit_codes)
        self.assertEqual(RECOGNIZED_ASSETS_LEVEL, "认可资产负债数据对比")
        self.assertEqual(second_levels(RECOGNIZED_ASSET_LIABILITY_LEVEL), ["资产构成情况", "负债构成情况"])
        self.assertIn("认可资产构成", chart_names(RECOGNIZED_ASSETS_LEVEL, "全部"))
        self.assertEqual(
            metric_codes_for_chart("认可资产构成"),
            (
                "CASH_LIQUID_ASSETS", "INVESTMENT_ASSETS",
                "SUBSIDIARY_JV_ASSOCIATE_EQUITY", "REINSURANCE_ASSETS",
                "RECEIVABLES_AND_PREPAYMENTS", "FIXED_ASSETS",
                "LAND_USE_RIGHTS", "SEPARATE_ACCOUNT_ASSETS",
                "OTHER_RECOGNIZED_ASSETS",
            ),
        )
        self.assertEqual(
            chart_names(RECOGNIZED_ASSET_LIABILITY_LEVEL, "负债构成情况"),
            ["认可负债构成"],
        )
        self.assertEqual(
            metric_codes_for_chart("认可负债构成"),
            (
                "RESERVE_LIABILITIES", "FINANCIAL_LIABILITIES",
                "PAYABLES_AND_ADVANCES", "PROVISIONS",
                "SEPARATE_ACCOUNT_LIABILITY", "CAPITAL_LIABILITIES",
                "OTHER_RECOGNIZED_LIABILITIES",
            ),
        )
        self.assertEqual(
            chart_names(MINIMUM_CAPITAL_LEVEL, "保险风险"),
            [
                "各类保险风险（寿）占比",
                "保险风险（寿）/认可负债率",
                "保险风险（非寿）/认可负债率",
            ],
        )
        self.assertNotIn(
            "各类保险风险（寿）占比",
            chart_names(MINIMUM_CAPITAL_LEVEL, "保险风险（寿）"),
        )
        self.assertEqual(
            chart_names(MINIMUM_CAPITAL_LEVEL, "市场风险"),
            [
                "各类市场风险占比",
                "利率风险/认可资产率",
                "权益价格风险/认可资产率",
            ],
        )
        self.assertEqual(
            chart_names(MINIMUM_CAPITAL_LEVEL, "信用风险"),
            [
                "各类信用风险占比",
                "利差风险/认可资产率",
                "对手违约风险/认可资产率",
            ],
        )
        removed_scatter_charts = {
            MARKET_RISK_ASSET_SCATTER,
            CREDIT_RISK_ASSET_SCATTER,
            INSURANCE_RISK_LIABILITY_SCATTER,
        }
        self.assertTrue(removed_scatter_charts.isdisjoint(
            chart_names(MINIMUM_CAPITAL_LEVEL, "全部")
        ))
        for chart_name in removed_scatter_charts:
            self.assertEqual(metric_codes_for_chart(chart_name), ())
        removed_ratio_sections = {
            "利率风险最低资本占认可资产率",
            "权益价格风险最低资本占认可资产率",
            "利差风险最低资本占认可资产率",
            "对手违约风险最低资本占认可资产率",
            "保险风险（寿）最低资本占认可负债率",
            "保险风险（非寿）最低资本占认可负债率",
        }
        self.assertTrue(removed_ratio_sections.isdisjoint(
            second_levels(MINIMUM_CAPITAL_LEVEL)
        ))
        self.assertEqual(
            metric_codes_for_chart("保险风险（寿）/认可负债率"),
            ("LIFE_INSURANCE_RISK_TO_LIABILITIES",),
        )
        self.assertEqual(
            metric_codes_for_chart("保险风险（非寿）/认可负债率"),
            ("NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",),
        )
        self.assertEqual(
            metric_codes_for_chart("利率风险/认可资产率"),
            ("INTEREST_RATE_RISK_TO_ASSETS",),
        )
        self.assertEqual(
            metric_codes_for_chart("权益价格风险/认可资产率"),
            ("EQUITY_RISK_TO_ASSETS",),
        )
        self.assertEqual(
            metric_codes_for_chart("利差风险/认可资产率"),
            ("SPREAD_RISK_TO_ASSETS",),
        )
        self.assertEqual(
            metric_codes_for_chart("对手违约风险/认可资产率"),
            ("COUNTERPARTY_RISK_TO_ASSETS",),
        )
        self.assertEqual(
            metric_codes_for_chart("核心资本/注册资本"),
            ("CORE_CAPITAL_TO_REGISTERED_CAPITAL",),
        )
        removed_duplicate_charts = {
            "寿险业务保险风险最低资本占比",
            "市场风险最低资本占比",
            "信用风险最低资本占比",
        }
        self.assertTrue(
            removed_duplicate_charts.isdisjoint(
                chart_names(MINIMUM_CAPITAL_LEVEL, "全部")
            )
        )
        self.assertNotIn(
            "寿险业务保险风险最低资本占比",
            chart_names(MINIMUM_CAPITAL_LEVEL, "保险风险（寿）"),
        )
        self.assertNotIn(
            "市场风险最低资本占比",
            chart_names(MINIMUM_CAPITAL_LEVEL, "市场风险"),
        )
        self.assertNotIn(
            "信用风险最低资本占比",
            chart_names(MINIMUM_CAPITAL_LEVEL, "信用风险"),
        )
        for chart_name in removed_duplicate_charts:
            self.assertEqual(metric_codes_for_chart(chart_name), ())
        self.assertNotIn(
            "调研公司分类列表",
            chart_names(COMPANY_OVERVIEW_LEVEL, "偿付能力披露整体情况"),
        )
        self.assertFalse(any(
            entry.chart_name == "调研公司分类列表" for entry in COMPANY_NAVIGATION
        ))
        self.assertNotIn(
            INDUSTRY_QUANT_CHART,
            chart_names(MINIMUM_CAPITAL_LEVEL, "全部"),
        )
        self.assertIn(
            INDUSTRY_QUANT_CHART,
            chart_names(OVERVIEW_LEVEL, "全部", industry=True),
        )
        self.assertTrue(any(entry.chart_name == INDUSTRY_QUANT_CHART for entry in INDUSTRY_NAVIGATION))
        capital_ratio_charts = chart_names(COMPANY_OVERVIEW_LEVEL, "资本充足率")
        self.assertIn("核心及综合充足率", capital_ratio_charts)
        self.assertIn("核心资本占比", capital_ratio_charts)
        policy_surplus_ratio_chart = "计入核心资本的保单未来盈余/核心资本的比例"
        core_share_index = capital_ratio_charts.index("核心资本占比")
        self.assertEqual(
            capital_ratio_charts[core_share_index + 1],
            policy_surplus_ratio_chart,
        )
        self.assertEqual(
            chart_names(COMPANY_OVERVIEW_LEVEL, "保单未来盈余"),
            [policy_surplus_ratio_chart],
        )
        self.assertEqual(
            chart_names(COMPANY_OVERVIEW_LEVEL, "全部").count(policy_surplus_ratio_chart),
            1,
        )
        self.assertEqual(
            capital_ratio_charts[-1],
            "资本使用效率与核心资本占比气泡图",
        )
        self.assertNotIn("核心资本", capital_ratio_charts)
        self.assertNotIn("核心充足率柱状图", capital_ratio_charts)
        self.assertNotIn("综合充足率柱状图", capital_ratio_charts)
        self.assertEqual(
            metric_codes_for_chart("核心及综合充足率"),
            ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"),
        )
        self.assertEqual(
            metric_codes_for_chart("资本使用效率与核心资本占比气泡图"),
            (
                "ACTUAL_CAPITAL",
                "RECOGNIZED_ASSETS",
                "REGISTERED_CAPITAL",
                "CORE_T1_CAPITAL",
                "CORE_T2_CAPITAL",
            ),
        )
        self.assertEqual(
            KPMG_BRIGHT_CHART_COLORS[:4],
            ("#FFA3DA", "#00B8F5", "#FD349C", "#269924"),
        )
        self.assertEqual(
            KPMG_PERIOD_CHART_COLORS[:3],
            ("#ACEAFF", "#00B8F5", "#B497FF"),
        )
        self.assertEqual(
            KPMG_BRIGHT_SERIES_COLORS[:3],
            ("#00B8F5", "#FD349C", "#00C0AE"),
        )
        self.assertEqual(
            KPMG_CAPITAL_TIER_COLORS,
            ("#ACEAFF", "#FFA3DA", "#00B8F5", "#B497FF"),
        )
        self.assertEqual(KPMG_CAPITAL_COMBO_COLORS, ("#1E49E2", "#00B8F5"))
        self.assertEqual(
            KPMG_QUANT_RISK_COLORS,
            ("#ACEAFF", "#00B8F5", "#FFA3DA", "#B497FF", "#63EBB2", "#00C0AE"),
        )
        forbidden_non_key_fills = {
            "#00338D",  # KPMG Blue
            "#1E49E2",  # Cobalt Blue
            "#0C233C",  # Dark Blue
            "#7213EA",  # Purple
            "#510DBC",  # Dark Purple
            "#AB0D82",  # Dark Pink
            "#098E7E",  # Dark Green
        }
        for palette in (
            KPMG_BRIGHT_CHART_COLORS,
            KPMG_BRIGHT_SERIES_COLORS,
            KPMG_CAPITAL_TIER_COLORS,
            KPMG_QUANT_RISK_COLORS,
        ):
            self.assertTrue(set(palette).isdisjoint(forbidden_non_key_fills))

        automatic_palettes = (
            KPMG_DEFAULT_COLORS,
            KPMG_CHART_COLORS,
            KPMG_LIGHT_CHART_COLORS,
            KPMG_BRIGHT_CHART_COLORS,
            KPMG_BRIGHT_SERIES_COLORS,
            KPMG_CAPITAL_TIER_COLORS,
            KPMG_PERIOD_CHART_COLORS,
            KPMG_QUANT_RISK_COLORS,
        )
        forbidden_pairs = (
            {"#510DBC", "#7213EA"},  # Dark Purple + Purple
            {"#76D2FF", "#ACEAFF"},  # Blue + Light Blue
        )
        for palette in automatic_palettes:
            for forbidden_pair in forbidden_pairs:
                self.assertFalse(forbidden_pair.issubset(set(palette)))

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
        self.assertEqual(ratio["二级模块"], "资本充足率")
        self.assertEqual(profit["一级模块"], "经营指标")
        self.assertEqual(profit["二级模块"], "经营成果")

    def test_apply_navigation_labels_preserves_unmapped_metrics(self):
        frame = pd.DataFrame([
            {"指标编码": "CORE_T1_CAPITAL", "一级模块": "旧一级", "二级模块": "旧二级"},
            {"指标编码": "UNMAPPED", "一级模块": "保留一级", "二级模块": "保留二级"},
        ])
        result = apply_navigation_labels(frame)
        self.assertEqual(result.iloc[0]["一级模块"], COMPANY_OVERVIEW_LEVEL)
        self.assertEqual(result.iloc[0]["二级模块"], "资本充足率")
        self.assertEqual(result.iloc[1]["一级模块"], "保留一级")

    def test_capital_detail_entries_remain_visible_without_metrics(self):
        available = {
            "CORE_T1_TO_ACTUAL_CAPITAL",
            "CORE_T2_TO_ACTUAL_CAPITAL",
            "ANC_T1_TO_ACTUAL_CAPITAL",
            "ANC_T2_TO_ACTUAL_CAPITAL",
        }
        self.assertIn(ACTUAL_CAPITAL_LEVEL, first_levels_for_codes(available))
        self.assertIn("核心资本", second_levels(ACTUAL_CAPITAL_LEVEL, available_codes=available))
        self.assertEqual(
            chart_names(ACTUAL_CAPITAL_LEVEL, "核心资本", available_codes=available),
            ["核心一级资本明细"],
        )

    def test_operating_quality_remains_visible_for_undisclosed_companies(self):
        self.assertIn(OPERATING_QUALITY_LEVEL, first_levels_for_codes([]))
        self.assertEqual(second_levels(OPERATING_QUALITY_LEVEL), ["业务质量指标", "投资质量指标"])
        self.assertEqual(len(chart_names(OPERATING_QUALITY_LEVEL, "全部", available_codes=[])), 8)

    def test_capital_stack_requires_all_four_metric_codes(self):
        only_core = {"CORE_T1_CAPITAL"}
        all_four = {"CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"}
        self.assertNotIn(
            "资本规模与结构",
            chart_names(ACTUAL_CAPITAL_LEVEL, "全部", available_codes=only_core),
        )
        self.assertIn(
            "资本规模与结构",
            chart_names(ACTUAL_CAPITAL_LEVEL, "全部", available_codes=all_four),
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
            "行业资本分级",
            selected_chart="资本规模与结构",
        )
        self.assertEqual(resolved, ["资本规模与结构"])
