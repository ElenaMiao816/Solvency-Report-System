from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from services.solvency_step7_quality import (
    anc_pie_figure,
    anc_stack_figure,
    combo_figure,
    complete_external_capital_detail_zeros,
    core_omitted_nonzero,
    core_waterfall_figure,
    capital_value,
    has_values,
    surrender_comparison_figure,
    period_scopes,
    radar_figure,
    radar_values,
    ratio_bar_figure,
    value_for,
)
from services.solvency_step7_quality_charts import company_quality_chart


def row(code: str, value: float | None, period: str = "2026Q2", scope: str = "本季度（末）数") -> dict:
    return {"公司": "甲公司", "报告期": period, "期间口径": scope, "指标编码": code, "数值": value}


class Step7QualityTests(unittest.TestCase):
    def test_investment_radar_hides_period_scope_explanatory_notes(self):
        source = (Path(__file__).resolve().parents[1] / "step7_solvency.py").read_text(encoding="utf-8")
        self.assertNotIn("仅展示最新报告期 {latest}；净资产收益率", source)
        self.assertNotIn("期间口径说明：如旧集成表仅保留", source)
        self.assertIn(
            'if missing_by_company and chart_name != "近三年平均投资收益率与综合投资收益率":',
            source,
        )

    def test_old_external_wide_capital_blanks_are_zero_in_charts(self):
        external = lambda code, value, period="2025Q4": {
            **row(code, value, period=period, scope="本季度末数"),
            "来源类型": "外部数据集", "单位": "万元",
        }
        old_data = pd.DataFrame([
            external("NET_ASSETS", 100),
            external("NON_RECOGNIZED_ASSET_BOOK_VALUE", -10),
            external("CORE_T1_CAPITAL", 90),
            external("ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", 20),
            external("ANC_T1_CAPITAL", 20),
            external("ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", 15, "2025Q2"),
            external("ANC_T1_CAPITAL", 15, "2025Q2"),
            {**row("ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", 5, period="2025Q4"),
             "公司": "乙公司", "来源类型": "PDF提取", "单位": "万元"},
            {**row("OTHER_CORE_T1_ADJUSTMENT", None, period="2025Q4"),
             "公司": "丙公司", "来源类型": "外部数据集", "披露状态": "未披露"},
        ])
        completed = complete_external_capital_detail_zeros(old_data)
        completed_again = complete_external_capital_detail_zeros(completed)
        self.assertEqual(len(completed), len(completed_again))
        company = completed.loc[completed["公司"].eq("甲公司")]
        figure, missing, difference = core_waterfall_figure(company, "2025Q4")
        self.assertIsNotNone(figure)
        self.assertEqual(missing, [])
        self.assertEqual(difference, 0)
        self.assertEqual(list(figure.data[0].y)[2:4], [0, 0])
        pie, pie_missing, _ = anc_pie_figure(company, "2025Q4")
        stack, stack_missing = anc_stack_figure(company, ["2025Q2", "2025Q4"])
        self.assertIsNotNone(pie)
        self.assertEqual(pie_missing, [])
        self.assertEqual(stack_missing, [])
        self.assertEqual(float(stack.data[0].y[0]), 0)
        inferred = company.loc[company["指标编码"].eq("DEFERRED_TAX_ASSET_ADJUSTMENT")]
        self.assertEqual(inferred.iloc[0]["披露状态"], "推定零值")
        self.assertEqual(len(completed.loc[completed["公司"].eq("乙公司")]), 1)
        self.assertEqual(len(completed.loc[completed["公司"].eq("丙公司")]), 1)

    def test_waterfall_bars_are_wider_and_still_scale_with_panel(self):
        frame = pd.DataFrame([
            row("FINANCIAL_STATEMENT_NET_ASSETS", 100),
            row("NON_RECOGNIZED_ASSET_BOOK_VALUE", -10),
            row("CORE_T1_CAPITAL", 90),
        ])
        figure, _, _ = core_waterfall_figure(frame, "2026Q2")
        figure.update_yaxes(range=[0, 120])
        for company_count, expected_fraction in ((3, 0.94), (7, 0.9), (12, 0.8)):
            with self.subTest(company_count=company_count):
                spec = company_quality_chart(figure, "甲公司", company_count, unit="十亿元").to_dict(validate=True)
                bars = spec["layer"][1]
                connector = spec["layer"][0]
                self.assertEqual(bars["mark"]["width"], {"band": expected_fraction})
                self.assertEqual(bars["encoding"]["x"]["scale"], {"paddingInner": 0.18, "paddingOuter": 0.6})
                self.assertEqual(connector["encoding"]["x"]["scale"], bars["encoding"]["x"]["scale"])
                self.assertNotIn("width", spec)

    def test_waterfall_uses_external_wide_net_assets_as_start(self):
        frame = pd.DataFrame([
            {**row("NET_ASSETS", 100, scope="本季度末数"), "来源类型": "外部数据集"},
            {**row("NON_RECOGNIZED_ASSET_BOOK_VALUE", -10, scope="本季度末数"), "来源类型": "外部数据集"},
            {**row("CORE_T1_CAPITAL", 90, scope="本季度末数"), "来源类型": "外部数据集"},
        ])
        figure, missing, difference = core_waterfall_figure(frame, "2026Q2")
        self.assertIsNotNone(figure)
        self.assertTrue(has_values(figure))
        self.assertEqual(figure.data[0].y[0], 100)
        self.assertEqual(figure.data[0].y[1], -10)
        self.assertEqual(figure.data[0].y[-1], 90)
        self.assertNotIn("净资产", missing)
        self.assertEqual(difference, 0)

    def test_waterfall_does_not_substitute_operating_net_assets_from_pdf(self):
        frame = pd.DataFrame([
            {**row("NET_ASSETS", 100), "来源类型": "PDF提取"},
            {**row("CORE_T1_CAPITAL", 90), "来源类型": "PDF提取"},
        ])
        figure, missing, _ = core_waterfall_figure(frame, "2026Q2")
        self.assertIsNone(figure)
        self.assertIn("净资产", missing)

    @unittest.skipUnless(
        Path(r"F:\CROSS\数据包\CROSS汇总表24Q4&25Q2Q4_0920.xlsx").exists(),
        "Local CROSS workbook unavailable",
    )
    def test_real_cross_wide_rows_render_core_waterfall(self):
        from services.solvency_dataset_adapter import convert_external_workbook
        from services.solvency_metric_registry import extend_taxonomy
        from services.solvency_normalizer import load_taxonomy

        root = Path(__file__).parents[1]
        report = Path(r"F:\CROSS\数据包\CROSS汇总表24Q4&25Q2Q4_0920.xlsx")
        taxonomy = extend_taxonomy(load_taxonomy(root / "config" / "solvency_taxonomy.xlsx"))
        converted = convert_external_workbook(report.read_bytes(), report.name, taxonomy).data
        old_package = converted.loc[converted["来源类型"].ne("宽表空白推定")].copy()
        repaired = complete_external_capital_detail_zeros(old_package)
        for company in ("中邮人寿", "中银三星", "农银人寿", "交银人寿", "工银安盛", "建信人寿", "招商信诺"):
            with self.subTest(company=company):
                rows = converted.loc[converted["原始公司名称"].eq(company)]
                figure, missing, _ = core_waterfall_figure(rows, "2025Q4")
                self.assertIsNotNone(figure)
                self.assertTrue(has_values(figure))
                self.assertNotIn("净资产", missing)
                self.assertEqual(len(figure.data[0].x), 7)
                self.assertNotIn("农业大灾风险准备金", missing)
                self.assertNotIn("合格负债类资本工具", missing)
                old_rows = repaired.loc[repaired["原始公司名称"].eq(company)]
                _, old_missing, _ = core_waterfall_figure(old_rows, "2025Q4")
                self.assertEqual(old_missing, [])
        for company in ("农银人寿", "交银人寿", "工银安盛"):
            with self.subTest(company=company):
                old_rows = repaired.loc[repaired["原始公司名称"].eq(company)]
                _, missing, _ = anc_pie_figure(old_rows, "2025Q4")
                _, period_missing = anc_stack_figure(old_rows, ["2024Q4", "2025Q2", "2025Q4"])
                self.assertEqual(missing, [])
                self.assertEqual(period_missing, [])
        for company in ("中邮人寿", "中银三星", "建信人寿", "招商信诺"):
            with self.subTest(company=company):
                old_rows = repaired.loc[repaired["原始公司名称"].eq(company)]
                pie, missing, _ = anc_pie_figure(old_rows, "2025Q4")
                _, period_missing = anc_stack_figure(old_rows, ["2024Q4", "2025Q2", "2025Q4"])
                self.assertIsNone(pie)
                self.assertEqual(missing, ["附属一级资本明细"])
                self.assertIn("2025Q4：附属一级资本明细", period_missing)

    def test_waterfall_combines_valuation_components_and_omits_two_steps(self):
        frame = pd.DataFrame([
            row("FINANCIAL_STATEMENT_NET_ASSETS", 100),
            row("NON_RECOGNIZED_ASSET_BOOK_VALUE", -10),
            row("LONG_TERM_EQUITY_VALUATION_DIFFERENCE", 4),
            row("CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT", 6),
            row("DEFERRED_TAX_ASSET_ADJUSTMENT", -2),
            row("POLICY_SURPLUS_CORE_T1", 3),
            row("OTHER_CORE_T1_ADJUSTMENT", -1),
            row("AGRICULTURAL_CATASTROPHE_RISK_RESERVE", 2),
            row("QUALIFYING_CORE_T1_LIABILITY_CAPITAL", 0),
            row("CORE_T1_CAPITAL", 102),
        ])
        figure, missing, difference = core_waterfall_figure(frame, "2026Q2")
        self.assertEqual(list(figure.data[0].y), [100, -10, 10, -2, 3, -1, 102])
        self.assertEqual(len(figure.data[0].x), 7)
        self.assertIn("长期股权投资及投资性房地产估值差额", figure.data[0].x)
        self.assertNotIn("农业大灾风险准备金", missing)
        self.assertNotIn("合格负债类资本工具", missing)
        self.assertEqual(core_omitted_nonzero(frame, "2026Q2"), [("农业大灾风险准备金", 2)])
        self.assertEqual(difference, 0)

    def test_combined_valuation_keeps_disclosed_part_and_names_missing_part(self):
        frame = pd.DataFrame([
            row("FINANCIAL_STATEMENT_NET_ASSETS", 100),
            row("LONG_TERM_EQUITY_VALUATION_DIFFERENCE", 7),
            row("CORE_T1_CAPITAL", 107),
        ])
        figure, missing, _ = core_waterfall_figure(frame, "2026Q2")
        self.assertEqual(figure.data[0].y[2], 7)
        self.assertIn("投资性房地产估值差额（税后）", missing)
        self.assertIn("部分未披露", figure.data[0].customdata[2])

    def test_value_rate_uses_matching_period_scope_and_missing_is_not_zero(self):
        frame = pd.DataFrame([
            row("NEW_BUSINESS_VALUE", 20), row("SIGNED_PREMIUM", 100), row("RENEWAL_PREMIUM", 60),
            row("NEW_BUSINESS_VALUE", 50, scope="本年累计数"),
            row("SIGNED_PREMIUM", 200, scope="本年累计数"),
            row("RENEWAL_PREMIUM", 150, scope="本年累计数"),
            row("NEW_BUSINESS_VALUE", 40, period="2026Q1"),
        ])
        axes = period_scopes(frame, ("NEW_BUSINESS_VALUE", "SIGNED_PREMIUM", "RENEWAL_PREMIUM"), ["2026Q1", "2026Q2"])
        figure, missing, invalid = combo_figure(
            frame, ["2026Q1", "2026Q2"], "NEW_BUSINESS_VALUE", "SIGNED_PREMIUM",
            derived_rate=True, axes=axes,
        )
        self.assertEqual(list(figure.data[1].y), [None, None, 50.0, 100.0])
        self.assertEqual(len(missing), 5)
        self.assertEqual(invalid, [])

    def test_radar_uses_separate_three_year_codes(self):
        frame = pd.DataFrame([
            row("ROE", 5), row("ROA", 2),
            row("INVESTMENT_RETURN", 3), row("COMPREHENSIVE_INVESTMENT_RETURN", 4),
            row("THREE_YEAR_AVG_INVESTMENT_RETURN", 6, scope="本季度"),
            row("THREE_YEAR_AVG_COMPREHENSIVE_INVESTMENT_RETURN", 7, scope="本季度"),
        ])
        values, missing = radar_values(frame, "2026Q2")
        self.assertEqual(values, [5, 2, 3, 4, 6, 7])
        self.assertEqual(missing, [])

        figure, _ = radar_figure(frame, "2026Q2")
        self.assertEqual(list(figure.data[0].theta)[:6], ["1", "2", "3", "4", "5", "6"])
        self.assertEqual(
            list(figure.data[0].customdata)[:6],
            [label for _, label, _ in (
                ("ROE", "净资产收益率", "return"),
                ("ROA", "总资产收益率", "return"),
                ("INVESTMENT_RETURN", "投资收益率（当季数）", "quarter"),
                ("COMPREHENSIVE_INVESTMENT_RETURN", "综合投资收益率（当季数）", "quarter"),
                ("INVESTMENT_RETURN", "近三年平均投资收益率", "three_year"),
                ("COMPREHENSIVE_INVESTMENT_RETURN", "近三年平均综合投资收益率", "three_year"),
            )],
        )
        self.assertIn("%{customdata}", figure.data[0].hovertemplate)
        figure.update_layout(polar=dict(radialaxis=dict(range=[0, 8])))
        chart_spec = company_quality_chart(figure, "甲公司", 7, unit="%").to_dict(validate=True)
        rows = [
            item
            for dataset in chart_spec.get("datasets", {}).values()
            for item in dataset
            if isinstance(item, dict)
        ]
        self.assertEqual(
            {item["名称"] for item in rows if "名称" in item},
            {"1", "2", "3", "4", "5", "6"},
        )
        self.assertIn("近三年平均综合投资收益率", {
            item["指标名称"] for item in rows if "指标名称" in item
        })
        self.assertIn("width - 34", str(chart_spec))

    def test_waterfall_preserves_signed_adjustments_and_reported_total(self):
        frame = pd.DataFrame([
            row("FINANCIAL_STATEMENT_NET_ASSETS", 100, scope="本季度末数"),
            row("NON_RECOGNIZED_ASSET_BOOK_VALUE", -10, scope="本季度末数"),
            row("POLICY_SURPLUS_CORE_T1", 5, scope="本季度末数"),
            row("CORE_T1_CAPITAL", 95, scope="本季度末数"),
        ])
        figure, missing, difference = core_waterfall_figure(frame, "2026Q2")
        self.assertIsNotNone(figure)
        self.assertEqual(difference, 0)
        self.assertIn("其他调整项目", missing)
        self.assertEqual(figure.data[0].y[0], 100)
        self.assertEqual(figure.data[0].y[-1], 95)

    def test_ancillary_pie_and_stack_do_not_impute_missing(self):
        frame = pd.DataFrame([row("ANC_T1_SUBORDINATED_TERM_DEBT", 20, scope="本季度末数")])
        pie, missing, negative = anc_pie_figure(frame, "2026Q2")
        stack, stack_missing = anc_stack_figure(frame, ["2026Q1", "2026Q2"])
        self.assertIsNotNone(pie)
        self.assertIn("其他附属一级资本", missing)
        self.assertEqual(negative, [])
        self.assertIsNone(stack.data[0].y[0])
        self.assertTrue(stack_missing)
        self.assertIsNone(value_for(frame, "OTHER_ANC_T1_CAPITAL", "2026Q2"))

    def test_ancillary_zero_only_section_is_undisclosed(self):
        from services.solvency_step7_quality import ANC_COMPONENTS
        frame = pd.DataFrame([
            row(code, 0, period="2025Q4", scope="本季度末数") for code, _ in ANC_COMPONENTS
        ] + [row("ANC_T1_CAPITAL", 100, period="2025Q4", scope="本季度末数")])
        pie, missing, negative = anc_pie_figure(frame, "2025Q4")
        stack, stack_missing = anc_stack_figure(frame, ["2025Q4"])
        self.assertIsNone(pie)
        self.assertEqual(missing, ["附属一级资本明细"])
        self.assertEqual(negative, [])
        self.assertEqual(stack_missing, ["2025Q4：附属一级资本明细"])
        self.assertTrue(all(value is None for trace in stack.data for value in trace.y))

        old_wide = pd.DataFrame([
            {**row("POLICY_SURPLUS_ANC_T1", 0, period="2025Q4", scope="本季度末数"),
             "来源类型": "外部数据集"},
            {**row("ANC_T1_CAPITAL", 100, period="2025Q4", scope="本季度末数"),
             "来源类型": "外部数据集"},
        ])
        completed = complete_external_capital_detail_zeros(old_wide)
        self.assertEqual(len(completed), len(old_wide))
        self.assertEqual(anc_pie_figure(completed, "2025Q4")[1], ["附属一级资本明细"])

    def test_capital_never_uses_prior_forecast_or_cumulative_columns(self):
        frame = pd.DataFrame([
            row("CORE_T1_CAPITAL", 100, scope="本季度末数"),
            row("CORE_T1_CAPITAL", 80, scope="上季度末数"),
            row("CORE_T1_CAPITAL", 60, scope="期初数"),
            row("CORE_T1_CAPITAL", 120, scope="下季度末预测数"),
            row("CORE_T1_CAPITAL", 777, scope="本年累计数"),
        ])
        self.assertEqual(capital_value(frame, "CORE_T1_CAPITAL", "2026Q2"), 100)
        self.assertIsNone(capital_value(frame.iloc[1:], "CORE_T1_CAPITAL", "2026Q2"))

    def test_cumulative_return_does_not_use_quarterly_or_three_year_value(self):
        frame = pd.DataFrame([
            row("INVESTMENT_RETURN", 1),
            row("INVESTMENT_RETURN", 2, scope="本年累计数"),
            row("INVESTMENT_RETURN", 8, scope="近三年平均"),
            row("COMPREHENSIVE_INVESTMENT_RETURN", 3),
        ])
        figure, missing, _ = combo_figure(frame, ["2026Q2"], "INVESTMENT_RETURN",
                                         "COMPREHENSIVE_INVESTMENT_RETURN", scope="cumulative")
        self.assertEqual(list(figure.data[0].y), [2])
        self.assertEqual(list(figure.data[1].y), [None])
        self.assertTrue(missing)

    def test_disclosed_zero_and_absence_and_zero_denominator_are_distinct(self):
        frame = pd.DataFrame([
            row("NEW_BUSINESS_VALUE", 0), row("SIGNED_PREMIUM", 0), row("RENEWAL_PREMIUM", 0),
        ])
        fig, missing, invalid = combo_figure(frame, ["2026Q2"], "NEW_BUSINESS_VALUE", "SIGNED_PREMIUM", derived_rate=True)
        self.assertEqual(list(fig.data[0].y), [0])
        self.assertEqual(list(fig.data[1].y), [None])
        self.assertTrue(has_values(fig))
        self.assertFalse(missing)
        self.assertIn("无法计算", invalid[0])

    def test_operating_ratio_bars_use_quarter_axis_and_formula_fallback(self):
        frame = pd.DataFrame([
            row("INSURANCE_CONTRACT_LIABILITY", 800),
            row("TOTAL_ASSETS", 1000),
            row("NET_ASSETS", 100),
            row("INSURANCE_REVENUE", 75),
            row("SIGNED_PREMIUM", 100),
        ])
        liabilities, missing, invalid = ratio_bar_figure(
            frame, ["2026Q2"], "INSURANCE_CONTRACT_LIABILITY_TO_TOTAL_LIABILITIES"
        )
        revenue, revenue_missing, revenue_invalid = ratio_bar_figure(
            frame, ["2026Q2"], "INSURANCE_REVENUE_TO_SIGNED_PREMIUM"
        )
        self.assertAlmostEqual(liabilities.data[0].y[0], 800 / 900 * 100)
        self.assertAlmostEqual(revenue.data[0].y[0], 75.0)
        self.assertEqual(list(liabilities.data[0].x), ["2026Q2"])
        self.assertFalse(missing or invalid or revenue_missing or revenue_invalid)

    def test_negative_pie_is_suppressed_and_signed_stack_is_kept(self):
        frame = pd.DataFrame([
            row("ANC_T1_SUBORDINATED_TERM_DEBT", 20),
            row("OTHER_ANC_T1_CAPITAL", -5),
        ])
        pie, _, negative = anc_pie_figure(frame, "2026Q2")
        stack, _ = anc_stack_figure(frame, ["2026Q2"])
        self.assertIsNone(pie)
        self.assertEqual(negative, ["其他附属一级资本"])
        self.assertEqual(list(stack.data[-1].y), [-5])

    def test_latest_period_missing_is_not_replaced_by_older_disclosure(self):
        frame = pd.DataFrame([row("ROE", 5, period="2026Q1")])
        values, missing = radar_values(frame, "2026Q2")
        self.assertEqual(values, [None] * 6)
        self.assertEqual(len(missing), 6)

    def test_explicit_undisclosed_status_overrides_placeholder_zero(self):
        frame = pd.DataFrame([{**row("ROE", 0), "披露状态": "未披露"}])
        self.assertIsNone(value_for(frame, "ROE", "2026Q2"))

    def test_legacy_point_scope_is_preserved_for_business_combos(self):
        frame = pd.DataFrame([
            row("SIGNED_PREMIUM", 100, scope="本季度末数"),
            row("NEW_BUSINESS_MARGIN", 5, scope="本季度末数"),
        ])
        fig, missing, _ = combo_figure(frame, ["2026Q2"], "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN")
        self.assertEqual(list(fig.data[0].y), [100])
        self.assertEqual(list(fig.data[1].y), [5])
        self.assertIn("本季度末数", fig.data[0].x[0])
        self.assertFalse(missing)

    def test_external_business_rows_from_old_converter_render_in_quarter_scope(self):
        frame = pd.DataFrame([
            {
                **row("SIGNED_PREMIUM", 3296829, scope="本季度末数"),
                "来源类型": "外部数据集", "来源工作表": "2026Q2",
            },
            {
                **row("NEW_BUSINESS_MARGIN", 0.0744, scope="本季度末数"),
                "来源类型": "外部数据集", "来源工作表": "2026Q2",
            },
        ])
        fig, missing, invalid = combo_figure(
            frame, ["2026Q2"], "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN",
            axes=[("2026Q2", "quarter")],
        )
        self.assertEqual(list(fig.data[0].y), [3296829])
        self.assertAlmostEqual(float(fig.data[1].y[0]), 7.44)
        self.assertEqual(missing, [])
        self.assertEqual(invalid, [])

    def test_business_combo_bar_labels_are_black_and_inside_bars(self):
        cases = (
            ("SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN", False),
            ("NEW_BUSINESS_VALUE", "SIGNED_PREMIUM", True),
        )
        for bar_code, line_code, derived in cases:
            with self.subTest(bar_code=bar_code):
                frame = pd.DataFrame([
                    row(bar_code, 100),
                    row(line_code, 10),
                ])
                figure, _, _ = combo_figure(
                    frame, ["2026Q2"], bar_code, line_code, derived_rate=derived,
                )
                figure.update_yaxes(range=[0, 120], secondary_y=False)
                figure.update_yaxes(range=[0, 20], secondary_y=True)
                spec = company_quality_chart(
                    figure, "甲公司", 7, unit="十亿元",
                ).to_dict(validate=True)
                bar_labels = spec["layer"][0]["layer"][1]
                self.assertEqual(bar_labels["mark"]["color"], "#000000")
                dataset = spec["datasets"][spec["data"]["name"]]
                self.assertEqual(dataset[0]["柱标签位置"], dataset[0]["柱值"] / 2)

    def test_legacy_point_return_cannot_be_assumed_to_be_quarterly(self):
        frame = pd.DataFrame([row("INVESTMENT_RETURN", 5, scope="本季度末数")])
        values, _ = radar_values(frame, "2026Q2")
        self.assertIsNone(values[2])

    def test_legacy_operating_metric_point_scope_is_recovered_for_radar(self):
        frame = pd.DataFrame([
            {**row("INVESTMENT_RETURN", 5, scope="本季度末数"), "来源工作表": "主要经营指标"},
            {**row("COMPREHENSIVE_INVESTMENT_RETURN", 6, scope="本季度末数"), "来源工作表": "OPERATING_METRICS"},
        ])
        values, missing = radar_values(frame, "2026Q2")
        self.assertEqual(values[2:4], [5, 6])
        self.assertNotIn("投资收益率（当季数）", missing)
        self.assertNotIn("综合投资收益率（当季数）", missing)

    def test_render_all_modules_keeps_undisclosed_company_in_legend_and_notes(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import pandas as pd
from step7_solvency import _render_quality_and_capital
from services.solvency_step7_chart_plans import SPECIAL_CHART_PLANS, QUALITY_AND_CAPITAL
rows = [{"公司":"未披露公司", "报告期":"2026Q2", "期间口径":"本季度末数",
         "指标编码":"CORE_T1_CAPITAL", "数值":None, "单位":"万元", "披露状态":"未披露"}]
for index, (name, plan) in enumerate(SPECIAL_CHART_PLANS.items()):
    if plan.kind == QUALITY_AND_CAPITAL:
        _render_quality_and_capital(pd.DataFrame(rows), name, periods=["2026Q2"],
            unit_mode="亿元", highlight_company="无", key_prefix=str(index),
            selected_companies=["未披露公司"])
'''
        at = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(at.exception)
        legends = [item.value for item in at.markdown if "未披露：未披露公司" in item.value or "未披露公司（未披露）" in item.value]
        self.assertEqual(len(legends), 9)
        self.assertEqual(len([item for item in at.caption if item.value.startswith("注：未披露——")]), 10)
        self.assertEqual(len(at.get("plotly_chart")), 0)
        captions = [item.value for item in at.caption]
        self.assertTrue(any("6 核心一级资本" in value for value in captions))
        self.assertFalse(any("终点采用披露的核心一级资本" in value for value in captions))
        self.assertFalse(any("上述公司有全部或部分指标未披露" in value for value in captions))

    def test_business_quality_charts_fix_scope_to_quarter_without_selector(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import pandas as pd
from step7_solvency import _render_quality_and_capital
rows = []
values = {
    "SIGNED_PREMIUM": (100, 240),
    "NEW_BUSINESS_MARGIN": (5, 6),
    "NEW_BUSINESS_VALUE": (20, 50),
    "RENEWAL_PREMIUM": (60, 140),
    "SURRENDER_RATE": (3, 4),
}
for code, (quarter, cumulative) in values.items():
    for scope, value in (("本季度数", quarter), ("本年累计数", cumulative)):
        rows.append({"公司":"甲公司", "报告期":"2026Q2", "期间口径":scope,
                     "指标编码":code, "指标名称":code, "数值":value,
                     "单位":"%" if code in {"NEW_BUSINESS_MARGIN", "SURRENDER_RATE"} else "万元",
                     "披露状态":"已披露"})
data = pd.DataFrame(rows)
for index, chart in enumerate((
    "签单保费与新业务利润率",
    "新业务价值与新业务价值率",
    "综合退保率",
)):
    _render_quality_and_capital(data, chart, periods=["2026Q2"],
        unit_mode="亿元", highlight_company="无", key_prefix=f"fixed_{index}",
        selected_companies=["甲公司"])
'''
        at = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(len(at.selectbox), 0)
        captions = [item.value for item in at.caption]
        self.assertTrue(any("期间口径：当季" in item for item in captions))

    def test_render_old_external_capital_package_does_not_label_zeros_undisclosed(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import pandas as pd
from step7_solvency import _render_quality_and_capital
def item(code, value):
    return {"公司":"甲公司", "报告期":"2025Q4", "期间口径":"本季度末数",
            "指标编码":code, "数值":value, "单位":"万元", "来源类型":"外部数据集"}
data = pd.DataFrame([
    item("NET_ASSETS",100), item("NON_RECOGNIZED_ASSET_BOOK_VALUE",-10),
    item("CORE_T1_CAPITAL",90), item("ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS",20),
    item("ANC_T1_CAPITAL",20),
])
for index, chart in enumerate(("核心一级资本明细", "附属一级资本明细")):
    _render_quality_and_capital(data, chart, periods=["2025Q4"],
        unit_mode="亿元", highlight_company="无", key_prefix=str(index),
        selected_companies=["甲公司"])
'''
        at = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(at.exception)
        self.assertFalse(any(item.value.startswith("注：未披露——") for item in at.caption))
        self.assertFalse(any("未披露：甲公司" in item.value for item in at.markdown))

    def test_render_zero_only_ancillary_details_as_undisclosed(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import pandas as pd
from step7_solvency import _render_quality_and_capital
totals = {"中邮人寿":24.13, "中银三星":4.99, "建信人寿":14.42, "招商信诺":11.49}
rows = []
for company, total in totals.items():
    for period in ("2024Q4", "2025Q2", "2025Q4"):
        for code, value in (("POLICY_SURPLUS_ANC_T1",0), ("ANC_T1_CAPITAL",total)):
            rows.append({"公司":company, "报告期":period, "期间口径":"本季度末数",
                         "指标编码":code, "数值":value, "单位":"十亿元", "来源类型":"外部数据集"})
_render_quality_and_capital(pd.DataFrame(rows), "附属一级资本明细", periods=["2024Q4", "2025Q2", "2025Q4"],
    unit_mode="十亿元", highlight_company="无", key_prefix="anc_zero",
    selected_companies=list(totals))
'''
        at = AppTest.from_string(source).run(timeout=30)
        self.assertFalse(at.exception)
        captions = [item.value for item in at.caption]
        notes = [value for value in captions if value.startswith("注：未披露——")]
        self.assertEqual(notes, [
            "注：未披露——中邮人寿、中银三星、建信人寿、招商信诺：2025Q4均未披露附属一级资本明细。"
        ])
        all_text = captions + [item.value for item in at.markdown]
        self.assertFalse(any("最近三期构成堆叠柱图" in value for value in all_text))
        self.assertFalse(any("均披露为 0" in value or "七类明细合计与披露" in value
                             for value in captions))

    def test_combo_period_ticks_only_show_quarters_and_hover_keeps_scope(self):
        frame = pd.DataFrame([row("SIGNED_PREMIUM", 100), row("NEW_BUSINESS_MARGIN", 5)])
        fig, _, _ = combo_figure(frame, ["2026Q2"], "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN")
        self.assertEqual(list(fig.layout.xaxis.ticktext), ["2026Q2"])
        self.assertIn("当季", fig.data[0].x[0])

    def test_surrender_companies_share_one_chart_and_missing_quarter_is_a_gap(self):
        periods = ["2025Q4", "2026Q1", "2026Q2"]
        frame = pd.DataFrame([row("SURRENDER_RATE", 1, period=periods[0]), row("SURRENDER_RATE", 2)])
        fig, missing = surrender_comparison_figure(frame, ["甲公司", "乙公司"], periods,
                     {"甲公司":"#00338D", "乙公司":"#00B8F5"}, scope="quarter")
        self.assertEqual(len(fig.data), 2)
        self.assertEqual(list(fig.data[0].x), periods)
        self.assertEqual(list(fig.data[0].y), [1, None, 2])
        self.assertFalse(fig.data[0].connectgaps)
        self.assertEqual(missing["乙公司"], periods)

    def test_tiny_negative_asset_share_stays_exact_and_is_visible_below_zero(self):
        from step7_solvency import _convert_unit
        from services.solvency_step7_charts import build_component_stack_chart, component_stack_proportion_axis_domain
        specs = [("INVESTMENT_ASSETS", "投资资产", "#00B8F5"),
                 ("OTHER_RECOGNIZED_ASSETS", "其他认可资产", "#F1C44D")]
        frame = pd.DataFrame([
            {**row("OTHER_RECOGNIZED_ASSETS", -23248.56967, "2025Q3"), "单位": "万元"},
            {**row("RECOGNIZED_ASSETS", 1598.336247, "2025Q3"), "单位": "亿元"},
            {**row("INVESTMENT_ASSETS", 1342.6, "2025Q3"), "单位": "亿元"},
        ])
        frame["公司"] = "中银三星"
        frame = _convert_unit(frame, "亿元")
        domain = component_stack_proportion_axis_domain(frame, specs, ["2025Q3"],
            label_denominator_code="RECOGNIZED_ASSETS", show_small_negative_labels=True)
        spec = build_component_stack_chart(frame, specs, ["2025Q3"], "认可资产构成",
            label_denominator_code="RECOGNIZED_ASSETS", plot_proportions=True,
            shared_y_domain=domain, show_small_negative_labels=True).to_dict()
        rows = {r["指标编码"]: r for r in spec["datasets"][spec["data"]["name"]]}
        negative = rows["OTHER_RECOGNIZED_ASSETS"]
        expected = -23248.56967 / 10000 / 1598.336247
        self.assertAlmostEqual(negative["绘图占比"], expected)
        self.assertEqual(negative["堆叠起点"], 0)
        self.assertAlmostEqual(negative["堆叠终点"], expected)
        self.assertEqual(negative["占比标签"], "-0.15%")
        self.assertLess(negative["标签位置"], negative["堆叠终点"])
        self.assertLess(domain[0], negative["标签位置"])
        self.assertTrue(negative["小额负值注释"])
        self.assertIsNone(spec["layer"][0]["encoding"]["y"]["stack"])
        self.assertEqual(spec["layer"][-1]["mark"]["type"], "rule")

    def test_recognized_assets_stack_has_separate_positive_and_negative_bounds(self):
        from services.solvency_step7_charts import build_component_stack_chart, component_stack_proportion_axis_domain
        specs = [("INVESTMENT_ASSETS", "投资资产", "#00B8F5"),
                 ("REINSURANCE_ASSETS", "再保险资产", "#269924"),
                 ("OTHER_RECOGNIZED_ASSETS", "其他认可资产", "#F1C44D")]
        frame = pd.DataFrame([{**row(code, value), "单位":"亿元"} for code, value in
            [("INVESTMENT_ASSETS",120),("REINSURANCE_ASSETS",-5),("OTHER_RECOGNIZED_ASSETS",-15),("RECOGNIZED_ASSETS",100)]])
        domain = component_stack_proportion_axis_domain(frame,specs,["2026Q2"],label_denominator_code="RECOGNIZED_ASSETS")
        spec = build_component_stack_chart(frame,specs,["2026Q2"],"认可资产构成",
               label_denominator_code="RECOGNIZED_ASSETS",plot_proportions=True,shared_y_domain=domain).to_dict()
        rows = {r["指标编码"]: r for r in spec["datasets"][spec["data"]["name"]]}
        self.assertAlmostEqual(rows["INVESTMENT_ASSETS"]["堆叠终点"],1.2)
        self.assertAlmostEqual(rows["REINSURANCE_ASSETS"]["堆叠起点"],0)
        self.assertAlmostEqual(rows["OTHER_RECOGNIZED_ASSETS"]["堆叠终点"],-.2)
        self.assertEqual(rows["OTHER_RECOGNIZED_ASSETS"]["占比标签"],"-15%")
        self.assertLess(domain[0],-.2)
        self.assertIsNone(spec["layer"][0]["encoding"]["y"]["stack"])


if __name__ == "__main__":
    unittest.main()
