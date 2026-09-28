from __future__ import annotations

import unittest
from pathlib import Path
import pandas as pd

from dashboard_components import (
    _format_key_solvency_overview_display,
    build_dashboard_header_html,
    build_report_back_cover_html,
    build_report_cover_html,
    build_key_solvency_overview_html,
    build_key_solvency_overview_table,
    calculate_industry_overview,
    profile_platform_copy,
)


ROOT = Path(__file__).resolve().parents[1]


def dashboard_rows() -> pd.DataFrame:
    rows = []
    companies = (
        ("甲人寿", "寿险", "大型公司", 160.0, 140.0),
        ("乙人寿", "寿险", "中型公司", 130.0, 120.0),
        ("丙健康", "健康险", "养老健康", 90.0, 80.0),
        ("行业合计", "行业合计", "全行业", 145.0, 125.0),
    )
    for company, company_type, peer_group, combined, core in companies:
        code = "INDUSTRY_LIFE_TOTAL" if company == "行业合计" else f"CODE_{company}"
        for metric_code, metric_name, value in (
            ("COMBINED_SOLVENCY_RATIO", "综合偿付能力充足率", combined),
            ("CORE_SOLVENCY_RATIO", "核心偿付能力充足率", core),
        ):
            rows.append({
                "公司": company,
                "公司类型": company_type,
                "公司统一编码": code,
                "同业分类": peer_group,
                "报告期": "2025Q4",
                "期间口径": "本季度末数",
                "指标编码": metric_code,
                "指标名称": metric_name,
                "数值": value,
                "单位": "%",
            })
    return pd.DataFrame(rows)


class DashboardComponentTests(unittest.TestCase):
    def test_key_solvency_overview_uses_prior_year_matching_period_and_keeps_missing_blank(self):
        rows = []
        values_by_period = {
            "2024Q4": {
                "CORE_SOLVENCY_RATIO": 120.0,
                "COMBINED_SOLVENCY_RATIO": 180.0,
                "ACTUAL_CAPITAL": 100.0,
                "MINIMUM_CAPITAL": 60.0,
                "RECOGNIZED_LIABILITIES": 300.0,
                "POLICY_SURPLUS_CORE_T1": 10.0,
                "POLICY_SURPLUS_CORE_T2": 2.0,
                "POLICY_SURPLUS_ANC_T1": 1.0,
                "POLICY_SURPLUS_ANC_T2": 1.0,
            },
            "2025Q4": {
                "CORE_SOLVENCY_RATIO": 130.0,
                "COMBINED_SOLVENCY_RATIO": 195.0,
                "ACTUAL_CAPITAL": 110.0,
                "MINIMUM_CAPITAL": 66.0,
                "RECOGNIZED_LIABILITIES": 330.0,
                "POLICY_SURPLUS_CORE_T1": 11.0,
                "POLICY_SURPLUS_CORE_T2": 2.2,
                "POLICY_SURPLUS_ANC_T1": 1.1,
                "POLICY_SURPLUS_ANC_T2": 1.1,
                "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL": 0.40,
                "MARKET_RISK_TO_QUANT_CAPITAL": 0.25,
                "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL": 0.30,
            },
            "2025Q2": {
                "CORE_SOLVENCY_RATIO": 999.0,
                "ACTUAL_CAPITAL": 999.0,
            },
        }
        for period, metrics in values_by_period.items():
            for code, value in metrics.items():
                rows.append({
                    "公司": "甲人寿",
                    "公司类型": "寿险",
                    "公司统一编码": "A",
                    "同业分类": "大型公司",
                    "报告期": period,
                    "期间口径": "期末数",
                    "指标编码": code,
                    "数值": value,
                })
        table, latest, prior = build_key_solvency_overview_table(pd.DataFrame(rows))
        self.assertEqual((latest, prior), ("2025Q4", "2024Q4"))
        self.assertEqual(
            table.columns.tolist(),
            [
                "公司名称",
                "核心资本充足率2025Q4",
                "核心资本充足率2024Q4",
                "综合资本充足率2025Q4",
                "综合资本充足率2024Q4",
                "实际资本2025Q4",
                "实际资本2024Q4",
                "保单未来盈余2025Q4",
                "保单未来盈余2024Q4",
                "保单未来盈余/核心资本比例 2025Q4",
                "市场风险占比 2025Q4",
                "保险风险占比 2025Q4",
                "认可负债余额2025Q4",
                "认可负债余额2024Q4",
            ],
        )
        self.assertAlmostEqual(table.iloc[0, 1], 130.0)
        self.assertAlmostEqual(table.iloc[0, 2], 120.0)
        self.assertAlmostEqual(table.iloc[0, 5], 110.0)
        self.assertAlmostEqual(table.iloc[0, 7], 15.4)
        self.assertAlmostEqual(table.iloc[0, 8], 14.0)
        self.assertAlmostEqual(table.iloc[0, 9], 0.40)
        self.assertAlmostEqual(table.iloc[0, 10], 0.25)
        self.assertAlmostEqual(table.iloc[0, 11], 0.30)
        self.assertAlmostEqual(table.iloc[0, 12], 330.0)
        self.assertAlmostEqual(table.iloc[0, 13], 300.0)

    def test_key_overview_has_no_duplicate_internal_heading(self):
        source = (ROOT / "dashboard_components.py").read_text(encoding="utf-8")
        self.assertNotIn('st.markdown("### :material/table_chart: 关键偿付数据概览")', source)

    def test_key_solvency_overview_sums_available_policy_surplus_layers(self):
        rows = []
        values_by_company = {
            "仅核心一级": {
                "POLICY_SURPLUS_CORE_T1": 337_959.0,
            },
            "披露三层": {
                "POLICY_SURPLUS_CORE_T1": 263_219.16,
                "POLICY_SURPLUS_CORE_T2": 8_163.89,
                "POLICY_SURPLUS_ANC_T1": 354_759.48,
            },
        }
        for company, policy_values in values_by_company.items():
            for period in ("2025Q2", "2026Q2"):
                metrics = {
                    "CORE_SOLVENCY_RATIO": 120.0,
                    "COMBINED_SOLVENCY_RATIO": 180.0,
                    "ACTUAL_CAPITAL": 1_000_000.0,
                    "RECOGNIZED_LIABILITIES": 5_000_000.0,
                    **policy_values,
                }
                for code, metric_value in metrics.items():
                    rows.append({
                        "公司": company,
                        "公司类型": "寿险",
                        "报告期": period,
                        "期间口径": "期末数",
                        "指标编码": code,
                        "数值": metric_value,
                    })

        table, latest, prior = build_key_solvency_overview_table(pd.DataFrame(rows))

        self.assertEqual((latest, prior), ("2026Q2", "2025Q2"))
        by_company = table.set_index("公司名称")
        self.assertAlmostEqual(by_company.loc["仅核心一级", "保单未来盈余2026Q2"], 337_959.0)
        self.assertAlmostEqual(by_company.loc["仅核心一级", "保单未来盈余2025Q2"], 337_959.0)
        expected_three_layers = 263_219.16 + 8_163.89 + 354_759.48
        self.assertAlmostEqual(by_company.loc["披露三层", "保单未来盈余2026Q2"], expected_three_layers)
        self.assertAlmostEqual(by_company.loc["披露三层", "保单未来盈余2025Q2"], expected_three_layers)

    def test_key_overview_merges_registered_legal_and_short_names_across_years(self):
        rows = [
            {
                "公司": company,
                "公司类型": "寿险",
                "报告期": period,
                "期间口径": "本季度末数",
                "指标编码": code,
                "数值": value,
            }
            for company, period, code, value in (
                ("工银安盛人寿保险有限公司", "2025Q1", "CORE_SOLVENCY_RATIO", 184.0),
                ("工银安盛", "2026Q1", "CORE_SOLVENCY_RATIO", 128.0),
                ("工银安盛人寿保险有限公司", "2025Q1", "COMBINED_SOLVENCY_RATIO", 248.0),
                ("工银安盛", "2026Q1", "COMBINED_SOLVENCY_RATIO", 187.0),
            )
        ]
        table, latest, prior = build_key_solvency_overview_table(pd.DataFrame(rows))
        self.assertEqual((latest, prior), ("2026Q1", "2025Q1"))
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["公司名称"], "工银安盛")
        self.assertEqual(table.iloc[0]["核心资本充足率2026Q1"], 128.0)
        self.assertEqual(table.iloc[0]["核心资本充足率2025Q1"], 184.0)
        self.assertEqual(calculate_industry_overview(pd.DataFrame(rows)).company_count, 1)

    def test_key_solvency_overview_html_matches_annual_table_style(self):
        display = pd.DataFrame([
            {
                "公司名称": "甲人寿<script>",
                "核心资本充足率2025Q4": "130.0%",
                "核心资本充足率2024Q4": "120.0%",
                "市场风险占比 2025Q4": "未披露",
            },
            {
                "公司名称": "乙人寿",
                "核心资本充足率2025Q4": "128.0%",
                "核心资本充足率2024Q4": "130.0%",
                "市场风险占比 2025Q4": "12.0%",
            },
        ])
        result = build_key_solvency_overview_html(
            display,
            latest_period="2025Q4",
            prior_period="2024Q4",
        )
        self.assertIn("font-family:sans-serif;font-size:10px", result)
        self.assertIn("background-color:#00338D;color:white", result)
        self.assertNotIn(">最新报告期</th>", result)
        self.assertNotIn(">去年同期</th>", result)
        self.assertIn("核心资本充足率<br>2025Q4", result)
        self.assertIn("核心资本充足率<br>2024Q4", result)
        self.assertNotIn("overflow-x:auto", result)
        self.assertNotIn("min-width:1850px", result)
        self.assertIn("<colgroup>", result)
        self.assertIn("table-layout:fixed", result)
        self.assertIn("@media print", result)
        self.assertIn("min-width:0!important", result)
        self.assertIn("white-space:normal!important", result)
        self.assertEqual(result.count("<thead><tr"), 1)
        self.assertIn("background-color:#CDCDCD", result)
        self.assertIn("background-color:#F8F9FA", result)
        self.assertIn("white-space:nowrap;word-break:keep-all", result)
        self.assertIn("甲人寿&lt;script&gt;", result)
        self.assertNotIn("甲人寿<script>", result)

        tracked = build_key_solvency_overview_html(display, "乙人寿")
        self.assertIn("background-color:rgba(0,51,141,0.03)", tracked)
        self.assertIn("border-left:1.5px solid #00338D", tracked)
        self.assertIn("border-right:1.5px solid #00338D", tracked)
        self.assertIn("font-weight:bold", tracked)

    def test_key_solvency_overview_formats_rates_and_amounts_by_metric_type(self):
        table = pd.DataFrame([{
            "公司名称": "甲人寿",
            "核心资本充足率2025Q4": 130.0,
            "综合资本充足率2024Q4": 180.0,
            "实际资本2025Q4": 110.0,
            "保单未来盈余/核心资本比例 2025Q4": 0.40,
            "市场风险占比 2025Q4": 0.25,
        }])
        display = _format_key_solvency_overview_display(table)
        self.assertEqual(display.iloc[0, 1], "130.0%")
        self.assertEqual(display.iloc[0, 2], "180.0%")
        self.assertEqual(display.iloc[0, 3], "110.00")
        self.assertEqual(display.iloc[0, 4], "40.0%")
        self.assertEqual(display.iloc[0, 5], "25.0%")
        result = build_key_solvency_overview_html(
            display,
            latest_period="2025Q4",
            prior_period="2024Q4",
        )
        self.assertIn("核心资本充足率<br>2025Q4", result)
        self.assertIn("综合资本充足率<br>2024Q4", result)
        self.assertIn(">130.0%</td>", result)
        self.assertIn(">110.00</td>", result)

    def test_profile_copy_changes_with_life_nonlife_and_annual_profiles(self):
        self.assertEqual(
            profile_platform_copy("LIFE_SOLVENCY", "寿险偿付能力季度报告", "QUARTERLY"),
            ("人身险公司偿付能力信息分享", "中国人身险公司偿付能力季度报告数据库"),
        )
        self.assertEqual(
            profile_platform_copy("NON_LIFE_SOLVENCY", "财险偿付能力季度报告", "QUARTERLY")[0],
            "财产险公司偿付能力信息分享",
        )
        self.assertEqual(
            profile_platform_copy("LIFE_ANNUAL", "寿险年度报告", "ANNUAL")[0],
            "人身险公司年度报告信息分享",
        )

    def test_industry_overview_excludes_industry_total_and_calculates_bands(self):
        overview = calculate_industry_overview(dashboard_rows())
        self.assertEqual(overview.report_period, "2025Q4")
        self.assertEqual(overview.period_scope, "本季度末数")
        self.assertEqual(overview.company_count, 3)
        self.assertEqual(overview.combined_median, 130.0)
        self.assertEqual(overview.core_median, 120.0)
        self.assertEqual(overview.sufficient_count, 1)
        self.assertEqual(overview.warning_count, 1)
        self.assertEqual(overview.insufficient_count, 1)

    def test_header_html_uses_background_image_and_escapes_dynamic_text(self):
        result = build_dashboard_header_html(
            title="人身险公司偿付能力信息分享",
            database_name="中国人身险公司偿付能力季度报告数据库",
            report_period="2025Q4",
            period_scope="本季度末数",
            company_count=71,
            target="甲人寿<script>",
            image_path=ROOT / "picture" / "bg_header.png",
        )
        self.assertIn("data:image/png;base64,", result)
        self.assertIn("background-size: 100% auto", result)
        self.assertIn("color: #ffffff !important", result)
        self.assertIn("71 家公司", result)
        self.assertIn("甲人寿&lt;script&gt;", result)
        self.assertNotIn("甲人寿<script>", result)

    def test_report_cover_and_back_use_local_annual_platform_images(self):
        cover = build_report_cover_html(
            title="偿付能力公司报告<script>",
            subtitle="2025Q4 · 三峡人寿",
            date_text="2026年8月",
            image_path=ROOT / "picture" / "标题页.png",
        )
        back = build_report_back_cover_html(ROOT / "picture" / "封底页.png")
        self.assertIn("data:image/png;base64,", cover)
        self.assertIn("solvency-print-cover--front", cover)
        self.assertIn("偿付能力公司报告&lt;script&gt;", cover)
        self.assertNotIn("偿付能力公司报告<script>", cover)
        self.assertIn("data:image/png;base64,", back)
        self.assertIn("solvency-print-cover--back", back)

    def test_major_financing_heading_omits_sequence_and_period(self):
        component_source = (ROOT / "dashboard_components.py").read_text(encoding="utf-8")
        self.assertIn(
            'st.markdown("### :material/assignment: 重大融资信息统计")',
            component_source,
        )
        self.assertNotIn("02 · 重大融资信息统计 ·", component_source)

    def test_print_modes_have_distinct_final_page_rules(self):
        component_source = (ROOT / "dashboard_components.py").read_text(encoding="utf-8")
        report_source = (ROOT / "step7_solvency.py").read_text(encoding="utf-8")
        self.assertIn("size: A4 portrait; margin: 10mm", component_source)
        self.assertIn("size: 338.67mm 190.5mm; margin: 0", component_source)
        self.assertIn("导出竖版 A4 PDF</button>", component_source)
        self.assertIn("竖版 A4 与横版 16:9 均导出封面、正文和封底", component_source)
        self.assertIn("doc.body.appendChild(style)", component_source)
        self.assertIn('view.dispatchEvent(new view.Event("resize"))', component_source)
        self.assertIn("await delay(360)", component_source)
        self.assertIn('[data-stale="true"]', component_source)
        self.assertIn("waitForStablePage", component_source)
        self.assertIn("consecutiveStableChecks >= 3", component_source)
        self.assertIn('doc.querySelector(\'[data-testid="stMain"]\')', component_source)
        self.assertIn("element.getClientRects().length > 0", component_source)
        self.assertIn('style.display !== "none"', component_source)
        self.assertIn('style.visibility !== "hidden"', component_source)
        self.assertNotIn(
            'doc.querySelector(\'[data-testid="stElementContainer"][data-stale="true"]\')',
            component_source,
        )
        self.assertIn("view.__solvencyDashboardPrintJob", component_source)
        self.assertIn("return () => {", component_source)
        self.assertIn('view.removeEventListener("afterprint", job.afterPrint)', component_source)
        self.assertIn("solvency-print-mode-portrait", component_source)
        self.assertIn("solvency-print-mode-widescreen", component_source)
        self.assertNotIn(
            "html.solvency-print-mode-portrait .solvency-print-cover {display:none!important;}",
            report_source,
        )

    def test_report_pages_do_not_override_selected_paper_size(self):
        for file_name in ("step7_solvency.py", "step8_solvency.py"):
            source = (ROOT / file_name).read_text(encoding="utf-8")
            self.assertNotIn("@page {size:338.67mm 190.5mm", source)
            self.assertIn("width:338.67mm!important; height:190.5mm!important", source)
            self.assertIn("object-fit:contain!important", source)


if __name__ == "__main__":
    unittest.main()
