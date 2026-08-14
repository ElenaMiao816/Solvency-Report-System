from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from dashboard_components import (
    PEER_CLASSIFICATION_STANDARD,
    build_dashboard_header_html,
    build_report_back_cover_html,
    build_report_cover_html,
    build_peer_classification_table,
    calculate_industry_overview,
    profile_platform_copy,
    render_peer_classification,
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

    def test_peer_classification_table_uses_existing_peer_labels(self):
        table = build_peer_classification_table(dashboard_rows())
        self.assertEqual(table.columns.tolist(), ["序号", "大型公司", "中型公司", "养老健康"])
        self.assertEqual(table.iloc[0]["大型公司"], "甲人寿")
        self.assertEqual(table.iloc[0]["中型公司"], "乙人寿")
        self.assertEqual(table.iloc[0]["养老健康"], "丙健康")
        self.assertIn("5,000 亿元", PEER_CLASSIFICATION_STANDARD)

    def test_peer_classification_renders_full_static_table_without_scrolling(self):
        with patch("dashboard_components.st") as streamlit:
            render_peer_classification(dashboard_rows())

        streamlit.table.assert_called_once()
        _, kwargs = streamlit.table.call_args
        self.assertEqual(kwargs["height"], "content")
        self.assertEqual(kwargs["width"], "stretch")
        self.assertTrue(kwargs["hide_index"])
        streamlit.dataframe.assert_not_called()

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

    def test_print_modes_have_distinct_final_page_rules(self):
        component_source = (ROOT / "dashboard_components.py").read_text(encoding="utf-8")
        self.assertIn("size: A4 portrait; margin: 10mm", component_source)
        self.assertIn("size: 338.67mm 190.5mm; margin: 8mm 12mm", component_source)
        self.assertIn("@page :first { margin: 0; }", component_source)
        self.assertIn("@page :last { margin: 0; }", component_source)
        self.assertIn("doc.body.appendChild(style)", component_source)
        self.assertIn("solvency-print-mode-portrait", component_source)
        self.assertIn("solvency-print-mode-widescreen", component_source)

    def test_report_pages_do_not_override_selected_paper_size(self):
        for file_name in ("step7_solvency.py", "step8_solvency.py"):
            source = (ROOT / file_name).read_text(encoding="utf-8")
            self.assertNotIn("@page {size:338.67mm 190.5mm", source)
            self.assertIn("width:338.67mm!important; height:190.5mm!important", source)
            self.assertIn("object-fit:contain!important", source)


if __name__ == "__main__":
    unittest.main()
