from __future__ import annotations

import io
import unittest

import pandas as pd

from services.solvency_step6_analysis import (
    COMPANY_REPORT,
    INDUSTRY_REPORT,
    NO_COMPANY_TYPE_FILTER,
    build_comparison_chart,
    companies_for_quick_selection,
    default_companies,
    filter_analysis_frame,
    format_chart_value,
    navigation_scope_frame,
    nonblank_values,
    prepare_analysis_frame,
    report_scope_frame,
    sort_report_periods,
    visualization_metric_frame,
)
from services.solvency_dataset_adapter import read_standard_workbook


def sample_rows() -> pd.DataFrame:
    rows = []
    for company, company_type, code, peer_group in (
        ("甲人寿", "寿险", "COMPANY_A", "大型公司"),
        ("乙健康", "健康险", "COMPANY_B", "养老健康"),
        ("行业合计", "行业合计", "INDUSTRY_LIFE_TOTAL", "全行业"),
    ):
        for period, value in (("2024Q4", 100.0), ("2025Q4", 120.0)):
            rows.append({
                "公司": company,
                "标准公司名称": company,
                "公司统一编码": code,
                "公司类型": company_type,
                "同业分类": peer_group,
                "报告类型": "LIFE_SOLVENCY",
                "报告期": period,
                "一级模块": "偿付能力",
                "二级模块": "资本",
                "指标编码": "ACTUAL_CAPITAL",
                "指标名称": "实际资本",
                "期间口径": "本季度末数",
                "数值": value,
                "单位": "万元",
            })
    return pd.DataFrame(rows)


class Step6AnalysisTests(unittest.TestCase):
    def test_chart_value_formatting_uses_metric_units(self):
        self.assertEqual(format_chart_value(5977438, "万元", "金额"), "5,977,438.00")
        self.assertEqual(format_chart_value(-6428230, "万元", "金额"), "(6,428,230.00)")
        self.assertEqual(format_chart_value(193.64, "%", "百分比"), "193.6%")
        self.assertEqual(format_chart_value(0.169, "倍", "比率"), "0.17倍")
        self.assertEqual(format_chart_value(1.2345, "‰", "百分比", 2), "1.23‰")

    def test_prepare_analysis_frame_filters_profile_and_non_numeric_rows(self):
        frame = sample_rows()
        blank = frame.iloc[[0]].copy()
        blank["报告类型"] = ""
        other = frame.iloc[[1]].copy()
        other["报告类型"] = "NON_LIFE_SOLVENCY"
        invalid = frame.iloc[[2]].copy()
        invalid["数值"] = "公式错误"
        result, skipped = prepare_analysis_frame(
            pd.concat([blank, other, invalid], ignore_index=True),
            "LIFE_SOLVENCY",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["报告类型"], "LIFE_SOLVENCY")
        self.assertEqual(skipped, ("NON_LIFE_SOLVENCY",))

    def test_prepare_analysis_frame_preserves_policy_surplus_unavailable_reason(self):
        row = sample_rows().iloc[[0]].copy()
        row["指标编码"] = "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"
        row["指标名称"] = "计入核心资本的保单未来盈余/核心资本的比例"
        row["数值"] = None
        row["披露状态"] = "无法计算"
        row["备注"] = "无法计算：计入核心二级资本的保单未来盈余未披露"

        result, _ = prepare_analysis_frame(row, "LIFE_SOLVENCY")

        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["数值"]))
        self.assertEqual(result.iloc[0]["披露状态"], "无法计算")

    def test_prepare_analysis_frame_repairs_old_full_short_company_codes(self):
        frame = sample_rows().iloc[[0, 1]].copy()
        frame["公司"] = ["工银安盛人寿保险有限公司", "工银安盛"]
        frame["标准公司名称"] = frame["公司"]
        frame["公司统一编码"] = ["OLD_FULL", "OLD_SHORT"]
        result, _ = prepare_analysis_frame(frame, "LIFE_SOLVENCY")
        self.assertEqual(result["公司"].tolist(), ["工银安盛", "工银安盛"])
        self.assertEqual(result["公司统一编码"].nunique(), 1)
        self.assertEqual(result["报告期"].tolist(), ["2024Q4", "2025Q4"])

    def test_company_and_industry_report_scopes(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        company = report_scope_frame(frame, COMPANY_REPORT)
        industry = report_scope_frame(frame, INDUSTRY_REPORT)
        self.assertNotIn("行业合计", company["公司"].tolist())
        self.assertIn("行业合计", industry["公司"].tolist())
        self.assertEqual(len(industry), len(frame))

    def test_filters_module_company_time_metric_and_period_scope(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        result = filter_analysis_frame(
            frame,
            level_one="关键偿付数据概览",
            level_two="资本充足率",
            company_types=["寿险"],
            peer_groups=["大型公司"],
            companies=["甲人寿"],
            periods=["2025Q4"],
            metric_code="ACTUAL_CAPITAL",
            period_scope="本季度末数",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["公司"], "甲人寿")
        self.assertEqual(result.iloc[0]["报告期"], "2025Q4")

    def test_navigation_scope_limits_metric_options_without_page_navigation(self):
        frame = pd.DataFrame(
            [
                {"一级模块": "实际资本", "二级模块": "行业资本分级", "指标编码": "A"},
                {"一级模块": "实际资本", "二级模块": "保单未来盈余", "指标编码": "B"},
                {"一级模块": "最低资本", "二级模块": "保险风险", "指标编码": "C"},
            ]
        )
        selected = navigation_scope_frame(frame, "实际资本", "行业资本分级")
        self.assertEqual(selected["指标编码"].tolist(), ["A"])
        whole_module = navigation_scope_frame(frame, "实际资本", "全部")
        self.assertEqual(whole_module["指标编码"].tolist(), ["A", "B"])
        print_all = navigation_scope_frame(
            frame,
            "一键显示全部（打印/导出）",
            "全部",
        )
        self.assertEqual(print_all["指标编码"].tolist(), ["A", "B", "C"])

    def test_quality_and_reconciliation_rows_are_not_visualization_metrics(self):
        frame = sample_rows()
        quality = frame.iloc[[0]].copy()
        quality["一级模块"] = "数据质量"
        quality["二级模块"] = "勾稽检查"
        quality["指标编码"] = "FEATURE_FACTOR_CHECK"
        quality["指标名称"] = "check特征系数"
        quality["数据类型"] = "校验"
        quality["指标属性"] = "校验"
        combined = pd.concat([frame, quality], ignore_index=True)
        prepared, _ = prepare_analysis_frame(combined, "LIFE_SOLVENCY")
        self.assertNotIn("FEATURE_FACTOR_CHECK", prepared["指标编码"].tolist())
        self.assertNotIn("数据质量", nonblank_values(prepared, "一级模块"))
        self.assertEqual(len(visualization_metric_frame(quality)), 0)

    def test_company_type_quick_selection_uses_peer_classification(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        company_frame = report_scope_frame(frame, COMPANY_REPORT)
        self.assertEqual(
            companies_for_quick_selection(company_frame, "养老健康"),
            ["乙健康"],
        )
        self.assertEqual(
            companies_for_quick_selection(company_frame, NO_COMPANY_TYPE_FILTER),
            ["乙健康", "甲人寿"],
        )

    def test_period_sort_and_default_companies(self):
        periods = sort_report_periods(["2025Q4", "2024Q4", "2025Q1"])
        self.assertEqual(periods, ["2024Q4", "2025Q1", "2025Q4"])
        companies = ["甲人寿", "乙健康", "行业合计"]
        self.assertEqual(default_companies(COMPANY_REPORT, companies), ["甲人寿"])
        self.assertEqual(
            default_companies(INDUSTRY_REPORT, companies),
            ["甲人寿", "乙健康", "行业合计"],
        )

    def test_builds_each_supported_chart(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        for chart_type in (
            "簇状柱状图",
            "折线图",
            "带直线和数据标记的散点图",
            "散点图",
            "横向条形图",
        ):
            chart = build_comparison_chart(
                frame,
                chart_type,
                ["2024Q4", "2025Q4"],
            )
            spec = chart.to_dict()
            self.assertIn("layer", spec)
            mark_types = {
                layer.get("mark", {}).get("type")
                for layer in spec["layer"]
                if isinstance(layer.get("mark"), dict)
            }
            self.assertIn("text", mark_types)
            self.assertNotIn("追踪状态", str(spec))
            self.assertNotIn("diamond", str(spec))
            self.assertIn("120.00", str(spec))
            if chart_type in {"簇状柱状图", "横向条形图"}:
                self.assertEqual(
                    spec["layer"][0]["encoding"]["color"]["scale"]["range"],
                    ["#FFA3DA", "#00B8F5", "#FD349C"],
                )

    def test_chart_supports_kpmg_legend_customization_and_layout(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        chart = build_comparison_chart(
            frame,
            "簇状柱状图",
            ["2024Q4", "2025Q4"],
            layout_mode="以公司为横轴",
            legend_label_map={"2024Q4": "基期", "2025Q4": "本期"},
            legend_color_map={"2024Q4": "#1E49E2", "2025Q4": "#00338D"},
            show_average=True,
            transparent=True,
        )
        spec = chart.to_dict()
        serialized = str(spec)
        self.assertIn("基期", serialized)
        self.assertIn("#1E49E2", serialized)
        self.assertIn("transparent", serialized)
        self.assertIn("rule", serialized)

    def test_line_chart_emphasizes_tracked_company(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        chart = build_comparison_chart(
            frame,
            "折线图",
            ["2024Q4", "2025Q4"],
            highlight_company="甲人寿",
        )
        serialized = str(chart.to_dict())
        self.assertIn("甲人寿", serialized)
        self.assertIn("condition", serialized)
        self.assertIn("90", serialized)

    def test_line_chart_can_stagger_close_labels_on_transparent_background(self):
        frame, _ = prepare_analysis_frame(sample_rows(), "LIFE_SOLVENCY")
        frame = frame[frame["指标编码"].eq("ACTUAL_CAPITAL")].copy()
        frame.loc[frame["公司"].eq("乙健康"), "数值"] = frame.loc[
            frame["公司"].eq("甲人寿"), "数值"
        ].to_numpy()
        chart = build_comparison_chart(
            frame,
            "折线图",
            ["2024Q4", "2025Q4"],
            transparent=True,
            avoid_label_overlap=True,
        )
        spec = chart.to_dict()
        serialized = str(spec)
        text_marks = [
            layer["mark"]
            for layer in spec["layer"]
            if isinstance(layer.get("mark"), dict) and layer["mark"].get("type") == "text"
        ]
        self.assertEqual(spec["background"], "transparent")
        self.assertEqual(spec["config"]["view"]["fill"], "transparent")
        self.assertGreaterEqual(len(text_marks), 2)
        self.assertIn("-12", serialized)
        self.assertIn("12", serialized)

    def test_reuploaded_industry_total_keeps_aggregate_identity(self):
        industry = sample_rows().query("公司 == '行业合计'").iloc[[0]]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            industry.to_excel(writer, sheet_name="标准数据", index=False)
        result = read_standard_workbook(output.getvalue(), "industry.xlsx")
        self.assertEqual(result.iloc[0]["公司"], "行业合计")
        self.assertEqual(result.iloc[0]["公司类型"], "行业合计")
        self.assertEqual(result.iloc[0]["公司统一编码"], "INDUSTRY_LIFE_TOTAL")


if __name__ == "__main__":
    unittest.main()
