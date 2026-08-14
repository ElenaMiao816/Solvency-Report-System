from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from step7_solvency import (
    DEFAULT_SORT_LABEL,
    STEP7_CHART_TYPE,
    _analysis_completion_url,
    _call_ai_analysis_cached,
    _company_color_map,
    _convert_unit,
    _metric_sort_options,
    _sort_companies_by_metric,
)
from step8_solvency import (
    _default_peer_group_styles,
    _distribution_stats,
    _ordered_peer_groups,
    _ranking_figure,
    _rgba,
    _trend_figure,
    _industry_quant_waterfall_figure,
)
from services.solvency_step7_charts import (
    build_capital_amount_combo,
    build_company_bar_trend_chart,
    build_effect_diverging_chart,
    build_matrix_chart,
    build_single_metric_trend_chart,
    build_single_metric_trend_charts,
)
from services.solvency_step7_chart_plans import (
    CAPITAL_AMOUNT_COMBO,
    COMPANY_BAR_TREND,
    EFFECT_DIVERGING,
    MARKET_CREDIT_MATRIX,
    SINGLE_METRIC_TREND,
    SOLVENCY_MATRIX,
    chart_plan_for,
)


class Step78ReportViewTests(unittest.TestCase):
    def test_step7_ai_completion_url_accepts_root_or_full_endpoint(self):
        self.assertEqual(
            _analysis_completion_url("https://api.example.com/v1/"),
            "https://api.example.com/v1/chat/completions",
        )
        self.assertEqual(
            _analysis_completion_url("https://api.example.com/v1/chat/completions"),
            "https://api.example.com/v1/chat/completions",
        )

    def test_step7_ai_analysis_uses_configured_openai_compatible_endpoint(self):
        class Response:
            ok = True
            status_code = 200

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "测试点评"}}]}

        _call_ai_analysis_cached.clear()
        with patch("step7_solvency.requests.post", return_value=Response()) as post:
            result = _call_ai_analysis_cached(
                "甲公司=180%；乙公司=150%；样本均值=165%",
                "综合偿付能力充足率",
                "2025Q4",
                "secret",
                "https://api.example.com/v1",
                "example-model",
            )
        self.assertEqual(result, "测试点评")
        self.assertEqual(post.call_args.args[0], "https://api.example.com/v1/chat/completions")
        self.assertIn("甲公司=180%", post.call_args.kwargs["json"]["messages"][0]["content"])

    def test_step7_amount_unit_conversion_preserves_non_amount_units(self):
        frame = pd.DataFrame(
            [
                {"数值": 10000.0, "单位": "万元"},
                {"数值": 180.0, "单位": "%"},
            ]
        )
        result = _convert_unit(frame, "亿元")
        self.assertEqual(result.iloc[0]["数值"], 1.0)
        self.assertEqual(result.iloc[0]["单位"], "亿元")
        self.assertEqual(result.iloc[1]["数值"], 180.0)
        self.assertEqual(result.iloc[1]["单位"], "%")

    def test_step7_supports_annual_report_display_units(self):
        frame = pd.DataFrame([{"数值": 1_000_000_000.0, "单位": "元"}])
        expected = {
            "十亿元": 1.0,
            "亿元": 10.0,
            "百万元": 1000.0,
            "十万元": 10000.0,
        }
        for unit, value in expected.items():
            converted = _convert_unit(frame, unit)
            self.assertEqual(converted.iloc[0]["数值"], value)
            self.assertEqual(converted.iloc[0]["单位"], unit)

    def test_step7_company_sort_uses_latest_metric_and_keeps_missing_last(self):
        frame = pd.DataFrame(
            [
                {"公司": "甲", "指标编码": "RATIO", "报告期": "2025Q4", "数值": 120},
                {"公司": "乙", "指标编码": "RATIO", "报告期": "2025Q4", "数值": 180},
                {"公司": "丙", "指标编码": "OTHER", "报告期": "2025Q4", "数值": 999},
            ]
        )
        self.assertEqual(
            _sort_companies_by_metric(
                frame, ["甲", "乙", "丙"], "RATIO", "2025Q4", descending=True
            ),
            ["乙", "甲", "丙"],
        )
        self.assertEqual(
            _sort_companies_by_metric(
                frame, ["甲", "乙", "丙"], "RATIO", "2025Q4", descending=False
            ),
            ["甲", "乙", "丙"],
        )

    def test_step7_uses_fixed_line_chart_and_metric_sort_options(self):
        frame = pd.DataFrame(
            [
                {"指标编码": "CORE", "指标名称": "核心偿付能力充足率"},
                {"指标编码": "COMBINED", "指标名称": "综合偿付能力充足率"},
            ]
        )
        options, lookup = _metric_sort_options(frame)
        self.assertEqual(STEP7_CHART_TYPE, "内置分析方案")
        self.assertEqual(options[0], DEFAULT_SORT_LABEL)
        self.assertEqual(lookup["核心偿付能力充足率"], "CORE")

    def test_step7_tracking_reserves_kpmg_pink_for_selected_company(self):
        default_colors = _company_color_map(["甲", "乙", "丙"])
        colors = _company_color_map(["甲", "乙", "丙"], "乙")
        self.assertEqual(default_colors["甲"], "#00338D")
        self.assertEqual(colors["乙"], "#FD349C")
        self.assertNotEqual(colors["甲"], "#FD349C")

    def test_step7_uses_built_in_chart_plan_registry(self):
        self.assertEqual(chart_plan_for("综合偿付能力充足率").kind, COMPANY_BAR_TREND)
        self.assertEqual(chart_plan_for("核心资本/注册资本率").kind, SINGLE_METRIC_TREND)
        self.assertEqual(chart_plan_for("四级资本规模与结构").kind, CAPITAL_AMOUNT_COMBO)
        self.assertEqual(chart_plan_for("风险分散效应").kind, EFFECT_DIVERGING)
        self.assertEqual(chart_plan_for("损失吸收效应").kind, EFFECT_DIVERGING)
        self.assertEqual(chart_plan_for("偿付能力矩阵").kind, SOLVENCY_MATRIX)
        self.assertEqual(chart_plan_for("市场—信用风险矩阵").kind, MARKET_CREDIT_MATRIX)

    def test_step7_new_chart_builders_produce_valid_specs(self):
        rows = []
        for company, peer, core, combined, market, credit in [
            ("甲", "大型公司", 120.0, 180.0, 0.7, 0.2),
            ("乙", "小型公司", 90.0, 140.0, 0.9, 0.4),
        ]:
            for period, shift in [("2025Q2", 0.0), ("2025Q4", 5.0)]:
                for code, name, value, unit in [
                    ("CORE_SOLVENCY_RATIO", "核心偿付能力充足率", core + shift, "%"),
                    ("COMBINED_SOLVENCY_RATIO", "综合偿付能力充足率", combined + shift, "%"),
                    ("MARKET_RISK_TO_QUANT_CAPITAL", "市场风险最低资本占比", market, "倍"),
                    ("CREDIT_RISK_TO_QUANT_CAPITAL", "信用风险最低资本占比", credit, "倍"),
                    ("DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL", "风险分散效应最低资本占比", -0.3, "倍"),
                    ("LOSS_ABSORPTION_TO_QUANT_CAPITAL", "损失吸收效应最低资本占比", -0.1, "倍"),
                ]:
                    rows.append({"公司": company, "同业分类": peer, "报告期": period, "指标编码": code, "指标名称": name, "数值": value, "单位": unit})
                for code, value in zip(
                    ["CORE_T1_TO_ACTUAL_CAPITAL", "CORE_T2_TO_ACTUAL_CAPITAL", "ANC_T1_TO_ACTUAL_CAPITAL", "ANC_T2_TO_ACTUAL_CAPITAL"],
                    [0.6, 0.1, 0.25, 0.05],
                ):
                    rows.append({"公司": company, "同业分类": peer, "报告期": period, "指标编码": code, "指标名称": code, "数值": value, "单位": "倍"})
        frame = pd.DataFrame(rows)
        periods = ["2025Q2", "2025Q4"]
        colors = {"甲": "#00338D", "乙": "#00B8F5"}
        trend = build_single_metric_trend_chart(frame, "COMBINED_SOLVENCY_RATIO", periods, colors, "甲")
        company_panels = build_company_bar_trend_chart(frame, "COMBINED_SOLVENCY_RATIO", periods, "甲")
        diversification = build_effect_diverging_chart(frame, "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL", periods)
        loss_absorption = build_effect_diverging_chart(frame, "LOSS_ABSORPTION_TO_QUANT_CAPITAL", periods)
        solvency_matrix, _ = build_matrix_chart(
            frame, "CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO", periods,
            "偿付能力矩阵", "甲", True, colors,
        )
        market_matrix, _ = build_matrix_chart(
            frame, "MARKET_RISK_TO_QUANT_CAPITAL", "CREDIT_RISK_TO_QUANT_CAPITAL",
            periods, "市场—信用风险矩阵", "甲", company_colors=colors,
        )
        charts = [trend, company_panels, diversification, loss_absorption, solvency_matrix, market_matrix]
        for chart in charts:
            self.assertIsInstance(chart.to_dict(validate=True), dict)

        trend_spec = trend.to_dict(validate=True)
        self.assertEqual(trend_spec["layer"][0]["mark"]["type"], "line")
        self.assertEqual(trend_spec["layer"][1]["mark"]["type"], "point")
        self.assertGreaterEqual(trend_spec["layer"][1]["mark"]["size"], 50)
        self.assertNotIn("stroke", trend_spec["layer"][1]["mark"])
        self.assertFalse(trend_spec["config"]["axis"]["grid"])
        self.assertTrue(trend_spec["config"]["axis"]["domain"])
        self.assertEqual(trend_spec["config"]["axis"]["domainColor"], "#D0D5DD")
        self.assertNotIn("facet", trend_spec)
        panel_spec = company_panels.to_dict(validate=True)
        self.assertEqual(panel_spec["facet"]["field"], "公司")
        self.assertEqual(panel_spec["spec"]["layer"][0]["mark"]["type"], "bar")
        panel_bar_color = panel_spec["spec"]["layer"][0]["encoding"]["color"]
        self.assertEqual(panel_bar_color["field"], "报告期")
        self.assertEqual(panel_bar_color["scale"]["domain"], periods)
        self.assertEqual(panel_bar_color["scale"]["range"], ["#00338D", "#1E49E2"])
        self.assertIsNone(panel_bar_color["legend"])
        panel_y_scale = panel_spec["spec"]["layer"][0]["encoding"]["y"]["scale"]
        self.assertTrue(panel_y_scale["zero"])
        self.assertEqual(panel_y_scale["domainMin"], 0)
        self.assertNotIn("padding", panel_y_scale)
        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["type"], "line")
        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["strokeWidth"], 2.6)
        self.assertEqual(panel_spec["spec"]["layer"][2]["mark"]["type"], "point")
        self.assertNotIn("stroke", panel_spec["spec"]["layer"][2]["mark"])
        self.assertNotIn("stroke", panel_spec["spec"]["layer"][3]["mark"])
        self.assertFalse(panel_spec["config"]["axis"]["grid"])
        self.assertEqual(panel_spec["config"]["axis"]["tickColor"], "#D0D5DD")
        self.assertNotIn("aggregate", panel_spec["spec"]["layer"][0]["encoding"]["y"])
        tracked_frame = panel_spec["spec"]["layer"][-1]
        self.assertEqual(tracked_frame["mark"]["type"], "rect")
        self.assertEqual(tracked_frame["mark"]["stroke"], "#B8BDC7")
        self.assertEqual(tracked_frame["mark"]["strokeWidth"], 2)
        self.assertEqual(tracked_frame["encoding"]["x"]["value"], -13)
        self.assertEqual(tracked_frame["encoding"]["y"]["value"], -18)
        self.assertEqual(
            panel_spec["facet"]["header"]["labelPadding"],
            20,
        )

        diversification_spec = diversification.to_dict(validate=True)
        diversification_dataset = diversification_spec["data"]["name"]
        diversification_rows = diversification_spec["datasets"][diversification_dataset]
        self.assertEqual(len(diversification_rows), 4)
        self.assertEqual(diversification_spec["spec"]["layer"][1]["mark"]["type"], "text")
        self.assertNotIn("data", diversification_spec["spec"]["layer"][0])
        self.assertNotIn("data", diversification_spec["spec"]["layer"][1])
        self.assertEqual(
            diversification_spec["spec"]["layer"][1]["encoding"]["y"]["field"],
            "标签位置",
        )

        matrix_spec = market_matrix.to_dict(validate=True)
        point_encoding = matrix_spec["layer"][0]["encoding"]
        self.assertNotIn("stroke", matrix_spec["layer"][0]["mark"])
        self.assertFalse(matrix_spec["config"]["axis"]["grid"])
        self.assertEqual(point_encoding["color"]["field"], "公司")
        self.assertEqual(point_encoding["color"]["legend"]["orient"], "right")
        self.assertEqual(point_encoding["color"]["legend"]["offset"], 24)
        self.assertEqual(matrix_spec["width"], 820)
        self.assertEqual(matrix_spec["height"], 540)
        matrix_color_map = dict(zip(
            point_encoding["color"]["scale"]["domain"],
            point_encoding["color"]["scale"]["range"],
        ))
        self.assertEqual(matrix_color_map, colors)

    def test_step7_capital_charts_add_contrast_labels_and_drop_empty_facets(self):
        rows = []
        for period in ["2025Q2", "2025Q4"]:
            for code, value in zip(
                ["CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"],
                [60.0, 10.0, 25.0, 5.0],
            ):
                rows.append({"公司": "完整公司", "报告期": period, "指标编码": code, "数值": value, "单位": "亿元"})
            rows.append({"公司": "完整公司", "报告期": period, "指标编码": "ACTUAL_CAPITAL", "数值": 100.0, "单位": "亿元"})
        rows.append({"公司": "仅有实际资本", "报告期": "2025Q4", "指标编码": "ACTUAL_CAPITAL", "数值": 80.0, "单位": "亿元"})
        amount_spec = build_capital_amount_combo(pd.DataFrame(rows), ["2025Q2", "2025Q4"]).to_dict(validate=True)
        dataset_name = amount_spec["data"]["name"]
        chart_companies = {row["公司"] for row in amount_spec["datasets"][dataset_name]}
        self.assertEqual(chart_companies, {"完整公司"})
        self.assertEqual(amount_spec["columns"], 1)
        label_encoding = amount_spec["spec"]["layer"][1]["encoding"]
        self.assertEqual(label_encoding["text"]["field"], "占比标签")
        self.assertEqual(label_encoding["y"]["field"], "标签位置")
        self.assertEqual(label_encoding["color"]["condition"]["value"], "#ACEAFF")
        self.assertEqual(label_encoding["color"]["value"], "#0C233C")

    def test_step7_trend_legends_show_undisclosed_periods_and_taller_plots(self):
        frame = pd.DataFrame([
            {"公司": "甲", "报告期": "2025Q2", "指标编码": "TEST", "指标名称": "测试指标", "数值": 1.0, "单位": "倍"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "TEST", "指标名称": "测试指标", "数值": 1.2, "单位": "倍"},
            {"公司": "乙", "报告期": "2025Q2", "指标编码": "TEST", "指标名称": "测试指标", "数值": 0.8, "单位": "倍"},
        ])
        periods = ["2025Q2", "2025Q3", "2025Q4"]
        trend_spec = build_single_metric_trend_chart(
            frame,
            "TEST",
            periods,
            {"甲": "#00338D", "乙": "#1E49E2"},
        ).to_dict(validate=True)
        self.assertEqual(trend_spec["height"], 420)
        self.assertEqual(trend_spec["layer"][0]["encoding"]["color"]["legend"]["title"], "公司")
        self.assertEqual(
            trend_spec["layer"][0]["encoding"]["color"]["legend"]["orient"],
            "top-right",
        )
        self.assertFalse(any(
            layer.get("encoding", {}).get("color", {}).get("scale", {}).get("domain") == ["未披露"]
            for layer in trend_spec["layer"]
        ))
        metric_dataset = trend_spec.get("data", trend_spec["layer"][0].get("data"))["name"]
        missing_rows = [
            row for row in trend_spec["datasets"][metric_dataset]
            if row["披露状态"] == "未披露"
        ]
        self.assertEqual(len(missing_rows), 3)
        self.assertTrue(all(row["数值"] is None for row in missing_rows))

        dense_rows = [
            {
                "公司": f"公司{company_index:02d}",
                "报告期": period,
                "指标编码": "TEST",
                "指标名称": "测试指标",
                "数值": company_index + period_index / 10,
                "单位": "倍",
            }
            for company_index in range(21)
            for period_index, period in enumerate(periods)
        ]
        dense_colors = {f"公司{index:02d}": "#00338D" for index in range(21)}
        grouped = build_single_metric_trend_charts(
            pd.DataFrame(dense_rows),
            "TEST",
            periods,
            dense_colors,
            "公司00",
        )
        self.assertEqual(len(grouped), 3)
        grouped_specs = [chart.to_dict(validate=True) for chart in grouped]
        y_domains = [spec["layer"][0]["encoding"]["y"]["scale"]["domain"] for spec in grouped_specs]
        self.assertTrue(all(domain == y_domains[0] for domain in y_domains))
        for spec in grouped_specs:
            dataset_name = spec.get("data", spec["layer"][0].get("data"))["name"]
            chart_companies = {row["公司"] for row in spec["datasets"][dataset_name]}
            self.assertIn("公司00", chart_companies)
            self.assertLessEqual(len(chart_companies), 8)
            self.assertIn("组", spec["title"])

        panel_spec = build_company_bar_trend_chart(frame, "TEST", periods).to_dict(validate=True)
        self.assertEqual(panel_spec["spec"]["height"], 285)
        self.assertEqual(panel_spec["spec"]["layer"][0]["mark"]["type"], "bar")
        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["type"], "line")
        self.assertLessEqual(len(panel_spec["spec"]["layer"]), 5)

    def test_step7_non_life_risk_to_liabilities_uses_five_decimal_labels(self):
        frame = pd.DataFrame([
            {
                "公司": company,
                "报告期": period,
                "指标编码": "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",
                "指标名称": "保险风险（非寿）/认可负债",
                "数值": value,
                "单位": "倍",
            }
            for company, period, value in [
                ("甲", "2025Q2", 0.0012345),
                ("甲", "2025Q4", 0.0013456),
                ("乙", "2025Q2", 0.0004567),
                ("乙", "2025Q4", 0.0005678),
            ]
        ])
        spec = build_single_metric_trend_chart(
            frame,
            "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",
            ["2025Q2", "2025Q4"],
            {"甲": "#00338D", "乙": "#1E49E2"},
            "甲",
        ).to_dict(validate=True)
        for layer_index in (2, 3, 4):
            self.assertEqual(
                spec["layer"][layer_index]["encoding"]["text"]["format"],
                ",.5f",
            )
        y_encoding = spec["layer"][0]["encoding"]["y"]
        self.assertEqual(y_encoding["axis"]["format"], ".5f")
        self.assertEqual(y_encoding["axis"]["tickCount"], 6)
        domain = y_encoding["scale"]["domain"]
        self.assertLess(domain[1] - domain[0], 0.002)

    def test_step7_capital_layout_avoids_trailing_blank_panels(self):
        def amount_rows(company_count: int) -> pd.DataFrame:
            return pd.DataFrame([
                {"公司": f"公司{index:02d}", "报告期": "2025Q4", "指标编码": code, "数值": value, "单位": "亿元"}
                for index in range(company_count)
                for code, value in zip(
                    ["CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL", "ACTUAL_CAPITAL"],
                    [70.0, 0.2, 29.7, 0.1, 100.0],
                )
            ])

        spec = build_capital_amount_combo(amount_rows(15), ["2025Q4"]).to_dict(validate=True)
        self.assertEqual(spec["columns"], 5)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset[:4]}
        self.assertEqual(labels["CORE_T1_CAPITAL"], "70.0%")
        self.assertEqual(labels["CORE_T2_CAPITAL"], "")
        seven_spec = build_capital_amount_combo(amount_rows(7), ["2025Q4"]).to_dict(validate=True)
        self.assertEqual(seven_spec["columns"], 7)

    def test_step8_industry_quant_waterfall_preserves_positive_and_negative_effects(self):
        codes = [
            "INDUSTRY_LIFE_INSURANCE_RISK", "INDUSTRY_NON_LIFE_INSURANCE_RISK",
            "INDUSTRY_MARKET_RISK", "INDUSTRY_CREDIT_RISK",
            "INDUSTRY_CAPITALIZABLE_DIVERSIFICATION_EFFECT", "INDUSTRY_LOSS_ABSORPTION",
            "INDUSTRY_CONTROL_RISK",
        ]
        values = [40.0, 5.0, 60.0, 20.0, -25.0, -8.0, -2.0]
        frame = pd.DataFrame({"指标编码": codes, "报告期": ["2025Q4"] * 7, "数值": values})
        figure = _industry_quant_waterfall_figure(frame, "2025Q4")
        self.assertEqual(list(figure.data[0].measure)[-1], "total")
        self.assertEqual(float(figure.data[0].y[-1]), sum(values))

    def test_step8_distribution_statistics_are_period_and_peer_group_scoped(self):
        frame = pd.DataFrame(
            {
                "报告期": ["2025Q4"] * 4,
                "同业分类": ["头部"] * 4,
                "数值": [100.0, 120.0, 140.0, 160.0],
            }
        )
        stats = _distribution_stats(frame)
        self.assertEqual(int(stats.iloc[0]["公司数"]), 4)
        self.assertEqual(float(stats.iloc[0]["最小值"]), 100.0)
        self.assertEqual(float(stats.iloc[0]["中位数"]), 130.0)
        self.assertEqual(float(stats.iloc[0]["最大值"]), 160.0)

    def test_step8_trend_uses_plotly_compatible_transparent_fill(self):
        stats = pd.DataFrame(
            {
                "报告期": ["2025Q3", "2025Q4"],
                "同业分类": ["头部", "头部"],
                "公司数": [3, 3],
                "最小值": [100.0, 110.0],
                "下四分位": [120.0, 130.0],
                "中位数": [140.0, 150.0],
                "上四分位": [160.0, 170.0],
                "最大值": [180.0, 190.0],
                "平均值": [140.0, 150.0],
            }
        )
        figure = _trend_figure(
            stats,
            "综合偿付能力充足率",
            "COMBINED_SOLVENCY_RATIO",
            ["2025Q3", "2025Q4"],
            {"头部": "#00338D"},
            {"头部": "头部"},
            1,
            True,
        )
        self.assertEqual(_rgba("#00338D"), "rgba(0,51,141,0.170)")
        self.assertEqual(figure.data[1].fillcolor, "rgba(0,51,141,0.170)")

    def test_step8_ranking_applies_selected_bar_gap(self):
        frame = pd.DataFrame([
            {
                "公司": "甲人寿",
                "公司类型": "寿险",
                "同业分类": "头部",
                "报告期": "2025Q4",
                "期间口径": "本季度末数",
                "数值": 160.0,
                "单位": "%",
                "数据类型": "百分比",
            },
        ])
        figure = _ranking_figure(
            frame,
            "综合偿付能力充足率",
            "COMBINED_SOLVENCY_RATIO",
            {"头部": "#00338D"},
            {"头部": "头部"},
            1,
            True,
            0.55,
        )
        self.assertEqual(figure.layout.bargap, 0.55)

    def test_step8_peer_groups_follow_annual_report_default_order(self):
        groups = ["小型", "外资", "头部", "其他", "银行系", "养老健康"]
        self.assertEqual(
            _ordered_peer_groups(groups),
            ["头部", "银行系", "外资", "养老健康", "小型", "其他"],
        )

    def test_step8_peer_group_styles_use_system_labels_and_stable_colors(self):
        colors, labels = _default_peer_group_styles(["头部", "银行系", "外资"])
        self.assertEqual(labels, {"头部": "头部", "银行系": "银行系", "外资": "外资"})
        self.assertEqual(list(colors), ["头部", "银行系", "外资"])
        self.assertEqual(len(set(colors.values())), 3)

    def test_step8_ranking_legend_follows_selected_peer_group_order(self):
        frame = pd.DataFrame(
            [
                {
                    "公司": "甲人寿",
                    "同业分类": "头部",
                    "报告期": "2025Q4",
                    "期间口径": "本季度末数",
                    "数值": 160.0,
                    "单位": "%",
                    "数据类型": "百分比",
                },
                {
                    "公司": "乙人寿",
                    "同业分类": "银行系",
                    "报告期": "2025Q4",
                    "期间口径": "本季度末数",
                    "数值": 150.0,
                    "单位": "%",
                    "数据类型": "百分比",
                },
            ]
        )
        figure = _ranking_figure(
            frame,
            "综合偿付能力充足率",
            "COMBINED_SOLVENCY_RATIO",
            {"头部": "#00338D", "银行系": "#00B8F5"},
            {"头部": "头部", "银行系": "银行系"},
            1,
            True,
            0.35,
            ["银行系", "头部"],
        )
        self.assertEqual([trace.name for trace in figure.data], ["银行系", "头部"])


if __name__ == "__main__":
    unittest.main()
