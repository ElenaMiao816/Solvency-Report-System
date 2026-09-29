from __future__ import annotations

import ast
import unittest
import inspect
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from step7_solvency import (
    BUBBLE_FULL_PANEL_WIDTH,
    BUBBLE_ZOOM_PANEL_WIDTH,
    COMPANY_CHART_RENDER_KEYS,
    COMPONENT_STACK_DENOMINATOR_CODES,
    COMPONENT_STACK_SPECS,
    DEFAULT_SORT_LABEL,
    QUANT_RISK_STACK_RATIO_CODES,
    RISK_SCATTER_DISPLAY_WIDTH,
    RISK_SCATTER_FULL_PANEL_WIDTH,
    RISK_SCATTER_ZOOM_PANEL_WIDTH,
    STEP7_CHART_TYPE,
    _analysis_completion_url,
    _call_ai_analysis_cached,
    _chart_without_internal_title,
    _chart_input_metric_codes,
    _bubble_company_color_map,
    _company_color_map,
    _company_scope_for_types,
    _convert_unit,
    _metric_sort_options,
    _missing_disclosure_note_lines,
    _render_capital_efficiency_bubble_fragment,
    _render_combination_analysis,
    _render_risk_ratio_scatter_fragment,
    _render_report_metric,
    _policy_surplus_coverage_note,
    _core_policy_surplus_coverage_note,
    _report_style,
    _sort_companies_by_metric,
    _unavailable_metric_details,
    show_step_7_solvency,
)
from step8_solvency import (
    SUPPLEMENTAL_SECTIONS,
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
    build_capital_efficiency_bubble_chart,
    capital_efficiency_bubble_default_domain,
    combine_capital_efficiency_bubble_charts,
    build_capital_ratio_combo_chart,
    capital_ratio_amount_axis_domain,
    build_company_bar_trend_chart,
    build_company_period_bar_chart,
    build_component_stack_chart,
    build_effect_diverging_chart,
    build_matrix_chart,
    build_single_metric_trend_chart,
    build_single_metric_trend_charts,
    build_solvency_ratio_combo_chart,
    metric_bar_axis_domain,
    risk_ratio_scatter_default_domain,
    combine_linked_matrix_charts,
    solvency_ratio_axis_domain,
)
from services.solvency_step7_chart_plans import (
    CAPITAL_AMOUNT_COMBO,
    CAPITAL_EFFICIENCY_BUBBLE,
    CAPITAL_RATIO_COMBO,
    CAPITAL_STRUCTURE_COMBO,
    COMPANY_BAR_TREND,
    COMPANY_PERIOD_BAR,
    COMPONENT_STACK,
    EFFECT_DIVERGING,
    MARKET_CREDIT_MATRIX,
    SINGLE_METRIC_TREND,
    SOLVENCY_MATRIX,
    SOLVENCY_RATIO_COMBO,
    TREND_WITH_COMPANY_BARS,
    chart_plan_for,
)


class Step78ReportViewTests(unittest.TestCase):
    def test_policy_surplus_stack_uses_renamed_title_and_explains_full_allocation(self):
        new_title = "计入各级资本的保单未来盈余构成占比"
        self.assertIn(new_title, COMPONENT_STACK_SPECS)
        self.assertNotIn("各级资本中的保单未来盈余占比", COMPONENT_STACK_SPECS)
        render_source = inspect.getsource(_render_combination_analysis)
        self.assertIn(new_title, render_source)
        self.assertIn(
            "100%表示该公司披露的保单未来盈余全部计入了同一个资本层级",
            render_source,
        )

    def test_all_altair_text_layers_disable_tooltips(self):
        project_root = Path(__file__).resolve().parents[1]
        for relative_path in (
            Path("services/solvency_step6_analysis.py"),
            Path("services/solvency_step7_charts.py"),
        ):
            source_path = project_root / relative_path
            tree = ast.parse(source_path.read_text(encoding="utf-8"))
            text_calls = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "mark_text"
            ]
            self.assertTrue(text_calls, str(relative_path))
            for call in text_calls:
                tooltip = next(
                    (keyword.value for keyword in call.keywords if keyword.arg == "tooltip"),
                    None,
                )
                self.assertIsInstance(tooltip, ast.Constant, str(relative_path))
                self.assertIs(tooltip.value, False, str(relative_path))

    def test_removed_peer_classification_board_is_not_a_step8_supplement(self):
        self.assertNotIn("调研公司分类列表", SUPPLEMENTAL_SECTIONS)

    def test_step7_chart_renderer_removes_repeated_metric_and_vega_titles(self):
        source = inspect.getsource(_render_report_metric)
        self.assertNotIn('st.markdown(f"#### {metric_name}")', source)
        self.assertIn("_chart_without_internal_title(chart)", source)

    def test_step7_print_keeps_company_cards_intact_and_screen_keeps_single_row(self):
        source = (Path(__file__).resolve().parents[1] / "step7_solvency.py").read_text(
            encoding="utf-8"
        )
        print_css = source[source.index("@media print {"):source.index("</style>")]
        self.assertIn('key=f"annual_company_grid_{key_prefix}"', source)
        self.assertIn('[class*="st-key-annual_company_grid_"]', print_css)
        self.assertIn('grid-template-columns:repeat(auto-fit,minmax(155px,1fr))', print_css)
        self.assertIn('--solvency-print-content-width:314.67mm', print_css)
        self.assertIn('grid-auto-flow:column!important', print_css)
        self.assertIn('grid-auto-columns:minmax(0,1fr)!important', print_css)
        self.assertIn('[class*="st-key-annual_company_panel_"] {', print_css)
        self.assertIn('break-inside:avoid-page!important', print_css)
        self.assertIn('> [data-testid="stLayoutWrapper"]', print_css)
        self.assertIn(
            '[data-testid="stHorizontalBlock"] > [data-testid="stColumn"]',
            print_css,
        )
        self.assertIn('justify-self:stretch!important', print_css)
        self.assertIn(
            '[class*="st-key-s7_report_module_"]:has([class*="st-key-s7_company_bars_page_"])',
            print_css,
        )
        self.assertIn('break-before:page!important; page-break-before:always!important', print_css)
        self.assertIn(
            'with st.container(key=f"s7_company_bars_page_{key_prefix}_{code}")',
            source,
        )
        self.assertIn('[data-testid="stElementContainer"]:has([class*="st-key-annual_company_panel_"])', print_css)
        self.assertIn('html.solvency-print-mode-widescreen .key-solvency-overview {', print_css)
        self.assertIn('width:100%!important; min-width:0!important; max-width:100%!important', print_css)
        self.assertIn('table-layout:fixed!important', print_css)
        self.assertIn('font-size:9px!important; padding:3px 2px!important', print_css)
        self.assertIn('width:var(--solvency-print-page-width)!important', print_css)
        self.assertIn('max-width:100%!important; height:auto!important', print_css)
        self.assertIn('min-height:280px!important', print_css)
        self.assertIn('width:100%!important; min-width:0!important; max-width:100%!important', print_css)
        self.assertIn('height:280px!important; min-height:280px!important; max-height:280px!important', print_css)
        self.assertNotIn(
            '[class*="st-key-s7_report_module_"] canvas,\n'
            '            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] svg,',
            print_css,
        )
        self.assertIn('margin:0 auto!important; padding:0!important', print_css)
        self.assertIn('height:var(--solvency-print-page-height)!important', print_css)
        self.assertIn('padding:8mm 12mm!important', print_css)
        self.assertIn('align-items:center!important', print_css)
        self.assertIn('justify-content:center!important', print_css)
        self.assertIn('transform:scale(var(--solvency-print-scale))!important', print_css)
        self.assertIn('transform-origin:center center!important', print_css)
        self.assertIn('page-break-before:auto!important', print_css)
        self.assertIn('font-size:30px!important', print_css)
        self.assertIn('margin:10px 0 8px!important', print_css)
        screen_css = source[source.index("<style>"):source.index("@media print {")]
        self.assertIn('st-key-annual_company_grid_', screen_css)
        self.assertIn('flex-wrap:nowrap!important', screen_css)
        self.assertIn('flex:1 1 0!important', screen_css)
        self.assertIn('html.solvency-print-mode-widescreen {', screen_css)
        self.assertIn('height:var(--solvency-print-page-height)!important', screen_css)
        self.assertIn('transform:scale(var(--solvency-print-scale))!important', screen_css)
        self.assertNotIn('table-layout:fixed!important', screen_css)

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

    def test_step7_amount_unit_conversion_is_vectorized(self):
        frame = pd.DataFrame(
            [
                {"数值": 10000.0, "单位": "万元"},
                {"数值": 180.0, "单位": "%"},
            ]
        )
        with patch.object(
            pd.DataFrame,
            "apply",
            side_effect=AssertionError("单位换算不应退回逐行 apply"),
        ):
            result = _convert_unit(frame, "亿元")
        self.assertEqual(result["数值"].tolist(), [1.0, 180.0])

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

    def test_step7_tracking_reserves_kpmg_blue_for_selected_company(self):
        default_colors = _company_color_map(["甲", "乙", "丙"])
        colors = _company_color_map(["甲", "乙", "丙"], "乙")
        self.assertEqual(default_colors["甲"], "#00338D")
        self.assertEqual(colors["乙"], "#00338D")
        self.assertNotEqual(colors["甲"], "#00338D")

    def test_capital_efficiency_bubbles_use_shared_company_colors(self):
        companies = ["甲", "乙", "丙"]

        self.assertEqual(
            _bubble_company_color_map(companies),
            _company_color_map(companies),
        )
        self.assertEqual(
            _bubble_company_color_map(companies, "乙"),
            _company_color_map(companies, "乙"),
        )

        renderer = getattr(
            _render_capital_efficiency_bubble_fragment,
            "__wrapped__",
            _render_capital_efficiency_bubble_fragment,
        )
        self.assertNotIn("_bubble_company_color_map", inspect.getsource(renderer))

        combination_source = inspect.getsource(_render_combination_analysis)
        metric_source = inspect.getsource(_render_report_metric)
        expected_mapping = '_company_color_map(frame["公司"], highlight_company)'
        self.assertIn(expected_mapping, combination_source)
        self.assertIn(expected_mapping, metric_source)

    def test_print_all_uses_stable_chart_keys_and_keeps_solvency_ratio_chart(self):
        self.assertIn("核心及综合充足率", COMPANY_CHART_RENDER_KEYS)
        self.assertNotEqual(
            COMPANY_CHART_RENDER_KEYS["核心及综合充足率"],
            COMPANY_CHART_RENDER_KEYS["综合充足率变化"],
        )
        source = inspect.getsource(show_step_7_solvency)
        self.assertIn("COMPANY_CHART_RENDER_KEYS.get", source)
        self.assertIn('key=f"s7_report_module_{chart_render_key}"', source)
        self.assertIn('key_prefix=f"s7_{chart_render_key}"', source)

    def test_step7_company_type_scope_controls_available_companies(self):
        frame = pd.DataFrame([
            {"公司": "甲公司", "同业分类": "大型公司"},
            {"公司": "乙公司", "同业分类": "银行系"},
            {"公司": "丙公司", "同业分类": "大型公司"},
            {"公司": "乙公司", "同业分类": "银行系"},
        ])
        scoped, companies = _company_scope_for_types(frame, ["银行系"])
        self.assertEqual(companies, ["乙公司"])
        self.assertEqual(set(scoped["同业分类"]), {"银行系"})

        all_scoped, all_companies = _company_scope_for_types(frame, ["全部"])
        self.assertEqual(all_companies, ["甲公司", "乙公司", "丙公司"])
        self.assertEqual(len(all_scoped), len(frame))

    def test_step7_company_type_widget_immediately_updates_company_widget(self):
        app = AppTest.from_string(
            '''
import pandas as pd
import streamlit as st
from step7_solvency import (
    _company_scope_for_types,
    _sync_company_selection_state,
)

@st.fragment
def render_filters():
    frame = pd.DataFrame([
        {"公司": "甲公司", "同业分类": "头部"},
        {"公司": "乙公司", "同业分类": "银行系"},
        {"公司": "丙公司", "同业分类": "外资"},
        {"公司": "丁公司", "同业分类": "养老健康"},
        {"公司": "戊公司", "同业分类": "小型"},
    ])
    st.session_state.setdefault("s7_company_types", ["全部"])
    selected_types = st.multiselect(
        "公司类型",
        ["全部", "头部", "银行系", "外资", "养老健康", "小型"],
        key="s7_company_types",
    )
    _, companies = _company_scope_for_types(frame, selected_types)
    company_key = _sync_company_selection_state(selected_types, companies)
    st.multiselect(
        "展示公司",
        companies,
        key=company_key,
    )

render_filters()
'''
        ).run()
        expected_by_type = {
            "头部": "甲公司",
            "银行系": "乙公司",
            "外资": "丙公司",
            "养老健康": "丁公司",
            "小型": "戊公司",
        }
        for company_type, company in expected_by_type.items():
            app.multiselect[0].set_value([company_type]).run()
            self.assertEqual(app.multiselect[1].values, [company])
            self.assertEqual(app.multiselect[1].value, [company])

        # A specific type must take precedence if "全部" is still present.
        app.multiselect[0].set_value(["全部", "银行系"]).run()
        self.assertEqual(app.multiselect[1].values, ["乙公司"])
        self.assertEqual(app.multiselect[1].value, ["乙公司"])

        app.multiselect[0].set_value(["头部", "外资"]).run()
        self.assertEqual(
            app.multiselect[1].values,
            ["甲公司", "丙公司"],
        )
        self.assertEqual(
            app.multiselect[1].value,
            ["甲公司", "丙公司"],
        )

        app.multiselect[0].set_value(["全部"]).run()
        self.assertEqual(
            app.multiselect[1].values,
            ["甲公司", "乙公司", "丙公司", "丁公司", "戊公司"],
        )

    def test_step7_full_report_filters_support_repeated_type_changes(self):
        app = AppTest.from_string(
            '''
import pandas as pd
from step7_solvency import show_step_7_solvency

companies = [
    ("甲公司", "头部"),
    ("乙公司", "银行系"),
    ("丙公司", "外资"),
    ("丁公司", "养老健康"),
    ("戊公司", "小型"),
]
frame = pd.DataFrame([
    {
        "公司": company,
        "公司类型": "寿险",
        "同业分类": peer_group,
        "报告期": "2025Q4",
        "指标编码": "CORE_SOLVENCY_RATIO",
        "指标名称": "核心偿付能力充足率",
        "数值": 1.5,
        "单位": "倍",
    }
    for company, peer_group in companies
])
show_step_7_solvency(frame)
'''
        ).run(timeout=20)

        def widget(label):
            return next(item for item in app.multiselect if item.label == label)

        expected = {
            "头部": "甲公司",
            "银行系": "乙公司",
            "外资": "丙公司",
            "养老健康": "丁公司",
            "小型": "戊公司",
        }
        for company_type, company in expected.items():
            widget("公司类型").set_value([company_type]).run(timeout=20)
            self.assertEqual(widget("展示公司").values, [company])
            self.assertEqual(widget("展示公司").value, [company])

        widget("公司类型").set_value(["全部"]).run(timeout=20)
        self.assertEqual(
            widget("展示公司").value,
            ["甲公司", "乙公司", "丙公司", "丁公司", "戊公司"],
        )

    def test_step7_uses_built_in_chart_plan_registry(self):
        self.assertEqual(chart_plan_for("核心及综合充足率").kind, SOLVENCY_RATIO_COMBO)
        self.assertEqual(chart_plan_for("综合充足率变化").kind, CAPITAL_RATIO_COMBO)
        self.assertEqual(chart_plan_for("核心资本占比").kind, CAPITAL_STRUCTURE_COMBO)
        self.assertEqual(
            chart_plan_for("资本使用效率与核心资本占比气泡图").kind,
            CAPITAL_EFFICIENCY_BUBBLE,
        )
        self.assertEqual(chart_plan_for("核心资本/注册资本").kind, SINGLE_METRIC_TREND)
        self.assertEqual(
            chart_plan_for("计入核心资本的保单未来盈余/核心资本的比例").kind,
            SINGLE_METRIC_TREND,
        )
        self.assertEqual(chart_plan_for("资本规模与结构").kind, CAPITAL_AMOUNT_COMBO)
        self.assertEqual(chart_plan_for("量化风险最低资本构成").kind, COMPONENT_STACK)

    def test_solvency_ratio_combo_explains_bar_and_line_encoding(self):
        source = inspect.getsource(_render_combination_analysis)
        self.assertIn("图表说明：柱状图为综合偿付能力充足率", source)
        self.assertIn("粉色折线为核心偿付能力充足率", source)
        self.assertIn("核心偿付能力充足率（%）", source)
        self.assertEqual(RISK_SCATTER_DISPLAY_WIDTH, 750)
        self.assertEqual(RISK_SCATTER_FULL_PANEL_WIDTH, BUBBLE_FULL_PANEL_WIDTH)
        self.assertEqual(RISK_SCATTER_ZOOM_PANEL_WIDTH, BUBBLE_ZOOM_PANEL_WIDTH)
        render_source = inspect.getsource(_render_combination_analysis)
        self.assertIn("_render_risk_ratio_scatter_fragment", render_source)
        self.assertIn('horizontal_alignment="center"', render_source)
        self.assertIn("width=RISK_SCATTER_DISPLAY_WIDTH", render_source)
        self.assertIn("核心资本占实际资本的比例", render_source)
        self.assertNotIn("核心资本占两者合计的比例", render_source)
        metric_render_source = inspect.getsource(_render_report_metric)
        self.assertIn("build_single_metric_trend_charts", metric_render_source)
        self.assertIn("company_period_bars", metric_render_source)

    def test_capital_efficiency_bubble_uses_core_registered_ratio_and_asset_size(self):
        periods = ["2025Q2", "2025Q4"]
        latest_values = {
            "甲": {
                "ACTUAL_CAPITAL": 20.0,
                "RECOGNIZED_ASSETS": 100.0,
                "REGISTERED_CAPITAL": 28.0,
                "CORE_T1_CAPITAL": 12.0,
                "CORE_T2_CAPITAL": 2.0,
            },
            "乙": {
                "ACTUAL_CAPITAL": 30.0,
                "RECOGNIZED_ASSETS": 120.0,
                "REGISTERED_CAPITAL": 36.0,
                "CORE_T1_CAPITAL": 15.0,
                "CORE_T2_CAPITAL": 3.0,
            },
        }
        rows = []
        for company, values in latest_values.items():
            for period in periods:
                for code, value in values.items():
                    rows.append({
                        "公司": company,
                        "同业分类": "大型公司" if company == "甲" else "小型公司",
                        "报告期": period,
                        "指标编码": code,
                        "指标名称": code,
                        "数值": value / 2 if period == "2025Q2" else value,
                        "单位": "亿元",
                    })
        chart, latest_period = build_capital_efficiency_bubble_chart(
            pd.DataFrame(rows),
            periods,
            {"甲": "#00338D", "乙": "#00B8F5"},
            "甲",
        )
        self.assertEqual(latest_period, "2025Q4")
        spec = chart.to_dict(validate=True)
        self.assertEqual(spec["layer"][0]["mark"]["type"], "rect")
        self.assertEqual(spec["layer"][0]["mark"]["stroke"], "#D9DEE7")
        self.assertEqual(spec["layer"][0]["mark"]["fillOpacity"], 0)
        rule_layers = [
            layer for layer in spec["layer"]
            if layer["mark"]["type"] == "rule"
        ]
        self.assertEqual(len(rule_layers), 2)
        for layer in rule_layers:
            self.assertEqual(layer["mark"]["color"], "#C9CED6")
            self.assertEqual(layer["mark"]["strokeDash"], [5, 4])
        bubble_layer = next(
            layer for layer in spec["layer"]
            if layer["mark"]["type"] == "circle"
        )
        self.assertEqual(
            bubble_layer["encoding"]["x"]["title"],
            "实际资本/注册资本（%）",
        )
        self.assertEqual(
            bubble_layer["encoding"]["y"]["title"],
            "核心资本/注册资本（%）",
        )
        self.assertNotIn("format", bubble_layer["encoding"]["x"]["axis"])
        self.assertEqual(
            bubble_layer["encoding"]["x"]["axis"]["labelExpr"],
            "format(datum.value * 100, '.1f')",
        )
        self.assertEqual(bubble_layer["encoding"]["x"]["axis"]["tickCount"], 6)
        self.assertTrue(bubble_layer["encoding"]["x"]["axis"]["labels"])
        self.assertTrue(bubble_layer["encoding"]["x"]["axis"]["ticks"])
        self.assertEqual(bubble_layer["encoding"]["x"]["axis"]["domainColor"], "#D9DEE7")
        self.assertEqual(
            bubble_layer["encoding"]["x"]["axis"]["title"],
            "实际资本/注册资本（%）",
        )
        self.assertTrue(spec["config"]["axisX"]["labels"])
        self.assertNotIn("format", bubble_layer["encoding"]["y"]["axis"])
        self.assertEqual(
            bubble_layer["encoding"]["y"]["axis"]["labelExpr"],
            "format(datum.value * 100, '.0f')",
        )
        self.assertTrue(bubble_layer["encoding"]["y"]["axis"]["labels"])
        self.assertTrue(bubble_layer["encoding"]["y"]["axis"]["ticks"])
        self.assertEqual(bubble_layer["encoding"]["size"]["field"], "气泡大小")
        self.assertEqual(
            bubble_layer["encoding"]["size"]["legend"]["title"],
            ["气泡大小代表", "认可资产金额（亿元）"],
        )
        self.assertEqual(
            bubble_layer["encoding"]["size"]["legend"]["tickCount"],
            3,
        )
        company_label_layers = [
            layer for layer in spec["layer"]
            if layer["mark"]["type"] == "text"
            and layer.get("encoding", {}).get("text", {}).get("field") == "公司"
        ]
        self.assertEqual(company_label_layers, [])
        self.assertEqual(
            bubble_layer["encoding"]["color"]["legend"]["title"],
            "公司",
        )
        self.assertIn(
            "公司",
            [tooltip["field"] for tooltip in bubble_layer["encoding"]["tooltip"]],
        )
        self.assertEqual(spec["width"], 650)
        self.assertEqual(spec["height"], 430)
        self.assertEqual(spec["title"]["text"], "全样本图")
        self.assertEqual(spec["title"]["anchor"], "start")
        self.assertEqual(spec["title"]["offset"], 8)
        dataset_name = bubble_layer["data"]["name"]
        plotted = {row["公司"]: row for row in spec["datasets"][dataset_name]}
        self.assertEqual(set(plotted), {"甲", "乙"})
        self.assertTrue(all(row["报告期"] == "2025Q4" for row in plotted.values()))
        self.assertAlmostEqual(plotted["甲"]["实际资本/注册资本"], 20.0 / 28.0)
        self.assertAlmostEqual(plotted["甲"]["核心资本/注册资本"], 0.50)
        self.assertAlmostEqual(plotted["甲"]["气泡大小"], 100.0)
        self.assertTrue(all("标签位置" not in row for row in plotted.values()))

    def test_capital_efficiency_overlap_zoom_selects_closest_pair(self):
        rows = []
        companies = {
            "建信人寿": (10.1, 100.0, 0.52),
            "农银人寿": (10.3, 100.0, 0.51),
            "中银三星": (10.4, 100.0, 0.69),
            "招商信诺": (12.8, 100.0, 0.64),
            "交银人寿": (12.4, 100.0, 0.78),
            "中邮人寿": (9.1, 100.0, 0.60),
            "工银安盛": (16.0, 100.0, 0.71),
        }
        for company, (actual, assets, core_share) in companies.items():
            values = {
                "ACTUAL_CAPITAL": actual,
                "RECOGNIZED_ASSETS": assets,
                "REGISTERED_CAPITAL": 100.0,
                "CORE_T1_CAPITAL": core_share * 100.0,
                "CORE_T2_CAPITAL": 0.0,
            }
            for code, value in values.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元",
                })
        chart, latest_period = build_capital_efficiency_bubble_chart(
            pd.DataFrame(rows),
            ["2025Q4"],
            {company: "#00338D" for company in companies},
            zoom_to_overlap_region=True,
            panel_label="局部样本放大",
            chart_width=420,
            chart_height=430,
            show_company_legend=False,
            show_axis_titles=False,
        )
        self.assertEqual(latest_period, "2025Q4")
        spec = chart.to_dict(validate=True)
        bubble_layer = next(
            layer for layer in spec["layer"]
            if layer["mark"]["type"] == "circle"
        )
        dataset_name = bubble_layer["data"]["name"]
        plotted_companies = {
            row["公司"] for row in spec["datasets"][dataset_name]
        }
        self.assertEqual(plotted_companies, set(companies))
        x_domain = bubble_layer["encoding"]["x"]["scale"]["domain"]
        y_domain = bubble_layer["encoding"]["y"]["scale"]["domain"]
        visible_companies = {
            row["公司"]
            for row in spec["datasets"][dataset_name]
            if x_domain[0] <= row["实际资本/注册资本"] <= x_domain[1]
            and y_domain[0] <= row["核心资本/注册资本"] <= y_domain[1]
        }
        self.assertEqual(visible_companies, {"建信人寿", "农银人寿", "中邮人寿"})
        selected_x = [companies[name][0] / 100 for name in visible_companies]
        selected_y = [companies[name][2] for name in visible_companies]
        self.assertAlmostEqual(
            x_domain[1] - x_domain[0],
            (max(selected_x) - min(selected_x)) * 1.35,
        )
        self.assertAlmostEqual(
            y_domain[1] - y_domain[0],
            (max(selected_y) - min(selected_y)) * 1.35,
        )
        self.assertIsNone(bubble_layer["encoding"]["color"]["legend"])
        self.assertEqual(
            bubble_layer["encoding"]["size"]["legend"]["title"],
            ["气泡大小代表", "认可资产金额（亿元）"],
        )
        self.assertEqual(bubble_layer["encoding"]["x"]["axis"]["title"], "")
        self.assertEqual(bubble_layer["encoding"]["y"]["axis"]["title"], "")
        self.assertEqual(
            bubble_layer["encoding"]["y"]["axis"]["labelExpr"],
            "format(datum.value * 100, '.1f')",
        )
        self.assertFalse(any(
            layer["mark"]["type"] == "rule" for layer in spec["layer"]
        ))
        self.assertEqual(spec["width"], 420)
        self.assertEqual(spec["height"], 430)
        self.assertEqual(spec["title"]["fontSize"], 12)
        self.assertEqual(spec["title"]["text"], "局部样本放大")
        self.assertEqual(spec["title"]["anchor"], "start")

    def test_capital_efficiency_overlap_zoom_expands_to_dense_cluster(self):
        companies = {
            "密集甲": (10.0, 0.500),
            "密集乙": (10.1, 0.505),
            "密集丙": (10.2, 0.508),
            "远端甲": (14.0, 0.700),
            "远端乙": (16.0, 0.800),
        }
        rows = []
        for company, (x_percent, core_share) in companies.items():
            actual = x_percent
            values = {
                "ACTUAL_CAPITAL": actual,
                "RECOGNIZED_ASSETS": 100.0,
                "REGISTERED_CAPITAL": 100.0,
                "CORE_T1_CAPITAL": 100.0 * core_share,
                "CORE_T2_CAPITAL": 0.0,
            }
            for code, value in values.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元",
                })
        chart, _ = build_capital_efficiency_bubble_chart(
            pd.DataFrame(rows),
            ["2025Q4"],
            {company: "#00338D" for company in companies},
            zoom_to_overlap_region=True,
        )
        spec = chart.to_dict(validate=True)
        bubble_layer = next(
            layer for layer in spec["layer"]
            if layer["mark"]["type"] == "circle"
        )
        dataset_name = bubble_layer["data"]["name"]
        plotted_companies = {
            row["公司"] for row in spec["datasets"][dataset_name]
        }
        self.assertEqual(plotted_companies, set(companies))
        x_domain = bubble_layer["encoding"]["x"]["scale"]["domain"]
        y_domain = bubble_layer["encoding"]["y"]["scale"]["domain"]
        visible_companies = {
            row["公司"]
            for row in spec["datasets"][dataset_name]
            if x_domain[0] <= row["实际资本/注册资本"] <= x_domain[1]
            and y_domain[0] <= row["核心资本/注册资本"] <= y_domain[1]
        }
        self.assertEqual(visible_companies, {"密集甲", "密集乙", "密集丙"})

    def test_capital_efficiency_overlap_zoom_keeps_full_scale_when_points_are_spread(self):
        rows = []
        companies = {
            "分散甲": (10.0, 0.50),
            "分散乙": (20.0, 0.70),
            "分散丙": (30.0, 0.90),
        }
        for company, (actual, core_share) in companies.items():
            for code, value in {
                "ACTUAL_CAPITAL": actual,
                "RECOGNIZED_ASSETS": 100.0,
                "REGISTERED_CAPITAL": 100.0,
                "CORE_T1_CAPITAL": 100.0 * core_share,
                "CORE_T2_CAPITAL": 0.0,
            }.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元",
                })
        frame = pd.DataFrame(rows)
        colors = {company: "#00338D" for company in companies}
        full_chart, _ = build_capital_efficiency_bubble_chart(
            frame, ["2025Q4"], colors,
        )
        local_chart, _ = build_capital_efficiency_bubble_chart(
            frame, ["2025Q4"], colors, zoom_to_overlap_region=True,
        )

        def bubble_domains(chart):
            spec = chart.to_dict(validate=True)
            bubble_layer = next(
                layer for layer in spec["layer"]
                if layer["mark"]["type"] == "circle"
            )
            return (
                bubble_layer["encoding"]["x"]["scale"]["domain"],
                bubble_layer["encoding"]["y"]["scale"]["domain"],
            )

        self.assertEqual(bubble_domains(local_chart), bubble_domains(full_chart))

    def test_capital_efficiency_combined_panels_share_exact_plot_height(self):
        rows = []
        for company, (actual, core_share) in {
            "甲": (10.0, 0.50),
            "乙": (10.2, 0.51),
            "丙": (16.0, 0.75),
        }.items():
            for code, value in {
                "ACTUAL_CAPITAL": actual,
                "RECOGNIZED_ASSETS": 100.0,
                "REGISTERED_CAPITAL": 100.0,
                "CORE_T1_CAPITAL": 100.0 * core_share,
                "CORE_T2_CAPITAL": 0.0,
            }.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元",
                })
        frame = pd.DataFrame(rows)
        colors = {"甲": "#00338D", "乙": "#1E49E2", "丙": "#7213EA"}
        full_chart, _ = build_capital_efficiency_bubble_chart(
            frame,
            ["2025Q4"],
            colors,
            chart_height=430,
            show_company_legend=True,
            apply_theme=False,
        )
        local_chart, _ = build_capital_efficiency_bubble_chart(
            frame,
            ["2025Q4"],
            colors,
            zoom_to_overlap_region=True,
            chart_height=430,
            show_axis_titles=False,
            show_company_legend=True,
            apply_theme=False,
        )
        spec = combine_capital_efficiency_bubble_charts(
            full_chart,
            local_chart,
        ).to_dict(validate=True)
        self.assertEqual(
            [panel["height"] for panel in spec["hconcat"]],
            [430, 430],
        )
        self.assertEqual(spec["resolve"]["scale"]["x"], "independent")
        self.assertEqual(spec["resolve"]["scale"]["y"], "independent")
        for panel in spec["hconcat"]:
            bubbles = next(
                layer for layer in panel["layer"]
                if layer["mark"]["type"] == "circle"
            )
            self.assertEqual(
                bubbles["encoding"]["color"]["legend"]["orient"],
                "right",
            )

    def test_capital_efficiency_linked_zoom_tracks_company_and_supports_fixed_domain(self):
        rows = []
        companies = {
            "追踪公司": (16.0, 0.71),
            "邻近甲": (15.4, 0.69),
            "邻近乙": (14.8, 0.73),
            "邻近丙": (13.9, 0.66),
            "远端甲": (9.0, 0.50),
            "远端乙": (10.0, 0.80),
        }
        for company, (actual, core_share) in companies.items():
            for code, value in {
                "ACTUAL_CAPITAL": actual,
                "RECOGNIZED_ASSETS": 100.0,
                "REGISTERED_CAPITAL": 100.0,
                "CORE_T1_CAPITAL": 100.0 * core_share,
                "CORE_T2_CAPITAL": 0.0,
            }.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元",
                })
        frame = pd.DataFrame(rows)
        colors = {company: "#00338D" for company in companies}
        initial_domain, latest_period = capital_efficiency_bubble_default_domain(
            frame,
            ["2025Q4"],
            colors,
            "追踪公司",
        )
        self.assertEqual(latest_period, "2025Q4")
        self.assertLessEqual(initial_domain["x"][0], 0.16)
        self.assertGreaterEqual(initial_domain["x"][1], 0.16)
        self.assertLessEqual(initial_domain["y"][0], 0.71)
        self.assertGreaterEqual(initial_domain["y"][1], 0.71)

        full_chart, _ = build_capital_efficiency_bubble_chart(
            frame,
            ["2025Q4"],
            colors,
            "追踪公司",
            apply_theme=False,
        )
        linked_local, _ = build_capital_efficiency_bubble_chart(
            frame,
            ["2025Q4"],
            colors,
            "追踪公司",
            zoom_to_overlap_region=True,
            linked_selection_name="capital_efficiency_zoom",
            apply_theme=False,
        )
        linked = combine_capital_efficiency_bubble_charts(
            full_chart,
            linked_local,
            selection_name="capital_efficiency_zoom",
            selection_domain=initial_domain,
        ).to_dict(validate=True)
        selection = next(
            param for param in linked["params"]
            if param["name"] == "capital_efficiency_zoom"
        )
        self.assertEqual(selection["value"], initial_domain)
        self.assertFalse(selection["select"]["clear"])
        local_bubbles = next(
            layer for layer in linked["hconcat"][1]["layer"]
            if layer["mark"]["type"] == "circle"
        )
        self.assertEqual(
            local_bubbles["encoding"]["x"]["scale"]["domain"],
            {"param": "capital_efficiency_zoom", "encoding": "x"},
        )
        self.assertEqual(
            local_bubbles["encoding"]["y"]["scale"]["domain"],
            {"param": "capital_efficiency_zoom", "encoding": "y"},
        )

        locked_local, _ = build_capital_efficiency_bubble_chart(
            frame,
            ["2025Q4"],
            colors,
            "追踪公司",
            zoom_to_overlap_region=True,
            zoom_domain_override=initial_domain,
            apply_theme=False,
        )
        locked = combine_capital_efficiency_bubble_charts(
            full_chart,
            locked_local,
        ).to_dict(validate=True)
        self.assertNotIn("params", locked)
        locked_bubbles = next(
            layer for layer in locked["hconcat"][1]["layer"]
            if layer["mark"]["type"] == "circle"
        )
        self.assertEqual(
            locked_bubbles["encoding"]["x"]["scale"]["domain"],
            initial_domain["x"],
        )
        self.assertEqual(
            locked_bubbles["encoding"]["y"]["scale"]["domain"],
            initial_domain["y"],
        )

    def test_capital_efficiency_dragging_stays_client_side(self):
        renderer = getattr(
            _render_capital_efficiency_bubble_fragment,
            "__wrapped__",
            _render_capital_efficiency_bubble_fragment,
        )
        source = inspect.getsource(renderer)
        self.assertNotIn("on_select=", source)
        self.assertNotIn("selection_mode=", source)
        self.assertIn('f"capital_efficiency_zoom_', source)
        style_source = inspect.getsource(_report_style)
        self.assertIn(".capital_efficiency_zoom_brush", style_source)
        self.assertIn(".capital_efficiency_zoom_brush_bg", style_source)

    def test_capital_efficiency_bubble_uses_page_capped_responsive_width(self):
        renderer = getattr(
            _render_capital_efficiency_bubble_fragment,
            "__wrapped__",
            _render_capital_efficiency_bubble_fragment,
        )
        source = inspect.getsource(renderer)
        self.assertIn("chart_width=BUBBLE_FULL_PANEL_WIDTH", source)
        self.assertIn("chart_width=BUBBLE_ZOOM_PANEL_WIDTH", source)
        self.assertIn('width="content"', source)
        self.assertLess(
            BUBBLE_FULL_PANEL_WIDTH + BUBBLE_ZOOM_PANEL_WIDTH,
            1120,
        )

    def test_core_capital_to_registered_capital_company_bars_share_axis_and_period_colors(self):
        code = "CORE_CAPITAL_TO_REGISTERED_CAPITAL"
        periods = ["2024Q4", "2025Q2", "2025Q4"]
        frame = pd.DataFrame([
            {
                "公司": company,
                "报告期": period,
                "指标编码": code,
                "指标名称": "核心资本/注册资本",
                "数值": value,
                "单位": "倍",
            }
            for company, values in (("甲", (0.30, 0.35, 0.40)), ("乙", (0.50, 0.60, 0.65)))
            for period, value in zip(periods, values)
        ])
        shared_domain = metric_bar_axis_domain(frame, code, periods)
        self.assertAlmostEqual(shared_domain[0], 0.0)
        self.assertAlmostEqual(shared_domain[1], 65.0 * 1.12)

        for company in ("甲", "乙"):
            spec = build_company_period_bar_chart(
                frame[frame["公司"].eq(company)],
                code,
                periods,
                shared_y_domain=shared_domain,
            ).to_dict(validate=True)
            self.assertEqual(spec["layer"][0]["mark"]["type"], "bar")
            self.assertEqual(
                spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
                list(shared_domain),
            )
            self.assertEqual(
                spec["layer"][1]["encoding"]["y"]["scale"]["domain"],
                list(shared_domain),
            )
            self.assertEqual(
                spec["layer"][0]["encoding"]["color"]["scale"]["domain"],
                periods,
            )
            self.assertEqual(
                spec["layer"][0]["encoding"]["color"]["scale"]["range"],
                ["#ACEAFF", "#00B8F5", "#B497FF"],
            )
            dataset = spec["datasets"][spec["data"]["name"]]
            self.assertTrue(all(row["单位"] == "%" for row in dataset))
            self.assertTrue(all(row["数值标签"].endswith("%") for row in dataset))

    def test_policy_surplus_core_ratio_company_bars_share_axis_and_period_colors(self):
        code = "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"
        periods = ["2024Q4", "2025Q2", "2025Q4"]
        frame = pd.DataFrame([
            {
                "公司": company,
                "报告期": period,
                "指标编码": code,
                "指标名称": "计入核心资本的保单未来盈余/核心资本的比例",
                "数值": value,
                "单位": "倍",
            }
            for company, values in (
                ("甲", (0.20, 0.30, 0.40)),
                ("乙", (0.35, 0.45, 0.55)),
            )
            for period, value in zip(periods, values)
        ])
        shared_domain = metric_bar_axis_domain(frame, code, periods)
        self.assertAlmostEqual(shared_domain[0], 0.0)
        self.assertAlmostEqual(shared_domain[1], 55.0 * 1.12)

        for company in ("甲", "乙"):
            spec = build_company_period_bar_chart(
                frame[frame["公司"].eq(company)],
                code,
                periods,
                shared_y_domain=shared_domain,
            ).to_dict(validate=True)
            self.assertEqual(
                spec["layer"][0]["encoding"]["x"]["field"],
                "报告期",
            )
            self.assertEqual(
                spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
                list(shared_domain),
            )
            self.assertEqual(
                spec["layer"][1]["encoding"]["y"]["scale"]["domain"],
                list(shared_domain),
            )
            self.assertEqual(
                spec["layer"][0]["encoding"]["color"]["scale"]["domain"],
                periods,
            )
            self.assertEqual(
                spec["layer"][0]["encoding"]["color"]["scale"]["range"],
                ["#ACEAFF", "#00B8F5", "#B497FF"],
            )
            dataset = spec["datasets"][spec["data"]["name"]]
            self.assertTrue(all(row["单位"] == "%" for row in dataset))
            self.assertTrue(all(row["数值标签"].endswith("%") for row in dataset))

    def test_solvency_ratio_combo_uses_one_axis_and_integer_labels(self):
        rows = []
        for company, core, combined in [
            ("甲", 119.6, 181.4),
            ("乙", 89.5, 210.6),
        ]:
            for period, shift in [
                ("2024Q4", 0.0),
                ("2025Q2", 3.0),
                ("2025Q4", 6.0),
            ]:
                rows.extend([
                    {
                        "公司": company,
                        "报告期": period,
                        "指标编码": "CORE_SOLVENCY_RATIO",
                        "指标名称": "核心偿付能力充足率",
                        "数值": core + shift,
                        "单位": "%",
                    },
                    {
                        "公司": company,
                        "报告期": period,
                        "指标编码": "COMBINED_SOLVENCY_RATIO",
                        "指标名称": "综合偿付能力充足率",
                        "数值": combined + shift,
                        "单位": "%",
                    },
                ])
        frame = pd.DataFrame(rows)
        periods = ["2024Q4", "2025Q2", "2025Q4"]
        shared_domain = solvency_ratio_axis_domain(frame, periods)
        self.assertEqual(shared_domain, (0.0, 250.0))

        specs = []
        for company in ("甲", "乙"):
            chart = build_solvency_ratio_combo_chart(
                frame[frame["公司"].eq(company)],
                periods,
                shared_y_domain=shared_domain,
            )
            specs.append(chart.to_dict(validate=True))
        for spec in specs:
            self.assertEqual(len(spec["layer"]), 5)
            self.assertEqual(spec["layer"][0]["mark"]["type"], "bar")
            self.assertFalse(spec["layer"][0]["encoding"]["y"]["axis"]["labels"])
            self.assertFalse(spec["layer"][0]["encoding"]["y"]["axis"]["ticks"])
            self.assertNotIn("color", spec["layer"][0]["mark"])
            self.assertEqual(
                spec["layer"][0]["encoding"]["color"]["scale"]["range"],
                ["#ACEAFF", "#00B8F5", "#B497FF"],
            )
            self.assertEqual(spec["layer"][2]["mark"]["type"], "line")
            self.assertEqual(spec["layer"][2]["mark"]["color"], "#FD349C")
            self.assertEqual(spec["layer"][1]["mark"]["color"], "#000000")
            self.assertEqual(spec["layer"][4]["mark"]["color"], "#000000")
            self.assertEqual(
                spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
                [0.0, 250.0],
            )
            self.assertEqual(
                spec["layer"][2]["encoding"]["y"]["scale"]["domain"],
                [0.0, 250.0],
            )
            self.assertEqual(
                spec["layer"][1]["encoding"]["text"]["field"],
                "数值标签",
            )
            self.assertEqual(
                spec["layer"][4]["encoding"]["text"]["field"],
                "数值标签",
            )

    def test_step7_combo_lines_convert_multiple_units_to_percent(self):
        rows = []
        for code, value, unit in (
            ("CORE_SOLVENCY_RATIO", 0.8, "倍"),
            ("COMBINED_SOLVENCY_RATIO", 1.2, "倍"),
            ("ACTUAL_CAPITAL", 120.0, "亿元"),
            ("MINIMUM_CAPITAL", 70.0, "亿元"),
        ):
            rows.append({
                "公司": "甲",
                "报告期": "2025Q4",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "单位": unit,
            })
        frame = pd.DataFrame(rows)

        solvency_spec = build_solvency_ratio_combo_chart(
            frame,
            ["2025Q4"],
        ).to_dict(validate=True)
        core_data_name = solvency_spec["layer"][2]["data"]["name"]
        core_row = solvency_spec["datasets"][core_data_name][0]
        self.assertEqual(core_row["数值"], 80.0)
        self.assertEqual(core_row["单位"], "%")
        self.assertEqual(core_row["数值标签"], "80%")

        capital_spec = build_capital_ratio_combo_chart(
            frame,
            {"实际资本": ("ACTUAL_CAPITAL",), "最低资本": ("MINIMUM_CAPITAL",)},
            ["2025Q4"],
            "综合充足率变化",
            ratio_code="COMBINED_SOLVENCY_RATIO",
            ratio_label="综合偿付能力充足率（%）",
        ).to_dict(validate=True)
        line_data_name = capital_spec["vconcat"][0]["data"]["name"]
        line_row = capital_spec["datasets"][line_data_name][0]
        self.assertEqual(line_row["线值"], 120.0)
        self.assertEqual(line_row["线标签"], "120.0%")

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
        self.assertEqual(panel_bar_color["scale"]["range"], ["#ACEAFF", "#00B8F5"])
        self.assertIsNone(panel_bar_color["legend"])
        self.assertEqual(
            panel_spec["spec"]["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"],
            0.5,
        )
        panel_y_scale = panel_spec["spec"]["layer"][0]["encoding"]["y"]["scale"]
        panel_y_axis = panel_spec["spec"]["layer"][0]["encoding"]["y"]["axis"]
        self.assertFalse(panel_y_axis["labels"])
        self.assertFalse(panel_y_axis["ticks"])
        self.assertTrue(panel_y_scale["zero"])
        self.assertEqual(panel_y_scale["domain"][0], 0)
        self.assertNotIn("padding", panel_y_scale)
        self.assertEqual(panel_spec["resolve"]["scale"]["y"], "shared")

        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["type"], "line")
        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["strokeWidth"], 2.6)
        self.assertEqual(panel_spec["spec"]["layer"][2]["mark"]["type"], "point")
        self.assertNotIn("stroke", panel_spec["spec"]["layer"][2]["mark"])
        self.assertNotIn("stroke", panel_spec["spec"]["layer"][3]["mark"])
        effect_spec = diversification.to_dict(validate=True)
        self.assertFalse(effect_spec["spec"]["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertEqual(
            effect_spec["spec"]["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"],
            0.5,
        )
        self.assertFalse(panel_spec["config"]["axis"]["grid"])
        self.assertEqual(panel_spec["config"]["axis"]["tickColor"], "#D0D5DD")
        self.assertNotIn("aggregate", panel_spec["spec"]["layer"][0]["encoding"]["y"])
        tracked_frame = panel_spec["spec"]["layer"][-1]
        self.assertEqual(tracked_frame["mark"]["type"], "rect")
        self.assertEqual(tracked_frame["mark"]["stroke"], "#00338D")
        self.assertEqual(tracked_frame["mark"]["strokeOpacity"], 0.35)
        self.assertEqual(tracked_frame["mark"]["strokeWidth"], 1.5)
        self.assertEqual(tracked_frame["mark"]["fill"], "#00338D")
        self.assertEqual(tracked_frame["mark"]["fillOpacity"], 0.03)
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

    def test_risk_asset_bubble_uses_adaptive_units_padding_and_asset_size(self):
        rows = []
        for company, interest, equity, assets in [
            ("甲", 0.012, 0.0002, 1_000.0),
            ("乙", 0.018, 0.0011, 2_000.0),
        ]:
            for period, shift in [("2025Q2", -0.005), ("2025Q4", 0.0)]:
                rows.extend([
                    {
                        "公司": company,
                        "同业分类": "主体样本",
                        "报告期": period,
                        "指标编码": "INTEREST_RATE_RISK_TO_ASSETS",
                        "指标名称": "利率风险/认可资产",
                        "数值": interest + shift,
                        "单位": "倍",
                    },
                    {
                        "公司": company,
                        "同业分类": "主体样本",
                        "报告期": period,
                        "指标编码": "EQUITY_RISK_TO_ASSETS",
                        "指标名称": "权益价格风险/认可资产",
                        "数值": equity + shift,
                        "单位": "倍",
                    },
                    {
                        "公司": company,
                        "同业分类": "主体样本",
                        "报告期": period,
                        "指标编码": "RECOGNIZED_ASSETS",
                        "指标名称": "认可资产",
                        "数值": assets,
                        "单位": "亿元",
                    },
                ])
        chart, latest_period = build_matrix_chart(
            pd.DataFrame(rows),
            "INTEREST_RATE_RISK_TO_ASSETS",
            "EQUITY_RISK_TO_ASSETS",
            ["2025Q2", "2025Q4"],
            "利率与权益价格风险占认可资产率气泡图",
            percentage_axes=True,
            bubble_size_code="RECOGNIZED_ASSETS",
            bubble_size_label="认可资产",
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["layer"][0]["data"]["name"]]
        self.assertEqual(latest_period, "2025Q4")
        self.assertEqual({row["公司"] for row in dataset}, {"甲", "乙"})
        latest_values = {
            row["公司"]: row["INTEREST_RATE_RISK_TO_ASSETS"]
            for row in dataset
        }
        self.assertAlmostEqual(latest_values["甲"], 1.2)
        self.assertAlmostEqual(latest_values["乙"], 1.8)
        latest_y_values = {
            row["公司"]: row["EQUITY_RISK_TO_ASSETS"]
            for row in dataset
        }
        self.assertAlmostEqual(latest_y_values["甲"], 0.2)
        self.assertAlmostEqual(latest_y_values["乙"], 1.1)
        self.assertEqual(spec["layer"][0]["encoding"]["size"]["field"], "气泡大小")
        self.assertEqual(
            spec["layer"][0]["encoding"]["size"]["legend"]["title"],
            ["气泡大小代表", "认可资产金额（亿元）"],
        )
        self.assertEqual(
            {row["公司"]: row["气泡大小"] for row in dataset},
            {"甲": 1_000.0, "乙": 2_000.0},
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["x"]["axis"]["format"],
            ".1~f",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["y"]["axis"]["format"],
            ".1~f",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["x"]["axis"]["tickCount"],
            6,
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["y"]["axis"]["tickCount"],
            6,
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["x"]["title"],
            "利率风险/认可资产（%）",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["y"]["title"],
            "权益价格风险/认可资产（‰）",
        )
        self.assertTrue(spec["layer"][0]["mark"]["clip"])
        x_domain = spec["layer"][0]["encoding"]["x"]["scale"]["domain"]
        y_domain = spec["layer"][0]["encoding"]["y"]["scale"]["domain"]
        self.assertLess(x_domain[0], min(latest_values.values()) - 0.10)
        self.assertGreater(x_domain[1], max(latest_values.values()) + 0.10)
        self.assertLess(y_domain[0], min(latest_y_values.values()) - 0.10)
        self.assertGreater(y_domain[1], max(latest_y_values.values()) + 0.10)
        self.assertEqual(
            spec["layer"][0]["encoding"]["tooltip"][2]["format"],
            ".2f",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["tooltip"][2]["title"],
            "利率风险/认可资产（%）",
        )
        self.assertEqual(
            spec["layer"][0]["encoding"]["tooltip"][3]["title"],
            "权益价格风险/认可资产（‰）",
        )

    def test_requested_risk_scatters_use_linked_full_and_local_views(self):
        rows = []
        companies = {
            "聚集甲": (0.100, 0.200, 1_000.0),
            "聚集乙": (0.101, 0.201, 1_100.0),
            "聚集丙": (0.102, 0.198, 1_200.0),
            "聚集丁": (0.103, 0.199, 1_300.0),
            "远端甲": (0.200, 0.400, 2_000.0),
            "远端乙": (0.300, 0.100, 3_000.0),
        }
        for company, (interest, equity, assets) in companies.items():
            for code, value in {
                "INTEREST_RATE_RISK_TO_ASSETS": interest,
                "EQUITY_RISK_TO_ASSETS": equity,
                "RECOGNIZED_ASSETS": assets,
            }.items():
                rows.append({
                    "公司": company,
                    "同业分类": "测试样本",
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": "亿元" if code == "RECOGNIZED_ASSETS" else "倍",
                })
        frame = pd.DataFrame(rows)
        colors = {company: "#00338D" for company in companies}
        selection_domain, latest_period = risk_ratio_scatter_default_domain(
            frame,
            "INTEREST_RATE_RISK_TO_ASSETS",
            "EQUITY_RISK_TO_ASSETS",
            ["2025Q4"],
            "利率与权益价格风险占认可资产率气泡图",
            company_colors=colors,
            bubble_size_code="RECOGNIZED_ASSETS",
            bubble_size_label="认可资产",
        )
        self.assertEqual(latest_period, "2025Q4")
        self.assertLess(selection_domain["x"][1], 20.0)
        full_chart, _ = build_matrix_chart(
            frame,
            "INTEREST_RATE_RISK_TO_ASSETS",
            "EQUITY_RISK_TO_ASSETS",
            ["2025Q4"],
            "利率与权益价格风险占认可资产率气泡图",
            company_colors=colors,
            percentage_axes=True,
            panel_label="全样本图",
            chart_width=RISK_SCATTER_FULL_PANEL_WIDTH,
            chart_height=430,
            linked_selection_name="risk_ratio_zoom_test",
            apply_theme=False,
            bubble_size_code="RECOGNIZED_ASSETS",
            bubble_size_label="认可资产",
        )
        local_chart, _ = build_matrix_chart(
            frame,
            "INTEREST_RATE_RISK_TO_ASSETS",
            "EQUITY_RISK_TO_ASSETS",
            ["2025Q4"],
            "利率与权益价格风险占认可资产率气泡图",
            company_colors=colors,
            percentage_axes=True,
            zoom_to_overlap_region=True,
            panel_label="局部样本放大",
            chart_width=RISK_SCATTER_ZOOM_PANEL_WIDTH,
            chart_height=430,
            linked_selection_name="risk_ratio_zoom_test",
            apply_theme=False,
            bubble_size_code="RECOGNIZED_ASSETS",
            bubble_size_label="认可资产",
            show_size_legend=False,
        )
        linked = combine_linked_matrix_charts(
            full_chart,
            local_chart,
            selection_name="risk_ratio_zoom_test",
            selection_domain=selection_domain,
        ).to_dict(validate=True)
        self.assertEqual([panel["title"]["text"] for panel in linked["hconcat"]], [
            "全样本图",
            "局部样本放大",
        ])
        selection = next(
            param for param in linked["params"]
            if param["name"] == "risk_ratio_zoom_test"
        )
        self.assertEqual(selection["value"], selection_domain)
        local_points = next(
            layer for layer in linked["hconcat"][1]["layer"]
            if layer["mark"]["type"] == "circle"
        )
        self.assertEqual(
            [layer["mark"]["type"] for layer in linked["hconcat"][0]["layer"]],
            ["circle", "rect", "rule", "rule"],
        )
        self.assertEqual(
            [layer["mark"]["type"] for layer in linked["hconcat"][1]["layer"]],
            ["rect", "circle"],
        )
        self.assertEqual(
            local_points["encoding"]["x"]["scale"]["domain"],
            {"param": "risk_ratio_zoom_test", "encoding": "x"},
        )
        self.assertEqual(
            linked["hconcat"][0]["layer"][0]["encoding"]["size"]["field"],
            "气泡大小",
        )
        self.assertEqual(linked["resolve"]["scale"]["size"], "shared")

    def test_risk_scatter_dragging_stays_browser_side_and_print_hides_brush(self):
        renderer = getattr(
            _render_risk_ratio_scatter_fragment,
            "__wrapped__",
            _render_risk_ratio_scatter_fragment,
        )
        source = inspect.getsource(renderer)
        self.assertNotIn("on_select=", source)
        self.assertNotIn("selection_mode=", source)
        self.assertIn('f"risk_ratio_zoom_', source)
        self.assertIn("chart_width=RISK_SCATTER_FULL_PANEL_WIDTH", source)
        self.assertIn("chart_width=RISK_SCATTER_ZOOM_PANEL_WIDTH", source)
        self.assertIn('width="content"', source)
        self.assertLess(
            RISK_SCATTER_FULL_PANEL_WIDTH + RISK_SCATTER_ZOOM_PANEL_WIDTH,
            1120,
        )
        style_source = inspect.getsource(_report_style)
        self.assertIn(".risk_ratio_zoom_brush", style_source)
        self.assertIn(".risk_ratio_zoom_brush_bg", style_source)

    def test_policy_surplus_core_ratio_labels_only_values_above_point_four(self):
        rows = []
        values = {
            ("甲", "2025Q2"): 0.39,
            ("甲", "2025Q4"): 0.40,
            ("乙", "2025Q2"): 0.41,
            ("乙", "2025Q4"): 0.55,
        }
        for (company, period), value in values.items():
            rows.append({
                "公司": company,
                "报告期": period,
                "指标编码": "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
                "指标名称": "计入核心资本的保单未来盈余/核心资本的比例",
                "数值": value,
                "单位": "倍",
            })
        chart = build_single_metric_trend_chart(
            pd.DataFrame(rows),
            "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
            ["2025Q2", "2025Q4"],
            {"甲": "#00338D", "乙": "#00B8F5"},
            "甲",
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        flags = {round(float(row["数值"]), 6): row for row in dataset if row.get("数值") is not None}
        self.assertFalse(flags[39.0]["显示标签"])
        self.assertFalse(flags[40.0]["显示标签"])
        self.assertTrue(flags[41.0]["显示标签"])
        self.assertTrue(flags[55.0]["显示标签"])
        self.assertEqual(flags[41.0]["数值标签"], "41.00%")
        self.assertTrue(all(row["单位"] == "%" for row in flags.values()))
        self.assertTrue(all(not row["是否全局最大"] for row in flags.values()))
        self.assertTrue(all(not row["是否全局最小"] for row in flags.values()))

    def test_policy_surplus_unavailable_details_include_company_period_and_reason(self):
        frame = pd.DataFrame([
            {
                "公司": "甲人寿",
                "报告期": "2026Q1",
                "指标编码": "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
                "指标名称": "计入核心资本的保单未来盈余/核心资本的比例",
                "数值": None,
                "披露状态": "无法计算",
                "备注": "无法计算：计入核心二级资本的保单未来盈余未披露",
            },
            {
                "公司": "乙人寿",
                "报告期": "2026Q1",
                "指标编码": "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
                "指标名称": "计入核心资本的保单未来盈余/核心资本的比例",
                "数值": 0.2,
                "披露状态": "已计算",
                "备注": "",
            },
        ])

        self.assertEqual(
            _unavailable_metric_details(
                frame,
                "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
                ["2026Q1"],
            ),
            [("甲人寿", "2026Q1", "计入核心二级资本的保单未来盈余未披露")],
        )

    def test_business_combo_missing_notes_group_companies_periods_and_metrics(self):
        companies = ["工银安盛", "建信人寿", "交银人寿", "农银人寿", "招商信诺", "中银三星", "中邮人寿"]
        details = [
            "2025Q1·本季度末数：签单保费",
            "2025Q1·本季度末数：新业务利润率",
            "2025Q3·本季度末数：签单保费",
            "2025Q3·本季度末数：新业务利润率",
        ]
        notes = _missing_disclosure_note_lines(
            "签单保费与新业务利润率",
            companies,
            {company: details for company in companies},
        )
        self.assertEqual(notes, [
            "注：未披露——工银安盛、建信人寿、交银人寿、农银人寿、招商信诺、中银三星、中邮人寿："
            "2025Q1、2025Q3均未披露签单保费及新业务利润率。"
        ])

        value_notes = _missing_disclosure_note_lines(
            "新业务价值与新业务价值率",
            companies[:2],
            {company: [
                "2025Q1·本季度末数：新业务价值",
                "2025Q1·本季度末数：签单保费（无法计算新业务价值率）",
            ] for company in companies[:2]},
        )
        self.assertEqual(value_notes, [
            "注：未披露——工银安盛、建信人寿：2025Q1均未披露新业务价值及签单保费（新业务价值率因此无法计算）。"
        ])

    def test_quality_chart_missing_notes_are_compact(self):
        companies = ["甲公司", "乙公司", "丙公司"]
        surrender = _missing_disclosure_note_lines(
            "综合退保率",
            companies,
            {company: ["2025Q1", "2025Q3"] for company in companies},
        )
        self.assertEqual(surrender, [
            "注：未披露——甲公司、乙公司、丙公司：2025Q1、2025Q3均未披露综合退保率。"
        ])

        radar_metrics = [
            "净资产收益率", "总资产收益率", "投资收益率（当季数）",
            "综合投资收益率（当季数）", "近三年平均投资收益率", "近三年平均综合投资收益率",
        ]
        radar = _missing_disclosure_note_lines(
            "投资质量六指标雷达图",
            companies,
            {company: radar_metrics for company in companies},
        )
        self.assertEqual(radar, [
            "注：未披露——甲公司、乙公司、丙公司：六项投资质量指标均未披露。"
        ])

        cumulative_details = [
            "2025Q1·累计：投资收益率",
            "2025Q1·累计：综合投资收益率",
            "2025Q3·累计：投资收益率",
            "2025Q3·累计：综合投资收益率",
        ]
        cumulative = _missing_disclosure_note_lines(
            "累计投资收益率与累计综合投资收益率",
            companies,
            {company: cumulative_details for company in companies},
        )
        self.assertEqual(cumulative, [
            "注：未披露——甲公司、乙公司、丙公司：2025Q1、2025Q3均未披露投资收益率及综合投资收益率。"
        ])

        three_year = _missing_disclosure_note_lines(
            "近三年平均投资收益率与综合投资收益率",
            companies,
            {
                "甲公司": ["2024Q4·近三年平均：投资收益率"],
                "乙公司": ["2025Q2·近三年平均：综合投资收益率"],
            },
        )
        self.assertEqual(three_year, [
            "注：未披露——甲公司、乙公司部分报告期的近三年平均投资收益率或综合投资收益率未披露。"
        ])

    def test_policy_surplus_ratio_note_explains_core_only_and_four_levels(self):
        codes = (
            "POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2",
            "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2",
        )
        core_only = ("中邮人寿", "中银三星", "建信人寿", "招商信诺")
        all_four = ("农银人寿", "交银人寿", "工银安盛")
        rows, ratios = [], []
        for company in (*core_only, *all_four):
            ratios.append({"公司": company, "报告期": "2025Q4", "数值": 0.1})
            for code, value in zip(codes, (10, 0, 0, 0) if company in core_only else (10, 0, 5, 2)):
                rows.append({
                    "公司": company, "报告期": "2025Q4", "期间口径": "本季度末数",
                    "指标编码": code, "数值": value, "披露状态": "已披露",
                    "来源类型": "外部数据集",
                })
        note = _policy_surplus_coverage_note(pd.DataFrame(rows), pd.DataFrame(ratios))
        self.assertIn("中邮人寿、中银三星、建信人寿、招商信诺：分子仅计入核心一级资本", note)
        self.assertIn("农银人寿、交银人寿、工银安盛：分子计入四类", note)

        # A source PDF's explicit zero is a disclosed level, unlike the wide table's zero placeholder.
        for row in rows:
            if row["公司"] == "中邮人寿" and row["指标编码"] != codes[0]:
                row["披露状态"] = "已披露为0"
        note = _policy_surplus_coverage_note(pd.DataFrame(rows), pd.DataFrame(ratios))
        self.assertNotIn("中邮人寿：分子仅计入核心一级资本", note)

    def test_core_policy_surplus_ratio_note_explains_core_t1_only_and_both_levels(self):
        codes = (
            "POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2",
            "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2",
        )
        core_t1_only = ("中邮人寿", "中银三星", "建信人寿", "招商信诺")
        both_core_levels = ("农银人寿", "交银人寿", "工银安盛")
        rows, ratios = [], []
        for company in (*core_t1_only, *both_core_levels):
            ratios.append({"公司": company, "报告期": "2025Q4", "数值": 0.1})
            values = (10, 0, 0, 0) if company in core_t1_only else (10, 0, 5, 2)
            for code, value in zip(codes, values):
                rows.append({
                    "公司": company, "报告期": "2025Q4", "期间口径": "本季度末数",
                    "指标编码": code, "数值": value, "披露状态": "已披露",
                    "来源类型": "外部数据集",
                })
        note = _core_policy_surplus_coverage_note(pd.DataFrame(rows), pd.DataFrame(ratios))
        self.assertIn(
            "中邮人寿、中银三星、建信人寿、招商信诺：分子仅计入核心一级资本的保单未来盈余，核心二级资本没有披露这个明细",
            note,
        )
        self.assertIn(
            "农银人寿、交银人寿、工银安盛：分子计入核心一级及核心二级资本的保单未来盈余",
            note,
        )

        for row in rows:
            if row["公司"] == "中邮人寿" and row["指标编码"] == "POLICY_SURPLUS_CORE_T2":
                row["披露状态"] = "已披露为0"
        note = _core_policy_surplus_coverage_note(pd.DataFrame(rows), pd.DataFrame(ratios))
        self.assertNotIn("中邮人寿：分子仅计入核心一级资本", note)

    def test_capital_ratio_combo_shares_amount_axis_within_each_chart_group(self):
        rows = []
        for company, actual, minimum, core_t1, core_t2, anc_t1, anc_t2, combined_ratio, core_ratio in [
            ("甲", 100.0, 50.0, 60.0, 10.0, 20.0, 10.0, 200.0, 140.0),
            ("乙", 300.0, 100.0, 180.0, 20.0, 80.0, 20.0, 300.0, 200.0),
        ]:
            for code, value, unit in [
                ("ACTUAL_CAPITAL", actual, "亿元"),
                ("MINIMUM_CAPITAL", minimum, "亿元"),
                ("CORE_T1_CAPITAL", core_t1, "亿元"),
                ("CORE_T2_CAPITAL", core_t2, "亿元"),
                ("ANC_T1_CAPITAL", anc_t1, "亿元"),
                ("ANC_T2_CAPITAL", anc_t2, "亿元"),
                ("COMBINED_SOLVENCY_RATIO", combined_ratio, "%"),
                ("CORE_SOLVENCY_RATIO", core_ratio, "%"),
            ]:
                rows.append({
                    "公司": company,
                    "报告期": "2025Q4",
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "单位": unit,
                })
        frame = pd.DataFrame(rows)
        periods = ["2025Q4"]
        combined_groups = {
            "实际资本": ("ACTUAL_CAPITAL",),
            "最低资本": ("MINIMUM_CAPITAL",),
        }
        core_groups = {
            "核心资本": ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
            "最低资本": ("MINIMUM_CAPITAL",),
        }
        capital_structure_groups = {
            "核心资本": ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
            "附属资本": ("ANC_T1_CAPITAL", "ANC_T2_CAPITAL"),
        }
        combined_domain = capital_ratio_amount_axis_domain(
            frame,
            combined_groups,
            periods,
        )
        core_domain = capital_ratio_amount_axis_domain(frame, core_groups, periods)
        capital_structure_domain = capital_ratio_amount_axis_domain(
            frame,
            capital_structure_groups,
            periods,
        )
        self.assertEqual(combined_domain, (0.0, 432.0))
        self.assertEqual(core_domain, (0.0, 324.0))
        self.assertEqual(capital_structure_domain, (0.0, 324.0))

        company_domains = []
        for company in ("甲", "乙"):
            spec = build_capital_ratio_combo_chart(
                frame[frame["公司"].eq(company)],
                combined_groups,
                periods,
                "综合充足率变化",
                ratio_code="COMBINED_SOLVENCY_RATIO",
                shared_amount_domain=combined_domain,
            ).to_dict(validate=True)
            bar_panel = spec["vconcat"][1]
            bar_domain = bar_panel["layer"][0]["encoding"]["y"]["scale"]["domain"]
            label_domain = bar_panel["layer"][1]["encoding"]["y"]["scale"]["domain"]
            self.assertEqual(bar_domain, [0.0, 432.0])
            self.assertEqual(label_domain, [0.0, 432.0])
            company_domains.append(bar_domain)
        self.assertEqual(company_domains[0], company_domains[1])

        core_company_domains = []
        for company in ("甲", "乙"):
            spec = build_capital_ratio_combo_chart(
                frame[frame["公司"].eq(company)],
                capital_structure_groups,
                periods,
                "核心资本占比",
                share_line=True,
                shared_amount_domain=capital_structure_domain,
            ).to_dict(validate=True)
            bar_panel = spec["vconcat"][1]
            bar_domain = bar_panel["layer"][0]["encoding"]["y"]["scale"]["domain"]
            self.assertEqual(bar_domain, [0.0, 324.0])
            core_company_domains.append(bar_domain)
        self.assertEqual(core_company_domains[0], core_company_domains[1])

        grouped_domain = capital_ratio_amount_axis_domain(
            frame,
            combined_groups,
            periods,
            stacked=False,
        )
        self.assertEqual(grouped_domain, (0.0, 324.0))
        grouped_spec = build_capital_ratio_combo_chart(
            frame[frame["公司"].eq("甲")],
            combined_groups,
            periods,
            "综合充足率变化",
            ratio_code="COMBINED_SOLVENCY_RATIO",
            shared_amount_domain=grouped_domain,
            bar_layout="grouped",
        ).to_dict(validate=True)
        grouped_bars = grouped_spec["vconcat"][1]["layer"][0]["encoding"]
        grouped_labels = grouped_spec["vconcat"][1]["layer"][1]["encoding"]
        self.assertEqual(grouped_bars["xOffset"]["field"], "组成类别")
        self.assertEqual(grouped_bars["xOffset"]["scale"]["paddingInner"], 0.0)
        self.assertEqual(grouped_bars["xOffset"]["scale"]["paddingOuter"], 0.04)
        self.assertEqual(
            grouped_spec["vconcat"][1]["layer"][0]["mark"]["width"],
            {"band": 0.94},
        )
        self.assertEqual(grouped_bars["y"]["field"], "数值")
        self.assertNotIn("y2", grouped_bars)
        self.assertEqual(grouped_labels["xOffset"]["field"], "组成类别")
        self.assertEqual(grouped_bars["y"]["scale"]["domain"], [0.0, 324.0])

    def test_capital_ratio_views_render_only_stacked_combo(self):
        rows = []
        values = {
            "ACTUAL_CAPITAL": 120.0,
            "MINIMUM_CAPITAL": 70.0,
            "CORE_T1_CAPITAL": 65.0,
            "CORE_T2_CAPITAL": 15.0,
            "COMBINED_SOLVENCY_RATIO": 171.4,
            "CORE_SOLVENCY_RATIO": 114.3,
        }
        for code, value in values.items():
            rows.append({
                "公司": "甲",
                "报告期": "2025Q4",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "单位": "%" if code.endswith("RATIO") else "亿元",
            })
        frame = pd.DataFrame(rows)

        for chart_name in ("综合充足率变化", "核心充足率变化"):
            with (
                patch("step7_solvency.st.caption") as caption,
                patch("step7_solvency.st.markdown") as markdown,
                patch("step7_solvency._render_chart_legend"),
                patch("step7_solvency._render_company_chart_grid") as render_grid,
            ):
                _render_combination_analysis(
                    frame,
                    chart_name,
                    periods=["2025Q4"],
                    unit_mode="亿元",
                    highlight_company="无",
                    key_prefix="capital_ratio_variants",
                )

            self.assertEqual(render_grid.call_count, 1)
            self.assertFalse(any("并列柱组合图" in str(call) for call in markdown.call_args_list))
            self.assertFalse(any("每个报告期的两项资本各自成柱" in str(call) for call in caption.call_args_list))

    def test_step7_excel_guide_bar_combo_and_signed_stack_validate(self):
        rows = []
        for company in ["甲", "乙"]:
            for period, shift in [("2024YE", 0.0), ("2025YE", 10.0)]:
                values = {
                    "CORE_SOLVENCY_RATIO": 120.0 + shift,
                    "COMBINED_SOLVENCY_RATIO": 180.0 + shift,
                    "ACTUAL_CAPITAL": 120.0 + shift,
                    "MINIMUM_CAPITAL": 70.0 + shift,
                    "CORE_T1_CAPITAL": 70.0 + shift,
                    "CORE_T2_CAPITAL": 10.0,
                    "ANC_T1_CAPITAL": 30.0,
                    "ANC_T2_CAPITAL": 10.0,
                    "INSURANCE_RISK_CAPITAL": 30.0,
                    "NON_LIFE_INSURANCE_RISK_CAPITAL": 5.0,
                    "MARKET_RISK_CAPITAL": 20.0,
                    "CREDIT_RISK_CAPITAL": 15.0,
                    "QUANT_RISK_DIVERSIFICATION_EFFECT": -8.0,
                    "CONTRACT_LOSS_ABSORPTION_EFFECT": -3.0,
                    "FEATURE_FACTOR_IMPACT": 2.0,
                    "CONTROL_RISK_CAPITAL": 4.0,
                }
                for code, value in values.items():
                    rows.append({
                        "公司": company,
                        "报告期": period,
                        "指标编码": code,
                        "指标名称": code,
                        "数值": value,
                        "单位": "%" if code.endswith("RATIO") else "亿元",
                    })
        frame = pd.DataFrame(rows)
        periods = ["2024YE", "2025YE"]
        company_frame = frame[frame["公司"].eq("甲")]
        bar = build_company_period_bar_chart(company_frame, "CORE_SOLVENCY_RATIO", periods)
        combo = build_capital_ratio_combo_chart(
            company_frame,
            {"实际资本": ("ACTUAL_CAPITAL",), "最低资本": ("MINIMUM_CAPITAL",)},
            periods,
            "综合充足率变化",
            ratio_code="COMBINED_SOLVENCY_RATIO",
            ratio_label="综合偿付能力充足率（%）",
        )
        stack = build_component_stack_chart(
            frame,
            (
                ("INSURANCE_RISK_CAPITAL", "保险风险（寿）", "#00338D"),
                ("MARKET_RISK_CAPITAL", "市场风险", "#7213EA"),
                ("QUANT_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", "#FD349C"),
            ),
            periods,
            "量化风险最低资本构成",
        )
        for chart in (bar, combo, stack):
            self.assertIsInstance(chart.to_dict(validate=True), dict)
        bar_spec = bar.to_dict(validate=True)
        self.assertNotIn("facet", bar_spec)
        self.assertEqual(bar_spec["layer"][0]["encoding"]["x"]["field"], "报告期")
        self.assertEqual(bar_spec["layer"][0]["mark"]["width"], {"band": 0.72})
        self.assertEqual(bar_spec["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"], 0.5)
        self.assertFalse(bar_spec["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertFalse(bar_spec["layer"][0]["encoding"]["y"]["axis"]["ticks"])
        self.assertIsNone(bar_spec["layer"][0]["encoding"]["color"]["legend"])
        self.assertEqual(bar_spec["config"]["font"], "Microsoft YaHei")
        self.assertEqual(bar_spec["config"]["title"]["anchor"], "middle")
        self.assertEqual(bar_spec["config"]["title"]["fontSize"], 13)
        self.assertEqual(bar_spec["config"]["title"]["color"], "#00338D")
        self.assertEqual(bar_spec["config"]["axis"]["labelFontSize"], 10)
        self.assertEqual(bar_spec["config"]["legend"]["labelFontSize"], 10)
        combo_spec = combo.to_dict(validate=True)
        self.assertIn("vconcat", combo_spec)
        self.assertEqual(combo_spec["title"], "甲")
        self.assertNotIn(
            "title",
            _chart_without_internal_title(combo).to_dict(validate=True),
        )
        self.assertEqual(combo_spec["vconcat"][0]["layer"][0]["mark"]["color"], "#FD349C")
        self.assertIsNone(combo_spec["vconcat"][0]["layer"][0]["encoding"]["x"]["axis"])
        self.assertFalse(combo_spec["vconcat"][0]["layer"][0]["encoding"]["y"]["axis"]["domain"])
        self.assertFalse(combo_spec["vconcat"][0]["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertFalse(combo_spec["vconcat"][0]["layer"][0]["encoding"]["y"]["axis"]["grid"])
        self.assertFalse(combo_spec["vconcat"][1]["layer"][0]["encoding"]["x"]["axis"]["domain"])
        self.assertEqual(combo_spec["vconcat"][1]["layer"][0]["mark"]["width"], {"band": 0.72})
        self.assertEqual(
            combo_spec["vconcat"][1]["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"],
            0.5,
        )
        self.assertFalse(combo_spec["vconcat"][1]["layer"][0]["encoding"]["y"]["axis"]["domain"])
        self.assertFalse(combo_spec["vconcat"][1]["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertFalse(combo_spec["vconcat"][1]["layer"][0]["encoding"]["y"]["axis"]["grid"])
        self.assertIsNone(combo_spec["vconcat"][1]["layer"][0]["encoding"]["color"]["legend"])
        self.assertEqual(
            combo_spec["vconcat"][1]["layer"][0]["encoding"]["color"]["scale"]["range"],
            ["#1E49E2", "#00B8F5"],
        )
        self.assertFalse(combo_spec["config"]["axisX"]["domain"])
        self.assertFalse(combo_spec["config"]["axisX"]["ticks"])
        self.assertTrue(combo_spec["config"]["axisX"]["labels"])
        self.assertFalse(combo_spec["config"]["axisY"]["domain"])
        combo_data_name = combo_spec["vconcat"][1]["data"]["name"]
        combo_rows = combo_spec["datasets"][combo_data_name]
        minimum = next(
            row for row in combo_rows
            if row["报告期"] == "2024YE" and row["组成类别"] == "最低资本"
        )
        actual = next(
            row for row in combo_rows
            if row["报告期"] == "2024YE" and row["组成类别"] == "实际资本"
        )
        self.assertEqual(minimum["堆叠起点"], 0.0)
        self.assertEqual(actual["堆叠起点"], minimum["堆叠终点"])
        self.assertEqual(actual["标签颜色"], "#FFFFFF")
        self.assertEqual(minimum["标签颜色"], "#0C233C")
        self.assertEqual(actual["数值标签"], "120")
        self.assertEqual(minimum["数值标签"], "70")
        stack_spec = stack.to_dict(validate=True)
        self.assertEqual(stack_spec["facet"]["field"], "公司")
        self.assertFalse(stack_spec["spec"]["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertFalse(stack_spec["spec"]["layer"][0]["encoding"]["y"]["axis"]["ticks"])
        self.assertEqual(stack_spec["spec"]["layer"][0]["mark"]["width"], {"band": 0.72})
        self.assertEqual(
            stack_spec["spec"]["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"],
            0.5,
        )
        self.assertIsNone(stack_spec["spec"]["layer"][0]["encoding"]["color"]["legend"])
        company_combo = build_capital_ratio_combo_chart(
            company_frame,
            {"实际资本": ("ACTUAL_CAPITAL",), "最低资本": ("MINIMUM_CAPITAL",)},
            periods,
            "综合充足率变化",
            ratio_code="COMBINED_SOLVENCY_RATIO",
        ).to_dict(validate=True)
        company_stack = build_component_stack_chart(
            company_frame,
            (("INSURANCE_RISK_CAPITAL", "保险风险（寿）", "#00338D"),),
            periods,
            "保险风险构成",
        ).to_dict(validate=True)
        self.assertNotIn("facet", company_combo)
        self.assertNotIn("facet", company_stack)
        self.assertEqual(company_combo["vconcat"][1]["layer"][0]["encoding"]["x"]["field"], "报告期")
        dense_bar_spec = build_company_period_bar_chart(
            company_frame,
            "CORE_SOLVENCY_RATIO",
            periods,
            dense_layout=True,
        ).to_dict(validate=True)
        dense_combo_spec = build_capital_ratio_combo_chart(
            company_frame,
            {"实际资本": ("ACTUAL_CAPITAL",), "最低资本": ("MINIMUM_CAPITAL",)},
            periods,
            "综合充足率变化",
            ratio_code="COMBINED_SOLVENCY_RATIO",
            dense_layout=True,
        ).to_dict(validate=True)
        self.assertEqual(
            dense_bar_spec["layer"][0]["encoding"]["x"]["axis"]["labelFontSize"],
            7,
        )
        self.assertEqual(dense_bar_spec["layer"][1]["mark"]["fontSize"], 8)
        self.assertEqual(
            dense_combo_spec["vconcat"][1]["layer"][0]["encoding"]["x"]["axis"]["labelFontSize"],
            7,
        )
        self.assertEqual(dense_combo_spec["vconcat"][1]["layer"][1]["mark"]["fontSize"], 8)
        dataset = stack_spec["datasets"][stack_spec["data"]["name"]]
        negative = next(row for row in dataset if row["指标编码"] == "QUANT_RISK_DIVERSIFICATION_EFFECT")
        self.assertLess(negative["堆叠终点"], 0)

    def test_quant_risk_stack_labels_match_ratio_trend_metrics(self):
        rows = [
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "QUANT_RISK_CAPITAL", "数值": 100.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "INSURANCE_RISK_CAPITAL", "数值": 30.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "MARKET_RISK_CAPITAL", "数值": 50.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "QUANT_RISK_DIVERSIFICATION_EFFECT", "数值": -10.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "CONTROL_RISK_CAPITAL", "数值": 4.0, "单位": "亿元"},
            # Reviewed ratios take priority so the stack labels are identical
            # to the corresponding trend-chart values.
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL", "数值": 0.31, "单位": "倍"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "MARKET_RISK_TO_QUANT_CAPITAL", "数值": 0.49, "单位": "倍"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL", "数值": -0.10, "单位": "倍"},
        ]
        frame = pd.DataFrame(rows)
        frame["指标名称"] = frame["指标编码"]
        specs = (
            ("INSURANCE_RISK_CAPITAL", "保险风险（寿）", "#ACEAFF"),
            ("MARKET_RISK_CAPITAL", "市场风险", "#B497FF"),
            ("QUANT_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", "#63EBB2"),
            ("CONTROL_RISK_CAPITAL", "控制风险", "#F1C44D"),
        )
        ratio_codes = {
            "INSURANCE_RISK_CAPITAL": "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
            "MARKET_RISK_CAPITAL": "MARKET_RISK_TO_QUANT_CAPITAL",
            "QUANT_RISK_DIVERSIFICATION_EFFECT": "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL",
        }
        chart = build_component_stack_chart(
            frame,
            specs,
            ["2025Q4"],
            "量化风险最低资本构成",
            label_ratio_codes=ratio_codes,
            label_denominator_code="QUANT_RISK_CAPITAL",
            plot_proportions=True,
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset}
        ratios = {row["指标编码"]: row["构成占比"] for row in dataset}
        self.assertEqual(labels["INSURANCE_RISK_CAPITAL"], "31%")
        self.assertEqual(labels["MARKET_RISK_CAPITAL"], "49%")
        self.assertEqual(labels["QUANT_RISK_DIVERSIFICATION_EFFECT"], "-10%")
        self.assertEqual(labels["CONTROL_RISK_CAPITAL"], "")
        self.assertAlmostEqual(ratios["INSURANCE_RISK_CAPITAL"], 0.31)
        insurance = next(
            row for row in dataset
            if row["指标编码"] == "INSURANCE_RISK_CAPITAL"
        )
        market = next(
            row for row in dataset
            if row["指标编码"] == "MARKET_RISK_CAPITAL"
        )
        diversification = next(
            row for row in dataset
            if row["指标编码"] == "QUANT_RISK_DIVERSIFICATION_EFFECT"
        )
        self.assertEqual(insurance["数值"], 30.0)
        self.assertAlmostEqual(insurance["堆叠终点"], 0.31)
        self.assertAlmostEqual(market["堆叠终点"], 0.80)
        self.assertAlmostEqual(diversification["堆叠终点"], -0.10)
        self.assertLess(spec["layer"][0]["encoding"]["y"]["scale"]["domain"][1], 2.0)

        trend = build_single_metric_trend_chart(
            frame,
            "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
            ["2025Q4"],
            {"甲": "#00338D"},
            "甲",
        ).to_dict(validate=True)
        self.assertEqual(trend["layer"][0]["encoding"]["y"]["axis"]["format"], ",.1f")
        self.assertEqual(trend["layer"][3]["encoding"]["text"]["field"], "数值标签")
        trend_dataset = trend["datasets"][trend["data"]["name"]]
        self.assertEqual(trend_dataset[0]["数值"], 31.0)
        self.assertEqual(trend_dataset[0]["数值标签"], "31.0%")

    def test_quant_risk_renderer_preserves_denominator_and_ratio_inputs(self):
        chart_name = "量化风险最低资本构成"
        values = {
            "INSURANCE_RISK_CAPITAL": 40.0,
            "NON_LIFE_INSURANCE_RISK_CAPITAL": 5.0,
            "MARKET_RISK_CAPITAL": 30.0,
            "CREDIT_RISK_CAPITAL": 25.0,
            "QUANT_RISK_DIVERSIFICATION_EFFECT": -8.0,
            "CONTRACT_LOSS_ABSORPTION_EFFECT": -2.0,
            "QUANT_RISK_CAPITAL": 90.0,
            "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL": 40.0 / 90.0,
            "NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL": 5.0 / 90.0,
            "MARKET_RISK_TO_QUANT_CAPITAL": 30.0 / 90.0,
            "CREDIT_RISK_TO_QUANT_CAPITAL": 25.0 / 90.0,
            "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL": -8.0 / 90.0,
            "LOSS_ABSORPTION_TO_QUANT_CAPITAL": -2.0 / 90.0,
        }
        frame = pd.DataFrame([
            {
                "公司": "测试公司",
                "报告期": "2025Q4",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "单位": "倍" if code.endswith("TO_QUANT_CAPITAL") else "亿元",
            }
            for code, value in values.items()
        ])

        input_codes = _chart_input_metric_codes(chart_name, COMPONENT_STACK)
        self.assertIn("QUANT_RISK_CAPITAL", input_codes)
        self.assertTrue(set(QUANT_RISK_STACK_RATIO_CODES.values()).issubset(input_codes))

        with (
            patch("step7_solvency.st.caption"),
            patch("step7_solvency._render_chart_legend"),
            patch("step7_solvency._render_company_chart_grid") as render_grid,
        ):
            _render_combination_analysis(
                frame,
                chart_name,
                periods=["2025Q4"],
                unit_mode="亿元",
                highlight_company="无",
                key_prefix="quant_stack_regression",
            )

        filtered_frame = render_grid.call_args.args[0]
        filtered_codes = set(filtered_frame["指标编码"].astype(str))
        self.assertIn("QUANT_RISK_CAPITAL", filtered_codes)
        self.assertTrue(
            set(QUANT_RISK_STACK_RATIO_CODES.values()).issubset(filtered_codes)
        )
        chart_factory = render_grid.call_args.args[1]
        self.assertIsInstance(
            chart_factory(filtered_frame).to_dict(validate=True),
            dict,
        )

    def test_life_insurance_risk_stack_uses_total_as_denominator_and_negative_diversification(self):
        chart_name = "各类保险风险（寿）占比"
        values = {
            "INSURANCE_RISK_CAPITAL": 100.0,
            "LOSS_OCCURRENCE_RISK_CAPITAL": 50.0,
            "SURRENDER_RISK_CAPITAL": 30.0,
            "EXPENSE_RISK_CAPITAL": 20.0,
            "LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT": 10.0,
        }
        frame = pd.DataFrame([
            {
                "公司": "测试公司",
                "报告期": "2025Q4",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "单位": "亿元",
            }
            for code, value in values.items()
        ])

        self.assertEqual(
            COMPONENT_STACK_DENOMINATOR_CODES[chart_name],
            "INSURANCE_RISK_CAPITAL",
        )
        self.assertIn(
            "INSURANCE_RISK_CAPITAL",
            _chart_input_metric_codes(chart_name, COMPONENT_STACK),
        )
        with (
            patch("step7_solvency.st.caption"),
            patch("step7_solvency._render_chart_legend"),
            patch("step7_solvency._render_company_chart_grid") as render_grid,
        ):
            _render_combination_analysis(
                frame,
                chart_name,
                periods=["2025Q4"],
                unit_mode="亿元",
                highlight_company="无",
                key_prefix="life_risk_stack_regression",
            )

        filtered_frame = render_grid.call_args.args[0]
        self.assertIn("INSURANCE_RISK_CAPITAL", set(filtered_frame["指标编码"]))
        chart_factory = render_grid.call_args.args[1]
        spec = chart_factory(filtered_frame).to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        rows = {row["指标编码"]: row for row in dataset}
        self.assertAlmostEqual(rows["LOSS_OCCURRENCE_RISK_CAPITAL"]["构成占比"], 0.50)
        self.assertAlmostEqual(rows["SURRENDER_RISK_CAPITAL"]["构成占比"], 0.30)
        self.assertAlmostEqual(rows["EXPENSE_RISK_CAPITAL"]["构成占比"], 0.20)
        self.assertAlmostEqual(
            rows["LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT"]["构成占比"],
            -0.10,
        )
        self.assertLess(
            rows["LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT"]["堆叠终点"],
            0,
        )

    def test_quant_risk_stack_excludes_feature_factor_and_control_risk(self):
        quant_specs = COMPONENT_STACK_SPECS["量化风险最低资本构成"]
        codes = {code for code, _, _ in quant_specs}
        self.assertEqual(
            codes,
            {
                "INSURANCE_RISK_CAPITAL",
                "NON_LIFE_INSURANCE_RISK_CAPITAL",
                "MARKET_RISK_CAPITAL",
                "CREDIT_RISK_CAPITAL",
                "QUANT_RISK_DIVERSIFICATION_EFFECT",
                "CONTRACT_LOSS_ABSORPTION_EFFECT",
            },
        )
        self.assertNotIn("FEATURE_FACTOR_IMPACT", codes)
        self.assertNotIn("CONTROL_RISK_CAPITAL", codes)
        self.assertEqual(
            [color for _, _, color in quant_specs],
            ["#ACEAFF", "#00B8F5", "#FFA3DA", "#B497FF", "#63EBB2", "#00C0AE"],
        )

    def test_policy_surplus_uses_the_same_blue_purple_capital_hierarchy(self):
        policy_specs = COMPONENT_STACK_SPECS["计入各级资本的保单未来盈余构成占比"]
        quant_specs = COMPONENT_STACK_SPECS["量化风险最低资本构成"]
        self.assertEqual(
            [color for _, _, color in policy_specs],
            ["#00B8F5", "#ACEAFF", "#B497FF", "#7213EA"],
        )
        market_specs = COMPONENT_STACK_SPECS["各类市场风险占比"]
        self.assertEqual(
            [color for _, _, color in market_specs],
            ["#FFA3DA", "#00B8F5", "#FD349C", "#F68D2E", "#ACEAFF", "#B497FF", "#63EBB2"],
        )
        life_specs = COMPONENT_STACK_SPECS["各类保险风险（寿）占比"]
        life_diversification_color = next(
            color for code, _, color in life_specs
            if code == "LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT"
        )
        quant_diversification_color = next(
            color for code, _, color in quant_specs
            if code == "QUANT_RISK_DIVERSIFICATION_EFFECT"
        )
        self.assertEqual(life_diversification_color, quant_diversification_color)
        credit_specs = COMPONENT_STACK_SPECS["各类信用风险占比"]
        market_diversification_color = next(
            color for code, _, color in market_specs
            if code == "MARKET_RISK_DIVERSIFICATION_EFFECT"
        )
        credit_diversification_color = next(
            color for code, _, color in credit_specs
            if code == "CREDIT_RISK_DIVERSIFICATION_EFFECT"
        )
        self.assertEqual(credit_diversification_color, market_diversification_color)

    def test_market_risk_stack_uses_market_total_and_includes_overseas_equity(self):
        rows = [
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "MARKET_RISK_CAPITAL", "数值": 100.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "INTEREST_RATE_RISK_CAPITAL", "数值": 30.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "EQUITY_RISK_CAPITAL", "数值": 20.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "REAL_ESTATE_RISK_CAPITAL", "数值": 10.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "OVERSEAS_FIXED_INCOME_RISK_CAPITAL", "数值": 5.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "OVERSEAS_EQUITY_RISK_CAPITAL", "数值": 8.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "FOREIGN_EXCHANGE_RISK_CAPITAL", "数值": 7.0, "单位": "亿元"},
            # Reports may disclose the deduction as a positive absolute amount;
            # the chart must normalize it to a negative stack segment.
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "MARKET_RISK_DIVERSIFICATION_EFFECT", "数值": 12.0, "单位": "亿元"},
        ]
        frame = pd.DataFrame(rows)
        frame["指标名称"] = frame["指标编码"]
        specs = COMPONENT_STACK_SPECS["各类市场风险占比"]
        chart = build_component_stack_chart(
            frame,
            specs,
            ["2025Q4"],
            "各类市场风险占比",
            label_denominator_code="MARKET_RISK_CAPITAL",
            negative_component_codes=("MARKET_RISK_DIVERSIFICATION_EFFECT",),
            plot_proportions=True,
            min_label_share=0.10,
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset}
        self.assertIn("OVERSEAS_EQUITY_RISK_CAPITAL", labels)
        self.assertEqual(labels["INTEREST_RATE_RISK_CAPITAL"], "30%")
        self.assertEqual(labels["REAL_ESTATE_RISK_CAPITAL"], "10%")
        self.assertEqual(labels["OVERSEAS_FIXED_INCOME_RISK_CAPITAL"], "")
        self.assertEqual(labels["OVERSEAS_EQUITY_RISK_CAPITAL"], "")
        self.assertEqual(labels["FOREIGN_EXCHANGE_RISK_CAPITAL"], "")
        self.assertEqual(labels["MARKET_RISK_DIVERSIFICATION_EFFECT"], "-12%")
        diversification = next(
            row for row in dataset
            if row["指标编码"] == "MARKET_RISK_DIVERSIFICATION_EFFECT"
        )
        overseas_equity = next(
            row for row in dataset
            if row["指标编码"] == "OVERSEAS_EQUITY_RISK_CAPITAL"
        )
        self.assertEqual(overseas_equity["数值"], 8.0)
        self.assertAlmostEqual(overseas_equity["堆叠终点"], 0.73)
        self.assertLess(diversification["堆叠终点"], 0)
        self.assertAlmostEqual(diversification["堆叠终点"], -0.12)

    def test_life_insurance_risk_stack_hides_labels_at_or_below_ten_percent(self):
        rows = [
            {"公司": "甲", "报告期": "2026Q2", "指标编码": "INSURANCE_RISK_CAPITAL", "数值": 100.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2026Q2", "指标编码": "LOSS_OCCURRENCE_RISK_CAPITAL", "数值": 79.9, "单位": "亿元"},
            {"公司": "甲", "报告期": "2026Q2", "指标编码": "SURRENDER_RISK_CAPITAL", "数值": 10.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2026Q2", "指标编码": "EXPENSE_RISK_CAPITAL", "数值": 10.1, "单位": "亿元"},
        ]
        frame = pd.DataFrame(rows)
        frame["指标名称"] = frame["指标编码"]
        chart = build_component_stack_chart(
            frame,
            COMPONENT_STACK_SPECS["各类保险风险（寿）占比"],
            ["2026Q2"],
            "各类保险风险（寿）占比",
            label_denominator_code="INSURANCE_RISK_CAPITAL",
            plot_proportions=True,
            min_label_share=0.10,
            hide_labels_at_threshold=True,
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset}
        self.assertEqual(labels["SURRENDER_RISK_CAPITAL"], "")
        self.assertEqual(labels["EXPENSE_RISK_CAPITAL"], "10%")
        self.assertLess(spec["layer"][0]["encoding"]["y"]["scale"]["domain"][1], 2.0)

    def test_credit_risk_stack_uses_credit_total_and_negative_diversification(self):
        rows = [
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "CREDIT_RISK_CAPITAL", "数值": 50.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "SPREAD_RISK_CAPITAL", "数值": 30.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "COUNTERPARTY_RISK_CAPITAL", "数值": 20.0, "单位": "亿元"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "CREDIT_RISK_DIVERSIFICATION_EFFECT", "数值": 10.0, "单位": "亿元"},
        ]
        frame = pd.DataFrame(rows)
        frame["指标名称"] = frame["指标编码"]
        chart = build_component_stack_chart(
            frame,
            COMPONENT_STACK_SPECS["各类信用风险占比"],
            ["2025Q4"],
            "各类信用风险占比",
            label_denominator_code="CREDIT_RISK_CAPITAL",
            negative_component_codes=("CREDIT_RISK_DIVERSIFICATION_EFFECT",),
            plot_proportions=True,
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset}
        self.assertEqual(labels["SPREAD_RISK_CAPITAL"], "60%")
        self.assertEqual(labels["COUNTERPARTY_RISK_CAPITAL"], "40%")
        self.assertEqual(labels["CREDIT_RISK_DIVERSIFICATION_EFFECT"], "-20%")
        diversification = next(
            row for row in dataset
            if row["指标编码"] == "CREDIT_RISK_DIVERSIFICATION_EFFECT"
        )
        counterparty = next(
            row for row in dataset
            if row["指标编码"] == "COUNTERPARTY_RISK_CAPITAL"
        )
        self.assertEqual(counterparty["数值"], 20.0)
        self.assertAlmostEqual(counterparty["堆叠终点"], 1.0)
        self.assertLess(diversification["堆叠终点"], 0)
        self.assertAlmostEqual(diversification["堆叠终点"], -0.20)
        self.assertLess(spec["layer"][0]["encoding"]["y"]["scale"]["domain"][1], 2.0)

    def test_all_stack_labels_hide_components_below_four_point_five_percent(self):
        frame = pd.DataFrame([
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "SPREAD_RISK_CAPITAL", "数值": 91.01, "单位": "亿元", "指标名称": "利差风险"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "COUNTERPARTY_RISK_CAPITAL", "数值": 4.49, "单位": "亿元", "指标名称": "交易对手违约风险"},
            {"公司": "甲", "报告期": "2025Q4", "指标编码": "CREDIT_RISK_DIVERSIFICATION_EFFECT", "数值": -4.50, "单位": "亿元", "指标名称": "风险分散效应"},
        ])
        chart = build_component_stack_chart(
            frame,
            COMPONENT_STACK_SPECS["各类信用风险占比"],
            ["2025Q4"],
            "各类信用风险占比",
            value_labels=True,
        )
        spec = chart.to_dict(validate=True)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset}
        self.assertEqual(labels["COUNTERPARTY_RISK_CAPITAL"], "")
        self.assertEqual(labels["CREDIT_RISK_DIVERSIFICATION_EFFECT"], "-5")

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
        self.assertNotIn("facet", amount_spec)
        self.assertFalse(amount_spec["layer"][0]["encoding"]["y"]["axis"]["labels"])
        self.assertFalse(amount_spec["layer"][0]["encoding"]["y"]["axis"]["ticks"])
        label_encoding = amount_spec["layer"][1]["encoding"]
        self.assertEqual(label_encoding["text"]["field"], "占比标签")
        self.assertEqual(label_encoding["y"]["field"], "标签位置")
        self.assertEqual(label_encoding["color"]["field"], "标签颜色")
        self.assertIsNone(label_encoding["color"]["scale"])
        label_colors = {row["资本类别"]: row["标签颜色"] for row in amount_spec["datasets"][dataset_name]}
        self.assertEqual(label_colors["核心一级资本"], "#0C233C")
        self.assertEqual(label_colors["附属二级资本"], "#FFFFFF")
        self.assertEqual(
            amount_spec["layer"][0]["encoding"]["color"]["scale"]["range"],
            ["#00B8F5", "#ACEAFF", "#B497FF", "#7213EA"],
        )
        self.assertEqual(
            amount_spec["layer"][0]["encoding"]["y"]["scale"]["domain"],
            [0.0, 1.0],
        )
        grouped_rows = pd.DataFrame(amount_spec["datasets"][dataset_name])
        for _, period_rows in grouped_rows.groupby(["公司", "报告期"]):
            self.assertAlmostEqual(period_rows["资本占比"].sum(), 1.0)
            self.assertAlmostEqual(period_rows["堆叠终点"].max(), 1.0)
        tooltip_fields = {
            item.get("field")
            for item in amount_spec["layer"][0]["encoding"]["tooltip"]
        }
        self.assertIn("资本占比", tooltip_fields)
        self.assertIn("数值", tooltip_fields)
        self.assertEqual(
            amount_spec["layer"][0]["encoding"]["x"]["scale"]["paddingOuter"],
            0.5,
        )

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
        self.assertEqual(len(grouped), 1)
        dense_spec = grouped[0].to_dict(validate=True)
        dataset_name = dense_spec.get("data", dense_spec["layer"][0].get("data"))["name"]
        chart_companies = {row["公司"] for row in dense_spec["datasets"][dataset_name]}
        self.assertEqual(chart_companies, set(dense_colors))
        self.assertEqual(dense_spec["height"], 420)
        self.assertNotIn("组", dense_spec["title"])

        panel_spec = build_company_bar_trend_chart(frame, "TEST", periods).to_dict(validate=True)
        self.assertEqual(panel_spec["spec"]["height"], 285)
        self.assertEqual(panel_spec["spec"]["layer"][0]["mark"]["type"], "bar")
        self.assertEqual(panel_spec["spec"]["layer"][1]["mark"]["type"], "line")
        self.assertLessEqual(len(panel_spec["spec"]["layer"]), 5)

    def test_step7_non_life_risk_to_liabilities_uses_permille_display(self):
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
                spec["layer"][layer_index]["encoding"]["text"]["field"],
                "数值标签",
            )
        y_encoding = spec["layer"][0]["encoding"]["y"]
        self.assertEqual(y_encoding["axis"]["format"], ".5f")
        self.assertEqual(y_encoding["axis"]["tickCount"], 6)
        domain = y_encoding["scale"]["domain"]
        self.assertLess(domain[1] - domain[0], 2.0)
        dataset = spec["datasets"][spec["data"]["name"]]
        disclosed = [row for row in dataset if row["数值"] is not None]
        self.assertTrue(all(row["单位"] == "‰" for row in disclosed))
        self.assertTrue(all(row["数值标签"].endswith("‰") for row in disclosed))
        first_value = next(
            row["数值"] for row in disclosed
            if row["公司"] == "甲" and row["报告期"] == "2025Q2"
        )
        self.assertAlmostEqual(first_value, 1.2345)
        self.assertIn("‰", y_encoding["title"])

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
        self.assertEqual(spec["columns"], 15)
        self.assertEqual(spec["spec"]["height"], 285)
        dataset = spec["datasets"][spec["data"]["name"]]
        labels = {row["指标编码"]: row["占比标签"] for row in dataset[:4]}
        self.assertEqual(labels["CORE_T1_CAPITAL"], "70%")
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
        self.assertEqual(list(colors.values()), ["#00B8F5", "#FD349C", "#00C0AE"])

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
