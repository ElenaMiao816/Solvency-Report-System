from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

from services.solvency_dataset_adapter import POLICY_SURPLUS_COMPONENT_CODES, canonical_period_scope
from dashboard_components import (
    company_detail_rows,
    render_key_solvency_overview,
    render_major_financing,
    render_report_analysis,
    render_report_back_cover,
    render_report_cover,
    render_report_footnote,
    render_report_notes_editor,
)
from services.llm_config import model_request_parameters, normalize_model_id
from services.llm_http import post_json_with_retry
from services.solvency_company_identity import display_company_names
from services.solvency_navigation import (
    COMPANY_NAVIGATION,
    CREDIT_RISK_ASSET_SCATTER,
    INSURANCE_RISK_LIABILITY_SCATTER,
    KPMG_BRIGHT_CHART_COLORS,
    KPMG_CAPITAL_COMBO_COLORS,
    KPMG_CHART_COLORS,
    KPMG_QUANT_RISK_COLORS,
    MARKET_RISK_ASSET_SCATTER,
    PRINT_ALL_LABEL,
    metric_codes_for_chart,
    resolve_chart_selection,
)
from services.solvency_step6_analysis import (
    filter_analysis_frame,
    format_chart_value,
    nonblank_values,
    sort_report_periods,
)
from services.solvency_step7_chart_plans import (
    COMPANY_BAR_TREND,
    COMPANY_PERIOD_BAR,
    CAPITAL_AMOUNT_COMBO,
    CAPITAL_EFFICIENCY_BUBBLE,
    CAPITAL_RATIO_COMBO,
    CAPITAL_STRUCTURE_COMBO,
    COMPONENT_STACK,
    EFFECT_DIVERGING,
    FINANCING_TABLE,
    KEY_METRICS_TABLE,
    MARKET_CREDIT_MATRIX,
    PENDING_DEFINITION,
    QUALITY_AND_CAPITAL,
    RISK_RATIO_SCATTER,
    SINGLE_METRIC_TREND,
    SOLVENCY_MATRIX,
    SOLVENCY_RATIO_COMBO,
    TREND_WITH_COMPANY_BARS,
    chart_plan_for,
)
from services.solvency_step7_charts import (
    CAPITAL_STRUCTURE_COLORS,
    build_capital_amount_combo,
    build_capital_efficiency_bubble_chart,
    capital_efficiency_bubble_default_domain,
    combine_capital_efficiency_bubble_charts,
    combine_linked_matrix_charts,
    build_capital_ratio_combo_chart,
    capital_ratio_amount_axis_domain,
    build_company_bar_trend_chart,
    build_company_period_bar_chart,
    build_component_stack_chart,
    build_effect_diverging_chart,
    build_matrix_chart,
    build_single_metric_trend_chart,
    build_single_metric_trend_charts,
    build_solvency_ratio_combo_overview_chart,
    convert_multiple_units_to_percent,
    component_stack_axis_domain,
    component_stack_proportion_axis_domain,
    metric_bar_axis_domain,
    report_period_combo_bar_color_map,
    report_period_color_map,
    risk_ratio_scatter_default_domain,
    solvency_ratio_axis_domain,
)
from services.solvency_report_notes import company_notes_template
from services.solvency_step7_quality import (
    ANC_COMPONENTS,
    COMPONENT_COLORS,
    SCOPE_LABELS,
    GREY,
    anc_pie_figure,
    combo_figure,
    complete_external_capital_detail_zeros,
    core_omitted_nonzero,
    core_waterfall_figure,
    capital_value,
    has_values,
    operating_quarter_value,
    period_scopes,
    radar_figure,
    ratio_bar_figure,
    value_for,
)
from services.solvency_step7_quality_charts import company_quality_chart, empty_company_chart


PICTURE_DIR = Path(__file__).resolve().parent / "picture"
STEP7_CHART_TYPE = "内置分析方案"
ALL_COMPANY_TYPES = "全部"
DEFAULT_SORT_LABEL = "默认（按列表原始顺序）"
SORT_DESCENDING = "降序（从大到小）"
SORT_ASCENDING = "升序（从小到大）"
TRACKED_COMPANY_COLOR = "#00338D"
RISK_DIVERSIFICATION_EFFECT_COLOR = KPMG_BRIGHT_CHART_COLORS[6]
OVERSEAS_FIXED_INCOME_RISK_COLOR = "#F68D2E"
RISK_SCATTER_DISPLAY_WIDTH = 750
RISK_SCATTER_FULL_PANEL_WIDTH = 570
RISK_SCATTER_ZOOM_PANEL_WIDTH = 330
BUBBLE_FULL_PANEL_WIDTH = 570
BUBBLE_ZOOM_PANEL_WIDTH = 330
COMPANY_CHART_RENDER_KEYS = {
    chart_name: f"chart_{index}"
    for index, chart_name in enumerate(
        dict.fromkeys(entry.chart_name for entry in COMPANY_NAVIGATION)
    )
}
RECOGNIZED_BALANCE_COMPONENT_COLORS = {
    "cash_or_provision": KPMG_BRIGHT_CHART_COLORS[0],
    "investment_or_financial": KPMG_BRIGHT_CHART_COLORS[1],
    "equity_or_capital": KPMG_BRIGHT_CHART_COLORS[2],
    "insurance_or_reserve": KPMG_BRIGHT_CHART_COLORS[3],
    "receivable_or_payable": KPMG_BRIGHT_CHART_COLORS[4],
    "fixed_assets": KPMG_BRIGHT_CHART_COLORS[5],
    "land_use_rights": KPMG_BRIGHT_CHART_COLORS[6],
    "separate_account": KPMG_BRIGHT_CHART_COLORS[7],
    "other_recognized": KPMG_BRIGHT_CHART_COLORS[8],
}
COMPONENT_STACK_SPECS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "计入各级资本的保单未来盈余构成占比": (
        ("POLICY_SURPLUS_CORE_T1", "计入核心一级资本", CAPITAL_STRUCTURE_COLORS["核心一级资本"]),
        ("POLICY_SURPLUS_CORE_T2", "计入核心二级资本", CAPITAL_STRUCTURE_COLORS["核心二级资本"]),
        ("POLICY_SURPLUS_ANC_T1", "计入附属一级资本", CAPITAL_STRUCTURE_COLORS["附属一级资本"]),
        ("POLICY_SURPLUS_ANC_T2", "计入附属二级资本", CAPITAL_STRUCTURE_COLORS["附属二级资本"]),
    ),
    "量化风险最低资本构成": (
        ("INSURANCE_RISK_CAPITAL", "保险风险（寿）", KPMG_QUANT_RISK_COLORS[0]),
        ("NON_LIFE_INSURANCE_RISK_CAPITAL", "保险风险（非寿）", KPMG_QUANT_RISK_COLORS[1]),
        ("MARKET_RISK_CAPITAL", "市场风险", KPMG_QUANT_RISK_COLORS[2]),
        ("CREDIT_RISK_CAPITAL", "信用风险", KPMG_QUANT_RISK_COLORS[3]),
        ("QUANT_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", KPMG_QUANT_RISK_COLORS[4]),
        ("CONTRACT_LOSS_ABSORPTION_EFFECT", "损失吸收效应", KPMG_QUANT_RISK_COLORS[5]),
    ),
    "各类保险风险（寿）占比": (
        ("LOSS_OCCURRENCE_RISK_CAPITAL", "损失发生风险", KPMG_BRIGHT_CHART_COLORS[0]),
        ("SURRENDER_RISK_CAPITAL", "退保风险", KPMG_BRIGHT_CHART_COLORS[1]),
        ("EXPENSE_RISK_CAPITAL", "费用风险", KPMG_BRIGHT_CHART_COLORS[2]),
        ("LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", RISK_DIVERSIFICATION_EFFECT_COLOR),
    ),
    "各类市场风险占比": (
        ("INTEREST_RATE_RISK_CAPITAL", "利率风险", KPMG_BRIGHT_CHART_COLORS[0]),
        ("EQUITY_RISK_CAPITAL", "权益价格风险", KPMG_BRIGHT_CHART_COLORS[1]),
        ("REAL_ESTATE_RISK_CAPITAL", "房地产价格风险", KPMG_BRIGHT_CHART_COLORS[2]),
        ("OVERSEAS_FIXED_INCOME_RISK_CAPITAL", "境外固定收益类资产价格风险", OVERSEAS_FIXED_INCOME_RISK_COLOR),
        ("OVERSEAS_EQUITY_RISK_CAPITAL", "境外权益类资产价格风险", KPMG_BRIGHT_CHART_COLORS[4]),
        ("FOREIGN_EXCHANGE_RISK_CAPITAL", "汇率风险", KPMG_BRIGHT_CHART_COLORS[5]),
        ("MARKET_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", RISK_DIVERSIFICATION_EFFECT_COLOR),
    ),
    "各类信用风险占比": (
        ("SPREAD_RISK_CAPITAL", "利差风险", KPMG_BRIGHT_CHART_COLORS[0]),
        ("COUNTERPARTY_RISK_CAPITAL", "交易对手违约风险", KPMG_BRIGHT_CHART_COLORS[1]),
        ("CREDIT_RISK_DIVERSIFICATION_EFFECT", "风险分散效应", RISK_DIVERSIFICATION_EFFECT_COLOR),
    ),
    "认可资产构成": (
        ("CASH_LIQUID_ASSETS", "现金及流动性管理工具", RECOGNIZED_BALANCE_COMPONENT_COLORS["cash_or_provision"]),
        ("INVESTMENT_ASSETS", "投资资产", RECOGNIZED_BALANCE_COMPONENT_COLORS["investment_or_financial"]),
        ("SUBSIDIARY_JV_ASSOCIATE_EQUITY", "在子公司合营企业和联营企业中的权益", RECOGNIZED_BALANCE_COMPONENT_COLORS["equity_or_capital"]),
        ("REINSURANCE_ASSETS", "再保险资产", RECOGNIZED_BALANCE_COMPONENT_COLORS["insurance_or_reserve"]),
        ("RECEIVABLES_AND_PREPAYMENTS", "应收及预付款项", RECOGNIZED_BALANCE_COMPONENT_COLORS["receivable_or_payable"]),
        ("FIXED_ASSETS", "固定资产", RECOGNIZED_BALANCE_COMPONENT_COLORS["fixed_assets"]),
        ("LAND_USE_RIGHTS", "土地使用权", RECOGNIZED_BALANCE_COMPONENT_COLORS["land_use_rights"]),
        ("SEPARATE_ACCOUNT_ASSETS", "独立账户资产", RECOGNIZED_BALANCE_COMPONENT_COLORS["separate_account"]),
        ("OTHER_RECOGNIZED_ASSETS", "其他认可资产", RECOGNIZED_BALANCE_COMPONENT_COLORS["other_recognized"]),
    ),
    "认可负债构成": (
        # Follow the recognized-asset legend's color order so matching
        # asset/liability meanings occupy the same relative legend position.
        ("PROVISIONS", "预计负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["cash_or_provision"]),
        ("FINANCIAL_LIABILITIES", "金融负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["investment_or_financial"]),
        ("CAPITAL_LIABILITIES", "资本性负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["equity_or_capital"]),
        ("RESERVE_LIABILITIES", "准备金负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["insurance_or_reserve"]),
        ("PAYABLES_AND_ADVANCES", "应付及预收款项", RECOGNIZED_BALANCE_COMPONENT_COLORS["receivable_or_payable"]),
        ("SEPARATE_ACCOUNT_LIABILITY", "独立账户负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["separate_account"]),
        ("OTHER_RECOGNIZED_LIABILITIES", "其它认可负债", RECOGNIZED_BALANCE_COMPONENT_COLORS["other_recognized"]),
    ),
}

QUANT_RISK_STACK_RATIO_CODES = {
    "INSURANCE_RISK_CAPITAL": "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
    "NON_LIFE_INSURANCE_RISK_CAPITAL": "NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
    "MARKET_RISK_CAPITAL": "MARKET_RISK_TO_QUANT_CAPITAL",
    "CREDIT_RISK_CAPITAL": "CREDIT_RISK_TO_QUANT_CAPITAL",
    "QUANT_RISK_DIVERSIFICATION_EFFECT": "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL",
    "CONTRACT_LOSS_ABSORPTION_EFFECT": "LOSS_ABSORPTION_TO_QUANT_CAPITAL",
}

COMPONENT_STACK_DENOMINATOR_CODES = {
    "量化风险最低资本构成": "QUANT_RISK_CAPITAL",
    "各类保险风险（寿）占比": "INSURANCE_RISK_CAPITAL",
    "各类市场风险占比": "MARKET_RISK_CAPITAL",
    "各类信用风险占比": "CREDIT_RISK_CAPITAL",
    "认可资产构成": "RECOGNIZED_ASSETS",
    "认可负债构成": "RECOGNIZED_LIABILITIES",
}

RISK_RATIO_SCATTER_SPECS = {
    MARKET_RISK_ASSET_SCATTER: (
        "INTEREST_RATE_RISK_TO_ASSETS",
        "EQUITY_RISK_TO_ASSETS",
        "RECOGNIZED_ASSETS",
        "认可资产",
    ),
    CREDIT_RISK_ASSET_SCATTER: (
        "SPREAD_RISK_TO_ASSETS",
        "COUNTERPARTY_RISK_TO_ASSETS",
        "RECOGNIZED_ASSETS",
        "认可资产",
    ),
    INSURANCE_RISK_LIABILITY_SCATTER: (
        "LIFE_INSURANCE_RISK_TO_LIABILITIES",
        "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",
        "RECOGNIZED_LIABILITIES",
        "认可负债",
    ),
}
RISK_SCATTER_DUAL_VIEW_CHARTS = frozenset({
    MARKET_RISK_ASSET_SCATTER,
    CREDIT_RISK_ASSET_SCATTER,
    INSURANCE_RISK_LIABILITY_SCATTER,
})


def _valid_state(key: str, options: Iterable[str], *, multiple: bool = False) -> None:
    values = list(options)
    if key not in st.session_state:
        return
    if multiple:
        st.session_state[key] = [value for value in st.session_state[key] if value in values]
    elif st.session_state[key] not in values:
        st.session_state.pop(key, None)


def _metric_title(frame: pd.DataFrame, code: str) -> str:
    rows = frame[frame["指标编码"].astype(str).eq(code)]
    return code if rows.empty else str(rows.iloc[0]["指标名称"])


def _metric_sort_options(frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return annual-platform-style metric labels and their metric codes."""
    if frame.empty:
        return [DEFAULT_SORT_LABEL], {}
    metrics = frame[["指标编码", "指标名称"]].drop_duplicates()
    duplicated_names = metrics["指标名称"].astype(str).duplicated(keep=False)
    options = [DEFAULT_SORT_LABEL]
    lookup: dict[str, str] = {}
    for duplicated, row in zip(duplicated_names, metrics.itertuples(index=False)):
        name = str(row.指标名称).strip() or str(row.指标编码).strip()
        code = str(row.指标编码).strip()
        label = f"{name}（{code}）" if duplicated else name
        if label and label not in lookup:
            options.append(label)
            lookup[label] = code
    return [DEFAULT_SORT_LABEL, *sorted(options[1:])], lookup


def _sort_companies_by_metric(
    frame: pd.DataFrame,
    companies: Iterable[str],
    metric_code: str,
    latest_period: str,
    *,
    descending: bool,
) -> list[str]:
    """Sort companies by the selected metric while keeping missing values last."""
    original = list(dict.fromkeys(str(company) for company in companies if str(company).strip()))
    if not original or not metric_code:
        return original
    rows = frame[
        frame["公司"].astype(str).isin(original)
        & frame["指标编码"].astype(str).eq(str(metric_code))
        & frame["报告期"].astype(str).eq(str(latest_period))
    ].copy()
    rows["数值"] = pd.to_numeric(rows["数值"], errors="coerce")
    values = rows.dropna(subset=["数值"]).groupby("公司", sort=False)["数值"].first().to_dict()
    present = [company for company in original if company in values]
    missing = [company for company in original if company not in values]
    return [*sorted(present, key=lambda company: values[company], reverse=descending), *missing]


def _company_color_map(companies: Iterable[str], highlight_company: str = "无") -> dict[str, str]:
    """Match the annual platform: reserve KPMG Blue for the tracked company."""
    ordered = list(dict.fromkeys(str(company) for company in companies if str(company).strip()))
    highlight = str(highlight_company or "").strip()
    has_highlight = highlight in ordered
    palette = (
        [color for color in KPMG_CHART_COLORS if color.upper() != TRACKED_COMPANY_COLOR]
        if has_highlight
        else list(KPMG_CHART_COLORS)
    )
    if not palette:
        palette = list(KPMG_CHART_COLORS)
    colors = {
        company: palette[index % len(palette)]
        for index, company in enumerate(ordered)
    }
    if has_highlight:
        colors[highlight] = TRACKED_COMPANY_COLOR
    return colors


def _bubble_company_color_map(
    companies: Iterable[str],
    highlight_company: str = "无",
) -> dict[str, str]:
    """Keep bubble colors aligned with the shared Step7 company palette."""
    return _company_color_map(companies, highlight_company)


def _ordered_companies(frame: pd.DataFrame) -> list[str]:
    """Keep companies in the integrated table's original display order."""
    if frame.empty or "公司" not in frame.columns:
        return []
    return list(
        dict.fromkeys(
            value
            for value in frame["公司"].fillna("").astype(str).str.strip()
            if value
        )
    )


def _company_scope_for_types(
    frame: pd.DataFrame,
    selected_types: Iterable[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Return the exact company scope implied by the company-type widget."""
    selected = [
        str(value).strip()
        for value in selected_types
        if str(value).strip()
    ]
    specific_types = [
        value for value in selected if value != ALL_COMPANY_TYPES
    ]
    if not specific_types:
        company_source = frame
    else:
        company_source = frame[
            frame["同业分类"].fillna("").astype(str).str.strip().isin(specific_types)
        ].copy()
    return company_source, _ordered_companies(company_source)


def _sync_company_selection_state(
    selected_types: Iterable[str],
    available_companies: Iterable[str],
    *,
    signature_key: str = "_s7_company_scope_signature",
) -> str:
    """Return a fresh downstream widget key whenever its upstream scope changes."""
    companies = list(dict.fromkeys(
        str(company).strip()
        for company in available_companies
        if str(company).strip()
    ))
    selected = tuple(sorted({
        str(value).strip()
        for value in selected_types
        if str(value).strip() and str(value).strip() != ALL_COMPANY_TYPES
    }))
    signature = (selected or (ALL_COMPANY_TYPES,), tuple(companies))
    revision_key = "_s7_company_selection_revision"
    st.session_state.setdefault(revision_key, 0)
    if st.session_state.get(signature_key) != signature:
        if signature_key in st.session_state:
            st.session_state[revision_key] += 1
        st.session_state[signature_key] = signature
    selection_key = f"s7_companies_scope_v3_{st.session_state[revision_key]}"
    current = st.session_state.get(selection_key, companies)
    st.session_state[selection_key] = [
        company for company in current if company in companies
    ]
    return selection_key


def _reset_company_selection_for_types(frame: pd.DataFrame) -> None:
    """Apply a company-type change before Streamlit redraws downstream widgets."""
    selected_types = st.session_state.get("s7_company_types", [ALL_COMPANY_TYPES])
    _, companies = _company_scope_for_types(frame, selected_types)
    selected = tuple(sorted({
        str(value).strip()
        for value in selected_types
        if str(value).strip() and str(value).strip() != ALL_COMPANY_TYPES
    }))
    revision_key = "_s7_company_selection_revision"
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1
    selection_key = f"s7_companies_scope_v3_{st.session_state[revision_key]}"
    st.session_state[selection_key] = list(companies)
    st.session_state["_s7_company_scope_signature"] = (
        selected or (ALL_COMPANY_TYPES,),
        tuple(companies),
    )


def _selected_chart_names(frame: pd.DataFrame) -> list[str]:
    first = st.session_state.get("company_nav_level_one", "")
    if first == PRINT_ALL_LABEL:
        return list(dict.fromkeys(entry.chart_name for entry in COMPANY_NAVIGATION))
    return resolve_chart_selection(
        first,
        st.session_state.get("company_nav_level_two", "全部"),
        st.session_state.get("company_nav_chart", ""),
    )


def _report_style() -> None:
    st.markdown(
        """
        <style>
        [class*="st-key-s7_report_module_"],
        [class*="st-key-s7_trend_group_"],
        .solvency-report-title, .solvency-module-title,
        .solvency-report-analysis, .solvency-report-footnote,
        .solvency-chart-legend {
            font-family:Microsoft YaHei, 微软雅黑, sans-serif;
        }
        [class*="st-key-s7_report_module_"] {
            font-size:14px;
        }
        [class*="st-key-s7_report_module_"] p,
        [class*="st-key-s7_report_module_"] label,
        [class*="st-key-s7_report_module_"] button,
        [class*="st-key-s7_report_module_"] input {
            font-family:Microsoft YaHei, 微软雅黑, sans-serif!important;
            font-size:14px;
        }
        [data-testid="stMainBlockContainer"] {
            max-width:none!important; padding-left:10px!important; padding-right:10px!important;
        }
        .solvency-report-title {
            color:#00338D; font-size:30px; font-weight:900;
            border-bottom:2px solid #00338D; padding-bottom:8px; margin:6px 0 18px;
        }
        .solvency-module-title {
            color:#1E293B; font-size:17px; font-weight:600; letter-spacing:1px;
            background:rgba(255,255,255,.45); border:1px solid rgba(255,255,255,.8);
            border-radius:8px; padding:8px 18px; width:fit-content;
            box-shadow:0 4px 12px rgba(0,0,0,.03), inset 0 1px 0 rgba(255,255,255,.5);
            margin:10px 0 18px;
        }
        [class*="st-key-s7_report_module_"] h3 {
            color:#1E293B!important; font-size:17px!important; font-weight:600!important;
            letter-spacing:1px; background:rgba(255,255,255,.45)!important;
            border:1px solid rgba(255,255,255,.8); border-radius:8px!important;
            padding:8px 18px!important; width:fit-content; margin:10px 0 18px!important;
            box-shadow:0 4px 12px rgba(0,0,0,.03), inset 0 1px 0 rgba(255,255,255,.5)!important;
        }
        .solvency-analysis-note {
            background:#F2F6FC; border-left:4px solid #00338D;
            border-radius:4px; padding:9px 13px; margin:6px 0 0; color:#243B53;
        }
        .solvency-analysis-spacer {height:16px; width:100%;}
        .solvency-chart-legend {
            display:flex; flex-wrap:wrap; justify-content:flex-end;
            align-items:center; gap:10px 18px;
            margin:2px 10px 8px 0; color:#0C233C; font-size:10px;
        }
        .solvency-chart-legend__item {display:inline-flex; align-items:center; gap:6px;}
        .solvency-chart-legend__symbol {
            --legend-color:#0C233C; display:inline-block; flex:0 0 auto;
        }
        .solvency-chart-legend__symbol--square {
            width:11px; height:11px; border-radius:2px; background:var(--legend-color);
        }
        .solvency-chart-legend__symbol--outline {
            width:13px; height:11px; border-radius:3px;
            border:2px solid var(--legend-color); background:transparent;
            box-sizing:border-box;
        }
        .solvency-chart-legend__symbol--line,
        .solvency-chart-legend__symbol--dash {
            width:22px; height:0; border-top:3px solid var(--legend-color);
        }
        .solvency-chart-legend__symbol--dash {border-top-style:dashed;}
        .solvency-analysis-ai {color:#D84315; font-size:12px; margin-top:5px;}
        .solvency-analysis-ai--error {color:#C00000;}
        .solvency-report-analysis {
            background:#F4F7FC; border-left:4px solid #00338D;
            border-radius:3px; padding:7px 10px; margin:4px 0 10px;
        }
        .solvency-report-analysis p {margin:2px 0; line-height:1.45;}
        .solvency-note-default {color:#0A1F5C; font-size:14px;}
        .solvency-note-custom {color:#1E49E2; font-size:14px; font-weight:600;}
        .solvency-report-footnote {color:#777; font-size:12px; font-style:italic; margin:4px 0 16px;}
        [class*="st-key-annual_company_panel_"] {
            border:1px solid #EAEAEA!important; border-radius:0!important;
            padding:14px 4px 2px!important; box-shadow:none!important;
            box-sizing:border-box!important;
        }
        [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] {
            flex-wrap:nowrap!important;
        }
        [class*="st-key-annual_company_grid_"] [data-testid="stColumn"] {
            min-width:0!important;
            flex:1 1 0!important;
        }
        .annual-company-title {
            color:#00338D; font-size:13px; font-weight:700; line-height:1.25;
            text-align:center; white-space:nowrap; min-height:17px;
            margin:0 0 30px; padding:0 4px;
        }
        [class*="st-key-annual_company_panel_"] [data-testid="stMarkdownContainer"]:has(.annual-company-title) {
            margin:0!important; padding:0!important;
        }
        [class*="st-key-annual_company_panel_gray_"] {
            background:rgba(200,200,200,.12)!important;
        }
        [class*="st-key-annual_company_panel_white_"] {
            background:rgba(255,255,255,0)!important;
        }
        [class*="st-key-annual_company_panel_tracked_"] {
            background:rgba(0,51,141,.03)!important;
            border:1.5px solid rgba(0,51,141,.35)!important;
        }
        [class*="st-key-s7_report_module_"] [data-testid="stTable"] table,
        [class*="st-key-s7_report_module_"] [data-testid="stDataFrame"] {
            font-family:Microsoft YaHei, 微软雅黑, sans-serif!important;
            font-size:11px!important;
        }
        [class*="st-key-s7_report_module_"] [data-testid="stTable"] thead th {
            background:#00338D; color:#FFFFFF; font-size:11px;
            font-weight:700; padding:6px 4px; border:1px solid #FFFFFF;
        }
        [class*="st-key-s7_report_module_"] [data-testid="stTable"] tbody td {
            font-size:11px; padding:4px; border:1px solid #EAEAEA;
        }
        [class*="st-key-s7_report_module_"] {break-inside:avoid-page; page-break-inside:avoid;}
        [class*="st-key-s7_trend_group_"] {break-inside:avoid-page; page-break-inside:avoid;}
        .solvency-print-cover {
            position:relative; width:100%; aspect-ratio:16/9; overflow:hidden;
            margin:0; padding:0; background:#00338D;
            -webkit-print-color-adjust:exact; print-color-adjust:exact;
        }
        .solvency-print-cover img, .solvency-cover-fallback {
            position:absolute; inset:0; width:100%; height:100%; object-fit:cover;
        }
        .solvency-cover-fallback {background:linear-gradient(135deg,#00338D,#1E49E2);}
        .solvency-print-cover__text {
            position:absolute; inset:0; z-index:2; display:flex; flex-direction:column;
            justify-content:center; align-items:flex-start; padding:0 8%; box-sizing:border-box;
            color:#fff; text-shadow:2px 2px 5px rgba(0,0,0,.45);
        }
        .solvency-print-cover__title {font-size:48px; font-weight:900; line-height:1.35;}
        .solvency-print-cover__subtitle {font-size:23px; font-weight:650; margin-top:14px;}
        .solvency-print-cover__date {font-size:19px; margin-top:18px;}
        .solvency-print-cover--front {page-break-after:always; break-after:page;}
        .solvency-print-cover--back {page-break-before:always; break-before:page;}
        /* The print button adds this class before opening the browser dialog.
           Keeping the final 16:9 geometry active during that short preflight lets
           Vega reflow and lets JavaScript measure the exact printable content. */
        html.solvency-print-mode-widescreen {
            --solvency-print-page-width:338.67mm;
            --solvency-print-page-height:190.5mm;
            --solvency-print-content-width:314.67mm;
            --solvency-print-content-height:174.5mm;
        }
        html.solvency-print-mode-widescreen,
        html.solvency-print-mode-widescreen body,
        html.solvency-print-mode-widescreen [data-testid="stAppViewContainer"],
        html.solvency-print-mode-widescreen [data-testid="stMain"] {
            width:var(--solvency-print-page-width)!important;
            max-width:var(--solvency-print-page-width)!important;
            margin:0 auto!important; padding:0!important;
            overflow:visible!important; box-sizing:border-box!important;
        }
        html.solvency-print-mode-widescreen [data-testid="stMainBlockContainer"] {
            width:var(--solvency-print-page-width)!important;
            min-width:var(--solvency-print-page-width)!important;
            max-width:var(--solvency-print-page-width)!important;
            margin:0 auto!important; padding:0!important;
            box-sizing:border-box!important; overflow:visible!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] {
            --solvency-print-scale:1;
            width:var(--solvency-print-page-width)!important;
            min-width:var(--solvency-print-page-width)!important;
            max-width:var(--solvency-print-page-width)!important;
            height:var(--solvency-print-page-height)!important;
            min-height:var(--solvency-print-page-height)!important;
            max-height:var(--solvency-print-page-height)!important;
            padding:8mm 12mm!important;
            margin:0 auto!important; box-sizing:border-box!important;
            overflow:hidden!important;
            display:flex!important; align-items:center!important;
            justify-content:center!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] > [data-testid="stVerticalBlock"] {
            width:var(--solvency-print-content-width)!important;
            min-width:var(--solvency-print-content-width)!important;
            max-width:var(--solvency-print-content-width)!important;
            flex:0 0 auto!important;
            transform:scale(var(--solvency-print-scale))!important;
            transform-origin:center center!important;
            gap:0.45rem!important;
            box-sizing:border-box!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] [data-testid="stElementContainer"],
        html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] [data-testid="stVegaLiteChart"] {
            width:100%!important; min-width:0!important; max-width:100%!important;
            box-sizing:border-box!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-annual_company_grid_"],
        html.solvency-print-mode-widescreen [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] {
            display:grid!important; width:100%!important;
            min-width:0!important; max-width:100%!important;
            margin-left:0!important; margin-right:0!important;
            grid-template-columns:none!important; grid-auto-flow:column!important;
            grid-auto-columns:minmax(0,1fr)!important; gap:8px!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width:100%!important; min-width:0!important; max-width:100%!important;
            justify-self:stretch!important; box-sizing:border-box!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-annual_company_panel_"] {
            width:100%!important; min-width:0!important; max-width:100%!important;
            height:100%!important; padding:12px 3px 2px!important;
        }
        html.solvency-print-mode-widescreen .annual-company-title {
            margin:0 0 12px!important;
        }
        html.solvency-print-mode-widescreen [class*="st-key-s7_company_bars_page_"] {
            break-before:auto!important; page-break-before:auto!important;
        }
        @media print {
            [data-testid="stHeader"], [data-testid="stSidebar"],
            [data-testid="stToolbar"], [data-testid="collapsedControl"],
            [data-testid="stExpander"], div[role="tablist"], h1,
            .platform-page-heading, .solvency-report-title,
            .solvency-no-print,
            [class*="st-key-capital_bubble_print_controls_"],
            [class*="st-key-risk_scatter_print_controls_"] {display:none!important;}
            .capital_efficiency_zoom_brush,
            .capital_efficiency_zoom_brush_bg,
            [class*="capital_efficiency_zoom_"][class*="_brush"],
            .risk_ratio_zoom_brush,
            .risk_ratio_zoom_brush_bg,
            [class*="risk_ratio_zoom_"][class*="_brush"] {
                display:none!important;
            }
            [data-testid="stElementContainer"]:has(.platform-page-heading),
            [data-testid="stElementContainer"]:has(.solvency-report-title) {display:none!important;}
            [class*="st-key-s7_report_module_"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-s7_report_module_"]:has(.solvency-grouped-trend) {
                break-inside:auto!important; page-break-inside:auto!important;
            }
            /* A dual-view metric deliberately spans two sheets: the trend stays
               on the first sheet and the complete company-bar group starts the next. */
            [class*="st-key-s7_report_module_"]:has([class*="st-key-s7_company_bars_page_"]) {
                break-inside:auto!important; page-break-inside:auto!important;
            }
            [class*="st-key-s7_company_bars_page_"] {
                break-before:page!important; page-break-before:always!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-s7_trend_group_"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-s7_report_module_"] [data-testid="stVegaLiteChart"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                max-width:100%!important;
            }
            /* Keep every company card intact.  The print-only grid packs cards
               tightly enough to avoid an otherwise nearly empty spill page. */
            [class*="st-key-annual_company_grid_"],
            [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] {
                display:grid!important;
                grid-template-columns:repeat(auto-fit,minmax(155px,1fr))!important;
                align-items:stretch!important; gap:8px!important;
                width:100%!important; min-width:0!important; max-width:100%!important;
                break-inside:auto!important; page-break-inside:auto!important;
            }
            [class*="st-key-annual_company_grid_"] > [data-testid="stLayoutWrapper"],
            [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stLayoutWrapper"],
            [class*="st-key-annual_company_grid_"] [data-testid="stElementContainer"]:has([class*="st-key-annual_company_panel_"]),
            [class*="st-key-annual_company_grid_"] [data-testid="stVerticalBlockBorderWrapper"]:has([class*="st-key-annual_company_panel_"]) {
                width:auto!important; min-width:0!important; max-width:100%!important;
                box-sizing:border-box!important;
            }
            /* st.columns keeps a flex-basis width after its parent becomes a
               print grid.  Fill the grid track instead of shrinking to it. */
            [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                width:100%!important; min-width:0!important; max-width:100%!important;
                justify-self:stretch!important; box-sizing:border-box!important;
            }
            [class*="st-key-annual_company_panel_"] {
                width:100%!important; min-width:0!important; max-width:100%!important;
                align-self:stretch!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-annual_company_panel_"] [data-testid="stVegaLiteChart"] {
                width:100%!important; min-width:0!important; max-width:100%!important;
                min-height:280px!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            [class*="st-key-annual_company_panel_"] [data-testid="stVegaLiteChart"] canvas,
            [class*="st-key-annual_company_panel_"] [data-testid="stVegaLiteChart"] svg {
                width:100%!important; min-width:0!important; max-width:100%!important;
                height:280px!important; min-height:280px!important; max-height:280px!important;
            }
            /* Every report module is a self-contained 16:9 sheet.  The print
               control measures the complete title/note/legend/chart block and
               sets --solvency-print-scale so browser, font and DPI differences
               cannot push any part onto another page. */
            html.solvency-print-mode-widescreen {
                --solvency-print-page-width:338.67mm;
                --solvency-print-page-height:190.5mm;
                --solvency-print-content-width:314.67mm;
                --solvency-print-content-height:174.5mm;
            }
            html.solvency-print-mode-widescreen,
            html.solvency-print-mode-widescreen body,
            html.solvency-print-mode-widescreen [data-testid="stAppViewContainer"],
            html.solvency-print-mode-widescreen [data-testid="stMain"] {
                width:var(--solvency-print-page-width)!important;
                max-width:var(--solvency-print-page-width)!important;
                margin:0 auto!important; padding:0!important;
                overflow:visible!important; box-sizing:border-box!important;
                -webkit-print-color-adjust:exact!important;
                print-color-adjust:exact!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stMainBlockContainer"] {
                width:var(--solvency-print-page-width)!important;
                min-width:var(--solvency-print-page-width)!important;
                max-width:var(--solvency-print-page-width)!important;
                margin:0 auto!important; padding:0!important;
                box-sizing:border-box!important; overflow:visible!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-annual_company_grid_"],
            html.solvency-print-mode-widescreen [class*="st-key-annual_company_grid_"] [data-testid="stHorizontalBlock"] {
                width:100%!important; min-width:0!important; max-width:100%!important;
                margin-left:0!important; margin-right:0!important;
                grid-template-columns:none!important;
                grid-auto-flow:column!important;
                grid-auto-columns:minmax(0,1fr)!important;
                gap:8px!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-annual_company_panel_"] {
                height:100%!important; padding:12px 3px 2px!important;
            }
            html.solvency-print-mode-widescreen .annual-company-title {
                margin:0 0 12px!important;
            }
            html.solvency-print-mode-widescreen .key-solvency-overview {
                width:100%!important; min-width:0!important; max-width:100%!important;
                margin-left:0!important; margin-right:0!important;
                table-layout:fixed!important;
            }
            html.solvency-print-mode-widescreen .key-solvency-overview th,
            html.solvency-print-mode-widescreen .key-solvency-overview td {
                min-width:0!important; max-width:none!important;
                white-space:normal!important; overflow-wrap:anywhere!important;
                word-break:break-word!important; box-sizing:border-box!important;
                font-size:9px!important; padding:3px 2px!important;
            }
            html.solvency-print-mode-widescreen .key-solvency-overview th:first-child,
            html.solvency-print-mode-widescreen .key-solvency-overview td:first-child {
                width:6.5%!important; white-space:nowrap!important;
                word-break:keep-all!important; overflow-wrap:normal!important;
            }
            .solvency-report-page-break {break-before:page; page-break-before:always;}
            html.solvency-print-mode-widescreen .solvency-print-cover {
                width:338.67mm!important; height:190.5mm!important; max-width:338.67mm!important;
                aspect-ratio:auto!important; margin:0 auto!important; padding:0!important;
                box-sizing:border-box!important; overflow:hidden!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            html.solvency-print-mode-widescreen .solvency-print-cover img {
                width:100%!important; height:100%!important; object-fit:contain!important;
                object-position:center!important; display:block!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stElementContainer"]:has(.solvency-print-cover),
            html.solvency-print-mode-widescreen [data-testid="stMarkdownContainer"]:has(.solvency-print-cover),
            html.solvency-print-mode-widescreen .stMarkdown:has(.solvency-print-cover) {
                width:338.67mm!important; max-width:338.67mm!important;
                height:190.5mm!important; margin:0 auto!important; padding:0!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stVerticalBlock"]:has(.solvency-print-cover) {
                gap:0!important; margin:0!important; padding:0!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] {
                --solvency-print-scale:1;
                width:var(--solvency-print-page-width)!important;
                min-width:var(--solvency-print-page-width)!important;
                max-width:var(--solvency-print-page-width)!important;
                height:var(--solvency-print-page-height)!important;
                min-height:var(--solvency-print-page-height)!important;
                max-height:var(--solvency-print-page-height)!important;
                padding:8mm 12mm!important;
                margin:0 auto!important; box-sizing:border-box!important;
                overflow:hidden!important;
                display:flex!important; align-items:center!important;
                justify-content:center!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] > [data-testid="stVerticalBlock"] {
                width:var(--solvency-print-content-width)!important;
                min-width:var(--solvency-print-content-width)!important;
                max-width:var(--solvency-print-content-width)!important;
                flex:0 0 auto!important;
                transform:scale(var(--solvency-print-scale))!important;
                transform-origin:center center!important;
                gap:0.45rem!important;
                box-sizing:border-box!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] [data-testid="stElementContainer"],
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] [data-testid="stVegaLiteChart"] {
                width:100%!important; min-width:0!important; max-width:100%!important;
                box-sizing:border-box!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_company_bars_page_"] {
                break-before:auto!important; page-break-before:auto!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] img {
                max-width:100%!important; height:auto!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] table {
                width:100%!important; max-width:100%!important;
                table-layout:fixed!important; box-sizing:border-box!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stElementContainer"]:has(.solvency-print-cover--front) {
                break-after:page!important; page-break-after:always!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stElementContainer"]:has(.solvency-print-cover--back) {
                break-before:page!important; page-break-before:always!important;
            }
            .solvency-module-title {
                color:#00338D!important; font-size:30px!important; font-weight:900!important;
                background:transparent!important; border:none!important; border-radius:0!important;
                box-shadow:none!important; padding:0!important; margin:10px 0 8px!important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_image_override_controls(
    chart_names: list[str],
    notes: dict[str, dict[str, str]],
) -> dict[str, dict[str, object]]:
    overrides = st.session_state.setdefault("s7_image_overrides", {})
    st.caption("手动上传图片（PNG/JPG）；选择对应图表后，图片将替代系统生成图表进入网页和打印报告。")
    uploads = st.file_uploader(
        "上传图表截图",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="s7_image_uploads",
    )
    filename_targets = {
        str(note.get("图片文件名", "")).strip(): chart_name
        for chart_name, note in notes.items()
        if str(note.get("图片文件名", "")).strip()
    }
    for index, upload in enumerate(uploads or []):
        state_key = f"s7_image_target_{index}"
        if state_key not in st.session_state:
            suggested_target = filename_targets.get(upload.name, "不覆盖/跳过")
            st.session_state[state_key] = (
                suggested_target if suggested_target in chart_names else "不覆盖/跳过"
            )
        c1, c2 = st.columns([1, 1.4], vertical_alignment="center")
        with c1:
            target = st.selectbox(
                f"{upload.name} 对应图表",
                ["不覆盖/跳过", *chart_names],
                key=state_key,
            )
        with c2:
            st.image(upload, width="stretch")
        if target != "不覆盖/跳过":
            overrides[target] = {
                "name": upload.name,
                "data": upload.getvalue(),
            }
    if overrides:
        st.caption("当前已覆盖：" + "、".join(overrides))
        disabled = st.multiselect(
            "本次暂不使用的图片覆盖",
            list(overrides),
            key="s7_disabled_image_overrides",
            placeholder="如需恢复系统图表，可在此选择",
        )
        return {name: value for name, value in overrides.items() if name not in disabled}
    return {}


def _convert_unit(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    result = frame.copy()
    target_yuan_map = {
        "十亿元": 1_000_000_000,
        "亿元": 100_000_000,
        "百万元": 1_000_000,
        "十万元": 100_000,
    }
    if target not in target_yuan_map:
        return result
    target_yuan = target_yuan_map[target]
    source_yuan = {
        "元": 1,
        "万元": 10_000,
        "十万元": 100_000,
        "百万元": 1_000_000,
        "亿元": 100_000_000,
        "十亿元": 1_000_000_000,
    }
    # This function runs whenever the company/category selection changes.  The
    # previous row-wise ``apply`` created a temporary Series for every record,
    # which made a full report redraw needlessly expensive.  Keep the same
    # conversion semantics while doing the work in vectorized pandas columns.
    numeric_values = pd.to_numeric(result["数值"], errors="coerce")
    clean_units = result["单位"].fillna("").astype(str).str.strip()
    source_scales = clean_units.map(source_yuan)
    amount_mask = source_scales.notna()
    result["数值"] = numeric_values
    result.loc[amount_mask, "数值"] = (
        numeric_values.loc[amount_mask]
        * source_scales.loc[amount_mask]
        / target_yuan
    )
    result.loc[amount_mask, "单位"] = target
    return result


def _analysis_completion_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("模型接口地址不能为空。")
    return value if value.endswith("/chat/completions") else f"{value}/chat/completions"


@st.cache_data(show_spinner=False, ttl="12h", max_entries=256)
def _call_ai_analysis_cached(
    data_text: str,
    metric_name: str,
    latest_period: str,
    api_key: str,
    base_url: str,
    model: str,
) -> str:
    prompt = (
        "你是资深四大保险精算顾问。请根据以下偿付能力同业对标数据，"
        "用一句中文给出专业点评，指出最高、最低、均值及值得关注的差异；"
        "不要编造数据，不超过80字。\n"
        f"报告期：{latest_period}\n指标：{metric_name}\n数据：{data_text}"
    )
    payload = {
        "model": normalize_model_id(base_url, model),
        "messages": [{"role": "user", "content": prompt}],
    }
    payload.update(model_request_parameters(base_url, model))
    response = post_json_with_retry(
        requests.post,
        _analysis_completion_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not response.ok:
        try:
            detail = response.json().get("error", {}).get("message", "")
        except Exception:
            detail = response.text
        raise RuntimeError(detail or f"HTTP {response.status_code}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型接口返回结构中没有 choices/message/content。") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content).strip()


def _ai_analysis_for_latest_period(
    latest: pd.DataFrame,
    metric_name: str,
    latest_period: str,
) -> tuple[str, bool]:
    settings = {
        "api_key": str(st.session_state.get("llm_api_key", "")).strip(),
        "base_url": str(st.session_state.get("llm_base_url", "")).strip(),
        "model": str(st.session_state.get("llm_model", "")).strip(),
    }
    if not all(settings.values()):
        return (
            "当前登录会话的 AI 配置不完整。请退出登录，在登录页填写 "
            "Base URL、模型名称和 API Key 后重新进入。"
        ), True
    company_values = "、".join(
        f"{row['公司']}={format_chart_value(row['数值'], row.get('单位', ''), row.get('数据类型', ''))}"
        for _, row in latest.sort_values("数值", ascending=False).iterrows()
    )
    sample = latest.iloc[0]
    average_text = format_chart_value(
        latest["数值"].mean(),
        sample.get("单位", ""),
        sample.get("数据类型", ""),
    )
    data_text = f"{company_values}；样本均值={average_text}"
    try:
        return _call_ai_analysis_cached(
            data_text,
            metric_name,
            latest_period,
            settings["api_key"],
            settings["base_url"],
            settings["model"],
        ), False
    except Exception as exc:
        return f"AI 分析失败：{exc}", True


def _unavailable_metric_details(
    frame: pd.DataFrame,
    code: str,
    periods: Iterable[str],
) -> list[tuple[str, str, str]]:
    if frame is None or frame.empty or code not in {
        "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
        "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",
    }:
        return []
    scoped = filter_analysis_frame(frame, metric_code=code, periods=periods)
    if scoped.empty:
        return []
    values = pd.to_numeric(scoped["数值"], errors="coerce")
    states = scoped["披露状态"].fillna("").astype(str).str.strip()
    scoped = scoped[values.isna() & states.eq("无法计算")].copy()
    if scoped.empty:
        return []
    details: list[tuple[str, str, str]] = []
    for _, row in scoped.drop_duplicates(["公司", "报告期"], keep="last").iterrows():
        reason = str(row.get("备注", "") or "计算依赖不足").strip()
        reason = re.sub(r"^无法计算[：:]?\s*", "", reason) or "计算依赖不足"
        details.append((str(row.get("公司", "")).strip(), str(row.get("报告期", "")).strip(), reason))
    return details


def _policy_surplus_coverage_note(source: pd.DataFrame, ratio_rows: pd.DataFrame) -> str:
    """Describe the numerator's disclosed capital levels for the plotted periods."""
    if source.empty or ratio_rows.empty:
        return ""
    pairs = {(str(row["公司"]), str(row["报告期"])) for _, row in ratio_rows.iterrows()}
    components = source.loc[
        source["指标编码"].astype(str).isin(POLICY_SURPLUS_COMPONENT_CODES)
        & source["期间口径"].map(canonical_period_scope).eq("本季度末数")
    ]
    labels = ("核心一级", "核心二级", "附属一级", "附属二级")
    by_kind: dict[str, dict[str, list[str]]] = {"core": {}, "all": {}, "partial": {}}
    partial_levels: dict[tuple[str, str], list[str]] = {}
    for (company, period), group in components.groupby(["公司", "报告期"], sort=False):
        company, period = str(company), str(period)
        if (company, period) not in pairs:
            continue
        available: dict[str, tuple[float, str]] = {}
        for code, rows in group.groupby("指标编码", sort=False):
            rows = rows.loc[~rows.get("来源类型", pd.Series("", index=rows.index)).fillna("").astype(str).eq("宽表空白推定")]
            rows = rows.loc[~rows.get("披露状态", pd.Series("", index=rows.index)).fillna("").astype(str).isin({"未披露", "不适用", "无法计算"})]
            numeric = pd.to_numeric(rows["数值"], errors="coerce").dropna()
            if len(numeric.unique()) == 1:
                selected = rows.loc[numeric.index[0]]
                available[str(code)] = (float(numeric.iloc[0]), str(selected.get("披露状态", "")))
        if not available:
            continue
        core = POLICY_SURPLUS_COMPONENT_CODES[0]
        others = POLICY_SURPLUS_COMPONENT_CODES[1:]
        only_core = core in available and (
            not any(code in available for code in others)
            or (all(code in available and available[code][0] == 0 for code in others)
                and not any(available[code][1] in {"已披露为0", "disclosed_zero"} for code in others))
        )
        if only_core:
            kind = "core"
        elif all(code in available for code in POLICY_SURPLUS_COMPONENT_CODES):
            kind = "all"
        else:
            kind = "partial"
            partial_levels[(company, period)] = [
                label for code, label in zip(POLICY_SURPLUS_COMPONENT_CODES, labels) if code in available
            ]
        by_kind[kind].setdefault(company, []).append(period)

    def company_list(kind: str) -> str:
        names = []
        for company, periods in by_kind[kind].items():
            selected = {period for name, period in pairs if name == company}
            suffix = "" if set(periods) == selected else f"（{'、'.join(sort_report_periods(periods))}）"
            names.append(company + suffix)
        return "、".join(names)

    notes = []
    if by_kind["core"]:
        notes.append(f"{company_list('core')}：分子仅计入核心一级资本的保单未来盈余，其余三个资本层级未披露")
    if by_kind["all"]:
        notes.append(f"{company_list('all')}：分子计入四类保单未来盈余")
    for company, periods in by_kind["partial"].items():
        for period in sort_report_periods(periods):
            included = partial_levels[(company, period)]
            notes.append(f"{company}（{period}）：分子计入已披露的{'、'.join(included)}保单未来盈余")
    return "；".join(notes)


def _core_policy_surplus_coverage_note(source: pd.DataFrame, ratio_rows: pd.DataFrame) -> str:
    """Describe whether the core-capital ratio uses core T1 only or both core levels."""
    if source.empty or ratio_rows.empty:
        return ""
    pairs = {(str(row["公司"]), str(row["报告期"])) for _, row in ratio_rows.iterrows()}
    components = source.loc[
        source["指标编码"].astype(str).isin(POLICY_SURPLUS_COMPONENT_CODES)
        & source["期间口径"].map(canonical_period_scope).eq("本季度末数")
    ]
    by_kind: dict[str, dict[str, list[str]]] = {"core_t1": {}, "both": {}, "partial": {}}
    for (company, period), group in components.groupby(["公司", "报告期"], sort=False):
        company, period = str(company), str(period)
        if (company, period) not in pairs:
            continue
        available: dict[str, tuple[float, str]] = {}
        for code, rows in group.groupby("指标编码", sort=False):
            rows = rows.loc[~rows.get("来源类型", pd.Series("", index=rows.index)).fillna("").astype(str).eq("宽表空白推定")]
            rows = rows.loc[~rows.get("披露状态", pd.Series("", index=rows.index)).fillna("").astype(str).isin({"未披露", "不适用", "无法计算"})]
            numeric = pd.to_numeric(rows["数值"], errors="coerce").dropna()
            if len(numeric.unique()) == 1:
                selected = rows.loc[numeric.index[0]]
                available[str(code)] = (float(numeric.iloc[0]), str(selected.get("披露状态", "")))

        core_t1, core_t2 = POLICY_SURPLUS_COMPONENT_CODES[:2]
        ancillary = POLICY_SURPLUS_COMPONENT_CODES[2:]
        if core_t1 not in available and core_t2 not in available:
            continue
        core_t2_is_explicit_zero = (
            core_t2 in available
            and available[core_t2][1] in {"已披露为0", "disclosed_zero"}
        )
        ancillary_has_value = any(
            code in available and available[code][0] != 0 for code in ancillary
        )
        core_t1_only = core_t1 in available and (
            core_t2 not in available
            or (
                available[core_t2][0] == 0
                and not core_t2_is_explicit_zero
                and not ancillary_has_value
            )
        )
        if core_t1_only:
            kind = "core_t1"
        elif core_t1 in available and core_t2 in available:
            kind = "both"
        else:
            kind = "partial"
        by_kind[kind].setdefault(company, []).append(period)

    def company_list(kind: str) -> str:
        names = []
        for company, company_periods in by_kind[kind].items():
            selected = {period for name, period in pairs if name == company}
            suffix = (
                "" if set(company_periods) == selected
                else f"（{'、'.join(sort_report_periods(company_periods))}）"
            )
            names.append(company + suffix)
        return "、".join(names)

    notes: list[str] = []
    if by_kind["core_t1"]:
        notes.append(
            f"{company_list('core_t1')}：分子仅计入核心一级资本的保单未来盈余，"
            "核心二级资本没有披露这个明细"
        )
    if by_kind["both"]:
        notes.append(
            f"{company_list('both')}：分子计入核心一级及核心二级资本的保单未来盈余"
        )
    if by_kind["partial"]:
        notes.append(
            f"{company_list('partial')}：仅披露核心二级资本的保单未来盈余"
        )
    return "；".join(notes)


def _render_report_metric(
    frame: pd.DataFrame,
    code: str,
    *,
    status_frame: pd.DataFrame | None = None,
    periods: list[str],
    show_labels: bool,
    decimals: int,
    unit_mode: str,
    transparent: bool,
    show_average: bool,
    highlight_company: str,
    enable_ai: bool,
    key_prefix: str,
    company_panels: bool = False,
    company_period_bars: bool = False,
) -> None:
    unavailable = _unavailable_metric_details(
        status_frame if status_frame is not None else frame,
        code,
        periods,
    )
    unavailable_companies = list(dict.fromkeys(company for company, _, _ in unavailable))
    unavailable_legend = (
        [("#B8BDC7", "无法计算：" + "、".join(unavailable_companies), "square")]
        if unavailable_companies
        else []
    )
    if unavailable:
        reason_text = "；".join(
            f"{company}（{period}）：{reason}"
            for company, period, reason in unavailable
        )
        st.caption("注：无法计算原因——" + reason_text)
    metric_frame = filter_analysis_frame(frame, metric_code=code, periods=periods)
    if metric_frame.empty:
        if unavailable_legend:
            _render_chart_legend(unavailable_legend)
        st.caption(f"{_metric_title(status_frame if status_frame is not None else frame, code)}：当前筛选范围没有可计算数据。")
        return
    coverage_note = ""
    if code in {
        "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
        "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",
    }:
        coverage_source = status_frame if status_frame is not None else frame
        coverage_note = (
            _core_policy_surplus_coverage_note(coverage_source, metric_frame)
            if code == "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"
            else _policy_surplus_coverage_note(coverage_source, metric_frame)
        )
        if coverage_note:
            st.caption("注：保单未来盈余计算口径——" + coverage_note + "。")
    metric_frame = _convert_unit(metric_frame, unit_mode)
    metric_frame = convert_multiple_units_to_percent(metric_frame)
    metric_name = str(metric_frame.iloc[0]["指标名称"])
    st.caption(
        f"公司 {metric_frame['公司'].nunique()} 家　｜　"
        f"报告期 {metric_frame['报告期'].nunique()} 个　｜　"
        f"记录 {len(metric_frame):,} 条"
    )
    ordered_periods = sort_report_periods(metric_frame["报告期"])
    latest_period = ordered_periods[-1]
    latest = metric_frame[metric_frame["报告期"].astype(str).eq(latest_period)].copy()
    latest["数值"] = pd.to_numeric(latest["数值"], errors="coerce")
    latest = latest.dropna(subset=["数值"]).drop_duplicates(subset=["公司"], keep="last")
    if not latest.empty:
        max_row = latest.loc[latest["数值"].idxmax()]
        min_row = latest.loc[latest["数值"].idxmin()]
        ai_html = ""
        if enable_ai:
            ai_text, ai_failed = _ai_analysis_for_latest_period(latest, metric_name, latest_period)
            ai_class = "solvency-analysis-ai solvency-analysis-ai--error" if ai_failed else "solvency-analysis-ai"
            ai_html = f"<div class='{ai_class}'><b>AI 分析：</b>{escape(ai_text)}</div>"
        st.markdown(
            "<div class='solvency-analysis-note'>"
            f"最新报告期 {latest_period}：最高为 {max_row['公司']} "
            f"（{format_chart_value(max_row['数值'], max_row.get('单位', ''), max_row.get('数据类型', ''), decimals)}），"
            f"最低为 {min_row['公司']} "
            f"（{format_chart_value(min_row['数值'], min_row.get('单位', ''), min_row.get('数据类型', ''), decimals)}）。"
            f"{ai_html}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='solvency-analysis-spacer' aria-hidden='true'></div>",
            unsafe_allow_html=True,
        )
    if company_panels:
        expected_count = metric_frame["公司"].nunique() * len(ordered_periods)
        disclosed_count = metric_frame.drop_duplicates(["公司", "报告期"]).shape[0]
        period_colors = report_period_color_map(ordered_periods)
        legend_items = [
            (period_colors[period], period, "square")
            for period in ordered_periods
        ]
        legend_items.append(("#0C233C", "趋势折线", "line"))
        if highlight_company in metric_frame["公司"].astype(str).unique():
            legend_items.append(("#B8BDC7", "特定追踪公司", "outline"))
        if disclosed_count < expected_count:
            legend_items.append(("#B8BDC7", "未披露", "square"))
        legend_items.extend(unavailable_legend)
        if code in {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}:
            legend_items.append(("#ED2124", "监管下限", "dash"))
        _render_chart_legend(legend_items)
        chart = build_company_bar_trend_chart(
            metric_frame,
            code,
            ordered_periods,
            highlight_company,
        )
        st.altair_chart(
            _chart_without_internal_title(chart),
            width="stretch",
            key=f"{key_prefix}_{code}_company_panels",
        )
    else:
        expected_count = metric_frame["公司"].nunique() * len(ordered_periods)
        disclosed_count = metric_frame.drop_duplicates(["公司", "报告期"]).shape[0]
        external_legend = []
        if disclosed_count < expected_count:
            external_legend.append(("#B8BDC7", "未披露", "square"))
        external_legend.extend(unavailable_legend)
        if code in {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}:
            external_legend.append(("#ED2124", "监管下限", "dash"))
        if external_legend:
            _render_chart_legend(external_legend)
        charts = build_single_metric_trend_charts(
            metric_frame,
            code,
            ordered_periods,
            _company_color_map(frame["公司"], highlight_company),
            highlight_company,
        )
        if len(charts) > 1:
            st.markdown(
                "<div class='solvency-grouped-trend' aria-hidden='true'></div>",
                unsafe_allow_html=True,
            )
        for group_index, chart in enumerate(charts, start=1):
            with st.container(key=f"s7_trend_group_{key_prefix}_{code}_{group_index}"):
                rendered_chart = _chart_without_internal_title(chart)
                if code in {
                    "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
                    "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",
                }:
                    rendered_chart = rendered_chart.properties(height=280)
                st.altair_chart(
                    rendered_chart,
                    width="stretch",
                    key=f"{key_prefix}_{code}_trend_{group_index}",
                )
    if company_period_bars:
        with st.container(key=f"s7_company_bars_page_{key_prefix}_{code}"):
            company_panel_count = metric_frame["公司"].nunique()
            shared_y_domain = metric_bar_axis_domain(metric_frame, code, ordered_periods)
            period_colors = report_period_color_map(ordered_periods)
            st.caption(
                "以下按公司展示各报告期柱状小图；所有公司统一使用同一纵轴范围，"
                "不同报告期使用不同颜色。"
            )
            _render_chart_legend([
                (period_colors[period], period, "square")
                for period in ordered_periods
            ] + unavailable_legend)
            _render_company_chart_grid(
                metric_frame,
                lambda company_frame: build_company_period_bar_chart(
                    company_frame,
                    code,
                    ordered_periods,
                    shared_y_domain=shared_y_domain,
                    dense_layout=company_panel_count >= 10,
                    panel_count=company_panel_count,
                ),
                key_prefix=f"{key_prefix}_{code}_period_bars",
                highlight_company=highlight_company,
            )
def _render_chart_legend(items: Iterable[tuple[str, str, str]]) -> None:
    legend_html = "".join(
        (
            f"<span class='solvency-chart-legend__item'>"
            f"<i class='solvency-chart-legend__symbol solvency-chart-legend__symbol--{shape}' "
            f"style='--legend-color:{color}'></i>{escape(label)}</span>"
        )
        for color, label, shape in items
    )
    st.markdown(
        f"<div class='solvency-chart-legend'>{legend_html}</div>",
        unsafe_allow_html=True,
    )


def _render_company_chart_grid(
    frame: pd.DataFrame,
    chart_factory: Callable[[pd.DataFrame], object],
    *,
    key_prefix: str,
    highlight_company: str = "无",
    panel_width: int | None = None,
    selected_companies: list[str] | None = None,
    panel_renderer: Callable[[str], None] | None = None,
) -> None:
    """Render equal company panels in one page-width row without scrolling."""
    companies = selected_companies if selected_companies is not None else _ordered_companies(frame)
    if not companies:
        st.warning("当前筛选范围没有可绘制的公司数据。")
        return
    company_count = len(companies)
    title_font_size = 9 if company_count >= 12 else 10 if company_count >= 8 else 11 if company_count >= 5 else 13
    column_gap = None if company_count >= 8 else "small"
    with st.container(key=f"annual_company_grid_{key_prefix}"):
        panel_columns = st.columns(
            company_count,
            gap=column_gap,
            vertical_alignment="top",
        )
        tracked_company = str(highlight_company or "").strip()
        for index, (column, company) in enumerate(zip(panel_columns, companies)):
            company_frame = frame[
                frame["公司"].fillna("").astype(str).eq(company)
            ].copy()
            try:
                chart = chart_factory(company_frame)
            except ValueError as exc:
                st.warning(str(exc))
                continue
            tone = (
                "tracked"
                if tracked_company and tracked_company != "无" and company == tracked_company
                else "gray" if index % 2 else "white"
            )
            with column:
                with st.container(
                    border=True,
                    width="stretch",
                    gap=None,
                    key=f"annual_company_panel_{tone}_{key_prefix}_{index}",
                ):
                    st.markdown(
                        "<div class='annual-company-title' "
                        f"style='font-size:{title_font_size}px'>{escape(company)}</div>",
                        unsafe_allow_html=True,
                    )
                    if panel_renderer is not None:
                        panel_renderer(company)
                    else:
                        st.altair_chart(
                            _chart_without_internal_title(chart),
                            width="stretch",
                            key=f"{key_prefix}_{index}",
                        )


def _chart_without_internal_title(chart: object) -> object:
    """Keep company titles outside Vega so Streamlit's chart toolbar cannot cover them."""
    copied = chart.copy(deep=True)
    copied.title = alt.Undefined
    return copied


@st.fragment
def _render_capital_efficiency_bubble_fragment(
    converted: pd.DataFrame,
    periods: list[str],
    company_colors: dict[str, str],
    highlight_company: str,
    key_prefix: str,
) -> None:
    """Keep magnifier interaction entirely inside Vega without Python reruns."""
    bubble_panel_height = 430
    state_prefix = "capital_efficiency_bubble_zoom"
    signature_key = f"{state_prefix}_signature"
    revision_key = f"{state_prefix}_revision"
    available_companies = tuple(
        dict.fromkeys(converted["公司"].dropna().astype(str))
    )
    requested_highlight = str(highlight_company or "").strip()
    tracked_company = (
        requested_highlight
        if requested_highlight in available_companies
        else ""
    )
    try:
        default_domain, latest_period = capital_efficiency_bubble_default_domain(
            converted,
            periods,
            company_colors,
            tracked_company,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return
    sample_signature = (
        latest_period,
        available_companies,
        tracked_company,
    )
    if (
        st.session_state.get(signature_key) != sample_signature
        or revision_key not in st.session_state
    ):
        st.session_state[signature_key] = sample_signature
        st.session_state[revision_key] = (
            int(st.session_state.get(revision_key, 0)) + 1
        )
    st.session_state.setdefault(revision_key, 0)
    # Keep the Vega selection entirely browser-side.  A revision-specific name
    # resets the brush without registering the chart as a Streamlit input.
    selection_name = (
        f"capital_efficiency_zoom_{st.session_state[revision_key]}"
    )
    try:
        full_chart, latest_period = build_capital_efficiency_bubble_chart(
            converted,
            periods,
            company_colors,
            tracked_company,
            panel_label="全样本图",
            chart_width=BUBBLE_FULL_PANEL_WIDTH,
            chart_height=bubble_panel_height,
            show_company_legend=True,
            apply_theme=False,
            linked_selection_name=selection_name,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        key=f"capital_bubble_print_controls_{key_prefix}",
    ):
        if st.button(
            "定位追踪公司" if tracked_company else "恢复默认范围",
            icon=":material/my_location:",
            key=f"{state_prefix}_reset",
        ):
            st.session_state[revision_key] += 1
            st.rerun(scope="fragment")
        st.caption(
            "拖动主图灰色选区可移动观察范围，滚轮可缩放；"
            "当前局部视图会直接用于打印，打印时自动隐藏选区框。"
        )

    overlap_chart = None
    overlap_error = ""
    try:
        overlap_chart, _ = build_capital_efficiency_bubble_chart(
            converted,
            periods,
            company_colors,
            tracked_company,
            zoom_to_overlap_region=True,
            panel_label="局部样本放大",
            chart_width=BUBBLE_ZOOM_PANEL_WIDTH,
            chart_height=bubble_panel_height,
            show_company_legend=True,
            show_axis_titles=False,
            apply_theme=False,
            linked_selection_name=selection_name,
        )
        overlap_chart.to_dict(validate=True)
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        overlap_error = str(exc)
    st.caption(
        f"展示 {latest_period}：横轴为实际资本/注册资本，纵轴为核心资本/注册资本，"
        "气泡面积代表认可资产规模；浅灰虚线为全样本横纵指标中位数。"
    )
    if overlap_chart is not None:
        bubble_pair = combine_capital_efficiency_bubble_charts(
            full_chart,
            overlap_chart,
            selection_name=selection_name,
            selection_domain=default_domain,
        )
        st.altair_chart(
            bubble_pair,
            width="content",
        )
    else:
        st.altair_chart(
            full_chart,
            width="content",
            key=f"{key_prefix}_capital_efficiency_bubble_full_v8",
        )
    if overlap_error:
        st.caption(f"局部放大图暂不可用：{overlap_error}")


@st.fragment
def _render_risk_ratio_scatter_fragment(
    converted: pd.DataFrame,
    periods: list[str],
    company_colors: dict[str, str],
    highlight_company: str,
    key_prefix: str,
    chart_name: str,
    x_code: str,
    y_code: str,
    bubble_size_code: str,
    denominator_label: str,
) -> None:
    """Render linked full/local risk bubbles without Python selection reruns."""
    panel_height = 430
    state_prefix = f"risk_ratio_scatter_zoom_{x_code.lower()}_{y_code.lower()}"
    signature_key = f"{state_prefix}_signature"
    revision_key = f"{state_prefix}_revision"
    available_companies = tuple(
        dict.fromkeys(converted["公司"].dropna().astype(str))
    )
    requested_highlight = str(highlight_company or "").strip()
    tracked_company = (
        requested_highlight
        if requested_highlight in available_companies
        else ""
    )
    sample_signature = (
        "adaptive_risk_bubble_axes_v2",
        chart_name,
        tuple(periods),
        available_companies,
        tracked_company,
    )
    if (
        st.session_state.get(signature_key) != sample_signature
        or revision_key not in st.session_state
    ):
        st.session_state[signature_key] = sample_signature
        st.session_state[revision_key] = (
            int(st.session_state.get(revision_key, 0)) + 1
        )
    st.session_state.setdefault(revision_key, 0)
    selection_name = (
        f"risk_ratio_zoom_{x_code.lower()}_{st.session_state[revision_key]}"
    )
    try:
        default_domain, latest_period = risk_ratio_scatter_default_domain(
            converted,
            x_code,
            y_code,
            periods,
            chart_name,
            tracked_company,
            company_colors,
            bubble_size_code=bubble_size_code,
            bubble_size_label=denominator_label,
        )
        full_chart, _ = build_matrix_chart(
            converted,
            x_code,
            y_code,
            periods,
            chart_name,
            tracked_company,
            company_colors=company_colors,
            percentage_axes=True,
            panel_label="全样本图",
            chart_width=RISK_SCATTER_FULL_PANEL_WIDTH,
            chart_height=panel_height,
            linked_selection_name=selection_name,
            apply_theme=False,
            bubble_size_code=bubble_size_code,
            bubble_size_label=denominator_label,
        )
    except ValueError as exc:
        st.warning(str(exc))
        return

    with st.container(
        horizontal=True,
        vertical_alignment="center",
        key=f"risk_scatter_print_controls_{x_code.lower()}",
    ):
        if st.button(
            "定位追踪公司" if tracked_company else "恢复默认范围",
            icon=":material/my_location:",
            key=f"{state_prefix}_reset",
        ):
            st.session_state[revision_key] += 1
            st.rerun(scope="fragment")
        st.caption(
            "拖动全样本图灰色选区可移动观察范围，滚轮可缩放；"
            "当前局部视图会直接用于打印，打印时自动隐藏选区框。"
        )

    local_chart = None
    local_error = ""
    try:
        local_chart, _ = build_matrix_chart(
            converted,
            x_code,
            y_code,
            periods,
            chart_name,
            tracked_company,
            company_colors=company_colors,
            percentage_axes=True,
            zoom_to_overlap_region=True,
            panel_label="局部样本放大",
            chart_width=RISK_SCATTER_ZOOM_PANEL_WIDTH,
            chart_height=panel_height,
            show_axis_titles=False,
            linked_selection_name=selection_name,
            apply_theme=False,
            bubble_size_code=bubble_size_code,
            bubble_size_label=denominator_label,
            show_size_legend=False,
        )
        local_chart.to_dict(validate=True)
    except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
        local_error = str(exc)

    st.caption(
        f"展示 {latest_period}；每个气泡代表一家公司，横纵轴均为占{denominator_label}的比例，"
        f"气泡面积代表{denominator_label}规模；"
        "横纵轴根据各指标量级自动采用%、‰或基点（bp），具体单位见坐标轴标题；"
        "浅灰虚线为全样本横纵指标中位数。"
    )
    if local_chart is not None:
        linked_chart = combine_linked_matrix_charts(
            full_chart,
            local_chart,
            selection_name=selection_name,
            selection_domain=default_domain,
        )
        st.altair_chart(linked_chart, width="content")
    else:
        st.altair_chart(
            full_chart,
            width="content",
            key=f"{key_prefix}_{x_code}_risk_scatter_full_v2",
        )
    if local_error:
        st.caption(f"局部放大图暂不可用：{local_error}")


def _missing_disclosure_note_lines(
    chart_name: str,
    companies: list[str],
    missing_by_company: dict[str, list[str]],
) -> list[str]:
    """Collapse repeated disclosure gaps into concise company-group notes."""
    if chart_name == "综合退保率":
        grouped_periods: dict[tuple[str, ...], list[str]] = {}
        for company in companies:
            missing_periods = missing_by_company.get(company)
            if missing_periods:
                key = tuple(sort_report_periods(dict.fromkeys(missing_periods)))
                grouped_periods.setdefault(key, []).append(company)
        return [
            f"注：未披露——{'、'.join(names)}：{'、'.join(periods)}均未披露综合退保率。"
            for periods, names in grouped_periods.items()
        ]

    if chart_name == "投资质量六指标雷达图":
        grouped_metrics: dict[tuple[str, ...], list[str]] = {}
        for company in companies:
            metrics = missing_by_company.get(company)
            if metrics:
                grouped_metrics.setdefault(tuple(dict.fromkeys(metrics)), []).append(company)
        notes: list[str] = []
        for metrics, names in grouped_metrics.items():
            detail = (
                "六项投资质量指标均未披露"
                if len(metrics) == 6
                else f"未披露{'、'.join(metrics)}"
            )
            notes.append(f"注：未披露——{'、'.join(names)}：{detail}。")
        return notes

    if chart_name == "近三年平均投资收益率与综合投资收益率":
        affected = [company for company in companies if missing_by_company.get(company)]
        if not affected:
            return []
        return [
            f"注：未披露——{'、'.join(affected)}部分报告期的近三年平均投资收益率或"
            "综合投资收益率未披露。"
        ]

    period_metric_charts = {
        "保险合同负债/总负债",
        "保险业务收入/签单保费",
        "签单保费与新业务利润率",
        "新业务价值与新业务价值率",
        "累计投资收益率与累计综合投资收益率",
    }
    if chart_name in period_metric_charts:
        grouped: dict[tuple[str, ...], list[str]] = {}
        for company in companies:
            details = missing_by_company.get(company)
            if details:
                grouped.setdefault(tuple(details), []).append(company)

        notes: list[str] = []
        for details, names in grouped.items():
            by_axis: dict[str, list[str]] = {}
            for detail in details:
                axis, separator, metric = str(detail).partition("：")
                if not separator:
                    axis, metric = "所选期间", str(detail)
                metric = metric.replace(
                    "签单保费（无法计算新业务价值率）",
                    "签单保费（新业务价值率因此无法计算）",
                )
                metric = metric.replace(
                    "续期签单保费（无法计算新业务价值率）",
                    "续期签单保费（新业务价值率因此无法计算）",
                )
                if metric not in by_axis.setdefault(axis, []):
                    by_axis[axis].append(metric)

            axes_by_metrics: dict[tuple[str, ...], list[str]] = {}
            for axis, metrics in by_axis.items():
                axes_by_metrics.setdefault(tuple(metrics), []).append(axis)
            phrases: list[str] = []
            for metrics, axes in axes_by_metrics.items():
                split_axes = [axis.split("·", 1) for axis in axes]
                same_scope = all(len(parts) == 2 for parts in split_axes) and len({parts[1] for parts in split_axes}) == 1
                axis_labels = [parts[0] for parts in split_axes] if same_scope else axes
                phrases.append(f"{'、'.join(axis_labels)}均未披露{'及'.join(metrics)}")
            notes.append(
                f"注：未披露——{'、'.join(names)}：{'；'.join(phrases)}。"
            )
        return notes

    if chart_name != "附属一级资本明细":
        return [
            f"注：未披露——{company}：" + "；".join(missing_by_company[company])
            for company in companies if company in missing_by_company
        ]

    grouped: dict[tuple[str, ...], list[str]] = {}
    fallback: list[str] = []
    for company in companies:
        details = missing_by_company.get(company)
        if not details:
            continue
        periods: list[str] = []
        for detail in details:
            period, separator, label = str(detail).partition("：")
            if not separator or label != "附属一级资本明细":
                fallback.append(f"注：未披露——{company}：" + "；".join(details))
                periods = []
                break
            periods.append(period)
        if periods:
            key = tuple(sort_report_periods(dict.fromkeys(periods)))
            grouped.setdefault(key, []).append(company)

    notes = [
        f"注：未披露——{'、'.join(names)}：{'、'.join(periods)}均未披露附属一级资本明细。"
        for periods, names in grouped.items()
    ]
    return [*notes, *fallback]


def _render_quality_and_capital(
    frame: pd.DataFrame,
    chart_name: str,
    *,
    periods: list[str],
    unit_mode: str,
    highlight_company: str,
    key_prefix: str,
    selected_companies: list[str] | None = None,
) -> None:
    """Render disclosed values for every selected company, including empty panels."""
    converted = convert_multiple_units_to_percent(_convert_unit(frame, unit_mode))
    if chart_name in {"核心一级资本明细", "附属一级资本明细"}:
        converted = complete_external_capital_detail_zeros(converted)
    companies = selected_companies if selected_companies is not None else _ordered_companies(converted)
    periods = sort_report_periods(periods)
    if not periods:
        st.info("当前没有可展示的报告期。")
        return
    latest = periods[-1]
    missing_by_company: dict[str, list[str]] = {}
    invalid_by_company: dict[str, list[str]] = {}
    figures: dict[str, list[tuple[str, object | None]]] = {}
    amount_labels = {
        "SIGNED_PREMIUM": "签单保费",
        "NEW_BUSINESS_VALUE": "新业务价值",
        "INVESTMENT_RETURN": "投资收益率",
    }
    line_labels = {
        "NEW_BUSINESS_MARGIN": "新业务利润率",
        "COMPREHENSIVE_INVESTMENT_RETURN": "综合投资收益率",
    }
    combo_specs = {
        "签单保费与新业务利润率": ("SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN", False, None),
        "新业务价值与新业务价值率": ("NEW_BUSINESS_VALUE", "SIGNED_PREMIUM", True, None),
        "累计投资收益率与累计综合投资收益率": ("INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN", False, "cumulative"),
        "近三年平均投资收益率与综合投资收益率": ("INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN", False, "three_year"),
    }
    ratio_bar_specs = {
        "保险合同负债/总负债": "INSURANCE_CONTRACT_LIABILITY_TO_TOTAL_LIABILITIES",
        "保险业务收入/签单保费": "INSURANCE_REVENUE_TO_SIGNED_PREMIUM",
    }
    period_colored_bar_charts = {
        *ratio_bar_specs,
        "签单保费与新业务利润率",
        "新业务价值与新业务价值率",
    }
    fixed_quarter_charts = {
        "签单保费与新业务利润率",
        "新业务价值与新业务价值率",
        "综合退保率",
    }
    global_axes = None
    if chart_name in combo_specs:
        bar_code, line_code, derived, scope = combo_specs[chart_name]
        if chart_name in fixed_quarter_charts:
            global_axes = [(period, "quarter") for period in periods]
        else:
            global_axes = (
                [(period, scope) for period in periods]
                if scope else period_scopes(
                    converted,
                    (bar_code, line_code, "RENEWAL_PREMIUM") if derived else (bar_code, line_code),
                    periods,
                )
            )
        if not scope and chart_name not in fixed_quarter_charts:
            kinds = list(dict.fromkeys(kind for _, kind in global_axes))
            selected_scope = st.selectbox("期间口径", kinds, format_func=SCOPE_LABELS.get,
                                          key=f"{key_prefix}_scope") if len(kinds) > 1 else kinds[0]
            global_axes = [(period, selected_scope) for period in periods]
        ratio_label = "新业务价值率" if derived else line_labels.get(line_code, "")
        if scope in {"cumulative", "three_year"} or chart_name in period_colored_bar_charts:
            period_colors = report_period_combo_bar_color_map(periods)
            bar_legend = [(period_colors[period], period, "square") for period in periods]
            if scope in {"cumulative", "three_year"}:
                st.caption("柱形展示投资收益率，粉色折线展示综合投资收益率；柱形颜色对应不同报告期。")
            else:
                st.caption("柱形颜色对应不同报告期；粉色折线展示对应比率。")
        else:
            bar_legend = [("#1E49E2", f"{amount_labels[bar_code]}（{unit_mode}，柱形）", "square")]
        _render_chart_legend([
            *bar_legend,
            ("#FD349C", f"{ratio_label}（%，折线）", "line"),
        ])
        if derived:
            st.caption("新业务价值率＝同公司、同报告期当季的新业务价值 ÷（签单保费－续期签单保费）× 100%。")
    elif chart_name in ratio_bar_specs:
        period_colors = report_period_combo_bar_color_map(periods)
        _render_chart_legend([
            (period_colors[period], period, "square")
            for period in periods
        ])
        formula = (
            "保险合同负债 ÷（总资产－净资产）× 100%"
            if chart_name == "保险合同负债/总负债"
            else "保险业务收入 ÷ 签单保费 × 100%"
        )
        st.caption(
            f"各公司按季度展示，柱形颜色对应不同报告期；计算公式：{formula}。"
            "缺失期间留空，不以 0 替代。"
        )
    elif chart_name == "综合退保率":
        selected_scope = "quarter"
        colors = _company_color_map(companies, highlight_company)
        missing: dict[str, list[str]] = {}
        trend_rows: list[dict[str, object]] = []
        for company in companies:
            company_frame = converted.loc[converted["公司"].fillna("").astype(str).eq(company)]
            for period in periods:
                value = operating_quarter_value(company_frame, "SURRENDER_RATE", period)
                if value is None:
                    missing.setdefault(company, []).append(period)
                trend_rows.append({
                    "公司": company,
                    "报告期": period,
                    "指标编码": "SURRENDER_RATE",
                    "指标名称": "综合退保率",
                    "数值": value,
                    "单位": "%",
                    "披露状态": "已披露" if value is not None else "未披露",
                })
        fully_missing = [
            company for company in companies
            if len(missing.get(company, [])) == len(periods)
        ]
        partly_missing = [
            company for company in companies
            if company in missing and company not in fully_missing
        ]
        disclosure_legend = []
        if fully_missing:
            disclosure_legend.append(("#B8BDC7", "未披露：" + "、".join(fully_missing), "square"))
        if partly_missing:
            disclosure_legend.append(("#B8BDC7", "部分未披露：" + "、".join(partly_missing), "square"))
        if disclosure_legend:
            _render_chart_legend(disclosure_legend)
        st.caption(f"综合退保率（%）；期间口径：{SCOPE_LABELS[selected_scope]}。每条折线代表一家公司，缺失期间留空。")
        trend_frame = pd.DataFrame(trend_rows)
        if not trend_frame.empty and trend_frame["数值"].notna().any():
            chart = build_single_metric_trend_chart(
                trend_frame,
                "SURRENDER_RATE",
                periods,
                colors,
                highlight_company,
            )
            st.altair_chart(
                _chart_without_internal_title(chart),
                width="stretch",
                key=f"{key_prefix}_surrender_all",
            )
        else:
            st.info("未披露可绘图数据")
        for note in _missing_disclosure_note_lines(chart_name, companies, missing):
            st.caption(note)
        return
    elif chart_name == "核心一级资本明细":
        st.caption(f"仅展示最新报告期 {latest}；调整项保留原表正负号，金额单位：{unit_mode}。")
        _render_chart_legend([("#1E49E2", "净资产 / 核心一级资本", "square"), ("#00C0AE", "增加项", "square"), ("#FD349C", "减少项", "square")])
        st.caption("瀑布图横轴：0 净资产；1 非认可资产；2 长期股权投资及投资性房地产估值差额；3 递延所得税资产；4 保单未来盈余；5 其他调整；6 核心一级资本。")
    elif chart_name == "附属一级资本明细":
        st.caption(f"饼图展示最新报告期 {latest}。金额单位：{unit_mode}。")
        _render_chart_legend([(color, label, "square") for color, (_, label) in zip(COMPONENT_COLORS, ANC_COMPONENTS)])
        st.caption("饼图占比以七类可用明细之和为分母；七类均为零或无值时按明细未披露处理；存在负值时不绘制饼图。")
    elif chart_name == "投资质量六指标雷达图":
        st.caption(
            "雷达图轴：1 净资产收益率；2 总资产收益率；3 投资收益率（当季数）；"
            "4 综合投资收益率（当季数）；5 近三年平均投资收益率；6 近三年平均综合投资收益率。"
        )

    for company in companies:
        company_frame = converted.loc[converted["公司"].fillna("").astype(str).eq(company)]
        missing: list[str] = []
        invalid: list[str] = []
        charts: list[tuple[str, object | None]] = []
        if chart_name in ratio_bar_specs:
            fig, missing, invalid = ratio_bar_figure(
                company_frame, periods, ratio_bar_specs[chart_name]
            )
            charts.append(("", fig))
        elif chart_name in combo_specs:
            fig, missing, invalid = combo_figure(
                company_frame, periods, bar_code, line_code,
                derived_rate=derived, scope=scope, axes=global_axes,
            )
            charts.append(("", fig if global_axes else None))
            if not global_axes:
                missing = ["所选期间全部指标"]
        elif chart_name == "投资质量六指标雷达图":
            fig, missing = radar_figure(company_frame, latest)
            charts.append(("", fig if len(missing) < 6 else None))
        elif chart_name == "核心一级资本明细":
            fig, missing, difference = core_waterfall_figure(company_frame, latest)
            charts.append(("", fig))
            omitted = core_omitted_nonzero(company_frame, latest)
            if omitted:
                invalid.append("图中省略的非零项目：" + "、".join(
                    f"{label} {value:,.2f} {unit_mode}" for label, value in omitted
                ))
            if difference is not None and abs(difference) > max(abs(float(fig.data[0].y[0])) * 1e-6, 0.01):
                invalid.append(f"披露终值与已披露调整项之和相差 {difference:,.2f} {unit_mode}")
        elif chart_name == "附属一级资本明细":
            pie, pie_missing, negative = anc_pie_figure(company_frame, latest)
            charts.append(("最新季度构成饼图", pie))
            missing = [f"{latest}：{item}" for item in pie_missing]
            if negative:
                invalid.append("含负值，饼图不绘制：" + "、".join(negative))
            total = capital_value(company_frame, "ANC_T1_CAPITAL", latest)
            values = [capital_value(company_frame, code, latest) for code, _ in ANC_COMPONENTS]
            if total is not None and not pie_missing:
                delta = total - sum(values)
                if abs(delta) > max(abs(total) * 1e-6, 0.01):
                    invalid.append(f"七类明细合计与披露的附属一级资本相差 {delta:,.2f} {unit_mode}")
        figures[company] = [(subtitle, fig if has_values(fig) else None) for subtitle, fig in charts]
        if missing:
            missing_by_company[company] = missing
        if invalid:
            invalid_by_company[company] = invalid

    if missing_by_company and chart_name != "近三年平均投资收益率与综合投资收益率":
        _render_chart_legend([(GREY, "未披露：" + "、".join(missing_by_company), "square")])

    # Use a common radar scale so panels can be compared, including negatives.
    if chart_name == "投资质量六指标雷达图":
        radar_points = [value for charts in figures.values() for _, fig in charts if fig is not None
                        for trace in fig.data for value in trace.r if value is not None]
        low, high = min([0, *radar_points]), max([0, *radar_points])
        padding = (high - low) * 0.1 or 1
        for charts in figures.values():
            for _, fig in charts:
                if fig is not None:
                    fig.update_layout(polar=dict(radialaxis=dict(range=[low - padding if low < 0 else 0, high + padding])))
    elif chart_name in ratio_bar_specs:
        chart_figures = [fig for charts in figures.values() for _, fig in charts if fig is not None]
        values = [value for fig in chart_figures for value in fig.data[0].y if value is not None]
        if values:
            low, high = min([0, *values]), max([0, *values])
            padding = (high - low) * 0.1 or 1
            common_range = [low - padding if low < 0 else 0, high + padding]
            for fig in chart_figures:
                fig.update_yaxes(range=common_range)
    elif chart_name in combo_specs:
        chart_figures = [fig for charts in figures.values() for _, fig in charts if fig is not None]
        def axis_range(values):
            low, high = min([0, *values]), max([0, *values])
            pad = (high - low) * 0.1 or 1
            return [low - pad if low < 0 else 0, high + pad]
        first_values = [v for fig in chart_figures for v in fig.data[0].y if v is not None]
        second_values = [v for fig in chart_figures for trace in fig.data[1:] for v in trace.y if v is not None]
        same_unit = chart_name in combo_specs and scope in {"cumulative", "three_year"}
        left_range = axis_range(first_values + second_values if same_unit else first_values)
        right_range = left_range if same_unit else axis_range(second_values)
        for fig in chart_figures:
            fig.update_yaxes(range=left_range, secondary_y=False)
            fig.update_yaxes(range=right_range, secondary_y=True)
    elif chart_name in {"核心一级资本明细", "附属一级资本明细"}:
        bounds = [0.0]
        amount_figures = [fig for charts in figures.values() for _, fig in charts
                          if fig is not None and fig.data[0].type != "pie"]
        for fig in amount_figures:
            if fig.data[0].type == "waterfall":
                running = 0.0
                for measure, value in zip(fig.data[0].measure, fig.data[0].y):
                    if value is not None:
                        running = value if measure == "absolute" else running + value
                        bounds.append(running)
            else:
                for index in range(len(fig.data[0].x)):
                    values = [trace.y[index] for trace in fig.data if trace.y[index] is not None]
                    bounds.extend([sum(max(0, v) for v in values), sum(min(0, v) for v in values)])
        low, high = min(bounds), max(bounds)
        padding = (high - low) * 0.1 or 1
        for fig in amount_figures:
            fig.update_yaxes(range=[low - padding if low < 0 else 0, high + padding])
    def render_panel(company: str, chart_index: int) -> None:
        _, fig = figures[company][chart_index]
        if fig is None:
            message = "未披露可绘图数据"
            if chart_name == "附属一级资本明细" and chart_index == 0 and company in invalid_by_company:
                message = "本期不绘制饼图\n原因见下方注释"
            chart = empty_company_chart(message, len(companies))
        else:
            chart = company_quality_chart(
                fig, company, len(companies), unit=unit_mode,
                percentage_bar=chart_name in ratio_bar_specs or (
                    chart_name in combo_specs and scope in {"cumulative", "three_year"}
                ),
                color_by_period=(
                    chart_name in period_colored_bar_charts
                    or (chart_name in combo_specs and scope in {"cumulative", "three_year"})
                ),
            )
        st.altair_chart(chart, width="stretch", key=f"{key_prefix}_{chart_index}_{company}")

    group_titles = (
        [f"最新季度构成饼图（{latest}）"]
        if chart_name == "附属一级资本明细" else [""]
    )
    for chart_index, group_title in enumerate(group_titles):
        with st.container(key=f"s7_trend_group_{key_prefix}_{chart_index}"):
            if group_title:
                st.markdown(f"**{group_title}**")
            _render_company_chart_grid(
                converted, lambda _: None, key_prefix=f"{key_prefix}_quality_{chart_index}",
                highlight_company=highlight_company, selected_companies=companies,
                panel_renderer=lambda company, index=chart_index: render_panel(company, index),
            )
    # Long notes sit below the row, so missing data cannot stretch one panel.
    for note in _missing_disclosure_note_lines(chart_name, companies, missing_by_company):
        st.caption(note)
    for company in companies:
        if company in invalid_by_company:
            st.caption(f"注：{company}：" + "；".join(invalid_by_company[company]))


def _chart_input_metric_codes(chart_name: str, plan_kind: str) -> tuple[str, ...]:
    """Keep calculation inputs that are not navigation display metrics."""
    codes = list(metric_codes_for_chart(chart_name))
    if plan_kind == COMPONENT_STACK:
        denominator_code = COMPONENT_STACK_DENOMINATOR_CODES.get(chart_name, "")
        if denominator_code:
            codes.append(denominator_code)
        if chart_name == "量化风险最低资本构成":
            codes.extend(QUANT_RISK_STACK_RATIO_CODES.values())
    if plan_kind == CAPITAL_RATIO_COMBO and chart_name == "核心充足率变化":
        codes.append("ACTUAL_CAPITAL")
    return tuple(dict.fromkeys(codes))


def _all_zero_component_companies(
    frame: pd.DataFrame,
    component_specs: Iterable[tuple[str, str, str]],
    periods: Iterable[str],
) -> list[str]:
    """Return companies whose complete disclosed component set is entirely zero."""
    codes = tuple(str(code) for code, _, _ in component_specs)
    period_set = {str(period) for period in periods}
    rows = frame.loc[
        frame["指标编码"].astype(str).isin(codes)
        & frame["报告期"].astype(str).isin(period_set)
    ].copy()
    if rows.empty:
        return []
    if "来源类型" in rows:
        rows = rows.loc[
            ~rows["来源类型"].fillna("").astype(str).eq("宽表空白推定")
        ]
    if "披露状态" in rows:
        rows = rows.loc[
            ~rows["披露状态"].fillna("").astype(str).isin(
                {"未披露", "不适用", "无法计算", "not_disclosed", "disclosed_na"}
            )
        ]
    rows["_零值检查"] = pd.to_numeric(rows["数值"], errors="coerce")
    rows = rows.dropna(subset=["_零值检查"])
    if rows.empty:
        return []

    zero_companies: list[str] = []
    for company in _ordered_companies(rows):
        company_rows = rows.loc[rows["公司"].fillna("").astype(str).eq(company)]
        if (
            set(company_rows["指标编码"].astype(str)) == set(codes)
            and company_rows["_零值检查"].abs().le(1e-12).all()
        ):
            zero_companies.append(company)
    return zero_companies


def _component_stack_disclosure_gaps(
    frame: pd.DataFrame,
    component_specs: Iterable[tuple[str, str, str]],
    periods: Iterable[str],
    denominator_code: str,
) -> tuple[list[str], list[str]]:
    """Classify unrenderable and partially disclosed composition companies."""
    companies = _ordered_companies(frame)
    component_codes = {str(code) for code, _, _ in component_specs}
    period_order = tuple(dict.fromkeys(str(period) for period in periods))
    period_set = set(period_order)
    if not companies or not component_codes or not period_set:
        return [], []

    target_codes = component_codes | {str(denominator_code)}
    rows = frame.loc[
        frame["指标编码"].fillna("").astype(str).isin(target_codes)
        & frame["报告期"].fillna("").astype(str).isin(period_set)
    ].copy()
    rows["_披露数值"] = pd.to_numeric(rows["数值"], errors="coerce")
    disclosed = rows["_披露数值"].notna()
    if "来源类型" in rows:
        disclosed &= ~rows["来源类型"].fillna("").astype(str).eq("宽表空白推定")
    if "披露状态" in rows:
        disclosed &= ~rows["披露状态"].fillna("").astype(str).isin(
            {"未披露", "不适用", "无法计算", "not_disclosed", "disclosed_na"}
        )
    rows = rows.loc[disclosed].drop_duplicates(
        ["公司", "报告期", "指标编码"], keep="last"
    )

    expected_component_pairs = {
        (period, code) for period in period_set for code in component_codes
    }
    unavailable: list[str] = []
    partial: list[str] = []
    for company in companies:
        company_rows = rows.loc[
            rows["公司"].fillna("").astype(str).eq(company)
        ]
        component_rows = company_rows.loc[
            company_rows["指标编码"].astype(str).isin(component_codes)
        ]
        component_pairs = set(zip(
            component_rows["报告期"].astype(str),
            component_rows["指标编码"].astype(str),
        ))
        component_periods = {period for period, _ in component_pairs}
        denominator_rows = company_rows.loc[
            company_rows["指标编码"].astype(str).eq(str(denominator_code))
            & company_rows["_披露数值"].ne(0)
        ]
        denominator_periods = set(denominator_rows["报告期"].astype(str))
        renderable_periods = component_periods & denominator_periods
        if not renderable_periods:
            unavailable.append(company)
        elif (
            component_pairs != expected_component_pairs
            or denominator_periods != period_set
        ):
            partial.append(company)
    return unavailable, partial


def _render_combination_analysis(
    frame: pd.DataFrame,
    chart_name: str,
    *,
    periods: list[str],
    unit_mode: str,
    highlight_company: str,
    key_prefix: str,
    financing_data: pd.DataFrame | None = None,
) -> None:
    """Render an approved multi-metric plan selected by chart name."""
    plan = chart_plan_for(chart_name)
    if plan.kind == KEY_METRICS_TABLE:
        render_key_solvency_overview(frame, highlight_company=highlight_company)
        return
    if plan.kind == FINANCING_TABLE:
        render_major_financing(financing_data)
        return
    if plan.kind == PENDING_DEFINITION:
        st.warning(
            f"“{chart_name}”在新版 Excel 中标记为“待定”，目前没有定义可绘图的明细组成指标。"
        )
        return
    if plan.kind == QUALITY_AND_CAPITAL:
        _render_quality_and_capital(
            frame, chart_name, periods=periods, unit_mode=unit_mode,
            highlight_company=highlight_company, key_prefix=key_prefix,
        )
        return
    required_codes = _chart_input_metric_codes(chart_name, plan.kind)
    chart_frame = (
        frame[frame["指标编码"].fillna("").astype(str).isin(required_codes)].copy()
        if required_codes
        else frame
    )
    converted = _convert_unit(chart_frame, unit_mode)
    company_colors = _company_color_map(frame["公司"], highlight_company)
    company_panel_count = converted["公司"].nunique()
    dense_company_panels = company_panel_count >= 10
    if plan.kind == SOLVENCY_RATIO_COMBO:
        shared_y_domain = solvency_ratio_axis_domain(converted, periods)
        period_colors = report_period_combo_bar_color_map(periods)
        st.caption(
            "图表说明：柱状图为综合偿付能力充足率；粉色折线为核心偿付能力充足率。"
            "各公司统一使用同一百分比纵轴范围，柱形颜色对应不同报告期。"
        )
        _render_chart_legend([
            *[
                (period_colors[period], period, "square")
                for period in periods
            ],
            ("#FD349C", "核心偿付能力充足率（%）", "line"),
        ])
        try:
            overview_chart = build_solvency_ratio_combo_overview_chart(
                converted,
                periods,
                company_order=_ordered_companies(converted),
                shared_y_domain=shared_y_domain,
                highlight_company=highlight_company,
            )
        except ValueError as exc:
            st.warning(str(exc))
        else:
            with st.container(
                key=f"annual_company_grid_{key_prefix}_solvency_ratio_combo"
            ):
                st.altair_chart(
                    overview_chart,
                    width="stretch",
                    key=f"{key_prefix}_solvency_ratio_combo_overview",
                )
    elif plan.kind == COMPANY_PERIOD_BAR:
        codes = metric_codes_for_chart(chart_name)
        if codes:
            shared_y_domain = metric_bar_axis_domain(
                converted,
                codes[0],
                periods,
            )
            period_colors = report_period_color_map(periods)
            _render_chart_legend([
                (period_colors[period], period, "square")
                for period in periods
            ])
            _render_company_chart_grid(
                converted,
                lambda company_frame: build_company_period_bar_chart(
                    company_frame,
                    codes[0],
                    periods,
                    shared_y_domain=shared_y_domain,
                    dense_layout=dense_company_panels,
                    panel_count=company_panel_count,
                ),
                key_prefix=f"{key_prefix}_{codes[0]}_bar",
                highlight_company=highlight_company,
            )
    elif plan.kind == CAPITAL_RATIO_COMBO:
        if chart_name == "综合充足率变化":
            amount_groups = {
                "实际资本": ("ACTUAL_CAPITAL",),
                "最低资本": ("MINIMUM_CAPITAL",),
            }
            ratio_code = "COMBINED_SOLVENCY_RATIO"
            ratio_label = "综合偿付能力充足率（%）"
        else:
            amount_groups = {
                "核心资本": ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
                "最低资本": ("MINIMUM_CAPITAL",),
            }
            ratio_code = "CORE_SOLVENCY_RATIO"
            ratio_label = "核心偿付能力充足率（%）"
        shared_amount_domain = capital_ratio_amount_axis_domain(
            converted,
            amount_groups,
            periods,
        )
        st.caption(
            "上方 KPMG Pink 折线展示对应充足率，下方堆叠柱展示资本规模；"
            "同组所有公司统一使用同一金额纵轴范围，便于直接比较资本规模。"
        )
        st.markdown("**堆叠柱组合图**")
        _render_chart_legend([
            *[
                (KPMG_CAPITAL_COMBO_COLORS[index], label, "square")
                for index, label in enumerate(amount_groups)
            ],
            ("#FD349C", ratio_label, "line"),
        ])
        _render_company_chart_grid(
            converted,
            lambda company_frame: build_capital_ratio_combo_chart(
                company_frame,
                amount_groups,
                periods,
                chart_name,
                ratio_code=ratio_code,
                ratio_label=ratio_label,
                shared_amount_domain=shared_amount_domain,
                dense_layout=dense_company_panels,
                panel_count=company_panel_count,
            ),
            key_prefix=f"{key_prefix}_{ratio_code}_combo",
            highlight_company=highlight_company,
        )
    elif plan.kind == CAPITAL_STRUCTURE_COMBO:
        capital_structure_groups = {
            "核心资本": ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
            "附属资本": ("ANC_T1_CAPITAL", "ANC_T2_CAPITAL"),
        }
        shared_amount_domain = capital_ratio_amount_axis_domain(
            converted,
            capital_structure_groups,
            periods,
        )
        st.caption(
            "上方 KPMG Pink 折线展示核心资本占实际资本的比例，下方堆叠柱展示核心资本与附属资本；"
            "同组所有公司统一使用同一金额纵轴范围，便于直接比较资本规模。"
        )
        _render_chart_legend([
            (KPMG_CAPITAL_COMBO_COLORS[0], "核心资本", "square"),
            (KPMG_CAPITAL_COMBO_COLORS[1], "附属资本", "square"),
            ("#FD349C", "核心资本占实际资本的比例", "line"),
        ])
        _render_company_chart_grid(
            converted,
            lambda company_frame: build_capital_ratio_combo_chart(
                company_frame,
                capital_structure_groups,
                periods,
                "核心资本与附属资本结构",
                ratio_label="核心资本占实际资本的比例（%）",
                share_line=True,
                shared_amount_domain=shared_amount_domain,
                dense_layout=dense_company_panels,
                panel_count=company_panel_count,
            ),
            key_prefix=f"{key_prefix}_core_capital_structure",
            highlight_company=highlight_company,
        )
    elif plan.kind == COMPONENT_STACK:
        specs = COMPONENT_STACK_SPECS.get(chart_name, ())
        if not specs:
            st.warning(f"“{chart_name}”尚未配置组成指标。")
            return
        if chart_name == "计入各级资本的保单未来盈余构成占比":
            st.caption(
                "各色块表示计入相应资本层级的保单未来盈余占该公司已披露保单未来盈余明细合计的比例；"
                "100%表示该公司披露的保单未来盈余全部计入了同一个资本层级，其他资本层级没有披露这个明细。"
            )
        elif chart_name == "量化风险最低资本构成":
            st.caption(
                "柱高按各构成项目/量化风险最低资本的比例展示，实际金额仅保留在悬浮信息中；"
                "数据标签与对应比例折线图一致，负向抵减项目位于零线下方。"
            )
        elif chart_name == "各类保险风险（寿）占比":
            st.caption(
                "柱高按各构成项目/寿险业务保险风险最低资本的比例展示，实际金额仅保留在悬浮信息中；"
                "负向风险分散效应位于零线下方。"
            )
        elif chart_name == "各类市场风险占比":
            st.caption(
                "柱高按各构成项目/市场风险最低资本的比例展示，实际金额仅保留在悬浮信息中；"
                "负向风险分散效应位于零线下方。"
            )
        elif chart_name == "各类信用风险占比":
            st.caption(
                "柱高按各构成项目/信用风险最低资本的比例展示，实际金额仅保留在悬浮信息中；"
                "负向风险分散效应位于零线下方。"
            )
        elif chart_name == "认可资产构成":
            st.caption(
                "柱高按各资产构成项目/认可资产合计的比例展示，实际金额仅保留在悬浮信息中；"
                "正数项目自零线向上堆叠，负数项目自零线向下堆叠，保留原始正负号；"
                "微小负占比保留真实柱高，并用零线下方的标签和引线标明；"
                "展示已登记指标编码的现金及流动性管理工具、投资资产、在子公司合营企业和联营企业中的权益、"
                "再保险资产、应收及预付款项、固定资产、土地使用权、独立账户资产、其他认可资产九类资产。"
                "缺失或未披露项目保持为空，不以0替代；灰色图例列示未披露或部分未披露的公司。"
            )
        elif chart_name == "认可负债构成":
            st.caption(
                "柱高按各负债构成项目/认可负债合计的比例展示，实际金额仅保留在悬浮信息中；"
                "正数项目自零线向上堆叠，负数项目自零线向下堆叠，并保留原始正负号；"
                "展示准备金负债、金融负债、应付及预收款项、预计负债、独立账户负债、"
                "资本性负债、其它认可负债七类负债。缺失或未披露项目保持为空，不以0替代；"
                "灰色图例列示未披露或部分未披露的公司。"
            )
        else:
            st.caption("按公司分面展示构成；负向抵减项目位于零线下方，所有颜色均来自 KPMG 色卡。")
        negative_component_codes = (
            (
                "QUANT_RISK_DIVERSIFICATION_EFFECT",
                "CONTRACT_LOSS_ABSORPTION_EFFECT",
            )
            if chart_name == "量化风险最低资本构成"
            else ("LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT",)
            if chart_name == "各类保险风险（寿）占比"
            else ("MARKET_RISK_DIVERSIFICATION_EFFECT",)
            if chart_name == "各类市场风险占比"
            else ("CREDIT_RISK_DIVERSIFICATION_EFFECT",)
            if chart_name == "各类信用风险占比"
            else ()
        )
        proportion_stack = chart_name in COMPONENT_STACK_DENOMINATOR_CODES
        all_zero_companies = (
            _all_zero_component_companies(converted, specs, periods)
            if chart_name == "各类保险风险（寿）占比"
            else []
        )
        disclosure_stack = chart_name in {"认可资产构成", "认可负债构成"}
        unavailable_companies, partial_companies = (
            _component_stack_disclosure_gaps(
                converted,
                specs,
                periods,
                COMPONENT_STACK_DENOMINATOR_CODES[chart_name],
            )
            if disclosure_stack
            else ([], [])
        )
        shared_stack_domain = (
            component_stack_proportion_axis_domain(
                converted,
                specs,
                periods,
                label_ratio_codes=(
                    QUANT_RISK_STACK_RATIO_CODES
                    if chart_name == "量化风险最低资本构成"
                    else None
                ),
                label_denominator_code=COMPONENT_STACK_DENOMINATOR_CODES[chart_name],
                negative_component_codes=negative_component_codes,
                show_small_negative_labels=disclosure_stack,
            )
            if proportion_stack
            else component_stack_axis_domain(
                converted,
                specs,
                periods,
                negative_component_codes=negative_component_codes,
            )
        )
        stack_legend = [
            (color, label, "square")
            for _, label, color in specs
        ]
        if unavailable_companies:
            stack_legend.append((
                "#B8BDC7", f"未披露：{'、'.join(unavailable_companies)}", "square"
            ))
        if partial_companies:
            stack_legend.append((
                "#B8BDC7", f"部分未披露：{'、'.join(partial_companies)}", "square"
            ))
        _render_chart_legend(stack_legend)
        _render_company_chart_grid(
            converted,
            lambda company_frame: (
                empty_company_chart(
                    "缺少可用于比例堆叠的构成占比",
                    company_panel_count,
                )
                if str(company_frame["公司"].iloc[0]) in all_zero_companies
                else empty_company_chart(
                    "未披露可绘图数据",
                    company_panel_count,
                )
                if str(company_frame["公司"].iloc[0]) in unavailable_companies
                else build_component_stack_chart(
                    company_frame,
                    specs,
                    periods,
                    chart_name,
                    label_ratio_codes=(
                        QUANT_RISK_STACK_RATIO_CODES
                        if chart_name == "量化风险最低资本构成"
                        else None
                    ),
                    label_denominator_code=(
                        COMPONENT_STACK_DENOMINATOR_CODES.get(chart_name, "")
                    ),
                    value_labels=False,
                    negative_component_codes=negative_component_codes,
                    dense_layout=dense_company_panels,
                    panel_count=company_panel_count,
                    shared_y_domain=shared_stack_domain,
                    plot_proportions=proportion_stack,
                    show_small_negative_labels=disclosure_stack,
                    avoid_label_overlap=(
                        chart_name == "计入各级资本的保单未来盈余构成占比"
                    ),
                    min_label_share=(
                        0.10
                        if chart_name in {"各类保险风险（寿）占比", "各类市场风险占比"}
                        else None
                    ),
                    hide_labels_at_threshold=chart_name == "各类保险风险（寿）占比",
                )
            ),
            key_prefix=f"{key_prefix}_component_stack",
            highlight_company=highlight_company,
        )
        if all_zero_companies:
            st.caption(
                f"注：{'、'.join(all_zero_companies)}的损失发生、退保、费用风险最低资本及"
                "风险分散效应均为0，"
                "缺少可用于比例堆叠的构成占比。"
            )
        if disclosure_stack and (unavailable_companies or partial_companies):
            subject = "认可资产" if chart_name == "认可资产构成" else "认可负债"
            notes: list[str] = []
            if unavailable_companies:
                notes.append(
                    f"未披露——{'、'.join(unavailable_companies)}：所选期间缺少可绘制的{subject}明细或合计"
                )
            if partial_companies:
                notes.append(
                    f"部分未披露——{'、'.join(partial_companies)}：部分报告期或构成项目缺失"
                )
            st.caption(f"注：{'；'.join(notes)}。缺失项目保持为空，不以0替代。")
    elif plan.kind == CAPITAL_AMOUNT_COMBO:
        st.caption(
            "100%堆叠柱显示核心一级、核心二级、附属一级和附属二级资本构成；"
            "所有公司和报告期的柱高一致，色块高度代表结构占比，悬浮可查看占比和金额。"
        )
        capital_codes = [
            "CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL",
            "ANC_T2_CAPITAL",
        ]
        expected_count = converted["公司"].nunique() * len(periods) * len(capital_codes)
        disclosed_count = converted[
            converted["指标编码"].astype(str).isin(capital_codes)
        ].drop_duplicates(["公司", "报告期", "指标编码"]).shape[0]
        capital_legend = [
            (color, label, "square")
            for label, color in CAPITAL_STRUCTURE_COLORS.items()
        ]
        if disclosed_count < expected_count:
            capital_legend.append(("#B8BDC7", "未披露", "square"))
        _render_chart_legend(capital_legend)
        _render_company_chart_grid(
            converted,
            lambda company_frame: build_capital_amount_combo(
                company_frame,
                periods,
                dense_layout=dense_company_panels,
                panel_count=company_panel_count,
            ),
            key_prefix=f"{key_prefix}_capital_amount",
            highlight_company=highlight_company,
        )
    elif plan.kind == EFFECT_DIVERGING:
        codes = metric_codes_for_chart(chart_name)
        if codes:
            st.caption("按公司展示抵减效应的跨期方向与幅度，零线用于区分正向增加和负向抵减。")
            st.altair_chart(
                _chart_without_internal_title(
                    build_effect_diverging_chart(converted, codes[0], periods)
                ),
                width="stretch",
                key=f"{key_prefix}_{codes[0]}_effect",
            )
    elif plan.kind == CAPITAL_EFFICIENCY_BUBBLE:
        _render_capital_efficiency_bubble_fragment(
            converted,
            periods,
            company_colors,
            highlight_company,
            key_prefix,
        )
    elif plan.kind == SOLVENCY_MATRIX:
        chart, _ = build_matrix_chart(
            converted,
            "CORE_SOLVENCY_RATIO",
            "COMBINED_SOLVENCY_RATIO",
            periods,
            "偿付能力矩阵",
            highlight_company,
            regulatory_lines=True,
            company_colors=company_colors,
        )
        st.altair_chart(
            _chart_without_internal_title(chart),
            width="stretch",
            key=f"{key_prefix}_solvency_matrix",
        )
        st.caption("横轴50%、纵轴100%的红色虚线分别为核心和综合偿付能力充足率监管下限。")
    elif plan.kind == MARKET_CREDIT_MATRIX:
        chart, _ = build_matrix_chart(
            converted,
            "MARKET_RISK_TO_QUANT_CAPITAL",
            "CREDIT_RISK_TO_QUANT_CAPITAL",
            periods,
            "市场—信用风险矩阵",
            highlight_company,
            company_colors=company_colors,
        )
        st.altair_chart(
            _chart_without_internal_title(chart),
            width="stretch",
            key=f"{key_prefix}_market_credit_matrix",
        )
        st.caption("深蓝虚线为样本中位数，用于识别市场风险和信用风险同时偏高的公司。")
    elif plan.kind == RISK_RATIO_SCATTER:
        x_code, y_code, bubble_size_code, denominator_label = (
            RISK_RATIO_SCATTER_SPECS[chart_name]
        )
        if chart_name in RISK_SCATTER_DUAL_VIEW_CHARTS:
            _render_risk_ratio_scatter_fragment(
                converted,
                periods,
                company_colors,
                highlight_company,
                key_prefix,
                chart_name,
                x_code,
                y_code,
                bubble_size_code,
                denominator_label,
            )
        else:
            chart, latest_period = build_matrix_chart(
                converted,
                x_code,
                y_code,
                periods,
                chart_name,
                highlight_company,
                company_colors=company_colors,
                percentage_axes=True,
                bubble_size_code=bubble_size_code,
                bubble_size_label=denominator_label,
            )
            centered_chart = _chart_without_internal_title(chart).properties(
                width=RISK_SCATTER_DISPLAY_WIDTH
            )
            st.caption(
                f"展示 {latest_period}；每个气泡代表一家公司，横纵轴均为占{denominator_label}的比例，"
                f"气泡面积代表{denominator_label}规模；"
                "深蓝虚线为样本横纵指标中位数。"
            )
            with st.container(horizontal_alignment="center"):
                st.altair_chart(
                    centered_chart,
                    width=RISK_SCATTER_DISPLAY_WIDTH,
                    key=f"{key_prefix}_risk_asset_scatter",
                )


def show_step_7_solvency(
    data: pd.DataFrame,
    financing_data: pd.DataFrame | None = None,
) -> None:
    """Company report page following the annual-platform report workflow."""
    _report_style()
    st.markdown("<div class='solvency-report-title'>公司级偿付能力对标报告</div>", unsafe_allow_html=True)
    if data is None or data.empty:
        st.info("请先在 Step5 确认集成数据，或在 Step6 上传集成表。")
        return
    status_frame = display_company_names(company_detail_rows(data))
    if status_frame.empty:
        st.info("当前数据没有公司明细记录。")
        return
    status_frame["数值"] = pd.to_numeric(status_frame["数值"], errors="coerce")
    frame = status_frame.dropna(subset=["数值"]).copy()
    all_periods = sort_report_periods(status_frame["报告期"])
    if not all_periods:
        st.info("当前数据没有可用报告期。")
        return

    notes = render_report_notes_editor(
        title="公司内容分析与注释输入",
        key_prefix="step7",
        template=company_notes_template(),
    )
    available_chart_names = list(dict.fromkeys(
        entry.chart_name for entry in COMPANY_NAVIGATION
    ))
    image_overrides: dict[str, dict[str, object]] = {}
    with st.expander("公司级图表设置与图片覆盖", expanded=False, icon=":material/tune:"):
        c0, c1, c2, c3, c4 = st.columns([1, 2, 1, 1, 1])
        peer_groups = nonblank_values(status_frame, "同业分类")
        type_options = [ALL_COMPANY_TYPES, *peer_groups]
        st.session_state.setdefault("s7_company_types", [ALL_COMPANY_TYPES])
        _valid_state("s7_company_types", type_options, multiple=True)
        with c0:
            selected_types_raw = st.multiselect(
                "公司类型",
                type_options,
                key="s7_company_types",
                on_change=_reset_company_selection_for_types,
                args=(status_frame,),
                help="沿用偿付能力平台的‘同业分类’字段进行筛选。",
            )
        company_source, all_companies = _company_scope_for_types(
            status_frame,
            selected_types_raw,
        )
        company_selection_key = _sync_company_selection_state(
            selected_types_raw,
            all_companies,
        )
        with c1:
            selected_companies = st.multiselect(
                "展示公司",
                all_companies,
                key=company_selection_key,
                placeholder="选择一家或多家公司",
            )
        with c2:
            unit_options = ["十亿元", "亿元", "百万元", "十万元"]
            _valid_state("s7_unit_mode", unit_options)
            unit_mode = st.selectbox(
                "显示单位",
                unit_options,
                key="s7_unit_mode",
            )
        with c3:
            highlight_options = ["无", *selected_companies]
            _valid_state("s7_highlight_company", highlight_options)
            highlight_company = st.selectbox(
                "特定追踪",
                highlight_options,
                key="s7_highlight_company",
            )
        with c4:
            enable_ai = st.toggle(
                "一键AI分析",
                value=False,
                key="s7_enable_ai",
                help="使用登录页配置的大模型接口，为当前图表生成一句同业对标点评。",
            )
            ai_data_consent = False
            if enable_ai:
                ai_data_consent = st.checkbox(
                    "确认将当前图表中的公司名称和指标数据发送至已配置的大模型服务",
                    value=False,
                    key="s7_ai_data_consent",
                )
                if not ai_data_consent:
                    st.caption("未确认前不会调用模型接口或发送数据。")

        st.divider()
        sort_options, sort_lookup = _metric_sort_options(company_source)
        _valid_state("s7_sort_field", sort_options)
        sc1, sc2, sc3 = st.columns([2, 2, 3])
        with sc1:
            sort_field = st.selectbox(
                "图表展示顺序依据",
                sort_options,
                key="s7_sort_field",
            )
        with sc2:
            sort_order = st.radio(
                "排序方向",
                [SORT_DESCENDING, SORT_ASCENDING],
                horizontal=True,
                key="s7_sort_order",
            )
        with sc3:
            st.caption("图表类型：按指标特征自动匹配（系统内置）")
        selected_companies = _sort_companies_by_metric(
            company_source,
            selected_companies,
            sort_lookup.get(sort_field, ""),
            all_periods[-1],
            descending=sort_order == SORT_DESCENDING,
        )

        st.markdown("#### 手动上传图片（PNG/JPG）")
        image_overrides = _render_image_override_controls(available_chart_names, notes)

    selected_periods = all_periods
    show_labels = True
    decimals = 2
    transparent = True
    show_average = False
    if not selected_companies:
        st.info("请在报告配置中至少选择一家展示公司。")
        return
    scoped = filter_analysis_frame(frame, companies=selected_companies, periods=selected_periods)
    scoped_status = filter_analysis_frame(
        status_frame,
        companies=selected_companies,
        periods=selected_periods,
    )
    scoped["_s7_company_order"] = pd.Categorical(
        scoped["公司"], categories=selected_companies, ordered=True
    )
    scoped = scoped.sort_values(["_s7_company_order", "报告期"]).drop(columns="_s7_company_order")

    chart_names = _selected_chart_names(scoped_status)
    if not chart_names:
        st.info("请在侧边栏公司报告导航中选择具体图表。")
        return
    print_all = st.session_state.get("company_nav_level_one") == PRINT_ALL_LABEL
    if print_all:
        today = date.today()
        period_label = (
            selected_periods[0]
            if len(selected_periods) == 1
            else f"{selected_periods[0]}–{selected_periods[-1]}"
        )
        company_label = (
            "、".join(selected_companies)
            if len(selected_companies) <= 4
            else f"{len(selected_companies)} 家公司"
        )
        render_report_cover(
            title="保险公司偿付能力公司对标报告",
            subtitle=f"{period_label} · {company_label}",
            date_text=f"{today.year}年{today.month}月",
            picture_dir=PICTURE_DIR,
        )
    for chart_index, chart_name in enumerate(chart_names):
        if print_all and chart_index:
            st.markdown("<div class='solvency-report-page-break'></div>", unsafe_allow_html=True)
        entries = [entry for entry in COMPANY_NAVIGATION if entry.chart_name == chart_name]
        level_one = entries[0].level_one if entries else ""
        level_two = entries[0].level_two if entries else ""
        chart_render_key = COMPANY_CHART_RENDER_KEYS.get(
            chart_name,
            f"chart_{chart_index}",
        )
        with st.container(key=f"s7_report_module_{chart_render_key}"):
            st.markdown(
                f"<div class='solvency-module-title'>"
                f"{level_one} - {level_two} - {chart_name}</div>",
                unsafe_allow_html=True,
            )
            chart_note = notes.get(chart_name, {})
            render_report_analysis(chart_note)
            override = image_overrides.get(chart_name)
            if override:
                st.image(
                    override["data"],
                    caption=f"图片覆盖：{override['name']}",
                    width="stretch",
                )
            else:
                plan = chart_plan_for(chart_name)
                available_codes = set(scoped_status["指标编码"].fillna("").astype(str))
                entry = entries[0] if entries else None
                missing_codes = (
                    [code for code in entry.metric_codes if code not in available_codes]
                    if entry is not None
                    else []
                )
                if entry is not None and entry.requires_all and missing_codes:
                    st.warning(
                        f"“{chart_name}”缺少 {len(missing_codes)} 个必要指标，暂无法按 Excel 指定格式完整绘图。"
                    )
                    st.caption("缺失指标编码：" + "、".join(missing_codes))
                elif plan.kind == QUALITY_AND_CAPITAL:
                    _render_quality_and_capital(
                        scoped_status, chart_name, periods=selected_periods,
                        unit_mode=unit_mode, highlight_company=highlight_company,
                        key_prefix=f"s7_{chart_render_key}", selected_companies=selected_companies,
                    )
                elif plan.kind in {SINGLE_METRIC_TREND, TREND_WITH_COMPANY_BARS, COMPANY_BAR_TREND}:
                    for code in metric_codes_for_chart(chart_name):
                        _render_report_metric(
                            scoped,
                            code,
                            status_frame=scoped_status,
                            periods=selected_periods,
                            show_labels=show_labels,
                            decimals=decimals,
                            unit_mode=unit_mode,
                            transparent=transparent,
                            show_average=show_average,
                            highlight_company=highlight_company,
                            enable_ai=enable_ai and ai_data_consent,
                            key_prefix=f"s7_{chart_render_key}",
                            company_panels=plan.kind == COMPANY_BAR_TREND,
                            company_period_bars=plan.kind == TREND_WITH_COMPANY_BARS,
                        )
                else:
                    _render_combination_analysis(
                        scoped,
                        chart_name,
                        periods=selected_periods,
                        unit_mode=unit_mode,
                        highlight_company=highlight_company,
                        key_prefix=f"s7_{chart_render_key}",
                        financing_data=financing_data,
                    )
            render_report_footnote(chart_note)

    if print_all:
        render_report_back_cover(picture_dir=PICTURE_DIR)

    with st.expander("查看报告底层数据"):
        st.dataframe(scoped, width="stretch", hide_index=True)
