from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_components import (
    company_detail_rows,
    render_industry_overview,
    render_major_financing,
    render_peer_classification,
    render_report_analysis,
    render_report_back_cover,
    render_report_cover,
    render_report_footnote,
    render_report_notes_editor,
)
from services.solvency_navigation import (
    INDUSTRY_NAVIGATION,
    INDUSTRY_QUANT_CHART,
    KPMG_DEFAULT_COLORS,
    OVERVIEW_LEVEL,
    PRINT_ALL_LABEL,
    metric_codes_for_chart,
    resolve_chart_selection,
)
from services.solvency_step6_analysis import (
    format_chart_value,
    nonblank_values,
    sort_report_periods,
)
from services.solvency_report_notes import industry_notes_template


REGULATORY_LIMITS = {
    "CORE_SOLVENCY_RATIO": 50.0,
    "COMBINED_SOLVENCY_RATIO": 100.0,
}
PICTURE_DIR = Path(__file__).resolve().parent / "picture"
SUPPLEMENTAL_SECTIONS = (
    "行业整体偿付能力概览",
    "调研公司分类列表",
    "重大融资信息统计",
)
PEER_GROUP_COLUMN = "同业分类"
DEFAULT_PEER_GROUP_ORDER = ("头部", "银行系", "外资", "养老健康", "小型")


def _rgba(color: str, alpha: float = 0.17) -> str:
    """Return a Plotly-compatible rgba color for a six-digit hex color."""
    text = str(color or "").strip()
    if len(text) == 7 and text.startswith("#"):
        try:
            red = int(text[1:3], 16)
            green = int(text[3:5], 16)
            blue = int(text[5:7], 16)
            opacity = min(1.0, max(0.0, float(alpha)))
            return f"rgba({red},{green},{blue},{opacity:.3f})"
        except ValueError:
            pass
    return "rgba(0,51,141,0.170)"


def _valid_state(key: str, options: Iterable[str], *, multiple: bool = False) -> None:
    values = list(options)
    if key not in st.session_state:
        return
    if multiple:
        st.session_state[key] = [value for value in st.session_state[key] if value in values]
    elif st.session_state[key] not in values:
        st.session_state.pop(key, None)


def _ordered_peer_groups(values: Iterable[str]) -> list[str]:
    """Return peer groups in the annual-report platform's display order."""
    present = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    ordered = [group for group in DEFAULT_PEER_GROUP_ORDER if group in present]
    return [*ordered, *(group for group in present if group not in ordered)]


def _default_peer_group_styles(
    peer_groups: Iterable[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return stable system labels and KPMG colors for peer-group charts."""
    ordered = list(peer_groups)
    colors = {
        peer_group: KPMG_DEFAULT_COLORS[index % len(KPMG_DEFAULT_COLORS)]
        for index, peer_group in enumerate(ordered)
    }
    labels = {peer_group: peer_group for peer_group in ordered}
    return colors, labels


def _report_style() -> None:
    st.markdown(
        """
        <style>
        .industry-report-title {
            color:#00338D; font-size:28px; font-weight:800;
            border-bottom:2px solid #00338D; padding-bottom:8px; margin:6px 0 18px;
        }
        .industry-module-title {color:#00338D; font-size:22px; font-weight:750; margin:18px 0 8px;}
        .industry-analysis-note {
            background:#F2F6FC; border-left:4px solid #008578;
            border-radius:4px; padding:9px 13px; margin:6px 0 12px; color:#243B53;
        }
        .solvency-report-analysis {
            background:#F4F7FC; border-left:4px solid #00338D;
            border-radius:3px; padding:7px 10px; margin:4px 0 10px;
        }
        .solvency-report-analysis p {margin:2px 0; line-height:1.45;}
        .solvency-note-default {color:#0A1F5C; font-size:13px;}
        .solvency-note-custom {color:#1E49E2; font-size:13px; font-weight:600;}
        .solvency-report-footnote {color:#777; font-size:12px; font-style:italic; margin:4px 0 16px;}
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
        @media print {
            [data-testid="stHeader"], [data-testid="stSidebar"],
            [data-testid="stToolbar"], [data-testid="collapsedControl"],
            [data-testid="stExpander"], div[role="tablist"], h1,
            .platform-page-heading, .industry-report-title,
            .solvency-no-print {display:none!important;}
            [data-testid="stElementContainer"]:has(.platform-page-heading),
            [data-testid="stElementContainer"]:has(.industry-report-title) {display:none!important;}
            .industry-report-module {break-inside:avoid; page-break-inside:avoid;}
            .industry-report-page-break {break-before:page; page-break-before:always;}
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
                height:190.5mm!important; margin:0!important; padding:0!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stVerticalBlock"]:has(.solvency-print-cover) {
                gap:0!important; margin:0!important; padding:0!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s8_report_module_"],
            html.solvency-print-mode-widescreen [class*="st-key-s8_section_"] {
                width:100%!important; max-width:100%!important;
                min-height:174.5mm!important; padding:0!important;
                margin:0!important; box-sizing:border-box!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stElementContainer"]:has(.solvency-print-cover--front) {
                break-after:page!important; page-break-after:always!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stElementContainer"]:has(.solvency-print-cover--back) {
                break-before:page!important; page-break-before:always!important;
            }
            html.solvency-print-mode-portrait [data-testid="stElementContainer"]:has(.solvency-print-cover),
            html.solvency-print-mode-portrait .solvency-print-cover {display:none!important;}
            .industry-module-title {font-size:30px!important; margin-top:0!important;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _selected_chart_names(frame: pd.DataFrame) -> list[str]:
    available = set(frame["指标编码"].fillna("").astype(str))
    if st.session_state.get("industry_nav_level_one") == PRINT_ALL_LABEL:
        return list(
            dict.fromkeys(
                entry.chart_name
                for entry in INDUSTRY_NAVIGATION
                if (
                    all(code in available for code in entry.metric_codes)
                    if entry.requires_all
                    else any(code in available for code in entry.metric_codes)
                )
            )
        )
    return resolve_chart_selection(
        OVERVIEW_LEVEL,
        st.session_state.get("industry_nav_level_two", "全部"),
        st.session_state.get("industry_nav_chart", ""),
        industry=True,
        available_codes=available,
    )


def _industry_total_rows(frame: pd.DataFrame) -> pd.DataFrame:
    company = frame["公司"].fillna("").astype(str).str.strip()
    company_type = frame["公司类型"].fillna("").astype(str).str.strip()
    company_code = frame["公司统一编码"].fillna("").astype(str).str.strip()
    return frame.loc[
        company.eq("行业合计")
        | company_type.eq("行业合计")
        | company_code.str.startswith("INDUSTRY_")
    ].copy()


def _industry_quant_waterfall_figure(
    frame: pd.DataFrame,
    period: str,
) -> go.Figure:
    codes = metric_codes_for_chart(INDUSTRY_QUANT_CHART, industry=True)
    rows = frame[
        frame["指标编码"].astype(str).isin(codes)
        & frame["报告期"].astype(str).eq(str(period))
    ].copy()
    rows["数值"] = pd.to_numeric(rows["数值"], errors="coerce")
    rows = rows.dropna(subset=["数值"])
    lookup = rows.groupby("指标编码", sort=False)["数值"].first().to_dict()
    labels = {
        "INDUSTRY_LIFE_INSURANCE_RISK": "寿险保险风险",
        "INDUSTRY_NON_LIFE_INSURANCE_RISK": "非寿险保险风险",
        "INDUSTRY_MARKET_RISK": "市场风险",
        "INDUSTRY_CREDIT_RISK": "信用风险",
        "INDUSTRY_CAPITALIZABLE_DIVERSIFICATION_EFFECT": "风险分散效应",
        "INDUSTRY_LOSS_ABSORPTION": "损失吸收",
        "INDUSTRY_CONTROL_RISK": "控制风险",
    }
    x = [labels[code] for code in codes if code in lookup]
    y = [lookup[code] for code in codes if code in lookup]
    total = sum(y)
    fig = go.Figure(
        go.Waterfall(
            name=str(period),
            orientation="v",
            measure=["relative"] * len(y) + ["total"],
            x=[*x, "量化风险最低资本"],
            y=[*y, total],
            text=[f"{value:,.0f}" for value in y] + [f"{total:,.0f}"],
            textposition="outside",
            connector={"line": {"color": "#9AA6B2"}},
            increasing={"marker": {"color": "#1E49E2"}},
            decreasing={"marker": {"color": "#FD349C"}},
            totals={"marker": {"color": "#00338D"}},
            hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{period} 行业量化风险最低资本构成",
        font={"family": "Microsoft YaHei", "color": "#0C233C"},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin={"t": 80, "l": 20, "r": 20, "b": 70},
        yaxis={"title": "金额"},
    )
    return fig


def _render_industry_quant_waterfalls(
    frame: pd.DataFrame,
    periods: list[str],
    key_prefix: str,
) -> None:
    available = [
        period
        for period in periods
        if not frame[frame["报告期"].astype(str).eq(str(period))].empty
    ]
    if not available:
        st.caption("当前报告期没有行业量化风险最低资本构成数据。")
        return
    st.caption("正值表示各类毛风险，负值表示风险分散、损失吸收及控制风险等抵减效应。")
    columns = st.columns(min(2, len(available)))
    for index, period in enumerate(available):
        with columns[index % len(columns)]:
            st.plotly_chart(
                _industry_quant_waterfall_figure(frame, period),
                width="stretch",
                key=f"{key_prefix}_{period}_waterfall",
            )


def _metric_name(frame: pd.DataFrame, code: str) -> str:
    rows = frame[frame["指标编码"].astype(str).eq(code)]
    return code if rows.empty else str(rows.iloc[0]["指标名称"])


def _period_scope_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "期间口径" not in frame.columns:
        return frame
    mask = frame["期间口径"].fillna("").astype(str).str.contains(
        "本季度末|期末|时点", regex=True, na=False
    )
    return frame.loc[mask].copy() if mask.any() else frame.copy()


def _distribution_stats(
    frame: pd.DataFrame,
    peer_group_order: Iterable[str] = (),
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    stats = (
        frame.groupby(["报告期", PEER_GROUP_COLUMN], dropna=False)["数值"]
        .agg(
            公司数="count",
            最小值="min",
            下四分位=lambda s: s.quantile(0.25),
            中位数="median",
            上四分位=lambda s: s.quantile(0.75),
            最大值="max",
            平均值="mean",
        )
        .reset_index()
    )
    order = list(peer_group_order) or _ordered_peer_groups(stats[PEER_GROUP_COLUMN])
    stats[PEER_GROUP_COLUMN] = pd.Categorical(
        stats[PEER_GROUP_COLUMN], order, ordered=True
    )
    return stats.sort_values([PEER_GROUP_COLUMN, "报告期"]).reset_index(drop=True)


def _trend_figure(
    stats: pd.DataFrame,
    metric_name: str,
    metric_code: str,
    period_order: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    decimals: int,
    show_labels: bool,
) -> go.Figure:
    fig = go.Figure()
    for peer_group in stats[PEER_GROUP_COLUMN].drop_duplicates():
        rows = stats[stats[PEER_GROUP_COLUMN].eq(peer_group)].copy()
        rows["报告期"] = pd.Categorical(rows["报告期"], period_order, ordered=True)
        rows = rows.sort_values("报告期")
        x = rows["报告期"].astype(str)
        color = colors.get(str(peer_group), "#00338D")
        display = labels.get(str(peer_group), str(peer_group))
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rows["上四分位"],
                mode="lines",
                line={"width": 0, "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rows["下四分位"],
                mode="lines",
                fill="tonexty",
                fillcolor=_rgba(color),
                line={"width": 0, "color": color},
                name=f"{display} 四分位区间",
                hovertemplate="报告期：%{x}<br>下四分位：%{y:,.2f}<extra></extra>",
                showlegend=False,
            )
        )
        text = [
            format_chart_value(value, "%" if "SOLVENCY_RATIO" in metric_code else "", "", decimals)
            for value in rows["中位数"]
        ]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=rows["中位数"],
                mode="lines+markers+text" if show_labels else "lines+markers",
                text=text if show_labels else None,
                textposition="top center",
                name=display,
                line={"color": color, "width": 3},
                marker={"size": 9, "color": color},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (rows["最大值"] - rows["中位数"]).clip(lower=0),
                    "arrayminus": (rows["中位数"] - rows["最小值"]).clip(lower=0),
                    "color": color,
                    "thickness": 1,
                    "width": 3,
                },
                customdata=rows[["公司数", "最小值", "下四分位", "上四分位", "最大值"]],
                hovertemplate=(
                    "报告期：%{x}<br>中位数：%{y:,.2f}<br>公司数：%{customdata[0]}"
                    "<br>最小值：%{customdata[1]:,.2f}<br>下四分位：%{customdata[2]:,.2f}"
                    "<br>上四分位：%{customdata[3]:,.2f}<br>最大值：%{customdata[4]:,.2f}<extra></extra>"
                ),
            )
        )
    limit = REGULATORY_LIMITS.get(metric_code)
    if limit is not None:
        fig.add_hline(
            y=limit,
            line_dash="dash",
            line_color="#ED2124",
            annotation_text=f"监管下限 {limit:g}%",
            annotation_font_color="#ED2124",
        )
    fig.update_layout(
        title=f"{metric_name}行业分布趋势（中位数、四分位区间及极值）",
        font={"family": "Microsoft YaHei", "color": "#0C233C"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        margin={"t": 100, "l": 20, "r": 20, "b": 20},
        yaxis={"tickformat": ",.1f", "ticksuffix": "%" if "SOLVENCY_RATIO" in metric_code else ""},
    )
    return fig


def _ranking_figure(
    frame: pd.DataFrame,
    metric_name: str,
    metric_code: str,
    colors: dict[str, str],
    labels: dict[str, str],
    decimals: int,
    show_labels: bool,
    bar_gap: float = 0.35,
    peer_group_order: Iterable[str] = (),
) -> go.Figure:
    ranking = frame.sort_values("数值", ascending=False).copy()
    fig = go.Figure()
    order = list(peer_group_order) or _ordered_peer_groups(ranking[PEER_GROUP_COLUMN])
    for peer_group in order:
        rows = ranking[ranking[PEER_GROUP_COLUMN].eq(peer_group)]
        if rows.empty:
            continue
        data_labels = [
            format_chart_value(value, row_unit, row_type, decimals)
            for value, row_unit, row_type in zip(rows["数值"], rows["单位"], rows["数据类型"])
        ]
        fig.add_trace(
            go.Bar(
                x=rows["公司"],
                y=rows["数值"],
                name=labels.get(str(peer_group), str(peer_group)),
                marker_color=colors.get(str(peer_group), "#00338D"),
                text=data_labels if show_labels else None,
                textposition="outside",
                cliponaxis=False,
                customdata=rows[[PEER_GROUP_COLUMN, "报告期", "期间口径"]],
                hovertemplate=(
                    "公司：%{x}<br>同业分类：%{customdata[0]}<br>报告期：%{customdata[1]}"
                    "<br>期间口径：%{customdata[2]}<br>数值：%{y:,.2f}<extra></extra>"
                ),
            )
        )
    limit = REGULATORY_LIMITS.get(metric_code)
    if limit is not None:
        fig.add_hline(
            y=limit, line_dash="dash", line_color="#ED2124",
            annotation_text=f"监管下限 {limit:g}%", annotation_font_color="#ED2124"
        )
    fig.update_layout(
        title=f"{metric_name}公司分布",
        barmode="group",
        font={"family": "Microsoft YaHei", "color": "#0C233C"},
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={"orientation": "h", "y": 1.12, "x": 1, "xanchor": "right"},
        xaxis={"tickangle": -35},
        yaxis={"tickformat": ",.1f", "ticksuffix": "%" if "SOLVENCY_RATIO" in metric_code else ""},
        margin={"t": 100, "l": 20, "r": 20, "b": 70},
        bargap=min(0.9, max(0.0, float(bar_gap))),
    )
    return fig


def _render_industry_metric(
    frame: pd.DataFrame,
    code: str,
    periods: list[str],
    peer_groups: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    decimals: int,
    print_mode: bool,
    key_prefix: str,
) -> None:
    metric = frame[
        frame["指标编码"].astype(str).eq(code)
        & frame["报告期"].astype(str).isin(periods)
        & frame[PEER_GROUP_COLUMN].astype(str).isin(peer_groups)
    ].copy()
    metric = _period_scope_rows(metric)
    metric["数值"] = pd.to_numeric(metric["数值"], errors="coerce")
    metric = metric.dropna(subset=["数值"])
    if metric.empty:
        st.caption(f"{_metric_name(frame, code)}：当前筛选范围无数据。")
        return
    name = str(metric.iloc[0]["指标名称"])
    stats = _distribution_stats(metric, peer_groups)
    st.markdown(f"#### {name}")
    if print_mode:
        show_labels, bar_gap = True, 0.35
    else:
        control_left, control_right = st.columns([1, 2])
        with control_left:
            show_labels = st.toggle(
                "显示数据标签",
                value=True,
                key=f"{key_prefix}_{code}_labels",
            )
        with control_right:
            bar_gap = st.slider(
                "柱子间距",
                min_value=0.05,
                max_value=0.80,
                value=0.35,
                step=0.05,
                key=f"{key_prefix}_{code}_bar_gap",
            )
    latest_period = sort_report_periods(metric["报告期"])[-1]
    latest = metric[metric["报告期"].astype(str).eq(latest_period)]
    c1, c2, c3 = st.columns(3)
    c1.metric("纳入公司", f"{latest['公司'].nunique():,} 家", border=True)
    c2.metric(
        "行业中位数",
        format_chart_value(latest["数值"].median(), latest.iloc[0].get("单位", ""), latest.iloc[0].get("数据类型", ""), decimals),
        border=True,
    )
    c3.metric(
        "行业平均值",
        format_chart_value(latest["数值"].mean(), latest.iloc[0].get("单位", ""), latest.iloc[0].get("数据类型", ""), decimals),
        border=True,
    )
    st.plotly_chart(
        _trend_figure(stats, name, code, periods, colors, labels, decimals, show_labels),
        width="stretch",
        key=f"{key_prefix}_{code}_trend",
    )
    st.plotly_chart(
        _ranking_figure(
            latest,
            name,
            code,
            colors,
            labels,
            decimals,
            show_labels,
            bar_gap,
            peer_groups,
        ),
        width="stretch",
        key=f"{key_prefix}_{code}_rank",
    )
    st.markdown(
        "<div class='industry-analysis-note'>"
        "趋势图中实线为中位数，浅色带为上下四分位区间，误差线表示最小值至最大值；"
        "偿付能力充足率同时显示监管下限，便于跨期观察分布变化。"
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"查看{name}统计明细"):
        st.dataframe(stats, width="stretch", hide_index=True)


def show_step_8_solvency(
    data: pd.DataFrame,
    financing_data: pd.DataFrame | None = None,
) -> None:
    """Industry report page following the annual-platform report workflow."""
    _report_style()
    st.markdown("<div class='industry-report-title'>行业偿付能力分析报告</div>", unsafe_allow_html=True)
    if data is None or data.empty:
        st.info("请先在 Step5 确认多公司集成数据。")
        return
    industry_totals = _industry_total_rows(data)
    frame = company_detail_rows(data).copy()
    if frame.empty:
        st.info("当前数据没有公司明细记录。")
        return
    frame["数值"] = pd.to_numeric(frame["数值"], errors="coerce")
    frame = frame.dropna(subset=["数值"])
    periods = sort_report_periods(frame["报告期"])
    peer_groups = _ordered_peer_groups(nonblank_values(frame, PEER_GROUP_COLUMN))
    if not periods or not peer_groups:
        st.info("当前数据缺少报告期或同业分类。")
        return

    notes = render_report_notes_editor(
        title="行业分析注释输入",
        key_prefix="step8",
        template=industry_notes_template(),
    )
    with st.expander("行业分析配置", expanded=False, icon=":material/tune:"):
        c1, c2, c3 = st.columns(3)
        _valid_state("s8_periods", periods, multiple=True)
        _valid_state("s8_peer_groups", peer_groups, multiple=True)
        with c1:
            selected_peer_groups = st.multiselect(
                "选择同业分类",
                peer_groups,
                default=peer_groups if "s8_peer_groups" not in st.session_state else None,
                key="s8_peer_groups",
            )
        with c2:
            selected_periods = st.multiselect(
                "选择报告期",
                periods,
                default=periods[-4:] if "s8_periods" not in st.session_state else None,
                key="s8_periods",
            )
        with c3:
            decimals = int(st.number_input("小数位数", 0, 4, 1, key="s8_decimals"))
            _valid_state("s8_supplemental_sections", SUPPLEMENTAL_SECTIONS, multiple=True)
            selected_sections = st.multiselect(
                "附加行业板块",
                SUPPLEMENTAL_SECTIONS,
                default=(
                    list(SUPPLEMENTAL_SECTIONS)
                    if "s8_supplemental_sections" not in st.session_state
                    else None
                ),
                key="s8_supplemental_sections",
            )

        _valid_state("s8_peer_group_order", peer_groups, multiple=True)
        selected_order = st.multiselect(
            "同业分类显示顺序",
            peer_groups,
            default=peer_groups if "s8_peer_group_order" not in st.session_state else None,
            key="s8_peer_group_order",
            help="按选中顺序控制图表图例及分类的展示顺序；未选分类会自动追加。",
        )
        peer_group_order = [
            *selected_order,
            *(group for group in peer_groups if group not in selected_order),
        ]
        selected_peer_groups = [
            group for group in peer_group_order if group in selected_peer_groups
        ]

    peer_group_colors, peer_group_labels = _default_peer_group_styles(peer_group_order)

    if not selected_peer_groups or not selected_periods:
        st.info("请在行业分析配置中至少选择一种同业分类和一个报告期。")
        return
    scoped = frame[
        frame["报告期"].astype(str).isin(selected_periods)
        & frame[PEER_GROUP_COLUMN].astype(str).isin(selected_peer_groups)
    ].copy()
    navigation_frame = pd.concat([scoped, industry_totals], ignore_index=True)
    chart_names = _selected_chart_names(navigation_frame)
    print_all = st.session_state.get("industry_nav_level_one") == PRINT_ALL_LABEL
    if print_all:
        today = date.today()
        period_label = (
            selected_periods[0]
            if len(selected_periods) == 1
            else f"{selected_periods[0]}–{selected_periods[-1]}"
        )
        render_report_cover(
            title="保险公司偿付能力行业分析报告",
            subtitle=f"{period_label} · {'、'.join(selected_peer_groups)}",
            date_text=f"{today.year}年{today.month}月",
            picture_dir=PICTURE_DIR,
        )

    rendered_sections = 0

    def start_report_page() -> None:
        nonlocal rendered_sections
        if print_all and rendered_sections:
            st.markdown(
                "<div class='industry-report-page-break'></div>",
                unsafe_allow_html=True,
            )
        rendered_sections += 1

    for section_name in selected_sections:
        start_report_page()
        with st.container(key=f"s8_section_{rendered_sections}"):
            render_report_analysis(notes.get(section_name, {}))
            if section_name == "行业整体偿付能力概览":
                render_industry_overview(scoped)
            elif section_name == "调研公司分类列表":
                render_peer_classification(scoped)
            else:
                financing_view = financing_data
                if isinstance(financing_view, pd.DataFrame) and not financing_view.empty:
                    financing_view = financing_view[
                        financing_view["季度"].astype(str).isin(selected_periods)
                    ].copy()
                render_major_financing(financing_view, print_mode=print_all)
            render_report_footnote(notes.get(section_name, {}))

    if not chart_names and not selected_sections:
        st.info("请在侧边栏行业分析导航中选择具体图表，或在配置中启用附加行业板块。")
        return
    for chart_index, chart_name in enumerate(chart_names):
        start_report_page()
        entries = [
            entry for entry in INDUSTRY_NAVIGATION
            if entry.chart_name == chart_name
        ]
        level_two = entries[0].level_two if entries else ""
        with st.container(key=f"s8_report_module_{chart_index}"):
            st.markdown(
                f"<div class='industry-module-title'>"
                f"{OVERVIEW_LEVEL} · {level_two} · {chart_name}</div>",
                unsafe_allow_html=True,
            )
            render_report_analysis(notes.get(chart_name, {}))
            if chart_name == INDUSTRY_QUANT_CHART:
                _render_industry_quant_waterfalls(
                    industry_totals,
                    selected_periods,
                    key_prefix=f"s8_{chart_index}",
                )
            else:
                for code in metric_codes_for_chart(chart_name, industry=True):
                    _render_industry_metric(
                        scoped,
                        code,
                        selected_periods,
                        selected_peer_groups,
                        peer_group_colors,
                        peer_group_labels,
                        decimals,
                        print_mode=print_all,
                        key_prefix=f"s8_{chart_index}",
                    )
            render_report_footnote(notes.get(chart_name, {}))

    if print_all:
        render_report_back_cover(picture_dir=PICTURE_DIR)

    with st.expander("查看行业报告底层数据"):
        st.dataframe(scoped, width="stretch", hide_index=True)
