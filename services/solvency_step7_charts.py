from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import altair as alt
import pandas as pd

from .solvency_navigation import (
    KPMG_BRIGHT_CHART_COLORS,
    KPMG_CAPITAL_COMBO_COLORS,
    KPMG_CAPITAL_TIER_COLORS,
    KPMG_CHART_COLORS,
    KPMG_PERIOD_CHART_COLORS,
)


TRANSPARENT = "transparent"
ANNUAL_REPORT_FONT = "Microsoft YaHei"
COMPANY_PANEL_HEIGHT = 250
ANNUAL_REPORT_PANEL_BORDER = "#EAEAEA"
REGULATORY_LIMITS = {
    "CORE_SOLVENCY_RATIO": 50.0,
    "COMBINED_SOLVENCY_RATIO": 100.0,
}
CAPITAL_AMOUNT_CODES = (
    "CORE_T1_CAPITAL",
    "CORE_T2_CAPITAL",
    "ANC_T1_CAPITAL",
    "ANC_T2_CAPITAL",
)
COMPONENT_LABELS = {
    "CORE_T1_CAPITAL": "核心一级资本",
    "CORE_T2_CAPITAL": "核心二级资本",
    "ANC_T1_CAPITAL": "附属一级资本",
    "ANC_T2_CAPITAL": "附属二级资本",
    "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL": "风险分散效应",
    "LOSS_ABSORPTION_TO_QUANT_CAPITAL": "损失吸收效应",
}
COMPONENT_COLORS = {
    "核心一级资本": KPMG_CAPITAL_TIER_COLORS[0],
    "核心二级资本": KPMG_CAPITAL_TIER_COLORS[1],
    "附属一级资本": KPMG_CAPITAL_TIER_COLORS[2],
    "附属二级资本": KPMG_CAPITAL_TIER_COLORS[3],
    "风险分散效应": KPMG_BRIGHT_CHART_COLORS[4],
    "损失吸收效应": KPMG_BRIGHT_CHART_COLORS[5],
}
CAPITAL_STRUCTURE_COLORS = {
    "核心一级资本": "#00B8F5",
    "核心二级资本": "#ACEAFF",
    "附属一级资本": "#B497FF",
    "附属二级资本": "#7213EA",
}
CAPITAL_COMPONENT_ORDER = {
    label: index
    for index, label in enumerate(list(COMPONENT_COLORS)[:4])
}
MIN_INSIDE_LABEL_SHARE = 0.045
UNDISCLOSED_COLOR = "#B8BDC7"
RESPONSIVE_BAR_WIDTH = alt.RelativeBandSize(0.72)
TREND_LABEL_FORMATS = {
    "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES": ",.5f",
    "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL": ",.1f",
    "NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL": ",.1f",
    "MARKET_RISK_TO_QUANT_CAPITAL": ",.1f",
    "CREDIT_RISK_TO_QUANT_CAPITAL": ",.1f",
    "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL": ",.1f",
    "LOSS_ABSORPTION_TO_QUANT_CAPITAL": ",.1f",
}
TREND_AXIS_FORMATS = {
    code: ",.1f"
    for code in (
        "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
        "NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",
        "MARKET_RISK_TO_QUANT_CAPITAL",
        "CREDIT_RISK_TO_QUANT_CAPITAL",
        "DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL",
        "LOSS_ABSORPTION_TO_QUANT_CAPITAL",
    )
}
COMPACT_TREND_SCALE_CODES = {
    "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",
}
COMPACT_POLICY_RATIO_CODES = {
    "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",
    "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",
}


def report_period_color_map(period_order: Iterable[str]) -> dict[str, str]:
    """Assign stable annual-report-style KPMG colors to report periods."""
    periods = list(dict.fromkeys(str(period) for period in period_order))
    return {
        period: KPMG_PERIOD_CHART_COLORS[index % len(KPMG_PERIOD_CHART_COLORS)]
        for index, period in enumerate(periods)
    }


def report_period_combo_bar_color_map(
    period_order: Iterable[str],
    *,
    line_color: str = "#FD349C",
) -> dict[str, str]:
    """Keep period bars distinct from the fixed line color in combo charts."""
    colors = report_period_color_map(period_order)
    return {
        period: "#098E7E" if color.upper() == line_color.upper() else color
        for period, color in colors.items()
    }


def _label_color_for_fill(color: str) -> str:
    """Use white labels only on the darker official KPMG fill colors."""
    dark_fills = {"#00338D", "#1E49E2", "#0C233C", "#7213EA", "#510DBC"}
    return "#FFFFFF" if str(color).upper() in dark_fills else "#0C233C"


def _whole_number_label(value: float) -> str:
    """Format chart labels as integers using conventional half-up rounding."""
    number = float(value)
    rounded = math.floor(abs(number) + 0.5)
    prefix = "-" if number < 0 and rounded else ""
    return f"{prefix}{rounded:,}"


def _whole_percent_label(value: float) -> str:
    return f"{_whole_number_label(float(value) * 100)}%"


def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["数值"] = pd.to_numeric(result["数值"], errors="coerce")
    return result.dropna(subset=["数值"])


def _transparent(chart: alt.Chart) -> alt.Chart:
    return chart.properties(background=TRANSPARENT).configure(
        font=ANNUAL_REPORT_FONT,
    ).configure_view(
        fill=TRANSPARENT,
        strokeOpacity=0,
    ).configure_axis(
        domain=True,
        domainColor="#D0D5DD",
        domainWidth=1,
        grid=False,
        labelFont=ANNUAL_REPORT_FONT,
        labelFontSize=10,
        labelColor="#0C233C",
        ticks=True,
        tickColor="#D0D5DD",
        titleFont=ANNUAL_REPORT_FONT,
        titleFontSize=11,
        titleColor="#0C233C",
    ).configure_axisY(
        domain=False,
        ticks=False,
    ).configure_axisX(
        domain=False,
        ticks=False,
        labels=True,
    ).configure_legend(
        labelFont=ANNUAL_REPORT_FONT,
        labelFontSize=10,
        labelColor="#0C233C",
        titleFont=ANNUAL_REPORT_FONT,
        titleFontSize=10,
        titleColor="#0C233C",
        symbolOpacity=1,
    ).configure_header(
        labelColor="#00338D",
        labelFont=ANNUAL_REPORT_FONT,
        labelFontSize=13,
        labelFontWeight="bold",
        labelPadding=10,
        titleFont=ANNUAL_REPORT_FONT,
        titleColor="#0C233C",
    ).configure_title(
        anchor="middle",
        color="#00338D",
        font=ANNUAL_REPORT_FONT,
        fontSize=13,
        fontWeight="bold",
        offset=12,
    )


def _metric_title(frame: pd.DataFrame, code: str) -> str:
    rows = frame[frame["指标编码"].astype(str).eq(code)]
    return code if rows.empty else str(rows.iloc[0].get("指标名称", code))


def _axis_title(frame: pd.DataFrame) -> str:
    units = [str(value).strip() for value in frame.get("单位", pd.Series(dtype=str)).dropna().unique() if str(value).strip()]
    return "数值" if not units else f"数值（{'、'.join(units)}）"


def convert_multiple_units_to_percent(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert ratio multiples to their Step 7 display units without mutating source data."""
    result = frame.copy()
    if "数值" not in result or "单位" not in result:
        return result
    units = result["单位"].fillna("").astype(str).str.strip()
    multiple_mask = units.eq("倍")
    metric_codes = result.get(
        "指标编码", pd.Series("", index=result.index)
    ).fillna("").astype(str).str.strip()
    non_life_liability_mask = metric_codes.eq(
        "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES"
    )
    permille_from_multiple = non_life_liability_mask & multiple_mask
    permille_from_percent = non_life_liability_mask & units.isin({"%", "％"})
    percent_mask = multiple_mask & ~non_life_liability_mask
    if not (percent_mask | permille_from_multiple | permille_from_percent).any():
        return result
    numeric = pd.to_numeric(result["数值"], errors="coerce")
    result["数值"] = numeric
    result.loc[percent_mask, "数值"] = numeric.loc[percent_mask] * 100.0
    result.loc[percent_mask, "单位"] = "%"
    result.loc[permille_from_multiple, "数值"] = (
        numeric.loc[permille_from_multiple] * 1000.0
    )
    result.loc[permille_from_percent, "数值"] = (
        numeric.loc[permille_from_percent] * 10.0
    )
    result.loc[permille_from_multiple | permille_from_percent, "单位"] = "‰"
    return result


def _with_metric_value_labels(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    """Add visible value labels, including a percent sign for percentage rows."""
    result = frame.copy()
    numeric = pd.to_numeric(result.get("数值", pd.Series(dtype=float)), errors="coerce")
    precision = (
        0
        if code == "SOLVENCY_RATIO_COMBO"
        else 5
        if code in COMPACT_TREND_SCALE_CODES
        else 1
        if code in TREND_AXIS_FORMATS
        else 2
    )
    units = result.get("单位", pd.Series("", index=result.index)).fillna("").astype(str).str.strip()
    result["数值标签"] = [
        ""
        if pd.isna(value)
        else f"{float(value):,.{precision}f}{'%' if unit in {'%', '％'} else '‰' if unit == '‰' else ''}"
        for value, unit in zip(numeric, units)
    ]
    return result


def _extreme_value_labels(frame: pd.DataFrame) -> pd.Series:
    """Format max/min point labels with two decimals for percent and per-mille."""
    numeric = pd.to_numeric(frame.get("数值", pd.Series(dtype=float)), errors="coerce")
    units = frame.get("单位", pd.Series("", index=frame.index)).fillna("").astype(str).str.strip()
    return pd.Series(
        [
            ""
            if pd.isna(value)
            else f"{float(value):,.2f}{'%' if unit in {'%', '％'} else '‰' if unit == '‰' else ''}"
            for value, unit in zip(numeric, units)
        ],
        index=frame.index,
    )


def _company_scale(
    companies: Iterable[str],
    colors: Mapping[str, str],
) -> alt.Scale:
    domain = list(dict.fromkeys(str(company) for company in companies))
    palette = [colors.get(company, KPMG_CHART_COLORS[index % len(KPMG_CHART_COLORS)]) for index, company in enumerate(domain)]
    return alt.Scale(domain=domain, range=palette)


def _company_legend_order(
    companies: Iterable[str],
    colors: Mapping[str, str],
) -> list[str]:
    """Use the shared color mapping as the canonical company legend order."""
    available = list(dict.fromkeys(str(company) for company in companies))
    available_set = set(available)
    mapped = [
        str(company)
        for company in colors
        if str(company) in available_set
    ]
    mapped_set = set(mapped)
    return [*mapped, *[company for company in available if company not in mapped_set]]


def _facet_layout(company_count: int) -> tuple[int, int, int]:
    """Keep annual-report-style company facets on one fixed-height row.

    More companies only reduce panel width and horizontal spacing. The chart
    builders retain their normal panel height so dense print pages do not
    become vertically sparse.
    """
    if company_count <= 1:
        return 1, 760, 22
    spacing = (
        5 if company_count >= 10
        else 8 if company_count >= 6
        else 12 if company_count >= 4
        else 22
    )
    available_width = 1120
    panel_width = max(
        52,
        (available_width - spacing * (company_count - 1)) // company_count,
    )
    return company_count, panel_width, spacing


def _panel_density_count(panel_count: int, dense_layout: bool = False) -> int:
    return max(int(panel_count or 1), 10 if dense_layout else 1)


def _panel_typography(
    panel_count: int,
    dense_layout: bool = False,
) -> tuple[int, int, int]:
    """Scale period labels, values, and company headings for one-row panels."""
    count = _panel_density_count(panel_count, dense_layout)
    if count >= 12:
        return 6, 7, 8
    if count >= 8:
        return 7, 8, 10
    if count >= 5:
        return 8, 9, 11
    return 9, 10, 12


def _responsive_bar_fraction(
    panel_count: int,
    dense_layout: bool = False,
) -> float:
    """Return the responsive share occupied by one bar in its period slot."""
    count = _panel_density_count(panel_count, dense_layout)
    return 0.56 if count >= 12 else 0.64 if count >= 8 else 0.68 if count >= 5 else 0.72


def _responsive_bar_width(
    panel_count: int,
    dense_layout: bool = False,
) -> alt.RelativeBandSize:
    """Narrow bars gradually as more company panels share the same row."""
    return alt.RelativeBandSize(
        _responsive_bar_fraction(panel_count, dense_layout)
    )


def _responsive_grouped_bar_width(
    panel_count: int,
    dense_layout: bool = False,
) -> alt.RelativeBandSize:
    """Keep paired bars broad and nearly touching within each report period."""
    count = _panel_density_count(panel_count, dense_layout)
    fraction = 0.8 if count >= 12 else 0.86 if count >= 8 else 0.9 if count >= 5 else 0.94
    return alt.RelativeBandSize(fraction)


def _compact_period_padding(
    panel_count: int = 1,
    dense_layout: bool = False,
) -> float:
    """Return the inner padding shared by all compact period band scales."""
    count = _panel_density_count(panel_count, dense_layout)
    return 0.58 if count >= 12 else 0.54 if count >= 8 else 0.5


def _compact_period_scale(
    panel_count: int = 1,
    dense_layout: bool = False,
) -> alt.Scale:
    """Keep report-period columns closer together inside every chart panel."""
    return alt.Scale(
        paddingInner=_compact_period_padding(panel_count, dense_layout),
        paddingOuter=0.5,
    )


def _hidden_value_axis() -> alt.Axis:
    """Keep the shared y-scale geometry while hiding its numeric tick labels."""
    return alt.Axis(title=None, labels=False, ticks=False)


def _capital_stack_positions(
    rows: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Calculate deterministic stack bounds and label positions per bar."""
    result = rows.copy()
    result["_资本顺序"] = result["资本类别"].map(CAPITAL_COMPONENT_ORDER)
    result = result.sort_values(["公司", "报告期", "_资本顺序"])
    groups = result.groupby(["公司", "报告期"], sort=False)[value_column]
    result["堆叠终点"] = groups.cumsum()
    result["堆叠起点"] = result["堆叠终点"] - result[value_column]
    result["标签位置"] = (result["堆叠起点"] + result["堆叠终点"]) / 2
    return result


def _signed_stack_positions(
    rows: pd.DataFrame,
    value_column: str,
    order_column: str,
) -> pd.DataFrame:
    """Calculate stack bounds while keeping negative effects below zero."""
    result = rows.sort_values(["公司", "报告期", order_column]).copy()
    values = pd.to_numeric(result[value_column], errors="coerce")
    result["_正值"] = values.clip(lower=0)
    result["_负值"] = values.clip(upper=0)
    groups = result.groupby(["公司", "报告期"], sort=False)
    positive_end = groups["_正值"].cumsum()
    negative_end = groups["_负值"].cumsum()
    positive_start = positive_end - result["_正值"]
    negative_start = negative_end - result["_负值"]
    positive = values.ge(0)
    result["堆叠终点"] = positive_end.where(positive, negative_end)
    result["堆叠起点"] = positive_start.where(positive, negative_start)
    result["标签位置"] = (result["堆叠起点"] + result["堆叠终点"]) / 2
    totals = groups[value_column].transform(lambda series: series.abs().sum())
    result["构成占比"] = values.abs() / totals.where(totals.ne(0))
    result["占比标签"] = result["构成占比"].map(
        lambda value: (
            ""
            if pd.isna(value) or value < MIN_INSIDE_LABEL_SHARE
            else _whole_percent_label(value)
        )
    )
    return result.drop(columns=["_正值", "_负值"])


def _metric_rows(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
) -> tuple[pd.DataFrame, list[str], str]:
    metric = convert_multiple_units_to_percent(
        _clean_numeric(frame[frame["指标编码"].astype(str).eq(code)])
    )
    periods = list(period_order)
    if metric.empty:
        raise ValueError(f"指标 {code} 没有可绘制数据。")
    metric = metric.drop_duplicates(
        subset=["公司", "报告期", "指标编码"],
        keep="last",
    ).copy()
    companies = list(dict.fromkeys(frame["公司"].dropna().astype(str)))
    grid = pd.MultiIndex.from_product(
        [companies, periods],
        names=["公司", "报告期"],
    ).to_frame(index=False)
    metric = grid.merge(metric, on=["公司", "报告期"], how="left", sort=False)
    metric["指标编码"] = metric["指标编码"].fillna(code)
    metric_name = _metric_title(frame, code)
    metric["指标名称"] = metric["指标名称"].fillna(metric_name)
    if "单位" in metric.columns:
        disclosed_units = metric["单位"].dropna()
        default_unit = "" if disclosed_units.empty else disclosed_units.iloc[0]
        metric["单位"] = metric["单位"].fillna(default_unit)
    metric["披露状态"] = metric["数值"].notna().map({True: "已披露", False: "未披露"})
    metric = _with_metric_value_labels(metric, code)
    return metric, periods, _metric_title(metric, code)


def _trend_y_domain(metric: pd.DataFrame, code: str) -> list[float]:
    values = pd.to_numeric(metric["数值"], errors="coerce").dropna().tolist()
    limit = REGULATORY_LIMITS.get(code)
    if limit is not None:
        values.append(float(limit))
    if not values:
        return [0.0, 1.0]
    low, high = min(values), max(values)
    span = high - low
    minimum_padding = 0.0000001 if code in COMPACT_TREND_SCALE_CODES else 0.01
    span_padding = 0.14 if code in COMPACT_POLICY_RATIO_CODES else 0.08
    magnitude_padding = 0.06 if code in COMPACT_POLICY_RATIO_CODES else 0.025
    padding = max(
        span * span_padding,
        max(abs(low), abs(high)) * magnitude_padding,
        minimum_padding,
    )
    return [low - padding, high + padding]


def _build_metric_trend_group(
    metric: pd.DataFrame,
    periods: list[str],
    name: str,
    code: str,
    company_colors: Mapping[str, str],
    highlight_company: str,
    y_domain: list[float],
    group_index: int = 1,
    group_count: int = 1,
) -> alt.Chart:
    companies = _company_legend_order(metric["公司"].astype(str), company_colors)
    scale = _company_scale(companies, company_colors)
    label_format = TREND_LABEL_FORMATS.get(code, ",.2f")
    if code in COMPACT_TREND_SCALE_CODES:
        y_axis = alt.Axis(format=".5f", tickCount=6)
    elif code in TREND_AXIS_FORMATS:
        y_axis = alt.Axis(format=TREND_AXIS_FORMATS[code])
    else:
        y_axis = alt.Axis()
    company_legend = alt.Legend(
        title="公司",
        orient="top-right",
        direction="vertical",
        columns=1,
        symbolSize=80,
    )
    highlight = str(highlight_company or "").strip()
    line_size = (
        alt.condition(alt.datum["公司"] == highlight, alt.value(4.0), alt.value(1.8))
        if highlight in companies else alt.value(2.0)
    )
    line_opacity = (
        alt.condition(alt.datum["公司"] == highlight, alt.value(1.0), alt.value(0.68))
        if highlight in companies else alt.value(0.72)
    )
    value_tooltip = alt.Tooltip("数值:Q", format=label_format)
    base = alt.Chart(metric)
    lines = base.mark_line().encode(
        x=alt.X("报告期:N", sort=periods, title="报告期"),
        y=alt.Y(
            "数值:Q",
            title=_axis_title(metric),
            axis=y_axis,
            scale=alt.Scale(domain=y_domain, zero=False, nice=False),
        ),
        color=alt.Color("公司:N", scale=scale, legend=company_legend),
        detail="公司:N",
        size=line_size,
        opacity=line_opacity,
        tooltip=["公司:N", "报告期:N", "指标名称:N", "披露状态:N", value_tooltip, "单位:N"],
    )
    point_size = (
        alt.condition(alt.datum["公司"] == highlight, alt.value(88), alt.value(58))
        if highlight in companies else alt.value(58)
    )
    points = base.mark_point(filled=True, size=58).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        color=alt.Color("公司:N", scale=scale, legend=None),
        size=point_size,
        opacity=line_opacity,
        tooltip=["公司:N", "报告期:N", "指标名称:N", "披露状态:N", value_tooltip, "单位:N"],
    )
    highlight_labels = base.transform_filter(
        alt.datum["显示标签"] & ~alt.datum["是否期间最大"] & ~alt.datum["是否期间最小"]
    ).mark_text(tooltip=False, dy=-10, fontSize=10, fontWeight="bold", color="#0C233C").encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        text=alt.Text("数值标签:N"),
    )
    max_labels = base.transform_filter(alt.datum["是否期间最大"]).mark_text(
        tooltip=False, dy=-11, fontSize=10, fontWeight="bold", color="#0C233C",
    ).encode(x=alt.X("报告期:N", sort=periods), y="数值:Q", text=alt.Text("极值标签:N"))
    min_labels = base.transform_filter(alt.datum["是否期间最小"]).mark_text(
        tooltip=False, dy=11, baseline="top", fontSize=10, fontWeight="bold", color="#0C233C",
    ).encode(x=alt.X("报告期:N", sort=periods), y="数值:Q", text=alt.Text("极值标签:N"))
    trend_layers: list[alt.Chart] = [lines, points, highlight_labels, max_labels, min_labels]
    limit = REGULATORY_LIMITS.get(code)
    if limit is not None:
        rule_data = pd.DataFrame({"监管下限": [limit]})
        trend_layers.append(
            alt.Chart(rule_data).mark_rule(color="#ED2124", strokeDash=[5, 4], strokeWidth=2).encode(
                y="监管下限:Q",
                tooltip=[alt.Tooltip("监管下限:Q", title="监管下限")],
            )
        )
    title = f"{name}跨期趋势"
    height = 420
    if group_count > 1:
        title = f"{title} · 第 {group_index}/{group_count} 组（{len(companies)} 家公司）"
        height = 350
    return _transparent(
        alt.layer(*trend_layers)
        .properties(title=title, height=height)
    )


def _prepare_metric_trend(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
    highlight_company: str,
) -> tuple[pd.DataFrame, list[str], str, list[float]]:
    metric, periods, name = _metric_rows(frame, code, period_order)
    highlight = str(highlight_company or "").strip()
    metric["是否追踪"] = metric["公司"].astype(str).eq(highlight)
    metric["是否期间最大"] = metric.groupby("报告期")["数值"].transform(
        lambda series: series.eq(series.max())
    )
    metric["是否期间最小"] = metric.groupby("报告期")["数值"].transform(
        lambda series: series.eq(series.min())
    )
    metric["显示标签"] = metric[["是否追踪", "是否期间最大", "是否期间最小"]].any(axis=1)
    metric["极值标签"] = _extreme_value_labels(metric)
    return metric, periods, name, _trend_y_domain(metric, code)


def build_single_metric_trend_chart(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
    company_colors: Mapping[str, str],
    highlight_company: str = "",
) -> alt.Chart:
    """Return one point-line trend chart (kept for small selections and callers)."""
    metric, periods, name, y_domain = _prepare_metric_trend(
        frame, code, period_order, highlight_company
    )
    return _build_metric_trend_group(
        metric,
        periods,
        name,
        code,
        company_colors,
        highlight_company,
        y_domain,
    )


def build_single_metric_trend_charts(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
    company_colors: Mapping[str, str],
    highlight_company: str = "",
) -> list[alt.Chart]:
    """Return one trend chart containing every selected company."""
    return [
        build_single_metric_trend_chart(
            frame,
            code,
            period_order,
            company_colors,
            highlight_company,
        )
    ]


def build_company_bar_trend_chart(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
    highlight_company: str = "",
) -> alt.Chart:
    """Return one compact bar + point-line panel per company."""
    metric, periods, name = _metric_rows(frame, code, period_order)
    highlight = str(highlight_company or "").strip()
    metric["是否追踪"] = metric["公司"].astype(str).eq(highlight)
    period_colors = report_period_color_map(periods)
    company_count = metric["公司"].nunique()
    columns, panel_width, facet_spacing = _facet_layout(company_count)
    panel_height = 270 if company_count == 1 else 285
    dense_layout = company_count >= 10
    period_font_size, value_font_size, company_font_size = _panel_typography(
        company_count,
        dense_layout,
    )
    period_scale = _compact_period_scale(company_count, dense_layout)
    shared_domain = metric_bar_axis_domain(metric, code, periods)
    y_scale = alt.Scale(domain=list(shared_domain), zero=True, nice=False)
    base = alt.Chart()
    bars = base.mark_bar(
        width=_responsive_bar_width(company_count, dense_layout),
        opacity=0.92,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            title=None,
            axis=alt.Axis(title=None, labelAngle=0, labelFontSize=period_font_size, labelLimit=55),
            scale=period_scale,
        ),
        y=alt.Y(
            "数值:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
        ),
        color=alt.Color(
            "报告期:N",
            scale=alt.Scale(
                domain=periods,
                range=[period_colors[period] for period in periods],
            ),
            legend=None,
        ),
        tooltip=["公司:N", "报告期:N", "指标名称:N", "披露状态:N", alt.Tooltip("数值:Q", format=",.2f"), "单位:N"],
    )
    line = base.mark_line(color="#0C233C", strokeWidth=2.6).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("数值:Q", scale=y_scale),
    )
    points = base.mark_point(
        filled=True, size=62, color="#0C233C",
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("数值:Q", scale=y_scale),
    )
    labels = base.mark_text(
        tooltip=False, dy=-12, fontSize=value_font_size, fontWeight="bold", color="#0C233C",
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("数值:Q", scale=y_scale),
        text=alt.Text("数值标签:N"),
    )
    layers: list[alt.Chart] = [bars, line, points, labels]
    limit = REGULATORY_LIMITS.get(code)
    if limit is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"监管下限": [limit]}))
            .mark_rule(color="#ED2124", strokeDash=[5, 4], strokeWidth=1.2)
            .encode(y=alt.Y("监管下限:Q", scale=y_scale))
        )
    tracked_frame = (
        base.transform_filter(alt.datum["是否追踪"])
        .transform_aggregate(追踪记录="count()", groupby=["公司"])
        .mark_rect(
            fill="#00338D",
            fillOpacity=0.03,
            stroke="#00338D",
            strokeOpacity=0.35,
            strokeWidth=1.5,
            cornerRadius=5,
        )
        .encode(
            x=alt.value(-13),
            x2=alt.value(panel_width + 13),
            y=alt.value(-18),
            y2=alt.value(panel_height + 13),
        )
    )
    layers.append(tracked_frame)
    chart = alt.layer(*layers, data=metric).properties(width=panel_width, height=panel_height)
    return _transparent(
        chart.facet(
            facet=alt.Facet(
                "公司:N",
                title="公司",
                header=alt.Header(labelFontSize=company_font_size, labelPadding=20),
            ),
            columns=columns,
            spacing=facet_spacing,
        ).resolve_scale(y="shared").properties(title=f"{name}公司跨期对标")
    )


def build_company_period_bar_chart(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
    *,
    shared_y_domain: tuple[float, float] | None = None,
    dense_layout: bool = False,
    panel_count: int = 1,
) -> alt.Chart:
    """Return one company's metric bars with report period on x."""
    metric, periods, name = _metric_rows(frame, code, period_order)
    metric = metric.dropna(subset=["数值"]).copy()
    if metric.empty:
        raise ValueError(f"{name}缺少可绘制的数据。")
    companies = list(dict.fromkeys(metric["公司"].dropna().astype(str)))
    if len(companies) != 1:
        raise ValueError(f"{name}需要按公司分别绘制。")
    company = companies[0]
    period_font_size, value_font_size, _ = _panel_typography(
        panel_count,
        dense_layout,
    )
    period_scale = _compact_period_scale(panel_count, dense_layout)
    period_colors = report_period_color_map(periods)
    value_format = TREND_LABEL_FORMATS.get(code, ",.2f")
    y_scale = (
        alt.Scale(domain=list(shared_y_domain), zero=True, nice=False)
        if shared_y_domain is not None
        else alt.Scale(zero=True)
    )
    bars = alt.Chart(metric).mark_bar(
        width=_responsive_bar_width(panel_count, dense_layout),
        opacity=0.94,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X(
            "报告期:N",
            title=None,
            sort=periods,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelFontSize=period_font_size,
                labelLimit=55,
            ),
            scale=period_scale,
        ),
        y=alt.Y(
            "数值:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
        ),
        color=alt.Color(
            "报告期:N",
            scale=alt.Scale(domain=periods, range=[period_colors[period] for period in periods]),
            legend=None,
        ),
        tooltip=["公司:N", "报告期:N", "指标名称:N", alt.Tooltip("数值:Q", format=value_format), "单位:N"],
    )
    labels = alt.Chart(metric).mark_text(
        tooltip=False,
        dy=-7,
        fontSize=value_font_size,
        fontWeight="bold",
        color="#0C233C",
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("数值:Q", scale=y_scale),
        text=alt.Text("数值标签:N"),
    )
    layers: list[alt.Chart] = [bars, labels]
    limit = REGULATORY_LIMITS.get(code)
    if limit is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"监管下限": [limit]}))
            .mark_rule(color="#ED2124", strokeDash=[5, 4], strokeWidth=2)
            .encode(y="监管下限:Q")
        )
    return _transparent(
        alt.layer(*layers).properties(title=company, height=250)
    )


def metric_bar_axis_domain(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
) -> tuple[float, float]:
    """Return one padded bar domain shared by every displayed company."""
    periods = {str(period) for period in period_order}
    rows = convert_multiple_units_to_percent(frame[
        frame["指标编码"].astype(str).eq(str(code))
        & frame["报告期"].astype(str).isin(periods)
    ])
    values = pd.to_numeric(rows.get("数值", pd.Series(dtype=float)), errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0
    lower = min(0.0, float(values.min()))
    upper = max(0.0, float(values.max()))
    padded_lower = lower * 1.08 if lower < 0 else 0.0
    padded_upper = upper * 1.12 if upper > 0 else 1.0
    return float(padded_lower), float(padded_upper)


def solvency_ratio_axis_domain(
    frame: pd.DataFrame,
    period_order: Iterable[str],
) -> tuple[float, float]:
    """Return one padded percentage domain shared by all displayed companies."""
    periods = {str(period) for period in period_order}
    rows = convert_multiple_units_to_percent(frame[
        frame["指标编码"].astype(str).isin(
            ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO")
        )
        & frame["报告期"].astype(str).isin(periods)
    ]).copy()
    values = pd.to_numeric(rows.get("数值", pd.Series(dtype=float)), errors="coerce")
    maximum = max(100.0, float(values.max())) if values.notna().any() else 100.0
    upper = max(125.0, math.ceil(maximum * 1.12 / 25.0) * 25.0)
    return 0.0, float(upper)


def _with_solvency_combo_label_offsets(
    core: pd.DataFrame,
    combined: pd.DataFrame,
    domain: tuple[float, float],
    value_font_size: int,
) -> pd.DataFrame:
    """Place a line label below its point when its bar label is too close."""
    combined_values = combined[["公司", "报告期", "数值"]].rename(
        columns={"数值": "综合充足率数值"}
    ).drop_duplicates(["公司", "报告期"], keep="last")
    result = core.merge(
        combined_values,
        on=["公司", "报告期"],
        how="left",
        validate="many_to_one",
    )
    domain_span = max(float(domain[1]) - float(domain[0]), 1.0)
    label_gap_pixels = max(18.0, float(value_font_size) * 2.2 + 4.0)
    value_gap_pixels = (
        (result["综合充足率数值"] - result["数值"]).abs()
        / domain_span
        * COMPANY_PANEL_HEIGHT
    )
    result["label_offset"] = value_gap_pixels.lt(label_gap_pixels).map(
        {True: value_font_size + 5, False: -10}
    )
    return result


def build_solvency_ratio_combo_chart(
    frame: pd.DataFrame,
    period_order: Iterable[str],
    *,
    shared_y_domain: tuple[float, float] | None = None,
    dense_layout: bool = False,
    panel_count: int = 1,
) -> alt.Chart:
    """Return one company panel with combined-ratio bars and a core-ratio line."""
    periods = list(dict.fromkeys(str(period) for period in period_order))
    codes = ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO")
    metric = convert_multiple_units_to_percent(
        _clean_numeric(frame[frame["指标编码"].astype(str).isin(codes)])
    )
    metric = _with_metric_value_labels(metric, "SOLVENCY_RATIO_COMBO")
    metric = metric[metric["报告期"].astype(str).isin(periods)].copy()
    companies = list(dict.fromkeys(metric["公司"].dropna().astype(str)))
    if len(companies) != 1:
        raise ValueError("核心及综合充足率需要按公司分别绘制。")
    company = companies[0]
    combined = metric[
        metric["指标编码"].astype(str).eq("COMBINED_SOLVENCY_RATIO")
    ].copy()
    core = metric[
        metric["指标编码"].astype(str).eq("CORE_SOLVENCY_RATIO")
    ].copy()
    if combined.empty or core.empty:
        raise ValueError(f"{company}缺少核心或综合偿付能力充足率数据。")

    domain = shared_y_domain or solvency_ratio_axis_domain(metric, periods)
    y_scale = alt.Scale(domain=list(domain), zero=True, nice=False)
    period_font_size, value_font_size, _ = _panel_typography(
        panel_count,
        dense_layout,
    )
    core = _with_solvency_combo_label_offsets(
        core,
        combined,
        domain,
        value_font_size,
    )
    period_scale = _compact_period_scale(panel_count, dense_layout)
    period_colors = report_period_combo_bar_color_map(periods)
    x = alt.X(
        "报告期:N",
        title=None,
        sort=periods,
        axis=alt.Axis(
            title=None,
            labelAngle=0,
            labelFontSize=period_font_size,
            labelLimit=55,
        ),
        scale=period_scale,
    )
    value_tooltip = alt.Tooltip("数值:Q", format=",.0f")

    bars = alt.Chart(combined).mark_bar(
        width=_responsive_bar_width(panel_count, dense_layout),
        opacity=0.9,
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=x,
        y=alt.Y(
            "数值:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
        ),
        color=alt.Color(
            "报告期:N",
            scale=alt.Scale(
                domain=periods,
                range=[period_colors[period] for period in periods],
            ),
            legend=None,
        ),
        tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
    )
    bar_labels = alt.Chart(combined).mark_text(
        tooltip=False,
        dy=-7,
        fontSize=value_font_size,
        fontWeight="bold",
        color="#000000",
    ).encode(
        x=x,
        y=alt.Y("数值:Q", scale=y_scale),
        text=alt.Text("数值标签:N"),
    )
    line = alt.Chart(core).mark_line(
        color="#FD349C",
        strokeWidth=2.6,
    ).encode(
        x=x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
    )
    points = alt.Chart(core).mark_point(
        color="#FD349C",
        filled=True,
        size=54,
    ).encode(
        x=x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
    )
    line_labels = alt.Chart(core).mark_text(
        tooltip=False,
        dy=alt.ExprRef(expr="datum.label_offset"),
        fontSize=value_font_size,
        fontWeight="bold",
        color="#000000",
    ).encode(
        x=x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        text=alt.Text("数值标签:N"),
    )
    return _transparent(
        alt.layer(
            bars,
            bar_labels,
            line,
            points,
            line_labels,
        ).properties(title=company, height=COMPANY_PANEL_HEIGHT)
    )


def build_solvency_ratio_combo_overview_chart(
    frame: pd.DataFrame,
    period_order: Iterable[str],
    *,
    company_order: Iterable[str] | None = None,
    shared_y_domain: tuple[float, float] | None = None,
    highlight_company: str = "无",
) -> alt.Chart:
    """Build all company solvency panels as one responsive Vega-Lite view.

    The former UI rendered one independent chart per company.  Apart from
    repeating the same pandas work, that forced the browser to initialise many
    Vega runtimes during every Streamlit rerun.  This grouped chart keeps the
    same period bars, core-ratio line, shared axis, labels, panel backgrounds,
    and company ordering while sending only one chart specification.
    """
    periods = list(dict.fromkeys(str(period) for period in period_order))
    codes = ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO")
    metric = convert_multiple_units_to_percent(
        _clean_numeric(frame[frame["指标编码"].astype(str).isin(codes)])
    )
    metric = _with_metric_value_labels(metric, "SOLVENCY_RATIO_COMBO")
    metric = metric[metric["报告期"].astype(str).isin(periods)].copy()
    discovered_companies = list(dict.fromkeys(metric["公司"].dropna().astype(str)))
    requested_companies = (
        list(dict.fromkeys(str(company) for company in company_order))
        if company_order is not None
        else discovered_companies
    )
    companies = [company for company in requested_companies if company in discovered_companies]
    if not companies or metric.empty:
        raise ValueError("核心及综合充足率缺少可绘制的数据。")
    metric = metric[metric["公司"].astype(str).isin(companies)].copy()
    metric["_报告期顺序"] = metric["报告期"].astype(str).map(
        {period: index for index, period in enumerate(periods)}
    )

    domain = shared_y_domain or solvency_ratio_axis_domain(metric, periods)
    overview_height = COMPANY_PANEL_HEIGHT + 90
    plot_top = 72
    plot_bottom = overview_height - 42
    y_scale = alt.Scale(
        domain=list(domain),
        range=[plot_bottom, plot_top],
        zero=True,
        nice=False,
    )
    panel_count = len(companies)
    dense_layout = panel_count >= 10
    period_font_size, value_font_size, company_font_size = _panel_typography(
        panel_count,
        dense_layout,
    )
    company_positions = {company: index for index, company in enumerate(companies)}
    period_positions = {period: index for index, period in enumerate(periods)}
    period_padding_inner = _compact_period_padding(panel_count, dense_layout)
    period_padding_outer = 0.5
    period_step = 1.0 / (
        len(periods)
        - period_padding_inner
        + 2 * period_padding_outer
    )
    period_band_width = period_step * (1 - period_padding_inner)
    metric["_面板序号"] = metric["公司"].astype(str).map(company_positions)
    metric["_横轴中心"] = (
        metric["_面板序号"]
        + period_step * period_padding_outer
        + metric["报告期"].astype(str).map(period_positions) * period_step
        + period_band_width / 2
    )
    bar_width = period_band_width * _responsive_bar_fraction(
        panel_count,
        dense_layout,
    )
    metric["_柱左"] = metric["_横轴中心"] - bar_width / 2
    metric["_柱右"] = metric["_横轴中心"] + bar_width / 2
    combined = metric[
        metric["指标编码"].astype(str).eq("COMBINED_SOLVENCY_RATIO")
    ].copy()
    core = metric[
        metric["指标编码"].astype(str).eq("CORE_SOLVENCY_RATIO")
    ].copy()
    core_offsets = _with_solvency_combo_label_offsets(
        core,
        combined,
        domain,
        value_font_size,
    )[["公司", "报告期", "label_offset"]]
    metric = metric.merge(
        core_offsets,
        on=["公司", "报告期"],
        how="left",
        validate="many_to_one",
    )

    panel_rows = pd.DataFrame(
        {
            "公司": companies,
            "面板序号": range(len(companies)),
            "_面板左": [index + 0.04 for index in range(len(companies))],
            "_面板右": [index + 0.96 for index in range(len(companies))],
            "_面板中心": [index + 0.5 for index in range(len(companies))],
        }
    )
    period_colors = report_period_combo_bar_color_map(periods)
    horizontal_scale = alt.Scale(
        domain=[0.0, float(panel_count)],
        zero=False,
        nice=False,
    )
    center_x = alt.X(
        "_横轴中心:Q",
        title=None,
        scale=horizontal_scale,
        axis=None,
    )
    value_tooltip = alt.Tooltip("数值:Q", format=",.0f")

    backgrounds = alt.Chart(panel_rows).mark_rect(
        stroke=ANNUAL_REPORT_PANEL_BORDER,
        strokeWidth=1,
        tooltip=False,
    ).encode(
        x=alt.X("_面板左:Q", title=None, scale=horizontal_scale, axis=None),
        x2=alt.X2("_面板右:Q"),
        y=alt.value(0),
        y2=alt.value(overview_height),
        color=alt.condition(
            "datum['面板序号'] % 2 === 1",
            alt.value("rgba(200,200,200,0.12)"),
            alt.value(TRANSPARENT),
        ),
    )
    company_titles = alt.Chart(panel_rows).mark_text(
        tooltip=False,
        baseline="top",
        fontSize=company_font_size,
        fontWeight="bold",
        color="#00338D",
    ).encode(
        x=alt.X("_面板中心:Q", title=None, scale=horizontal_scale, axis=None),
        y=alt.value(18),
        text=alt.Text("公司:N"),
    )
    bars = (
        alt.Chart(metric)
        .transform_filter(alt.datum["指标编码"] == "COMBINED_SOLVENCY_RATIO")
        .mark_bar(
            orient="vertical",
            opacity=0.9,
            cornerRadiusTopLeft=2,
            cornerRadiusTopRight=2,
        )
        .encode(
            x=alt.X("_柱左:Q", title=None, scale=horizontal_scale, axis=None),
            x2=alt.X2("_柱右:Q"),
            y=alt.Y("数值:Q", title=None, axis=_hidden_value_axis(), scale=y_scale),
            y2=alt.Y2(datum=0),
            color=alt.Color(
                "报告期:N",
                scale=alt.Scale(
                    domain=periods,
                    range=[period_colors[period] for period in periods],
                ),
                legend=None,
            ),
            tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
        )
    )
    bar_labels = (
        alt.Chart(metric)
        .transform_filter(alt.datum["指标编码"] == "COMBINED_SOLVENCY_RATIO")
        .mark_text(
            tooltip=False,
            dy=-7,
            fontSize=value_font_size,
            fontWeight="bold",
            color="#000000",
        )
        .encode(
            x=center_x,
            y=alt.Y("数值:Q", scale=y_scale),
            text=alt.Text("数值标签:N"),
        )
    )
    core_rows = alt.Chart(metric).transform_filter(
        alt.datum["指标编码"] == "CORE_SOLVENCY_RATIO"
    )
    line = core_rows.mark_line(
        color="#FD349C",
        strokeWidth=2.6,
    ).encode(
        x=center_x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        detail=alt.Detail("公司:N"),
        order=alt.Order("_报告期顺序:Q", sort="ascending"),
        tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
    )
    points = core_rows.mark_point(
        color="#FD349C",
        filled=True,
        size=54,
    ).encode(
        x=center_x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        tooltip=["公司:N", "报告期:N", "指标名称:N", value_tooltip, "单位:N"],
    )
    line_labels = core_rows.mark_text(
        tooltip=False,
        dy=alt.ExprRef(expr="datum.label_offset"),
        fontSize=value_font_size,
        fontWeight="bold",
        color="#000000",
    ).encode(
        x=center_x,
        y=alt.Y("数值:Q", axis=None, scale=y_scale),
        text=alt.Text("数值标签:N"),
    )
    period_labels = (
        alt.Chart(metric)
        .transform_filter(alt.datum["指标编码"] == "COMBINED_SOLVENCY_RATIO")
        .mark_text(
            tooltip=False,
            dy=15,
            baseline="top",
            fontSize=period_font_size,
            color="#0C233C",
        )
        .encode(
            x=center_x,
            y=alt.value(plot_bottom),
            text=alt.Text("报告期:N"),
        )
    )
    return _transparent(
        alt.layer(
            backgrounds,
            company_titles,
            bars,
            bar_labels,
            line,
            points,
            line_labels,
            period_labels,
        ).properties(
            height=overview_height,
            padding={"top": 0, "bottom": 0, "left": 0, "right": 0},
        )
    )


def build_capital_ratio_combo_chart(
    frame: pd.DataFrame,
    amount_groups: Mapping[str, tuple[str, ...]],
    period_order: Iterable[str],
    title: str,
    *,
    ratio_code: str = "",
    ratio_label: str = "",
    share_line: bool = False,
    shared_amount_domain: tuple[float, float] | None = None,
    dense_layout: bool = False,
    panel_count: int = 1,
    bar_layout: str = "stacked",
) -> alt.Chart:
    """Build one company panel: ratio line above stacked or grouped capital bars."""
    if bar_layout not in {"stacked", "grouped"}:
        raise ValueError("bar_layout must be 'stacked' or 'grouped'.")
    periods = list(period_order)
    source_codes = tuple(code for codes in amount_groups.values() for code in codes)
    source = _clean_numeric(frame[frame["指标编码"].astype(str).isin(source_codes)])
    pivot = source.pivot_table(
        index=["公司", "报告期"],
        columns="指标编码",
        values="数值",
        aggfunc="first",
    ).reset_index()
    component_rows: list[dict[str, object]] = []
    unit_values = [str(value).strip() for value in source.get("单位", pd.Series(dtype=str)).dropna() if str(value).strip()]
    amount_unit = unit_values[0] if unit_values else ""
    stack_labels = list(amount_groups)
    if "最低资本" in stack_labels:
        stack_labels = ["最低资本", *[label for label in stack_labels if label != "最低资本"]]
    stack_order = {label: index for index, label in enumerate(stack_labels)}
    for row in pivot.to_dict("records"):
        for label, codes in amount_groups.items():
            values = [pd.to_numeric(row.get(code), errors="coerce") for code in codes]
            if any(pd.isna(value) for value in values):
                continue
            component_rows.append({
                "公司": row["公司"],
                "报告期": row["报告期"],
                "记录类型": "资本构成",
                "组成类别": label,
                "组成顺序": stack_order[label],
                "数值": float(sum(values)),
                "单位": amount_unit,
            })
    components = pd.DataFrame(component_rows)
    if components.empty:
        raise ValueError(f"{title}缺少可绘制的资本金额。")
    companies = list(dict.fromkeys(components["公司"].dropna().astype(str)))
    if len(companies) != 1:
        raise ValueError(f"{title}需要按公司分别绘制。")
    company = companies[0]
    period_font_size, value_font_size, _ = _panel_typography(
        panel_count,
        dense_layout,
    )
    components = _signed_stack_positions(components, "数值", "组成顺序")
    if bar_layout == "grouped":
        components["标签位置"] = pd.to_numeric(components["数值"], errors="coerce") / 2
    components["数值标签"] = components.apply(
        lambda row: (
            ""
            if pd.isna(row["数值"])
            or (
                bar_layout == "stacked"
                and (
                    pd.isna(row["构成占比"])
                    or abs(float(row["构成占比"])) < MIN_INSIDE_LABEL_SHARE
                )
            )
            else _whole_number_label(row["数值"])
        ),
        axis=1,
    )

    if share_line:
        wide = components.pivot_table(
            index=["公司", "报告期"], columns="组成类别", values="数值", aggfunc="first"
        ).reset_index()
        first_label = next(iter(amount_groups))
        total = wide[list(amount_groups)].sum(axis=1, min_count=len(amount_groups))
        wide["线值"] = wide[first_label] / total.where(total.ne(0)) * 100
        line_rows = wide[["公司", "报告期", "线值"]].dropna().copy()
    else:
        line_rows = convert_multiple_units_to_percent(
            _clean_numeric(
                frame[frame["指标编码"].astype(str).eq(ratio_code)]
            )
        )[["公司", "报告期", "数值"]].drop_duplicates(
            ["公司", "报告期"], keep="last"
        )
        line_rows = line_rows.rename(columns={"数值": "线值"})
    line_rows["线标签"] = line_rows["线值"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):,.1f}%"
    )
    colors = [
        KPMG_CAPITAL_COMBO_COLORS[index % len(KPMG_CAPITAL_COMBO_COLORS)]
        for index in range(len(amount_groups))
    ]
    labels = list(amount_groups)
    label_color_map = {
        label: _label_color_for_fill(color)
        for label, color in zip(labels, colors)
    }
    components["标签颜色"] = components["组成类别"].map(label_color_map).fillna("#0C233C")
    period_axis = alt.Axis(
        title=None,
        domain=False,
        ticks=False,
        labelAngle=0,
        labelPadding=8,
        labelColor="#7A8496",
        labelFontSize=period_font_size,
        labelLimit=80,
    )
    amount_axis = alt.Axis(
        title=None,
        domain=False,
        ticks=False,
        labels=False,
        grid=False,
    )
    ratio_axis = alt.Axis(
        title=None,
        domain=False,
        ticks=False,
        labels=False,
        grid=False,
    )
    amount_domain = shared_amount_domain or capital_ratio_amount_axis_domain(
        frame,
        amount_groups,
        periods,
        stacked=bar_layout == "stacked",
    )
    amount_scale = alt.Scale(domain=list(amount_domain), zero=True, nice=False)
    combo_period_scale = _compact_period_scale(panel_count, dense_layout)
    period_x = alt.X(
        "报告期:N",
        sort=periods,
        axis=period_axis,
        scale=combo_period_scale,
    )
    category_offset = alt.XOffset(
        "组成类别:N",
        sort=labels,
        scale=alt.Scale(paddingInner=0.0, paddingOuter=0.04),
    )
    bar_encoding: dict[str, object] = {
        "x": period_x,
        "y": alt.Y(
            "堆叠终点:Q" if bar_layout == "stacked" else "数值:Q",
            axis=amount_axis,
            scale=amount_scale,
            stack=None,
        ),
        "color": alt.Color(
            "组成类别:N",
            scale=alt.Scale(domain=labels, range=colors),
            legend=None,
        ),
        "tooltip": [
            "公司:N",
            "报告期:N",
            "组成类别:N",
            alt.Tooltip("数值:Q", format=",.2f"),
            "单位:N",
        ],
    }
    if bar_layout == "stacked":
        bar_encoding["y2"] = alt.Y2("堆叠起点:Q")
    else:
        bar_encoding["xOffset"] = category_offset
    bars = alt.Chart(components).mark_bar(
        width=(
            _responsive_grouped_bar_width(panel_count, dense_layout)
            if bar_layout == "grouped"
            else _responsive_bar_width(panel_count, dense_layout)
        )
    ).encode(**bar_encoding)
    label_encoding: dict[str, object] = {
        "x": alt.X("报告期:N", sort=periods, scale=combo_period_scale),
        "y": alt.Y("标签位置:Q", axis=amount_axis, scale=amount_scale),
        "text": "数值标签:N",
        "color": alt.Color("标签颜色:N", scale=None, legend=None),
    }
    if bar_layout == "grouped":
        label_encoding["xOffset"] = category_offset
    inside_labels = alt.Chart(components).mark_text(
        tooltip=False,
        fontSize=value_font_size,
        fontWeight="bold",
        baseline="middle",
    ).encode(**label_encoding)
    bar_panel = alt.layer(bars, inside_labels).properties(height=190)
    if line_rows.empty:
        return _transparent(bar_panel.properties(title=company))

    line = alt.Chart(line_rows).mark_line(
        color="#FD349C",
        strokeWidth=2.6,
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            axis=None,
            scale=combo_period_scale,
        ),
        y=alt.Y(
            "线值:Q",
            axis=ratio_axis,
            scale=alt.Scale(zero=False, padding=18),
        ),
        tooltip=["公司:N", "报告期:N", alt.Tooltip("线值:Q", title=ratio_label or "比例", format=",.2f")],
    )
    points = alt.Chart(line_rows).mark_point(
        color="#FD349C",
        filled=True,
        shape="triangle-up",
        size=68,
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            axis=None,
            scale=combo_period_scale,
        ),
        y=alt.Y("线值:Q", axis=ratio_axis, scale=alt.Scale(zero=False, padding=18)),
        tooltip=["公司:N", "报告期:N", alt.Tooltip("线值:Q", title=ratio_label or "比例", format=",.2f")],
    )
    line_labels = alt.Chart(line_rows).mark_text(
        tooltip=False,
        dy=-14,
        color="#FD349C",
        fontSize=value_font_size,
        fontWeight="bold",
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            axis=None,
            scale=combo_period_scale,
        ),
        y=alt.Y("线值:Q", axis=ratio_axis, scale=alt.Scale(zero=False, padding=18)),
        text="线标签:N",
    )
    line_panel = alt.layer(line, points, line_labels).properties(height=72)
    return _transparent(
        alt.vconcat(line_panel, bar_panel, spacing=4)
        .resolve_scale(x="shared")
        .properties(title=company)
    )


def capital_ratio_amount_axis_domain(
    frame: pd.DataFrame,
    amount_groups: Mapping[str, tuple[str, ...]],
    period_order: Iterable[str],
    *,
    stacked: bool = True,
) -> tuple[float, float]:
    """Return one shared amount domain for stacked totals or individual bars."""
    periods = {str(period) for period in period_order}
    source_codes = tuple(code for codes in amount_groups.values() for code in codes)
    source = _clean_numeric(frame[frame["指标编码"].astype(str).isin(source_codes)])
    source = source[source["报告期"].astype(str).isin(periods)]
    pivot = source.pivot_table(
        index=["公司", "报告期"],
        columns="指标编码",
        values="数值",
        aggfunc="first",
    ).reset_index()
    positive_totals: list[float] = []
    negative_totals: list[float] = []
    for row in pivot.to_dict("records"):
        positive_total = 0.0
        negative_total = 0.0
        has_value = False
        for codes in amount_groups.values():
            values = [pd.to_numeric(row.get(code), errors="coerce") for code in codes]
            if any(pd.isna(value) for value in values):
                continue
            group_value = float(sum(values))
            if stacked:
                positive_total += max(group_value, 0.0)
                negative_total += min(group_value, 0.0)
            else:
                positive_total = max(positive_total, max(group_value, 0.0))
                negative_total = min(negative_total, min(group_value, 0.0))
            has_value = True
        if has_value:
            positive_totals.append(positive_total)
            negative_totals.append(negative_total)
    if not positive_totals:
        return 0.0, 1.0
    lower = min(0.0, min(negative_totals))
    upper = max(0.0, max(positive_totals))
    padded_lower = lower * 1.08 if lower < 0 else 0.0
    padded_upper = upper * 1.08 if upper > 0 else 1.0
    return float(padded_lower), float(padded_upper)


def component_stack_axis_domain(
    frame: pd.DataFrame,
    component_specs: Iterable[tuple[str, str, str]],
    period_order: Iterable[str],
    *,
    negative_component_codes: Iterable[str] = (),
) -> tuple[float, float]:
    """Return one signed stack domain shared by every displayed company."""
    codes = {str(code) for code, _, _ in component_specs}
    periods = {str(period) for period in period_order}
    rows = _clean_numeric(
        frame[
            frame["指标编码"].astype(str).isin(codes)
            & frame["报告期"].astype(str).isin(periods)
        ]
    ).drop_duplicates(["公司", "报告期", "指标编码"], keep="last")
    if rows.empty:
        return 0.0, 1.0
    negative_codes = {str(code) for code in negative_component_codes}
    if negative_codes:
        negative_mask = rows["指标编码"].astype(str).isin(negative_codes)
        rows.loc[negative_mask, "数值"] = -pd.to_numeric(
            rows.loc[negative_mask, "数值"], errors="coerce"
        ).abs()
    rows["正向值"] = pd.to_numeric(rows["数值"], errors="coerce").clip(lower=0)
    rows["负向值"] = pd.to_numeric(rows["数值"], errors="coerce").clip(upper=0)
    totals = rows.groupby(["公司", "报告期"], dropna=False)[["正向值", "负向值"]].sum()
    lower = min(0.0, float(totals["负向值"].min()))
    upper = max(0.0, float(totals["正向值"].max()))
    padded_lower = lower * 1.08 if lower < 0 else 0.0
    padded_upper = upper * 1.08 if upper > 0 else 1.0
    return float(padded_lower), float(padded_upper)


def _attach_component_stack_ratios(
    rows: pd.DataFrame,
    frame: pd.DataFrame,
    label_ratio_codes: Mapping[str, str] | None,
    label_denominator_code: str,
) -> pd.DataFrame:
    """Attach reviewed component ratios, falling back to amount/denominator."""
    result = rows.copy()
    result["标签占比"] = pd.NA
    ratio_codes = dict(label_ratio_codes or {})
    ratio_to_component = {
        ratio_code: component_code
        for component_code, ratio_code in ratio_codes.items()
    }
    ratio_rows = _clean_numeric(
        frame[frame["指标编码"].astype(str).isin(ratio_to_component)]
    )
    if not ratio_rows.empty:
        ratio_rows = ratio_rows.drop_duplicates(
            ["公司", "报告期", "指标编码"], keep="last"
        )[["公司", "报告期", "指标编码", "数值"]].copy()
        ratio_rows["指标编码"] = ratio_rows["指标编码"].map(ratio_to_component)
        ratio_rows = ratio_rows.rename(columns={"数值": "折线指标占比"})
        result = result.merge(
            ratio_rows,
            on=["公司", "报告期", "指标编码"],
            how="left",
            validate="many_to_one",
        )
        result["标签占比"] = pd.to_numeric(result["折线指标占比"], errors="coerce")
        result = result.drop(columns="折线指标占比")
    if label_denominator_code:
        denominator_rows = _clean_numeric(
            frame[frame["指标编码"].astype(str).eq(label_denominator_code)]
        )
        if not denominator_rows.empty:
            denominator_rows = denominator_rows.drop_duplicates(
                ["公司", "报告期"], keep="last"
            )[["公司", "报告期", "数值"]].rename(columns={"数值": "标签分母"})
            result = result.merge(
                denominator_rows,
                on=["公司", "报告期"],
                how="left",
                validate="many_to_one",
            )
            denominator = pd.to_numeric(result["标签分母"], errors="coerce")
            fallback_ratio = (
                pd.to_numeric(result["数值"], errors="coerce")
                / denominator.where(denominator.ne(0))
            )
            result["标签占比"] = pd.to_numeric(
                result["标签占比"], errors="coerce"
            ).fillna(fallback_ratio)
            result = result.drop(columns="标签分母")
    result["构成占比"] = pd.to_numeric(result["标签占比"], errors="coerce")
    return result


def component_stack_proportion_axis_domain(
    frame: pd.DataFrame,
    component_specs: Iterable[tuple[str, str, str]],
    period_order: Iterable[str],
    *,
    label_ratio_codes: Mapping[str, str] | None = None,
    label_denominator_code: str = "",
    negative_component_codes: Iterable[str] = (),
    show_small_negative_labels: bool = False,
) -> tuple[float, float]:
    """Return one signed ratio-stack domain shared by every displayed company."""
    codes = {str(code) for code, _, _ in component_specs}
    periods = {str(period) for period in period_order}
    rows = _clean_numeric(
        frame[
            frame["指标编码"].astype(str).isin(codes)
            & frame["报告期"].astype(str).isin(periods)
        ]
    ).drop_duplicates(["公司", "报告期", "指标编码"], keep="last")
    if rows.empty:
        return 0.0, 1.0
    negative_codes = {str(code) for code in negative_component_codes}
    if negative_codes:
        negative_mask = rows["指标编码"].astype(str).isin(negative_codes)
        rows.loc[negative_mask, "数值"] = -pd.to_numeric(
            rows.loc[negative_mask, "数值"], errors="coerce"
        ).abs()
    rows = _attach_component_stack_ratios(
        rows,
        frame,
        label_ratio_codes,
        label_denominator_code,
    )
    rows["绘图占比"] = pd.to_numeric(rows["构成占比"], errors="coerce")
    negative_amount = pd.to_numeric(rows["数值"], errors="coerce").lt(0)
    rows.loc[negative_amount, "绘图占比"] = -rows.loc[
        negative_amount, "绘图占比"
    ].abs()
    rows = rows[rows["绘图占比"].notna()].copy()
    if rows.empty:
        return 0.0, 1.0
    rows["正向占比"] = rows["绘图占比"].clip(lower=0)
    rows["负向占比"] = rows["绘图占比"].clip(upper=0)
    totals = rows.groupby(["公司", "报告期"], dropna=False)[
        ["正向占比", "负向占比"]
    ].sum()
    lower = min(0.0, float(totals["负向占比"].min()))
    upper = max(0.0, float(totals["正向占比"].max()))
    padded_lower = lower * 1.08 if lower < 0 else 0.0
    padded_upper = upper * 1.08 if upper > 0 else 1.0
    if show_small_negative_labels:
        small_negative = rows["绘图占比"].lt(0) & rows["绘图占比"].abs().lt(MIN_INSIDE_LABEL_SHARE)
        counts = small_negative.groupby([rows["公司"], rows["报告期"]]).sum()
        if counts.max() > 0:
            # Reserve annotation space without enlarging the actual bar values.
            label_step = 0.08 * max(padded_upper, abs(lower), 1.0)
            padded_lower = min(padded_lower, lower - label_step * (int(counts.max()) + 1))
    return float(padded_lower), float(padded_upper)


def _with_component_stack_label_offsets(
    rows: pd.DataFrame,
    stack_domain: tuple[float, float],
    value_font_size: int,
    panel_height: int,
) -> pd.DataFrame:
    """Move labels out of stack segments that are too short on screen."""
    result = rows.copy()
    result["label_offset"] = 0.0
    domain_span = max(float(stack_domain[1]) - float(stack_domain[0]), 1e-12)
    segment_pixels = (
        (result["堆叠终点"] - result["堆叠起点"]).abs()
        / domain_span
        * panel_height
    )
    label_height = max(14.0, float(value_font_size) + 6.0)
    external = result["占比标签"].fillna("").ne("") & segment_pixels.lt(label_height)
    if not external.any():
        return result

    group_keys = [result["公司"], result["报告期"]]
    positive_top = result["堆叠终点"].where(result["堆叠终点"].gt(0), 0).groupby(
        group_keys,
        sort=False,
    ).transform("max")
    negative_bottom = result["堆叠终点"].where(result["堆叠终点"].lt(0), 0).groupby(
        group_keys,
        sort=False,
    ).transform("min")
    positive_external = external & result["堆叠终点"].ge(0)
    negative_external = external & result["堆叠终点"].lt(0)
    positive_rank = positive_external.groupby(group_keys, sort=False).cumsum()
    negative_rank = negative_external.groupby(group_keys, sort=False).cumsum()

    result.loc[positive_external, "标签位置"] = positive_top[positive_external]
    result.loc[positive_external, "label_offset"] = (
        -label_height * positive_rank[positive_external]
    )
    result.loc[negative_external, "标签位置"] = negative_bottom[negative_external]
    result.loc[negative_external, "label_offset"] = (
        label_height * negative_rank[negative_external]
    )
    result.loc[external, "标签颜色"] = "#0C233C"
    return result


def build_component_stack_chart(
    frame: pd.DataFrame,
    component_specs: Iterable[tuple[str, str, str]],
    period_order: Iterable[str],
    title: str,
    *,
    label_ratio_codes: Mapping[str, str] | None = None,
    label_denominator_code: str = "",
    value_labels: bool = False,
    negative_component_codes: Iterable[str] = (),
    dense_layout: bool = False,
    panel_count: int = 1,
    shared_y_domain: tuple[float, float] | None = None,
    plot_proportions: bool = False,
    min_label_share: float | None = None,
    hide_labels_at_threshold: bool = False,
    show_small_negative_labels: bool = False,
    avoid_label_overlap: bool = False,
) -> alt.Chart:
    """Build a signed component stack with KPMG-only semantic colors."""
    label_share_threshold = (
        MIN_INSIDE_LABEL_SHARE
        if min_label_share is None
        else max(0.0, float(min_label_share))
    )
    def hide_share_label(value: object) -> bool:
        if pd.isna(value):
            return True
        share = abs(float(value))
        return (
            share <= label_share_threshold
            if hide_labels_at_threshold
            else share < label_share_threshold
        )
    specs = list(component_specs)
    labels = [label for _, label, _ in specs]
    colors = [color for _, _, color in specs]
    code_to_label = {code: label for code, label, _ in specs}
    code_order = {code: index for index, (code, _, _) in enumerate(specs)}
    rows = _clean_numeric(frame[frame["指标编码"].astype(str).isin(code_to_label)]).copy()
    if rows.empty:
        raise ValueError(f"{title}缺少可绘制的组成指标。")
    rows = rows.drop_duplicates(["公司", "报告期", "指标编码"], keep="last")
    negative_codes = {str(code) for code in negative_component_codes}
    if negative_codes:
        negative_mask = rows["指标编码"].astype(str).isin(negative_codes)
        rows.loc[negative_mask, "数值"] = -pd.to_numeric(
            rows.loc[negative_mask, "数值"], errors="coerce"
        ).abs()
    rows["组成类别"] = rows["指标编码"].map(code_to_label)
    rows["组成顺序"] = rows["指标编码"].map(code_order)
    label_color_map = {
        label: _label_color_for_fill(color)
        for label, color in zip(labels, colors)
    }
    rows["标签颜色"] = rows["组成类别"].map(label_color_map).fillna("#0C233C")
    rows = _signed_stack_positions(rows, "数值", "组成顺序")
    ratio_codes = dict(label_ratio_codes or {})
    if value_labels:
        rows["占比标签"] = rows.apply(
            lambda row: (
                ""
                if pd.isna(row["数值"])
                or hide_share_label(row["构成占比"])
                else _whole_number_label(row["数值"])
            ),
            axis=1,
        )
    elif ratio_codes or label_denominator_code:
        rows = _attach_component_stack_ratios(
            rows,
            frame,
            ratio_codes,
            label_denominator_code,
        )
        rows["占比标签"] = rows["构成占比"].map(
            lambda value: (
                ""
                if hide_share_label(value)
                else _whole_percent_label(value)
            )
        )
    if plot_proportions:
        if "构成占比" not in rows:
            raise ValueError(f"{title}缺少可用于比例堆叠的构成占比。")
        rows["绘图占比"] = pd.to_numeric(rows["构成占比"], errors="coerce")
        negative_amount = pd.to_numeric(rows["数值"], errors="coerce").lt(0)
        rows.loc[negative_amount, "绘图占比"] = -rows.loc[
            negative_amount, "绘图占比"
        ].abs()
        rows = rows[rows["绘图占比"].notna()].copy()
        if rows.empty:
            raise ValueError(f"{title}缺少可用于比例堆叠的构成占比。")
        rows = _signed_stack_positions(rows, "绘图占比", "组成顺序")
        rows["构成占比"] = rows["绘图占比"]
        rows["占比标签"] = rows["构成占比"].map(
            lambda value: (
                ""
                if hide_share_label(value)
                else _whole_percent_label(value)
            )
        )
    periods = list(period_order)
    company_count = rows["公司"].nunique()
    facet_columns, panel_width, facet_spacing = _facet_layout(company_count)
    density_count = panel_count if panel_count > 1 else company_count
    period_font_size, value_font_size, _ = _panel_typography(
        density_count,
        dense_layout,
    )
    panel_height = 285
    period_scale = _compact_period_scale(density_count, dense_layout)
    stack_domain = shared_y_domain or (
        component_stack_proportion_axis_domain(
            frame,
            specs,
            periods,
            label_ratio_codes=ratio_codes,
            label_denominator_code=label_denominator_code,
            negative_component_codes=negative_component_codes,
            show_small_negative_labels=show_small_negative_labels,
        )
        if plot_proportions
        else component_stack_axis_domain(
            frame,
            specs,
            periods,
            negative_component_codes=negative_component_codes,
        )
    )
    y_scale = alt.Scale(domain=list(stack_domain), zero=True, nice=False)
    if avoid_label_overlap:
        rows = _with_component_stack_label_offsets(
            rows,
            stack_domain,
            value_font_size,
            panel_height,
        )
    else:
        rows["label_offset"] = 0.0
    rows["小额负值注释"] = False
    if plot_proportions and show_small_negative_labels:
        small_negative = rows["绘图占比"].lt(0) & rows["绘图占比"].abs().lt(MIN_INSIDE_LABEL_SHARE)
        rows.loc[small_negative, "小额负值注释"] = True
        # Keep the signed stack endpoints exact; only move the text below it.
        negative_floor = rows.groupby(["公司", "报告期"])["堆叠终点"].transform("min").clip(upper=0)
        ranks = small_negative.groupby([rows["公司"], rows["报告期"]]).cumsum()
        label_step = 0.08 * max(stack_domain[1], abs(float(negative_floor.min())), 1.0)
        rows.loc[small_negative, "标签位置"] = (negative_floor - ranks * label_step)[small_negative]
        rows.loc[small_negative, "占比标签"] = rows.loc[small_negative, "绘图占比"].map(
            lambda value: f"{value * 100:.2f}%" if abs(value) >= 0.0001 else f"{value * 100:.2g}%"
        )
        rows.loc[small_negative, "标签颜色"] = "#0C233C"
        rows["注释引线终点"] = rows["标签位置"] + label_step * 0.3
    base = alt.Chart()
    tooltips: list[alt.Tooltip | str] = [
        "公司:N",
        "报告期:N",
        "组成类别:N",
        alt.Tooltip("数值:Q", format=",.2f"),
    ]
    if not value_labels:
        tooltips.append(alt.Tooltip("构成占比:Q", format=".2%"))
    tooltips.append("单位:N")
    bars = base.mark_bar(
        width=_responsive_bar_width(density_count, dense_layout)
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            title=None,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelFontSize=period_font_size,
                labelLimit=55,
            ),
            scale=period_scale,
        ),
        y=alt.Y(
            "堆叠终点:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
            stack=None,
        ),
        y2=alt.Y2("堆叠起点:Q"),
        color=alt.Color(
            "组成类别:N",
            scale=alt.Scale(domain=labels, range=colors),
            legend=None,
        ),
        tooltip=tooltips,
    )
    labels_chart = base.transform_filter(alt.datum["占比标签"] != "").mark_text(
        tooltip=False,
        fontSize=value_font_size,
        fontWeight="bold",
        baseline="middle",
        dy=alt.ExprRef(expr="datum.label_offset"),
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("标签位置:Q", scale=y_scale),
        text="占比标签:N",
        color=alt.Color("标签颜色:N", scale=None, legend=None),
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(
        color="#0C233C", strokeWidth=1
    ).encode(y=alt.Y("零线:Q", scale=y_scale))
    layers = [bars, zero, labels_chart]
    if plot_proportions and show_small_negative_labels:
        leaders = base.transform_filter(alt.datum["小额负值注释"]).mark_rule(
            color="#65758B", strokeWidth=1,
        ).encode(
            x=alt.X("报告期:N", sort=periods, scale=period_scale),
            y=alt.Y("堆叠终点:Q", scale=y_scale),
            y2=alt.Y2("注释引线终点:Q"),
            tooltip=tooltips,
        )
        layers.append(leaders)
    chart = alt.layer(*layers, data=rows).properties(width=panel_width, height=panel_height)
    if company_count == 1:
        company = str(rows["公司"].dropna().astype(str).iloc[0])
        return _transparent(
            alt.layer(*layers, data=rows).properties(
                height=panel_height,
                title=company,
            )
        )
    return _transparent(
        chart.facet(facet=alt.Facet("公司:N", title="公司"), columns=facet_columns, spacing=facet_spacing)
        .resolve_scale(y="shared")
        .properties(title=title)
    )
def build_capital_amount_combo(
    frame: pd.DataFrame,
    period_order: Iterable[str],
    *,
    dense_layout: bool = False,
    panel_count: int = 1,
) -> alt.Chart:
    source = frame[frame["指标编码"].astype(str).isin(CAPITAL_AMOUNT_CODES)].copy()
    rows = _clean_numeric(source)
    periods = list(period_order)
    component_counts = (
        rows
        .groupby(["公司", "报告期"])["指标编码"]
        .nunique()
    )
    valid_keys = component_counts[component_counts.eq(len(CAPITAL_AMOUNT_CODES))].index
    row_keys = pd.MultiIndex.from_frame(rows[["公司", "报告期"]])
    rows = rows[row_keys.isin(valid_keys)].copy()
    if rows.empty:
        raise ValueError("资本规模与结构缺少完整的四级资本数据。")
    rows["资本类别"] = rows["指标编码"].map(COMPONENT_LABELS)
    rows["标签颜色"] = rows["资本类别"].map(
        lambda label: _label_color_for_fill(CAPITAL_STRUCTURE_COLORS.get(label, "#ACEAFF"))
    )
    rows["资本合计"] = rows.groupby(["公司", "报告期"])["数值"].transform("sum")
    rows = rows[rows["资本合计"].abs().gt(1e-12)].copy()
    if rows.empty:
        raise ValueError("资本规模与结构缺少可计算占比的四级资本数据。")
    rows["资本占比"] = rows["数值"] / rows["资本合计"]
    rows["占比标签"] = rows["资本占比"].map(
        lambda value: (
            ""
            if pd.isna(value) or abs(value) < MIN_INSIDE_LABEL_SHARE
            else _whole_percent_label(value)
        )
    )
    # This is a 100% structure chart: amount remains available in tooltips,
    # while normalized shares determine segment and total bar heights.
    rows = _capital_stack_positions(rows, "资本占比")
    company_count = rows["公司"].nunique()
    facet_columns, panel_width, facet_spacing = _facet_layout(company_count)
    density_count = panel_count if panel_count > 1 else company_count
    period_font_size, value_font_size, _ = _panel_typography(
        density_count,
        dense_layout,
    )
    period_scale = _compact_period_scale(density_count, dense_layout)
    y_scale = alt.Scale(domain=[0.0, 1.0], zero=True, nice=False)
    panel_height = 285
    # Bind the company data only at the layered-chart level. If each child
    # layer carries the full dataset, Vega-Lite's outer facet cannot isolate
    # companies and every panel stacks all companies on the same bars.
    base = alt.Chart()
    bars = base.transform_filter(
        alt.FieldOneOfPredicate(field="指标编码", oneOf=list(CAPITAL_AMOUNT_CODES))
    ).mark_bar(width=_responsive_bar_width(density_count, dense_layout)).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            title=None,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelFontSize=period_font_size,
                labelLimit=55,
            ),
            scale=period_scale,
        ),
        y=alt.Y(
            "堆叠终点:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
        ),
        y2=alt.Y2("堆叠起点:Q"),
        color=alt.Color(
            "资本类别:N",
            scale=alt.Scale(
                domain=list(CAPITAL_STRUCTURE_COLORS),
                range=list(CAPITAL_STRUCTURE_COLORS.values()),
            ),
            legend=None,
        ),
        order=alt.Order("资本类别:N"),
        tooltip=[
            "公司:N",
            "报告期:N",
            "资本类别:N",
            alt.Tooltip("资本占比:Q", title="结构占比", format=".1%"),
            alt.Tooltip("数值:Q", title="金额", format=",.2f"),
            "单位:N",
        ],
    )
    labels = base.transform_filter(
        alt.FieldOneOfPredicate(field="指标编码", oneOf=list(CAPITAL_AMOUNT_CODES))
    ).transform_filter(
        alt.datum["占比标签"] != ""
    ).mark_text(
        tooltip=False,
        fontSize=value_font_size,
        fontWeight="bold",
        baseline="middle",
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("标签位置:Q", scale=y_scale),
        text=alt.Text("占比标签:N"),
        color=alt.Color("标签颜色:N", scale=None, legend=None),
    )
    chart = alt.layer(bars, labels, data=rows)
    if company_count == 1:
        company = str(rows["公司"].dropna().astype(str).iloc[0])
        return _transparent(
            chart.properties(height=270, title=company)
        )
    return _transparent(
        chart
        .properties(width=panel_width, height=panel_height)
        .facet(facet=alt.Facet("公司:N", title="公司"), columns=facet_columns, spacing=facet_spacing)
        .resolve_scale(y="shared")
        .properties(title="资本规模与结构")
    )


def build_effect_diverging_chart(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
) -> alt.Chart:
    """Show one effect per company with direction, magnitude, and time trend."""
    rows = _clean_numeric(frame[frame["指标编码"].astype(str).eq(code)])
    if rows.empty:
        raise ValueError(f"指标 {code} 没有可绘制数据。")
    rows = rows.drop_duplicates(
        subset=["公司", "报告期", "指标编码"],
        keep="last",
    ).copy()
    rows["效应类型"] = rows["指标编码"].map(COMPONENT_LABELS)
    rows["标签位置"] = rows["数值"] / 2
    periods = list(period_order)
    company_count = rows["公司"].nunique()
    facet_columns, panel_width, facet_spacing = _facet_layout(company_count)
    panel_height = 230
    dense_layout = company_count >= 10
    period_font_size, value_font_size, company_font_size = _panel_typography(
        company_count,
        dense_layout,
    )
    period_scale = _compact_period_scale(company_count, dense_layout)
    shared_domain = metric_bar_axis_domain(rows, code, periods)
    y_scale = alt.Scale(domain=list(shared_domain), zero=True, nice=False)
    color = COMPONENT_COLORS.get(COMPONENT_LABELS.get(code, ""), KPMG_BRIGHT_CHART_COLORS[1])
    label_color = _label_color_for_fill(color)
    # Keep the shared dataset on the layered chart so the outer company facet
    # filters each bar and label layer to that company's rows.
    base = alt.Chart()
    bars = base.mark_bar(
        width=_responsive_bar_width(company_count, dense_layout),
        color=color,
        opacity=0.92,
        cornerRadius=2,
    ).encode(
        x=alt.X(
            "报告期:N",
            sort=periods,
            title=None,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelFontSize=period_font_size,
                labelLimit=55,
            ),
            scale=period_scale,
        ),
        y=alt.Y(
            "数值:Q",
            title=None,
            axis=_hidden_value_axis(),
            scale=y_scale,
        ),
        tooltip=["公司:N", "报告期:N", "效应类型:N", alt.Tooltip("数值:Q", format=".2%")],
    )
    labels = base.mark_text(
        tooltip=False,
        baseline="middle",
        fontSize=value_font_size,
        fontWeight="bold",
        color=label_color,
    ).encode(
        x=alt.X("报告期:N", sort=periods, scale=period_scale),
        y=alt.Y("标签位置:Q", scale=y_scale),
        text=alt.Text("数值:Q", format=".1%"),
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(
        color="#0C233C", strokeWidth=1.1,
    ).encode(y=alt.Y("零线:Q", scale=y_scale))
    name = _metric_title(rows, code)
    chart = alt.layer(bars, labels, zero, data=rows).properties(
        width=panel_width,
        height=panel_height,
    )
    return _transparent(
        chart.facet(
            facet=alt.Facet(
                "公司:N",
                title="公司",
                header=alt.Header(labelFontSize=company_font_size),
            ),
            columns=facet_columns,
            spacing=facet_spacing,
        ).resolve_scale(y="shared").properties(title=f"{name}跨期发散分析")
    )


def build_capital_efficiency_bubble_chart(
    frame: pd.DataFrame,
    period_order: Iterable[str],
    company_colors: Mapping[str, str],
    highlight_company: str = "",
    *,
    zoom_to_overlap_region: bool = False,
    panel_label: str = "全样本图",
    chart_width: int = 650,
    chart_height: int = 430,
    show_company_legend: bool = True,
    show_size_legend: bool = True,
    show_axis_titles: bool = True,
    apply_theme: bool = True,
    linked_selection_name: str = "",
    zoom_domain_override: Mapping[str, Iterable[float]] | None = None,
) -> tuple[alt.Chart, str]:
    """Plot latest-period actual/registered and core/registered capital with asset size."""
    periods = list(dict.fromkeys(str(period) for period in period_order))
    if not periods:
        raise ValueError("缺少可绘制的报告期。")
    latest_period = periods[-1]
    required_codes = (
        "ACTUAL_CAPITAL",
        "RECOGNIZED_ASSETS",
        "REGISTERED_CAPITAL",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
    )
    rows = frame[
        frame["指标编码"].astype(str).isin(required_codes)
        & frame["报告期"].astype(str).eq(latest_period)
    ].copy()
    if "同业分类" not in rows.columns:
        rows["同业分类"] = ""
    rows = _clean_numeric(rows)
    asset_units = [
        str(value).strip()
        for value in rows.loc[
            rows["指标编码"].astype(str).eq("RECOGNIZED_ASSETS"), "单位"
        ].dropna().unique()
        if str(value).strip()
    ]
    asset_unit = asset_units[0] if asset_units else ""
    pivot = rows.pivot_table(
        index=["公司", "同业分类"],
        columns="指标编码",
        values="数值",
        aggfunc="first",
    ).reset_index()
    missing_columns = [code for code in required_codes if code not in pivot.columns]
    if missing_columns:
        raise ValueError("最新报告期缺少气泡图所需指标。")
    pivot = pivot.dropna(subset=list(required_codes)).copy()
    pivot = pivot[
        pivot["REGISTERED_CAPITAL"].ne(0)
        & pivot["RECOGNIZED_ASSETS"].ne(0)
    ].copy()
    pivot["实际资本/注册资本"] = (
        pivot["ACTUAL_CAPITAL"] / pivot["REGISTERED_CAPITAL"]
    )
    pivot["核心资本/注册资本"] = (
        (pivot["CORE_T1_CAPITAL"] + pivot["CORE_T2_CAPITAL"])
        / pivot["REGISTERED_CAPITAL"]
    )
    pivot["认可资产"] = pivot["RECOGNIZED_ASSETS"]
    pivot["气泡大小"] = pivot["RECOGNIZED_ASSETS"].abs()
    pivot["报告期"] = latest_period
    finite_columns = ["实际资本/注册资本", "核心资本/注册资本", "气泡大小"]
    finite_mask = pivot[finite_columns].apply(
        lambda column: column.map(lambda value: math.isfinite(float(value)))
    ).all(axis=1)
    pivot = pivot[finite_mask & pivot["气泡大小"].gt(0)].copy()
    if pivot.empty:
        raise ValueError(f"{latest_period} 缺少可绘制的完整公司数据。")
    full_sample_x_median = float(pivot["实际资本/注册资本"].median())
    full_sample_y_median = float(pivot["核心资本/注册资本"].median())
    zoom_domain_source = pivot
    zoom_applied = False
    highlight = str(highlight_company or "").strip()

    if zoom_to_overlap_region and len(pivot) >= 2:
        x_field = "实际资本/注册资本"
        y_field = "核心资本/注册资本"
        x_span = max(float(pivot[x_field].max() - pivot[x_field].min()), 1e-9)
        y_span = max(float(pivot[y_field].max() - pivot[y_field].min()), 1e-9)
        row_indices = list(pivot.index)

        def display_distance_between(left_index: object, right_index: object) -> float:
            x_distance = (
                float(pivot.at[left_index, x_field] - pivot.at[right_index, x_field])
                / x_span
            )
            y_distance = (
                float(pivot.at[left_index, y_field] - pivot.at[right_index, y_field])
                / y_span
            )
            return math.sqrt(x_distance ** 2 + (y_distance * 0.65) ** 2)

        highlight_indices = list(
            pivot.index[pivot["公司"].astype(str).eq(highlight)]
        )
        if highlight_indices:
            seed_index = highlight_indices[0]
            nearby = sorted(
                (
                    display_distance_between(row_index, seed_index),
                    row_index,
                )
                for row_index in row_indices
            )
            context_count = min(len(pivot), 4)
            selected_indices = [
                row_index for _, row_index in nearby[:context_count]
            ]
            zoom_domain_source = pivot.loc[selected_indices]
            zoom_applied = True
        else:
            closest_pair: tuple[object, object] | None = None
            closest_distance = math.inf
            for left_position, left_index in enumerate(row_indices[:-1]):
                for right_index in row_indices[left_position + 1:]:
                    display_distance = display_distance_between(
                        left_index,
                        right_index,
                    )
                    if display_distance < closest_distance:
                        closest_distance = display_distance
                        closest_pair = (left_index, right_index)
            if closest_pair is None or closest_distance > 0.12:
                closest_pair = None
            cluster_radius = min(max(closest_distance * 2.2, 0.06), 0.16)
            nearby = []
            for row_index in row_indices:
                distance_to_seed = (
                    min(
                        display_distance_between(row_index, seed_index)
                        for seed_index in closest_pair
                    )
                    if closest_pair is not None
                    else math.inf
                )
                nearby.append((distance_to_seed, row_index))
            nearby.sort(key=lambda item: item[0])
            dense_count = sum(
                1 for distance, _ in nearby if distance <= cluster_radius
            )
            if closest_pair is not None:
                context_count = min(
                    len(pivot),
                    min(6, max(3, dense_count)),
                )
                selected_indices = [
                    row_index for _, row_index in nearby[:context_count]
                ]
                zoom_domain_source = pivot.loc[selected_indices]
                zoom_applied = True

    pivot = pivot.sort_values(
        ["实际资本/注册资本", "核心资本/注册资本"],
        kind="stable",
    ).reset_index(drop=True)

    pivot["是否追踪"] = pivot["公司"].astype(str).eq(highlight)
    companies = _company_legend_order(pivot["公司"].astype(str), company_colors)
    company_scale = _company_scale(companies, company_colors)
    company_legend = (
        alt.Legend(
            title="公司",
            orient="right",
            direction="vertical",
            columns=1,
            symbolSize=110,
            offset=24,
            padding=8,
        )
        if show_company_legend
        else None
    )
    size_title = "认可资产" if not asset_unit else f"认可资产（{asset_unit}）"
    size_legend_title = (
        ["气泡大小代表", "认可资产金额"]
        if not asset_unit
        else ["气泡大小代表", f"认可资产金额（{asset_unit}）"]
    )
    size_legend = (
        alt.Legend(
            title=size_legend_title,
            orient="right",
            direction="vertical",
            tickCount=3,
            format=",.1f",
            offset=24,
            padding=8,
        )
        if show_size_legend
        else None
    )
    def padded_domain(field: str) -> list[float]:
        source = zoom_domain_source if zoom_applied else pivot
        lower = float(source[field].min())
        upper = float(source[field].max())
        span = upper - lower
        if zoom_applied:
            center = (lower + upper) / 2
            if span > 1e-12:
                domain_span = span * 1.35
            else:
                full_span = float(pivot[field].max() - pivot[field].min())
                domain_span = (
                    full_span * 0.12
                    if full_span > 1e-12
                    else max(abs(center) * 0.08, 1e-4)
                )
            return [center - domain_span / 2, center + domain_span / 2]
        padding = max(
            span * 0.08,
            max(abs(lower), abs(upper)) * 0.025,
            0.005,
        )
        return [lower - padding, upper + padding]

    x_domain = padded_domain("实际资本/注册资本")
    y_domain = padded_domain("核心资本/注册资本")

    def overridden_domain(key: str, fallback: list[float]) -> list[float]:
        if not zoom_domain_override or key not in zoom_domain_override:
            return fallback
        values = [float(value) for value in zoom_domain_override[key]]
        if (
            len(values) != 2
            or not all(math.isfinite(value) for value in values)
            or values[0] >= values[1]
        ):
            return fallback
        return values

    x_domain = overridden_domain("x", x_domain)
    y_domain = overridden_domain("y", y_domain)
    selection_name = str(linked_selection_name or "").strip()
    if zoom_to_overlap_region and selection_name:
        x_scale = alt.Scale(
            domain={"param": selection_name, "encoding": "x"},
            zero=False,
            nice=False,
        )
        y_scale = alt.Scale(
            domain={"param": selection_name, "encoding": "y"},
            zero=False,
            nice=False,
        )
    else:
        x_scale = alt.Scale(domain=x_domain, zero=False, nice=False)
        y_scale = alt.Scale(domain=y_domain, zero=False, nice=False)
    border_color = "#D9DEE7"
    reference_color = "#C9CED6"
    x_title = "实际资本/注册资本（%）" if show_axis_titles else ""
    y_title = "核心资本/注册资本（%）" if show_axis_titles else ""
    x_axis = alt.Axis(
        title=x_title,
        labelExpr="format(datum.value * 100, '.1f')",
        tickCount=6,
        labelOverlap=True,
        labels=True,
        domain=True,
        domainColor=border_color,
        domainWidth=1,
        ticks=True,
        tickColor=border_color,
    )
    y_axis = alt.Axis(
        title=y_title,
        labelExpr=(
            "format(datum.value * 100, '.1f')"
            if zoom_to_overlap_region
            else "format(datum.value * 100, '.0f')"
        ),
        tickCount=7,
        labels=True,
        domain=True,
        domainColor=border_color,
        domainWidth=1,
        ticks=True,
        tickColor=border_color,
    )
    base = alt.Chart(pivot)
    bubbles = base.mark_circle(
        opacity=0.78,
        clip=zoom_to_overlap_region,
    ).encode(
        x=alt.X(
            "实际资本/注册资本:Q",
            title=x_title,
            axis=x_axis,
            scale=x_scale,
        ),
        y=alt.Y(
            "核心资本/注册资本:Q",
            title=y_title,
            axis=y_axis,
            scale=y_scale,
        ),
        size=alt.Size(
            "气泡大小:Q",
            scale=alt.Scale(range=[180, 1800]),
            legend=size_legend,
        ),
        color=alt.Color("公司:N", scale=company_scale, legend=company_legend),
        stroke=alt.condition(
            alt.datum["是否追踪"],
            alt.value("#00338D"),
            alt.value("#FFFFFF"),
        ),
        strokeWidth=alt.condition(
            alt.datum["是否追踪"],
            alt.value(3.0),
            alt.value(1.0),
        ),
        tooltip=[
            "公司:N",
            "同业分类:N",
            "报告期:N",
            alt.Tooltip("实际资本/注册资本:Q", format=".1%"),
            alt.Tooltip("核心资本/注册资本:Q", format=".1%"),
            alt.Tooltip("认可资产:Q", title=size_title, format=",.2f"),
        ],
    )
    if selection_name:
        border = alt.Chart(pd.DataFrame({"边框": [1]})).mark_rect(
            fillOpacity=0,
            stroke=border_color,
            strokeWidth=1,
        ).encode(
            x=alt.value(0),
            x2=alt.value(chart_width),
            y=alt.value(0),
            y2=alt.value(chart_height),
        )
    else:
        border = alt.Chart(pd.DataFrame({
            "横轴起点": [x_domain[0]],
            "横轴终点": [x_domain[1]],
            "纵轴起点": [y_domain[0]],
            "纵轴终点": [y_domain[1]],
        })).mark_rect(
            fillOpacity=0,
            stroke=border_color,
            strokeWidth=1,
        ).encode(
            x=alt.X("横轴起点:Q", axis=x_axis, scale=x_scale),
            x2="横轴终点:Q",
            y=alt.Y("纵轴起点:Q", axis=y_axis, scale=y_scale),
            y2="纵轴终点:Q",
        )
    reference_values = pd.DataFrame({
        "横轴中位数": [full_sample_x_median],
        "纵轴中位数": [full_sample_y_median],
    })
    vertical_reference = alt.Chart(reference_values).mark_rule(
        color=reference_color,
        strokeDash=[5, 4],
        strokeWidth=1,
    ).encode(
        x=alt.X("横轴中位数:Q", axis=x_axis, scale=x_scale),
        y=alt.Y(
            datum=y_domain[0],
            type="quantitative",
            axis=y_axis,
            scale=y_scale,
        ),
        y2=alt.Y2(datum=y_domain[1]),
    )
    horizontal_reference = alt.Chart(reference_values).mark_rule(
        color=reference_color,
        strokeDash=[5, 4],
        strokeWidth=1,
    ).encode(
        x=alt.X(
            datum=x_domain[0],
            type="quantitative",
            axis=x_axis,
            scale=x_scale,
        ),
        x2=alt.X2(datum=x_domain[1]),
        y=alt.Y("纵轴中位数:Q", axis=y_axis, scale=y_scale),
    )
    if selection_name and not zoom_to_overlap_region:
        # The interval selection must inherit the bubble fields instead of the
        # border layer's synthetic start/end fields so Streamlit can persist
        # the selected data-domain for print locking.
        chart_layers = [bubbles, border, vertical_reference, horizontal_reference]
    else:
        chart_layers = [border]
        if not zoom_to_overlap_region:
            chart_layers.extend([vertical_reference, horizontal_reference])
        chart_layers.append(bubbles)
    chart = alt.layer(*chart_layers).properties(
        title=alt.TitleParams(
            text=panel_label,
            anchor="start",
            color="#667085",
            fontSize=12,
            fontWeight="bold",
            offset=8,
        ),
        width=chart_width,
        height=chart_height,
    )
    return (_transparent(chart) if apply_theme else chart), latest_period


def capital_efficiency_bubble_chart_domain(
    chart: alt.Chart,
) -> dict[str, list[float]]:
    """Read the linked x/y window from an already-built zoom chart."""
    spec = chart.to_dict(validate=True)
    bubble_layer = next(
        layer
        for layer in spec["layer"]
        if layer.get("mark", {}).get("type") == "circle"
    )
    return {
        "x": list(bubble_layer["encoding"]["x"]["scale"]["domain"]),
        "y": list(bubble_layer["encoding"]["y"]["scale"]["domain"]),
    }


def capital_efficiency_bubble_default_domain(
    frame: pd.DataFrame,
    period_order: Iterable[str],
    company_colors: Mapping[str, str],
    highlight_company: str = "",
) -> tuple[dict[str, list[float]], str]:
    """Return the initial linked-zoom window around the tracked company."""
    chart, latest_period = build_capital_efficiency_bubble_chart(
        frame,
        period_order,
        company_colors,
        highlight_company,
        zoom_to_overlap_region=True,
        show_company_legend=False,
        show_size_legend=False,
        show_axis_titles=False,
        apply_theme=False,
    )
    return capital_efficiency_bubble_chart_domain(chart), latest_period


def combine_capital_efficiency_bubble_charts(
    full_chart: alt.Chart,
    overlap_chart: alt.Chart,
    *,
    spacing: int = 20,
    selection_name: str = "",
    selection_domain: Mapping[str, Iterable[float]] | None = None,
) -> alt.HConcatChart:
    """Place both bubble panels on one canvas so their plot baselines align."""
    linked_selection = str(selection_name or "").strip()
    if linked_selection and selection_domain:
        x_domain = [float(value) for value in selection_domain.get("x", ())]
        y_domain = [float(value) for value in selection_domain.get("y", ())]
        if len(x_domain) == 2 and len(y_domain) == 2:
            brush = alt.selection_interval(
                name=linked_selection,
                encodings=["x", "y"],
                value={"x": x_domain, "y": y_domain},
                clear=False,
                translate=True,
                zoom=True,
                mark=alt.BrushConfig(
                    fill="#D9DEE7",
                    fillOpacity=0.12,
                    stroke="#98A2B3",
                    strokeDash=[4, 3],
                    strokeOpacity=0.95,
                    strokeWidth=1.2,
                ),
            )
            full_chart = full_chart.add_params(brush)
    combined = alt.hconcat(
        full_chart,
        overlap_chart,
        spacing=spacing,
    ).resolve_scale(
        x="independent",
        y="independent",
        color="shared",
        size="shared",
    )
    return _transparent(combined)


def _compact_scatter_axis(values: pd.Series) -> alt.Axis:
    """Use sparse, range-aware ticks so rounded scatter labels never repeat."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return alt.Axis(format=".2~f", tickCount=6)
    span = float(numeric.max() - numeric.min())
    if span > 0:
        target_step = span / 5
        decimal_places = max(
            0,
            min(6, int(math.ceil(-math.log10(target_step)))),
        )
    else:
        magnitude = abs(float(numeric.iloc[0]))
        decimal_places = 2 if magnitude < 1 else 1
    return alt.Axis(format=f".{decimal_places}~f", tickCount=6)


def _ratio_axis_display_scale(values: pd.Series) -> tuple[float, str]:
    """Choose a readable display unit for a raw 0-1 ratio series."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().abs()
    maximum = float(numeric.max()) if not numeric.empty else 0.0
    if maximum >= 0.01:
        return 100.0, "%"
    if maximum >= 0.001:
        return 1_000.0, "‰"
    if maximum > 0:
        return 10_000.0, "bp"
    return 100.0, "%"


def _scatter_zoom_source(
    pivot: pd.DataFrame,
    x_code: str,
    y_code: str,
    highlight_company: str = "",
) -> tuple[pd.DataFrame, bool]:
    """Choose a dense local window, prioritizing the tracked company."""
    if len(pivot) < 2:
        return pivot, False
    x_span = max(float(pivot[x_code].max() - pivot[x_code].min()), 1e-9)
    y_span = max(float(pivot[y_code].max() - pivot[y_code].min()), 1e-9)
    row_indices = list(pivot.index)

    def display_distance(left_index: object, right_index: object) -> float:
        x_distance = float(
            pivot.at[left_index, x_code] - pivot.at[right_index, x_code]
        ) / x_span
        y_distance = float(
            pivot.at[left_index, y_code] - pivot.at[right_index, y_code]
        ) / y_span
        return math.sqrt(x_distance ** 2 + (y_distance * 0.65) ** 2)

    highlight = str(highlight_company or "").strip()
    highlight_indices = list(
        pivot.index[pivot["公司"].astype(str).eq(highlight)]
    )
    if highlight_indices:
        seed_index = highlight_indices[0]
        nearby = sorted(
            (display_distance(row_index, seed_index), row_index)
            for row_index in row_indices
        )
        selected = [row_index for _, row_index in nearby[:min(len(pivot), 4)]]
        return pivot.loc[selected], True

    closest_pair: tuple[object, object] | None = None
    closest_distance = math.inf
    for left_position, left_index in enumerate(row_indices[:-1]):
        for right_index in row_indices[left_position + 1:]:
            distance = display_distance(left_index, right_index)
            if distance < closest_distance:
                closest_distance = distance
                closest_pair = (left_index, right_index)
    if closest_pair is None or closest_distance > 0.12:
        return pivot, False

    cluster_radius = min(max(closest_distance * 2.2, 0.06), 0.16)
    nearby = sorted(
        (
            min(display_distance(row_index, seed_index) for seed_index in closest_pair),
            row_index,
        )
        for row_index in row_indices
    )
    dense_count = sum(1 for distance, _ in nearby if distance <= cluster_radius)
    context_count = min(len(pivot), min(6, max(3, dense_count)))
    selected = [row_index for _, row_index in nearby[:context_count]]
    return pivot.loc[selected], True


def _scatter_padded_domain(
    full_sample: pd.DataFrame,
    domain_source: pd.DataFrame,
    field: str,
    *,
    zoomed: bool,
    protect_bubbles: bool = False,
) -> list[float]:
    lower = float(domain_source[field].min())
    upper = float(domain_source[field].max())
    span = upper - lower
    if zoomed:
        center = (lower + upper) / 2
        if span > 1e-12:
            domain_span = span * (1.55 if protect_bubbles else 1.35)
        else:
            full_span = float(full_sample[field].max() - full_sample[field].min())
            domain_span = (
                full_span * (0.20 if protect_bubbles else 0.12)
                if full_span > 1e-12
                else max(
                    abs(center) * (0.15 if protect_bubbles else 0.08),
                    0.01,
                )
            )
        return [center - domain_span / 2, center + domain_span / 2]
    padding = max(
        span * (0.18 if protect_bubbles else 0.08),
        max(abs(lower), abs(upper)) * (0.06 if protect_bubbles else 0.025),
        0.01,
    )
    return [lower - padding, upper + padding]


def _scatter_domain_override(
    domain_override: Mapping[str, Iterable[float]] | None,
    key: str,
    fallback: list[float],
) -> list[float]:
    if not domain_override or key not in domain_override:
        return fallback
    values = [float(value) for value in domain_override[key]]
    if (
        len(values) != 2
        or not all(math.isfinite(value) for value in values)
        or values[0] >= values[1]
    ):
        return fallback
    return values


def build_matrix_chart(
    frame: pd.DataFrame,
    x_code: str,
    y_code: str,
    period_order: Iterable[str],
    title: str,
    highlight_company: str = "",
    regulatory_lines: bool = False,
    company_colors: Mapping[str, str] | None = None,
    percentage_axes: bool = False,
    zoom_to_overlap_region: bool = False,
    panel_label: str = "",
    chart_width: int = 820,
    chart_height: int = 540,
    show_company_legend: bool = True,
    show_axis_titles: bool = True,
    linked_selection_name: str = "",
    zoom_domain_override: Mapping[str, Iterable[float]] | None = None,
    apply_theme: bool = True,
    bubble_size_code: str = "",
    bubble_size_label: str = "",
    show_size_legend: bool = True,
) -> tuple[alt.Chart, str]:
    size_code = str(bubble_size_code or "").strip()
    required_codes = [x_code, y_code]
    if size_code:
        required_codes.append(size_code)
    rows = _clean_numeric(
        frame[frame["指标编码"].astype(str).isin(required_codes)]
    )
    periods = list(period_order)
    latest_period = periods[-1]
    latest = rows[rows["报告期"].astype(str).eq(latest_period)]
    size_units = (
        [
            str(value).strip()
            for value in latest.loc[
                latest["指标编码"].astype(str).eq(size_code), "单位"
            ].dropna().unique()
            if str(value).strip()
        ]
        if size_code
        else []
    )
    size_unit = size_units[0] if size_units else ""
    pivot = latest.pivot_table(
        index=["公司", "同业分类"], columns="指标编码", values="数值", aggfunc="first"
    ).reset_index()
    missing_columns = [code for code in required_codes if code not in pivot.columns]
    if missing_columns:
        raise ValueError(f"{latest_period} 缺少可绘制的完整公司数据。")
    pivot = pivot.dropna(subset=required_codes).copy()
    if size_code:
        pivot["气泡大小"] = pd.to_numeric(
            pivot[size_code], errors="coerce"
        ).abs()
        finite_size = pivot["气泡大小"].map(
            lambda value: math.isfinite(float(value))
        )
        pivot = pivot[finite_size & pivot["气泡大小"].gt(0)].copy()
    x_unit = ""
    y_unit = ""
    if percentage_axes:
        if size_code:
            x_multiplier, x_unit = _ratio_axis_display_scale(pivot[x_code])
            y_multiplier, y_unit = _ratio_axis_display_scale(pivot[y_code])
        else:
            x_multiplier, y_multiplier = 100.0, 100.0
            x_unit, y_unit = "%", "%"
        pivot[x_code] = (
            pd.to_numeric(pivot[x_code], errors="coerce") * x_multiplier
        )
        pivot[y_code] = (
            pd.to_numeric(pivot[y_code], errors="coerce") * y_multiplier
        )
    if pivot.empty:
        raise ValueError(f"{latest_period} 缺少可绘制的完整公司数据。")
    full_sample_x_median = float(pivot[x_code].median())
    full_sample_y_median = float(pivot[y_code].median())
    zoom_source, zoom_applied = (
        _scatter_zoom_source(pivot, x_code, y_code, highlight_company)
        if zoom_to_overlap_region
        else (pivot, False)
    )
    x_domain = _scatter_domain_override(
        zoom_domain_override,
        "x",
        _scatter_padded_domain(
            pivot,
            zoom_source,
            x_code,
            zoomed=zoom_applied,
            protect_bubbles=bool(size_code),
        ),
    )
    y_domain = _scatter_domain_override(
        zoom_domain_override,
        "y",
        _scatter_padded_domain(
            pivot,
            zoom_source,
            y_code,
            zoomed=zoom_applied,
            protect_bubbles=bool(size_code),
        ),
    )
    selection_name = str(linked_selection_name or "").strip()
    if zoom_to_overlap_region and selection_name:
        x_scale = alt.Scale(
            domain={"param": selection_name, "encoding": "x"},
            zero=False,
            nice=False,
        )
        y_scale = alt.Scale(
            domain={"param": selection_name, "encoding": "y"},
            zero=False,
            nice=False,
        )
    elif zoom_to_overlap_region or zoom_domain_override or size_code:
        x_scale = alt.Scale(domain=x_domain, zero=False, nice=False)
        y_scale = alt.Scale(domain=y_domain, zero=False, nice=False)
    else:
        x_scale = alt.Scale(zero=False)
        y_scale = alt.Scale(zero=False)
    highlight = str(highlight_company or "").strip()
    pivot["是否追踪"] = pivot["公司"].astype(str).eq(highlight)
    companies = list(dict.fromkeys(pivot["公司"].astype(str)))
    scale = _company_scale(companies, company_colors or {})
    legend = (
        alt.Legend(
            title="公司",
            orient="right",
            direction="vertical",
            columns=1,
            symbolSize=110,
            offset=24,
            padding=8,
        )
        if show_company_legend
        else None
    )
    x_title = _metric_title(rows, x_code)
    y_title = _metric_title(rows, y_code)
    if percentage_axes:
        x_title = f"{x_title}（{x_unit}）"
        y_title = f"{y_title}（{y_unit}）"
    if not show_axis_titles:
        x_title = ""
        y_title = ""
    axis_source = zoom_source if zoom_to_overlap_region and zoom_applied else pivot
    x_axis = _compact_scatter_axis(axis_source[x_code]) if percentage_axes else alt.Axis()
    y_axis = _compact_scatter_axis(axis_source[y_code]) if percentage_axes else alt.Axis()
    tooltip_format = ".2f" if percentage_axes else ",.2f"
    tooltip = [
        "公司:N",
        "同业分类:N",
        alt.Tooltip(f"{x_code}:Q", title=x_title, format=tooltip_format),
        alt.Tooltip(f"{y_code}:Q", title=y_title, format=tooltip_format),
    ]
    if size_code:
        size_label = str(bubble_size_label or "").strip() or _metric_title(
            rows, size_code
        )
        size_title = size_label if not size_unit else f"{size_label}（{size_unit}）"
        size_legend_title = (
            ["气泡大小代表", f"{size_label}金额"]
            if not size_unit
            else ["气泡大小代表", f"{size_label}金额（{size_unit}）"]
        )
        size_encoding = alt.Size(
            "气泡大小:Q",
            scale=alt.Scale(range=[180, 1800]),
            legend=(
                alt.Legend(
                    title=size_legend_title,
                    orient="right",
                    direction="vertical",
                    tickCount=3,
                    format=",.1f",
                    offset=24,
                    padding=8,
                )
                if show_size_legend
                else None
            ),
        )
        tooltip.append(
            alt.Tooltip(f"{size_code}:Q", title=size_title, format=",.2f")
        )
    else:
        size_encoding = alt.condition(
            alt.datum["是否追踪"], alt.value(230), alt.value(115)
        )
    points = alt.Chart(pivot).mark_circle(
        opacity=0.96,
        clip=bool(size_code) or zoom_to_overlap_region,
    ).encode(
        x=alt.X(
            f"{x_code}:Q",
            title=x_title,
            scale=x_scale,
            axis=x_axis,
        ),
        y=alt.Y(
            f"{y_code}:Q",
            title=y_title,
            scale=y_scale,
            axis=y_axis,
        ),
        color=alt.Color("公司:N", scale=scale, legend=legend),
        size=size_encoding,
        stroke=alt.condition(
            alt.datum["是否追踪"], alt.value("#00338D"), alt.value("#FFFFFF")
        ),
        strokeWidth=alt.condition(
            alt.datum["是否追踪"], alt.value(3.0), alt.value(1.0)
        ),
        tooltip=tooltip,
    )
    border = None
    if selection_name:
        border = alt.Chart(pd.DataFrame({"边框": [1]})).mark_rect(
            fillOpacity=0,
            stroke="#D9DEE7",
            strokeWidth=1,
        ).encode(
            x=alt.value(0),
            x2=alt.value(chart_width),
            y=alt.value(0),
            y2=alt.value(chart_height),
        )
    if zoom_to_overlap_region and selection_name:
        # Match the working bubble-chart structure exactly: the local panel has
        # one dynamic-domain data layer plus a pixel border. Independent median
        # rule datasets would otherwise participate in scale merging and pin the
        # local axes instead of following the interval selection.
        layers: list[alt.Chart] = [border, points]
    else:
        # Keep points first so the interval selection inherits the actual x/y
        # fields rather than a synthetic border or reference-line field.
        layers = [points]
        if border is not None:
            layers.append(border)
    if regulatory_lines and not zoom_to_overlap_region:
        layers.extend([
            alt.Chart(pd.DataFrame({"核心监管线": [50.0]})).mark_rule(color="#ED2124", strokeDash=[5, 4]).encode(x="核心监管线:Q"),
            alt.Chart(pd.DataFrame({"综合监管线": [100.0]})).mark_rule(color="#ED2124", strokeDash=[5, 4]).encode(y="综合监管线:Q"),
        ])
    elif not zoom_to_overlap_region:
        reference_color = "#C9CED6" if panel_label else "#0C233C"
        layers.extend([
            alt.Chart(pd.DataFrame({"横轴中位数": [full_sample_x_median]})).mark_rule(color=reference_color, strokeDash=[6, 4]).encode(x="横轴中位数:Q"),
            alt.Chart(pd.DataFrame({"纵轴中位数": [full_sample_y_median]})).mark_rule(color=reference_color, strokeDash=[6, 4]).encode(y="纵轴中位数:Q"),
        ])
    chart_title: str | alt.TitleParams = f"{latest_period} {title}"
    if panel_label:
        chart_title = alt.TitleParams(
            text=panel_label,
            anchor="start",
            color="#667085",
            fontSize=12,
            fontWeight="bold",
            offset=8,
        )
    chart = alt.layer(*layers).properties(
        title=chart_title,
        width=chart_width,
        height=chart_height,
    )
    return (_transparent(chart) if apply_theme else chart), latest_period


def matrix_chart_domain(
    chart: alt.Chart,
    x_code: str,
    y_code: str,
) -> dict[str, list[float]]:
    """Read the fixed x/y domain from a locally zoomed matrix chart."""
    spec = chart.to_dict(validate=True)
    points = next(
        layer
        for layer in spec["layer"]
        if layer.get("mark", {}).get("type") == "circle"
    )
    return {
        "x": list(points["encoding"]["x"]["scale"]["domain"]),
        "y": list(points["encoding"]["y"]["scale"]["domain"]),
    }


def risk_ratio_scatter_default_domain(
    frame: pd.DataFrame,
    x_code: str,
    y_code: str,
    period_order: Iterable[str],
    title: str,
    highlight_company: str = "",
    company_colors: Mapping[str, str] | None = None,
    bubble_size_code: str = "",
    bubble_size_label: str = "",
) -> tuple[dict[str, list[float]], str]:
    """Return the initial linked local window for a risk-ratio bubble chart."""
    chart, latest_period = build_matrix_chart(
        frame,
        x_code,
        y_code,
        period_order,
        title,
        highlight_company,
        company_colors=company_colors,
        percentage_axes=True,
        zoom_to_overlap_region=True,
        show_company_legend=False,
        show_axis_titles=False,
        apply_theme=False,
        bubble_size_code=bubble_size_code,
        bubble_size_label=bubble_size_label,
        show_size_legend=False,
    )
    return matrix_chart_domain(chart, x_code, y_code), latest_period


def combine_linked_matrix_charts(
    full_chart: alt.Chart,
    local_chart: alt.Chart,
    *,
    selection_name: str,
    selection_domain: Mapping[str, Iterable[float]],
    spacing: int = 20,
) -> alt.HConcatChart:
    """Combine full and local risk panels with a browser-side brush."""
    x_domain = [float(value) for value in selection_domain.get("x", ())]
    y_domain = [float(value) for value in selection_domain.get("y", ())]
    if len(x_domain) == 2 and len(y_domain) == 2:
        brush = alt.selection_interval(
            name=selection_name,
            encodings=["x", "y"],
            value={"x": x_domain, "y": y_domain},
            clear=False,
            translate=True,
            zoom=True,
            mark=alt.BrushConfig(
                fill="#D9DEE7",
                fillOpacity=0.12,
                stroke="#98A2B3",
                strokeDash=[4, 3],
                strokeOpacity=0.95,
                strokeWidth=1.2,
            ),
        )
        full_chart = full_chart.add_params(brush)
    return _transparent(
        alt.hconcat(full_chart, local_chart, spacing=spacing).resolve_scale(
            x="independent",
            y="independent",
            color="shared",
            size="shared",
        )
    )
