from __future__ import annotations

from collections.abc import Iterable, Mapping

import altair as alt
import pandas as pd

from .solvency_navigation import KPMG_CHART_COLORS


TRANSPARENT = "transparent"
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
    "核心一级资本": "#00338D",
    "核心二级资本": "#1E49E2",
    "附属一级资本": "#7213EA",
    "附属二级资本": "#00B8F5",
    "风险分散效应": "#1E49E2",
    "损失吸收效应": "#FD349C",
}
CAPITAL_COMPONENT_ORDER = {
    label: index
    for index, label in enumerate(list(COMPONENT_COLORS)[:4])
}
MIN_INSIDE_LABEL_SHARE = 0.04
UNDISCLOSED_COLOR = "#B8BDC7"
TREND_GROUP_THRESHOLD = 11
TREND_GROUP_MAX_PEERS = 7
TREND_LABEL_FORMATS = {
    "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES": ",.5f",
}
COMPACT_TREND_SCALE_CODES = {
    "NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",
}


def report_period_color_map(period_order: Iterable[str]) -> dict[str, str]:
    """Assign stable KPMG colors to report periods in display order."""
    periods = list(dict.fromkeys(str(period) for period in period_order))
    return {
        period: KPMG_CHART_COLORS[index % len(KPMG_CHART_COLORS)]
        for index, period in enumerate(periods)
    }


def _clean_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["数值"] = pd.to_numeric(result["数值"], errors="coerce")
    return result.dropna(subset=["数值"])


def _transparent(chart: alt.Chart) -> alt.Chart:
    return chart.properties(background=TRANSPARENT).configure_view(
        fill=TRANSPARENT,
        strokeOpacity=0,
    ).configure_axis(
        domain=True,
        domainColor="#D0D5DD",
        domainWidth=1,
        grid=False,
        labelColor="#0C233C",
        ticks=True,
        tickColor="#D0D5DD",
        titleColor="#0C233C",
    ).configure_legend(
        labelColor="#0C233C",
        titleColor="#0C233C",
        symbolOpacity=1,
    ).configure_header(
        labelColor="#0C233C",
        labelFontWeight="bold",
        titleColor="#0C233C",
    ).configure_title(
        color="#0C233C",
        fontWeight="bold",
    )


def _metric_title(frame: pd.DataFrame, code: str) -> str:
    rows = frame[frame["指标编码"].astype(str).eq(code)]
    return code if rows.empty else str(rows.iloc[0].get("指标名称", code))


def _axis_title(frame: pd.DataFrame) -> str:
    units = [str(value).strip() for value in frame.get("单位", pd.Series(dtype=str)).dropna().unique() if str(value).strip()]
    return "数值" if not units else f"数值（{'、'.join(units)}）"


def _company_scale(
    companies: Iterable[str],
    colors: Mapping[str, str],
) -> alt.Scale:
    domain = list(dict.fromkeys(str(company) for company in companies))
    palette = [colors.get(company, KPMG_CHART_COLORS[index % len(KPMG_CHART_COLORS)]) for index, company in enumerate(domain)]
    return alt.Scale(domain=domain, range=palette)


def _facet_layout(company_count: int) -> tuple[int, int]:
    """Choose a compact grid that uses the report width without tiny panels."""
    if company_count <= 1:
        return 1, 760
    if company_count == 2:
        return 2, 470
    if company_count == 3:
        return 3, 300
    if company_count == 4:
        return 2, 470
    if company_count == 5:
        return 5, 190
    if company_count == 6:
        return 3, 300
    for columns in range(min(7, company_count), 1, -1):
        if company_count % columns == 0 and company_count // columns <= 3:
            return columns, max(140, 980 // columns)
    # Prime-sized selections (such as the seven companies in the report
    # screenshot) stay on one row so Vega-Lite does not create a trailing
    # empty facet with a misleading report-period axis.
    return company_count, max(140, 980 // max(company_count, 1))


def _capital_label_color() -> alt.condition:
    """Use KPMG Light Blue on dark fills and Dark Blue on Pacific Blue."""
    return alt.condition(
        alt.FieldOneOfPredicate(
            field="资本类别",
            oneOf=["核心一级资本", "核心二级资本", "附属一级资本"],
        ),
        alt.value("#ACEAFF"),
        alt.value("#0C233C"),
    )


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


def _metric_rows(
    frame: pd.DataFrame,
    code: str,
    period_order: Iterable[str],
) -> tuple[pd.DataFrame, list[str], str]:
    metric = _clean_numeric(frame[frame["指标编码"].astype(str).eq(code)])
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
    padding = max(span * 0.08, max(abs(low), abs(high)) * 0.025, minimum_padding)
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
    companies = list(dict.fromkeys(metric["公司"].astype(str)))
    scale = _company_scale(companies, company_colors)
    label_format = TREND_LABEL_FORMATS.get(code, ",.2f")
    y_axis = (
        alt.Axis(format=".5f", tickCount=6)
        if code in COMPACT_TREND_SCALE_CODES
        else alt.Axis()
    )
    company_legend = alt.Legend(
        title="公司",
        orient="top-right",
        direction="vertical",
        columns=1,
        symbolSize=80,
    )
    highlight = str(highlight_company or "").strip()
    line_size = (
        alt.condition(alt.datum["公司"] == highlight, alt.value(3.0), alt.value(1.8))
        if highlight in companies else alt.value(2.0)
    )
    line_opacity = (
        alt.condition(alt.datum["公司"] == highlight, alt.value(1.0), alt.value(0.68))
        if highlight in companies else alt.value(0.72)
    )
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
        tooltip=["公司:N", "报告期:N", "指标名称:N", "披露状态:N", "数值:Q", "单位:N"],
    )
    points = base.mark_point(filled=True, size=58).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        color=alt.Color("公司:N", scale=scale, legend=None),
        opacity=line_opacity,
        tooltip=["公司:N", "报告期:N", "指标名称:N", "披露状态:N", "数值:Q", "单位:N"],
    )
    highlight_labels = base.transform_filter(
        alt.datum["显示标签"] & ~alt.datum["是否全局最大"] & ~alt.datum["是否全局最小"]
    ).mark_text(dy=-10, fontSize=10, fontWeight="bold", color="#0C233C").encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        text=alt.Text("数值:Q", format=label_format),
    )
    max_labels = base.transform_filter(alt.datum["是否全局最大"]).mark_text(
        dy=-11, fontSize=10, fontWeight="bold", color="#0C233C",
    ).encode(x=alt.X("报告期:N", sort=periods), y="数值:Q", text=alt.Text("数值:Q", format=label_format))
    min_labels = base.transform_filter(alt.datum["是否全局最小"]).mark_text(
        dy=11, baseline="top", fontSize=10, fontWeight="bold", color="#0C233C",
    ).encode(x=alt.X("报告期:N", sort=periods), y="数值:Q", text=alt.Text("数值:Q", format=label_format))
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
    metric["是否全局最大"] = metric["数值"].eq(metric["数值"].max())
    metric["是否全局最小"] = metric["数值"].eq(metric["数值"].min())
    metric["显示标签"] = metric[["是否追踪", "是否全局最大", "是否全局最小"]].any(axis=1)
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
    """Split dense selections into readable groups with a shared y-domain.

    When a tracked company is selected it is repeated in every dense group so
    each peer group can be compared against the same highlighted benchmark.
    """
    metric, periods, name, y_domain = _prepare_metric_trend(
        frame, code, period_order, highlight_company
    )
    companies = list(dict.fromkeys(metric["公司"].astype(str)))
    if len(companies) <= TREND_GROUP_THRESHOLD:
        company_groups = [companies]
    else:
        highlight = str(highlight_company or "").strip()
        tracked = [highlight] if highlight in companies else []
        peers = [company for company in companies if company != highlight]
        group_size = TREND_GROUP_MAX_PEERS if tracked else TREND_GROUP_MAX_PEERS + 1
        group_count = max(1, (len(peers) + group_size - 1) // group_size)
        base_size, extra = divmod(len(peers), group_count)
        company_groups = []
        start = 0
        for index in range(group_count):
            current_size = base_size + (1 if index < extra else 0)
            company_groups.append(peers[start:start + current_size])
            start += current_size
        if tracked:
            company_groups = [tracked + group for group in company_groups]
    charts: list[alt.Chart] = []
    group_count = len(company_groups)
    for group_index, group in enumerate(company_groups, start=1):
        group_metric = metric[metric["公司"].astype(str).isin(group)].copy()
        charts.append(_build_metric_trend_group(
            group_metric,
            periods,
            name,
            code,
            company_colors,
            highlight_company,
            y_domain,
            group_index,
            group_count,
        ))
    return charts


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
    columns, panel_width = _facet_layout(company_count)
    facet_rows = max(1, (company_count + columns - 1) // columns)
    panel_height = 285 if facet_rows == 1 else 235 if facet_rows == 2 else 195
    base = alt.Chart()
    bars = base.mark_bar(opacity=0.92, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("报告期:N", sort=periods, title="报告期", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(
            "数值:Q",
            title=_axis_title(metric),
            scale=alt.Scale(zero=True, domainMin=0),
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
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
    )
    points = base.mark_point(
        filled=True, size=62, color="#0C233C",
    ).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
    )
    labels = base.mark_text(
        dy=-12, fontSize=11, fontWeight="bold", color="#0C233C",
    ).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        text=alt.Text("数值:Q", format=",.2f"),
    )
    layers: list[alt.Chart] = [bars, line, points, labels]
    limit = REGULATORY_LIMITS.get(code)
    if limit is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"监管下限": [limit]}))
            .mark_rule(color="#ED2124", strokeDash=[5, 4], strokeWidth=1.2)
            .encode(y="监管下限:Q")
        )
    tracked_frame = (
        base.transform_filter(alt.datum["是否追踪"])
        .transform_aggregate(追踪记录="count()", groupby=["公司"])
        .mark_rect(
            fillOpacity=0,
            stroke="#B8BDC7",
            strokeWidth=2,
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
                header=alt.Header(labelFontSize=12, labelPadding=20),
            ),
            columns=columns,
            spacing=22,
        ).resolve_scale(y="independent").properties(title=f"{name}公司跨期对标")
    )


def build_capital_amount_combo(
    frame: pd.DataFrame,
    period_order: Iterable[str],
) -> alt.Chart:
    codes = [*CAPITAL_AMOUNT_CODES, "ACTUAL_CAPITAL"]
    source = frame[frame["指标编码"].astype(str).isin(codes)].copy()
    expected = pd.MultiIndex.from_product(
        [list(dict.fromkeys(frame["公司"].dropna().astype(str))), list(period_order)],
        names=["公司", "报告期"],
    )
    disclosure_counts = source.groupby(["公司", "报告期"])["指标编码"].nunique()
    has_undisclosed = disclosure_counts.reindex(expected, fill_value=0).lt(len(codes)).any()
    rows = _clean_numeric(source)
    periods = list(period_order)
    capital_mask = rows["指标编码"].astype(str).isin(CAPITAL_AMOUNT_CODES)
    component_counts = (
        rows[capital_mask]
        .groupby(["公司", "报告期"])["指标编码"]
        .nunique()
    )
    valid_keys = component_counts[component_counts.eq(len(CAPITAL_AMOUNT_CODES))].index
    row_keys = pd.MultiIndex.from_frame(rows[["公司", "报告期"]])
    rows = rows[row_keys.isin(valid_keys)].copy()
    capital_mask = rows["指标编码"].astype(str).isin(CAPITAL_AMOUNT_CODES)
    rows["资本类别"] = rows["指标编码"].map(COMPONENT_LABELS)
    rows["_资本组成值"] = rows["数值"].where(capital_mask)
    capital_totals = rows.groupby(["公司", "报告期"])["_资本组成值"].transform("sum")
    rows["资本占比"] = rows["数值"].where(capital_mask) / capital_totals
    rows["占比标签"] = rows["资本占比"].map(
        lambda value: (
            ""
            if pd.isna(value) or abs(value) < MIN_INSIDE_LABEL_SHARE
            else f"{value:.1%}"
        )
    )
    capital_rows = _capital_stack_positions(rows[capital_mask].copy(), "数值")
    total_rows = rows[~capital_mask].copy()
    rows = pd.concat([capital_rows, total_rows], ignore_index=True, sort=False)
    company_count = rows["公司"].nunique()
    facet_columns, panel_width = _facet_layout(company_count)
    facet_rows = max(1, (company_count + facet_columns - 1) // facet_columns)
    panel_height = 285 if facet_rows == 1 else 235 if facet_rows == 2 else 195
    # Bind the company data only at the layered-chart level. If each child
    # layer carries the full dataset, Vega-Lite's outer facet cannot isolate
    # companies and every panel stacks all companies on the same bars.
    base = alt.Chart()
    bars = base.transform_filter(
        alt.FieldOneOfPredicate(field="指标编码", oneOf=list(CAPITAL_AMOUNT_CODES))
    ).mark_bar().encode(
        x=alt.X("报告期:N", sort=periods, title="报告期"),
        y=alt.Y("堆叠终点:Q", title=_axis_title(rows)),
        y2=alt.Y2("堆叠起点:Q"),
        color=alt.Color(
            "资本类别:N",
            scale=alt.Scale(domain=list(COMPONENT_COLORS)[:4], range=list(COMPONENT_COLORS.values())[:4]),
            legend=None,
        ),
        order=alt.Order("资本类别:N"),
        tooltip=["公司:N", "报告期:N", "资本类别:N", alt.Tooltip("数值:Q", format=",.2f"), "单位:N"],
    )
    labels = base.transform_filter(
        alt.FieldOneOfPredicate(field="指标编码", oneOf=list(CAPITAL_AMOUNT_CODES))
    ).transform_filter(
        alt.datum["占比标签"] != ""
    ).mark_text(fontSize=10, fontWeight="bold", baseline="middle").encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("标签位置:Q"),
        text=alt.Text("占比标签:N"),
        color=_capital_label_color(),
    )
    total_line = base.transform_filter(
        alt.datum["指标编码"] == "ACTUAL_CAPITAL"
    ).mark_line(
        color="#FD349C",
        strokeWidth=2.4,
        point=alt.OverlayMarkDef(size=38, filled=True, color="#FD349C"),
    ).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("数值:Q"),
        tooltip=["公司:N", "报告期:N", alt.Tooltip("数值:Q", title="实际资本", format=",.2f"), "单位:N"],
    )
    return _transparent(
        alt.layer(bars, labels, total_line, data=rows)
        .properties(width=panel_width, height=panel_height)
        .facet(facet=alt.Facet("公司:N", title="公司"), columns=facet_columns, spacing=24)
        .resolve_scale(y="independent")
        .properties(title="四级资本规模与结构（粉色折线为实际资本总额）")
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
    facet_columns, panel_width = _facet_layout(company_count)
    facet_rows = max(1, (company_count + facet_columns - 1) // facet_columns)
    panel_height = 230 if facet_rows == 1 else 180 if facet_rows == 2 else 145
    color = COMPONENT_COLORS.get(COMPONENT_LABELS.get(code, ""), "#1E49E2")
    label_color = "#ACEAFF" if color == "#1E49E2" else "#0C233C"
    # Keep the shared dataset on the layered chart so the outer company facet
    # filters each bar and label layer to that company's rows.
    base = alt.Chart()
    bars = base.mark_bar(color=color, opacity=0.92, cornerRadius=2).encode(
        x=alt.X("报告期:N", sort=periods, title="报告期", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y(
            "数值:Q",
            title="占量化风险最低资本比例",
            axis=alt.Axis(format=".1%"),
            scale=alt.Scale(zero=True, padding=18),
        ),
        tooltip=["公司:N", "报告期:N", "效应类型:N", alt.Tooltip("数值:Q", format=".2%")],
    )
    labels = base.mark_text(
        baseline="middle",
        fontSize=10,
        fontWeight="bold",
        color=label_color,
    ).encode(
        x=alt.X("报告期:N", sort=periods),
        y=alt.Y("标签位置:Q"),
        text=alt.Text("数值:Q", format=".1%"),
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(
        color="#0C233C", strokeWidth=1.1,
    ).encode(y="零线:Q")
    name = _metric_title(rows, code)
    chart = alt.layer(bars, labels, zero, data=rows).properties(
        width=panel_width,
        height=panel_height,
    )
    return _transparent(
        chart.facet(
            facet=alt.Facet("公司:N", title="公司", header=alt.Header(labelFontSize=12)),
            columns=facet_columns,
            spacing=22,
        ).resolve_scale(y="independent").properties(title=f"{name}跨期发散分析")
    )


def build_matrix_chart(
    frame: pd.DataFrame,
    x_code: str,
    y_code: str,
    period_order: Iterable[str],
    title: str,
    highlight_company: str = "",
    regulatory_lines: bool = False,
    company_colors: Mapping[str, str] | None = None,
) -> tuple[alt.Chart, str]:
    rows = _clean_numeric(frame[frame["指标编码"].astype(str).isin([x_code, y_code])])
    periods = list(period_order)
    latest_period = periods[-1]
    latest = rows[rows["报告期"].astype(str).eq(latest_period)]
    pivot = latest.pivot_table(
        index=["公司", "同业分类"], columns="指标编码", values="数值", aggfunc="first"
    ).reset_index().dropna(subset=[x_code, y_code])
    highlight = str(highlight_company or "").strip()
    pivot["是否追踪"] = pivot["公司"].astype(str).eq(highlight)
    companies = list(dict.fromkeys(pivot["公司"].astype(str)))
    scale = _company_scale(companies, company_colors or {})
    legend = alt.Legend(
        title="公司",
        orient="right",
        direction="vertical",
        columns=1,
        symbolSize=110,
        offset=24,
        padding=8,
    )
    points = alt.Chart(pivot).mark_circle(opacity=0.96).encode(
        x=alt.X(f"{x_code}:Q", title=_metric_title(rows, x_code), scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y_code}:Q", title=_metric_title(rows, y_code), scale=alt.Scale(zero=False)),
        color=alt.Color("公司:N", scale=scale, legend=legend),
        size=alt.condition(alt.datum["是否追踪"], alt.value(230), alt.value(115)),
        tooltip=["公司:N", "同业分类:N", alt.Tooltip(f"{x_code}:Q", format=",.2f"), alt.Tooltip(f"{y_code}:Q", format=",.2f")],
    )
    layers: list[alt.Chart] = [points]
    if regulatory_lines:
        layers.extend([
            alt.Chart(pd.DataFrame({"核心监管线": [50.0]})).mark_rule(color="#ED2124", strokeDash=[5, 4]).encode(x="核心监管线:Q"),
            alt.Chart(pd.DataFrame({"综合监管线": [100.0]})).mark_rule(color="#ED2124", strokeDash=[5, 4]).encode(y="综合监管线:Q"),
        ])
    else:
        layers.extend([
            alt.Chart(pd.DataFrame({"横轴中位数": [pivot[x_code].median()]})).mark_rule(color="#0C233C", strokeDash=[6, 4]).encode(x="横轴中位数:Q"),
            alt.Chart(pd.DataFrame({"纵轴中位数": [pivot[y_code].median()]})).mark_rule(color="#0C233C", strokeDash=[6, 4]).encode(y="纵轴中位数:Q"),
        ])
    return _transparent(
        alt.layer(*layers).properties(
            title=f"{latest_period} {title}",
            width=820,
            height=540,
        )
    ), latest_period
