"""Render quality/capital data with the existing annual-report chart geometry.

The figures supplied here carry the already selected data and shared axis ranges.
All displayed company panels use the same Vega renderer and sizing helpers as
the core/combined solvency chart, including responsive band widths.
"""
from __future__ import annotations

import math

import altair as alt
import pandas as pd

from .solvency_step7_charts import (
    COMPANY_PANEL_HEIGHT, MIN_INSIDE_LABEL_SHARE, _compact_period_scale,
    _hidden_value_axis, _panel_typography, _responsive_bar_width,
    _responsive_grouped_bar_width,
    _transparent, _whole_number_label, report_period_combo_bar_color_map,
)


def _scale(axis):
    return alt.Scale(domain=list(axis.range), zero=True, nice=False)


def _period_x(periods, count, *, scale=None):
    period_font, _, _ = _panel_typography(count)
    return alt.X("报告期:N", title=None, sort=periods,
                 scale=scale if scale is not None else _compact_period_scale(count),
                 axis=alt.Axis(title=None, labelAngle=0, labelFontSize=period_font, labelLimit=55))


def _tooltip():
    return ["公司:N", "报告期:N", "指标名称:N", alt.Tooltip("数值:Q", format=",.2f"), "单位:N"]


def empty_company_chart(message, count):
    _, font, _ = _panel_typography(count)
    return _transparent(alt.Chart(pd.DataFrame({"说明": [message]})).mark_text(
        fontSize=font, color="#7A8496", lineBreak="\n", lineHeight=font + 4,
    ).encode(text="说明:N").properties(height=COMPANY_PANEL_HEIGHT))


def company_quality_chart(fig, company: str, company_count: int, *, unit: str,
                          percentage_bar: bool = False, color_by_period: bool = False):
    """Use original chart parameters for every new company panel."""
    kind = fig.data[0].type
    if kind == "pie":
        return _pie(fig, company, company_count, unit)
    if kind == "scatterpolar":
        return _radar(fig, company, company_count)
    if kind == "waterfall":
        return _waterfall(fig, company, company_count, unit)
    if kind == "bar" and len(fig.data) == 1:
        return _bar(
            fig, company, company_count, "%" if percentage_bar else unit,
            color_by_period=color_by_period,
            label_color="#000000" if percentage_bar else "#FFFFFF",
        )
    if len(fig.data) == 2 and fig.data[1].type == "scatter":
        return _combo(fig, company, company_count, "%" if percentage_bar else unit,
                      color_by_period=color_by_period)
    return _stack(fig, company, company_count, unit)


def _bar(
    fig,
    company,
    count,
    unit,
    *,
    color_by_period=False,
    label_color="#FFFFFF",
):
    trace = fig.data[0]
    periods = list(fig.layout.xaxis.ticktext or trace.x)
    rows = pd.DataFrame([
        {
            "公司": company,
            "报告期": period,
            "指标名称": trace.name,
            "数值": value,
            "单位": unit,
            "标签位置": None if value is None else value / 2,
            "标签": "" if value is None else (
                f"{value:.1f}%" if unit == "%" else _whole_number_label(value)
            ),
        }
        for period, value in zip(periods, trace.y)
    ])
    x, scale = _period_x(periods, count), _scale(fig.layout.yaxis)
    _, font, _ = _panel_typography(count)
    base = alt.Chart(rows)
    bars = base.mark_bar(
        width=_responsive_bar_width(count), opacity=0.9,
        cornerRadiusTopLeft=2, cornerRadiusTopRight=2, color=trace.marker.color,
    ).encode(
        x=x,
        y=alt.Y("数值:Q", title=None, axis=_hidden_value_axis(), scale=scale),
        tooltip=["公司:N", "报告期:N", "指标名称:N",
                 alt.Tooltip("数值:Q", title=f"数值（{unit}）", format=",.2f")],
    )
    if color_by_period:
        period_colors = report_period_combo_bar_color_map(periods)
        bars = bars.encode(color=alt.Color(
            "报告期:N",
            scale=alt.Scale(
                domain=periods,
                range=[period_colors[period] for period in periods],
            ),
            legend=None,
        ))
    labels = base.mark_text(
        tooltip=False, fontSize=font, fontWeight="bold", color=label_color,
    ).encode(
        x=x, y=alt.Y("标签位置:Q", scale=scale), text="标签:N",
    )
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(
        color="#0C233C", strokeWidth=1,
    ).encode(y=alt.Y("零线:Q", scale=scale))
    return _transparent(alt.layer(bars, labels, zero).properties(height=COMPANY_PANEL_HEIGHT))


def _combo(fig, company, count, unit, *, color_by_period=False):
    bar, line = fig.data
    periods = list(fig.layout.xaxis.ticktext or bar.x)
    rows = pd.DataFrame([
        {"公司": company, "报告期": period, "期间口径": scope, "柱值": amount,
         "线值": rate, "柱标签": "" if amount is None else
         (f"{amount:.1f}%" if unit == "%" else _whole_number_label(amount)),
         "线标签": "" if rate is None else f"{rate:.1f}%", "顺序": index}
        for index, (period, scope, amount, rate) in enumerate(zip(periods, bar.x, bar.y, line.y))
    ])
    rows["线段"] = rows["线值"].isna().cumsum()
    x = _period_x(periods, count)
    _, font, _ = _panel_typography(count)
    amount_scale, rate_scale = _scale(fig.layout.yaxis), _scale(fig.layout.yaxis2)
    amount_low, amount_high = fig.layout.yaxis.range
    rate_low, rate_high = fig.layout.yaxis2.range
    business_amount_bar = str(bar.name) in {"签单保费", "新业务价值"}
    if business_amount_bar:
        rows["柱标签位置"] = rows["柱值"] / 2
        rows["柱标签偏移"] = 0
    else:
        close_labels = (
            ((rows["柱值"] - amount_low) / (amount_high - amount_low)
             - (rows["线值"] - rate_low) / (rate_high - rate_low)).abs()
            < (font + 12) / COMPANY_PANEL_HEIGHT
        )
        rows["柱标签位置"] = rows["柱值"].where(~close_labels, rows["柱值"] / 2)
        rows["柱标签偏移"] = close_labels.map({True: 0, False: -7})
    bar_tip = ["公司:N", "报告期:N", "期间口径:N",
               alt.Tooltip("柱值:Q", title=f"{bar.name}（{unit}）", format=",.2f")]
    line_tip = ["公司:N", "报告期:N", "期间口径:N",
                alt.Tooltip("线值:Q", title=f"{line.name}（%）", format=",.2f")]
    base = alt.Chart(rows)
    bars = base.mark_bar(width=_responsive_bar_width(count), opacity=0.9,
                         cornerRadiusTopLeft=2, cornerRadiusTopRight=2,
                         color=bar.marker.color).encode(
        x=x, y=alt.Y("柱值:Q", title=None, axis=_hidden_value_axis(), scale=amount_scale), tooltip=bar_tip)
    if color_by_period:
        period_colors = report_period_combo_bar_color_map(periods)
        bars = bars.encode(color=alt.Color(
            "报告期:N", scale=alt.Scale(domain=periods, range=[period_colors[p] for p in periods]),
            legend=None,
        ))
    bar_labels = base.mark_text(tooltip=False, dy=alt.ExprRef(expr="datum['柱标签偏移']"),
                                fontSize=font, fontWeight="bold",
                                color="#000000").encode(
        x=x, y=alt.Y("柱标签位置:Q", scale=amount_scale), text="柱标签:N")
    # Each run of disclosed values is a separate path, even on older Vega-Lite.
    line_chart = base.mark_line(color="#FD349C", strokeWidth=2.6).encode(
        x=x, y=alt.Y("线值:Q", title=None, axis=None, scale=rate_scale),
        detail="线段:N", order="顺序:Q", tooltip=line_tip)
    points = base.mark_point(color="#FD349C", filled=True, size=54).encode(
        x=x, y=alt.Y("线值:Q", axis=None, scale=rate_scale), tooltip=line_tip)
    labels = base.mark_text(tooltip=False, dy=-10, fontSize=font, fontWeight="bold", color="#000000").encode(
        x=x, y=alt.Y("线值:Q", axis=None, scale=rate_scale), text="线标签:N")
    amounts = alt.layer(bars, bar_labels)
    rates = alt.layer(line_chart, points, labels)
    return _transparent(alt.layer(amounts, rates).resolve_scale(y="independent").properties(height=COMPANY_PANEL_HEIGHT))


def _stack(fig, company, count, unit):
    periods = list(fig.data[0].x)
    records = []
    for index, period in enumerate(periods):
        positive = negative = 0.0
        total = sum(abs(trace.y[index]) for trace in fig.data if trace.y[index] is not None)
        for trace in fig.data:
            value = trace.y[index]
            if value is None:
                continue
            start = positive if value >= 0 else negative
            end = start + value
            if value >= 0:
                positive = end
            else:
                negative = end
            share = abs(value) / total if total else 0
            records.append({"公司": company, "报告期": period, "指标名称": trace.name,
                            "数值": value, "单位": unit, "起点": start, "终点": end,
                            "标签位置": (start + end) / 2, "颜色": trace.marker.color,
                            "标签": _whole_number_label(value) if share >= MIN_INSIDE_LABEL_SHARE else ""})
    rows = pd.DataFrame(records)
    x, scale = _period_x(periods, count), _scale(fig.layout.yaxis)
    _, font, _ = _panel_typography(count)
    base = alt.Chart(rows)
    bars = base.mark_bar(width=_responsive_bar_width(count)).encode(
        x=x, y=alt.Y("终点:Q", title=None, axis=_hidden_value_axis(), scale=scale, stack=None),
        y2="起点:Q", color=alt.Color("颜色:N", scale=None, legend=None), tooltip=_tooltip())
    labels = base.mark_text(tooltip=False, fontSize=font, fontWeight="bold", baseline="middle", color="#0C233C").encode(
        x=x, y=alt.Y("标签位置:Q", scale=scale), text="标签:N")
    zero = alt.Chart(pd.DataFrame({"零线": [0]})).mark_rule(color="#0C233C", strokeWidth=1).encode(
        y=alt.Y("零线:Q", scale=scale))
    return _transparent(alt.layer(bars, labels, zero).properties(height=COMPANY_PANEL_HEIGHT))


def _waterfall(fig, company, count, unit):
    trace = fig.data[0]
    running, records = 0.0, []
    for index, (measure, value, label) in enumerate(zip(trace.measure, trace.y, trace.customdata)):
        start = 0 if measure == "absolute" else running
        end = None if value is None else (value if measure == "absolute" else running + value)
        if end is not None:
            running = end
        records.append({"公司": company, "报告期": str(index), "指标名称": label,
                        "数值": value, "单位": unit, "起点": start if end is not None else None,
                        "终点": end, "颜色": "#1E49E2" if measure == "absolute" else
                        "#00C0AE" if value is not None and value >= 0 else "#FD349C",
                        "标签": "" if value is None or count >= 5 else _whole_number_label(value)})
    rows = pd.DataFrame(records)
    periods = list(rows["报告期"])
    # Keep the seven steps close together while every bar remains a fraction
    # of its responsive category band, not a fixed pixel width.
    waterfall_x_scale = alt.Scale(paddingInner=0.18, paddingOuter=0.6)
    x, scale = _period_x(periods, count, scale=waterfall_x_scale), _scale(fig.layout.yaxis)
    _, font, _ = _panel_typography(count)
    base = alt.Chart(rows)
    bars = base.mark_bar(width=_responsive_grouped_bar_width(count)).encode(
        x=x, y=alt.Y("终点:Q", title=None, axis=_hidden_value_axis(), scale=scale, stack=None),
        y2="起点:Q", color=alt.Color("颜色:N", scale=None, legend=None),
        tooltip=["公司:N", alt.Tooltip("报告期:N", title="步骤"), "指标名称:N",
                 alt.Tooltip("数值:Q", format=",.2f"), "单位:N"])
    labels = base.mark_text(tooltip=False, dy=-7, fontSize=font, fontWeight="bold", color="#0C233C").encode(
        x=x, y=alt.Y("终点:Q", scale=scale), text="标签:N")
    connectors = []
    for previous, current in zip(records, records[1:]):
        if previous["终点"] is not None and current["终点"] is not None and current["报告期"] != periods[-1]:
            connectors.append({"前项": previous["报告期"], "后项": current["报告期"], "连接值": previous["终点"]})
    connector = alt.Chart(pd.DataFrame(connectors, columns=["前项", "后项", "连接值"])).mark_rule(
        color="#B8BDC7", strokeWidth=1).encode(
        x=alt.X("前项:N", sort=periods, scale=waterfall_x_scale), x2="后项:N",
        y=alt.Y("连接值:Q", scale=scale))
    return _transparent(alt.layer(connector, bars, labels).properties(height=COMPANY_PANEL_HEIGHT))


def _pie(fig, company, count, unit):
    trace = fig.data[0]
    total = sum(trace.values)
    rows = pd.DataFrame([{"公司": company, "指标名称": label, "数值": value, "单位": unit,
                          "占比": value / total, "颜色": color, "顺序": index}
                         for index, (label, value, color) in enumerate(zip(trace.labels, trace.values, trace.marker.colors))])
    _, font, _ = _panel_typography(count)
    # Vega width/height signals update on browser/container resize. No fixed radius.
    radius = "max(1, min(width, height) / 2 - 8)"
    base = alt.Chart(rows).encode(theta=alt.Theta("数值:Q", stack=True), order="顺序:Q")
    arcs = base.mark_arc(outerRadius=alt.ExprRef(expr=radius)).encode(
        color=alt.Color("颜色:N", scale=None, legend=None),
        tooltip=["公司:N", "指标名称:N", alt.Tooltip("数值:Q", format=",.2f"), "单位:N",
                 alt.Tooltip("占比:Q", format=".2%")])
    labels = base.mark_text(radius=alt.ExprRef(expr=f"({radius}) * 0.65"),
                           fontSize=font, fontWeight="bold", color="#0C233C").encode(
        text=alt.condition(f"datum['占比'] >= {MIN_INSIDE_LABEL_SHARE}", alt.Text("占比:Q", format=".1%"), alt.value("")))
    return _transparent(alt.layer(arcs, labels).properties(height=COMPANY_PANEL_HEIGHT))


def _radar(fig, company, count):
    trace = fig.data[0]
    values = list(trace.r)[:6]
    customdata = list(trace.customdata)[:6] if trace.customdata is not None else list(trace.theta)[:6]
    full_labels = [str(label).replace("<br>", "") for label in customdata]
    labels = [str(index) for index in range(1, 7)]
    low, high = fig.layout.polar.radialaxis.range
    period_font, _, _ = _panel_typography(count)
    # Numeric axis labels need much less margin than the former full names, so
    # the hexagon can use more of each responsive company panel.
    radius = "max(6, min(width - 34, height - 36) / 2)"
    def position(chart, extra=0):
        return chart.transform_calculate(
            px=f"width / 2 + datum.nx * (({radius}) + {extra})",
            py=f"height / 2 + datum.ny * (({radius}) + {extra})",
        ).encode(x=alt.X("px:Q", scale=None, axis=None), y=alt.Y("py:Q", scale=None, axis=None))
    directions = [(math.cos(math.pi / 2 + i * math.pi / 3), -math.sin(math.pi / 2 + i * math.pi / 3)) for i in range(6)]
    grid_rows = [{"nx": nx * level / 4, "ny": ny * level / 4, "环": level, "顺序": i}
                 for level in range(1, 5) for i, (nx, ny) in enumerate(directions + directions[:1])]
    grid = position(alt.Chart(pd.DataFrame(grid_rows))).mark_line(color="#E7ECF5", strokeWidth=1).encode(detail="环:N", order="顺序:Q")
    spokes = position(alt.Chart(pd.DataFrame([{"nx": nx, "ny": ny} for nx, ny in directions]))).mark_rule(color="#E7ECF5").encode(
        x2=alt.value(alt.ExprRef(expr="width / 2")), y2=alt.value(alt.ExprRef(expr="height / 2")))
    points = []
    for index, (value, (nx, ny)) in enumerate(zip(values, directions)):
        if value is None:
            continue
        distance = (value - low) / (high - low)
        points.append({"nx": nx * distance, "ny": ny * distance, "顺序": index,
                       "公司": company, "指标名称": full_labels[index], "数值": value})
    point_rows = pd.DataFrame(points)
    tip = ["公司:N", "指标名称:N", alt.Tooltip("数值:Q", format=".2f", title="收益率（%）")]
    dots = position(alt.Chart(point_rows)).mark_point(color="#1E49E2", filled=True, size=36).encode(tooltip=tip)
    edges = []
    by_index = {p["顺序"]: p for p in points}
    for i in range(6):
        if i in by_index and (i + 1) % 6 in by_index:
            edges.extend([{**by_index[j], "边": i, "顶点": k} for k, j in enumerate([i, (i + 1) % 6])])
    lines = position(alt.Chart(pd.DataFrame(edges, columns=["nx", "ny", "边", "顶点", "公司", "指标名称", "数值"])) ).mark_line(
        color="#1E49E2", strokeWidth=2.6).encode(detail="边:N", order="顶点:Q", tooltip=tip)
    title_rows = pd.DataFrame([{"nx": nx, "ny": ny, "名称": label} for (nx, ny), label in zip(directions, labels)])
    titles = position(alt.Chart(title_rows), 10).mark_text(
        fontSize=period_font, lineBreak="\n", lineHeight=period_font + 2, color="#0C233C",
        align=alt.ExprRef(expr="datum.nx > 0.1 ? 'left' : datum.nx < -0.1 ? 'right' : 'center'"),
        baseline=alt.ExprRef(expr="datum.ny < -0.9 ? 'bottom' : datum.ny > 0.9 ? 'top' : 'middle'"),
    ).encode(text="名称:N")
    layers = [grid, spokes]
    if len(points) == 6:
        area = position(alt.Chart(point_rows)).mark_line(
            interpolate="linear-closed", fill="#1E49E2", fillOpacity=0.12,
            stroke="#1E49E2", strokeWidth=2.6,
        ).encode(order="顺序:Q")
        layers.append(area)
    layers.extend([lines, dots, titles])
    if count <= 4:
        radial_labels = pd.DataFrame([{"nx": 0, "ny": -level / 4,
                                      "刻度": f"{low + (high-low)*level/4:.1f}%"} for level in range(1, 5)])
        layers.append(position(alt.Chart(radial_labels)).mark_text(
            fontSize=period_font, align="left", dx=3, color="#7A8496",
        ).encode(text="刻度:N"))
    return _transparent(alt.layer(*layers).properties(height=COMPANY_PANEL_HEIGHT))
