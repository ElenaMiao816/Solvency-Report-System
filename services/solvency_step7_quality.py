"""Data selection and Plotly figures for Step 7 operating quality and capital detail."""
from __future__ import annotations

from collections.abc import Iterable
import math
import re

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BLUE = "#1E49E2"
PINK = "#FD349C"
GREY = "#B8BDC7"
COMPONENT_COLORS = ("#ACEAFF", "#00B8F5", "#B497FF", "#FD349C", "#63EBB2", "#00C0AE", "#F1C44D")

CORE_STEPS = (
    (("NON_RECOGNIZED_ASSET_BOOK_VALUE",), "非认可资产账面价值"),
    (("LONG_TERM_EQUITY_VALUATION_DIFFERENCE", "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT"),
     "长期股权投资及投资性房地产估值差额"),
    (("DEFERRED_TAX_ASSET_ADJUSTMENT",), "递延所得税资产调整"),
    (("POLICY_SURPLUS_CORE_T1",), "计入核心一级的保单未来盈余"),
    (("OTHER_CORE_T1_ADJUSTMENT",), "其他调整项目"),
)
CORE_COMPONENT_LABELS = {
    "LONG_TERM_EQUITY_VALUATION_DIFFERENCE": "长期股权投资估值差额",
    "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT": "投资性房地产估值差额（税后）",
}
CORE_OMITTED_STEPS = (
    ("AGRICULTURAL_CATASTROPHE_RISK_RESERVE", "农业大灾风险准备金"),
    ("QUALIFYING_CORE_T1_LIABILITY_CAPITAL", "合格负债类资本工具"),
)
ANC_COMPONENTS = (
    ("ANC_T1_SUBORDINATED_TERM_DEBT", "次级定期债务"),
    ("ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", "资本补充债券"),
    ("ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT", "可转换次级债"),
    ("ANC_T1_DEFERRED_TAX_ASSET", "递延所得税资产"),
    ("ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE", "投资性房地产公允价值增值"),
    ("POLICY_SURPLUS_ANC_T1", "计入附属一级的保单未来盈余"),
    ("OTHER_ANC_T1_CAPITAL", "其他附属一级资本"),
)
RADAR_METRICS = (
    ("ROE", "净资产收益率", "return"),
    ("ROA", "总资产收益率", "return"),
    ("INVESTMENT_RETURN", "投资收益率（当季数）", "quarter"),
    ("COMPREHENSIVE_INVESTMENT_RETURN", "综合投资收益率（当季数）", "quarter"),
    ("INVESTMENT_RETURN", "近三年平均投资收益率", "three_year"),
    ("COMPREHENSIVE_INVESTMENT_RETURN", "近三年平均综合投资收益率", "three_year"),
)
METRIC_LABELS = {
    "SIGNED_PREMIUM": "签单保费", "NEW_BUSINESS_VALUE": "新业务价值",
    "RENEWAL_PREMIUM": "续期签单保费", "INSURANCE_REVENUE": "保险业务收入",
    "INSURANCE_CONTRACT_LIABILITY_TO_TOTAL_LIABILITIES": "保险合同负债/总负债",
    "INSURANCE_REVENUE_TO_SIGNED_PREMIUM": "保险业务收入/签单保费",
    "NEW_BUSINESS_MARGIN": "新业务利润率", "INVESTMENT_RETURN": "投资收益率",
    "COMPREHENSIVE_INVESTMENT_RETURN": "综合投资收益率",
}
SCOPE_LABELS = {"quarter": "当季", "point": "本季度末数", "cumulative": "累计", "three_year": "近三年平均"}
BUSINESS_QUARTER_CODES = frozenset({
    "SURRENDER_RATE", "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN", "NEW_BUSINESS_VALUE",
    "RENEWAL_PREMIUM", "INSURANCE_REVENUE", "INSURANCE_CONTRACT_LIABILITY",
    "TOTAL_ASSETS", "NET_ASSETS", "INSURANCE_CONTRACT_LIABILITY_TO_TOTAL_LIABILITIES",
    "INSURANCE_REVENUE_TO_SIGNED_PREMIUM", "NEW_BUSINESS_VALUE_RATE",
})
BUSINESS_QUARTER_RATIO_CODES = frozenset({"SURRENDER_RATE", "NEW_BUSINESS_MARGIN"})
CORE_FULL_LABELS = (
    "各项非认可资产的账面价值",
    "长期股权投资的认可价值与账面价值的差额＋投资性房地产公允价值增值（扣除减值、折旧及所得税影响）",
    "递延所得税资产（由经营性亏损引起的递延所得税资产除外）",
    "计入核心一级资本的保单未来盈余",
    "银保监会规定的其他调整项目",
)


def has_values(fig: go.Figure | None) -> bool:
    if fig is None:
        return False
    for trace in fig.data:
        for field in ("y", "r", "values"):
            values = getattr(trace, field, None)
            if values is not None and any(value is not None and pd.notna(value) for value in values):
                return True
    return False


def _scope_kind(scope: object) -> str:
    text = re.sub(r"\s+", "", str(scope))
    if any(term in text for term in ("上季", "上年", "去年", "期初", "年初", "预测", "下季", "同期")):
        return "excluded"
    if "近三年" in text and "平均" in text:
        return "three_year"
    if "累计" in text:
        return "cumulative"
    if "季度末" in text or text in {"期末数", "期末"}:
        return "point"
    if "季度数" in text or "季度（末）数" in text or "当季" in text or "本季数" in text or text == "本季度":
        return "quarter"
    return "other"


def _values(frame: pd.DataFrame, code: str, period: str, scope: str | None = None) -> pd.DataFrame:
    rows = frame.loc[
        frame["指标编码"].astype(str).eq(code)
        & frame["报告期"].astype(str).eq(str(period))
    ].copy()
    if scope is not None:
        rows = rows.loc[rows["期间口径"].map(_scope_kind).eq(scope)]
    rows["数值"] = pd.to_numeric(rows["数值"], errors="coerce")
    if "披露状态" in rows:
        rows = rows.loc[~rows["披露状态"].isin(["未披露", "不适用", "无法计算", "not_disclosed", "disclosed_na"])]
    rows = rows.loc[~rows["期间口径"].map(_scope_kind).eq("excluded")]
    return rows.loc[rows["数值"].map(lambda value: pd.notna(value) and math.isfinite(value))]


def value_for(frame: pd.DataFrame, code: str, period: str, scope: str | None = None) -> float | None:
    """Return a disclosed value only; never treat an absent row as zero."""
    rows = _values(frame, code, period, scope)
    if rows.empty:
        return None
    if scope is None and len(rows) > 1:
        for preferred in ("point", "cumulative", "quarter", "three_year", "other"):
            match = rows.loc[rows["期间口径"].map(_scope_kind).eq(preferred)]
            if not match.empty:
                rows = match
                break
    unique = rows["数值"].unique()
    # Conflicting duplicates must not depend on input row order.
    return float(unique[0]) if len(unique) == 1 else None


def legacy_operating_quarter_value(frame: pd.DataFrame, code: str, period: str) -> float | None:
    """Recover old STEP3 rows whose operating-quarter scope was collapsed to period-end."""
    rows = _values(frame, code, period, "point")
    if rows.empty or "来源工作表" not in rows.columns:
        return None
    source = rows["来源工作表"].fillna("").astype(str)
    external_business = pd.Series(False, index=rows.index)
    if code in BUSINESS_QUARTER_CODES and "来源类型" in rows.columns:
        external_business = rows["来源类型"].fillna("").astype(str).isin({"外部数据集"})
    rows = rows.loc[
        source.str.contains("主要经营指标", regex=False)
        | source.str.upper().eq("OPERATING_METRICS")
        | external_business
    ]
    if rows.empty:
        return None
    values = rows["数值"].copy()
    if code in BUSINESS_QUARTER_RATIO_CODES and "来源类型" in rows.columns:
        external = rows["来源类型"].fillna("").astype(str).eq("外部数据集")
        values.loc[external] = values.loc[external] * 100.0
    unique = values.unique()
    return float(unique[0]) if len(unique) == 1 else None


def operating_quarter_value(frame: pd.DataFrame, code: str, period: str) -> float | None:
    """Read current-quarter operating data, including rows created by older converters."""
    value = value_for(frame, code, period, "quarter")
    if value is not None:
        return value
    return legacy_operating_quarter_value(frame, code, period)


def capital_value(frame: pd.DataFrame, code: str, period: str) -> float | None:
    for scope in ("point", "quarter"):
        value = value_for(frame, code, period, scope)
        if value is not None:
            return value
    return None


def complete_external_capital_detail_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the wide-table blank-as-zero convention to older STEP5 data.

    Earlier conversions omitted blank capital-detail cells altogether. Add
    chart-only zero rows for absent components when the same company's period
    has another numeric detail from the external wide table. Explicit missing
    rows and non-external sources are left as they are.
    """
    required = {"公司", "报告期", "指标编码", "期间口径", "数值", "来源类型"}
    if frame.empty or not required.issubset(frame.columns):
        return frame
    core_labels = {
        code: CORE_COMPONENT_LABELS.get(code, label)
        for codes, label in CORE_STEPS for code in codes
    }
    core_labels.update({code: label for code, label in CORE_OMITTED_STEPS})
    sections = (("core", core_labels), ("ancillary", dict(ANC_COMPONENTS)))
    additions: list[pd.Series] = []
    for (_, report_period), group in frame.groupby(["公司", "报告期"], sort=False, dropna=False):
        period = str(report_period)
        external = group.loc[
            group["来源类型"].fillna("").astype(str).isin({"外部数据集", "宽表空白推定"})
        ]
        if external.empty:
            continue
        existing_codes = set(group["指标编码"].dropna().astype(str))
        for section, labels in sections:
            present = [code for code in labels if capital_value(external, code, period) is not None]
            if not present:
                continue
            if section == "ancillary" and not any(
                capital_value(external, code, period) for code in present
            ):
                continue
            anchor = next(
                (rows.iloc[0] for scope in ("point", "quarter")
                 if not (rows := _values(external, present[0], period, scope)).empty),
                None,
            )
            if anchor is None:
                continue
            for code, label in labels.items():
                if code in existing_codes:
                    continue
                row = anchor.copy()
                row["指标编码"] = code
                row["指标名称"] = label
                row["数值"] = 0.0
                row["来源类型"] = "宽表空白推定"
                row["指标属性"] = "推定"
                row["披露状态"] = "推定零值"
                row["原始披露值"] = ""
                row["备注"] = "旧版宽表转窄表遗漏空白明细，同组有数值，图表按0处理"
                additions.append(row)
    if not additions:
        return frame
    return pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)


def three_year_value(frame: pd.DataFrame, code: str, period: str) -> float | None:
    dedicated = {
        "INVESTMENT_RETURN": "THREE_YEAR_AVG_INVESTMENT_RETURN",
        "COMPREHENSIVE_INVESTMENT_RETURN": "THREE_YEAR_AVG_COMPREHENSIVE_INVESTMENT_RETURN",
    }[code]
    value = value_for(frame, dedicated, period)
    return value if value is not None else value_for(frame, code, period, "three_year")


def period_scopes(frame: pd.DataFrame, codes: Iterable[str], periods: Iterable[str]) -> list[tuple[str, str]]:
    chosen = frame.loc[
        frame["指标编码"].isin(codes) & frame["报告期"].astype(str).isin([str(p) for p in periods])
    ]
    present = {(str(row["报告期"]), _scope_kind(row["期间口径"])) for _, row in chosen.iterrows()}
    kinds = [kind for kind in ("quarter", "point", "cumulative") if any(k == kind for _, k in present)] or ["quarter"]
    return [(str(period), kind) for period in periods for kind in kinds]


def _base_figure(height: int = 290) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        height=height, margin=dict(l=34, r=35, t=12, b=70),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color="#0C233C"), showlegend=False,
    )
    return fig


def combo_figure(
    frame: pd.DataFrame, periods: list[str], bar_code: str, line_code: str,
    *, derived_rate: bool = False, scope: str | None = None,
    axes: list[tuple[str, str]] | None = None,
) -> tuple[go.Figure, list[str], list[str]]:
    """Amount/rate combo with matched period and period-scope values."""
    if axes is None:
        axis_codes = (bar_code, line_code, "RENEWAL_PREMIUM") if derived_rate else (bar_code, line_code)
        axes = [(str(p), scope) for p in periods] if scope else period_scopes(frame, axis_codes, periods)
    labels, bars, lines, missing, invalid = [], [], [], [], []
    for period, kind in axes:
        label = f"{period}·{SCOPE_LABELS[kind]}"
        labels.append(label)
        def scoped_value(code: str) -> float | None:
            if kind == "three_year":
                return three_year_value(frame, code, period)
            if kind == "quarter" and code in BUSINESS_QUARTER_CODES:
                return operating_quarter_value(frame, code, period)
            return value_for(frame, code, period, kind)

        amount = scoped_value(bar_code)
        if derived_rate:
            premium = scoped_value("SIGNED_PREMIUM")
            renewal = scoped_value("RENEWAL_PREMIUM")
            new_business_premium = (
                None if premium is None or renewal is None else premium - renewal
            )
            if amount is not None and new_business_premium is not None and new_business_premium != 0:
                rate = amount / new_business_premium * 100
            else:
                rate = None
                if new_business_premium == 0:
                    invalid.append(f"{label}：签单保费－续期签单保费为 0，新业务价值率无法计算")
        else:
            rate = scoped_value(line_code)
        bars.append(amount)
        lines.append(rate)
        if amount is None:
            missing.append(f"{label}：{METRIC_LABELS[bar_code]}")
        if derived_rate and premium is None:
            missing.append(f"{label}：签单保费（无法计算新业务价值率）")
        if derived_rate and renewal is None:
            missing.append(f"{label}：续期签单保费（无法计算新业务价值率）")
        elif not derived_rate and rate is None:
            missing.append(f"{label}：{METRIC_LABELS[line_code]}")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=bars, marker_color=BLUE, name=METRIC_LABELS[bar_code], hovertemplate="%{x}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=lines, mode="lines+markers", line=dict(color=PINK, width=2), marker=dict(size=6), name="新业务价值率" if derived_rate else METRIC_LABELS[line_code], connectgaps=False, hovertemplate="%{x}<br>%{y:,.2f}%<extra>%{fullData.name}</extra>"), secondary_y=True)
    fig.update_layout(height=290, margin=dict(l=34, r=34, t=12, b=70), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#0C233C"), showlegend=False)
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=labels,
                     ticktext=[period for period, _ in axes],
                     automargin=True, categoryorder="array", categoryarray=labels)
    fig.update_yaxes(title_text="金额" if bar_code in {"SIGNED_PREMIUM", "NEW_BUSINESS_VALUE"} else "%", secondary_y=False, showgrid=True, gridcolor="#E7ECF5")
    fig.update_yaxes(title_text="%", secondary_y=True, showgrid=False)
    return fig, missing, invalid


def ratio_bar_figure(
    frame: pd.DataFrame,
    periods: list[str],
    metric_code: str,
) -> tuple[go.Figure, list[str], list[str]]:
    """Quarterly company-panel bar chart for the two operating ratios."""
    values: list[float | None] = []
    missing: list[str] = []
    invalid: list[str] = []
    for period in periods:
        value = operating_quarter_value(frame, metric_code, period)
        if value is None and metric_code == "INSURANCE_CONTRACT_LIABILITY_TO_TOTAL_LIABILITIES":
            liability = operating_quarter_value(frame, "INSURANCE_CONTRACT_LIABILITY", period)
            assets = operating_quarter_value(frame, "TOTAL_ASSETS", period)
            equity = operating_quarter_value(frame, "NET_ASSETS", period)
            denominator = None if assets is None or equity is None else assets - equity
            if liability is not None and denominator not in {None, 0}:
                value = liability / denominator * 100.0
            elif denominator == 0:
                invalid.append(f"{period}：总负债（总资产－净资产）为 0，比例无法计算")
        elif value is None and metric_code == "INSURANCE_REVENUE_TO_SIGNED_PREMIUM":
            revenue = operating_quarter_value(frame, "INSURANCE_REVENUE", period)
            premium = operating_quarter_value(frame, "SIGNED_PREMIUM", period)
            if revenue is not None and premium not in {None, 0}:
                value = revenue / premium * 100.0
            elif premium == 0:
                invalid.append(f"{period}：签单保费为 0，比例无法计算")
        values.append(value)
        if value is None:
            missing.append(f"{period}：{METRIC_LABELS[metric_code]}")

    fig = _base_figure()
    fig.add_trace(go.Bar(
        x=periods, y=values, marker_color=BLUE, name=METRIC_LABELS[metric_code],
        hovertemplate="%{x}<br>%{y:,.2f}%<extra>%{fullData.name}</extra>",
    ))
    fig.update_xaxes(
        tickangle=0, tickmode="array", tickvals=periods, ticktext=periods,
        automargin=True, categoryorder="array", categoryarray=periods,
    )
    fig.update_yaxes(title_text="%", ticksuffix="%", showgrid=True, gridcolor="#E7ECF5")
    return fig, missing, invalid


def surrender_figure(frame: pd.DataFrame, periods: list[str], axes: list[tuple[str, str]] | None = None) -> tuple[go.Figure, list[str]]:
    axes = axes if axes is not None else period_scopes(frame, ("SURRENDER_RATE",), periods)
    labels = [f"{p}·{SCOPE_LABELS[scope]}" for p, scope in axes]
    values = [value_for(frame, "SURRENDER_RATE", p, scope) for p, scope in axes]
    fig = _base_figure()
    fig.add_trace(go.Scatter(x=labels, y=values, mode="lines+markers", line=dict(color=BLUE, width=2), marker=dict(size=7), connectgaps=False, hovertemplate="%{x}<br>%{y:,.2f}%<extra></extra>"))
    fig.update_xaxes(tickangle=0, tickmode="array", tickvals=labels,
                     ticktext=[period for period, _ in axes],
                     automargin=True, categoryorder="array", categoryarray=labels)
    fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="#E7ECF5")
    return fig, [label for label, value in zip(labels, values) if value is None]


def radar_values(frame: pd.DataFrame, latest_period: str) -> tuple[list[float | None], list[str]]:
    values, missing = [], []
    for code, label, scope in RADAR_METRICS:
        if scope == "return":
            value = value_for(frame, code, latest_period, "quarter")
            if value is None:
                value = value_for(frame, code, latest_period, "cumulative")
            if value is None:
                value = value_for(frame, code, latest_period, "point")
        else:
            value = three_year_value(frame, code, latest_period) if scope == "three_year" else value_for(frame, code, latest_period, scope)
            if value is None and scope == "quarter":
                value = legacy_operating_quarter_value(frame, code, latest_period)
        values.append(value)
        if value is None:
            missing.append(label)
    return values, missing


def radar_figure(frame: pd.DataFrame, latest_period: str) -> tuple[go.Figure, list[str]]:
    values, missing = radar_values(frame, latest_period)
    full_labels = [label for _, label, _ in RADAR_METRICS]
    labels = [str(index) for index in range(1, len(RADAR_METRICS) + 1)]
    close_shape = not missing
    fig = _base_figure(330)
    fig.update_layout(margin=dict(l=48, r=48, t=35, b=20))
    fig.add_trace(go.Scatterpolar(
        r=values + ([values[0]] if close_shape else []),
        theta=labels + ([labels[0]] if close_shape else []),
        customdata=full_labels + ([full_labels[0]] if close_shape else []),
        mode="lines+markers", fill="toself" if not missing else "none",
        line=dict(color=BLUE, width=2), marker=dict(size=6), connectgaps=False,
        hovertemplate="%{customdata}<br>%{r:,.2f}%<extra></extra>",
    ))
    fig.update_layout(polar=dict(radialaxis=dict(ticksuffix="%", showline=True, gridcolor="#E7ECF5")))
    return fig, missing


def core_waterfall_figure(frame: pd.DataFrame, period: str) -> tuple[go.Figure | None, list[str], float | None]:
    start = capital_value(frame, "FINANCIAL_STATEMENT_NET_ASSETS", period)
    if start is None and "来源类型" in frame.columns:
        # CROSS wide tables have one net-assets column beside the core-capital
        # components; STEP5 retains its established NET_ASSETS code.
        wide_rows = frame.loc[frame["来源类型"].fillna("").astype(str).eq("外部数据集")]
        start = capital_value(wide_rows, "NET_ASSETS", period)
    end = capital_value(frame, "CORE_T1_CAPITAL", period)
    steps: list[tuple[str, float | None, str]] = []
    missing: list[str] = []
    for (codes, label), full_label in zip(CORE_STEPS, CORE_FULL_LABELS):
        components = [(code, capital_value(frame, code, period)) for code in codes]
        disclosed = [value for _, value in components if value is not None]
        value = sum(disclosed) if disclosed else None
        absent = [CORE_COMPONENT_LABELS.get(code, label) for code, item in components if item is None]
        missing.extend(absent if disclosed else [label])
        steps.append((label, value, full_label + ("（部分未披露）" if disclosed and absent else "")))
    if start is None:
        missing.insert(0, "净资产")
    if end is None:
        missing.append("核心一级资本")
    if start is None or end is None:
        return None, missing, None
    included = [value for _, value, _ in steps if value is not None]
    fig = go.Figure(go.Waterfall(
        x=["净资产", *[label + ("（未披露）" if value is None else "") for label, value, _ in steps], "核心一级资本"],
        y=[start, *[value for _, value, _ in steps], end],
        measure=["absolute", *(["relative"] * len(steps)), "absolute"],
        text=[f"{value:,.2f}" if value is not None else "未披露" for value in [start, *[value for _, value, _ in steps], end]],
        textposition="outside", textfont=dict(size=10),
        increasing=dict(marker=dict(color="#00C0AE")), decreasing=dict(marker=dict(color=PINK)),
        totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color=GREY)),
        customdata=["净资产", *[full_label for _, _, full_label in steps], "核心一级资本"],
        hovertemplate="%{customdata}<br>%{y:,.2f}<extra></extra>",
    ))
    fig.update_layout(height=440, margin=dict(l=45, r=20, t=30, b=120), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10), automargin=True)
    fig.update_yaxes(showgrid=True, gridcolor="#E7ECF5")
    omitted = [capital_value(frame, code, period) for code, _ in CORE_OMITTED_STEPS]
    difference = end - start - sum(included) - sum(value for value in omitted if value is not None)
    return fig, missing, difference


def core_omitted_nonzero(frame: pd.DataFrame, period: str) -> list[tuple[str, float]]:
    """Surface omitted items when the source discloses a nonzero amount."""
    values = [(label, capital_value(frame, code, period)) for code, label in CORE_OMITTED_STEPS]
    return [(label, value) for label, value in values if value is not None and value != 0]


def anc_detail_undisclosed(frame: pd.DataFrame, period: str) -> bool:
    """Zero-only or absent composition is not evidence of ancillary T1 detail."""
    return not any(
        value is not None and value != 0
        for code, _ in ANC_COMPONENTS
        if (value := capital_value(frame, code, period)) is not None
    )


def anc_pie_figure(frame: pd.DataFrame, period: str) -> tuple[go.Figure | None, list[str], list[str]]:
    if anc_detail_undisclosed(frame, period):
        return None, ["附属一级资本明细"], []
    values = [(label, capital_value(frame, code, period)) for code, label in ANC_COMPONENTS]
    missing = [label for label, value in values if value is None]
    negative = [label for label, value in values if value is not None and value < 0]
    positive = [(label, value) for label, value in values if value is not None and value > 0]
    # Negative components cannot form meaningful pie proportions.
    if negative or not positive:
        return None, missing, negative
    fig = go.Figure(go.Pie(
        labels=[label for label, _ in positive], values=[value for _, value in positive],
        marker=dict(colors=[COMPONENT_COLORS[[name for _, name in ANC_COMPONENTS].index(label)] for label, _ in positive]),
        textinfo="percent", hovertemplate="%{label}<br>%{value:,.2f}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
    return fig, missing, negative


def anc_stack_figure(frame: pd.DataFrame, periods: list[str]) -> tuple[go.Figure, list[str]]:
    selected = periods[-3:]
    fig = _base_figure(340)
    fig.update_layout(barmode="relative", margin=dict(l=45, r=15, t=10, b=40), showlegend=False)
    undisclosed = {period for period in selected if anc_detail_undisclosed(frame, period)}
    missing = [f"{period}：附属一级资本明细" for period in selected if period in undisclosed]
    for index, (code, label) in enumerate(ANC_COMPONENTS):
        values = [None if period in undisclosed else capital_value(frame, code, period) for period in selected]
        missing.extend(f"{period}：{label}" for period, value in zip(selected, values)
                       if period not in undisclosed and value is None)
        fig.add_trace(go.Bar(x=selected, y=values, name=label, marker_color=COMPONENT_COLORS[index], hovertemplate="%{x}<br>%{y:,.2f}<extra>%{fullData.name}</extra>"))
    fig.update_yaxes(showgrid=True, gridcolor="#E7ECF5")
    return fig, missing


def surrender_comparison_figure(
    frame: pd.DataFrame, companies: list[str], periods: list[str],
    colors: dict[str, str], *, scope: str, highlight_company: str = "无",
) -> tuple[go.Figure, dict[str, list[str]]]:
    """One company per line with a common scope and explicit gaps."""
    fig = _base_figure(380)
    missing = {}
    for company in companies:
        company_frame = frame.loc[frame["公司"].astype(str).eq(company)]
        values = [value_for(company_frame, "SURRENDER_RATE", period, scope) for period in periods]
        gaps = [period for period, value in zip(periods, values) if value is None]
        if gaps:
            missing[company] = gaps
        fig.add_trace(go.Scatter(
            x=periods, y=values, name=company, mode="lines+markers",
            line=dict(color=colors[company], width=3 if company == highlight_company else 2),
            marker=dict(size=6), connectgaps=False,
            customdata=[SCOPE_LABELS[scope]] * len(periods),
            hovertemplate="%{x}<br>%{customdata}<br>%{y:.2f}%<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(margin=dict(l=50, r=20, t=20, b=40), hovermode="x unified")
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=periods, title=None)
    fig.update_yaxes(ticksuffix="%", title="综合退保率", gridcolor="#E7ECF5", zeroline=True)
    return fig, missing


def compact_company_figure(fig: go.Figure, company_count: int) -> go.Figure:
    """Match the existing page-width company panels without a fixed bar width."""
    font_size = 8 if company_count >= 8 else 9 if company_count >= 5 else 10
    kind = fig.data[0].type
    fig.update_layout(
        height=330, autosize=True, margin=dict(l=4, r=4, t=24, b=30),
        font=dict(family="Microsoft YaHei", size=font_size, color="#0C233C"),
        bargap=0.5, showlegend=False,
    )
    if kind == "scatterpolar":
        fig.update_layout(margin=dict(l=45, r=45, t=35, b=35))
        labels = list(fig.data[0].theta)
        short = ["净资产<br>收益率", "总资产<br>收益率", "当季资产<br>收益率",
                 "当季综合<br>收益率", "三年平均<br>投资收益率", "三年平均<br>综合收益率"]
        fig.update_layout(polar=dict(
            angularaxis=dict(tickmode="array", tickvals=labels[:6], ticktext=short, tickfont=dict(size=font_size)),
            radialaxis=dict(showticklabels=company_count <= 4),
        ))
        return fig
    if kind == "pie":
        fig.update_traces(textfont_size=font_size)
        return fig
    fig.update_yaxes(title=None, showticklabels=False, showgrid=False,
                     zeroline=True, zerolinecolor="#0C233C", zerolinewidth=1, automargin=False)
    fig.update_xaxes(title=None, tickfont=dict(size=font_size), tickangle=0, automargin=False)
    if kind == "waterfall":
        fig.update_xaxes(tickmode="array", tickvals=list(fig.data[0].x),
                         ticktext=[str(i) for i in range(len(fig.data[0].x))])
        fig.update_traces(textposition="none" if company_count >= 5 else "outside", selector=dict(type="waterfall"))
    elif len(fig.data) == 2 and kind == "bar":
        fig.data[0].update(texttemplate="%{y:.1f}", textposition="auto", textfont_size=font_size)
        fig.data[1].update(mode="lines+markers+text", texttemplate="%{y:.1f}%",
                           textposition="top center", textfont_size=font_size)
    return fig
