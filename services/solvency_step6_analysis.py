from __future__ import annotations

import re
from collections.abc import Iterable

import altair as alt
import pandas as pd

from .solvency_navigation import KPMG_DEFAULT_COLORS
from .solvency_normalizer import STANDARD_COLUMNS, standardize_uploaded_frame


COMPANY_REPORT = "公司报告"
INDUSTRY_REPORT = "行业报告"
INDUSTRY_TOTAL = "行业合计"
VISUALIZATION_EXCLUDED_MODULES = ("数据质量", "勾稽检查")
NO_COMPANY_TYPE_FILTER = "不按公司类型筛选"

CHART_TYPES = (
    "簇状柱状图",
    "折线图",
    "带直线和数据标记的散点图",
    "散点图",
    "横向条形图",
)


def format_chart_value(
    value: object,
    unit: object = "",
    data_type: object = "",
    decimals: int | None = None,
) -> str:
    """Format visible chart labels according to the metric's semantic unit."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return ""

    number = float(numeric)
    unit_text = str(unit or "").strip()
    type_text = str(data_type or "").strip()
    is_amount = type_text == "金额" or unit_text in {"元", "万元", "亿元"}
    is_percent = unit_text in {"%", "％"} or type_text == "百分比"
    is_multiple = unit_text == "倍"
    is_count = type_text == "数量" or unit_text in {"人", "家", "个", "笔"}

    if is_amount:
        precision = 2 if decimals is None else max(0, int(decimals))
        formatted = f"{abs(number):,.{precision}f}"
        return f"({formatted})" if number < 0 else formatted
    if is_percent:
        precision = 1 if decimals is None else max(0, int(decimals))
        return f"{number:,.{precision}f}%"
    if is_multiple:
        precision = 2 if decimals is None else max(0, int(decimals))
        return f"{number:,.{precision}f}倍"
    if is_count:
        suffix = unit_text if unit_text else ""
        return f"{number:,.0f}{suffix}"

    precision = 2 if decimals is None else max(0, int(decimals))
    formatted = f"{number:,.{precision}f}"
    if decimals is None:
        formatted = formatted.rstrip("0").rstrip(".")
    return f"{formatted}{unit_text}" if unit_text else formatted


def _axis_number_format(units: list[str], data_types: list[str]) -> str:
    unit_set = set(units)
    type_set = set(data_types)
    if unit_set.intersection({"%", "％"}) or "百分比" in type_set:
        return ",.1f"
    if unit_set.intersection({"人", "家", "个", "笔"}) or "数量" in type_set:
        return ",.0f"
    return ",.2f"


def visualization_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep audit rows in STEP5 while excluding them from visualization choices."""
    if frame.empty:
        return frame.copy()
    excluded = pd.Series(False, index=frame.index)
    for column in ("一级模块", "二级模块"):
        if column in frame.columns:
            text = frame[column].fillna("").astype(str).str.strip()
            excluded |= text.isin(VISUALIZATION_EXCLUDED_MODULES)
            excluded |= text.str.contains("数据质量|勾稽检查", regex=True, na=False)
    for column in ("数据类型", "指标属性"):
        if column in frame.columns:
            text = frame[column].fillna("").astype(str).str.strip()
            excluded |= text.eq("校验") | text.str.contains("勾稽检查", na=False)
    if "指标编码" in frame.columns:
        codes = frame["指标编码"].fillna("").astype(str).str.upper()
        excluded |= codes.str.contains("CHECK", regex=False, na=False)
    return frame.loc[~excluded].copy()


def prepare_analysis_frame(
    frame: pd.DataFrame | None,
    report_profile_id: str = "",
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Validate the common schema and isolate the requested report profile."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS), ()

    result = standardize_uploaded_frame(frame)
    text_columns = [
        "公司",
        "公司类型",
        "同业分类",
        "报告类型",
        "报告期",
        "一级模块",
        "二级模块",
        "指标编码",
        "指标名称",
        "期间口径",
        "单位",
    ]
    for column in text_columns:
        result[column] = result[column].fillna("").astype(str).str.strip()

    profile_id = str(report_profile_id or "").strip()
    skipped: tuple[str, ...] = ()
    if profile_id:
        result.loc[result["报告类型"] == "", "报告类型"] = profile_id
        mismatched = sorted(
            value
            for value in result["报告类型"].unique().tolist()
            if value and value != profile_id
        )
        skipped = tuple(mismatched)
        result = result[result["报告类型"] == profile_id].copy()

    result["数值"] = pd.to_numeric(result["数值"], errors="coerce")
    required = ["公司", "指标编码", "指标名称", "报告期", "数值"]
    result = result.dropna(subset=["数值"])
    for column in required[:-1]:
        result = result[result[column] != ""]
    result = visualization_metric_frame(result)
    return result.reset_index(drop=True), skipped


def is_industry_total(frame: pd.DataFrame) -> pd.Series:
    company = frame["公司"].fillna("").astype(str).str.strip()
    company_type = frame["公司类型"].fillna("").astype(str).str.strip()
    company_code = frame["公司统一编码"].fillna("").astype(str).str.strip()
    return (
        company.eq(INDUSTRY_TOTAL)
        | company_type.eq(INDUSTRY_TOTAL)
        | company_code.str.startswith("INDUSTRY_")
    )


def report_scope_frame(frame: pd.DataFrame, report_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if report_mode == COMPANY_REPORT:
        return frame.loc[~is_industry_total(frame)].copy()
    return frame.copy()


def nonblank_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    values = frame[column].fillna("").astype(str).str.strip()
    return sorted(value for value in values.unique().tolist() if value)


def companies_for_quick_selection(
    frame: pd.DataFrame,
    selected_classification: str = NO_COMPANY_TYPE_FILTER,
) -> list[str]:
    """Return comparison companies for the annual-report style quick selector."""
    if frame.empty:
        return []
    scoped = frame
    selection = str(selected_classification or "").strip()
    if selection and selection != NO_COMPANY_TYPE_FILTER:
        if "同业分类" not in scoped.columns:
            return []
        peer_group = scoped["同业分类"].fillna("").astype(str).str.strip()
        scoped = scoped.loc[peer_group.eq(selection)]
    return nonblank_values(scoped, "公司")


def _period_sort_key(value: str) -> tuple[int, int, str]:
    text = str(value or "").strip()
    quarter = re.search(r"(?P<year>20\d{2})\s*[-_/年]?\s*[Qq第]?\s*(?P<quarter>[1-4])", text)
    if quarter:
        return int(quarter.group("year")), int(quarter.group("quarter")), text
    year = re.search(r"20\d{2}", text)
    if year:
        return int(year.group()), 5, text
    return 9999, 9, text


def sort_report_periods(values: Iterable[object]) -> list[str]:
    unique = {str(value).strip() for value in values if str(value).strip()}
    return sorted(unique, key=_period_sort_key)


def default_companies(report_mode: str, companies: Iterable[str]) -> list[str]:
    options = [str(company) for company in companies if str(company).strip()]
    if report_mode == COMPANY_REPORT:
        return options[:1]
    actual = [company for company in options if company != INDUSTRY_TOTAL]
    selected = actual[:5]
    if INDUSTRY_TOTAL in options:
        selected.append(INDUSTRY_TOTAL)
    return selected


def default_periods(periods: Iterable[str], limit: int = 8) -> list[str]:
    ordered = sort_report_periods(periods)
    return ordered[-limit:]


def filter_analysis_frame(
    frame: pd.DataFrame,
    *,
    level_one: str = "",
    level_two: str = "",
    company_types: Iterable[str] = (),
    peer_groups: Iterable[str] = (),
    companies: Iterable[str] = (),
    periods: Iterable[str] = (),
    metric_code: str = "",
    period_scope: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    filters = {
        "一级模块": level_one,
        "二级模块": level_two,
        "指标编码": metric_code,
        "期间口径": period_scope,
    }
    for column, value in filters.items():
        if value:
            result = result[result[column] == value]
    collection_filters = {
        "公司类型": list(company_types),
        "同业分类": list(peer_groups),
        "公司": list(companies),
        "报告期": list(periods),
    }
    for column, values in collection_filters.items():
        if values:
            result = result[result[column].isin(values)]
    return result.copy()


def navigation_scope_frame(
    frame: pd.DataFrame,
    level_one: str = "",
    level_two: str = "全部",
    *,
    print_all_label: str = "一键显示全部（打印/导出）",
) -> pd.DataFrame:
    """Limit visual metrics to the current report-navigation scope.

    The print-all entry intentionally means the full visualization dataset.  The
    synthetic ``全部`` second level likewise keeps every metric under the selected
    first-level module.
    """
    if frame.empty:
        return frame.copy()
    first = str(level_one or "").strip()
    second = str(level_two or "").strip()
    if not first or first == print_all_label:
        return frame.copy()
    result = frame
    if "一级模块" in result.columns:
        result = result[result["一级模块"].fillna("").astype(str).str.strip().eq(first)]
    if second and second != "全部" and "二级模块" in result.columns:
        result = result[result["二级模块"].fillna("").astype(str).str.strip().eq(second)]
    return result.copy()


def metric_options(frame: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    if frame.empty:
        return [], {}
    metrics = (
        frame[["指标编码", "指标名称"]]
        .drop_duplicates()
        .sort_values(["指标名称", "指标编码"])
    )
    labels: list[str] = []
    lookup: dict[str, str] = {}
    for row in metrics.itertuples(index=False):
        label = f"{row.指标名称}（{row.指标编码}）"
        labels.append(label)
        lookup[label] = row.指标编码
    return labels, lookup


def _staggered_label_offsets(
    chart_data: pd.DataFrame,
    x_field: str,
) -> pd.Series:
    """Stagger labels only when values at the same x position are close."""
    offsets = pd.Series(-12, index=chart_data.index, dtype="int64")
    for _, group in chart_data.groupby(x_field, sort=False, dropna=False):
        numeric = pd.to_numeric(group["数值"], errors="coerce").dropna().sort_values()
        if len(numeric) < 2:
            continue
        span = float(numeric.max() - numeric.min())
        magnitude = max(float(numeric.abs().max()), 1.0)
        threshold = max(span * 0.06, magnitude * 0.01, 1e-9)
        clusters: list[list[object]] = []
        current: list[object] = []
        previous: float | None = None
        for index, value in numeric.items():
            number = float(value)
            if previous is None or number - previous <= threshold:
                current.append(index)
            else:
                clusters.append(current)
                current = [index]
            previous = number
        if current:
            clusters.append(current)
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            for position, index in enumerate(cluster):
                distance = 12 + 16 * (position // 2)
                offsets.loc[index] = -distance if position % 2 == 0 else distance
    return offsets


def build_comparison_chart(
    frame: pd.DataFrame,
    chart_type: str,
    period_order: Iterable[str],
    show_labels: bool = True,
    decimals: int | None = None,
    layout_mode: str = "以报告期为横轴",
    legend_label_map: dict[str, str] | None = None,
    legend_color_map: dict[str, str] | None = None,
    y_axis_title: str = "",
    transparent: bool = False,
    show_average: bool = False,
    average_color: str = "#ED2124",
    highlight_company: str = "",
    avoid_label_overlap: bool = False,
) -> alt.Chart:
    if chart_type not in CHART_TYPES:
        raise ValueError(f"不支持的图表类型：{chart_type}")
    chart_data = frame.copy()
    chart_data["数值"] = pd.to_numeric(chart_data["数值"], errors="coerce")
    chart_data = chart_data.dropna(subset=["数值"])
    rename_map = {str(key): str(value) for key, value in (legend_label_map or {}).items()}
    color_map = {str(key): str(value) for key, value in (legend_color_map or {}).items()}
    company_axis = layout_mode == "以公司为横轴"
    legend_source = "报告期" if company_axis else "公司"
    x_field = "公司" if company_axis else "报告期"
    offset_field = "报告期" if company_axis else "公司"
    series_detail = "报告期" if company_axis else "公司"
    chart_data["图例项"] = chart_data[legend_source].map(
        lambda item: rename_map.get(str(item), str(item))
    )
    periods = list(period_order)
    units = nonblank_values(chart_data, "单位")
    data_types = nonblank_values(chart_data, "数据类型")
    axis_format = _axis_number_format(units, data_types)
    y_title = y_axis_title.strip() or ("数值" if not units else f"数值（{'、'.join(units)}）")
    chart_data["数据标签"] = chart_data.apply(
        lambda row: format_chart_value(
            row["数值"],
            row.get("单位", ""),
            row.get("数据类型", ""),
            decimals,
        ),
        axis=1,
    )
    tooltip = [
        alt.Tooltip("公司:N"),
        alt.Tooltip("公司类型:N"),
        alt.Tooltip("报告期:N"),
        alt.Tooltip("指标名称:N"),
        alt.Tooltip("期间口径:N"),
        alt.Tooltip("数据标签:N", title="数值"),
        alt.Tooltip("单位:N"),
    ]
    legend_domain: list[str] = []
    legend_range: list[str] = []
    for index, item in enumerate(dict.fromkeys(chart_data[legend_source].astype(str))):
        legend_domain.append(rename_map.get(item, item))
        legend_range.append(color_map.get(item, KPMG_DEFAULT_COLORS[index % len(KPMG_DEFAULT_COLORS)]))
    base = alt.Chart(chart_data).encode(
        color=alt.Color(
            "图例项:N",
            title="公司" if legend_source == "公司" else "报告期",
            scale=alt.Scale(domain=legend_domain, range=legend_range),
        ),
        tooltip=tooltip,
    )
    value_text = alt.Text("数据标签:N")
    if chart_type == "簇状柱状图":
        bars = base.mark_bar().encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None, title=x_field),
            xOffset=alt.XOffset(f"{offset_field}:N", sort=periods if offset_field == "报告期" else None),
            y=alt.Y(
                "数值:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=axis_format),
            ),
        )
        labels = alt.Chart(chart_data).mark_text(
            dy=-7,
            align="center",
            baseline="bottom",
            fontSize=12,
            fontWeight="bold",
        ).encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None),
            xOffset=alt.XOffset(f"{offset_field}:N", sort=periods if offset_field == "报告期" else None),
            y=alt.Y("数值:Q", scale=alt.Scale(zero=False)),
            text=value_text,
            color=alt.value("#1f2937"),
        )
        chart = bars + labels if show_labels else bars
    elif chart_type == "横向条形图":
        bars = base.mark_bar().encode(
            y=alt.Y(f"{x_field}:N", title=x_field, sort="-x"),
            yOffset=alt.YOffset(f"{offset_field}:N", sort=periods if offset_field == "报告期" else None),
            x=alt.X(
                "数值:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=axis_format),
            ),
        )
        labels = alt.Chart(chart_data).mark_text(
            dx=7,
            align="left",
            baseline="middle",
            fontSize=12,
            fontWeight="bold",
        ).encode(
            y=alt.Y(f"{x_field}:N", sort="-x"),
            yOffset=alt.YOffset(f"{offset_field}:N", sort=periods if offset_field == "报告期" else None),
            x=alt.X("数值:Q", scale=alt.Scale(zero=False)),
            text=value_text,
            color=alt.value("#1f2937"),
        )
        chart = bars + labels if show_labels else bars
    elif chart_type == "散点图":
        points = base.mark_point(filled=True, opacity=0.9).encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None, title=x_field),
            y=alt.Y(
                "数值:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=axis_format),
            ),
            size=alt.value(90),
            shape=alt.value("circle"),
        )
        labels = alt.Chart(chart_data).mark_text(
            dy=-12,
            fontSize=11,
            fontWeight="bold",
        ).encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None),
            y=alt.Y("数值:Q", scale=alt.Scale(zero=False)),
            text=value_text,
            detail=alt.Detail(f"{series_detail}:N"),
            color=alt.value("#1f2937"),
        )
        chart = points + labels if show_labels else points
    else:
        point = chart_type == "带直线和数据标记的散点图"
        highlight = str(highlight_company or "").strip()
        line_size = alt.value(2)
        point_size = alt.value(85 if point else 45)
        if highlight and series_detail == "公司":
            highlight_predicate = alt.FieldEqualPredicate(field="公司", equal=highlight)
            line_size = alt.condition(highlight_predicate, alt.value(4), alt.value(2))
            point_size = alt.condition(
                highlight_predicate,
                alt.value(130 if point else 90),
                alt.value(85 if point else 45),
            )
        line = base.mark_line(
            interpolate="linear",
        ).encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None, title=x_field),
            y=alt.Y(
                "数值:Q",
                title=y_title,
                scale=alt.Scale(zero=False),
                axis=alt.Axis(format=axis_format),
            ),
            detail=alt.Detail(f"{series_detail}:N"),
            size=line_size,
        )
        points = base.mark_point(filled=True).encode(
            x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None),
            y=alt.Y("数值:Q", scale=alt.Scale(zero=False)),
            size=point_size,
            shape=alt.value("circle"),
        )
        chart = line + points
        if show_labels:
            if avoid_label_overlap:
                chart_data["_标签偏移"] = _staggered_label_offsets(chart_data, x_field)
                label_layers: list[alt.Chart] = []
                for dy in sorted(chart_data["_标签偏移"].unique()):
                    label_rows = chart_data[chart_data["_标签偏移"].eq(dy)]
                    label_layers.append(
                        alt.Chart(label_rows).mark_text(
                            dy=int(dy),
                            baseline="bottom" if dy < 0 else "top",
                            fontSize=10,
                            fontWeight="bold",
                        ).encode(
                            x=alt.X(
                                f"{x_field}:N",
                                sort=periods if x_field == "报告期" else None,
                            ),
                            y=alt.Y("数值:Q", scale=alt.Scale(zero=False)),
                            text=value_text,
                            detail=alt.Detail(f"{series_detail}:N"),
                            color=alt.value("#1f2937"),
                        )
                    )
                chart = alt.layer(line, points, *label_layers)
            else:
                labels = alt.Chart(chart_data).mark_text(
                    dy=-12,
                    fontSize=11,
                    fontWeight="bold",
                ).encode(
                    x=alt.X(f"{x_field}:N", sort=periods if x_field == "报告期" else None),
                    y=alt.Y("数值:Q", scale=alt.Scale(zero=False)),
                    text=value_text,
                    detail=alt.Detail(f"{series_detail}:N"),
                    color=alt.value("#1f2937"),
                )
                chart = line + points + labels
    if show_average:
        average = alt.Chart(chart_data).mark_rule(
            color=average_color,
            strokeDash=[7, 5],
            strokeWidth=2,
        ).encode(y=alt.Y("mean(数值):Q"))
        chart = chart + average
    background = "transparent" if transparent else "white"
    return (
        chart.properties(height=430, background=background)
        .configure_view(fill="transparent" if transparent else "white", strokeOpacity=0)
        .interactive(bind_y=False)
    )
