from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from dashboard_components import (
    company_detail_rows,
    render_report_analysis,
    render_report_back_cover,
    render_report_cover,
    render_report_footnote,
    render_report_notes_editor,
)
from services.llm_config import model_request_parameters, normalize_model_id
from services.llm_http import post_json_with_retry
from services.solvency_navigation import (
    COMPANY_NAVIGATION,
    KPMG_CHART_COLORS,
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
    CAPITAL_AMOUNT_COMBO,
    EFFECT_DIVERGING,
    MARKET_CREDIT_MATRIX,
    SINGLE_METRIC_TREND,
    SOLVENCY_MATRIX,
    chart_plan_for,
)
from services.solvency_step7_charts import (
    build_capital_amount_combo,
    build_company_bar_trend_chart,
    build_effect_diverging_chart,
    build_matrix_chart,
    build_single_metric_trend_chart,
    build_single_metric_trend_charts,
    report_period_color_map,
)
from services.solvency_report_notes import company_notes_template


PICTURE_DIR = Path(__file__).resolve().parent / "picture"
STEP7_CHART_TYPE = "内置分析方案"
ALL_COMPANY_TYPES = "全部"
DEFAULT_SORT_LABEL = "默认（按列表原始顺序）"
SORT_DESCENDING = "降序（从大到小）"
SORT_ASCENDING = "升序（从小到大）"


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
    """Use KPMG Primary colors first and reserve KPMG Pink for tracking."""
    ordered = list(dict.fromkeys(str(company) for company in companies if str(company).strip()))
    highlight = str(highlight_company or "").strip()
    has_highlight = highlight in ordered
    palette = (
        [color for color in KPMG_CHART_COLORS if color.upper() != "#FD349C"]
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
        colors[highlight] = "#FD349C"
    return colors


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


def _selected_chart_names(frame: pd.DataFrame) -> list[str]:
    first = st.session_state.get("company_nav_level_one", "")
    available = set(frame["指标编码"].fillna("").astype(str))
    if first == PRINT_ALL_LABEL:
        return list(dict.fromkeys(
            entry.chart_name
            for entry in COMPANY_NAVIGATION
            if any(code in available for code in entry.metric_codes)
        ))
    return resolve_chart_selection(
        first,
        st.session_state.get("company_nav_level_two", "全部"),
        st.session_state.get("company_nav_chart", ""),
        available_codes=available,
    )


def _report_style() -> None:
    st.markdown(
        """
        <style>
        .solvency-report-title {
            color:#00338D; font-size:28px; font-weight:800;
            border-bottom:2px solid #00338D; padding-bottom:8px; margin:6px 0 18px;
        }
        .solvency-module-title {
            color:#00338D; font-size:22px; font-weight:750; margin:18px 0 8px;
        }
        .solvency-analysis-note {
            background:#F2F6FC; border-left:4px solid #00338D;
            border-radius:4px; padding:9px 13px; margin:6px 0 0; color:#243B53;
        }
        .solvency-analysis-spacer {height:16px; width:100%;}
        .solvency-chart-legend {
            display:flex; flex-wrap:wrap; justify-content:flex-end;
            align-items:center; gap:10px 18px;
            margin:2px 10px 8px 0; color:#0C233C; font-size:13px;
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
        .solvency-note-default {color:#0A1F5C; font-size:13px;}
        .solvency-note-custom {color:#1E49E2; font-size:13px; font-weight:600;}
        .solvency-report-footnote {color:#777; font-size:12px; font-style:italic; margin:4px 0 16px;}
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
        @media print {
            [data-testid="stHeader"], [data-testid="stSidebar"],
            [data-testid="stToolbar"], [data-testid="collapsedControl"],
            [data-testid="stExpander"], div[role="tablist"], h1,
            .platform-page-heading, .solvency-report-title,
            .solvency-no-print {display:none!important;}
            [data-testid="stElementContainer"]:has(.platform-page-heading),
            [data-testid="stElementContainer"]:has(.solvency-report-title) {display:none!important;}
            [class*="st-key-s7_report_module_"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-s7_report_module_"]:has(.solvency-grouped-trend) {
                break-inside:auto!important; page-break-inside:auto!important;
            }
            [class*="st-key-s7_trend_group_"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                overflow:visible!important;
            }
            [class*="st-key-s7_report_module_"] [data-testid="stVegaLiteChart"] {
                break-inside:avoid-page!important; page-break-inside:avoid!important;
                max-width:100%!important;
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
                height:190.5mm!important; margin:0!important; padding:0!important;
                break-inside:avoid-page!important; page-break-inside:avoid!important;
            }
            html.solvency-print-mode-widescreen [data-testid="stVerticalBlock"]:has(.solvency-print-cover) {
                gap:0!important; margin:0!important; padding:0!important;
            }
            html.solvency-print-mode-widescreen [class*="st-key-s7_report_module_"] {
                width:100%!important; max-width:100%!important;
                min-height:0!important; padding:0!important;
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
            .solvency-module-title {font-size:30px!important; margin-top:0!important;}
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
    result["数值"] = result.apply(
        lambda row: (
            pd.to_numeric(pd.Series([row["数值"]]), errors="coerce").iloc[0]
            * source_yuan.get(str(row.get("单位", "")).strip(), target_yuan)
            / target_yuan
        ),
        axis=1,
    )
    amount_mask = result["单位"].astype(str).isin(source_yuan)
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
        return "请先在 Step 1 的“大模型接口设置”中完整填写接口地址、模型名称和 API Key。", True
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


def _render_report_metric(
    frame: pd.DataFrame,
    code: str,
    *,
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
) -> None:
    metric_frame = filter_analysis_frame(frame, metric_code=code, periods=periods)
    if metric_frame.empty:
        st.caption(f"{_metric_title(frame, code)}：当前筛选范围无数据。")
        return
    metric_frame = _convert_unit(metric_frame, unit_mode)
    metric_name = str(metric_frame.iloc[0]["指标名称"])
    st.markdown(f"#### {metric_name}")
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
        if code in {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}:
            legend_items.append(("#ED2124", "监管下限", "dash"))
        _render_chart_legend(legend_items)
        chart = build_company_bar_trend_chart(
            metric_frame,
            code,
            ordered_periods,
            highlight_company,
        )
        st.altair_chart(chart, width="stretch", key=f"{key_prefix}_{code}_company_panels")
    else:
        expected_count = metric_frame["公司"].nunique() * len(ordered_periods)
        disclosed_count = metric_frame.drop_duplicates(["公司", "报告期"]).shape[0]
        external_legend = []
        if disclosed_count < expected_count:
            external_legend.append(("#B8BDC7", "未披露", "square"))
        if code in {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}:
            external_legend.append(("#ED2124", "监管下限", "dash"))
        if external_legend:
            _render_chart_legend(external_legend)
        charts = build_single_metric_trend_charts(
            metric_frame,
            code,
            ordered_periods,
            _company_color_map(metric_frame["公司"], highlight_company),
            highlight_company,
        )
        if len(charts) > 1:
            st.markdown(
                "<div class='solvency-grouped-trend' aria-hidden='true'></div>",
                unsafe_allow_html=True,
            )
        for group_index, chart in enumerate(charts, start=1):
            with st.container(key=f"s7_trend_group_{key_prefix}_{code}_{group_index}"):
                st.altair_chart(
                    chart,
                    width="stretch",
                    key=f"{key_prefix}_{code}_trend_{group_index}",
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


def _render_combination_analysis(
    frame: pd.DataFrame,
    chart_name: str,
    *,
    periods: list[str],
    unit_mode: str,
    highlight_company: str,
    key_prefix: str,
) -> None:
    """Render an approved multi-metric plan selected by chart name."""
    plan = chart_plan_for(chart_name)
    converted = _convert_unit(frame, unit_mode)
    company_colors = _company_color_map(converted["公司"], highlight_company)
    if plan.kind == CAPITAL_AMOUNT_COMBO:
        st.caption("堆叠柱显示四级资本构成，KPMG Pink 折线显示实际资本总额；占比不足4%的薄色块请悬浮查看。")
        capital_codes = [
            "CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL",
            "ANC_T2_CAPITAL", "ACTUAL_CAPITAL",
        ]
        expected_count = converted["公司"].nunique() * len(periods) * len(capital_codes)
        disclosed_count = converted[
            converted["指标编码"].astype(str).isin(capital_codes)
        ].drop_duplicates(["公司", "报告期", "指标编码"]).shape[0]
        capital_legend = [
            ("#00338D", "核心一级资本", "square"),
            ("#1E49E2", "核心二级资本", "square"),
            ("#7213EA", "附属一级资本", "square"),
            ("#00B8F5", "附属二级资本", "square"),
            ("#FD349C", "实际资本总额", "line"),
        ]
        if disclosed_count < expected_count:
            capital_legend.append(("#B8BDC7", "未披露", "square"))
        _render_chart_legend(capital_legend)
        st.altair_chart(
            build_capital_amount_combo(converted, periods),
            width="stretch",
            key=f"{key_prefix}_capital_amount",
        )
    elif plan.kind == EFFECT_DIVERGING:
        codes = metric_codes_for_chart(chart_name)
        if codes:
            st.caption("按公司展示抵减效应的跨期方向与幅度，零线用于区分正向增加和负向抵减。")
            st.altair_chart(
                build_effect_diverging_chart(converted, codes[0], periods),
                width="stretch",
                key=f"{key_prefix}_{codes[0]}_effect",
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
        st.altair_chart(chart, width="stretch", key=f"{key_prefix}_solvency_matrix")
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
        st.altair_chart(chart, width="stretch", key=f"{key_prefix}_market_credit_matrix")
        st.caption("深蓝虚线为样本中位数，用于识别市场风险和信用风险同时偏高的公司。")


def show_step_7_solvency(data: pd.DataFrame) -> None:
    """Company report page following the annual-platform report workflow."""
    _report_style()
    st.markdown("<div class='solvency-report-title'>公司级偿付能力对标报告</div>", unsafe_allow_html=True)
    if data is None or data.empty:
        st.info("请先在 Step5 确认集成数据，或在 Step6 上传集成表。")
        return
    frame = company_detail_rows(data).copy()
    if frame.empty:
        st.info("当前数据没有公司明细记录。")
        return
    frame["数值"] = pd.to_numeric(frame["数值"], errors="coerce")
    frame = frame.dropna(subset=["数值"])
    all_periods = sort_report_periods(frame["报告期"])
    if not all_periods:
        st.info("当前数据没有可用报告期。")
        return

    notes = render_report_notes_editor(
        title="公司内容分析与注释输入",
        key_prefix="step7",
        template=company_notes_template(),
    )
    available_codes = set(frame["指标编码"].fillna("").astype(str))
    available_chart_names = list(dict.fromkeys(
        entry.chart_name
        for entry in COMPANY_NAVIGATION
        if any(code in available_codes for code in entry.metric_codes)
    ))
    image_overrides: dict[str, dict[str, object]] = {}
    with st.expander("公司级图表设置与图片覆盖", expanded=False, icon=":material/tune:"):
        c0, c1, c2, c3, c4 = st.columns([1, 2, 1, 1, 1])
        peer_groups = nonblank_values(frame, "同业分类")
        type_options = [ALL_COMPANY_TYPES, *peer_groups]
        st.session_state.setdefault("s7_company_types", [ALL_COMPANY_TYPES])
        _valid_state("s7_company_types", type_options, multiple=True)
        with c0:
            selected_types_raw = st.multiselect(
                "公司类型",
                type_options,
                key="s7_company_types",
                help="沿用偿付能力平台的‘同业分类’字段进行筛选。",
            )
        if not selected_types_raw or ALL_COMPANY_TYPES in selected_types_raw:
            company_source = frame
        else:
            company_source = frame[
                frame["同业分类"].astype(str).isin(selected_types_raw)
            ].copy()
        all_companies = _ordered_companies(company_source)
        type_signature = tuple(selected_types_raw)
        if st.session_state.get("_s7_previous_company_types") != type_signature:
            st.session_state["s7_companies"] = all_companies
            st.session_state["_s7_previous_company_types"] = type_signature
        _valid_state("s7_companies", all_companies, multiple=True)
        with c1:
            selected_companies = st.multiselect(
                "展示公司", all_companies, key="s7_companies", placeholder="选择一家或多家公司"
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
                help="使用 Step 1/2 已配置的大模型接口，为当前图表生成一句同业对标点评。",
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
    scoped["_s7_company_order"] = pd.Categorical(
        scoped["公司"], categories=selected_companies, ordered=True
    )
    scoped = scoped.sort_values(["_s7_company_order", "报告期"]).drop(columns="_s7_company_order")

    chart_names = _selected_chart_names(scoped)
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
        with st.container(key=f"s7_report_module_{chart_index}"):
            st.markdown(
                f"<div class='solvency-module-title'>"
                f"{level_one} · {level_two} · {chart_name}</div>",
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
                if plan.kind in {SINGLE_METRIC_TREND, COMPANY_BAR_TREND}:
                    for code in metric_codes_for_chart(chart_name):
                        _render_report_metric(
                            scoped,
                            code,
                            periods=selected_periods,
                            show_labels=show_labels,
                            decimals=decimals,
                            unit_mode=unit_mode,
                            transparent=transparent,
                            show_average=show_average,
                            highlight_company=highlight_company,
                            enable_ai=enable_ai and ai_data_consent,
                            key_prefix=f"s7_{chart_index}",
                            company_panels=plan.kind == COMPANY_BAR_TREND,
                        )
                else:
                    _render_combination_analysis(
                        scoped,
                        chart_name,
                        periods=selected_periods,
                        unit_mode=unit_mode,
                        highlight_company=highlight_company,
                        key_prefix=f"s7_{chart_index}",
                    )
            render_report_footnote(chart_note)

    if print_all:
        render_report_back_cover(picture_dir=PICTURE_DIR)

    with st.expander("查看报告底层数据"):
        st.dataframe(scoped, width="stretch", hide_index=True)
