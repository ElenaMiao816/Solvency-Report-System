from __future__ import annotations

import base64
import hashlib
import html
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from services.solvency_financing_analysis import summarize_financing
from services.solvency_navigation import KPMG_CATEGORIES
from services.solvency_report_notes import (
    NOTE_COLUMNS,
    normalize_notes_frame,
    notes_lookup,
    notes_workbook_bytes,
    overlay_notes,
    read_notes_workbook,
)
from services.solvency_step6_analysis import sort_report_periods


PEER_CLASSIFICATION_STANDARD = (
    "规模分类：认可资产规模大于 5,000 亿元为大型公司，小于 500 亿元为小型公司，"
    "介于两者之间为中型公司；银行系、外资系、养老健康沿用调研样本标签。"
)
PEER_GROUP_ORDER = (
    "大型公司",
    "中型公司",
    "小型公司",
    "银行系",
    "外资系",
    "养老健康",
)


_PRINT_COMPONENT = st.components.v2.component(
    "solvency_dashboard_print_control",
    html="""
<div id="print-dashboard-controls">
  <button class="print-dashboard print-dashboard--portrait" data-mode="portrait" type="button">🖨️ 导出竖版 A4 PDF（仅正文）</button>
  <button class="print-dashboard print-dashboard--widescreen" data-mode="widescreen" type="button">🖨️ 导出横版 16:9 PDF</button>
</div>
""",
    css="""
#print-dashboard-controls { display:flex; flex-direction:column; gap:8px; }
.print-dashboard {
  width: 100%;
  min-height: 42px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 0;
  border-radius: 6px;
  color: #ffffff;
  font: 600 14px/1.2 var(--st-font, sans-serif);
  cursor: pointer;
}
.print-dashboard--portrait { background:#00338D; }
.print-dashboard--widescreen { background:#098E7E; }
.print-dashboard:hover { filter:brightness(1.08); }
.print-dashboard:focus-visible {
  outline: 2px solid var(--st-primary-color, #00338d);
  outline-offset: 2px;
}
""",
    js="""
export default function (component) {
  const { parentElement } = component
  const buttons = parentElement.querySelectorAll(".print-dashboard")
  buttons.forEach((button) => button.onclick = () => {
    const view = parentElement.ownerDocument?.defaultView ?? window
    const doc = view.document
    const root = doc.documentElement
    const mode = button.dataset.mode === "portrait" ? "portrait" : "widescreen"
    root.classList.remove(
      "solvency-print-mode-portrait",
      "solvency-print-mode-widescreen",
    )
    root.classList.add(`solvency-print-mode-${mode}`)
    doc.getElementById("dynamic-solvency-print-style")?.remove()
    const style = doc.createElement("style")
    style.id = "dynamic-solvency-print-style"
    style.textContent = mode === "portrait"
      ? "@page { size: A4 portrait; margin: 10mm; }"
      : [
          "@page { size: 338.67mm 190.5mm; margin: 8mm 12mm; }",
          "@page :first { margin: 0; }",
          "@page :last { margin: 0; }",
        ].join(" ")
    // Streamlit injects page-level styles inside the document body. Appending
    // here makes the selected paper rule the last rule in cascade order.
    doc.body.appendChild(style)
    view.addEventListener("afterprint", () => {
      root.classList.remove(
        "solvency-print-mode-portrait",
        "solvency-print-mode-widescreen",
      )
      style.remove()
    }, { once: true })
    setTimeout(() => view.print(), 160)
  })
}
""",
)


@dataclass(frozen=True)
class IndustryOverview:
    report_period: str
    period_scope: str
    company_count: int
    combined_median: float | None
    core_median: float | None
    sufficient_count: int
    warning_count: int
    insufficient_count: int


def _text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="string")
    return frame[column].fillna("").astype(str).str.strip()


def company_detail_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    result = frame.copy()
    company = _text_series(result, "公司")
    company_type = _text_series(result, "公司类型")
    company_code = _text_series(result, "公司统一编码")
    industry = (
        company.eq("行业合计")
        | company_type.eq("行业合计")
        | company_code.str.startswith("INDUSTRY_")
    )
    return result.loc[~industry & company.ne("")].copy()


def preferred_period_scope(frame: pd.DataFrame, requested: str = "") -> str:
    scopes = _text_series(frame, "期间口径")
    available = [value for value in scopes.unique().tolist() if value]
    if requested and requested in available:
        return requested
    for preferred in ("本季度末数", "期末数", "本年累计数", "本季度数"):
        if preferred in available:
            return preferred
    return available[0] if available else ""


def profile_platform_copy(
    profile_id: str,
    profile_name: str,
    frequency: str,
) -> tuple[str, str]:
    profile_text = f"{profile_id} {profile_name}".lower()
    if "non_life" in profile_text or "财产" in profile_text or "财险" in profile_text:
        sector = "财产险公司"
    elif "life" in profile_text or "寿险" in profile_text or "人身" in profile_text:
        sector = "人身险公司"
    else:
        clean_name = str(profile_name or "保险公司").replace("季度报告", "").replace("年度报告", "").strip()
        sector = clean_name or "保险公司"

    annual = str(frequency or "").upper() == "ANNUAL" or "年报" in profile_text or "年度报告" in profile_text
    if annual:
        report_label = "年度报告"
        database_label = "年度报告数据库"
    else:
        report_label = "偿付能力"
        database_label = "偿付能力季度报告数据库"
    return (
        f"{sector}{report_label}信息分享",
        f"中国{sector}{database_label}",
    )


def format_period_label(periods: Iterable[object]) -> str:
    ordered = sort_report_periods(periods)
    if not ordered:
        return "-"
    if len(ordered) == 1:
        return ordered[0]
    if len(ordered) <= 3:
        return "、".join(ordered)
    return f"{ordered[0]}–{ordered[-1]}"


def format_target_label(targets: Iterable[object] | str) -> str:
    if isinstance(targets, str):
        return targets.strip() or "全部公司"
    values = [str(value).strip() for value in targets if str(value).strip()]
    if not values:
        return "全部公司"
    if len(values) <= 3:
        return "、".join(values)
    return f"{len(values)} 家公司"


@lru_cache(maxsize=8)
def _image_data_uri(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_dashboard_header_html(
    *,
    title: str,
    database_name: str,
    report_period: str,
    period_scope: str,
    company_count: int,
    target: str,
    image_path: str | Path,
) -> str:
    image_uri = _image_data_uri(str(Path(image_path).resolve()))
    background = (
        f"url('{image_uri}')"
        if image_uri
        else "linear-gradient(90deg, #1525ff 0%, #1525ff 67%, #071b69 67%, #071b69 100%)"
    )
    safe_title = html.escape(str(title))
    details = (
        f"{database_name}　·　报告期：{report_period or '-'}　·　"
        f"口径：{period_scope or '-'}　·　{int(company_count):,} 家公司　·　"
        f"目标：{target or '全部公司'}"
    )
    safe_details = html.escape(details)
    return f"""
    <style>
      .solvency-dashboard-hero {{
        min-height: 178px;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        padding: 32px 34px;
        margin: 0 0 18px 0;
        border-radius: 8px;
        background-image: {background};
        background-size: 100% auto;
        background-position: center top;
        background-repeat: no-repeat;
        color: #ffffff;
      }}
      .solvency-dashboard-hero__content {{
        width: 72%;
      }}
      .solvency-dashboard-hero__title {{
        margin: 0 0 16px 0;
        color: #ffffff !important;
        font-size: clamp(25px, 2.6vw, 40px);
        line-height: 1.2;
        font-weight: 700;
        letter-spacing: 0.02em;
      }}
      .solvency-dashboard-hero__details {{
        margin: 0;
        color: #ffffff !important;
        font-size: clamp(12px, 1.15vw, 16px);
        line-height: 1.6;
        font-weight: 500;
      }}
      @media (max-width: 760px) {{
        .solvency-dashboard-hero {{ min-height: 150px; padding: 24px; }}
        .solvency-dashboard-hero__content {{ width: 78%; }}
      }}
    </style>
    <section class="solvency-dashboard-hero" aria-label="{safe_title}">
      <div class="solvency-dashboard-hero__content">
        <div
          class="solvency-dashboard-hero__title"
          role="heading"
          aria-level="1"
          style="color: #ffffff !important;"
        >{safe_title}</div>
        <p class="solvency-dashboard-hero__details" style="color: #ffffff !important;">{safe_details}</p>
      </div>
    </section>
    """


def build_report_cover_html(
    *,
    title: str,
    subtitle: str,
    date_text: str,
    image_path: str | Path,
) -> str:
    image_uri = _image_data_uri(str(Path(image_path).resolve()))
    background = (
        f'<img src="{image_uri}" alt="" />'
        if image_uri
        else '<div class="solvency-cover-fallback"></div>'
    )
    return f"""
    <div class="solvency-print-cover solvency-print-cover--front">
      {background}
      <div class="solvency-print-cover__text">
        <div class="solvency-print-cover__title">{html.escape(str(title))}</div>
        <div class="solvency-print-cover__subtitle">{html.escape(str(subtitle))}</div>
        <div class="solvency-print-cover__date">{html.escape(str(date_text))}</div>
      </div>
    </div>
    """


def build_report_back_cover_html(image_path: str | Path) -> str:
    image_uri = _image_data_uri(str(Path(image_path).resolve()))
    background = (
        f'<img src="{image_uri}" alt="" />'
        if image_uri
        else '<div class="solvency-cover-fallback"></div>'
    )
    return f"""
    <div class="solvency-print-cover solvency-print-cover--back">
      {background}
    </div>
    """


def render_report_cover(
    *,
    title: str,
    subtitle: str,
    date_text: str,
    picture_dir: str | Path,
) -> None:
    st.markdown(
        build_report_cover_html(
            title=title,
            subtitle=subtitle,
            date_text=date_text,
            image_path=Path(picture_dir) / "标题页.png",
        ),
        unsafe_allow_html=True,
    )


def render_report_back_cover(*, picture_dir: str | Path) -> None:
    st.markdown(
        build_report_back_cover_html(Path(picture_dir) / "封底页.png"),
        unsafe_allow_html=True,
    )


def render_report_notes_editor(
    *,
    title: str,
    key_prefix: str,
    template: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    """Render the annual-platform notes workflow with an offline template."""
    source_key = f"{key_prefix}_notes_source"
    digest_key = f"{key_prefix}_notes_upload_digest"
    editor_key = f"{key_prefix}_notes_editor"
    if source_key not in st.session_state:
        st.session_state[source_key] = normalize_notes_frame(template)

    with st.expander(title, expanded=False, icon=":material/edit_note:"):
        st.caption(
            "先下载模板填写，也可上传后直接在页面中修改分析内容和注释；"
            "一级、二级模块及图表对应关系保持锁定。"
        )
        st.download_button(
            "下载分析注释模板",
            data=notes_workbook_bytes(template),
            file_name=f"{key_prefix}_图片内容分析和注释模板.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            icon=":material/download:",
            key=f"{key_prefix}_notes_template_download",
        )
        upload = st.file_uploader(
            "上传分析注释表 Excel",
            type=["xlsx"],
            key=f"{key_prefix}_notes_upload",
        )
        if upload is not None:
            payload = upload.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            if st.session_state.get(digest_key) != digest:
                try:
                    uploaded = read_notes_workbook(payload, upload.name)
                    st.session_state[source_key] = overlay_notes(template, uploaded)
                    st.session_state[digest_key] = digest
                    st.session_state.pop(editor_key, None)
                    st.success(f"已读取 {len(uploaded):,} 条分析注释配置。")
                except Exception as exc:
                    st.error(f"分析注释表读取失败：{exc}")

        source = st.session_state[source_key]
        edited = st.data_editor(
            source,
            key=editor_key,
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            disabled=["模块ID", "一级分类", "二级分类", "对应图表名称"],
            column_config={
                "模块ID": st.column_config.TextColumn(width="small"),
                "一级分类": st.column_config.TextColumn(width="medium"),
                "二级分类": st.column_config.TextColumn(width="medium"),
                "对应图表名称": st.column_config.TextColumn(width="large", pinned=True),
                "分析内容-默认": st.column_config.TextColumn(width="large"),
                "分析内容-自定义": st.column_config.TextColumn(width="large"),
                "注释内容": st.column_config.TextColumn(width="large"),
                "图片文件名": st.column_config.TextColumn(width="medium"),
            },
        )
    return notes_lookup(edited.reindex(columns=NOTE_COLUMNS))


def render_report_analysis(note: dict[str, str] | None) -> None:
    note = note or {}
    default_text = str(note.get("分析内容-默认", "") or "").strip()
    custom_text = str(note.get("分析内容-自定义", "") or "").strip()
    if not default_text and not custom_text:
        return
    paragraphs = []
    if default_text:
        paragraphs.append(
            f'<p class="solvency-note-default">{html.escape(default_text)}</p>'
        )
    if custom_text:
        paragraphs.append(
            f'<p class="solvency-note-custom">{html.escape(custom_text)}</p>'
        )
    st.markdown(
        '<div class="solvency-report-analysis">' + "".join(paragraphs) + "</div>",
        unsafe_allow_html=True,
    )


def render_report_footnote(note: dict[str, str] | None) -> None:
    text = str((note or {}).get("注释内容", "") or "").strip()
    if text:
        st.markdown(
            f'<p class="solvency-report-footnote">注：{html.escape(text)}</p>',
            unsafe_allow_html=True,
        )


def render_dashboard_header(
    placeholder,
    data: pd.DataFrame | None,
    *,
    profile_id: str,
    profile_name: str,
    frequency: str,
    image_path: str | Path,
    periods: Iterable[object] = (),
    period_scope: str = "",
    targets: Iterable[object] | str = (),
) -> None:
    title, database_name = profile_platform_copy(profile_id, profile_name, frequency)
    detail = company_detail_rows(data)
    company_count = int(_text_series(detail, "公司").nunique()) if not detail.empty else 0
    available_periods = periods or _text_series(detail, "报告期").tolist()
    report_period = format_period_label(available_periods)
    if not period_scope and not detail.empty:
        selected_period = sort_report_periods(_text_series(detail, "报告期").tolist())
        period_frame = detail[
            _text_series(detail, "报告期").eq(selected_period[-1])
        ] if selected_period else detail
        period_scope = preferred_period_scope(period_frame)
    header_html = build_dashboard_header_html(
        title=title,
        database_name=database_name,
        report_period=report_period,
        period_scope=period_scope,
        company_count=company_count,
        target=format_target_label(targets),
        image_path=image_path,
    )
    placeholder.html(header_html, width="stretch")


def calculate_industry_overview(
    data: pd.DataFrame | None,
    *,
    report_period: str = "",
    period_scope: str = "",
) -> IndustryOverview:
    detail = company_detail_rows(data)
    if detail.empty:
        return IndustryOverview("", "", 0, None, None, 0, 0, 0)
    periods = sort_report_periods(_text_series(detail, "报告期").tolist())
    selected_period = report_period if report_period in periods else (periods[-1] if periods else "")
    period_frame = detail[_text_series(detail, "报告期").eq(selected_period)].copy()
    selected_scope = preferred_period_scope(period_frame, period_scope)
    if selected_scope:
        period_frame = period_frame[
            _text_series(period_frame, "期间口径").eq(selected_scope)
        ].copy()
    period_frame["数值_numeric"] = pd.to_numeric(period_frame.get("数值"), errors="coerce")

    def metric_values(code: str) -> pd.Series:
        rows = period_frame[
            _text_series(period_frame, "指标编码").eq(code)
        ].dropna(subset=["数值_numeric"])
        rows = rows.drop_duplicates(subset=["公司"], keep="last")
        return rows.set_index("公司")["数值_numeric"]

    combined = metric_values("COMBINED_SOLVENCY_RATIO")
    core = metric_values("CORE_SOLVENCY_RATIO")
    company_count = int(combined.index.nunique())
    if company_count == 0:
        company_count = int(_text_series(period_frame, "公司").nunique())
    combined_median = float(combined.median()) if not combined.empty else None
    core_median = float(core.median()) if not core.empty else None
    return IndustryOverview(
        report_period=selected_period,
        period_scope=selected_scope,
        company_count=company_count,
        combined_median=combined_median,
        core_median=core_median,
        sufficient_count=int((combined >= 150).sum()),
        warning_count=int(((combined >= 100) & (combined < 150)).sum()),
        insufficient_count=int((combined < 100).sum()),
    )


def build_peer_classification_table(data: pd.DataFrame | None) -> pd.DataFrame:
    detail = company_detail_rows(data)
    if detail.empty:
        return pd.DataFrame(columns=["序号", *PEER_GROUP_ORDER])
    periods = sort_report_periods(_text_series(detail, "报告期").tolist())
    if periods:
        detail = detail[_text_series(detail, "报告期").eq(periods[-1])].copy()
    pairs = pd.DataFrame({
        "公司": _text_series(detail, "公司"),
        "同业分类": _text_series(detail, "同业分类"),
    })
    pairs = pairs[(pairs["公司"] != "") & (pairs["同业分类"] != "")]
    pairs = pairs.drop_duplicates(subset=["公司"], keep="last")
    present_groups = pairs["同业分类"].unique().tolist()
    ordered_groups = [group for group in PEER_GROUP_ORDER if group in present_groups]
    ordered_groups.extend(sorted(group for group in present_groups if group not in PEER_GROUP_ORDER))
    values_by_group = {
        group: sorted(pairs.loc[pairs["同业分类"] == group, "公司"].tolist())
        for group in ordered_groups
    }
    max_rows = max((len(values) for values in values_by_group.values()), default=0)
    result = pd.DataFrame({"序号": range(1, max_rows + 1)})
    for group in ordered_groups:
        values = values_by_group[group]
        result[group] = values + [""] * (max_rows - len(values))
    return result


def render_industry_overview(data: pd.DataFrame | None) -> IndustryOverview:
    overview = calculate_industry_overview(data)
    st.markdown(
        f"### :material/account_balance: 01 · 行业整体偿付能力概览 · "
        f"{overview.report_period or '-'}"
    )
    first_row = st.columns(3)
    first_row[0].metric("纳入公司数", f"{overview.company_count:,} 家", border=True)
    first_row[1].metric(
        "综合偿付能力充足率中位数",
        "-" if overview.combined_median is None else f"{overview.combined_median:.1f}%",
        border=True,
    )
    first_row[2].metric(
        "核心偿付能力充足率中位数",
        "-" if overview.core_median is None else f"{overview.core_median:.1f}%",
        border=True,
    )
    denominator = max(overview.company_count, 1)
    second_row = st.columns(3)
    second_row[0].metric(
        "充足（≥150%）",
        f"{overview.sufficient_count:,} 家",
        f"占比 {overview.sufficient_count / denominator:.0%}",
        delta_color="green",
        delta_arrow="off",
        border=True,
    )
    second_row[1].metric(
        "预警（100%–150%）",
        f"{overview.warning_count:,} 家",
        f"占比 {overview.warning_count / denominator:.0%}",
        delta_color="gray",
        delta_arrow="off",
        border=True,
    )
    second_row[2].metric(
        "不达标（<100%）",
        f"{overview.insufficient_count:,} 家",
        f"占比 {overview.insufficient_count / denominator:.0%}",
        delta_color="red",
        delta_arrow="off",
        border=True,
    )
    if overview.period_scope:
        st.caption(f"概览采用“{overview.period_scope}”口径；分档依据综合偿付能力充足率。")
    return overview


def render_major_financing(
    data: pd.DataFrame | None,
    *,
    print_mode: bool = False,
) -> None:
    summary = summarize_financing(data)
    st.markdown(
        f"### :material/assignment: 02 · 重大融资信息统计 · "
        f"{summary.latest_period or '-'}"
    )
    st.info(
        "本页统计各公司在各季度的增资或发债事项，以及综合偿付能力充足率的变动情况。"
    )
    metrics = st.columns(3)
    metrics[0].metric("融资事件总数", f"{summary.event_count:,}", border=True)
    metrics[1].metric("涉及公司数", f"{summary.company_count:,}", border=True)
    metrics[2].metric("最新季度", summary.latest_period or "-", border=True)

    st.divider()
    st.markdown("#### :material/bar_chart: 重大融资信息明细")
    if not isinstance(data, pd.DataFrame) or data.empty:
        st.info("请在左侧边栏上传重大融资信息 Excel。")
        return

    periods = list(reversed(sort_report_periods(data["季度"].tolist())))
    def render_period_table(period: str, period_frame: pd.DataFrame) -> None:
        display = period_frame[
            ["公司名称", "增资/发债", "季度总变动", "增资/发债的影响"]
        ].reset_index(drop=True)
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={
                "公司名称": st.column_config.TextColumn("公司名称", pinned=True, width="small"),
                "增资/发债": st.column_config.TextColumn("增资/发债", width="large"),
                "季度总变动": st.column_config.TextColumn("季度总变动", width="medium"),
                "增资/发债的影响": st.column_config.TextColumn("增资/发债的影响", width="medium"),
            },
        )

    for period in periods:
        period_frame = data[data["季度"] == period].copy()
        company_count = int(period_frame["公司名称"].nunique())
        label = f"{period}（{company_count} 家公司）"
        if print_mode:
            st.markdown(f"#### {label}")
            render_period_table(period, period_frame)
        else:
            with st.expander(
                f":material/calendar_month: {label}",
                expanded=period == summary.latest_period,
            ):
                render_period_table(period, period_frame)


def render_print_control(*, key: str = "solvency_dashboard_print") -> None:
    st.markdown("### :material/print: 打印/导出 PDF")
    st.info(
        "竖版 A4 只导出正文，不套用 16:9 标题页和封底页；"
        "横版 16:9 用于完整报告。打印时请勾选“背景图形”以保留颜色。"
    )
    _PRINT_COMPONENT(key=key, width="stretch", height=96)


def render_kpmg_palette() -> None:
    with st.expander("查看 KPMG 官方色卡", icon=":material/palette:"):
        for category, colors in KPMG_CATEGORIES.items():
            st.markdown(f"**{category}**")
            swatches = "".join(
                (
                    '<span style="display:inline-flex;align-items:center;gap:4px;'
                    'margin:0 14px 8px 0;font-size:13px;">'
                    f'<span style="width:14px;height:14px;border-radius:3px;'
                    f'border:1px solid #d7dce5;background:{html.escape(color)};"></span>'
                    f'{html.escape(name)} <b>({html.escape(color)})</b></span>'
                )
                for name, color in colors.items()
            )
            st.html(f'<div style="display:flex;flex-wrap:wrap;">{swatches}</div>')


def render_peer_classification(data: pd.DataFrame | None) -> pd.DataFrame:
    table = build_peer_classification_table(data)
    st.markdown("### :material/apartment: 调研公司分类列表")
    st.caption(PEER_CLASSIFICATION_STANDARD)
    if table.empty:
        st.info("当前数据没有可展示的同业分类。")
    else:
        st.table(
            table,
            width="stretch",
            height="content",
            hide_index=True,
        )
    return table
