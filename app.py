from __future__ import annotations

import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import fitz

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from services.solvency_ai_table_extractor import (
    extraction_logs_frame,
    reconstructed_workbook_bytes,
)
from services.solvency_hybrid_pipeline import (
    extract_tables_hybrid,
    locate_tables_hybrid,
)
from services.solvency_normalizer import STANDARD_COLUMNS, load_taxonomy, normalize_tables, standardize_uploaded_frame
from services.solvency_pdf_locator import PageMatch, extract_report_metadata, load_page_features
from services.solvency_validator import load_validation_rules, validate_standard_data
from step7_solvency import show_step_7_solvency
from step8_solvency import show_step_8_solvency


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
TEMPLATE_DIR = ROOT / "templates"
FEATURE_CONFIG = CONFIG_DIR / "solvency_page_features.json"
TAXONOMY_FILE = CONFIG_DIR / "solvency_taxonomy.xlsx"
RULES_FILE = CONFIG_DIR / "solvency_validation_rules.xlsx"
COMPANY_URL_FILE = CONFIG_DIR / "solvency_company_urls.xlsx"
STANDARD_TEMPLATE = TEMPLATE_DIR / "偿付能力季度报告标准目标表.xlsx"


st.set_page_config(page_title="偿付能力报告平台", page_icon="🛡️", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    h1, h2, h3 {color:#00338D;}
    [data-testid="stMetric"] {background:#F4F7FC; border-left:4px solid #00338D; padding:12px; border-radius:5px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "pdf_bytes": None,
        "pdf_name": "",
        "metadata": {},
        "page_matches": [],
        "auto_page_matches": [],
        "raw_tables": [],
        "table_candidates": {},
        "selected_table_candidates": {},
        "ai_extraction_logs": [],
        "ai_workbook_bytes": b"",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_model": "",
        "llm_api_key": "",
        "auto_vision_retry": True,
        "force_vision_mode": False,
        "standard_data": pd.DataFrame(columns=STANDARD_COLUMNS),
        "integrated_data": pd.DataFrame(columns=STANDARD_COLUMNS),
        "validation_results": pd.DataFrame(),
        "monitor_results": pd.DataFrame(),
        "monitor_single_result": None,
        "edited_pages": {},
        "locator_config_version": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def read_taxonomy() -> pd.DataFrame:
    return load_taxonomy(TAXONOMY_FILE)


@st.cache_data(show_spinner=False)
def read_rules() -> pd.DataFrame:
    return load_validation_rules(RULES_FILE)


@st.cache_data(show_spinner=False)
def read_companies() -> pd.DataFrame:
    companies = pd.read_excel(COMPANY_URL_FILE, sheet_name="公司网址", header=2)
    required = ["公司", "公司类别", "偿付能力披露地址"]
    missing = [column for column in required if column not in companies.columns]
    if missing:
        raise ValueError(f"公司网址配置缺少字段：{', '.join(missing)}")
    companies = companies[required].copy()
    for column in required:
        companies[column] = companies[column].fillna("").astype(str).str.strip()
    return companies[companies["公司"] != ""].drop_duplicates(subset=["公司"], keep="last").reset_index(drop=True)


def target_period_terms(year: int, quarter: str) -> list[str]:
    quarter_number = int(quarter[-1])
    chinese_number = {1: "一", 2: "二", 3: "三", 4: "四"}[quarter_number]
    return [
        f"{year}{quarter}",
        f"{year}年第{quarter_number}季度",
        f"{year}年第{chinese_number}季度",
        f"{year}年{quarter_number}季度",
        f"{year}年{chinese_number}季度",
    ]


def check_company_report(row: pd.Series, year: int, quarter: str, timeout: int = 15) -> dict:
    company = str(row.get("公司", "")).strip()
    category = str(row.get("公司类别", "")).strip()
    url = str(row.get("偿付能力披露地址", "")).strip()
    checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    base = {
        "公司": company,
        "公司类别": category,
        "目标报告期": f"{year}{quarter}",
        "检查结果": "",
        "HTTP状态": "",
        "匹配关键词": "",
        "披露地址": url,
        "检查时间": checked_at,
        "说明": "",
    }
    if not url.lower().startswith(("http://", "https://")):
        return {**base, "检查结果": "未配置有效链接", "说明": "请维护系统公司网址配置。"}

    try:
        response = requests.get(
            url,
            timeout=(5, timeout),
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            },
        )
        base["HTTP状态"] = response.status_code
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" in content_type or response.content[:4] == b"%PDF":
            metadata = extract_report_metadata(response.content)
            matched = metadata.get("报告年度") == year and metadata.get("报告季度") == quarter
            return {
                **base,
                "检查结果": "已更新" if matched else "PDF报告期不匹配",
                "匹配关键词": str(metadata.get("报告期", "")),
                "说明": "链接直接返回PDF，已按PDF首页报告期判断。",
            }

        soup = BeautifulSoup(response.text, "html.parser")
        visible_text = soup.get_text(" ", strip=True)
        link_text = " ".join(
            f"{tag.get_text(' ', strip=True)} {tag.get('href', '')}"
            for tag in soup.find_all("a")
        )
        searchable = re.sub(r"\s+", "", f"{visible_text} {link_text}")
        terms = target_period_terms(year, quarter)
        matched_terms = [term for term in terms if term.lower() in searchable.lower()]
        has_solvency = "偿付能力" in searchable or "solvency" in searchable.lower()

        if matched_terms and has_solvency:
            result = "已更新"
            note = "页面中同时发现目标报告期和偿付能力关键词。"
        elif matched_terms:
            result = "疑似已更新，需人工核对"
            note = "发现目标报告期，但未在静态页面文本中发现偿付能力关键词。"
        else:
            result = "未发现目标报告"
            note = "静态页面未发现目标报告期；动态加载页面或反爬虫网站需人工打开核对。"
        return {
            **base,
            "检查结果": result,
            "匹配关键词": "、".join(matched_terms),
            "说明": note,
        }
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", "")
        return {
            **base,
            "HTTP状态": status,
            "检查结果": "访问失败，需人工核对",
            "说明": str(exc)[:300],
        }
    except Exception as exc:
        return {**base, "检查结果": "检查异常，需人工核对", "说明": str(exc)[:300]}


@st.cache_data(show_spinner=False)
def render_pdf_page(pdf_bytes: bytes, page_number: int, zoom: float = 1.8) -> bytes:
    """将指定物理页渲染为PNG，用于人工核实。"""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"页码 {page_number} 超出文档范围。")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")
    finally:
        document.close()


def parse_page_numbers(raw_value: str, total_pages: int) -> tuple[list[int], list[str]]:
    """解析逗号、分号或空格分隔的物理页码。"""
    valid_pages: list[int] = []
    invalid_values: list[str] = []
    for token in re.split(r"[,，;；\s]+", str(raw_value).strip()):
        if not token:
            continue
        if not token.isdigit():
            invalid_values.append(token)
            continue
        page_number = int(token)
        if page_number == 0:
            continue
        if 1 <= page_number <= total_pages:
            if page_number not in valid_pages:
                valid_pages.append(page_number)
        else:
            invalid_values.append(token)
    return valid_pages, invalid_values


def dataframe_to_xlsx(frame: pd.DataFrame, sheet_name: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


initialize_state()

st.title("🛡️ 偿付能力报告处理与分析平台")
st.caption("第一阶段版本：季度报告监控、核心表定位、标准化、勾稽、多公司集成及基础分析")

tabs = st.tabs([
    "STEP0 报告监控",
    "STEP1 页码定位",
    "STEP2 表格提取",
    "STEP3 标准化",
    "STEP4 勾稽检查",
    "STEP5 数据集成",
    "STEP6 自定义分析",
    "STEP7 公司报告",
    "STEP8 行业分析",
])


with tabs[0]:
    st.subheader("偿付能力季度报告监控")
    companies = read_companies()

    year_col, quarter_col = st.columns(2)
    target_year = int(year_col.number_input("报告年度", 2020, 2050, 2026))
    target_quarter = quarter_col.selectbox("报告季度", ["Q1", "Q2", "Q3", "Q4"])

    all_categories = sorted(companies["公司类别"].dropna().unique().tolist())
    selected_categories = st.multiselect(
        "公司类别",
        all_categories,
        default=all_categories,
        key="monitor_company_categories",
    )
    filtered_companies = companies[
        companies["公司类别"].isin(selected_categories)
    ].reset_index(drop=True)

    st.caption(
        f"系统公司范围共 {len(companies)} 家；当前筛选 {len(filtered_companies)} 家。"
        "运行时仅使用偿付能力系统自己的公司网址配置。"
    )
    st.dataframe(
        filtered_companies,
        width="stretch",
        hide_index=True,
        column_config={
            "偿付能力披露地址": st.column_config.LinkColumn(
                "偿付能力披露地址",
                display_text="打开页面",
            )
        },
    )

    st.markdown("#### 逐个查看")
    if filtered_companies.empty:
        st.warning("当前类别筛选下没有公司。")
    else:
        selected_company = st.selectbox(
            "选择需要检查的公司",
            filtered_companies["公司"].tolist(),
            key="monitor_selected_company",
        )
        selected_row = filtered_companies[
            filtered_companies["公司"] == selected_company
        ].iloc[0]

        link_col, check_col = st.columns(2)
        link_col.link_button(
            "打开该公司披露页面",
            selected_row["偿付能力披露地址"],
            width="stretch",
        )
        if check_col.button(
            "检查该公司是否更新",
            key="monitor_one",
            type="primary",
            width="stretch",
        ):
            with st.spinner(f"正在检查 {selected_company}..."):
                st.session_state.monitor_single_result = check_company_report(
                    selected_row,
                    target_year,
                    target_quarter,
                )

        if st.session_state.monitor_single_result:
            single_result = st.session_state.monitor_single_result
            status = single_result["检查结果"]
            if status == "已更新":
                st.success(f"{single_result['公司']}：{status}")
            elif "失败" in status or "异常" in status:
                st.error(f"{single_result['公司']}：{status}")
            else:
                st.warning(f"{single_result['公司']}：{status}")
            st.dataframe(
                pd.DataFrame([single_result]),
                width="stretch",
                hide_index=True,
                column_config={
                    "披露地址": st.column_config.LinkColumn(
                        "披露地址",
                        display_text="打开页面",
                    )
                },
            )

    st.markdown("#### 所有公司检查")
    with st.expander("批量检查设置"):
        worker_count = st.slider(
            "并发检查数量",
            min_value=1,
            max_value=8,
            value=4,
            help="数量越大速度越快，但部分保险公司网站可能触发访问限制。",
        )
        request_timeout = st.slider(
            "单个网站读取超时（秒）",
            min_value=5,
            max_value=30,
            value=15,
        )

    if st.button(
        f"检查当前筛选的全部 {len(filtered_companies)} 家公司",
        key="monitor_all",
        disabled=filtered_companies.empty,
        width="stretch",
    ):
        progress = st.progress(0, text="正在准备批量检查...")
        result_rows = []
        records = list(filtered_companies.iterrows())

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    check_company_report,
                    row,
                    target_year,
                    target_quarter,
                    request_timeout,
                ): order
                for order, (_, row) in enumerate(records)
            }
            for completed, future in enumerate(as_completed(future_map), start=1):
                result = future.result()
                result["_排序"] = future_map[future]
                result_rows.append(result)
                progress.progress(
                    completed / len(future_map),
                    text=f"已检查 {completed}/{len(future_map)}：{result['公司']}",
                )

        st.session_state.monitor_results = (
            pd.DataFrame(result_rows)
            .sort_values("_排序")
            .drop(columns="_排序")
            .reset_index(drop=True)
        )
        progress.empty()

    if not st.session_state.monitor_results.empty:
        results = st.session_state.monitor_results
        summary = results["检查结果"].value_counts()
        summary_columns = st.columns(min(len(summary), 4))
        for index, (status, count) in enumerate(summary.items()):
            summary_columns[index % len(summary_columns)].metric(status, int(count))

        result_filter = st.multiselect(
            "筛选检查结果",
            results["检查结果"].drop_duplicates().tolist(),
            default=results["检查结果"].drop_duplicates().tolist(),
            key="monitor_result_filter",
        )
        result_view = results[results["检查结果"].isin(result_filter)]
        st.dataframe(
            result_view,
            width="stretch",
            hide_index=True,
            column_config={
                "披露地址": st.column_config.LinkColumn(
                    "披露地址",
                    display_text="打开页面",
                )
            },
        )
        st.download_button(
            "下载本次检查结果",
            dataframe_to_xlsx(results, "报告更新检查"),
            f"偿付能力报告更新检查_{target_year}{target_quarter}.xlsx",
            width="stretch",
        )



with tabs[1]:
    st.subheader("📑 智能页码定位")
    st.caption("系统结合页面语义推断与表格结构雷达定位目标表；跨页页码可在左侧人工修改，右侧同步显示PDF原页。")

    feature_config = load_page_features(FEATURE_CONFIG)
    table_configs = feature_config.get("tables", [])
    config_version = str(feature_config.get("version", ""))
    if st.session_state.locator_config_version != config_version:
        st.session_state.locator_config_version = config_version
        st.session_state.auto_page_matches = []
        st.session_state.page_matches = []
        st.session_state.edited_pages = {}
        st.session_state.raw_tables = []
        st.session_state.ai_extraction_logs = []
        st.session_state.ai_workbook_bytes = b""
        st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
        for item in table_configs:
            st.session_state.pop(f"page_edit_{item['table_id']}", None)
    table_config_by_name = {item["table_name"]: item for item in table_configs}

    uploaded_pdf = st.file_uploader(
        "拖拽或选择一份偿付能力季度报告摘要 PDF",
        type=["pdf"],
        key="solvency_pdf",
    )

    if uploaded_pdf is None:
        st.info("上传PDF并启动智能定位后，此处将显示可编辑页码和对应页面预览。")
    else:
        incoming = uploaded_pdf.getvalue()
        if incoming != st.session_state.pdf_bytes:
            st.session_state.pdf_bytes = incoming
            st.session_state.pdf_name = uploaded_pdf.name
            st.session_state.metadata = extract_report_metadata(incoming)
            st.session_state.auto_page_matches = []
            st.session_state.page_matches = []
            st.session_state.edited_pages = {}
            st.session_state.raw_tables = []
            st.session_state.table_candidates = {}
            st.session_state.selected_table_candidates = {}
            st.session_state.ai_extraction_logs = []
            st.session_state.ai_workbook_bytes = b""
            st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
            for item in table_configs:
                st.session_state.pop(f"page_edit_{item['table_id']}", None)

        total_pages = int(st.session_state.metadata.get("页数", 0) or 0)
        st.caption(f"当前文件：{uploaded_pdf.name}　|　文档共 {total_pages} 页")
        with st.expander("查看自动识别的报告基本信息"):
            st.json(st.session_state.metadata)

        col_left, col_spacer, col_right = st.columns([1, 0.05, 1.2])

        with col_left:
            st.markdown("#### 检索目标设定")
            all_table_names = [item["table_name"] for item in table_configs]
            selected_table_names = st.multiselect(
                "请选择需要定位的报表：",
                all_table_names,
                default=all_table_names,
                key="solvency_target_tables",
            )
            selected_configs = [
                table_config_by_name[name]
                for name in selected_table_names
                if name in table_config_by_name
            ]

            with st.expander("大模型接口设置（STEP1和STEP2共用）", expanded=False):
                st.text_input(
                    "OpenAI兼容接口地址",
                    key="llm_base_url",
                    help="填写兼容 /chat/completions 的接口根地址。",
                )
                st.text_input(
                    "模型名称",
                    key="llm_model",
                    placeholder="填写实际可用的模型名称",
                )
                st.text_input("API Key", type="password", key="llm_api_key")
                st.caption("页码定位和逐页表格提取共用此设置；密钥仅保存在当前浏览器会话。")
            if st.button(
                "启动智能定位",
                type="primary",
                key="locate_tables",
                width="stretch",
            ):
                if not selected_configs:
                    st.error("请至少选择一张报表。")
                elif not all(
                    str(st.session_state.get(key, "")).strip()
                    for key in ("llm_base_url", "llm_model", "llm_api_key")
                ):
                    st.error("请先在上方填写接口地址、模型名称和API Key。混合智能定位需要语义模型参与。")
                else:
                    with st.spinner("正在执行大模型页面推断和Python表格雷达扫描..."):
                        selected_feature_config = {
                            **feature_config,
                            "tables": selected_configs,
                        }
                        matches, locator_message = locate_tables_hybrid(
                            incoming,
                            selected_feature_config,
                            api_key=st.session_state.llm_api_key,
                            base_url=st.session_state.llm_base_url,
                            model=st.session_state.llm_model,
                        )
                        st.session_state.auto_page_matches = matches
                        st.session_state.page_matches = matches
                        st.session_state.edited_pages = {
                            item.table_id: list(item.pages)
                            for item in matches
                        }
                        for item in matches:
                            st.session_state[f"page_edit_{item.table_id}"] = ", ".join(map(str, item.pages))
                    st.success("定位完成！请核对页码并结合右侧页面预览进行校准。")
                    if "失败" in locator_message:
                        st.warning(locator_message)
                    else:
                        st.info(locator_message)

            if st.session_state.auto_page_matches:
                st.markdown("---")
                st.markdown("#### 结果核对")
                st.caption("若表格跨页，请用逗号分隔物理页码，例如：26, 27。未找到时可直接手工输入。")

                edited_pages: dict[str, list[int]] = {}
                updated_matches: list[PageMatch] = []
                for match in st.session_state.auto_page_matches:
                    input_key = f"page_edit_{match.table_id}"
                    if input_key not in st.session_state:
                        st.session_state[input_key] = ", ".join(map(str, match.pages))
                    raw_value = st.text_input(match.table_name, key=input_key)
                    valid_pages, invalid_values = parse_page_numbers(raw_value, total_pages)
                    edited_pages[match.table_id] = valid_pages
                    updated_matches.append(
                        PageMatch(
                            table_id=match.table_id,
                            table_name=match.table_name,
                            pages=valid_pages,
                            score=match.score,
                            evidence=match.evidence,
                        )
                    )
                    auto_text = ", ".join(map(str, match.pages)) if match.pages else "未自动找到"
                    evidence_text = match.evidence or "无明确关键词证据"
                    st.caption(f"自动定位：{auto_text}　|　识别依据：{evidence_text}")
                    if invalid_values:
                        st.warning(f"以下页码或内容无效，已忽略：{', '.join(invalid_values)}")

                st.session_state.edited_pages = edited_pages
                st.session_state.page_matches = updated_matches

                if st.button(
                    "确认页码，进入下一步",
                    key="confirm_solvency_pages",
                    width="stretch",
                ):
                    valid_count = sum(bool(item.pages) for item in updated_matches)
                    st.success(f"页码已确认：5类目标中有 {valid_count} 类配置了有效页码。请前往 STEP2 提取表格。")

        with col_right:
            st.markdown("#### 页面预览")
            if st.session_state.auto_page_matches and st.session_state.page_matches:
                preview_matches = st.session_state.page_matches
                preview_ids = [item.table_id for item in preview_matches]
                preview_name_map = {item.table_id: item.table_name for item in preview_matches}
                selected_preview_id = st.selectbox(
                    "选择要预览的报表：",
                    preview_ids,
                    format_func=lambda table_id: preview_name_map[table_id],
                    key="solvency_preview_table",
                )
                selected_match = next(
                    item for item in preview_matches
                    if item.table_id == selected_preview_id
                )
                pages_to_preview = selected_match.pages

                if not pages_to_preview:
                    st.info("尚未配置有效页码，请在左侧输入物理页码。")
                else:
                    if len(pages_to_preview) > 1:
                        current_page = st.radio(
                            "该报表包含多页，请切换预览：",
                            pages_to_preview,
                            horizontal=True,
                            key=f"preview_page_{selected_preview_id}",
                        )
                    else:
                        current_page = pages_to_preview[0]

                    try:
                        preview_image = render_pdf_page(incoming, current_page)
                        st.image(
                            preview_image,
                            caption=f"当前预览：第 {current_page} 页 / 共 {total_pages} 页",
                            width="stretch",
                        )
                    except Exception as exc:
                        st.warning(f"页面预览失败：{exc}")
            else:
                st.info("启动智能定位后，此处将显示对应PDF页面。")


with tabs[2]:
    st.subheader("表格智能转换")
    st.caption(
        "采用逐页结构化提取：每个物理页独立处理，失败页单独进行图片扫描或网格纠错，"
        "所有页面成功后再按人工确认的页码顺序确定性拼接；任何一页失败都不会输出残缺跨页表。"
    )
    if not st.session_state.page_matches:
        st.info("请先在 STEP1 上传报告、完成页码定位并人工确认物理页码。")
    else:
        with st.expander("逐页提取与重试设置", expanded=not bool(st.session_state.raw_tables)):
            configured_model = st.session_state.get("llm_model", "") or "未设置"
            st.caption(f"使用STEP1中的接口设置，当前模型：{configured_model}")
            option_cols = st.columns(2)
            with option_cols[0]:
                st.toggle(
                    "单页提取失败时自动图片扫描重试",
                    key="auto_vision_retry",
                    help="只重试失败的物理页；模型不支持图片时会改用该页的加强版文本纠错。",
                )
            with option_cols[1]:
                st.toggle(
                    "所有页面直接使用图片扫描模式",
                    key="force_vision_mode",
                    help="适用于扫描版或纯图片PDF；所选模型必须支持图片输入。",
                )
            st.caption("API Key仅保存在当前浏览器会话内，不会写入配置文件或导出的Excel。")

        if st.button("开始智能提取", type="primary", key="extract_tables"):
            if not all(
                str(st.session_state.get(key, "")).strip()
                for key in ("llm_base_url", "llm_model", "llm_api_key")
            ):
                st.error("请先完整填写接口地址、模型名称和API Key。")
            else:
                progress_logs = []
                try:
                    with st.status("正在逐页提取并按页码拼接...", expanded=True) as extraction_status:
                        log_slot = st.empty()

                        def report_progress(log):
                            progress_logs.append(log)
                            log_slot.dataframe(
                                extraction_logs_frame(progress_logs),
                                width="stretch",
                                hide_index=True,
                            )

                        bundle = extract_tables_hybrid(
                            st.session_state.pdf_bytes,
                            st.session_state.page_matches,
                            api_key=st.session_state.llm_api_key,
                            base_url=st.session_state.llm_base_url,
                            model=st.session_state.llm_model,
                            auto_vision_retry=st.session_state.auto_vision_retry,
                            force_vision=st.session_state.force_vision_mode,
                            progress_callback=report_progress,
                        )
                        st.session_state.raw_tables = bundle.tables
                        st.session_state.ai_extraction_logs = bundle.logs
                        st.session_state.ai_workbook_bytes = reconstructed_workbook_bytes(bundle)
                        st.session_state.table_candidates = {}
                        st.session_state.selected_table_candidates = {}
                        st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)

                        failed = [item for item in bundle.logs if item.status == "失败"]
                        if bundle.tables:
                            extraction_status.update(
                                label=f"智能提取完成：生成 {len(bundle.tables)} 张标准化中间表",
                                state="complete",
                                expanded=False,
                            )
                        else:
                            extraction_status.update(
                                label="智能提取未生成可用表格",
                                state="error",
                                expanded=True,
                            )
                        if failed:
                            st.warning(f"仍有 {len(failed)} 次最终提取失败，请查看下方提取日志。")
                except Exception as exc:
                    st.session_state.ai_extraction_logs = progress_logs
                    st.error(f"智能提取中止：{exc}")

    if st.session_state.ai_extraction_logs:
        st.markdown("##### 提取与重试日志")
        st.dataframe(
            extraction_logs_frame(st.session_state.ai_extraction_logs),
            width="stretch",
            hide_index=True,
        )

    if st.session_state.raw_tables:
        tables = st.session_state.raw_tables

        def format_reconstructed_table(index: int) -> str:
            item = tables[index]
            pages = item.source_pages or [item.page]
            page_label = "、".join(map(str, pages))
            return f"{item.table_name} - 第{page_label}页 - {item.strategy}"

        selected_table_index = st.selectbox(
            "标准化中间表预览",
            range(len(tables)),
            format_func=format_reconstructed_table,
            key="ai_table_preview",
        )
        selected_table = tables[selected_table_index]
        source_pages = selected_table.source_pages or [selected_table.page]

        metric_cols = st.columns(3)
        metric_cols[0].metric("处理方式", selected_table.strategy)
        metric_cols[1].metric("来源页数", len(source_pages))
        metric_cols[2].metric("质量评分", f"{selected_table.quality_score:.1f}")
        if selected_table.evidence:
            st.caption(f"自动校验依据：{selected_table.evidence}")

        table_col, pdf_col = st.columns([1.15, 1], gap="large")
        with table_col:
            st.markdown("##### 网格化重构结果")
            st.dataframe(selected_table.to_frame(), width="stretch", hide_index=True)
        with pdf_col:
            selected_source_page = st.selectbox(
                "PDF原页核实",
                source_pages,
                format_func=lambda value: f"第 {value} 页",
                key=f"ai_source_page_{selected_table.table_id}",
            )
            try:
                st.image(
                    render_pdf_page(st.session_state.pdf_bytes, selected_source_page),
                    caption=f"PDF物理第 {selected_source_page} 页",
                    width="stretch",
                )
            except Exception as exc:
                st.warning(f"PDF页面预览失败：{exc}")

        if st.session_state.ai_workbook_bytes:
            st.download_button(
                "下载STEP2标准化提取Excel",
                st.session_state.ai_workbook_bytes,
                "偿付能力报告_STEP2标准化提取.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

with tabs[3]:
    st.subheader("偿付能力标准化")
    if STANDARD_TEMPLATE.exists():
        st.download_button("下载标准目标表模板", STANDARD_TEMPLATE.read_bytes(), STANDARD_TEMPLATE.name)
    company_type = st.selectbox("公司类型", ["寿险", "财险", "再保险", "养老险", "健康险", "其他"], key="standard_company_type")
    if st.session_state.raw_tables and st.button("生成标准化长表", type="primary", key="normalize_tables"):
        st.session_state.standard_data = normalize_tables(
            st.session_state.raw_tables,
            read_taxonomy(),
            st.session_state.metadata,
            company_type,
        )
    if not st.session_state.standard_data.empty:
        st.dataframe(st.session_state.standard_data, use_container_width=True, hide_index=True)
        st.download_button("下载标准化数据", dataframe_to_xlsx(st.session_state.standard_data, "标准数据"), "偿付能力标准数据.xlsx")


with tabs[4]:
    st.subheader("偿付能力勾稽检查")
    if st.session_state.standard_data.empty:
        st.info("请先完成 STEP3 标准化。")
    elif st.button("执行勾稽检查", type="primary", key="validate_data"):
        st.session_state.validation_results = validate_standard_data(st.session_state.standard_data, read_rules())
    if not st.session_state.validation_results.empty:
        st.dataframe(st.session_state.validation_results, use_container_width=True, hide_index=True)
        st.download_button("下载勾稽结果", dataframe_to_xlsx(st.session_state.validation_results, "勾稽结果"), "偿付能力勾稽结果.xlsx")


with tabs[5]:
    st.subheader("多公司、多季度数据集成")
    uploads = st.file_uploader("上传一个或多个标准数据文件", type=["xlsx"], accept_multiple_files=True, key="integrated_uploads")
    if uploads and st.button("合并标准数据", type="primary", key="merge_standard_data"):
        frames = []
        for upload in uploads:
            candidate = pd.read_excel(upload, sheet_name="标准数据", header=2)
            frames.append(standardize_uploaded_frame(candidate))
        st.session_state.integrated_data = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["公司", "报告期", "指标编码", "期间口径", "来源页码"], keep="last"
        )
    if not st.session_state.integrated_data.empty:
        st.dataframe(st.session_state.integrated_data, use_container_width=True, hide_index=True)
        st.download_button("下载行业集成数据", dataframe_to_xlsx(st.session_state.integrated_data, "标准数据"), "偿付能力行业集成数据.xlsx")


with tabs[6]:
    st.subheader("自定义偿付能力对标分析")
    source = st.session_state.integrated_data if not st.session_state.integrated_data.empty else st.session_state.standard_data
    if source.empty:
        st.info("请先完成标准化或数据集成。")
    else:
        metrics = source[["指标编码", "指标名称"]].drop_duplicates().sort_values("指标名称")
        selected_metric = st.selectbox("选择指标", metrics["指标名称"].tolist(), key="custom_metric")
        code = metrics.loc[metrics["指标名称"] == selected_metric, "指标编码"].iloc[0]
        view = source[source["指标编码"] == code].copy()
        view["数值"] = pd.to_numeric(view["数值"], errors="coerce")
        view = view.dropna(subset=["数值"])
        if view.empty:
            st.warning("该指标没有可绘制的数值。")
        else:
            import plotly.express as px
            fig = px.line(view.sort_values("报告期"), x="报告期", y="数值", color="公司", markers=True, facet_col="期间口径" if view["期间口径"].nunique() <= 3 else None)
            st.plotly_chart(fig, use_container_width=True)


analysis_source = st.session_state.integrated_data if not st.session_state.integrated_data.empty else st.session_state.standard_data
with tabs[7]:
    show_step_7_solvency(analysis_source)
with tabs[8]:
    show_step_8_solvency(analysis_source)

