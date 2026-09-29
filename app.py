from __future__ import annotations

# Incremented when Step7 filter state handling changes. Keeping the revision in
# the entry point makes Streamlit reload the active session without restarting
# the server or discarding the user's integrated data.
STEP7_FILTER_RUNTIME_REVISION = 3

import io
import hashlib
import hmac
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from time import perf_counter
import fitz

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

from dashboard_components import render_kpmg_palette, render_print_control
from services.solvency_locator_roles import LOCATOR_ROLE_LABELS
from services.report_profiles import (
    ProfileValidationError,
    ReportProfile,
    load_profile_registry,
    load_profile_workbook,
    profile_workbook_bytes,
)
from services.solvency_vlm_v2_pipeline import (
    VLMV2LocatorRun,
    evaluate_vlm_v2_step3_gate,
    extract_metrics_vlm_v2,
    locate_tables_vlm_v2,
    read_vlm_v2_extracted_tables,
    refresh_vlm_v2_cross_table_checks,
    vlm_v2_to_extracted_tables,
    vlm_v2_gold_evaluation,
    vlm_v2_workbook_bytes,
)
from services.solvency_vlm_v2_benchmark import (
    blind_scores_workbook_bytes,
    blind_test_template_bytes,
    combine_blind_scores,
    match_blind_case,
    read_blind_gold_workbook,
    score_blind_case,
)
from services.solvency_dataset_adapter import (
    add_missing_derived_metrics,
    convert_external_workbook,
    read_standard_workbook,
    standard_workbook_bytes,
)
from services.solvency_company_identity import load_peer_group_config, resolve_peer_group
from services.solvency_step3_standardizer import (
    infer_step3_session_metadata,
    infer_step3_upload_metadata,
    normalize_report_period,
    read_target_template,
    result_workbook_bytes,
    standardize_to_target,
    step3_metric_catalog,
    target_template_workbook_bytes,
)
from services.solvency_table_extractor import ExtractedTable
from services.solvency_display import dataframe_for_display
from services.solvency_step6_analysis import (
    CHART_TYPES,
    COMPANY_REPORT,
    INDUSTRY_REPORT,
    NO_COMPANY_TYPE_FILTER,
    build_comparison_chart,
    companies_for_quick_selection,
    default_periods,
    filter_analysis_frame,
    metric_options,
    nonblank_values,
    prepare_analysis_frame,
    report_scope_frame,
    sort_report_periods,
    visualization_metric_frame,
)
from services.solvency_metric_registry import extend_taxonomy
from services.solvency_filing_catalog import extend_filing_taxonomy
from services.solvency_financing_analysis import read_major_financing_workbook
from services.solvency_navigation import (
    COMPANY_FIRST_LEVELS,
    KPMG_DEFAULT_COLORS,
    OVERVIEW_LEVEL,
    PRINT_ALL_LABEL,
    chart_names,
    first_levels_for_codes,
    metric_codes_for_chart,
    resolve_chart_selection,
    second_levels,
)
from services.solvency_gold_standard import (
    evaluate_gold_case,
    find_gold_case,
    load_gold_manifest,
)
from services.solvency_normalizer import (
    STANDARD_COLUMNS,
    load_taxonomy,
    narrow_table_view,
    upgrade_standard_frame,
)
from services.solvency_pdf_locator import (
    PageMatch,
    extract_report_metadata,
    report_identity_warning,
)
from services.solvency_validator import (
    load_validation_rules,
    validate_standard_data,
)
from services.solvency_validation_workbook import (
    validation_display_frame,
    validation_workbook_bytes,
)
from step7_solvency import show_step_7_solvency
from step8_solvency import show_step_8_solvency


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
PROFILE_DIR = CONFIG_DIR / "report_profiles"
DEFAULT_PROFILE_ID = "LIFE_SOLVENCY"
STANDARD_SCHEMA_VERSION = 2
GOLD_MANIFEST = ROOT / "gold_standard" / "manifest.json"
PEER_GROUP_CONFIG = CONFIG_DIR / "solvency_peer_groups.json"
WORKFLOW_TAB_LABELS = (
    "STEP0 报告监控",
    "STEP1 页码定位",
    "STEP2 表格提取",
    "STEP3 标准化",
    "STEP4 勾稽检查",
    "STEP5 数据集成",
    "STEP6 自定义分析",
    "STEP7 公司报告",
    "STEP8 行业分析",
)
ANALYSIS_TAB_LABELS = WORKFLOW_TAB_LABELS[6:]


st.set_page_config(page_title="保险公司报告分析处理平台", page_icon="🛡️", layout="wide")


LOGIN_AI_PROVIDERS = {
    "阿里云百炼 (通义千问)": (
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen-plus",
    ),
    "DeepSeek (深度求索)": ("https://api.deepseek.com/v1", "deepseek-chat"),
    "月之暗面 (Kimi)": ("https://api.moonshot.cn/v1", "moonshot-v1-8k"),
    "智谱AI (GLM-4)": ("https://open.bigmodel.cn/api/paas/v4", "glm-4"),
    "OpenAI (ChatGPT)": ("https://api.openai.com/v1", "gpt-4o"),
    "自定义私有化节点": ("", ""),
}
DEFAULT_LOGIN_AI_PROVIDER = "阿里云百炼 (通义千问)"
NORMAL_USER_ROLE = "普通用户"
PROJECT_MEMBER_ROLE = "项目组成员"


def _initialize_login_state() -> None:
    """Initialize authentication and shared model settings for this browser tab."""
    default_base_url, default_model = LOGIN_AI_PROVIDERS[DEFAULT_LOGIN_AI_PROVIDER]
    defaults = {
        "logged_in": False,
        "user_role": None,
        "llm_provider": DEFAULT_LOGIN_AI_PROVIDER,
        "llm_base_url": default_base_url,
        "llm_model": default_model,
        "llm_api_key": "",
        "login_base_url_input": default_base_url,
        "login_model_input": default_model,
        "login_api_key_input": "",
        "login_user_role": NORMAL_USER_ROLE,
        "login_security_code": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _provider_name_from_base_url(base_url: str) -> str:
    normalized_url = str(base_url or "").strip().rstrip("/").lower()
    for provider, (provider_url, _) in LOGIN_AI_PROVIDERS.items():
        if normalized_url == provider_url.strip().rstrip("/").lower():
            return provider
    return "自定义私有化节点"


def _expected_login_code(role: str) -> str:
    """Allow deployments to override the reference platform's two access codes."""
    env_name = (
        "SOLVENCY_PROJECT_ACCESS_CODE"
        if role == PROJECT_MEMBER_ROLE
        else "SOLVENCY_USER_ACCESS_CODE"
    )
    fallback = "KPMG666" if role == PROJECT_MEMBER_ROLE else "KPMG1234"
    return os.getenv(env_name, fallback)


def _logout() -> None:
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.llm_api_key = ""
    st.session_state.login_api_key_input = ""
    st.session_state.login_security_code = ""
    st.session_state.pop("workflow_main_tab", None)
    st.session_state.pop("profile_settings_tabs", None)


def _render_login_page() -> None:
    st.markdown(
        """
        <style>
        header {visibility: hidden;}
        .stApp {background: linear-gradient(135deg, #040B16 0%, #0A1931 50%, #002266 100%); color: #E2E8F0;}
        [data-testid="column"]:nth-of-type(2) {background: rgba(255,255,255,0.08); backdrop-filter: blur(25px); -webkit-backdrop-filter: blur(25px); border-radius: 20px; border: 1px solid rgba(0,243,255,0.3); box-shadow: 0 15px 35px rgba(0,0,0,0.5), inset 0 0 15px rgba(0,243,255,0.15); padding: 40px 30px; margin-top: 6vh;}
        .title-glow {background: linear-gradient(90deg, #FFFFFF 0%, #76D2FF 50%, #00F3FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; filter: drop-shadow(0 0 15px rgba(0,243,255,0.6));}
        div[data-baseweb="input"]>div, div[data-baseweb="select"]>div {background: #F4F6FA!important; border: 1px solid rgba(0,243,255,0.3)!important; border-radius: 8px!important;}
        div[data-baseweb="input"]>div:focus-within, div[data-baseweb="select"]>div:focus-within {border-color: #00F3FF!important; box-shadow: 0 0 12px rgba(0,243,255,0.5)!important;}
        div[data-testid="stRadio"] label p {color: #FFFFFF!important; font-weight: bold!important; font-size: 14px!important;}
        label p, .stSelectbox label p, .stTextInput label p {color: #00F3FF!important; letter-spacing: 1px!important;}
        div[data-baseweb="input"] input {color: #31333F!important; -webkit-text-fill-color: #31333F!important; caret-color: #00338D!important; font-size: 14px!important;}
        div[data-baseweb="input"] input::placeholder {color: #7A8290!important; -webkit-text-fill-color: #7A8290!important; opacity: 1!important;}
        div[data-baseweb="select"] span {color: #31333F!important; -webkit-text-fill-color: #31333F!important; font-size: 14px!important;}
        div[data-baseweb="select"] svg {color: #31333F!important; fill: #31333F!important;}
        button[kind="primary"] {background: linear-gradient(90deg, #0044CC, #0088FF)!important; border: 1px solid rgba(0,243,255,0.5)!important; box-shadow: 0 0 15px rgba(0,136,255,0.4)!important; color: white!important; font-weight: bold!important; letter-spacing: 2px!important; border-radius: 8px!important; margin-top: 5px!important;}
        button[kind="primary"]:hover {box-shadow: 0 0 25px rgba(0,243,255,0.8)!important; transform: scale(1.02);}
        [data-testid="stPopover"] {display: flex; justify-content: flex-end;}
        [data-testid="stPopover"] button {background: transparent!important; border: none!important; box-shadow: none!important; color: #94A3B8!important; font-size: 12px!important; font-weight: normal!important; padding: 0!important; min-height: 0!important; text-decoration: underline; margin-top: 8px; margin-bottom: 15px;}
        [data-testid="stPopover"] button:hover {color: #00F3FF!important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        st.markdown(
            "<div style='text-align:center; margin-bottom:30px;'>"
            "<h1 style='font-size:32px; margin:0;'><span class='title-glow'>"
            "偿付能力数智分析平台</span></h1>"
            "<p style='color:#76D2FF; font-size:11px; letter-spacing:4px; "
            "margin-top:8px; font-weight:bold;'>SOLVENCY INTELLIGENCE</p></div>",
            unsafe_allow_html=True,
        )
        selected_role = st.radio(
            "访问权限",
            [NORMAL_USER_ROLE, PROJECT_MEMBER_ROLE],
            horizontal=True,
            key="login_user_role",
            label_visibility="collapsed",
        )
        st.text_input(
            "安全验证",
            type="password",
            key="login_security_code",
            placeholder="请输入内部安全码",
        )
        st.text_input(
            "RPC 接口地址",
            key="login_base_url_input",
            placeholder="https://example.com/v1",
            help="填写兼容 /chat/completions 的接口根地址。",
        )
        st.text_input(
            "指定模型版本",
            key="login_model_input",
            placeholder="填写实际可用的模型名称",
        )
        st.text_input(
            "请填写您使用AI的API Key",
            type="password",
            key="login_api_key_input",
            placeholder=" sk-... (不填则仅开放报告检测、数据合并及可视化本地分析功能)",
        )
        if "api.openai.com" in str(st.session_state.login_base_url_input).lower():
            st.markdown(
                "<p style='font-size:11px; color:#F87171; text-align:right; "
                "margin-top:4px; margin-bottom:0;'>⚠️ 严禁向境外节点传输涉密数据</p>",
                unsafe_allow_html=True,
            )
        with st.popover("如何获取API key?"):
            st.markdown(
                "**1.** 前往各大模型官方开放平台注册账号。\n\n"
                "**2.** 在控制台生成 `sk-` 开头的密钥。\n\n"
                "**3.** 复制填入上方输入框即可体验 AI 功能。\n\n"
                "登录页中的接口地址、模型版本和 API Key 由 Step1、Step2 与 Step7 共用。"
            )

        if st.button("启 动 系 统", type="primary", width="stretch"):
            supplied_code = str(st.session_state.login_security_code)
            if not hmac.compare_digest(supplied_code, _expected_login_code(selected_role)):
                st.error(f"❌ 拒绝访问：{selected_role}安全码错误")
            else:
                st.session_state.logged_in = True
                st.session_state.user_role = selected_role
                st.session_state.llm_base_url = str(
                    st.session_state.login_base_url_input
                ).strip()
                st.session_state.llm_provider = _provider_name_from_base_url(
                    st.session_state.llm_base_url
                )
                st.session_state.llm_model = str(
                    st.session_state.login_model_input
                ).strip()
                st.session_state.llm_api_key = str(
                    st.session_state.login_api_key_input
                ).strip()
                st.session_state.pop("workflow_main_tab", None)
                st.session_state.pop("profile_settings_tabs", None)
                st.rerun()

        st.markdown(
            "<div style='text-align:center; color:#94A3B8; font-size:11px; "
            "margin-top:30px; letter-spacing:1px;'>系统版本：v4.0 © 2026<br>"
            "保险报告处理与分析平台</div>",
            unsafe_allow_html=True,
        )


_initialize_login_state()
if not st.session_state.logged_in:
    _render_login_page()
    st.stop()

st.markdown(
    """
    <style>
    /* 保留 Streamlit 为固定顶栏提供的原生顶部安全间距；不要在这里缩小
       padding-top，否则首个 Profile 操作区会被 Deploy/菜单工具栏覆盖。 */
    [data-testid="stMainBlockContainer"] {padding-bottom: 2rem;}
    [data-testid="stAppViewContainer"] {background:#F4F7FC;}
    h1, h2, h3 {color:#00338D;}
    .platform-page-heading h1 {margin:0 0 0.35rem;}
    .platform-profile-caption {
      margin:0 0 0.75rem; color:rgba(49,51,63,.6); font-size:0.875rem;
    }
    [data-testid="stMetric"] {background:#F4F7FC; border-left:4px solid #00338D; padding:12px; border-radius:5px;}
    /* Remove obsolete STEP7 elements left behind during a rerun without hiding
       the current module merely because one nested chart is still stale. */
    [class*="st-key-s7_report_module_"][data-stale="true"],
    [class*="st-key-s7_report_module_"]
      [data-testid="stElementContainer"][data-stale="true"],
    [data-testid="stElementContainer"][data-stale="true"]:has(
      [class*="st-key-s7_report_module_"]
    ) {
      display:none!important;
    }
    @page { size: 13.333in 7.5in; margin: 0.35in; }
    @media print {
      [data-testid="stSidebar"],
      [data-testid="stHeader"],
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="stStatusWidget"],
      [data-testid="stFileUploader"],
      [data-testid="stSegmentedControl"],
      [data-testid="stSelectbox"],
      [data-testid="stMultiSelect"],
      [data-testid="stButton"],
      [data-testid="stDownloadButton"],
      div[role="tablist"],
      .platform-page-heading,
      footer { display: none !important; }
      [data-testid="stElementContainer"]:has(.platform-page-heading) {
        display: none !important;
      }
      html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        width: 100% !important;
        max-width: none !important;
        overflow: visible !important;
        background: #ffffff !important;
      }
      [data-testid="stMainBlockContainer"] {
        width: 100% !important;
        max-width: none !important;
        padding: 0 !important;
      }
      [data-testid="stExpander"] details > div { display: block !important; }
      [data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stExpander"] {
        break-inside: avoid-page;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def new_state_defaults() -> dict[str, object]:
    """Return fresh per-session values shared by initialization and profile reset."""
    return {
        "pdf_bytes": None,
        "pdf_name": "",
        "metadata": {},
        "page_matches": [],
        "auto_page_matches": [],
        "vlm_v2_locator_run": None,
        "vlm_v2_extraction_run": None,
        "vlm_v2_locator_elapsed": 0.0,
        "vlm_v2_extraction_elapsed": 0.0,
        "vlm_v2_workbook_bytes": b"",
        "vlm_v2_blind_summary": pd.DataFrame(),
        "vlm_v2_blind_details": pd.DataFrame(),
        "vlm_v2_blind_workbook_bytes": b"",
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
        "standard_workbook_bytes": b"",
        "standard_target_catalog": pd.DataFrame(),
        "standard_target_summary": pd.DataFrame(),
        "standardization_warnings": [],
        "standard_template_label": "",
        "integrated_data": pd.DataFrame(columns=STANDARD_COLUMNS),
        "integration_preview": pd.DataFrame(columns=STANDARD_COLUMNS),
        "integration_sheet_summary": pd.DataFrame(),
        "integration_mapping_summary": pd.DataFrame(),
        "integration_logic_checks": pd.DataFrame(),
        "integration_warnings": [],
        "integration_preview_mode": "",
        "step6_analysis_source": pd.DataFrame(columns=STANDARD_COLUMNS),
        "major_financing_data": pd.DataFrame(),
        "standard_schema_version": 0,
        "validation_results": pd.DataFrame(),
        "monitor_results": pd.DataFrame(),
        "monitor_single_result": None,
        "edited_pages": {},
        "pages_confirmed": False,
        "normalization_diagnostics": pd.DataFrame(),
        "locator_config_version": "",
        "extraction_grid_cache": {},
        "extraction_image_cache": {},
        "extraction_cache_lock": Lock(),
        "active_profile_id": DEFAULT_PROFILE_ID,
        "active_profile_runtime": "",
    }


def initialize_state() -> None:
    for key, value in new_state_defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.standard_schema_version < STANDARD_SCHEMA_VERSION:
        for key in ("standard_data", "integrated_data", "integration_preview"):
            st.session_state[key] = upgrade_standard_frame(st.session_state.get(key))
        st.session_state.standard_schema_version = STANDARD_SCHEMA_VERSION


def reset_profile_results() -> None:
    """Clear report-specific state when the user switches Profile."""
    empty_values = new_state_defaults()
    for key in (
        "llm_base_url",
        "llm_model",
        "llm_api_key",
        "auto_vision_retry",
        "force_vision_mode",
        "standard_schema_version",
        "extraction_cache_lock",
        "active_profile_id",
    ):
        empty_values.pop(key)
    for key, value in empty_values.items():
        st.session_state[key] = value
    st.session_state.pop("solvency_target_tables", None)
    for key in list(st.session_state):
        if str(key).startswith("page_edit_"):
            del st.session_state[key]


@st.cache_data(show_spinner=False)
def read_profile_registry() -> dict[str, ReportProfile]:
    return load_profile_registry(PROFILE_DIR, ROOT)


@st.cache_data(show_spinner=False)
def read_peer_groups(
    path: Path,
    modified_ns: int,
) -> tuple[dict[str, str], str]:
    del modified_ns
    return load_peer_group_config(path)


@st.cache_data(show_spinner=False)
def read_extracted_tables_workbook(
    workbook_bytes: bytes,
    profile_tables: list[dict],
) -> list[ExtractedTable]:
    """从 STEP2 下载的标准化提取 Excel 重建提取表格，供 STEP3 直接标准化。

    优先读取 VLM v2 指标结果并复核前置门槛；兼容旧版「表名_P页码」工作表。
    """
    if not workbook_bytes:
        return []

    def _sanitize(value: str) -> str:
        return re.sub(r"[\\/*?:\[\]]", "", str(value).strip())

    name_to_id = {
        _sanitize(str(item.get("table_name", ""))): str(item.get("table_id", ""))
        for item in profile_tables
        if item.get("table_name") and item.get("table_id")
    }
    ordered_names = sorted(name_to_id, key=len, reverse=True)

    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    if "VLM_v2指标结果" in excel.sheet_names:
        return read_vlm_v2_extracted_tables(excel)
    tables: list[ExtractedTable] = []
    table_index = 0
    for sheet_name in excel.sheet_names:
        if sheet_name in {"提取日志", "单位信息", "报告元信息"}:
            continue
        frame = pd.read_excel(excel, sheet_name=sheet_name, header=None)
        rows = [
            ["" if pd.isna(cell) else str(cell).strip() for cell in row]
            for row in frame.values.tolist()
        ]
        rows = [row for row in rows if row and row[0] != "【单位备注】"]
        rows = [row for row in rows if any(cell for cell in row)]
        if len(rows) < 2:
            continue
        sheet_key = _sanitize(sheet_name)
        table_id = ""
        for candidate in ordered_names:
            if sheet_key.startswith(candidate):
                table_id = name_to_id[candidate]
                break
        if not table_id:
            continue
        page_match = re.search(r"_P(\d+)", sheet_key)
        page = int(page_match.group(1)) if page_match else table_index + 1
        tables.append(
            ExtractedTable(
                table_id=table_id,
                table_name=sheet_name,
                page=page,
                table_index=table_index,
                rows=rows,
                strategy="手动上传",
                quality_score=0.0,
                evidence="来自本地 STEP2 标准化提取 Excel",
                source_pages=[page],
                unit_records=[],
                profile_strategy_id="",
            )
        )
        table_index += 1
    return tables


@st.cache_data(show_spinner=False)
def read_step3_upload_metadata(
    workbook_bytes: bytes,
    filename: str,
    company_items: tuple[tuple[str, str], ...],
    peer_group_items: tuple[tuple[str, str], ...],
    default_peer_group: str,
    context_metadata_items: tuple[tuple[str, str], ...],
    context_filename: str,
):
    company_candidates = tuple(
        {"company_name": name, "company_type": company_type}
        for name, company_type in company_items
    )
    return infer_step3_upload_metadata(
        workbook_bytes,
        filename,
        company_candidates,
        peer_group_map=dict(peer_group_items),
        default_peer_group=default_peer_group,
        context_metadata=dict(context_metadata_items),
        context_filename=context_filename,
    )


@st.cache_data(show_spinner=False)
def read_uploaded_profile(
    workbook_bytes: bytes,
    source_name: str,
) -> ReportProfile:
    return load_profile_workbook(
        workbook_bytes,
        project_root=ROOT,
        source_name=source_name,
    )


@st.cache_data(show_spinner=False)
def export_profile_workbook(profile: ReportProfile) -> bytes:
    return profile_workbook_bytes(profile)


@st.cache_data(show_spinner=False)
def prepare_visualization_source(
    frame: pd.DataFrame,
    report_profile_id: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    enriched = (
        add_missing_derived_metrics(frame)
        if report_profile_id == "LIFE_SOLVENCY"
        else frame
    )
    return prepare_analysis_frame(enriched, report_profile_id)


@st.cache_data(show_spinner=False)
def read_step6_analysis_workbook(
    workbook_bytes: bytes,
    source_name: str,
    report_profile_id: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    frame = read_standard_workbook(workbook_bytes, source_name)
    return prepare_visualization_source(frame, report_profile_id)


@st.cache_data(show_spinner=False, max_entries=8)
def read_major_financing_upload(
    workbook_bytes: bytes,
    source_name: str,
) -> pd.DataFrame:
    return read_major_financing_workbook(workbook_bytes, source_name)


@st.cache_data(show_spinner=False)
def read_taxonomy(path: str) -> pd.DataFrame:
    return load_taxonomy(path)


def extraction_taxonomy(profile: ReportProfile, path: str) -> pd.DataFrame:
    taxonomy = profile.taxonomy_frame() if profile.field_dictionary else read_taxonomy(path)
    return extend_filing_taxonomy(taxonomy) if profile.profile_id == 'LIFE_SOLVENCY' else taxonomy


@st.cache_data(show_spinner=False)
def read_rules(path: str) -> pd.DataFrame:
    return load_validation_rules(path)


@st.cache_data(show_spinner=False)
def read_gold_manifest() -> dict:
    return load_gold_manifest(GOLD_MANIFEST) if GOLD_MANIFEST.exists() else {"cases": []}


@st.cache_data(show_spinner=False)
def read_companies(
    source_path: str,
    sheet_name: str,
    header: int,
    company_column: str,
    company_type_column: str,
    report_url_column: str,
) -> pd.DataFrame:
    companies = pd.read_excel(
        source_path,
        sheet_name=sheet_name,
        header=header,
    )
    required = [company_column, company_type_column, report_url_column]
    missing = [column for column in required if column not in companies.columns]
    if missing:
        raise ValueError(f"公司网址配置缺少字段：{', '.join(missing)}")
    companies = companies[required].rename(columns={
        company_column: "公司",
        company_type_column: "公司类别",
        report_url_column: "报告披露地址",
    })
    for column in ("公司", "公司类别", "报告披露地址"):
        companies[column] = companies[column].fillna("").astype(str).str.strip()
    return companies[companies["公司"] != ""].drop_duplicates(subset=["公司"], keep="last").reset_index(drop=True)


def prepare_embedded_companies(profile: ReportProfile) -> pd.DataFrame:
    companies = profile.company_frame()
    if companies.empty:
        return companies
    columns = {
        str(profile.monitoring["company_name_column"]): "公司",
        str(profile.monitoring["company_type_column"]): "公司类别",
        str(profile.monitoring["report_url_column"]): "报告披露地址",
    }
    companies = companies.rename(columns=columns)
    for column in ("公司", "公司类别", "报告披露地址"):
        companies[column] = companies[column].fillna("").astype(str).str.strip()
    return companies[companies["公司"] != ""].drop_duplicates(
        subset=["公司"], keep="last",
    ).reset_index(drop=True)


def keep_valid_widget_state(
    key: str,
    options: list[str],
    *,
    multiple: bool = False,
) -> None:
    """Discard stale selections when upstream Step6 filters change."""
    if key not in st.session_state:
        return
    if multiple:
        current = st.session_state.get(key) or []
        st.session_state[key] = [value for value in current if value in options]
    elif st.session_state.get(key) not in options:
        del st.session_state[key]


def render_all_chart_navigation(chart_options: list[str]) -> None:
    """Render read-only chart rows with the same visual rhythm as sidebar radios."""
    if not chart_options:
        return
    rows = "".join(
        (
            '<div class="kpmg-nav-all-item">'
            '<span class="kpmg-nav-all-dot" aria-hidden="true"></span>'
            f'<span>{html.escape(str(name))}</span>'
            "</div>"
        )
        for name in chart_options
    )
    st.markdown(
        (
            '<div class="kpmg-nav-all-charts">'
            '<div class="kpmg-nav-all-label">具体图表（全部展示）</div>'
            f"{rows}</div>"
        ),
        unsafe_allow_html=True,
    )


def navigation_metric_codes(frame: pd.DataFrame) -> list[str]:
    if "指标编码" not in frame.columns:
        return []
    return frame["指标编码"].dropna().astype(str).str.strip().tolist()


def render_report_navigation(
    *,
    title: str,
    container_key: str,
    state_prefix: str,
    available_codes: list[str],
    print_key: str,
    industry: bool = False,
) -> None:
    with st.container(key=container_key):
        st.markdown(f"### {title}")

    first_options = (
        first_levels_for_codes(available_codes, industry=True)
        if industry
        else COMPANY_FIRST_LEVELS
    )
    level_one_key = f"{state_prefix}_level_one"
    keep_valid_widget_state(level_one_key, first_options)
    main_nav = st.radio("📁 一级模块", first_options, key=level_one_key)
    if main_nav == PRINT_ALL_LABEL:
        render_print_control(key=print_key)
        return

    navigation_level = OVERVIEW_LEVEL if industry else main_nav
    chart_key = f"{state_prefix}_chart"
    second_options = [
        "全部",
        *second_levels(
            navigation_level,
            industry=industry,
            available_codes=available_codes if industry else None,
        ),
    ]
    level_two_key = f"{state_prefix}_level_two"
    previous_level_one_key = f"_{state_prefix}_previous_level_one"
    if st.session_state.get(previous_level_one_key) != main_nav:
        st.session_state[previous_level_one_key] = main_nav
        st.session_state[level_two_key] = (
            second_options[1] if len(second_options) > 1 else "全部"
        )
        st.session_state.pop(chart_key, None)
    keep_valid_widget_state(level_two_key, second_options)
    second_nav = st.radio("📂 二级模块", second_options, key=level_two_key)
    chart_options = chart_names(
        navigation_level,
        second_nav,
        industry=industry,
        available_codes=available_codes if industry else None,
    )
    if second_nav == "全部":
        st.session_state.pop(chart_key, None)
        render_all_chart_navigation(chart_options)
    else:
        keep_valid_widget_state(chart_key, chart_options)
    if chart_options and second_nav != "全部":
        st.radio("具体图表", chart_options, key=chart_key)


def render_major_financing_upload(profile_id: str) -> None:
    st.markdown("### 补充数据")
    financing_upload = st.file_uploader(
        "上传重大融资信息 Excel",
        type=["xlsx"],
        key=f"major_financing_upload_{profile_id}",
        help="应包含季度、公司名称、增资/发债及综合充足率变动字段。",
    )
    if financing_upload is None:
        return
    try:
        st.session_state.major_financing_data = read_major_financing_upload(
            financing_upload.getvalue(),
            financing_upload.name,
        )
        st.caption(
            f"已读取 {len(st.session_state.major_financing_data):,} 条融资事件。"
        )
    except Exception as exc:
        st.error(f"重大融资信息读取失败：{exc}")


def render_company_report_workspace(
    data: pd.DataFrame,
) -> None:
    """Render Step7 with normal page reruns so settings control all content below."""
    show_step_7_solvency(data, st.session_state.major_financing_data)


def render_industry_report_workspace(
    data: pd.DataFrame,
) -> None:
    """Render Step8 with the same normal page-rerun model as Step6 and Step7."""
    show_step_8_solvency(data, st.session_state.major_financing_data)


def target_period_terms(
    year: int,
    period: str,
    frequency: str = "QUARTERLY",
) -> list[str]:
    if frequency.upper() == "ANNUAL":
        return [
            f"{year}年度",
            f"{year}年年度报告",
            f"{year}年年报",
            f"annualreport{year}",
        ]
    quarter = period
    quarter_number = int(quarter[-1])
    chinese_number = {1: "一", 2: "二", 3: "三", 4: "四"}[quarter_number]
    return [
        f"{year}{quarter}",
        f"{year}年第{quarter_number}季度",
        f"{year}年第{chinese_number}季度",
        f"{year}年{quarter_number}季度",
        f"{year}年{chinese_number}季度",
    ]


def check_company_report(
    row: pd.Series,
    year: int,
    period: str,
    *,
    frequency: str = "QUARTERLY",
    report_name: str = "目标报告",
    report_terms: tuple[str, ...] = (),
    timeout: int = 15,
) -> dict:
    company = str(row.get("公司", "")).strip()
    peer_group = str(row.get("同业分类", "")).strip()
    url = str(row.get("报告披露地址", "")).strip()
    checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    target_period = str(year) if frequency.upper() == "ANNUAL" else f"{year}{period}"
    base = {
        "公司": company,
        "同业分类": peer_group,
        "报告类型": report_name,
        "目标报告期": target_period,
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
            matched = metadata.get("报告年度") == year
            if frequency.upper() != "ANNUAL":
                matched = matched and metadata.get("报告季度") == period
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
        terms = target_period_terms(year, period, frequency)
        matched_terms = [term for term in terms if term.lower() in searchable.lower()]
        normalized_report_terms = tuple(
            item for item in report_terms if str(item).strip()
        )
        has_report_term = (
            any(
                str(term).lower() in searchable.lower()
                for term in normalized_report_terms
            )
            if normalized_report_terms
            else True
        )

        if matched_terms and has_report_term:
            result = "已更新"
            note = "页面中同时发现目标报告期和报告类型关键词。"
        elif matched_terms:
            result = "疑似已更新，需人工核对"
            note = "发现目标报告期，但未在静态页面文本中发现报告类型关键词。"
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


LOCATOR_SOURCE_LABELS = {
    "local": "正文标题 / 首尾边界",
    "semantic": "大模型标题语义",
    "vision": "图片标题确认",
    "directory": "目录页码换算",
    "radar": "表格结构雷达",
    **LOCATOR_ROLE_LABELS,
}


def locator_candidate_rows(match: PageMatch) -> list[dict[str, object]]:
    """Expose every locator channel for transparent human page review."""
    rows = [{
        "候选来源": "系统收敛结果（推荐）",
        "物理页码": list(match.pages),
        "用途": "当前推荐提取范围；仍需结合右侧PDF人工确认",
    }]
    for source_id, label in LOCATOR_SOURCE_LABELS.items():
        pages = sorted(set(match.sources.get(source_id, [])))
        if not pages:
            continue
        rows.append({
            "候选来源": label,
            "物理页码": pages,
            "用途": (
                "高置信正文证据"
                if source_id == "local"
                else "辅助候选，采用前请核对右侧PDF原页"
            ),
        })
    return rows


def apply_locator_candidate(input_key: str, pages: list[int]) -> None:
    st.session_state[input_key] = ", ".join(map(str, pages))
    st.session_state.pages_confirmed = False


def render_locator_evidence(match: PageMatch, input_key: str) -> None:
    """Keep long evidence and candidate controls in one collapsed detail area."""
    with st.expander("查看定位证据 / 目录 / 标题 / 结构候选来源", expanded=False):
        auto_text = ", ".join(map(str, match.pages)) if match.pages else "未自动找到"
        st.caption(f"自动定位：{auto_text}")
        st.caption(f"识别依据：{match.evidence or '无明确关键词证据'}")
        if match.review_required:
            st.warning(match.review_reason or "请人工核对候选范围")
        candidate_rows = locator_candidate_rows(match)
        display_frame = pd.DataFrame(candidate_rows)
        display_frame["物理页码"] = display_frame["物理页码"].map(
            lambda pages: ", ".join(map(str, pages)) or "未找到"
        )
        st.dataframe(display_frame, width="stretch", hide_index=True)
        selected_candidate = st.selectbox(
            "选择一个候选范围", range(len(candidate_rows)),
            format_func=lambda index: (
                f"{candidate_rows[index]['候选来源']}："
                f"{', '.join(map(str, candidate_rows[index]['物理页码'])) or '未找到'}"
            ), key=f"locator_candidate_{match.table_id}",
        )
        st.button(
            "采用所选候选页", key=f"apply_locator_candidate_{match.table_id}",
            on_click=apply_locator_candidate,
            args=(input_key, list(candidate_rows[selected_candidate]["物理页码"])),
            width="stretch",
        )


@st.cache_data(show_spinner=False, max_entries=8)
def dataframe_to_xlsx(frame: pd.DataFrame, sheet_name: str) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


@st.cache_data(show_spinner=False, max_entries=12)
def cached_convert_external_workbook(
    workbook_bytes: bytes,
    filename: str,
    taxonomy: pd.DataFrame,
    company_type_items: tuple[tuple[str, str], ...],
    report_profile_id: str,
):
    return convert_external_workbook(
        workbook_bytes,
        filename,
        taxonomy,
        dict(company_type_items),
        report_profile_id=report_profile_id,
    )


@st.cache_data(show_spinner=False, max_entries=128)
def cached_read_standard_workbook(
    workbook_bytes: bytes,
    filename: str,
) -> pd.DataFrame:
    return read_standard_workbook(workbook_bytes, filename)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_complete_step5_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return add_missing_derived_metrics(frame)


@st.cache_data(show_spinner=False, max_entries=8)
def validation_results_to_xlsx(
    results: pd.DataFrame,
    step3_workbook: bytes,
) -> bytes:
    return validation_workbook_bytes(results, step3_workbook)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_step3_target_template(
    target_catalog: pd.DataFrame,
    profile_name: str,
) -> bytes:
    return target_template_workbook_bytes(target_catalog, profile_name)


def sync_analysis_navigation_from_tab() -> None:
    target = st.session_state.get("workflow_main_tab", "")
    if target in ANALYSIS_TAB_LABELS:
        st.session_state.analysis_page_navigation = target


initialize_state()

is_project_member = st.session_state.user_role == PROJECT_MEMBER_ROLE
with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
    st.caption(
        f"当前身份：{st.session_state.user_role}　｜　"
        f"AI：{st.session_state.get('llm_provider', '自定义节点')} / "
        f"{st.session_state.get('llm_model', '') or '未配置'}"
    )
    st.button(
        "退出登录",
        icon=":material/logout:",
        key="logout",
        on_click=_logout,
        width="content",
    )

if not is_project_member:
    st.info(
        "普通用户通道仅开放 STEP6 自定义分析与 STEP7 公司报告。",
        icon=":material/lock:",
    )

profiles = read_profile_registry()
with st.expander(
    "报告 Profile 设置",
    expanded=False,
    icon=":material/tune:",
):
    profile_settings_tab_labels = (
        "切换报告类型",
        "上传高级配置",
        "下载或核对",
    )
    if is_project_member:
        profile_switch_tab, profile_upload_tab, profile_review_tab = st.tabs(
            profile_settings_tab_labels,
            default=profile_settings_tab_labels[0],
            key="profile_settings_tabs",
        )
    else:
        profile_switch_tab = st.container()

    if is_project_member:
        with profile_upload_tab:
            st.caption(
                "目标表和定位关键词可以通过配置工作簿维护。上传后先进行结构与引用校验，"
                "仅在当前浏览器会话中生效，不会覆盖项目内的正式配置。"
            )
            uploaded_profile_file = st.file_uploader(
                "上传报告 Profile 配置工作簿",
                type=["xlsx"],
                key="report_profile_upload",
            )
            if uploaded_profile_file is not None:
                uploaded_bytes = uploaded_profile_file.getvalue()
                try:
                    uploaded_profile = read_uploaded_profile(
                        uploaded_bytes,
                        uploaded_profile_file.name,
                    )
                    profiles[uploaded_profile.profile_id] = uploaded_profile
                    st.success(
                        f"配置校验通过：{uploaded_profile.profile_name}，"
                        f"共 {len(uploaded_profile.tables)} 张目标表。"
                    )
                except ProfileValidationError as exc:
                    st.error(f"配置工作簿未启用：{exc}")

    profile_ids = list(profiles)
    if st.session_state.active_profile_id not in profile_ids:
        st.session_state.active_profile_id = profile_ids[0]

    with profile_switch_tab:
        st.segmented_control(
            "当前报告类型",
            profile_ids,
            format_func=lambda profile_id: profiles[profile_id].profile_name,
            selection_mode="single",
            required=True,
            key="active_profile_id",
            on_change=reset_profile_results,
            width="stretch",
            help="寿险与财险使用相互隔离的公司范围、目标表、关键词和比较数据。",
        )
        st.caption(
            "切换后会清空当前未保存的 PDF、页码、提取表格和标准化结果，"
            "防止不同 Profile 的数据混用。"
        )

    active_profile = profiles[st.session_state.active_profile_id]
    if st.session_state.active_profile_runtime != active_profile.runtime_version:
        st.session_state.active_profile_runtime = active_profile.runtime_version
        st.session_state.locator_config_version = ""
        st.session_state.monitor_results = pd.DataFrame()
        st.session_state.monitor_single_result = None
        st.session_state.pop("solvency_target_tables", None)

    if is_project_member:
        with profile_review_tab:
            st.caption(
                f"Profile：{active_profile.config_version}　|　"
                f"工作簿架构：v{active_profile.workbook_schema_version}　|　"
                f"比较范围：同一 profile 内跨公司、跨期　|　"
                f"目标表：{len(active_profile.tables)} 张"
            )
            st.caption(
                f"v2 配置：版式变体 {len(active_profile.layout_variants)} 条，"
                f"字段字典 {len(active_profile.field_dictionary)} 条，"
                f"完整性规则 {len(active_profile.completeness_rules)} 条，"
                f"公司来源 {len(active_profile.companies)} 家，"
                f"Gold 样本 {len(active_profile.gold_samples)} 条。"
                "旧版三表工作簿仍可上传。"
            )
            st.dataframe(
                pd.DataFrame(active_profile.tables).reindex(
                    columns=["table_id", "table_name", "max_pages"],
                ),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "下载当前 Profile 配置工作簿",
                export_profile_workbook(active_profile),
                f"{active_profile.profile_id}_profile.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
            )

taxonomy_path = (
    active_profile.resource_path("normalization", "taxonomy_file")
    if not active_profile.field_dictionary
    else None
)
rules_path = active_profile.resource_path("validation", "validation_rules_file")
company_source_path = (
    active_profile.resource_path("monitoring", "company_source_file")
    if not active_profile.companies
    else None
)

st.markdown(
    (
        '<section class="platform-page-heading">'
        '<h1>保险报告处理与分析平台</h1>'
        '<p class="platform-profile-caption">'
        f"当前 profile：{active_profile.profile_name}（{active_profile.profile_id}）｜"
        "支持同一报告类型内跨公司、跨期比较"
        "</p></section>"
    ),
    unsafe_allow_html=True,
)

visible_workflow_labels = (
    WORKFLOW_TAB_LABELS
    if is_project_member
    else WORKFLOW_TAB_LABELS[6:8]
)
visible_tabs = st.tabs(
    visible_workflow_labels,
    default=visible_workflow_labels[0],
    key="workflow_main_tab",
    on_change=sync_analysis_navigation_from_tab,
)
tabs = (
    {index: tab for index, tab in enumerate(visible_tabs)}
    if is_project_member
    else {6: visible_tabs[0], 7: visible_tabs[1]}
)
active_workflow_tab = st.session_state.get(
    "workflow_main_tab",
    visible_workflow_labels[0],
)
analysis_tabs_active = active_workflow_tab in ANALYSIS_TAB_LABELS
if analysis_tabs_active:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid #D9E2F1; }
        [data-testid="stSidebar"] h3 { color:#0C233C !important; font-size:18px !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] { margin-bottom:8px; }
        [data-testid="stExpander"] details {
          border:1px solid #D5DEEC; border-radius:8px; background:#F7F9FC;
        }
        [data-testid="stExpander"] details[open] { border-color:#B8C9E5; }
        [data-testid="stExpander"] summary { color:#0C233C; font-weight:650; }
        .st-key-kpmg_company_nav_title, .st-key-kpmg_industry_nav_title {
          background:#FFFFFF; border-radius:8px; padding:8px 10px;
          box-shadow:0 4px 14px rgba(12,35,60,.06); margin:8px 0 10px 0;
        }
        .kpmg-nav-all-charts {
          margin:0 0 8px; color:#0C233C; font-family:inherit;
          font-size:0.875rem; font-weight:400; line-height:1.4;
        }
        .kpmg-nav-all-label { margin:0 0 4px; font:inherit; }
        .kpmg-nav-all-item {
          display:flex; align-items:center; gap:8px; min-height:24px;
          margin:0; padding:0; font:inherit;
        }
        .kpmg-nav-all-dot {
          width:14px; height:14px; flex:0 0 14px; box-sizing:border-box;
          border:1px solid #CDD4DE; border-radius:50%; background:#F0F2F6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if is_project_member:
    with tabs[0]:
        st.subheader(f"{active_profile.profile_name}监控")
        companies = prepare_embedded_companies(active_profile)
        if companies.empty:
            companies = read_companies(
                str(company_source_path),
                str(active_profile.monitoring["company_source_sheet"]),
                int(active_profile.monitoring.get("company_source_header", 0) or 0),
                str(active_profile.monitoring["company_name_column"]),
                str(active_profile.monitoring["company_type_column"]),
                str(active_profile.monitoring["report_url_column"]),
            )
        monitor_peer_group_map, monitor_default_peer_group = read_peer_groups(
            PEER_GROUP_CONFIG,
            PEER_GROUP_CONFIG.stat().st_mtime_ns,
        )
        companies = companies.copy()
        companies["同业分类"] = companies["公司"].map(
            lambda name: resolve_peer_group(
                name,
                monitor_peer_group_map,
                monitor_default_peer_group,
            )
        )

        year_col, quarter_col = st.columns(2)
        target_year = int(year_col.number_input("报告年度", 2020, 2050, 2026))
        if active_profile.frequency.upper() == "ANNUAL":
            target_period = "ANNUAL"
            quarter_col.text_input("报告期间", value="年度", disabled=True)
        else:
            target_period = quarter_col.selectbox(
                "报告季度",
                ["Q1", "Q2", "Q3", "Q4"],
            )

        all_categories = sorted(companies["同业分类"].dropna().unique().tolist())
        selected_categories = st.multiselect(
            "同业分类",
            all_categories,
            default=all_categories,
            key="monitor_peer_groups",
        )
        filtered_companies = companies[
            companies["同业分类"].isin(selected_categories)
        ].reset_index(drop=True)

        st.caption(
            f"系统公司范围共 {len(companies)} 家；当前筛选 {len(filtered_companies)} 家。"
            f"运行时使用 {active_profile.profile_id} 的公司来源配置。"
        )
        st.dataframe(
            filtered_companies[["公司", "同业分类", "报告披露地址"]],
            width="stretch",
            hide_index=True,
            column_config={
                "报告披露地址": st.column_config.LinkColumn(
                    "报告披露地址",
                    display_text="打开页面",
                )
            },
        )

        st.markdown("#### 逐个查看")
        if filtered_companies.empty:
            st.warning("当前同业分类筛选下没有公司。")
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
                selected_row["报告披露地址"],
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
                        target_period,
                        frequency=active_profile.frequency,
                        report_name=active_profile.profile_name,
                        report_terms=tuple(
                            active_profile.monitoring.get("report_terms", [])
                        ),
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
                        target_period,
                        frequency=active_profile.frequency,
                        report_name=active_profile.profile_name,
                        report_terms=tuple(
                            active_profile.monitoring.get("report_terms", [])
                        ),
                        timeout=request_timeout,
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
                f"{active_profile.profile_id}_报告更新检查_"
                f"{target_year}{'' if target_period == 'ANNUAL' else target_period}.xlsx",
                width="stretch",
            )



    with tabs[1]:
        st.subheader("📑 智能页码定位")
        st.caption("系统使用VLM v2直接浏览整份PDF并定位目标表；跨页页码可在左侧人工修改，右侧同步显示PDF原页。")

        feature_config = dict(active_profile.feature_config)
        table_configs = feature_config.get("tables", [])
        config_version = active_profile.runtime_version
        if st.session_state.locator_config_version != config_version:
            st.session_state.locator_config_version = config_version
            st.session_state.auto_page_matches = []
            st.session_state.page_matches = []
            st.session_state.edited_pages = {}
            st.session_state.pages_confirmed = False
            st.session_state.raw_tables = []
            st.session_state.ai_extraction_logs = []
            st.session_state.ai_workbook_bytes = b""
            st.session_state.vlm_v2_locator_run = None
            st.session_state.vlm_v2_extraction_run = None
            st.session_state.vlm_v2_locator_elapsed = 0.0
            st.session_state.vlm_v2_extraction_elapsed = 0.0
            st.session_state.vlm_v2_workbook_bytes = b""
            st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
            st.session_state.validation_results = pd.DataFrame()
            st.session_state.normalization_diagnostics = pd.DataFrame()
            for item in table_configs:
                st.session_state.pop(f"page_edit_{item['table_id']}", None)
        table_config_by_name = {item["table_name"]: item for item in table_configs}

        uploaded_pdf = st.file_uploader(
            f"拖拽或选择一份{active_profile.profile_name} PDF",
            type=["pdf"],
            key=f"report_pdf_{active_profile.profile_id}",
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
                st.session_state.pages_confirmed = False
                st.session_state.raw_tables = []
                st.session_state.table_candidates = {}
                st.session_state.selected_table_candidates = {}
                st.session_state.ai_extraction_logs = []
                st.session_state.ai_workbook_bytes = b""
                st.session_state.vlm_v2_locator_run = None
                st.session_state.vlm_v2_extraction_run = None
                st.session_state.vlm_v2_locator_elapsed = 0.0
                st.session_state.vlm_v2_extraction_elapsed = 0.0
                st.session_state.vlm_v2_workbook_bytes = b""
                st.session_state.extraction_grid_cache = {}
                st.session_state.extraction_image_cache = {}
                st.session_state.extraction_cache_lock = Lock()
                st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
                st.session_state.validation_results = pd.DataFrame()
                st.session_state.normalization_diagnostics = pd.DataFrame()
                for item in table_configs:
                    st.session_state.pop(f"page_edit_{item['table_id']}", None)

            total_pages = int(st.session_state.metadata.get("页数", 0) or 0)
            st.caption(f"当前文件：{uploaded_pdf.name}　|　文档共 {total_pages} 页")
            identity_warning = report_identity_warning(
                uploaded_pdf.name,
                st.session_state.metadata,
            )
            if identity_warning:
                st.warning(identity_warning, icon=":material/warning:")
            gold_case = find_gold_case(incoming, read_gold_manifest())
            if gold_case:
                st.success(
                    f"已匹配真实PDF金标准：{gold_case['case_id']}。"
                    "定位和提取结果会自动显示基准对比。"
                )
            with st.expander("查看自动识别的报告基本信息"):
                st.json(st.session_state.metadata)

            col_left, col_spacer, col_right = st.columns([1, 0.05, 1.2])

            with col_left:
                st.markdown("#### 检索目标设定")
                all_table_names = [item["table_name"] for item in table_configs]
                required_table_names = [
                    item["table_name"]
                    for item in table_configs
                    if item.get("required", True)
                ]
                selected_table_names = st.multiselect(
                    "请选择需要定位的报表：",
                    all_table_names,
                    default=required_table_names,
                    key="solvency_target_tables",
                )
                selected_configs = [
                    table_config_by_name[name]
                    for name in selected_table_names
                    if name in table_config_by_name
                ]

                configured_provider = st.session_state.get("llm_provider", "自定义节点")
                configured_model = st.session_state.get("llm_model", "") or "未配置"
                st.caption(
                    f"AI 配置来自登录页：{configured_provider} / {configured_model}。"
                    "Step1、Step2 与 Step7 共用该配置；如需更改，请退出后重新登录。"
                )
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
                        st.error(
                            "当前登录会话的 AI 配置不完整。请退出登录，在登录页填写 "
                            "Base URL、模型名称和 API Key 后重新进入。"
                        )
                    else:
                        try:
                            vlm_v2_taxonomy = extraction_taxonomy(active_profile, str(taxonomy_path))
                            started = perf_counter()
                            with st.status(
                                "VLM v2正在分批浏览整份PDF并定位目标表...",
                                expanded=True,
                            ) as locator_status:
                                locator_progress = st.progress(
                                    0,
                                    text="正在生成每批最多6页的定位拼图...",
                                )

                                def update_locator_progress(event):
                                    phase = str(event.get("phase", ""))
                                    elapsed = float(event.get("elapsed_seconds", 0.0) or 0.0)
                                    if phase == "locator_started":
                                        total = max(1, int(event.get("total_batches", 0) or 0))
                                        locator_progress.progress(
                                            5,
                                            text=f"开始定位，共{total}批，每批最多6页 · 已用时{elapsed:.1f}秒",
                                        )
                                    elif phase == "locator_batch_completed":
                                        completed = int(event.get("completed_batches", 0) or 0)
                                        total = max(1, int(event.get("total_batches", 0) or 0))
                                        batch_index = int(event.get("batch_index", 0) or 0)
                                        round_name = str(event.get("round_name", "首轮"))
                                        status_text = str(event.get("status", ""))
                                        pages = event.get("visible_pages", [])
                                        page_text = (
                                            f"{min(pages)}-{max(pages)}页"
                                            if pages else "未知页"
                                        )
                                        base = 10 if round_name == "首轮" else 82
                                        span = 70 if round_name == "首轮" else 14
                                        locator_progress.progress(
                                            min(96, base + round(span * completed / total)),
                                            text=(
                                                f"{round_name}完成 {completed}/{total}："
                                                f"批次{batch_index}（{page_text}，{status_text}）"
                                                f" · 已用时{elapsed:.1f}秒"
                                            ),
                                        )
                                        locator_status.write(
                                            f"{round_name}批次{batch_index}：{page_text}，{status_text}"
                                        )
                                    elif phase == "locator_retry_started":
                                        indexes = "、".join(
                                            str(value) for value in event.get("batch_indexes", [])
                                        )
                                        locator_progress.progress(
                                            82,
                                            text=f"仅重试超时批次：{indexes} · 已用时{elapsed:.1f}秒",
                                        )
                                    elif phase == "locator_completed":
                                        locator_progress.progress(
                                            100,
                                            text=f"页面定位批次处理完成 · 总用时{elapsed:.1f}秒",
                                        )

                                vlm_run = locate_tables_vlm_v2(
                                    incoming,
                                    selected_configs,
                                    vlm_v2_taxonomy,
                                    api_key=st.session_state.llm_api_key,
                                    base_url=st.session_state.llm_base_url,
                                    model=st.session_state.llm_model,
                                    timeout=90,
                                    request_max_attempts=2,
                                    pages_per_sheet=6,
                                    sheets_per_call=1,
                                    max_workers=2,
                                    progress_callback=update_locator_progress,
                                )
                                scan_diagnostics = vlm_run.diagnostics[
                                    vlm_run.diagnostics["轮次"].isin(["首轮", "超时重试"])
                                ]
                                latest_batches = (
                                    scan_diagnostics.sort_values("调用序号", kind="stable")
                                    .groupby("批次序号", sort=False)
                                    .tail(1)
                                )
                                unresolved_batches = latest_batches[
                                    latest_batches["状态"] != "成功"
                                ]
                                locator_status.update(
                                    label=(
                                        "VLM v2定位完成"
                                        if unresolved_batches.empty
                                        else f"VLM v2定位完成，{len(unresolved_batches)}个批次仍需人工核验"
                                    ),
                                    state="complete" if unresolved_batches.empty else "error",
                                    expanded=not unresolved_batches.empty,
                                )
                            matches = list(vlm_run.matches)
                            st.session_state.vlm_v2_locator_run = vlm_run
                            st.session_state.vlm_v2_locator_elapsed = perf_counter() - started
                            st.session_state.vlm_v2_extraction_run = None
                            st.session_state.vlm_v2_extraction_elapsed = 0.0
                            st.session_state.vlm_v2_workbook_bytes = b""
                            st.session_state.auto_page_matches = matches
                            st.session_state.page_matches = matches
                            st.session_state.pages_confirmed = False
                            st.session_state.edited_pages = {
                                item.table_id: list(item.pages)
                                for item in matches
                            }
                            for item in matches:
                                st.session_state[f"page_edit_{item.table_id}"] = ", ".join(map(str, item.pages))
                            located_count = sum(bool(item.pages) for item in matches)
                            if located_count:
                                st.success(
                                    f"定位完成：{located_count}类目标已找到候选页，"
                                    f"用时{st.session_state.vlm_v2_locator_elapsed:.1f}秒。"
                                    "请结合右侧PDF原页进行校准并确认。"
                                )
                            else:
                                st.warning("尚未自动找到目标页，请直接填写物理页码。")
                            if not unresolved_batches.empty:
                                failed_pages = "、".join(
                                    unresolved_batches["扫描物理页"].astype(str).tolist()
                                )
                                st.warning(
                                    f"以下页面批次在超时重试后仍未成功：{failed_pages}。"
                                    "其他批次的定位结果已经保留，请在下方结合PDF原页人工补充。"
                                )
                                with st.expander("查看未成功的定位批次", expanded=False):
                                    st.dataframe(
                                        unresolved_batches[[
                                            "批次序号", "扫描物理页", "轮次", "状态", "错误",
                                        ]],
                                        width="stretch",
                                        hide_index=True,
                                    )
                        except Exception as exc:
                            st.error(f"智能定位失败：{exc}")

                if st.session_state.auto_page_matches:
                    st.markdown("---")
                    st.markdown("#### 结果核对")
                    st.caption("若表格跨页，请用逗号分隔物理页码，例如：26, 27。未找到时可直接手工输入。")

                    edited_pages: dict[str, list[int]] = {}
                    updated_matches: list[PageMatch] = []
                    conflict_matches = [
                        item
                        for item in st.session_state.auto_page_matches
                        if item.review_required
                    ]
                    if conflict_matches:
                        st.warning(
                            "以下目标需要人工核对，具体原因见各目标下方的折叠定位证据："
                            + "、".join(item.table_name for item in conflict_matches)
                        )
                    saved_locator = st.session_state.vlm_v2_locator_run
                    if saved_locator is not None and not saved_locator.diagnostics.empty:
                        with st.expander("查看定位批次与补查记录", expanded=False):
                            st.caption("请求超时、模型漏回与未命中分别记录；未定位不代表报告未披露。")
                            st.dataframe(saved_locator.diagnostics, width="stretch", hide_index=True)
                    for match in st.session_state.auto_page_matches:
                        input_key = f"page_edit_{match.table_id}"
                        if input_key not in st.session_state:
                            st.session_state[input_key] = ", ".join(map(str, match.pages))
                        raw_value = st.text_input(match.table_name, key=input_key)
                        valid_pages, invalid_values = parse_page_numbers(raw_value, total_pages)
                        previous_pages = st.session_state.edited_pages.get(match.table_id)
                        if previous_pages is not None and previous_pages != valid_pages:
                            st.session_state.pages_confirmed = False
                        edited_pages[match.table_id] = valid_pages
                        updated_matches.append(
                            PageMatch(
                                table_id=match.table_id,
                                table_name=match.table_name,
                                pages=valid_pages,
                                score=match.score,
                                evidence=match.evidence,
                                review_required=match.review_required,
                                review_reason=match.review_reason,
                                sources=match.sources,
                                strategy_id=match.strategy_id,
                                table_config=match.table_config,
                            )
                        )
                        render_locator_evidence(match, input_key)
                        if match.review_required:
                            st.warning("此目标需要人工核对，原因见上方折叠的定位证据。")
                        if invalid_values:
                            st.warning(f"以下页码或内容无效，已忽略：{', '.join(invalid_values)}")

                    st.session_state.edited_pages = edited_pages
                    st.session_state.page_matches = updated_matches

                    if gold_case:
                        st.markdown("##### 真实PDF金标准 - 页码定位")
                        st.dataframe(
                            evaluate_gold_case(
                                gold_case,
                                matches=updated_matches,
                            ),
                            width="stretch",
                            hide_index=True,
                        )

                    if st.button(
                        "确认页码，进入下一步",
                        key="confirm_solvency_pages",
                        width="stretch",
                    ):
                        st.session_state.pages_confirmed = True
                        locator_run = st.session_state.vlm_v2_locator_run
                        if locator_run is not None:
                            st.session_state.vlm_v2_locator_run = VLMV2LocatorRun(
                                matches=tuple(updated_matches),
                                diagnostics=locator_run.diagnostics,
                                model_calls=locator_run.model_calls,
                                page_count=locator_run.page_count,
                            )
                        valid_count = sum(bool(item.pages) for item in updated_matches)
                        conflict_note = (
                            "；定位冲突已由人工确认"
                            if conflict_matches else ""
                        )
                        st.success(
                            f"页码已确认：{len(updated_matches)}类目标中有"
                            f" {valid_count} 类配置了有效页码{conflict_note}。"
                            "请前往 STEP2 提取表格。"
                        )

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
            "VLM v2从STEP1人工确认的PDF页面直接提取标准指标，完成确定性业务校验，"
            "并只对失败指标进行一次高清局部重试。"
        )

        with st.expander("VLM v2 批量盲测评分", expanded=False):
            st.caption(
                "先运行全部PDF，再读取金标准统一评分；金标准内容不会进入任何模型提示词。"
                "评分拆分页码、披露状态、标准数值、本季度口径、调用次数和耗时。"
            )
            st.download_button(
                "下载盲测金标准模板",
                blind_test_template_bytes(),
                "VLM_v2批量盲测金标准模板.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="vlm_v2_blind_template_download",
            )
            blind_gold_upload = st.file_uploader(
                "上传已填写的盲测金标准",
                type=["xlsx"],
                key="vlm_v2_blind_gold_upload",
            )
            blind_pdf_uploads = st.file_uploader(
                "上传未参与开发的公司PDF（可多选）",
                type=["pdf"],
                accept_multiple_files=True,
                key="vlm_v2_blind_pdf_uploads",
            )
            if st.button(
                "运行批量盲测并评分",
                key="vlm_v2_blind_run",
                type="primary",
                disabled=not blind_gold_upload or not blind_pdf_uploads,
                width="stretch",
            ):
                try:
                    gold_cases = read_blind_gold_workbook(blind_gold_upload.getvalue())
                    blind_taxonomy = extraction_taxonomy(active_profile, str(taxonomy_path))
                    blind_configs = [dict(item) for item in active_profile.tables]
                    scores = []
                    progress = st.progress(0, text="准备批量盲测…")
                    for index, uploaded_pdf in enumerate(blind_pdf_uploads, start=1):
                        pdf_bytes = uploaded_pdf.getvalue()
                        gold_case = match_blind_case(uploaded_pdf.name, pdf_bytes, gold_cases)
                        if gold_case is None:
                            raise ValueError(f"{uploaded_pdf.name} 未在金标准中唯一匹配到样本。")
                        progress.progress(
                            (index - 1) / len(blind_pdf_uploads),
                            text=f"正在处理 {index}/{len(blind_pdf_uploads)}：{uploaded_pdf.name}",
                        )
                        started = perf_counter()
                        blind_locator = locate_tables_vlm_v2(
                            pdf_bytes, blind_configs, blind_taxonomy,
                            api_key=st.session_state.llm_api_key,
                            base_url=st.session_state.llm_base_url,
                            model=st.session_state.llm_model,
                        )
                        blind_extraction = extract_metrics_vlm_v2(
                            pdf_bytes, blind_locator.matches, blind_configs, blind_taxonomy,
                            api_key=st.session_state.llm_api_key,
                            base_url=st.session_state.llm_base_url,
                            model=st.session_state.llm_model,
                            auto_retry=True,
                        )
                        scores.append(score_blind_case(
                            gold_case, blind_locator, blind_extraction,
                            filename=uploaded_pdf.name,
                            elapsed_seconds=perf_counter() - started,
                        ))
                    combined = combine_blind_scores(scores)
                    st.session_state.vlm_v2_blind_summary = combined.summary
                    st.session_state.vlm_v2_blind_details = combined.details
                    st.session_state.vlm_v2_blind_workbook_bytes = blind_scores_workbook_bytes(combined)
                    progress.progress(1.0, text="批量盲测评分完成")
                except Exception as exc:
                    st.error(f"批量盲测失败：{exc}")
            if not st.session_state.vlm_v2_blind_summary.empty:
                st.dataframe(st.session_state.vlm_v2_blind_summary, width="stretch", hide_index=True)
                with st.expander("查看逐项评分明细"):
                    st.dataframe(dataframe_for_display(st.session_state.vlm_v2_blind_details), width="stretch", hide_index=True)
                st.download_button(
                    "下载批量盲测评分结果",
                    st.session_state.vlm_v2_blind_workbook_bytes,
                    "VLM_v2批量盲测评分.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    width="stretch",
                    key="vlm_v2_blind_result_download",
                )

        if not st.session_state.page_matches or not st.session_state.pages_confirmed:
            st.info("请先在 STEP1 上传报告、完成页码定位并人工确认物理页码。")
        else:
            configured_model = st.session_state.get("llm_model", "") or "未设置"
            st.caption(
                f"使用登录页中的AI设置，当前模型：{configured_model}。"
                "VLM v2统一处理扫描版和带文字层PDF；单次请求最多4页、90秒，"
                "接口最多尝试2次；单表超时不影响其他结果，仅重试超时表和未通过校验的指标。"
            )

            if st.button("开始智能提取", type="primary", key="extract_tables"):
                if not all(
                    str(st.session_state.get(key, "")).strip()
                    for key in ("llm_base_url", "llm_model", "llm_api_key")
                ):
                    st.error(
                        "当前登录会话的 AI 配置不完整。请退出登录，在登录页填写 "
                        "Base URL、模型名称和 API Key 后重新进入。"
                    )
                else:
                    try:
                        st.session_state.vlm_v2_extraction_run = None
                        st.session_state.vlm_v2_workbook_bytes = b""
                        st.session_state.ai_workbook_bytes = b""
                        st.session_state.raw_tables = []
                        extraction_started = perf_counter()
                        with st.status("正在从已确认页面提取标准指标...", expanded=True) as extraction_status:
                            progress_bar = st.progress(
                                0,
                                text="正在读取目标指标定义并识别本季度口径...",
                            )

                            def update_extraction_progress(event):
                                phase = str(event.get("phase", ""))
                                elapsed = float(event.get("elapsed_seconds", 0.0) or 0.0)
                                elapsed_text = f"{elapsed:.1f}秒"
                                if phase == "rendering_started":
                                    page_count = int(event.get("page_count", 0) or 0)
                                    progress_bar.progress(
                                        2,
                                        text=f"正在统一渲染并缓存{page_count}个物理页 · 已用时{elapsed_text}",
                                    )
                                elif phase == "rendering_completed":
                                    page_count = int(event.get("page_count", 0) or 0)
                                    progress_bar.progress(
                                        8,
                                        text=f"页面缓存完成，共{page_count}页 · 已用时{elapsed_text}",
                                    )
                                elif phase == "initial_started":
                                    total = int(event.get("total_tables", 0) or 0)
                                    progress_bar.progress(
                                        10,
                                        text=f"开始首轮提取，共{total}张目标表 · 已用时{elapsed_text}",
                                    )
                                elif phase == "initial_completed":
                                    completed = int(event.get("completed_tables", 0) or 0)
                                    total = max(1, int(event.get("total_tables", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    percent = min(70, 10 + round(60 * completed / total))
                                    progress_bar.progress(
                                        percent,
                                        text=f"首轮已完成 {completed}/{total}：{table_name} · 已用时{elapsed_text}",
                                    )
                                    extraction_status.write(
                                        f"首轮完成 {completed}/{total}：{table_name}（{elapsed_text}）"
                                    )
                                elif phase == "initial_failed":
                                    completed = int(event.get("completed_tables", 0) or 0)
                                    total = max(1, int(event.get("total_tables", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    status = str(event.get("status", "失败"))
                                    percent = min(70, 10 + round(60 * completed / total))
                                    progress_bar.progress(
                                        percent,
                                        text=(
                                            f"首轮已处理 {completed}/{total}：{table_name}{status}，"
                                            f"其他表继续 · 已用时{elapsed_text}"
                                        ),
                                    )
                                    extraction_status.write(
                                        f"首轮{status} {completed}/{total}：{table_name}；"
                                        f"已保留其他成功结果（{elapsed_text}）"
                                    )
                                elif phase == "timeout_retry_started":
                                    names = "、".join(event.get("retry_table_names", []))
                                    progress_bar.progress(
                                        70,
                                        text=f"仅重试首轮失败请求批次：{names} · 已用时{elapsed_text}",
                                    )
                                    extraction_status.write(
                                        f"开始重试失败请求批次：{names}（{elapsed_text}）"
                                    )
                                elif phase == "timeout_retry_completed":
                                    completed = int(event.get("completed_retries", 0) or 0)
                                    total = max(1, int(event.get("retry_total", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    progress_bar.progress(
                                        71,
                                        text=(
                                            f"失败请求重试成功 {completed}/{total}：{table_name}"
                                            f" · 已用时{elapsed_text}"
                                        ),
                                    )
                                    extraction_status.write(
                                        f"失败请求重试成功 {completed}/{total}：{table_name}（{elapsed_text}）"
                                    )
                                elif phase == "timeout_retry_failed":
                                    completed = int(event.get("completed_retries", 0) or 0)
                                    total = max(1, int(event.get("retry_total", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    status = str(event.get("status", "失败"))
                                    progress_bar.progress(
                                        71,
                                        text=(
                                            f"失败请求重试{status} {completed}/{total}：{table_name}；"
                                            f"将保留其他结果 · 已用时{elapsed_text}"
                                        ),
                                    )
                                    extraction_status.write(
                                        f"失败请求重试{status}：{table_name}；其他成功结果不受影响"
                                        f"（{elapsed_text}）"
                                    )
                                elif phase == "validation_completed":
                                    retry_total = int(event.get("retry_total", 0) or 0)
                                    text = (
                                        f"业务校验完成，需要定向重试{retry_total}张表"
                                        if retry_total else "业务校验完成，无需重试"
                                    )
                                    progress_bar.progress(72, text=f"{text} · 已用时{elapsed_text}")
                                    extraction_status.write(f"{text}（{elapsed_text}）")
                                elif phase == "retry_started":
                                    names = "、".join(event.get("retry_table_names", []))
                                    progress_bar.progress(
                                        74,
                                        text=f"正在受控并行重试：{names} · 已用时{elapsed_text}",
                                    )
                                elif phase == "retry_completed":
                                    completed = int(event.get("completed_retries", 0) or 0)
                                    total = max(1, int(event.get("retry_total", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    percent = min(96, 74 + round(22 * completed / total))
                                    progress_bar.progress(
                                        percent,
                                        text=f"重试已完成 {completed}/{total}：{table_name} · 已用时{elapsed_text}",
                                    )
                                    extraction_status.write(
                                        f"定向重试完成 {completed}/{total}：{table_name}（{elapsed_text}）"
                                    )
                                elif phase == "retry_failed":
                                    completed = int(event.get("completed_retries", 0) or 0)
                                    total = max(1, int(event.get("retry_total", 0) or 0))
                                    table_name = str(event.get("table_name", "目标表"))
                                    status = str(event.get("status", "失败"))
                                    percent = min(96, 74 + round(22 * completed / total))
                                    progress_bar.progress(
                                        percent,
                                        text=(
                                            f"定向重试{status} {completed}/{total}：{table_name}；"
                                            f"保留首轮结果 · 已用时{elapsed_text}"
                                        ),
                                    )
                                    extraction_status.write(
                                        f"定向重试{status}：{table_name}；已保留首轮结果"
                                        f"（{elapsed_text}）"
                                    )
                                elif phase == "completed":
                                    progress_bar.progress(
                                        100,
                                        text=f"提取与业务校验全部完成 · 总用时{elapsed_text}",
                                    )

                            vlm_v2_taxonomy = extraction_taxonomy(active_profile, str(taxonomy_path))
                            vlm_table_configs = [
                                dict(item.table_config)
                                for item in st.session_state.page_matches
                            ]
                            extraction_run = extract_metrics_vlm_v2(
                                st.session_state.pdf_bytes,
                                st.session_state.page_matches,
                                vlm_table_configs,
                                vlm_v2_taxonomy,
                                api_key=st.session_state.llm_api_key,
                                base_url=st.session_state.llm_base_url,
                                model=st.session_state.llm_model,
                                timeout=90,
                                request_max_attempts=2,
                                auto_retry=True,
                                max_workers=3,
                                retry_max_workers=2,
                                max_pages_per_request=4,
                                progress_callback=update_extraction_progress,
                            )
                            extraction_elapsed = perf_counter() - extraction_started
                            step3_gate = evaluate_vlm_v2_step3_gate(extraction_run)
                            st.session_state.vlm_v2_extraction_run = extraction_run
                            st.session_state.vlm_v2_extraction_elapsed = extraction_elapsed
                            st.session_state.vlm_v2_workbook_bytes = vlm_v2_workbook_bytes(
                                st.session_state.vlm_v2_locator_run,
                                extraction_run,
                                report_metadata=st.session_state.metadata,
                                source_filename=st.session_state.pdf_name,
                            )
                            st.session_state.ai_workbook_bytes = st.session_state.vlm_v2_workbook_bytes
                            st.session_state.ai_extraction_logs = []
                            st.session_state.raw_tables = (
                                vlm_v2_to_extracted_tables(extraction_run)
                                if step3_gate.passed else []
                            )
                            st.session_state.table_candidates = {}
                            st.session_state.selected_table_candidates = {}
                            st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
                            st.session_state.validation_results = pd.DataFrame()
                            st.session_state.normalization_diagnostics = pd.DataFrame()
                            if step3_gate.passed:
                                extraction_status.update(
                                    label=(
                                        f"智能提取完成：STEP3前置门槛已通过，"
                                        f"用时{extraction_elapsed:.1f}秒"
                                    ),
                                    state="complete",
                                    expanded=False,
                                )
                            else:
                                extraction_status.update(
                                    label=(
                                        f"指标提取完成，但仍有校验项未通过，"
                                        f"用时{extraction_elapsed:.1f}秒"
                                    ),
                                    state="error",
                                    expanded=True,
                                )
                    except Exception as exc:
                        st.error(f"智能提取中止：{exc}")

        extraction_run = st.session_state.vlm_v2_extraction_run
        if extraction_run is not None:
            refreshed_run = refresh_vlm_v2_cross_table_checks(extraction_run)
            if refreshed_run is not extraction_run:
                extraction_run = refreshed_run
                st.session_state.vlm_v2_extraction_run = extraction_run
                st.session_state.vlm_v2_workbook_bytes = vlm_v2_workbook_bytes(
                    st.session_state.vlm_v2_locator_run, extraction_run,
                    report_metadata=st.session_state.metadata,
                    source_filename=st.session_state.pdf_name,
                )
                st.session_state.ai_workbook_bytes = st.session_state.vlm_v2_workbook_bytes
                if evaluate_vlm_v2_step3_gate(extraction_run).passed:
                    st.session_state.raw_tables = vlm_v2_to_extracted_tables(extraction_run)
            step3_gate = evaluate_vlm_v2_step3_gate(extraction_run)
            found_count = int(
                extraction_run.records["状态"].isin({"found", "disclosed_zero"}).sum()
            ) if not extraction_run.records.empty else 0
            failed_count = int(
                (extraction_run.validations["状态"] == "失败").sum()
            ) if not extraction_run.validations.empty else 0

            st.markdown("### 📊 提取结果预览")
            if st.session_state.ai_workbook_bytes:
                original_pdf_stem = re.sub(
                    r'[\\/:*?"<>|]+',
                    "_",
                    Path(st.session_state.pdf_name or "").stem,
                ).strip(" ._")
                step2_download_name = "偿付能力报告_STEP2结构化提取.xlsx"
                if original_pdf_stem:
                    step2_download_name = (
                        f"偿付能力报告_STEP2结构化提取_{original_pdf_stem[:80]}.xlsx"
                    )
                st.download_button(
                    "一键下载结构化提取表（Excel）",
                    st.session_state.ai_workbook_bytes,
                    step2_download_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    icon=":material/download:",
                    width="stretch",
                    key="download_step2_vlm_result",
                )

            records = extraction_run.records.copy()
            if not records.empty:
                preview_tabs = []
                preview_groups = []
                for table_id, group in records.groupby("目标表ID", sort=False):
                    table_name = str(group.iloc[0]["目标表名称"] or table_id)
                    preview_tabs.append(table_name)
                    preview_groups.append(group)
                for table_tab, group in zip(st.tabs(preview_tabs), preview_groups):
                    with table_tab:
                        preview = group[[
                            "指标编码", "指标名称", "状态", "原始值", "单位",
                            "标准数值", "标准单位", "期间口径", "物理页码",
                            "原始标签", "证据原文",
                        ]].copy()
                        preview["状态"] = preview["状态"].map({
                            "found": "已披露",
                            "disclosed_zero": "0",
                            "disclosed_na": "不适用",
                            "not_disclosed": "未披露",
                        }).fillna(preview["状态"])
                        st.dataframe(
                            dataframe_for_display(preview),
                            width="stretch",
                            hide_index=True,
                            height=min(720, 140 + len(preview) * 36),
                            column_config={
                                "标准数值": st.column_config.NumberColumn(format="%.6g"),
                                "物理页码": st.column_config.NumberColumn(format="%d"),
                            },
                        )

            with st.expander("查看提取质量与STEP3前置门槛", expanded=not step3_gate.passed):
                result_metrics = st.columns(4)
                result_metrics[0].metric("可用指标", found_count)
                result_metrics[1].metric("校验失败项", failed_count)
                result_metrics[2].metric(
                    "模型调用",
                    f"{extraction_run.model_calls}（重试{extraction_run.retry_calls}）",
                )
                result_metrics[3].metric(
                    "提取耗时",
                    f"{st.session_state.vlm_v2_extraction_elapsed:.1f}秒",
                )
                if step3_gate.passed:
                    st.success(step3_gate.summary)
                else:
                    st.error(step3_gate.summary)
                st.dataframe(step3_gate.checks, width="stretch", hide_index=True)
                validation_view = extraction_run.validations.copy()
                if not validation_view.empty:
                    validation_view["_排序"] = validation_view["状态"].map(
                        {"失败": 0, "通过": 1}
                    ).fillna(2)
                    validation_view = validation_view.sort_values("_排序").drop(columns="_排序")
                    st.dataframe(dataframe_for_display(validation_view), width="stretch", hide_index=True)
                if extraction_run.logs:
                    st.caption("　".join(extraction_run.logs))

            gold_case = find_gold_case(
                st.session_state.pdf_bytes,
                read_gold_manifest(),
            )
            if gold_case:
                with st.expander("真实PDF金标准 - 指标提取结果"):
                    st.dataframe(
                        vlm_v2_gold_evaluation(
                            gold_case,
                            st.session_state.vlm_v2_locator_run,
                            extraction_run,
                        ),
                        width="stretch",
                        hide_index=True,
                    )

    with tabs[3]:
        st.subheader(f"{active_profile.profile_name}目标表标准填报")
        with st.container(border=True):
            st.markdown("#### 功能说明")
            st.write(
                "按目标指标清单将 STEP2 提取结果填入标准窄表，并自动完成单位换算与派生指标计算。"
                "输出字段和可用指标范围与 STEP5 的精确映射口径完全一致。"
            )

        step3_source_mode = st.segmented_control(
            "STEP3 输入来源",
            ["使用 STEP2 提取结果", "上传本地提取表格"],
            default="使用 STEP2 提取结果",
            key="step3_source_mode",
            help=(
                "可沿用本次VLM v2正式提取结果，或上传本地STEP2结构化提取Excel。"
            ),
        )
        uploaded_source_file = None
        uploaded_source_pdf = None
        if step3_source_mode == "上传本地提取表格":
            uploaded_source_file = st.file_uploader(
                "上传 STEP2 标准化提取 Excel",
                type=["xlsx"],
                key="step3_source_upload",
                help=(
                    "需使用 STEP2 下载的“STEP2标准化提取.xlsx”，"
                    "或与其工作表结构一致的提取表格。"
                ),
            )
            uploaded_source_pdf = st.file_uploader(
                "原始报告 PDF（用于核对保单未来盈余披露，可选）",
                type=["pdf"],
                key="step3_source_pdf_upload",
                help="上传与 STEP2 表格对应的原始 PDF，可纠正旧提取结果中误判为零的未列示子项。",
            )
        uploaded_source_bytes = (
            uploaded_source_file.getvalue() if uploaded_source_file is not None else b""
        )
        vlm_step3_gate = evaluate_vlm_v2_step3_gate(
            st.session_state.vlm_v2_extraction_run
        ) if step3_source_mode == "使用 STEP2 提取结果" else None
        source_read_error = ""
        if step3_source_mode == "使用 STEP2 提取结果":
            source_tables = st.session_state.raw_tables
        else:
            source_tables = []
            if uploaded_source_file is not None:
                try:
                    source_tables = read_extracted_tables_workbook(
                        uploaded_source_bytes, active_profile.tables,
                    )
                    if not source_tables:
                        source_read_error = (
                            "文件已上传，但未找到可填报的指标数据。请上传 STEP2 下载的完整结构化提取 Excel；"
                            "支持 VLM v2 指标结果和旧版逐表工作簿。"
                        )
                except Exception as exc:
                    source_read_error = f"上传文件读取失败：{exc}"
        source_file_name = (
            uploaded_source_file.name
            if step3_source_mode == "上传本地提取表格" and uploaded_source_file is not None
            else st.session_state.pdf_name
        ) or "本地提取"
        source_basename = Path(source_file_name).stem
        current_pdf_name = st.session_state.pdf_name or ""
        if step3_source_mode == "使用 STEP2 提取结果":
            source_pdf_bytes = st.session_state.pdf_bytes
        elif uploaded_source_pdf is not None:
            source_pdf_bytes = uploaded_source_pdf.getvalue()
        elif current_pdf_name and Path(current_pdf_name).stem in source_basename:
            source_pdf_bytes = st.session_state.pdf_bytes
        else:
            source_pdf_bytes = b""

        include_derived = active_profile.profile_id == "LIFE_SOLVENCY"
        taxonomy = (
            active_profile.taxonomy_frame()
            if active_profile.field_dictionary
            else read_taxonomy(str(taxonomy_path))
        )
        if include_derived:
            taxonomy = extend_taxonomy(taxonomy)
        system_target_catalog = step3_metric_catalog(
            taxonomy,
            include_derived=include_derived,
            use_checklists=active_profile.profile_id == 'LIFE_SOLVENCY',
        )

        if vlm_step3_gate is not None:
            with st.container(border=True):
                st.markdown("#### STEP2 前置质量门槛")
                if vlm_step3_gate.passed:
                    st.success(vlm_step3_gate.summary)
                else:
                    st.error(vlm_step3_gate.summary)
                st.dataframe(vlm_step3_gate.checks, width="stretch", hide_index=True)

        use_default_target = st.toggle(
            "使用系统默认目标表",
            value=True,
            key="step3_use_default_target",
            help="默认目标表按填报清单列出披露指标，并保留系统派生指标；无对应披露的指标也会保留在标准窄表中。",
        )
        target_upload = None
        if use_default_target:
            with st.container(horizontal=True, vertical_alignment="center"):
                st.caption(
                    f"当前默认目标表包含 {len(system_target_catalog):,} 个指标，"
                    "可下载后将不需要的指标设为“不启用”。"
                )
                st.download_button(
                    "下载系统目标表",
                    cached_step3_target_template(
                        system_target_catalog,
                        active_profile.profile_name,
                    ),
                    f"{active_profile.profile_id}_STEP3目标表.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    icon=":material/download:",
                    key=f"download_step3_target_{active_profile.profile_id}",
                )
        else:
            target_upload = st.file_uploader(
                "上传自定义目标表",
                type=["xlsx"],
                key="step3_target_upload",
                help="建议使用系统目标表修改“启用”列；指标编码不得超出 STEP3 正式指标字典。",
            )

        company_type = active_profile.company_types[0]
        peer_group_map, default_peer_group = read_peer_groups(
            PEER_GROUP_CONFIG,
            PEER_GROUP_CONFIG.stat().st_mtime_ns,
        )

        step3_company_name = ""
        step3_report_period = ""
        step3_peer_group_override = ""
        step3_company_type = company_type
        step3_detected_metadata = None
        company_items = tuple(
            (str(row.get("公司", "")), str(row.get("公司类别", "")))
            for row in companies.to_dict("records")
        )
        if step3_source_mode == "上传本地提取表格":
            source_key = (
                hashlib.sha256(uploaded_source_bytes).hexdigest()[:12]
                if uploaded_source_bytes
                else "empty"
            )
            if uploaded_source_bytes and not source_read_error:
                step3_detected_metadata = read_step3_upload_metadata(
                    uploaded_source_bytes,
                    source_file_name,
                    company_items,
                    tuple(sorted(peer_group_map.items())),
                    default_peer_group,
                    tuple(
                        (key, str(st.session_state.metadata.get(key, "") or ""))
                        for key in ("公司", "报告年度", "报告季度", "报告期")
                    ),
                    st.session_state.pdf_name,
                )
        else:
            source_key = (
                hashlib.sha256(source_pdf_bytes).hexdigest()[:12]
                if source_pdf_bytes
                else hashlib.sha256(source_file_name.encode("utf-8")).hexdigest()[:12]
            )
            step3_detected_metadata = infer_step3_session_metadata(
                st.session_state.metadata,
                source_file_name,
                tuple(
                    {"company_name": name, "company_type": item_type}
                    for name, item_type in company_items
                ),
                peer_group_map=peer_group_map,
                default_peer_group=default_peer_group,
            )
        if step3_detected_metadata is not None:
            step3_company_type = step3_detected_metadata.company_type or company_type

        with st.container(border=True):
            st.markdown("#### 报告元信息自动识别")
            if step3_source_mode == "上传本地提取表格":
                st.caption(
                    "系统优先读取工作簿内嵌的报告元信息；若其报告期与原PDF及上传文件名"
                    "共同指向的季度冲突，则采用一致的来源期间。也兼容表内季度或期末日期。"
                    "识别结果仍可人工修改。"
                )
            else:
                st.caption("系统优先采用PDF识别结果；缺失时根据PDF文件名补全。请核对后填报。")
            if step3_detected_metadata is not None:
                for warning in step3_detected_metadata.warnings:
                    st.warning(warning)
            else:
                st.info("上传 STEP2 标准化提取 Excel 后将自动识别。")
            mode_key = "upload" if step3_source_mode == "上传本地提取表格" else "step2"
            company_widget_key = f"step3_company_name_{mode_key}_{source_key}"
            period_widget_key = f"step3_report_period_{mode_key}_{source_key}"
            peer_group_widget_key = f"step3_peer_group_{mode_key}_{source_key}"
            metadata_signature_key = f"step3_metadata_signature_v3_{mode_key}_{source_key}"
            metadata_values_key = f"step3_metadata_values_{mode_key}_{source_key}"
            detected_company = (
                step3_detected_metadata.company if step3_detected_metadata else ""
            )
            detected_period = (
                step3_detected_metadata.report_period if step3_detected_metadata else ""
            )
            detected_peer_group = (
                step3_detected_metadata.peer_group if step3_detected_metadata else ""
            )
            detected_signature = (detected_company, detected_period, detected_peer_group)
            if (
                st.session_state.get(metadata_signature_key) != detected_signature
                or metadata_values_key not in st.session_state
            ):
                st.session_state[metadata_values_key] = {
                    company_widget_key: detected_company,
                    period_widget_key: detected_period,
                    peer_group_widget_key: detected_peer_group or "自动匹配（按公司名称）",
                }
                for widget_key, value in st.session_state[metadata_values_key].items():
                    st.session_state[widget_key] = value
                st.session_state[metadata_signature_key] = detected_signature
            # Widget keys disappear when switching input modes; retain reviewed values.
            for widget_key, value in st.session_state[metadata_values_key].items():
                st.session_state.setdefault(widget_key, value)
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                step3_company_name = st.text_input(
                    "公司名称",
                    key=company_widget_key,
                    help="用于反查公司标准名称、统一编码与同业分类。",
                ).strip()
            with meta_col2:
                step3_report_period = st.text_input(
                    "报告期",
                    key=period_widget_key,
                    placeholder="如 2026Q2",
                    help="报告期，如 2026Q2、2026年2季度。",
                ).strip()
            with meta_col3:
                peer_group_choices = sorted(
                    {group for group in peer_group_map.values() if group}
                    | ({default_peer_group} if default_peer_group else set())
                )
                peer_group_options = ["自动匹配（按公司名称）", *peer_group_choices]
                if st.session_state.get(peer_group_widget_key) not in peer_group_options:
                    st.session_state[peer_group_widget_key] = "自动匹配（按公司名称）"
                step3_peer_group = st.selectbox(
                    "同业分类",
                    peer_group_options,
                    key=peer_group_widget_key,
                    help="默认按公司名称自动匹配；匹配不到时可手动指定。",
                )
                step3_peer_group_override = (
                    "" if step3_peer_group.startswith("自动匹配") else step3_peer_group
                )
            st.session_state[metadata_values_key] = {
                company_widget_key: step3_company_name,
                period_widget_key: step3_report_period,
                peer_group_widget_key: step3_peer_group,
            }

        standardize_submitted = st.button(
            "启动目标表标准填报",
            type="primary",
            icon=":material/auto_fix_high:",
            disabled=(
                not source_tables
                or not step3_company_name
                or (not use_default_target and target_upload is None)
            ),
            key="step3_standardize_target",
        )

        if not source_tables:
            if source_read_error:
                st.error(source_read_error)
            elif step3_source_mode == "上传本地提取表格":
                st.info("请上传 STEP2 标准化提取 Excel，或切回“使用 STEP2 提取结果”。")
            else:
                st.info("请先在STEP2完成VLM v2提取并通过全部前置门槛。")
        elif not use_default_target and target_upload is None:
            st.info("请上传自定义目标表，或切回系统默认目标表。")
        elif not step3_company_name:
            st.warning("请先在上方“报告元信息自动识别”中填写公司名称。")

        if standardize_submitted:
            st.session_state.standard_data = pd.DataFrame(columns=STANDARD_COLUMNS)
            st.session_state.validation_results = pd.DataFrame()
            st.session_state.standard_workbook_bytes = b""
            st.session_state.normalization_diagnostics = pd.DataFrame()
            st.session_state.standard_target_summary = pd.DataFrame()
            st.session_state.standardization_warnings = []
            with st.status("正在进行目标表标准填报…", expanded=True) as status_box:
                try:
                    status_box.write("正在读取目标指标清单并校验 STEP3 正式指标范围…")
                    if use_default_target:
                        target_catalog = system_target_catalog
                        template_warnings: tuple[str, ...] = ()
                        template_label = "系统默认目标表"
                    else:
                        target_catalog, template_warnings = read_target_template(
                            target_upload.getvalue(),
                            target_upload.name,
                            system_target_catalog,
                        )
                        template_label = target_upload.name

                    status_box.write("正在匹配披露指标、识别期间口径并统一计量单位…")
                    effective_period = st.session_state.metadata.get("报告期", "")
                    metadata = dict(st.session_state.metadata)
                    if step3_source_mode == "上传本地提取表格":
                        metadata = (
                            step3_detected_metadata.to_metadata()
                            if step3_detected_metadata is not None
                            else {}
                        )
                        effective_period = str(metadata.get("报告期", "") or "")
                    elif step3_detected_metadata is not None:
                        metadata.update(step3_detected_metadata.to_metadata())
                        effective_period = str(metadata.get("报告期", "") or "")
                    # A cleared/stale widget must not erase a recognized company.
                    metadata["公司"] = (
                        step3_company_name or str(metadata.get("公司", "") or "").strip()
                    )
                    if step3_report_period:
                        period_year, period_quarter, normalized_period = normalize_report_period(
                            step3_report_period
                        )
                        effective_period = normalized_period or step3_report_period
                        metadata["报告期"] = effective_period
                        metadata["报告年度"] = period_year
                        metadata["报告季度"] = period_quarter
                    if not str(metadata.get("公司", "") or "").strip():
                        raise ValueError("公司名称尚未识别，请在报告元信息中补充公司名称后再填报。")
                    metadata["来源文件"] = source_file_name
                    metadata["导入批次"] = f"{source_basename}:{effective_period}"
                    result = standardize_to_target(
                        source_tables,
                        taxonomy,
                        metadata,
                        step3_company_type,
                        target_catalog,
                        report_profile_id=active_profile.profile_id,
                        allowed_company_types=active_profile.company_types,
                        peer_group=step3_peer_group_override,
                        peer_group_map=peer_group_map,
                        default_peer_group=default_peer_group,
                        include_derived=include_derived,
                        source_pdf_bytes=source_pdf_bytes,
                        template_warnings=template_warnings,
                    )

                    status_box.write("正在生成 STEP5 可直接读取的标准窄表工作簿…")
                    st.session_state.standard_data = result.data
                    st.session_state.normalization_diagnostics = result.diagnostics
                    st.session_state.standard_target_catalog = result.target_catalog
                    st.session_state.standard_target_summary = result.target_summary
                    st.session_state.standardization_warnings = list(result.warnings)
                    st.session_state.standard_template_label = template_label
                    st.session_state.standard_workbook_bytes = result_workbook_bytes(result)
                    status_box.update(
                        label="目标表标准填报完成",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    status_box.update(
                        label="目标表标准填报失败",
                        state="error",
                        expanded=True,
                    )
                    st.error(f"标准填报失败：{exc}")

        if not st.session_state.standard_target_summary.empty:
            summary = st.session_state.standard_target_summary
            filled_targets = int((summary["填报状态"] == "已填报").sum())
            missing_targets = int(summary['填报状态'].isin({'未披露', '无法计算', '未填报'}).sum())
            with st.container(horizontal=True):
                st.metric("目标指标", len(summary), border=True)
                st.metric("已填报指标", filled_targets, border=True)
                st.metric("未填报指标", missing_targets, border=True)
                st.metric("窄表记录", len(st.session_state.standard_data), border=True)
            st.caption(f"本次使用：{st.session_state.standard_template_label}")
            for warning in st.session_state.standardization_warnings:
                st.warning(warning)
            if missing_targets:
                with st.expander("查看未填报目标指标", expanded=True):
                    st.dataframe(
                        summary[summary['填报状态'].isin({'未披露', '无法计算', '未填报'})],
                        width="stretch",
                        hide_index=True,
                    )

        if not st.session_state.normalization_diagnostics.empty:
            with st.expander("查看未匹配或歧义诊断"):
                st.dataframe(
                    st.session_state.normalization_diagnostics,
                    width="stretch",
                    hide_index=True,
                )

        if not st.session_state.standard_data.empty:
            derived_count = int(
                st.session_state.standard_data["指标属性"].isin(["计算", "校验"]).sum()
            )
            converted_rows = int(
                st.session_state.standard_data["备注"]
                .astype(str)
                .str.contains("已换算为", regex=False)
                .sum()
            )
            pending_unit_rows = int(
                st.session_state.standard_data["备注"]
                .astype(str)
                .str.contains("未识别原始单位", regex=False)
                .sum()
            )
            with st.container(horizontal=True):
                st.metric("派生及校验记录", derived_count, border=True)
                st.metric("单位换算记录", converted_rows, border=True)
                st.metric("单位待核对记录", pending_unit_rows, border=True)
            st.markdown("#### 标准窄表预览")
            st.caption(
                "页面预览显示已计算数值；下载后的 Excel 会在系统计算指标的“数值”单元格中保留公式，"
                "并提供“计算依据”工作表用于追溯。未取得的披露指标在数值单元格显示“未披露”；原表横杠及数字0均按0处理。"
            )
            step3_narrow_display = narrow_table_view(st.session_state.standard_data)
            # The preview column contains numbers and disclosure text; keep the
            # mixed display Arrow-safe without changing numeric Excel cells.
            step3_narrow_display['数值'] = step3_narrow_display['数值'].map(
                lambda value: '' if pd.isna(value) else str(value))
            st.dataframe(dataframe_for_display(step3_narrow_display), width="stretch", hide_index=True)
            st.download_button(
                "下载已填报的标准窄表",
                (
                    st.session_state.standard_workbook_bytes
                    or standard_workbook_bytes(st.session_state.standard_data)
                ),
                f"{source_basename}_STEP3标准窄表.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )


    with tabs[4]:
        st.subheader(f"{active_profile.profile_name}勾稽检查")
        st.caption(
            "数据规范检查覆盖 STEP3 标准窄表全部记录；六项关键指标完整性、公式勾稽和"
            "跨表一致性仅检查当前期末组（本季度末数、期末数、本季度（末）数）。"
            "检查过程只读，不会自动修改标准化结果；下载工作簿会接续保留 STEP3 公式，"
            "并附带检查公式、Excel 差异及容差复核。"
        )
        if st.session_state.standard_data.empty:
            st.info("请先完成 STEP3 标准化。")
        elif st.button(
            "执行勾稽检查",
            type="primary",
            key="validate_data",
            icon=":material/fact_check:",
        ):
            st.session_state.validation_results = validate_standard_data(
                st.session_state.standard_data,
                read_rules(str(rules_path)),
                taxonomy,
            )
        validation_results = st.session_state.validation_results
        required_result_columns = {"check_type", "rule_name", "status", "severity"}
        if not validation_results.empty and not required_result_columns.issubset(validation_results.columns):
            st.info("Step4 规则已升级，请重新点击“执行勾稽检查”生成新版结果。")
        elif not validation_results.empty:
            status_counts = validation_results["status"].astype(str).value_counts().to_dict()
            summary_columns = st.columns(6)
            summary_columns[0].metric("检查项", f"{len(validation_results):,}")
            summary_columns[1].metric("通过", f"{status_counts.get('通过', 0):,}")
            summary_columns[2].metric("未通过", f"{status_counts.get('未通过', 0):,}")
            summary_columns[3].metric("缺失", f"{status_counts.get('缺失', 0):,}")
            summary_columns[4].metric("需复核", f"{status_counts.get('需复核', 0):,}")
            summary_columns[5].metric(
                "不适用/错误",
                f"{status_counts.get('不适用', 0) + status_counts.get('执行错误', 0):,}",
            )

            blocking_count = sum(
                status_counts.get(status, 0)
                for status in ("未通过", "缺失", "执行错误")
            )
            if blocking_count:
                st.warning(
                    f"发现 {blocking_count:,} 项未通过、缺失或执行错误。"
                    "请根据处理建议返回 STEP2/STEP3 修正后重新检查。"
                )
            elif status_counts.get("需复核", 0):
                st.info("公式勾稽未发现阻断项，但仍有重复记录等事项需要人工复核。")
            else:
                st.success("第一阶段勾稽检查已通过，可以继续进入 STEP5 数据集成。")

            available_statuses = [
                status for status in ("未通过", "缺失", "需复核", "执行错误", "通过", "不适用")
                if status_counts.get(status, 0)
            ]
            issue_statuses = [
                status for status in ("未通过", "缺失", "需复核", "执行错误")
                if status in available_statuses
            ]
            filter_columns = st.columns(2)
            selected_statuses = filter_columns[0].multiselect(
                "状态筛选",
                available_statuses,
                default=issue_statuses or available_statuses,
                key="step4_status_filter",
            )
            available_types = list(dict.fromkeys(validation_results["check_type"].astype(str)))
            selected_types = filter_columns[1].multiselect(
                "检查类别",
                available_types,
                default=available_types,
                key="step4_type_filter",
            )
            filtered_results = validation_results[
                validation_results["status"].astype(str).isin(selected_statuses)
                & validation_results["check_type"].astype(str).isin(selected_types)
            ].copy()
            display_results = validation_display_frame(filtered_results)
            display_columns = {
                "check_type": "检查类别",
                "severity": "未通过时级别",
                "rule_name": "规则名称",
                "company": "公司",
                "report_period": "报告期",
                "period": "期间组",
                "actual": "披露值/实际数",
                "expected": "系统计算值/应有数",
                "difference": "差异",
                "tolerance": "容差",
                "status": "状态",
                "notes": "检查说明",
                "involved_metrics": "涉及指标编码",
                "source_pages": "来源页码",
                "suggestion": "处理建议",
            }
            st.dataframe(
                display_results[list(display_columns)].rename(columns=display_columns),
                width="stretch",
                hide_index=True,
                column_config={
                    "披露值/实际数": st.column_config.NumberColumn(format="%.4f"),
                    "系统计算值/应有数": st.column_config.NumberColumn(format="%.4f"),
                    "差异": st.column_config.NumberColumn(format="%.4f"),
                    "容差": st.column_config.NumberColumn(format="%.4f"),
                },
            )
            st.download_button(
                "下载勾稽检查工作簿",
                validation_results_to_xlsx(
                    validation_results,
                    (
                        st.session_state.standard_workbook_bytes
                        or standard_workbook_bytes(st.session_state.standard_data)
                    ),
                ),
                "偿付能力_STEP4勾稽检查结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                icon=":material/download:",
            )


    with tabs[5]:
        st.subheader("多公司、多期间数据集成")
        st.caption(
            "两种接入方式互斥；确认后将以本次预览结果作为后续 "
            f"STEP6-STEP8 的 {active_profile.profile_id} 集成数据。"
        )
        integration_mode = st.segmented_control(
            "数据接入方式",
            ["标准窄表", "外部宽表转窄表"],
            default="标准窄表",
            key="integration_mode",
            width="stretch",
        )

        if integration_mode == "标准窄表":
            uploads = st.file_uploader(
                "上传一个或多个标准窄表文件",
                type=["xlsx"],
                accept_multiple_files=True,
                key="integrated_standard_uploads",
            )
            if uploads and st.button(
                "读取并预览标准窄表", type="primary", key="preview_standard_data"
            ):
                integration_started = perf_counter()
                integration_progress = st.progress(0.0, text=f"准备读取 {len(uploads)} 个文件…")
                try:
                    frames = []
                    summaries = []
                    skipped_profiles: set[str] = set()
                    for file_index, upload in enumerate(uploads, start=1):
                        integration_progress.progress(
                            (file_index - 1) / (len(uploads) + 1),
                            text=f"正在读取 {file_index}/{len(uploads)}：{upload.name} · 已耗时 {perf_counter() - integration_started:.1f} 秒",
                        )
                        standardized = cached_read_standard_workbook(
                            upload.getvalue(), upload.name
                        )
                        standardized["报告类型"] = (
                            standardized["报告类型"]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                            .replace("", active_profile.profile_id)
                        )
                        mismatched = set(
                            standardized.loc[
                                standardized["报告类型"] != active_profile.profile_id,
                                "报告类型",
                            ].astype(str)
                        )
                        skipped_profiles.update(mismatched)
                        standardized = standardized[
                            standardized["报告类型"] == active_profile.profile_id
                        ]
                        if not standardized.empty:
                            frames.append(standardized)
                        summaries.append({
                            "来源文件": upload.name,
                            "数据类型": "标准窄表",
                            "记录数": len(standardized),
                        })
                        integration_progress.progress(
                            file_index / (len(uploads) + 1),
                            text=f"已读取 {file_index}/{len(uploads)} 个文件 · 已耗时 {perf_counter() - integration_started:.1f} 秒",
                        )
                    integration_preview = (
                        pd.concat(frames, ignore_index=True)
                        if frames
                        else pd.DataFrame(columns=STANDARD_COLUMNS)
                    )
                    integration_progress.progress(
                        len(uploads) / (len(uploads) + 1), text="文件读取完成，正在合并并补充可计算指标…"
                    )
                    st.session_state.integration_preview = (
                        cached_complete_step5_metrics(integration_preview)
                        if active_profile.profile_id == "LIFE_SOLVENCY"
                        else integration_preview
                    )
                    st.session_state.integration_sheet_summary = pd.DataFrame(summaries)
                    st.session_state.integration_mapping_summary = pd.DataFrame()
                    st.session_state.integration_logic_checks = pd.DataFrame()
                    st.session_state.integration_warnings = (
                        ["已跳过其他报告类型的数据：" + "、".join(sorted(skipped_profiles))]
                        if skipped_profiles
                        else []
                    )
                    st.session_state.integration_preview_mode = "标准窄表"
                    integration_progress.progress(
                        1.0, text=f"集成预览完成：{len(uploads)} 个文件，{len(st.session_state.integration_preview):,} 条记录 · 共 {perf_counter() - integration_started:.1f} 秒"
                    )
                except Exception as exc:
                    integration_progress.empty()
                    st.error(f"标准窄表读取失败：{exc}")
        else:
            if active_profile.profile_id != "LIFE_SOLVENCY":
                st.info("当前外部 CROSS 宽表映射仅适用于寿险偿付能力 Profile。")
                uploads = []
            else:
                uploads = st.file_uploader(
                    "上传一个或多个外部宽表文件",
                    type=["xlsx"],
                    accept_multiple_files=True,
                    key="integrated_external_uploads",
                )
                st.caption(
                    "转换时将同步生成“行业合计”窄表记录：金额指标按有效公司求和，"
                    "比率和倍数按行业合计分子、分母重新计算；行业风险构成采用底稿中的展示名称。"
                )
            if uploads and st.button(
                "精确映射并转换预览", type="primary", key="preview_external_data"
            ):
                try:
                    base_taxonomy = (
                        active_profile.taxonomy_frame()
                        if active_profile.field_dictionary
                        else read_taxonomy(str(taxonomy_path))
                    )
                    taxonomy = extend_taxonomy(base_taxonomy)
                    company_type_map = dict(zip(companies["公司"], companies["公司类别"]))
                    company_type_items = tuple(sorted(company_type_map.items()))
                    with st.spinner(f"正在转换 {len(uploads)} 个宽表文件…"):
                        converted = [
                            cached_convert_external_workbook(
                                upload.getvalue(),
                                upload.name,
                                taxonomy,
                                company_type_items,
                                active_profile.profile_id,
                            )
                            for upload in uploads
                        ]
                    integration_preview = pd.concat(
                        [item.data for item in converted], ignore_index=True
                    )
                    st.session_state.integration_preview = cached_complete_step5_metrics(
                        integration_preview
                    )
                    st.session_state.integration_sheet_summary = pd.concat(
                        [item.sheet_summary for item in converted], ignore_index=True
                    )
                    st.session_state.integration_mapping_summary = pd.concat(
                        [item.mapping_summary for item in converted], ignore_index=True
                    ).drop_duplicates(subset=["来源字段", "指标编码"], keep="last")
                    st.session_state.integration_logic_checks = pd.concat(
                        [item.logic_checks for item in converted], ignore_index=True
                    )
                    st.session_state.integration_warnings = [
                        warning for item in converted for warning in item.warnings
                    ]
                    st.session_state.integration_preview_mode = "外部宽表转窄表"
                except Exception as exc:
                    st.error(f"外部宽表转换失败：{exc}")

    preview = st.session_state.integration_preview
    step5_is_open = bool(tabs[5].open)
    identity_columns = {"原始公司名称", "标准公司名称", "公司统一编码"}
    if step5_is_open and not identity_columns.issubset(preview.columns):
        preview = upgrade_standard_frame(preview)
        st.session_state.integration_preview = preview
    if (
        step5_is_open
        and not preview.empty
        and st.session_state.integration_preview_mode == integration_mode
    ):
            st.markdown("#### 集成前预览")
            identity_changes = preview.loc[
                preview["原始公司名称"].astype(str).str.strip()
                != preview["标准公司名称"].astype(str).str.strip(),
                ["原始公司名称", "标准公司名称", "公司统一编码", "公司类型"],
            ].drop_duplicates()
            if not identity_changes.empty:
                st.info(
                    f"已识别 {len(identity_changes):,} 组历史公司名称，"
                    "后续跨期分析将按标准公司名称和统一编码归集。"
                )
                with st.expander("查看公司历史名称映射"):
                    st.dataframe(identity_changes, width="stretch", hide_index=True)
            if not st.session_state.integration_sheet_summary.empty:
                st.dataframe(
                    st.session_state.integration_sheet_summary,
                    width="stretch",
                    hide_index=True,
                )
            for warning in st.session_state.integration_warnings:
                st.warning(warning)
            if not st.session_state.integration_mapping_summary.empty:
                with st.expander("查看指标精确映射"):
                    st.dataframe(
                        st.session_state.integration_mapping_summary,
                        width="stretch",
                        hide_index=True,
                    )
            if not st.session_state.integration_logic_checks.empty:
                with st.expander("查看派生指标逻辑校验", expanded=True):
                    st.dataframe(
                        st.session_state.integration_logic_checks,
                        width="stretch",
                        hide_index=True,
                    )
            st.caption(
                f"转换后共 {len(preview):,} 条窄表记录。确认后将替换当前集成数据，"
                "不与其他接入方式并行合并。"
            )
            st.dataframe(
                dataframe_for_display(narrow_table_view(preview.head(1000))),
                width="stretch",
                hide_index=True,
            )
            if st.button(
                "确认采用该数据集",
                type="primary",
                key="confirm_integrated_data",
            ):
                st.session_state.integrated_data = preview.copy()
                st.success("已更新集成数据，STEP6-STEP8 将使用本次确认的数据集。")
    if step5_is_open and not st.session_state.integrated_data.empty:
        st.markdown("#### 当前已确认集成数据")
        display_limit = 1000
        integrated_data = st.session_state.integrated_data
        st.caption(
            f"当前共 {len(integrated_data):,} 条记录；页面仅展示前 "
            f"{min(display_limit, len(integrated_data)):,} 条，下载文件仍包含全部记录。"
        )
        integrated_narrow_display = narrow_table_view(integrated_data.head(display_limit))
        st.dataframe(
            dataframe_for_display(integrated_narrow_display.head(display_limit)),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "下载行业集成数据",
            # Generate the complete workbook only on download, not on every
            # preview/rerun. Capture this dataset, not mutable session state.
            lambda frame=integrated_data.copy(): standard_workbook_bytes(frame),
            "偿付能力行业集成数据.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click="ignore",
        )

raw_analysis_source = (
    st.session_state.integrated_data
    if not st.session_state.integrated_data.empty
    else st.session_state.standard_data
)
shared_analysis_source, _ = prepare_visualization_source(
    raw_analysis_source,
    active_profile.profile_id,
)


with tabs[6]:
    st.subheader("自定义偿付能力对标分析")
    source_mode = st.radio(
        "数据源选择",
        ["直接引用集成后的数据", "上传集成表 Excel"],
        horizontal=True,
        key="step6_data_source_choice",
    )

    source = pd.DataFrame(columns=STANDARD_COLUMNS)
    source_label = ""
    skipped_profiles: tuple[str, ...] = ()
    if source_mode == "直接引用集成后的数据":
        source, skipped_profiles = prepare_visualization_source(
            st.session_state.integrated_data,
            active_profile.profile_id,
        )
        source_label = "STEP5 已确认集成数据"
    else:
        step6_upload = st.file_uploader(
            "上传行业集成目标表",
            type=["xlsx"],
            key=f"step6_integrated_upload_{active_profile.profile_id}",
            help="优先读取“标准数据”工作表；至少应包含公司、指标编码、指标名称和数值字段。",
        )
        if step6_upload is not None:
            try:
                source, skipped_profiles = read_step6_analysis_workbook(
                    step6_upload.getvalue(),
                    step6_upload.name,
                    active_profile.profile_id,
                )
                source_label = step6_upload.name
                st.success(f"已读取 {len(source):,} 条可分析记录。")
            except Exception as exc:
                st.error(f"集成表读取失败：{exc}")

    if skipped_profiles:
        st.warning(
            "已跳过不属于当前报告 Profile 的记录："
            + "、".join(skipped_profiles)
        )
    st.session_state.step6_analysis_source = source.copy()

    navigation_source = report_scope_frame(
        visualization_metric_frame(source),
        COMPANY_REPORT,
    )
    available_metric_codes = navigation_metric_codes(navigation_source)
    industry_navigation_source = report_scope_frame(
        visualization_metric_frame(source),
        INDUSTRY_REPORT,
    )
    industry_available_metric_codes = navigation_metric_codes(
        industry_navigation_source
    )

    if analysis_tabs_active:
        with st.sidebar:
            render_report_navigation(
                title="公司报告导航",
                container_key="kpmg_company_nav_title",
                state_prefix="company_nav",
                available_codes=available_metric_codes,
                print_key=f"company_report_print_{active_profile.profile_id}",
            )
            if is_project_member:
                render_report_navigation(
                    title="行业分析导航",
                    container_key="kpmg_industry_nav_title",
                    state_prefix="industry_nav",
                    available_codes=industry_available_metric_codes,
                    print_key=f"industry_report_print_{active_profile.profile_id}",
                    industry=True,
                )
            render_major_financing_upload(active_profile.profile_id)

    if navigation_source.empty:
        if not source.empty:
            st.info("当前数据仅包含校验类记录；这些记录保留在 STEP5，但不会进入可视化指标。")
        elif source_mode == "直接引用集成后的数据":
            st.info("STEP5 尚无已确认的集成数据，请先完成集成，或切换为上传集成表 Excel。")
        elif source_label == "":
            st.info("请上传一份标准窄表格式的集成 Excel。")
    else:
        render_kpmg_palette()
        selected_level_one = st.session_state.get("company_nav_level_one", "")
        selected_level_two = st.session_state.get("company_nav_level_two", "全部")
        module_frame = navigation_source.copy()
        if selected_level_one and selected_level_one != PRINT_ALL_LABEL:
            selected_charts = resolve_chart_selection(
                selected_level_one,
                selected_level_two,
                st.session_state.get("company_nav_chart", ""),
            )
            selected_codes = list(dict.fromkeys(
                code
                for chart_name in selected_charts
                for code in metric_codes_for_chart(chart_name)
            ))
            module_frame = module_frame[
                module_frame["指标编码"].astype(str).str.strip().isin(selected_codes)
            ]
        if module_frame.empty and not navigation_source.empty:
            st.info("当前一级/二级模块下没有可视化指标，请在侧边栏选择其他模块。")
        if not module_frame.empty:
            with st.expander(
                "核心配置面板",
                expanded=True,
                icon=":material/tune:",
            ):
                settings_col, layout_col, unit_col = st.columns([1.5, 1, 1])
                metric_labels, metric_lookup = metric_options(module_frame)
                keep_valid_widget_state("step6_metric", metric_labels)
                with settings_col:
                    chart_type = st.selectbox(
                        "图表类型",
                        CHART_TYPES,
                        key="step6_chart_type",
                    )
                    selected_metric = st.selectbox(
                        "选择显示指标",
                        metric_labels,
                        key="step6_metric",
                        disabled=not metric_labels,
                        placeholder="暂无可用指标",
                    ) if metric_labels else ""

                    peer_group_options = nonblank_values(module_frame, "同业分类")
                    company_type_filter_options = [
                        NO_COMPANY_TYPE_FILTER,
                        *peer_group_options,
                    ]
                    keep_valid_widget_state(
                        "step6_company_type_quick_select",
                        company_type_filter_options,
                    )
                    selected_company_type = st.selectbox(
                        "按公司类型快速选择",
                        company_type_filter_options,
                        key="step6_company_type_quick_select",
                    )
                    all_company_options = companies_for_quick_selection(
                        module_frame,
                    )
                    selected_type_companies = companies_for_quick_selection(
                        module_frame,
                        selected_company_type,
                    )
                    previous_company_type = st.session_state.get(
                        "_step6_previous_company_type_quick_select"
                    )
                    if previous_company_type != selected_company_type:
                        if selected_company_type == NO_COMPANY_TYPE_FILTER:
                            current_companies = st.session_state.get("step6_companies", [])
                            retained_companies = [
                                company
                                for company in current_companies
                                if company in all_company_options
                            ]
                            st.session_state.step6_companies = (
                                retained_companies or all_company_options[:2]
                            )
                        else:
                            st.session_state.step6_companies = selected_type_companies
                        st.session_state._step6_previous_company_type_quick_select = (
                            selected_company_type
                        )
                    elif "step6_companies" not in st.session_state:
                        st.session_state.step6_companies = all_company_options[:2]
                    keep_valid_widget_state(
                        "step6_companies",
                        all_company_options,
                        multiple=True,
                    )
                    selected_comparison_companies = st.multiselect(
                        "选择对比公司",
                        all_company_options,
                        key="step6_companies",
                        placeholder="选择一家或多家公司",
                    )

                comparison_frame = (
                    module_frame[
                        module_frame["公司"].isin(selected_comparison_companies)
                    ].copy()
                    if selected_comparison_companies
                    else module_frame.iloc[0:0].copy()
                )
                with layout_col:
                    layout_mode = st.radio(
                        "布局视角",
                        ["以公司为横轴", "以报告期为横轴"],
                        key="step6_layout_mode",
                    )
                    decimals = st.number_input(
                        "小数位数",
                        min_value=0,
                        max_value=4,
                        value=2,
                        step=1,
                        key="step6_decimals",
                    )
                    show_labels = st.toggle(
                        "显示数据标签",
                        value=True,
                        key="step6_show_labels",
                    )

                selected_metric_code = metric_lookup.get(selected_metric, "")
                metric_preview = filter_analysis_frame(
                    module_frame,
                    metric_code=selected_metric_code,
                )
                preview_units = nonblank_values(metric_preview, "单位")
                unit_options = ["原始数值"]
                if any(unit in {"元", "万元", "亿元"} for unit in preview_units):
                    unit_options.extend(["亿元", "十亿元"])
                if any(unit == "倍" for unit in preview_units):
                    unit_options.append("百分比(%)")
                keep_valid_widget_state("step6_unit_mode", unit_options)
                with unit_col:
                    unit_mode = st.selectbox(
                        "数值单位换算",
                        unit_options,
                        key="step6_unit_mode",
                    )
                    y_axis_title = st.text_input(
                        "Y轴单位显示修改",
                        value="",
                        placeholder="留空则使用指标单位",
                        key="step6_y_axis_title",
                    )
                    transparent = st.toggle(
                        "开启透明背景模式",
                        value=False,
                        key="step6_transparent",
                    )
                    show_average = st.toggle(
                        "平均值线",
                        value=False,
                        key="step6_show_average",
                    )
                    average_color = (
                        st.color_picker(
                            "基准线颜色",
                            value="#ED2124",
                            key="step6_average_color",
                        )
                        if show_average
                        else "#ED2124"
                    )

                period_options = sort_report_periods(comparison_frame["报告期"].tolist())
                keep_valid_widget_state(
                    "step6_periods",
                    period_options,
                    multiple=True,
                )
                time_col, scope_col = st.columns([2, 1])
                with time_col:
                    selected_periods = st.multiselect(
                        "对比时间",
                        period_options,
                        default=(
                            default_periods(period_options)
                            if "step6_periods" not in st.session_state
                            else None
                        ),
                        key="step6_periods",
                        placeholder="选择报告期",
                    )

                period_frame = (
                    comparison_frame[
                        comparison_frame["报告期"].isin(selected_periods)
                    ].copy()
                    if selected_periods
                    else comparison_frame.iloc[0:0].copy()
                )
                scope_options = nonblank_values(period_frame, "期间口径")
                keep_valid_widget_state("step6_period_scope", scope_options)
                with scope_col:
                    period_scope = st.selectbox(
                        "期间口径",
                        scope_options,
                        key="step6_period_scope",
                        disabled=not scope_options,
                        placeholder="暂无期间口径",
                    ) if scope_options else ""
                scoped_frame = filter_analysis_frame(
                    period_frame,
                    period_scope=period_scope,
                )

            chart_frame = filter_analysis_frame(
                scoped_frame,
                metric_code=selected_metric_code,
            )
            if chart_frame.empty:
                st.info("请至少选择一个公司、一个报告期和一个可用指标。")
            else:
                chart_frame = chart_frame.copy()
                units = nonblank_values(chart_frame, "单位")
                chart_plot_frame = chart_frame.copy()
                if unit_mode in {"亿元", "十亿元"}:
                    target_yuan = 100_000_000 if unit_mode == "亿元" else 1_000_000_000
                    source_yuan = {"元": 1, "万元": 10_000, "亿元": 100_000_000}
                    chart_plot_frame["数值"] = chart_plot_frame.apply(
                        lambda row: (
                            pd.to_numeric(pd.Series([row["数值"]]), errors="coerce").iloc[0]
                            * source_yuan.get(str(row.get("单位", "")).strip(), target_yuan)
                            / target_yuan
                        ),
                        axis=1,
                    )
                    chart_plot_frame["单位"] = unit_mode
                elif unit_mode == "百分比(%)":
                    multiple_mask = chart_plot_frame["单位"].astype(str).str.strip().eq("倍")
                    chart_plot_frame.loc[multiple_mask, "数值"] = (
                        pd.to_numeric(
                            chart_plot_frame.loc[multiple_mask, "数值"],
                            errors="coerce",
                        )
                        * 100
                    )
                    chart_plot_frame.loc[multiple_mask, "单位"] = "%"

                duplicate_columns = ["公司", "报告期", "指标编码", "期间口径"]
                duplicate_count = int(chart_frame.duplicated(duplicate_columns, keep=False).sum())
                if duplicate_count:
                    st.warning(
                        f"当前筛选结果中有 {duplicate_count:,} 条重复粒度记录，"
                        "图表将全部保留展示，请检查集成数据。"
                    )
                if len(units) > 1:
                    st.warning("同一指标存在多个单位：" + "、".join(units))

                st.markdown("#### :material/palette: 自定义图例标签与颜色")
                legend_items = (
                    sort_report_periods(chart_plot_frame["报告期"].tolist())
                    if layout_mode == "以公司为横轴"
                    else nonblank_values(chart_plot_frame, "公司")
                )
                legend_label_map: dict[str, str] = {}
                legend_color_map: dict[str, str] = {}
                legend_columns = st.columns(min(4, max(1, len(legend_items))))
                for index, item in enumerate(legend_items):
                    with legend_columns[index % len(legend_columns)]:
                        st.caption(f"原始值：{item}")
                        legend_label_map[item] = st.text_input(
                            "显示名称",
                            value=item,
                            key=f"step6_legend_label_{layout_mode}_{index}",
                        )
                        legend_color_map[item] = st.color_picker(
                            "选择颜色",
                            value=KPMG_DEFAULT_COLORS[index % len(KPMG_DEFAULT_COLORS)],
                            key=f"step6_legend_color_{layout_mode}_{index}",
                        )

                metric_name = str(chart_frame.iloc[0]["指标名称"])
                st.markdown(f"#### {metric_name}")
                st.caption(
                    f"数据源：{source_label or '当前上传文件'}　｜　"
                    f"对比公司：{len(selected_comparison_companies)} 家　｜　"
                    f"对比期间：{chart_frame['报告期'].nunique()} 个　｜　"
                    f"记录数：{len(chart_frame)} 条"
                )
                chart = build_comparison_chart(
                    chart_plot_frame,
                    chart_type,
                    sort_report_periods(chart_frame["报告期"].tolist()),
                    show_labels=show_labels,
                    decimals=int(decimals),
                    layout_mode=layout_mode,
                    legend_label_map=legend_label_map,
                    legend_color_map=legend_color_map,
                    y_axis_title=y_axis_title,
                    transparent=transparent,
                    show_average=show_average,
                    average_color=average_color,
                )
                st.altair_chart(chart, width="stretch")

                with st.expander("查看图表明细数据"):
                    display_columns = [
                        "公司",
                        "公司类型",
                        "同业分类",
                        "报告期",
                        "一级模块",
                        "二级模块",
                        "指标编码",
                        "指标名称",
                        "期间口径",
                        "数值",
                        "单位",
                        "来源类型",
                        "来源文件",
                        "来源工作表",
                        "计算逻辑",
                    ]
                    st.dataframe(
                        chart_frame[display_columns].sort_values(["报告期", "公司"]),
                        width="stretch",
                        hide_index=True,
                    )

with tabs[7]:
    if active_workflow_tab == WORKFLOW_TAB_LABELS[7]:
        company_report_source = st.session_state.step6_analysis_source
        if company_report_source.empty:
            company_report_source = shared_analysis_source
        render_company_report_workspace(company_report_source)
if is_project_member:
    with tabs[8]:
        if active_workflow_tab == WORKFLOW_TAB_LABELS[8]:
            render_industry_report_workspace(shared_analysis_source)

