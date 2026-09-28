from __future__ import annotations

"""Template-driven STEP3 standardization helpers.

The UI deliberately keeps the annual-report platform's target-template workflow,
while the output remains the canonical narrow table consumed by STEP5.
"""

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from .solvency_dataset_adapter import (
    POLICY_SURPLUS_RATIO_COMPONENTS,
    append_derived_metrics,
    canonical_period_scope,
    supported_metric_catalog,
    write_standard_workbook_sheets,
)
from .solvency_metric_registry import CUSTOM_METRICS_BY_CODE, DERIVED_METRICS
from .solvency_normalizer import NARROW_TABLE_COLUMNS, STANDARD_COLUMNS, normalize_tables, normalize_company_type, standardize_uploaded_frame
from .solvency_filing_catalog import filing_target_details, THREE_YEAR_TARGETS, THREE_YEAR_TARGET_CODES
from .solvency_company_identity import (
    canonical_company_name,
    company_display_name,
    resolve_company_identity_from_text,
    resolve_company_identity,
    resolve_peer_group,
)
from .solvency_table_extractor import ExtractedTable
from .solvency_pdf_locator import (
    POLICY_SURPLUS_SOURCE_LABELS,
    extract_report_metadata,
    visible_policy_surplus_codes,
)


TARGET_SHEET_NAME = "指标清单"
STANDARD_SHEET_NAME = "标准数据"
TARGET_TEMPLATE_COLUMNS = [
    "启用",
    "指标编码",
    "指标名称",
    "一级模块",
    "二级模块",
    "标准单位",
    "数据类型",
    "指标属性",
    "计算逻辑",
    "目标表ID", "来源清单", "填报规则", "期间口径",
]


@dataclass(frozen=True)
class Step3StandardizationResult:
    data: pd.DataFrame
    formula_source_data: pd.DataFrame
    target_catalog: pd.DataFrame
    diagnostics: pd.DataFrame
    target_summary: pd.DataFrame
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Step3UploadMetadata:
    company: str = ""
    company_type: str = ""
    report_year: int | None = None
    report_quarter: str = ""
    report_period: str = ""
    disclosure_date: str = ""
    peer_group: str = ""
    company_source: str = ""
    period_source: str = ""
    warnings: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, object]:
        return {
            "公司": self.company,
            "报告年度": self.report_year,
            "报告季度": self.report_quarter,
            "报告期": self.report_period,
            "披露日期": self.disclosure_date,
        }


_QUARTER_MAP = {
    "一": "Q1", "1": "Q1",
    "二": "Q2", "2": "Q2",
    "三": "Q3", "3": "Q3",
    "四": "Q4", "4": "Q4",
}
_METADATA_KEY_ALIASES = {
    "公司": "公司",
    "公司名称": "公司名称",
    "公司全称": "公司名称",
    "报告年度": "报告年度",
    "报表年度": "报告年度",
    "年度": "报告年度",
    "报告季度": "报告季度",
    "报表季度": "报告季度",
    "季度": "报告季度",
    "报告期": "报告期",
    "报告期间": "报告期",
    "报表期间": "报告期",
    "披露日期": "披露日期",
    "发布日期": "披露日期",
    "来源文件": "来源文件",
    "原PDF文件": "来源文件",
    "原始PDF": "来源文件",
}
_METADATA_SHEET_NAMES = {"报告元信息", "报告信息", "元信息", "METADATA"}


def normalize_report_period(value: object) -> tuple[int | None, str, str]:
    """Normalize explicit quarterly period text without matching digits in the year."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    patterns = (
        r"(?<!\d)(20\d{2})\s*[-_/.年]?\s*Q\s*([1-4])(?!\d)",
        r"(?<!\d)(20\d{2})\s*[-_/.年]?\s*([1-4])\s*Q(?!\d)",
        r"(?<!\d)(20\d{2})\s*(?:年|年度)?\s*第?\s*([1-4一二三四])\s*(?:季度|季报|季)",
        r"(?<!\d)Q\s*([1-4])\s*[-_/年]?\s*(20\d{2})(?!\d)",
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            if index == 3:
                year = int(match.group(2))
                quarter_symbol = match.group(1)
            else:
                year = int(match.group(1))
                quarter_symbol = match.group(2)
            quarter = _QUARTER_MAP.get(quarter_symbol, "")
            return year, quarter, f"{year}{quarter}" if quarter else ""
    return None, "", ""


def _normalize_report_year(value: object) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", unicodedata.normalize("NFKC", str(value or "")))
    return int(match.group(1)) if match else None


def _normalize_report_quarter(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    match = re.fullmatch(r"(?:第\s*)?(?:Q\s*)?([1-4一二三四])(?:\s*(?:季度|季报|季))?", text)
    return _QUARTER_MAP.get(match.group(1), "") if match else ""


def _period_from_embedded_metadata(
    embedded: Mapping[str, str],
) -> tuple[int | None, str, str]:
    year, quarter, period = normalize_report_period(embedded.get("报告期", ""))
    if period:
        return year, quarter, period
    year = _normalize_report_year(embedded.get("报告年度", ""))
    quarter = _normalize_report_quarter(embedded.get("报告季度", ""))
    return year, quarter, f"{year}{quarter}" if year and quarter else ""


def _date_candidates(text: str) -> list[tuple[str, int, int, int, str]]:
    candidates: list[tuple[str, int, int, int, str]] = []
    normalized = unicodedata.normalize("NFKC", text)
    pattern = re.compile(
        r"(?<!\d)(20\d{2})\s*[-/.年]\s*(1[0-2]|0?[1-9])\s*[-/.月]\s*"
        r"(3[01]|[12]\d|0?[1-9])\s*日?"
    )
    for match in pattern.finditer(normalized):
        year, month, day = map(int, match.groups())
        line_start = normalized.rfind("\n", 0, match.start()) + 1
        line_end = normalized.find("\n", match.end())
        if line_end < 0:
            line_end = len(normalized)
        context = normalized[line_start:line_end]
        candidates.append((f"{year:04d}-{month:02d}-{day:02d}", year, month, day, context))
    return candidates


def _period_from_dates(text: str) -> tuple[int | None, str, str, str]:
    candidates = _date_candidates(text)
    if not candidates:
        return None, "", "", ""

    frequencies = Counter(candidate[0] for candidate in candidates)
    scored: list[tuple[int, int, int, int, str]] = []
    quarter_ends = {(3, 31), (6, 30), (9, 30), (12, 31)}
    for date_text, year, month, day, context in candidates:
        report_context = bool(re.search(r"报告期|报告日|截至|期末|本季度末|年末", context))
        disclosure_context = bool(re.search(r"披露日期|发布日期|公告日期|出具日期", context))
        is_quarter_end = (month, day) in quarter_ends
        if disclosure_context and not report_context:
            continue
        if not is_quarter_end and not report_context:
            continue
        score = frequencies[date_text] * 2 + (6 if is_quarter_end else 0) + (5 if report_context else 0)
        scored.append((score, year, month, day, date_text))
    if not scored:
        return None, "", "", ""
    _, year, month, _, date_text = max(scored)
    quarter = f"Q{min(4, (month - 1) // 3 + 1)}"
    return year, quarter, f"{year}{quarter}", date_text


def _normalize_metadata_key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    compact = re.sub(r"[\s:：]", "", text)
    return _METADATA_KEY_ALIASES.get(compact, "")


def _normalize_date(value: object) -> str:
    candidates = _date_candidates(str(value or ""))
    return candidates[0][0] if candidates else str(value or "").strip()


def _workbook_text_and_metadata(workbook_bytes: bytes) -> tuple[str, dict[str, str]]:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    fragments: list[str] = []
    embedded: dict[str, str] = {}
    for sheet_name in excel.sheet_names:
        frame = pd.read_excel(
            excel,
            sheet_name=sheet_name,
            header=None,
            nrows=250,
            dtype=str,
            keep_default_na=False,
        )
        rows = [[str(cell).strip() for cell in row] for row in frame.values.tolist()]
        fragments.append(sheet_name)
        fragments.extend(" | ".join(cell for cell in row if cell) for row in rows)
        if unicodedata.normalize("NFKC", sheet_name).strip().upper() not in _METADATA_SHEET_NAMES:
            continue
        for row in rows:
            key = _normalize_metadata_key(row[0]) if row else ""
            if len(row) >= 2 and key and row[1]:
                embedded[key] = row[1]
        if len(rows) >= 2:
            for index, raw_key in enumerate(rows[0]):
                key = _normalize_metadata_key(raw_key)
                if key and index < len(rows[1]) and rows[1][index]:
                    embedded[key] = rows[1][index]
    return "\n".join(fragment for fragment in fragments if fragment), embedded


def _configured_company_match(
    text: str,
    companies: Sequence[Mapping[str, object]],
) -> tuple[str, str]:
    normalized_text = canonical_company_name(text)
    matches: list[tuple[int, str, str]] = []
    for item in companies:
        name = str(item.get("company_name") or item.get("公司名称") or "").strip()
        company_type = str(item.get("company_type") or item.get("公司类型") or "").strip()
        key = canonical_company_name(name)
        if key and key in normalized_text:
            matches.append((len(key), name, company_type))
        display_key = canonical_company_name(company_display_name(name))
        if display_key and display_key != key and display_key in normalized_text:
            matches.append((len(display_key), name, company_type))
    if matches:
        _, name, company_type = max(matches, key=lambda item: item[0])
        return name, company_type

    company_type_map = {
        str(item.get("company_name") or item.get("公司名称") or "").strip():
        str(item.get("company_type") or item.get("公司类型") or "").strip()
        for item in companies
    }
    identity = resolve_company_identity_from_text(text, company_type_map)
    if identity is None:
        return "", ""
    identity_key = canonical_company_name(identity.standard_name)
    for item in companies:
        name = str(item.get("company_name") or item.get("公司名称") or "").strip()
        if canonical_company_name(name) == identity_key:
            company_type = str(
                item.get("company_type") or item.get("公司类型") or identity.company_type
            ).strip()
            return name, company_type
    return "", ""


def _fallback_company_from_text(text: str) -> str:
    match = re.search(
        r"公司名称\s*[:：]\s*([^\r\n|，,;；]+?)(?=\s+20\d{2}|\s+单位|$)",
        text,
    )
    return match.group(1).strip() if match else ""


def infer_step3_session_metadata(
    metadata: Mapping[str, object],
    filename: str,
    companies: Sequence[Mapping[str, object]],
    *,
    peer_group_map: Mapping[str, str] | None = None,
    default_peer_group: str = "",
) -> Step3UploadMetadata:
    """Recover STEP2 session identity when PDF cover-text recognition missed it."""
    raw_company = str(metadata.get("公司", "") or "").strip()
    company, company_type = _configured_company_match(raw_company, companies)
    if not company:
        company = raw_company
    company_source = "PDF报告元信息" if company else ""

    filename_company, filename_company_type = _configured_company_match(
        Path(filename or "").stem, companies
    )
    if not filename_company and peer_group_map:
        filename_company, filename_company_type = _configured_company_match(
            Path(filename or "").stem,
            tuple({"company_name": name} for name in peer_group_map),
        )
    warnings: list[str] = []
    if not company and filename_company:
        company, company_type = filename_company, filename_company_type
        company_source = "PDF文件名"
    elif company and filename_company and canonical_company_name(company) != canonical_company_name(filename_company):
        warnings.append(
            f"PDF元信息识别为“{company}”，文件名识别为“{filename_company}”；请核对公司名称。"
        )

    year, quarter, period = _period_from_embedded_metadata({
        key: str(metadata.get(key, "") or "")
        for key in ("报告年度", "报告季度", "报告期")
    })
    period_source = "PDF报告元信息" if period else ""
    filename_year, filename_quarter, filename_period = normalize_report_period(filename)
    if not period and filename_period:
        year, quarter, period = filename_year, filename_quarter, filename_period
        period_source = "PDF文件名"
    elif period and filename_period and period != filename_period:
        warnings.append(
            f"PDF元信息报告期为“{period}”，文件名为“{filename_period}”；请核对报告期。"
        )
    if not company:
        warnings.append("未能自动识别公司名称，请在下方补充。")
    if not period:
        warnings.append("未能自动识别报告期，请在下方补充。")
    return Step3UploadMetadata(
        company=company,
        company_type=company_type,
        report_year=year,
        report_quarter=quarter,
        report_period=period,
        disclosure_date=str(metadata.get("披露日期", "") or ""),
        peer_group=resolve_peer_group(company, peer_group_map, default_peer_group) if company else "",
        company_source=company_source,
        period_source=period_source,
        warnings=tuple(warnings),
    )


def infer_step3_upload_metadata(
    workbook_bytes: bytes,
    filename: str,
    companies: Sequence[Mapping[str, object]],
    *,
    peer_group_map: Mapping[str, str] | None = None,
    default_peer_group: str = "",
    context_metadata: Mapping[str, object] | None = None,
    context_filename: str = "",
) -> Step3UploadMetadata:
    """Infer STEP3 company and reporting period from workbook content, then filename."""
    if not workbook_bytes:
        return Step3UploadMetadata()

    workbook_text, embedded = _workbook_text_and_metadata(workbook_bytes)
    embedded_company = embedded.get("公司") or embedded.get("公司名称") or ""
    embedded_source_file = embedded.get("来源文件", "")
    company, company_type = _configured_company_match(embedded_company, companies)
    company_source = "工作簿元信息" if company else ""
    if not company:
        company, company_type = _configured_company_match(embedded_source_file, companies)
        company_source = "原PDF文件名" if company else ""
    if not company:
        company, company_type = _configured_company_match(workbook_text, companies)
        company_source = "工作簿内容" if company else ""
    if not company and embedded_company:
        company = embedded_company
        company_source = "工作簿元信息"
    if not company:
        fallback_company = _fallback_company_from_text(workbook_text)
        company, company_type = _configured_company_match(fallback_company, companies)
        company = company or fallback_company
        company_source = "工作簿内容" if company else ""

    filename_text = Path(filename or "").stem
    filename_company, filename_company_type = _configured_company_match(filename_text, companies)
    warnings: list[str] = []
    if company and filename_company and company != filename_company:
        warnings.append(
            f"工作簿识别为“{company}”，文件名识别为“{filename_company}”；已采用工作簿内容。"
        )
    if not company and filename_company:
        company, company_type = filename_company, filename_company_type
        company_source = "文件名"

    year, quarter, period = _period_from_embedded_metadata(embedded)
    period_source = "工作簿元信息" if period else ""
    if not period:
        year, quarter, period = normalize_report_period(embedded_source_file)
        period_source = "原PDF文件名" if period else ""
    if not period:
        year, quarter, period = normalize_report_period(workbook_text)
        period_source = "工作簿内容" if period else ""
    disclosure_date = _normalize_date(embedded.get("披露日期", ""))
    date_year, date_quarter, date_period, detected_date = _period_from_dates(workbook_text)
    if not period and date_period:
        year, quarter, period = date_year, date_quarter, date_period
        period_source = "工作簿日期"
    if not disclosure_date:
        disclosure_date = detected_date

    filename_year, filename_quarter, filename_period = normalize_report_period(filename_text)
    source_year, source_quarter, source_period = normalize_report_period(embedded_source_file)
    if (
        period and source_period and filename_period
        and source_period == filename_period != period
    ):
        warnings.append(
            f"工作簿元信息报告期为“{period}”，原PDF与上传文件名均为“{source_period}”；"
            "已采用一致的文件来源期间。"
        )
        year, quarter, period = source_year, source_quarter, source_period
        period_source = "原PDF与上传文件名一致"
    elif period and filename_period and period != filename_period:
        warnings.append(
            f"工作簿识别报告期为“{period}”，文件名识别为“{filename_period}”；已采用工作簿内容。"
        )
    if not period and filename_period:
        year, quarter, period = filename_year, filename_quarter, filename_period
        period_source = "文件名"

    context = dict(context_metadata or {})
    context_company_text = str(context.get("公司", "") or "")
    context_company, _ = _configured_company_match(context_company_text, companies)
    if not context_company:
        context_company, _ = _configured_company_match(context_filename, companies)
    context_year, context_quarter, context_period = _period_from_embedded_metadata({
        "报告年度": str(context.get("报告年度", "") or ""),
        "报告季度": str(context.get("报告季度", "") or ""),
        "报告期": str(context.get("报告期", "") or ""),
    })
    if not context_period:
        context_year, context_quarter, context_period = normalize_report_period(context_filename)
    if not period and company and context_company == company and context_period:
        year, quarter, period = context_year, context_quarter, context_period
        period_source = "当前PDF会话（公司一致）"

    date_year_match = re.match(r"(20\d{2})-", disclosure_date)
    if year and date_year_match and not year <= int(date_year_match.group(1)) <= year + 1:
        warnings.append(f"工作簿披露日期“{disclosure_date}”与报告期“{period}”不符，已清空。")
        disclosure_date = ""

    if not company:
        warnings.append("未能自动识别公司名称，请人工补充。")
    if not period:
        warnings.append(
            "未能自动识别报告期：工作簿中没有报告期、年度与季度组合或可核验的期末日期，"
            "文件名中也未包含季度，请人工补充。"
        )
    peer_group = (
        resolve_peer_group(company, peer_group_map, default_peer_group)
        if company
        else ""
    )
    return Step3UploadMetadata(
        company=company,
        company_type=company_type,
        report_year=year,
        report_quarter=quarter,
        report_period=period,
        disclosure_date=disclosure_date,
        peer_group=peer_group,
        company_source=company_source,
        period_source=period_source,
        warnings=tuple(warnings),
    )


def step3_metric_catalog(
    taxonomy: pd.DataFrame,
    *,
    include_derived: bool,
    use_checklists: bool = False,
) -> pd.DataFrame:
    """Return canonical filing targets, optionally scoped to the supplied lists."""
    catalog = supported_metric_catalog(
        taxonomy,
        include_derived=include_derived,
        include_step3_only=True,
    ).copy()
    if use_checklists:
        details = filing_target_details()
        ordered = list(details)
        if include_derived:
            ordered.extend(d.code for d in DERIVED_METRICS if d.code not in ordered)
        catalog = catalog[catalog['指标编码'].isin(ordered)].copy()
        catalog['_order'] = catalog['指标编码'].map({code:i for i, code in enumerate(ordered)})
        catalog = catalog.sort_values('_order', kind='stable').drop(columns='_order')
        for column in ('目标表ID', '来源清单', '填报规则'):
            catalog[column] = catalog['指标编码'].map(lambda code: details.get(code, {}).get(column, ''))
    catalog.insert(0, "启用", "是")
    catalog['期间口径'] = catalog['指标编码'].map(
        lambda code: '本季度' if code in THREE_YEAR_TARGET_CODES else '本季度末数')
    return catalog.reindex(columns=TARGET_TEMPLATE_COLUMNS).fillna('')


def _find_header_row(preview: pd.DataFrame, required_header: str) -> int | None:
    for index, row in preview.iterrows():
        values = {str(value).strip() for value in row if not pd.isna(value)}
        if required_header in values:
            return int(index)
    return None


def _enabled(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"否", "不启用", "false", "0", "n", "no"}


def read_target_template(
    workbook_bytes: bytes,
    filename: str,
    supported_catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Read a target list and constrain it to STEP5-supported metric codes."""
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    sheet_name = (
        TARGET_SHEET_NAME
        if TARGET_SHEET_NAME in excel.sheet_names
        else STANDARD_SHEET_NAME
        if STANDARD_SHEET_NAME in excel.sheet_names
        else excel.sheet_names[0]
    )
    preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=20)
    header_row = _find_header_row(preview, "指标编码")
    if header_row is None:
        raise ValueError(f"{filename} 未找到“指标编码”表头。")

    frame = pd.read_excel(excel, sheet_name=sheet_name, header=header_row).fillna("")
    frame.columns = [str(column).strip() for column in frame.columns]
    if "启用" in frame.columns:
        frame = frame[frame["启用"].map(_enabled)]
    raw_codes = [
        str(value).strip()
        for value in frame["指标编码"]
        if str(value).strip()
    ]
    codes = list(dict.fromkeys(raw_codes))
    if not codes:
        raise ValueError(f"{filename} 未配置任何已启用的目标指标。")

    supported_codes = set(supported_catalog["指标编码"].astype(str))
    unknown_codes = [code for code in codes if code not in supported_codes]
    if unknown_codes:
        raise ValueError(
            "目标表包含 STEP3 正式指标字典之外的编码：" + "、".join(unknown_codes)
        )

    indexed = supported_catalog.set_index("指标编码", drop=False)
    selected = pd.DataFrame([indexed.loc[code] for code in codes]).reset_index(drop=True)
    selected["启用"] = "是"

    warnings: list[str] = []
    duplicate_count = len(raw_codes) - len(codes)
    if duplicate_count > 0:
        warnings.append(f"目标表中 {duplicate_count} 条重复指标已按指标编码去重。")
    return selected.reindex(columns=TARGET_TEMPLATE_COLUMNS), tuple(warnings)


def _dependency_closure(target_codes: set[str]) -> set[str]:
    derived_by_code = {definition.code: definition for definition in DERIVED_METRICS}
    required = set(target_codes)
    pending = list(target_codes)
    while pending:
        code = pending.pop()
        definition = derived_by_code.get(code)
        if definition is None:
            continue
        for dependency in definition.dependencies:
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return required


def _standardize_step3_period_scopes(
    frame: pd.DataFrame,
    metadata: Mapping[str, object],
) -> pd.DataFrame:
    """Collapse equivalent current-quarter labels before calculating metrics."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    _, _, report_period = normalize_report_period(metadata.get("报告期", ""))
    if not report_period:
        year = _normalize_report_year(metadata.get("报告年度", ""))
        quarter = _normalize_report_quarter(metadata.get("报告季度", ""))
        report_period = f"{year}{quarter}" if year and quarter else ""

    def normalize_scope(value: object) -> str:
        text = str(value or "").strip()
        if any(term in text for term in ("近三年", "累计", "预测", "上季度", "期初")):
            return canonical_period_scope(text)
        canonical = canonical_period_scope(text)
        if canonical == "本季度末数":
            return canonical
        _, _, explicit_period = normalize_report_period(text)
        if report_period and explicit_period == report_period:
            return "本季度末数"
        quarter_end_dates = {(3, 31): "Q1", (6, 30): "Q2", (9, 30): "Q3", (12, 31): "Q4"}
        for _, year, month, day, _ in _date_candidates(text):
            quarter = quarter_end_dates.get((month, day), "")
            if report_period and quarter and f"{year}{quarter}" == report_period:
                return "本季度末数"
        return canonical

    result["期间口径"] = result["期间口径"].map(normalize_scope)
    return result


def _verify_policy_surplus_against_pdf(
    frame: pd.DataFrame,
    metadata: Mapping[str, object],
    source_pdf_bytes: bytes,
) -> tuple[pd.DataFrame, list[str]]:
    """Correct legacy STEP2 zero claims only when the matching PDF is supplied."""
    pdf_metadata = extract_report_metadata(source_pdf_bytes)
    pdf_period = str(pdf_metadata.get("报告期", "") or "")
    report_period = str(metadata.get("报告期", "") or "")
    if pdf_period and report_period and pdf_period != report_period:
        raise ValueError(f"原始PDF报告期为{pdf_period}，与STEP3报告期{report_period}不一致。")
    pdf_company = canonical_company_name(company_display_name(pdf_metadata.get("公司", "")))
    report_company = canonical_company_name(company_display_name(metadata.get("公司", "")))
    if pdf_company and report_company and pdf_company != report_company:
        raise ValueError("原始PDF公司与STEP3报告公司不一致。")

    capital_rows = frame.loc[frame["一级模块"].astype(str).eq("实际资本")]
    numeric_pages = pd.to_numeric(capital_rows.get("来源页码", pd.Series(dtype=object)), errors="coerce")
    pages = sorted({int(page) for page in numeric_pages.dropna() if page >= 1})
    visible_codes = visible_policy_surplus_codes(source_pdf_bytes, pages or None)
    if visible_codes is None:
        return frame, []
    result = frame.copy()
    corrected: list[str] = []
    for code in POLICY_SURPLUS_SOURCE_LABELS:
        if code in visible_codes:
            continue
        mask = result["指标编码"].astype(str).eq(code) & pd.to_numeric(
            result["数值"], errors="coerce"
        ).notna()
        if not mask.any():
            continue
        result.loc[mask, "数值"] = float("nan")
        result.loc[mask, "披露状态"] = "未披露"
        result.loc[mask, "原始披露值"] = ""
        result.loc[mask, "来源页码"] = float("nan")
        result.loc[mask, "备注"] = "经原始PDF实际资本表核对：未列示该项保单未来盈余。"
        corrected.append(code)
    return result, corrected


def _deduplicate_step3_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep one best record for each company/report/metric/period combination."""
    if frame.empty:
        return frame.copy()
    result = frame.copy()
    keys = [
        column
        for column in ("报告类型", "公司统一编码", "报告期", "指标编码", "期间口径")
        if column in result.columns
    ]
    if not keys:
        return result
    values = result.get("数值", pd.Series(index=result.index, dtype=object))
    result["_有值"] = values.notna() & values.astype(str).str.strip().ne("")
    state_rank = {
        "已披露为0": 5,
        "已披露": 4,
        "已计算": 3,
        "不适用": 2,
        "未披露": 1,
        "无法计算": 0,
    }
    states = result.get("披露状态", pd.Series(index=result.index, dtype=str))
    result["_状态优先级"] = states.fillna("").astype(str).map(state_rank).fillna(0)
    result["_原始顺序"] = range(len(result))
    result = result.sort_values(
        [*keys, "_有值", "_状态优先级", "_原始顺序"],
        ascending=[True] * len(keys) + [False, False, True],
        kind="stable",
    )
    result = result.drop_duplicates(keys, keep="first").sort_values(
        "_原始顺序", kind="stable"
    )
    return result.drop(columns=["_有值", "_状态优先级", "_原始顺序"])


def _derived_unavailable_reason(
    code: str,
    normalized: pd.DataFrame,
    taxonomy: pd.DataFrame,
) -> str:
    definitions = {definition.code: definition for definition in DERIVED_METRICS}
    definition = definitions.get(code)
    if definition is None:
        return "计算依赖不足"

    names = {
        str(row.get("指标编码", "")).strip(): str(row.get("指标名称", "")).strip()
        for _, row in taxonomy.iterrows()
    }
    names.update({
        metric_code: metric.name
        for metric_code, metric in CUSTOM_METRICS_BY_CODE.items()
    })
    current = normalized.copy()
    if not current.empty:
        current = current[
            current["期间口径"].map(canonical_period_scope).eq("本季度末数")
        ]

    missing: list[str] = []
    values: dict[str, float] = {}
    optional_components = POLICY_SURPLUS_RATIO_COMPONENTS.get(code, ())
    for dependency in definition.dependencies:
        rows = current[current["指标编码"].astype(str).eq(dependency)]
        numeric = pd.to_numeric(rows.get("数值", pd.Series(dtype=float)), errors="coerce").dropna()
        if not numeric.empty:
            values[dependency] = float(numeric.iloc[0])
            continue
        if dependency in optional_components:
            continue
        label = names.get(dependency, dependency)
        states = set(rows.get("披露状态", pd.Series(dtype=str)).fillna("").astype(str))
        suffix = "披露为不适用" if "不适用" in states else "未披露"
        missing.append(f"{label}{suffix}")
    if missing:
        return "无法计算：" + "、".join(missing)

    if optional_components and not any(item in values for item in optional_components):
        return "无法计算：相关保单未来盈余层级均未披露"

    if code == "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL":
        denominator = values.get("CORE_T1_CAPITAL", 0.0) + values.get("CORE_T2_CAPITAL", 0.0)
        if denominator == 0:
            return "无法计算：核心资本为0"
    if code == "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES" and values.get(
        "INSURANCE_CONTRACT_LIABILITY", 0.0
    ) == 0:
        return "无法计算：保险合同负债为0"
    return "无法计算：依赖指标期间口径不一致或数值无效"


def _target_summary(
    target_catalog: pd.DataFrame,
    data: pd.DataFrame,
) -> pd.DataFrame:
    available = data[data['数值'].notna() & data['数值'].astype(str).str.strip().ne('')]
    counts = available["指标编码"].astype(str).value_counts() if not available.empty else pd.Series(dtype=int)
    summary = target_catalog[
        ["指标编码", "指标名称", "一级模块", "二级模块", "指标属性"]
    ].copy()
    summary["填报记录数"] = summary["指标编码"].map(counts).fillna(0).astype(int)
    states = data.drop_duplicates('指标编码').set_index('指标编码')['披露状态'].to_dict() if not data.empty else {}
    summary['填报状态'] = [
        '已填报' if count else states.get(code, '未披露')
        for code, count in zip(summary['指标编码'], summary['填报记录数'])
    ]
    return summary


def standardize_to_target(
    tables: list[ExtractedTable],
    taxonomy: pd.DataFrame,
    metadata: dict,
    company_type: str,
    target_catalog: pd.DataFrame,
    *,
    report_profile_id: str,
    allowed_company_types: Sequence[str],
    peer_group: str = "",
    peer_group_map: Mapping[str, str] | None = None,
    default_peer_group: str = "",
    include_derived: bool,
    template_warnings: Sequence[str] = (),
    source_pdf_bytes: bytes | None = None,
) -> Step3StandardizationResult:
    """Fill a target list and return the canonical STEP5-compatible narrow table."""
    target_codes = set(target_catalog["指标编码"].astype(str).str.strip())
    required_codes = _dependency_closure(target_codes)
    derived_codes = {definition.code for definition in DERIVED_METRICS}
    normalization_codes = required_codes - derived_codes
    # Keep old STEP2 exports and legacy per-table Excel files readable even when
    # the user enables only the new three-year targets, not quarterly returns.
    normalization_codes.update(old for old, (new, _) in THREE_YEAR_TARGETS.items() if new in target_codes)
    scoped_taxonomy = taxonomy[
        taxonomy["指标编码"].astype(str).str.strip().isin(normalization_codes)
    ].copy()

    diagnostics: list[dict] = []
    normalized = normalize_tables(
        tables,
        scoped_taxonomy,
        metadata,
        company_type,
        report_profile_id=report_profile_id,
        diagnostics=diagnostics,
        allowed_company_types=allowed_company_types,
        peer_group=peer_group,
        peer_group_map=peer_group_map,
        default_peer_group=default_peer_group,
    )
    corrected_policy_codes: list[str] = []
    if source_pdf_bytes and required_codes.intersection(POLICY_SURPLUS_SOURCE_LABELS):
        normalized, corrected_policy_codes = _verify_policy_surplus_against_pdf(
            normalized, metadata, source_pdf_bytes,
        )
    for old_code, (new_code, name) in THREE_YEAR_TARGETS.items():
        if new_code not in target_codes:
            continue
        mask = normalized['指标编码'].eq(old_code) & normalized['期间口径'].astype(str).str.contains('近三年平均', regex=False)
        normalized.loc[mask, '指标编码'] = new_code
        normalized.loc[mask, '指标名称'] = name
        normalized.loc[mask, '二级模块'] = '近三年投资收益率'
        normalized.loc[mask, '期间口径'] = '本季度'
    # Three-year averaging is the metric's meaning, not the report period.
    # Also accept previously coded exports with the old average-period label.
    average_mask = normalized['指标编码'].isin(THREE_YEAR_TARGET_CODES) & normalized['期间口径'].isin(
        ['近三年平均', '本季度', '本季度末数', '本季度数'])
    normalized.loc[average_mask, '期间口径'] = '本季度'
    quarterly_mask = ~normalized['指标编码'].isin(THREE_YEAR_TARGET_CODES)
    normalized.loc[quarterly_mask] = _standardize_step3_period_scopes(
        normalized.loc[quarterly_mask],
        metadata,
    )
    normalized = _deduplicate_step3_metrics(normalized).reset_index(drop=True)
    enriched = append_derived_metrics(normalized) if include_derived else normalized
    output_codes = set(target_codes)
    registered_rows = normalized[
        normalized["指标编码"].astype(str).eq("REGISTERED_CAPITAL")
    ]
    if (
        "REGISTERED_CAPITAL" in required_codes
        and pd.to_numeric(registered_rows.get("数值", pd.Series(dtype=float)), errors="coerce").notna().any()
    ):
        output_codes.add("REGISTERED_CAPITAL")
    data = enriched[
        enriched["指标编码"].astype(str).str.strip().isin(output_codes)
    ].copy()
    data = standardize_uploaded_frame(data)
    data = _deduplicate_step3_metrics(data).reset_index(drop=True)
    current_codes = set(data.loc[~data['期间口径'].astype(str).str.contains('近三年|上季度|期初|预测|累计', regex=True), '指标编码'])
    identity = resolve_company_identity(metadata.get('公司', ''), fallback_company_type=normalize_company_type(company_type, allowed_company_types))
    group_name = peer_group or resolve_peer_group(identity.standard_name, peer_group_map, default_peer_group)
    missing_rows = []
    for _, target in target_catalog.iterrows():
        code = str(target['指标编码'])
        if code in current_codes:
            continue
        calculated = code in derived_codes
        missing_rows.append({
            '公司': identity.standard_name, '原始公司名称': identity.original_name,
            '标准公司名称': identity.standard_name, '公司统一编码': identity.company_code,
            '公司类型': identity.company_type, '同业分类': group_name,
            '报告类型': report_profile_id, '报告年度': metadata.get('报告年度'),
            '报告季度': metadata.get('报告季度', ''), '报告期': metadata.get('报告期', ''),
            '披露日期': metadata.get('披露日期', ''), '一级模块': target['一级模块'],
            '二级模块': target['二级模块'], '指标编码': code, '指标名称': target['指标名称'],
            '期间口径': '本季度' if code in THREE_YEAR_TARGET_CODES else '本季度末数', '数值': float('nan'), '单位': target['标准单位'],
            '数据类型': target['数据类型'], '是否预测': '否', '来源页码': '',
            '原始披露值': '', '备注': _derived_unavailable_reason(code, normalized, taxonomy) if calculated else 'Step2中无对应近三年平均指标披露' if code in THREE_YEAR_TARGET_CODES else 'Step2中无对应本季度指标披露',
            '来源类型': '系统计算' if calculated else '报告提取', '指标属性': target['指标属性'],
            '来源文件': metadata.get('来源文件', ''), '来源工作表': '',
            '导入批次': metadata.get('导入批次', ''), '计算逻辑': target.get('计算逻辑', ''),
            '披露状态': '无法计算' if calculated else '未披露',
        })
    if missing_rows:
        data = pd.concat([data, pd.DataFrame(missing_rows)], ignore_index=True)
    formula_source_data = enriched[
        enriched["指标编码"].astype(str).str.strip().isin(required_codes)
    ].copy().reindex(columns=STANDARD_COLUMNS).reset_index(drop=True)
    if not data.empty:
        target_order = {
            code: index
            for index, code in enumerate(target_catalog["指标编码"].astype(str))
        }
        data["_目标顺序"] = data["指标编码"].map(target_order)
        data = data.sort_values(
            ["_目标顺序", "期间口径", "来源页码"],
            kind="stable",
        ).drop(columns="_目标顺序")
    data = data.reindex(columns=STANDARD_COLUMNS).reset_index(drop=True)
    diagnostic_frame = pd.DataFrame(diagnostics)
    summary = _target_summary(target_catalog, data)
    warnings = list(template_warnings)
    if corrected_policy_codes:
        warnings.append(
            "已按原始PDF核对并纠正未列示的保单未来盈余子项："
            + "、".join(POLICY_SURPLUS_SOURCE_LABELS[code] for code in corrected_policy_codes)
        )
    missing_count = int(summary['填报状态'].isin({'未披露', '无法计算'}).sum())
    if missing_count:
        warnings.append(
            f"目标表中有 {missing_count} 个指标未取得数值，已在标准窄表保留并标记未披露或无法计算。"
        )
    return Step3StandardizationResult(
        data=data,
        formula_source_data=formula_source_data,
        target_catalog=target_catalog.copy(),
        diagnostics=diagnostic_frame,
        target_summary=summary,
        warnings=tuple(warnings),
    )


def _format_workbook(writer: pd.ExcelWriter) -> None:
    for worksheet in writer.book.worksheets:
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions
        for column_cells in worksheet.iter_cols():
            values = [str(cell.value or "") for cell in column_cells[:200]]
            width = min(max(max((len(value) for value in values), default=0) + 2, 10), 32)
            worksheet.column_dimensions[column_cells[0].column_letter].width = width


def target_template_workbook_bytes(
    target_catalog: pd.DataFrame,
    profile_name: str,
) -> bytes:
    output = io.BytesIO()
    instructions = pd.DataFrame(
        [
            ("用途", f"{profile_name} STEP3 目标指标配置；输出可直接进入 STEP5。"),
            ("选择指标", "在“指标清单”中将不需要的指标“启用”改为“否”，不要修改指标编码。"),
            ("窄表字段", "“标准数据”页保留所有启用目标；未取得的披露指标在数值单元格填写“未披露”；原表横杠与数字0统一填0并保留原值；明确不适用与派生指标无法计算保持区分。"),
            ("单位口径", "按各指标标准单位输出；百分比保留披露百分数，例如 130.98 表示 130.98%；文本指标保留原文。"),
        ],
        columns=["项目", "说明"],
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="使用说明", index=False)
        target_catalog.reindex(columns=TARGET_TEMPLATE_COLUMNS).to_excel(
            writer,
            sheet_name=TARGET_SHEET_NAME,
            index=False,
        )
        pd.DataFrame(columns=NARROW_TABLE_COLUMNS).to_excel(
            writer,
            sheet_name=STANDARD_SHEET_NAME,
            index=False,
        )
        _format_workbook(writer)
    return output.getvalue()


def result_workbook_bytes(result: Step3StandardizationResult) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        write_standard_workbook_sheets(
            writer,
            result.data,
            formula_source=result.formula_source_data,
        )
        result.target_summary.to_excel(writer, sheet_name="填报汇总", index=False)
        if result.diagnostics.empty:
            pd.DataFrame(columns=["状态", "原因"]).to_excel(
                writer,
                sheet_name="匹配诊断",
                index=False,
            )
        else:
            result.diagnostics.to_excel(writer, sheet_name="匹配诊断", index=False)
        _format_workbook(writer)
    return output.getvalue()
