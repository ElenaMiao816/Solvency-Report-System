from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .solvency_step6_analysis import sort_report_periods


REQUIRED_FINANCING_HEADERS = {"季度", "公司名称", "增资/发债"}
OUTPUT_COLUMNS = [
    "季度",
    "公司名称",
    "增资/发债",
    "季度总变动",
    "增资/发债的影响",
    "来源文件",
    "来源工作表",
    "来源行号",
]
QUARTER_PATTERN = re.compile(
    r"^(?P<year>20\d{2})\s*[-_/年]?\s*[Qq第]?\s*(?P<quarter>[1-4])(?:季度)?$"
)
TOTAL_CHANGE_PATTERN = re.compile(r"季度总变动\s*[：:]\s*([^\r\n]+)")
FINANCING_IMPACT_PATTERN = re.compile(
    r"增资\s*[/／]\s*发债的影响\s*[：:]\s*([^\r\n]+)"
)


@dataclass(frozen=True)
class FinancingSummary:
    event_count: int
    company_count: int
    latest_period: str


def _text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_quarter(value: Any) -> str:
    text = _text(value)
    match = QUARTER_PATTERN.fullmatch(text)
    if match is None:
        return ""
    return f"{match.group('year')}Q{match.group('quarter')}"


def split_solvency_change(value: Any) -> tuple[str, str]:
    text = _text(value)
    total_match = TOTAL_CHANGE_PATTERN.search(text)
    impact_match = FINANCING_IMPACT_PATTERN.search(text)
    total = total_match.group(1).strip() if total_match else "未披露"
    impact = impact_match.group(1).strip() if impact_match else "未披露"
    return total, impact


def normalize_financing_frame(
    frame: pd.DataFrame,
    *,
    source_filename: str,
    source_sheet: str,
    header_row: int = 0,
) -> pd.DataFrame:
    working = frame.copy()
    working.columns = [str(column).strip() for column in working.columns]
    missing = REQUIRED_FINANCING_HEADERS - set(working.columns)
    has_change = "综合充足率变动" in working.columns
    has_split_change = {"季度总变动", "增资/发债的影响"}.issubset(working.columns)
    if missing or not (has_change or has_split_change):
        required = "季度、公司名称、增资/发债，以及综合充足率变动（或两个拆分字段）"
        raise ValueError(f"重大融资工作表缺少必要字段，需要包含：{required}。")

    records: list[dict[str, Any]] = []
    invalid_period_rows: list[int] = []
    for index, row in working.iterrows():
        company = _text(row.get("公司名称"))
        financing = _text(row.get("增资/发债"))
        raw_period = _text(row.get("季度"))
        if not company and not financing and not raw_period:
            continue
        period = normalize_quarter(raw_period)
        source_row = int(index) + header_row + 2
        if not period:
            invalid_period_rows.append(source_row)
            continue
        if not company or not financing:
            continue

        if has_split_change:
            total_change = _text(row.get("季度总变动")) or "未披露"
            financing_impact = _text(row.get("增资/发债的影响")) or "未披露"
        else:
            total_change, financing_impact = split_solvency_change(
                row.get("综合充足率变动")
            )
        records.append({
            "季度": period,
            "公司名称": company,
            "增资/发债": financing,
            "季度总变动": total_change,
            "增资/发债的影响": financing_impact,
            "来源文件": source_filename,
            "来源工作表": source_sheet,
            "来源行号": source_row,
        })

    if invalid_period_rows:
        rows = "、".join(str(value) for value in invalid_period_rows[:10])
        suffix = "等" if len(invalid_period_rows) > 10 else ""
        raise ValueError(f"以下行的季度无法识别：{rows}{suffix}。")
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    if result.empty:
        raise ValueError("重大融资工作表中没有可展示的有效记录。")
    period_order = {value: index for index, value in enumerate(sort_report_periods(result["季度"]))}
    result["_period_order"] = result["季度"].map(period_order)
    return (
        result.sort_values(["_period_order", "公司名称", "来源行号"], ascending=[False, True, True])
        .drop(columns="_period_order")
        .reset_index(drop=True)
    )


def read_major_financing_workbook(
    workbook_bytes: bytes,
    source_filename: str,
) -> pd.DataFrame:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    for sheet_name in excel.sheet_names:
        preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=12)
        for index, row in preview.iterrows():
            values = {str(value).strip() for value in row if not pd.isna(value)}
            if REQUIRED_FINANCING_HEADERS.issubset(values):
                frame = pd.read_excel(excel, sheet_name=sheet_name, header=int(index))
                return normalize_financing_frame(
                    frame,
                    source_filename=source_filename,
                    source_sheet=str(sheet_name),
                    header_row=int(index),
                )
    raise ValueError("未找到重大融资信息表头，需要包含：季度、公司名称、增资/发债。")


def summarize_financing(frame: pd.DataFrame | None) -> FinancingSummary:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return FinancingSummary(0, 0, "")
    periods = sort_report_periods(frame.get("季度", pd.Series(dtype="string")).tolist())
    companies = frame.get("公司名称", pd.Series(dtype="string")).fillna("").astype(str).str.strip()
    return FinancingSummary(
        event_count=len(frame),
        company_count=int(companies[companies != ""].nunique()),
        latest_period=periods[-1] if periods else "",
    )
