from __future__ import annotations

import re
from typing import Sequence


THREE_YEAR_RETURN_TABLE_ID = "THREE_YEAR_INVESTMENT_RETURN"
THREE_YEAR_RETURN_HEADER = ["项目", "数值"]
THREE_YEAR_RETURN_LABELS = (
    "近三年平均投资收益率",
    "近三年平均综合投资收益率",
)

_VALUE_PATTERN = r"[+\-－−]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?\s*[％%]?"
_ORDINARY_LABEL_PATTERN = r"(?:近三年)?(?:平均)?(?<!综合)投资收益率"
_COMPREHENSIVE_LABEL_PATTERN = r"(?:近三年)?(?:平均)?综合投资收益率"


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _clean_value(value: str) -> str:
    return (
        re.sub(r"\s+", "", value)
        .replace("％", "%")
        .replace("，", ",")
        .replace("－", "-")
        .replace("−", "-")
    )


def _extract_after_label(text: str, label_pattern: str) -> str:
    match = re.search(
        rf"(?:{label_pattern})\s*(?:为|是|[:：=])?\s*({_VALUE_PATTERN})",
        text,
        flags=re.I,
    )
    return _clean_value(match.group(1)) if match else ""


def _extract_from_rows(
    rows: Sequence[Sequence[object]],
    label_pattern: str,
) -> str:
    for row in rows:
        row_text = " ".join(str(cell or "") for cell in row)
        value = _extract_after_label(row_text, label_pattern)
        if value:
            return value

    if rows:
        header = [str(cell or "") for cell in rows[0]]
        for column_index, cell in enumerate(header):
            if not re.search(label_pattern, _compact(cell), flags=re.I):
                continue
            for row in rows[1:]:
                if column_index >= len(row):
                    continue
                value_match = re.search(_VALUE_PATTERN, str(row[column_index] or ""))
                if value_match:
                    return _clean_value(value_match.group(0))

    full_text = "\n".join(
        " ".join(str(cell or "") for cell in row)
        for row in rows
    )
    return _extract_after_label(full_text, label_pattern)


def normalize_three_year_return_rows(
    table_id: str,
    rows: list[list[str]],
) -> tuple[list[list[str]], str]:
    """Convert prose or irregular disclosure into the canonical two-row table."""
    if table_id != THREE_YEAR_RETURN_TABLE_ID or not rows:
        return rows, ""

    comprehensive = _extract_from_rows(rows, _COMPREHENSIVE_LABEL_PATTERN)
    ordinary = _extract_from_rows(rows, _ORDINARY_LABEL_PATTERN)
    if not ordinary or not comprehensive:
        return rows, ""

    normalized = [
        list(THREE_YEAR_RETURN_HEADER),
        [THREE_YEAR_RETURN_LABELS[0], ordinary],
        [THREE_YEAR_RETURN_LABELS[1], comprehensive],
    ]
    if rows == normalized:
        return rows, ""
    return normalized, "已将近三年投资收益率的句式或非标准版式归一化为两行标准数据"
