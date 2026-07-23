from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from .solvency_table_extractor import ExtractedTable


STANDARD_COLUMNS = [
    "公司", "公司类型", "报告年度", "报告季度", "报告期", "披露日期",
    "一级模块", "二级模块", "行次", "指标编码", "指标名称", "期间口径",
    "数值", "单位", "数据类型", "是否预测", "来源页码", "原始披露值", "备注",
]

LIFE_COMPANY_TYPES = ("寿险", "健康险", "养老险")

_COMPANY_TYPE_ALIASES = {
    "寿险": "寿险",
    "健康": "健康险",
    "健康险": "健康险",
    "养老": "养老险",
    "养老险": "养老险",
}

_CURRENCY_UNIT_IN_YUAN = {
    "元": 1.0,
    "千元": 1_000.0,
    "万元": 10_000.0,
    "亿元": 100_000_000.0,
}

_INLINE_UNIT_PATTERN = re.compile(
    r"[（(]\s*(亿元|万元|千元|元|%|％|百分比|百分点|人|户|件|次|级)\s*[）)]"
)


PERIOD_TERMS = [
    "本季度末数", "上季度末数", "下季度末预测数", "本季度数", "上季度数",
    "本年累计数", "期末数", "期初数", "未来3个月", "未来12个月",
    "账面价值", "非认可", "认可价值",
]


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="指标字典", header=2)
    frame = frame.fillna("")
    for column in frame.columns:
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def parse_numeric(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip().replace(",", "").replace("，", "")
    if text in {"", "-", "—", "--", "不适用", "未披露"}:
        return np.nan
    text = text.replace("％", "%")
    if text.endswith("%"):
        text = text[:-1].strip()
    if (text.startswith("(") and text.endswith(")")) or (text.startswith("（") and text.endswith("）")):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        return np.nan


def _normalize_label(value: str) -> str:
    text = _INLINE_UNIT_PATTERN.sub("", str(value or ""))
    return re.sub(r"[\s：:（）()、，,。·—\-_/]", "", text)


def normalize_company_type(company_type: str) -> str:
    normalized = _COMPANY_TYPE_ALIASES.get(str(company_type or "").strip())
    if normalized not in LIFE_COMPANY_TYPES:
        supported = "、".join(LIFE_COMPANY_TYPES)
        raise ValueError(f"公司类型仅支持：{supported}")
    return normalized


def _canonical_unit(value: str) -> str:
    text = re.sub(r"[\s：:（）()]", "", str(value or "")).replace("％", "%")
    if "百分比" in text:
        return "%"
    if "百分点" in text:
        return "百分点"
    for unit in ("亿元", "万元", "千元", "元", "%", "人", "户", "件", "次", "级"):
        if text.endswith(unit):
            return unit
    return text


def _unit_from_text(value: str) -> str:
    matches = list(_INLINE_UNIT_PATTERN.finditer(str(value or "")))
    return _canonical_unit(matches[-1].group(1)) if matches else ""


def _select_unit(records, target_unit: str) -> str:
    units = list(dict.fromkeys(
        _canonical_unit(record.normalized_unit)
        for record in records
        if _canonical_unit(record.normalized_unit)
    ))
    if target_unit in _CURRENCY_UNIT_IN_YUAN:
        units = [unit for unit in units if unit in _CURRENCY_UNIT_IN_YUAN]
    elif target_unit == "%":
        units = [unit for unit in units if unit == "%"]
    return units[0] if len(units) == 1 else ""


def _source_unit(
    table: ExtractedTable,
    label: str,
    header: str,
    target_unit: str,
) -> str:
    normalized_label = _normalize_label(label)
    header_parts = [part for part in str(header or "").split("/") if part]
    normalized_headers = {_normalize_label(part) for part in header_parts}

    row_records = [
        record
        for record in table.unit_records
        if record.scope == "行级" and _normalize_label(record.target) == normalized_label
    ]
    selected = _select_unit(row_records, target_unit)
    if selected:
        return selected
    inline_row_unit = _unit_from_text(label)
    if inline_row_unit:
        return inline_row_unit

    column_records = [
        record
        for record in table.unit_records
        if record.scope == "列级" and _normalize_label(record.target) in normalized_headers
    ]
    selected = _select_unit(column_records, target_unit)
    if selected:
        return selected
    for part in reversed(header_parts):
        inline_column_unit = _unit_from_text(part)
        if inline_column_unit:
            return inline_column_unit

    table_records = [record for record in table.unit_records if record.scope == "表级"]
    return _select_unit(table_records, target_unit)


def _convert_unit(value: float, source_unit: str, target_unit: str) -> tuple[float, str]:
    if (
        source_unit not in _CURRENCY_UNIT_IN_YUAN
        or target_unit not in _CURRENCY_UNIT_IN_YUAN
        or source_unit == target_unit
    ):
        return value, ""
    converted = (
        value
        * _CURRENCY_UNIT_IN_YUAN[source_unit]
        / _CURRENCY_UNIT_IN_YUAN[target_unit]
    )
    return converted, f"原单位：{source_unit}；已换算为{target_unit}"


def _taxonomy_candidates(taxonomy: pd.DataFrame) -> list[tuple[int, list[str]]]:
    candidates = []
    for index, row in taxonomy.iterrows():
        aliases = [row.get("指标名称", "")]
        aliases += [item.strip() for item in str(row.get("别名", "")).split("|") if item.strip()]
        candidates.append((index, [_normalize_label(alias) for alias in aliases if alias]))
    return candidates


def match_metric(label: str, taxonomy: pd.DataFrame, candidates=None):
    normalized = _normalize_label(label)
    if not normalized:
        return None
    candidates = candidates or _taxonomy_candidates(taxonomy)
    best_index, best_score = None, 0.0
    for index, aliases in candidates:
        for alias in aliases:
            if not alias:
                continue
            if alias in normalized or normalized in alias:
                score = min(len(alias), len(normalized)) / max(len(alias), len(normalized)) + 0.5
            else:
                score = SequenceMatcher(None, normalized, alias).ratio()
            if score > best_score:
                best_index, best_score = index, score
    return taxonomy.loc[best_index] if best_index is not None and best_score >= 0.72 else None


def _period_header(row: list[str], index: int) -> str:
    current = str(row[index]).strip() if index < len(row) else ""
    for term in PERIOD_TERMS:
        if term in current:
            return term
    return current or f"列{index + 1}"


def normalize_tables(
    tables: list[ExtractedTable],
    taxonomy: pd.DataFrame,
    metadata: dict,
    company_type: str = "寿险",
) -> pd.DataFrame:
    company_type = normalize_company_type(company_type)
    records: list[dict] = []
    candidates = _taxonomy_candidates(taxonomy)
    for table in tables:
        rows = table.rows
        if len(rows) < 2:
            continue
        second_header = rows[1] if len(rows) > 1 else []
        is_multiheader = any(
            term in "".join(second_header)
            for term in ("账面价值", "非认可", "认可价值")
        )
        header_rows = rows[:2] if is_multiheader else rows[:1]
        data_start = 2 if is_multiheader else 1
        if is_multiheader:
            filled_first_header = []
            last_header = ""
            for value in rows[0]:
                last_header = value or last_header
                filled_first_header.append(last_header)
            header_rows = [filled_first_header, rows[1]]
        max_width = max(len(row) for row in rows)
        headers = []
        for column_index in range(max_width):
            parts = []
            for header_row in header_rows:
                value = header_row[column_index] if column_index < len(header_row) else ""
                if value and value not in parts:
                    parts.append(value)
            headers.append("/".join(parts))

        for row in rows[data_start:]:
            row = row + [""] * (max_width - len(row))
            row_number = row[0] if re.fullmatch(r"\d+(?:\.\d+)*\*?", row[0].strip()) else ""
            label_index = 1 if row_number and max_width > 1 else 0
            label = row[label_index].strip()
            metric = match_metric(label, taxonomy, candidates)
            if metric is None:
                continue
            value_start = label_index + 1
            for column_index in range(value_start, max_width):
                raw_value = row[column_index]
                numeric_value = parse_numeric(raw_value)
                data_type = str(metric.get("数据类型", "金额"))
                target_unit = str(metric.get("标准单位", "")).strip()
                note = ""
                if data_type in {"文本", "评级", "布尔"}:
                    value = raw_value.strip()
                    if not value:
                        continue
                else:
                    if pd.isna(numeric_value):
                        continue
                    value = numeric_value
                    source_unit = _source_unit(
                        table,
                        label,
                        headers[column_index],
                        target_unit,
                    )
                    if target_unit in _CURRENCY_UNIT_IN_YUAN and not source_unit:
                        note = "未识别原始单位，数值未换算"
                    else:
                        value, note = _convert_unit(value, source_unit, target_unit)
                period = _period_header(headers, column_index)
                if period.endswith("/认可价值"):
                    period = period.removesuffix("/认可价值")
                elif period.endswith("/账面价值"):
                    period = period.removesuffix("/账面价值") + "-账面价值"
                elif period.endswith("/非认可"):
                    period = period.removesuffix("/非认可") + "-非认可"
                records.append({
                    "公司": metadata.get("公司", ""),
                    "公司类型": company_type,
                    "报告年度": metadata.get("报告年度"),
                    "报告季度": metadata.get("报告季度", ""),
                    "报告期": metadata.get("报告期", ""),
                    "披露日期": metadata.get("披露日期", ""),
                    "一级模块": metric.get("一级模块", table.table_name),
                    "二级模块": metric.get("二级模块", ""),
                    "行次": row_number,
                    "指标编码": metric.get("指标编码", ""),
                    "指标名称": metric.get("指标名称", label),
                    "期间口径": period,
                    "数值": value,
                    "单位": target_unit,
                    "数据类型": data_type,
                    "是否预测": "是" if "预测" in period else "否",
                    "来源页码": "、".join(map(str, table.source_pages)) if table.source_pages else table.page,
                    "原始披露值": raw_value,
                    "备注": note,
                })
    return pd.DataFrame(records, columns=STANDARD_COLUMNS)


def standardize_uploaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    return result[STANDARD_COLUMNS]

