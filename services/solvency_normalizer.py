from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .solvency_table_extractor import ExtractedTable
from .solvency_company_identity import apply_company_identities, resolve_company_identity
from .solvency_navigation import apply_navigation_labels


STANDARD_COLUMNS = [
    "公司", "原始公司名称", "标准公司名称", "公司统一编码", "公司类型", "同业分类",
    "报告类型", "报告年度", "报告季度", "报告期", "披露日期",
    "一级模块", "二级模块", "行次", "指标编码", "指标名称", "期间口径",
    "数值", "单位", "数据类型", "是否预测", "来源页码", "原始披露值", "备注",
    "来源类型", "指标属性", "来源文件", "来源工作表", "导入批次", "计算逻辑",
]

LIFE_COMPANY_TYPES = ("寿险", "健康险", "养老险")
NON_LIFE_COMPANY_TYPES = ("财险",)

_COMPANY_TYPE_ALIASES = {
    "寿险": "寿险",
    "健康": "健康险",
    "健康险": "健康险",
    "养老": "养老险",
    "养老险": "养老险",
    "财险": "财险",
    "财产险": "财险",
    "财产保险": "财险",
    "非寿险": "财险",
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
    "本季度末数", "上季度末数", "下季度末预测数", "下季度预测数",
    "基本情景下的下季度预测数", "本季度数", "上季度数",
    "本年累计数", "期末数", "期初数", "未来3个月", "未来12个月",
    "账面价值", "非认可", "认可价值",
]
PERIOD_ALIASES = {
    "下季度预测数": "下季度末预测数",
    "基本情景下的下季度预测数": "下季度末预测数",
}

TABLE_ALLOWED_CODES = {
    "SOLVENCY_MAIN": {
        "RECOGNIZED_ASSETS",
        "RECOGNIZED_LIABILITIES",
        "ACTUAL_CAPITAL",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "ANC_T1_CAPITAL",
        "ANC_T2_CAPITAL",
        "MINIMUM_CAPITAL",
        "QUANT_RISK_CAPITAL",
        "CONTROL_RISK_CAPITAL",
        "ADDITIONAL_CAPITAL",
        "CORE_SOLVENCY_SURPLUS",
        "COMBINED_SOLVENCY_SURPLUS",
        "CORE_SOLVENCY_RATIO",
        "COMBINED_SOLVENCY_RATIO",
    },
    "ACTUAL_CAPITAL": {
        "ACTUAL_CAPITAL",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "ANC_T1_CAPITAL",
        "ANC_T2_CAPITAL",
        "POLICY_SURPLUS_CORE_T1",
        "POLICY_SURPLUS_CORE_T2",
        "POLICY_SURPLUS_ANC_T1",
        "POLICY_SURPLUS_ANC_T2",
    },
    "THREE_YEAR_INVESTMENT_RETURN": {
        "INVESTMENT_RETURN",
        "COMPREHENSIVE_INVESTMENT_RETURN",
    },
}


@dataclass(frozen=True)
class MetricMatchDecision:
    metric: pd.Series | None
    score: float
    runner_up_score: float
    ambiguous: bool
    reason: str
    candidate_name: str = ""
    runner_up_name: str = ""


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


def normalize_company_type(
    company_type: str,
    allowed_company_types: Sequence[str] | None = None,
) -> str:
    raw = str(company_type or "").strip()
    normalized = _COMPANY_TYPE_ALIASES.get(raw, raw)
    supported_types = tuple(allowed_company_types or LIFE_COMPANY_TYPES)
    if normalized not in supported_types:
        supported = "、".join(supported_types)
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


def resolve_metric_match(
    label: str,
    taxonomy: pd.DataFrame,
    candidates=None,
    *,
    table_id: str = "",
    minimum_score: float = 0.72,
    ambiguity_margin: float = 0.08,
) -> MetricMatchDecision:
    normalized = _normalize_label(label)
    if not normalized:
        return MetricMatchDecision(None, 0.0, 0.0, False, "项目名称为空")
    candidates = candidates or _taxonomy_candidates(taxonomy)
    allowed_codes = TABLE_ALLOWED_CODES.get(table_id)
    if table_id == "OPERATING_METRICS":
        allowed_codes = set(
            taxonomy.loc[taxonomy["一级模块"] == "经营指标", "指标编码"].astype(str)
        )
    elif table_id == "MINIMUM_CAPITAL":
        allowed_codes = set(
            taxonomy.loc[taxonomy["一级模块"] == "最低资本", "指标编码"].astype(str)
        )

    ranked: list[tuple[float, bool, int]] = []
    for index, aliases in candidates:
        code = str(taxonomy.loc[index].get("指标编码", "")).strip()
        if allowed_codes is not None and code not in allowed_codes:
            continue
        candidate_score = 0.0
        exact = False
        for alias in aliases:
            if not alias:
                continue
            if alias == normalized:
                score = 2.0
                exact = True
            elif alias in normalized or normalized in alias:
                score = min(len(alias), len(normalized)) / max(len(alias), len(normalized)) + 0.5
            else:
                score = SequenceMatcher(None, normalized, alias).ratio()
            candidate_score = max(candidate_score, score)
        if candidate_score:
            ranked.append((candidate_score, exact, index))
    ranked.sort(key=lambda item: (-item[0], -int(item[1]), item[2]))
    if not ranked:
        return MetricMatchDecision(None, 0.0, 0.0, False, "目标表范围内没有候选指标")

    best_score, exact, best_index = ranked[0]
    runner_score = ranked[1][0] if len(ranked) > 1 else 0.0
    runner_exact = ranked[1][1] if len(ranked) > 1 else False
    best_name = str(taxonomy.loc[best_index].get("指标名称", ""))
    runner_name = (
        str(taxonomy.loc[ranked[1][2]].get("指标名称", ""))
        if len(ranked) > 1 else ""
    )
    if best_score < minimum_score:
        return MetricMatchDecision(
            None,
            best_score,
            runner_score,
            False,
            f"最佳匹配得分低于{minimum_score:.2f}",
            best_name,
            runner_name,
        )
    ambiguous = (
        len(ranked) > 1
        and runner_score >= minimum_score
        and best_score - runner_score < ambiguity_margin
        and (not exact or runner_exact)
    )
    if ambiguous:
        return MetricMatchDecision(
            None,
            best_score,
            runner_score,
            True,
            f"前两候选分差小于{ambiguity_margin:.2f}",
            best_name,
            runner_name,
        )
    return MetricMatchDecision(
        taxonomy.loc[best_index],
        best_score,
        runner_score,
        False,
        "精确别名匹配" if exact else "唯一高置信度匹配",
        best_name,
        runner_name,
    )


def match_metric(label: str, taxonomy: pd.DataFrame, candidates=None):
    """Backward-compatible high-confidence metric matcher."""
    return resolve_metric_match(label, taxonomy, candidates).metric


def _period_header(row: list[str], index: int) -> str:
    current = str(row[index]).strip() if index < len(row) else ""
    for term in PERIOD_TERMS:
        if term in current:
            return PERIOD_ALIASES.get(term, term)
    return current or f"列{index + 1}"


def normalize_tables(
    tables: list[ExtractedTable],
    taxonomy: pd.DataFrame,
    metadata: dict,
    company_type: str = "寿险",
    report_profile_id: str = "LIFE_SOLVENCY",
    diagnostics: list[dict] | None = None,
    allowed_company_types: Sequence[str] | None = None,
    peer_group: str = "",
) -> pd.DataFrame:
    company_type = normalize_company_type(company_type, allowed_company_types)
    company_identity = resolve_company_identity(
        metadata.get("公司", ""),
        fallback_company_type=company_type,
    )
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
            decision = resolve_metric_match(
                label,
                taxonomy,
                candidates,
                table_id=table.table_id,
            )
            metric = decision.metric
            if metric is None:
                if diagnostics is not None and label:
                    diagnostics.append({
                        "目标表": table.table_name,
                        "来源页码": "、".join(map(str, table.source_pages or [table.page])),
                        "原始项目": label,
                        "状态": "歧义" if decision.ambiguous else "未匹配",
                        "原因": decision.reason,
                        "最佳候选": decision.candidate_name,
                        "最佳得分": round(decision.score, 3),
                        "第二候选": decision.runner_up_name,
                        "第二得分": round(decision.runner_up_score, 3),
                    })
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
                    "公司": company_identity.standard_name,
                    "原始公司名称": company_identity.original_name,
                    "标准公司名称": company_identity.standard_name,
                    "公司统一编码": company_identity.company_code,
                    "公司类型": company_identity.company_type,
                    "同业分类": peer_group,
                    "报告类型": report_profile_id,
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
                    "来源类型": "报告提取",
                    "指标属性": "披露",
                    "来源文件": metadata.get("来源文件", ""),
                    "来源工作表": table.table_name,
                    "导入批次": metadata.get("导入批次", ""),
                    "计算逻辑": "",
                })
    return apply_navigation_labels(pd.DataFrame(records, columns=STANDARD_COLUMNS))


def standardize_uploaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    return apply_navigation_labels(result[STANDARD_COLUMNS])


def upgrade_standard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Upgrade a saved/session DataFrame to the current standard schema."""
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return apply_company_identities(standardize_uploaded_frame(frame))

