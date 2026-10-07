from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .solvency_table_extractor import ExtractedTable
from .solvency_company_identity import (
    apply_company_identities,
    resolve_company_identity,
    resolve_peer_group,
)
from .solvency_disclosure_normalizer import normalize_three_year_return_rows_core
from .solvency_navigation import apply_navigation_labels
from .solvency_filing_catalog import filing_codes, filing_row_code
from .solvency_disclosure_policy import is_dash_value, is_disclosed_zero


STANDARD_COLUMNS = [
    "公司", "原始公司名称", "标准公司名称", "公司统一编码", "公司类型", "同业分类",
    "报告类型", "报告年度", "报告季度", "报告期", "披露日期",
    "一级模块", "二级模块", "行次", "指标编码", "指标名称", "期间口径",
    "数值", "单位", "数据类型", "是否预测", "来源页码", "原始披露值", "备注",
    "来源类型", "指标属性", "来源文件", "来源工作表", "导入批次", "计算逻辑", "披露状态",
]

# User-facing STEP3/STEP5 exchange schema.  The wider STANDARD_COLUMNS schema
# remains the internal source of truth so identity, provenance and profile
# fields stay available to validation and downstream report logic.
NARROW_TABLE_COLUMNS = [
    "公司",
    "同业分类",
    "报告期",
    "一级模块",
    "二级模块",
    "指标编码",
    "指标名称",
    "期间口径",
    "数值",
    "单位",
    "数据类型",
    "来源类型",
    "指标属性",
    "计算逻辑",
    "披露状态",
]

LIFE_COMPANY_TYPES = ("寿险", "健康险", "养老险")
NON_LIFE_COMPANY_TYPES = ("财险",)
CURRENT_QUARTER_BALANCE_METRIC_CODES = frozenset({
    "TOTAL_ASSETS",
    "INSURANCE_CONTRACT_LIABILITY",
})

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
    r"[（(]\s*(?:单位\s*[:：]?\s*)?"
    r"(亿元|万元|千元|元|%|％|百分比|百分点|人|户|件|次|级)\s*[）)]"
)

_PERIOD_UNIT_GROUP_PATTERN = re.compile(
    r"[（(]\s*(?:单位\s*[:：]?\s*)?"
    r"(?:(?:亿元|万元|千元|元|%|％|百分比|百分点|人|户|件|次|级)"
    r"(?:\s*[,，、/＋+]\s*)?)+\s*[）)]"
)

_DEFERRED_TAX_ASSET_LABEL = "递延所得税资产（由经营性亏损引起的递延所得税资产除外）"


PERIOD_TERMS = [
    "基本情景下的下季度末预测", "基本情景下的下季度预测",
    "下季度末预测", "下季度预测", "本季度末", "上季度末",
    "本季度末数", "上季度末数", "下季度末预测数", "下季度预测数",
    "基本情景下的下季度预测数", "本季度数", "上季度数",
    "本年度累计数", "本年累计数", "期末数", "期初数", "未来3个月", "未来12个月",
    "账面价值", "非认可", "认可价值",
]
PERIOD_ALIASES = {
    "本季度末": "本季度末数",
    "上季度末": "上季度末数",
    "下季度末预测": "下季度末预测数",
    "下季度预测": "下季度末预测数",
    "基本情景下的下季度末预测": "下季度末预测数",
    "基本情景下的下季度预测": "下季度末预测数",
    "下季度预测数": "下季度末预测数",
    "基本情景下的下季度预测数": "下季度末预测数",
    "本年度累计数": "本年累计数",
}

SOURCE_AGGREGATE_DEDUPE_CODES = {"ACTUAL_CAPITAL"}
MINIMUM_CAPITAL_EXACT_ONLY_CODES = {
    "QUANT_RISK_CAPITAL",
    "CONTROL_RISK_CAPITAL",
    "ADDITIONAL_CAPITAL",
    "MINIMUM_CAPITAL",
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
        "RECOGNIZED_ASSETS",
        "RECOGNIZED_LIABILITIES",
        "ACTUAL_CAPITAL",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "ANC_T1_CAPITAL",
        "ANC_T2_CAPITAL",
        "NET_ASSETS",
        "POLICY_SURPLUS_CORE_T1",
        "POLICY_SURPLUS_CORE_T2",
        "POLICY_SURPLUS_ANC_T1",
        "POLICY_SURPLUS_ANC_T2",
        "FINANCIAL_STATEMENT_ASSETS",
        "FINANCIAL_STATEMENT_LIABILITIES",
        "FINANCIAL_STATEMENT_NET_ASSETS",
        "NET_ASSET_ADJUSTMENT",
        "NON_RECOGNIZED_ASSET_BOOK_VALUE",
        "LONG_TERM_EQUITY_VALUATION_DIFFERENCE",
        "DEFERRED_TAX_ASSET_ADJUSTMENT",
        "OTHER_CORE_T1_ADJUSTMENT",
        "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
        "AGRICULTURAL_CATASTROPHE_RISK_RESERVE",
        "QUALIFYING_CORE_T1_LIABILITY_CAPITAL",
        "CORE_T2_PREFERRED_SHARES",
        "OTHER_CORE_T2_CAPITAL",
        "CORE_T2_EXCESS_DEDUCTION",
        "ANC_T1_SUBORDINATED_TERM_DEBT",
        "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS",
        "ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT",
        "ANC_T1_DEFERRED_TAX_ASSET",
        "ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE",
        "OTHER_ANC_T1_CAPITAL",
        "ANC_T1_EXCESS_DEDUCTION",
        "EMERGENCY_OTHER_ANC_T2_CAPITAL",
        "ANC_T2_EXCESS_DEDUCTION",
    },
    "THREE_YEAR_INVESTMENT_RETURN": {
        "INVESTMENT_RETURN",
        "COMPREHENSIVE_INVESTMENT_RETURN",
    },
    "RECOGNIZED_ASSETS": {
        "RECOGNIZED_ASSETS",
        "CASH_LIQUID_ASSETS",
        "INVESTMENT_ASSETS",
        "REINSURANCE_ASSETS",
        "SUBSIDIARY_JV_ASSOCIATE_EQUITY",
        "RECEIVABLES_AND_PREPAYMENTS",
        "FIXED_ASSETS",
        "LAND_USE_RIGHTS",
        "SEPARATE_ACCOUNT_ASSETS",
        "OTHER_RECOGNIZED_ASSETS",
    },
    "RECOGNIZED_LIABILITIES": {
        "RECOGNIZED_LIABILITIES",
        "RESERVE_LIABILITIES",
        "UNEARNED_PREMIUM_RESERVE",
        "LIFE_UNEARNED_PREMIUM_RESERVE",
        "NON_LIFE_UNEARNED_PREMIUM_RESERVE",
        "OUTSTANDING_CLAIMS_RESERVE",
        "IBNR_RESERVE",
        "FINANCIAL_LIABILITIES",
        "SECURITIES_SOLD_UNDER_REPURCHASE",
        "POLICYHOLDER_DEPOSITS_INVESTMENTS",
        "DERIVATIVE_FINANCIAL_LIABILITIES",
        "OTHER_FINANCIAL_LIABILITIES",
        "PAYABLES_AND_ADVANCES",
        "POLICY_DIVIDENDS_PAYABLE",
        "CLAIMS_PAYABLE",
        "PREMIUMS_RECEIVED_IN_ADVANCE",
        "REINSURANCE_PAYABLES",
        "COMMISSIONS_PAYABLE",
        "EMPLOYEE_BENEFITS_PAYABLE",
        "TAXES_PAYABLE",
        "REINSURANCE_DEPOSITS_RECEIVED",
        "OTHER_PAYABLES_AND_ADVANCES",
        "PROVISIONS",
        "SEPARATE_ACCOUNT_LIABILITY",
        "CAPITAL_LIABILITIES",
        "OTHER_RECOGNIZED_LIABILITIES",
        "DEFERRED_TAX_LIABILITIES",
        "CASH_VALUE_GUARANTEE",
        "INCOME_TAX_RESERVE",
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


def _normalize_operating_label(value: str) -> str:
    text = re.sub(
        r"^\s*(?:[（(][一二三四五六七八九十百\d]+[）)]|\d+(?:\.\d+)*[.、])\s*",
        "",
        str(value or ""),
    )
    return _normalize_label(text)


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
    checklist_code = filing_row_code(table_id, label)
    checklist_metric = taxonomy[taxonomy['指标编码'].eq(checklist_code)]
    if not checklist_metric.empty:
        return MetricMatchDecision(checklist_metric.iloc[0], 2.0, 0.0, False, '按填报清单及来源表精确匹配')
    if table_id == "MINIMUM_CAPITAL" and re.search(
        r"未考虑.*特征系数|特征系数.*调整前",
        str(label or ""),
    ):
        return MetricMatchDecision(
            None,
            0.0,
            0.0,
            False,
            "特征系数调整前金额不等同于量化风险最低资本",
        )
    exact_only = table_id == "OPERATING_METRICS"
    normalized = (
        _normalize_operating_label(label)
        if exact_only
        else _normalize_label(label)
    )
    if not normalized:
        return MetricMatchDecision(None, 0.0, 0.0, False, "项目名称为空")
    candidates = candidates or _taxonomy_candidates(taxonomy)
    allowed_codes = TABLE_ALLOWED_CODES.get(table_id)
    if allowed_codes is not None:
        allowed_codes = set(allowed_codes) | filing_codes(table_id)
    if table_id == "OPERATING_METRICS":
        allowed_codes = set(
            taxonomy.loc[taxonomy["一级模块"] == "经营指标", "指标编码"].astype(str)
        )
    elif table_id == "MINIMUM_CAPITAL":
        allowed_codes = set(
            taxonomy.loc[taxonomy["一级模块"] == "最低资本", "指标编码"].astype(str)
        )
        allowed_codes.add("MINIMUM_CAPITAL")
    if allowed_codes is not None:
        allowed_codes |= filing_codes(table_id)

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
            elif exact_only:
                score = 0.0
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
    best_code = str(taxonomy.loc[best_index].get("指标编码", "")).strip()
    runner_name = (
        str(taxonomy.loc[ranked[1][2]].get("指标名称", ""))
        if len(ranked) > 1 else ""
    )
    if (
        table_id == "MINIMUM_CAPITAL"
        and best_code in MINIMUM_CAPITAL_EXACT_ONLY_CODES
        and not exact
    ):
        return MetricMatchDecision(
            None,
            best_score,
            runner_score,
            False,
            "最低资本汇总指标仅接受精确名称或正式别名",
            best_name,
            runner_name,
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


def _period_header(row: list[str], index: int, *, table_id: str = "") -> str:
    if table_id == "THREE_YEAR_INVESTMENT_RETURN":
        return "近三年平均"
    current = str(row[index]).strip() if index < len(row) else ""
    compact = re.sub(r"\s+", "", current).replace("(", "（").replace(")", "）")
    compact = _PERIOD_UNIT_GROUP_PATTERN.sub("", compact).strip("/：:")
    if table_id == "OPERATING_METRICS":
        if any(term in compact for term in ("本季度（末）数", "本季度（末）", "本季度数")):
            return "本季度数"
        if any(term in compact for term in (
            "本年累计（末）数", "本年累计（末）",
            "本年度累计（末）数", "本年度累计（末）",
            "本年累计数", "本年度累计数",
        )):
            return "本年累计数"
    for term in PERIOD_TERMS:
        if term in compact:
            return PERIOD_ALIASES.get(term, term)
    return compact or f"列{index + 1}"


def _recognized_assets_subcolumn(header: str) -> str:
    """返回认可资产表两级表头列的二级口径（账面价值/非认可价值/认可价值）。"""
    compact = re.sub(r"\s+", "", str(header or ""))
    if compact.endswith("/账面价值"):
        return "账面价值"
    if compact.endswith("/非认可价值") or compact.endswith("/非认可"):
        return "非认可价值"
    if compact.endswith("/认可价值"):
        return "认可价值"
    return ""


def _normalize_canonical_table(table, taxonomy, metadata, identity, peer_group, report_profile_id):
    """Keep VLM's canonical IDs and disclosure states; never fuzzy-rematch them."""
    by_code = taxonomy.drop_duplicates('指标编码').set_index('指标编码')
    result = []
    for raw in table.metric_records:
        code = str(raw.get('指标编码', ''))
        if code not in by_code.index:
            continue
        metric = by_code.loc[code]
        status = str(raw.get('状态', 'not_disclosed'))
        dtype = str(metric.get('数据类型', '金额'))
        if status in {'found', 'disclosed_na', 'disclosed_zero'} and dtype not in {'文本', '评级', '布尔'} and is_disclosed_zero(raw.get('原始值')):
            status = 'disclosed_zero'
        value = np.nan
        if status == 'found':
            value = raw.get('原始值', '') if dtype in {'文本', '评级', '布尔'} else parse_numeric(raw.get('标准数值'))
        elif status == 'disclosed_zero':
            value = 0.0
        period = str(raw.get('期间口径') or '本季度末数')
        if str(raw.get('指标语义键', '')).endswith(':THREE_YEAR_AVERAGE'):
            period = '近三年平均'
        elif (
            code in CURRENT_QUARTER_BALANCE_METRIC_CODES
            or table.table_id == 'OPERATING_METRICS'
        ) and any(
            term in period for term in ('本季度数', '本季度（末）数', '本季度(末)数', '当季数')
        ) and '累计' not in period:
            period = '本季度数'
        elif any(term in period for term in ('本季度', '期末', '本期')) and not any(term in period for term in ('上季度', '期初', '预测', '累计')):
            period = '本季度末数'
        state = {'found':'已披露', 'disclosed_zero':'已披露为0', 'disclosed_na':'不适用', 'not_disclosed':'未披露'}.get(status, '未披露')
        result.append({
            '公司': identity.standard_name, '原始公司名称': identity.original_name,
            '标准公司名称': identity.standard_name, '公司统一编码': identity.company_code,
            '公司类型': identity.company_type, '同业分类': peer_group, '报告类型': report_profile_id,
            '报告年度': metadata.get('报告年度'), '报告季度': metadata.get('报告季度', ''),
            '报告期': metadata.get('报告期', ''), '披露日期': metadata.get('披露日期', ''),
            '一级模块': metric.get('一级模块', ''), '二级模块': metric.get('二级模块', ''),
            '行次': '', '指标编码': code, '指标名称': metric['指标名称'], '期间口径': period,
            '数值': value, '单位': metric.get('标准单位', ''), '数据类型': dtype,
            '是否预测': '是' if '预测' in period else '否',
            '来源页码': raw.get('物理页码') or '',
            '原始披露值': raw.get('原始值') if raw.get('原始值') is not None else '',
            '备注': ('原始披露为横杠，按填报规则转换为0。' if status == 'disclosed_zero' and is_dash_value(raw.get('原始值')) else '原始披露为数值0。' if status == 'disclosed_zero' else '')
                    + str(raw.get('证据原文') or ''),
            '来源类型': '报告提取', '指标属性': '披露', '来源文件': metadata.get('来源文件', ''),
            '来源工作表': table.table_name, '导入批次': metadata.get('导入批次', ''),
            '计算逻辑': '', '披露状态': state,
        })
    return result


def normalize_tables(
    tables: list[ExtractedTable],
    taxonomy: pd.DataFrame,
    metadata: dict,
    company_type: str = "寿险",
    report_profile_id: str = "LIFE_SOLVENCY",
    diagnostics: list[dict] | None = None,
    allowed_company_types: Sequence[str] | None = None,
    peer_group: str = "",
    peer_group_map: Mapping[str, str] | None = None,
    default_peer_group: str = "",
) -> pd.DataFrame:
    company_type = normalize_company_type(company_type, allowed_company_types)
    company_identity = resolve_company_identity(
        metadata.get("公司", ""),
        fallback_company_type=company_type,
    )
    resolved_peer_group = str(peer_group or "").strip()
    if not resolved_peer_group and peer_group_map:
        resolved_peer_group = resolve_peer_group(
            metadata.get("公司", ""),
            peer_group_map,
            fallback_peer_group=default_peer_group,
        )
    records: list[dict] = []
    candidates = _taxonomy_candidates(taxonomy)
    for table in tables:
        if getattr(table, 'metric_records', None):
            records.extend(_normalize_canonical_table(
                table, taxonomy, metadata, company_identity, resolved_peer_group, report_profile_id,
            ))
            continue
        rows = table.rows
        if table.table_id == "THREE_YEAR_INVESTMENT_RETURN":
            rows, _ = normalize_three_year_return_rows_core(table.table_id, rows)
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

        deferred_tax_signs: set[int] = set()
        if table.table_id == "ACTUAL_CAPITAL":
            target_label = _normalize_label(_DEFERRED_TAX_ASSET_LABEL)
            for source_row in rows[data_start:]:
                padded = source_row + [""] * (max_width - len(source_row))
                source_row_number = (
                    padded[0]
                    if re.fullmatch(r"\d+(?:\.\d+)*\*?", padded[0].strip())
                    else ""
                )
                source_label_index = 1 if source_row_number and max_width > 1 else 0
                source_label = padded[source_label_index].strip()
                if source_row_number or _normalize_label(source_label) != target_label:
                    continue
                for raw_value in padded[source_label_index + 1:]:
                    numeric_value = parse_numeric(raw_value)
                    if pd.isna(numeric_value) or numeric_value == 0:
                        continue
                    deferred_tax_signs.add(1 if numeric_value > 0 else -1)

        for row in rows[data_start:]:
            row = row + [""] * (max_width - len(row))
            row_number = row[0] if re.fullmatch(r"\d+(?:\.\d+)*\*?", row[0].strip()) else ""
            label_index = 1 if row_number and max_width > 1 else 0
            label = row[label_index].strip()
            decision = None
            contextual_code = filing_row_code(table.table_id, label, row_number)
            contextual_metric = taxonomy[taxonomy['指标编码'].eq(contextual_code)]
            if not contextual_metric.empty:
                decision = MetricMatchDecision(contextual_metric.iloc[0], 2.0, 0.0, False, '按来源表和清单行次匹配')
            elif (
                table.table_id == "MINIMUM_CAPITAL"
                and row_number.startswith("3.")
                and "附加资本" in label
            ):
                decision = MetricMatchDecision(
                    None,
                    0.0,
                    0.0,
                    False,
                    "附加资本明细行不作为附加资本合计重复入表",
                )
            elif table.table_id == "ACTUAL_CAPITAL" and row_number:
                contextual = resolve_metric_match(
                    f"{row_number} {label}",
                    taxonomy,
                    candidates,
                    table_id=table.table_id,
                )
                if contextual.metric is not None and contextual.score == 2.0:
                    decision = contextual
            elif (
                table.table_id == "ACTUAL_CAPITAL"
                and deferred_tax_signs == {-1, 1}
                and _normalize_label(label) == _normalize_label(_DEFERRED_TAX_ASSET_LABEL)
            ):
                row_values = [parse_numeric(value) for value in row[label_index + 1:]]
                numeric_values = [value for value in row_values if not pd.isna(value) and value != 0]
                if numeric_values and all(value > 0 for value in numeric_values):
                    ancillary = taxonomy.loc[
                        taxonomy["指标编码"].astype(str).eq("ANC_T1_DEFERRED_TAX_ASSET")
                    ]
                    if not ancillary.empty:
                        decision = MetricMatchDecision(
                            ancillary.iloc[-1],
                            2.0,
                            0.0,
                            False,
                            "同表存在正负两组同名递延所得税资产，正值组按附属一级资本识别",
                            str(ancillary.iloc[-1].get("指标名称", "")),
                            "",
                        )
            if decision is None:
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
                if table.table_id in {"RECOGNIZED_ASSETS", "RECOGNIZED_LIABILITIES"} and (
                    _recognized_assets_subcolumn(headers[column_index])
                    in ("账面价值", "非认可价值")
                ):
                    # S03/S04主要指标仅取“认可价值”口径，其他价值列仅用于核对。
                    continue
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
                    marker = str(raw_value or "").strip()
                    explicit_zero = False
                    if pd.isna(numeric_value):
                        if is_dash_value(marker):
                            value = 0.0
                            explicit_zero = True
                            note = "原始披露为横线，按填报规则转换为0"
                        elif marker in {'不适用', '<不适用>'}:
                            value = np.nan
                            note = '原始披露明确为不适用，保留不适用状态'
                        else:
                            continue
                    else:
                        value = numeric_value
                    source_unit = _source_unit(
                        table,
                        label,
                        headers[column_index],
                        target_unit,
                    )
                    if explicit_zero:
                        pass
                    elif target_unit in _CURRENCY_UNIT_IN_YUAN and not source_unit:
                        note = "未识别原始单位，数值未换算"
                    else:
                        value, note = _convert_unit(value, source_unit, target_unit)
                period = _period_header(headers, column_index, table_id=table.table_id)
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
                    "同业分类": resolved_peer_group,
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
                    "披露状态": ('不适用' if pd.isna(value) else '已披露为0' if data_type not in {'文本', '评级', '布尔'} and value == 0 else '已披露'),
                })
    result = pd.DataFrame(records, columns=STANDARD_COLUMNS)
    if not result.empty:
        dedupe_mask = result["指标编码"].astype(str).isin(SOURCE_AGGREGATE_DEDUPE_CODES)
        dedupe_key = [
            "公司统一编码",
            "报告期",
            "来源文件",
            "来源工作表",
            "指标编码",
            "期间口径",
            "数值",
        ]
        deduped = result.loc[dedupe_mask].drop_duplicates(subset=dedupe_key, keep="last")
        result = pd.concat([result.loc[~dedupe_mask], deduped], ignore_index=True).sort_index()
    return apply_navigation_labels(result.reindex(columns=STANDARD_COLUMNS))


def standardize_uploaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    # Display/export uses a literal marker; arithmetic always works with NaN.
    missing_value = result['数值'].astype(str).str.strip().eq('未披露')
    result['披露状态'] = result['披露状态'].fillna('').astype(object)
    result.loc[missing_value, '数值'] = np.nan
    result.loc[missing_value, '披露状态'] = '未披露'
    blank = result['披露状态'].fillna('').astype(str).str.strip().eq('')
    present = result['数值'].notna() & result['数值'].astype(str).str.strip().ne('')
    calculated = result['来源类型'].eq('系统计算')
    result.loc[blank & present & ~calculated, '披露状态'] = '已披露'
    result.loc[blank & present & calculated, '披露状态'] = '已计算'
    return apply_navigation_labels(result[STANDARD_COLUMNS])


def narrow_table_view(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return the compact STEP3/STEP5 display and workbook exchange schema."""
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(columns=NARROW_TABLE_COLUMNS)
    result = frame.copy()
    for column in NARROW_TABLE_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    missing = result['披露状态'].eq('未披露') & (
        result['数值'].isna() | result['数值'].astype(str).str.strip().isin({'', '未披露'}))
    result['数值'] = result['数值'].astype(object)
    result.loc[missing, '数值'] = '未披露'
    return result.reindex(columns=NARROW_TABLE_COLUMNS)


def upgrade_standard_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Upgrade a saved/session DataFrame to the current standard schema."""
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return apply_company_identities(standardize_uploaded_frame(frame))
