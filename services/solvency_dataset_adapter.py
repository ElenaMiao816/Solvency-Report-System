from __future__ import annotations

import io
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .solvency_metric_registry import (
    DERIVED_METRICS,
    INDUSTRY_METRICS_BY_SOURCE_CODE,
)
from .solvency_normalizer import STANDARD_COLUMNS, parse_numeric, standardize_uploaded_frame
from .solvency_company_identity import apply_company_identities, resolve_company_identity


QUARTER_SHEET_PATTERN = re.compile(r"^(?P<year>20\d{2})(?P<quarter>Q[1-4])$", re.IGNORECASE)
REQUIRED_STANDARD_HEADERS = {"公司", "指标编码", "指标名称", "数值"}
IDENTIFIER_COLUMNS = {"分类", "公司"}
PERCENT_FROM_RATIO_CODES = {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}
INDUSTRY_TOTAL_COMPANY_CODE = "INDUSTRY_LIFE_TOTAL"
INDUSTRY_OUTPUT_CODES = (
    "COMBINED_SOLVENCY_RATIO",
    "CORE_SOLVENCY_RATIO",
    "COMBINED_SOLVENCY_SURPLUS",
    "CORE_SOLVENCY_SURPLUS",
    "RECOGNIZED_ASSETS",
    "RECOGNIZED_LIABILITIES",
    "ACTUAL_CAPITAL",
    "CORE_T1_CAPITAL",
    "CORE_T2_CAPITAL",
    "ANC_T1_CAPITAL",
    "ANC_T2_CAPITAL",
    "MINIMUM_CAPITAL",
    "TOTAL_ASSETS",
    "QUANT_RISK_CAPITAL",
    "INSURANCE_RISK_CAPITAL",
    "NON_LIFE_INSURANCE_RISK_CAPITAL",
    "MARKET_RISK_CAPITAL",
    "CREDIT_RISK_CAPITAL",
    "QUANT_RISK_DIVERSIFICATION_EFFECT",
    "CONTRACT_LOSS_ABSORPTION_EFFECT",
    "CONTROL_RISK_CAPITAL",
    "INSURANCE_CONTRACT_LIABILITY",
    "SEPARATE_ACCOUNT_LIABILITY",
    "POLICY_SURPLUS_CORE_T1",
    "POLICY_SURPLUS_CORE_T2",
    "POLICY_SURPLUS_ANC_T1",
    "POLICY_SURPLUS_ANC_T2",
    "FEATURE_FACTOR_IMPACT",
    "ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS",
    "MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES",
    "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",
    "RECOGNIZED_ASSETS_TO_ACTUAL_CAPITAL",
    "RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL",
)
INDUSTRY_RATIO_FORMULAS = {
    "COMBINED_SOLVENCY_RATIO": "行业实际资本合计/行业最低资本合计×100",
    "CORE_SOLVENCY_RATIO": "(行业核心一级资本合计+行业核心二级资本合计)/行业最低资本合计×100",
    "ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS": "行业实际资本合计/行业认可资产合计",
    "MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES": "行业最低资本合计/行业认可负债合计",
    "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES": "行业四类保单未来盈余合计/行业保险合同负债合计",
    "RECOGNIZED_ASSETS_TO_ACTUAL_CAPITAL": "行业认可资产合计/行业实际资本合计",
    "RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL": "行业认可资产合计/行业最低资本合计",
}


@dataclass
class ExternalConversionResult:
    data: pd.DataFrame
    sheet_summary: pd.DataFrame
    mapping_summary: pd.DataFrame
    logic_checks: pd.DataFrame
    warnings: list[str]


def canonical_metric_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("％", "%")
    return re.sub(r"\s+", "", text).strip()


def canonical_period_scope(value: Any) -> str:
    text = str(value or "").strip()
    if "预测" in text:
        return "下季度末预测数"
    if "上季度末" in text or "期初" in text:
        return "上季度末数"
    if "本季度末" in text or "期末" in text or "本季度（末）" in text:
        return "本季度末数"
    return text


def _number(value: Any) -> float | None:
    numeric = parse_numeric(value)
    if pd.isna(numeric) or not math.isfinite(float(numeric)):
        return None
    return float(numeric)


def _safe_sum(values: dict[str, Any], codes: tuple[str, ...]) -> float | None:
    items = [_number(values.get(code)) for code in codes]
    if any(item is None for item in items):
        return None
    return float(sum(items))


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0 if numerator == 0 else None
    return float(numerator / denominator)


def calculate_derived_values(values: dict[str, Any]) -> dict[str, Any]:
    """Calculate the CROSS derived fields from canonical base metric codes."""
    result: dict[str, Any] = {}
    get = lambda code: _number(result.get(code, values.get(code)))

    core_policy_surplus = _safe_sum(values, ("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2"))
    all_policy_surplus = _safe_sum(
        values,
        ("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2"),
    )
    quant_before_factor = _safe_sum(
        values,
        (
            "INSURANCE_RISK_CAPITAL",
            "NON_LIFE_INSURANCE_RISK_CAPITAL",
            "MARKET_RISK_CAPITAL",
            "CREDIT_RISK_CAPITAL",
            "QUANT_RISK_DIVERSIFICATION_EFFECT",
            "CONTRACT_LOSS_ABSORPTION_EFFECT",
        ),
    )

    result["POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL"] = _safe_divide(core_policy_surplus, get("ACTUAL_CAPITAL"))
    ratio = result["POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL"]
    if ratio is None:
        result["POLICY_SURPLUS_CORE_BAND"] = None
    elif ratio <= 0:
        result["POLICY_SURPLUS_CORE_BAND"] = "小于等于0%"
    elif ratio <= 0.20:
        result["POLICY_SURPLUS_CORE_BAND"] = "(0,20%]"
    elif ratio <= 0.35:
        result["POLICY_SURPLUS_CORE_BAND"] = "(20%,35%]"
    else:
        result["POLICY_SURPLUS_CORE_BAND"] = "大于35%"

    quant_capital = get("QUANT_RISK_CAPITAL")
    result["FEATURE_FACTOR_IMPACT"] = (
        None if quant_capital is None or quant_before_factor is None else quant_capital - quant_before_factor
    )
    result["ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS"] = _safe_divide(get("ACTUAL_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["CORE_T1_TO_ACTUAL_CAPITAL"] = _safe_divide(get("CORE_T1_CAPITAL"), get("ACTUAL_CAPITAL"))
    result["CORE_T2_TO_ACTUAL_CAPITAL"] = _safe_divide(get("CORE_T2_CAPITAL"), get("ACTUAL_CAPITAL"))
    result["ANC_T1_TO_ACTUAL_CAPITAL"] = _safe_divide(get("ANC_T1_CAPITAL"), get("ACTUAL_CAPITAL"))
    result["ANC_T2_TO_ACTUAL_CAPITAL"] = _safe_divide(get("ANC_T2_CAPITAL"), get("ACTUAL_CAPITAL"))
    result["MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES"] = _safe_divide(get("MINIMUM_CAPITAL"), get("RECOGNIZED_LIABILITIES"))
    liabilities = _safe_sum(values, ("INSURANCE_CONTRACT_LIABILITY", "SEPARATE_ACCOUNT_LIABILITY"))
    result["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"] = _safe_divide(all_policy_surplus, liabilities)
    result["RECOGNIZED_ASSETS_TO_ACTUAL_CAPITAL"] = _safe_divide(get("RECOGNIZED_ASSETS"), get("ACTUAL_CAPITAL"))
    result["RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL"] = _safe_divide(get("RECOGNIZED_ASSETS"), get("MINIMUM_CAPITAL"))
    result["RECOGNIZED_ASSETS_TO_REGISTERED_CAPITAL"] = _safe_divide(get("RECOGNIZED_ASSETS"), get("REGISTERED_CAPITAL"))
    result["ACTUAL_CAPITAL_TO_REGISTERED_CAPITAL"] = _safe_divide(get("ACTUAL_CAPITAL"), get("REGISTERED_CAPITAL"))
    result["FEATURE_FACTOR_CHECK"] = _safe_divide(get("FEATURE_FACTOR_IMPACT"), quant_before_factor)
    result["LIFE_INSURANCE_RISK_TO_LIABILITIES"] = _safe_divide(get("INSURANCE_RISK_CAPITAL"), get("RECOGNIZED_LIABILITIES"))
    result["NON_LIFE_INSURANCE_RISK_TO_LIABILITIES"] = _safe_divide(get("NON_LIFE_INSURANCE_RISK_CAPITAL"), get("RECOGNIZED_LIABILITIES"))
    result["MARKET_RISK_TO_ASSETS"] = _safe_divide(get("MARKET_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["CREDIT_RISK_TO_ASSETS"] = _safe_divide(get("CREDIT_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    core_capital = _safe_sum(values, ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"))
    result["CORE_CAPITAL_TO_REGISTERED_CAPITAL"] = _safe_divide(core_capital, get("REGISTERED_CAPITAL"))
    result["POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"] = _safe_divide(core_policy_surplus, core_capital)
    result["CORE_T1_POLICY_SURPLUS_SHARE"] = _safe_divide(get("POLICY_SURPLUS_CORE_T1"), get("CORE_T1_CAPITAL"))
    result["ANC_T1_POLICY_SURPLUS_SHARE"] = _safe_divide(get("POLICY_SURPLUS_ANC_T1"), get("ANC_T1_CAPITAL"))
    result["INTEREST_RATE_RISK_TO_ASSETS"] = _safe_divide(get("INTEREST_RATE_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["EQUITY_RISK_TO_ASSETS"] = _safe_divide(get("EQUITY_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["SPREAD_RISK_TO_ASSETS"] = _safe_divide(get("SPREAD_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["COUNTERPARTY_RISK_TO_ASSETS"] = _safe_divide(get("COUNTERPARTY_RISK_CAPITAL"), get("RECOGNIZED_ASSETS"))
    result["LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL"] = _safe_divide(get("INSURANCE_RISK_CAPITAL"), quant_capital)
    result["NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL"] = _safe_divide(get("NON_LIFE_INSURANCE_RISK_CAPITAL"), quant_capital)
    result["MARKET_RISK_TO_QUANT_CAPITAL"] = _safe_divide(get("MARKET_RISK_CAPITAL"), quant_capital)
    result["CREDIT_RISK_TO_QUANT_CAPITAL"] = _safe_divide(get("CREDIT_RISK_CAPITAL"), quant_capital)
    result["DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL"] = _safe_divide(get("QUANT_RISK_DIVERSIFICATION_EFFECT"), quant_capital)
    result["LOSS_ABSORPTION_TO_QUANT_CAPITAL"] = _safe_divide(get("CONTRACT_LOSS_ABSORPTION_EFFECT"), quant_capital)
    result["TOTAL_ASSETS_TO_REGISTERED_CAPITAL"] = _safe_divide(get("TOTAL_ASSETS"), get("REGISTERED_CAPITAL"))
    return result


def calculate_industry_values(company_totals: dict[str, Any]) -> dict[str, Any]:
    """Calculate the populated industry row in the PPT draft ``汇总`` sheet.

    Amount fields are supplied as company totals. Ratios are recalculated from
    those totals instead of averaging company ratios. Two formulas in the draft
    contradict their labels, so this implementation follows the metric meaning:
    separate-account liabilities are summed and recognized-assets/minimum-capital
    uses recognized assets as its numerator.
    """
    result = dict(company_totals)
    get = lambda code: _number(result.get(code))

    result["COMBINED_SOLVENCY_RATIO"] = _safe_divide(
        get("ACTUAL_CAPITAL"), get("MINIMUM_CAPITAL")
    )
    core_capital = _safe_sum(result, ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"))
    result["CORE_SOLVENCY_RATIO"] = _safe_divide(core_capital, get("MINIMUM_CAPITAL"))
    for code in PERCENT_FROM_RATIO_CODES:
        if result.get(code) is not None:
            result[code] = float(result[code]) * 100

    all_policy_surplus = _safe_sum(
        result,
        (
            "POLICY_SURPLUS_CORE_T1",
            "POLICY_SURPLUS_CORE_T2",
            "POLICY_SURPLUS_ANC_T1",
            "POLICY_SURPLUS_ANC_T2",
        ),
    )
    result["ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS"] = _safe_divide(
        get("ACTUAL_CAPITAL"), get("RECOGNIZED_ASSETS")
    )
    result["MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES"] = _safe_divide(
        get("MINIMUM_CAPITAL"), get("RECOGNIZED_LIABILITIES")
    )
    result["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"] = _safe_divide(
        all_policy_surplus, get("INSURANCE_CONTRACT_LIABILITY")
    )
    result["RECOGNIZED_ASSETS_TO_ACTUAL_CAPITAL"] = _safe_divide(
        get("RECOGNIZED_ASSETS"), get("ACTUAL_CAPITAL")
    )
    result["RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL"] = _safe_divide(
        get("RECOGNIZED_ASSETS"), get("MINIMUM_CAPITAL")
    )
    return result


def _industry_company_totals(
    frame: pd.DataFrame,
    code_by_column: dict[str, str],
) -> dict[str, float]:
    source_column_by_code: dict[str, str] = {}
    for column, code in code_by_column.items():
        source_column_by_code.setdefault(code, column)

    ratio_codes = set(INDUSTRY_RATIO_FORMULAS)
    totals: dict[str, float] = {}
    for code in INDUSTRY_OUTPUT_CODES:
        if code in ratio_codes:
            continue
        column = source_column_by_code.get(code)
        if column is None:
            continue
        numeric = frame[column].map(_number).dropna()
        if not numeric.empty:
            totals[code] = float(numeric.sum())
    return totals


def _industry_metric_records(
    frame: pd.DataFrame,
    code_by_column: dict[str, str],
    lookup: dict[str, dict],
    *,
    filename: str,
    sheet_name: str,
    report_profile_id: str,
    year: int,
    quarter: str,
    report_period: str,
) -> list[dict]:
    totals = _industry_company_totals(frame, code_by_column)
    values = calculate_industry_values(totals)
    metrics_by_code: dict[str, dict] = {}
    for metric in lookup.values():
        metrics_by_code.setdefault(str(metric.get("指标编码", "")), metric)

    records: list[dict] = []
    for source_code in INDUSTRY_OUTPUT_CODES:
        value = values.get(source_code)
        if value is None or not math.isfinite(float(value)):
            continue
        source_metric = metrics_by_code.get(source_code)
        if source_metric is None:
            continue

        renamed = INDUSTRY_METRICS_BY_SOURCE_CODE.get(source_code)
        if renamed is None:
            metric_code = source_code
            metric_name = source_metric["指标名称"]
            level1 = source_metric["一级模块"]
            level2 = source_metric["二级模块"]
            unit = source_metric["单位"]
            data_type = source_metric["数据类型"]
            rename_note = ""
        else:
            metric_code = renamed.code
            metric_name = renamed.name
            level1 = renamed.level1
            level2 = renamed.level2
            unit = renamed.unit
            data_type = renamed.data_type
            rename_note = f"；行业指标由“{source_metric['指标名称']}”重命名"

        formula = INDUSTRY_RATIO_FORMULAS.get(
            source_code,
            f"所有有效公司的{source_metric['指标名称']}求和（忽略空值）",
        )
        records.append({
            "公司": "行业合计",
            "原始公司名称": "行业合计",
            "标准公司名称": "行业合计",
            "公司统一编码": INDUSTRY_TOTAL_COMPANY_CODE,
            "公司类型": "行业合计",
            "同业分类": "全行业",
            "报告类型": report_profile_id,
            "报告年度": year,
            "报告季度": quarter,
            "报告期": report_period,
            "披露日期": "",
            "一级模块": level1,
            "二级模块": level2,
            "行次": "",
            "指标编码": metric_code,
            "指标名称": metric_name,
            "期间口径": "本季度末数",
            "数值": float(value),
            "单位": unit,
            "数据类型": data_type,
            "是否预测": "否",
            "来源页码": "",
            "原始披露值": "",
            "备注": f"按PPT底稿“汇总”页行业口径，由本工作表公司数据计算{rename_note}",
            "来源类型": "系统计算",
            "指标属性": "行业计算",
            "来源文件": filename,
            "来源工作表": sheet_name,
            "导入批次": f"{Path(filename).stem}:{report_period}",
            "计算逻辑": formula,
        })
    return records


def append_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    source = apply_company_identities(standardize_uploaded_frame(frame))
    if source.empty:
        return source
    if "报告类型" not in source.columns:
        source["报告类型"] = ""
    source = source[~source["指标编码"].isin([item.code for item in DERIVED_METRICS])].copy()
    source["_期间组"] = source["期间口径"].map(canonical_period_scope)
    grouping = ["报告类型", "公司统一编码", "报告年度", "报告季度", "报告期", "_期间组"]
    derived_records: list[dict] = []

    for _, group in source.groupby(grouping, dropna=False, sort=False):
        values: dict[str, Any] = {}
        for code, code_rows in group.groupby("指标编码", sort=False):
            candidates = pd.to_numeric(code_rows["数值"], errors="coerce").dropna()
            if not candidates.empty:
                values[str(code)] = float(candidates.iloc[0])
        calculated = calculate_derived_values(values)
        base = group.iloc[0]
        source_pages = "、".join(dict.fromkeys(str(item) for item in group["来源页码"] if str(item).strip()))
        source_sheets = "、".join(dict.fromkeys(str(item) for item in group["来源工作表"] if str(item).strip()))
        for definition in DERIVED_METRICS:
            value = calculated.get(definition.code)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                continue
            derived_records.append({
                "公司": base.get("公司", ""),
                "原始公司名称": base.get("原始公司名称", ""),
                "标准公司名称": base.get("标准公司名称", ""),
                "公司统一编码": base.get("公司统一编码", ""),
                "公司类型": base.get("公司类型", ""),
                "同业分类": base.get("同业分类", ""),
                "报告类型": base.get("报告类型", ""),
                "报告年度": base.get("报告年度", ""),
                "报告季度": base.get("报告季度", ""),
                "报告期": base.get("报告期", ""),
                "披露日期": base.get("披露日期", ""),
                "一级模块": definition.level1,
                "二级模块": definition.level2,
                "行次": "",
                "指标编码": definition.code,
                "指标名称": definition.name,
                "期间口径": base.get("_期间组", "本季度末数"),
                "数值": value,
                "单位": definition.unit,
                "数据类型": definition.data_type,
                "是否预测": "是" if "预测" in str(base.get("_期间组", "")) else "否",
                "来源页码": source_pages,
                "原始披露值": "",
                "备注": "由标准化基础指标自动计算",
                "来源类型": "系统计算",
                "指标属性": definition.attribute,
                "来源文件": base.get("来源文件", ""),
                "来源工作表": source_sheets,
                "导入批次": base.get("导入批次", ""),
                "计算逻辑": definition.formula,
            })

    source = source.drop(columns=["_期间组"])
    if not derived_records:
        return standardize_uploaded_frame(source)
    return pd.concat(
        [standardize_uploaded_frame(source), pd.DataFrame(derived_records, columns=STANDARD_COLUMNS)],
        ignore_index=True,
    )


def add_missing_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill calculable derived metrics without replacing values supplied by the user.

    ``append_derived_metrics`` is intentionally authoritative for STEP3: it removes
    existing derived rows and recalculates them.  Integrated workbooks need a gentler
    policy because a reviewed external data set may already contain selected ratios.
    This helper keeps every supplied row and appends only derived metric keys that are
    absent and can be calculated from the available base indicators.
    """
    source = apply_company_identities(standardize_uploaded_frame(frame))
    if source.empty:
        return source

    derived_codes = {item.code for item in DERIVED_METRICS}
    calculated = append_derived_metrics(source)
    candidates = calculated[calculated["指标编码"].isin(derived_codes)].copy()
    if candidates.empty:
        return source

    existing = source[source["指标编码"].isin(derived_codes)].copy()
    key_columns = [
        "报告类型",
        "公司统一编码",
        "报告年度",
        "报告季度",
        "报告期",
        "_期间组",
        "指标编码",
    ]

    def row_keys(rows: pd.DataFrame) -> pd.Series:
        keyed = rows.copy()
        keyed["_期间组"] = keyed["期间口径"].map(canonical_period_scope)
        values = keyed[key_columns].fillna("").astype(str)
        return values.apply(tuple, axis=1)

    existing_keys = set(row_keys(existing).tolist()) if not existing.empty else set()
    missing_mask = ~row_keys(candidates).isin(existing_keys)
    additions = candidates.loc[missing_mask]
    if additions.empty:
        return source
    return pd.concat([source, additions], ignore_index=True)


def _metric_lookup(taxonomy: pd.DataFrame) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for _, row in taxonomy.iterrows():
        names = [row.get("指标名称", "")]
        names.extend(str(row.get("别名", "")).split("|"))
        payload = {
            "指标编码": str(row.get("指标编码", "")).strip(),
            "指标名称": str(row.get("指标名称", "")).strip(),
            "一级模块": str(row.get("一级模块", "")).strip(),
            "二级模块": str(row.get("二级模块", "")).strip(),
            "单位": str(row.get("标准单位", "")).strip(),
            "数据类型": str(row.get("数据类型", "")).strip(),
            "指标属性": "披露",
            "计算逻辑": "",
        }
        for name in names:
            key = canonical_metric_name(name)
            if key:
                lookup.setdefault(key, payload)
    for definition in DERIVED_METRICS:
        lookup[canonical_metric_name(definition.name)] = {
            "指标编码": definition.code,
            "指标名称": definition.name,
            "一级模块": definition.level1,
            "二级模块": definition.level2,
            "单位": definition.unit,
            "数据类型": definition.data_type,
            "指标属性": definition.attribute,
            "计算逻辑": definition.formula,
        }
    return lookup


def _external_value(raw_value: Any, metric: dict) -> Any:
    if metric["数据类型"] in {"文本", "评级", "布尔"}:
        return "" if pd.isna(raw_value) else str(raw_value).strip()
    numeric = _number(raw_value)
    if numeric is None:
        return np.nan
    if metric["指标编码"] in PERCENT_FROM_RATIO_CODES:
        text = str(raw_value).strip()
        return numeric if text.endswith(("%", "％")) else numeric * 100
    return numeric


def _company_type(value: Any) -> str:
    text = str(value or "").strip()
    return {"养老": "养老险", "健康": "健康险"}.get(text, text or "未分类")


def convert_external_workbook(
    workbook_bytes: bytes,
    filename: str,
    taxonomy: pd.DataFrame,
    company_type_map: dict[str, str] | None = None,
    report_profile_id: str = "LIFE_SOLVENCY",
) -> ExternalConversionResult:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    quarter_sheets = [name for name in excel.sheet_names if QUARTER_SHEET_PATTERN.fullmatch(str(name).strip())]
    if not quarter_sheets:
        raise ValueError("未找到名称形如 2025Q4 的季度数据工作表。")

    lookup = _metric_lookup(taxonomy)
    company_type_map = {str(key).strip(): _company_type(value) for key, value in (company_type_map or {}).items()}
    output: list[dict] = []
    sheet_rows: list[dict] = []
    mapping_rows: dict[str, dict] = {}
    logic_rows: list[dict] = []
    warnings: list[str] = []
    unknown_companies: set[str] = set()

    for sheet_name in quarter_sheets:
        match = QUARTER_SHEET_PATTERN.fullmatch(sheet_name.strip())
        assert match is not None
        year = int(match.group("year"))
        quarter = match.group("quarter").upper()
        report_period = f"{year}{quarter}"
        frame = pd.read_excel(excel, sheet_name=sheet_name, header=0)
        frame.columns = [str(column).strip() for column in frame.columns]
        missing_identifiers = IDENTIFIER_COLUMNS - set(frame.columns)
        if missing_identifiers:
            raise ValueError(f"工作表 {sheet_name} 缺少字段：{', '.join(sorted(missing_identifiers))}")
        frame = frame[frame["公司"].notna() & (frame["公司"].astype(str).str.strip() != "")].copy()
        metric_columns = [column for column in frame.columns if column not in IDENTIFIER_COLUMNS]
        unmapped = [column for column in metric_columns if canonical_metric_name(column) not in lookup]
        if unmapped:
            raise ValueError(f"工作表 {sheet_name} 存在未精确映射指标：{', '.join(unmapped)}")

        code_by_column = {column: lookup[canonical_metric_name(column)]["指标编码"] for column in metric_columns}
        for column in metric_columns:
            metric = lookup[canonical_metric_name(column)]
            mapping_rows.setdefault(column, {
                "来源字段": column,
                "匹配方式": "指标名称精确匹配",
                **metric,
            })

        failed_logic: dict[str, list[float]] = {item.code: [] for item in DERIVED_METRICS}
        checked_logic: dict[str, int] = {item.code: 0 for item in DERIVED_METRICS}
        output_start = len(output)
        skipped_values: dict[str, int] = {}
        for _, row in frame.iterrows():
            company = str(row["公司"]).strip()
            peer_group = str(row.get("分类", "")).strip()
            identity = resolve_company_identity(company, company_type_map)
            if identity.company_type == "未分类":
                unknown_companies.add(company)
            raw_by_code = {
                code_by_column[column]: _number(row[column])
                for column in metric_columns
                if _number(row[column]) is not None
            }
            calculated = calculate_derived_values(raw_by_code)
            for definition in DERIVED_METRICS:
                source_column = next((column for column, code in code_by_column.items() if code == definition.code), None)
                if not source_column:
                    continue
                actual = row[source_column]
                expected = calculated.get(definition.code)
                if expected is None or pd.isna(actual):
                    continue
                checked_logic[definition.code] += 1
                if definition.data_type == "文本":
                    if str(actual).strip() != str(expected).strip():
                        failed_logic[definition.code].append(float("nan"))
                else:
                    actual_number = _number(actual)
                    if actual_number is None:
                        failed_logic[definition.code].append(float("nan"))
                    else:
                        difference = abs(actual_number - float(expected))
                        tolerance = 1e-8 * max(1.0, abs(float(expected)))
                        if difference > tolerance:
                            failed_logic[definition.code].append(difference)

            for column in metric_columns:
                metric = lookup[canonical_metric_name(column)]
                value = _external_value(row[column], metric)
                if metric["数据类型"] not in {"文本", "评级", "布尔"} and pd.isna(value):
                    skipped_values[column] = skipped_values.get(column, 0) + 1
                    continue
                if metric["数据类型"] in {"文本", "评级", "布尔"} and value == "":
                    skipped_values[column] = skipped_values.get(column, 0) + 1
                    continue
                output.append({
                    "公司": identity.standard_name,
                    "原始公司名称": identity.original_name,
                    "标准公司名称": identity.standard_name,
                    "公司统一编码": identity.company_code,
                    "公司类型": identity.company_type,
                    "同业分类": peer_group,
                    "报告类型": report_profile_id,
                    "报告年度": year,
                    "报告季度": quarter,
                    "报告期": report_period,
                    "披露日期": "",
                    "一级模块": metric["一级模块"],
                    "二级模块": metric["二级模块"],
                    "行次": "",
                    "指标编码": metric["指标编码"],
                    "指标名称": metric["指标名称"],
                    "期间口径": "本季度末数",
                    "数值": value,
                    "单位": metric["单位"],
                    "数据类型": metric["数据类型"],
                    "是否预测": "否",
                    "来源页码": "",
                    "原始披露值": row[column],
                    "备注": "外部宽表按指标名称精确映射",
                    "来源类型": "外部数据集",
                    "指标属性": metric["指标属性"],
                    "来源文件": filename,
                    "来源工作表": sheet_name,
                    "导入批次": f"{Path(filename).stem}:{report_period}",
                    "计算逻辑": metric["计算逻辑"],
                })

        company_record_count = len(output) - output_start
        industry_records = _industry_metric_records(
            frame,
            code_by_column,
            lookup,
            filename=filename,
            sheet_name=sheet_name,
            report_profile_id=report_profile_id,
            year=year,
            quarter=quarter,
            report_period=report_period,
        )
        output.extend(industry_records)

        for definition in DERIVED_METRICS:
            if checked_logic[definition.code] == 0:
                continue
            differences = [item for item in failed_logic[definition.code] if math.isfinite(item)]
            logic_rows.append({
                "来源工作表": sheet_name,
                "指标编码": definition.code,
                "指标名称": definition.name,
                "校验行数": checked_logic[definition.code],
                "不一致行数": len(failed_logic[definition.code]),
                "最大绝对差异": max(differences) if differences else 0.0,
                "状态": "通过" if not failed_logic[definition.code] else "需复核",
                "计算逻辑": definition.formula,
            })
        sheet_rows.append({
            "来源文件": filename,
            "来源工作表": sheet_name,
            "报告期": report_period,
            "公司数": int(frame["公司"].nunique()),
            "指标数": len(metric_columns),
            "源数据单元格数": len(frame) * len(metric_columns),
            "转换记录数": len(output) - output_start,
            "公司转换记录数": company_record_count,
            "行业指标数": len(industry_records),
            "跳过空值或错误值": sum(skipped_values.values()),
        })
        if skipped_values:
            details = "、".join(f"{name} {count} 条" for name, count in skipped_values.items())
            warnings.append(f"工作表 {sheet_name} 跳过 {sum(skipped_values.values())} 个空值或错误值：{details}")

    if unknown_companies:
        warnings.append(f"{len(unknown_companies)} 家公司未在公司主数据中找到，已标记为“未分类”：{'、'.join(sorted(unknown_companies))}")
    logic_frame = pd.DataFrame(logic_rows)
    if not logic_frame.empty and (logic_frame["状态"] != "通过").any():
        warnings.append("部分外部派生指标与系统计算逻辑不一致，请在确认集成前查看逻辑校验结果。")

    return ExternalConversionResult(
        data=pd.DataFrame(output, columns=STANDARD_COLUMNS),
        sheet_summary=pd.DataFrame(sheet_rows),
        mapping_summary=pd.DataFrame(mapping_rows.values()),
        logic_checks=logic_frame,
        warnings=warnings,
    )


def read_standard_workbook(workbook_bytes: bytes, filename: str) -> pd.DataFrame:
    excel = pd.ExcelFile(io.BytesIO(workbook_bytes))
    sheet_name = "标准数据" if "标准数据" in excel.sheet_names else excel.sheet_names[0]
    preview = pd.read_excel(excel, sheet_name=sheet_name, header=None, nrows=12)
    header_row = None
    for index, row in preview.iterrows():
        values = {str(value).strip() for value in row if not pd.isna(value)}
        if REQUIRED_STANDARD_HEADERS.issubset(values):
            header_row = int(index)
            break
    if header_row is None:
        raise ValueError(f"{filename} 未找到标准窄表表头。")
    frame = pd.read_excel(excel, sheet_name=sheet_name, header=header_row)
    frame.columns = [str(column).strip() for column in frame.columns]
    result = apply_company_identities(standardize_uploaded_frame(frame))
    blank_source = result["来源类型"].astype(str).str.strip() == ""
    result.loc[blank_source, "来源类型"] = "标准窄表上传"
    blank_file = result["来源文件"].astype(str).str.strip() == ""
    result.loc[blank_file, "来源文件"] = filename
    blank_sheet = result["来源工作表"].astype(str).str.strip() == ""
    result.loc[blank_sheet, "来源工作表"] = sheet_name
    blank_batch = result["导入批次"].astype(str).str.strip() == ""
    result.loc[blank_batch, "导入批次"] = Path(filename).stem
    return result
