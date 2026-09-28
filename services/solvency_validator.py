from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .solvency_metric_registry import extend_taxonomy
from .solvency_normalizer import STANDARD_COLUMNS


EXTRA_VALIDATION_RULES = [
    {
        "规则ID": "DUPLICATE_ACTUAL_CAPITAL",
        "规则名称": "实际资本跨表一致性",
        "适用期间": "本季度末数|期末数|本季度（末）数",
        "容差": 1,
        "容差单位": "万元",
        "启用": "是",
        "规则说明": "主要指标表与实际资本表披露的实际资本应一致",
    },
    {
        "规则ID": "DUPLICATE_MINIMUM_CAPITAL",
        "规则名称": "最低资本跨表一致性",
        "适用期间": "本季度末数|期末数|本季度（末）数",
        "容差": 1,
        "容差单位": "万元",
        "启用": "是",
        "规则说明": "主要指标表与最低资本表披露的最低资本应一致",
    },
    {
        "规则ID": "DUPLICATE_RECOGNIZED_ASSETS",
        "规则名称": "认可资产跨表一致性",
        "适用期间": "本季度末数|期末数|本季度（末）数",
        "容差": 1,
        "容差单位": "万元",
        "启用": "是",
        "规则说明": "主要指标表与认可资产表披露的认可资产应一致",
    },
]


PERIOD_GROUP_ALIASES: dict[str, tuple[str, ...]] = {
    "当前期末": ("本季度末数", "期末数", "本季度（末）数"),
    "上期末/期初": ("上季度末数", "期初数"),
    "下季度预测": (
        "下季度末预测数",
        "下季度预测数",
        "基本情景下的下季度预测数",
    ),
    "本季度": ("本季度数",),
    "上季度": ("上季度数",),
    "本年累计": ("本年累计数", "年度累计数"),
}
CURRENT_PERIOD_GROUP = "当前期末"
PERIOD_ALIAS_TO_GROUP = {
    alias: group
    for group, aliases in PERIOD_GROUP_ALIASES.items()
    for alias in aliases
}


REQUIRED_CORE_METRICS: tuple[str, ...] = (
    "RECOGNIZED_ASSETS",
    "RECOGNIZED_LIABILITIES",
    "ACTUAL_CAPITAL",
    "MINIMUM_CAPITAL",
    "CORE_SOLVENCY_RATIO",
    "COMBINED_SOLVENCY_RATIO",
)


RULE_METRICS: dict[str, tuple[str, ...]] = {
    "ACTUAL_CAPITAL_COMPONENTS": (
        "ACTUAL_CAPITAL",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "ANC_T1_CAPITAL",
        "ANC_T2_CAPITAL",
    ),
    "ACTUAL_CAPITAL_BALANCE": (
        "ACTUAL_CAPITAL",
        "RECOGNIZED_ASSETS",
        "RECOGNIZED_LIABILITIES",
    ),
    "CORE_SURPLUS": (
        "CORE_SOLVENCY_SURPLUS",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "MINIMUM_CAPITAL",
    ),
    "COMBINED_SURPLUS": (
        "COMBINED_SOLVENCY_SURPLUS",
        "ACTUAL_CAPITAL",
        "MINIMUM_CAPITAL",
    ),
    "CORE_RATIO": (
        "CORE_SOLVENCY_RATIO",
        "CORE_T1_CAPITAL",
        "CORE_T2_CAPITAL",
        "MINIMUM_CAPITAL",
    ),
    "COMBINED_RATIO": (
        "COMBINED_SOLVENCY_RATIO",
        "ACTUAL_CAPITAL",
        "MINIMUM_CAPITAL",
    ),
    "MINIMUM_CAPITAL": (
        "MINIMUM_CAPITAL",
        "QUANT_RISK_CAPITAL",
        "CONTROL_RISK_CAPITAL",
        "ADDITIONAL_CAPITAL",
    ),
    "DUPLICATE_ACTUAL_CAPITAL": ("ACTUAL_CAPITAL",),
    "DUPLICATE_MINIMUM_CAPITAL": ("MINIMUM_CAPITAL",),
    "DUPLICATE_RECOGNIZED_ASSETS": ("RECOGNIZED_ASSETS",),
}


RESULT_COLUMNS = [
    "check_type",
    "severity",
    "rule_id",
    "rule_name",
    "company",
    "report_period",
    "period",
    "actual",
    "expected",
    "difference",
    "tolerance",
    "status",
    "notes",
    "involved_metrics",
    "source_pages",
    "suggestion",
]


@dataclass
class ValidationResult:
    check_type: str
    severity: str
    rule_id: str
    rule_name: str
    company: str
    report_period: str
    period: str
    actual: float | None
    expected: float | None
    difference: float | None
    tolerance: float
    status: str
    notes: str
    involved_metrics: str
    source_pages: str
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_validation_rules(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name="勾稽规则", header=2).fillna("")
    existing = set(frame.get("规则ID", pd.Series(dtype=str)).astype(str))
    additions = [item for item in EXTRA_VALIDATION_RULES if item["规则ID"] not in existing]
    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    return frame.fillna("")


def validation_period_group(value: Any) -> str:
    period = str(value or "").strip()
    return PERIOD_ALIAS_TO_GROUP.get(period, period)


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_tolerance(value: Any) -> float:
    number = _numeric(value)
    return 0.0 if number is None else number


def _source_pages(
    frame: pd.DataFrame,
    codes: Iterable[str] = (),
    period_group: str = "",
) -> str:
    scoped = frame
    code_values = {str(code) for code in codes if str(code).strip()}
    if code_values:
        scoped = scoped[scoped["指标编码"].astype(str).isin(code_values)]
    if period_group:
        scoped = scoped[scoped["_验证期间组"].astype(str).eq(period_group)]
    pages = [
        str(value).strip()
        for value in scoped.get("来源页码", pd.Series(dtype=str))
        if str(value).strip() and str(value).strip().lower() != "nan"
    ]
    return "、".join(dict.fromkeys(pages))


def _metric_values(frame: pd.DataFrame, code: str, period_group: str) -> list[float]:
    rows = frame[
        frame["指标编码"].astype(str).eq(code)
        & frame["_验证期间组"].astype(str).eq(period_group)
    ]
    return [
        float(value)
        for value in pd.to_numeric(rows["数值"], errors="coerce").dropna()
        if np.isfinite(float(value))
    ]


def _metric_value(frame: pd.DataFrame, code: str, period_group: str) -> float | None:
    values = _metric_values(frame, code, period_group)
    return values[0] if values else None


def _sum_metrics(frame: pd.DataFrame, codes: Iterable[str], period_group: str) -> float | None:
    values = [_metric_value(frame, code, period_group) for code in codes]
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None))


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator * 100


def _evaluate_business_rule(
    frame: pd.DataFrame,
    rule_id: str,
    period_group: str,
) -> tuple[float | None, float | None]:
    value = lambda code: _metric_value(frame, code, period_group)
    if rule_id == "ACTUAL_CAPITAL_COMPONENTS":
        return value("ACTUAL_CAPITAL"), _sum_metrics(
            frame,
            ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"),
            period_group,
        )
    if rule_id == "ACTUAL_CAPITAL_BALANCE":
        assets = value("RECOGNIZED_ASSETS")
        liabilities = value("RECOGNIZED_LIABILITIES")
        return value("ACTUAL_CAPITAL"), (
            None if assets is None or liabilities is None else assets - liabilities
        )
    if rule_id == "CORE_SURPLUS":
        core_capital = _sum_metrics(
            frame, ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"), period_group
        )
        minimum = value("MINIMUM_CAPITAL")
        return value("CORE_SOLVENCY_SURPLUS"), (
            None if core_capital is None or minimum is None else core_capital - minimum
        )
    if rule_id == "COMBINED_SURPLUS":
        actual_capital = value("ACTUAL_CAPITAL")
        minimum = value("MINIMUM_CAPITAL")
        return value("COMBINED_SOLVENCY_SURPLUS"), (
            None if actual_capital is None or minimum is None else actual_capital - minimum
        )
    if rule_id == "CORE_RATIO":
        core_capital = _sum_metrics(
            frame, ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL"), period_group
        )
        return value("CORE_SOLVENCY_RATIO"), _ratio(core_capital, value("MINIMUM_CAPITAL"))
    if rule_id == "COMBINED_RATIO":
        return value("COMBINED_SOLVENCY_RATIO"), _ratio(
            value("ACTUAL_CAPITAL"), value("MINIMUM_CAPITAL")
        )
    if rule_id == "MINIMUM_CAPITAL":
        return value("MINIMUM_CAPITAL"), _sum_metrics(
            frame,
            ("QUANT_RISK_CAPITAL", "CONTROL_RISK_CAPITAL", "ADDITIONAL_CAPITAL"),
            period_group,
        )
    if rule_id in {
        "DUPLICATE_ACTUAL_CAPITAL",
        "DUPLICATE_MINIMUM_CAPITAL",
        "DUPLICATE_RECOGNIZED_ASSETS",
    }:
        values = _metric_values(frame, RULE_METRICS[rule_id][0], period_group)
        if len(values) < 2:
            return None, None
        return max(values), min(values)
    raise KeyError(f"未注册的勾稽规则：{rule_id}")


def _entity_groups(frame: pd.DataFrame):
    working = frame.copy()
    for column in STANDARD_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working["_验证公司"] = (
        working["标准公司名称"].fillna("").astype(str).str.strip()
        .mask(lambda values: values.eq(""), working["公司"].fillna("").astype(str).str.strip())
    )
    working["_验证公司键"] = (
        working["公司统一编码"].fillna("").astype(str).str.strip()
        .mask(lambda values: values.eq(""), working["_验证公司"])
    )
    working["_验证报告期"] = working["报告期"].fillna("").astype(str).str.strip()
    fallback_period = (
        working["报告年度"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
        + working["报告季度"].fillna("").astype(str).str.strip()
    )
    working["_验证报告期"] = working["_验证报告期"].mask(
        working["_验证报告期"].eq(""), fallback_period
    )
    working["_验证报告类型"] = working["报告类型"].fillna("").astype(str).str.strip()
    working["_验证期间组"] = working["期间口径"].map(validation_period_group)
    grouping = ["_验证报告类型", "_验证公司键", "_验证报告期"]
    for _, group in working.groupby(grouping, dropna=False, sort=False):
        yield group.copy()


def _result(
    *,
    check_type: str,
    severity: str,
    rule_id: str,
    rule_name: str,
    group: pd.DataFrame,
    period: str = "",
    actual: Any = None,
    expected: Any = None,
    difference: Any = None,
    tolerance: float = 0.0,
    status: str,
    notes: str = "",
    involved_metrics: Iterable[str] = (),
    source_pages: str = "",
    suggestion: str = "",
) -> ValidationResult:
    company = str(group["_验证公司"].iloc[0]).strip() if not group.empty else ""
    report_period = str(group["_验证报告期"].iloc[0]).strip() if not group.empty else ""
    return ValidationResult(
        check_type=check_type,
        severity=severity,
        rule_id=rule_id,
        rule_name=rule_name,
        company=company,
        report_period=report_period,
        period=period,
        actual=_numeric(actual),
        expected=_numeric(expected),
        difference=_numeric(difference),
        tolerance=float(tolerance),
        status=status,
        notes=notes,
        involved_metrics="、".join(
            dict.fromkeys(str(code) for code in involved_metrics if str(code).strip())
        ),
        source_pages=source_pages,
        suggestion=suggestion,
    )


def _issue_summary(rows: pd.DataFrame, columns: Iterable[str], limit: int = 5) -> str:
    details: list[str] = []
    for _, row in rows.head(limit).iterrows():
        parts = [str(row.get(column, "")).strip() for column in columns]
        details.append(" / ".join(part for part in parts if part))
    return "；".join(details) + ("等" if len(rows) > limit else "")


def _schema_checks(group: pd.DataFrame, taxonomy: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    taxonomy_frame = taxonomy.copy().fillna("")
    taxonomy_frame["指标编码"] = taxonomy_frame["指标编码"].astype(str).str.strip()
    taxonomy_frame = taxonomy_frame[taxonomy_frame["指标编码"] != ""].drop_duplicates(
        subset="指标编码", keep="last"
    )
    taxonomy_by_code = taxonomy_frame.set_index("指标编码", drop=False)

    codes = group["指标编码"].fillna("").astype(str).str.strip()
    required_fields = (
        "公司", "标准公司名称", "公司统一编码", "公司类型", "同业分类",
        "报告类型", "报告年度", "报告季度", "报告期", "指标编码",
        "指标名称", "期间口径", "数据类型",
    )
    missing_by_field = {
        field: int(group[field].fillna("").astype(str).str.strip().eq("").sum())
        for field in required_fields
    }
    expected_units = codes.map(
        lambda code: str(taxonomy_by_code.loc[code].get("标准单位", "")).strip()
        if code in taxonomy_by_code.index else ""
    )
    missing_units = int(
        (
            group["单位"].fillna("").astype(str).str.strip().eq("")
            & expected_units.ne("")
        ).sum()
    )
    if missing_units:
        missing_by_field["单位"] = missing_units
    missing_by_field = {field: count for field, count in missing_by_field.items() if count}
    missing_total = sum(missing_by_field.values())
    results.append(_result(
        check_type="数据规范", severity="错误", rule_id="SCHEMA_REQUIRED_FIELDS",
        rule_name="标准窄表必填字段完整性", group=group, actual=missing_total,
        expected=0, difference=missing_total,
        status="通过" if missing_total == 0 else "未通过",
        notes=("所有必填字段均已填报" if missing_total == 0 else "；".join(
            f"{field}缺失{count}条" for field, count in missing_by_field.items()
        )),
        suggestion="返回 Step3 补齐公司、报告期或指标元数据。" if missing_total else "",
    ))

    unknown_rows = group.loc[~codes.isin(taxonomy_by_code.index)]
    results.append(_result(
        check_type="数据规范", severity="错误", rule_id="SCHEMA_KNOWN_METRIC",
        rule_name="指标编码存在于正式指标字典", group=group, actual=len(unknown_rows),
        expected=0, difference=len(unknown_rows),
        status="通过" if unknown_rows.empty else "未通过",
        notes=("全部指标编码均已登记" if unknown_rows.empty else "未登记指标：" + "、".join(
            dict.fromkeys(unknown_rows["指标编码"].astype(str))
        )),
        suggestion="返回 Step3 修正指标映射；新指标须先进入正式指标字典。" if not unknown_rows.empty else "",
    ))

    known_rows = group.loc[codes.isin(taxonomy_by_code.index)].copy()
    comparisons = (
        ("SCHEMA_METRIC_NAME", "指标编码与指标名称一致", "指标名称", "指标名称"),
        ("SCHEMA_UNIT_MATCH", "指标单位符合标准单位", "单位", "标准单位"),
        ("SCHEMA_DATA_TYPE", "指标数据类型符合字典", "数据类型", "数据类型"),
    )
    for rule_id, rule_name, source_column, taxonomy_column in comparisons:
        mismatch_indices = []
        for index, row in known_rows.iterrows():
            definition = taxonomy_by_code.loc[str(row["指标编码"]).strip()]
            if str(row.get(source_column, "")).strip() != str(definition.get(taxonomy_column, "")).strip():
                mismatch_indices.append(index)
        mismatches = known_rows.loc[mismatch_indices]
        results.append(_result(
            check_type="数据规范", severity="错误", rule_id=rule_id,
            rule_name=rule_name, group=group, actual=len(mismatches), expected=0,
            difference=len(mismatches), status="通过" if mismatches.empty else "未通过",
            notes="检查通过" if mismatches.empty else _issue_summary(
                mismatches, ("指标编码", source_column)
            ),
            involved_metrics=mismatches["指标编码"].astype(str).tolist(),
            source_pages=_source_pages(mismatches),
            suggestion="返回 Step3 按指标字典统一名称、单位和数据类型。" if not mismatches.empty else "",
        ))

    invalid_period_indices = []
    for index, row in known_rows.iterrows():
        definition = taxonomy_by_code.loc[str(row["指标编码"]).strip()]
        allowed = {
            item.strip() for item in str(definition.get("允许期间口径", "")).split("|")
            if item.strip()
        }
        if allowed and str(row.get("期间口径", "")).strip() not in allowed:
            invalid_period_indices.append(index)
    invalid_periods = known_rows.loc[invalid_period_indices]
    results.append(_result(
        check_type="数据规范", severity="错误", rule_id="SCHEMA_ALLOWED_PERIOD",
        rule_name="期间口径符合指标字典", group=group, actual=len(invalid_periods),
        expected=0, difference=len(invalid_periods),
        status="通过" if invalid_periods.empty else "未通过",
        notes="全部期间口径均在允许范围内" if invalid_periods.empty else _issue_summary(
            invalid_periods, ("指标编码", "期间口径")
        ),
        involved_metrics=invalid_periods["指标编码"].astype(str).tolist(),
        source_pages=_source_pages(invalid_periods),
        suggestion="返回 Step2 核对表头列位，或在 Step3 修正期间口径。" if not invalid_periods.empty else "",
    ))

    numeric_types = {"金额", "百分比", "数量", "倍数"}
    invalid_numeric_indices = []
    for index, row in known_rows.iterrows():
        definition = taxonomy_by_code.loc[str(row["指标编码"]).strip()]
        if str(definition.get("数据类型", "")).strip() in numeric_types and _numeric(row.get("数值")) is None:
            invalid_numeric_indices.append(index)
    invalid_numeric = known_rows.loc[invalid_numeric_indices]
    results.append(_result(
        check_type="数据规范", severity="错误", rule_id="SCHEMA_NUMERIC_VALUE",
        rule_name="数值字段可按指标类型解析", group=group, actual=len(invalid_numeric),
        expected=0, difference=len(invalid_numeric),
        status="通过" if invalid_numeric.empty else "未通过",
        notes="金额、百分比、数量和倍数均可正常解析" if invalid_numeric.empty else _issue_summary(
            invalid_numeric, ("指标编码", "数值")
        ),
        involved_metrics=invalid_numeric["指标编码"].astype(str).tolist(),
        source_pages=_source_pages(invalid_numeric),
        suggestion="返回 Step2 核对原始值和列错位，再重新执行 Step3。" if not invalid_numeric.empty else "",
    ))

    source_key = ["来源文件", "来源工作表", "指标编码", "_验证期间组"]
    exact_duplicate_count = int(group.duplicated(subset=[*source_key, "数值"], keep="first").sum())
    results.append(_result(
        check_type="数据规范", severity="警告", rule_id="SCHEMA_EXACT_DUPLICATE",
        rule_name="同一来源不存在完全重复记录", group=group,
        actual=exact_duplicate_count, expected=0, difference=exact_duplicate_count,
        status="通过" if exact_duplicate_count == 0 else "需复核",
        notes="未发现重复记录" if exact_duplicate_count == 0 else f"发现{exact_duplicate_count}条完全重复记录",
        suggestion="核对 Step2 是否重复提取了同一表格或同一页。" if exact_duplicate_count else "",
    ))

    conflicts: list[pd.DataFrame] = []
    for _, rows in group.groupby(source_key, dropna=False, sort=False):
        values = pd.to_numeric(rows["数值"], errors="coerce").dropna().unique()
        if len(values) > 1:
            conflicts.append(rows)
    conflict_rows = pd.concat(conflicts, ignore_index=True) if conflicts else group.iloc[0:0]
    results.append(_result(
        check_type="数据规范", severity="错误", rule_id="SCHEMA_CONFLICTING_DUPLICATE",
        rule_name="同一来源同一指标不存在冲突值", group=group,
        actual=len(conflicts), expected=0, difference=len(conflicts),
        status="通过" if not conflicts else "未通过",
        notes="未发现同源冲突值" if not conflicts else _issue_summary(
            conflict_rows, ("来源工作表", "指标编码", "期间口径", "数值")
        ),
        involved_metrics=conflict_rows.get("指标编码", pd.Series(dtype=str)).astype(str).tolist(),
        source_pages=_source_pages(conflict_rows),
        suggestion="返回 Step2 检查重复行、跨页拼接或列错位。" if conflicts else "",
    ))
    return results


def _completeness_checks(group: pd.DataFrame, taxonomy: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    taxonomy_names = (
        taxonomy.fillna("").drop_duplicates(subset="指标编码", keep="last")
        .set_index("指标编码")["指标名称"].astype(str).to_dict()
    )
    present_groups = {str(value).strip() for value in group["_验证期间组"] if str(value).strip()}
    target_groups: list[str] = [CURRENT_PERIOD_GROUP] if present_groups else []

    for period_group in target_groups:
        for code in REQUIRED_CORE_METRICS:
            available = bool(_metric_values(group, code, period_group))
            metric_name = taxonomy_names.get(code, code)
            results.append(_result(
                check_type="完整性", severity="错误",
                rule_id=f"REQUIRED_CORE_METRIC:{code}",
                rule_name=f"关键指标完整性：{metric_name}", group=group,
                period=period_group, actual=1 if available else 0, expected=1,
                difference=0 if available else -1, status="通过" if available else "缺失",
                notes="已取得可用数值" if available else f"{period_group}缺少{metric_name}",
                involved_metrics=(code,), source_pages=_source_pages(group, (code,), period_group),
                suggestion="返回 Step2 检查是否漏提，或返回 Step3 检查指标映射。" if not available else "",
            ))
    return results


def _business_checks(group: pd.DataFrame, rules: pd.DataFrame) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    present_periods = {
        str(value).strip() for value in group["_验证期间组"] if str(value).strip()
    }
    periods = [CURRENT_PERIOD_GROUP] if CURRENT_PERIOD_GROUP in present_periods else []
    for _, rule in rules.iterrows():
        rule_id = str(rule.get("规则ID", "")).strip()
        if rule_id not in RULE_METRICS or str(rule.get("启用", "是")).strip() == "否":
            continue
        allowed_periods = {
            validation_period_group(item.strip())
            for item in str(rule.get("适用期间", "")).split("|")
            if item.strip()
        }
        target_periods = [period for period in periods if not allowed_periods or period in allowed_periods]
        tolerance = _safe_tolerance(rule.get("容差", 0))
        duplicate_rule = rule_id.startswith("DUPLICATE_")
        for period_group in target_periods:
            try:
                actual, expected = _evaluate_business_rule(group, rule_id, period_group)
                if actual is None or expected is None:
                    if duplicate_rule:
                        status, notes, suggestion = (
                            "不适用", "同一期间组仅发现一处有效披露，无法执行跨表一致性检查", ""
                        )
                    else:
                        status, notes, suggestion = (
                            "缺失", "缺少勾稽所需指标，或公式分母为零",
                            "返回 Step2/Step3 补齐公式依赖指标并核对数值。",
                        )
                    difference = None
                else:
                    difference = float(actual - expected)
                    status = "通过" if abs(difference) <= tolerance else "未通过"
                    notes = "检查通过" if status == "通过" else "披露值与系统计算值的差异超过容差"
                    suggestion = "" if status == "通过" else "核对来源页中的披露值、单位和资本构成明细。"
            except Exception as exc:
                actual, expected, difference = None, None, None
                status, notes, suggestion = "执行错误", str(exc), "检查规则配置或联系系统维护人员。"
            metrics = RULE_METRICS[rule_id]
            results.append(_result(
                check_type="跨表一致性" if duplicate_rule else "公式勾稽",
                severity="错误", rule_id=rule_id,
                rule_name=str(rule.get("规则名称", rule_id)), group=group,
                period=period_group, actual=actual, expected=expected,
                difference=difference, tolerance=tolerance, status=status,
                notes=notes, involved_metrics=metrics,
                source_pages=_source_pages(group, metrics, period_group), suggestion=suggestion,
            ))
    return results


def validate_standard_data(
    frame: pd.DataFrame,
    rules: pd.DataFrame,
    taxonomy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run first-stage STEP4 checks against canonical STEP3 narrow-table data."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    rule_frame = rules if isinstance(rules, pd.DataFrame) else pd.DataFrame()
    taxonomy_frame = taxonomy if isinstance(taxonomy, pd.DataFrame) else pd.DataFrame()
    if not taxonomy_frame.empty:
        taxonomy_frame = extend_taxonomy(taxonomy_frame)
    results: list[ValidationResult] = []
    for group in _entity_groups(frame):
        if not taxonomy_frame.empty:
            results.extend(_schema_checks(group, taxonomy_frame))
            results.extend(_completeness_checks(group, taxonomy_frame))
        if not rule_frame.empty:
            results.extend(_business_checks(group, rule_frame))
    return pd.DataFrame([item.to_dict() for item in results], columns=RESULT_COLUMNS)


def validation_status_summary(results: pd.DataFrame) -> pd.DataFrame:
    order = ("通过", "未通过", "缺失", "需复核", "不适用", "执行错误")
    counts = (
        results.get("status", pd.Series(dtype=str)).fillna("").astype(str).value_counts().to_dict()
    )
    return pd.DataFrame([{"状态": status, "数量": int(counts.get(status, 0))} for status in order])
