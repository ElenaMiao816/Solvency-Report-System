from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    rule_id: str
    rule_name: str
    period: str
    actual: float | None
    expected: float | None
    difference: float | None
    tolerance: float
    status: str
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_validation_rules(path: str | Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name="勾稽规则", header=2).fillna("")


def _metric_value(frame: pd.DataFrame, code: str, period: str):
    rows = frame[(frame["指标编码"] == code) & (frame["期间口径"] == period)]
    if rows.empty:
        return np.nan
    values = pd.to_numeric(rows["数值"], errors="coerce").dropna()
    return values.iloc[0] if not values.empty else np.nan


def validate_standard_data(frame: pd.DataFrame, rules: pd.DataFrame) -> pd.DataFrame:
    results: list[ValidationResult] = []
    periods = sorted({str(item) for item in frame["期间口径"].dropna() if str(item).strip()})
    rule_functions = {
        "ACTUAL_CAPITAL_COMPONENTS": lambda p: (
            _metric_value(frame, "ACTUAL_CAPITAL", p),
            sum(_metric_value(frame, code, p) for code in ["CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"]),
        ),
        "ACTUAL_CAPITAL_BALANCE": lambda p: (
            _metric_value(frame, "ACTUAL_CAPITAL", p),
            _metric_value(frame, "RECOGNIZED_ASSETS", p) - _metric_value(frame, "RECOGNIZED_LIABILITIES", p),
        ),
        "CORE_SURPLUS": lambda p: (
            _metric_value(frame, "CORE_SOLVENCY_SURPLUS", p),
            _metric_value(frame, "CORE_T1_CAPITAL", p) + _metric_value(frame, "CORE_T2_CAPITAL", p) - _metric_value(frame, "MINIMUM_CAPITAL", p),
        ),
        "COMBINED_SURPLUS": lambda p: (
            _metric_value(frame, "COMBINED_SOLVENCY_SURPLUS", p),
            _metric_value(frame, "ACTUAL_CAPITAL", p) - _metric_value(frame, "MINIMUM_CAPITAL", p),
        ),
        "CORE_RATIO": lambda p: (
            _metric_value(frame, "CORE_SOLVENCY_RATIO", p),
            (_metric_value(frame, "CORE_T1_CAPITAL", p) + _metric_value(frame, "CORE_T2_CAPITAL", p)) / _metric_value(frame, "MINIMUM_CAPITAL", p) * 100,
        ),
        "COMBINED_RATIO": lambda p: (
            _metric_value(frame, "COMBINED_SOLVENCY_RATIO", p),
            _metric_value(frame, "ACTUAL_CAPITAL", p) / _metric_value(frame, "MINIMUM_CAPITAL", p) * 100,
        ),
        "MINIMUM_CAPITAL": lambda p: (
            _metric_value(frame, "MINIMUM_CAPITAL", p),
            _metric_value(frame, "QUANT_RISK_CAPITAL", p) + _metric_value(frame, "CONTROL_RISK_CAPITAL", p) + _metric_value(frame, "ADDITIONAL_CAPITAL", p),
        ),
    }

    for _, rule in rules.iterrows():
        rule_id = str(rule.get("规则ID", "")).strip()
        if rule_id not in rule_functions or str(rule.get("启用", "是")).strip() == "否":
            continue
        tolerance = float(rule.get("容差", 0) or 0)
        allowed_periods = [item.strip() for item in str(rule.get("适用期间", "")).split("|") if item.strip()]
        target_periods = [period for period in periods if not allowed_periods or period in allowed_periods]
        for period in target_periods:
            try:
                actual, expected = rule_functions[rule_id](period)
                if pd.isna(actual) or pd.isna(expected):
                    status, difference, notes = "缺失", None, "缺少勾稽所需指标"
                else:
                    difference = float(actual - expected)
                    status = "通过" if abs(difference) <= tolerance else "未通过"
                    notes = "" if status == "通过" else "差异超过容差"
            except Exception as exc:
                actual, expected, difference, status, notes = None, None, None, "错误", str(exc)
            results.append(ValidationResult(
                rule_id=rule_id,
                rule_name=str(rule.get("规则名称", rule_id)),
                period=period,
                actual=None if actual is None or pd.isna(actual) else float(actual),
                expected=None if expected is None or pd.isna(expected) else float(expected),
                difference=difference,
                tolerance=tolerance,
                status=status,
                notes=notes,
            ))
    return pd.DataFrame([item.to_dict() for item in results])

