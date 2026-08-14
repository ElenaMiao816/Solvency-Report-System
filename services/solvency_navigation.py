from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


PRINT_ALL_LABEL = "一键显示全部（打印/导出）"
OVERVIEW_LEVEL = "行业整体偿付能力概览"
COMPANY_OVERVIEW_LEVEL = "偿付能力概览"
ACTUAL_CAPITAL_LEVEL = "实际资本指标"
MINIMUM_CAPITAL_LEVEL = "最低资本指标"

KPMG_CATEGORIES = {
    "Primary Colors": {
        "KPMG Blue": "#00338D",
        "Cobalt Blue": "#1E49E2",
        "Dark Blue": "#0C233C",
        "Light Blue": "#ACEAFF",
        "Pacific Blue": "#00B8F5",
        "Purple": "#7213EA",
        "Pink": "#FD349C",
    },
    "Accent Colors": {
        "Blue": "#76D2FF",
        "Dark Purple": "#510DBC",
        "Light Purple": "#B497FF",
        "Dark Pink": "#AB0D82",
        "Light Pink": "#FFA3DA",
        "Dark Green": "#098E7E",
        "Green": "#00C0AE",
        "Light Green": "#63EBB2",
    },
    "Traffic Light": {
        "Red": "#ED2124",
        "Amber": "#F1C44D",
        "Positive Green": "#269924",
    },
}
KPMG_DEFAULT_COLORS = tuple(
    color
    for category in ("Primary Colors", "Accent Colors")
    for color in KPMG_CATEGORIES[category].values()
)
# Primary colors are always consumed before accents. Within the primary group,
# the saturated colors come first so dense line charts remain legible.
KPMG_PRIMARY_CHART_COLORS = (
    "#00338D",
    "#1E49E2",
    "#0C233C",
    "#7213EA",
    "#FD349C",
    "#00B8F5",
    "#ACEAFF",
)
KPMG_CHART_COLORS = (
    *KPMG_PRIMARY_CHART_COLORS,
    "#510DBC",
    "#AB0D82",
    "#098E7E",
    "#00C0AE",
    "#76D2FF",
    "#B497FF",
    "#63EBB2",
    "#FFA3DA",
    "#ED2124",
    "#F1C44D",
    "#269924",
)


@dataclass(frozen=True)
class NavigationEntry:
    level_one: str
    level_two: str
    chart_name: str
    metric_codes: tuple[str, ...]
    requires_all: bool = False


COMPANY_NAVIGATION: tuple[NavigationEntry, ...] = (
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "偿付能力充足率整体分布", "综合偿付能力充足率", ("COMBINED_SOLVENCY_RATIO",)),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "偿付能力充足率整体分布", "核心偿付能力充足率", ("CORE_SOLVENCY_RATIO",)),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "偿付能力充足率整体分布", "偿付能力矩阵", ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"), True),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "资本使用效率", "核心资本/注册资本率", ("CORE_CAPITAL_TO_REGISTERED_CAPITAL",)),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "资本使用效率", "实际资本/认可资产率", ("ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS",)),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "资本规模与结构", "四级资本规模与结构", ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"), True),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "保单未来盈余", "计入核心资本的保单未来盈余/核心资本的比例", ("POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",)),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "核心一级/附属一级资本中保单未来盈余占比情况", "核心一级资本中的保单未来盈余比例", ("CORE_T1_POLICY_SURPLUS_SHARE",)),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "核心一级/附属一级资本中保单未来盈余占比情况", "附属一级资本中的保单未来盈余比例", ("ANC_T1_POLICY_SURPLUS_SHARE",)),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "保单未来盈余/保险合同负债（存量保单盈利能力）", "保单未来盈余/保险合同负债（存量保单盈利能力）", ("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险最低资本情况", "寿险业务保险风险最低资本占比", ("LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险最低资本情况", "非寿险业务保险风险最低资本占比", ("NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场和信用风险最低资本情况", "市场风险最低资本占比", ("MARKET_RISK_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场和信用风险最低资本情况", "信用风险最低资本占比", ("CREDIT_RISK_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场和信用风险最低资本情况", "市场—信用风险矩阵", ("MARKET_RISK_TO_QUANT_CAPITAL", "CREDIT_RISK_TO_QUANT_CAPITAL"), True),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场风险最低资本占认可资产率", "利率风险/认可资产率", ("INTEREST_RATE_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场风险最低资本占认可资产率", "权益价格风险/认可资产率", ("EQUITY_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "信用风险最低资本占认可资产率", "利差风险/认可资产率", ("SPREAD_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "信用风险最低资本占认可资产率", "对手违约风险/认可资产率", ("COUNTERPARTY_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "风险分散效应和损失吸收", "风险分散效应", ("DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "风险分散效应和损失吸收", "损失吸收效应", ("LOSS_ABSORPTION_TO_QUANT_CAPITAL",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险最低资本占认可负债率", "保险风险（寿）/认可负债率", ("LIFE_INSURANCE_RISK_TO_LIABILITIES",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险最低资本占认可负债率", "保险风险（非寿）/认可负债率", ("NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",)),
)

INDUSTRY_QUANT_CHART = "量化风险最低资本构成（行业合计）"
INDUSTRY_NAVIGATION: tuple[NavigationEntry, ...] = (
    NavigationEntry(OVERVIEW_LEVEL, "偿付能力充足率整体分布", "综合偿付能力充足率", ("COMBINED_SOLVENCY_RATIO",)),
    NavigationEntry(OVERVIEW_LEVEL, "偿付能力充足率整体分布", "核心偿付能力充足率", ("CORE_SOLVENCY_RATIO",)),
    NavigationEntry(OVERVIEW_LEVEL, "资本使用效率", "核心资本/注册资本率", ("CORE_CAPITAL_TO_REGISTERED_CAPITAL",)),
    NavigationEntry(OVERVIEW_LEVEL, "资本使用效率", "实际资本/认可资产率", ("ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS",)),
    NavigationEntry(OVERVIEW_LEVEL, "量化风险最低资本构成", INDUSTRY_QUANT_CHART, (
        "INDUSTRY_LIFE_INSURANCE_RISK", "INDUSTRY_NON_LIFE_INSURANCE_RISK", "INDUSTRY_MARKET_RISK",
        "INDUSTRY_CREDIT_RISK", "INDUSTRY_CAPITALIZABLE_DIVERSIFICATION_EFFECT",
        "INDUSTRY_LOSS_ABSORPTION", "INDUSTRY_CONTROL_RISK",
    ), True),
)

COMPANY_FIRST_LEVELS = tuple(dict.fromkeys(entry.level_one for entry in COMPANY_NAVIGATION)) + (PRINT_ALL_LABEL,)
INDUSTRY_FIRST_LEVELS = (OVERVIEW_LEVEL, PRINT_ALL_LABEL)


def _available_entries(
    *,
    level_one: str | None = None,
    industry: bool = False,
    available_codes: Iterable[object] | None = None,
) -> list[NavigationEntry]:
    source = INDUSTRY_NAVIGATION if industry else COMPANY_NAVIGATION
    target_level = OVERVIEW_LEVEL if industry else level_one
    rows = [
        entry
        for entry in source
        if target_level is None or entry.level_one == target_level
    ]
    if available_codes is None:
        return rows
    available = {str(code).strip() for code in available_codes if str(code).strip()}
    return [
        entry
        for entry in rows
        if (
            all(code in available for code in entry.metric_codes)
            if entry.requires_all
            else any(code in available for code in entry.metric_codes)
        )
    ]


def first_levels_for_codes(
    available_codes: Iterable[object],
    *,
    industry: bool = False,
) -> tuple[str, ...]:
    rows = _available_entries(industry=industry, available_codes=available_codes)
    levels = tuple(dict.fromkeys(entry.level_one for entry in rows))
    if industry and levels:
        levels = (OVERVIEW_LEVEL,)
    return (*levels, PRINT_ALL_LABEL)


def second_levels(
    level_one: str,
    *,
    industry: bool = False,
    available_codes: Iterable[object] | None = None,
) -> list[str]:
    source = _available_entries(
        level_one=level_one,
        industry=industry,
        available_codes=available_codes,
    )
    return list(dict.fromkeys(entry.level_two for entry in source))


def chart_names(
    level_one: str,
    level_two: str = "全部",
    *,
    industry: bool = False,
    available_codes: Iterable[object] | None = None,
) -> list[str]:
    rows = _available_entries(
        level_one=level_one,
        industry=industry,
        available_codes=available_codes,
    )
    if level_two and level_two != "全部":
        rows = [entry for entry in rows if entry.level_two == level_two]
    return list(dict.fromkeys(entry.chart_name for entry in rows))


def resolve_chart_selection(
    level_one: str,
    level_two: str = "全部",
    selected_chart: str = "",
    *,
    industry: bool = False,
    available_codes: Iterable[object] | None = None,
) -> list[str]:
    """Resolve navigation state into the chart list that should be rendered."""
    options = chart_names(
        level_one,
        level_two,
        industry=industry,
        available_codes=available_codes,
    )
    if not level_two or level_two == "全部":
        return options
    selected = str(selected_chart or "").strip()
    if selected in options:
        return [selected]
    return options[:1]


def metric_codes_for_chart(chart_name: str, *, industry: bool = False) -> tuple[str, ...]:
    codes: list[str] = []
    source = INDUSTRY_NAVIGATION if industry else COMPANY_NAVIGATION
    for entry in source:
        if entry.chart_name == chart_name:
            codes.extend(entry.metric_codes)
    return tuple(dict.fromkeys(codes))


def navigation_labels_by_code() -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    for entry in COMPANY_NAVIGATION:
        for code in entry.metric_codes:
            labels.setdefault(code, (entry.level_one, entry.level_two))
    return labels


def apply_navigation_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the approved report-navigation labels without dropping other metrics."""
    result = frame.copy()
    if result.empty or "指标编码" not in result.columns:
        return result
    labels = navigation_labels_by_code()
    codes = result["指标编码"].fillna("").astype(str).str.strip()
    for code, (level_one, level_two) in labels.items():
        mask = codes.eq(code)
        if mask.any():
            result.loc[mask, "一级模块"] = level_one
            result.loc[mask, "二级模块"] = level_two
    return result


def ordered_available_codes(codes: Iterable[object]) -> list[str]:
    available = {str(code).strip() for code in codes if str(code).strip()}
    ordered = [
        code
        for entry in COMPANY_NAVIGATION
        for code in entry.metric_codes
        if code in available
    ]
    return list(dict.fromkeys(ordered))
