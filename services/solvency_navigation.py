from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


PRINT_ALL_LABEL = "一键显示全部（打印/导出）"
OVERVIEW_LEVEL = "行业整体偿付能力概览"
COMPANY_OVERVIEW_LEVEL = "关键偿付数据概览"
ACTUAL_CAPITAL_LEVEL = "实际资本数据对比"
MINIMUM_CAPITAL_LEVEL = "最低资本数据对比"
RECOGNIZED_ASSETS_LEVEL = "认可资产数据对比"
OPERATING_QUALITY_LEVEL = "主要经营指标对比"
APPENDIX_LEVEL = "附录"
MARKET_RISK_ASSET_SCATTER = "利率与权益价格风险占认可资产率气泡图"
CREDIT_RISK_ASSET_SCATTER = "利差与对手违约风险占认可资产率气泡图"
INSURANCE_RISK_LIABILITY_SCATTER = "寿险与非寿险保险风险占认可负债率气泡图"

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
# Keep these official colors available in the palette reference, but exclude
# them from automatic chart palettes.  Dark Purple is too close to Purple;
# Blue is too close to Light Blue when both are rendered in the same chart.
KPMG_AUTOMATIC_CHART_EXCLUDED_COLORS = frozenset({"#510DBC", "#76D2FF"})
KPMG_DEFAULT_COLORS = tuple(
    color
    for category in ("Primary Colors", "Accent Colors")
    for color in KPMG_CATEGORIES[category].values()
    if color.upper() not in KPMG_AUTOMATIC_CHART_EXCLUDED_COLORS
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
    "#AB0D82",
    "#098E7E",
    "#00C0AE",
    "#B497FF",
    "#63EBB2",
    "#FFA3DA",
    "#ED2124",
    "#F1C44D",
    "#269924",
)
# Prefer these official KPMG light/bright tones for filled chart areas.  Lines,
# reference rules, and highlighted series continue to use the saturated chart
# palette above so they remain visible on the light report background.
KPMG_LIGHT_CHART_COLORS = (
    "#ACEAFF",  # Light Blue
    "#FD349C",  # Pink
    "#B497FF",  # Light Purple
    "#FFA3DA",  # Light Pink
    "#63EBB2",  # Light Green
    "#00B8F5",  # Pacific Blue
    "#00C0AE",  # Green
    "#F1C44D",  # Amber
)
# Annual-report-style bright fill palette for non-key bar and stack charts.
# Deep KPMG Blue, Cobalt Blue and all Dark-series shades are intentionally
# excluded so dark-navy value labels remain clear without a white outline.
KPMG_BRIGHT_CHART_COLORS = (
    "#FFA3DA",  # Light Pink
    "#00B8F5",  # Pacific Blue
    "#FD349C",  # Pink
    "#269924",  # Positive Green
    "#ACEAFF",  # Light Blue
    "#B497FF",  # Light Purple
    "#63EBB2",  # Light Green
    "#00C0AE",  # Green
    "#F1C44D",  # Amber
)
# Report periods should progress from a light baseline to saturated recent
# periods while remaining consistent across every company panel.
KPMG_PERIOD_CHART_COLORS = (
    "#ACEAFF",  # Light Blue
    "#00B8F5",  # Pacific Blue
    "#B497FF",  # Light Purple
    "#FD349C",  # Pink
    "#00338D",  # KPMG Blue
)
# Bright variants remain legible when the same category color is reused by
# both a bar chart and a line chart (notably the Step 8 peer-group views).
KPMG_BRIGHT_SERIES_COLORS = (
    "#00B8F5",
    "#FD349C",
    "#00C0AE",
    "#269924",
    "#B497FF",
    "#FFA3DA",
    "#63EBB2",
    "#ACEAFF",
    "#F1C44D",
)
# Calm bright-blue hierarchy for four-level capital stacks.  This keeps related
# capital layers in one family while avoiding dark fills and white labels.
KPMG_CAPITAL_TIER_COLORS = (
    "#ACEAFF",  # Light Blue
    "#FFA3DA",  # Light Pink
    "#00B8F5",  # Pacific Blue
    "#B497FF",  # Light Purple
)
# Two-color capital-adequacy combination charts: Cobalt anchors the main
# capital amount while Pacific Blue keeps the second stack segment bright.
KPMG_CAPITAL_COMBO_COLORS = (
    "#1E49E2",  # Cobalt Blue
    "#00B8F5",  # Pacific Blue
)
# Quantified-risk stacks use blues for positive risk capital and greens for
# deductions.  Light Purple closes the positive stack without a heavy dark cap.
KPMG_QUANT_RISK_COLORS = (
    "#ACEAFF",  # Insurance risk (life)
    "#00B8F5",  # Insurance risk (non-life)
    "#FFA3DA",  # Market risk
    "#B497FF",  # Credit risk
    "#63EBB2",  # Diversification effect
    "#00C0AE",  # Loss-absorption effect
)


@dataclass(frozen=True)
class NavigationEntry:
    level_one: str
    level_two: str
    chart_name: str
    metric_codes: tuple[str, ...]
    requires_all: bool = False
    always_available: bool = False


COMPANY_NAVIGATION: tuple[NavigationEntry, ...] = (
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "偿付能力披露整体情况",
        "关键偿付数据概览",
        (
            "CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO", "ACTUAL_CAPITAL",
            "POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2",
            "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2",
            "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL", "MARKET_RISK_TO_QUANT_CAPITAL",
            "LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL", "RECOGNIZED_LIABILITIES",
        ),
        always_available=True,
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "核心及综合充足率",
        ("CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"),
        True,
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "综合充足率变化",
        ("ACTUAL_CAPITAL", "MINIMUM_CAPITAL", "COMBINED_SOLVENCY_RATIO"),
        True,
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "核心充足率变化",
        ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "MINIMUM_CAPITAL", "CORE_SOLVENCY_RATIO"),
        True,
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "核心资本占比",
        ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"),
        True,
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "计入核心资本的保单未来盈余/核心资本的比例",
        ("POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",),
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "保单未来盈余",
        "计入核心资本的保单未来盈余/核心资本的比例",
        ("POLICY_SURPLUS_CORE_TO_CORE_CAPITAL",),
    ),
    NavigationEntry(
        COMPANY_OVERVIEW_LEVEL,
        "资本充足率",
        "资本使用效率与核心资本占比气泡图",
        (
            "ACTUAL_CAPITAL", "RECOGNIZED_ASSETS", "REGISTERED_CAPITAL",
            "CORE_T1_CAPITAL", "CORE_T2_CAPITAL",
        ),
        True,
    ),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "资本使用效率", "核心资本/注册资本", ("CORE_CAPITAL_TO_REGISTERED_CAPITAL",)),
    NavigationEntry(COMPANY_OVERVIEW_LEVEL, "资本使用效率", "实际资本/认可资产率", ("ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS",)),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "行业资本分级", "资本规模与结构", ("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"), True),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "核心资本", "核心一级资本明细", (
        "FINANCIAL_STATEMENT_NET_ASSETS", "NON_RECOGNIZED_ASSET_BOOK_VALUE",
        "LONG_TERM_EQUITY_VALUATION_DIFFERENCE", "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
        "DEFERRED_TAX_ASSET_ADJUSTMENT", "AGRICULTURAL_CATASTROPHE_RISK_RESERVE",
        "POLICY_SURPLUS_CORE_T1", "QUALIFYING_CORE_T1_LIABILITY_CAPITAL",
        "OTHER_CORE_T1_ADJUSTMENT", "CORE_T1_CAPITAL",
    ), always_available=True),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "附属资本", "附属一级资本明细", (
        "ANC_T1_SUBORDINATED_TERM_DEBT", "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS",
        "ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT", "ANC_T1_DEFERRED_TAX_ASSET",
        "ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE", "POLICY_SURPLUS_ANC_T1",
        "OTHER_ANC_T1_CAPITAL", "ANC_T1_CAPITAL",
    ), always_available=True),
    NavigationEntry(
        ACTUAL_CAPITAL_LEVEL,
        "保单未来盈余",
        "计入各级资本的保单未来盈余构成占比",
        ("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2"),
        True,
    ),
    NavigationEntry(ACTUAL_CAPITAL_LEVEL, "存量保单盈利能力", "保单未来盈余/保险合同负债（存量保单盈利能力）", ("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES",)),
    NavigationEntry(
        MINIMUM_CAPITAL_LEVEL,
        "风险构成情况",
        "量化风险最低资本构成",
        (
            "INSURANCE_RISK_CAPITAL", "NON_LIFE_INSURANCE_RISK_CAPITAL",
            "MARKET_RISK_CAPITAL", "CREDIT_RISK_CAPITAL",
            "QUANT_RISK_DIVERSIFICATION_EFFECT", "CONTRACT_LOSS_ABSORPTION_EFFECT",
        ),
        True,
    ),
    NavigationEntry(
        MINIMUM_CAPITAL_LEVEL,
        "保险风险",
        "各类保险风险（寿）占比",
        (
            "LOSS_OCCURRENCE_RISK_CAPITAL", "SURRENDER_RISK_CAPITAL",
            "EXPENSE_RISK_CAPITAL", "LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT",
        ),
        True,
    ),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险", "保险风险（寿）/认可负债率", ("LIFE_INSURANCE_RISK_TO_LIABILITIES",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "保险风险", "保险风险（非寿）/认可负债率", ("NON_LIFE_INSURANCE_RISK_TO_LIABILITIES",)),
    NavigationEntry(
        MINIMUM_CAPITAL_LEVEL,
        "市场风险",
        "各类市场风险占比",
        (
            "INTEREST_RATE_RISK_CAPITAL", "EQUITY_RISK_CAPITAL",
            "REAL_ESTATE_RISK_CAPITAL", "OVERSEAS_FIXED_INCOME_RISK_CAPITAL",
            "OVERSEAS_EQUITY_RISK_CAPITAL", "FOREIGN_EXCHANGE_RISK_CAPITAL",
            "MARKET_RISK_DIVERSIFICATION_EFFECT", "MARKET_RISK_CAPITAL",
        ),
        True,
    ),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场风险", "利率风险/认可资产率", ("INTEREST_RATE_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "市场风险", "权益价格风险/认可资产率", ("EQUITY_RISK_TO_ASSETS",)),
    NavigationEntry(
        MINIMUM_CAPITAL_LEVEL,
        "信用风险",
        "各类信用风险占比",
        (
            "SPREAD_RISK_CAPITAL", "COUNTERPARTY_RISK_CAPITAL",
            "CREDIT_RISK_DIVERSIFICATION_EFFECT", "CREDIT_RISK_CAPITAL",
        ),
        True,
    ),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "信用风险", "利差风险/认可资产率", ("SPREAD_RISK_TO_ASSETS",)),
    NavigationEntry(MINIMUM_CAPITAL_LEVEL, "信用风险", "对手违约风险/认可资产率", ("COUNTERPARTY_RISK_TO_ASSETS",)),
    NavigationEntry(
        RECOGNIZED_ASSETS_LEVEL,
        "资产构成情况",
        "认可资产构成",
        (
            "CASH_LIQUID_ASSETS", "INVESTMENT_ASSETS",
            "SUBSIDIARY_JV_ASSOCIATE_EQUITY", "REINSURANCE_ASSETS",
            "RECEIVABLES_AND_PREPAYMENTS", "FIXED_ASSETS",
            "LAND_USE_RIGHTS", "SEPARATE_ACCOUNT_ASSETS",
            "OTHER_RECOGNIZED_ASSETS",
        ),
        True,
    ),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "业务质量指标", "签单保费与新业务利润率", ("SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN"), always_available=True),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "业务质量指标", "新业务价值与新业务价值率", ("NEW_BUSINESS_VALUE", "SIGNED_PREMIUM"), always_available=True),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "业务质量指标", "综合退保率", ("SURRENDER_RATE",), always_available=True),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "投资质量指标", "投资质量六指标雷达图", ("ROE", "ROA", "INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN", "THREE_YEAR_AVG_INVESTMENT_RETURN", "THREE_YEAR_AVG_COMPREHENSIVE_INVESTMENT_RETURN"), always_available=True),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "投资质量指标", "累计投资收益率与累计综合投资收益率", ("INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN"), always_available=True),
    NavigationEntry(OPERATING_QUALITY_LEVEL, "投资质量指标", "近三年平均投资收益率与综合投资收益率", ("INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN", "THREE_YEAR_AVG_INVESTMENT_RETURN", "THREE_YEAR_AVG_COMPREHENSIVE_INVESTMENT_RETURN"), always_available=True),
    NavigationEntry(APPENDIX_LEVEL, "重大融资信息", "增资发债信息统计", (), always_available=True),
)

INDUSTRY_QUANT_CHART = "量化风险最低资本构成（行业合计）"
INDUSTRY_NAVIGATION: tuple[NavigationEntry, ...] = (
    NavigationEntry(OVERVIEW_LEVEL, "偿付能力充足率整体分布", "综合偿付能力充足率", ("COMBINED_SOLVENCY_RATIO",)),
    NavigationEntry(OVERVIEW_LEVEL, "偿付能力充足率整体分布", "核心偿付能力充足率", ("CORE_SOLVENCY_RATIO",)),
    NavigationEntry(OVERVIEW_LEVEL, "资本使用效率", "注册资本/核心资本率", ("REGISTERED_CAPITAL_TO_CORE_CAPITAL",)),
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
        if entry.always_available or (
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
        if entry.always_available:
            continue
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
