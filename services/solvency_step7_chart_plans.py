from __future__ import annotations

from dataclasses import dataclass

from .solvency_navigation import COMPANY_NAVIGATION


SINGLE_METRIC_TREND = "single_metric_trend"
COMPANY_BAR_TREND = "company_bar_trend"
CAPITAL_AMOUNT_COMBO = "capital_amount_combo"
EFFECT_DIVERGING = "effect_diverging"
SOLVENCY_MATRIX = "solvency_matrix"
MARKET_CREDIT_MATRIX = "market_credit_matrix"


@dataclass(frozen=True)
class ChartPlan:
    kind: str
    description: str


SPECIAL_CHART_PLANS: dict[str, ChartPlan] = {
    "综合偿付能力充足率": ChartPlan(
        COMPANY_BAR_TREND,
        "按公司分面的跨期柱状图与数据点折线",
    ),
    "核心偿付能力充足率": ChartPlan(
        COMPANY_BAR_TREND,
        "按公司分面的跨期柱状图与数据点折线",
    ),
    "市场风险最低资本占比": ChartPlan(
        COMPANY_BAR_TREND,
        "按公司分面的跨期柱状图与数据点折线",
    ),
    "信用风险最低资本占比": ChartPlan(
        COMPANY_BAR_TREND,
        "按公司分面的跨期柱状图与数据点折线",
    ),
    "四级资本规模与结构": ChartPlan(
        CAPITAL_AMOUNT_COMBO,
        "四级资本金额、结构占比与实际资本总额",
    ),
    "风险分散效应": ChartPlan(
        EFFECT_DIVERGING,
        "风险分散效应跨期发散柱状图",
    ),
    "损失吸收效应": ChartPlan(
        EFFECT_DIVERGING,
        "损失吸收效应跨期发散柱状图",
    ),
    "偿付能力矩阵": ChartPlan(
        SOLVENCY_MATRIX,
        "核心与综合偿付能力充足率矩阵",
    ),
    "市场—信用风险矩阵": ChartPlan(
        MARKET_CREDIT_MATRIX,
        "市场与信用风险最低资本占比矩阵",
    ),
}


CHART_PLAN_REGISTRY: dict[str, ChartPlan] = {
    entry.chart_name: SPECIAL_CHART_PLANS.get(
        entry.chart_name,
        ChartPlan(SINGLE_METRIC_TREND, "跨期趋势图"),
    )
    for entry in COMPANY_NAVIGATION
}


def chart_plan_for(chart_name: str) -> ChartPlan:
    return CHART_PLAN_REGISTRY.get(
        str(chart_name),
        ChartPlan(SINGLE_METRIC_TREND, "跨期趋势图"),
    )
