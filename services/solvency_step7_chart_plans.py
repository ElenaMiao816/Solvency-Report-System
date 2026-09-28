from __future__ import annotations

from dataclasses import dataclass

from .solvency_navigation import (
    COMPANY_NAVIGATION,
)


SINGLE_METRIC_TREND = "single_metric_trend"
TREND_WITH_COMPANY_BARS = "trend_with_company_bars"
COMPANY_BAR_TREND = "company_bar_trend"
COMPANY_PERIOD_BAR = "company_period_bar"
SOLVENCY_RATIO_COMBO = "solvency_ratio_combo"
CAPITAL_AMOUNT_COMBO = "capital_amount_combo"
CAPITAL_RATIO_COMBO = "capital_ratio_combo"
CAPITAL_STRUCTURE_COMBO = "capital_structure_combo"
CAPITAL_EFFICIENCY_BUBBLE = "capital_efficiency_bubble"
COMPONENT_STACK = "component_stack"
EFFECT_DIVERGING = "effect_diverging"
SOLVENCY_MATRIX = "solvency_matrix"
MARKET_CREDIT_MATRIX = "market_credit_matrix"
RISK_RATIO_SCATTER = "risk_ratio_scatter"
KEY_METRICS_TABLE = "key_metrics_table"
FINANCING_TABLE = "financing_table"
PENDING_DEFINITION = "pending_definition"
QUALITY_AND_CAPITAL = "quality_and_capital"


@dataclass(frozen=True)
class ChartPlan:
    kind: str
    description: str


SPECIAL_CHART_PLANS: dict[str, ChartPlan] = {
    "关键偿付数据概览": ChartPlan(
        KEY_METRICS_TABLE,
        "最新报告期与去年同期关键偿付数据对照表",
    ),
    "核心及综合充足率": ChartPlan(
        SOLVENCY_RATIO_COMBO,
        "按公司和报告期展示综合偿付能力充足率柱形与核心偿付能力充足率折线",
    ),
    "综合充足率变化": ChartPlan(
        CAPITAL_RATIO_COMBO,
        "实际资本与最低资本堆叠柱组合综合偿付能力充足率折线",
    ),
    "核心充足率变化": ChartPlan(
        CAPITAL_RATIO_COMBO,
        "核心资本与最低资本堆叠柱组合核心偿付能力充足率折线",
    ),
    "核心资本占比": ChartPlan(
        CAPITAL_STRUCTURE_COMBO,
        "核心资本与附属资本堆叠柱组合核心资本占实际资本比例折线",
    ),
    "资本使用效率与核心资本占比气泡图": ChartPlan(
        CAPITAL_EFFICIENCY_BUBBLE,
        "最新报告期实际资本/注册资本、核心资本/注册资本与认可资产规模气泡图",
    ),
    "核心资本/注册资本": ChartPlan(
        SINGLE_METRIC_TREND,
        "多公司跨期折线图",
    ),
    "计入核心资本的保单未来盈余/核心资本的比例": ChartPlan(
        SINGLE_METRIC_TREND,
        "多公司跨期折线图",
    ),
    "资本规模与结构": ChartPlan(
        CAPITAL_AMOUNT_COMBO,
        "核心一级、核心二级、附属一级和附属二级资本堆叠图",
    ),
    "核心一级资本明细": ChartPlan(QUALITY_AND_CAPITAL, "最新报告期核心一级资本瀑布图"),
    "附属一级资本明细": ChartPlan(QUALITY_AND_CAPITAL, "最新报告期构成饼图"),
    "签单保费与新业务利润率": ChartPlan(QUALITY_AND_CAPITAL, "公司小图：签单保费柱形与新业务利润率折线"),
    "新业务价值与新业务价值率": ChartPlan(QUALITY_AND_CAPITAL, "公司小图：新业务价值柱形与价值率折线"),
    "综合退保率": ChartPlan(QUALITY_AND_CAPITAL, "全部所选公司的综合退保率折线对比"),
    "投资质量六指标雷达图": ChartPlan(QUALITY_AND_CAPITAL, "最新季度六项投资质量指标"),
    "累计投资收益率与累计综合投资收益率": ChartPlan(QUALITY_AND_CAPITAL, "累计投资收益率柱形与累计综合投资收益率折线"),
    "近三年平均投资收益率与综合投资收益率": ChartPlan(QUALITY_AND_CAPITAL, "近三年平均投资收益率柱形与平均综合投资收益率折线"),
    "计入各级资本的保单未来盈余构成占比": ChartPlan(
        COMPONENT_STACK,
        "计入各级资本的保单未来盈余构成占比堆叠图",
    ),
    "量化风险最低资本构成": ChartPlan(
        COMPONENT_STACK,
        "量化风险最低资本正负向构成堆叠图",
    ),
    "各类保险风险（寿）占比": ChartPlan(
        COMPONENT_STACK,
        "寿险保险风险最低资本构成堆叠图",
    ),
    "各类市场风险占比": ChartPlan(
        COMPONENT_STACK,
        "市场风险最低资本构成堆叠图",
    ),
    "各类信用风险占比": ChartPlan(
        COMPONENT_STACK,
        "信用风险最低资本构成堆叠图",
    ),
    "认可资产构成": ChartPlan(
        COMPONENT_STACK,
        "认可资产构成堆叠图（现金及流动性管理工具、投资资产、在子公司合营企业和联营企业中的权益、再保险资产、应收及预付款项、固定资产、土地使用权、独立账户资产、其他认可资产）",
    ),
    "增资发债信息统计": ChartPlan(
        FINANCING_TABLE,
        "重大融资信息表",
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
