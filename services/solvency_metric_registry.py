from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MetricDefinition:
    code: str
    name: str
    level1: str
    level2: str
    unit: str
    data_type: str
    attribute: str
    aliases: tuple[str, ...] = ()
    formula: str = ""
    dependencies: tuple[str, ...] = ()
    external_transform: str = "identity"

    def taxonomy_row(self) -> dict:
        return {
            "指标编码": self.code,
            "指标名称": self.name,
            "别名": "|".join(self.aliases),
            "一级模块": self.level1,
            "二级模块": self.level2,
            "标准单位": self.unit,
            "数据类型": self.data_type,
            "核心指标": "否",
            "允许期间口径": "本季度末数|上季度末数|期末数|期初数",
            "说明": self.formula,
        }


SUPPLEMENTAL_BASE_METRICS = (
    MetricDefinition("NON_LIFE_INSURANCE_RISK_CAPITAL", "非寿险业务保险风险最低资本合计", "最低资本", "保险风险", "万元", "金额", "披露"),
    MetricDefinition("QUANT_RISK_DIVERSIFICATION_EFFECT", "量化风险分散效应", "最低资本", "量化风险", "万元", "金额", "披露"),
    MetricDefinition("CONTRACT_LOSS_ABSORPTION_EFFECT", "特定类别保险合同损失吸收效应", "最低资本", "量化风险", "万元", "金额", "披露"),
    MetricDefinition("REAL_ESTATE_RISK_CAPITAL", "市场风险-房地产价格风险最低资本", "最低资本", "市场风险", "万元", "金额", "披露", ("房地产价格风险最低资本",)),
    MetricDefinition("OVERSEAS_FIXED_INCOME_RISK_CAPITAL", "市场风险-境外固定收益类资产价格风险最低资本", "最低资本", "市场风险", "万元", "金额", "披露", ("境外固定收益类资产价格风险最低资本",)),
    MetricDefinition("OVERSEAS_EQUITY_RISK_CAPITAL", "市场风险-境外权益类资产价格风险最低资本", "最低资本", "市场风险", "万元", "金额", "披露", ("境外权益类资产价格风险最低资本",)),
    MetricDefinition("FOREIGN_EXCHANGE_RISK_CAPITAL", "市场风险-汇率风险最低资本", "最低资本", "市场风险", "万元", "金额", "披露", ("汇率风险最低资本",)),
    MetricDefinition("MARKET_RISK_DIVERSIFICATION_EFFECT", "市场风险-风险分散效应", "最低资本", "市场风险", "万元", "金额", "披露"),
    MetricDefinition("CREDIT_RISK_DIVERSIFICATION_EFFECT", "信用风险-风险分散效应", "最低资本", "信用风险", "万元", "金额", "披露"),
    MetricDefinition("SEPARATE_ACCOUNT_LIABILITY", "独立账户负债", "经营指标", "资产负债", "万元", "金额", "披露"),
    MetricDefinition("POLICY_SURPLUS_CORE_T1", "计入核心一级资本的保单未来盈余", "实际资本", "保单未来盈余", "万元", "金额", "披露"),
    MetricDefinition("POLICY_SURPLUS_CORE_T2", "计入核心二级资本的保单未来盈余", "实际资本", "保单未来盈余", "万元", "金额", "披露"),
    MetricDefinition("POLICY_SURPLUS_ANC_T1", "计入附属一级资本的保单未来盈余", "实际资本", "保单未来盈余", "万元", "金额", "披露"),
    MetricDefinition("POLICY_SURPLUS_ANC_T2", "计入附属二级资本的保单未来盈余", "实际资本", "保单未来盈余", "万元", "金额", "披露"),
    MetricDefinition("REGISTERED_CAPITAL", "注册资本", "经营指标", "资本结构", "万元", "金额", "披露"),
)


DERIVED_METRICS = (
    MetricDefinition(
        "POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL", "保单未来盈余/核心资本", "派生指标", "保单未来盈余", "倍", "比率", "计算",
        formula="(计入核心一级资本的保单未来盈余+计入核心二级资本的保单未来盈余)/实际资本",
        dependencies=("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "ACTUAL_CAPITAL"),
    ),
    MetricDefinition(
        "POLICY_SURPLUS_CORE_BAND", "保单未来盈余/核心资本分布", "派生指标", "保单未来盈余", "", "文本", "计算",
        formula="按保单未来盈余/核心资本划分：<=0、(0,20%]、(20%,35%]、>35%",
        dependencies=("POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL",),
    ),
    MetricDefinition(
        "FEATURE_FACTOR_IMPACT", "考虑特征系数影响", "派生指标", "风险结构", "万元", "金额", "计算",
        formula="量化风险最低资本-(寿险保险风险+非寿险保险风险+市场风险+信用风险+量化风险分散效应+损失吸收效应)",
        dependencies=("QUANT_RISK_CAPITAL", "INSURANCE_RISK_CAPITAL", "NON_LIFE_INSURANCE_RISK_CAPITAL", "MARKET_RISK_CAPITAL", "CREDIT_RISK_CAPITAL", "QUANT_RISK_DIVERSIFICATION_EFFECT", "CONTRACT_LOSS_ABSORPTION_EFFECT"),
    ),
    MetricDefinition("ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS", "实际资本/认可资产", "派生指标", "资本效率", "倍", "比率", "计算", formula="实际资本/认可资产", dependencies=("ACTUAL_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES", "最低资本/认可负债", "派生指标", "资本效率", "倍", "比率", "计算", formula="最低资本/认可负债", dependencies=("MINIMUM_CAPITAL", "RECOGNIZED_LIABILITIES")),
    MetricDefinition(
        "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES", "保单未来盈余/保险合同负债", "派生指标", "保单未来盈余", "倍", "比率", "计算",
        formula="四类保单未来盈余之和/(保险合同负债+独立账户负债)",
        dependencies=("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2", "INSURANCE_CONTRACT_LIABILITY", "SEPARATE_ACCOUNT_LIABILITY"),
    ),
    MetricDefinition("RECOGNIZED_ASSETS_TO_ACTUAL_CAPITAL", "认可资产/实际资本", "派生指标", "资本效率", "倍", "比率", "计算", formula="认可资产/实际资本", dependencies=("RECOGNIZED_ASSETS", "ACTUAL_CAPITAL")),
    MetricDefinition("RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL", "认可资产/最低资本", "派生指标", "资本效率", "倍", "比率", "计算", formula="认可资产/最低资本", dependencies=("RECOGNIZED_ASSETS", "MINIMUM_CAPITAL")),
    MetricDefinition("RECOGNIZED_ASSETS_TO_REGISTERED_CAPITAL", "认可资产/注册资本", "派生指标", "资本结构", "倍", "比率", "计算", formula="认可资产/注册资本", dependencies=("RECOGNIZED_ASSETS", "REGISTERED_CAPITAL")),
    MetricDefinition("ACTUAL_CAPITAL_TO_REGISTERED_CAPITAL", "实际资本/注册资本", "派生指标", "资本结构", "倍", "比率", "计算", formula="实际资本/注册资本", dependencies=("ACTUAL_CAPITAL", "REGISTERED_CAPITAL")),
    MetricDefinition(
        "FEATURE_FACTOR_CHECK", "check特征系数", "数据质量", "勾稽检查", "倍", "校验", "校验",
        formula="考虑特征系数影响/(寿险保险风险+非寿险保险风险+市场风险+信用风险+量化风险分散效应+损失吸收效应)",
        dependencies=("FEATURE_FACTOR_IMPACT", "INSURANCE_RISK_CAPITAL", "NON_LIFE_INSURANCE_RISK_CAPITAL", "MARKET_RISK_CAPITAL", "CREDIT_RISK_CAPITAL", "QUANT_RISK_DIVERSIFICATION_EFFECT", "CONTRACT_LOSS_ABSORPTION_EFFECT"),
    ),
    MetricDefinition("LIFE_INSURANCE_RISK_TO_LIABILITIES", "保险风险（寿）/认可负债", "派生指标", "风险结构", "倍", "比率", "计算", formula="寿险业务保险风险最低资本合计/认可负债", dependencies=("INSURANCE_RISK_CAPITAL", "RECOGNIZED_LIABILITIES")),
    MetricDefinition("NON_LIFE_INSURANCE_RISK_TO_LIABILITIES", "保险风险（非寿）/认可负债", "派生指标", "风险结构", "倍", "比率", "计算", formula="非寿险业务保险风险最低资本合计/认可负债", dependencies=("NON_LIFE_INSURANCE_RISK_CAPITAL", "RECOGNIZED_LIABILITIES")),
    MetricDefinition("MARKET_RISK_TO_ASSETS", "市场风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="市场风险最低资本合计/认可资产", dependencies=("MARKET_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("CREDIT_RISK_TO_ASSETS", "信用风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="信用风险最低资本合计/认可资产", dependencies=("CREDIT_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition(
        "CORE_CAPITAL_TO_REGISTERED_CAPITAL", "核心资本/注册资本", "派生指标", "资本结构", "倍", "比率", "计算",
        formula="(核心一级资本+核心二级资本)/注册资本", dependencies=("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "REGISTERED_CAPITAL"),
    ),
    MetricDefinition("CORE_T1_POLICY_SURPLUS_SHARE", "核心一级资本中的保单未来盈余占比", "派生指标", "保单未来盈余", "倍", "比率", "计算", formula="计入核心一级资本的保单未来盈余/核心一级资本", dependencies=("POLICY_SURPLUS_CORE_T1", "CORE_T1_CAPITAL")),
    MetricDefinition("ANC_T1_POLICY_SURPLUS_SHARE", "附属一级资本中保单未来盈余占比", "派生指标", "保单未来盈余", "倍", "比率", "计算", formula="计入附属一级资本的保单未来盈余/附属一级资本", dependencies=("POLICY_SURPLUS_ANC_T1", "ANC_T1_CAPITAL")),
    MetricDefinition("INTEREST_RATE_RISK_TO_ASSETS", "利率风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="利率风险最低资本/认可资产", dependencies=("INTEREST_RATE_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("EQUITY_RISK_TO_ASSETS", "权益价格风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="权益价格风险最低资本/认可资产", dependencies=("EQUITY_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("SPREAD_RISK_TO_ASSETS", "利差风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="利差风险最低资本/认可资产", dependencies=("SPREAD_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("COUNTERPARTY_RISK_TO_ASSETS", "交易对手违约风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="交易对手违约风险最低资本/认可资产", dependencies=("COUNTERPARTY_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("TOTAL_ASSETS_TO_REGISTERED_CAPITAL", "总资产/注册资本", "派生指标", "资本结构", "倍", "比率", "计算", formula="总资产/注册资本", dependencies=("TOTAL_ASSETS", "REGISTERED_CAPITAL")),
)


INDUSTRY_METRICS = (
    MetricDefinition(
        "INDUSTRY_LIFE_INSURANCE_RISK", "保险风险（寿）", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的寿险业务保险风险最低资本合计求和",
        dependencies=("INSURANCE_RISK_CAPITAL",),
    ),
    MetricDefinition(
        "INDUSTRY_NON_LIFE_INSURANCE_RISK", "保险风险（非寿）", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的非寿险业务保险风险最低资本合计求和",
        dependencies=("NON_LIFE_INSURANCE_RISK_CAPITAL",),
    ),
    MetricDefinition(
        "INDUSTRY_MARKET_RISK", "市场风险", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的市场风险最低资本合计求和",
        dependencies=("MARKET_RISK_CAPITAL",),
    ),
    MetricDefinition(
        "INDUSTRY_CREDIT_RISK", "信用风险", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的信用风险最低资本合计求和",
        dependencies=("CREDIT_RISK_CAPITAL",),
    ),
    MetricDefinition(
        "INDUSTRY_CAPITALIZABLE_DIVERSIFICATION_EFFECT", "可资本化风险分散效应", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的量化风险分散效应求和",
        dependencies=("QUANT_RISK_DIVERSIFICATION_EFFECT",),
    ),
    MetricDefinition(
        "INDUSTRY_LOSS_ABSORPTION", "损失吸收", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的特定类别保险合同损失吸收效应求和",
        dependencies=("CONTRACT_LOSS_ABSORPTION_EFFECT",),
    ),
    MetricDefinition(
        "INDUSTRY_CONTROL_RISK", "控制风险", "行业指标", "最低资本构成", "万元", "金额", "行业计算",
        formula="所有有效公司的控制风险最低资本求和",
        dependencies=("CONTROL_RISK_CAPITAL",),
    ),
)

INDUSTRY_METRICS_BY_SOURCE_CODE = {
    definition.dependencies[0]: definition for definition in INDUSTRY_METRICS
}


ALL_CUSTOM_METRICS = SUPPLEMENTAL_BASE_METRICS + DERIVED_METRICS + INDUSTRY_METRICS
CUSTOM_METRICS_BY_CODE = {item.code: item for item in ALL_CUSTOM_METRICS}
CUSTOM_METRICS_BY_NAME = {item.name: item for item in ALL_CUSTOM_METRICS}


def extend_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Add source metrics needed by the CROSS data set without modifying the Excel dictionary."""
    additions = pd.DataFrame([item.taxonomy_row() for item in SUPPLEMENTAL_BASE_METRICS])
    existing_codes = set(taxonomy.get("指标编码", pd.Series(dtype=str)).astype(str))
    additions = additions[~additions["指标编码"].isin(existing_codes)]
    return pd.concat([taxonomy, additions], ignore_index=True).fillna("")
