from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from .solvency_filing_catalog import extend_filing_taxonomy


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
    step5_wide_mapping: str = "是"

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
            "允许期间口径": "本季度末数|上季度末数|下季度末预测数|期末数|期初数",
            "说明": self.formula,
            "STEP5宽表映射": self.step5_wide_mapping,
        }


SUPPLEMENTAL_BASE_METRICS = (
    MetricDefinition("NON_LIFE_INSURANCE_RISK_CAPITAL", "非寿险业务保险风险最低资本合计", "最低资本", "保险风险", "万元", "金额", "披露"),
    MetricDefinition(
        "LOSS_OCCURRENCE_RISK_CAPITAL",
        "寿险业务保险风险-损失发生风险最低资本",
        "最低资本",
        "保险风险",
        "万元",
        "金额",
        "披露",
        ("损失发生风险最低资本",),
    ),
    MetricDefinition(
        "EXPENSE_RISK_CAPITAL",
        "寿险业务保险风险-费用风险最低资本",
        "最低资本",
        "保险风险",
        "万元",
        "金额",
        "披露",
        ("费用风险最低资本",),
    ),
    MetricDefinition(
        "LIFE_INSURANCE_RISK_DIVERSIFICATION_EFFECT",
        "寿险业务保险风险-风险分散效应",
        "最低资本",
        "保险风险",
        "万元",
        "金额",
        "披露",
        (
            "寿险业务保险风险间的相关性效应",
            "寿险业务保险风险分散效应",
        ),
    ),
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
    MetricDefinition(
        "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
        "投资性房地产公允价值增值（核心一级资本调整）",
        "实际资本", "核心一级资本调整", "万元", "金额", "披露",
        ("投资性房地产（包括保险公司以物权方式或通过子公司等方式持有的投资性房地产）的公允价值增值（扣除减值、折旧及所得税影响）",),
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "AGRICULTURAL_CATASTROPHE_RISK_RESERVE",
        "对农业保险提取的大灾风险准备金",
        "实际资本", "核心一级资本调整", "万元", "金额", "披露",
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "QUALIFYING_CORE_T1_LIABILITY_CAPITAL",
        "符合核心一级资本标准的负债类资本工具且按规定可计入核心一级资本的金额",
        "实际资本", "核心一级资本调整", "万元", "金额", "披露",
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "CORE_T2_PREFERRED_SHARES", "核心二级资本-优先股",
        "实际资本", "核心二级资本明细", "万元", "金额", "披露",
        ("优先股",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "OTHER_CORE_T2_CAPITAL", "其他核心二级资本",
        "实际资本", "核心二级资本明细", "万元", "金额", "披露",
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "CORE_T2_EXCESS_DEDUCTION", "核心二级资本超限额应扣除部分",
        "实际资本", "核心二级资本明细", "万元", "金额", "披露",
        ("2.4 减：超限额应扣除的部分",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_SUBORDINATED_TERM_DEBT", "附属一级资本-次级定期债务",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("次级定期债务",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", "附属一级资本-资本补充债券",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("资本补充债券",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT", "附属一级资本-可转换次级债",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("可转换次级债",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_DEFERRED_TAX_ASSET",
        "附属一级资本-递延所得税资产（由经营性亏损引起的递延所得税资产除外）",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("3.4 递延所得税资产（由经营性亏损引起的递延所得税资产除外）",),
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE",
        "投资性房地产公允价值增值可计入附属一级资本的金额",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("投资性房地产（包括保险公司以物权方式或通过子公司等方式持有的投资性房地产）公允价值增值可计入附属一级资本的金额（扣除减值、折旧及所得税影响）",),
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "OTHER_ANC_T1_CAPITAL", "其他附属一级资本",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T1_EXCESS_DEDUCTION", "附属一级资本超限额应扣除部分",
        "实际资本", "附属一级资本明细", "万元", "金额", "披露",
        ("3.8 减：超限额应扣除的部分",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "EMERGENCY_OTHER_ANC_T2_CAPITAL", "应急资本等其他附属二级资本",
        "实际资本", "附属二级资本明细", "万元", "金额", "披露",
        step5_wide_mapping="否",
    ),
    MetricDefinition(
        "ANC_T2_EXCESS_DEDUCTION", "附属二级资本超限额应扣除部分",
        "实际资本", "附属二级资本明细", "万元", "金额", "披露",
        ("4.3 减：超限额应扣除的部分",), step5_wide_mapping="否",
    ),
    MetricDefinition(
        "REGISTERED_CAPITAL", "注册资本", "经营指标", "资本结构", "万元", "金额", "披露",
        ("注册资本金", "公司注册资本", "注册资本（万元）", "注册资本（亿元）"),
    ),
)


DERIVED_METRICS = (
    MetricDefinition(
        "POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL", "保单未来盈余/核心资本", "派生指标", "保单未来盈余", "倍", "比率", "计算",
        formula="已披露的核心一级及核心二级保单未来盈余之和/实际资本；未披露层级不计入",
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
    MetricDefinition("CORE_T1_TO_ACTUAL_CAPITAL", "核心一级资本占比", "实际资本", "核心资本占比分布", "倍", "比率", "计算", formula="核心一级资本/实际资本", dependencies=("CORE_T1_CAPITAL", "ACTUAL_CAPITAL")),
    MetricDefinition("CORE_T2_TO_ACTUAL_CAPITAL", "核心二级资本占比", "实际资本", "核心资本占比分布", "倍", "比率", "计算", formula="核心二级资本/实际资本", dependencies=("CORE_T2_CAPITAL", "ACTUAL_CAPITAL")),
    MetricDefinition("ANC_T1_TO_ACTUAL_CAPITAL", "附属一级资本占比", "实际资本", "附属资本占比分布", "倍", "比率", "计算", formula="附属一级资本/实际资本", dependencies=("ANC_T1_CAPITAL", "ACTUAL_CAPITAL")),
    MetricDefinition("ANC_T2_TO_ACTUAL_CAPITAL", "附属二级资本占比", "实际资本", "附属资本占比分布", "倍", "比率", "计算", formula="附属二级资本/实际资本", dependencies=("ANC_T2_CAPITAL", "ACTUAL_CAPITAL")),
    MetricDefinition(
        "POLICY_SURPLUS_CORE_TO_CORE_CAPITAL", "计入核心资本的保单未来盈余/核心资本的比例", "实际资本", "保单未来盈余", "倍", "比率", "计算",
        formula="(已披露的核心一级保单未来盈余+已披露的核心二级保单未来盈余)/(核心一级资本+核心二级资本)；核心二级未披露时仅以核心一级保单未来盈余为分子",
        dependencies=("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
    ),
    MetricDefinition("MINIMUM_CAPITAL_TO_RECOGNIZED_LIABILITIES", "最低资本/认可负债", "派生指标", "资本效率", "倍", "比率", "计算", formula="最低资本/认可负债", dependencies=("MINIMUM_CAPITAL", "RECOGNIZED_LIABILITIES")),
    MetricDefinition(
        "POLICY_SURPLUS_TO_INSURANCE_LIABILITIES", "保单未来盈余/保险合同负债", "派生指标", "保单未来盈余", "倍", "比率", "计算",
        formula="已披露的各级资本保单未来盈余之和/保险合同负债；未披露层级不计入",
        dependencies=("POLICY_SURPLUS_CORE_T1", "POLICY_SURPLUS_CORE_T2", "POLICY_SURPLUS_ANC_T1", "POLICY_SURPLUS_ANC_T2", "INSURANCE_CONTRACT_LIABILITY"),
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
    MetricDefinition("LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL", "寿险业务保险风险最低资本占比", "最低资本", "保险风险最低资本情况", "倍", "比率", "计算", formula="寿险业务保险风险最低资本合计/量化风险最低资本", dependencies=("INSURANCE_RISK_CAPITAL", "QUANT_RISK_CAPITAL")),
    MetricDefinition("NON_LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL", "非寿险业务保险风险最低资本占比", "最低资本", "保险风险最低资本情况", "倍", "比率", "计算", formula="非寿险业务保险风险最低资本合计/量化风险最低资本", dependencies=("NON_LIFE_INSURANCE_RISK_CAPITAL", "QUANT_RISK_CAPITAL")),
    MetricDefinition("MARKET_RISK_TO_QUANT_CAPITAL", "市场风险最低资本占比", "最低资本", "市场和信用风险最低资本情况", "倍", "比率", "计算", formula="市场风险最低资本合计/量化风险最低资本", dependencies=("MARKET_RISK_CAPITAL", "QUANT_RISK_CAPITAL")),
    MetricDefinition("CREDIT_RISK_TO_QUANT_CAPITAL", "信用风险最低资本占比", "最低资本", "市场和信用风险最低资本情况", "倍", "比率", "计算", formula="信用风险最低资本合计/量化风险最低资本", dependencies=("CREDIT_RISK_CAPITAL", "QUANT_RISK_CAPITAL")),
    MetricDefinition("DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL", "风险分散效应最低资本占比", "最低资本", "风险分散效应和损失吸收", "倍", "比率", "计算", formula="量化风险分散效应/量化风险最低资本", dependencies=("QUANT_RISK_DIVERSIFICATION_EFFECT", "QUANT_RISK_CAPITAL")),
    MetricDefinition("LOSS_ABSORPTION_TO_QUANT_CAPITAL", "损失吸收效应最低资本占比", "最低资本", "风险分散效应和损失吸收", "倍", "比率", "计算", formula="特定类别保险合同损失吸收效应/量化风险最低资本", dependencies=("CONTRACT_LOSS_ABSORPTION_EFFECT", "QUANT_RISK_CAPITAL")),
    MetricDefinition("MARKET_RISK_TO_ASSETS", "市场风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="市场风险最低资本合计/认可资产", dependencies=("MARKET_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition("CREDIT_RISK_TO_ASSETS", "信用风险/认可资产", "派生指标", "风险结构", "倍", "比率", "计算", formula="信用风险最低资本合计/认可资产", dependencies=("CREDIT_RISK_CAPITAL", "RECOGNIZED_ASSETS")),
    MetricDefinition(
        "CORE_CAPITAL_TO_REGISTERED_CAPITAL", "核心资本/注册资本", "派生指标", "资本结构", "倍", "比率", "计算",
        formula="(核心一级资本+核心二级资本)/注册资本", dependencies=("CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "REGISTERED_CAPITAL"),
    ),
    MetricDefinition(
        "REGISTERED_CAPITAL_TO_CORE_CAPITAL", "注册资本/核心资本率", "派生指标", "资本结构", "倍", "比率", "计算",
        formula="注册资本/(核心一级资本+核心二级资本)", dependencies=("REGISTERED_CAPITAL", "CORE_T1_CAPITAL", "CORE_T2_CAPITAL"),
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


METRIC_ALIAS_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "QUANT_RISK_CAPITAL": (
        "可资本化风险最低资本",
        "量化风险最低资本（考虑特征系数后）",
    ),
    "ADDITIONAL_CAPITAL": ("附加资本合计",),
    "MINIMUM_CAPITAL": ("最低资本合计", "最低资本总额"),
    "ANC_T1_DEFERRED_TAX_ASSET": (
        "21 递延所得税资产（由经营性亏损引起的递延所得税资产除外）",
        "递延所得税资产（由经营性亏损引起的递延所得税资产除外）-附属一级资本",
    ),
}


# These capital components are present as explicit columns in CROSS wide
# workbooks. Keep the remaining STEP3-only filing rows out of STEP5 mapping.
STEP5_CROSS_CAPITAL_DETAIL_CODES = {
    "NON_RECOGNIZED_ASSET_BOOK_VALUE",
    "LONG_TERM_EQUITY_VALUATION_DIFFERENCE",
    "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
    "DEFERRED_TAX_ASSET_ADJUSTMENT",
    "QUALIFYING_CORE_T1_LIABILITY_CAPITAL",
    "OTHER_CORE_T1_ADJUSTMENT",
    "ANC_T1_SUBORDINATED_TERM_DEBT",
    "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS",
    "ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT",
    "ANC_T1_DEFERRED_TAX_ASSET",
    "ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE",
    "OTHER_ANC_T1_CAPITAL",
}


METRIC_PERIOD_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "INVESTMENT_RETURN": ("本季度数", "近三年平均"),
    "COMPREHENSIVE_INVESTMENT_RETURN": ("本季度数", "近三年平均"),
    "NET_ASSETS": (
        "本季度末数",
        "上季度末数",
        "下季度末预测数",
        "期末数",
        "期初数",
    ),
}


def _merge_pipe_values(current: object, additions: tuple[str, ...]) -> str:
    values = [
        item.strip()
        for item in str(current or "").split("|")
        if item.strip()
    ]
    values.extend(item for item in additions if item and item not in values)
    return "|".join(values)


def extend_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Return the complete system taxonomy used by STEP3 and STEP4."""
    additions = pd.DataFrame([
        item.taxonomy_row()
        for item in (*SUPPLEMENTAL_BASE_METRICS, *DERIVED_METRICS)
    ])
    existing_codes = set(taxonomy.get("指标编码", pd.Series(dtype=str)).astype(str))
    additions = additions[~additions["指标编码"].isin(existing_codes)]
    result = pd.concat([taxonomy, additions], ignore_index=True).fillna("")
    for code, aliases in METRIC_ALIAS_EXTENSIONS.items():
        mask = result["指标编码"].astype(str).eq(code)
        if mask.any():
            result.loc[mask, "别名"] = result.loc[mask, "别名"].map(
                lambda value: _merge_pipe_values(value, aliases)
            )
    for code, periods in METRIC_PERIOD_EXTENSIONS.items():
        mask = result["指标编码"].astype(str).eq(code)
        if mask.any():
            result.loc[mask, "允许期间口径"] = result.loc[mask, "允许期间口径"].map(
                lambda value: _merge_pipe_values(value, periods)
            )
    detail_mask = result["指标编码"].astype(str).isin(STEP5_CROSS_CAPITAL_DETAIL_CODES)
    result.loc[detail_mask, "STEP5宽表映射"] = "是"
    actual_capital_mask = result["一级模块"].astype(str).eq("实际资本")
    if actual_capital_mask.any():
        result.loc[actual_capital_mask, "允许期间口径"] = result.loc[
            actual_capital_mask, "允许期间口径"
        ].map(
            lambda value: _merge_pipe_values(value, ("本季度末数", "上季度末数"))
        )
    return extend_filing_taxonomy(result)
