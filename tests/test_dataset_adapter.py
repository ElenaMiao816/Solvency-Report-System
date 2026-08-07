from __future__ import annotations

import io
import unittest

import pandas as pd

from services.solvency_dataset_adapter import (
    append_derived_metrics,
    calculate_derived_values,
    calculate_industry_values,
    convert_external_workbook,
    read_standard_workbook,
)
from services.solvency_normalizer import STANDARD_COLUMNS


class DatasetAdapterTests(unittest.TestCase):
    def test_calculates_cross_derived_metrics_with_multiples(self):
        values = {
            "RECOGNIZED_ASSETS": 500.0,
            "RECOGNIZED_LIABILITIES": 300.0,
            "ACTUAL_CAPITAL": 200.0,
            "MINIMUM_CAPITAL": 100.0,
            "QUANT_RISK_CAPITAL": 90.0,
            "INSURANCE_RISK_CAPITAL": 40.0,
            "NON_LIFE_INSURANCE_RISK_CAPITAL": 5.0,
            "MARKET_RISK_CAPITAL": 30.0,
            "CREDIT_RISK_CAPITAL": 20.0,
            "QUANT_RISK_DIVERSIFICATION_EFFECT": -10.0,
            "CONTRACT_LOSS_ABSORPTION_EFFECT": -4.0,
            "INSURANCE_CONTRACT_LIABILITY": 250.0,
            "SEPARATE_ACCOUNT_LIABILITY": 50.0,
            "POLICY_SURPLUS_CORE_T1": 20.0,
            "POLICY_SURPLUS_CORE_T2": 10.0,
            "POLICY_SURPLUS_ANC_T1": 5.0,
            "POLICY_SURPLUS_ANC_T2": 1.0,
            "REGISTERED_CAPITAL": 50.0,
            "CORE_T1_CAPITAL": 120.0,
            "CORE_T2_CAPITAL": 20.0,
            "ANC_T1_CAPITAL": 50.0,
            "INTEREST_RATE_RISK_CAPITAL": 10.0,
            "EQUITY_RISK_CAPITAL": 8.0,
            "SPREAD_RISK_CAPITAL": 7.0,
            "COUNTERPARTY_RISK_CAPITAL": 6.0,
            "TOTAL_ASSETS": 600.0,
        }
        result = calculate_derived_values(values)
        self.assertAlmostEqual(result["ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS"], 0.4)
        self.assertAlmostEqual(result["FEATURE_FACTOR_IMPACT"], 9.0)
        self.assertAlmostEqual(result["FEATURE_FACTOR_CHECK"], 9.0 / 81.0)
        self.assertAlmostEqual(result["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"], 36.0 / 300.0)
        self.assertEqual(result["POLICY_SURPLUS_CORE_BAND"], "(0,20%]")

    def test_appends_derived_and_lineage_columns(self):
        rows = []
        base_values = {
            "ACTUAL_CAPITAL": 20.0,
            "RECOGNIZED_ASSETS": 100.0,
            "POLICY_SURPLUS_CORE_T1": 2.0,
            "POLICY_SURPLUS_CORE_T2": 0.0,
        }
        for code, value in base_values.items():
            rows.append({
                "公司": "测试人寿",
                "公司类型": "寿险",
                "同业分类": "小型公司",
                "报告年度": 2025,
                "报告季度": "Q4",
                "报告期": "2025Q4",
                "期间口径": "期末数",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
                "来源页码": "3",
                "来源文件": "测试.pdf",
                "来源工作表": "主要指标",
            })
        result = append_derived_metrics(pd.DataFrame(rows))
        ratio = result[result["指标编码"] == "ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS"].iloc[0]
        self.assertAlmostEqual(ratio["数值"], 0.2)
        self.assertEqual(ratio["单位"], "倍")
        self.assertEqual(ratio["来源类型"], "系统计算")
        self.assertEqual(ratio["期间口径"], "本季度末数")
        self.assertTrue(ratio["计算逻辑"])

    def test_reads_standard_headers_from_first_or_third_row(self):
        base = pd.DataFrame([{"公司": "测试人寿", "指标编码": "A", "指标名称": "指标A", "数值": 1}])
        for startrow in (0, 2):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                base.to_excel(writer, sheet_name="标准数据", index=False, startrow=startrow)
            result = read_standard_workbook(output.getvalue(), f"header_{startrow}.xlsx")
            self.assertEqual(result.iloc[0]["公司"], "测试人寿")
            self.assertEqual(list(result.columns), STANDARD_COLUMNS)

    def test_converts_external_wide_table_with_exact_logic_check(self):
        external = pd.DataFrame([{
            "分类": "小型公司",
            "公司": "测试人寿",
            "认可资产": 100.0,
            "实际资本": 20.0,
            "实际资本/认可资产": 0.2,
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            external.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame([
            {"指标编码": "RECOGNIZED_ASSETS", "指标名称": "认可资产", "一级模块": "主要指标", "二级模块": "偿付能力", "标准单位": "万元", "数据类型": "金额", "别名": ""},
            {"指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "一级模块": "主要指标", "二级模块": "偿付能力", "标准单位": "万元", "数据类型": "金额", "别名": ""},
        ])
        result = convert_external_workbook(
            output.getvalue(),
            "external.xlsx",
            taxonomy,
            {"测试人寿": "寿险"},
        )
        self.assertEqual(len(result.data), 7)
        ratio = result.data[result.data["指标编码"] == "ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS"].iloc[0]
        self.assertEqual(ratio["单位"], "倍")
        self.assertEqual(ratio["同业分类"], "小型公司")
        self.assertEqual(ratio["公司类型"], "寿险")
        self.assertEqual(ratio["原始公司名称"], "测试人寿")
        self.assertEqual(ratio["标准公司名称"], "测试人寿")
        self.assertTrue(ratio["公司统一编码"])
        self.assertEqual(result.logic_checks.iloc[0]["状态"], "通过")
        industry = result.data[result.data["公司"] == "行业合计"]
        self.assertEqual(len(industry), 4)
        self.assertEqual(result.sheet_summary.iloc[0]["行业指标数"], 4)

    def test_calculates_industry_totals_ratios_and_renamed_risk_metrics(self):
        totals = {
            "RECOGNIZED_ASSETS": 400.0,
            "RECOGNIZED_LIABILITIES": 320.0,
            "ACTUAL_CAPITAL": 110.0,
            "CORE_T1_CAPITAL": 60.0,
            "CORE_T2_CAPITAL": 12.0,
            "MINIMUM_CAPITAL": 60.0,
            "INSURANCE_CONTRACT_LIABILITY": 270.0,
            "SEPARATE_ACCOUNT_LIABILITY": 50.0,
            "POLICY_SURPLUS_CORE_T1": 20.0,
            "POLICY_SURPLUS_CORE_T2": 2.0,
            "POLICY_SURPLUS_ANC_T1": 5.0,
            "POLICY_SURPLUS_ANC_T2": 1.0,
        }
        calculated = calculate_industry_values(totals)
        self.assertAlmostEqual(calculated["COMBINED_SOLVENCY_RATIO"], 110.0 / 60.0 * 100)
        self.assertAlmostEqual(calculated["CORE_SOLVENCY_RATIO"], 72.0 / 60.0 * 100)
        self.assertAlmostEqual(calculated["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"], 28.0 / 270.0)
        self.assertAlmostEqual(calculated["RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL"], 400.0 / 60.0)
        self.assertEqual(calculated["SEPARATE_ACCOUNT_LIABILITY"], 50.0)

        external = pd.DataFrame([
            {
                "分类": "小型公司", "公司": "甲人寿", "认可资产": 100.0,
                "认可负债": 80.0, "实际资本": 20.0, "核心一级资本": 10.0,
                "核心二级资本": 2.0, "最低资本": 10.0,
                "寿险业务保险风险最低资本合计": 5.0,
            },
            {
                "分类": "大型公司", "公司": "乙人寿", "认可资产": 300.0,
                "认可负债": 240.0, "实际资本": 90.0, "核心一级资本": 50.0,
                "核心二级资本": 10.0, "最低资本": 50.0,
                "寿险业务保险风险最低资本合计": 10.0,
            },
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            external.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame([
            {"指标编码": code, "指标名称": name, "一级模块": "主要指标", "二级模块": "偿付能力", "标准单位": unit, "数据类型": data_type, "别名": ""}
            for code, name, unit, data_type in (
                ("COMBINED_SOLVENCY_RATIO", "综合偿付能力充足率", "%", "百分比"),
                ("CORE_SOLVENCY_RATIO", "核心偿付能力充足率", "%", "百分比"),
                ("RECOGNIZED_ASSETS", "认可资产", "万元", "金额"),
                ("RECOGNIZED_LIABILITIES", "认可负债", "万元", "金额"),
                ("ACTUAL_CAPITAL", "实际资本", "万元", "金额"),
                ("CORE_T1_CAPITAL", "核心一级资本", "万元", "金额"),
                ("CORE_T2_CAPITAL", "核心二级资本", "万元", "金额"),
                ("MINIMUM_CAPITAL", "最低资本", "万元", "金额"),
                ("INSURANCE_RISK_CAPITAL", "寿险业务保险风险最低资本合计", "万元", "金额"),
            )
        ])
        result = convert_external_workbook(
            output.getvalue(),
            "industry.xlsx",
            taxonomy,
            {"甲人寿": "寿险", "乙人寿": "寿险"},
        )
        industry = result.data[result.data["公司"] == "行业合计"].set_index("指标编码")
        self.assertAlmostEqual(industry.loc["COMBINED_SOLVENCY_RATIO", "数值"], 110.0 / 60.0 * 100)
        self.assertAlmostEqual(industry.loc["RECOGNIZED_ASSETS_TO_MINIMUM_CAPITAL", "数值"], 400.0 / 60.0)
        self.assertEqual(industry.loc["INDUSTRY_LIFE_INSURANCE_RISK", "指标名称"], "保险风险（寿）")
        self.assertEqual(industry.loc["INDUSTRY_LIFE_INSURANCE_RISK", "数值"], 15.0)
        self.assertEqual(industry.loc["INDUSTRY_LIFE_INSURANCE_RISK", "来源类型"], "系统计算")
        self.assertEqual(industry.loc["INDUSTRY_LIFE_INSURANCE_RISK", "指标属性"], "行业计算")
        self.assertIn("重命名", industry.loc["INDUSTRY_LIFE_INSURANCE_RISK", "备注"])


if __name__ == "__main__":
    unittest.main()
