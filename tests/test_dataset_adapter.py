from __future__ import annotations

import io
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from services.solvency_dataset_adapter import (
    add_missing_derived_metrics,
    append_derived_metrics,
    calculate_derived_values,
    calculate_industry_values,
    convert_external_workbook,
    read_standard_workbook,
    standard_workbook_bytes,
)
from services.solvency_normalizer import NARROW_TABLE_COLUMNS, STANDARD_COLUMNS
from services.solvency_normalizer import load_taxonomy
from services.solvency_metric_registry import extend_taxonomy


CROSS_WORKBOOK = Path(r"F:\CROSS\数据包\CROSS汇总表24Q4&25Q2Q4_0920.xlsx")


class DatasetAdapterTests(unittest.TestCase):
    def test_capital_detail_blanks_become_zero_when_section_has_other_values(self):
        taxonomy_path = Path(__file__).parents[1] / "config" / "solvency_taxonomy.xlsx"
        taxonomy = extend_taxonomy(load_taxonomy(taxonomy_path))
        codes = (
            "NET_ASSETS", "CORE_T1_CAPITAL", "NON_RECOGNIZED_ASSET_BOOK_VALUE",
            "POLICY_SURPLUS_CORE_T1", "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
            "ANC_T1_CAPITAL", "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS", "OTHER_ANC_T1_CAPITAL",
        )
        labels = {
            code: taxonomy.loc[taxonomy["指标编码"].eq(code), "指标名称"].iloc[0]
            for code in codes
        }
        def source_row(
            company, core_total, core_property, anc_total, anc_other, *,
            core_policy=5, anc_bonds=20, net_assets=100, non_recognized=-10,
        ):
            values = {
                "NET_ASSETS": net_assets, "CORE_T1_CAPITAL": core_total,
                "NON_RECOGNIZED_ASSET_BOOK_VALUE": non_recognized,
                "POLICY_SURPLUS_CORE_T1": core_policy,
                "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT": core_property,
                "ANC_T1_CAPITAL": anc_total,
                "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS": anc_bonds,
                "OTHER_ANC_T1_CAPITAL": anc_other,
            }
            return {"分类": "测试", "公司": company, **{labels[code]: value for code, value in values.items()}}

        source = pd.DataFrame([
            source_row("甲人寿", 95, None, 20, None),
            source_row("乙人寿", 95, None, 100, None),
            source_row("丙人寿", 95, "-", 20, None),
            source_row("丁人寿", 95, None, 0, None, anc_bonds=None),
            source_row("戊人寿", 95, None, 100, None, anc_bonds=0),
            source_row("己人寿", 99996, None, 20, None, net_assets=100000),
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="2025Q4", index=False)
        result = convert_external_workbook(output.getvalue(), "capital.xlsx", taxonomy)
        converted = result.data

        def rows(company, code):
            return converted.loc[converted["原始公司名称"].eq(company) & converted["指标编码"].eq(code)]

        for company in ("甲人寿", "乙人寿", "丁人寿", "己人寿"):
            inferred = rows(company, "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT")
            self.assertEqual(len(inferred), 1)
            self.assertEqual(float(inferred.iloc[0]["数值"]), 0)
            self.assertEqual(inferred.iloc[0]["来源类型"], "宽表空白推定")
            self.assertEqual(inferred.iloc[0]["指标属性"], "推定")
            self.assertEqual(inferred.iloc[0]["披露状态"], "推定零值")
            self.assertTrue(pd.isna(inferred.iloc[0]["原始披露值"]))
        self.assertTrue(rows("丙人寿", "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT").empty)
        self.assertEqual(float(rows("甲人寿", "OTHER_ANC_T1_CAPITAL").iloc[0]["数值"]), 0)
        self.assertTrue(rows("乙人寿", "OTHER_ANC_T1_CAPITAL").empty)
        self.assertTrue(rows("丁人寿", "OTHER_ANC_T1_CAPITAL").empty)
        self.assertTrue(rows("戊人寿", "OTHER_ANC_T1_CAPITAL").empty)
        self.assertEqual(result.sheet_summary.iloc[0]["资本明细与合计不一致组数"], 1)
        self.assertTrue(any("相关空白保持未披露" in warning for warning in result.warnings))

    @unittest.skipUnless(CROSS_WORKBOOK.exists(), "Local CROSS workbook unavailable")
    def test_real_cross_workbook_maps_capital_details_and_investment_scopes(self):
        taxonomy_path = Path(__file__).parents[1] / "config" / "solvency_taxonomy.xlsx"
        taxonomy = extend_taxonomy(load_taxonomy(taxonomy_path))
        result = convert_external_workbook(CROSS_WORKBOOK.read_bytes(), CROSS_WORKBOOK.name, taxonomy)

        self.assertEqual(set(result.sheet_summary["来源工作表"]), {"2024Q4", "2025Q2", "2025Q4"})
        self.assertEqual(result.sheet_summary["公司数"].sum(), 223)
        mapped = result.mapping_summary.set_index("来源字段")
        self.assertEqual(mapped.loc["投资收益率（累计数）", "指标编码"], "INVESTMENT_RETURN")
        self.assertEqual(mapped.loc["投资收益率（当季数）", "指标编码"], "INVESTMENT_RETURN")
        self.assertEqual(mapped.loc["递延所得税资产（由经营性亏损引起的递延所得税资产除外）-附属一级资本", "指标编码"], "ANC_T1_DEFERRED_TAX_ASSET")
        self.assertEqual(mapped.loc["资本补充债券", "指标编码"], "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS")
        self.assertEqual(mapped.loc["递延所得税资产（由经营性亏损引起的递延所得税资产除外）", "指标编码"], "DEFERRED_TAX_ASSET_ADJUSTMENT")

        source = pd.read_excel(CROSS_WORKBOOK, sheet_name="2025Q4", nrows=1)
        company = source.iloc[0]["公司"]
        rows = result.data.loc[
            result.data["原始公司名称"].eq(company)
            & result.data["报告期"].eq("2025Q4")
        ]
        def amount(code, scope):
            match = rows.loc[rows["指标编码"].eq(code) & rows["期间口径"].eq(scope)]
            self.assertEqual(len(match), 1, (code, scope))
            return float(match.iloc[0]["数值"])

        self.assertAlmostEqual(amount("INVESTMENT_RETURN", "本年累计数"), 3.9)
        self.assertAlmostEqual(amount("COMPREHENSIVE_INVESTMENT_RETURN", "本年累计数"), 3.61)
        self.assertAlmostEqual(amount("INVESTMENT_RETURN", "本季度数"), 1.49)
        self.assertAlmostEqual(amount("COMPREHENSIVE_INVESTMENT_RETURN", "本季度数"), 0.58)
        self.assertAlmostEqual(amount("NON_RECOGNIZED_ASSET_BOOK_VALUE", "本季度末数"), 463.01)

        core = result.data.loc[
            result.data["原始公司名称"].eq("中邮人寿")
            & result.data["报告期"].eq("2025Q4")
            & result.data["指标编码"].eq("CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT")
        ]
        self.assertEqual(len(core), 1)
        self.assertEqual(float(core.iloc[0]["数值"]), 0)
        self.assertEqual(core.iloc[0]["来源类型"], "宽表空白推定")
        ancillary = result.data.loc[
            result.data["原始公司名称"].eq("中邮人寿")
            & result.data["报告期"].eq("2025Q4")
            & result.data["指标编码"].eq("OTHER_ANC_T1_CAPITAL")
        ]
        self.assertTrue(ancillary.empty)

    def test_external_investment_scopes_preserve_percentage_points(self):
        source = pd.DataFrame([{
            "分类": "小型公司", "公司": "测试人寿",
            "投资收益率": 3.9,
            "投资收益率（累计数）": "3.9%",
            "投资收益率（当季数）": 0.0149,
            "近三年平均投资收益率": 0.052,
            "近三年平均综合投资收益率": "6.3%",
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame([
            {
                "指标编码": "INVESTMENT_RETURN", "指标名称": "投资收益率",
                "一级模块": "经营指标", "二级模块": "投资收益率",
                "标准单位": "%", "数据类型": "比例", "别名": "",
                "STEP5宽表映射": "是",
            },
            {
                "指标编码": "COMPREHENSIVE_INVESTMENT_RETURN", "指标名称": "综合投资收益率",
                "一级模块": "经营指标", "二级模块": "综合投资收益率",
                "标准单位": "%", "数据类型": "比例", "别名": "",
                "STEP5宽表映射": "是",
            },
        ])
        result = convert_external_workbook(output.getvalue(), "rates.xlsx", taxonomy)
        actual = result.data.loc[result.data["指标编码"].eq("INVESTMENT_RETURN")]
        values = dict(zip(actual["期间口径"], actual["数值"]))
        self.assertAlmostEqual(values["本季度末数"], 3.9)
        self.assertAlmostEqual(values["本年累计数"], 3.9)
        self.assertAlmostEqual(values["本季度数"], 1.49)
        self.assertAlmostEqual(values["近三年平均"], 5.2)
        comprehensive = result.data.loc[
            result.data["指标编码"].eq("COMPREHENSIVE_INVESTMENT_RETURN")
        ]
        self.assertEqual(comprehensive.iloc[0]["期间口径"], "近三年平均")
        self.assertAlmostEqual(float(comprehensive.iloc[0]["数值"]), 6.3)
        self.assertNotIn("源比例小数", result.mapping_summary.columns)

    def test_external_business_quality_columns_are_current_quarter_values(self):
        source = pd.DataFrame([{
            "分类": "银行系", "公司": "工银安盛",
            "综合退保率": 0.0078,
            "签单保费": 3296829,
            "新业务利润率": 0.0744,
            "新业务价值": 82603,
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="2026Q2", index=False)
        taxonomy_path = Path(__file__).parents[1] / "config" / "solvency_taxonomy.xlsx"
        taxonomy = extend_taxonomy(load_taxonomy(taxonomy_path))

        result = convert_external_workbook(output.getvalue(), "business.xlsx", taxonomy)
        rows = result.data.set_index("指标编码")

        for code in ("SURRENDER_RATE", "SIGNED_PREMIUM", "NEW_BUSINESS_MARGIN", "NEW_BUSINESS_VALUE"):
            self.assertEqual(rows.loc[code, "期间口径"], "本季度数")
        self.assertAlmostEqual(float(rows.loc["SIGNED_PREMIUM", "数值"]), 3296829)
        self.assertAlmostEqual(float(rows.loc["NEW_BUSINESS_MARGIN", "数值"]), 7.44)
        self.assertAlmostEqual(float(rows.loc["SURRENDER_RATE", "数值"]), 0.78)

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
        self.assertAlmostEqual(result["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"], 36.0 / 250.0)
        self.assertAlmostEqual(result["CORE_T1_TO_ACTUAL_CAPITAL"], 120.0 / 200.0)
        self.assertAlmostEqual(result["CORE_T2_TO_ACTUAL_CAPITAL"], 20.0 / 200.0)
        self.assertAlmostEqual(result["ANC_T1_TO_ACTUAL_CAPITAL"], 50.0 / 200.0)
        self.assertAlmostEqual(result["POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"], 30.0 / 140.0)
        self.assertAlmostEqual(result["REGISTERED_CAPITAL_TO_CORE_CAPITAL"], 50.0 / 140.0)
        self.assertAlmostEqual(result["LIFE_INSURANCE_RISK_TO_QUANT_CAPITAL"], 40.0 / 90.0)
        self.assertAlmostEqual(result["DIVERSIFICATION_EFFECT_TO_QUANT_CAPITAL"], -10.0 / 90.0)
        self.assertEqual(result["POLICY_SURPLUS_CORE_BAND"], "(0,20%]")
        legacy_only = calculate_derived_values({"CORE_CAPITAL_TO_REGISTERED_CAPITAL": 2.8})
        self.assertAlmostEqual(legacy_only["REGISTERED_CAPITAL_TO_CORE_CAPITAL"], 1.0 / 2.8)

    def test_policy_surplus_ratios_distinguish_missing_from_disclosed_zero(self):
        common = {
            "POLICY_SURPLUS_CORE_T1": 20.0,
            "POLICY_SURPLUS_ANC_T1": 5.0,
            "CORE_T1_CAPITAL": 100.0,
            "CORE_T2_CAPITAL": 20.0,
            "INSURANCE_CONTRACT_LIABILITY": 250.0,
        }
        missing = calculate_derived_values(common)
        self.assertAlmostEqual(missing["POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"], 20 / 120)
        self.assertAlmostEqual(missing["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"], 25 / 250)
        self.assertIsNone(calculate_derived_values({"INSURANCE_CONTRACT_LIABILITY": 250})["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"])
        self.assertIsNone(calculate_derived_values({**common, "INSURANCE_CONTRACT_LIABILITY": 0})["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"])

        disclosed_zero = calculate_derived_values({
            **common,
            "POLICY_SURPLUS_CORE_T2": 0.0,
            "POLICY_SURPLUS_ANC_T2": 0.0,
        })
        self.assertAlmostEqual(
            disclosed_zero["POLICY_SURPLUS_CORE_TO_CORE_CAPITAL"],
            20.0 / 120.0,
        )
        self.assertAlmostEqual(
            disclosed_zero["POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"],
            25.0 / 250.0,
        )

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

    def test_policy_surplus_derived_metrics_bridge_opening_and_closing_periods(self):
        rows = []
        period_values = {
            "期初数": {
                "ACTUAL_CAPITAL": 200.0,
                "POLICY_SURPLUS_CORE_T1": 20.0,
                "POLICY_SURPLUS_CORE_T2": 10.0,
            },
            "期末数": {
                "ACTUAL_CAPITAL": 100.0,
                "POLICY_SURPLUS_CORE_T1": 30.0,
                "POLICY_SURPLUS_CORE_T2": 10.0,
            },
        }
        for period, values in period_values.items():
            for code, value in values.items():
                rows.append({
                    "公司": "测试人寿",
                    "公司类型": "寿险",
                    "同业分类": "小型公司",
                    "报告年度": 2026,
                    "报告季度": "Q1",
                    "报告期": "2026Q1",
                    "期间口径": period,
                    "指标编码": code,
                    "指标名称": code,
                    "数值": value,
                    "来源页码": "25",
                    "来源文件": "测试.pdf",
                    "来源工作表": "S02-实际资本明细表",
                })

        result = append_derived_metrics(pd.DataFrame(rows))
        ratios = result[
            result["指标编码"].eq("POLICY_SURPLUS_CORE_TO_ACTUAL_CAPITAL")
        ].set_index("期间口径")["数值"]
        bands = result[
            result["指标编码"].eq("POLICY_SURPLUS_CORE_BAND")
        ].set_index("期间口径")["数值"]

        self.assertAlmostEqual(ratios["上季度末数"], 0.15)
        self.assertAlmostEqual(ratios["本季度末数"], 0.40)
        self.assertEqual(bands["上季度末数"], "(0,20%]")
        self.assertEqual(bands["本季度末数"], "大于35%")

    def test_fills_missing_derived_metrics_without_overwriting_reviewed_values(self):
        rows = [
            {
                "公司": "测试人寿",
                "报告类型": "LIFE_SOLVENCY",
                "报告年度": 2025,
                "报告季度": "Q4",
                "报告期": "2025Q4",
                "期间口径": "本季度末数",
                "指标编码": code,
                "指标名称": code,
                "数值": value,
            }
            for code, value in (
                ("ACTUAL_CAPITAL", 200.0),
                ("CORE_T1_CAPITAL", 120.0),
                ("CORE_T2_CAPITAL", 20.0),
                ("CORE_T1_TO_ACTUAL_CAPITAL", 0.61),
            )
        ]
        result = add_missing_derived_metrics(pd.DataFrame(rows))
        reviewed = result[result["指标编码"] == "CORE_T1_TO_ACTUAL_CAPITAL"]
        generated = result[result["指标编码"] == "CORE_T2_TO_ACTUAL_CAPITAL"]
        self.assertEqual(len(reviewed), 1)
        self.assertAlmostEqual(reviewed.iloc[0]["数值"], 0.61)
        self.assertEqual(len(generated), 1)
        self.assertAlmostEqual(generated.iloc[0]["数值"], 0.1)
        self.assertEqual(generated.iloc[0]["来源类型"], "系统计算")

    def test_recalculates_policy_surplus_ratio_from_disclosed_components(self):
        rows = [{
            "公司": "中银三星", "报告类型": "LIFE_SOLVENCY",
            "报告年度": 2025, "报告季度": "Q4", "报告期": "2025Q4",
            "期间口径": "本季度末数", "指标编码": code,
            "指标名称": code, "数值": value, "披露状态": status,
        } for code, value, status in (
            ("POLICY_SURPLUS_CORE_T1", 20.0, "已披露"),
            ("INSURANCE_CONTRACT_LIABILITY", 250.0, "已披露"),
            ("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES", None, "无法计算"),
        )]
        result = add_missing_derived_metrics(pd.DataFrame(rows))
        ratio = result.loc[result["指标编码"].eq("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES")]
        self.assertEqual(len(ratio), 1)
        self.assertAlmostEqual(ratio.iloc[0]["数值"], 20 / 250)
        self.assertEqual(ratio.iloc[0]["披露状态"], "已计算")

        reviewed = pd.DataFrame(rows)
        reviewed.loc[reviewed["指标编码"].eq("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES"), "数值"] = 0.09
        result = add_missing_derived_metrics(reviewed)
        ratio = result.loc[result["指标编码"].eq("POLICY_SURPLUS_TO_INSURANCE_LIABILITIES")].iloc[0]
        self.assertAlmostEqual(ratio["数值"], 20 / 250)
        self.assertAlmostEqual(ratio["原始披露值"], 0.09)
        self.assertEqual(ratio["来源类型"], "系统计算")
        self.assertIn("统一口径重算", ratio["备注"])

    def test_external_conversion_reports_only_nonblank_invalid_values(self):
        external = pd.DataFrame([
            {"分类": "测试", "公司": "甲人寿", "认可资产": 100.0},
            {"分类": "测试", "公司": "乙人寿", "认可资产": None},
            {"分类": "测试", "公司": "丙人寿", "认可资产": "无法识别"},
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            external.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame([{
            "指标编码": "RECOGNIZED_ASSETS", "指标名称": "认可资产",
            "一级模块": "主要指标", "二级模块": "偿付能力",
            "标准单位": "万元", "数据类型": "金额", "别名": "",
        }])
        result = convert_external_workbook(output.getvalue(), "invalid.xlsx", taxonomy)
        summary = result.sheet_summary.iloc[0]
        self.assertEqual(summary["空值或未披露标记数"], 1)
        self.assertEqual(summary["无效值数"], 1)
        self.assertTrue(any("非空值无法解析" in warning for warning in result.warnings))
        self.assertFalse(any("空值或错误值" in warning for warning in result.warnings))

    def test_step5_converts_legacy_core_to_registered_ratio_to_reversed_ratio(self):
        external = pd.DataFrame([{
            "分类": "小型公司",
            "公司": "测试人寿",
            "核心资本/注册资本": 2.8,
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            external.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame(columns=[
            "指标编码", "指标名称", "一级模块", "二级模块", "标准单位", "数据类型", "别名",
        ])
        converted = convert_external_workbook(
            output.getvalue(),
            "legacy_ratio.xlsx",
            taxonomy,
            {"测试人寿": "寿险"},
        )
        reversed_rows = converted.data[
            converted.data["指标编码"].eq("REGISTERED_CAPITAL_TO_CORE_CAPITAL")
        ]
        self.assertEqual(len(reversed_rows), 1)
        self.assertAlmostEqual(reversed_rows.iloc[0]["数值"], 1.0 / 2.8)
        self.assertEqual(reversed_rows.iloc[0]["指标名称"], "注册资本/核心资本率")
        self.assertEqual(reversed_rows.iloc[0]["来源类型"], "系统计算")
        self.assertEqual(reversed_rows.iloc[0]["计算逻辑"], "1/(核心资本/注册资本)")

    def test_reads_standard_headers_from_first_or_third_row(self):
        base = pd.DataFrame([{"公司": "测试人寿", "指标编码": "A", "指标名称": "指标A", "数值": 1}])
        for startrow in (0, 2):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                base.to_excel(writer, sheet_name="标准数据", index=False, startrow=startrow)
            result = read_standard_workbook(output.getvalue(), f"header_{startrow}.xlsx")
            self.assertEqual(result.iloc[0]["公司"], "测试人寿")
            self.assertEqual(list(result.columns), STANDARD_COLUMNS)

    def test_step5_wide_mapping_excludes_step3_only_metrics(self):
        external = pd.DataFrame([{
            "分类": "小型公司",
            "公司": "测试人寿",
            "实际资本": 20.0,
            "对净资产的调整额": 3.0,
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            external.to_excel(writer, sheet_name="2025Q4", index=False)
        taxonomy = pd.DataFrame([
            {"指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "一级模块": "主要指标", "二级模块": "偿付能力", "标准单位": "万元", "数据类型": "金额", "别名": "", "STEP5宽表映射": "是"},
            {"指标编码": "NET_ASSET_ADJUSTMENT", "指标名称": "对净资产的调整额", "一级模块": "实际资本", "二级模块": "核心一级资本调整", "标准单位": "万元", "数据类型": "金额", "别名": "", "STEP5宽表映射": "否"},
        ])

        with self.assertRaisesRegex(ValueError, "未精确映射指标"):
            convert_external_workbook(
                output.getvalue(),
                "external.xlsx",
                taxonomy,
                {"测试人寿": "寿险"},
            )

    def test_step5_standard_narrow_upload_preserves_step3_only_metrics(self):
        source = pd.DataFrame([{
            "公司": "测试人寿",
            "指标编码": "NET_ASSET_ADJUSTMENT",
            "指标名称": "对净资产的调整额",
            "数值": -3.0,
            "单位": "万元",
        }])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="标准数据", index=False)

        result = read_standard_workbook(output.getvalue(), "step3_standard.xlsx")

        self.assertEqual(result.iloc[0]["指标编码"], "NET_ASSET_ADJUSTMENT")
        self.assertEqual(result.iloc[0]["数值"], -3.0)

    def test_compact_standard_workbook_round_trip_restores_hidden_system_fields(self):
        source = pd.DataFrame([{
            "公司": "尚未登记的新公司",
            "公司统一编码": "CUSTOM_001",
            "公司类型": "健康险",
            "同业分类": "小型公司",
            "报告类型": "LIFE_SOLVENCY",
            "报告期": "2025Q4",
            "一级模块": "主要指标",
            "二级模块": "偿付能力",
            "指标编码": "ACTUAL_CAPITAL",
            "指标名称": "实际资本",
            "期间口径": "期末数",
            "数值": 100.0,
            "单位": "万元",
            "数据类型": "金额",
            "来源类型": "外部数据集",
            "指标属性": "披露",
            "计算逻辑": "",
        }])
        workbook = standard_workbook_bytes(source)
        visible = pd.read_excel(io.BytesIO(workbook), sheet_name="标准数据")
        workbook_model = load_workbook(io.BytesIO(workbook), read_only=False)
        restored = read_standard_workbook(workbook, "compact.xlsx")

        self.assertEqual(list(visible.columns), NARROW_TABLE_COLUMNS)
        self.assertEqual(workbook_model["_系统字段"].sheet_state, "hidden")
        self.assertEqual(restored.iloc[0]["公司统一编码"], "CUSTOM_001")
        self.assertEqual(restored.iloc[0]["公司类型"], "健康险")
        self.assertEqual(restored.iloc[0]["报告类型"], "LIFE_SOLVENCY")

    def test_uploaded_old_company_codes_do_not_split_registered_aliases(self):
        source = pd.DataFrame([
            {
                "公司": "工银安盛人寿保险有限公司",
                "公司统一编码": "LEGACY_FULL_CODE",
                "公司类型": "寿险",
                "报告期": "2025Q1",
                "指标编码": "ACTUAL_CAPITAL",
                "指标名称": "实际资本",
                "数值": 100.0,
            },
            {
                "公司": "工银安盛",
                "公司统一编码": "LEGACY_SHORT_CODE",
                "公司类型": "寿险",
                "报告期": "2026Q1",
                "指标编码": "ACTUAL_CAPITAL",
                "指标名称": "实际资本",
                "数值": 110.0,
            },
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            source.to_excel(writer, sheet_name="标准数据", index=False)
        result = read_standard_workbook(output.getvalue(), "mixed_names.xlsx")
        self.assertEqual(result["公司"].tolist(), ["工银安盛", "工银安盛"])
        self.assertEqual(result["公司统一编码"].nunique(), 1)
        self.assertEqual(result["原始公司名称"].tolist(), [
            "工银安盛人寿保险有限公司", "工银安盛",
        ])

    def test_derived_metrics_bridge_old_full_and_short_name_records(self):
        rows = [
            {
                "公司": company,
                "原始公司名称": company,
                "标准公司名称": company,
                "公司统一编码": old_code,
                "公司类型": "寿险",
                "报告类型": "LIFE_SOLVENCY",
                "报告年度": 2026,
                "报告季度": "Q1",
                "报告期": "2026Q1",
                "期间口径": "本季度末数",
                "指标编码": metric,
                "指标名称": metric,
                "数值": value,
            }
            for company, old_code, metric, value in (
                ("工银安盛人寿保险有限公司", "OLD_FULL", "ACTUAL_CAPITAL", 20.0),
                ("工银安盛", "OLD_SHORT", "RECOGNIZED_ASSETS", 100.0),
            )
        ]
        result = add_missing_derived_metrics(pd.DataFrame(rows))
        ratio = result[result["指标编码"].eq("ACTUAL_CAPITAL_TO_RECOGNIZED_ASSETS")]
        self.assertEqual(len(ratio), 1)
        self.assertAlmostEqual(ratio.iloc[0]["数值"], 0.2)
        self.assertEqual(result["公司统一编码"].nunique(), 1)

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
