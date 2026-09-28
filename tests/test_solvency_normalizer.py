from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from services.solvency_normalizer import (
    TABLE_ALLOWED_CODES,
    load_taxonomy,
    normalize_tables,
    resolve_metric_match,
)
from services.solvency_metric_registry import extend_taxonomy
from services.solvency_table_extractor import ExtractedTable, UnitRecord


ROOT = Path(__file__).resolve().parents[1]


class SolvencyNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = pd.DataFrame([
            {
                "指标编码": "CORE_T1_CAPITAL",
                "指标名称": "核心一级资本",
                "别名": "核心一级资本金额|核心一级资本（万元）",
                "一级模块": "实际资本",
                "二级模块": "资本构成",
                "标准单位": "万元",
                "数据类型": "金额",
            },
            {
                "指标编码": "SOLVENCY_RATIO",
                "指标名称": "综合偿付能力充足率",
                "别名": "",
                "一级模块": "主要指标",
                "二级模块": "偿付能力",
                "标准单位": "%",
                "数据类型": "百分比",
            },
        ]).fillna("")
        self.metadata = {
            "公司": "测试人寿保险有限公司",
            "报告年度": 2026,
            "报告季度": "Q1",
            "报告期": "2026Q1",
        }

    @staticmethod
    def _unit(scope: str, target: str, unit: str) -> UnitRecord:
        return UnitRecord(
            scope=scope,
            target=target,
            raw_unit=unit,
            normalized_unit=unit,
            source_page=1,
        )

    def _normalize(self, rows, units, company_type="寿险"):
        table = ExtractedTable(
            table_id="TEST",
            table_name="测试表",
            page=1,
            table_index=1,
            rows=rows,
            source_pages=[1],
            unit_records=units,
        )
        return normalize_tables(
            [table],
            self.taxonomy,
            self.metadata,
            company_type,
        )

    def test_table_yuan_is_converted_to_ten_thousand_yuan(self):
        result = self._normalize(
            [
                ["项目", "期末数"],
                ["核心一级资本", "100,000,000"],
                ["综合偿付能力充足率", "130%"],
            ],
            [
                self._unit("表级", "", "元"),
                self._unit("行级", "综合偿付能力充足率", "%"),
            ],
        )

        capital = result[result["指标编码"] == "CORE_T1_CAPITAL"].iloc[0]
        ratio = result[result["指标编码"] == "SOLVENCY_RATIO"].iloc[0]
        self.assertEqual(capital["数值"], 10_000)
        self.assertEqual(capital["单位"], "万元")
        self.assertEqual(capital["原始披露值"], "100,000,000")
        self.assertIn("原单位：元", capital["备注"])
        self.assertEqual(ratio["数值"], 130)
        self.assertEqual(ratio["单位"], "%")
        self.assertEqual(ratio["备注"], "")

    def test_row_unit_overrides_table_unit(self):
        result = self._normalize(
            [
                ["项目", "期末数"],
                ["核心一级资本（万元）", "123.45"],
            ],
            [
                self._unit("表级", "", "元"),
                self._unit("行级", "核心一级资本（万元）", "万元"),
            ],
        )

        self.assertEqual(result.iloc[0]["数值"], 123.45)
        self.assertEqual(result.iloc[0]["备注"], "")

    def test_structured_row_unit_converts_when_output_label_has_no_unit(self):
        result = self._normalize(
            [
                ["项目", "期末数"],
                ["核心一级资本", "100,000,000"],
            ],
            [
                self._unit("行级", "核心一级资本", "元"),
            ],
        )

        self.assertEqual(result.iloc[0]["数值"], 10_000)
        self.assertEqual(result.iloc[0]["单位"], "万元")
        self.assertIn("原单位：元", result.iloc[0]["备注"])

    def test_column_unit_overrides_table_unit(self):
        result = self._normalize(
            [
                ["项目", "期末数（元）"],
                ["核心一级资本", "25,000"],
            ],
            [
                self._unit("表级", "", "万元"),
                self._unit("列级", "期末数（元）", "元"),
            ],
        )

        self.assertEqual(result.iloc[0]["数值"], 2.5)
        self.assertIn("已换算为万元", result.iloc[0]["备注"])

    def test_missing_amount_unit_is_flagged_without_conversion(self):
        result = self._normalize(
            [
                ["项目", "期末数"],
                ["核心一级资本", "25,000"],
            ],
            [],
        )

        self.assertEqual(result.iloc[0]["数值"], 25_000)
        self.assertEqual(result.iloc[0]["备注"], "未识别原始单位，数值未换算")

    def test_mixed_table_units_are_selected_by_metric_type(self):
        result = self._normalize(
            [
                ["项目", "期末数"],
                ["核心一级资本", "100,000,000"],
                ["综合偿付能力充足率", "130"],
            ],
            [
                self._unit("表级", "", "元"),
                self._unit("表级", "", "%"),
            ],
        )

        capital = result[result["指标编码"] == "CORE_T1_CAPITAL"].iloc[0]
        ratio = result[result["指标编码"] == "SOLVENCY_RATIO"].iloc[0]
        self.assertEqual(capital["报告类型"], "LIFE_SOLVENCY")
        self.assertEqual(capital["数值"], 10_000)
        self.assertIn("原单位：元", capital["备注"])
        self.assertEqual(ratio["数值"], 130)
        self.assertEqual(ratio["单位"], "%")
        self.assertEqual(ratio["备注"], "")

    def test_only_life_company_types_are_accepted(self):
        result = self._normalize(
            [["项目", "期末数（万元）"], ["核心一级资本", "1"]],
            [],
            company_type="健康",
        )
        self.assertEqual(result.iloc[0]["公司类型"], "健康险")

        with self.assertRaisesRegex(ValueError, "公司类型仅支持"):
            self._normalize(
                [["项目", "期末数（万元）"], ["核心一级资本", "1"]],
                [],
                company_type="财险",
            )

    def test_profile_can_explicitly_enable_non_life_company_type(self):
        table = ExtractedTable(
            table_id="TEST",
            table_name="测试表",
            page=1,
            table_index=1,
            rows=[["项目", "期末数（万元）"], ["核心一级资本", "1"]],
            source_pages=[1],
        )
        result = normalize_tables(
            [table],
            self.taxonomy,
            self.metadata,
            "财产保险",
            report_profile_id="NON_LIFE_SOLVENCY",
            allowed_company_types=("财险",),
        )
        self.assertEqual(result.iloc[0]["公司类型"], "财险")
        self.assertEqual(result.iloc[0]["报告类型"], "NON_LIFE_SOLVENCY")

    def test_actual_capital_accepts_step3_only_disclosure_metrics(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "FINANCIAL_STATEMENT_ASSETS",
                "指标名称": "财务报表资产总额",
                "别名": "财务报表资产合计",
                "一级模块": "实际资本",
                "二级模块": "汇总口径",
                "标准单位": "万元",
                "数据类型": "金额",
                "STEP5宽表映射": "否",
            },
            {
                "指标编码": "NET_ASSET_ADJUSTMENT",
                "指标名称": "对净资产的调整额",
                "别名": "净资产调整额",
                "一级模块": "实际资本",
                "二级模块": "核心一级资本调整",
                "标准单位": "万元",
                "数据类型": "金额",
                "STEP5宽表映射": "否",
            },
        ]).fillna("")
        table = ExtractedTable(
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本明细表",
            page=1,
            table_index=1,
            rows=[
                ["项目", "期末数（万元）"],
                ["财务报表资产总额", "100"],
                ["对净资产的调整额", "-3"],
            ],
            source_pages=[1],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(
            result["指标编码"].tolist(),
            ["FINANCIAL_STATEMENT_ASSETS", "NET_ASSET_ADJUSTMENT"],
        )
        self.assertEqual(result["数值"].tolist(), [100.0, -3.0])

    def test_actual_capital_detail_rows_are_all_standardized_and_disambiguated(self):
        taxonomy = extend_taxonomy(pd.DataFrame([{
            "指标编码": "DEFERRED_TAX_ASSET_ADJUSTMENT",
            "指标名称": "递延所得税资产（由经营性亏损引起的递延所得税资产除外）",
            "别名": "",
            "一级模块": "实际资本",
            "二级模块": "核心一级资本调整",
            "标准单位": "万元",
            "数据类型": "金额",
            "STEP5宽表映射": "否",
        }]).fillna(""))
        table = ExtractedTable(
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本表明细表",
            page=29,
            table_index=1,
            rows=[
                ["行次", "项目", "期末数（万元）"],
                ["1.2.3", "投资性房地产（包括保险公司以物权方式或通过子公司等方式持有的投资性房地产）的公允价值增值（扣除减值、折旧及所得税影响）", "1"],
                ["1.2.5", "对农业保险提取的大灾风险准备金", "2"],
                ["1.2.6", "计入核心一级资本的保单未来盈余", "3"],
                ["1.2.7", "符合核心一级资本标准的负债类资本工具且按规定可计入核心一级资本的金额", "4"],
                ["2.1", "优先股", "5"],
                ["2.2", "计入核心二级资本的保单未来盈余", "6"],
                ["2.3", "其他核心二级资本", "7"],
                ["2.4", "减：超限额应扣除的部分", "181,609.06"],
                ["3.1", "次级定期债务", "8"],
                ["3.2", "资本补充债券", "9"],
                ["3.3", "可转换次级债", "10"],
                ["3.4", "递延所得税资产（由经营性亏损引起的递延所得税资产除外）", "1"],
                ["3.5", "投资性房地产（包括保险公司以物权方式或通过子公司等方式持有的投资性房地产）公允价值增值可计入附属一级资本的金额（扣除减值、折旧及所得税影响）", "11"],
                ["3.6", "计入附属一级资本的保单未来盈余", "12"],
                ["3.7", "其他附属一级资本", "13"],
                ["3.8", "减：超限额应扣除的部分", "2"],
                ["4.1", "应急资本等其他附属二级资本", "14"],
                ["4.2", "计入附属二级资本的保单未来盈余", "15"],
                ["4.3", "减：超限额应扣除的部分", "3"],
            ],
            source_pages=[29],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(result["指标编码"].tolist(), [
            "CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT",
            "AGRICULTURAL_CATASTROPHE_RISK_RESERVE",
            "POLICY_SURPLUS_CORE_T1",
            "QUALIFYING_CORE_T1_LIABILITY_CAPITAL",
            "CORE_T2_PREFERRED_SHARES",
            "POLICY_SURPLUS_CORE_T2",
            "OTHER_CORE_T2_CAPITAL",
            "CORE_T2_EXCESS_DEDUCTION",
            "ANC_T1_SUBORDINATED_TERM_DEBT",
            "ANC_T1_CAPITAL_SUPPLEMENTARY_BONDS",
            "ANC_T1_CONVERTIBLE_SUBORDINATED_DEBT",
            "ANC_T1_DEFERRED_TAX_ASSET",
            "ANC_T1_INVESTMENT_PROPERTY_FAIR_VALUE",
            "POLICY_SURPLUS_ANC_T1",
            "OTHER_ANC_T1_CAPITAL",
            "ANC_T1_EXCESS_DEDUCTION",
            "EMERGENCY_OTHER_ANC_T2_CAPITAL",
            "POLICY_SURPLUS_ANC_T2",
            "ANC_T2_EXCESS_DEDUCTION",
        ])

    def test_actual_capital_sequential_row_21_maps_to_ancillary_deferred_tax_asset(self):
        label = "递延所得税资产（由经营性亏损引起的递延所得税资产除外）"
        taxonomy = extend_taxonomy(pd.DataFrame([{
            "指标编码": "DEFERRED_TAX_ASSET_ADJUSTMENT",
            "指标名称": label,
            "别名": "",
            "一级模块": "实际资本",
            "二级模块": "核心一级资本调整",
            "标准单位": "万元",
            "数据类型": "金额",
            "STEP5宽表映射": "否",
        }]).fillna(""))
        table = ExtractedTable(
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本明细表",
            page=25,
            table_index=1,
            rows=[
                ["行次", "项目", "期末数（万元）", "期初数（万元）"],
                ["7", label, "-873,374.41", "-376,407.24"],
                ["21", label, "873,374.41", "376,407.24"],
            ],
            source_pages=[25],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(
            set(result.loc[result["行次"].eq("7"), "指标编码"]),
            {"DEFERRED_TAX_ASSET_ADJUSTMENT"},
        )
        self.assertEqual(
            set(result.loc[result["行次"].eq("21"), "指标编码"]),
            {"ANC_T1_DEFERRED_TAX_ASSET"},
        )

    def test_capital_totals_are_mapped_and_dash_components_become_zero(self):
        taxonomy = extend_taxonomy(pd.DataFrame([
            {
                "指标编码": code,
                "指标名称": name,
                "别名": "",
                "一级模块": level1,
                "二级模块": "偿付能力",
                "标准单位": "万元",
                "数据类型": "金额",
                "允许期间口径": "本季度末数|期末数",
            }
            for code, name, level1 in [
                ("QUANT_RISK_CAPITAL", "量化风险最低资本", "最低资本"),
                ("CONTROL_RISK_CAPITAL", "控制风险最低资本", "最低资本"),
                ("ADDITIONAL_CAPITAL", "附加资本", "最低资本"),
                ("MINIMUM_CAPITAL", "最低资本", "最低资本"),
                ("ANC_T2_CAPITAL", "附属二级资本", "实际资本"),
            ]
        ]).fillna(""))
        table = ExtractedTable(
            table_id="SOLVENCY_MAIN",
            table_name="偿付能力充足率指标",
            page=11,
            table_index=1,
            rows=[
                ["指标名称", "本季度末数"],
                ["附属二级资本", "-"],
                ["可资本化风险最低资本", "316,607.17"],
                ["控制风险最低资本", "13,496.57"],
                ["附加资本", "—"],
                ["最低资本", "330,103.74"],
            ],
            source_pages=[11],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")
        values = result.set_index("指标编码")["数值"].to_dict()

        self.assertEqual(values["ANC_T2_CAPITAL"], 0.0)
        self.assertEqual(values["ADDITIONAL_CAPITAL"], 0.0)
        self.assertEqual(values["QUANT_RISK_CAPITAL"], 316_607.17)
        self.assertEqual(values["MINIMUM_CAPITAL"], 330_103.74)

    def test_unadjusted_quantitative_capital_is_not_mapped_to_reported_total(self):
        taxonomy = extend_taxonomy(pd.DataFrame([
            {
                "指标编码": code,
                "指标名称": name,
                "别名": "",
                "一级模块": level1,
                "二级模块": "量化风险",
                "标准单位": "万元",
                "数据类型": "金额",
                "允许期间口径": "期末数|期初数",
            }
            for code, name, level1 in [
                ("QUANT_RISK_CAPITAL", "量化风险最低资本", "最低资本"),
                ("ADDITIONAL_CAPITAL", "附加资本", "最低资本"),
                ("MINIMUM_CAPITAL", "最低资本", "主要指标"),
            ]
        ]).fillna(""))
        diagnostics = []
        table = ExtractedTable(
            table_id="MINIMUM_CAPITAL",
            table_name="S05-最低资本表",
            page=35,
            table_index=1,
            rows=[
                ["行次", "项目", "期末数"],
                ["1", "量化风险最低资本", "316,607.17"],
                ["1*", "量化风险最低资本（未考虑特征系数前）", "333,270.70"],
                ["3", "附加资本", "-"],
                ["3.1", "逆周期附加资本", "-"],
                ["3.4", "其他附加资本", "-"],
                ["4", "最低资本", "330,103.74"],
            ],
            source_pages=[35],
        )

        result = normalize_tables(
            [table], taxonomy, self.metadata, "寿险", diagnostics=diagnostics
        )

        self.assertEqual(
            result["指标编码"].tolist(),
            ["QUANT_RISK_CAPITAL", "QUANT_RISK_CAPITAL_BEFORE_FACTOR", "ADDITIONAL_CAPITAL",
             "COUNTERCYCLICAL_ADDITIONAL_CAPITAL", "OTHER_ADDITIONAL_CAPITAL", "MINIMUM_CAPITAL"],
        )
        values = result.set_index('指标编码')
        self.assertAlmostEqual(values.loc['QUANT_RISK_CAPITAL_BEFORE_FACTOR', '数值'], 333270.70)
        self.assertAlmostEqual(values.loc['QUANT_RISK_CAPITAL', '数值'], 316607.17)
        self.assertEqual(values.loc['OTHER_ADDITIONAL_CAPITAL', '披露状态'], '已披露为0')
        self.assertEqual(values.loc['OTHER_ADDITIONAL_CAPITAL', '数值'], 0)

    def test_operating_metrics_use_exact_labels_and_canonical_periods(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "SIGNED_PREMIUM",
                "指标名称": "签单保费",
                "别名": "",
                "一级模块": "经营指标",
                "二级模块": "规模类",
                "标准单位": "万元",
                "数据类型": "金额",
            },
            {
                "指标编码": "AGENT_COUNT",
                "指标名称": "期末个人营销员数量",
                "别名": "",
                "一级模块": "经营指标",
                "二级模块": "规模类",
                "标准单位": "人",
                "数据类型": "数量",
            },
        ]).fillna("")
        table = ExtractedTable(
            table_id="OPERATING_METRICS",
            table_name="主要经营指标",
            page=13,
            table_index=1,
            rows=[
                ["指标名称", "本季度数（万元）", "本年度累计数（万元）"],
                ["1.签单保费", "297,677.41", "297,677.41"],
                ["2.新单首年期交签单保费", "20,591.32", "20,591.32"],
                ["6.分渠道的签单保费", "297,677.41", "297,677.41"],
                ["7.期末个人营销员数量（单位：人）", "63", "63"],
            ],
            source_pages=[13],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(result["指标编码"].tolist(), [
            "SIGNED_PREMIUM", "SIGNED_PREMIUM", "AGENT_COUNT", "AGENT_COUNT"
        ])
        self.assertEqual(
            result["期间口径"].tolist(),
            ["本季度数", "本年累计数", "本季度数", "本年累计数"],
        )

    def test_operating_period_headers_drop_mixed_unit_suffixes(self):
        taxonomy = pd.DataFrame([{
            "指标编码": "SIGNED_PREMIUM",
            "指标名称": "签单保费",
            "别名": "",
            "一级模块": "经营指标",
            "二级模块": "保费",
            "标准单位": "万元",
            "数据类型": "金额",
        }]).fillna("")
        table = ExtractedTable(
            table_id="OPERATING_METRICS",
            table_name="主要经营指标",
            page=14,
            table_index=1,
            rows=[
                ["指标名称", "本季度（末）数（万元，%）", "本年累计（末）数（万元，%）"],
                ["签单保费", "7,077,949.57", "7,077,949.57"],
            ],
            source_pages=[14],
            unit_records=[self._unit("表级", "", "万元")],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(result["期间口径"].tolist(), ["本季度数", "本年累计数"])

    def test_period_headers_without_number_drop_unit_suffixes(self):
        taxonomy = pd.DataFrame([{
            "指标编码": "RECOGNIZED_ASSETS",
            "指标名称": "认可资产",
            "别名": "",
            "一级模块": "主要指标",
            "二级模块": "偿付能力",
            "标准单位": "万元",
            "数据类型": "金额",
        }]).fillna("")
        table = ExtractedTable(
            table_id="SOLVENCY_MAIN",
            table_name="偿付能力充足率指标",
            page=12,
            table_index=1,
            rows=[
                ["指标名称", "上季度末（万元）", "本季度末（万元）", "下季度末预测（万元）"],
                ["认可资产", "1", "2", "3"],
            ],
            source_pages=[12],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(
            result["期间口径"].tolist(),
            ["上季度末数", "本季度末数", "下季度末预测数"],
        )

    def test_actual_capital_deferred_tax_without_row_numbers_uses_sign_pair_context(self):
        label = "递延所得税资产（由经营性亏损引起的递延所得税资产除外）"
        taxonomy = extend_taxonomy(pd.DataFrame([{
            "指标编码": "DEFERRED_TAX_ASSET_ADJUSTMENT",
            "指标名称": label,
            "别名": "",
            "一级模块": "实际资本",
            "二级模块": "核心一级资本调整",
            "标准单位": "万元",
            "数据类型": "金额",
        }]).fillna(""))
        table = ExtractedTable(
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本明细表",
            page=25,
            table_index=1,
            rows=[
                ["项目", "上季度末（万元）", "本季度末（万元）"],
                [label, "-376,407.24", "-873,374.41"],
                [label, "376,407.24", "873,374.41"],
            ],
            source_pages=[25],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(
            result["指标编码"].tolist(),
            [
                "DEFERRED_TAX_ASSET_ADJUSTMENT",
                "DEFERRED_TAX_ASSET_ADJUSTMENT",
                "ANC_T1_DEFERRED_TAX_ASSET",
                "ANC_T1_DEFERRED_TAX_ASSET",
            ],
        )
        self.assertEqual(
            result["期间口径"].tolist(),
            ["上季度末数", "本季度末数", "上季度末数", "本季度末数"],
        )
        allowed = taxonomy.loc[
            taxonomy["指标编码"].astype(str).isin(result["指标编码"]),
            "允许期间口径",
        ].astype(str)
        self.assertTrue(allowed.str.contains("本季度末数", regex=False).all())
        self.assertTrue(allowed.str.contains("上季度末数", regex=False).all())

    def test_three_year_return_rows_are_normalized_before_metric_mapping(self):
        taxonomy = extend_taxonomy(pd.DataFrame([
            {
                "指标编码": code,
                "指标名称": name,
                "别名": "",
                "一级模块": "经营指标",
                "二级模块": "收益率",
                "标准单位": "%",
                "数据类型": "百分比",
                "允许期间口径": "本季度（末）数|本年累计数|年度累计数",
            }
            for code, name in [
                ("INVESTMENT_RETURN", "投资收益率"),
                ("COMPREHENSIVE_INVESTMENT_RETURN", "综合投资收益率"),
            ]
        ]).fillna(""))
        table = ExtractedTable(
            table_id="THREE_YEAR_INVESTMENT_RETURN",
            table_name="近三年（综合）投资收益率",
            page=14,
            table_index=1,
            rows=[
                ["近三年平均投资收益率", "4.55%"],
                ["近三年平均综合投资收益率", "4.34%"],
            ],
            source_pages=[14],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(
            result["指标编码"].tolist(),
            ["INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN"],
        )
        self.assertEqual(result["期间口径"].tolist(), ["近三年平均", "近三年平均"])
        self.assertEqual(result["数值"].tolist(), [4.55, 4.34])

    def test_duplicate_actual_capital_totals_are_collapsed_within_one_source(self):
        taxonomy = pd.DataFrame([{
            "指标编码": "ACTUAL_CAPITAL",
            "指标名称": "实际资本",
            "别名": "实际资本合计",
            "一级模块": "实际资本",
            "二级模块": "汇总",
            "标准单位": "万元",
            "数据类型": "金额",
        }]).fillna("")
        table = ExtractedTable(
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本明细表",
            page=28,
            table_index=1,
            rows=[
                ["行次", "项目", "期末数", "期初数"],
                ["6", "实际资本（=2-4）", "501,783.84", "477,155.28"],
                ["5", "实际资本合计", "501,783.84", "477,155.28"],
            ],
            source_pages=[28, 29, 30],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["期间口径"]), {"期末数", "期初数"})

    def test_recognized_assets_multiheader_extracts_recognized_value_only(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "CASH_LIQUID_ASSETS",
                "指标名称": "现金及流动性管理工具",
                "别名": "现金及流动性管理工具认可价值",
                "一级模块": "认可资产",
                "二级模块": "资产构成",
                "标准单位": "万元",
                "数据类型": "金额",
            },
            {
                "指标编码": "RECOGNIZED_ASSETS",
                "指标名称": "认可资产",
                "别名": "认可资产合计|认可资产总额",
                "一级模块": "主要指标",
                "二级模块": "偿付能力",
                "标准单位": "万元",
                "数据类型": "金额",
            },
        ]).fillna("")
        table = ExtractedTable(
            table_id="RECOGNIZED_ASSETS",
            table_name="S03-认可资产表",
            page=35,
            table_index=1,
            rows=[
                ["行次", "项目", "期末数", "", "", "期初数", "", ""],
                ["", "", "账面价值", "非认可价值", "认可价值",
                 "账面价值", "非认可价值", "认可价值"],
                ["1", "现金及流动性管理工具", "100", "20", "80", "90", "18", "72"],
                ["2", "认可资产合计", "", "", "800", "", "", "720"],
            ],
            source_pages=[35],
        )

        result = normalize_tables([table], taxonomy, self.metadata, "寿险")

        cash = result[result["指标编码"] == "CASH_LIQUID_ASSETS"]
        self.assertEqual(len(cash), 2)
        self.assertEqual(set(cash["期间口径"]), {"期末数", "期初数"})
        cash_by_period = {
            row["期间口径"]: row["数值"] for _, row in cash.iterrows()
        }
        self.assertEqual(cash_by_period["期末数"], 80)
        self.assertEqual(cash_by_period["期初数"], 72)

        recognized = result[result["指标编码"] == "RECOGNIZED_ASSETS"]
        self.assertEqual(len(recognized), 2)
        recognized_by_period = {
            row["期间口径"]: row["数值"] for _, row in recognized.iterrows()
        }
        self.assertEqual(recognized_by_period["期末数"], 800)
        self.assertEqual(recognized_by_period["期初数"], 720)

    def test_recognized_assets_new_component_metrics_are_cataloged(self):
        taxonomy = load_taxonomy(ROOT / "config" / "solvency_taxonomy.xlsx")
        new_metrics = {
            "SUBSIDIARY_JV_ASSOCIATE_EQUITY": "在子公司合营企业和联营企业中的权益",
            "RECEIVABLES_AND_PREPAYMENTS": "应收及预付款项",
            "FIXED_ASSETS": "固定资产",
            "LAND_USE_RIGHTS": "土地使用权",
            "SEPARATE_ACCOUNT_ASSETS": "独立账户资产",
            "OTHER_RECOGNIZED_ASSETS": "其他认可资产",
        }
        allowed = TABLE_ALLOWED_CODES["RECOGNIZED_ASSETS"]
        for code in new_metrics:
            self.assertIn(code, allowed)

        codes = set(taxonomy["指标编码"].astype(str))
        for code, name in new_metrics.items():
            self.assertIn(code, codes)
            decision = resolve_metric_match(
                name,
                taxonomy,
                table_id="RECOGNIZED_ASSETS",
            )
            self.assertIsNotNone(decision.metric, name)
            self.assertEqual(decision.metric["指标编码"], code)


if __name__ == "__main__":
    unittest.main()
