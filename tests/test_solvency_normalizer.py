from __future__ import annotations

import unittest

import pandas as pd

from services.solvency_normalizer import normalize_tables
from services.solvency_table_extractor import ExtractedTable, UnitRecord


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


if __name__ == "__main__":
    unittest.main()
