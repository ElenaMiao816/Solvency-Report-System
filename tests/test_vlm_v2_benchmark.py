from __future__ import annotations

import unittest

import pandas as pd

from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_benchmark import (
    VLMV2BlindGoldCase,
    blind_test_template_bytes,
    read_blind_gold_workbook,
    score_blind_case,
)
from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS,
    VALIDATION_COLUMNS,
    VLMV2ExtractionRun,
    VLMV2LocatorRun,
)


class VLMV2BenchmarkTests(unittest.TestCase):
    def test_template_round_trip_has_required_sheets(self):
        data = blind_test_template_bytes()
        excel = pd.ExcelFile(data)
        self.assertEqual(
            set(excel.sheet_names),
            {"使用说明", "样本", "页码金标准", "指标金标准"},
        )
        with self.assertRaisesRegex(ValueError, "没有有效样本"):
            read_blind_gold_workbook(data)

    def test_case_scoring_separates_page_value_period_and_gate(self):
        case = VLMV2BlindGoldCase(
            case_id="case-1",
            filename="sample.pdf",
            sha256="",
            pages=pd.DataFrame([{"样本ID": "case-1", "目标表ID": "SOLVENCY_MAIN", "物理页码": "3"}]),
            metrics=pd.DataFrame([{
                "样本ID": "case-1", "目标表ID": "SOLVENCY_MAIN", "指标编码": "ACTUAL_CAPITAL",
                "期望状态": "披露数值", "期望标准数值": 100.0, "标准单位": "万元", "期间口径": "本季度末",
                "相对容差": 0.005, "绝对容差": 0.01,
            }]),
        )
        locator = VLMV2LocatorRun((PageMatch("SOLVENCY_MAIN", "主要指标", [3], 99, "e"),), pd.DataFrame(), 1, 3)
        record = dict(zip(METRIC_COLUMNS, [
            "SOLVENCY_MAIN", "主要指标", "ACTUAL_CAPITAL", "实际资本", "ACTUAL_CAPITAL", "found", "100", 100.0,
            "万元", 100.0, "万元", "本季度末", "实际资本", "实际资本", "本季度末", 3, "实际资本100", .99, "直提",
        ]))
        validations = pd.DataFrame([dict(zip(VALIDATION_COLUMNS, ["x", "", "", "通过", "", "", "", ""]))])
        extraction = VLMV2ExtractionRun(pd.DataFrame([record]), validations, (), 1, 0)
        score = score_blind_case(case, locator, extraction, filename="sample.pdf", elapsed_seconds=2.5)
        row = score.summary.iloc[0]
        self.assertEqual(row["总体检查准确率"], 1.0)
        self.assertEqual(row["STEP3门槛"], "通过")


if __name__ == "__main__":
    unittest.main()
