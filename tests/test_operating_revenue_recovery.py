from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from services.solvency_operating_recovery import recover_insurance_revenue
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS, VLMV2TargetCard, _extract_card_records,
    _normalize_number_and_unit, validate_vlm_v2_metrics,
)
from tests.test_vlm_v2_pipeline import FakeResponse


PDF = Path(r"F:\CROSS\V2\中银三星2025Q3偿付能力季度报告摘要.pdf")
AVIVA_PDF = PDF.parent / "中英人寿2026Q1偿付能力季度报告摘要.pdf"


@unittest.skipUnless(PDF.exists(), "Local report unavailable")
class OperatingRevenueRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf_bytes = PDF.read_bytes()

    def test_continued_table_reads_current_quarter_and_previous_page_unit(self):
        for selected in ([10], [11], [10, 11]):
            with self.subTest(selected=selected):
                item = recover_insurance_revenue(self.pdf_bytes, selected)
                self.assertIsNotNone(item)
                self.assertEqual(item["page"], 11)
                self.assertEqual(item["period_label"], "本季度数")
                self.assertEqual(item["value_raw"], "10,670,639,397.99")
                self.assertEqual(item["unit"], "元")
                self.assertIn("经营指标（元，%）", item["evidence_text"])
                self.assertIn("物理页10", item["evidence_text"])
                value, unit = _normalize_number_and_unit(item["value_raw"], item["unit"], "万元")
                self.assertAlmostEqual(value, 1067063.939799, places=6)
                self.assertEqual(unit, "万元")

    def test_unrelated_pages_do_not_supply_required_value(self):
        self.assertIsNone(recover_insurance_revenue(self.pdf_bytes, [1]))

    @unittest.skipUnless(AVIVA_PDF.exists(), "Local Aviva report unavailable")
    def test_standalone_unit_above_aviva_operating_table(self):
        item = recover_insurance_revenue(AVIVA_PDF.read_bytes(), [16])
        self.assertIsNotNone(item)
        self.assertEqual(item["page"], 16)
        self.assertEqual(item["period_label"], "本季度数")
        self.assertEqual(item["value_raw"], "7,855,050,801.15")
        self.assertEqual(item["unit"], "元")
        self.assertIn("单位：元", item["evidence_text"])
        value, unit = _normalize_number_and_unit(item["value_raw"], item["unit"], "万元")
        self.assertAlmostEqual(value, 785505.080115, places=6)
        self.assertEqual(unit, "万元")

    @unittest.skipUnless(AVIVA_PDF.exists(), "Local Aviva report unavailable")
    def test_step2_recovers_aviva_model_omission(self):
        metric = dict(metric_id="INSURANCE_REVENUE", name="保险业务收入", expected_unit="万元",
                      data_type="金额", semantic_key="INSURANCE_REVENUE")
        card = VLMV2TargetCard("OPERATING_METRICS", "主要经营指标", "", (metric,))
        match = PageMatch("OPERATING_METRICS", "主要经营指标", [16], 99, "")

        def omit(_url, *, headers, json, timeout):
            return FakeResponse({"metrics": []})

        rows, count = _extract_card_records(
            AVIVA_PDF.read_bytes(), match, card,
            api_key="test", base_url="https://example.invalid", model="test",
            timeout=30, request_max_attempts=1, post_func=omit,
            page_image_cache={16: "data:image/jpeg;base64,page16"},
        )
        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["状态"], "found")
        self.assertEqual(rows[0]["原始值"], "7,855,050,801.15")
        self.assertEqual(rows[0]["单位"], "元")
        self.assertEqual(rows[0]["物理页码"], 16)
        checks = validate_vlm_v2_metrics(pd.DataFrame(rows, columns=METRIC_COLUMNS), [card])
        required = checks.loc[checks["校验ID"].eq("OPERATING_METRICS:INSURANCE_REVENUE:REQUIRED")]
        self.assertEqual(required.iloc[0]["状态"], "通过")

    def test_other_company_units_are_read_from_their_own_reports(self):
        samples = (
            ("中信保诚2026Q1偿付能力季度报告摘要.pdf", "亿元", "152"),
            ("中美联泰2026Q1偿付能力季度报告摘要.pdf", "万元", "1,059,446.64"),
        )
        for name, unit, raw in samples:
            report = PDF.parent / name
            if not report.exists():
                continue
            with self.subTest(name=name):
                import fitz
                with fitz.open(report) as doc:
                    pages = [i + 1 for i, page in enumerate(doc) if "保险业务收入" in page.get_text()]
                item = recover_insurance_revenue(report.read_bytes(), pages[:2])
                self.assertIsNotNone(item)
                self.assertEqual(item["unit"], unit)
                self.assertEqual(item["value_raw"], raw)

    def test_step2_recovers_model_omission_without_annual_column(self):
        metric = dict(metric_id="INSURANCE_REVENUE", name="保险业务收入", expected_unit="万元",
                      data_type="金额", semantic_key="INSURANCE_REVENUE")
        card = VLMV2TargetCard("OPERATING_METRICS", "主要经营指标", "", (metric,))
        match = PageMatch("OPERATING_METRICS", "主要经营指标", [10], 99, "")

        def omit(_url, *, headers, json, timeout):
            return FakeResponse({"metrics": []})

        rows, count = _extract_card_records(
            self.pdf_bytes, match, card,
            api_key="test", base_url="https://example.invalid", model="test",
            timeout=30, request_max_attempts=1, post_func=omit,
            page_image_cache={10: "data:image/jpeg;base64,page10"},
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["状态"], "found")
        self.assertEqual(rows[0]["物理页码"], 11)
        self.assertEqual(rows[0]["原始值"], "10,670,639,397.99")
        self.assertEqual(rows[0]["单位"], "元")
        self.assertAlmostEqual(rows[0]["标准数值"], 1067063.939799, places=6)
        self.assertEqual(rows[0]["提取轮次"], "源PDF文字表格补提")
        checks = validate_vlm_v2_metrics(pd.DataFrame(rows, columns=METRIC_COLUMNS), [card])
        required = checks.loc[checks["校验ID"].eq("OPERATING_METRICS:INSURANCE_REVENUE:REQUIRED")]
        self.assertEqual(required.iloc[0]["状态"], "通过")


if __name__ == "__main__":
    unittest.main()
