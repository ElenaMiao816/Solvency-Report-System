import unittest
from unittest.mock import patch

from services.solvency_pdf_locator import extract_report_metadata


class ReportMetadataTests(unittest.TestCase):
    def test_cover_tail_digits_beat_company_history_year(self):
        page_texts = [
            (
                "保险公司偿付能力报告\n"
                "华贵人寿保险股份有限公司\n"
                "ÄêµÚ ¼¾¶È\n"
                "2026            1"
            ),
            "目录",
            "公司开业时间：2017 年 2 月",
        ]

        with patch(
            "services.solvency_pdf_locator.extract_page_texts",
            return_value=page_texts,
        ):
            metadata = extract_report_metadata(b"placeholder")

        self.assertEqual(metadata["报告年度"], 2026)
        self.assertEqual(metadata["报告季度"], "Q1")
        self.assertEqual(metadata["报告期"], "2026Q1")

    def test_limited_liability_insurer_name_is_extracted_from_cover(self):
        page_texts = [
            (
                "保险公司偿付能力季度报告摘要\n"
                "中国人民养老保险有限责任公司\n"
                "2026年第1季度"
            ),
        ]

        with patch(
            "services.solvency_pdf_locator.extract_page_texts",
            return_value=page_texts,
        ):
            metadata = extract_report_metadata(b"placeholder")

        self.assertEqual(metadata["公司"], "中国人民养老保险有限责任公司")
        self.assertEqual(metadata["报告期"], "2026Q1")

    def test_reordered_cover_period_does_not_use_establishment_year_or_date(self):
        page_texts = [
            "偿付能力季度报告摘要 中银三星人寿保险有限公司 年第一季度2025",
            "公司信息 开业时间 2005年5月26日",
        ]
        with patch(
            "services.solvency_pdf_locator.extract_page_texts",
            return_value=page_texts,
        ):
            metadata = extract_report_metadata(b"placeholder")

        self.assertEqual(metadata["报告期"], "2025Q1")
        self.assertEqual(metadata["披露日期"], "")


if __name__ == "__main__":
    unittest.main()
