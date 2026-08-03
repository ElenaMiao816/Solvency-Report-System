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


if __name__ == "__main__":
    unittest.main()
