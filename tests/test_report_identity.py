from __future__ import annotations

import unittest

from services.solvency_pdf_locator import report_identity_warning


class ReportIdentityWarningTests(unittest.TestCase):
    def test_warns_when_filename_and_pdf_company_conflict(self):
        warning = report_identity_warning(
            "横琴人寿2026Q1偿付能力季度报告摘要.pdf",
            {"公司": "太平人寿保险有限公司"},
        )

        self.assertIn("横琴人寿", warning)
        self.assertIn("太平人寿保险有限公司", warning)
        self.assertIn("继续按PDF正文内容定位", warning)

    def test_accepts_matching_company_short_name(self):
        warning = report_identity_warning(
            "太平人寿2026Q1偿付能力季度报告摘要.pdf",
            {"公司": "太平人寿保险有限公司"},
        )

        self.assertEqual("", warning)

    def test_accepts_company_brand_without_insurance_type(self):
        warning = report_identity_warning(
            "德华安顾2026Q1偿付能力季度报告摘要.pdf",
            {"公司": "德华安顾人寿保险有限公司"},
        )

        self.assertEqual("", warning)

    def test_ignores_generic_filename(self):
        warning = report_identity_warning(
            "2026Q1偿付能力季度报告摘要.pdf",
            {"公司": "太平人寿保险有限公司"},
        )

        self.assertEqual("", warning)


if __name__ == "__main__":
    unittest.main()
