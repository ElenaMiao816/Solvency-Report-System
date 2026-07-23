from __future__ import annotations

import unittest

from services.solvency_pdf_locator import _expand_to_item_boundaries
from services.solvency_table_boundaries import (
    TableBoundaryError,
    boundary_items,
    enforce_output_boundaries,
)
from services.solvency_table_extractor import TABLE_SIGNATURES


class OperatingMetricsBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.header = [
            "\u6307\u6807\u540d\u79f0",
            "\u672c\u5b63\u5ea6\u6570",
            "\u672c\u5e74\u5ea6\u7d2f\u8ba1\u6570",
        ]

    def test_limited_disclosure_can_end_at_comprehensive_return(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u51c0\u5229\u6da6", "10", "18"],
            ["\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", "3.2%", "3.1%"],
            ["\u524d\u4e94\u5927\u4ea7\u54c1\u7684\u4fe1\u606f", "", ""],
        ]

        bounded, note = enforce_output_boundaries("OPERATING_METRICS", rows)

        self.assertEqual(
            [row[0] for row in bounded],
            [
                "\u6307\u6807\u540d\u79f0",
                "\u4fdd\u9669\u4e1a\u52a1\u6536\u5165",
                "\u51c0\u5229\u6da6",
                "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387",
            ],
        )
        self.assertIn("\u5b9e\u9645\u7ec8\u6b62\u9879\u76ee\uff1a\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", note)
        self.assertIn("\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", boundary_items("OPERATING_METRICS"))

    def test_full_disclosure_prefers_attrition_over_fallback(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", "3.2%", "3.1%"],
            ["\u6548\u76ca\u7c7b\u6307\u6807", "", ""],
            ["\u7efc\u5408\u9000\u4fdd\u7387", "1.1%", "1.0%"],
            ["\u54c1\u8d28\u7c7b\u6307\u6807", "", ""],
            ["\u8425\u9500\u5458\u8131\u843d\u7387", "8%", "9%"],
            ["\u4e0b\u4e00\u5f20\u8868", "", ""],
        ]

        bounded, note = enforce_output_boundaries("OPERATING_METRICS", rows)

        labels = [row[0] for row in bounded]
        self.assertIn("\u7efc\u5408\u9000\u4fdd\u7387", labels)
        self.assertEqual(labels[-1], "\u8425\u9500\u5458\u8131\u843d\u7387")
        self.assertIn("\u5b9e\u9645\u7ec8\u6b62\u9879\u76ee\uff1a\u8425\u9500\u5458\u8131\u843d\u7387", note)

    def test_missing_all_supported_end_items_still_fails(self):
        rows = [
            self.header,
            ["\u4fdd\u9669\u4e1a\u52a1\u6536\u5165", "100", "190"],
            ["\u51c0\u5229\u6da6", "10", "18"],
        ]

        with self.assertRaises(TableBoundaryError):
            enforce_output_boundaries("OPERATING_METRICS", rows)

    def test_candidate_signature_requires_only_disclosure_start(self):
        self.assertEqual(TABLE_SIGNATURES["OPERATING_METRICS"], ("\u4fdd\u9669\u4e1a\u52a1\u6536\u5165",))


class OperatingMetricsLocatorTests(unittest.TestCase):
    table = {"table_id": "OPERATING_METRICS", "max_pages": 5}

    def test_limited_disclosure_closes_at_fallback_page(self):
        page_texts = [
            "\u4e3b\u8981\u7ecf\u8425\u6307\u6807\n\u4fdd\u9669\u4e1a\u52a1\u6536\u5165 100 90",
            "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387 3.2% 3.1%",
            "\u524d\u4e94\u5927\u4ea7\u54c1\u7684\u4fe1\u606f",
        ]

        expanded = _expand_to_item_boundaries(page_texts, self.table, [(1, 8.0, ["anchor"])])

        self.assertEqual([page for page, _, _ in expanded], [1, 2])
        self.assertIn("\u7ec8\u6b62\u9879\u76ee\uff1a\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387", expanded[-1][2])

    def test_full_disclosure_closes_at_preferred_later_page(self):
        page_texts = [
            "\u4e3b\u8981\u7ecf\u8425\u6307\u6807\n\u4fdd\u9669\u4e1a\u52a1\u6536\u5165 100 90",
            "\u7efc\u5408\u6295\u8d44\u6536\u76ca\u7387 3.2% 3.1%",
            "\u6548\u76ca\u7c7b\u6307\u6807\n\u7efc\u5408\u9000\u4fdd\u7387 1.1% 1.0%",
            "\u54c1\u8d28\u7c7b\u6307\u6807\n\u8425\u9500\u5458\u8131\u843d\u7387 8% 9%",
        ]

        expanded = _expand_to_item_boundaries(page_texts, self.table, [(1, 8.0, ["anchor"])])

        self.assertEqual([page for page, _, _ in expanded], [1, 2, 3, 4])
        self.assertIn("\u7ec8\u6b62\u9879\u76ee\uff1a\u8425\u9500\u5458\u8131\u843d\u7387", expanded[-1][2])


if __name__ == "__main__":
    unittest.main()
