from __future__ import annotations

import unittest

from services.solvency_disclosure_normalizer import (
    THREE_YEAR_RETURN_TABLE_ID,
    normalize_three_year_return_rows,
)
from services.solvency_table_boundaries import enforce_output_boundaries


class ThreeYearReturnDisclosureTests(unittest.TestCase):
    def test_sentence_disclosure_becomes_two_canonical_rows(self):
        rows = [[
            "近三年平均投资收益率为6.74%，近三年平均综合投资收益率为8.92%。"
        ]]

        normalized, note = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            rows,
        )

        self.assertEqual(normalized, [
            ["项目", "数值"],
            ["近三年平均投资收益率", "6.74%"],
            ["近三年平均综合投资收益率", "8.92%"],
        ])
        self.assertIn("句式", note)

    def test_comprehensive_value_is_not_mistaken_for_ordinary_value(self):
        rows = [[
            "近三年平均综合投资收益率：8.92%；近三年平均投资收益率：6.74%。"
        ]]

        normalized, _ = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            rows,
        )

        self.assertEqual(normalized[1][1], "6.74%")
        self.assertEqual(normalized[2][1], "8.92%")

    def test_two_column_layout_is_supported(self):
        rows = [
            ["近三年投资收益率", "近三年综合投资收益率"],
            ["6.74％", "8.92％"],
        ]

        normalized, _ = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            rows,
        )

        self.assertEqual(normalized[1][1], "6.74%")
        self.assertEqual(normalized[2][1], "8.92%")

    def test_normalized_sentence_passes_existing_business_boundaries(self):
        normalized, _ = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            [["投资收益率为6.74%，综合投资收益率为8.92%。"]],
        )

        bounded, note = enforce_output_boundaries(
            THREE_YEAR_RETURN_TABLE_ID,
            normalized,
        )

        self.assertEqual(bounded, normalized)
        self.assertIn("实际终止项目", note)


if __name__ == "__main__":
    unittest.main()
