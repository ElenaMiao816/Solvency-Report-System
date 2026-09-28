from __future__ import annotations

import unittest

from services.solvency_ai_table_extractor import (
    PageGrid,
    _slice_grid_for_target,
    _source_completeness_profile,
)
from services.solvency_disclosure_normalizer import (
    THREE_YEAR_RETURN_TABLE_ID,
    normalize_three_year_return_rows,
)
from services.solvency_hybrid_pipeline import (
    _deterministic_three_year_rows,
    _merge_page_rows,
)
from services.solvency_pdf_locator import (
    _expand_to_item_boundaries,
    _three_year_page_has_explicit_value,
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

    def test_two_separate_sentences_become_two_canonical_rows(self):
        rows = [
            ["近三年平均投资收益率4.00%。"],
            ["近三年平均综合投资收益率为4.93%。"],
        ]

        normalized, _ = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            rows,
        )
        bounded, _ = enforce_output_boundaries(
            THREE_YEAR_RETURN_TABLE_ID,
            normalized,
        )

        self.assertEqual(bounded, [
            ["项目", "数值"],
            ["近三年平均投资收益率", "4.00%"],
            ["近三年平均综合投资收益率", "4.93%"],
        ])

    def test_quarterly_return_pair_before_title_cannot_replace_disclosure(self):
        page_texts = [
            "主要经营指标\n（九）投资收益率 0.53% 0.53%\n"
            "（十）综合投资收益率 0.36% 0.36%",
            "（五）近三年（综合）投资收益率\n"
            "近三年平均投资收益率4.00%。\n"
            "近三年平均综合投资收益率为4.93%。",
        ]
        selected = [(2, 20.0, ["标题：近三年（综合）投资收益率"])]

        expanded = _expand_to_item_boundaries(
            page_texts,
            {
                "table_id": THREE_YEAR_RETURN_TABLE_ID,
                "max_pages": 2,
            },
            selected,
        )

        self.assertEqual([page for page, _, _ in expanded], [2])

    def test_company_prefixed_sentences_beat_prior_quarterly_rows(self):
        page_texts = [
            "主要经营指标\n"
            "投资收益率 0.61% 0.61%\n"
            "综合投资收益率 1.55% 1.55%\n"
            "注：投资收益率、综合投资收益率根据公司财务报告数据。",
            "（四）近三年平均投资收益率\n"
            "公司近三年平均投资收益率为 4.76%。\n"
            "（五）近三年平均综合投资收益率\n"
            "公司近三年平均综合投资收益率实际为 6.18%。",
        ]
        table = {
            "table_id": THREE_YEAR_RETURN_TABLE_ID,
            "max_pages": 2,
            "title_terms": [
                "近三年平均投资收益率",
                "近三年平均综合投资收益率",
            ],
        }

        expanded = _expand_to_item_boundaries(
            page_texts,
            table,
            [(2, 32.0, ["近三年平均投资收益率"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [2])
        self.assertIn("边界版式：明确近三年双项披露", expanded[0][2])
        self.assertFalse(_three_year_page_has_explicit_value(page_texts[0]))
        self.assertTrue(_three_year_page_has_explicit_value(page_texts[1]))

        normalized, _ = normalize_three_year_return_rows(
            THREE_YEAR_RETURN_TABLE_ID,
            [
                ["公司近三年平均投资收益率为 4.76%。"],
                ["公司近三年平均综合投资收益率实际为 6.18%。"],
            ],
        )
        self.assertEqual(normalized, [
            ["项目", "数值"],
            ["近三年平均投资收益率", "4.76%"],
            ["近三年平均综合投资收益率", "6.18%"],
        ])

    def test_cross_page_explicit_pair_beats_quarterly_pair_on_anchor_page(self):
        page_texts = [
            "（四）近三年平均投资收益率\n"
            "近三年平均投资收益率 2.85%",
            "（五）近三年平均综合投资收益率\n"
            "近三年平均综合投资收益率 4.00%\n"
            "人身保险公司主要经营指标\n"
            "（九）投资收益率 1.21% 1.21%\n"
            "（十）综合投资收益率 -0.02% -0.02%",
        ]
        selected = [(2, 25.0, ["近三年平均综合投资收益率"])]

        expanded = _expand_to_item_boundaries(
            page_texts,
            {
                "table_id": THREE_YEAR_RETURN_TABLE_ID,
                "max_pages": 2,
            },
            selected,
        )

        self.assertEqual([page for page, _, _ in expanded], [1, 2])
        self.assertIn("边界版式：明确近三年双项披露", expanded[0][2])

    def test_split_title_values_beat_quarterly_rows_and_footnote_mentions(self):
        page_texts = [
            "（四）主要经营指标\n"
            "九、投资收益率（%） 0.11 0.11\n"
            "十、综合投资收益率（%） 0.27 0.27\n"
            "注：上述指标包括近三年平均投资收益率和"
            "近三年平均综合投资收益率",
            "（五）近三年平均投资收益率\n"
            "\n"
            "5.35%。\n"
            "2025年 2024年 2023年\n"
            "近3年投资收益率 5.85% 6.35% 3.86%\n"
            "（六）近三年平均综合投资收益率\n"
            "\n"
            "5.78%。\n"
            "近3年综合投资收益率 5.52% 6.83% 5.01%",
        ]

        expanded = _expand_to_item_boundaries(
            page_texts,
            {
                "table_id": THREE_YEAR_RETURN_TABLE_ID,
                "max_pages": 2,
            },
            [(1, 20.0, ["脚注和季度收益率候选"])],
        )

        self.assertEqual([page for page, _, _ in expanded], [2])
        self.assertIn("边界版式：明确近三年双项披露", expanded[0][2])
        self.assertFalse(_three_year_page_has_explicit_value(page_texts[0]))
        self.assertTrue(_three_year_page_has_explicit_value(page_texts[1]))

    def test_split_title_values_exclude_annual_breakdown_rows(self):
        grid_text = "\n".join([
            "000|（四）主要经营指标",
            "001|九、投资收益率（%） 0.11 0.11",
            "002|十、综合投资收益率（%） 0.27 0.27",
            "003|（五）近三年平均投资收益率",
            "004|5.35%。",
            "005|2025年 2024年 2023年",
            "006|近3年投资收益率 5.85% 6.35% 3.86%",
            "007|（六）近三年平均综合投资收益率",
            "008|5.78%。",
            "009|2025年 2024年 2023年",
            "010|近3年综合投资收益率 5.52% 6.83% 5.01%",
        ])
        grid = PageGrid(14, grid_text, 60)

        sliced = _slice_grid_for_target(
            THREE_YEAR_RETURN_TABLE_ID,
            grid_text,
        )
        expected_rows, required_terms = _source_completeness_profile(
            THREE_YEAR_RETURN_TABLE_ID,
            [grid],
        )

        self.assertIn("近三年平均投资收益率 5.35%", sliced)
        self.assertIn("近三年平均综合投资收益率 5.78%", sliced)
        self.assertNotIn("5.85%", sliced)
        self.assertNotIn("5.52%", sliced)
        self.assertNotIn("2025年", sliced)
        self.assertEqual(expected_rows, 2)
        self.assertEqual(required_terms, (
            "近三年平均投资收益率",
            "近三年平均综合投资收益率",
        ))

    def test_second_page_slice_keeps_explicit_value_not_quarterly_metrics(self):
        grid_text = "\n".join([
            "000|（五）近三年平均综合投资收益率",
            "001|近三年平均综合投资收益率 4.00%",
            "002|（六）人身保险公司主要经营指标",
            "003|指标名称 本季度数 本年度累计数",
            "004|（九）投资收益率 1.21% 1.21%",
            "005|（十）综合投资收益率 -0.02% -0.02%",
        ])

        sliced = _slice_grid_for_target(
            THREE_YEAR_RETURN_TABLE_ID,
            grid_text,
        )

        self.assertIn("近三年平均综合投资收益率 4.00%", sliced)
        self.assertNotIn("1.21%", sliced)
        self.assertNotIn("-0.02%", sliced)

    def test_source_profile_ignores_narrative_products_and_channels(self):
        grid_text = "\n".join([
            "000|综合投资收益率、近三年平均投资收益率、"
            "近三年平均综合投资收益率依据老准则结果",
            "001|按照偿付能力监管规则第18号编制",
            "002|前五大产品的信息如下",
            "003|产品名称 产品类型 签单保费",
            "004|产品甲 普通寿险 108,920,435.76",
            "005|分渠道的签单保费如下",
            "006|银保渠道 174,619,579.92 174,619,579.92",
            "007|（四）近三年（综合）投资收益率",
            "008|指标名称",
            "009|近三年平均投资收益率（%） 4.37%",
            "010|近三年平均综合投资收益率（%） 4.98%",
        ])
        grid = PageGrid(16, grid_text, 84)

        expected_rows, required_terms = _source_completeness_profile(
            THREE_YEAR_RETURN_TABLE_ID,
            [grid],
        )

        self.assertEqual(expected_rows, 2)
        self.assertEqual(required_terms, (
            "近三年平均投资收益率",
            "近三年平均综合投资收益率",
        ))

    def test_section_title_does_not_become_a_duplicate_numeric_row(self):
        grid_text = "\n".join([
            "028|（五）近三年（综合）投资收益率",
            "029|近三年平均投资收益率 3.28%",
            "030|近三年平均综合投资收益率 3.66%",
            "031|9",
        ])
        grid = PageGrid(11, grid_text, 20)

        sliced = _slice_grid_for_target(
            THREE_YEAR_RETURN_TABLE_ID,
            grid_text,
        )
        rows, note = _deterministic_three_year_rows(grid)
        expected_rows, _ = _source_completeness_profile(
            THREE_YEAR_RETURN_TABLE_ID,
            [grid],
        )

        self.assertEqual(len(sliced.splitlines()), 2)
        self.assertNotIn("（五）近三年（综合）投资收益率", sliced)
        self.assertEqual(expected_rows, 2)
        self.assertEqual(rows, [
            ["项目", "数值"],
            ["近三年平均投资收益率", "3.28%"],
            ["近三年平均综合投资收益率", "3.66%"],
        ])
        self.assertIn("归一化", note)
        self.assertEqual(
            _merge_page_rows(
                THREE_YEAR_RETURN_TABLE_ID,
                "近三年（综合）投资收益率",
                [11],
                {11: rows},
            ),
            rows,
        )

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
