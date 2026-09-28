import unittest

from services.solvency_ai_table_extractor import (
    ExtractionQualityError,
    PageGrid,
    _instructions,
    _slice_grid_for_target,
    _source_completeness_profile,
    _source_item_labels,
    _validate,
)
from services.table_strategy_handlers import CompletenessRequest
from services.table_strategy_registry import resolve_table_strategy
from services.solvency_normalizer import _period_header
from services.solvency_hybrid_pipeline import _merge_page_rows


class SolvencyMainForecastTests(unittest.TestCase):
    def setUp(self):
        self.grid = "\n".join([
            "000|行次  项目                    期末数      期初数",
            "001|4.2   控制风险最低资本        7,032.96    6,582.73",
            "002|5     核心偿付能力溢额        27,116.93   52,852.48",
            "003|6     核心偿付能力充足率       120.28%     142.24%",
            "004|7     综合偿付能力溢额        46,828.90   72,779.22",
            "005|8     综合偿付能力充足率       135.03%     158.16%",
            "006|2.基本情景下的下季度预测数",
            "007|行次  项目                    下季度预测数",
            "008|1     实际资本                173,360.81",
            "009|2     核心资本                153,648.84",
            "010|3     最低资本                141,770.70",
            "011|4     核心偿付能力溢额        11,878.14",
            "012|5     核心偿付能力充足率       108.38%",
            "013|6     综合偿付能力溢额        31,590.11",
            "014|7     综合偿付能力充足率       122.28%",
            "015|（二）流动性风险监管指标",
            "016|LCR1  120.46%  105.26%",
        ])

    def test_grid_slice_keeps_forecast_and_stops_before_liquidity(self):
        sliced = _slice_grid_for_target("SOLVENCY_MAIN", self.grid)

        self.assertIn("基本情景下的下季度预测数", sliced)
        self.assertIn("核心资本", sliced)
        self.assertIn("122.28%", sliced)
        self.assertNotIn("流动性风险监管指标", sliced)
        self.assertNotIn("LCR1", sliced)

    def test_source_profile_requires_forecast_when_disclosed(self):
        grids = [PageGrid(11, self.grid, 120)]

        expected_rows, required_terms = _source_completeness_profile(
            "SOLVENCY_MAIN",
            grids,
        )
        labels = _source_item_labels("SOLVENCY_MAIN", grids)

        self.assertGreaterEqual(expected_rows, 12)
        self.assertIn("预测数", required_terms)
        self.assertIn("核心资本", labels)

    def test_prompt_requires_one_wide_comparable_table(self):
        prompt = _instructions(
            "SOLVENCY_MAIN",
            "偿付能力充足率指标",
            [10, 11],
        )

        self.assertIn("指标名称、本季度末数、上季度末数、下季度末预测数", prompt)
        self.assertIn("同名项目时必须合并到同一行", prompt)
        self.assertIn("核心资本", prompt)

    def test_completeness_rejects_missing_forecast_column(self):
        strategy = resolve_table_strategy("SOLVENCY_MAIN")
        rows = [
            ["指标名称", "本季度末数", "上季度末数"],
            ["认可资产", "3,238,122.19", "3,037,228.88"],
            ["实际资本", "180,525.01", "197,916.49"],
            ["最低资本", "133,696.11", "125,137.27"],
            ["核心偿付能力充足率", "120.28%", "142.24%"],
            ["综合偿付能力充足率", "135.03%", "158.16%"],
        ]

        with self.assertRaisesRegex(ExtractionQualityError, "预测数"):
            strategy.validate_completeness(CompletenessRequest(
                mode="full_table",
                table_id="SOLVENCY_MAIN",
                rows=rows,
                validator=_validate,
                validator_kwargs={
                    "expected_source_rows": 0,
                    "source_required_terms": ("预测数",),
                    "source_item_labels": (),
                    "source_section_titles": (),
                },
            ))

    def test_original_forecast_header_normalizes_to_comparable_period(self):
        self.assertEqual(
            _period_header(["指标名称", "下季度预测数"], 1),
            "下季度末预测数",
        )
        self.assertEqual(
            _period_header(
                ["指标名称", "基本情景下的下季度预测数"],
                1,
            ),
            "下季度末预测数",
        )

    def test_merge_discards_source_row_numbers_missing_from_standard_header(self):
        rows = _merge_page_rows(
            "SOLVENCY_MAIN",
            "偿付能力充足率指标",
            [10],
            {10: [
                ["指标名称", "本季度末数", "上季度末数", "下季度末预测数"],
                ["1", "认可资产", "59,762,185,553.49", "55,045,435,318.22", "63,600,204,450.48"],
                ["3.1", "核心一级资本", "2,855,607,010.41", "2,911,714,989.71", "2,835,233,069.07"],
                ["8", "综合偿付能力充足率", "184.61%", "207.24%", "164.97%"],
            ]},
        )

        self.assertEqual(
            rows[0],
            ["指标名称", "本季度末数", "上季度末数", "下季度末预测数"],
        )
        self.assertEqual(rows[1][0], "认可资产")
        self.assertEqual(rows[1][-1], "63,600,204,450.48")
        self.assertTrue(all(len(row) == 4 for row in rows))


if __name__ == "__main__":
    unittest.main()
