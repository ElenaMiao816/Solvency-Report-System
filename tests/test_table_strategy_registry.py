from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

from services.solvency_pdf_locator import locate_tables
from services.table_strategy_handlers import (
    BoundaryRequest,
    CompletenessRequest,
    PostprocessRequest,
    PromptRequest,
    RECOGNIZED_ASSETS_V2_POSTPROCESS_HANDLER,
    _simplify_recognized_assets_columns,
)
from services.table_strategy_registry import (
    GENERIC_GRID_STRATEGY_ID,
    STRATEGY_ACTUAL_CAPITAL,
    STRATEGY_MINIMUM_CAPITAL,
    STRATEGY_OPERATING_METRICS,
    STRATEGY_RECOGNIZED_ASSETS,
    STRATEGY_RECOGNIZED_ASSETS_V2,
    STRATEGY_SOLVENCY_MAIN,
    STRATEGY_THREE_YEAR_RETURN,
    StrategyRegistryError,
    registered_table_strategies,
    resolve_table_strategy,
)


class TableStrategyRegistryTests(unittest.TestCase):
    def test_existing_life_solvency_tables_are_registered(self):
        expected = {
            "SOLVENCY_MAIN": STRATEGY_SOLVENCY_MAIN,
            "OPERATING_METRICS": STRATEGY_OPERATING_METRICS,
            "ACTUAL_CAPITAL": STRATEGY_ACTUAL_CAPITAL,
            "THREE_YEAR_INVESTMENT_RETURN": STRATEGY_THREE_YEAR_RETURN,
            "MINIMUM_CAPITAL": STRATEGY_MINIMUM_CAPITAL,
            "RECOGNIZED_ASSETS": STRATEGY_RECOGNIZED_ASSETS_V2,
        }

        self.assertGreaterEqual(len(registered_table_strategies()), 8)
        for table_id, strategy_id in expected.items():
            with self.subTest(table_id=table_id):
                strategy = resolve_table_strategy(table_id)
                self.assertEqual(strategy.strategy_id, strategy_id)
                self.assertEqual(strategy.table_id, table_id)

    def test_recognized_assets_v1_remains_registered_as_fallback(self):
        strategy = resolve_table_strategy(
            "RECOGNIZED_ASSETS",
            STRATEGY_RECOGNIZED_ASSETS,
        )
        self.assertEqual(strategy.strategy_id, STRATEGY_RECOGNIZED_ASSETS)
        self.assertEqual(strategy.table_id, "RECOGNIZED_ASSETS")

    def test_recognized_assets_v2_is_default(self):
        strategy = resolve_table_strategy("RECOGNIZED_ASSETS")
        self.assertEqual(strategy.strategy_id, STRATEGY_RECOGNIZED_ASSETS_V2)

    def test_simplify_recognized_assets_columns_keeps_recognized_value_only(self):
        rows = [
            ["行次", "项目", "期末数", "", "", "期初数", "", ""],
            ["", "", "账面价值", "非认可价值", "认可价值",
             "账面价值", "非认可价值", "认可价值"],
            ["1", "现金及流动性管理工具", "100", "20", "80", "90", "18", "72"],
            ["2", "认可资产合计", "", "", "800", "", "", "720"],
        ]
        simplified, changed = _simplify_recognized_assets_columns(rows)

        self.assertTrue(changed)
        self.assertEqual(simplified, [
            ["行次", "项目", "期末数", "期初数"],
            ["1", "现金及流动性管理工具", "80", "72"],
            ["2", "认可资产合计", "800", "720"],
        ])

    def test_simplify_recognized_assets_columns_skips_single_level_table(self):
        rows = [
            ["行次", "项目", "期末数", "期初数"],
            ["1", "现金及流动性管理工具", "80", "72"],
        ]
        simplified, changed = _simplify_recognized_assets_columns(rows)

        self.assertFalse(changed)
        self.assertEqual(simplified, rows)

    def test_recognized_assets_v2_postprocess_simplifies_and_normalizes_total(self):
        rows = [
            ["行次", "项目", "期末数", "", "", "期初数", "", ""],
            ["", "", "账面价值", "非认可价值", "认可价值",
             "账面价值", "非认可价值", "认可价值"],
            ["1", "现金及流动性管理工具", "100", "20", "80", "90", "18", "72"],
            ["合计", "", "", "", "800", "", "", "720"],
        ]
        result, notes = RECOGNIZED_ASSETS_V2_POSTPROCESS_HANDLER(
            PostprocessRequest(table_id="RECOGNIZED_ASSETS", rows=rows)
        )

        self.assertEqual(result, [
            ["行次", "项目", "期末数", "期初数"],
            ["1", "现金及流动性管理工具", "80", "72"],
            ["认可资产合计", "", "800", "720"],
        ])
        self.assertTrue(any("化简" in note for note in notes))
        self.assertTrue(any("认可资产合计" in note for note in notes))

    def test_unregistered_table_uses_generic_strategy(self):
        strategy = resolve_table_strategy("NEW_SIMPLE_TABLE")

        self.assertEqual(strategy.strategy_id, GENERIC_GRID_STRATEGY_ID)
        self.assertEqual(strategy.table_id, "NEW_SIMPLE_TABLE")

    def test_strategy_cannot_be_assigned_to_another_table(self):
        with self.assertRaisesRegex(
            StrategyRegistryError,
            "仅适用于 ACTUAL_CAPITAL",
        ):
            resolve_table_strategy(
                "MINIMUM_CAPITAL",
                STRATEGY_ACTUAL_CAPITAL,
            )

    def test_dispatch_injects_bound_table_id_into_legacy_handler(self):
        strategy = resolve_table_strategy("ACTUAL_CAPITAL")

        def legacy_handler(*, table_id: str, value: int) -> tuple[str, int]:
            return table_id, value

        result = strategy.run(
            "extract_page",
            legacy_handler,
            value=7,
        )

        self.assertEqual(result, ("ACTUAL_CAPITAL", 7))

    def test_dispatch_stamps_strategy_on_extraction_result(self):
        strategy = resolve_table_strategy("MINIMUM_CAPITAL")

        @dataclass
        class Result:
            profile_strategy_id: str = ""

        result = strategy.run(
            "reconstruct_table",
            lambda *, table_id: Result(),
        )

        self.assertEqual(result.profile_strategy_id, STRATEGY_MINIMUM_CAPITAL)

    @patch(
        "services.solvency_pdf_locator.extract_page_texts",
        return_value=["实际资本表\n核心一级资本 100\n实际资本合计 200"],
    )
    def test_locator_carries_configured_strategy_to_page_match(self, _mock_text):
        matches = locate_tables(
            b"pdf",
            {
                "tables": [
                    {
                        "table_id": "ACTUAL_CAPITAL",
                        "table_name": "实际资本表",
                        "strategy_id": STRATEGY_ACTUAL_CAPITAL,
                        "title_terms": ["实际资本表"],
                        "content_terms": ["核心一级资本", "实际资本合计"],
                        "minimum_score": 1,
                        "max_pages": 1,
                    }
                ]
            },
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].strategy_id, STRATEGY_ACTUAL_CAPITAL)

    def test_prompt_handler_is_selected_by_strategy(self):
        strategy = resolve_table_strategy("THREE_YEAR_INVESTMENT_RETURN")
        prompt = strategy.build_prompt(PromptRequest(
            mode="single_page",
            table_id=strategy.table_id,
            table_name="近三年投资收益率",
            page_number=8,
        ))

        self.assertIn("一句或两句", prompt)
        self.assertIn("近三年平均综合投资收益率|数值", prompt)

    def test_profile_prompt_overrides_registered_compatibility_prompt(self):
        strategy = resolve_table_strategy(
            "THREE_YEAR_INVESTMENT_RETURN",
            table_config={
                "prompt_single_page_note": "PROFILE_V2_PROMPT_MARKER",
            },
        )
        prompt = strategy.build_prompt(PromptRequest(
            mode="single_page",
            table_id=strategy.table_id,
            table_name="近三年投资收益率",
            page_number=8,
        ))

        self.assertIn("PROFILE_V2_PROMPT_MARKER", prompt)

    def test_profile_postprocess_actions_can_disable_legacy_special_cleaning(self):
        calls = []

        def normalize(table_id, rows):
            calls.append(table_id)
            return rows, "normalized"

        strategy = resolve_table_strategy(
            "THREE_YEAR_INVESTMENT_RETURN",
            table_config={"postprocess_actions": ["trim_adjacent_rows"]},
        )
        rows, notes = strategy.postprocess(PostprocessRequest(
            table_id=strategy.table_id,
            rows=[["项目", "数值"]],
            normalize_three_year=normalize,
        ))

        self.assertEqual(calls, [])
        self.assertEqual(rows, [["项目", "数值"]])
        self.assertEqual(notes, ())

    def test_boundary_handler_controls_operating_grid_variant(self):
        calls: list[tuple[str, bool]] = []

        def engine(
            table_id: str,
            grid_text: str,
            *,
            operating_metrics: bool,
        ) -> str:
            calls.append((table_id, operating_metrics))
            return grid_text

        strategy = resolve_table_strategy("OPERATING_METRICS")
        result = strategy.enforce_boundaries(BoundaryRequest(
            mode="grid_slice",
            table_id=strategy.table_id,
            grid_text="grid",
            engine=engine,
        ))

        self.assertEqual(result, "grid")
        self.assertEqual(calls, [("OPERATING_METRICS", True)])

    def test_postprocess_handlers_only_run_registered_special_cleaning(self):
        normalized = []
        recovered = []

        def normalize(table_id, rows):
            normalized.append(table_id)
            return rows, "normalized"

        def recover(table_id, rows, _grids):
            recovered.append(table_id)
            return rows, "recovered"

        three_year = resolve_table_strategy(
            "THREE_YEAR_INVESTMENT_RETURN"
        )
        _, three_year_notes = three_year.postprocess(PostprocessRequest(
            table_id=three_year.table_id,
            rows=[["项目", "数值"]],
            normalize_three_year=normalize,
            recover_minimum_capital=recover,
        ))
        actual_capital = resolve_table_strategy("ACTUAL_CAPITAL")
        actual_capital.postprocess(PostprocessRequest(
            table_id=actual_capital.table_id,
            rows=[["项目", "数值"]],
            normalize_three_year=normalize,
            recover_minimum_capital=recover,
        ))

        self.assertEqual(normalized, ["THREE_YEAR_INVESTMENT_RETURN"])
        self.assertEqual(recovered, [])
        self.assertEqual(three_year_notes, ("normalized",))

    def test_completeness_handler_owns_signature_requirement(self):
        def validator(
            table_id,
            rows,
            ragged_ratio,
            *,
            required_signature_hits,
        ):
            return table_id, required_signature_hits

        main = resolve_table_strategy("SOLVENCY_MAIN")
        operating = resolve_table_strategy("OPERATING_METRICS")
        main_result = main.validate_completeness(CompletenessRequest(
            mode="full_table",
            table_id=main.table_id,
            rows=[["项目", "数值"]],
            validator=validator,
            signatures=("核心偿付能力充足率", "综合偿付能力充足率"),
        ))
        operating_result = operating.validate_completeness(
            CompletenessRequest(
                mode="full_table",
                table_id=operating.table_id,
                rows=[["项目", "数值"]],
                validator=validator,
                signatures=("保险业务收入", "净利润"),
            )
        )

        self.assertEqual(main_result, ("SOLVENCY_MAIN", 2))
        self.assertEqual(operating_result, ("OPERATING_METRICS", 1))

    def test_actual_capital_uses_wrapped_label_recall_tolerance(self):
        def validator(
            table_id,
            rows,
            ragged_ratio,
            *,
            required_signature_hits,
            source_item_recall_ratio,
        ):
            return table_id, required_signature_hits, source_item_recall_ratio

        strategy = resolve_table_strategy("ACTUAL_CAPITAL")
        result = strategy.validate_completeness(CompletenessRequest(
            mode="full_table",
            table_id=strategy.table_id,
            rows=[["行次", "项目", "期末数", "期初数"]],
            validator=validator,
            signatures=("核心一级资本", "实际资本合计"),
        ))

        self.assertEqual(result, ("ACTUAL_CAPITAL", 1, 0.80))


if __name__ == "__main__":
    unittest.main()
