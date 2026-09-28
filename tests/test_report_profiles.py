from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from services.report_profiles import (
    LIST_CONFIG_FIELDS,
    load_profile_registry,
    load_profile_workbook,
    profile_workbook_bytes,
)
from services.solvency_hybrid_pipeline import _locator_prompt


ROOT = Path(__file__).resolve().parents[1]


class ReportProfileTests(unittest.TestCase):
    def test_default_life_solvency_profile_loads_existing_targets(self):
        profiles = load_profile_registry(
            ROOT / "config" / "report_profiles",
            ROOT,
        )
        profile = profiles["LIFE_SOLVENCY"]

        self.assertEqual(profile.profile_name, "寿险偿付能力季度报告")
        self.assertEqual(profile.workbook_schema_version, "2.0")
        self.assertEqual(profile.config_version, "2.0")
        self.assertEqual(profile.analysis["comparison_scope"], "WITHIN_PROFILE")
        self.assertEqual(len(profile.tables), 7)
        self.assertEqual(len(profile.layout_variants), 13)
        self.assertEqual(len(profile.completeness_rules), 6)
        self.assertTrue(
            all(table.get("strategy_id") for table in profile.tables)
        )
        self.assertEqual(
            {table["table_id"] for table in profile.tables},
            {
                "SOLVENCY_MAIN",
                "OPERATING_METRICS",
                "REGISTERED_CAPITAL",
                "ACTUAL_CAPITAL",
                "THREE_YEAR_INVESTMENT_RETURN",
                "MINIMUM_CAPITAL",
                "RECOGNIZED_ASSETS",
            },
        )

    def test_profile_workbook_roundtrip_preserves_locator_inputs(self):
        profile = load_profile_registry(
            ROOT / "config" / "report_profiles",
            ROOT,
        )["LIFE_SOLVENCY"]
        restored = load_profile_workbook(
            profile_workbook_bytes(profile),
            project_root=ROOT,
        )

        self.assertEqual(restored.profile_id, profile.profile_id)
        self.assertEqual(restored.company_types, profile.company_types)
        self.assertEqual(
            restored.feature_config["version"],
            profile.feature_config["version"],
        )
        self.assertEqual(len(restored.tables), len(profile.tables))
        original_by_id = {
            table["table_id"]: table
            for table in profile.tables
        }
        for restored_table in restored.tables:
            original = original_by_id[restored_table["table_id"]]
            self.assertEqual(
                restored_table["table_name"],
                original["table_name"],
            )
            self.assertEqual(
                restored_table.get("max_pages"),
                original.get("max_pages"),
            )
            self.assertEqual(
                restored_table.get("strategy_id"),
                original.get("strategy_id"),
            )
            for config_field in LIST_CONFIG_FIELDS.values():
                self.assertEqual(
                    restored_table.get(config_field, []),
                    original.get(config_field, []),
                )
            for stage_field in (
                "canonical_headers",
                "prompt_full_table_note",
                "prompt_single_page_note",
                "boundary_actions",
                "postprocess_actions",
            ):
                self.assertEqual(
                    restored_table.get(stage_field),
                    original.get(stage_field),
                )
        self.assertEqual(len(restored.layout_variants), 13)
        self.assertEqual(len(restored.completeness_rules), 6)

    def test_life_v2_tables_expose_profile_driven_stage_parameters(self):
        profile = load_profile_registry(
            ROOT / "config" / "report_profiles",
            ROOT,
        )["LIFE_SOLVENCY"]
        tables = {table["table_id"]: table for table in profile.tables}

        self.assertEqual(
            tables["SOLVENCY_MAIN"]["boundary_actions"],
            ["extend_solvency_forecast"],
        )
        self.assertTrue(
            tables["OPERATING_METRICS"]["completeness_track_sections"]
        )
        self.assertEqual(
            tables["ACTUAL_CAPITAL"]["postprocess_actions"],
            ["normalize_actual_capital_total", "trim_adjacent_rows"],
        )
        self.assertEqual(
            tables["THREE_YEAR_INVESTMENT_RETURN"]["canonical_headers"],
            ["项目", "数值"],
        )
        self.assertTrue(
            tables["MINIMUM_CAPITAL"]["boundary_require_value_for_items"]
        )
        self.assertEqual(
            tables["RECOGNIZED_ASSETS"]["postprocess_actions"],
            ["simplify_recognized_assets_columns", "normalize_recognized_assets_total", "trim_adjacent_rows"],
        )
        self.assertEqual(
            tables["RECOGNIZED_ASSETS"]["canonical_headers"],
            ["行次", "项目", "期末数", "期初数"],
        )

    def test_life_v2_static_workbook_embeds_default_configuration(self):
        workbook_path = (
            ROOT / "config" / "report_profiles"
            / "LIFE_SOLVENCY_profile_v2.xlsx"
        )
        restored = load_profile_workbook(
            workbook_path.read_bytes(),
            project_root=ROOT,
            source_name=workbook_path.name,
        )

        self.assertEqual(restored.profile_id, "LIFE_SOLVENCY")
        self.assertEqual(restored.workbook_schema_version, "2.0")
        self.assertEqual(restored.config_version, "2.0")
        self.assertEqual(len(restored.tables), 6)
        self.assertEqual(len(restored.layout_variants), 13)
        self.assertEqual(len(restored.completeness_rules), 6)
        self.assertEqual(len(restored.taxonomy_frame()), 88)
        self.assertEqual(len(restored.company_frame()), 92)
        tables = {table["table_id"]: table for table in restored.tables}
        self.assertEqual(
            tables["ACTUAL_CAPITAL"]["postprocess_actions"],
            ["normalize_actual_capital_total", "trim_adjacent_rows"],
        )
        with pd.ExcelFile(workbook_path) as excel:
            sheets = excel.sheet_names
        self.assertEqual(
            sheets,
            [
                "报告类型", "目标表", "定位关键词", "版式变体", "字段字典",
                "完整性规则", "公司来源", "Gold样本",
            ],
        )

    def test_non_life_pilot_loads_v2_inputs_and_generic_operating_strategy(self):
        profile = load_profile_registry(
            ROOT / "config" / "report_profiles",
            ROOT,
        )["NON_LIFE_SOLVENCY"]

        self.assertEqual(profile.workbook_schema_version, "2.0")
        self.assertEqual(profile.company_types, ("财险",))
        self.assertEqual(len(profile.tables), 6)
        self.assertEqual(len(profile.layout_variants), 5)
        self.assertGreaterEqual(len(profile.field_dictionary), 15)
        self.assertEqual(len(profile.companies), 2)
        operating = next(
            table for table in profile.tables
            if table["table_id"] == "NON_LIFE_OPERATING_METRICS"
        )
        self.assertEqual(operating["strategy_id"], "generic.grid_table.v1")
        self.assertEqual(operating["minimum_rows"], 5)
        self.assertTrue(operating["boundary_variants"])
        registered_capital = next(
            table for table in profile.tables
            if table["table_id"] == "REGISTERED_CAPITAL"
        )
        self.assertEqual(
            registered_capital["strategy_id"],
            "generic.grid_table.v1",
        )
        self.assertEqual(registered_capital["max_pages"], 1)

    def test_non_life_v2_workbook_roundtrip_embeds_all_configuration_sheets(self):
        workbook_path = (
            ROOT / "config" / "report_profiles"
            / "NON_LIFE_SOLVENCY_profile_v2.xlsx"
        )
        restored = load_profile_workbook(
            workbook_path.read_bytes(),
            project_root=ROOT,
            source_name=workbook_path.name,
        )

        self.assertEqual(restored.profile_id, "NON_LIFE_SOLVENCY")
        self.assertEqual(restored.workbook_schema_version, "2.0")
        self.assertEqual(len(restored.layout_variants), 5)
        self.assertGreaterEqual(len(restored.taxonomy_frame()), 15)
        self.assertEqual(len(restored.company_frame()), 2)
        with pd.ExcelFile(workbook_path) as excel:
            sheets = excel.sheet_names
        self.assertEqual(
            sheets,
            [
                "报告类型", "目标表", "定位关键词", "版式变体", "字段字典",
                "完整性规则", "公司来源", "Gold样本",
            ],
        )

    def test_locator_prompt_uses_profile_identity_and_dynamic_targets(self):
        prompt = _locator_prompt(
            target_names=["目标表A", "目标表B"],
            boundary_hints="- 目标表A：从A到B",
            radar_hints="- 目标表A：第3页",
            directory_hints={"A": [3]},
            scan_text="---PDF物理第3页---",
            profile_context={
                "profile_name": "测试报告",
                "prompt_role": "测试报告审阅专家",
                "instructions": ["仅使用测试口径。"],
            },
        )

        self.assertIn("测试报告审阅专家", prompt)
        self.assertIn("当前报告 profile：测试报告", prompt)
        self.assertIn("仅使用测试口径", prompt)
        self.assertIn('"目标表A"', prompt)
        self.assertIn("目录/标题双通道候选", prompt)
        self.assertIn("目录驱动", prompt)
        self.assertIn("标题驱动", prompt)
        self.assertNotIn("五类目标表", prompt)


if __name__ == "__main__":
    unittest.main()
