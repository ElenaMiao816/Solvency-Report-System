from __future__ import annotations

import unittest
from pathlib import Path

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
        self.assertEqual(profile.analysis["comparison_scope"], "WITHIN_PROFILE")
        self.assertEqual(len(profile.tables), 5)
        self.assertTrue(
            all(table.get("strategy_id") for table in profile.tables)
        )
        self.assertEqual(
            {table["table_id"] for table in profile.tables},
            {
                "SOLVENCY_MAIN",
                "OPERATING_METRICS",
                "ACTUAL_CAPITAL",
                "THREE_YEAR_INVESTMENT_RETURN",
                "MINIMUM_CAPITAL",
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
        self.assertNotIn("五类目标表", prompt)


if __name__ == "__main__":
    unittest.main()
