from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from services.solvency_company_identity import (
    apply_company_identities,
    company_display_name,
    display_company_names,
    load_peer_group_config,
    resolve_company_identity,
    resolve_company_identity_from_text,
    resolve_peer_group,
)
from services.solvency_normalizer import STANDARD_COLUMNS, upgrade_standard_frame


class CompanyIdentityTests(unittest.TestCase):
    def test_chart_names_use_short_labels_without_mutating_source_data(self):
        full_names = [
            "中银三星人寿保险有限公司",
            "中邮人寿保险股份有限公司",
            "招商信诺人寿保险有限公司",
            "建信人寿保险股份有限公司",
            "未知保险公司有限公司",
        ]
        frame = pd.DataFrame({"公司": full_names, "数值": [1, 2, 3, 4, 5]})
        display = display_company_names(frame)

        self.assertEqual(
            display["公司"].tolist(),
            ["中银三星", "中邮人寿", "招商信诺", "建信人寿", "未知"],
        )
        self.assertEqual(frame["公司"].tolist(), full_names)
        self.assertEqual(company_display_name("新华保险"), "新华保险")

    def test_colliding_display_names_keep_distinct_legal_names(self):
        frame = pd.DataFrame({"公司": ["甲保险有限公司", "甲保险股份有限公司"]})
        self.assertEqual(display_company_names(frame)["公司"].tolist(), frame["公司"].tolist())
        self.assertNotEqual(
            resolve_company_identity("甲保险有限公司").company_code,
            resolve_company_identity("甲保险股份有限公司").company_code,
        )

    def test_registered_legal_and_short_names_share_one_identity(self):
        full = resolve_company_identity("工银安盛人寿保险有限公司")
        short = resolve_company_identity("工银安盛")
        self.assertEqual(full.standard_name, "工银安盛")
        self.assertEqual(full.company_code, short.company_code)
        frame = pd.DataFrame([
            {"公司": "工银安盛人寿保险有限公司", "报告期": "2025Q1"},
            {"公司": "工银安盛", "报告期": "2026Q1"},
        ])
        result = apply_company_identities(frame)
        self.assertEqual(result["公司"].tolist(), ["工银安盛", "工银安盛"])
        self.assertEqual(result["公司统一编码"].nunique(), 1)
        self.assertEqual(result["原始公司名称"].tolist(), [
            "工银安盛人寿保险有限公司", "工银安盛",
        ])

    def test_screenshot_company_pairs_resolve_to_seven_not_fourteen(self):
        pairs = (
            ("工银安盛人寿保险有限公司", "工银安盛"),
            ("建信人寿保险股份有限公司", "建信人寿"),
            ("交银人寿保险有限公司", "交银人寿"),
            ("农银人寿保险股份有限公司", "农银人寿"),
            ("招商信诺人寿保险有限公司", "招商信诺"),
            ("中银三星人寿保险有限公司", "中银三星"),
            ("中邮人寿保险股份有限公司", "中邮人寿"),
        )
        names = [name for pair in pairs for name in pair]
        resolved = [resolve_company_identity(name) for name in names]
        self.assertEqual(len({item.standard_name for item in resolved}), 7)
        self.assertEqual(len({item.company_code for item in resolved}), 7)

    def test_historical_and_current_names_share_identity(self):
        company_types = {"中韩人寿": "寿险"}
        historical = resolve_company_identity("中韩人寿", company_types)
        current = resolve_company_identity("东方嘉富人寿", company_types)

        self.assertEqual(historical.standard_name, "东方嘉富人寿")
        self.assertEqual(current.standard_name, "东方嘉富人寿")
        self.assertEqual(historical.company_code, current.company_code)
        self.assertEqual(current.company_type, "寿险")
        self.assertEqual(historical.matched_by, "历史名称映射")

    def test_frame_preserves_original_name_and_standardizes_company(self):
        frame = pd.DataFrame([
            {"公司": "中韩人寿", "公司类型": "寿险", "报告期": "2024Q4"},
            {"公司": "东方嘉富人寿", "公司类型": "", "报告期": "2025Q4"},
        ])
        result = apply_company_identities(frame, {"中韩人寿": "寿险"})

        self.assertEqual(result["公司"].tolist(), ["东方嘉富人寿", "东方嘉富人寿"])
        self.assertEqual(result["原始公司名称"].tolist(), ["中韩人寿", "东方嘉富人寿"])
        self.assertEqual(result["公司统一编码"].nunique(), 1)
        self.assertEqual(result["公司类型"].tolist(), ["寿险", "寿险"])

    def test_peer_group_resolves_alias_and_full_legal_name(self):
        peer_groups = {
            "东方嘉富人寿": "小型公司",
            "中国人寿": "大型公司",
        }

        self.assertEqual(resolve_peer_group("中韩人寿", peer_groups), "小型公司")
        self.assertEqual(
            resolve_peer_group("中国人寿保险股份有限公司", peer_groups),
            "大型公司",
        )
        self.assertEqual(resolve_peer_group("未配置人寿", peer_groups), "其他")

    def test_people_insurance_pension_legal_name_resolves_to_configured_short_name(self):
        company_types = {"人保养老": "养老险"}
        identity = resolve_company_identity(
            "中国人民养老保险有限责任公司",
            company_types,
        )
        text_identity = resolve_company_identity_from_text(
            "中国人民养老保险有限责任公司2026年第一季度报告",
            company_types,
        )

        self.assertEqual(identity.standard_name, "人保养老")
        self.assertEqual(identity.company_type, "养老险")
        self.assertIsNotNone(text_identity)
        self.assertEqual(text_identity.standard_name, "人保养老")
        self.assertEqual(
            resolve_peer_group(
                "中国人民养老保险有限责任公司",
                {"人保养老": "养老健康"},
            ),
            "养老健康",
        )

    def test_frame_fills_blank_peer_group_but_preserves_reviewed_value(self):
        frame = pd.DataFrame([
            {"公司": "中韩人寿", "公司类型": "寿险", "同业分类": ""},
            {"公司": "中国人寿", "公司类型": "寿险", "同业分类": "自定义"},
        ])
        result = apply_company_identities(
            frame,
            peer_group_map={"东方嘉富人寿": "小型公司", "中国人寿": "大型公司"},
            default_peer_group="其他",
        )

        self.assertEqual(result["同业分类"].tolist(), ["小型公司", "自定义"])

    def test_peer_group_config_rejects_duplicate_company_membership(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "peer_groups.json"
            path.write_text(
                '{"groups":{"大型公司":["甲公司"],"银行系":["甲公司"]}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "同时属于"):
                load_peer_group_config(path)

    def test_upgrades_legacy_session_frame_to_current_schema(self):
        legacy = pd.DataFrame([{
            "公司": "中韩人寿",
            "公司类型": "寿险",
            "报告期": "2024Q4",
            "指标编码": "ACTUAL_CAPITAL",
            "数值": 1.0,
        }])
        result = upgrade_standard_frame(legacy)

        self.assertEqual(list(result.columns), STANDARD_COLUMNS)
        self.assertEqual(result.iloc[0]["原始公司名称"], "中韩人寿")
        self.assertEqual(result.iloc[0]["标准公司名称"], "东方嘉富人寿")
        self.assertEqual(result.iloc[0]["公司统一编码"], "LIFE_ORIENTAL_JIAFU")


if __name__ == "__main__":
    unittest.main()
