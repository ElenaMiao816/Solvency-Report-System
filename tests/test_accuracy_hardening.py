import unittest
from unittest.mock import Mock, patch

import pandas as pd

from services.solvency_ai_table_extractor import (
    ExtractionQualityError,
    PageGrid,
    _item_recall_profile,
    _source_completeness_profile,
    _source_item_labels,
)
from services.solvency_gold_standard import (
    evaluate_locator,
    find_gold_case,
    pdf_sha256,
)
from services.solvency_hybrid_pipeline import (
    PAGE_HIGH_RES_VISION_MODE,
    _extract_one_page,
    _locate_scan_directory_with_vision,
    _locator_conflict,
    _validate_cross_page_columns,
)
from services.solvency_normalizer import resolve_metric_match
from services.solvency_pdf_locator import (
    PageMatch,
    locate_directory_candidates,
)
from services.solvency_validator import validate_standard_data


class DirectoryLocatorTests(unittest.TestCase):
    def test_directory_printed_pages_are_converted_to_physical_pages(self):
        page_texts = [""] * 20
        page_texts[2] = "目录\n三、主要指标 ........ 5\n九、实际资本 ........ 14"
        page_texts[3] = "正文\n1"
        page_texts[7] = "偿付能力充足率指标\n5"
        page_texts[16] = "实际资本表\n14"
        tables = [
            {
                "table_id": "SOLVENCY_MAIN",
                "table_name": "偿付能力充足率指标",
                "title_terms": ["主要指标", "偿付能力充足率指标"],
            },
            {
                "table_id": "ACTUAL_CAPITAL",
                "table_name": "S02-实际资本明细表",
                "title_terms": ["实际资本", "实际资本表"],
            },
        ]
        candidates, message = locate_directory_candidates(page_texts, tables)
        self.assertEqual(candidates["SOLVENCY_MAIN"], [8])
        self.assertEqual(candidates["ACTUAL_CAPITAL"], [17])
        self.assertIn("偏移量=3", message)

    @patch("services.solvency_hybrid_pipeline._render_locator_images")
    @patch("services.solvency_hybrid_pipeline._call_chat")
    def test_scan_directory_is_parsed_by_vision(
        self,
        mock_chat,
        mock_render,
    ):
        tables = [{
            "table_id": "MINIMUM_CAPITAL",
            "table_name": "S05-最低资本表",
            "title_terms": ["最低资本表"],
        }]
        mock_render.return_value = [(page, f"image-{page}") for page in range(1, 11)]
        mock_chat.return_value = '{"S05-最低资本表":[20]}'
        candidates, message = _locate_scan_directory_with_vision(
            b"pdf",
            tables,
            list(range(1, 21)),
            20,
            api_key="key",
            base_url="https://example.test/v1",
            model="vision",
            timeout=30,
            post_func=Mock(),
        )
        self.assertEqual(candidates["MINIMUM_CAPITAL"], [20])
        self.assertIn("视觉解析生成1类候选", message)
        self.assertEqual(mock_render.call_args.args[1], list(range(1, 11)))


class LocatorConflictTests(unittest.TestCase):
    def test_closed_boundary_disagreement_is_isolated(self):
        required, reason = _locator_conflict(
            local_pages=[8],
            ai_pages=[8, 9],
            visual_pages=[8],
            directory_pages=[8],
            boundary_closed=True,
        )
        self.assertTrue(required)
        self.assertIn("不一致", reason)

    def test_consistent_sources_do_not_require_review(self):
        required, reason = _locator_conflict(
            local_pages=[8],
            ai_pages=[8],
            visual_pages=[8],
            directory_pages=[8],
            boundary_closed=True,
        )
        self.assertFalse(required)
        self.assertEqual(reason, "")

    def test_directory_only_evidence_requires_review(self):
        required, reason = _locator_conflict(
            local_pages=[],
            ai_pages=[],
            visual_pages=[],
            directory_pages=[20],
            boundary_closed=False,
        )
        self.assertTrue(required)
        self.assertIn("仅有目录候选", reason)


class CompletenessTests(unittest.TestCase):
    def test_source_item_recall_ignores_table_metadata_with_dates(self):
        grid = PageGrid(
            27,
            "\n".join([
                "001|公司名称：长城人寿保险股份有限公司 2026-03-31 单位：万元",
                "002|1 财务报表资产总额 17,677,527.30 17,275,442.21",
                "003|2 认可资产总额 17,582,518.95 17,321,539.94",
                "004|5 实际资本合计 1,390,038.35 1,630,168.62",
            ]),
            20,
        )
        labels = _source_item_labels("ACTUAL_CAPITAL", [grid])
        self.assertNotIn("公司名称长城人寿保险股份有限公司", labels)
        self.assertEqual(labels, (
            "财务报表资产总额",
            "认可资产总额",
            "实际资本合计",
        ))

    def test_source_item_recall_detects_missing_business_rows(self):
        grid = PageGrid(
            1,
            "\n".join([
                "001|认可资产 100 90",
                "002|认可负债 80 70",
                "003|实际资本 20 20",
                "004|最低资本 10 9",
                "005|综合偿付能力充足率 200% 220%",
            ]),
            20,
        )
        labels = _source_item_labels("SOLVENCY_MAIN", [grid])
        rows = [
            ["项目", "期末数", "期初数"],
            ["认可资产", "100", "90"],
            ["实际资本", "20", "20"],
            ["综合偿付能力充足率", "200%", "220%"],
        ]
        recall, missing = _item_recall_profile(rows, labels)
        self.assertLess(recall, 0.9)
        self.assertIn("认可负债", missing)
        self.assertIn("最低资本", missing)

    def test_operating_footnote_continuations_are_not_business_rows(self):
        grids = [
            PageGrid(
                13,
                "\n".join([
                    "001|指标名称 本季度数 本年度累计数",
                    "002|（一）保险业务收入（万元） 8,990,429.26 8,990,429.26",
                    "003|（十）综合投资收益率 0.40% 0.40%",
                    "004|3 注：下表指标根据公司财务报告数据披露，依据2017年准则编制",
                    "005|报告数据披露根据2020年修订的第22号和第25号准则",
                ]),
                30,
            ),
            PageGrid(
                14,
                "\n".join([
                    "001|6.5 其他渠道 60,254.87 60,254.87",
                    "002|（十三）品质类指标 -- --",
                    "003|5.营销员脱落率 0.00% 0.00%",
                    "004|（五）近三年（综合）投资收益率",
                ]),
                20,
            ),
        ]

        labels = _source_item_labels("OPERATING_METRICS", grids)
        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            grids,
        )
        output_rows = [
            ["指标名称", "本季度数", "本年度累计数"],
            ["（一）保险业务收入（万元）", "8,990,429.26", "8,990,429.26"],
            ["（十）综合投资收益率", "0.40%", "0.40%"],
            ["6.5 其他渠道", "60,254.87", "60,254.87"],
            ["5.营销员脱落率", "0.00%", "0.00%"],
        ]
        recall, missing = _item_recall_profile(output_rows, labels)

        self.assertEqual(expected_rows, 4)
        self.assertNotIn("报告数据披露根据", "".join(labels))
        self.assertEqual(recall, 1.0)
        self.assertEqual(missing, ())

    def test_cross_page_column_shift_is_rejected(self):
        page_rows = {
            1: [
                ["行次", "项目", "期末数", "期初数"],
                ["1", "核心一级资本", "100", "90"],
            ],
            2: [
                ["项目", "期末数", "期初数"],
                ["附属一级资本", "10", "9"],
            ],
        }
        with self.assertRaises(ExtractionQualityError):
            _validate_cross_page_columns([1, 2], page_rows)

    def test_consistent_cross_page_columns_pass(self):
        page_rows = {
            1: [
                ["行次", "项目", "期末数", "期初数"],
                ["1", "核心一级资本", "100", "90"],
            ],
            2: [
                ["行次", "项目", "期末数", "期初数"],
                ["2", "附属一级资本", "10", "9"],
            ],
        }
        self.assertIn("一致", _validate_cross_page_columns([1, 2], page_rows))


class AdaptiveVisionTests(unittest.TestCase):
    @patch("services.solvency_hybrid_pipeline.recover_unit_records_from_images")
    @patch("services.solvency_hybrid_pipeline._render_images")
    @patch("services.solvency_hybrid_pipeline._call_chat")
    def test_standard_image_failure_retries_at_three_times_zoom(
        self,
        mock_chat,
        mock_render,
        mock_recover,
    ):
        mock_render.side_effect = [
            [(9, "data:image/jpeg;base64,low")],
            [(9, "data:image/jpeg;base64,high")],
        ]
        mock_chat.side_effect = [
            RuntimeError("普通图片识别失败"),
            "近三年平均投资收益率|6.74%\n"
            "近三年平均综合投资收益率|8.92%",
        ]
        mock_recover.return_value = []
        rows, mode, _score, logs, _units = _extract_one_page(
            pdf_bytes=b"pdf",
            table_id="THREE_YEAR_INVESTMENT_RETURN",
            table_name="近三年（综合）投资收益率",
            page_number=9,
            grid=PageGrid(9, "", 0),
            api_key="key",
            base_url="https://example.test/v1",
            model="vision",
            auto_vision_retry=True,
            force_vision=False,
            timeout=30,
            post_func=Mock(),
        )
        self.assertEqual(mode, PAGE_HIGH_RES_VISION_MODE)
        self.assertEqual(len(rows), 3)
        self.assertEqual(mock_render.call_args_list[1].kwargs["zoom"], 3.0)
        self.assertTrue(any(item.mode == PAGE_HIGH_RES_VISION_MODE for item in logs))


class MetricAmbiguityTests(unittest.TestCase):
    def test_close_fuzzy_candidates_are_quarantined(self):
        taxonomy = pd.DataFrame([
            {"指标编码": "A", "指标名称": "指标甲A", "别名": ""},
            {"指标编码": "B", "指标名称": "指标甲B", "别名": ""},
        ])
        decision = resolve_metric_match("指标甲", taxonomy)
        self.assertTrue(decision.ambiguous)
        self.assertIsNone(decision.metric)

    def test_duplicate_exact_aliases_are_quarantined(self):
        taxonomy = pd.DataFrame([
            {"指标编码": "A", "指标名称": "完全同名指标", "别名": ""},
            {"指标编码": "B", "指标名称": "另一指标", "别名": "完全同名指标"},
        ])
        decision = resolve_metric_match("完全同名指标", taxonomy)
        self.assertTrue(decision.ambiguous)
        self.assertIsNone(decision.metric)

    def test_table_scope_prevents_cross_table_mapping(self):
        taxonomy = pd.DataFrame([
            {
                "指标编码": "ROE",
                "指标名称": "投资收益率",
                "别名": "",
                "一级模块": "经营指标",
            },
            {
                "指标编码": "INVESTMENT_RETURN",
                "指标名称": "近三年平均投资收益率",
                "别名": "投资收益率",
                "一级模块": "经营指标",
            },
        ])
        decision = resolve_metric_match(
            "投资收益率",
            taxonomy,
            table_id="THREE_YEAR_INVESTMENT_RETURN",
        )
        self.assertEqual(decision.metric["指标编码"], "INVESTMENT_RETURN")


class ExpandedValidationTests(unittest.TestCase):
    def test_duplicate_actual_capital_rule_detects_difference(self):
        frame = pd.DataFrame([
            {"指标编码": "ACTUAL_CAPITAL", "期间口径": "期末数", "数值": 100},
            {"指标编码": "ACTUAL_CAPITAL", "期间口径": "期末数", "数值": 103},
        ])
        rules = pd.DataFrame([{
            "规则ID": "DUPLICATE_ACTUAL_CAPITAL",
            "规则名称": "实际资本跨表一致性",
            "适用期间": "期末数",
            "容差": 1,
            "启用": "是",
        }])
        result = validate_standard_data(frame, rules)
        self.assertEqual(result.iloc[0]["status"], "未通过")
        self.assertEqual(result.iloc[0]["difference"], 3)


class GoldStandardTests(unittest.TestCase):
    def test_hash_match_and_locator_metrics(self):
        pdf_bytes = b"real-pdf-placeholder"
        case = {
            "case_id": "sample",
            "sha256": pdf_sha256(pdf_bytes),
            "tables": {"SOLVENCY_MAIN": {"pages": [8]}},
        }
        self.assertEqual(
            find_gold_case(pdf_bytes, {"cases": [case]})["case_id"],
            "sample",
        )
        result = evaluate_locator(
            case,
            [PageMatch("SOLVENCY_MAIN", "主表", [8], 10, "")],
        )
        self.assertEqual(result.iloc[0]["页码召回率"], 1.0)
        self.assertTrue(result.iloc[0]["是否完全一致"])


if __name__ == "__main__":
    unittest.main()
