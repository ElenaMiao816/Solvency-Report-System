from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from services.solvency_ai_table_extractor import PageGrid
from services.solvency_hybrid_pipeline import (
    PAGE_VISION_MODE,
    _extract_one_page,
    _low_text_page_numbers,
    locate_tables_hybrid,
)
from services.solvency_pdf_locator import PageMatch


TABLE_ID = "THREE_YEAR_INVESTMENT_RETURN"
TABLE_NAME = "近三年（综合）投资收益率"
TABLE_CONFIG = {
    "table_id": TABLE_ID,
    "table_name": TABLE_NAME,
    "title_terms": [TABLE_NAME],
    "content_terms": ["投资收益率", "综合投资收益率"],
    "max_pages": 2,
}


class FakeResponse:
    def __init__(self, content: str, *, ok: bool = True, status_code: int = 200):
        self.ok = ok
        self.status_code = status_code
        self.text = content
        self._content = content

    def json(self):
        if not self.ok:
            return {"error": {"message": self._content}}
        return {"choices": [{"message": {"content": self._content}}]}


class ScannedPdfLocatorTests(unittest.TestCase):
    def test_low_text_detection_is_page_level(self):
        self.assertEqual(
            _low_text_page_numbers(["", "仅页码6", "这是一个包含足够可检索文字的正常PDF页面内容。" * 2]),
            [1, 2],
        )

    @patch("services.solvency_hybrid_pipeline._render_locator_images")
    @patch("services.solvency_hybrid_pipeline._solvency_radar")
    @patch("services.solvency_hybrid_pipeline.locate_tables")
    def test_pure_scan_uses_two_stage_vision_and_returns_physical_page(
        self,
        mock_local,
        mock_radar,
        mock_render,
    ):
        mock_local.return_value = [PageMatch(TABLE_ID, TABLE_NAME, [], 0.0, "")]
        mock_radar.return_value = (
            ["", "", ""],
            {TABLE_ID: []},
            "- Python雷达未找到强候选页",
        )
        mock_render.side_effect = lambda _pdf, pages, **_kwargs: [
            (page, f"data:image/jpeg;base64,page{page}") for page in pages
        ]
        calls = []

        def fake_post(_url, **kwargs):
            calls.append(kwargs["json"])
            return FakeResponse(json.dumps({TABLE_NAME: [2]}, ensure_ascii=False))

        matches, message = locate_tables_hybrid(
            b"scan-pdf",
            {"tables": [TABLE_CONFIG]},
            api_key="key",
            base_url="https://example.test/v1",
            model="vision-model",
            post_func=fake_post,
        )

        self.assertEqual(matches[0].pages, [2])
        self.assertIn("图片候选扫描：2", matches[0].evidence)
        self.assertIn("图片候选确认：2", matches[0].evidence)
        self.assertIn("3/3个低文本页", message)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any(
            isinstance(message_item.get("content"), list)
            for payload in calls
            for message_item in payload["messages"]
        ))
        details = [
            item["image_url"]["detail"]
            for payload in calls
            for message_item in payload["messages"]
            if isinstance(message_item.get("content"), list)
            for item in message_item["content"]
            if item.get("type") == "image_url"
        ]
        self.assertIn("low", details)
        self.assertIn("high", details)

    @patch("services.solvency_hybrid_pipeline._render_locator_images")
    @patch("services.solvency_hybrid_pipeline._solvency_radar")
    @patch("services.solvency_hybrid_pipeline.locate_tables")
    def test_non_vision_model_returns_clear_scan_failure(
        self,
        mock_local,
        mock_radar,
        mock_render,
    ):
        mock_local.return_value = [PageMatch(TABLE_ID, TABLE_NAME, [], 0.0, "")]
        mock_radar.return_value = (["", ""], {TABLE_ID: []}, "no hints")
        mock_render.return_value = [
            (1, "data:image/jpeg;base64,page1"),
            (2, "data:image/jpeg;base64,page2"),
        ]

        def fake_post(_url, **_kwargs):
            return FakeResponse(
                "image input is not supported",
                ok=False,
                status_code=400,
            )

        matches, message = locate_tables_hybrid(
            b"scan-pdf",
            {"tables": [TABLE_CONFIG]},
            api_key="key",
            base_url="https://example.test/v1",
            model="text-only-model",
            post_func=fake_post,
        )

        self.assertEqual(matches[0].pages, [])
        self.assertIn("图片页码定位部分失败", message)
        self.assertIn("image input is not supported", message)


class ScannedPdfExtractionTests(unittest.TestCase):
    @patch("services.solvency_hybrid_pipeline._render_images")
    def test_image_sentence_is_normalized_before_validation(self, mock_render):
        mock_render.return_value = [(9, "data:image/jpeg;base64,page9")]

        def fake_post(_url, **_kwargs):
            return FakeResponse(
                "近三年平均投资收益率为6.74%，近三年平均综合投资收益率为8.92%。"
            )

        rows, mode, _score, logs, units = _extract_one_page(
            pdf_bytes=b"scan-pdf",
            table_id=TABLE_ID,
            table_name=TABLE_NAME,
            page_number=9,
            grid=PageGrid(9, "", 0),
            api_key="key",
            base_url="https://example.test/v1",
            model="vision-model",
            auto_vision_retry=True,
            force_vision=False,
            timeout=30,
            post_func=fake_post,
        )

        self.assertEqual(mode, PAGE_VISION_MODE)
        self.assertEqual(rows, [
            ["项目", "数值"],
            ["近三年平均投资收益率", "6.74%"],
            ["近三年平均综合投资收益率", "8.92%"],
        ])
        self.assertEqual(logs[-1].status, "成功")
        self.assertIn("归一化", logs[-1].message)
        self.assertTrue(all(item.normalized_unit == "%" for item in units))

    @patch("services.solvency_hybrid_pipeline._render_images")
    def test_image_unit_recovery_preserves_inline_source_unit(self, mock_render):
        mock_render.return_value = [(8, "data:image/jpeg;base64,page8")]
        responses = [
            "\n".join([
                "行次|项目|本季度末数|上季度末数",
                "5|实际资本合计|3,593,086,798.99|3,435,399,358.42",
            ]),
            json.dumps({
                "unit_records": [{
                    "scope": "行级",
                    "target": "实际资本合计",
                    "raw_unit": "元",
                    "source_page": 8,
                    "source_text": "实际资本（元）",
                    "confidence": "高",
                }],
            }, ensure_ascii=False),
        ]
        calls = []

        def fake_post(_url, **kwargs):
            calls.append(kwargs["json"])
            return FakeResponse(responses.pop(0))

        rows, mode, _score, logs, units = _extract_one_page(
            pdf_bytes=b"scan-pdf",
            table_id="ACTUAL_CAPITAL",
            table_name="S02-实际资本明细表",
            page_number=8,
            grid=PageGrid(8, "", 0),
            api_key="key",
            base_url="https://example.test/v1",
            model="vision-model",
            auto_vision_retry=True,
            force_vision=False,
            timeout=30,
            post_func=fake_post,
        )

        self.assertEqual(mode, PAGE_VISION_MODE)
        self.assertEqual(rows[1][1], "实际资本合计")
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].target, "实际资本合计")
        self.assertEqual(units[0].normalized_unit, "元")
        self.assertEqual(units[0].source_text, "实际资本（元）")
        self.assertIn("单位专用补提取获得1条明确单位", logs[-1].message)


if __name__ == "__main__":
    unittest.main()
