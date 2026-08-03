from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from services.solvency_ai_table_extractor import AIExtractionBundle, PageGrid
from services.solvency_hybrid_pipeline import (
    PAGE_VISION_MODE,
    extract_tables_hybrid,
)
from services.solvency_pdf_locator import PageMatch
from services.solvency_table_extractor import ExtractedTable


class ExtractionSchedulingTests(unittest.TestCase):
    def test_parallel_tables_preserve_results_and_reuse_page_artifacts(self):
        matches = [
            PageMatch("TABLE_A", "表A", [1], 10.0, "证据A"),
            PageMatch("TABLE_B", "表B", [1], 10.0, "证据B"),
        ]
        state_lock = threading.Lock()
        active = 0
        peak_active = 0

        def fake_build_grids(_pdf_bytes, pages):
            return [PageGrid(page, "", 0) for page in pages]

        def fake_render(_pdf_bytes, pages, **_kwargs):
            return [(page, f"image-{page}") for page in pages]

        def fake_extract_one_page(**kwargs):
            nonlocal active, peak_active
            kwargs["render_func"](
                kwargs["pdf_bytes"],
                [kwargs["page_number"]],
            )
            with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                time.sleep(0.06)
            finally:
                with state_lock:
                    active -= 1
            rows = [
                ["项目", "数值"],
                [kwargs["table_id"], "1"],
            ]
            return rows, PAGE_VISION_MODE, 95.0, [], []

        def fake_merge(_table_id, _table_name, pages, page_rows):
            return page_rows[pages[0]]

        patchers = [
            patch(
                "services.solvency_hybrid_pipeline.build_pdf_grids",
                side_effect=fake_build_grids,
            ),
            patch(
                "services.solvency_hybrid_pipeline._render_images",
                side_effect=fake_render,
            ),
            patch(
                "services.solvency_hybrid_pipeline._extract_one_page",
                side_effect=fake_extract_one_page,
            ),
            patch(
                "services.solvency_hybrid_pipeline._merge_page_rows",
                side_effect=fake_merge,
            ),
            patch(
                "services.solvency_hybrid_pipeline.enforce_output_boundaries",
                side_effect=lambda _table_id, rows, require_complete: (
                    rows,
                    "边界完整",
                ),
            ),
            patch(
                "services.solvency_hybrid_pipeline._source_completeness_profile",
                return_value=(0, []),
            ),
            patch(
                "services.solvency_hybrid_pipeline._source_item_labels",
                return_value=[],
            ),
            patch(
                "services.solvency_hybrid_pipeline._validate",
                return_value=(95.0, "完整性通过"),
            ),
            patch(
                "services.solvency_hybrid_pipeline.extract_unit_records",
                return_value=[],
            ),
            patch(
                "services.solvency_hybrid_pipeline.merge_unit_records",
                return_value=[],
            ),
        ]
        mocks = [patcher.start() for patcher in patchers]
        self.addCleanup(lambda: [patcher.stop() for patcher in reversed(patchers)])

        parallel_started = time.perf_counter()
        parallel = extract_tables_hybrid(
            b"pdf",
            matches,
            api_key="key",
            base_url="https://example.test/v1",
            model="vision",
        )
        parallel_elapsed = time.perf_counter() - parallel_started
        parallel_signature = [
            (table.table_id, table.rows, table.quality_score)
            for table in parallel.tables
        ]

        self.assertEqual([item[0] for item in parallel_signature], ["TABLE_A", "TABLE_B"])
        self.assertEqual(peak_active, 2)
        self.assertEqual(mocks[0].call_count, 1)
        self.assertEqual(mocks[1].call_count, 1)

        active = 0
        peak_active = 0
        mocks[0].reset_mock()
        mocks[1].reset_mock()
        serial_started = time.perf_counter()
        serial = extract_tables_hybrid(
            b"pdf",
            matches,
            api_key="key",
            base_url="https://example.test/v1",
            model="vision",
            _parallel_tables=False,
        )
        serial_elapsed = time.perf_counter() - serial_started
        serial_signature = [
            (table.table_id, table.rows, table.quality_score)
            for table in serial.tables
        ]

        self.assertEqual(serial_signature, parallel_signature)
        self.assertEqual(peak_active, 1)
        self.assertEqual(mocks[0].call_count, 1)
        self.assertEqual(mocks[1].call_count, 1)
        self.assertLess(parallel_elapsed, serial_elapsed * 0.8)

    def test_full_table_fallback_reuses_cached_page_artifacts(self):
        matches = [PageMatch("TABLE_A", "表A", [1], 10.0, "证据A")]

        def fake_build_grids(_pdf_bytes, pages):
            return [PageGrid(page, "grid", 10) for page in pages]

        def fake_render(_pdf_bytes, pages, **kwargs):
            zoom = kwargs.get("zoom", 1.8)
            quality = kwargs.get("jpeg_quality", 84)
            return [
                (page, f"image-{page}-{zoom}-{quality}")
                for page in pages
            ]

        def fake_extract_one_page(**kwargs):
            kwargs["render_func"](
                kwargs["pdf_bytes"],
                [kwargs["page_number"]],
            )
            return None, "", 0.0, [], []

        def fake_full_table_reconstruction(pdf_bytes, _matches, **kwargs):
            grids = kwargs["_grid_builder"](pdf_bytes, [1])
            standard_images = kwargs["_render_func"](pdf_bytes, [1])
            high_res_images = kwargs["_render_func"](
                pdf_bytes,
                [1],
                zoom=3.0,
                jpeg_quality=92,
            )
            self.assertEqual([grid.page_number for grid in grids], [1])
            self.assertEqual(
                standard_images,
                [(1, "image-1-1.8-84")],
            )
            self.assertEqual(
                high_res_images,
                [(1, "image-1-3.0-92")],
            )
            return AIExtractionBundle(
                tables=[ExtractedTable(
                    table_id="TABLE_A",
                    table_name="表A",
                    page=1,
                    table_index=1,
                    rows=[["项目", "数值"], ["项目A", "1"]],
                    source_pages=[1],
                )],
                logs=[],
            )

        with (
            patch(
                "services.solvency_hybrid_pipeline.build_pdf_grids",
                side_effect=fake_build_grids,
            ) as build_mock,
            patch(
                "services.solvency_hybrid_pipeline._render_images",
                side_effect=fake_render,
            ) as render_mock,
            patch(
                "services.solvency_hybrid_pipeline._extract_one_page",
                side_effect=fake_extract_one_page,
            ),
            patch(
                "services.solvency_hybrid_pipeline.extract_tables_with_llm",
                side_effect=fake_full_table_reconstruction,
            ),
        ):
            bundle = extract_tables_hybrid(
                b"pdf",
                matches,
                api_key="key",
                base_url="https://example.test/v1",
                model="vision",
            )

        self.assertEqual(len(bundle.tables), 1)
        self.assertEqual(bundle.tables[0].rows, [["项目", "数值"], ["项目A", "1"]])
        self.assertEqual(build_mock.call_count, 1)
        self.assertEqual(render_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
