from __future__ import annotations

import unittest
from pathlib import Path


class VLMV2FormalFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")

    def test_step1_uses_vlm_but_keeps_manual_confirmation_and_pdf_preview(self):
        self.assertIn("vlm_run = locate_tables_vlm_v2(", self.source)
        self.assertNotIn("locate_tables_hybrid(", self.source)
        self.assertIn("sheets_per_call=1", self.source)
        self.assertIn("max_workers=2", self.source)
        self.assertIn("progress_callback=update_locator_progress", self.source)
        self.assertIn('"其他批次的定位结果已经保留', self.source)
        self.assertIn('"确认页码，进入下一步"', self.source)
        self.assertIn('st.markdown("#### 页面预览")', self.source)
        self.assertIn("render_pdf_page(", self.source)

    def test_step2_uses_vlm_and_tabbed_result_preview(self):
        self.assertIn("extraction_run = extract_metrics_vlm_v2(", self.source)
        self.assertNotIn("extract_tables_hybrid(", self.source)
        self.assertIn("progress_callback=update_extraction_progress", self.source)
        self.assertIn("request_max_attempts=2", self.source)
        self.assertIn("retry_max_workers=2", self.source)
        self.assertIn("max_pages_per_request=4", self.source)
        self.assertIn('st.markdown("### 📊 提取结果预览")', self.source)
        self.assertIn("st.tabs(preview_tabs)", self.source)
        self.assertIn('"一键下载结构化提取表（Excel）"', self.source)
        self.assertIn('"disclosed_zero": "0"', self.source)
        self.assertIn('"disclosed_na": "不适用"', self.source)

    def test_step3_mixed_value_preview_renders_missing_text_and_zero(self):
        import ast
        import pandas as pd
        from streamlit.testing.v1 import AppTest
        tree = ast.parse(self.source)
        nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                 any((isinstance(target, ast.Name) and target.id == 'step3_narrow_display') or
                     (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                      and target.value.id == 'step3_narrow_display') for target in node.targets)]
        snippet = '\n'.join([
            'import streamlit as st', 'import pandas as pd',
            'from services.solvency_normalizer import narrow_table_view',
            *[ast.unparse(node) for node in sorted(nodes, key=lambda node: node.lineno)],
            'st.dataframe(step3_narrow_display, width="stretch", hide_index=True)',
        ])
        app = AppTest.from_string(snippet)
        app.session_state['standard_data'] = pd.DataFrame([
            {'数值': None, '披露状态': '未披露'},
            {'数值': 0, '披露状态': '已披露为0'},
        ])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.dataframe[0].value['数值'].tolist(), ['未披露', '0.0'])

    def test_step3_uses_formal_step2_result_without_duplicate_vlm_source(self):
        self.assertNotIn('"使用 VLM v2 结果"', self.source)
        self.assertIn(
            ') if step3_source_mode == "使用 STEP2 提取结果" else None',
            self.source,
        )


if __name__ == "__main__":
    unittest.main()
