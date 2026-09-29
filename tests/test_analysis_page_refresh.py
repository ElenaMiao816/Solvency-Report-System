from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_decorator_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return ""


class AnalysisPageRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        cls.app_tree = ast.parse(cls.app_source)

    def test_step7_and_step8_follow_normal_full_page_reruns(self):
        functions = {
            node.name: node
            for node in self.app_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for function_name in (
            "render_company_report_workspace",
            "render_industry_report_workspace",
        ):
            decorators = {
                _decorator_name(item)
                for item in functions[function_name].decorator_list
            }
            self.assertNotIn("st.fragment", decorators)

    def test_step6_to_step8_share_company_and_industry_navigation(self):
        shared_sidebar = next(
            node
            for node in ast.walk(self.app_tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "analysis_tabs_active"
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "render_report_navigation"
                for child in ast.walk(node)
            )
        )
        navigation_calls = [
            child
            for child in ast.walk(shared_sidebar)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "render_report_navigation"
        ]
        self.assertEqual(len(navigation_calls), 2)
        titles = {
            keyword.value.value
            for call in navigation_calls
            for keyword in call.keywords
            if keyword.arg == "title" and isinstance(keyword.value, ast.Constant)
        }
        self.assertEqual(titles, {"公司报告导航", "行业分析导航"})

    def test_step0_to_step8_share_the_same_page_background(self):
        background_rule = (
            '[data-testid="stAppViewContainer"] {background:#F4F7FC;}'
        )
        global_style_source = self.app_source.split(
            "analysis_tabs_active =", 1
        )[0]
        self.assertIn(background_rule, global_style_source)
        self.assertEqual(self.app_source.count(background_rule), 1)

    def test_bubble_drag_selection_remains_browser_side(self):
        source = (ROOT / "step7_solvency.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        renderer = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_render_capital_efficiency_bubble_fragment"
        )
        renderer_source = ast.get_source_segment(source, renderer) or ""
        self.assertIn("st.fragment", {_decorator_name(item) for item in renderer.decorator_list})
        self.assertNotIn("on_select=", renderer_source)
        self.assertNotIn("selection_mode=", renderer_source)

    def test_only_stale_step7_elements_are_hidden_without_hiding_current_module(self):
        unsafe_parent_selector = '[class*="st-key-s7_report_module_"]:has('
        self.assertNotIn(unsafe_parent_selector, self.app_source)
        stale_child_selector = (
            '[class*="st-key-s7_report_module_"]\n'
            '      [data-testid="stElementContainer"][data-stale="true"]'
        )
        self.assertIn(stale_child_selector, self.app_source)
        self.assertIn(
            '[data-testid="stElementContainer"][data-stale="true"]',
            self.app_source,
        )
        self.assertIn(
            '[data-testid="stElementContainer"][data-stale="true"]:has(',
            self.app_source,
        )
        self.assertIn(
            '[class*="st-key-s7_report_module_"][data-stale="true"]',
            self.app_source,
        )
        self.assertNotIn(
            '[data-testid="stMainBlockContainer"]:has(',
            self.app_source,
        )
        self.assertNotIn("opacity:1!important; transition:none!important", self.app_source)
        cleanup_start = self.app_source.index(stale_child_selector)
        cleanup_end = self.app_source.index("@media print", cleanup_start)
        cleanup_rule = self.app_source[cleanup_start:cleanup_end]
        self.assertIn("display:none!important", cleanup_rule)
        self.assertNotIn("@media screen", cleanup_rule)


if __name__ == "__main__":
    unittest.main()
