import unittest

from services.solvency_ai_table_extractor import (
    PageGrid,
    _instructions,
    _slice_grid_for_target,
    _source_completeness_profile,
    _source_item_labels,
)
from services.solvency_hybrid_pipeline import _pipe_prompt


class HuahuiOperatingLayoutTests(unittest.TestCase):
    def setUp(self):
        self.first_page = "\n".join([
            "040|（三）主要经营指标",
            "041|指标名称 本季度数 本年累计数",
            "042|保险业务收入（元） 102,704.00 102,704.00",
            "043|净利润（元） -9,455,369.16 -9,455,369.16",
            "044|保险合同负债 未到期责任准备金 - -",
        ])
        self.continuation_page = "\n".join([
            "000|（元） 未决赔款准备金 - -",
            "001|寿险责任准备金 62,499,595.39 62,499,595.39",
            "002|长期健康险责任准备金 7,694,218.50 7,694,218.50",
            "003|基本每股收益（元） - -",
            "004|净资产收益率 -1.64% -1.64%",
            "005|总资产收益率 -1.38% -1.38%",
            "006|投资收益率 0.37% 0.37%",
            "007|综合投资收益率 0.37% 0.37%",
            "008|效益类指标 新业务利润率 0.00% 0.00%",
            "009|规模类指标 签单保费（元） 102,704.00 102,704.00",
            "010|品质类指标 营销员脱落率 NA NA",
            "011|其中，规模类指标——签单保费占前五位的产品信息：",
            "012|产品名称 产品类型 本年度签单保费（元）",
            "013|华汇人寿乐享安泰两全保险 两全保险 53,390.00",
            "014|（四）近三年平均（综合）投资收益率",
        ])

    def test_continuation_starts_with_reserve_rows_and_stops_before_products(self):
        sliced = _slice_grid_for_target(
            "OPERATING_METRICS",
            self.continuation_page,
        )

        self.assertTrue(sliced.startswith("000|（元） 未决赔款准备金"))
        self.assertIn("寿险责任准备金", sliced)
        self.assertIn("长期健康险责任准备金", sliced)
        self.assertIn("基本每股收益", sliced)
        self.assertIn("营销员脱落率", sliced)
        self.assertNotIn("产品名称", sliced)
        self.assertNotIn("近三年平均", sliced)

    def test_source_profile_keeps_placeholder_rows_without_title_pollution(self):
        grids = [
            PageGrid(8, self.first_page, 40),
            PageGrid(9, self.continuation_page, 60),
        ]

        expected_rows, _ = _source_completeness_profile(
            "OPERATING_METRICS",
            grids,
        )
        labels = _source_item_labels("OPERATING_METRICS", grids)

        self.assertGreaterEqual(expected_rows, 11)
        self.assertIn("保险业务收入元", labels)
        self.assertTrue(any("未决赔款准备金" in label for label in labels))
        self.assertIn("基本每股收益元", labels)
        self.assertFalse(any("主要经营指标指标名称" in label for label in labels))

    def test_prompt_calls_out_cross_page_liability_children(self):
        prompt = _instructions(
            "OPERATING_METRICS",
            "主要经营指标",
            [8, 9],
        )

        self.assertIn("跨页合并单元格必须续接", prompt)
        self.assertIn("未决赔款准备金", prompt)
        self.assertIn("基本每股收益", prompt)
        self.assertIn("类别标题必须按原顺序", prompt)

        page_prompt = _pipe_prompt(
            "OPERATING_METRICS",
            "主要经营指标",
            9,
        )
        self.assertIn("类别标题必须按原顺序", page_prompt)


if __name__ == "__main__":
    unittest.main()
