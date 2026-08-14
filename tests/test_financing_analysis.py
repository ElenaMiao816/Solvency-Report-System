from __future__ import annotations

import io
import unittest

import pandas as pd

from services.solvency_financing_analysis import (
    normalize_financing_frame,
    read_major_financing_workbook,
    split_solvency_change,
    summarize_financing,
)


class FinancingAnalysisTests(unittest.TestCase):
    def test_splits_total_change_and_financing_impact(self):
        total, impact = split_solvency_change(
            "季度总变动：+42%\n增资/发债的影响：未单独披露"
        )
        self.assertEqual(total, "+42%")
        self.assertEqual(impact, "未单独披露")

    def test_reads_workbook_and_calculates_summary(self):
        frame = pd.DataFrame([
            {
                "季度": "2025Q4",
                "公司名称": "甲人寿",
                "增资/发债": "增资10亿元",
                "综合充足率变动": "季度总变动：+10%\n增资/发债的影响：+9%",
            },
            {
                "季度": "2024Q4",
                "公司名称": "乙人寿",
                "增资/发债": "5亿元资本补充债",
                "综合充足率变动": "季度总变动：-1%\n增资/发债的影响：未单独披露",
            },
        ])
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="融资信息", index=False, startrow=2)
        result = read_major_financing_workbook(output.getvalue(), "financing.xlsx")
        summary = summarize_financing(result)
        self.assertEqual(result.iloc[0]["季度"], "2025Q4")
        self.assertEqual(result.iloc[0]["季度总变动"], "+10%")
        self.assertEqual(result.iloc[0]["增资/发债的影响"], "+9%")
        self.assertEqual(result.iloc[0]["来源行号"], 4)
        self.assertEqual(summary.event_count, 2)
        self.assertEqual(summary.company_count, 2)
        self.assertEqual(summary.latest_period, "2025Q4")

    def test_accepts_pre_split_change_columns(self):
        result = normalize_financing_frame(
            pd.DataFrame([{
                "季度": "2025年第2季度",
                "公司名称": "甲人寿",
                "增资/发债": "增资",
                "季度总变动": "+5%",
                "增资/发债的影响": "无",
            }]),
            source_filename="split.xlsx",
            source_sheet="Sheet1",
        )
        self.assertEqual(result.iloc[0]["季度"], "2025Q2")
        self.assertEqual(result.iloc[0]["增资/发债的影响"], "无")

    def test_rejects_unrecognized_quarter(self):
        with self.assertRaisesRegex(ValueError, "季度无法识别"):
            normalize_financing_frame(
                pd.DataFrame([{
                    "季度": "待定",
                    "公司名称": "甲人寿",
                    "增资/发债": "增资",
                    "综合充足率变动": "季度总变动：+5%",
                }]),
                source_filename="bad.xlsx",
                source_sheet="Sheet1",
            )


if __name__ == "__main__":
    unittest.main()
