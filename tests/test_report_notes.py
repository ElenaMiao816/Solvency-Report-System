from __future__ import annotations

import unittest

import pandas as pd

from services.solvency_report_notes import (
    NOTE_COLUMNS,
    company_notes_template,
    industry_notes_template,
    notes_lookup,
    notes_workbook_bytes,
    overlay_notes,
    read_notes_workbook,
)


class ReportNotesTests(unittest.TestCase):
    def test_pending_capital_notes_migrate_to_new_detail_charts(self):
        old = pd.DataFrame([
            {"对应图表名称": "核心资本明细占比-待定", "注释内容": "原核心资本注释"},
            {"对应图表名称": "附属资本明细占比-待定", "注释内容": "原附属资本注释"},
        ])
        lookup = notes_lookup(overlay_notes(company_notes_template(), old))
        self.assertEqual(lookup["核心一级资本明细"]["注释内容"], "原核心资本注释")
        self.assertEqual(lookup["附属一级资本明细"]["注释内容"], "原附属资本注释")

    def test_company_and_industry_templates_follow_navigation(self):
        company = company_notes_template()
        industry = industry_notes_template()
        self.assertEqual(company.columns.tolist(), NOTE_COLUMNS)
        self.assertIn("资本规模与结构", company["对应图表名称"].tolist())
        self.assertNotIn("资本结构占比", company["对应图表名称"].tolist())
        self.assertIn("核心及综合充足率", company["对应图表名称"].tolist())
        self.assertIn("核心资本占比", company["对应图表名称"].tolist())
        self.assertIn("资本使用效率与核心资本占比气泡图", company["对应图表名称"].tolist())
        self.assertNotIn("核心资本", company["对应图表名称"].tolist())
        self.assertNotIn("核心充足率柱状图", company["对应图表名称"].tolist())
        self.assertNotIn("综合充足率柱状图", company["对应图表名称"].tolist())
        self.assertIn("综合充足率变化", company["对应图表名称"].tolist())
        self.assertIn("核心资本/注册资本", company["对应图表名称"].tolist())
        self.assertNotIn("注册资本/核心资本率", company["对应图表名称"].tolist())
        self.assertNotIn("核心资本/注册资本率", company["对应图表名称"].tolist())
        self.assertIn("量化风险最低资本构成", company["对应图表名称"].tolist())
        self.assertEqual(
            industry["对应图表名称"].tolist()[:2],
            ["行业整体偿付能力概览", "重大融资信息统计"],
        )
        self.assertNotIn("调研公司分类列表", company["对应图表名称"].tolist())
        self.assertNotIn("调研公司分类列表", industry["对应图表名称"].tolist())
        self.assertIn("量化风险最低资本构成（行业合计）", industry["对应图表名称"].tolist())

    def test_notes_workbook_round_trip_and_overlay(self):
        template = company_notes_template()
        target = template.iloc[[0]].copy()
        target.loc[:, "分析内容-自定义"] = "测试分析"
        target.loc[:, "注释内容"] = "测试注释"
        payload = notes_workbook_bytes(target)
        uploaded = read_notes_workbook(payload, "notes.xlsx")
        merged = overlay_notes(template, uploaded)
        lookup = notes_lookup(merged)
        chart_name = str(target.iloc[0]["对应图表名称"])
        self.assertEqual(lookup[chart_name]["分析内容-自定义"], "测试分析")
        self.assertEqual(lookup[chart_name]["注释内容"], "测试注释")

    def test_notes_alias_headers_are_supported(self):
        source = pd.DataFrame([{
            "模块ID": "A1",
            "一级模块": "实际资本",
            "二级模块": "核心资本占比分布",
            "具体图表": "核心一级资本占比",
            "分析内容": "自定义内容",
            "注释": "底部注释",
        }])
        payload = notes_workbook_bytes(
            source.rename(columns={
                "一级模块": "一级分类",
                "二级模块": "二级分类",
                "具体图表": "对应图表名称",
                "分析内容": "分析内容-自定义",
                "注释": "注释内容",
            })
        )
        result = read_notes_workbook(payload)
        self.assertEqual(result.iloc[0]["对应图表名称"], "核心一级资本占比")

    def test_old_capital_chart_notes_migrate_to_retained_chart(self):
        uploaded = pd.DataFrame([{
            "模块ID": "OLD",
            "一级分类": "实际资本指标",
            "二级分类": "行业资本分级",
            "对应图表名称": "资本分级行业分布",
            "分析内容-自定义": "旧版资本分析",
        }])
        merged = overlay_notes(company_notes_template(), uploaded)
        lookup = notes_lookup(merged)
        self.assertEqual(lookup["资本规模与结构"]["分析内容-自定义"], "旧版资本分析")

    def test_old_combined_effect_note_migrates_to_new_combined_chart(self):
        uploaded = pd.DataFrame([{
            "模块ID": "OLD_EFFECT",
            "一级分类": "最低资本指标",
            "二级分类": "风险分散效应和损失吸收",
            "对应图表名称": "风险分散效应和损失吸收",
            "分析内容-自定义": "旧版效应分析",
        }])
        lookup = notes_lookup(overlay_notes(company_notes_template(), uploaded))
        self.assertEqual(lookup["量化风险最低资本构成"]["分析内容-自定义"], "旧版效应分析")

    def test_old_solvency_ratio_note_migrates_to_combo_chart(self):
        uploaded = pd.DataFrame([{
            "模块ID": "OLD_RATIO",
            "一级分类": "关键偿付数据概览",
            "二级分类": "资本充足率",
            "对应图表名称": "综合充足率柱状图",
            "分析内容-自定义": "旧版充足率分析",
        }])
        lookup = notes_lookup(overlay_notes(company_notes_template(), uploaded))
        self.assertEqual(
            lookup["核心及综合充足率"]["分析内容-自定义"],
            "旧版充足率分析",
        )

    def test_old_core_capital_note_migrates_to_ratio_chart(self):
        uploaded = pd.DataFrame([{
            "模块ID": "OLD_CORE_CAPITAL",
            "一级分类": "关键偿付数据概览",
            "二级分类": "资本充足率",
            "对应图表名称": "核心资本",
            "分析内容-自定义": "旧版核心资本分析",
        }])
        lookup = notes_lookup(overlay_notes(company_notes_template(), uploaded))
        self.assertEqual(
            lookup["核心资本占比"]["分析内容-自定义"],
            "旧版核心资本分析",
        )

    def test_old_capital_efficiency_note_migrates_to_core_registered_chart(self):
        uploaded = pd.DataFrame([{
            "模块ID": "OLD_CAPITAL_EFFICIENCY",
            "一级分类": "关键偿付数据概览",
            "二级分类": "资本使用效率",
            "对应图表名称": "核心资本/注册资本率",
            "分析内容-自定义": "旧版资本使用效率分析",
        }])
        lookup = notes_lookup(overlay_notes(company_notes_template(), uploaded))
        self.assertEqual(
            lookup["核心资本/注册资本"]["分析内容-自定义"],
            "旧版资本使用效率分析",
        )


if __name__ == "__main__":
    unittest.main()
