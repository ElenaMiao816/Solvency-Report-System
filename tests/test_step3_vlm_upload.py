from __future__ import annotations

import ast
import io
import re
import unittest
from pathlib import Path

import pandas as pd

from services.solvency_table_extractor import ExtractedTable
from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS, VALIDATION_COLUMNS, VLMV2ExtractionRun,
    read_vlm_v2_extracted_tables, vlm_v2_to_extracted_tables, vlm_v2_workbook_bytes,
)


class Step3VLMUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Execute the real upload reader without starting the entire authenticated app.
        cls.app_tree = ast.parse((Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8"))
        reader = next(node for node in cls.app_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "read_extracted_tables_workbook")
        reader.decorator_list = []
        namespace = dict(io=io, re=re, pd=pd, ExtractedTable=ExtractedTable,
                         read_vlm_v2_extracted_tables=read_vlm_v2_extracted_tables)
        exec(compile(ast.Module(body=[reader], type_ignores=[]), "upload_reader", "exec"), namespace)
        cls.read_upload = staticmethod(namespace[reader.name])

    def run_data(self):
        records = pd.DataFrame([
            {"目标表ID": "ACTUAL_CAPITAL", "目标表名称": "S02-实际资本明细表",
             "指标编码": "ACTUAL_CAPITAL", "指标名称": "实际资本", "状态": "found",
             "原始值": "468,123.68", "数值": 468123.68, "标准数值": 468123.68,
             "标准单位": "万元", "物理页码": 20, "期间口径": "本季度末数",
             "原始标签": "实际资本", "证据原文": "实际资本 468,123.68"},
            {"目标表ID": "MINIMUM_CAPITAL", "目标表名称": "S05-最低资本表",
             "指标编码": "ADDITIONAL_CAPITAL", "指标名称": "附加资本", "状态": "disclosed_zero",
             "原始值": "-", "数值": 0, "标准数值": 0, "标准单位": "万元",
             "物理页码": 23, "期间口径": "本季度末数", "原始标签": "附加资本",
             "证据原文": "附加资本 -"},
        ]).reindex(columns=METRIC_COLUMNS).fillna("")
        validations = pd.DataFrame(columns=VALIDATION_COLUMNS)
        return VLMV2ExtractionRun(records, validations, (), 0, 0)

    def test_download_upload_round_trip_matches_live_step2(self):
        run = self.run_data()
        uploaded = self.read_upload(vlm_v2_workbook_bytes(None, run), [])
        self.assertEqual(uploaded, vlm_v2_to_extracted_tables(run))
        self.assertEqual(len(uploaded), 2)
        self.assertEqual(float(uploaded[1].rows[1][1]), 0)
        self.assertEqual(uploaded[0].source_pages, [20])

    def test_uploaded_business_validation_failure_is_blocked(self):
        run = self.run_data()
        run.validations.loc[0] = ["CAPITAL_CHECK", "ACTUAL_CAPITAL", "ACTUAL_CAPITAL",
                                  "失败", "合计", 1, 2, "不平"]
        with self.assertRaisesRegex(ValueError, "确定性业务校验"):
            self.read_upload(vlm_v2_workbook_bytes(None, run), [])

    def test_old_uploaded_dash_na_is_upgraded_to_numeric_zero(self):
        run = self.run_data()
        run.records.loc[0, ['原始值', '状态', '数值', '标准数值']] = ['—', 'disclosed_na', '', '']
        uploaded = self.read_upload(vlm_v2_workbook_bytes(None, run), [])
        self.assertEqual(uploaded, vlm_v2_to_extracted_tables(run))
        record = uploaded[0].metric_records[0]
        self.assertEqual(record['状态'], 'disclosed_zero')
        self.assertEqual(record['标准数值'], 0)
        self.assertEqual(record['原始值'], '—')

    def test_upload_rechecks_period_instead_of_trusting_saved_gate(self):
        run = self.run_data()
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            run.records.loc[0, "期间口径"] = "本年度累计数"
            run.records.to_excel(writer, sheet_name="VLM_v2指标结果", index=False)
            run.validations.to_excel(writer, sheet_name="确定性校验", index=False)
            pd.DataFrame([{"状态": "通过"}]).to_excel(writer, sheet_name="STEP3前置门槛", index=False)
        with self.assertRaisesRegex(ValueError, "本季度期间口径"):
            self.read_upload(output.getvalue(), [])

    def test_missing_validation_sheet_has_actionable_error(self):
        output = io.BytesIO()
        self.run_data().records.to_excel(output, sheet_name="VLM_v2指标结果", index=False)
        with self.assertRaisesRegex(ValueError, "缺少.*确定性校验"):
            self.read_upload(output.getvalue(), [])

    def test_legacy_per_table_workbook_still_imports(self):
        output = io.BytesIO()
        pd.DataFrame([["项目", "期末数"], ["实际资本", "100"], ["【单位备注】", "万元"]]).to_excel(
            output, sheet_name="S02-实际资本明细表_P20", index=False, header=False)
        tables = self.read_upload(output.getvalue(), [
            {"table_id": "ACTUAL_CAPITAL", "table_name": "S02-实际资本明细表"}])
        self.assertEqual(tables[0].page, 20)
        self.assertEqual(tables[0].rows, [["项目", "期末数"], ["实际资本", "100"]])

    def test_actual_button_is_enabled_for_imported_workbook(self):
        from streamlit.testing.v1 import AppTest
        button = next(node for node in ast.walk(self.app_tree) if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "standardize_submitted" for t in node.targets))
        source = "\n".join([
            "import streamlit as st",
            "source_tables = st.session_state['tables']",
            "use_default_target = True",
            "target_upload = None",
            "step3_company_name = '测试公司'",
            ast.unparse(button),
            "if standardize_submitted: st.success('填报已触发')",
        ])
        app = AppTest.from_string(source)
        app.session_state["tables"] = self.read_upload(vlm_v2_workbook_bytes(None, self.run_data()), [])
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button[0].disabled)
        app.button[0].click().run()
        self.assertEqual(app.success[0].value, "填报已触发")


if __name__ == "__main__":
    unittest.main()
