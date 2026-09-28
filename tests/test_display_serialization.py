import ast
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
from openpyxl import load_workbook

from services.solvency_display import dataframe_for_display


class DisplaySerializationTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            '实际值': ['页面请求未完成', 2234689577.59, 'not_disclosed', None],
            '期望值': ['全部确认页成功读取', 0.0, '可核验', None],
            '数值': [100.5, '详见下表', '未披露', None],
            '标准数值': [0.0, 10.25, None, None],
        })

    def test_exact_reported_arrow_failures_are_fixed_without_mutation(self):
        source = self.fixture()
        original = source.copy(deep=True)
        with self.assertRaises((pa.ArrowInvalid, pa.ArrowTypeError)):
            pa.Table.from_pandas(source)
        display = dataframe_for_display(source)
        pa.Table.from_pandas(display)
        pd.testing.assert_frame_equal(source, original)
        self.assertEqual(display['数值'].tolist(), ['100.5', '详见下表', '未披露', ''])
        self.assertEqual(display['实际值'].iloc[1], '2234689577.59')
        self.assertEqual(display['标准数值'].iloc[0], 0)
        self.assertTrue(pd.api.types.is_numeric_dtype(display['标准数值']))

    def test_export_keeps_original_numbers_and_text(self):
        source = self.fixture()
        dataframe_for_display(source)
        output = io.BytesIO()
        source.to_excel(output, index=False)
        sheet = load_workbook(io.BytesIO(output.getvalue())).active
        self.assertEqual(sheet['C2'].value, 100.5)
        self.assertEqual(sheet['C2'].data_type, 'n')
        self.assertEqual(sheet['C3'].value, '详见下表')
        self.assertEqual(sheet['D2'].value, 0)

    def test_real_step2_dataframe_calls_do_not_need_streamlit_arrow_fallback(self):
        from streamlit.testing.v1 import AppTest
        tree = ast.parse((Path(__file__).parents[1] / 'app.py').read_text(encoding='utf-8'))
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'dataframe' and call.args:
                    expr = ast.unparse(call.args[0])
                    if expr in {'dataframe_for_display(validation_view)', 'dataframe_for_display(preview)'}:
                        calls.append(ast.unparse(node))
        self.assertEqual(len(calls), 2)
        source = '\n'.join([
            'import streamlit as st',
            'from services.solvency_display import dataframe_for_display',
            'validation_view = st.session_state["fixture"]',
            'preview = validation_view.copy()', *calls,
        ])
        app = AppTest.from_string(source)
        app.session_state['fixture'] = self.fixture()
        with patch('streamlit.dataframe_util.fix_arrow_incompatible_column_types', side_effect=AssertionError('unexpected Arrow fallback')):
            app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.dataframe), 2)

    def test_empty_numeric_and_categorical_frames(self):
        for frame in [pd.DataFrame(), pd.DataFrame({'x': [0, 1, None]}),
                      pd.DataFrame({'x': pd.Categorical([1.5, '未披露', None])})]:
            pa.Table.from_pandas(dataframe_for_display(frame))
