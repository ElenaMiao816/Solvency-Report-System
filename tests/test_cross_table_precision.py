import io
import unittest

import pandas as pd

from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS, VALIDATION_COLUMNS, VLMV2ExtractionRun,
    validate_vlm_v2_metrics, evaluate_vlm_v2_step3_gate,
    refresh_vlm_v2_cross_table_checks, read_vlm_v2_extracted_tables,
    vlm_v2_workbook_bytes,
)


class CrossTablePrecisionTests(unittest.TestCase):
    def records(self, pairs=None, unit='万元'):
        pairs = pairs or {
            'QUANT_RISK_CAPITAL': [129138.041881, 129138.041882],
            'MINIMUM_CAPITAL': [137622.411233, 137622.411234],
        }
        rows = []
        for code, values in pairs.items():
            for i, value in enumerate(values):
                row = dict.fromkeys(METRIC_COLUMNS, '')
                row.update({'目标表ID': f'TABLE_{i}', '目标表名称': f'来源表{i}',
                    '指标编码': code, '指标名称': code, '指标语义键': code,
                    '状态': 'found', '数值': value, '标准数值': value,
                    '原始值': str(value), '单位': unit, '标准单位': unit,
                    '物理页码': i + 1, '原始标签': code, '证据原文': f'{code} {value}',
                    '期间口径': '本季度数', '置信度': .99})
                rows.append(row)
        return pd.DataFrame(rows, columns=METRIC_COLUMNS)

    def checks(self, records):
        checks = validate_vlm_v2_metrics(records, [])
        return checks[checks['规则'].eq('跨表一致性')]

    def old_run(self):
        records = self.records()
        validations = validate_vlm_v2_metrics(records, [])
        mask = validations['规则'].eq('跨表一致性')
        validations.loc[mask, '状态'] = '失败'
        validations.loc[mask, '说明'] = '同一指标在不同来源表中数值冲突。'
        return VLMV2ExtractionRun(records, validations, (), 0, 0)

    def test_beida_fangzheng_screenshot_values_pass_without_rounding_source(self):
        records = self.records()
        original = records.copy(deep=True)
        checks = self.checks(records)
        self.assertEqual(len(checks), 2)
        self.assertTrue(checks['状态'].eq('通过').all())
        self.assertTrue(checks['说明'].str.contains('精度差异可忽略').all())
        self.assertTrue(checks['期望值'].str.contains('0.0001万元', regex=False).all())
        pd.testing.assert_frame_equal(records, original)

    def test_currency_tolerance_is_one_yuan_not_relative_to_balance(self):
        for unit, delta in [('元', 1), ('千元', .001), ('万元', .0001), ('百万元', .000001), ('亿元', .00000001)]:
            with self.subTest(unit=unit):
                self.assertEqual(self.checks(self.records({'X': [1, 1 + delta]}, unit)).iloc[0]['状态'], '通过')
                self.assertEqual(self.checks(self.records({'X': [1, 1 + delta * 2]}, unit)).iloc[0]['状态'], '失败')
        self.assertEqual(self.checks(self.records({'X': [1000000000, 1000000001]})).iloc[0]['状态'], '失败')

    def test_icbc_allianz_integer_detail_agrees_with_precise_summary(self):
        records = self.records({'ANC_T2_CAPITAL': [1222.0, 1222.4832]})
        records.loc[0, '原始值'] = '1,222'
        records.loc[0, '证据原文'] = '附属二级资本 本季度末数 1,222 单位：万元'
        records.loc[1, '原始值'] = '1222.4832'
        records.loc[1, '证据原文'] = '附属二级资本 本季度数 1222.4832 单位：万元'

        check = self.checks(records).iloc[0]

        self.assertEqual(check['状态'], '通过')
        self.assertIn('四舍五入后逐对一致', check['期望值'])
        self.assertIn('精度差异可忽略', check['说明'])

    def test_integer_precision_without_literal_unit_evidence_stays_strict(self):
        records = self.records({'ANC_T2_CAPITAL': [1222.0, 1222.4832]})
        records.loc[0, '原始值'] = '1,222'
        records.loc[0, '证据原文'] = '附属二级资本 1,222'
        records.loc[1, '原始值'] = '1222.4832'
        records.loc[1, '证据原文'] = '附属二级资本 1222.4832 万元'

        self.assertEqual(self.checks(records).iloc[0]['状态'], '失败')

    def test_percentage_counts_and_three_value_range(self):
        self.assertEqual(self.checks(self.records({'RATE': [120, 120.0000005]}, '%')).iloc[0]['状态'], '通过')
        self.assertEqual(self.checks(self.records({'RATE': [120, 120.01]}, '%')).iloc[0]['状态'], '失败')
        self.assertEqual(self.checks(self.records({'COUNT': [100, 101]}, '人')).iloc[0]['状态'], '失败')
        self.assertEqual(self.checks(self.records({'X': [1, 1.00008, 1.00016]})).iloc[0]['状态'], '失败')

    def test_incompatible_units_and_nonfinite_numbers_do_not_pass(self):
        records = self.records({'X': [1, 1.0000001]})
        records.loc[1, '标准单位'] = '亿元'
        self.assertEqual(self.checks(records).iloc[0]['状态'], '失败')
        self.assertEqual(self.checks(self.records({'X': [1, float('inf')]})).iloc[0]['状态'], '失败')

    def test_saved_run_gate_and_export_recheck_only_precision_conflicts(self):
        run = self.old_run()
        original = run.validations.copy(deep=True)
        updated = refresh_vlm_v2_cross_table_checks(run)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertFalse(updated.validations['状态'].eq('失败').any())
        pd.testing.assert_frame_equal(run.validations, original)
        self.assertIs(refresh_vlm_v2_cross_table_checks(updated), updated)
        exported = pd.read_excel(io.BytesIO(vlm_v2_workbook_bytes(None, run)), sheet_name='确定性校验')
        self.assertFalse(exported['状态'].eq('失败').any())
        run.validations.loc[len(run.validations)] = ['REQUEST', 'TABLE_0', '', '失败', '请求覆盖完整性', '', '', '仍有失败请求']
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)

    def test_old_step2_upload_no_longer_requires_model_rerun(self):
        run = self.old_run()
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            run.records.to_excel(writer, sheet_name='VLM_v2指标结果', index=False)
            run.validations.to_excel(writer, sheet_name='确定性校验', index=False)
        tables = read_vlm_v2_extracted_tables(pd.ExcelFile(io.BytesIO(stream.getvalue())))
        self.assertEqual(len(tables), 2)
        values = [row['标准数值'] for table in tables for row in table.metric_records]
        self.assertIn(129138.041881, values)
        self.assertIn(129138.041882, values)

    def test_material_conflict_and_unverifiable_old_failure_stay_blocked(self):
        run = self.old_run()
        run.records.loc[1, '标准数值'] += 1
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        single = VLMV2ExtractionRun(run.records.iloc[:1], run.validations, (), 0, 0)
        self.assertFalse(evaluate_vlm_v2_step3_gate(single).passed)
