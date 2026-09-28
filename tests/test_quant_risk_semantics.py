import io
import json
import unittest
from unittest.mock import patch

import pandas as pd

from services.solvency_source_semantics import quant_risk_source_error
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import (
    _candidate_rank, _merge_page_records, _retry_plan, extract_metrics_vlm_v2,
    evaluate_vlm_v2_step3_gate, validate_vlm_v2_metrics, refresh_vlm_v2_cross_table_checks,
    read_vlm_v2_extracted_tables, vlm_v2_to_extracted_tables,
)
from tests.test_vlm_v2_pipeline import FakeResponse


MAIN = 'QUANT_RISK_CAPITAL'
BEFORE = 'QUANT_RISK_CAPITAL_BEFORE_FACTOR'
MAIN_LABEL = '量化风险最低资本'
BEFORE_LABEL = '量化风险最低资本（未考虑特征系数前）'


class QuantRiskSourceTests(unittest.TestCase):
    def test_before_factor_variants_cannot_fill_main(self):
        for suffix in ['（未考虑特征系数前）', '(未考虑特征系数)', '（不考虑特征系数）',
                       '（考虑特征系数前）', '（调整前）', '（未考虑公司特征系数）']:
            self.assertTrue(quant_risk_source_error(MAIN, MAIN_LABEL + suffix))
            self.assertFalse(quant_risk_source_error(BEFORE, MAIN_LABEL + suffix))
        for path in [['1*', MAIN_LABEL], '1＊ > 量化风险最低资本', '行次1*']:
            self.assertTrue(quant_risk_source_error(MAIN, MAIN_LABEL, path))
            self.assertFalse(quant_risk_source_error(BEFORE, MAIN_LABEL, path))

    def test_correct_labels_and_reverse_mismatch(self):
        for label in [MAIN_LABEL, MAIN_LABEL + '（考虑特征系数后）', MAIN_LABEL + '（调整后）']:
            self.assertFalse(quant_risk_source_error(MAIN, label))
            self.assertTrue(quant_risk_source_error(BEFORE, label))
        self.assertFalse(quant_risk_source_error('OTHER', BEFORE_LABEL))
        self.assertFalse(quant_risk_source_error(MAIN, MAIN_LABEL, ['1'], BEFORE_LABEL + '相邻行文字'))
        self.assertTrue(quant_risk_source_error(MAIN, '', '', BEFORE_LABEL))

    def test_valid_source_beats_higher_confidence_wrong_candidate(self):
        wrong = {'status': 'found', 'source_label': BEFORE_LABEL, 'confidence': .999}
        correct = {'status': 'found', 'source_label': MAIN_LABEL, 'confidence': .8}
        self.assertGreater(_candidate_rank(correct, MAIN), _candidate_rank(wrong, MAIN))
        def row(label):
            return {'目标表ID': 'MINIMUM_CAPITAL', '指标语义键': MAIN, '指标编码': MAIN,
                    '原始标签': label, '状态': 'found'}
        self.assertEqual(_merge_page_records([row(BEFORE_LABEL), row(MAIN_LABEL)])[0]['原始标签'], MAIN_LABEL)


class QuantRiskPipelineTests(unittest.TestCase):
    def run_case(self, correct_retry=True, wrong_code=MAIN, auto_retry=True):
        # Values transcribed from the user-provided Caixin screenshot. No API call.
        values = {
            MAIN: (MAIN_LABEL, '4,468,722,380.78'),
            BEFORE: (BEFORE_LABEL, '4,703,918,295.56'),
            'CONTROL_RISK_CAPITAL': ('控制风险最低资本', '71,487,301.09'),
            'ADDITIONAL_CAPITAL': ('附加资本', '0.00'),
            'MINIMUM_CAPITAL': ('最低资本', '4,540,209,681.87'),
        }
        taxonomy = pd.DataFrame([{'指标编码': code, '指标名称': label, '别名': label,
            '一级模块': '最低资本', '二级模块': '最低资本明细', '标准单位': '万元',
            '数据类型': '金额', '允许期间口径': '本季度末数'} for code, (label, _) in values.items()])
        config = {'table_id': 'MINIMUM_CAPITAL', 'table_name': 'S05-最低资本表'}
        match = PageMatch('MINIMUM_CAPITAL', config['table_name'], [37, 38], 99, '', table_config=config)
        requests = []
        def post(_url, *, headers, json: dict, timeout):
            prompt = json['messages'][0]['content'][0]['text']
            payload = __import__('json').loads(prompt.split('目标定义：\n')[1].split('\n页面顺序')[0])
            codes = [m['metric_id'] for m in payload['target_metrics']]
            requests.append(codes)
            if MAIN in codes or BEFORE in codes:
                self.assertIn('禁止把1*复制给行1', prompt)
                for metric in payload['target_metrics']:
                    if metric['metric_id'] in {MAIN, BEFORE}:
                        self.assertTrue(metric['source_matching_rule'])
            if len(requests) > 1:
                self.assertIn('上一轮未通过校验', prompt)
                self.assertIn('特征系数', prompt)
            metrics = []
            for code in codes:
                source = (BEFORE if code == MAIN else MAIN) if code == wrong_code and (len(requests) == 1 or not correct_retry) else code
                label, value = values[source]
                metrics.append(dict(metric_id=code, status='found', value_raw=value,
                    source_label=label, row_header_path=['1*' if source == BEFORE else '1', label],
                    unit='元', page=37, period_label='期末数', evidence_text=f'{label} {value}元', confidence=.99))
            return FakeResponse({'metrics': metrics})
        with patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=[
            (37, 'data:image/jpeg;base64,page37'), (38, 'data:image/jpeg;base64,page38')]):
            run = extract_metrics_vlm_v2(b'cached', [match], [config], taxonomy,
                api_key='test', base_url='https://example.test/v1', model='test',
                post_func=post, auto_retry=auto_retry, max_workers=1)
        return run, requests

    def test_wrong_main_is_retried_alone_and_both_amounts_retained(self):
        run, calls = self.run_case()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], [MAIN])
        rows = run.records.set_index('指标编码')
        self.assertAlmostEqual(rows.loc[MAIN, '标准数值'], 446872.238078)
        self.assertAlmostEqual(rows.loc[BEFORE, '标准数值'], 470391.829556)
        self.assertEqual(rows.loc[MAIN, '原始值'], '4,468,722,380.78')
        self.assertEqual(run.retry_calls, 1)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        tables = vlm_v2_to_extracted_tables(run)
        self.assertAlmostEqual(next(r for r in tables[0].metric_records if r['指标编码'] == MAIN)['标准数值'], 446872.238078)

    def test_reverse_mismatch_retries_only_before_factor(self):
        run, calls = self.run_case(wrong_code=BEFORE)
        self.assertEqual(calls[1], [BEFORE])
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_persistent_wrong_row_remains_blocked_not_made_undisclosed(self):
        run, calls = self.run_case(correct_retry=False)
        self.assertEqual(len(calls), 2)
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        check = run.validations[run.validations['规则'].eq('来源指标语义') & run.validations['指标编码'].eq(MAIN)].iloc[0]
        self.assertEqual(check['状态'], '失败')
        self.assertIn('行1*', check['说明'])
        row = run.records[run.records['指标编码'].eq(MAIN)].iloc[0]
        self.assertEqual(row['状态'], 'found')
        self.assertEqual(row['原始值'], '4,703,918,295.56')

    def test_same_values_are_allowed_when_sources_are_distinct(self):
        run, _ = self.run_case()
        row = run.records[run.records['指标编码'].eq(MAIN)].iloc[0]
        run.records.loc[run.records['指标编码'].eq(BEFORE), ['原始值', '数值', '标准数值']] = [row['原始值'], row['数值'], row['标准数值']]
        checks = validate_vlm_v2_metrics(run.records, [])
        self.assertTrue(checks.loc[checks['规则'].eq('来源指标语义'), '状态'].eq('通过').all())

    def test_legacy_upload_cannot_bypass_source_guard(self):
        run, _ = self.run_case(auto_retry=False)
        legacy = run.validations[~run.validations['规则'].eq('来源指标语义')].copy()
        legacy['状态'] = '通过'  # Simulates a previously accepted workbook.
        run = __import__('dataclasses').replace(run, validations=legacy)
        original = run.records.copy(deep=True)
        updated = refresh_vlm_v2_cross_table_checks(run)
        self.assertFalse(evaluate_vlm_v2_step3_gate(updated).passed)
        pd.testing.assert_frame_equal(original, updated.records)
        self.assertIs(refresh_vlm_v2_cross_table_checks(updated), updated)
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            run.records.to_excel(writer, sheet_name='VLM_v2指标结果', index=False)
            legacy.to_excel(writer, sheet_name='确定性校验', index=False)
        with self.assertRaises(ValueError):
            read_vlm_v2_extracted_tables(pd.ExcelFile(io.BytesIO(stream.getvalue())))

    def test_retry_plan_preserves_unrelated_failures(self):
        run, _ = self.run_case(auto_retry=False)
        validations = run.validations.copy()
        validations.loc[len(validations)] = ['OTHER:NUMBER', 'OTHER', 'OTHER_CODE', '失败', '数值可解析', '', '', '']
        plan = _retry_plan(validations, run.records)
        self.assertEqual(plan['MINIMUM_CAPITAL'], {MAIN})
        self.assertEqual(plan['OTHER'], {'OTHER_CODE'})

    def test_missing_old_checks_cannot_hide_semantic_mismatch(self):
        run, _ = self.run_case(auto_retry=False)
        run = __import__('dataclasses').replace(run, validations=pd.DataFrame())
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
