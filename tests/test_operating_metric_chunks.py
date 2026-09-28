import json as jsonlib
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import requests

import services.solvency_vlm_v2_pipeline as vlm_pipeline
from services.solvency_filing_catalog import extend_filing_taxonomy
from services.solvency_normalizer import load_taxonomy
from services.solvency_operating_recovery import recover_surrender_rate
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import extract_metrics_vlm_v2, evaluate_vlm_v2_step3_gate
from tests.test_vlm_v2_pipeline import FakeResponse


class OperatingMetricChunkTests(unittest.TestCase):
    JIANXIN_Q2 = Path(r'F:\CROSS\V1\建信人寿2025Q2偿付能力季度报告摘要.pdf')

    def run_case(self, failure='', permanent=False, failed_code='INSURANCE_REVENUE'):
        taxonomy = extend_filing_taxonomy(load_taxonomy(Path(__file__).parents[1] / 'config' / 'solvency_taxonomy.xlsx'))
        config = {'table_id': 'OPERATING_METRICS', 'table_name': '主要经营指标'}
        match = PageMatch('OPERATING_METRICS', '主要经营指标', [10, 11], 99, '', table_config=config)
        attempts = Counter()
        def post(_url, *, headers, json, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            payload = jsonlib.loads(prompt.split('目标定义：\n')[1].split('\n页面顺序')[0])
            metrics = payload['target_metrics']
            codes = tuple(m['metric_id'] for m in metrics)
            self.assertLessEqual(len(codes), 8)
            self.assertEqual(payload['response_mode'], 'sparse_disclosures')
            self.assertIn('本季度数/本季度（末）数', prompt)
            self.assertIn('本年度累计数/本年累计数', prompt)
            self.assertEqual(json['thinking'], {'type': 'disabled'})
            self.assertEqual(timeout, 90)
            if 'TOP_FIVE_PRODUCTS' in codes:
                self.assertEqual(codes, ('TOP_FIVE_PRODUCTS',))
            attempts[codes] += 1
            if failure and failed_code in codes and (permanent or attempts[codes] == 1):
                if failure == 'timeout':
                    raise requests.ReadTimeout('operating timeout fixture')
                if failure == 'connection':
                    raise requests.ConnectionError('connection fixture')
                return FakeResponse({'unexpected': []})
            return FakeResponse({'metrics': [dict(metric_id=m['metric_id'], status='found',
                value_raw='产品甲：100万元' if m['metric_id'] == 'TOP_FIVE_PRODUCTS' else '0',
                unit=m['expected_unit'], period_label='本季度数', source_label=m['name'],
                page=10, evidence_text=m['name'] + ' 本季度数 0', confidence=.99) for m in metrics]})
        with patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=[
            (10, 'data:image/jpeg;base64,page10'), (11, 'data:image/jpeg;base64,page11')
        ]) as render:
            run = extract_metrics_vlm_v2(b'fixture', [match], [config], taxonomy,
                api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6', post_func=post)
            self.assertEqual(render.call_count, 1)
        return run, attempts

    def test_small_numeric_batches_and_independent_text_batch(self):
        run, attempts = self.run_case()
        self.assertEqual(len(run.records), 29)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(run.model_calls, 5)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertEqual(run.records.set_index('指标编码').loc['TOP_FIVE_PRODUCTS', '原始值'], '产品甲：100万元')

    def test_transient_errors_retry_only_failed_group(self):
        for failure in ['timeout', 'connection', 'invalid']:
            with self.subTest(failure=failure):
                run, attempts = self.run_case(failure)
                self.assertEqual(sorted(attempts.values()), [1, 1, 1, 1, 2])
                self.assertEqual(len(run.records), 29)
                self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_permanent_failure_keeps_other_results_and_blocks_gate(self):
        run, attempts = self.run_case('timeout', permanent=True)
        self.assertEqual(sorted(attempts.values()), [1, 1, 1, 1, 2])
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertTrue(run.validations['校验ID'].eq('OPERATING_METRICS:REQUEST_COVERAGE').any())
        self.assertIn('TOP_FIVE_PRODUCTS', set(run.records['指标编码']))

    def test_text_timeout_does_not_discard_revenue(self):
        run, attempts = self.run_case('timeout', permanent=True, failed_code='TOP_FIVE_PRODUCTS')
        revenue = run.records.set_index('指标编码').loc['INSURANCE_REVENUE']
        self.assertEqual(revenue['标准数值'], 0)
        self.assertEqual(revenue['状态'], 'disclosed_zero')
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)

    def test_previous_surrender_rate_is_rejected_and_current_value_is_recovered(self):
        taxonomy = extend_filing_taxonomy(
            load_taxonomy(Path(__file__).parents[1] / 'config' / 'solvency_taxonomy.xlsx')
        )
        config = {'table_id': 'OPERATING_METRICS', 'table_name': '主要经营指标'}
        card = vlm_pipeline.build_vlm_v2_target_cards([config], taxonomy)[0]
        match = PageMatch('OPERATING_METRICS', '主要经营指标', [14], 99, '', table_config=config)
        previous = {
            'metric_id': 'SURRENDER_RATE', 'status': 'found', 'value_raw': '0.54%',
            'unit': '%', 'period_label': '上季度数', 'source_label': '综合退保率',
            'column_header_path': ['上季度数'], 'page': 14,
            'evidence_text': '综合退保率｜上季度数 0.54%', 'confidence': .99,
        }
        current = {
            **previous, 'value_raw': '0.90%', 'period_label': '本季度数',
            'column_header_path': ['本季度数'],
            'evidence_text': '综合退保率｜本季度数 0.90%',
            'recovery_mode': '源PDF表格本期列补提',
        }
        with patch.object(vlm_pipeline, '_call_vlm_json', return_value={'metrics': [previous]}), patch.object(
            vlm_pipeline, 'recover_surrender_rate', return_value=current,
        ):
            rows, _ = vlm_pipeline._extract_card_records(
                b'fixture', match, card,
                api_key='test', base_url='https://example.test/v1', model='vision-test',
                timeout=90, request_max_attempts=1, post_func=None,
                metric_codes={'SURRENDER_RATE'},
                page_image_cache={14: 'data:image/jpeg;base64,page14'},
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['期间口径'], '本季度数')
        self.assertEqual(rows[0]['原始值'], '0.90%')
        self.assertAlmostEqual(rows[0]['标准数值'], 0.9)

    @unittest.skipUnless(JIANXIN_Q2.exists(), 'Local Jianxin 2025Q2 PDF unavailable')
    def test_jianxin_q2_source_grid_recovers_current_surrender_rate(self):
        recovered = recover_surrender_rate(self.JIANXIN_Q2.read_bytes(), [14])
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered['metric_id'], 'SURRENDER_RATE')
        self.assertEqual(recovered['period_label'], '本季度数')
        self.assertEqual(recovered['value_raw'], '0.90%')
        self.assertEqual(recovered['page'], 14)
