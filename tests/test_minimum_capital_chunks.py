"""Allianz Q1 current-column fixture, visually checked on physical pages 32–33.

Network responses are controlled: these regressions test orchestration, units,
zero rules and gate behavior, not a paid/live model accuracy benchmark.
"""
import json
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import requests

from services.solvency_filing_catalog import filing_entries, extend_filing_taxonomy
from services.solvency_normalizer import load_taxonomy
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import extract_metrics_vlm_v2, evaluate_vlm_v2_step3_gate
from tests.test_vlm_v2_pipeline import FakeResponse


class MinimumCapitalChunkTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = extend_filing_taxonomy(load_taxonomy(
            Path(__file__).parents[1] / 'config' / 'solvency_taxonomy.xlsx'))
        self.config = {'table_id': 'MINIMUM_CAPITAL', 'table_name': 'S05-最低资本表'}
        self.match = PageMatch('MINIMUM_CAPITAL', 'S05-最低资本表', [32, 33], 99, '', table_config=self.config)
        entries = [e for e in filing_entries() if e['table_id'] == 'MINIMUM_CAPITAL']
        raw_values = [
            '2122955098.71', '2234689577.59', '1385820219.95', '869633455.80',
            '940038689.25', '156539092.96', '580391018.06', '11230809.18',
            '11230809.18', '-', '-', '2229391207.97', '1801506231.30',
            '1383248339.64', '7903677.74', '-', '407783799.64', '50308542.07',
            '1421359382.42', '992886541.32', '324789765.62', '860571275.73',
            '192474500.03', '1197914062.91', '1186725137.92', '1186725137.92',
            '6934517554.98', '4029851.39', '-', '-', '-', '-', '-', '2126984950.10',
        ]
        self.fixture = {e['code']: (e['name'], raw, 32 if i < 27 else 33)
                        for i, (e, raw) in enumerate(zip(entries, raw_values))}

    def run_case(self, failure='', permanent=False, attempts_allowed=2, real_pdf=None,
                 oversized_retry=False):
        attempts, seen = Counter(), []
        def post(_url, *, headers, json: dict, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            payload = __import__('json').loads(prompt.split('目标定义：\n')[1].split('\n页面顺序')[0])
            codes = tuple(m['metric_id'] for m in payload['target_metrics'])
            self.assertLessEqual(len(codes), 12)
            self.assertEqual(payload['response_mode'], 'sparse_disclosures')
            self.assertIn('页面顺序及物理页码：[32, 33]', prompt)
            self.assertIn('不得把目标标准单位当作原表单位', prompt)
            self.assertEqual(len([c for c in content if c['type'] == 'image_url']), 2)
            self.assertEqual(json['thinking'], {'type': 'disabled'})
            self.assertEqual(timeout, 90)
            attempts[codes] += 1
            seen.append(codes)
            if oversized_retry and 'CREDIT_RISK_CAPITAL' in codes and len(codes) > 6:
                raise requests.ReadTimeout('test oversized minimum batch')
            if failure and 'CREDIT_RISK_CAPITAL' in codes and (permanent or attempts[codes] == 1):
                if failure == 'timeout':
                    raise requests.ReadTimeout('test minimum chunk timeout')
                if failure == 'network':
                    raise requests.ConnectionError('test connection failure')
                return FakeResponse({})
            metrics = []
            for code in codes:
                name, value, page = self.fixture[code]
                metrics.append(dict(metric_id=code, status='found', value_raw=value, unit='元',
                    period_label='本季度数', source_label=name, page=page,
                    evidence_text=f'{name} 本季度数 {value}元（单位见物理页32）', confidence=.99))
            return FakeResponse({'metrics': metrics})

        kwargs = dict(api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6',
                      post_func=post, request_max_attempts=attempts_allowed, auto_retry=True)
        if real_pdf:
            run = extract_metrics_vlm_v2(real_pdf.read_bytes(), [self.match], [self.config], self.taxonomy, **kwargs)
        else:
            with patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=[
                (p, f'data:image/jpeg;base64,page{p}') for p in [32, 33]
            ]) as render:
                run = extract_metrics_vlm_v2(b'cached', [self.match], [self.config], self.taxonomy, **kwargs)
                self.assertEqual(render.call_count, 1)
        return run, attempts, seen

    def test_all_34_metrics_units_dash_zero_and_capital_equation(self):
        run, attempts, seen = self.run_case()
        self.assertEqual(len(seen), 3)
        self.assertEqual(len(run.records), 34)
        rows = run.records.set_index('指标编码')
        for code, (_, value, page) in self.fixture.items():
            self.assertAlmostEqual(rows.loc[code, '标准数值'], 0 if value == '-' else float(value) / 10000)
            self.assertEqual(rows.loc[code, '物理页码'], page)
        self.assertEqual(rows.loc['ADDITIONAL_CAPITAL', '状态'], 'disclosed_zero')
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_only_failed_chunk_is_retried(self):
        for failure in ['timeout', 'network', 'malformed']:
            with self.subTest(failure=failure):
                run, attempts, seen = self.run_case(failure)
                self.assertEqual(sorted(attempts.values()), [1, 1, 2])
                self.assertEqual(run.model_calls, 4)
                self.assertEqual(run.retry_calls, 1)
                self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
                self.assertEqual(len(run.records), 34)

    def test_permanent_failure_keeps_success_and_blocks_false_non_disclosure(self):
        run, attempts, seen = self.run_case('timeout', permanent=True)
        self.assertEqual(sorted(attempts.values()), [1, 1, 1, 1, 2])
        self.assertEqual(run.model_calls, 6)
        self.assertEqual(run.retry_calls, 3)
        successful = {code for codes in seen if 'CREDIT_RISK_CAPITAL' not in codes for code in codes}
        self.assertEqual(set(run.records['指标编码']), successful)
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        coverage = run.validations[run.validations['规则'].eq('请求覆盖完整性')].iloc[0]
        self.assertIn('test minimum chunk timeout', coverage['说明'])

    def test_repeated_oversized_failure_is_recovered_with_smaller_batches(self):
        run, attempts, seen = self.run_case(oversized_retry=True)
        self.assertEqual(sorted(attempts.values()), [1, 1, 1, 1, 2])
        self.assertEqual(run.model_calls, 6)
        self.assertEqual(run.retry_calls, 3)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertEqual(len(run.records), 34)
        self.assertTrue(any(len(codes) == 6 for codes in seen))

    def test_retry_limit_one_does_not_repeat_failed_chunk(self):
        run, attempts, seen = self.run_case('timeout', attempts_allowed=1)
        self.assertEqual(len(seen), 3)
        self.assertTrue(all(count == 1 for count in attempts.values()))
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)

    @unittest.skipUnless(Path(r'F:\CROSS\V1\安联人寿2026Q1偿付能力季度报告摘要.pdf').exists(), 'local PDF unavailable')
    def test_real_allianz_pdf_rendering_with_controlled_response(self):
        run, _, _ = self.run_case(real_pdf=Path(r'F:\CROSS\V1\安联人寿2026Q1偿付能力季度报告摘要.pdf'))
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)


if __name__ == '__main__':
    unittest.main()
