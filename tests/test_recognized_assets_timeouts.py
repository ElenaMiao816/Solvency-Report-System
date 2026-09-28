import re
import unittest
from collections import Counter
from json import loads
from pathlib import Path
from unittest.mock import patch

import requests

from services.solvency_filing_catalog import extend_filing_taxonomy
from services.solvency_normalizer import load_taxonomy
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import extract_metrics_vlm_v2, evaluate_vlm_v2_step3_gate
from tests.test_vlm_v2_pipeline import FakeResponse


class RecognizedAssetsTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = extend_filing_taxonomy(load_taxonomy(
            Path(__file__).parents[1] / 'config' / 'solvency_taxonomy.xlsx'))
        self.config = {'table_id': 'RECOGNIZED_ASSETS', 'table_name': 'S03-认可资产表'}
        self.match = PageMatch('RECOGNIZED_ASSETS', 'S03-认可资产表', [31, 32, 33], 99, '', table_config=self.config)
        # Independently read from the supplied Aixin PDF's current recognised-value column.
        self.values = {
            31: {'CASH_LIQUID_ASSETS': 217106.50, 'INVESTMENT_ASSETS': 3056642.59},
            32: {'SUBSIDIARY_JV_ASSOCIATE_EQUITY': 7804.60, 'REINSURANCE_ASSETS': 2325709.53,
                 'RECEIVABLES_AND_PREPAYMENTS': 105748.75, 'FIXED_ASSETS': 197.03},
            33: {'OTHER_RECOGNIZED_ASSETS': 2639.31, 'RECOGNIZED_ASSETS': 5715848.30},
        }

    def run_case(self, fail_page=None, permanent=False, pdf=None, values=None):
        attempts = Counter()
        calls = []
        values = values or self.values

        def post(_url, *, headers, json, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            pages = re.search(r'页面顺序及物理页码：\[([^]]+)\]', prompt).group(1)
            self.assertNotIn(',', pages)
            page = int(pages)
            focused = '认可价值专列视图' in prompt
            self.assertEqual(sum(item['type'] == 'image_url' for item in content), 1 if page == 31 or focused else 2)
            if page != 31 and not focused:
                self.assertIn('物理页31的表头/单位上下文', prompt)
            self.assertIn('"response_mode": "sparse_disclosures"', prompt)
            self.assertIn('不得读取账面价值', prompt)
            self.assertEqual(json['thinking'], {'type': 'disabled'})
            self.assertEqual(timeout, 90)
            attempts[page] += 1
            calls.append(page)
            if page == fail_page and (permanent or attempts[page] == 1):
                raise requests.ReadTimeout(f'page {page} timeout')
            return FakeResponse({'metrics': [dict(
                metric_id=code, status='found', value_raw=str(value), unit='万元',
                period_label='期末数', source_label=code, page=page,
                row_header_path=[code], column_header_path=['期末数', '认可价值'],
                evidence_text=f'{code} 期末认可价值 {value}万元', confidence=.99,
            ) for code, value in values.get(page, {}).items()]})

        kwargs = dict(api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6',
                      post_func=post, auto_retry=True, request_max_attempts=2, max_workers=2)
        if pdf is not None:
            result = extract_metrics_vlm_v2(pdf, [self.match], [self.config], self.taxonomy, **kwargs)
        else:
            with patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=[
                (page, f'data:image/jpeg;base64,page{page}') for page in self.match.pages
            ]) as render:
                result = extract_metrics_vlm_v2(b'cached-pdf', [self.match], [self.config], self.taxonomy, **kwargs)
                self.assertEqual(render.call_count, 1)
        return result, attempts, calls

    def test_timeout_retries_only_failed_page_and_preserves_other_pages(self):
        run, attempts, calls = self.run_case(fail_page=32)
        self.assertEqual(calls[:3], [31, 32, 33])
        self.assertGreater(len(calls[3:]), 1)
        self.assertTrue(all(page == 32 for page in calls[3:]))
        self.assertEqual(attempts, {31: 1, 32: 1 + len(calls[3:]), 33: 1})
        self.assertEqual(run.model_calls, len(calls))
        self.assertEqual(run.retry_calls, len(calls) - 3)
        self.assertEqual(len(run.records), 49)
        self.assertFalse(run.records['指标编码'].duplicated().any())
        rows = run.records.set_index('指标编码')
        self.assertEqual(rows.loc['REINSURANCE_ASSETS', '标准数值'], 2325709.53)
        self.assertEqual(rows.loc['RECOGNIZED_ASSETS', '标准数值'], 5715848.30)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_permanent_page_timeout_keeps_results_and_blocks_gate(self):
        run, attempts, calls = self.run_case(fail_page=32, permanent=True)
        self.assertEqual(calls[:3], [31, 32, 33])
        self.assertGreater(len(calls[3:]), 1)
        self.assertTrue(all(page == 32 for page in calls[3:]))
        self.assertEqual(run.records.set_index('指标编码').loc['RECOGNIZED_ASSETS', '标准数值'], 5715848.30)
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertTrue(run.validations['规则'].eq('请求覆盖完整性').any())

    def test_optional_page_timeout_is_not_mislabeled_as_non_disclosure(self):
        values = {31: {**self.values[31], **self.values[32], **self.values[33]}}
        run, attempts, calls = self.run_case(fail_page=33, permanent=True, values=values)
        required = run.validations[run.validations['规则'].eq('目标覆盖')]
        self.assertTrue(required['状态'].eq('通过').all())
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertEqual(attempts[31], 1)
        self.assertEqual(attempts[32], 1)
        self.assertEqual(attempts[33], 1 + len(calls[3:]))

    @unittest.skipUnless(Path(r'F:\CROSS\V1\爱心人寿2026Q1偿付能力季度报告摘要.pdf').exists(), 'local PDF fixture unavailable')
    def test_real_aixin_pdf_rendering_with_stubbed_model(self):
        # Real page rendering and full pipeline; no paid/network model invocation.
        run, attempts, calls = self.run_case(pdf=Path(
            r'F:\CROSS\V1\爱心人寿2026Q1偿付能力季度报告摘要.pdf').read_bytes())
        self.assertEqual(calls, [31, 32, 33])
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    @unittest.skipUnless(Path(r'F:\CROSS\V2\中银三星2026Q1偿付能力季度报告摘要.pdf').exists(), 'local PDF fixture unavailable')
    def test_real_zhongyin_samsung_failed_dense_page_uses_bounded_recovery(self):
        pdf = Path(r'F:\CROSS\V2\中银三星2026Q1偿付能力季度报告摘要.pdf').read_bytes()
        match = PageMatch(
            'RECOGNIZED_ASSETS', 'S03-认可资产表', [20, 21, 22], 99, '',
            table_config=self.config,
        )
        calls = []
        recovery_batches = []
        focused_pages = set()
        values = {
            20: {'CASH_LIQUID_ASSETS': ('现金及流动性管理工具', '3,122,643,681.88')},
            21: {
                'INVESTMENT_ASSETS': ('投资资产', '135,662,995,545.62'),
                'REINSURANCE_ASSETS': ('再保险资产', '15,931,639,035.04'),
            },
            22: {'RECOGNIZED_ASSETS': ('资产合计', '157,798,333,808.05')},
        }

        def post(_url, *, headers, json, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            page = int(re.search(r'页面顺序及物理页码：\[([^]]+)\]', prompt).group(1))
            payload = loads(re.search(r'目标定义：\s*(.*?)\n页面顺序', prompt, re.S).group(1))
            codes = {item['metric_id'] for item in payload['target_metrics']}
            calls.append((page, frozenset(codes)))
            if '认可价值专列视图' in prompt:
                focused_pages.add(page)
            if page == 21 and len(codes) > 10:
                raise requests.ReadTimeout('dense recognized-assets page timeout')
            if page == 21:
                recovery_batches.append(frozenset(codes))
                self.assertLessEqual(len(codes), 10)
                self.assertIn('认可价值专列视图', prompt)
                self.assertIn('按指标小批提取', prompt)
            metrics = []
            for code, (label, value) in values.get(page, {}).items():
                if code not in codes:
                    continue
                # Reproduce the original shared-page failure when S03 is not
                # isolated from the S04 table that begins lower on page 22.
                if page == 22 and code == 'RECOGNIZED_ASSETS' and page not in focused_pages:
                    value = '176,074,903,082.82'
                metrics.append(dict(
                    metric_id=code, status='found', value_raw=value, unit='元',
                    period_label='期末数', source_label=label, page=page,
                    row_header_path=[label], column_header_path=['期末数', '认可价值'],
                    evidence_text=f'{label} 期末认可价值 {value} 元', confidence=.99,
                ))
            return FakeResponse({
                'metrics': metrics,
                'table_unit': '元',
                'unit_page': 20,
                'unit_evidence': '单位：元',
            })

        run = extract_metrics_vlm_v2(
            pdf, [match], [self.config], self.taxonomy,
            api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6',
            post_func=post, auto_retry=True, request_max_attempts=2, max_workers=1,
        )
        self.assertGreater(len(recovery_batches), 1)
        self.assertEqual(recovery_batches[0], frozenset({
            'CASH_LIQUID_ASSETS', 'INVESTMENT_ASSETS',
            'REINSURANCE_ASSETS', 'RECOGNIZED_ASSETS',
        }))
        rows = run.records.set_index('指标编码')
        self.assertAlmostEqual(rows.loc['INVESTMENT_ASSETS', '标准数值'], 13566299.554562)
        self.assertAlmostEqual(rows.loc['REINSURANCE_ASSETS', '标准数值'], 1593163.903504)
        self.assertAlmostEqual(rows.loc['RECOGNIZED_ASSETS', '标准数值'], 15779833.380805)
        self.assertIn(22, focused_pages)
        self.assertFalse(run.validations['规则'].eq('请求覆盖完整性').any())
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)


if __name__ == '__main__':
    unittest.main()
