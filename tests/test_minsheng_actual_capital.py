from json import loads
import re
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import requests

from services.solvency_filing_catalog import extend_filing_taxonomy
from services.solvency_normalizer import load_taxonomy
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import (
    _render_page_images, extract_metrics_vlm_v2, evaluate_vlm_v2_step3_gate,
)
from tests.test_vlm_v2_pipeline import FakeResponse


PDF = Path(r'F:\CROSS\V1\民生人寿2026Q1偿付能力季度报告摘要.pdf')
CORE = {'ACTUAL_CAPITAL','CORE_T1_CAPITAL','CORE_T2_CAPITAL','ANC_T1_CAPITAL','ANC_T2_CAPITAL'}
# Independently checked against physical page 29, current-period column, 万元.
VALUES = {
    'ACTUAL_CAPITAL': ('实际资本合计','1,857,087.05'),
    'CORE_T1_CAPITAL': ('核心一级资本','1,440,973.21'),
    'CORE_T2_CAPITAL': ('核心二级资本','0.00'),
    'ANC_T1_CAPITAL': ('附属一级资本','416,113.85'),
    'ANC_T2_CAPITAL': ('附属二级资本','0.00'),
    'FINANCIAL_STATEMENT_NET_ASSETS': ('净资产','1,423,854.85'),
    'NET_ASSET_ADJUSTMENT': ('对净资产的调整额','17,118.35'),
    'NON_RECOGNIZED_ASSET_BOOK_VALUE': ('各项非认可资产的账面价值','-18,021.86'),
    'LONG_TERM_EQUITY_VALUATION_DIFFERENCE': ('长期股权投资的认可价值与账面价值的差额','-321,790.30'),
    'CORE_T1_INVESTMENT_PROPERTY_FAIR_VALUE_ADJUSTMENT': ('投资性房地产公允价值增值','0.00'),
    'DEFERRED_TAX_ASSET_ADJUSTMENT': ('递延所得税资产','-46,048.90'),
    'AGRICULTURAL_CATASTROPHE_RISK_RESERVE': ('对农业保险提取的大灾风险准备金','0.00'),
    'POLICY_SURPLUS_CORE_T1': ('计入核心一级资本的保单未来盈余','402,979.42'),
    'QUALIFYING_CORE_T1_LIABILITY_CAPITAL': ('符合核心一级资本标准的负债类资本工具','0.00'),
    'OTHER_CORE_T1_ADJUSTMENT': ('银保监会规定的其他调整项目','0.00'),
}


class MinshengActualCapitalTests(unittest.TestCase):
    def setUp(self):
        self.taxonomy = extend_filing_taxonomy(load_taxonomy(Path(__file__).parents[1] / 'config/solvency_taxonomy.xlsx'))
        self.config = dict(table_id='ACTUAL_CAPITAL', table_name='S02-实际资本表')
        self.match = PageMatch('ACTUAL_CAPITAL', 'S02-实际资本表', [29], 99, '', table_config=self.config)

    def run_case(self, failure=None, core_failure=True, permanent=False, native=False):
        attempts = Counter()
        calls = []
        image_ids = []
        data = PDF.read_bytes() if native else b'test-pdf'
        cache = list(_render_page_images(data, [29], zoom=2, jpeg_quality=84)) if native else [(29,'data:image/jpeg;base64,cached')]
        def post(url, *, headers, json: dict, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            payload = loads(re.search(r'目标定义：\s*(.*?)\n页面顺序', prompt, re.S)[1])
            codes = frozenset(item['metric_id'] for item in payload['target_metrics'])
            self.assertLessEqual(len(codes), 10)
            if not calls:
                self.assertEqual(codes, CORE)
            calls.append(codes)
            attempts[codes] += 1
            images = [part['image_url']['url'] for part in content if part['type']=='image_url']
            self.assertEqual(images, [cache[0][1]])
            image_ids.append(images[0])
            self.assertEqual(timeout, 90)
            self.assertEqual(json['thinking'], {'type':'disabled'})
            self.assertIn('不能因金额为零省略指标', prompt)
            chosen = codes == CORE if core_failure else 'FINANCIAL_STATEMENT_NET_ASSETS' in codes
            if failure and chosen and (permanent or attempts[codes] == 1):
                if failure == 'timeout':
                    raise requests.ReadTimeout('test S02 timeout')
                if failure == 'http':
                    raise requests.HTTPError('test HTTP 503')
                return FakeResponse({'metrics':None})
            return FakeResponse({'metrics': [dict(metric_id=code, status='found', value_raw=VALUES[code][1],
                unit='万元', period_label='期末数', source_label=VALUES[code][0],
                column_header_path=['期末数'], page=29, evidence_text=' '.join(VALUES[code])+' 万元', confidence=.99)
                for code in sorted(codes) if code in VALUES]})
        with (
            patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=cache) as render,
            patch('services.solvency_vlm_v2_pipeline.visible_policy_surplus_codes', return_value=None),
        ):
            result = extract_metrics_vlm_v2(data, [self.match], [self.config], self.taxonomy,
                api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6',
                post_func=post, auto_retry=True, request_max_attempts=2, max_workers=2)
            self.assertEqual(render.call_count, 1)
        self.assertEqual(len(set(image_ids)),1)
        return result, calls, attempts

    def test_normal_bounded_chunks_keep_zeros_and_absent_details_distinct(self):
        run, calls, attempts = self.run_case()
        self.assertEqual([len(codes) for codes in calls], [5,10,10,5])
        self.assertEqual(len(set.union(*(set(codes) for codes in calls))), 30)
        self.assertEqual(run.model_calls,4)
        self.assertEqual(run.retry_calls,0)
        rows = run.records.set_index('指标编码')
        self.assertAlmostEqual(rows.loc['ACTUAL_CAPITAL','标准数值'],1857087.05)
        for code in ('CORE_T2_CAPITAL','ANC_T2_CAPITAL'):
            self.assertEqual(rows.loc[code,'状态'],'disclosed_zero')
            self.assertEqual(rows.loc[code,'标准数值'],0)
        self.assertEqual(rows.loc['ANC_T1_SUBORDINATED_TERM_DEBT','状态'],'not_disclosed')
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_failed_core_batch_retried_without_repeating_successful_details(self):
        for failure in ('timeout','http','invalid'):
            with self.subTest(failure=failure):
                run, calls, attempts = self.run_case(failure)
                self.assertEqual(len(calls),5)
                self.assertEqual(calls[-1],CORE)
                self.assertEqual(attempts[frozenset(CORE)],2)
                self.assertEqual(run.retry_calls,1)
                self.assertEqual(run.model_calls,5)
                self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
                self.assertFalse(run.records['指标编码'].duplicated().any())

    def test_failed_detail_batch_does_not_erase_core_values(self):
        run, calls, attempts = self.run_case('timeout', core_failure=False)
        self.assertEqual(len(calls),5)
        self.assertEqual(calls[-1],calls[1])
        self.assertEqual(attempts[frozenset(CORE)],1)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)

    def test_permanent_detail_failure_keeps_core_but_blocks_coverage(self):
        run, calls, attempts = self.run_case('invalid', core_failure=False, permanent=True)
        self.assertEqual(len(calls),5)
        rows = run.records.set_index('指标编码')
        self.assertAlmostEqual(rows.loc['ACTUAL_CAPITAL','标准数值'],1857087.05)
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertIn('请求覆盖完整性',set(run.validations['规则']))

    def test_permanent_core_timeout_still_keeps_successful_details(self):
        run, calls, attempts = self.run_case('timeout', permanent=True)
        self.assertEqual(len(calls),5)
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertAlmostEqual(run.records.set_index('指标编码').loc['FINANCIAL_STATEMENT_NET_ASSETS','标准数值'],1423854.85)
        self.assertTrue(any('超时' in log for log in run.logs))

    @unittest.skipUnless(PDF.exists(),'Local Minsheng PDF unavailable')
    def test_original_pdf_rendering_with_stubbed_model(self):
        run, calls, attempts = self.run_case('timeout', native=True)
        self.assertEqual(len(calls),5)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)


if __name__ == '__main__':
    unittest.main()
