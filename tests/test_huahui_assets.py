import base64
import io
import re
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import fitz
import pandas as pd
import requests
from PIL import Image

from services.solvency_asset_columns import build_recognized_views, recognized_column_error
from services.solvency_locator_roles import page_role
from services.solvency_unit_context import recognized_unit_contexts
from services.solvency_vlm_v2_pipeline import (
    _render_page_images, _candidate_rank, extract_metrics_vlm_v2,
    evaluate_vlm_v2_step3_gate, validate_vlm_v2_metrics,
)
from tests import test_recognized_assets_timeouts as assets_fixture
from tests.test_vlm_v2_pipeline import FakeResponse


PDF = Path(r'F:\CROSS\V1\华汇人寿2026Q1偿付能力季度报告摘要.pdf')


class AssetColumnGuards(unittest.TestCase):
    def test_detail_titles_are_primary_and_other_tables_are_not(self):
        self.assertEqual(page_role('RECOGNIZED_ASSETS', '（二）认可资产明细表', ''), 'primary')
        self.assertEqual(page_role('RECOGNIZED_LIABILITIES', 'S04-认可负债表', ''), 'primary')
        self.assertEqual(page_role('RECOGNIZED_LIABILITIES', 'S05-最低资本表', '', 'primary'), 'reference')
        for title in ('非认可资产明细表', '认可负债明细表'):
            self.assertEqual(page_role('RECOGNIZED_ASSETS', title, '', 'primary'), 'reference')

    def test_wrong_columns_are_blocked_even_without_cross_table_comparison(self):
        for label in ('期末数 > 账面价值', '本季度末 > 非认可价值', '期初数 > 认可价值'):
            self.assertTrue(recognized_column_error(label))
        self.assertTrue(recognized_column_error([]))
        self.assertFalse(recognized_column_error(['期末数', '认可价值']))
        row = dict(目标表ID='RECOGNIZED_ASSETS', 指标编码='RECOGNIZED_ASSETS', 状态='found',
                   原始值='671620316.03', 标准数值=67162.031603, 标准单位='万元', 单位='元',
                   列表头路径='期末数 > 账面价值', 期间口径='期末数', 物理页码=20, 原始标签='合计', 证据原文='合计', 置信度=.99)
        checks = validate_vlm_v2_metrics(pd.DataFrame([row]), [])
        failed = checks[checks['状态'].eq('失败')]
        self.assertIn('来源指标语义', set(failed['规则']))
        bad = dict(status='found', confidence=1, column_header_path=['账面价值'])
        good = dict(status='found', confidence=.8, column_header_path=['认可价值'])
        self.assertGreater(_candidate_rank(good, 'RECOGNIZED_ASSETS', 'RECOGNIZED_ASSETS'),
                           _candidate_rank(bad, 'RECOGNIZED_ASSETS', 'RECOGNIZED_ASSETS'))

    def test_scanned_or_invalid_pdf_retains_full_image_path(self):
        self.assertEqual(build_recognized_views(b'not a pdf', [1], {}), {})


@unittest.skipUnless(PDF.exists(), 'Local Huahui PDF unavailable')
class HuahuiRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pdf = PDF.read_bytes()
        cls.cache = dict(_render_page_images(cls.pdf, [18,19,20,21,22], zoom=2, jpeg_quality=84))
        cls.views = build_recognized_views(cls.pdf, [18,19,20,21,22], cls.cache)

    def test_full_continuation_coverage_without_entering_nonrecognized_table(self):
        self.assertEqual(set(self.views), {18,19,20})
        units = recognized_unit_contexts(self.pdf, [18,19,20,21,22])
        self.assertEqual(set(units), {18,19,20})
        self.assertEqual(units[20], ('元',18,'单位：元'))
        for page, (url, note) in self.views.items():
            content = base64.b64decode(url.split(',')[1])
            self.assertLess(len(content), 750000)
            with Image.open(io.BytesIO(content)) as image:
                self.assertLessEqual(max(image.size), 1800)
            self.assertIn(f'物理页{page}', note)
            self.assertIn('期末数 > 认可价值', note)

    def test_unconfirmed_gaps_do_not_inherit_columns(self):
        views = build_recognized_views(self.pdf, [18,20], self.cache)
        self.assertEqual(set(views), {18})

    def test_changed_continuation_geometry_falls_back(self):
        with fitz.open(stream=self.pdf, filetype='pdf') as doc:
            doc[18].set_cropbox(fitz.Rect(10, 0, doc[18].rect.width, doc[18].rect.height))
            changed = doc.tobytes()
        self.assertNotIn(19, build_recognized_views(changed, [18,19,20], self.cache))

    def run_case(self, failure=None, permanent=False, retry_wrong_column=False):
        fixture = assets_fixture.RecognizedAssetsTimeoutTests()
        fixture.setUp()
        fixture.match = replace(fixture.match, pages=[18,19,20])
        values = {18: {'CASH_LIQUID_ASSETS':'123,311,642.95', 'INVESTMENT_ASSETS':'430,300,000.00'},
                  19: {'REINSURANCE_ASSETS':'152,763.30'}, 20: {'RECOGNIZED_ASSETS':'580,141,255.20'}}
        attempts = Counter()
        calls = []
        def post(url, *, headers, json, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            page = int(re.search(r'页面顺序及物理页码：\[(\d+)\]', prompt)[1])
            calls.append(page)
            attempts[page] += 1
            images = [part['image_url']['url'] for part in content if part['type']=='image_url']
            self.assertEqual(images, [self.views[page][0]])
            self.assertEqual(json['thinking'], {'type':'disabled'})
            if page == 19 and failure and (permanent or attempts[page] == 1):
                if failure == 'timeout':
                    raise requests.ReadTimeout('test timeout')
                if failure == 'http':
                    raise requests.HTTPError('test HTTP 503')
                return FakeResponse({'metrics': None})
            metrics = [dict(metric_id=code, status='found', value_raw=value, unit='元',
                            period_label='期末数', source_label='合计' if code=='RECOGNIZED_ASSETS' else code,
                            column_header_path=['期末数', '认可价值'], page=page,
                            evidence_text=f'{code} 期末认可价值 {value}元', confidence=.99)
                       for code, value in values[page].items()]
            if retry_wrong_column and page == 20 and attempts[page] == 1:
                metrics[0].update(value_raw='671,620,316.03', column_header_path=['期末数','账面价值'])
            return FakeResponse({'metrics':metrics})
        with patch('services.solvency_vlm_v2_pipeline._render_page_images', return_value=list(self.cache.items())) as render, \
             patch('services.solvency_vlm_v2_pipeline.build_recognized_views', wraps=build_recognized_views) as focus:
            result = extract_metrics_vlm_v2(self.pdf, [fixture.match], [fixture.config], fixture.taxonomy,
                api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6', post_func=post,
                auto_retry=True, request_max_attempts=2)
            self.assertEqual(render.call_count, 1)
            self.assertEqual(focus.call_count, 1)
        return result, calls

    def test_page_failure_isolation_and_targeted_recovery(self):
        for failure in ('timeout', 'http', 'invalid'):
            with self.subTest(failure=failure):
                run, calls = self.run_case(failure)
                self.assertEqual(calls[:3], [18,19,20])
                self.assertGreater(len(calls[3:]), 1)
                self.assertTrue(all(page == 19 for page in calls[3:]))
                self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
                rows = run.records.set_index('指标编码')
                self.assertAlmostEqual(rows.loc['RECOGNIZED_ASSETS','标准数值'], 58014.12552)
                self.assertAlmostEqual(rows.loc['REINSURANCE_ASSETS','标准数值'], 15.27633)
                self.assertFalse(run.records['指标编码'].duplicated().any())

    def test_permanent_failure_cannot_be_declared_genuine_non_disclosure(self):
        run, calls = self.run_case('invalid', permanent=True)
        self.assertEqual(calls[:3], [18,19,20])
        self.assertGreater(len(calls[3:]), 1)
        self.assertTrue(all(page == 19 for page in calls[3:]))
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertIn('请求覆盖完整性', set(run.validations['规则']))
        self.assertAlmostEqual(run.records.set_index('指标编码').loc['RECOGNIZED_ASSETS','标准数值'], 58014.12552)

    def test_wrong_column_triggers_targeted_validation_retry(self):
        run, calls = self.run_case(retry_wrong_column=True)
        self.assertEqual(calls, [18,19,20,18,19,20])
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertAlmostEqual(run.records.set_index('指标编码').loc['RECOGNIZED_ASSETS','标准数值'], 58014.12552)


if __name__ == '__main__':
    unittest.main()
