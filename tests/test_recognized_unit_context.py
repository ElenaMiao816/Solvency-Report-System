import re
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
import pandas as pd

from services.solvency_pdf_locator import PageMatch
from services.solvency_unit_context import (
    recognized_unit_contexts, model_unit_context, apply_unit_context,
)
from services.solvency_vlm_v2_pipeline import (
    VLMV2TargetCard, _extract_card_records, _normalize_number_and_unit,
    validate_vlm_v2_metrics, METRIC_COLUMNS,
)
from tests.test_vlm_v2_pipeline import FakeResponse


PDF = Path(r'F:\CROSS\V1\弘康人寿2026Q1偿付能力季度报告摘要.pdf')
BOC_PDF = Path(r'F:\CROSS\V2\中银三星2025Q3偿付能力季度报告摘要.pdf')


def unit_pdf(texts):
    with fitz.open() as doc:
        for text in texts:
            page = doc.new_page()
            page.insert_text((40, 80), text, fontname='china-s', fontsize=16)
        return doc.tobytes()


class UnitContextTests(unittest.TestCase):
    def test_continuation_inherits_only_same_adjacent_table(self):
        data = unit_pdf(['认可资产表\n单位：元\n现金及流动性管理工具', '再保险资产', '认可负债表\n单位：万元'])
        context = recognized_unit_contexts(data, [1, 2, 3])
        self.assertEqual(context[2], ('元', 1, '单位：元'))
        self.assertNotIn(3, context)

    def test_unit_change_and_conflicting_header(self):
        data = unit_pdf(['认可资产表\n单位：元', '单位：万元\n其他认可资产', '单位：元\n单位：万元'])
        context = recognized_unit_contexts(data, [1, 2, 3])
        self.assertEqual(context[2], ('万元', 2, '单位：万元'))
        self.assertEqual(context[3][0], '')
        unknown = unit_pdf(['认可资产表\n单位：元', '单位：美元\n其他认可资产'])
        self.assertEqual(recognized_unit_contexts(unknown, [1,2])[2][0], '')

    def test_shared_page_uses_unit_after_its_own_title(self):
        data = unit_pdf(['实际资本表\n单位：万元\n认可资产表\n单位：元'])
        self.assertEqual(recognized_unit_contexts(data, [1])[1][0], '元')

    def test_table_title_parenthesized_unit_is_inherited_and_conflicts_blocked(self):
        data = unit_pdf(['S03-认可资产表（元）\n现金及流动性管理工具', '信托计划\n认可资产合计'])
        context = recognized_unit_contexts(data, [1, 2])
        self.assertEqual(context[1][0], '元')
        self.assertEqual(context[2][0], '元')
        self.assertIn('认可资产表（元）', context[2][2])
        conflicting = unit_pdf(['S03-认可资产表（元）\n单位：万元'])
        self.assertEqual(recognized_unit_contexts(conflicting, [1])[1][0], '')

    def test_continuation_total_before_next_table_keeps_unit_on_that_page_only(self):
        data = unit_pdf([
            'S03-认可资产表（元）\n现金及流动性管理工具',
            '信托计划 8,854,107,910.35\n资产合计 159,833,624,742.10\nS04-认可负债表（元）',
            '认可负债表\n单位：万元',
        ])
        context = recognized_unit_contexts(data, [1, 2, 3])
        self.assertEqual(context[2][0], '元')
        self.assertNotIn(3, context)

    def test_unknown_currency_scale_cannot_silently_become_wanyuan(self):
        for unit in ['', '未知', '美元', '万元左右']:
            self.assertEqual(_normalize_number_and_unit('95047859471.59', unit, '万元'), (None, '万元'))
        self.assertEqual(_normalize_number_and_unit('10000', '人民币元', '万元'), (1, '万元'))
        self.assertEqual(_normalize_number_and_unit('95万元', '元', '万元'), (95, '万元'))

    def test_explicit_zero_is_not_changed_to_non_disclosure(self):
        self.assertEqual(_normalize_number_and_unit('-', '', '万元'), (0, '万元'))
        self.assertEqual(_normalize_number_and_unit('0', '', '万元'), (0, '万元'))

    def test_scanned_header_requires_unit_page_and_matching_quote(self):
        response = dict(table_unit='元', unit_page=22, unit_evidence='单位：元')
        self.assertEqual(model_unit_context(response, [24,22]), ('元',22,'单位：元'))
        self.assertIsNone(model_unit_context(response, [24]))
        response['table_unit'] = '万元'
        self.assertIsNone(model_unit_context(response, [24,22]))

    def test_model_table_title_quote_is_valid_but_conflicting_units_are_not(self):
        response = dict(table_unit='元', unit_page=20, unit_evidence='S03-认可资产表（元）')
        self.assertEqual(model_unit_context(response, [20, 21]), ('元', 20, 'S03-认可资产表（元）'))
        self.assertIsNone(model_unit_context(response, [21]))
        response['unit_evidence'] += '；单位：万元'
        self.assertIsNone(model_unit_context(response, [20, 21]))

    def test_unverified_model_unit_is_cleared_and_literal_evidence_survives(self):
        item = dict(unit='万元', value_raw='159,833,624,742.10', evidence_text='认可资产合计 159,833,624,742.10')
        result = apply_unit_context(item, None, require_explicit=True)
        self.assertEqual(result['unit'], '')
        self.assertIn('未采用模型推测单位', result['evidence_text'])
        result = apply_unit_context(dict(item, evidence_text='认可资产合计 159,833,624,742.10元'), None, require_explicit=True)
        self.assertEqual(result['unit'], '元')

    @unittest.skipUnless(BOC_PDF.exists(), 'Local source PDF unavailable')
    def test_boc_samsung_title_unit_reaches_mixed_table_continuation(self):
        # Page 21 closes S03 and starts S04; its S03 rows still inherit 元.
        context = recognized_unit_contexts(BOC_PDF.read_bytes(), [20, 21, 22])
        self.assertEqual(context[20][0], '元')
        self.assertEqual(context[21][0], '元')
        self.assertNotIn(22, context)

    def test_source_value_and_row_specific_unit_take_priority(self):
        context = ('元',22,'单位：元')
        item = apply_unit_context(dict(unit='元', value_raw='95万元'), context)
        self.assertEqual(item['unit'], '万元')
        item = apply_unit_context(dict(unit='元', value_raw='95', source_label='其他认可资产（万元）'), context)
        self.assertEqual(item['unit'], '万元')

    def test_audit_keeps_original_model_unit_without_changing_raw_value(self):
        item = apply_unit_context(dict(unit='万元', value_raw='95,047,859,471.59', evidence_text='认可资产合计'), ('元',22,'单位：元'))
        self.assertEqual(item['value_raw'], '95,047,859,471.59')
        self.assertEqual(item['unit'], '元')
        self.assertIn('模型原单位“万元”', item['evidence_text'])


class RecognizedUnitPipelineTests(unittest.TestCase):
    @unittest.skipUnless(BOC_PDF.exists(), 'Local source PDF unavailable')
    def test_boc_samsung_header_corrects_every_asset_row(self):
        metrics = tuple(dict(metric_id=code, name=name, expected_unit='万元', data_type='金额', semantic_key=code)
                        for code, name in [('RECOGNIZED_ASSETS', '认可资产'), ('OTHER_RECOGNIZED_ASSETS', '其他认可资产')])
        card = VLMV2TargetCard('RECOGNIZED_ASSETS', '认可资产表', '', metrics)
        match = PageMatch(card.table_id, card.table_name, [21], 99, '')
        cache = {20: 'data:image/jpeg;base64,header20', 21: 'data:image/jpeg;base64,page21'}

        def run(header, source_bytes=None):
            def post(url, *, headers, json, timeout):
                response = dict(metrics=[
                    dict(metric_id=code, status='found', value_raw=raw, unit='万元',
                         period_label='期末数', source_label=label, page=21,
                         column_header_path=['期末数', '认可价值'], evidence_text=f'{label} {raw}', confidence=.99)
                    for code, label, raw in [
                        ('RECOGNIZED_ASSETS', '认可资产合计', '159,833,624,742.10'),
                        ('OTHER_RECOGNIZED_ASSETS', '其他认可资产', '-232,485,696.69'),
                    ]
                ])
                if header:
                    response.update(table_unit='元', unit_page=20, unit_evidence='S03-认可资产表（元）')
                return FakeResponse(response)

            return _extract_card_records(
                BOC_PDF.read_bytes() if source_bytes is None else source_bytes,
                match, card, api_key='test', base_url='https://example.invalid',
                model='test', timeout=30, request_max_attempts=1, post_func=post,
                page_image_cache=cache, unit_context_pages=[20, 21],
                recognized_view_cache={}, max_pages_per_request=2,
            )[0]

        good = {row['指标编码']: row for row in run(True)}
        self.assertEqual({row['单位'] for row in good.values()}, {'元'})
        self.assertAlmostEqual(good['RECOGNIZED_ASSETS']['标准数值'], 15983362.47421, places=5)
        self.assertAlmostEqual(good['OTHER_RECOGNIZED_ASSETS']['标准数值'], -23248.569669, places=5)
        native = run(False)
        self.assertEqual({row['单位'] for row in native}, {'元'})
        scanned_header = run(True, b'no-text-layer')
        self.assertEqual({row['单位'] for row in scanned_header}, {'元'})
        unverified = run(False, b'no-text-layer')
        self.assertTrue(all(row['单位'] == '' and row['标准数值'] is None for row in unverified))
        checks = validate_vlm_v2_metrics(pd.DataFrame(unverified, columns=METRIC_COLUMNS), [card])
        self.assertTrue(((checks['规则'] == '数值可解析') & (checks['状态'] == '失败')).any())

    def extract(self, native=False, retry=False, header=True, numeric_error=False):
        metric = dict(metric_id='RECOGNIZED_ASSETS', name='认可资产', expected_unit='万元', data_type='金额', semantic_key='RECOGNIZED_ASSETS')
        card = VLMV2TargetCard('RECOGNIZED_ASSETS', '认可资产表', '', (metric,))
        match = PageMatch(card.table_id, card.table_name, [24] if retry else [22,23,24], 99, '')
        cache = {p:f'data:image/jpeg;base64,page{p}' for p in [22,23,24]}
        calls = []
        def post(url, *, headers, json, timeout):
            content = json['messages'][0]['content']
            prompt = content[0]['text']
            page = int(re.search(r'页面顺序及物理页码：\[(\d+)\]', prompt)[1])
            calls.append(page)
            images = [part['image_url']['url'] for part in content if part['type']=='image_url']
            self.assertEqual(len(images), 1 if page==22 else 2)
            if page!=22:
                self.assertEqual(images[-1], cache[22])
                self.assertIn('禁止重复提取上下文页金额', prompt)
            value = '95,047,859,471.59' if not numeric_error else '96,047,859,471.59'
            items = [dict(metric_id='RECOGNIZED_ASSETS', status='found', value_raw=value,
                           unit='万元', period_label='期末数', source_label='合计', page=page,
                           column_header_path=['期末数', '认可价值'],
                           evidence_text='合计 期末认可价值 ' + value, confidence=.99)] if page==24 else []
            # A hallucinated value from context must never enter current-page results.
            if page!=22:
                items.append(dict(metric_id='RECOGNIZED_ASSETS', status='found', value_raw='1', unit='万元',
                                  source_label='上下文页错误返回', column_header_path=['期末数', '认可价值'],
                                  evidence_text='not current page', page=22, confidence=1))
            response = dict(metrics=items)
            if header:
                response.update(table_unit='元', unit_page=22, unit_evidence='单位：元')
            return FakeResponse(response)
        data = PDF.read_bytes() if native else b'no-text-layer'
        with patch('services.solvency_vlm_v2_pipeline._render_page_images') as render:
            rows, count = _extract_card_records(data, match, card, api_key='test', base_url='https://api.moonshot.cn/v1',
                model='kimi-k2.6', timeout=90, request_max_attempts=1, post_func=post,
                page_image_cache=cache, unit_context_pages=[22,23,24])
            render.assert_not_called()
        return rows[0], count, calls

    @unittest.skipUnless(PDF.exists(), 'Local PDF unavailable')
    def test_hongkang_native_header_corrects_wrong_model_unit(self):
        row, count, calls = self.extract(native=True, header=False)
        self.assertEqual(calls, [22,23,24])
        self.assertEqual(row['单位'], '元')
        self.assertAlmostEqual(row['标准数值'], 9504785.947159, places=6)
        self.assertIn('物理页22', row['证据原文'])
        self.assertEqual(count, 3)

    def test_scanned_context_header_corrects_wrong_model_unit(self):
        row, _, _ = self.extract()
        self.assertAlmostEqual(row['标准数值'], 9504785.947159, places=6)

    def test_retry_only_failed_page_retains_original_header_context(self):
        row, count, calls = self.extract(retry=True)
        self.assertEqual(calls, [24])
        self.assertEqual(count, 1)
        self.assertEqual(row['单位'], '元')

    def test_real_value_error_still_fails_cross_table_check(self):
        row, _, _ = self.extract(numeric_error=True)
        main = dict(row, **{'目标表ID':'SOLVENCY_MAIN', '原始值':'95,047,859,471.59',
                            '标准数值':9504785.947159, '物理页码':9})
        checks = validate_vlm_v2_metrics(pd.DataFrame([main,row], columns=METRIC_COLUMNS), [])
        cross = checks[checks['规则'].eq('跨表一致性')]
        self.assertEqual(cross.iloc[0]['状态'], '失败')

    def test_corrected_native_amount_agrees_with_main_table(self):
        row, _, _ = self.extract()
        main = dict(row, **{'目标表ID':'SOLVENCY_MAIN', '标准数值':9504785.947159, '物理页码':9})
        checks = validate_vlm_v2_metrics(pd.DataFrame([main,row], columns=METRIC_COLUMNS), [])
        self.assertFalse(((checks['规则']=='跨表一致性') & (checks['状态']=='失败')).any())
