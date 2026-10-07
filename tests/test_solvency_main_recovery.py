import unittest
from pathlib import Path
from unittest.mock import patch

import services.solvency_vlm_v2_pipeline as vlm_pipeline
from services.solvency_pdf_locator import PageMatch
from services.solvency_main_recovery import recover_solvency_main_metrics


PDF = Path(r'F:\CROSS\V1\工银安盛2026Q2偿付能力季度报告摘要.pdf')


def word(x, y, text, block=0, line=0, order=0):
    return (x, y, x + max(12, len(text) * 7), y + 12, text, block, line, order)


class FakePage:
    def __init__(self, text='', words=()):
        self.text = text
        self.words = list(words)

    def get_text(self, kind=None):
        return self.words if kind == 'words' else self.text


class FakeDoc:
    def __init__(self, pages):
        self.pages = pages

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class SolvencyMainRecoveryTests(unittest.TestCase):
    def test_pipeline_replaces_model_non_disclosure_with_exact_source_row(self):
        metrics = tuple({
            'metric_id': code,
            'name': code,
            'expected_unit': '%' if 'RATIO' in code else '万元',
        } for code in ('MINIMUM_CAPITAL', 'CORE_SOLVENCY_RATIO', 'COMBINED_SOLVENCY_RATIO'))
        card = vlm_pipeline.VLMV2TargetCard('SOLVENCY_MAIN', '偿付能力主表', '', metrics)
        match = PageMatch('SOLVENCY_MAIN', '偿付能力主表', [1, 2], 99, '主表跨页')
        model_rows = [
            {'metric_id': metric['metric_id'], 'status': 'not_disclosed'}
            for metric in metrics
        ]
        recovered = {
            'MINIMUM_CAPITAL': {
                'metric_id': 'MINIMUM_CAPITAL', 'status': 'found', 'value_raw': '3,098,166',
                'unit': '万元', 'period_label': '本季度数', 'source_label': '最低资本',
                'row_header_path': ['最低资本'], 'column_header_path': ['本季度数'],
                'page': 2, 'evidence_text': '最低资本｜本季度数 3,098,166；单位依据：单位：万元',
                'confidence': 1.0, 'recovery_mode': '源PDF文字行补提',
            }
        }
        with patch.object(vlm_pipeline, '_call_vlm_json', return_value={'metrics': model_rows}), \
             patch.object(vlm_pipeline, 'recover_solvency_main_metrics', return_value=recovered) as fallback:
            rows, calls = vlm_pipeline._extract_card_records(
                b'pdf', match, card, api_key='key', base_url='https://example.test',
                model='model', timeout=1, request_max_attempts=1, post_func=None,
                page_image_cache={1: 'data:image/jpeg;base64,x', 2: 'data:image/jpeg;base64,y'},
                split_metrics=False,
            )
        fallback.assert_called_once()
        self.assertEqual(calls, 1)
        by_code = {row['指标编码']: row for row in rows}
        self.assertEqual(by_code['MINIMUM_CAPITAL']['状态'], 'found')
        self.assertEqual(by_code['MINIMUM_CAPITAL']['标准数值'], 3098166.0)
        self.assertEqual(by_code['MINIMUM_CAPITAL']['提取轮次'], '源PDF文字行补提')
        self.assertEqual(by_code['CORE_SOLVENCY_RATIO']['状态'], 'not_disclosed')

    def test_exact_rows_use_leftmost_current_value_and_title_page_unit(self):
        title = FakePage('（一）偿付能力充足率指标\n单位：万元\n项目 本季度数 上季度数')
        rows = FakePage(words=[
            word(10, 10, '4'), word(60, 11, '最低资本'), word(260, 10, '3,098,166'), word(360, 10, '3,042,540'),
            word(10, 30, '6'), word(60, 31, '核心偿付能力充足率'), word(260, 30, '136%'), word(360, 30, '128%'),
            word(10, 50, '8'), word(60, 51, '综合偿付能力充足率'), word(260, 50, '195%'), word(360, 50, '187%'),
            word(10, 70, '4.1'), word(60, 71, '量化风险最低资本'), word(260, 70, '3,104,431'),
        ])
        with patch('services.solvency_main_recovery.fitz.open', return_value=FakeDoc([title, rows])):
            result = recover_solvency_main_metrics(b'pdf', [1, 2])
        self.assertEqual(result['MINIMUM_CAPITAL']['value_raw'], '3,098,166')
        self.assertEqual(result['MINIMUM_CAPITAL']['unit'], '万元')
        self.assertEqual(result['MINIMUM_CAPITAL']['page'], 2)
        self.assertEqual(result['CORE_SOLVENCY_RATIO']['value_raw'], '136%')
        self.assertEqual(result['COMBINED_SOLVENCY_RATIO']['value_raw'], '195%')
        self.assertNotEqual(result['MINIMUM_CAPITAL']['value_raw'], '3,104,431')

    def test_conflicting_exact_rows_remain_unresolved(self):
        title = FakePage('偿付能力充足率指标\n单位：万元\n本季度数')
        first = [word(60, 10, '最低资本'), word(260, 10, '100')]
        second = [word(60, 10, '最低资本'), word(260, 10, '200')]
        with patch('services.solvency_main_recovery.fitz.open', return_value=FakeDoc([
            title, FakePage(words=first), FakePage(words=second),
        ])):
            result = recover_solvency_main_metrics(b'pdf', [1, 2, 3], {'MINIMUM_CAPITAL'})
        self.assertNotIn('MINIMUM_CAPITAL', result)

    @unittest.skipUnless(PDF.exists(), 'Local PDF unavailable')
    def test_real_icbc_allianz_continuation_rows(self):
        result = recover_solvency_main_metrics(PDF.read_bytes(), [10, 11])
        self.assertEqual(result['MINIMUM_CAPITAL']['value_raw'], '3,098,166')
        self.assertEqual(result['MINIMUM_CAPITAL']['unit'], '万元')
        self.assertEqual(result['CORE_SOLVENCY_RATIO']['value_raw'], '136%')
        self.assertEqual(result['COMBINED_SOLVENCY_RATIO']['value_raw'], '195%')
        self.assertTrue(all(result[code]['page'] == 11 for code in result))


if __name__ == '__main__':
    unittest.main()
