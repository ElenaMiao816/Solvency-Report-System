import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from services.solvency_pdf_locator import PageMatch
from services.solvency_unit_context import apply_item_unit_evidence
from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS,
    VLMV2TargetCard,
    _extract_card_records,
)


ICBC_AXA_PDF = Path(r'F:\CROSS\V1\工银安盛2026Q1偿付能力季度报告摘要.pdf')


class RegisteredCapitalUnitEvidenceTests(unittest.TestCase):
    def test_all_supported_company_source_units_override_model_guess(self):
        for unit in ('元', '千元', '万元', '百万元', '亿元'):
            with self.subTest(unit=unit):
                item = apply_item_unit_evidence({
                    'value_raw': '125.05',
                    'unit': '万元' if unit != '万元' else '亿元',
                    'evidence_text': f'注册资本（营运资金）：125.05{unit}人民币',
                }, require_explicit=True)
                self.assertEqual(item['unit'], unit)

    def test_matching_value_does_not_borrow_an_unrelated_amount_unit(self):
        item = apply_item_unit_evidence({
            'value_raw': '125.05',
            'unit': '万元',
            'evidence_text': '注册资本：125.05亿元；实收资本：80万元',
        }, require_explicit=True)
        self.assertEqual(item['unit'], '亿元')

    def test_unverified_model_unit_is_blocked_instead_of_assumed(self):
        item = apply_item_unit_evidence({
            'value_raw': '125.05',
            'unit': '万元',
            'evidence_text': '注册资本：125.05',
        }, require_explicit=True)
        self.assertEqual(item['unit'], '')
        self.assertIn('未采用模型推测单位', item['evidence_text'])

    def test_conflicting_units_for_same_value_are_blocked(self):
        item = apply_item_unit_evidence({
            'value_raw': '100',
            'unit': '万元',
            'evidence_text': '注册资本100万元；另一处写为100亿元',
        }, require_explicit=True)
        self.assertEqual(item['unit'], '')

    def test_chinese_amount_literal_supplies_its_own_unit(self):
        item = apply_item_unit_evidence({
            'value_raw': '伍拾亿元整',
            'unit': '万元',
            'evidence_text': '注册资本：伍拾亿元整',
        }, require_explicit=True)
        self.assertEqual(item['unit'], '亿元')

    @unittest.skipUnless(ICBC_AXA_PDF.exists(), 'Local ICBC-AXA source PDF unavailable')
    def test_icbc_axa_real_page_corrects_wan_yuan_to_yi_yuan(self):
        pdf = ICBC_AXA_PDF.read_bytes()
        card = VLMV2TargetCard('REGISTERED_CAPITAL', '注册资本', '', ({
            'metric_id': 'REGISTERED_CAPITAL',
            'name': '注册资本',
            'data_type': '金额',
            'expected_unit': '万元',
        },))
        match = PageMatch(
            table_id=card.table_id,
            table_name=card.table_name,
            pages=[3],
            score=99,
            evidence='公司信息 注册资本（营运资金）',
            table_config={},
        )
        response = {'metrics': [{
            'metric_id': 'REGISTERED_CAPITAL',
            'status': 'found',
            'value_raw': '125.05',
            'unit': '万元',
            'page': 3,
            'period_label': '',
            'source_label': '注册资本（营运资金）',
            'evidence_text': '注册资本（营运资金）：125.05亿元人民币',
            'confidence': .99,
        }]}
        with patch(
            'services.solvency_vlm_v2_pipeline._call_vlm_json',
            return_value=response,
        ) as model:
            rows, calls = _extract_card_records(
                pdf,
                match,
                card,
                api_key='test',
                base_url='https://example.test',
                model='test',
                timeout=90,
                request_max_attempts=1,
                post_func=None,
            )

        self.assertEqual(calls, 1)
        self.assertEqual(len(model.call_args.kwargs['image_urls']), 1)
        row = pd.DataFrame(rows, columns=METRIC_COLUMNS).iloc[0]
        self.assertEqual(row['单位'], '亿元')
        self.assertEqual(row['标准单位'], '万元')
        self.assertEqual(row['标准数值'], 1_250_500.0)


if __name__ == '__main__':
    unittest.main()
