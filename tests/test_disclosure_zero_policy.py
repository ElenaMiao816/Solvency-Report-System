import unittest

from services.solvency_disclosure_policy import is_disclosed_zero
from services.solvency_vlm_v2_pipeline import _normalized_model_status, _normalize_number_and_unit


class DisclosureZeroPolicyTests(unittest.TestCase):
    def test_dash_variants_and_numeric_zero_are_explicit_zero(self):
        for raw in ['-', '--', '---', '—', '——', '–', '−', '－', '﹣', ' — ', 0, 0.0, '0.00', '0%', '０.００％', '-0', '(0.00)']:
            with self.subTest(raw=raw):
                self.assertTrue(is_disclosed_zero(raw))
                self.assertEqual(_normalized_model_status({'status': 'found', 'value_raw': raw}, 'NET_ASSETS'), 'disclosed_zero')

    def test_absence_and_not_applicable_are_not_zero(self):
        for raw in ['', None, '不适用', '<不适用>', '未披露', '1', '-1']:
            with self.subTest(raw=raw):
                self.assertFalse(is_disclosed_zero(raw))
        self.assertEqual(_normalized_model_status({'status': 'not_disclosed', 'value_raw': '-'}, 'NET_ASSETS'), 'not_disclosed')
        self.assertEqual(_normalized_model_status({'status': 'disclosed_na', 'value_raw': '<不适用>'}, 'NET_ASSETS'), 'disclosed_na')
        self.assertEqual(_normalized_model_status({'status': 'found', 'value_raw': '-'}, 'TOP_FIVE_PRODUCTS'), 'found')

    def test_dash_zero_is_not_limited_to_capital_whitelist(self):
        for code in ['POLICY_SURPLUS_CORE_T2', 'RECOGNIZED_ASSETS', 'INVESTMENT_RETURN', 'AGENT_COUNT']:
            with self.subTest(code=code):
                self.assertEqual(_normalize_number_and_unit('—', '元', '万元', code), (0, '万元'))
