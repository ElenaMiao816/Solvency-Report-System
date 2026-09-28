import io
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import fitz
import pandas as pd

from services.solvency_amounts import chinese_money_yuan
from services.solvency_pdf_locator import PageMatch
from services.solvency_vlm_v2_pipeline import (
    METRIC_COLUMNS, VLMV2ExtractionRun, VLMV2TargetCard,
    _normalize_number_and_unit, _parse_number, _cross_table_check,
    _extract_card_records,
    validate_vlm_v2_metrics, refresh_vlm_v2_cross_table_checks,
    evaluate_vlm_v2_step3_gate, vlm_v2_to_extracted_tables,
    vlm_v2_workbook_bytes, read_vlm_v2_extracted_tables,
)


# Manually checked against Caixin 2026Q1 physical pages 16, 34, 35, 37, 38.
CAIXIN = {
    'CORE_T1_CAPITAL': ('377,916.20', '3,779,161,979.81'),
    'RECOGNIZED_ASSETS': ('5,999,945.60', '59,999,455,957.35'),
    'QUANT_RISK_CAPITAL': ('446,872.24', '4,468,722,380.78'),
    'MINIMUM_CAPITAL': ('454,020.97', '4,540,209,681.87'),
    'CORE_T2_CAPITAL': ('534.88', '5,348,782.37'),
    'ACTUAL_CAPITAL': ('756,902.15', '7,569,021,524.36'),
    'CONTROL_RISK_CAPITAL': ('7,148.73', '71,487,301.09'),
    'ANC_T1_CAPITAL': ('378,451.08', '3,784,510,762.19'),
}


def record(code, raw, unit, table='SUMMARY', page=16):
    standard, _ = _normalize_number_and_unit(raw, unit, '万元')
    row = dict.fromkeys(METRIC_COLUMNS, '')
    row.update({'目标表ID': table, '目标表名称': table, '指标编码': code,
                '指标语义键': code, '指标名称': code, '原始值': raw, '单位': unit,
                '数值': _parse_number(raw, unit), '标准数值': standard, '标准单位': '万元',
                '状态': 'found', '期间口径': '本季度末数', '物理页码': page,
                '原始标签': code, '证据原文': f'{code} {raw} {unit}', '置信度': .99})
    return row


def pair(code='CORE_T1_CAPITAL'):
    summary, detail = CAIXIN[code]
    return pd.DataFrame([record(code, summary, '万元'), record(code, detail, '元', 'DETAIL', 34)])


class ChineseMoneyTests(unittest.TestCase):
    def test_chinese_uppercase_amounts(self):
        cases = {'伍拾亿元整': '5000000000', '人民币伍拾亿元正': '5000000000',
                 '五十亿元': '5000000000', '伍拾億圓整': '5000000000',
                 '壹亿零叁佰万元整': '103000000', '壹仟零伍拾万元整': '10500000',
                 '壹佰零贰元叁角肆分': '102.34', '零元伍分': '0.05', '零元整': '0'}
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(chinese_money_yuan(raw), Decimal(expected))

    def test_uncertain_and_malformed_amounts_are_not_guessed(self):
        for raw in ['约伍拾亿元', '伍拾亿元以上', '注册资本伍拾亿元，实收贰拾亿元',
                    '伍拾亿元至陆拾亿元', '伍拾', '一百二元', '壹佰仟元',
                    '伍拾亿万元', '壹万亿元', '一二三元', '伍拾亿元50', '50亿元', '-', None]:
            with self.subTest(raw=raw):
                self.assertIsNone(chinese_money_yuan(raw))

    def test_explicit_chinese_scale_applied_once(self):
        for source in ['', '元', '万元', '亿元']:
            for target, expected in [('元', 5000000000), ('万元', 500000), ('亿元', 50)]:
                self.assertEqual(_normalize_number_and_unit('伍拾亿元整', source, target), (expected, target))
        self.assertEqual(_parse_number('伍拾亿元整', '亿元'), 50)
        self.assertEqual(_normalize_number_and_unit('伍拾亿元整', '元', '%'), (None, '%'))
        for raw in ['伍拾亿元至60亿元', '伍拾亿元50', '五十亿元以上50', '约伍拾亿元']:
            self.assertEqual(_normalize_number_and_unit(raw, '亿元', '万元'), (None, '万元'))

    def test_numeric_dash_and_small_negative_regression(self):
        self.assertEqual(_normalize_number_and_unit('-0.01', '元', '万元')[0], -.000001)
        for raw in ['-', '0', '0.00']:
            self.assertEqual(_normalize_number_and_unit(raw, '元', '万元')[0], 0)
        self.assertEqual(_normalize_number_and_unit('50', '亿元', '万元')[0], 500000)


class CaixinPrecisionTests(unittest.TestCase):
    def test_all_eight_pdf_pairs_pass_without_changing_values(self):
        for code in CAIXIN:
            with self.subTest(code=code):
                records = pair(code)
                original = records.copy(deep=True)
                result = _cross_table_check(code, records)
                self.assertEqual(result['状态'], '通过')
                self.assertIn('原文披露精度', result['说明'])
                pd.testing.assert_frame_equal(records, original)

    def test_material_difference_still_fails(self):
        records = pair()
        records.loc[0, ['原始值', '标准数值']] = ['377,916.21', 377916.21]
        self.assertEqual(_cross_table_check('X', records)['状态'], '失败')

    def test_coarse_row_does_not_hide_precise_source_conflict(self):
        records = pair()
        records.loc[2] = record('CORE_T1_CAPITAL', '3,779,161,990.81', '元', 'DETAIL_2')
        self.assertEqual(_cross_table_check('X', records)['状态'], '失败')

    def test_missing_or_inconsistent_raw_evidence_and_units_stay_blocked(self):
        for column, value in [('原始值', ''), ('原始值', '377,916.19'), ('单位', '亿元'), ('标准单位', '亿元')]:
            records = pair()
            records.loc[0, column] = value
            self.assertEqual(_cross_table_check('X', records)['状态'], '失败')

    def test_same_precision_sources_and_boundary(self):
        records = pd.DataFrame([record('X', '1.00', '万元'), record('X', '1.01', '万元', 'OTHER')])
        self.assertEqual(_cross_table_check('X', records)['状态'], '失败')
        for fine, expected in [('10049.99', '通过'), ('10050.00', '失败'), ('9950.00', '通过')]:
            records = pd.DataFrame([record('X', '1.00', '万元'), record('X', fine, '元', 'OTHER')])
            self.assertEqual(_cross_table_check('X', records)['状态'], expected)

    def test_explicit_decimal_precision_in_same_unit_and_yuan_target(self):
        rows = pd.DataFrame([record('X', '377916.20', '万元'), record('X', '377916.197981', '万元', 'OTHER')])
        self.assertEqual(_cross_table_check('X', rows)['状态'], '通过')
        rows['标准数值'] = rows['标准数值'] * 10000
        rows['标准单位'] = '元'
        self.assertEqual(_cross_table_check('X', rows)['状态'], '通过')

    def old_run(self):
        records = pd.concat([pair(code) for code in CAIXIN], ignore_index=True)
        registered = record('REGISTERED_CAPITAL', '伍拾亿元整', '亿元', 'REGISTERED_CAPITAL', 2)
        registered['标准数值'] = registered['数值'] = None
        records.loc[len(records)] = registered
        card = VLMV2TargetCard('REGISTERED_CAPITAL', '注册资本', '', ({
            'metric_id': 'REGISTERED_CAPITAL', 'data_type': '金额', 'expected_unit': '万元'},))
        checks = validate_vlm_v2_metrics(records, [card])
        checks.loc[checks['规则'].eq('跨表一致性'), '状态'] = '失败'
        return VLMV2ExtractionRun(records, checks, (), 0, 0)

    def test_saved_nine_failures_repaired_and_step3_receives_numeric_capital(self):
        run = self.old_run()
        self.assertEqual(run.validations['状态'].eq('失败').sum(), 9)
        refreshed = refresh_vlm_v2_cross_table_checks(run)
        self.assertFalse(refreshed.validations['状态'].eq('失败').any())
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
        self.assertTrue(pd.isna(run.records.iloc[-1]['标准数值']))
        self.assertIs(refresh_vlm_v2_cross_table_checks(refreshed), refreshed)
        for tables in [vlm_v2_to_extracted_tables(run), read_vlm_v2_extracted_tables(pd.ExcelFile(io.BytesIO(vlm_v2_workbook_bytes(None, run))))]:
            capital = next(t for t in tables if t.table_id == 'REGISTERED_CAPITAL').metric_records[0]
            self.assertEqual(capital['标准数值'], 500000)
            self.assertEqual(capital['数值'], 50)
            self.assertEqual(capital['原始值'], '伍拾亿元整')

    def test_original_legacy_workbook_is_repaired_on_upload(self):
        run = self.old_run()
        stream = io.BytesIO()
        with pd.ExcelWriter(stream, engine='openpyxl') as writer:
            run.records.to_excel(writer, sheet_name='VLM_v2指标结果', index=False)
            run.validations.to_excel(writer, sheet_name='确定性校验', index=False)
        tables = read_vlm_v2_extracted_tables(pd.ExcelFile(io.BytesIO(stream.getvalue())))
        self.assertEqual(next(t for t in tables if t.table_id == 'REGISTERED_CAPITAL').metric_records[0]['标准数值'], 500000)

    def test_unrelated_failures_are_not_cleared(self):
        for rule in ['请求覆盖完整性', '本季度期间口径', '证据完整性']:
            run = self.old_run()
            run.validations.loc[len(run.validations)] = ['OTHER', 'REGISTERED_CAPITAL', '', '失败', rule, '', '', '待核验']
            self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)

    def test_newly_parseable_capital_conflict_is_not_hidden(self):
        run = self.old_run()
        run.records.loc[len(run.records)] = record('REGISTERED_CAPITAL', '40.00', '亿元', 'OTHER')
        self.assertFalse(evaluate_vlm_v2_step3_gate(run).passed)

    @unittest.skipUnless(Path(r'F:\CROSS\V1\财信人寿2026Q1偿付能力季度报告摘要.pdf').exists(), 'Local source PDF unavailable')
    def test_real_pdf_page_render_and_mocked_extraction_pipeline(self):
        pdf = Path(r'F:\CROSS\V1\财信人寿2026Q1偿付能力季度报告摘要.pdf').read_bytes()
        with fitz.open(stream=pdf, filetype='pdf') as doc:
            self.assertIn('伍拾亿元整', doc[1].get_text())
        card = VLMV2TargetCard('REGISTERED_CAPITAL', '注册资本', '', ({
            'metric_id': 'REGISTERED_CAPITAL', 'name': '注册资本',
            'data_type': '金额', 'expected_unit': '万元'},))
        match = PageMatch(table_id=card.table_id, table_name=card.table_name, pages=[2], score=99,
                          evidence='公司基本情况 注册资本', table_config={})
        response = {'metrics': [{'metric_id': 'REGISTERED_CAPITAL', 'status': 'found',
            'value_raw': '伍拾亿元整', 'unit': '亿元', 'page': 2, 'period_label': '本季度末数',
            'source_label': '注册资本', 'evidence_text': '注册资本：伍拾亿元整', 'confidence': .99}]}
        with patch('services.solvency_vlm_v2_pipeline._call_vlm_json', return_value=response) as model:
            rows, calls = _extract_card_records(pdf, match, card, api_key='test', base_url='https://example.test',
                model='test', timeout=90, request_max_attempts=1, post_func=None)
        self.assertEqual(calls, 1)
        self.assertEqual(len(model.call_args.kwargs['image_urls']), 1)
        records = pd.DataFrame(rows, columns=METRIC_COLUMNS)
        self.assertEqual(records.iloc[0]['标准数值'], 500000)
        self.assertEqual(records.iloc[0]['数值'], 50)
        run = VLMV2ExtractionRun(records, validate_vlm_v2_metrics(records, [card]), (), calls, 0)
        self.assertTrue(evaluate_vlm_v2_step3_gate(run).passed)
