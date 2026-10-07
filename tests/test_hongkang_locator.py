import base64
import io
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import requests

from services.solvency_locator_recovery import (
    _catalog_title_candidate, bounded_review_pages, latest_scan_failures, operating_section_closed,
    render_review_sheets, text_page_catalog,
)
from services.solvency_vlm_v2_pipeline import locate_tables_vlm_v2
from tests.test_hengqin_locator import NAMES, hit
from tests.test_vlm_v2_pipeline import FakeResponse, minimal_taxonomy, image_only_pdf


PDF = Path(r'F:\CROSS\V1\弘康人寿2026Q1偿付能力季度报告摘要.pdf')
ICBC_ALLIANZ_PDF = Path(r'F:\CROSS\V1\工银安盛2026Q2偿付能力季度报告摘要.pdf')
CATALOG = [dict(page=21, titles=['九、实际资本', '（一）实际资本表']),
           dict(page=22, titles=['（二）认可资产表']), dict(page=25, titles=['（三）认可负债表']),
           dict(page=26, titles=['十、最低资本'])]
EXPECTED = dict(SOLVENCY_MAIN=[9], OPERATING_METRICS=[11], REGISTERED_CAPITAL=[2],
                ACTUAL_CAPITAL=[21], RECOGNIZED_ASSETS=[22, 23, 24],
                THREE_YEAR_INVESTMENT_RETURN=[11], MINIMUM_CAPITAL=[26])


class RecoveryHelpersTests(unittest.TestCase):
    def test_regulator_form_codes_are_valid_catalog_titles(self):
        for title in ('S02-实际资本表', 'S03-认可资产表', 'S04-认可负债表', 'S05-最低资本表'):
            self.assertTrue(_catalog_title_candidate(title))
        for non_title in ('2026年第2季度', '40,331,114', '说明：本表数据如下'):
            self.assertFalse(_catalog_title_candidate(non_title))

    def test_main_table_hit_rechecks_one_unlabelled_continuation_page(self):
        hits = [hit(10, '偿付能力充足率指标')]
        self.assertEqual(bounded_review_pages('SOLVENCY_MAIN', hits, [], 24), [10, 11])
        complete = hits + [hit(11, '主表续页', 'continuation')]
        self.assertEqual(bounded_review_pages('SOLVENCY_MAIN', complete, [], 24), [])

    def test_heading_candidates_exclude_summary_toc_and_liabilities(self):
        catalog = CATALOG + [dict(page=3, titles=['实际资本表…………18']),
                             dict(page=9, titles=['偿付能力充足率指标'])]
        # Catalog input from vision must also reject dotted TOC titles.
        self.assertEqual(bounded_review_pages('ACTUAL_CAPITAL', [], catalog, 26), [21])
        self.assertEqual(bounded_review_pages('RECOGNIZED_ASSETS', [], catalog, 26), [22, 23, 24])

    def test_partial_table_window_requires_visual_recheck(self):
        self.assertEqual(bounded_review_pages('RECOGNIZED_ASSETS', [hit(22, '认可资产表')], CATALOG, 26), [22, 23, 24])
        complete = [dict(hit(p, '认可资产表'), role='primary') for p in [22, 23, 24]]
        self.assertEqual(bounded_review_pages('RECOGNIZED_ASSETS', complete, CATALOG, 26), [])

    def test_candidate_window_is_bounded_for_unknown_end(self):
        catalog = [dict(page=p, titles=['实际资本表']) for p in [5, 15, 25]]
        self.assertLessEqual(len(bounded_review_pages('ACTUAL_CAPITAL', [], catalog, 40)), 6)

    def test_operating_groups_optional_only_with_visible_boundary(self):
        item = {'section_end': dict(confirmed=True, page=11, next_heading='（五）近三年（综合）投资收益率', evidence='经营指标表后紧接下一独立小节')}
        hits = [hit(11, '主要经营指标')]
        self.assertTrue(operating_section_closed(item, hits, [7, 8, 9, 10, 11, 12]))
        item['section_end']['page'] = 19
        self.assertFalse(operating_section_closed(item, hits, [11, 19]))
        item['section_end']['page'] = 11
        item['section_end']['next_heading'] = '品质类指标'
        self.assertFalse(operating_section_closed(item, hits, [11]))

    def test_focused_success_cannot_mask_failed_global_batch(self):
        rows = [dict(批次序号=4, 轮次='首轮', 状态='超时'),
                dict(批次序号=4, 轮次='超时重试', 状态='超时'),
                dict(批次序号=4, 轮次='经营指标边界复核', 状态='成功')]
        self.assertEqual(len(latest_scan_failures(rows)), 1)

    def test_no_text_layer_still_has_visual_recovery_candidates(self):
        self.assertEqual([c for c in text_page_catalog(image_only_pdf()) if c['titles']], [])
        self.assertEqual(bounded_review_pages('ACTUAL_CAPITAL', [], [dict(page=1, titles=['实际资本表'])], 2), [1, 2])

    @unittest.skipUnless(PDF.exists(), 'Local PDF unavailable')
    def test_real_pdf_title_hints_and_large_landscape_rendering(self):
        data = PDF.read_bytes()
        catalog = text_page_catalog(data)
        self.assertEqual(bounded_review_pages('ACTUAL_CAPITAL', [], catalog, 26), [21])
        self.assertEqual(bounded_review_pages('RECOGNIZED_ASSETS', [], catalog, 26), [22, 23, 24])
        sheets = render_review_sheets(data, [21, 22, 23, 24])
        self.assertEqual([p for p, _ in sheets], [(21, 22), (23, 24)])
        for _, url in sheets:
            raw = base64.b64decode(url.split(',')[1])
            self.assertLessEqual(len(raw), 1_500_000)
            with Image.open(io.BytesIO(raw)) as im:
                self.assertGreaterEqual(im.width, 1500)

    @unittest.skipUnless(ICBC_ALLIANZ_PDF.exists(), 'Local PDF unavailable')
    def test_icbc_allianz_s04_title_is_recoverable(self):
        catalog = text_page_catalog(ICBC_ALLIANZ_PDF.read_bytes())
        page_23 = next(item for item in catalog if item['page'] == 23)
        self.assertIn('S04-认可负债表', page_23['titles'])
        self.assertEqual(bounded_review_pages('RECOGNIZED_LIABILITIES', [], catalog, 24), [23])
        self.assertEqual(
            bounded_review_pages('SOLVENCY_MAIN', [hit(10, '偿付能力充足率指标')], catalog, 24),
            [10, 11],
        )


class HongkangPipelineTests(unittest.TestCase):
    def test_main_table_continuation_is_visually_recovered(self):
        calls = []

        def post(url, *, headers, json: dict, timeout):
            prompt = json['messages'][0]['content'][0]['text']
            focused = '这是缺失表/无标题续页高清复核' in prompt
            calls.append(focused)
            hits = [hit(1, '偿付能力充足率指标')]
            if focused:
                hits.append(hit(2, '最低资本及偿付能力充足率', 'continuation'))
            return FakeResponse(dict(targets=[dict(
                table_id='SOLVENCY_MAIN', found=True, page_hits=hits,
            )], page_catalog=[]))

        configs = [dict(table_id='SOLVENCY_MAIN', table_name='偿付能力充足率指标')]
        sheets = [((1, 2), 'data:image/jpeg;base64,test')]
        review = lambda data, pages: [(tuple(pages), 'data:image/jpeg;base64,test')]
        with patch('services.solvency_vlm_v2_pipeline.render_vlm_v2_contact_sheets', return_value=sheets), \
             patch('services.solvency_vlm_v2_pipeline.text_page_catalog', return_value=[]), \
             patch('services.solvency_vlm_v2_pipeline.render_review_sheets', side_effect=review):
            run = locate_tables_vlm_v2(
                b'pdf', configs, minimal_taxonomy(), api_key='test',
                base_url='https://api.moonshot.cn/v1', model='kimi-k2.6', post_func=post,
            )
        self.assertEqual(run.matches[0].pages, [1, 2])
        self.assertEqual(calls, [False, True])

    def run_case(self, *, scan_timeout=False, review_timeout=False, partial=False,
                 reject=False, visual_catalog=False, no_hints=False, real=False):
        calls = []
        fixture = {'SOLVENCY_MAIN': [hit(9, '偿付能力充足率指标')],
                   'OPERATING_METRICS': [hit(11, '主要经营指标', groups=['main'])],
                   'REGISTERED_CAPITAL': [hit(2, '注册资本')],
                   'THREE_YEAR_INVESTMENT_RETURN': [hit(11, '近三年收益率')],
                   'MINIMUM_CAPITAL': [hit(26, '最低资本表')],
                   'ACTUAL_CAPITAL': [], 'RECOGNIZED_ASSETS': [hit(22, '认可资产表')] if partial else []}
        recovery = {'ACTUAL_CAPITAL': [hit(21, '实际资本表')],
                    'RECOGNIZED_ASSETS': [hit(22, '认可资产表'), hit(23, '再保险资产及应收款项', 'continuation'), hit(24, '独立账户资产及认可资产合计', 'continuation')]}

        def post(url, *, headers, json: dict, timeout):
            prompt = json['messages'][0]['content'][0]['text']
            pages = __import__('json').loads(re.search(r'当前批次包含物理页：(\[[^\]]*\])', prompt)[1])
            cards = __import__('json').loads(re.search(r'目标定义：\n(.+?)\n\n要求：', prompt, re.S)[1])
            focused = '这是缺失表/无标题续页高清复核' in prompt
            calls.append((pages, focused))
            self.assertEqual(json['thinking'], {'type': 'disabled'})
            self.assertEqual(sum(c['type'] == 'image_url' for c in json['messages'][0]['content']), 1)
            if focused:
                self.assertLessEqual(len(pages), 2)
                if review_timeout:
                    raise requests.ReadTimeout('review timeout')
            elif scan_timeout and 21 in pages:
                raise requests.ReadTimeout('scan timeout')
            targets = []
            for card in cards:
                code = card['table_id']
                hits = [h for h in (recovery if focused else fixture).get(code, []) if h['page'] in pages]
                if focused and reject:
                    hits = []
                item = dict(table_id=code, found=bool(hits), page_hits=hits)
                if code == 'OPERATING_METRICS' and hits:
                    item['section_end'] = dict(confirmed=True, page=11, next_heading='近三年（综合）投资收益率', evidence='完整经营表后为近三年收益率独立小节')
                targets.append(item)
            catalog = [c for c in CATALOG if c['page'] in pages] if visual_catalog else []
            return FakeResponse(dict(targets=targets, page_catalog=catalog))

        def render(data, pages):
            return [(tuple(pages[i:i+2]), 'data:image/jpeg;base64,test') for i in range(0, len(pages), 2)]

        kwargs = dict(api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6', post_func=post)
        configs = [dict(table_id=c, table_name=n) for c, n in NAMES.items()]
        if real:
            result = locate_tables_vlm_v2(PDF.read_bytes(), configs, minimal_taxonomy(), **kwargs)
        else:
            sheets = [(tuple(range(i, min(i + 6, 27))), 'data:image/jpeg;base64,test') for i in range(1, 27, 6)]
            with patch('services.solvency_vlm_v2_pipeline.render_vlm_v2_contact_sheets', return_value=sheets), \
                 patch('services.solvency_vlm_v2_pipeline.text_page_catalog', return_value=[] if visual_catalog or no_hints else CATALOG), \
                 patch('services.solvency_vlm_v2_pipeline.render_review_sheets', side_effect=render) as rendering:
                result = locate_tables_vlm_v2(b'pdf', configs, minimal_taxonomy(), **kwargs)
                self.assertLessEqual(rendering.call_count, 1)
        return result, calls

    def test_missing_details_recovered_and_operating_no_false_warning(self):
        run, calls = self.run_case()
        self.assertEqual({m.table_id:m.pages for m in run.matches}, EXPECTED)
        self.assertFalse(any(m.review_required for m in run.matches))
        self.assertEqual(run.model_calls, 8)
        self.assertEqual([p for p, focused in calls if focused], [[9,10], [21,22], [23,24]])

    def test_scanned_visual_catalog_recovers_same_missing_ranges(self):
        run, _ = self.run_case(visual_catalog=True)
        self.assertEqual({m.table_id:m.pages for m in run.matches}, EXPECTED)

    def test_incomplete_asset_range_is_rechecked(self):
        run, _ = self.run_case(partial=True)
        self.assertEqual(next(m.pages for m in run.matches if m.table_id=='RECOGNIZED_ASSETS'), [22,23,24])

    def test_scan_timeout_preserves_success_and_reports_unfinished_scan(self):
        run, _ = self.run_case(scan_timeout=True, no_hints=True)
        missing = next(m for m in run.matches if m.table_id=='ACTUAL_CAPITAL')
        self.assertIn('19-24', missing.review_reason)
        self.assertIn('扫描未完成', missing.review_reason)
        self.assertEqual(next(m.pages for m in run.matches if m.table_id=='SOLVENCY_MAIN'), [9])

    def test_review_timeout_does_not_discard_successful_tables(self):
        run, _ = self.run_case(review_timeout=True, partial=True)
        asset = next(m for m in run.matches if m.table_id=='RECOGNIZED_ASSETS')
        self.assertEqual(asset.pages, [22])
        self.assertIn('高清复核请求失败', asset.review_reason)

    def test_text_hints_alone_never_become_final_matches(self):
        run, _ = self.run_case(reject=True)
        asset = next(m for m in run.matches if m.table_id=='RECOGNIZED_ASSETS')
        self.assertEqual(asset.pages, [])
        self.assertTrue(asset.review_required)
        self.assertEqual(asset.sources['vlm_recovery_candidate'], [22,23,24])

    @unittest.skipUnless(PDF.exists(), 'Local PDF unavailable')
    def test_real_pdf_with_controlled_model_responses(self):
        run, _ = self.run_case(real=True)
        self.assertEqual({m.table_id:m.pages for m in run.matches}, EXPECTED)
        self.assertEqual(run.page_count, 26)
