import ast
import json
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

from services.solvency_locator_roles import locator_page_hits, page_role, select_locator_pages
from services.solvency_vlm_v2_pipeline import locate_tables_vlm_v2
from tests.test_vlm_v2_pipeline import minimal_taxonomy, FakeResponse


NAMES = {'SOLVENCY_MAIN': '偿付能力充足率指标', 'OPERATING_METRICS': '主要经营指标',
         'REGISTERED_CAPITAL': '注册资本', 'ACTUAL_CAPITAL': 'S02-实际资本明细表',
         'RECOGNIZED_ASSETS': 'S03-认可资产表', 'THREE_YEAR_INVESTMENT_RETURN': '近三年收益率',
         'MINIMUM_CAPITAL': 'S05-最低资本表'}
EXPECTED = {'SOLVENCY_MAIN': [12], 'OPERATING_METRICS': [13, 14, 15], 'REGISTERED_CAPITAL': [2],
            'ACTUAL_CAPITAL': [28], 'RECOGNIZED_ASSETS': [28],
            'THREE_YEAR_INVESTMENT_RETURN': [15], 'MINIMUM_CAPITAL': [30]}


def hit(page, title, role='primary', groups=(), confidence=.95):
    return dict(page=page, source_title=title, evidence=title + ' 本季度数据行',
                role=role, coverage_groups=list(groups), confidence=confidence)


class LocatorSourceRoleTests(unittest.TestCase):
    def test_summary_is_not_a_detail_table_even_when_model_calls_it_primary(self):
        for table in ('ACTUAL_CAPITAL', 'MINIMUM_CAPITAL', 'RECOGNIZED_ASSETS'):
            self.assertEqual(page_role(table, '（一）偿付能力充足率指标', '包含资本分级和最低资本合计', 'primary'), 'summary')
        self.assertEqual(page_role('RECOGNIZED_ASSETS', '非认可资产表', '', 'primary'), 'reference')
        self.assertEqual(page_role('SOLVENCY_MAIN', '偿付能力变化及其原因分析', '', 'primary'), 'reference')

    def test_alternate_titles_and_shared_pages_work_without_form_codes(self):
        for code, title in [('ACTUAL_CAPITAL', '（一）实际资本表'), ('ACTUAL_CAPITAL', '十、实际资本'),
                            ('MINIMUM_CAPITAL', '十一、最低资本'), ('RECOGNIZED_ASSETS', '（二）认可资产表')]:
            self.assertEqual(page_role(code, title, ''), 'primary')
        self.assertEqual(page_role('OPERATING_METRICS', '报告期内签单保费占前五位的产品', ''), 'primary')
        self.assertEqual(page_role('OPERATING_METRICS', '品质类指标', '', 'continuation'), 'continuation')

    def test_summary_only_toc_and_missing_evidence_require_review(self):
        for title in ('目录', '偿付能力充足率指标', ''):
            items = locator_page_hits({'table_id': 'ACTUAL_CAPITAL', 'found': True,
                'page_hits': [dict(hit(3, title), evidence='')]}, [3])
            pages, _, reason, _, _ = select_locator_pages('ACTUAL_CAPITAL', items)
            self.assertTrue(reason)
            if title:
                self.assertEqual(pages, [])

    def test_physical_page_bounds_false_strings_and_confidence(self):
        self.assertEqual(locator_page_hits({'found': 'false', 'pages': [1]}, [1]), [])
        result = locator_page_hits({'table_id': 'SOLVENCY_MAIN', 'found': True,
            'page_hits': [hit(0, '偿付能力指标'), hit(2, '偿付能力指标'), hit(31, '偿付能力指标'), hit(True, '偿付能力指标')]}, [1, 2])
        self.assertEqual([h['page'] for h in result], [2])
        result += locator_page_hits({'table_id': 'SOLVENCY_MAIN', 'found': True,
            'page_hits': [hit(3, '偿付能力指标续表', 'continuation', confidence=.3)]}, [3])
        self.assertTrue(select_locator_pages('SOLVENCY_MAIN', result)[2])

    def test_continuations_across_batch_boundary_are_not_dropped(self):
        records = [dict(hit(18, '实际资本表'), role='primary'),
                   dict(hit(19, '附属二级资本明细', 'continuation'), role='continuation')]
        self.assertEqual(select_locator_pages('ACTUAL_CAPITAL', records)[0], [18, 19])


class HengqinLocatorTests(unittest.TestCase):
    def run_case(self, *, legacy=False, omit_tail=False, boundary_timeout=False, real_pdf=False):
        # Manually verified physical-page fixture. No production page constants.
        fixture = {
            'REGISTERED_CAPITAL': [hit(2, '公司信息')],
            'SOLVENCY_MAIN': [hit(12, '偿付能力充足率指标'), hit(24, '偿付能力变化及其原因分析')],
            'OPERATING_METRICS': [hit(13, '主要经营指标', groups=['main']),
                                  hit(14, '效益类指标、规模类指标', 'continuation', ['benefit', 'scale'])],
            'ACTUAL_CAPITAL': [hit(12, '偿付能力充足率指标'), hit(28, '（一）实际资本表')],
            'RECOGNIZED_ASSETS': [hit(28, '（二）认可资产表')],
            'THREE_YEAR_INVESTMENT_RETURN': [hit(15, '近三年（综合）投资收益率')],
            'MINIMUM_CAPITAL': [hit(12, '偿付能力充足率指标'), hit(30, '十一、最低资本')],
        }
        tail = hit(15, '前五位产品、品质类指标', 'continuation', ['products', 'quality'])
        if not omit_tail and not legacy:
            fixture['OPERATING_METRICS'].append(tail)
        calls = []
        def post(_url, *, headers, json: dict, timeout):
            prompt = json['messages'][0]['content'][0]['text']
            pages = __import__('json').loads(re.search(r'当前批次包含物理页：(\[[^\]]*\])', prompt)[1])
            cards = __import__('json').loads(re.search(r'目标定义：\n(.+?)\n\n要求：', prompt, re.S)[1])
            ids = [card['table_id'] for card in cards]
            calls.append((pages, ids))
            self.assertEqual(json['thinking'], {'type': 'disabled'})
            self.assertEqual(sum(c['type'] == 'image_url' for c in json['messages'][0]['content']), 1)
            self.assertLessEqual(len(pages), 6)
            self.assertIn('一页可同时包含多个目标表', prompt)
            self.assertIn('禁止用页脚印刷页码', prompt)
            focused = ids == ['OPERATING_METRICS']
            if focused and boundary_timeout:
                raise requests.ReadTimeout('boundary test timeout')
            targets = []
            for table_id in ids:
                relevant = ([tail] if focused else fixture[table_id])
                relevant = [h for h in relevant if h['page'] in pages]
                if legacy and not focused:
                    for h in relevant:
                        targets.append(dict(table_id=table_id, found=True, pages=[h['page']],
                            source_title=h['source_title'], evidence=h['evidence'], confidence=h['confidence']))
                elif relevant:
                    targets.append(dict(table_id=table_id, found=True, page_hits=relevant))
            if focused:
                # A local operating review must not overwrite other targets.
                targets.append(dict(table_id='ACTUAL_CAPITAL', found=True,
                    page_hits=[hit(pages[0], '实际资本表')]))
            return FakeResponse({'targets': targets})
        configs = [dict(table_id=key, table_name=name) for key, name in NAMES.items()]
        kwargs = dict(api_key='test', base_url='https://api.moonshot.cn/v1', model='kimi-k2.6', post_func=post)
        if real_pdf:
            run = locate_tables_vlm_v2(Path(r'F:\CROSS\V1\横琴人寿2026Q1偿付能力季度报告摘要.pdf').read_bytes(), configs, minimal_taxonomy(), **kwargs)
        else:
            sheets = [(tuple(range(i, min(i + 6, 31))), 'data:image/jpeg;base64,test') for i in range(1, 31, 6)]
            review = lambda data, pages: [
                (tuple(pages[i:i + 2]), 'data:image/jpeg;base64,test')
                for i in range(0, len(pages), 2)
            ]
            with patch('services.solvency_vlm_v2_pipeline.render_vlm_v2_contact_sheets', return_value=sheets) as render, \
                 patch('services.solvency_vlm_v2_pipeline.render_review_sheets', side_effect=review):
                run = locate_tables_vlm_v2(b'cached', configs, minimal_taxonomy(), **kwargs)
                self.assertEqual(render.call_count, 1)
        return run, calls

    def test_primary_tables_selected_summaries_retained_as_auxiliary(self):
        run, calls = self.run_case()
        self.assertEqual({m.table_id: m.pages for m in run.matches}, EXPECTED)
        self.assertEqual(run.model_calls, 6)
        self.assertFalse(any(m.review_required for m in run.matches))
        capital = next(m for m in run.matches if m.table_id == 'ACTUAL_CAPITAL')
        self.assertEqual(capital.sources['vlm_summary'], [12])
        self.assertIn('物理页12[辅助/未采用/summary]', capital.evidence)

    def test_screenshot_legacy_results_are_filtered_and_missing_tail_reviewed(self):
        run, calls = self.run_case(legacy=True)
        self.assertEqual({m.table_id: m.pages for m in run.matches}, EXPECTED)
        self.assertEqual(run.model_calls, 7)
        focused = [call for call in calls if call[1] == ['OPERATING_METRICS']]
        self.assertEqual(focused, [([13, 14, 15, 16, 17, 18], ['OPERATING_METRICS'])])

    def test_one_bounded_recheck_recovers_missing_operating_page(self):
        run, calls = self.run_case(omit_tail=True)
        self.assertEqual({m.table_id: m.pages for m in run.matches}, EXPECTED)
        self.assertEqual(run.model_calls, 7)

    def test_boundary_timeout_keeps_success_and_requests_manual_review(self):
        run, calls = self.run_case(omit_tail=True, boundary_timeout=True)
        operating = next(m for m in run.matches if m.table_id == 'OPERATING_METRICS')
        self.assertEqual(operating.pages, [13, 14])
        self.assertTrue(operating.review_required)
        self.assertIn('边界复核失败', operating.review_reason)
        self.assertEqual(run.model_calls, 7)

    @unittest.skipUnless(Path(r'F:\CROSS\V1\横琴人寿2026Q1偿付能力季度报告摘要.pdf').exists(), 'Local PDF unavailable')
    def test_real_pdf_rendering_with_controlled_visual_responses(self):
        run, _ = self.run_case(real_pdf=True)
        self.assertEqual(run.page_count, 30)
        self.assertEqual({m.table_id: m.pages for m in run.matches}, EXPECTED)


class LocatorEvidenceUiTests(unittest.TestCase):
    def test_evidence_is_inside_one_collapsed_expander_and_candidate_still_applies(self):
        source = (Path(__file__).parents[1] / 'app.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        names = {'locator_candidate_rows', 'apply_locator_candidate', 'render_locator_evidence'}
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
                 or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LOCATOR_SOURCE_LABELS' for t in n.targets)]
        snippet = '\n'.join(['import streamlit as st', 'import pandas as pd',
            'from services.solvency_pdf_locator import PageMatch',
            'from services.solvency_locator_roles import LOCATOR_ROLE_LABELS',
            *[ast.unparse(n) for n in nodes],
            "st.session_state.setdefault('page_edit_ACTUAL_CAPITAL', '28')",
            "st.session_state.setdefault('pages_confirmed', True)",
            "st.text_input('S02-实际资本明细表', key='page_edit_ACTUAL_CAPITAL')",
            "match = PageMatch('ACTUAL_CAPITAL', 'S02', [28], 95, '主表第28页；汇总第12页', sources={'vlm_primary':[28], 'vlm_summary':[12]})",
            "render_locator_evidence(match, 'page_edit_ACTUAL_CAPITAL')"])
        app = AppTest.from_string(snippet).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.expander), 1)
        self.assertFalse(app.expander[0].proto.expanded)
        self.assertEqual(len(app.expander[0].caption), 2)
        self.assertEqual(len(app.caption), 2)
        self.assertIn('汇总页', app.dataframe[0].value.iloc[2]['候选来源'])
        app.selectbox[0].set_value(2).run()
        app.button[0].click().run()
        self.assertEqual(app.text_input[0].value, '12')
        self.assertFalse(app.session_state['pages_confirmed'])
