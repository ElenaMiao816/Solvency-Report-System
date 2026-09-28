"""Auditable filing targets transcribed from the five supplied checklists."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

CATALOG_PATH = Path(__file__).parents[1] / 'config' / 'solvency_filing_checklists.json'

# These remain separate from the five supplied disclosure lists. STEP2's older
# exports use the quarterly code plus a THREE_YEAR_AVERAGE semantic/period key.
THREE_YEAR_TARGETS = {
    'INVESTMENT_RETURN': ('THREE_YEAR_AVG_INVESTMENT_RETURN', '近三年平均投资收益率'),
    'COMPREHENSIVE_INVESTMENT_RETURN': (
        'THREE_YEAR_AVG_COMPREHENSIVE_INVESTMENT_RETURN', '近三年平均综合投资收益率'),
}
THREE_YEAR_TARGET_CODES = {code for code, _ in THREE_YEAR_TARGETS.values()}


def filing_entries() -> list[dict]:
    return [entry for entry in json.loads(CATALOG_PATH.read_text(encoding='utf-8'))['records']
            if entry['kind'] == 'metric']


def filing_codes(table_id: str = '') -> set[str]:
    return {entry['code'] for entry in filing_entries()
            if not table_id or entry['table_id'] == table_id}


def filing_label(value: object) -> str:
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'^\([一二三四五六七八九十]+\)|^\d+[.、](?=13个月)', '', text)
    text = re.sub(r'^\d+(?:\.\d+)*[.、]|^6\.[1-5](?=\D)', '', text)
    return re.sub(r'\(单位[:：]人\)', '', text)


def filing_row_code(table_id: str, label: str, row_number: str = '') -> str:
    """Resolve duplicate labels only within their source table and (if needed) row."""
    matches = {entry['code'] for entry in filing_entries()
               if entry['table_id'] == table_id
               and filing_label(label) in {filing_label(entry['source_label']), filing_label(entry['name'])}
               and (not row_number or entry['row_number'] == row_number)}
    return next(iter(matches)) if len(matches) == 1 else ''


def extend_filing_taxonomy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    result = taxonomy.copy()
    existing = set(result['指标编码'].astype(str))
    additions = []
    for entry in filing_entries():
        code = entry['code']
        if code not in existing:
            additions.append({
                '指标编码': code, '指标名称': entry['name'], '别名': entry['name'],
                '一级模块': entry['level1'], '二级模块': entry['level2'],
                '标准单位': entry['unit'], '数据类型': entry['data_type'], '核心指标': '否',
                '允许期间口径': '本季度末数|本季度数|本季度|期末数', 'STEP5宽表映射': '否',
                '说明': f"{entry['source_file']} {entry['source_sheet']} 第{entry['source_row']}行",
            })
            existing.add(code)
    for code, name in THREE_YEAR_TARGETS.values():
        if code not in existing:
            additions.append({
                '指标编码': code, '指标名称': name, '别名': name,
                '一级模块': '经营指标', '二级模块': '近三年投资收益率',
                '标准单位': '%', '数据类型': '比例', '核心指标': '否',
                '允许期间口径': '本季度', 'STEP5宽表映射': '否',
                '说明': '本报告披露的近三年平均值；不以本季度收益率或单年数据替代，不自行求平均。',
            })
    if additions:
        result = pd.concat([result, pd.DataFrame(additions)], ignore_index=True).fillna('')
    # A component of additional capital must never be treated as its total.
    mask = result['指标编码'].eq('ADDITIONAL_CAPITAL')
    result.loc[mask, '别名'] = result.loc[mask, '别名'].map(
        lambda value: '|'.join(x for x in str(value).split('|') if x != '其他附加资本'))
    return result


def filing_target_details() -> dict[str, dict[str, str]]:
    details: dict[str, dict[str, str]] = {}
    for entry in filing_entries():
        item = details.setdefault(entry['code'], {'目标表ID': '', '来源清单': '', '填报规则': ''})
        for key, value in (('目标表ID', entry['table_id']), ('来源清单', entry['source_file'])):
            parts = item[key].split('|') if item[key] else []
            if value not in parts:
                parts.append(value)
            item[key] = '|'.join(parts)
        item['填报规则'] = (
            '按来源表、指标编码及行次匹配本季度实际披露；认可资产取认可价值；'
            '未匹配保留记录且数值单元格填写未披露；原表数值为横杠或数字0时统一填0并保留原值；明确不适用不填0。'
        )
    for code, _ in THREE_YEAR_TARGETS.values():
        details[code] = {
            '目标表ID': 'THREE_YEAR_INVESTMENT_RETURN',
            '来源清单': '补充目标：近三年平均投资收益率',
            '填报规则': '期间口径为本季度，填本季度报告明确披露的近三年平均值；与普通本季度收益率独立编码；'
                        '不以单年或本季度收益率替代，不自行求平均；无对应披露保留行并标记未披露。',
        }
    return details


TEXT_FILING_CODES = {entry['code'] for entry in filing_entries() if entry['data_type'] == '文本'}
