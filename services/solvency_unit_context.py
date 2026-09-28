"""Evidence-based monetary-unit recovery for VLM extraction results."""
from decimal import Decimal, InvalidOperation
import re

import fitz


MONEY_UNITS = ('百万元', '亿元', '万元', '千元', '元')
UNIT_PATTERN = r'单位[:：](?:人民币)?(百万元|亿元|万元|千元|元)(?:人民币)?'
RECOGNIZED_TITLE_UNIT_PATTERN = r'(?<!非)认可资产(?:明细)?表[（(](?:单位[:：])?(?:人民币)?(百万元|亿元|万元|千元|元)[）)]'
_CHINESE_DIGITS = '零〇一二三四五六七八九两兩壹贰貳叁參肆伍陆陸柒捌玖拾佰仟萬万億亿'
_NUMBER_WITH_UNIT = re.compile(
    r'(?<![\d.])([-+]?\d[\d,，]*(?:\.\d+)?)\s*'
    r'(百万元|亿元|万元|千元|元)(?:人民币)?'
)


def canonical_unit(value):
    return re.sub(r'人民币|RMB|CNY|\s+', '', str(value or ''), flags=re.I)


def literal_money_unit(value):
    match = re.search(
        rf'(?:\d|[{_CHINESE_DIGITS}])\s*(百万元|亿元|万元|千元|元)'
        r'(?:人民币|整|正)?\s*$',
        str(value or ''),
    )
    return match.group(1) if match else ''


def _plain_decimal(value):
    text = re.sub(r'[\s,，]', '', str(value or ''))
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?', text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def evidence_money_unit(value_raw, evidence_text):
    """Return the one unit explicitly attached to ``value_raw`` in evidence.

    The value match prevents an unrelated amount elsewhere in the evidence from
    donating its unit. Conflicting representations stay unresolved instead of
    silently choosing a scale.
    """
    literal = literal_money_unit(value_raw)
    if literal:
        return literal
    raw_number = _plain_decimal(value_raw)
    if raw_number is None:
        return ''
    evidence = re.sub(r'\s+', '', str(evidence_text or ''))
    units = {
        match.group(2)
        for match in _NUMBER_WITH_UNIT.finditer(evidence)
        if _plain_decimal(match.group(1)) == raw_number
    }
    return next(iter(units)) if len(units) == 1 else ''


def apply_item_unit_evidence(item, require_explicit=False):
    """Reconcile a model unit with the amount's literal source evidence.

    When ``require_explicit`` is true, an unverified model unit is cleared so a
    numeric validation/retry is triggered rather than accepting a guessed scale.
    """
    result = dict(item)
    old_unit = canonical_unit(item.get('unit'))
    verified = evidence_money_unit(item.get('value_raw'), item.get('evidence_text'))
    if verified:
        result['unit'] = verified
        if old_unit != verified:
            note = f'单位依据：证据原文中金额紧邻单位“{verified}”'
            if old_unit:
                note += f'；模型原单位“{old_unit}”校正为“{verified}”'
            result['evidence_text'] = str(item.get('evidence_text') or '') + '；' + note
    elif require_explicit:
        result['unit'] = ''
        result['evidence_text'] = (
            str(item.get('evidence_text') or '')
            + '；金额缺少可核验的原始单位，未采用模型推测单位'
        )
    else:
        result['unit'] = old_unit
    return result


def recognized_unit_contexts(pdf_bytes, pages):
    """Read only explicit in-table unit labels, inheriting within adjacent pages.

    No guessed scale, cross-table ratio, company name or fixed page number.
    Multiple conflicting units or a different table stop inheritance.
    """
    contexts = {}
    previous = None
    previous_page = None
    try:
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            for number in sorted(set(pages)):
                if not 1 <= number <= len(doc):
                    continue
                if previous_page is not None and number != previous_page + 1:
                    previous = None
                text = re.sub(r'\s+', '', doc[number - 1].get_text())
                heading = re.search(r'(?<!非)认可资产(?:明细)?表', text)
                other = re.search(r'非认可资产(?:明细)?表|认可负债(?:明细)?表|实际资本(?:明细)?表|最低资本(?:明细)?表', text)
                if heading:
                    previous = None
                    region = text[heading.end():]
                    stop = re.search(r'非认可资产(?:明细)?表|认可负债(?:明细)?表|实际资本(?:明细)?表|最低资本(?:明细)?表', region)
                    if stop:
                        region = region[:stop.start()]
                elif other:
                    # The last S03 page may contain its continuation first and
                    # the next table later. Retain the inherited unit for that
                    # page only when S03's closing total precedes the new title.
                    # Never carry it into the following page or into S04 rows.
                    prefix = text[:other.start()]
                    if (previous and re.search(r'(?:认可)?资产(?:总)?合计|(?<![负债])合计', prefix)
                            and re.search(r'\d[\d,，]*\.\d+', prefix)):
                        contexts[number] = previous
                    previous = None
                    previous_page = number
                    continue
                else:
                    # Unit labels on a continuation can change scale explicitly.
                    region = text
                labels = list(re.finditer(UNIT_PATTERN, region))
                # A source title such as “S03-认可资产表（元）” is itself an
                # explicit table-unit declaration, even without “单位：”.
                title_unit = (
                    re.match(r'[（(](?:单位[:：])?(?:人民币)?(百万元|亿元|万元|千元|元)[）)]', region)
                    if heading else None
                )
                units = {m.group(1) for m in labels}
                if title_unit:
                    units.add(title_unit.group(1))
                if len(units) == 1:
                    quote = (heading.group() + title_unit.group()) if title_unit else labels[0].group()
                    previous = (next(iter(units)), number, quote)
                elif len(units) > 1:
                    previous = ('', number, '本表出现多个冲突单位，需核验')
                elif re.search(r'单位[:：]', region):
                    previous = ('', number, '本表明确标注了不受支持的单位，需核验')
                if previous:
                    contexts[number] = previous
                previous_page = number
    except (RuntimeError, ValueError):
        return {}
    return contexts


def model_unit_context(response, visible_pages):
    """Accept a separately quoted table header, not a target's expected unit."""
    unit = canonical_unit(response.get('table_unit'))
    page = response.get('unit_page')
    evidence = re.sub(r'\s+', '', str(response.get('unit_evidence', '')))
    labels = {m.group(1) for m in re.finditer(UNIT_PATTERN, evidence)}
    labels.update(m.group(1) for m in re.finditer(RECOGNIZED_TITLE_UNIT_PATTERN, evidence))
    if type(page) is int and page in visible_pages and unit in MONEY_UNITS and labels == {unit}:
        return unit, page, evidence
    return None


def apply_unit_context(item, context, *, require_explicit=False):
    result = dict(item)
    old_unit = str(item.get('unit') or '')
    literal = literal_money_unit(item.get('value_raw'))
    paths = ' '.join(str(item.get(key) or '') for key in ('source_label', 'row_header_path', 'column_header_path'))
    row_units = set(re.findall(r'[（(](?:人民币)?(百万元|亿元|万元|千元|元)[）)]', paths))
    evidence_unit = evidence_money_unit(item.get('value_raw'), item.get('evidence_text'))
    if literal:
        result['unit'] = literal
    elif len(row_units) == 1:
        result['unit'] = next(iter(row_units))
        result['evidence_text'] = str(item.get('evidence_text') or '') + '；单位依据：原始行/列表头明确标注的单位'
    elif len(row_units) > 1:
        result['unit'] = ''
        result['evidence_text'] = str(item.get('evidence_text') or '') + '；行/列表头单位冲突，需人工核验'
    elif evidence_unit:
        result['unit'] = evidence_unit
        result['evidence_text'] = str(item.get('evidence_text') or '') + '；单位依据：原始金额紧邻单位'
    elif context:
        unit, page, evidence = context
        result['unit'] = unit
        note = f'单位依据：物理页{page}同表表头“{evidence}”'
        if canonical_unit(old_unit) != unit:
            note += f'；模型原单位“{old_unit or "空"}”校正为“{unit}”'
        result['evidence_text'] = str(item.get('evidence_text') or '') + '；' + note
    elif require_explicit:
        result['unit'] = ''
        result['evidence_text'] = str(item.get('evidence_text') or '') + '；同表表头及原始行均无可核验单位，未采用模型推测单位'
    else:
        result['unit'] = canonical_unit(old_unit)
    return result
