"""Conservative, deterministic helpers for explicitly disclosed RMB amounts."""

import math
import re
from decimal import Decimal, ROUND_HALF_UP


AMOUNT_UNIT_FACTORS = {'元': 1, '千元': 1000, '万元': 10000, '百万元': 1000000, '亿元': 100000000}
_AMOUNT_LITERAL = re.compile(
    r'(?<![\d.,])(?:[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|'
    r'\((?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?\))(?![\d.,])'
)
_DIGITS = dict(zip('零一二三四五六七八九', range(10)))
_TRANSLATE = str.maketrans('〇壹贰貳叁參肆伍陆陸柒捌玖拾佰仟萬億圓圆兩两',
                           '零一二二三三四五六六七八九十百千万亿元元二二')


def _chinese_section(text):
    """Parse one section below 10,000; reject shorthand/ambiguous digit runs."""
    total, digit, last_unit = 0, None, 10000
    for char in text:
        if char in _DIGITS:
            value = _DIGITS[char]
            if digit not in (None, 0):
                raise ValueError('Adjacent Chinese digits')
            digit = value
        else:
            unit = {'十': 10, '百': 100, '千': 1000}[char]
            if unit >= last_unit or digit == 0:
                raise ValueError('Invalid place order')
            if digit is None and not (unit == 10 and total == 0):
                raise ValueError('Missing coefficient')
            total += (1 if digit is None else digit) * unit
            last_unit, digit = unit, None
    # “一百二” could mean 102 or 120: require an explicit zero or ten.
    if digit and last_unit > 10 and total and not re.search(r'零[一二三四五六七八九]$', text):
        raise ValueError('Ambiguous shorthand')
    return total + (digit or 0)


def _chinese_integer(text):
    total, last_scale, start = 0, 10**12, 0
    for match in re.finditer('[亿万]', text):
        scale = {'亿': 10**8, '万': 10**4}[match.group()]
        section = text[start:match.start()]
        if scale >= last_scale or not section:
            raise ValueError('Invalid section order')
        coefficient = _chinese_section(section)
        if not coefficient:
            raise ValueError('Zero section')
        total += coefficient * scale
        last_scale, start = scale, match.end()
    tail = text[start:]
    return total + (_chinese_section(tail) if tail else 0)


def chinese_money_yuan(value):
    """Return Decimal yuan only for a complete, unambiguous Chinese RMB literal.

    Explicit 元/圆 is required. Prose, ranges, approximate amounts and mixed
    Arabic/Chinese literals are not guessed. Embedded 亿/万 are applied once.
    """
    text = re.sub(r'\s+', '', str(value)).translate(_TRANSLATE)
    text = re.sub(r'^人民币', '', text)
    match = re.fullmatch(r'([零一二三四五六七八九十百千万亿]+)元(?:[整正]|([零一二三四五六七八九])角(?:([零一二三四五六七八九])分)?|零?([一二三四五六七八九])分)?', text)
    if not match:
        return None
    try:
        integer = _chinese_integer(match[1])
        jiao = _DIGITS.get(match[2], 0)
        fen = _DIGITS.get(match[3] or match[4], 0)
        return Decimal(integer) + Decimal(jiao) / 10 + Decimal(fen) / 100
    except (KeyError, ValueError):
        return None


def _evidence_confirms_source_precision(row, number, digits, source_unit):
    """Require the low-precision literal and its unit to be explicit in evidence."""
    evidence = re.sub(r'\s+', '', str(row.get('证据原文', ''))).replace('，', ',')
    evidence_units = set(re.findall(r'百万元|万元|千元|亿元|元', evidence))
    if source_unit not in evidence_units:
        return False
    for match in _AMOUNT_LITERAL.finditer(evidence):
        token = match.group()
        negative = token.startswith('(') and token.endswith(')')
        if negative:
            token = token[1:-1]
        token_digits = len(token.partition('.')[2])
        try:
            token_number = Decimal(token.replace(',', '')) * (-1 if negative else 1)
        except Exception:
            continue
        if token_digits == digits and token_number == number:
            return True
    return False


def _source_decimal(row, standard_unit):
    """Verify source amount and recover its explicit decimal precision."""
    source_unit = str(row.get('单位', '')).strip()
    raw = re.sub(r'\s+', '', str(row.get('原始值', ''))).replace('，', ',')
    raw = re.sub(r'^(?:人民币|[￥¥])', '', raw)
    embedded = next((u for u in sorted(AMOUNT_UNIT_FACTORS, key=len, reverse=True) if raw.endswith(u)), '')
    if embedded:
        if source_unit and source_unit != embedded:
            return None
        source_unit, raw = embedded, raw[:-len(embedded)]
    if source_unit not in AMOUNT_UNIT_FACTORS:
        return None
    if raw.startswith('(') and raw.endswith(')'):
        raw = '-' + raw[1:-1]
    if not re.fullmatch(r'[-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?', raw):
        return None
    digits = len(raw.partition('.')[2])
    source_number = Decimal(raw.replace(',', ''))
    # Integer/one-decimal values can reflect a source table's displayed
    # precision. Accept that coarser precision only when the evidence repeats
    # the exact literal and source unit; this prevents a model-shortened value
    # from silently widening the cross-table tolerance.
    if digits < 2 and not _evidence_confirms_source_precision(
        row, source_number, digits, source_unit,
    ):
        return None
    factor = Decimal(AMOUNT_UNIT_FACTORS[source_unit]) / Decimal(AMOUNT_UNIT_FACTORS[standard_unit])
    exact = source_number * factor
    numeric = float(row['标准数值'])
    if not math.isfinite(numeric) or abs(float(exact) - numeric) > max(2 * math.ulp(numeric), 1e-12):
        return None
    return exact, (Decimal(1).scaleb(-digits) * factor).normalize()


def source_rounding_agrees(left, right, standard_unit):
    """Check a pair at the coarser *verified source* precision, not a relative tolerance.

    Call pairwise so a coarse summary cannot mask two conflicting precise rows.
    Same-precision sources must use the normal strict baseline comparison.
    """
    if standard_unit not in AMOUNT_UNIT_FACTORS:
        return False
    a, b = _source_decimal(left, standard_unit), _source_decimal(right, standard_unit)
    if a is None or b is None or a[1] == b[1]:
        return False
    coarse, fine = (a, b) if a[1] > b[1] else (b, a)
    return fine[0].quantize(coarse[1], rounding=ROUND_HALF_UP) == coarse[0]
