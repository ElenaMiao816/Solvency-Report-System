"""Deterministic disambiguation of similarly named source rows (not values)."""
import re
import unicodedata


QUANT_RISK_RULES = {
    'QUANT_RISK_CAPITAL': (
        '仅取量化风险最低资本合计（考虑特征系数后），通常为行1。'
        '禁止取行1*、未考虑/不考虑特征系数、考虑特征系数前或调整前金额；'
        '即使数值相同也不得共用调整前来源。必须保留完整原始行标签及行次。'
    ),
    'QUANT_RISK_CAPITAL_BEFORE_FACTOR': (
        '仅取量化风险最低资本（未考虑特征系数前），通常为行1*。'
        '来源须明确注明未考虑/不考虑特征系数、考虑特征系数前或调整前；'
        '禁止复制普通量化风险最低资本行1。行次仅作辅助，以原文含义为准。'
    ),
}


def _text(value):
    if isinstance(value, (list, tuple)):
        return '>'.join(_text(v) for v in value)
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(value or '')))


def quant_risk_source_error(code, label='', row_path='', evidence=''):
    """Return a retry reason for an explicit mismatch; never infer from equality.

    Labels/row headers take priority over evidence, which may quote nearby rows.
    Row 1* alone can identify the starred row but is not a universal row number.
    """
    if code not in QUANT_RISK_RULES:
        return ''
    source = _text(label) + '>' + _text(row_path)
    if not _text(label) and not _text(row_path):
        source = _text(evidence)
    before = bool(re.search(r'(?:未|不)考虑(?:公司)?特征系数|考虑(?:公司)?特征系数(?:之)?前|调整前', source))
    starred = bool(re.search(r'(?:^|[>(（:：])(?:行次?|第)?1\*(?:行|[)）>、.:：]|$|量化)', source))
    after = bool(re.search(r'(?:已考虑(?:公司)?特征系数|考虑(?:公司)?特征系数(?:之)?后|调整后)', source))
    # An explicit meaning overrides company-specific numbering, but conflicting
    # before/after wording itself must be reread.
    if code == 'QUANT_RISK_CAPITAL' and (before or (starred and not after)):
        return '原始行标签指向未考虑特征系数前（行1*），不能填入QUANT_RISK_CAPITAL；请重读普通量化风险最低资本（通常行1）本期值。'
    if code == 'QUANT_RISK_CAPITAL_BEFORE_FACTOR' and (after or not (before or starred)):
        return 'QUANT_RISK_CAPITAL_BEFORE_FACTOR必须有未考虑特征系数前/调整前或行1*的来源证据；不能使用普通量化风险最低资本行。'
    return ''
