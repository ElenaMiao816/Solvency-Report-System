"""Shared, explicit-value rules; absent rows and empty cells are never zero."""
import re
import unicodedata


DASH_MARKERS = frozenset({'-', '--', '---', '—', '——', '–', '−', '－', '﹣'})


def is_dash_value(value) -> bool:
    return re.sub(r'\s+', '', str(value if value is not None else '')) in DASH_MARKERS


def is_explicit_zero(value) -> bool:
    text = unicodedata.normalize('NFKC', str(value if value is not None else '')).strip()
    text = text.replace(',', '').replace(' ', '').removesuffix('%')
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    try:
        return float(text) == 0.0
    except (ValueError, TypeError):
        return False


def is_disclosed_zero(value) -> bool:
    return is_dash_value(value) or is_explicit_zero(value)
