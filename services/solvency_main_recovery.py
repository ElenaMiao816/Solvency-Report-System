"""Conservative text-layer recovery for required solvency-summary rows.

The visual model remains the primary extractor.  This helper is used only
when one of three required current-period rows is missing.  It accepts an
exact row label and the leftmost value to its right; conflicts or missing
source-unit evidence remain unresolved.
"""
from __future__ import annotations

import re
from collections import defaultdict

import fitz


_TARGET_LABELS = {
    "MINIMUM_CAPITAL": {"最低资本", "最低资本合计"},
    "CORE_SOLVENCY_RATIO": {"核心偿付能力充足率", "核心偿付充足率"},
    "COMBINED_SOLVENCY_RATIO": {"综合偿付能力充足率", "综合偿付充足率"},
}
_RATIO_CODES = {"CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO"}
_AMOUNT_UNITS = r"百万元|亿元|万元|千元|元"
_AMOUNT = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?$")
_PERCENT = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?[%％]$")
_CELL_VALUE = re.compile(r"(?:[-+]?\d[\d,，]*(?:\.\d+)?[%％]?|[-—–－])$")
_UNIT = re.compile(rf"单位[:：]?({_AMOUNT_UNITS})")
_TITLE = re.compile(r"偿付能力(?:充足率)?(?:主要)?指标")
_PERIODS = ("本季度末数", "本季度（末）数", "本季度数", "本报告期数", "期末数")


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _normalized_label(value: object) -> str:
    text = _compact(value)
    text = re.sub(r"^[（(]?\d+(?:\.\d+)?[）).、]?", "", text)
    text = re.sub(r"[（(][%％][）)]$", "", text)
    return text


def _row_words(page: fitz.Page) -> list[list[tuple]]:
    """Cluster words into visual rows without relying on PDF table borders."""
    words = [word for word in page.get_text("words") if len(word) >= 8 and _compact(word[4])]
    rows: list[list[tuple]] = []
    centers: list[float] = []
    for word in sorted(words, key=lambda item: ((item[1] + item[3]) / 2, item[0])):
        center = (word[1] + word[3]) / 2
        index = next((i for i, current in enumerate(centers) if abs(current - center) <= 3.5), None)
        if index is None:
            rows.append([word])
            centers.append(center)
        else:
            rows[index].append(word)
            centers[index] = sum((item[1] + item[3]) / 2 for item in rows[index]) / len(rows[index])
    return [sorted(row, key=lambda item: item[0]) for row in rows]


def _source_context(doc: fitz.Document, pages: list[int]) -> tuple[str, str, int, str]:
    """Return the unit and first current-period header from the main-table title page."""
    for page_number in pages:
        text = doc[page_number - 1].get_text()
        compact = _compact(text)
        title = _TITLE.search(compact)
        if title is None:
            continue
        scope = compact[title.start():title.start() + 1200]
        unit = _UNIT.search(scope)
        period = next((label for label in _PERIODS if label in scope), "")
        if unit:
            return unit.group(1), unit.group(), page_number, period
        return "", "", 0, period
    return "", "", 0, ""


def recover_solvency_main_metrics(
    pdf_bytes: bytes,
    selected_pages: list[int],
    metric_codes: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Recover exact required rows from the selected summary-table pages."""
    wanted = set(metric_codes or _TARGET_LABELS) & set(_TARGET_LABELS)
    if not selected_pages or not wanted:
        return {}
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = sorted({page for page in selected_pages if 1 <= page <= len(doc)})
            source_unit, unit_quote, unit_page, period = _source_context(doc, pages)
            for page_number in pages:
                page = doc[page_number - 1]
                for row in _row_words(page):
                    non_values = [word for word in row if _CELL_VALUE.fullmatch(_compact(word[4])) is None]
                    label = _normalized_label("".join(_compact(word[4]) for word in non_values))
                    metric_id = next(
                        (code for code in wanted if label in _TARGET_LABELS[code]),
                        None,
                    )
                    if metric_id is None or not non_values:
                        continue
                    label_right = max(float(word[2]) for word in non_values)
                    pattern = _PERCENT if metric_id in _RATIO_CODES else _AMOUNT
                    values = [
                        _compact(word[4]) for word in row
                        if float(word[0]) > label_right and pattern.fullmatch(_compact(word[4]))
                    ]
                    if not values:
                        continue
                    raw = values[0]
                    unit = "%" if metric_id in _RATIO_CODES else source_unit
                    if not unit:
                        continue
                    evidence = f"{label}｜{period or '本报告期当前列'} {raw}"
                    if metric_id not in _RATIO_CODES:
                        evidence += f"；单位依据：物理页{unit_page}“{unit_quote}”"
                    candidates[metric_id].append({
                        "metric_id": metric_id,
                        "status": "found",
                        "value_raw": raw,
                        "unit": unit,
                        "period_label": period,
                        "source_label": label,
                        "row_header_path": [label],
                        "column_header_path": [period] if period else [],
                        "page": page_number,
                        "evidence_text": evidence,
                        "confidence": 1.0,
                        "recovery_mode": "源PDF文字行补提",
                    })
    except (RuntimeError, ValueError, OSError, IndexError, AttributeError, TypeError):
        return {}

    recovered: dict[str, dict[str, object]] = {}
    for metric_id, items in candidates.items():
        distinct = {(item["value_raw"], item["unit"]) for item in items}
        if len(distinct) == 1:
            recovered[metric_id] = items[0]
    return recovered
