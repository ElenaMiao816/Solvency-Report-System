"""Conservative text-layer recovery for operating-table values.

The visual model remains the primary extractor. This helper only returns an
explicit current-quarter insurance-revenue cell with a verified source unit.
"""
from __future__ import annotations

import re

import fitz


_AMOUNT_UNITS = r"百万元|亿元|万元|千元|元"
_OPERATING_UNIT = re.compile(
    rf"(?:主要|其他)?经营指标(?:表)?[（(](?:单位[:：])?({_AMOUNT_UNITS})"
    r"(?:[,，、/][%％])?[）)]"
)
_OPERATING_TITLE = re.compile(r"(?:主要|其他)经营指标")
_STANDALONE_UNIT = re.compile(rf"单位[:：]({_AMOUNT_UNITS})")
_ROW_UNIT = re.compile(rf"[（(]({_AMOUNT_UNITS})[）)]")
_NUMBER = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?$")
_PERCENT = re.compile(r"[-+]?\d[\d,，]*(?:\.\d+)?[%％]$")


def _compact(value: object) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _current_column(header: list[object]) -> tuple[int, int] | None:
    labels = [i for i, cell in enumerate(header) if _compact(cell) in {"指标名称", "指标", "项目"}]
    current = [
        i for i, cell in enumerate(header)
        if _compact(cell) in {"本季度数", "本季度（末）数", "本报告期数"}
    ]
    if len(labels) == len(current) == 1 and labels[0] < current[0]:
        return labels[0], current[0]
    return None


def _source_unit(doc: fitz.Document, page_number: int, table_top: float, row_label: str) -> tuple[str, str, int]:
    row_unit = _ROW_UNIT.search(row_label)
    if row_unit:
        return row_unit.group(1), row_unit.group(), page_number
    page = doc[page_number - 1]
    before_table = page.get_text(clip=fitz.Rect(0, 0, page.rect.width, table_top))
    for source_page, text in (
        (page_number, before_table),
        (page_number - 1, doc[page_number - 2].get_text() if page_number > 1 else ""),
    ):
        compact = _compact(text)
        titles = list(_OPERATING_TITLE.finditer(compact))
        if not titles:
            continue
        title_start = titles[-1].start()
        evidence = [
            (match.group(1), match.group())
            for match in _OPERATING_UNIT.finditer(compact, title_start)
        ]
        evidence.extend(
            (match.group(1), match.group())
            for match in _STANDALONE_UNIT.finditer(compact, titles[-1].end())
            if match.start() - titles[-1].end() <= 120
        )
        if len({unit for unit, _ in evidence}) > 1:
            return "", "", 0
        if evidence:
            unit, quote = evidence[-1]
            return unit, quote, source_page
    return "", "", 0


def recover_insurance_revenue(pdf_bytes: bytes, selected_pages: list[int]) -> dict[str, object] | None:
    """Read only the source row and current column on selected/next pages.

    Conflicting candidate cells or absent source-unit evidence remain unresolved.
    The annual cumulative column is never used as a substitute.
    """
    if not selected_pages:
        return None
    candidates: list[dict[str, object]] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = sorted({p for number in selected_pages for p in (number, number + 1) if 1 <= p <= len(doc)})
            for number in pages:
                page = doc[number - 1]
                for table in page.find_tables().tables:
                    rows = table.extract()
                    for header_index, header in enumerate(rows[:3]):
                        columns = _current_column(header)
                        if columns is None:
                            continue
                        label_column, value_column = columns
                        for row in rows[header_index + 1:]:
                            if len(row) <= value_column:
                                continue
                            label = _compact(row[label_column])
                            name = re.sub(r"^[（(][一二三四五六七八九十]+[）)]|^\d+[.、]", "", label)
                            name = _ROW_UNIT.sub("", name)
                            if name not in {"保险业务收入", "保险业务收入合计"}:
                                continue
                            raw = _compact(row[value_column])
                            if not _NUMBER.fullmatch(raw):
                                continue
                            unit, unit_quote, unit_page = _source_unit(doc, number, table.bbox[1], label)
                            if not unit:
                                continue
                            period = _compact(header[value_column])
                            candidates.append({
                                "metric_id": "INSURANCE_REVENUE", "status": "found",
                                "value_raw": raw, "unit": unit, "period_label": period,
                                "source_label": label, "row_header_path": [label],
                                "column_header_path": [period], "page": number,
                                "evidence_text": f"{label}｜{period} {raw}；单位依据：物理页{unit_page}“{unit_quote}”",
                                "confidence": 1.0,
                                "recovery_mode": "源PDF文字表格补提",
                            })
                        break
    except (RuntimeError, ValueError, OSError, IndexError, AttributeError, TypeError):
        return None
    distinct = {(item["value_raw"], item["unit"]) for item in candidates}
    return candidates[0] if len(distinct) == 1 else None


def recover_surrender_rate(pdf_bytes: bytes, selected_pages: list[int]) -> dict[str, object] | None:
    """Recover the current-quarter comprehensive surrender rate.

    Some PDFs have a broken Chinese text map even though PyMuPDF still
    reconstructs the table grid and numeric cells.  The regulatory liquidity
    table places comprehensive surrender rate on row 2 and the current-quarter
    value in the first numeric column.  The structural fallback is accepted
    only for a four-column row whose two value cells are both percentages.
    """
    if not selected_pages:
        return None
    candidates: list[dict[str, object]] = []
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            pages = sorted({
                page
                for number in selected_pages
                for page in (number, number + 1)
                if 1 <= page <= len(doc)
            })
            for number in pages:
                for table in doc[number - 1].find_tables().tables:
                    rows = table.extract()
                    if not rows:
                        continue
                    for header_index, header in enumerate(rows[:3]):
                        current_columns = [
                            index for index, cell in enumerate(header)
                            if _compact(cell) in {"本季度数", "本季度（末）数", "本报告期数"}
                        ]
                        label_columns = [
                            index for index, cell in enumerate(header)
                            if _compact(cell) in {"指标名称", "指标", "项目", "流动性风险监测指标名称"}
                        ]
                        explicit_columns = (
                            (label_columns[0], current_columns[0])
                            if len(label_columns) == len(current_columns) == 1
                            and label_columns[0] < current_columns[0]
                            else None
                        )
                        structural_columns = (1, 2) if len(header) == 4 else None
                        columns = explicit_columns or structural_columns
                        if columns is None:
                            continue
                        label_column, value_column = columns
                        for row in rows[header_index + 1:]:
                            if len(row) <= value_column:
                                continue
                            label = _compact(row[label_column])
                            direct_match = "综合退保率" in label
                            structural_match = (
                                structural_columns is not None
                                and len(row) == 4
                                and _compact(row[0]) == "2"
                                and "%" in label.replace("％", "%")
                                and _PERCENT.fullmatch(_compact(row[2])) is not None
                                and _PERCENT.fullmatch(_compact(row[3])) is not None
                            )
                            if not (direct_match or structural_match):
                                continue
                            raw = _compact(row[value_column])
                            if _PERCENT.fullmatch(raw) is None:
                                continue
                            period = _compact(header[value_column]) if explicit_columns else "本季度数"
                            source_label = label if direct_match else "综合退保率（%）（年累计）"
                            candidates.append({
                                "metric_id": "SURRENDER_RATE",
                                "status": "found",
                                "value_raw": raw,
                                "unit": "%",
                                "period_label": period,
                                "source_label": source_label,
                                "row_header_path": [source_label],
                                "column_header_path": [period],
                                "page": number,
                                "evidence_text": f"{source_label}｜{period} {raw}",
                                "confidence": 1.0,
                                "recovery_mode": "源PDF表格本期列补提",
                            })
                        break
    except (RuntimeError, ValueError, OSError, IndexError, AttributeError, TypeError):
        return None
    distinct = {(item["value_raw"], item["page"]) for item in candidates}
    return candidates[0] if len(distinct) == 1 else None
