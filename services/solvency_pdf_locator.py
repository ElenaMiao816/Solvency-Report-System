from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pdfplumber

from .solvency_table_boundaries import (
    TABLE_ITEM_BOUNDARIES,
    end_item_groups,
    text_has_item,
)


@dataclass(frozen=True)
class PageMatch:
    table_id: str
    table_name: str
    pages: list[int]
    score: float
    evidence: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_page_features(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_page_texts(pdf_bytes: bytes) -> list[str]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [
            (page.extract_text(layout=True, x_tolerance=2, y_tolerance=3) or "")
            for page in pdf.pages
        ]


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _term_hits(text: str, terms: Iterable[str]) -> list[str]:
    compact = _compact(text)
    return [term for term in terms if term and _compact(term) in compact]


def _numeric_table_lines(text: str) -> int:
    return sum(
        len(re.findall(r"(?<!\d)[(（\-]?\d[\d,，.％%]*[)）]?", line)) >= 2
        for line in (text or "").splitlines()
    )


def _leading_numeric_table_lines(text: str, limit: int = 14) -> int:
    return sum(
        len(re.findall(r"(?<!\d)[(（\-]?\d[\d,，.％%]*[)）]?", line)) >= 2
        for line in (text or "").splitlines()[:limit]
    )

def _numeric_widths(text: str, *, leading_only: bool = False, limit: int = 14) -> list[int]:
    """Return numeric-cell counts for table-like lines."""
    lines = (text or "").splitlines()
    if leading_only:
        lines = lines[:limit]
    widths: list[int] = []
    for line in lines:
        count = len(re.findall(r"(?<!\d)[(（\-]?\d[\d,，.％%]*[)）]?", line))
        if count >= 2:
            widths.append(count)
    return widths


def _single_row_tail_continuation(
    previous_text: str,
    text: str,
    table: dict,
    boundary_positions: list[int],
) -> bool:
    """Recognize a final continuation page containing only one table row."""
    previous_widths = _numeric_widths(previous_text)
    leading_lines = (text or "").splitlines()[:14]
    compact_text = _compact(text)
    boundary_position = min(boundary_positions) if boundary_positions else len(compact_text)
    leading_widths: list[tuple[int, int]] = []
    for line in leading_lines:
        compact_line = _compact(line)
        position = compact_text.find(compact_line) if compact_line else -1
        if position < 0 or position >= boundary_position:
            break
        if re.search(r"20\d{2}年.*季度", compact_line):
            continue
        count = len(re.findall(r"(?<!\d)[(（\-]?\d[\d,，.％%]*[)）]?", line))
        if count >= 2:
            leading_widths.append((position, count))
    if not previous_widths or len(leading_widths) != 1:
        return False
    first_numeric_position, leading_width = leading_widths[0]
    tolerance = int(table.get("continuation_width_tolerance", 1))
    if not any(abs(leading_width - width) <= tolerance for width in previous_widths[-6:]):
        return False
    return first_numeric_position >= 0 and (
        not boundary_positions or first_numeric_position < min(boundary_positions)
    )

def _anchor_profile(text: str, table: dict) -> tuple[float, list[str]]:
    title_hits = _term_hits(text, table.get("title_terms", table.get("required_any", [])))
    content_hits = _term_hits(text, table.get("content_terms", table.get("optional", [])))
    header_hits = _term_hits(text, table.get("header_terms", []))
    required_all = table.get("required_all", [])
    if required_all and len(_term_hits(text, required_all)) != len(required_all):
        return 0.0, []

    numeric_lines = _numeric_table_lines(text)
    if title_hits:
        minimum_structure = int(table.get("anchor_min_structure_hits", 1))
        if len(content_hits) + len(header_hits) < minimum_structure and numeric_lines < 2:
            return 0.0, []
    else:
        if len(content_hits) < int(table.get("anchor_without_title_content_hits", 2)):
            return 0.0, []
        if len(header_hits) < int(table.get("anchor_without_title_header_hits", 1)):
            return 0.0, []

    score = (
        len(title_hits) * 8.0
        + len(content_hits) * 2.5
        + len(header_hits) * 1.5
        + min(numeric_lines, 8) * 0.6
    )
    return score, [*title_hits, *content_hits, *header_hits]


def _continuation_profile(
    previous_text: str,
    text: str,
    table: dict,
    other_table_titles: Iterable[str],
) -> tuple[float, list[str]]:
    title_hits = _term_hits(text, table.get("title_terms", []))
    content_hits = _term_hits(text, table.get("content_terms", []))
    header_hits = _term_hits(text, table.get("header_terms", []))
    continuation_hits = _term_hits(text, table.get("continuation_terms", []))
    stop_hits = _term_hits(text, table.get("stop_terms", []))
    other_title_hits = _term_hits(text, other_table_titles)
    numeric_lines = _numeric_table_lines(text)
    leading_numeric_lines = _leading_numeric_table_lines(text)

    own_section = str(table.get("section_code", "")).upper().replace("-", "")
    section_codes = {
        re.sub(r"\s+", "", code).upper()
        for code in re.findall(r"S\s*0?\d+", text or "", flags=re.IGNORECASE)
    }
    if own_section:
        section_codes.discard(own_section)

    compact_text = _compact(text)
    boundary_terms = [*stop_hits, *other_title_hits, *sorted(section_codes)]
    boundary_positions = [
        compact_text.find(_compact(term))
        for term in boundary_terms
        if term and compact_text.find(_compact(term)) >= 0
    ]
    single_row_tail = bool(table.get("allow_single_row_tail", True)) and _single_row_tail_continuation(
        previous_text, text, table, boundary_positions
    )
    own_markers = [*title_hits, *content_hits, *header_hits, *continuation_hits]
    own_positions = [
        compact_text.find(_compact(term))
        for term in own_markers
        if term and compact_text.find(_compact(term)) >= 0
    ]
    has_own_content_before_boundary = bool(
        boundary_positions and own_positions and min(own_positions) < min(boundary_positions)
    )
    if (
        boundary_positions
        and not title_hits
        and not has_own_content_before_boundary
        and not single_row_tail
    ):
        return 0.0, boundary_terms

    structural_hits = len(content_hits) + len(header_hits) + len(continuation_hits)
    generic_table_continuation = (
        not boundary_positions
        and leading_numeric_lines >= int(table.get("continuation_generic_numeric_lines", 3))
    )
    if not title_hits:
        if (
            structural_hits < int(table.get("continuation_min_structure_hits", 2))
            and not generic_table_continuation
            and not single_row_tail
        ):
            return 0.0, []
        if (
            numeric_lines < int(table.get("continuation_min_numeric_lines", 2))
            and not single_row_tail
        ):
            return 0.0, []

    score = (
        len(title_hits) * 5.0
        + len(content_hits) * 2.0
        + len(header_hits) * 1.5
        + len(continuation_hits)
        + min(numeric_lines, 8) * 0.8
        + (3.0 if generic_table_continuation else 0.0)
        + (5.0 if single_row_tail else 0.0)
    )
    evidence = [*title_hits, *content_hits, *header_hits, *continuation_hits]
    if single_row_tail:
        evidence.append("页首单行续表（列数与上一页一致）")
    return score, evidence

def _expand_to_item_boundaries(
    page_texts: list[str],
    table: dict,
    selected: list[tuple[int, float, list[str]]],
) -> list[tuple[int, float, list[str]]]:
    """Use first/last business items to deterministically close cross-page ranges."""
    table_id = str(table.get("table_id", ""))
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id, {})
    start_items = tuple(boundary.get("start_items", ()))
    end_groups = end_item_groups(table_id)
    if not start_items or not end_groups:
        return selected

    start_pages = [
        index
        for index, text in enumerate(page_texts, start=1)
        if any(text_has_item(text, item, require_value=True) for item in start_items)
    ]
    if not start_pages:
        return selected

    anchor = selected[0][0] if selected else start_pages[0]
    start_page = min(start_pages, key=lambda page: (abs(page - anchor), page))
    max_pages = max(1, int(table.get("max_pages", 1)))

    selected_end_group: tuple[str, ...] | None = None
    eligible_ends: list[int] = []
    for candidate_group in end_groups:
        end_pages = [
            index
            for index, text in enumerate(page_texts, start=1)
            if any(text_has_item(text, item, require_value=True) for item in candidate_group)
        ]
        eligible = [
            page
            for page in end_pages
            if start_page <= page < start_page + max_pages
        ]
        if eligible:
            selected_end_group = candidate_group
            eligible_ends = eligible
            break
    if not eligible_ends or selected_end_group is None:
        return selected

    end_page = min(eligible_ends)
    if end_page - start_page + 1 > max_pages:
        return selected

    existing = {page: (score, hits) for page, score, hits in selected}
    expanded: list[tuple[int, float, list[str]]] = []
    for page in range(start_page, end_page + 1):
        score, hits = existing.get(page, (5.0, []))
        page_hits = list(hits)
        if page == start_page:
            page_hits.append(f"起始项目：{start_items[0]}")
        if page == end_page:
            page_hits.append(f"终止项目：{selected_end_group[0]}")
        if start_page < page < end_page:
            page_hits.append("首尾项目之间的连续物理页")
        expanded.append((page, max(score, 5.0), page_hits))
    return expanded

def locate_tables(pdf_bytes: bytes, feature_config: dict) -> list[PageMatch]:
    page_texts = extract_page_texts(pdf_bytes)
    results: list[PageMatch] = []
    tables = feature_config.get("tables", [])
    for table in tables:
        anchors: list[tuple[int, float, list[str]]] = []
        for index, text in enumerate(page_texts, start=1):
            score, hits = _anchor_profile(text, table)
            if score >= float(table.get("minimum_score", 3.0)):
                anchors.append((index, score, hits))

        anchors.sort(key=lambda item: (-item[1], item[0]))
        selected: list[tuple[int, float, list[str]]] = anchors[:1]
        max_pages = max(1, int(table.get("max_pages", 1)))
        if table.get("include_continuation") and selected and max_pages > 1:
            other_titles = [
                term
                for other in tables
                if other.get("table_id") != table.get("table_id")
                for term in other.get("title_terms", [])
            ]
            while len(selected) < max_pages:
                previous_page = selected[-1][0]
                terminal_terms = list(table.get("terminal_terms", []))
                if len(selected) > 1:
                    terminal_terms.extend(table.get("stop_terms", []))
                    terminal_terms.extend(other_titles)
                terminal_hits = _term_hits(page_texts[previous_page - 1], terminal_terms)
                if terminal_hits:
                    break
                next_page = previous_page + 1
                if next_page > len(page_texts):
                    break
                continuation_score, continuation_hits = _continuation_profile(
                    page_texts[previous_page - 1],
                    page_texts[next_page - 1],
                    table,
                    other_titles,
                )
                if continuation_score < float(table.get("continuation_min_score", 4.5)):
                    break
                selected.append((next_page, continuation_score, continuation_hits))

        selected = _expand_to_item_boundaries(page_texts, table, selected)
        pages = [item[0] for item in selected]
        best_score = max((item[1] for item in selected), default=0.0)
        evidence = "、".join(dict.fromkeys(hit for item in selected for hit in item[2]))
        results.append(PageMatch(
            table_id=str(table["table_id"]),
            table_name=str(table["table_name"]),
            pages=pages,
            score=round(best_score, 2),
            evidence=evidence,
        ))
    return results


def extract_report_metadata(pdf_bytes: bytes) -> dict:
    page_texts = extract_page_texts(pdf_bytes)
    head = "\n".join(page_texts[:3])
    compact = _compact(head)

    year_match = re.search(r"(20\d{2})年", compact)
    quarter_match = re.search(r"第?([一二三四1234])季度", compact)
    split_period_match = re.search(r"年第季度(20\d{2})([1-4])", compact)
    quarter_map = {"一": "Q1", "1": "Q1", "二": "Q2", "2": "Q2", "三": "Q3", "3": "Q3", "四": "Q4", "4": "Q4"}
    company_match = re.search(r"([^\n]{2,40}(?:保险股份有限公司|保险有限公司))", page_texts[0] if page_texts else "")
    date_match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", compact)

    year = int(year_match.group(1)) if year_match else (
        int(split_period_match.group(1)) if split_period_match else None
    )
    quarter_symbol = quarter_match.group(1) if quarter_match else (split_period_match.group(2) if split_period_match else "")
    quarter = quarter_map.get(quarter_symbol)
    return {
        "公司": company_match.group(1).strip() if company_match else "",
        "报告年度": year,
        "报告季度": quarter,
        "报告期": f"{year}{quarter}" if year and quarter else "",
        "披露日期": "-".join(date_match.groups()) if date_match else "",
        "页数": len(page_texts),
    }
