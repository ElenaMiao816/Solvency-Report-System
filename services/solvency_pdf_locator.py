from __future__ import annotations

import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber

from .solvency_table_boundaries import (
    boundary_variants,
    end_item_groups,
    line_has_item,
)


@dataclass(frozen=True)
class PageMatch:
    table_id: str
    table_name: str
    pages: list[int]
    score: float
    evidence: str
    review_required: bool = False
    review_reason: str = ""
    sources: dict[str, list[int]] = field(default_factory=dict)
    strategy_id: str = ""
    table_config: dict = field(default_factory=dict)

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


def _following_standalone_percent(lines: list[str], line_number: int) -> bool:
    next_line = next(
        (
            re.sub(r"\s+", "", candidate)
            for candidate in lines[line_number + 1:line_number + 4]
            if candidate.strip()
        ),
        "",
    )
    return bool(re.fullmatch(
        r"[+\-－−]?\d+(?:\.\d+)?[％%][。.；;]?",
        next_line,
    ))


def _three_year_line_has_explicit_value(line: str, item: str) -> bool:
    """Accept narrative disclosure rows without accepting label-only footnotes."""
    compact_line = _compact(line)
    compact_item = _compact(item)
    if "近三年" not in compact_item:
        return False
    item_position = compact_line.find(compact_item)
    if item_position < 0:
        return False
    remainder = compact_line[item_position + len(compact_item):]
    return bool(re.search(
        r"[+\-－−]?\d[\d,，]*(?:\.\d+)?[％%]",
        remainder,
    ))


def _three_year_page_has_explicit_value(text: str) -> bool:
    """Reject quarterly rows and footnotes that only mention three-year labels."""
    explicit_items = (
        "近三年平均投资收益率",
        "近三年投资收益率",
        "近三年平均综合投资收益率",
        "近三年综合投资收益率",
    )
    lines = str(text or "").splitlines()
    return any(
        (
            line_has_item(line, item, require_value=True)
            or _three_year_line_has_explicit_value(line, item)
            or (
                line_has_item(line, item)
                and _following_standalone_percent(lines, line_number)
            )
        )
        for line_number, line in enumerate(lines)
        for item in explicit_items
    )


def _excluded_detail_page(text: str, table: dict) -> bool:
    """Identify a dedicated non-target detail page without dropping its parent row.

    Some operating-metrics forms keep a placeholder such as “前五大产品的信息”
    inside the required main table, then disclose the product details on a
    separate page with the same title.  The parent page must remain because
    valid metrics continue after the placeholder; only the dedicated detail
    page is excluded.
    """
    excluded_item_hits = _term_hits(text, table.get("exclude_item_terms", []))
    detail_hits = _term_hits(text, table.get("exclude_page_terms", []))
    if (
        not excluded_item_hits
        or len(detail_hits) < int(table.get("exclude_page_min_hits", 2))
    ):
        return False
    boundary_hits = _term_hits(
        text,
        [
            *table.get("start_item_terms", []),
            *table.get("end_item_terms", []),
            *table.get("terminal_terms", []),
        ],
    )
    return not boundary_hits


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
    if _excluded_detail_page(text, table):
        return 0.0, list(table.get("exclude_item_terms", []))
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
    if _excluded_detail_page(text, table):
        return 0.0, list(table.get("exclude_item_terms", []))
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
    variants = tuple(table.get("boundary_variants") or boundary_variants(table_id))
    if not variants:
        return selected

    anchor = selected[0][0] if selected else 1
    max_pages = max(1, int(table.get("max_pages", 1)))
    three_year_title_context = False
    if table_id == "THREE_YEAR_INVESTMENT_RETURN":
        title_terms = tuple(
            dict.fromkeys(
                [
                    *(
                        str(term)
                        for term in table.get("title_terms", ())
                        if str(term).strip()
                    ),
                    *(
                        str(term)
                        for variant in variants
                        for term in variant.get("title_terms", ())
                        if str(term).strip()
                    ),
                ]
            )
        )
        anchor_text = page_texts[anchor - 1] if 1 <= anchor <= len(page_texts) else ""
        compact_anchor_text = _compact(anchor_text)
        three_year_title_context = any(
            _compact(term) in compact_anchor_text for term in title_terms
        )

    def item_is_allowed(page_number: int, item: str) -> bool:
        if table_id != "THREE_YEAR_INVESTMENT_RETURN":
            return True
        if "近三年" in _compact(item):
            return True
        if not three_year_title_context:
            return True
        # Generic labels are only compatibility fallbacks. Once a near-three-year
        # title is found, do not let an earlier quarterly operating-metric pair
        # replace the disclosure. Keep same-page/continuation-page shorthand valid.
        return anchor <= page_number < anchor + max_pages

    def occurrences(items: tuple[str, ...]) -> list[tuple[int, int]]:
        hits: list[tuple[int, int]] = []
        for page_number, text in enumerate(page_texts, start=1):
            lines = (text or "").splitlines()
            for line_number, line in enumerate(lines):
                for item in items:
                    if not item_is_allowed(page_number, item):
                        continue
                    matched = line_has_item(line, item, require_value=True)
                    if (
                        not matched
                        and table_id == "THREE_YEAR_INVESTMENT_RETURN"
                        and _three_year_line_has_explicit_value(line, item)
                    ):
                        matched = True
                    if (
                        not matched
                        and table_id == "THREE_YEAR_INVESTMENT_RETURN"
                        and line_has_item(line, item)
                    ):
                        matched = _following_standalone_percent(
                            lines,
                            line_number,
                        )
                    if matched:
                        hits.append((page_number, line_number))
                        break
        return hits

    def range_has_formal_section_heading(start: int, end: int) -> bool:
        if table_id != "MINIMUM_CAPITAL":
            return False
        configured_titles = tuple(
            _compact(str(term))
            for term in table.get("title_terms", ())
            if str(term).strip()
        )
        numbered_heading = re.compile(
            r"^(?:第)?(?:[一二三四五六七八九十百]+|\d+)"
            r"(?:[、.．]|[）)])(?:S05[-－—]?)?"
            r"最低资本(?:表|明细表)?$"
        )
        for line in (page_texts[start - 1] or "").splitlines():
            compact_line = _compact(line)
            if not compact_line:
                continue
            if numbered_heading.fullmatch(compact_line):
                return True
            if any(compact_line == title for title in configured_titles):
                return True
        return False

    selected_variant: dict | None = None
    selected_start_items: tuple[str, ...] = ()
    selected_end_group: tuple[str, ...] = ()
    start_page = 0
    end_page = 0
    closed_ranges: list[dict] = []
    for variant_index, boundary in enumerate(variants):
        start_items = tuple(boundary.get("start_items", ()))
        end_groups = end_item_groups(table_id, boundary)
        if not start_items or not end_groups:
            continue
        start_hits = occurrences(start_items)
        if not start_hits:
            continue
        for candidate_group_index, candidate_group in enumerate(end_groups):
            for candidate_start_page, candidate_start_line in start_hits:
                eligible = [
                    (candidate_end_page, candidate_end_line)
                    for candidate_end_page, candidate_end_line
                    in occurrences(candidate_group)
                    if (
                        candidate_start_page
                        <= candidate_end_page
                        < candidate_start_page + max_pages
                    )
                    and (
                        candidate_end_page > candidate_start_page
                        or candidate_end_line >= candidate_start_line
                    )
                ]
                if not eligible:
                    continue
                candidate_end_page, candidate_end_line = min(eligible)
                # A formal S05 section must outrank any higher-priority layout
                # variant found in an earlier solvency-summary table. Within
                # the same heading class, keep anchor proximity ahead of range
                # length so a compact summary cannot replace the disclosure.
                closed_ranges.append({
                    "boundary_priority": int(boundary.get("priority", 0)),
                    "formal_section_heading": range_has_formal_section_heading(
                        candidate_start_page,
                        candidate_end_page,
                    ),
                    "anchor_distance": abs(candidate_start_page - anchor),
                    "variant_index": variant_index,
                    "group_index": candidate_group_index,
                    "span": candidate_end_page - candidate_start_page,
                    "start_page": candidate_start_page,
                    "end_page": candidate_end_page,
                    "end_group": candidate_group,
                    "boundary": boundary,
                    "start_items": start_items,
                })

    if not closed_ranges:
        return selected
    best_range = min(
        closed_ranges,
        key=lambda item: (
            -int(item["formal_section_heading"]),
            -item["boundary_priority"],
            item["anchor_distance"],
            item["variant_index"],
            item["group_index"],
            item["span"],
            item["start_page"],
        ),
    )
    selected_variant = best_range["boundary"]
    selected_start_items = best_range["start_items"]
    selected_end_group = best_range["end_group"]
    start_page = best_range["start_page"]
    end_page = best_range["end_page"]

    existing = {page: (score, hits) for page, score, hits in selected}
    expanded: list[tuple[int, float, list[str]]] = []
    for page in range(start_page, end_page + 1):
        score, hits = existing.get(page, (5.0, []))
        page_hits = list(hits)
        if page == start_page:
            page_hits.append(f"起始项目：{selected_start_items[0]}")
        if page == end_page:
            page_hits.append(f"终止项目：{selected_end_group[0]}")
        if start_page < page < end_page:
            page_hits.append("首尾项目之间的连续物理页")
        if page == start_page:
            page_hits.append(
                f"边界版式：{selected_variant.get('name', '标准版式')}"
            )
            if best_range["formal_section_heading"]:
                page_hits.append("正式章节标题")
        expanded.append((page, max(score, 5.0), page_hits))
    return expanded


def _printed_footer_page(text: str) -> int | None:
    """Read a report's printed page number from the last few text lines."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in reversed(lines[-10:]):
        match = re.fullmatch(r"(?:第\s*)?(\d{1,3})(?:\s*页)?", line)
        if match:
            return int(match.group(1))
    return None


def locate_directory_candidates(
    page_texts: list[str],
    tables: list[dict],
    *,
    max_directory_pages: int = 12,
) -> tuple[dict[str, list[int]], str]:
    """Convert table-of-contents printed pages into physical-page candidates.

    The conversion is accepted only when an offset can be measured from page
    footers or from a target heading that also appears in the report body.
    Directory hits are candidates, never final proof of table membership.
    """
    result = {str(item["table_id"]): [] for item in tables}
    toc_pages = [
        index
        for index, text in enumerate(page_texts[:max_directory_pages], start=1)
        if any(term in _compact(text).lower() for term in ("目录", "contents"))
    ]
    if not toc_pages:
        return result, "未检测到可解析目录"

    printed_by_id: dict[str, list[int]] = {key: [] for key in result}
    for physical_page in toc_pages:
        for line in page_texts[physical_page - 1].splitlines():
            compact_line = _compact(line)
            page_numbers = [
                int(value)
                for value in re.findall(r"(?<!\d)(\d{1,3})(?!\d)", line)
            ]
            if not page_numbers:
                continue
            printed_page = page_numbers[-1]
            for table in tables:
                table_id = str(table["table_id"])
                aliases = [
                    str(table.get("table_name", "")),
                    *[str(item) for item in table.get("title_terms", [])],
                ]
                if any(_compact(alias) in compact_line for alias in aliases if alias):
                    printed_by_id[table_id].append(printed_page)

    offset_votes: list[int] = []
    for physical_page, text in enumerate(page_texts, start=1):
        printed_page = _printed_footer_page(text)
        if printed_page is None:
            continue
        offset = physical_page - printed_page
        if 0 <= offset <= 30:
            offset_votes.append(offset)

    # A body title aligned with a directory entry is a second independent way
    # to measure the cover/front-matter offset.
    toc_page_set = set(toc_pages)
    for table in tables:
        table_id = str(table["table_id"])
        aliases = [
            str(table.get("table_name", "")),
            *[str(item) for item in table.get("title_terms", [])],
        ]
        for printed_page in printed_by_id[table_id]:
            for physical_page, text in enumerate(page_texts, start=1):
                if physical_page in toc_page_set:
                    continue
                compact_text = _compact(text)
                if any(_compact(alias) in compact_text for alias in aliases if alias):
                    offset = physical_page - printed_page
                    if 0 <= offset <= 30:
                        offset_votes.append(offset)

    if not offset_votes:
        return result, "检测到目录条目，但无法可靠换算印刷页码与PDF物理页码"

    offset, votes = Counter(offset_votes).most_common(1)[0]
    total_pages = len(page_texts)
    for table_id, printed_pages in printed_by_id.items():
        result[table_id] = sorted({
            printed_page + offset
            for printed_page in printed_pages
            if 1 <= printed_page + offset <= total_pages
        })
    located = sum(bool(pages) for pages in result.values())
    return result, f"目录物理页偏移量={offset}（{votes}条证据），生成{located}类候选"

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
            strategy_id=str(table.get("strategy_id", "")),
            table_config=dict(table),
        ))
    return results


def extract_report_metadata(pdf_bytes: bytes) -> dict:
    page_texts = extract_page_texts(pdf_bytes)
    head = "\n".join(page_texts[:3])
    compact = _compact(head)
    cover_compact = _compact(page_texts[0]) if page_texts else ""

    cover_period_match = re.search(
        r"(20\d{2})年第?([一二三四1234])季度",
        cover_compact,
    )
    split_period_match = re.search(
        r"年第季度(20\d{2})([1-4])",
        cover_compact,
    )
    # Some embedded cover fonts decode the Chinese caption as mojibake while
    # leaving the year and quarter digits readable and adjacent (for example
    # "...20261"). Keep this fallback deliberately limited to the cover tail.
    compact_digit_period_match = re.search(
        r"(?<!\d)(20\d{2})([1-4])$",
        cover_compact,
    )
    report_period_match = (
        cover_period_match
        or split_period_match
        or compact_digit_period_match
    )
    year_match = re.search(r"(20\d{2})年", compact)
    quarter_match = re.search(r"第?([一二三四1234])季度", compact)
    quarter_map = {"一": "Q1", "1": "Q1", "二": "Q2", "2": "Q2", "三": "Q3", "3": "Q3", "四": "Q4", "4": "Q4"}
    company_match = re.search(r"([^\n]{2,40}(?:保险股份有限公司|保险有限公司))", page_texts[0] if page_texts else "")
    date_match = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", compact)

    year = int(report_period_match.group(1)) if report_period_match else (
        int(year_match.group(1)) if year_match else None
    )
    quarter_symbol = (
        report_period_match.group(2)
        if report_period_match
        else (quarter_match.group(1) if quarter_match else "")
    )
    quarter = quarter_map.get(quarter_symbol)
    return {
        "公司": company_match.group(1).strip() if company_match else "",
        "报告年度": year,
        "报告季度": quarter,
        "报告期": f"{year}{quarter}" if year and quarter else "",
        "披露日期": "-".join(date_match.groups()) if date_match else "",
        "页数": len(page_texts),
    }


def report_identity_warning(filename: str, metadata: dict) -> str:
    """Warn when a company-prefixed filename conflicts with the PDF body.

    The check is deliberately conservative: generic filenames are accepted,
    and only a clear company token immediately before the report year is
    compared with the company extracted from the PDF cover.
    """
    company = _compact(str(metadata.get("公司", "")))
    if not company:
        return ""

    stem = Path(filename or "").stem
    compact_stem = _compact(stem)
    compact_stem = re.sub(
        r"^(?:修订版|最终版|正式版|扫描版|披露版|附件\d*)",
        "",
        compact_stem,
    )
    filename_match = re.match(
        r"([\u4e00-\u9fffA-Za-z·]{2,30})(?=20\d{2})",
        compact_stem,
    )
    if not filename_match:
        return ""

    filename_company = filename_match.group(1)
    company_core = re.sub(
        r"(?:保险股份有限公司|保险有限责任公司|保险有限公司|"
        r"股份有限公司|有限责任公司|有限公司)$",
        "",
        company,
    )
    if (
        filename_company in company_core
        or company_core in filename_company
    ):
        return ""

    return (
        f"文件名显示“{filename_company}”，但PDF正文识别为“{company}”。"
        "请确认上传文件是否正确；系统将继续按PDF正文内容定位。"
    )
