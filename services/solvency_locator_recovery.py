"""Bounded candidate discovery for visual rechecks; hints never become matches."""
import re
import base64

import fitz

from .solvency_locator_roles import page_role


DETAIL_IDS = {'ACTUAL_CAPITAL', 'RECOGNIZED_ASSETS', 'RECOGNIZED_LIABILITIES', 'MINIMUM_CAPITAL'}
RECOVERY_IDS = DETAIL_IDS | {'SOLVENCY_MAIN'}


def _catalog_title_candidate(line):
    """Keep short headings, including regulator form codes such as S04."""
    if not 2 < len(line) < 45 or re.search(r'[.．·…]{2}|[：:]|[，。；]', line):
        return False
    # Two adjacent digits normally identify a date/value rather than a heading.
    # S02-S05 are official form prefixes, however, and must remain searchable.
    return bool(re.match(r'^S0?\d{1,2}[-—－]', line, re.I)) or not re.search(r'\d{2,}', line)


def render_review_sheets(pdf_bytes, pages):
    """At most two large pages per image, retaining orientation and page labels."""
    sheets = []
    with fitz.open(stream=pdf_bytes, filetype='pdf') as source:
        for start in range(0, len(pages), 2):
            numbers = tuple(pages[start:start + 2])
            heights = [min(1800, 1700 * source[p - 1].rect.height / source[p - 1].rect.width) for p in numbers]
            with fitz.open() as composite:
                sheet = composite.new_page(width=1740, height=sum(heights) + 60 * len(numbers) + 20)
                y = 20
                for number, height in zip(numbers, heights):
                    sheet.draw_rect(fitz.Rect(20, y, 1720, y + 40), fill=(0, .2, .55))
                    sheet.insert_text((34, y + 28), f'PDF PHYSICAL PAGE {number}', fontsize=24, color=(1, 1, 1))
                    sheet.show_pdf_page(fitz.Rect(20, y + 40, 1720, y + 40 + height), source, number - 1)
                    y += height + 60
                pix = sheet.get_pixmap(alpha=False)
                payload = pix.tobytes('jpeg', jpg_quality=78)
                # Keep the request image bounded, including noisy scanned pages.
                while len(payload) > 1_500_000 and pix.width > 900:
                    pix = fitz.Pixmap(pix, int(pix.width * .85), int(pix.height * .85))
                    payload = pix.tobytes('jpeg', jpg_quality=68)
                sheets.append((numbers, 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')))
    return sheets


def text_page_catalog(pdf_bytes):
    """Optional text-layer title hints. Scanned PDFs return an empty catalog."""
    try:
        with fitz.open(stream=pdf_bytes, filetype='pdf') as doc:
            catalog = []
            for index, page in enumerate(doc):
                lines = [re.sub(r'\s+', '', line) for line in page.get_text().splitlines()]
                if any(line in {'目录', '目錄', 'CONTENTS'} for line in lines[:10]):
                    continue
                # Short headings only: never harvest prose mentions or TOC entries.
                titles = [line for line in lines[:35] if _catalog_title_candidate(line)]
                catalog.append({'page': index + 1, 'titles': titles})
            return catalog
    except (RuntimeError, ValueError):
        return []


def bounded_review_pages(table_id, hits, catalog, page_count, limit=6):
    """Propose a small window; every page still requires VLM confirmation."""
    primary = sorted({h['page'] for h in hits if h['role'] in {'primary', 'continuation'}})
    primary_starts = sorted({h['page'] for h in hits if h['role'] == 'primary'})
    title_pages = {}
    for entry in catalog:
        page = entry.get('page')
        if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= page_count:
            continue
        titles = entry.get('titles', [])
        if isinstance(titles, list):
            if any(re.sub(r'\s+', '', str(t)) in {'目录', '目錄', 'CONTENTS'} for t in titles):
                continue
            title_pages.setdefault(page, []).extend(t for t in titles if isinstance(t, str)
                                                   and not re.search(r'[.．·…]{2}', t))
    anchors = sorted(p for p, titles in title_pages.items()
                     if any(page_role(table_id, title, '') == 'primary' for title in titles))
    # SOLVENCY_MAIN often starts near the bottom of one page and continues on
    # the next page without repeating its heading. A confirmed first-page hit
    # is therefore a safe bounded anchor, but never a final page guess: the
    # focused visual pass must still confirm every continuation page.
    if table_id == 'SOLVENCY_MAIN':
        anchors += [page for page in primary_starts if page not in anchors]
    # Also recheck an unrecognized candidate reported by the first visual pass.
    anchors += sorted({h['page'] for h in hits if h['role'] == 'uncertain'} - set(anchors))
    candidates = set()
    for anchor in anchors:
        window = set()
        stop = anchor + (2 if table_id == 'SOLVENCY_MAIN' else 4)
        for page in range(anchor, min(stop, page_count + 1)):
            titles = title_pages.get(page, [])
            if table_id != 'SOLVENCY_MAIN' and page > anchor and any(
                page_role(other, title, '') == 'primary'
                or bool(re.search(r'非认可资产表|认可负债表', title))
                for title in titles for other in DETAIL_IDS - {table_id}
            ):
                break
            window.add(page)
        if not window.issubset(primary):
            candidates.update(window)
        if len(candidates) >= limit:
            return sorted(candidates)[:limit]
    return sorted(candidates)


def operating_section_closed(item, hits, visible_pages):
    """Missing optional groups are not missing pages when a boundary is seen."""
    boundary = item.get('section_end')
    data_pages = [h['page'] for h in hits if h['role'] in {'primary', 'continuation'}]
    if not isinstance(boundary, dict) or boundary.get('confirmed') is not True or not data_pages:
        return False
    page = boundary.get('page')
    title = re.sub(r'\s+', '', str(boundary.get('next_heading', '')))
    return (type(page) is int and page in visible_pages and max(data_pages) <= page <= max(data_pages) + 1
            and bool(str(boundary.get('evidence', '')).strip())
            and bool(re.search(r'近三年|风险管理能力|风险综合评级|重大事项|管理层分析|实际资本|最低资本|外部机构', title)))


def latest_scan_failures(diagnostics):
    """A successful focused review cannot erase an unsuccessful global batch."""
    latest = {}
    for row in diagnostics:
        if row['轮次'] in {'首轮', '超时重试'}:
            latest[row['批次序号']] = row
    return [row for row in latest.values() if row['状态'] != '成功']
