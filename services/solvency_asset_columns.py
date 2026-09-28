"""Conservative, source-grid-based visual context for recognised assets.

The VLM still reads all values. Geometry only selects the original label and
current recognised-value pixels; it never supplies or calculates metric values.
Uncertain/scanned/changed layouts retain the original full-page visual path.
"""
import base64
import io
import re

import fitz
from PIL import Image


def _compact(value):
    return re.sub(r'\s+', '', str(value or ''))


def recognized_column_error(path):
    text = _compact(path)
    if re.search(r'非认可价值|账面价值|帐面价值|期初|上季度|上期|上年|预测', text):
        return '认可资产须读取本期认可价值，不能使用账面、非认可或上一期列。'
    if not re.search(r'(?<!非)认可价值', text):
        return '认可资产来源列必须明确标注“认可价值”；缺少列证据时不能接纳该数值。'
    return ''


def _header(table):
    """Require an unambiguous two-level current-period/recognised header."""
    rows = table.extract()
    cells = [(str(value or ''), rect) for row, geometry in zip(rows[:4], table.rows[:4])
             for value, rect in zip(row, geometry.cells) if rect]
    current = [(value, rect) for value, rect in cells
               if re.fullmatch(r'期末(?:数)?|本季度(?:末)?(?:数)?|本期(?:末)?(?:数)?', _compact(value))]
    labels = [rect for value, rect in cells if _compact(value) in {'项目', '资产项目', '资产名称'}]
    matches = []
    for period, group in current:
        for value, rect in cells:
            if (_compact(value) == '认可价值' and group[0] <= rect[0] + 1
                    and rect[2] <= group[2] + 1 and rect[1] >= group[3] - 1):
                matches.append((period, rect, group))
    if len(matches) != 1 or len(labels) != 1:
        return None
    period, selected, group = matches[0]
    label = labels[0]
    if label[2] > selected[0] or selected[3] < label[3] - 1:
        return None
    return dict(period=_compact(period), period_group=group, selected=selected, label_end=label[2],
                left=table.bbox[0], right=table.bbox[2], top=table.bbox[1], bottom=selected[3])


def _aligned(table, header):
    if abs(table.bbox[0] - header['left']) > 1 or abs(table.bbox[2] - header['right']) > 1:
        return False
    # Every row must retain the same label and value boundaries. Do not inherit
    # a column number across a changed grid or guess from similar amounts.
    edges = [header['label_end'], header['selected'][0], header['selected'][2]]
    for row in table.rows:
        boundaries = [edge for cell in row.cells if cell for edge in (cell[0], cell[2])]
        if any(not any(abs(edge - boundary) <= 1 for boundary in boundaries) for edge in edges):
            return False
    return True


def _image(url):
    with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as image:
        return image.convert('RGB')


def _crop(image, rect, size):
    sx, sy = image.width / size[0], image.height / size[1]
    return image.crop((round(rect[0] * sx), round(rect[1] * sy),
                       round(rect[2] * sx), round(rect[3] * sy)))


def build_recognized_views(pdf_bytes, pages, page_cache):
    """Reuse rendered images, attach original headers to aligned continuations.

    Return {physical_page: (image_url, prompt_context)}; no automatic page
    extension, no company/page-number constants, no new rasterisation.
    """
    result = {}
    try:
        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
    except (RuntimeError, ValueError):
        return result
    header = None
    previous = None
    with doc:
        for number in sorted(set(pages)):
            if not 1 <= number <= len(doc):
                continue
            if previous is not None and number != previous + 1:
                header = None
            previous = number
            page = doc[number - 1]
            text = _compact(page.get_text())
            title = re.search(r'(?<!非)认可资产(?:明细)?表', text)
            other = re.search(r'非认可资产(?:明细)?表|认可负债(?:明细)?表|最低资本(?:明细)?表', text)
            other_rects = [
                rect
                for label in ('非认可资产表', '非认可资产明细表', '认可负债表', '认可负债明细表',
                              '最低资本表', '最低资本明细表')
                for rect in page.search_for(label)
            ]
            other_top = min((rect.y0 for rect in other_rects), default=None)
            if other and header is None and not title:
                continue
            try:
                tables = page.find_tables().tables
                if title:
                    candidates = [(table, _header(table)) for table in tables]
                    candidates = [(table, item) for table, item in candidates if item]
                    if other_top is not None:
                        candidates = [
                            (table, item) for table, item in candidates
                            if table.bbox[3] <= other_top + 1
                        ]
                    header = None
                    if len(candidates) == 1:
                        table, header = candidates[0]
                        header.update(page=number, size=(page.rect.width, page.rect.height))
                        group = fitz.Rect(header['period_group'])
                        captions = [rect for rect in page.search_for(header['period']) if group.contains(rect)]
                        if len(captions) == 1 and captions[0].width <= header['selected'][2] - header['selected'][0]:
                            center = (captions[0].x0 + captions[0].x1) / 2
                            half_width = (header['selected'][2] - header['selected'][0]) / 2
                            header['period_clip'] = (center - half_width, group.y0, center + half_width, group.y1)
                elif header:
                    candidates = [table for table in tables if _aligned(table, header)]
                    if other_top is not None:
                        # A physical page can end S03 and start S04. Preserve
                        # only the aligned S03 continuation above the next title.
                        candidates = [
                            table for table in candidates
                            if table.bbox[3] <= other_top + 1
                        ]
                    if len(candidates) != 1 or (page.rect.width, page.rect.height) != header['size']:
                        header = None
                        continue
                    table = candidates[0]
                    if any(re.search(r'账面价值|帐面价值|认可价值', _compact(value))
                           for row in table.extract()[:4] for value in row):
                        # Repeated headers might reorder columns without moving
                        # grid lines. Use the full page rather than the old order.
                        header = None
                        continue
                if not header:
                    continue
                original = _image(page_cache[number])
                first = _image(page_cache[header['page']])
                x0, _, x1, _ = header['selected']
                size = header['size']
                # Keep row labels (including totals and dashes) and exactly the
                # chosen numeric column. All pieces remain original PDF pixels.
                pieces = []
                for image, top, bottom in (
                    (first, header['top'], header['bottom']),
                    (original, header['bottom'] if number == header['page'] else table.bbox[1], table.bbox[3]),
                ):
                    left = _crop(image, (header['left'], top, header['label_end'], bottom), size)
                    right = _crop(image, (x0, top, x1, bottom), size)
                    if not pieces and header.get('period_clip'):
                        # Copy the actual parent caption at its original scale,
                        # rather than silently dropping the current-period label.
                        right.paste(_crop(first, header['period_clip'], size), (0, 0))
                    strip = Image.new('RGB', (left.width + right.width, max(left.height, right.height)), 'white')
                    strip.paste(left, (0, 0))
                    strip.paste(right, (left.width, 0))
                    pieces.append(strip)
                combined = Image.new('RGB', (max(p.width for p in pieces), sum(p.height for p in pieces)), 'white')
                y = 0
                for piece in pieces:
                    combined.paste(piece, (0, y))
                    y += piece.height
                if max(combined.size) > 1800:
                    combined.thumbnail((1800, 1800))
                output = io.BytesIO()
                combined.save(output, format='JPEG', quality=84)
                if len(output.getvalue()) > 750000:
                    continue
                context = (f'本图为物理页{number}按原表网格生成的认可价值专列视图，'
                           f'保留原始行次、项目及“{header["period"]} > 认可价值”一列；'
                           f'表头来自物理页{header["page"]}，只作列定位，不含该页其他行数值。'
                           '只从本图数据行读取金额，不能推算账面价值或复制上下文金额。'
                           '合计行对应RECOGNIZED_ASSETS，再保险资产行对应REINSURANCE_ASSETS；'
                           '只返回本图实际出现的指标。column_header_path必须保留上述期间及认可价值。')
                result[number] = ('data:image/jpeg;base64,' + base64.b64encode(output.getvalue()).decode(), context)
                if other:
                    # Do not let S03 column geometry leak into S04 or later
                    # tables after the recognised-assets continuation ends.
                    header = None
            except (RuntimeError, ValueError, KeyError, IndexError, OSError, AttributeError):
                # Invalid text/grid/cache means original-image VLM fallback.
                header = None
    return result
