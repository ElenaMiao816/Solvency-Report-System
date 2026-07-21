from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import dataclass
from typing import Callable

import fitz
import pandas as pd
import pdfplumber
import requests
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .solvency_pdf_locator import PageMatch
from .solvency_table_boundaries import (
    TABLE_ITEM_BOUNDARIES,
    TableBoundaryError,
    boundary_instruction,
    boundary_items,
    enforce_output_boundaries,
    item_hits_in_rows,
    line_has_item,
)
from .solvency_table_extractor import (
    TABLE_EXCLUSIONS,
    TABLE_HEADERS,
    TABLE_SIGNATURES,
    TABLE_START_MARKERS,
    ExtractedTable,
)

GRID_COLUMNS = 180
TEXT_MODE = "大模型网格重构"
VISION_MODE = "图片扫描重试"
TEXT_RETRY_MODE = "网格纠错重试"
TABLE_COMPLETENESS_TERMS = {
    "SOLVENCY_MAIN": (
        "核心偿付能力充足率",
        "综合偿付能力充足率",
        "实际资本",
        "最低资本",
    ),
}

TABLE_MIN_NUMERIC_ROWS = {
    "SOLVENCY_MAIN": 4,
}
SOURCE_ROW_COVERAGE_RATIO = 0.60

AI_TABLE_END_MARKERS = {
    "SOLVENCY_MAIN": ("监管指标名称", "流动性覆盖率", "LCR1"),
    "OPERATING_METRICS": (
        "流动性风险监管指标", "流动性风险监测指标",
        "近三年综合投资收益率", "近三年（综合）投资收益率",
    ),
    "ACTUAL_CAPITAL": ("S03", "认可资产表", "认可负债表", "最低资本表"),
    "THREE_YEAR_INVESTMENT_RETURN": ("S02", "实际资本表", "实际资本明细表", "最低资本表"),
    "MINIMUM_CAPITAL": ("S06", "风险综合评级", "偿付能力风险管理评估", "风险管理能力"),
}


class ExtractionQualityError(RuntimeError):
    pass


@dataclass
class ExtractionLog:
    table_id: str
    table_name: str
    pages: list[int]
    mode: str
    status: str
    message: str

    def to_dict(self) -> dict:
        return {
            "目标表": self.table_name,
            "物理页码": "、".join(map(str, self.pages)),
            "识别方式": self.mode,
            "状态": self.status,
            "说明": self.message,
        }


@dataclass
class AIExtractionBundle:
    tables: list[ExtractedTable]
    logs: list[ExtractionLog]


@dataclass
class PageGrid:
    page_number: int
    grid_text: str
    word_count: int


def _clean(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u3000", " ")).strip()


def _compact(value: str) -> str:
    return re.sub(r"[\s：:（）()、，,。·—\-_/]", "", str(value or ""))


def _group_matches(matches: list[PageMatch]) -> list[tuple[str, str, list[int]]]:
    grouped: dict[str, dict] = {}
    order: list[str] = []
    for match in matches:
        if match.table_id not in grouped:
            grouped[match.table_id] = {"name": match.table_name, "pages": []}
            order.append(match.table_id)
        grouped[match.table_id]["pages"].extend(match.pages)
    return [
        (table_id, grouped[table_id]["name"], sorted(set(grouped[table_id]["pages"])))
        for table_id in order
    ]


def _words_to_grid(page, columns: int = GRID_COLUMNS) -> tuple[str, int]:
    words = page.extract_words(
        x_tolerance=2, y_tolerance=3, keep_blank_chars=False, use_text_flow=False
    )
    if not words:
        return "", 0
    groups: list[dict] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        top = float(word["top"])
        if not groups or abs(top - groups[-1]["top"]) > 3.5:
            groups.append({"top": top, "words": [word]})
        else:
            group = groups[-1]
            group["words"].append(word)
            group["top"] = (group["top"] * (len(group["words"]) - 1) + top) / len(group["words"])

    width = max(float(page.width), 1.0)
    lines: list[str] = []
    for row_number, group in enumerate(groups):
        canvas = [" "] * columns
        for word in sorted(group["words"], key=lambda item: float(item["x0"])):
            text = _clean(word.get("text", ""))
            column = min(columns - 1, max(0, int(float(word["x0"]) / width * (columns - 1))))
            while column < columns and canvas[column] != " ":
                column += 1
            for character in text:
                if column >= columns:
                    break
                canvas[column] = character
                column += 1
        line = "".join(canvas).rstrip()
        if line:
            lines.append(f"{row_number:03d}|{line}")
    return "\n".join(lines), len(words)


def build_pdf_grids(pdf_bytes: bytes, pages: list[int]) -> list[PageGrid]:
    result: list[PageGrid] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number in pages:
            if page_number < 1 or page_number > len(pdf.pages):
                raise ValueError(f"页码 {page_number} 超出PDF范围。")
            grid, word_count = _words_to_grid(pdf.pages[page_number - 1])
            result.append(PageGrid(page_number, grid, word_count))
    return result


def _instructions(table_id: str, table_name: str, pages: list[int]) -> str:
    signatures = "、".join(TABLE_SIGNATURES.get(table_id, ())) or "无固定关键词"
    headers = "、".join(TABLE_HEADERS.get(table_id, ())) or "以原报告为准"
    exclusions = "、".join(TABLE_EXCLUSIONS.get(table_id, ())) or "无"
    completeness = "、".join(TABLE_COMPLETENESS_TERMS.get(table_id, ())) or "以原页面完整行数为准"
    scope_note = ""
    if table_id == "OPERATING_METRICS":
        scope_note = (
            "本目标是组合经营指标。无论原报告合并成一张表，还是拆分为主要经营指标、"
            "效益类指标、规模类指标、品质类指标四张相邻表，都必须全部提取并按原顺序合并。"
            "综合退保率属于品质类指标时必须保留。分类标题可作为无数值分组行保留，"
            "不得因为表名不同而遗漏后三类指标。"
        )
    return f"""
目标表ID：{table_id}
目标表名称：{table_name}
物理页码：{pages}
核心内容关键词：{signatures}
常见表头：{headers}
不得混入的相邻表内容：{exclusions}
完整性重点字段：{completeness}（原页面存在时必须全部保留）
组合范围补充：{scope_note}
项目边界约束：{boundary_instruction(table_id)}

请对输入的多页PDF进行网格化对齐重构：
1. 只提取目标表，不得混入同页其他表或正文。
2. 同一张表跨页时按页码顺序自动拼接，删除续表重复表名和重复表头。
3. 修复合并单元格造成的字段拆分，但不得推测或编造数值。
4. 每行必须和表头列严格对齐；缺失值用空字符串。
5. 保留金额、百分比、负号、括号和单位，不做换算。
6. 一旦遇到不得混入的内容或下一张表标题，立即停止；边界行及其后内容不得写入rows。
7. 只输出JSON对象，不输出Markdown或解释文字。

JSON格式：
{{"table_id":"{table_id}","table_name":"{table_name}",
 "columns":["列1","列2"],"rows":[["数据1","数据2"]],
 "notes":"可选的简短重构说明"}}
""".strip()


def _slice_grid_for_target(table_id: str, grid_text: str) -> str:
    lines = grid_text.splitlines()
    if not lines:
        return grid_text

    compact_lines = [_compact(line.partition("|")[2]) for line in lines]
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id, {})
    start_items = tuple(boundary.get("start_items", ()))
    end_items = tuple(boundary.get("end_items", ()))
    title_terms = tuple(boundary.get("title_terms", ()))

    title_index = next(
        (
            index
            for index, line in enumerate(compact_lines)
            if any(_compact(term) in line for term in title_terms)
        ),
        None,
    )
    search_from = title_index + 1 if title_index is not None else 0

    start_hit = None
    for item in start_items:
        start_hit = next(
            (
                index
                for index, line in enumerate(lines[search_from:], start=search_from)
                if line_has_item(line, item)
            ),
            None,
        )
        if start_hit is not None:
            break

    if start_hit is not None:
        lower_bound = title_index if title_index is not None else 0
        start = max(lower_bound, start_hit - 1)
    else:
        start = title_index if title_index is not None else 0

    end_search_from = start_hit + 1 if start_hit is not None else start
    end_hit = next(
        (
            index
            for index, line in enumerate(
                lines[end_search_from:], start=end_search_from
            )
            if any(line_has_item(line, item) for item in end_items)
        ),
        None,
    )
    if end_hit is not None:
        end = end_hit + 1
    else:
        if start_hit is None and title_index is None:
            for index, line in enumerate(compact_lines):
                if any(_compact(term) in line for term in TABLE_START_MARKERS.get(table_id, ())):
                    start = index
                    break
        end = len(lines)
        for index in range(start + 1, len(lines)):
            if any(_compact(term) in compact_lines[index] for term in AI_TABLE_END_MARKERS.get(table_id, ())):
                end = index
                break
    return "\n".join(lines[start:end])
def _source_completeness_profile(
    table_id: str,
    grids: list[PageGrid] | None,
) -> tuple[int, tuple[str, ...]]:
    """Estimate source rows and the boundary items visibly present in the source."""
    source_lines: list[str] = []
    for grid in grids or []:
        source_lines.extend(_slice_grid_for_target(table_id, grid.grid_text).splitlines())

    source_text = _compact("".join(line.partition("|")[2] for line in source_lines))
    required_terms = [
        term
        for term in TABLE_COMPLETENESS_TERMS.get(table_id, ())
        if _compact(term) in source_text
    ]
    for item in boundary_items(table_id):
        if any(
            line_has_item(line, item, require_value=True)
            for line in source_lines
        ):
            required_terms.append(item)

    numeric_rows = 0
    for line in source_lines:
        content = line.partition("|")[2]
        numeric_cells = re.findall(r"(?<!\d)[(（\-]?\d[\d,，.％%]*[)）]?", content)
        if len(numeric_cells) >= 2 and re.search(r"[\u4e00-\u9fffA-Za-z]", content):
            numeric_rows += 1
        elif table_id == "THREE_YEAR_INVESTMENT_RETURN" and (
            "%" in content or "％" in content
        ) and any(line_has_item(line, item) for item in boundary_items(table_id)):
            numeric_rows += 1
    return numeric_rows, tuple(dict.fromkeys(required_terms))
def _trim_adjacent_rows(table_id: str, rows: list[list[str]]) -> tuple[list[list[str]], str]:
    markers = AI_TABLE_END_MARKERS.get(table_id, ())
    if len(rows) < 2 or not markers:
        return rows, ""
    for index, row in enumerate(rows[1:], start=1):
        row_text = _compact("".join(row))
        hit = next((_compact(term) for term in markers if _compact(term) in row_text), "")
        if hit:
            trimmed = rows[:index]
            if len(trimmed) < 2:
                raise ExtractionQualityError(f"目标表边界出现在首个数据行：{hit}。")
            return trimmed, f"已在相邻表边界“{hit}”前自动截断"
    return rows, ""

def _grid_messages(table_id: str, table_name: str, pages: list[int], grids: list[PageGrid]) -> list[dict]:
    blocks = "\n\n".join(
        f"===== PDF物理第 {item.page_number} 页 =====\n{_slice_grid_for_target(table_id, item.grid_text)}"
        for item in grids
    )
    return [
        {"role": "system", "content": "你是保险公司偿付能力报告表格重构专家。必须依据页面网格恢复表格，不能编造披露数据。"},
        {"role": "user", "content": f"{_instructions(table_id, table_name, pages)}\n\n{blocks}"},
    ]


def _grid_repair_messages(
    table_id: str,
    table_name: str,
    pages: list[int],
    grids: list[PageGrid],
    previous_error: str,
) -> list[dict]:
    messages = _grid_messages(table_id, table_name, pages, grids)
    messages[-1]["content"] += f"""

上一轮重构未通过本地质量检查，失败原因：{previous_error}
请重新检查列对齐并返回完整JSON。特别注意：
- “行次/序号”列只是行编号，不是金额列；不得把它判作数值错位。
- 实际资本明细表应保留“行次、项目、期末数、期初数”等完整列。
- 组合经营指标必须覆盖主要经营指标、效益类、规模类和品质类全部相邻子表。
- 偿付能力充足率指标不得只返回核心/综合充足率两行，必须保留实际资本、最低资本等整张表的所有数据行。
"""
    return messages


def _is_unsupported_vision_error(error: Exception | str) -> bool:
    message = str(error).lower().replace("`", "")
    signals = (
        "unknown variant image_url",
        "expected text",
        "image input is not supported",
        "does not support image",
        "unsupported image",
        "unsupported content type",
    )
    return any(signal in message for signal in signals)

def _render_images(pdf_bytes: bytes, pages: list[int]) -> list[tuple[int, str]]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: list[tuple[int, str]] = []
    try:
        for page_number in pages:
            if page_number < 1 or page_number > len(document):
                raise ValueError(f"页码 {page_number} 超出PDF范围。")
            pixmap = document.load_page(page_number - 1).get_pixmap(
                matrix=fitz.Matrix(1.8, 1.8), alpha=False
            )
            try:
                image = pixmap.tobytes("jpeg", jpg_quality=84)
                mime = "image/jpeg"
            except (TypeError, ValueError):
                image = pixmap.tobytes("png")
                mime = "image/png"
            result.append((page_number, f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"))
    finally:
        document.close()
    return result


def _vision_messages(table_id: str, table_name: str, pages: list[int], images: list[tuple[int, str]]) -> list[dict]:
    content: list[dict] = [{
        "type": "text",
        "text": _instructions(table_id, table_name, pages) + "\n以下图片按物理页码顺序排列，请直接读取图片网格。",
    }]
    for page_number, data_url in images:
        content.extend([
            {"type": "text", "text": f"PDF物理第 {page_number} 页"},
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
        ])
    return [
        {"role": "system", "content": "你是偿付能力报告图片表格识别专家。严格按图片网格恢复跨页表格，不得编造数据。"},
        {"role": "user", "content": content},
    ]


def _completion_url(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    if not value:
        raise ValueError("模型接口地址不能为空。")
    return value if value.endswith("/chat/completions") else f"{value}/chat/completions"


def _response_error(response) -> str:
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict):
            return _clean(error.get("message", ""))
    except Exception:
        pass
    return _clean(getattr(response, "text", ""))[:500]


def _call_model(
    *, api_key: str, base_url: str, model: str, messages: list[dict],
    timeout: int, post_func: Callable | None,
    source_grids: list[PageGrid] | None = None,
) -> str:
    post = post_func or requests.post
    payload = {
        "model": model.strip(), "messages": messages, "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"}
    response = post(_completion_url(base_url), headers=headers, json=payload, timeout=timeout)
    if not getattr(response, "ok", False) and int(getattr(response, "status_code", 0) or 0) in {400, 422}:
        payload.pop("response_format", None)
        response = post(_completion_url(base_url), headers=headers, json=payload, timeout=timeout)
    if not getattr(response, "ok", False):
        raise RuntimeError(f"模型接口调用失败：{_response_error(response) or 'HTTP错误'}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型接口返回结构中没有choices/message/content。") from exc
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)

def _parse_json(content: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ExtractionQualityError("模型没有返回可解析的JSON对象。")
        try:
            payload = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ExtractionQualityError(f"模型JSON解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise ExtractionQualityError("模型返回结果不是JSON对象。")
    for key in ("result", "table", "data"):
        if isinstance(payload.get(key), dict):
            payload = payload[key]
            break
    if isinstance(payload.get("tables"), list) and payload["tables"]:
        payload = payload["tables"][0]
    return payload


def _header_similarity(row: list[str], columns: list[str]) -> float:
    width = max(len(row), len(columns))
    if not width:
        return 0.0
    left = row + [""] * (width - len(row))
    right = columns + [""] * (width - len(columns))
    matches = sum(bool(_compact(a)) and _compact(a) == _compact(b) for a, b in zip(left, right))
    return matches / width


def _coerce_table(payload: dict) -> tuple[list[list[str]], float, str]:
    columns = payload.get("columns") or payload.get("headers") or payload.get("header")
    raw_rows = payload.get("rows") or payload.get("records")
    if not isinstance(raw_rows, list):
        raise ExtractionQualityError("模型结果缺少rows数组。")
    if not isinstance(columns, list) or not columns:
        first_dict = next((row for row in raw_rows if isinstance(row, dict)), None)
        if first_dict is None:
            raise ExtractionQualityError("模型结果缺少columns数组。")
        columns = list(first_dict)
    columns = [_clean(value) or f"列{index + 1}" for index, value in enumerate(columns)]
    width = len(columns)
    normalized: list[list[str]] = []
    ragged = 0
    for raw_row in raw_rows:
        if isinstance(raw_row, dict) and isinstance(raw_row.get("cells"), list):
            cells = raw_row["cells"]
        elif isinstance(raw_row, dict):
            cells = [raw_row.get(column, "") for column in columns]
        elif isinstance(raw_row, list):
            cells = raw_row
        else:
            continue
        cleaned = [_clean(value) for value in cells]
        if not any(cleaned):
            continue
        ragged += len(cleaned) != width
        cleaned = (cleaned + [""] * width)[:width]
        if _header_similarity(cleaned, columns) >= 0.6:
            continue
        if normalized and cleaned == normalized[-1]:
            continue
        normalized.append(cleaned)
    if not normalized:
        raise ExtractionQualityError("模型结果没有有效数据行。")
    return [columns, *normalized], ragged / max(len(normalized), 1), _clean(payload.get("notes", ""))


def _validate(
    table_id: str,
    rows: list[list[str]],
    ragged_ratio: float,
    *,
    expected_source_rows: int = 0,
    source_required_terms: tuple[str, ...] = (),
) -> tuple[float, str]:
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ExtractionQualityError("重构结果少于2行或2列。")
    if ragged_ratio > 0.2:
        raise ExtractionQualityError(f"超过20%的数据行列数与表头不一致（{ragged_ratio:.0%}）。")
    flat = _compact("".join(cell for row in rows for cell in row))
    required = TABLE_SIGNATURES.get(table_id, ())
    required_hits = [term for term in required if _compact(term) in flat]
    minimum = len(required) if table_id == "SOLVENCY_MAIN" else min(1, len(required))
    if required and len(required_hits) < minimum:
        raise ExtractionQualityError(f"缺少目标表核心内容：{'、'.join(required)}。")
    exclusion_hits = [term for term in TABLE_EXCLUSIONS.get(table_id, ()) if _compact(term) in flat]
    if exclusion_hits:
        raise ExtractionQualityError(f"疑似混入相邻表内容：{'、'.join(exclusion_hits)}。")

    completeness_terms = source_required_terms or TABLE_COMPLETENESS_TERMS.get(table_id, ())
    semantic_items = set(boundary_items(table_id))
    exact_hits = set(item_hits_in_rows(rows, semantic_items))
    missing_completeness = [
        term for term in completeness_terms
        if (
            term not in exact_hits
            if term in semantic_items
            else _compact(term) not in flat
        )
    ]
    if missing_completeness:
        raise ExtractionQualityError(
            f"整表提取不完整，缺少原页面项目：{'、'.join(missing_completeness)}。"
        )

    data_rows = rows[1:]
    header_text = _compact("".join(rows[0][:2]))
    row_number_table = any(term in header_text for term in ("行次", "序号", "编号"))
    numeric_rows = 0
    shifted_rows = 0
    for row in data_rows:
        value_start = 2 if row_number_table and len(row) > 2 else 1
        has_number = any(re.search(r"\d", cell) for cell in row[value_start:])
        numeric_rows += bool(has_number)
        if not row_number_table:
            shifted_rows += bool(re.search(r"\d", row[0]) and not has_number)
    configured_minimum = int(TABLE_MIN_NUMERIC_ROWS.get(table_id, 1))
    minimum_numeric_rows = (
        min(configured_minimum, expected_source_rows)
        if expected_source_rows > 0
        else configured_minimum
    )
    if numeric_rows < minimum_numeric_rows:
        raise ExtractionQualityError(
            f"整表提取不完整：仅识别到{numeric_rows}条数值行，至少应有{minimum_numeric_rows}条。"
        )
    if expected_source_rows >= 3:
        coverage_minimum = max(
            minimum_numeric_rows,
            min(expected_source_rows, math.ceil(expected_source_rows * SOURCE_ROW_COVERAGE_RATIO)),
        )
        if numeric_rows < coverage_minimum:
            raise ExtractionQualityError(
                f"整表覆盖不足：原PDF网格约有{expected_source_rows}条数值行，"
                f"模型仅返回{numeric_rows}条，至少应返回{coverage_minimum}条。"
            )
    if len(data_rows) >= 3 and numeric_rows / len(data_rows) < 0.25:
        raise ExtractionQualityError("有效数值行比例过低，疑似列错位。")
    if shifted_rows >= 2 and shifted_rows / len(data_rows) > 0.35:
        raise ExtractionQualityError("数值集中在首列，疑似网格列错位。")

    header_hits = [term for term in TABLE_HEADERS.get(table_id, ()) if _compact(term) in flat]
    score = min(99.0, 65 + len(required_hits) * 8 + len(header_hits) * 3 + min(numeric_rows, 15) * 0.6)
    evidence_parts = [*required_hits, *header_hits]
    if row_number_table:
        evidence_parts.append("首列已识别为行次，不参与数值错位判断")
    evidence = "、".join(dict.fromkeys(evidence_parts))
    return round(score, 2), evidence or "跨页结构与数值列校验通过"


def _reconstruct(
    *, table_id: str, table_name: str, pages: list[int], mode: str,
    messages: list[dict], api_key: str, base_url: str, model: str,
    timeout: int, post_func: Callable | None,
    source_grids: list[PageGrid] | None = None,
) -> ExtractedTable:
    payload = _parse_json(_call_model(
        api_key=api_key, base_url=base_url, model=model, messages=messages,
        timeout=timeout, post_func=post_func,
    ))
    rows, ragged_ratio, notes = _coerce_table(payload)
    rows, adjacent_note = _trim_adjacent_rows(table_id, rows)
    try:
        rows, item_boundary_note = enforce_output_boundaries(
            table_id, rows, require_complete=True
        )
    except TableBoundaryError as exc:
        raise ExtractionQualityError(str(exc)) from exc
    boundary_note = "；".join(
        item for item in (adjacent_note, item_boundary_note) if item
    )
    expected_source_rows, source_required_terms = _source_completeness_profile(
        table_id, source_grids
    )
    score, evidence = _validate(
        table_id,
        rows,
        ragged_ratio,
        expected_source_rows=expected_source_rows,
        source_required_terms=source_required_terms,
    )
    if boundary_note:
        evidence = f"{evidence}；{boundary_note}"
    if notes:
        evidence = f"{evidence}；{notes}"
    return ExtractedTable(
        table_id=table_id, table_name=table_name, page=pages[0], table_index=1,
        rows=rows, strategy=mode, quality_score=score, evidence=evidence,
        source_pages=pages,
    )


def extract_tables_with_llm(
    pdf_bytes: bytes,
    matches: list[PageMatch],
    *,
    api_key: str,
    base_url: str,
    model: str,
    auto_vision_retry: bool = True,
    force_vision: bool = False,
    timeout: int = 180,
    post_func: Callable | None = None,
    progress_callback: Callable[[ExtractionLog], None] | None = None,
) -> AIExtractionBundle:
    if not pdf_bytes:
        raise ValueError("请先上传PDF。")
    if not matches:
        raise ValueError("请先完成STEP1页码定位。")
    if not api_key.strip() or not base_url.strip() or not model.strip():
        raise ValueError("请完整填写模型接口地址、模型名称和API Key。")

    tables: list[ExtractedTable] = []
    logs: list[ExtractionLog] = []

    def add_log(log: ExtractionLog) -> None:
        logs.append(log)
        if progress_callback:
            progress_callback(log)

    for table_id, table_name, pages in _group_matches(matches):
        grids: list[PageGrid] = []
        text_error = ""
        if not force_vision:
            try:
                grids = build_pdf_grids(pdf_bytes, pages)
                if sum(item.word_count for item in grids) < max(12, len(pages) * 6):
                    raise ExtractionQualityError("页面可检索文字不足，判断为扫描件或文字层异常。")
                table = _reconstruct(
                    table_id=table_id, table_name=table_name, pages=pages, mode=TEXT_MODE,
                    messages=_grid_messages(table_id, table_name, pages, grids),
                    source_grids=grids,
                    api_key=api_key, base_url=base_url, model=model,
                    timeout=timeout, post_func=post_func,
                )
                tables.append(table)
                add_log(ExtractionLog(
                    table_id, table_name, pages, TEXT_MODE, "成功",
                    f"已完成多页网格重构和自动拼接，质量评分 {table.quality_score:.1f}。",
                ))
                continue
            except Exception as exc:
                text_error = str(exc)
                add_log(ExtractionLog(
                    table_id, table_name, pages, TEXT_MODE,
                    "触发图片重试" if auto_vision_retry else "失败", text_error,
                ))

        if not (force_vision or auto_vision_retry):
            continue

        try:
            images = _render_images(pdf_bytes, pages)
            table = _reconstruct(
                table_id=table_id, table_name=table_name, pages=pages, mode=VISION_MODE,
                messages=_vision_messages(table_id, table_name, pages, images),
                source_grids=grids,
                api_key=api_key, base_url=base_url, model=model,
                timeout=timeout, post_func=post_func,
            )
            tables.append(table)
            add_log(ExtractionLog(
                table_id, table_name, pages, VISION_MODE, "成功",
                f"图片扫描重试成功，质量评分 {table.quality_score:.1f}。",
            ))
            continue
        except Exception as vision_exc:
            vision_error = str(vision_exc)
            vision_unsupported = _is_unsupported_vision_error(vision_exc)
            if vision_unsupported:
                add_log(ExtractionLog(
                    table_id, table_name, pages, VISION_MODE, "接口不支持图片",
                    "当前模型接口仅接受文本消息，已自动改用加强版网格纠错重试。",
                ))
            else:
                prefix = f"网格模式失败：{text_error}；" if text_error else ""
                add_log(ExtractionLog(
                    table_id, table_name, pages, VISION_MODE, "失败，转网格纠错",
                    f"{prefix}图片扫描重试未通过：{vision_error}",
                ))

            if not grids:
                try:
                    grids = build_pdf_grids(pdf_bytes, pages)
                except Exception:
                    grids = []
            searchable_words = sum(item.word_count for item in grids)
            if searchable_words < max(12, len(pages) * 6):
                add_log(ExtractionLog(
                    table_id, table_name, pages, TEXT_RETRY_MODE, "失败",
                    "PDF没有足够文字层，图片重试也未成功。请改用支持图片的视觉模型，或人工核对页码后重试。",
                ))
                continue

            try:
                repair_reason = "；".join(item for item in (text_error, vision_error) if item)
                table = _reconstruct(
                    table_id=table_id, table_name=table_name, pages=pages, mode=TEXT_RETRY_MODE,
                    messages=_grid_repair_messages(
                        table_id, table_name, pages, grids, repair_reason
                    ),
                    source_grids=grids,
                    api_key=api_key, base_url=base_url, model=model,
                    timeout=timeout, post_func=post_func,
                )
                tables.append(table)
                add_log(ExtractionLog(
                    table_id, table_name, pages, TEXT_RETRY_MODE, "成功",
                    f"加强版网格纠错成功，质量评分 {table.quality_score:.1f}。",
                ))
            except Exception as repair_exc:
                add_log(ExtractionLog(
                    table_id, table_name, pages, TEXT_RETRY_MODE, "失败",
                    f"图片重试和网格纠错均未通过完整性检查：{repair_exc}",
                ))
    return AIExtractionBundle(tables=tables, logs=logs)

def extraction_logs_frame(logs: list[ExtractionLog]) -> pd.DataFrame:
    columns = ["目标表", "物理页码", "识别方式", "状态", "说明"]
    return pd.DataFrame([item.to_dict() for item in logs], columns=columns)


def reconstructed_workbook_bytes(bundle: AIExtractionBundle) -> bytes:
    output = io.BytesIO()
    used: set[str] = {"提取日志"}
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        extraction_logs_frame(bundle.logs).to_excel(writer, sheet_name="提取日志", index=False)
        for table in bundle.tables:
            pages = table.source_pages or [table.page]
            page_label = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
            base = re.sub(r"[\\/*?:\[\]]", "", f"{table.table_name}_P{page_label}")[:31] or "Table"
            name, suffix = base, 1
            while name in used:
                suffix += 1
                name = f"{base[:27]}_{suffix}"
            used.add(name)
            table.to_frame().to_excel(writer, sheet_name=name, index=False, header=False)

        fill = PatternFill("solid", fgColor="00338D")
        font = Font(color="FFFFFF", bold=True)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for cell in sheet[1]:
                cell.fill, cell.font = fill, font
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for index, cells in enumerate(sheet.iter_cols(1, sheet.max_column), start=1):
                length = max((len(str(cell.value or "")) for cell in cells), default=8)
                sheet.column_dimensions[get_column_letter(index)].width = min(max(length + 2, 10), 42)
    return output.getvalue()