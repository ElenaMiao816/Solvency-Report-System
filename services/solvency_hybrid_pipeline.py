from __future__ import annotations

import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import fitz
import pdfplumber
import requests

from .solvency_ai_table_extractor import (
    AIExtractionBundle,
    AI_TABLE_END_MARKERS,
    ExtractionLog,
    ExtractionQualityError,
    extract_tables_with_llm,
    PageGrid,
    TEXT_MODE,
    TEXT_RETRY_MODE,
    VISION_MODE,
    _compact,
    _is_unsupported_vision_error,
    _render_images,
    _source_completeness_profile,
    _validate,
    _slice_grid_for_target,
    build_pdf_grids,
)
from .solvency_table_boundaries import (
    TableBoundaryError,
    boundary_instruction,
    boundary_items,
    enforce_output_boundaries,
    item_hits_in_rows,
)
from .solvency_pdf_locator import (
    PageMatch,
    _numeric_table_lines,
    _term_hits,
    locate_tables,
)
from .solvency_table_extractor import (
    TABLE_EXCLUSIONS,
    TABLE_HEADERS,
    TABLE_SIGNATURES,
    TABLE_START_MARKERS,
    ExtractedTable,
)

PAGE_TEXT_MODE = "逐页文本结构化提取"
PAGE_VISION_MODE = "逐页图像结构化提取"
PAGE_RETRY_MODE = "逐页网格纠错"
CROSS_PAGE_MERGE_MODE = "跨页一致性拼接"
FULL_TABLE_RECONSTRUCTION_MODE = "多页整表结构化重构"


def _completion_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        raise ValueError("模型接口地址不能为空。")
    return value if value.endswith("/chat/completions") else f"{value}/chat/completions"


def _response_error(response) -> str:
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict):
            return str(error.get("message", "")).strip()
    except Exception:
        pass
    return str(getattr(response, "text", "") or "")[:500]


def _call_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict],
    timeout: int,
    post_func: Callable | None = None,
    json_mode: bool = False,
) -> str:
    post = post_func or requests.post
    payload = {
        "model": model.strip(),
        "messages": messages,
        "temperature": 0,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    response = post(
        _completion_url(base_url),
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if (
        json_mode
        and not getattr(response, "ok", False)
        and int(getattr(response, "status_code", 0) or 0) in {400, 422}
    ):
        payload.pop("response_format", None)
        response = post(
            _completion_url(base_url),
            headers=headers,
            json=payload,
            timeout=timeout,
        )
    if not getattr(response, "ok", False):
        raise RuntimeError(f"模型接口调用失败：{_response_error(response) or 'HTTP错误'}")
    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("模型接口返回结构中没有choices/message/content。") from exc
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _parse_json_object(content: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(content).strip(), flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("页码定位模型没有返回JSON对象。")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise RuntimeError("页码定位模型结果不是JSON对象。")
    return value


def _physical_pages(value, total_pages: int) -> list[int]:
    if isinstance(value, int):
        values = [value]
    elif isinstance(value, str):
        values = [int(item) for item in re.findall(r"\d+", value)]
    elif isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, int):
                values.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                values.append(int(item.strip()))
    else:
        values = []
    return sorted({item for item in values if 1 <= item <= total_pages})


def _solvency_radar(
    pdf_bytes: bytes,
    feature_config: dict,
) -> tuple[list[str], dict[str, list[int]], str]:
    tables = feature_config.get("tables", [])
    hints: dict[str, list[int]] = {str(item["table_id"]): [] for item in tables}
    summaries: list[str] = []
    page_texts: list[str] = []
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for index in range(len(document)):
            page = document.load_page(index)
            raw_text = page.get_text("text") or ""
            page_texts.append(raw_text)
            compressed = " ".join(raw_text.replace("\x00", " ").split())
            summaries.append(f"---PDF物理第{index + 1}页---\n{compressed[:1600]}")
            compact_text = re.sub(r"\s+", "", raw_text)
            try:
                has_table = bool(page.find_tables().tables)
            except Exception:
                has_table = False
            numeric_lines = _numeric_table_lines(raw_text)
            continued = any(term in compact_text for term in ("续表", "（续）", "(续)"))
            table_like = has_table or numeric_lines >= 2 or continued
            for table in tables:
                title_hits = _term_hits(raw_text, table.get("title_terms", []))
                content_hits = _term_hits(raw_text, table.get("content_terms", []))
                header_hits = _term_hits(raw_text, table.get("header_terms", []))
                continuation_hits = _term_hits(raw_text, table.get("continuation_terms", []))
                strong_anchor = bool(title_hits) and (
                    len(content_hits) + len(header_hits) >= 1 or numeric_lines >= 2
                )
                content_anchor = len(content_hits) >= 2 and (
                    bool(header_hits) or table_like
                )
                explicit_continuation = continued and table_like and bool(
                    content_hits or header_hits or continuation_hits
                )
                if strong_anchor or content_anchor or explicit_continuation:
                    hints[str(table["table_id"])].append(index + 1)
    finally:
        document.close()
    hint_lines = []
    by_id = {str(item["table_id"]): item for item in tables}
    for table_id, pages in hints.items():
        if pages:
            hint_lines.append(
                f"- {by_id[table_id]['table_name']}：候选物理页码 {sorted(set(pages))}"
            )
    return page_texts, hints, "\n".join(hint_lines) or "- Python雷达未找到强候选页"


def _merge_locator_pages(
    local_pages: list[int],
    ai_pages: list[int],
    radar_pages: list[int],
    max_pages: int,
) -> list[int]:
    pages = sorted(set(local_pages))
    candidates = sorted(set(ai_pages + radar_pages))
    if not pages:
        pages = sorted(set(ai_pages)) or sorted(set(radar_pages))
    changed = True
    while changed and len(pages) < max_pages:
        changed = False
        for page in candidates:
            if page in pages:
                continue
            if not pages or any(abs(page - existing) == 1 for existing in pages):
                pages.append(page)
                pages.sort()
                changed = True
                if len(pages) >= max_pages:
                    break
    if len(pages) > max_pages:
        local_anchor = min(local_pages) if local_pages else min(pages)
        pages = sorted(pages, key=lambda value: (abs(value - local_anchor), value))[:max_pages]
        pages.sort()
    return pages


def locate_tables_hybrid(
    pdf_bytes: bytes,
    feature_config: dict,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 180,
    post_func: Callable | None = None,
) -> tuple[list[PageMatch], str]:
    """Hybrid page localization: semantic inference plus structural table radar."""
    local_matches = locate_tables(pdf_bytes, feature_config)
    local_by_id = {item.table_id: item for item in local_matches}
    page_texts, radar_hints, hint_text = _solvency_radar(pdf_bytes, feature_config)
    tables = feature_config.get("tables", [])
    target_names = [str(item["table_name"]) for item in tables]
    boundary_hints = "\n".join(
        f"- {item['table_name']}：{boundary_instruction(str(item['table_id']))}"
        for item in tables
    )
    scan_text = "\n\n".join(
        f"---PDF物理第{index + 1}页---\n{' '.join(text.split())[:1600]}"
        for index, text in enumerate(page_texts)
    )
    prompt = f"""你是保险公司偿付能力季度报告审阅专家。请定位下列五类目标表的PDF物理页码。

【执行方法】
1. 同时使用页面标题、表头、数据项目和下方Python表格雷达候选页。
2. 页码必须是PDF阅读器显示的物理页码，不是报告印刷页码。
3. 表格跨页时必须返回全部连续页；续页可能没有表名，也可能只有一条数据。
4. 如果某一页顶部仍有上一页表格的一条或数条数据、下方才开始新表，该页仍属于上一张表。
5. “主要经营指标”包括相邻的主要经营指标、效益类指标、规模类指标、品质类指标四部分。
6. “S02-实际资本明细表”也可能写作“实际资本表”；“S05-最低资本表”也可能写作“最低资本表”。
7. 只输出JSON对象，必须包含全部目标表名；找不到时返回空数组。

目标表：
{json.dumps(target_names, ensure_ascii=False)}

【首尾项目边界】
{boundary_hints}

【Python表格雷达候选页】
{hint_text}

【逐页文本摘要】
{scan_text}

输出示例：
{{"偿付能力充足率指标":[16],"主要经营指标":[17,18],"S02-实际资本明细表":[25,26]}}"""
    ai_error = ""
    try:
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
            post_func=post_func,
            json_mode=True,
        )
        ai_result = _parse_json_object(content)
    except Exception as exc:
        ai_result = {}
        ai_error = str(exc)

    total_pages = len(page_texts)
    results: list[PageMatch] = []
    for table in tables:
        table_id = str(table["table_id"])
        table_name = str(table["table_name"])
        local = local_by_id.get(table_id)
        local_pages = list(local.pages) if local else []
        ai_pages = _physical_pages(ai_result.get(table_name, []), total_pages)
        radar_pages = sorted(set(radar_hints.get(table_id, [])))
        boundary_closed = bool(
            local
            and "起始项目：" in str(local.evidence)
            and "终止项目：" in str(local.evidence)
        )
        pages = (
            local_pages
            if boundary_closed
            else _merge_locator_pages(
                local_pages,
                ai_pages,
                radar_pages,
                max(1, int(table.get("max_pages", 1))),
            )
        )
        evidence_parts = []
        if local and local.evidence:
            evidence_parts.append(local.evidence)
        if boundary_closed:
            evidence_parts.append("首尾项目已闭合，页码范围以项目边界为准")
        if ai_pages:
            evidence_parts.append(f"大模型页码推断：{','.join(map(str, ai_pages))}")
        if radar_pages:
            evidence_parts.append(f"Python表格雷达：{','.join(map(str, radar_pages))}")
        if ai_error:
            evidence_parts.append("大模型定位失败，已使用Python雷达兜底")
        results.append(
            PageMatch(
                table_id=table_id,
                table_name=table_name,
                pages=pages,
                score=max(float(local.score) if local else 0.0, 10.0 if ai_pages else 0.0),
                evidence="；".join(evidence_parts) or "未找到明确页码证据",
            )
        )
    if ai_error:
        return results, f"语义页码推断失败，已使用表格结构雷达完成兜底定位：{ai_error}"
    return results, "已完成混合智能定位：页面语义推断 + 表格结构雷达复核。"


def _pipe_prompt(table_id: str, table_name: str, page_number: int, retry_reason: str = "") -> str:
    signatures = "、".join(TABLE_SIGNATURES.get(table_id, ())) or "以原页面为准"
    headers = "、".join(TABLE_HEADERS.get(table_id, ())) or "以原页面为准"
    exclusions = "、".join(TABLE_EXCLUSIONS.get(table_id, ())) or "无"
    operating_note = ""
    if table_id == "OPERATING_METRICS":
        operating_note = (
            "主要经营指标由主要经营指标、效益类指标、规模类指标、品质类指标组成；"
            "本页出现其中任一部分都必须完整提取。若续页只有“营销员脱落率”一行，"
            "该行仍必须提取。不得输出“前五大产品的信息”项目或其产品明细。"
        )
    elif table_id == "THREE_YEAR_INVESTMENT_RETURN":
        operating_note = (
            "该目标表的数据区固定为两类各一行：投资收益率、综合投资收益率。"
            "项目名称可能带有或省略“近三年”“平均”等修饰语，均须按原文保留；"
            "不得混入备注、主要经营指标或前五大产品明细。"
        )

    retry_note = f"上一轮失败原因：{retry_reason}\n请逐行重新核对。" if retry_reason else ""
    unit_note = (
        "\u82e5\u539f\u9875\u8868\u683c\u4e0a\u65b9\u6216\u8868\u5934\u6807\u6709\u7edf\u4e00\u5355\u4f4d\uff0c\u5fc5\u987b\u628a\u5355\u4f4d\u9644\u52a0\u5230\u76f8\u5e94\u6570\u503c\u5217\u8868\u5934\u4e2d\uff0c"
        "\u4f8b\u5982\u201c\u671f\u672b\u6570\uff08\u5143\uff09\u201d\uff1b\u9879\u76ee\u540d\u81ea\u5e26\u5355\u4f4d\u6309\u539f\u6587\u4fdd\u7559\uff0c\u4e0d\u5f97\u628a\u5355\u4f4d\u8bf4\u660e\u4f5c\u4e3a\u6570\u636e\u884c\u3002"
    )
    return f"""你是四大会计师事务所的偿付能力报告数字化审阅专家。
当前任务：只提取PDF物理第{page_number}页中属于【{table_name}】的表格内容。

核心项目：{signatures}
常见表头：{headers}
不得混入：{exclusions}
项目边界：{boundary_instruction(table_id)}
{operating_note}
{retry_note}

{unit_note}
【必须遵守】
1. 这是逐页任务。即使本页是续页、没有表名、只有最后一条数据，也必须提取该条数据。
2. 如果本页下方开始下一张表，只提取边界之前属于目标表的内容。
3. 每个单元格必须使用英文竖线“|”分隔；空单元格也必须用“|”占位。
4. 保留全部科目、行次、数字、百分号、负号、逗号和括号，不得概括、删除或编造。
5. 多级表头按原顺序保留；续页重复表头可以保留，系统会在拼接时去重。
6. 只输出竖线分隔的纯文本，不输出Markdown代码块、说明或JSON。
"""


def _text_page_messages(
    table_id: str,
    table_name: str,
    page_number: int,
    grid: PageGrid,
    retry_reason: str = "",
) -> list[dict]:
    sliced = _slice_grid_for_target(table_id, grid.grid_text)
    prompt = _pipe_prompt(table_id, table_name, page_number, retry_reason)
    prompt += (
        "\n下面每行开头的三位数字和第一个竖线是页面坐标行号，不属于表格单元格；"
        "请忽略坐标行号后再重构。\n\n" + sliced
    )
    return [{"role": "user", "content": prompt}]


def _vision_page_messages(
    table_id: str,
    table_name: str,
    page_number: int,
    image_url: str,
) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": _pipe_prompt(table_id, table_name, page_number)},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
        ],
    }]


def _parse_pipe_rows(content: str) -> list[list[str]]:
    text = re.sub(r"^```(?:csv|text)?\s*|\s*```$", "", str(content).strip(), flags=re.I)
    rows: list[list[str]] = []
    max_width = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"[\s|:\-]+", line) and "-" in line:
            continue
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]
        cells = [re.sub(r"\s+", " ", item).strip() for item in line.split("|")]
        if not any(cells):
            continue
        rows.append(cells)
        max_width = max(max_width, len(cells))
    if not rows:
        raise ExtractionQualityError("模型没有返回可解析的竖线分隔表格。")
    return [row + [""] * (max_width - len(row)) for row in rows]


def _row_text(row: list[str]) -> str:
    return _compact("".join(row))


def _numeric_output_rows(rows: list[list[str]]) -> int:
    return sum(
        any(re.search(r"\d", cell) for cell in row[1:])
        for row in rows
        if len(row) >= 2
    )


def _trim_page_rows(table_id: str, rows: list[list[str]]) -> list[list[str]]:
    markers = AI_TABLE_END_MARKERS.get(table_id, ())
    end = len(rows)
    for index, row in enumerate(rows):
        row_text = _row_text(row)
        if any(_compact(marker) in row_text for marker in markers):
            end = index
            break
    trimmed = rows[:end]
    if not trimmed:
        raise ExtractionQualityError("目标表在相邻表边界之前没有有效行。")
    return trimmed


def _validate_single_page(
    table_id: str,
    rows: list[list[str]],
    grid: PageGrid,
) -> tuple[float, int, int]:
    rows = _trim_page_rows(table_id, rows)
    expected_rows, source_terms = _source_completeness_profile(table_id, [grid])
    semantic_items = set(boundary_items(table_id))
    exact_hits = set(item_hits_in_rows(rows, semantic_items))
    flat = _compact("".join(cell for row in rows for cell in row))
    missing_terms = [
        term for term in source_terms
        if (
            term not in exact_hits
            if term in semantic_items
            else _compact(term) not in flat
        )
    ]
    if missing_terms:
        raise ExtractionQualityError(
            f"本页提取遗漏源PDF项目：{'、'.join(missing_terms)}。"
        )
    actual_rows = _numeric_output_rows(rows)
    if expected_rows > 0:
        minimum = max(1, math.ceil(expected_rows * 0.55))
        if actual_rows < minimum:
            raise ExtractionQualityError(
                f"本页提取不完整：原PDF网格约有{expected_rows}条数值行，"
                f"模型仅返回{actual_rows}条，至少应返回{minimum}条。"
            )
    elif len(rows) < 1:
        raise ExtractionQualityError("本页未提取到有效表格行。")
    score = min(99.0, 70.0 + min(actual_rows, 20) * 1.2)
    return score, expected_rows, actual_rows


def _header_score(table_id: str, row: list[str]) -> int:
    text = _row_text(row)
    return sum(_compact(term) in text for term in TABLE_HEADERS.get(table_id, ()))


def _is_same_header(left: list[str], right: list[str]) -> bool:
    width = max(len(left), len(right))
    if not width:
        return False
    a = left + [""] * (width - len(left))
    b = right + [""] * (width - len(right))
    matches = sum(
        bool(_compact(x)) and _compact(x) == _compact(y)
        for x, y in zip(a, b)
    )
    return matches / width >= 0.5


def _merge_page_rows(
    table_id: str,
    table_name: str,
    pages: list[int],
    page_rows: dict[int, list[list[str]]],
) -> list[list[str]]:
    ordered = [(page, page_rows[page]) for page in pages if page in page_rows]
    if not ordered:
        raise ExtractionQualityError("没有可拼接的逐页结果。")
    header_candidates = [
        (score, page_index, row_index, row)
        for page_index, (_, rows) in enumerate(ordered)
        for row_index, row in enumerate(rows)
        if (score := _header_score(table_id, row)) > 0 and len(row) >= 2
    ]
    if header_candidates:
        _, header_page_index, header_row_index, header = max(
            header_candidates,
            key=lambda item: (item[0], -item[1], -item[2]),
        )
    else:
        header_page_index, header_row_index = 0, -1
        width = max(len(row) for _, rows in ordered for row in rows)
        header = [f"列{index + 1}" for index in range(width)]

    merged_data: list[list[str]] = []
    category_titles = {"效益类指标", "规模类指标", "品质类指标"}
    for page_index, (_, rows) in enumerate(ordered):
        start = header_row_index + 1 if page_index == header_page_index else 0
        for row in rows[start:]:
            if _is_same_header(row, header) or _header_score(table_id, row) >= 2:
                continue
            nonempty = [cell for cell in row if _compact(cell)]
            row_text = _row_text(row)
            if (
                len(nonempty) <= 1
                and _compact(table_name) in row_text
                and not any(_compact(title) in row_text for title in category_titles)
            ):
                continue
            if not nonempty:
                continue
            if merged_data and row == merged_data[-1]:
                continue
            merged_data.append(row)
    if not merged_data:
        raise ExtractionQualityError("逐页结果去重后没有数据行。")
    width = max(len(header), *(len(row) for row in merged_data))
    if len(header) < width:
        header = header + [f"列{index + 1}" for index in range(len(header), width)]
    return [
        header[:width],
        *[(row + [""] * (width - len(row)))[:width] for row in merged_data],
    ]


def _extract_one_page(
    *,
    pdf_bytes: bytes,
    table_id: str,
    table_name: str,
    page_number: int,
    grid: PageGrid,
    api_key: str,
    base_url: str,
    model: str,
    auto_vision_retry: bool,
    force_vision: bool,
    timeout: int,
    post_func: Callable | None,
) -> tuple[list[list[str]] | None, str, float, list[ExtractionLog]]:
    logs: list[ExtractionLog] = []
    text_error = ""
    if not force_vision:
        try:
            if grid.word_count < 6:
                raise ExtractionQualityError("本页可检索文字不足。")
            content = _call_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=_text_page_messages(table_id, table_name, page_number, grid),
                timeout=timeout,
                post_func=post_func,
            )
            rows = _trim_page_rows(table_id, _parse_pipe_rows(content))
            score, expected, actual = _validate_single_page(table_id, rows, grid)
            logs.append(ExtractionLog(
                table_id, table_name, [page_number], PAGE_TEXT_MODE, "成功",
                f"逐页提取完成；源网格约{expected}条数值行，返回{actual}条。",
            ))
            return rows, PAGE_TEXT_MODE, score, logs
        except Exception as exc:
            text_error = str(exc)
            logs.append(ExtractionLog(
                table_id, table_name, [page_number], PAGE_TEXT_MODE,
                "触发图片重试" if auto_vision_retry else "失败", text_error,
            ))
    if not (force_vision or auto_vision_retry):
        return None, "", 0.0, logs

    vision_error = ""
    try:
        images = _render_images(pdf_bytes, [page_number])
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_vision_page_messages(
                table_id, table_name, page_number, images[0][1]
            ),
            timeout=timeout,
            post_func=post_func,
        )
        rows = _trim_page_rows(table_id, _parse_pipe_rows(content))
        score, expected, actual = _validate_single_page(table_id, rows, grid)
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_VISION_MODE, "成功",
            f"逐页图片提取完成；源网格约{expected}条数值行，返回{actual}条。",
        ))
        return rows, PAGE_VISION_MODE, score, logs
    except Exception as exc:
        vision_error = str(exc)
        unsupported = _is_unsupported_vision_error(exc)
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_VISION_MODE,
            "接口不支持图片，转文本纠错" if unsupported else "失败，转文本纠错",
            vision_error,
        ))

    if grid.word_count < 6:
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "失败",
            "本页没有足够文字层，图片模式也未成功。",
        ))
        return None, "", 0.0, logs
    try:
        reason = "；".join(item for item in (text_error, vision_error) if item)
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_text_page_messages(
                table_id, table_name, page_number, grid, reason
            ),
            timeout=timeout,
            post_func=post_func,
        )
        rows = _trim_page_rows(table_id, _parse_pipe_rows(content))
        score, expected, actual = _validate_single_page(table_id, rows, grid)
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "成功",
            f"逐页文本纠错完成；源网格约{expected}条数值行，返回{actual}条。",
        ))
        return rows, PAGE_RETRY_MODE, score, logs
    except Exception as exc:
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "失败", str(exc),
        ))
        return None, "", 0.0, logs


def extract_tables_hybrid(
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
    """Per-page structured extraction followed by deterministic cross-page merge."""
    if not pdf_bytes:
        raise ValueError("请先上传PDF。")
    if not matches:
        raise ValueError("请先完成STEP1页码定位。")
    if not api_key.strip() or not base_url.strip() or not model.strip():
        raise ValueError("请完整填写模型接口地址、模型名称和API Key。")

    logs: list[ExtractionLog] = []
    tables: list[ExtractedTable] = []

    def add_log(log: ExtractionLog) -> None:
        logs.append(log)
        if progress_callback:
            progress_callback(log)
    def run_full_table_reconstruction(match: PageMatch, reason: str) -> bool:
        pages = sorted(set(match.pages))
        add_log(ExtractionLog(
            match.table_id,
            match.table_name,
            pages,
            FULL_TABLE_RECONSTRUCTION_MODE,
            "启动整表重构",
            reason,
        ))
        try:
            fallback = extract_tables_with_llm(
                pdf_bytes,
                [match],
                api_key=api_key,
                base_url=base_url,
                model=model,
                auto_vision_retry=auto_vision_retry,
                force_vision=False,
                timeout=timeout,
                post_func=post_func,
            )
        except Exception as exc:
            add_log(ExtractionLog(
                match.table_id,
                match.table_name,
                pages,
                FULL_TABLE_RECONSTRUCTION_MODE,
                "失败",
                f"整表结构化重构异常：{exc}",
            ))
            return False
        for fallback_log in fallback.logs:
            add_log(fallback_log)
        if not fallback.tables:
            detail = fallback.logs[-1].message if fallback.logs else "未生成可用表格"
            add_log(ExtractionLog(
                match.table_id,
                match.table_name,
                pages,
                FULL_TABLE_RECONSTRUCTION_MODE,
                "失败",
                f"整表结构化重构未通过：{detail}",
            ))
            return False
        table = fallback.tables[0]
        table.strategy = f"{FULL_TABLE_RECONSTRUCTION_MODE}（{table.strategy}）"
        table.evidence = f"逐页处理未通过后完成整表重构；{table.evidence}"
        tables.append(table)
        add_log(ExtractionLog(
            match.table_id,
            match.table_name,
            pages,
            FULL_TABLE_RECONSTRUCTION_MODE,
            "成功",
            f"已从全部物理页联合重构，生成{len(table.rows) - 1}条数据行。",
        ))
        return True

    for match in matches:
        pages = sorted(set(match.pages))
        if not pages:
            continue
        grids = build_pdf_grids(pdf_bytes, pages)
        grid_by_page = {item.page_number: item for item in grids}
        page_rows: dict[int, list[list[str]]] = {}
        page_modes: dict[int, str] = {}
        page_scores: dict[int, float] = {}
        workers = 2 if force_vision else min(5, max(1, len(pages)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    _extract_one_page,
                    pdf_bytes=pdf_bytes,
                    table_id=match.table_id,
                    table_name=match.table_name,
                    page_number=page,
                    grid=grid_by_page[page],
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    auto_vision_retry=auto_vision_retry,
                    force_vision=force_vision,
                    timeout=timeout,
                    post_func=post_func,
                ): page
                for page in pages
            }
            for future in as_completed(future_map):
                page = future_map[future]
                try:
                    rows, mode, score, page_logs = future.result()
                except Exception as exc:
                    rows, mode, score = None, "", 0.0
                    page_logs = [ExtractionLog(
                        match.table_id, match.table_name, [page], PAGE_TEXT_MODE,
                        "失败", f"逐页任务异常：{exc}",
                    )]
                for log in page_logs:
                    add_log(log)
                if rows:
                    page_rows[page] = rows
                    page_modes[page] = mode
                    page_scores[page] = score

        missing_pages = [page for page in pages if page not in page_rows]
        if missing_pages:
            reason = f"逐页提取未完整覆盖物理页{missing_pages}，切换至多页联合重构。"
            if run_full_table_reconstruction(match, reason):
                continue
            add_log(ExtractionLog(
                match.table_id, match.table_name, pages, CROSS_PAGE_MERGE_MODE, "失败",
                f"以下物理页未成功提取，整表重构亦未通过：{missing_pages}",
            ))
            continue
        try:
            merged_rows = _merge_page_rows(
                match.table_id, match.table_name, pages, page_rows
            )
            try:
                merged_rows, boundary_note = enforce_output_boundaries(
                    match.table_id, merged_rows, require_complete=True
                )
            except TableBoundaryError as boundary_exc:
                raise ExtractionQualityError(str(boundary_exc)) from boundary_exc
            expected_rows, source_terms = _source_completeness_profile(match.table_id, grids)
            quality, evidence = _validate(
                match.table_id,
                merged_rows,
                0.0,
                expected_source_rows=expected_rows,
                source_required_terms=source_terms,
            )
            modes = [page_modes[page] for page in pages]
            strategy = CROSS_PAGE_MERGE_MODE + "（" + " / ".join(dict.fromkeys(modes)) + "）"
            tables.append(ExtractedTable(
                table_id=match.table_id,
                table_name=match.table_name,
                page=pages[0],
                table_index=1,
                rows=merged_rows,
                strategy=strategy,
                quality_score=min(quality, sum(page_scores.values()) / len(page_scores)),
                evidence=f"{evidence}；{boundary_note}；逐页提取成功后按物理页码{pages}拼接",
                source_pages=pages,
                unit_records=extract_unit_records(match.table_id, merged_rows, grids),
            ))
            add_log(ExtractionLog(
                match.table_id, match.table_name, pages, CROSS_PAGE_MERGE_MODE, "成功",
                f"{len(pages)}页均提取成功并按页码拼接；合并后{len(merged_rows)-1}条数据行。",
            ))
        except Exception as exc:
            reason = f"逐页拼接结果未通过整表校验，切换至多页联合重构：{exc}"
            if run_full_table_reconstruction(match, reason):
                continue
            add_log(ExtractionLog(
                match.table_id, match.table_name, pages, CROSS_PAGE_MERGE_MODE, "失败",
                f"跨页拼接及整表重构均未通过：{exc}",
            ))
    return AIExtractionBundle(tables=tables, logs=logs)