from __future__ import annotations

import base64
import json
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable, Mapping

import fitz
import requests

from .llm_config import model_request_parameters, normalize_model_id
from .llm_http import post_json_with_retry
from .solvency_ai_table_extractor import (
    AIExtractionBundle,
    AI_TABLE_END_MARKERS,
    ExtractionLog,
    ExtractionQualityError,
    extract_tables_with_llm,
    PageGrid,
    SOURCE_ITEM_RECALL_RATIO,
    _cell_has_disclosed_value,
    _compact,
    _item_recall_profile,
    _is_unsupported_vision_error,
    _render_images,
    _source_completeness_profile,
    _source_item_labels,
    _source_section_titles,
    _validate,
    _validate_source_section_order,
    _slice_grid_for_target,
    build_pdf_grids,
    extract_unit_records,
    merge_unit_records,
    recover_unit_records_from_images,
    recover_minimum_capital_terminal_rows_core,
    unit_recovery_needed,
)
from .solvency_table_boundaries import (
    TableBoundaryError,
    boundary_instruction_for_table,
    boundary_items,
    enforce_output_boundaries,
    item_hits_in_rows,
)
from .solvency_pdf_locator import (
    PageMatch,
    _excluded_detail_page,
    _numeric_table_lines,
    _term_hits,
    _three_year_page_has_explicit_value,
    locate_directory_candidates,
    locate_tables,
)
from .solvency_disclosure_normalizer import (
    normalize_three_year_return_rows_core,
)
from .solvency_table_extractor import (
    TABLE_EXCLUSIONS,
    TABLE_HEADERS,
    TABLE_SIGNATURES,
    ExtractedTable,
    UnitRecord,
)
from .table_strategy_handlers import (
    BoundaryRequest,
    CompletenessRequest,
    PostprocessRequest,
    PromptRequest,
)
from .table_strategy_registry import (
    active_table_strategy,
    resolve_table_strategy,
)

PAGE_TEXT_MODE = "逐页文本结构化提取"
PAGE_VISION_MODE = "逐页图像结构化提取"
PAGE_HIGH_RES_VISION_MODE = "逐页3倍高清图像结构化提取"
PAGE_RETRY_MODE = "逐页网格纠错"
CROSS_PAGE_MERGE_MODE = "跨页一致性拼接"
FULL_TABLE_RECONSTRUCTION_MODE = "多页整表结构化重构"
LOCATOR_VISION_BATCH_SIZE = 6
LOW_TEXT_MIN_CHARACTERS = 24
TABLE_EXTRACTION_MAX_WORKERS = 2


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
        "model": normalize_model_id(base_url, model),
        "messages": messages,
    }
    payload.update(model_request_parameters(base_url, model))
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    response = post_json_with_retry(
        post,
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
        response = post_json_with_retry(
            post,
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


def _low_text_page_numbers(
    page_texts: list[str],
    minimum_characters: int = LOW_TEXT_MIN_CHARACTERS,
) -> list[int]:
    """Return physical pages whose searchable text layer is absent or unusable."""
    return [
        index
        for index, text in enumerate(page_texts, start=1)
        if len(_compact(text)) < minimum_characters
    ]


def _render_locator_images(
    pdf_bytes: bytes,
    pages: list[int],
    *,
    zoom: float,
    quality: int,
) -> list[tuple[int, str]]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    result: list[tuple[int, str]] = []
    try:
        for page_number in pages:
            if page_number < 1 or page_number > len(document):
                continue
            pixmap = document.load_page(page_number - 1).get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=False,
            )
            try:
                image = pixmap.tobytes("jpeg", jpg_quality=quality)
                mime = "image/jpeg"
            except (TypeError, ValueError):
                image = pixmap.tobytes("png")
                mime = "image/png"
            encoded = base64.b64encode(image).decode("ascii")
            result.append((page_number, f"data:{mime};base64,{encoded}"))
    finally:
        document.close()
    return result


def _profile_prompt_values(
    profile_context: Mapping[str, object] | None,
) -> tuple[str, str, list[str]]:
    context = dict(profile_context or {})
    profile_name = str(
        context.get("profile_name") or "保险公司偿付能力季度报告"
    ).strip()
    prompt_role = str(
        context.get("prompt_role") or f"{profile_name}审阅专家"
    ).strip()
    raw_instructions = context.get("instructions", [])
    instructions = [
        str(item).strip()
        for item in (
            raw_instructions
            if isinstance(raw_instructions, (list, tuple))
            else [raw_instructions]
        )
        if str(item).strip()
    ]
    return profile_name, prompt_role, instructions


def _profile_instruction_block(
    profile_context: Mapping[str, object] | None,
) -> str:
    _, _, instructions = _profile_prompt_values(profile_context)
    if not instructions:
        return "- 无额外 profile 专用规则，以目标表配置和项目边界为准。"
    return "\n".join(f"- {instruction}" for instruction in instructions)


def _vision_locator_messages(
    tables: list[dict],
    pages: list[int],
    images: list[tuple[int, str]],
    *,
    stage: str,
    candidate_hints: dict[str, list[int]] | None = None,
    profile_context: Mapping[str, object] | None = None,
) -> list[dict]:
    target_spec = [
        {
            "table_id": str(item["table_id"]),
            "table_name": str(item["table_name"]),
            "max_pages": max(1, int(item.get("max_pages", 1))),
            "boundary": boundary_instruction_for_table(item),
        }
        for item in tables
    ]
    hint_text = json.dumps(candidate_hints or {}, ensure_ascii=False)
    profile_name, prompt_role, _ = _profile_prompt_values(profile_context)
    profile_rules = _profile_instruction_block(profile_context)
    prompt = f"""你是{prompt_role}，负责图片页码定位。
当前报告 profile：{profile_name}
当前阶段：{stage}
本次只检查PDF物理页码：{pages}

目标表及边界：
{json.dumps(target_spec, ensure_ascii=False)}

上一阶段候选页（仅供复核，不得盲从）：
{hint_text}

profile 专用规则：
{profile_rules}

要求：
1. 直接阅读随后提供的页面图片，页码以每张图片前标注的PDF物理页码为准。
2. 同时依据标题、表头、首尾数据项目判断；跨页表必须返回本批次中属于该表的全部页。
3. 只能返回本次提供的物理页码。未在本批次出现的目标返回空数组。
4. 只输出JSON对象；键使用table_name，值为物理页码数组，并包含全部目标。
"""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for page_number, image_url in images:
        content.extend([
            {"type": "text", "text": f"PDF物理第 {page_number} 页"},
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                    "detail": "low" if stage.startswith("低分辨率") else "high",
                },
            },
        ])
    return [
        {
            "role": "system",
            "content": (
                f"你负责从{profile_name}页面图片中定位目标披露，"
                "不得根据常见页码猜测。"
            ),
        },
        {"role": "user", "content": content},
    ]


def _vision_directory_messages(
    tables: list[dict],
    pages: list[int],
    images: list[tuple[int, str]],
    *,
    profile_context: Mapping[str, object] | None = None,
) -> list[dict]:
    targets = [
        {
            "table_id": str(item["table_id"]),
            "table_name": str(item["table_name"]),
            "aliases": [
                str(item.get("table_name", "")),
                *[str(term) for term in item.get("title_terms", [])],
            ],
        }
        for item in tables
    ]
    profile_name, _, _ = _profile_prompt_values(profile_context)
    prompt = f"""你负责读取扫描版{profile_name}前部页面中的目录。
本次图片对应PDF物理页码：{pages}

目标披露：
{json.dumps(targets, ensure_ascii=False)}

要求：
1. 先确认图片中确实存在“目录”页，并读取目标披露对应的报告印刷页码。
2. 利用同批图片中可见的正文标题或页脚印刷页码，计算“PDF物理页码-报告印刷页码”的固定偏移。
3. 只有偏移有可靠视觉证据时才换算；无法可靠换算的目标返回空数组。
4. 返回的是换算后的PDF物理页码，可以超出本批图片范围；不得把目录页本身当成目标页。
5. 目录只是候选证据，不要求图片中已经出现目标表正文。
6. 只输出JSON对象；键使用table_name，值为物理页码数组，并包含全部目标。
"""
    content: list[dict] = [{"type": "text", "text": prompt}]
    for page_number, image_url in images:
        content.extend([
            {"type": "text", "text": f"PDF物理第 {page_number} 页"},
            {
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "low"},
            },
        ])
    return [
        {
            "role": "system",
            "content": "你只解析目录及印刷页码到PDF物理页码的可证实换算，不猜测页码。",
        },
        {"role": "user", "content": content},
    ]


def _locate_scan_directory_with_vision(
    pdf_bytes: bytes,
    tables: list[dict],
    low_text_pages: list[int],
    total_pages: int,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    post_func: Callable | None,
    max_front_pages: int = 10,
    profile_context: Mapping[str, object] | None = None,
) -> tuple[dict[str, list[int]], str]:
    """Parse directory candidates when the report front matter is image-only."""
    result = {str(item["table_id"]): [] for item in tables}
    front_pages = [
        page for page in sorted(set(low_text_pages))
        if page <= min(total_pages, max_front_pages)
    ]
    if not front_pages:
        return result, ""
    try:
        images = _render_locator_images(
            pdf_bytes,
            front_pages,
            zoom=0.9,
            quality=72,
        )
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_vision_directory_messages(
                tables,
                front_pages,
                images,
                profile_context=profile_context,
            ),
            timeout=timeout,
            post_func=post_func,
            json_mode=True,
        )
        payload = _parse_json_object(content)
        for table in tables:
            table_id = str(table["table_id"])
            table_name = str(table["table_name"])
            raw_pages = payload.get(table_name, payload.get(table_id, []))
            result[table_id] = _physical_pages(raw_pages, total_pages)
        located = sum(bool(pages) for pages in result.values())
        return result, f"扫描目录视觉解析生成{located}类候选"
    except Exception as exc:
        return result, f"扫描目录视觉解析失败：{exc}"


def _batched(values: list[int], size: int) -> list[list[int]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _run_vision_locator_stage(
    pdf_bytes: bytes,
    tables: list[dict],
    pages: list[int],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    post_func: Callable | None,
    stage: str,
    zoom: float,
    quality: int,
    candidate_hints: dict[str, list[int]] | None = None,
    profile_context: Mapping[str, object] | None = None,
) -> tuple[dict[str, list[int]], list[str]]:
    result = {str(item["table_id"]): [] for item in tables}
    errors: list[str] = []
    for batch in _batched(sorted(set(pages)), LOCATOR_VISION_BATCH_SIZE):
        try:
            images = _render_locator_images(
                pdf_bytes,
                batch,
                zoom=zoom,
                quality=quality,
            )
            content = _call_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=_vision_locator_messages(
                    tables,
                    batch,
                    images,
                    stage=stage,
                    candidate_hints=candidate_hints,
                    profile_context=profile_context,
                ),
                timeout=timeout,
                post_func=post_func,
                json_mode=True,
            )
            payload = _parse_json_object(content)
            for table in tables:
                table_id = str(table["table_id"])
                table_name = str(table["table_name"])
                raw_pages = payload.get(table_name, payload.get(table_id, []))
                located = _physical_pages(raw_pages, max(batch))
                result[table_id].extend(page for page in located if page in batch)
        except Exception as exc:
            errors.append(f"{stage}物理页{batch}失败：{exc}")
            if _is_unsupported_vision_error(exc):
                break
    return {
        table_id: sorted(set(located_pages))
        for table_id, located_pages in result.items()
    }, errors


def _locate_low_text_pages_with_vision(
    pdf_bytes: bytes,
    tables: list[dict],
    low_text_pages: list[int],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    post_func: Callable | None,
    profile_context: Mapping[str, object] | None = None,
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[str]]:
    low_candidates, errors = _run_vision_locator_stage(
        pdf_bytes,
        tables,
        low_text_pages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        post_func=post_func,
        stage="低分辨率候选扫描",
        zoom=0.9,
        quality=72,
        profile_context=profile_context,
    )
    candidate_pages = sorted({
        page
        for pages in low_candidates.values()
        for page in pages
    })
    if not candidate_pages:
        return low_candidates, low_candidates, errors

    low_text_set = set(low_text_pages)
    confirmation_pages = sorted({
        nearby
        for page in candidate_pages
        for nearby in (page - 1, page, page + 1)
        if nearby in low_text_set
    })
    confirmed, confirmation_errors = _run_vision_locator_stage(
        pdf_bytes,
        tables,
        confirmation_pages,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
        post_func=post_func,
        stage="高分辨率候选确认",
        zoom=1.8,
        quality=86,
        candidate_hints=low_candidates,
        profile_context=profile_context,
    )
    needs_high_resolution = bool(confirmation_errors) or any(
        low_candidates.get(table_id) and not confirmed.get(table_id)
        for table_id in low_candidates
    )
    high_resolution: dict[str, list[int]] = {}
    high_resolution_errors: list[str] = []
    if needs_high_resolution:
        high_resolution, high_resolution_errors = _run_vision_locator_stage(
            pdf_bytes,
            tables,
            confirmation_pages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            post_func=post_func,
            stage="3倍高清候选复核",
            zoom=3.0,
            quality=92,
            candidate_hints={
                table_id: confirmed.get(table_id) or low_candidates.get(table_id, [])
                for table_id in low_candidates
            },
            profile_context=profile_context,
        )
    final = {
        table_id: (
            high_resolution.get(table_id)
            or confirmed.get(table_id)
            or low_candidates.get(table_id, [])
        )
        for table_id in low_candidates
    }
    return low_candidates, final, [
        *errors,
        *confirmation_errors,
        *high_resolution_errors,
    ]


def _radar_content_anchor(
    table_id: str,
    content_hits: list[str],
    header_hits: list[str],
    terminal_hits: list[str],
    table_like: bool,
) -> bool:
    if table_id == "OPERATING_METRICS":
        # Narrative sections often mention assets, premiums, or other generic
        # operating terms together. Require a real table header, or the
        # distinctive terminal row on a continuation page.
        return (
            len(content_hits) >= 2
            and (bool(header_hits) or (bool(terminal_hits) and table_like))
        )
    return len(content_hits) >= 2 and (bool(header_hits) or table_like)


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
                table_id = str(table["table_id"])
                if _excluded_detail_page(raw_text, table):
                    continue
                title_hits = _term_hits(raw_text, table.get("title_terms", []))
                content_hits = _term_hits(raw_text, table.get("content_terms", []))
                header_hits = _term_hits(raw_text, table.get("header_terms", []))
                continuation_hits = _term_hits(raw_text, table.get("continuation_terms", []))
                terminal_hits = _term_hits(raw_text, table.get("terminal_terms", []))
                strong_anchor = bool(title_hits) and (
                    len(content_hits) + len(header_hits) >= 1 or numeric_lines >= 2
                )
                content_anchor = _radar_content_anchor(
                    table_id,
                    content_hits,
                    header_hits,
                    terminal_hits,
                    table_like,
                )
                explicit_continuation = continued and table_like and bool(
                    content_hits or header_hits or continuation_hits
                )
                if strong_anchor or content_anchor or explicit_continuation:
                    hints[table_id].append(index + 1)
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
    directory_pages: list[int] | None = None,
) -> list[int]:
    """Merge page evidence without allowing radar-only edge expansion.

    Local boundary analysis and model-confirmed pages are primary evidence.
    The Python radar is deliberately weaker: when primary evidence exists it
    may fill a missing page inside an already confirmed range, but it may not
    extend either edge. It remains a fallback when both primary sources fail.
    """
    directory_candidates = sorted(set(directory_pages or []))
    pages = sorted(set(local_pages))
    ai_candidates = sorted(set(ai_pages))
    if not pages:
        pages = ai_candidates or directory_candidates or sorted(set(radar_pages))

    # Model-confirmed pages may extend a local range when they are contiguous.
    changed = True
    while changed and len(pages) < max_pages:
        changed = False
        for page in ai_candidates:
            if page in pages:
                continue
            if not pages or any(abs(page - existing) == 1 for existing in pages):
                pages.append(page)
                pages.sort()
                changed = True
                if len(pages) >= max_pages:
                    break

    # Radar is candidate generation only. It can repair an internal hole but
    # cannot append a merely adjacent false-positive page.
    if pages and (local_pages or ai_pages):
        lower, upper = min(pages), max(pages)
        for page in sorted(set(radar_pages)):
            if len(pages) >= max_pages:
                break
            if lower < page < upper and page not in pages:
                pages.append(page)
        pages.sort()

    if len(pages) > max_pages:
        local_anchor = min(local_pages) if local_pages else min(pages)
        pages = sorted(pages, key=lambda value: (abs(value - local_anchor), value))[:max_pages]
        pages.sort()
    return pages


def _locator_conflict(
    *,
    local_pages: list[int],
    ai_pages: list[int],
    visual_pages: list[int],
    directory_pages: list[int],
    boundary_closed: bool,
) -> tuple[bool, str]:
    """Flag materially inconsistent locator evidence for human isolation."""
    sources = {
        "首尾项目闭合": sorted(set(local_pages)),
        "大模型语义": sorted(set(ai_pages)),
        "图片确认": sorted(set(visual_pages)),
    }
    nonempty = [(name, pages) for name, pages in sources.items() if pages]
    reasons: list[str] = []
    if boundary_closed and local_pages:
        local_set = set(local_pages)
        for name, pages in nonempty:
            if name == "首尾项目闭合":
                continue
            if set(pages) != local_set:
                reasons.append(
                    f"首尾项目闭合页{sorted(local_set)}与{name}页{pages}不一致"
                )
    elif len(nonempty) >= 2:
        for index, (left_name, left_pages) in enumerate(nonempty):
            for right_name, right_pages in nonempty[index + 1:]:
                left, right = set(left_pages), set(right_pages)
                if left.isdisjoint(right):
                    reasons.append(
                        f"{left_name}页{left_pages}与{right_name}页{right_pages}无交集"
                    )

    primary_pages = sorted({
        page
        for _, pages in nonempty
        for page in pages
    })
    if directory_pages and primary_pages:
        if all(
            min(abs(candidate - primary) for primary in primary_pages) > 1
            for candidate in directory_pages
        ):
            reasons.append(
                f"目录候选页{sorted(set(directory_pages))}与其他证据相距超过1页"
            )
    elif directory_pages and not primary_pages:
        reasons.append(
            f"仅有目录候选页{sorted(set(directory_pages))}，尚无正文证据"
        )
    return bool(reasons), "；".join(reasons)


def _locator_prompt(
    *,
    target_names: list[str],
    boundary_hints: str,
    radar_hints: str,
    directory_hints: Mapping[str, list[int]],
    scan_text: str,
    profile_context: Mapping[str, object] | None = None,
) -> str:
    profile_name, prompt_role, _ = _profile_prompt_values(profile_context)
    profile_rules = _profile_instruction_block(profile_context)
    example = {
        name: [14 + index]
        for index, name in enumerate(target_names[:3])
    }
    return f"""你是{prompt_role}。请定位下列目标表的PDF物理页码。
当前报告 profile：{profile_name}

【通用执行方法】
1. 页面标题、表头、首尾数据项目和明确的续表关系是主要证据；下方Python表格雷达可能误报，只能作为待复核候选，不能单独证明页面属于目标表。
2. 页码必须是PDF阅读器显示的物理页码，不是报告印刷页码。
3. 表格跨页时必须返回全部连续页；续页可能没有表名，也可能只有一条数据。
4. 如果某一页顶部仍有上一页表格的一条或数条数据、下方才开始新表，该页仍属于上一张表。
5. 对“近三年（综合）投资收益率”，必须找到近三年项目对应的真实平均值；主要经营指标中的季度
   “投资收益率/综合投资收益率”以及脚注中仅提及近三年项目名称，都不能作为该目标页。标题和
   平均值可能分居相邻行，此时仍属于一个完整项目。

【profile 专用规则】
{profile_rules}

【输出要求】
1. 输出前逐页自检：每个返回页必须有该目标的标题、表头、边界项目，或与前页表格连续的数据行；仅出现通用词、章节叙述或与候选页相邻都不算证据。
2. 只输出JSON对象，必须包含全部目标表名；找不到时返回空数组。

目标表：
{json.dumps(target_names, ensure_ascii=False)}

【首尾项目边界】
{boundary_hints}

【Python表格雷达候选页】
{radar_hints}

【目录换算候选页（仅作先验，不得单独证明）】
{json.dumps(directory_hints, ensure_ascii=False)}

【逐页文本摘要】
{scan_text}

输出示例：
{json.dumps(example, ensure_ascii=False)}"""


def locate_tables_hybrid(
    pdf_bytes: bytes,
    feature_config: dict,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 180,
    post_func: Callable | None = None,
    profile_context: Mapping[str, object] | None = None,
) -> tuple[list[PageMatch], str]:
    """Locate target disclosures through text radar and two-stage page vision."""
    local_matches = locate_tables(pdf_bytes, feature_config)
    local_by_id = {item.table_id: item for item in local_matches}
    page_texts, radar_hints, hint_text = _solvency_radar(pdf_bytes, feature_config)
    tables = feature_config.get("tables", [])
    directory_hints, directory_message = locate_directory_candidates(
        page_texts,
        tables,
    )
    low_text_pages = _low_text_page_numbers(page_texts)
    if low_text_pages and not any(directory_hints.values()):
        visual_directory_hints, visual_directory_message = (
            _locate_scan_directory_with_vision(
                pdf_bytes,
                tables,
                low_text_pages,
                len(page_texts),
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
                post_func=post_func,
                profile_context=profile_context,
            )
        )
        if any(visual_directory_hints.values()):
            directory_hints = visual_directory_hints
        if visual_directory_message:
            directory_message += f"；{visual_directory_message}"
    low_text_set = set(low_text_pages)
    searchable_pages = [
        page
        for page in range(1, len(page_texts) + 1)
        if page not in low_text_set
    ]
    target_names = [str(item["table_name"]) for item in tables]
    boundary_hints = "\n".join(
        f"- {item['table_name']}：{boundary_instruction_for_table(item)}"
        for item in tables
    )
    scan_text = "\n\n".join(
        f"---PDF物理第{index + 1}页---\n{' '.join(text.split())[:1600]}"
        for index, text in enumerate(page_texts)
        if index + 1 in searchable_pages
    )
    prompt = _locator_prompt(
        target_names=target_names,
        boundary_hints=boundary_hints,
        radar_hints=hint_text,
        directory_hints=directory_hints,
        scan_text=scan_text,
        profile_context=profile_context,
    )
    ai_error = ""
    ai_result: dict = {}
    if searchable_pages:
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
            ai_error = str(exc)

    vision_candidates = {str(item["table_id"]): [] for item in tables}
    vision_pages = {str(item["table_id"]): [] for item in tables}
    vision_errors: list[str] = []
    if low_text_pages:
        vision_candidates, vision_pages, vision_errors = _locate_low_text_pages_with_vision(
            pdf_bytes,
            tables,
            low_text_pages,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout=timeout,
            post_func=post_func,
            profile_context=profile_context,
        )

    total_pages = len(page_texts)
    results: list[PageMatch] = []
    for table in tables:
        table_id = str(table["table_id"])
        table_name = str(table["table_name"])
        local = local_by_id.get(table_id)
        local_pages = list(local.pages) if local else []
        raw_ai_pages = _physical_pages(ai_result.get(table_name, []), total_pages)
        detail_filtered_ai_pages = [
            page
            for page in raw_ai_pages
            if not _excluded_detail_page(page_texts[page - 1], table)
        ]
        semantic_lookalike_pages: list[int] = []
        ai_pages = detail_filtered_ai_pages
        local_boundary_closed = bool(
            local
            and "起始项目：" in str(local.evidence)
            and "终止项目：" in str(local.evidence)
        )
        if (
            table_id == "THREE_YEAR_INVESTMENT_RETURN"
            and local_boundary_closed
        ):
            semantic_lookalike_pages = [
                page
                for page in detail_filtered_ai_pages
                if (
                    page not in local_pages
                    and not _three_year_page_has_explicit_value(
                        page_texts[page - 1]
                    )
                )
            ]
            ai_pages = [
                page
                for page in detail_filtered_ai_pages
                if page not in semantic_lookalike_pages
            ]
        visual_candidates = [
            page
            for page in vision_candidates.get(table_id, [])
            if not _excluded_detail_page(page_texts[page - 1], table)
        ]
        visual_pages = [
            page
            for page in vision_pages.get(table_id, [])
            if not _excluded_detail_page(page_texts[page - 1], table)
        ]
        excluded_detail_pages = sorted(
            set(raw_ai_pages) - set(detail_filtered_ai_pages)
        )
        radar_pages = sorted(set(radar_hints.get(table_id, [])))
        directory_pages = sorted(set(directory_hints.get(table_id, [])))
        boundary_closed = local_boundary_closed
        pages = (
            local_pages
            if boundary_closed
            else _merge_locator_pages(
                local_pages,
                sorted(set(ai_pages + visual_pages)),
                radar_pages,
                max(1, int(table.get("max_pages", 1))),
                directory_pages,
            )
        )
        review_required, review_reason = _locator_conflict(
            local_pages=local_pages,
            ai_pages=ai_pages,
            visual_pages=visual_pages,
            directory_pages=directory_pages,
            boundary_closed=boundary_closed,
        )
        evidence_parts = []
        if local and local.evidence:
            evidence_parts.append(local.evidence)
        if boundary_closed:
            evidence_parts.append("首尾项目已闭合，页码范围以项目边界为准")
        if ai_pages:
            evidence_parts.append(f"大模型页码推断：{','.join(map(str, ai_pages))}")
        if excluded_detail_pages:
            evidence_parts.append(
                "已隔离非目标明细页："
                + ",".join(map(str, excluded_detail_pages))
            )
        if semantic_lookalike_pages:
            evidence_parts.append(
                "已排除仅含季度收益率或脚注提及的近三年伪候选页："
                + ",".join(map(str, semantic_lookalike_pages))
            )
        if visual_candidates:
            evidence_parts.append(
                f"图片候选扫描：{','.join(map(str, visual_candidates))}"
            )
        if visual_pages:
            evidence_parts.append(
                f"图片候选确认：{','.join(map(str, visual_pages))}"
            )
        if radar_pages:
            evidence_parts.append(
                f"Python表格雷达候选（仅补洞或兜底）：{','.join(map(str, radar_pages))}"
            )
        if directory_pages:
            evidence_parts.append(
                f"目录换算候选：{','.join(map(str, directory_pages))}"
            )
        if review_required:
            evidence_parts.append(f"证据冲突，必须人工确认：{review_reason}")
        if ai_error:
            evidence_parts.append("大模型定位失败，已使用Python雷达兜底")
        results.append(
            PageMatch(
                table_id=table_id,
                table_name=table_name,
                pages=pages,
                score=max(
                    float(local.score) if local else 0.0,
                    10.0 if ai_pages else 0.0,
                    12.0 if visual_pages else 0.0,
                ),
                evidence="；".join(evidence_parts) or "未找到明确页码证据",
                review_required=review_required,
                review_reason=review_reason,
                sources={
                    "local": local_pages,
                    "semantic": ai_pages,
                    "vision": visual_pages,
                    "directory": directory_pages,
                    "radar": radar_pages,
                },
                strategy_id=str(table.get("strategy_id", "")),
                table_config=dict(table),
            )
        )
    status_parts: list[str] = []
    if low_text_pages:
        status_parts.append(
            f"检测到{len(low_text_pages)}/{total_pages}个低文本页，已启用两阶段图片页码定位"
        )
    if searchable_pages:
        status_parts.append("已完成可检索页面语义推断与表格结构雷达复核")
    status_parts.append(directory_message)
    if ai_error:
        status_parts.append(f"文本语义定位失败，已保留其他定位结果：{ai_error}")
    if vision_errors:
        status_parts.append("图片页码定位部分失败：" + "；".join(vision_errors))
    conflict_count = sum(item.review_required for item in results)
    if conflict_count:
        status_parts.append(f"{conflict_count}类目标存在定位证据冲突，已隔离等待人工确认")
    if not status_parts:
        status_parts.append("已完成混合智能定位")
    return results, "；".join(status_parts) + "。"


def _pipe_prompt(
    table_id: str,
    table_name: str,
    page_number: int,
    retry_reason: str = "",
    table_config: Mapping[str, object] | None = None,
) -> str:
    config = dict(table_config or {})
    signatures = tuple(
        config.get("content_terms")
        or config.get("completeness_terms")
        or TABLE_SIGNATURES.get(table_id, ())
    )
    headers = tuple(
        config.get("header_terms") or TABLE_HEADERS.get(table_id, ())
    )
    configured_exclusions = tuple(dict.fromkeys([
        *config.get("exclude_item_terms", ()),
        *config.get("profile_exclude_items", ()),
        *config.get("stop_terms", ()),
    ]))
    exclusions = configured_exclusions or TABLE_EXCLUSIONS.get(table_id, ())
    strategy = active_table_strategy(table_id)
    return strategy.build_prompt(PromptRequest(
        mode="single_page",
        table_id=table_id,
        table_name=table_name,
        signatures=signatures,
        headers=headers,
        exclusions=exclusions,
        boundary_text=boundary_instruction_for_table(config or {"table_id": table_id}),
        page_number=page_number,
        retry_reason=retry_reason,
        table_config=config,
    ))


def _text_page_messages(
    table_id: str,
    table_name: str,
    page_number: int,
    grid: PageGrid,
    retry_reason: str = "",
    table_config: Mapping[str, object] | None = None,
) -> list[dict]:
    sliced = _slice_grid_for_target(
        table_id,
        grid.grid_text,
        table_config,
    )
    prompt = _pipe_prompt(
        table_id, table_name, page_number, retry_reason, table_config,
    )
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
    table_config: Mapping[str, object] | None = None,
) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": _pipe_prompt(
                table_id, table_name, page_number, table_config=table_config,
            )},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
        ],
    }]


def _parse_pipe_rows_with_shape(content: str) -> tuple[list[list[str]], float]:
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
    ragged_rows = sum(len(row) != max_width for row in rows)
    padded = [row + [""] * (max_width - len(row)) for row in rows]
    return padded, ragged_rows / max(len(rows), 1)


def _parse_pipe_rows(content: str) -> list[list[str]]:
    """Backward-compatible parser used by utility callers."""
    rows, _ = _parse_pipe_rows_with_shape(content)
    return rows


def _row_text(row: list[str]) -> str:
    return _compact("".join(row))


def _numeric_output_rows(rows: list[list[str]]) -> int:
    return sum(
        any(_cell_has_disclosed_value(cell) for cell in row[1:])
        for row in rows
        if len(row) >= 2
    )


def _trim_page_rows(
    table_id: str,
    rows: list[list[str]],
    table_config: Mapping[str, object] | None = None,
) -> list[list[str]]:
    config = dict(
        table_config
        or active_table_strategy(table_id).table_config
        or {}
    )
    markers = tuple(
        config.get("stop_terms") or AI_TABLE_END_MARKERS.get(table_id, ())
    )
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


def _prepare_page_rows(
    table_id: str,
    content: str,
    source_grids: list[PageGrid] | None = None,
) -> tuple[list[list[str]], str, float]:
    strategy = active_table_strategy(table_id)
    config = dict(strategy.table_config)
    rows, ragged_ratio = _parse_pipe_rows_with_shape(content)
    rows, postprocess_notes = strategy.postprocess(PostprocessRequest(
        table_id=table_id,
        rows=rows,
        source_grids=source_grids,
        normalize_three_year=normalize_three_year_return_rows_core,
        recover_minimum_capital=recover_minimum_capital_terminal_rows_core,
    ))
    if any("归一化" in note for note in postprocess_notes):
        ragged_ratio = 0.0
    note = "；".join(postprocess_notes)
    return _trim_page_rows(table_id, rows, config), note, ragged_ratio


def _validate_single_page_core(
    table_id: str,
    rows: list[list[str]],
    grid: PageGrid,
    ragged_ratio: float = 0.0,
    source_item_recall_ratio: float = SOURCE_ITEM_RECALL_RATIO,
    table_config: Mapping[str, object] | None = None,
) -> tuple[float, int, int]:
    config = dict(table_config or {})
    rows = _trim_page_rows(table_id, rows, config)
    if ragged_ratio > 0.20:
        raise ExtractionQualityError(
            f"本页超过20%的输出行列数不一致（{ragged_ratio:.0%}）。"
        )
    expected_rows, source_terms = _source_completeness_profile(
        table_id, [grid], config
    )
    semantic_items = set(boundary_items(table_id, config))
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
    source_labels = _source_item_labels(table_id, [grid], config)
    source_sections = _source_section_titles(table_id, [grid], config)
    _validate_source_section_order(rows, source_sections)
    item_recall, missing_items = _item_recall_profile(rows, source_labels)
    if len(source_labels) >= 4 and item_recall < source_item_recall_ratio:
        raise ExtractionQualityError(
            f"本页项目行召回率不足：{item_recall:.0%}，要求至少"
            f"{source_item_recall_ratio:.0%}；疑似遗漏："
            f"{'、'.join(missing_items[:8])}。"
        )
    score = min(99.0, 70.0 + min(actual_rows, 20) * 1.2)
    return score, expected_rows, actual_rows


def _validate_single_page(
    table_id: str,
    rows: list[list[str]],
    grid: PageGrid,
    ragged_ratio: float = 0.0,
) -> tuple[float, int, int]:
    strategy = active_table_strategy(table_id)
    return strategy.validate_completeness(CompletenessRequest(
        mode="single_page",
        table_id=table_id,
        rows=rows,
        validator=_validate_single_page_core,
        validator_kwargs={
            "grid": grid,
            "ragged_ratio": ragged_ratio,
        },
    ))


def _header_score(table_id: str, row: list[str]) -> int:
    text = _row_text(row)
    config = active_table_strategy(table_id).table_config
    headers = tuple(
        config.get("header_terms") or TABLE_HEADERS.get(table_id, ())
    )
    return sum(_compact(term) in text for term in headers)


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


def _meaningful_width(rows: list[list[str]]) -> int:
    return max(
        (
            max(
                (index + 1 for index, cell in enumerate(row) if _compact(cell)),
                default=0,
            )
            for row in rows
        ),
        default=0,
    )


def _value_column_profile(rows: list[list[str]]) -> tuple[int, ...]:
    if not rows:
        return ()
    header_text = _compact("".join(rows[0][:2]))
    row_number_table = any(term in header_text for term in ("行次", "序号", "编号"))
    counts: Counter[int] = Counter()
    numeric_rows = 0
    for row in rows[1:]:
        value_start = 2 if row_number_table and len(row) > 2 else 1
        positions = {
            index
            for index, cell in enumerate(row[value_start:], start=value_start)
            if re.search(r"\d", cell)
        }
        if not positions:
            continue
        numeric_rows += 1
        counts.update(positions)
    minimum_hits = max(1, math.ceil(numeric_rows * 0.30))
    return tuple(sorted(
        index for index, hits in counts.items() if hits >= minimum_hits
    ))


def _validate_cross_page_columns(
    pages: list[int],
    page_rows: dict[int, list[list[str]]],
) -> str:
    profiles = {
        page: (
            _meaningful_width(page_rows[page]),
            _value_column_profile(page_rows[page]),
        )
        for page in pages
        if page in page_rows
    }
    if len(profiles) <= 1:
        return "单页表无需跨页列一致性检查"
    widths = [width for width, _ in profiles.values() if width]
    if widths and max(widths) - min(widths) > 1:
        raise ExtractionQualityError(
            "跨页列数不一致："
            + "；".join(
                f"第{page}页{width}列"
                for page, (width, _) in profiles.items()
            )
        )
    reference_page, (_, reference_columns) = next(iter(profiles.items()))
    for page, (_, columns) in list(profiles.items())[1:]:
        if not reference_columns or not columns:
            continue
        if len(set(reference_columns) ^ set(columns)) > 1:
            raise ExtractionQualityError(
                f"跨页数值列位置不一致：第{reference_page}页"
                f"{reference_columns}，第{page}页{columns}。"
            )
    return "跨页列数与数值列位置一致"


def _merge_page_rows(
    table_id: str,
    table_name: str,
    pages: list[int],
    page_rows: dict[int, list[list[str]]],
) -> list[list[str]]:
    ordered = [(page, page_rows[page]) for page in pages if page in page_rows]
    if not ordered:
        raise ExtractionQualityError("没有可拼接的逐页结果。")
    _validate_cross_page_columns(pages, page_rows)
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
    render_func: Callable | None = None,
    table_config: Mapping[str, object] | None = None,
) -> tuple[
    list[list[str]] | None,
    str,
    float,
    list[ExtractionLog],
    list[UnitRecord],
]:
    logs: list[ExtractionLog] = []
    render_images = render_func or _render_images
    text_error = ""
    if not force_vision:
        try:
            if grid.word_count < 6:
                raise ExtractionQualityError("本页可检索文字不足。")
            content = _call_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=_text_page_messages(
                    table_id, table_name, page_number, grid,
                    table_config=table_config,
                ),
                timeout=timeout,
                post_func=post_func,
            )
            rows, normalization_note, ragged_ratio = _prepare_page_rows(
                table_id, content, [grid]
            )
            score, expected, actual = _validate_single_page(
                table_id, rows, grid, ragged_ratio
            )
            detail = f"逐页提取完成；源网格约{expected}条数值行，返回{actual}条。"
            if normalization_note:
                detail += normalization_note + "。"
            logs.append(ExtractionLog(
                table_id, table_name, [page_number], PAGE_TEXT_MODE, "成功",
                detail,
            ))
            return (
                rows,
                PAGE_TEXT_MODE,
                score,
                logs,
                extract_unit_records(table_id, rows, [grid]),
            )
        except Exception as exc:
            text_error = str(exc)
            logs.append(ExtractionLog(
                table_id, table_name, [page_number], PAGE_TEXT_MODE,
                "触发图片重试" if auto_vision_retry else "失败", text_error,
            ))
    if not (force_vision or auto_vision_retry):
        return None, "", 0.0, logs, []

    vision_error = ""
    try:
        images = render_images(pdf_bytes, [page_number])
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_vision_page_messages(
                table_id, table_name, page_number, images[0][1], table_config
            ),
            timeout=timeout,
            post_func=post_func,
        )
        rows, normalization_note, ragged_ratio = _prepare_page_rows(
            table_id, content, [grid]
        )
        score, expected, actual = _validate_single_page(
            table_id, rows, grid, ragged_ratio
        )
        detail = f"逐页图片提取完成；源网格约{expected}条数值行，返回{actual}条。"
        if normalization_note:
            detail += normalization_note + "。"
        page_units = extract_unit_records(table_id, rows, [grid])
        if unit_recovery_needed(rows, page_units):
            try:
                recovered_units = recover_unit_records_from_images(
                    table_id=table_id,
                    table_name=table_name,
                    pages=[page_number],
                    rows=rows,
                    images=images,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    timeout=timeout,
                    post_func=post_func,
                )
                page_units = merge_unit_records(page_units, recovered_units)
                if recovered_units:
                    detail += f"单位专用补提取获得{len(recovered_units)}条明确单位。"
                else:
                    detail += "单位专用补提取未发现明确单位，已保留待核对状态。"
            except Exception as unit_exc:
                detail += f"单位专用补提取失败，已保留待核对状态：{unit_exc}。"
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_VISION_MODE, "成功",
            detail,
        ))
        return rows, PAGE_VISION_MODE, score, logs, page_units
    except Exception as exc:
        vision_error = str(exc)
        unsupported = _is_unsupported_vision_error(exc)
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_VISION_MODE,
            "接口不支持图片，转文本纠错" if unsupported else "失败，转3倍高清图片",
            vision_error,
        ))

    if not unsupported:
        try:
            high_res_images = render_images(
                pdf_bytes,
                [page_number],
                zoom=3.0,
                jpeg_quality=92,
            )
            content = _call_chat(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=_vision_page_messages(
                    table_id, table_name, page_number, high_res_images[0][1],
                    table_config,
                ),
                timeout=timeout,
                post_func=post_func,
            )
            rows, normalization_note, ragged_ratio = _prepare_page_rows(
                table_id, content, [grid]
            )
            score, expected, actual = _validate_single_page(
                table_id, rows, grid, ragged_ratio
            )
            detail = (
                f"3倍高清图片提取完成；源网格约{expected}条数值行，"
                f"返回{actual}条。"
            )
            if normalization_note:
                detail += normalization_note + "。"
            page_units = extract_unit_records(table_id, rows, [grid])
            if unit_recovery_needed(rows, page_units):
                try:
                    recovered_units = recover_unit_records_from_images(
                        table_id=table_id,
                        table_name=table_name,
                        pages=[page_number],
                        rows=rows,
                        images=high_res_images,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        timeout=timeout,
                        post_func=post_func,
                    )
                    page_units = merge_unit_records(page_units, recovered_units)
                except Exception as unit_exc:
                    detail += f"单位补提取失败，保留待核对：{unit_exc}。"
            logs.append(ExtractionLog(
                table_id,
                table_name,
                [page_number],
                PAGE_HIGH_RES_VISION_MODE,
                "成功",
                detail,
            ))
            return rows, PAGE_HIGH_RES_VISION_MODE, score, logs, page_units
        except Exception as high_res_exc:
            vision_error = f"{vision_error}；3倍高清重试：{high_res_exc}"
            logs.append(ExtractionLog(
                table_id,
                table_name,
                [page_number],
                PAGE_HIGH_RES_VISION_MODE,
                "失败，转文本纠错",
                str(high_res_exc),
            ))

    if grid.word_count < 6:
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "失败",
            "本页没有足够文字层，图片模式也未成功。",
        ))
        return None, "", 0.0, logs, []
    try:
        reason = "；".join(item for item in (text_error, vision_error) if item)
        content = _call_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            messages=_text_page_messages(
                table_id, table_name, page_number, grid, reason, table_config
            ),
            timeout=timeout,
            post_func=post_func,
        )
        rows, normalization_note, ragged_ratio = _prepare_page_rows(
            table_id, content, [grid]
        )
        score, expected, actual = _validate_single_page(
            table_id, rows, grid, ragged_ratio
        )
        detail = f"逐页文本纠错完成；源网格约{expected}条数值行，返回{actual}条。"
        if normalization_note:
            detail += normalization_note + "。"
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "成功",
            detail,
        ))
        return (
            rows,
            PAGE_RETRY_MODE,
            score,
            logs,
            extract_unit_records(table_id, rows, [grid]),
        )
    except Exception as exc:
        logs.append(ExtractionLog(
            table_id, table_name, [page_number], PAGE_RETRY_MODE, "失败", str(exc),
        ))
        return None, "", 0.0, logs, []


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
    _parallel_tables: bool = True,
    _shared_grid_cache: dict[int, PageGrid] | None = None,
    _shared_image_cache: dict[tuple[int, float, int], tuple[int, str]] | None = None,
    _cache_lock: Lock | None = None,
    _page_worker_limit: int | None = None,
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
    grid_cache = _shared_grid_cache if _shared_grid_cache is not None else {}
    image_cache = _shared_image_cache if _shared_image_cache is not None else {}
    cache_lock = _cache_lock or Lock()

    def add_log(log: ExtractionLog) -> None:
        logs.append(log)
        if progress_callback:
            progress_callback(log)

    def cached_grids(pages: list[int]) -> list[PageGrid]:
        ordered_pages = sorted(set(pages))
        with cache_lock:
            missing = [page for page in ordered_pages if page not in grid_cache]
            if missing:
                for grid in build_pdf_grids(pdf_bytes, missing):
                    grid_cache.setdefault(grid.page_number, grid)
            return [grid_cache[page] for page in ordered_pages]

    def cached_render_images(
        source_pdf_bytes: bytes,
        pages: list[int],
        *,
        zoom: float = 1.8,
        jpeg_quality: int = 84,
    ) -> list[tuple[int, str]]:
        ordered_pages = list(dict.fromkeys(pages))
        keys = [
            (page, float(zoom), int(jpeg_quality))
            for page in ordered_pages
        ]
        with cache_lock:
            missing_pages = [
                page
                for page, key in zip(ordered_pages, keys)
                if key not in image_cache
            ]
            if missing_pages:
                rendered = _render_images(
                    source_pdf_bytes,
                    missing_pages,
                    zoom=zoom,
                    jpeg_quality=jpeg_quality,
                )
                for page, image_url in rendered:
                    image_cache.setdefault(
                        (page, float(zoom), int(jpeg_quality)),
                        (page, image_url),
                    )
            return [image_cache[key] for key in keys]

    def cached_grid_builder(
        _source_pdf_bytes: bytes,
        pages: list[int],
    ) -> list[PageGrid]:
        return cached_grids(pages)

    active_matches = [match for match in matches if sorted(set(match.pages))]
    if _parallel_tables and len(active_matches) > 1:
        completed: dict[int, AIExtractionBundle] = {}
        table_workers = min(TABLE_EXTRACTION_MAX_WORKERS, len(active_matches))
        with ThreadPoolExecutor(max_workers=table_workers) as executor:
            future_map = {
                executor.submit(
                    extract_tables_hybrid,
                    pdf_bytes,
                    [match],
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    auto_vision_retry=auto_vision_retry,
                    force_vision=force_vision,
                    timeout=timeout,
                    post_func=post_func,
                    progress_callback=None,
                    _parallel_tables=False,
                    _shared_grid_cache=grid_cache,
                    _shared_image_cache=image_cache,
                    _cache_lock=cache_lock,
                    _page_worker_limit=2,
                ): index
                for index, match in enumerate(active_matches)
            }
            for future in as_completed(future_map):
                index = future_map[future]
                bundle = future.result()
                completed[index] = bundle
                if progress_callback:
                    for log in bundle.logs:
                        progress_callback(log)
        for index in range(len(active_matches)):
            bundle = completed[index]
            tables.extend(bundle.tables)
            logs.extend(bundle.logs)
        return AIExtractionBundle(tables=tables, logs=logs)

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
                _grid_builder=cached_grid_builder,
                _render_func=cached_render_images,
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
        table_strategy = resolve_table_strategy(
            match.table_id,
            match.strategy_id or None,
            match.table_config,
        )
        grids = cached_grids(pages)
        grid_by_page = {item.page_number: item for item in grids}
        page_rows: dict[int, list[list[str]]] = {}
        page_modes: dict[int, str] = {}
        page_scores: dict[int, float] = {}
        page_unit_records: dict[int, list[UnitRecord]] = {}
        default_workers = 2 if force_vision else min(5, max(1, len(pages)))
        workers = (
            min(default_workers, max(1, _page_worker_limit))
            if _page_worker_limit is not None
            else default_workers
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    table_strategy.run,
                    "extract_page",
                    _extract_one_page,
                    pdf_bytes=pdf_bytes,
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
                    render_func=cached_render_images,
                    table_config=match.table_config,
                ): page
                for page in pages
            }
            for future in as_completed(future_map):
                page = future_map[future]
                try:
                    rows, mode, score, page_logs, units = future.result()
                except Exception as exc:
                    rows, mode, score, units = None, "", 0.0, []
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
                    page_unit_records[page] = units

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
            merged_rows = table_strategy.run(
                "merge_pages",
                _merge_page_rows,
                match.table_name,
                pages,
                page_rows,
                inject_table_id="positional",
            )
            try:
                merged_rows, boundary_note = table_strategy.enforce_boundaries(
                    BoundaryRequest(
                    mode="output_rows",
                    table_id=match.table_id,
                    rows=merged_rows,
                    require_complete=True,
                    engine=enforce_output_boundaries,
                ))
            except TableBoundaryError as boundary_exc:
                raise ExtractionQualityError(str(boundary_exc)) from boundary_exc
            expected_rows, source_terms = _source_completeness_profile(
                match.table_id, grids, match.table_config
            )
            configured_terms = tuple(
                match.table_config.get("completeness_terms", ())
            )
            if configured_terms:
                source_terms = configured_terms
            source_labels = _source_item_labels(
                match.table_id, grids, match.table_config
            )
            source_sections = _source_section_titles(
                match.table_id, grids, match.table_config
            )
            quality, evidence = table_strategy.validate_completeness(
                CompletenessRequest(
                    mode="full_table",
                    table_id=match.table_id,
                    rows=merged_rows,
                    ragged_ratio=0.0,
                    signatures=tuple(
                        match.table_config.get("content_terms")
                        or TABLE_SIGNATURES.get(match.table_id, ())
                    ),
                    validator=_validate,
                    validator_kwargs={
                        "expected_source_rows": expected_rows,
                        "source_required_terms": source_terms,
                        "source_item_labels": source_labels,
                        "source_section_titles": source_sections,
                        "configured_required": tuple(
                            match.table_config.get("content_terms", ())
                        ),
                        "configured_exclusions": tuple(dict.fromkeys([
                            *match.table_config.get("exclude_item_terms", ()),
                            *match.table_config.get("profile_exclude_items", ()),
                            *match.table_config.get("stop_terms", ()),
                        ])),
                        "configured_minimum_rows": int(
                            match.table_config.get("minimum_rows", 0) or 0
                        ),
                    },
                )
            )
            modes = [page_modes[page] for page in pages]
            strategy = CROSS_PAGE_MERGE_MODE + "（" + " / ".join(dict.fromkeys(modes)) + "）"
            merged_units = merge_unit_records(
                extract_unit_records(match.table_id, merged_rows, grids),
                *[page_unit_records.get(page, []) for page in pages],
            )
            tables.append(ExtractedTable(
                table_id=match.table_id,
                table_name=match.table_name,
                page=pages[0],
                table_index=1,
                rows=merged_rows,
                strategy=strategy,
                quality_score=min(quality, sum(page_scores.values()) / len(page_scores)),
                evidence=(
                    f"{evidence}；{boundary_note}；跨页列结构一致性校验通过；"
                    f"逐页提取成功后按物理页码{pages}拼接"
                ),
                source_pages=pages,
                unit_records=merged_units,
                profile_strategy_id=table_strategy.strategy_id,
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
