from __future__ import annotations

"""VLM-first extraction with bounded source checks for required disclosures."""

import base64
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Callable, Mapping, Sequence

import fitz
import pandas as pd
import requests
from PIL import Image

from .llm_config import normalize_model_id, vlm_request_parameters
from .llm_http import post_json_with_retry
from .solvency_pdf_locator import (
    PageMatch,
    POLICY_SURPLUS_SOURCE_LABELS,
    visible_policy_surplus_codes,
)
from .solvency_table_extractor import ExtractedTable
from .solvency_filing_catalog import filing_codes, filing_entries, TEXT_FILING_CODES
from .solvency_disclosure_policy import is_disclosed_zero
from .solvency_amounts import chinese_money_yuan, source_rounding_agrees
from .solvency_source_semantics import QUANT_RISK_RULES, quant_risk_source_error
from .solvency_asset_columns import build_recognized_views, recognized_column_error
from .solvency_locator_roles import LOCATOR_SCOPE_RULES, locator_page_hits, select_locator_pages
from .solvency_unit_context import (
    canonical_unit, literal_money_unit, recognized_unit_contexts,
    model_unit_context, apply_unit_context, apply_item_unit_evidence,
)
from .solvency_locator_recovery import (
    DETAIL_IDS, bounded_review_pages, latest_scan_failures,
    operating_section_closed, text_page_catalog, render_review_sheets,
)
from .solvency_metric_registry import CUSTOM_METRICS_BY_CODE
from .solvency_operating_recovery import recover_insurance_revenue, recover_surrender_rate


LOCATOR_MODE = "VLM v2全页视觉定位"
EXTRACTION_MODE = "VLM v2标准指标直提"
RETRY_MODE = "VLM v2失败指标定向重提"

METRIC_COLUMNS = [
    "目标表ID", "目标表名称", "指标编码", "指标名称", "指标语义键", "状态", "原始值", "数值",
    "单位", "标准数值", "标准单位", "期间口径", "原始标签", "行表头路径",
    "列表头路径", "物理页码", "证据原文", "置信度", "提取轮次",
]
VALIDATION_COLUMNS = [
    "校验ID", "目标表ID", "指标编码", "状态", "规则", "实际值", "期望值", "说明",
]

_TABLE_DESCRIPTIONS = {
    "SOLVENCY_MAIN": (
        "偿付能力核心摘要，披露认可资产、认可负债、实际资本、最低资本以及核心和综合"
        "偿付能力充足率；可能叫主要指标、偿付能力状况或监管指标摘要。"
    ),
    "OPERATING_METRICS": (
        "主要经营指标，披露保险业务收入、利润、资产负债、投资收益率、新业务价值、"
        "保费或渠道指标；前五大产品的信息按有证据的文本保留，分类标题不充当数值。"
    ),
    "REGISTERED_CAPITAL": (
        "公司基本信息、公司基本情况或公司概况中明确披露的注册资本金额。只定位直接"
        "出现‘注册资本’或‘注册资本金’及其金额和单位的页面；不得把实收资本、股本、"
        "实际资本、核心资本或其他偿付能力资本误认为注册资本。"
    ),
    "ACTUAL_CAPITAL": (
        "实际资本及资本分级构成，可能同时包含财务报表资产负债汇总、核心一级资本"
        "调整以及核心/附属各级资本明细。"
    ),
    "RECOGNIZED_ASSETS": (
        "认可资产构成明细，通常含账面价值、非认可价值、认可价值等多级表头，"
        "从现金及流动性管理工具等资产类别到认可资产合计。"
    ),
    "THREE_YEAR_INVESTMENT_RETURN": (
        "近三年平均投资收益率和近三年平均综合投资收益率，可能是两行表格、同页两句话"
        "或相邻页文字；普通本季度投资收益率不是本目标。"
    ),
    "MINIMUM_CAPITAL": (
        "最低资本及其量化风险、控制风险、附加资本构成，可能继续展开保险、市场和"
        "信用风险最低资本汇总并跨页披露。"
    ),
}

_TABLE_MODULES = {
    "SOLVENCY_MAIN": {"主要指标"},
    "OPERATING_METRICS": {"经营指标"},
    "ACTUAL_CAPITAL": {"实际资本"},
    "RECOGNIZED_ASSETS": {"认可资产"},
    "MINIMUM_CAPITAL": {"最低资本"},
}

_TABLE_EXTRA_CODES = {
    "REGISTERED_CAPITAL": {"REGISTERED_CAPITAL"},
    "ACTUAL_CAPITAL": {"ACTUAL_CAPITAL"},
    "RECOGNIZED_ASSETS": {"RECOGNIZED_ASSETS"},
    "MINIMUM_CAPITAL": {"MINIMUM_CAPITAL"},
    "THREE_YEAR_INVESTMENT_RETURN": {
        "INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN",
    },
}

_REQUIRED_CODES = {
    "SOLVENCY_MAIN": {
        "RECOGNIZED_ASSETS", "RECOGNIZED_LIABILITIES", "ACTUAL_CAPITAL",
        "MINIMUM_CAPITAL", "CORE_SOLVENCY_RATIO", "COMBINED_SOLVENCY_RATIO",
    },
    "OPERATING_METRICS": {"INSURANCE_REVENUE"},
    "REGISTERED_CAPITAL": {"REGISTERED_CAPITAL"},
    "ACTUAL_CAPITAL": {
        "CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL",
        "ANC_T2_CAPITAL", "ACTUAL_CAPITAL",
    },
    "RECOGNIZED_ASSETS": {
        "CASH_LIQUID_ASSETS", "INVESTMENT_ASSETS", "REINSURANCE_ASSETS",
        "RECOGNIZED_ASSETS",
    },
    "THREE_YEAR_INVESTMENT_RETURN": {
        "INVESTMENT_RETURN", "COMPREHENSIVE_INVESTMENT_RETURN",
    },
    "MINIMUM_CAPITAL": {
        "QUANT_RISK_CAPITAL", "INSURANCE_RISK_CAPITAL", "MARKET_RISK_CAPITAL",
        "CREDIT_RISK_CAPITAL", "CONTROL_RISK_CAPITAL", "ADDITIONAL_CAPITAL",
        "MINIMUM_CAPITAL",
    },
}

_THREE_YEAR_METRIC_NAMES = {
    "INVESTMENT_RETURN": "近三年平均投资收益率",
    "COMPREHENSIVE_INVESTMENT_RETURN": "近三年平均综合投资收益率",
}

_METRIC_ALIAS_OVERRIDES = {
    "FINANCIAL_STATEMENT_NET_ASSETS": ("净资产", "财务报表净资产", "净资产合计"),
}

_REGISTRY_FALLBACK_CODES = {"REGISTERED_CAPITAL"}

_AMOUNT_UNIT_FACTORS = {
    "元": 1.0,
    "千元": 1_000.0,
    "万元": 10_000.0,
    "百万元": 1_000_000.0,
    "亿元": 100_000_000.0,
}
_COUNT_UNIT_FACTORS = {
    "人": 1.0,
    "万人": 10_000.0,
    "户": 1.0,
    "万户": 10_000.0,
    "件": 1.0,
    "万件": 10_000.0,
}


@dataclass(frozen=True)
class VLMV2TargetCard:
    table_id: str
    table_name: str
    description: str
    metrics: tuple[Mapping[str, object], ...]

    def prompt_payload(
        self,
        *,
        metric_codes: set[str] | None = None,
        compact: bool = False,
    ) -> dict:
        selected = [
            dict(metric)
            for metric in self.metrics
            if metric_codes is None or str(metric["metric_id"]) in metric_codes
        ]
        if compact:
            selected = [
                {
                    "metric_id": metric["metric_id"],
                    "name": metric["name"],
                    "aliases": metric.get("aliases", []),
                    "expected_unit": metric.get("expected_unit", ""),
                    "source_label": metric.get("source_label", ""),
                    "source_row_number": metric.get("source_row_number", ""),
                    "source_matching_rule": QUANT_RISK_RULES.get(str(metric['metric_id']), ''),
                }
                for metric in selected
            ]
        return {
            "table_id": self.table_id,
            "configured_name": self.table_name,
            "semantic_definition": self.description,
            "response_mode": "sparse_disclosures" if compact else "complete",
            "target_metrics": selected,
        }


@dataclass(frozen=True)
class VLMV2LocatorRun:
    matches: tuple[PageMatch, ...]
    diagnostics: pd.DataFrame
    model_calls: int
    page_count: int


@dataclass(frozen=True)
class VLMV2ExtractionRun:
    records: pd.DataFrame
    validations: pd.DataFrame
    logs: tuple[str, ...]
    model_calls: int
    retry_calls: int


@dataclass(frozen=True)
class VLMV2QualityGate:
    passed: bool
    checks: pd.DataFrame
    summary: str


def _clean(value: object) -> str:
    return str(value if value is not None else "").strip()


def _is_non_current_period_label(value: object) -> bool:
    """Return whether a model-selected source column is not the current quarter."""
    text = _clean(value)
    return bool(text and re.search(r"上季度|上期|上年|去年|同期|期初|下季度|预测|累计", text))


def _operating_period_kind(value: object) -> str:
    """Classify the two independently retained operating-metric periods."""
    text = _clean(value)
    if re.search(r"本年度?累计|本年累计|年度累计|累计数", text):
        return "cumulative"
    if re.search(r"本季度|当季|本报告期|本期|期末", text):
        return "current"
    return "other"


def _period_is_invalid(value: object, table_id: object) -> bool:
    """Reject historical/forecast columns while allowing operating cumulative values."""
    text = _clean(value)
    if _clean(table_id) == "OPERATING_METRICS":
        return bool(text and re.search(r"上季度|上期|上年|去年|同期|期初|下季度|预测", text))
    return _is_non_current_period_label(text)


def _record_period_text(row: Mapping[str, object]) -> str:
    return _clean(row.get("期间口径") or row.get("period_label")) or _path_text(
        row.get("列表头路径") or row.get("column_header_path")
    )


def _candidate_storage_key(
    table_id: str,
    metric_id: str,
    item: Mapping[str, object],
) -> tuple[str, str]:
    if table_id != "OPERATING_METRICS":
        return metric_id, ""
    period_text = _clean(item.get("period_label")) or _path_text(item.get("column_header_path"))
    period_kind = _operating_period_kind(period_text)
    return metric_id, period_kind if period_kind != "other" else (period_text or "unspecified")


def build_vlm_v2_target_cards(
    table_configs: Sequence[Mapping[str, object]],
    taxonomy: pd.DataFrame,
) -> tuple[VLMV2TargetCard, ...]:
    """Build semantic target cards without company-specific layout rules."""
    required_columns = {
        "指标编码", "指标名称", "别名", "一级模块", "二级模块",
        "标准单位", "数据类型", "允许期间口径",
    }
    missing = required_columns - set(taxonomy.columns)
    if missing:
        raise ValueError("指标字典缺少字段：" + "、".join(sorted(missing)))

    cards: list[VLMV2TargetCard] = []
    for table in table_configs:
        table_id = _clean(table.get("table_id"))
        table_name = _clean(table.get("table_name"))
        modules = _TABLE_MODULES.get(table_id, set())
        extra_codes = _TABLE_EXTRA_CODES.get(table_id, set()) | filing_codes(table_id)
        selected = taxonomy[
            taxonomy["一级模块"].astype(str).isin(modules)
            | taxonomy["指标编码"].astype(str).isin(extra_codes)
        ]
        selected_codes = set(selected["指标编码"].astype(str))
        fallback_codes = (
            extra_codes & _REGISTRY_FALLBACK_CODES
        ) - selected_codes
        fallback_rows = [
            CUSTOM_METRICS_BY_CODE[code].taxonomy_row()
            for code in sorted(fallback_codes)
            if code in CUSTOM_METRICS_BY_CODE
        ]
        if fallback_rows:
            selected = pd.concat(
                [selected, pd.DataFrame(fallback_rows)],
                ignore_index=True,
            ).fillna("")
        checklist_rows = {item['code']: item for item in filing_entries() if item['table_id'] == table_id}
        if checklist_rows:
            selected = selected[selected['指标编码'].isin(checklist_rows)]
        metrics: list[dict[str, object]] = []
        for _, row in selected.iterrows():
            metric_id = _clean(row.get("指标编码"))
            metric_name = _THREE_YEAR_METRIC_NAMES.get(metric_id) \
                if table_id == "THREE_YEAR_INVESTMENT_RETURN" else ""
            aliases = [
                item.strip()
                for item in _clean(row.get("别名")).split("|")
                if item.strip()
            ]
            if metric_name:
                aliases = list(dict.fromkeys([metric_name, *aliases]))
            aliases = list(dict.fromkeys([
                *aliases,
                *_METRIC_ALIAS_OVERRIDES.get(metric_id, ()),
            ]))
            semantic_key = (
                f"{metric_id}:THREE_YEAR_AVERAGE"
                if table_id == "THREE_YEAR_INVESTMENT_RETURN"
                else metric_id
            )
            metrics.append({
                "metric_id": metric_id,
                "semantic_key": semantic_key,
                "name": metric_name or _clean(row.get("指标名称")),
                "aliases": aliases,
                "module": _clean(row.get("二级模块")),
                "expected_unit": _clean(row.get("标准单位")),
                "data_type": _clean(row.get("数据类型")),
                "allowed_period_labels": [
                    item.strip()
                    for item in _clean(row.get("允许期间口径")).split("|")
                    if item.strip()
                ],
                "required": metric_id in _REQUIRED_CODES.get(table_id, set()),
                "source_label": checklist_rows.get(metric_id, {}).get('source_label', ''),
                "source_row_number": checklist_rows.get(metric_id, {}).get('row_number', ''),
            })
        cards.append(VLMV2TargetCard(
            table_id=table_id,
            table_name=table_name,
            description=_TABLE_DESCRIPTIONS.get(
                table_id,
                f"从报告中定位并提取与“{table_name}”业务含义一致的披露内容。",
            ),
            metrics=tuple(metrics),
        ))
    return tuple(cards)


def _completion_url(base_url: str) -> str:
    value = _clean(base_url).rstrip("/")
    if not value:
        raise ValueError("模型接口地址不能为空。")
    return value if value.endswith("/chat/completions") else f"{value}/chat/completions"


def _response_error(response) -> str:
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict):
            return _clean(error.get("message"))
    except Exception:
        pass
    return _clean(getattr(response, "text", ""))[:500]


def _call_vlm_json(
    *,
    prompt: str,
    image_urls: Sequence[str],
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    max_attempts: int,
    post_func: Callable | None,
) -> dict:
    content: list[dict] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_url}}
        for image_url in image_urls
    )
    payload = {
        "model": normalize_model_id(base_url, model),
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
    }
    payload.update(vlm_request_parameters(base_url, model))
    headers = {
        "Authorization": f"Bearer {_clean(api_key)}",
        "Content-Type": "application/json",
    }
    post = post_func or requests.post
    response = post_json_with_retry(
        post,
        _completion_url(base_url),
        headers=headers,
        json=payload,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    if (
        not getattr(response, "ok", False)
        and int(getattr(response, "status_code", 0) or 0) in {400, 422}
    ):
        payload.pop("response_format", None)
        response = post_json_with_retry(
            post,
            _completion_url(base_url),
            headers=headers,
            json=payload,
            timeout=timeout,
            max_attempts=max_attempts,
        )
    if not getattr(response, "ok", False):
        raise RuntimeError(
            "VLM v2模型接口调用失败：" + (_response_error(response) or "HTTP错误")
        )
    try:
        content_value = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("VLM v2模型接口未返回choices/message/content。") from exc
    if isinstance(content_value, list):
        content_value = "".join(
            _clean(item.get("text")) if isinstance(item, dict) else _clean(item)
            for item in content_value
        )
    text = _clean(content_value)
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("VLM v2返回的内容不是有效JSON对象。") from exc
    if not isinstance(value, dict):
        raise RuntimeError("VLM v2必须返回JSON对象。")
    return value


def _image_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_vlm_v2_contact_sheets(
    pdf_bytes: bytes,
    *,
    pages_per_sheet: int = 6,
    jpeg_quality: int = 78,
) -> list[tuple[tuple[int, ...], str]]:
    """Render every PDF page visually; this works even when no text layer exists."""
    if not pdf_bytes:
        return []
    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    sheets: list[tuple[tuple[int, ...], str]] = []
    per_sheet = max(1, min(int(pages_per_sheet), 6))
    columns = 2
    rows = math.ceil(per_sheet / columns)
    sheet_width = 1800
    cell_width = 870
    cell_height = 1180
    label_height = 42
    gap = 20
    sheet_height = rows * (cell_height + label_height + gap) + gap
    try:
        for start in range(0, source.page_count, per_sheet):
            page_numbers = tuple(
                range(start + 1, min(source.page_count, start + per_sheet) + 1)
            )
            contact = fitz.open()
            page = contact.new_page(width=sheet_width, height=sheet_height)
            for index, page_number in enumerate(page_numbers):
                column = index % columns
                row = index // columns
                x0 = gap + column * (cell_width + gap)
                y0 = gap + row * (cell_height + label_height + gap)
                page.draw_rect(
                    fitz.Rect(x0, y0, x0 + cell_width, y0 + label_height),
                    color=(0.0, 0.2, 0.55),
                    fill=(0.0, 0.2, 0.55),
                )
                page.insert_text(
                    (x0 + 14, y0 + 29),
                    f"PDF PHYSICAL PAGE {page_number}",
                    fontsize=22,
                    color=(1, 1, 1),
                )
                target = fitz.Rect(
                    x0,
                    y0 + label_height,
                    x0 + cell_width,
                    y0 + label_height + cell_height,
                )
                page.show_pdf_page(target, source, page_number - 1, keep_proportion=True)
            pixmap = page.get_pixmap(alpha=False)
            sheets.append((
                page_numbers,
                _image_data_url(pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)),
            ))
            contact.close()
    finally:
        source.close()
    return sheets


def _render_page_images(
    pdf_bytes: bytes,
    pages: Sequence[int],
    *,
    zoom: float,
    jpeg_quality: int,
    max_dimension: int = 1800,
    max_image_bytes: int = 750_000,
) -> list[tuple[int, str]]:
    source = fitz.open(stream=pdf_bytes, filetype="pdf")
    rendered: list[tuple[int, str]] = []
    try:
        for page_number in sorted(set(int(page) for page in pages)):
            if not 1 <= page_number <= source.page_count:
                continue
            page = source.load_page(page_number - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            jpeg = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
            if (
                max(pixmap.width, pixmap.height) > max_dimension
                or len(jpeg) > max_image_bytes
            ):
                with Image.open(io.BytesIO(jpeg)) as image:
                    image = image.convert("RGB")
                    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    for quality in (jpeg_quality, 76, 68, 60, 52):
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=quality, optimize=True)
                        jpeg = buffer.getvalue()
                        if len(jpeg) <= max_image_bytes:
                            break
                    while len(jpeg) > max_image_bytes and min(image.size) > 600:
                        image = image.resize(
                            (
                                max(1, round(image.width * 0.85)),
                                max(1, round(image.height * 0.85)),
                            ),
                            Image.Resampling.LANCZOS,
                        )
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=52, optimize=True)
                        jpeg = buffer.getvalue()
            rendered.append((
                page_number,
                _image_data_url(jpeg),
            ))
    finally:
        source.close()
    return rendered


def _locator_prompt(
    cards: Sequence[VLMV2TargetCard],
    visible_pages: Sequence[int],
) -> str:
    card_payload = [
        {
            "table_id": card.table_id,
            "configured_name": card.table_name,
            "semantic_definition": card.description,
            "source_scope": LOCATOR_SCOPE_RULES.get(card.table_id, ''),
            "representative_metrics": [
                metric["name"] for metric in card.metrics if metric.get("required")
            ][:12],
        }
        for card in cards
    ]
    return f"""你是保险公司偿付能力报告的视觉审阅专家。

你看到的是PDF页面拼图，每个页面顶部有“PDF PHYSICAL PAGE N”物理页码。
本任务必须仅根据页面图像判断，不依赖PDF文字层，因此扫描版PDF同样适用。
当前批次包含物理页：{list(visible_pages)}。

请按业务语义定位目标，而不是要求标题逐字相同。目标定义：
{json.dumps(card_payload, ensure_ascii=False)}

要求：
1. 检查当前批次每一页；表名改变、横竖表、无边框或文字句式披露都要识别。
2. 返回当前批次中真正包含目标数据的全部页面；仅目录提及或正文引用不能算命中。
3. 跨页续表即使没有重复标题，也应根据表头、行项目和连续性识别。
4. 只读蓝色PDF PHYSICAL PAGE标签，禁止用页脚印刷页码、目录页码或拼图格序号替代。没有发现时found=false。
5. 每个候选物理页单独返回page_hits、标题、角色与简短证据，不能将多个不同来源页共用一段证据。confidence为0到1。
6. role分别为primary主表、continuation续表/分项、summary汇总数、reference变动分析/引用、toc目录、uncertain待核验。
   找到部分同名指标不等于找到目标明细表；S02/S03/S05不得将主要指标摘要作为primary。
   不要求印有S02/S03/S05编码，实际资本/认可资产/最低资本章节的等义明细表也可命中。
7. 一页可同时包含多个目标表，逐目标独立返回，禁止一页只归一个目标。不得自动把相邻页全部当作续表。
8. OPERATING_METRICS检查主要、效益、规模/渠道、前五位产品、品质五组；分别用coverage_groups中的main/benefit/scale/products/quality记录当前页实际可见分组。各公司披露不同，不要求五组齐全，不得因缺少可选分组否认已定位主表。只见标题或填表说明不能算有数据。
   特别检查近三年收益率前后同页的产品/品质表，仍归OPERATING_METRICS；近三年收益率另归THREE_YEAR_INVESTMENT_RETURN。
9. 每页evidence只写标题与2-3个区分性行标签，无需抄数值或整段文字。
10. 另用page_catalog逐页记录可见表名/章节标题（非目录），即使targets没有命中也记录；横向宽表及无标题续页不能忽略。
11. 经营指标表已经结束且可见下一独立章节时，返回section_end：confirmed=true、page、next_heading、evidence。必须看见实际边界，不得因未见产品/品质而猜测结束。
12. 每个请求目标都返回一项；没有命中明确found=false。不得把漏回目标当作未披露。

只返回JSON：
{{"targets":[{{"table_id":"...","found":true,"page_hits":[{{"page":1,
"role":"primary","source_title":"原页面标题","evidence":"可见区分性行标签",
"coverage_groups":[],"confidence":0.95}}],
"section_end":{{"confirmed":false,"page":0,"next_heading":"","evidence":""}}}}],
"page_catalog":[{{"page":1,"titles":["当前可见的章节或表格标题"]}}]}}
""".strip()


def locate_tables_vlm_v2(
    pdf_bytes: bytes,
    table_configs: Sequence[Mapping[str, object]],
    taxonomy: pd.DataFrame,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 90,
    request_max_attempts: int = 2,
    post_func: Callable | None = None,
    pages_per_sheet: int = 6,
    sheets_per_call: int = 1,
    max_workers: int = 2,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> VLMV2LocatorRun:
    """Locate targets in bounded page batches and retry only timed-out batches."""
    started_at = perf_counter()
    if not all((_clean(api_key), _clean(base_url), _clean(model))):
        raise ValueError("请完整填写模型接口地址、模型名称和API Key。")
    cards = build_vlm_v2_target_cards(table_configs, taxonomy)
    sheets = render_vlm_v2_contact_sheets(
        pdf_bytes,
        pages_per_sheet=pages_per_sheet,
    )
    if not sheets:
        raise ValueError("PDF没有可渲染页面。")

    candidate_hits = {card.table_id: [] for card in cards}
    page_catalog = []
    operating_closed = False
    operating_end_notes = []
    recovery_errors = {}
    missing_response_ids = set()
    diagnostics: list[dict[str, object]] = []
    valid_ids = set(candidate_hits)
    # Keep the compatibility argument, but formally cap every API call at one
    # six-page contact sheet so a single visual request cannot grow unchecked.
    group_size = min(max(1, int(sheets_per_call)), 1)
    groups = [
        sheets[start:start + group_size]
        for start in range(0, len(sheets), group_size)
    ]
    model_calls = 0

    def run_batch(
        batch_index: int,
        group: Sequence[tuple[tuple[int, ...], str]],
        round_name: str,
        focus_cards=None,
    ) -> dict[str, object]:
        visible_pages = [page for pages, _ in group for page in pages]
        try:
            prompt = _locator_prompt(focus_cards or cards, visible_pages)
            if round_name == '缺失表/续页高清复核':
                prompt += '\n这是缺失表/无标题续页高清复核。以下只是候选区间，不是已确认页码；逐页检查行项目、列结构，排除其他表。标题页不在本图时仍可根据连续行号确认续页。候选区间：' + json.dumps(recovery_ranges, ensure_ascii=False)
            response = _call_vlm_json(
                prompt=prompt,
                image_urls=[image_url for _, image_url in group],
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
                # The second attempt is deliberately deferred until every
                # first-round batch has finished, instead of blocking here.
                max_attempts=1,
                post_func=post_func,
            )
            if not isinstance(response.get('targets'), list):
                raise ValueError('定位响应缺少有效targets列表，不能视为未找到。')
            return {
                "batch_index": batch_index,
                "group": group,
                "visible_pages": visible_pages,
                "round": round_name,
                "response": response,
                "target_ids": {card.table_id for card in (focus_cards or cards)},
                "error": None,
                "timed_out": False,
            }
        except Exception as exc:
            return {
                "batch_index": batch_index,
                "group": group,
                "visible_pages": visible_pages,
                "round": round_name,
                "response": None,
                "error": exc,
                "timed_out": isinstance(exc, requests.Timeout),
            }

    def execute_batches(
        batch_specs: Sequence[tuple[int, Sequence[tuple[tuple[int, ...], str]]]],
        round_name: str,
        focus_cards=None,
    ) -> list[dict[str, object]]:
        if not batch_specs:
            return []
        worker_count = min(max(1, int(max_workers)), len(batch_specs), 2)
        results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_batch, batch_index, group, round_name, focus_cards): batch_index
                for batch_index, group in batch_specs
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                _notify_progress(
                    progress_callback,
                    started_at,
                    "locator_batch_completed",
                    round_name=round_name,
                    completed_batches=completed,
                    total_batches=len(batch_specs),
                    batch_index=result["batch_index"],
                    visible_pages=result["visible_pages"],
                    status=(
                        "超时"
                        if result["timed_out"]
                        else "失败"
                        if result["error"] is not None
                        else "成功"
                    ),
                )
        return results

    def consume_result(result: Mapping[str, object]) -> None:
        nonlocal model_calls, operating_closed
        model_calls += 1
        visible_pages = list(result["visible_pages"])
        response = result.get("response")
        error = result.get("error")
        timed_out = bool(result.get("timed_out"))
        batch_hits: list[str] = []
        omitted = set()
        if isinstance(response, Mapping):
            returned = {str(item.get('table_id', '')) for item in response.get('targets', []) if isinstance(item, Mapping)}
            omitted = set(result.get('target_ids', valid_ids)) - returned
            missing_response_ids.update(omitted)
            for entry in response.get('page_catalog', []) if isinstance(response.get('page_catalog'), list) else []:
                if isinstance(entry, dict) and type(entry.get('page')) is int and entry['page'] in visible_pages:
                    page_catalog.append(entry)
            for item in response.get("targets", []):
                if not isinstance(item, Mapping):
                    continue
                table_id = _clean(item.get("table_id"))
                if table_id not in valid_ids or table_id not in result.get('target_ids', valid_ids):
                    continue
                hits = locator_page_hits(item, visible_pages)
                candidate_hits[table_id].extend(hits)
                if table_id == 'OPERATING_METRICS':
                    closed = operating_section_closed(item, hits, visible_pages)
                    operating_closed |= closed
                    if closed:
                        boundary = item['section_end']
                        operating_end_notes.append(f"经营指标结束边界：物理页{boundary['page']}，下一章节{boundary['next_heading']}；{boundary['evidence']}")
                pages = sorted({hit['page'] for hit in hits})
                if not pages:
                    continue
                batch_hits.append(f"{table_id}:{','.join(map(str, pages))}")
        diagnostics.append({
            "调用序号": model_calls,
            "批次序号": int(result["batch_index"]),
            "轮次": _clean(result.get("round")),
            "扫描物理页": f"{min(visible_pages)}-{max(visible_pages)}",
            "页面数量": len(visible_pages),
            "状态": "超时" if timed_out else "失败" if error is not None else "成功",
            "命中目标": "；".join(batch_hits) or "无",
            "错误": _clean(error)[:500],
            "漏回目标": '、'.join(sorted(omitted)),
        })

    initial_specs = [(index, group) for index, group in enumerate(groups, start=1)]
    _notify_progress(
        progress_callback,
        started_at,
        "locator_started",
        total_batches=len(initial_specs),
        pages_per_batch=max(len(pages) for pages, _ in sheets),
    )
    initial_results = execute_batches(initial_specs, "首轮")
    for result in initial_results:
        consume_result(result)

    timed_out_results = [result for result in initial_results if result["timed_out"]]
    if int(request_max_attempts) > 1 and timed_out_results:
        retry_specs = [
            (int(result["batch_index"]), result["group"])
            for result in timed_out_results
        ]
        _notify_progress(
            progress_callback,
            started_at,
            "locator_retry_started",
            total_batches=len(retry_specs),
            batch_indexes=[index for index, _ in retry_specs],
        )
        retry_results = execute_batches(retry_specs, "超时重试")
        for result in retry_results:
            consume_result(result)

    # A single bounded boundary review for operating sub-sections. Reuse the
    # existing contact sheets; do not rescan the full PDF or guess neighbor hits.
    operating_hits = candidate_hits.get('OPERATING_METRICS', [])
    data_hits = [h for h in operating_hits if h['role'] in {'primary', 'continuation'}]
    coverage = {str(group) for hit in data_hits for group in hit.get('coverage_groups', [])
                if isinstance(hit.get('coverage_groups'), list)}
    scope_review_incomplete = False
    if data_hits and not operating_closed and not {'products', 'quality'}.issubset(coverage):
        last_page = max(hit['page'] for hit in data_hits)
        boundary_specs = [(i, g) for i, g in initial_specs
                          if any(last_page <= p <= last_page + 1 for pages, _ in g for p in pages)][:2]
        focus = tuple(card for card in cards if card.table_id == 'OPERATING_METRICS')
        scope_results = execute_batches(boundary_specs, '经营指标边界复核', focus)
        for result in scope_results:
            consume_result(result)
        scope_review_incomplete = any(result['error'] is not None for result in scope_results)

    # Optional title hints are not locations. Recheck only unresolved detail
    # ranges using larger pages, with the same VLM and at most two workers.
    catalog = page_catalog + text_page_catalog(pdf_bytes)
    page_count = sum(len(pages) for pages, _ in sheets)
    recovery_ranges = {
        card.table_id: bounded_review_pages(card.table_id, candidate_hits[card.table_id], catalog, page_count)
        for card in cards if card.table_id in DETAIL_IDS
    }
    recovery_pages = sorted({p for pages in recovery_ranges.values() for p in pages})[:12]
    if recovery_pages:
        review_sheets = render_review_sheets(pdf_bytes, recovery_pages)
        review_specs = [(len(initial_specs) + i, [sheet]) for i, sheet in enumerate(review_sheets, 1)]
        focus = tuple(card for card in cards if recovery_ranges.get(card.table_id))
        for result in execute_batches(review_specs, '缺失表/续页高清复核', focus):
            consume_result(result)
            if result['error'] is not None:
                for card in focus:
                    if set(result['visible_pages']) & set(recovery_ranges[card.table_id]):
                        recovery_errors[card.table_id] = '高清复核请求失败，已保留成功页码，不能判为未披露。'

    scan_failures = latest_scan_failures(diagnostics)

    _notify_progress(
        progress_callback,
        started_at,
        "locator_completed",
        total_batches=len(initial_specs),
        model_calls=model_calls,
        failed_batches=len(scan_failures),
    )

    config_by_id = {_clean(item.get("table_id")): dict(item) for item in table_configs}
    matches: list[PageMatch] = []
    for card in cards:
        pages, confidence, review_reason, sources, reason = select_locator_pages(
            card.table_id, candidate_hits[card.table_id],
        )
        if card.table_id == 'OPERATING_METRICS' and operating_end_notes:
            reason += '\n' + '\n'.join(dict.fromkeys(operating_end_notes))
        if card.table_id == 'OPERATING_METRICS' and scope_review_incomplete:
            review_reason = '经营指标边界复核失败，已保留首轮页码，请核对产品/品质续页。' + review_reason
        elif card.table_id == 'OPERATING_METRICS' and data_hits and not operating_closed:
            final_coverage = {str(group) for hit in candidate_hits[card.table_id]
                              if hit['role'] in {'primary', 'continuation'}
                              for group in hit['coverage_groups']}
            if not {'products', 'quality'}.issubset(final_coverage):
                review_reason = '未确认产品/品质分项范围（可能未披露或漏页），请人工核验。' + review_reason
        if card.table_id in recovery_errors:
            review_reason = recovery_errors[card.table_id] + review_reason
        if not pages:
            if scan_failures:
                ranges = '、'.join(row['扫描物理页'] for row in scan_failures)
                review_reason = f'扫描未完成：物理页{ranges}的批次请求失败/超时；未定位不等于未披露。' + review_reason
            elif card.table_id in missing_response_ids:
                review_reason = '模型响应漏回此目标，尚不能确认是否披露；请人工核对。' + review_reason
            else:
                review_reason += '；未定位不等于未披露，请结合原PDF核对。'
        if recovery_ranges.get(card.table_id):
            sources['vlm_recovery_candidate'] = recovery_ranges[card.table_id]
            unreviewed = set(recovery_ranges[card.table_id]) - set(recovery_pages)
            if unreviewed:
                review_reason += f'；达到高清复核页数上限，候选{sorted(unreviewed)}尚需人工核验。'
        review_required = bool(review_reason)
        matches.append(PageMatch(
            table_id=card.table_id,
            table_name=card.table_name,
            pages=pages,
            score=round(confidence * 100, 1),
            evidence=reason or "VLM v2未返回可核验证据",
            review_required=review_required,
            review_reason=review_reason,
            sources={LOCATOR_MODE: pages, **sources},
            strategy_id="VLM_V2_SEMANTIC",
            table_config=config_by_id.get(card.table_id, {}),
        ))
    return VLMV2LocatorRun(
        matches=tuple(matches),
        diagnostics=pd.DataFrame(diagnostics).sort_values(
            ["批次序号", "调用序号"],
            kind="stable",
        ).reset_index(drop=True),
        model_calls=model_calls,
        page_count=sum(len(pages) for pages, _ in sheets),
    )


def _metric_prompt(
    card: VLMV2TargetCard,
    pages: Sequence[int],
    *,
    metric_codes: set[str] | None = None,
    retry_reason: str = "",
) -> str:
    sparse_response = card.table_id in {"ACTUAL_CAPITAL", "RECOGNIZED_ASSETS", "MINIMUM_CAPITAL", "OPERATING_METRICS"}
    payload = card.prompt_payload(
        metric_codes=metric_codes,
        compact=sparse_response,
    )
    retry_note = f"\n上一轮未通过校验：{retry_reason}\n请只重新核对这些失败指标。" if retry_reason else ""
    response_scope = (
        "1. 采用稀疏返回：只返回页面中明确披露的target_metrics；未披露指标不要逐条返回，"
        "系统会确定性补为not_disclosed。已披露指标严格区分：正常数值status=found；"
        "指标在原表中存在且数值格为横杠或明确数字0时，统一status=disclosed_zero，保留来源原值；"
        "仅原文明确写不适用时status=disclosed_na。"
        "源表明确列出指标但本期数值格为空时，只有能够确认该项无适用金额，才返回"
        "status=disclosed_na并保留行标签和证据；否则省略该指标。"
        "严禁推算或编造。"
        if sparse_response
        else
        "1. 每个target_metrics指标都返回一条，并严格区分：正常数值status=found；"
        "原表存在且数值为横杠或数字0时统一status=disclosed_zero，保留原值；"
        "原文明确写不适用时status=disclosed_na；"
        "报告完全没有该指标时status=not_disclosed。严禁推算或编造。"
    )
    status_values = (
        "found|disclosed_zero|disclosed_na"
        if sparse_response
        else "found|disclosed_zero|disclosed_na|not_disclosed"
    )
    if card.table_id == "OPERATING_METRICS":
        period_rule = (
            "3. 经营指标必须同时提取原表明确披露的‘本季度数/本季度（末）数’和"
            "‘本年度累计数/本年累计数/年度累计数’。同一指标两列均有披露时返回两条记录，"
            "metric_id保持相同，period_label和column_header_path分别写对应原始表头；"
            "即使第一季度两列数值完全相同，也必须分别返回。不得读取上季度、期初或预测列，"
            "不得把累计数复制成当季数，也不得把当季数复制成累计数。"
            "综合退保率若位于同页的‘流动性风险监测指标’表，必须读取‘本季度数’列，"
            "也就是指标名称右侧第一列数值；禁止读取紧邻其后的‘上季度数’列。"
        )
    elif card.table_id == 'ACTUAL_CAPITAL':
        period_rule = (
            '3. 实际资本按指标小批提取，只返回本批target_metrics，不补写其他批次。'
            '读取期末数/本季度末，不能读取期初数。核心一级、核心二级、附属一级、'
            '附属二级及实际资本合计是独立行；明确的0.00和横杠必须返回disclosed_zero，'
            '不能因金额为零省略指标。保单未来盈余的四个子项必须各有本身的原表行；'
            '只披露核心一级子项时，其他三项应省略，不能借资本类别汇总行或填报清单行次'
            '凭空补出横杠。实际资本合计不是净资产。'
            '同表跨页时沿用输入首页明确标注的原始单位；保留实际行标签和期末列表头。'
        )
    elif card.table_id == 'RECOGNIZED_ASSETS':
        period_rule = (
            '3. 认可资产只取本季度末/期末的认可价值，不能读取账面价值、非认可价值或期初列。'
            '失败页恢复时会按指标小批提取，每次只返回本批target_metrics，不要补写其他批次。'
            '同表续页没有单位时，沿用输入图片中该表首页明确标注的单位；'
            '明确的0和横杠必须返回disclosed_zero，不能因金额为零省略指标。'
            '保留实际行次、行标签以及“本季度末/期末 > 认可价值”的完整列表头路径。'
        )
    elif card.table_id == 'MINIMUM_CAPITAL':
        period_rule = (
            '3. 最低资本只取本季度数/本季度末列，不能取上季度数。'
            '这是指标分批提取，每次只返回本批target_metrics，不要补写其他批次。'
            '同表续页没有单位时，沿用输入图片中该表首页明确标注的单位，并在证据中注明单位来源页。'
            '不得把目标标准单位当作原表单位。量化风险最低资本行1与调整前行1*必须区分。'
            'QUANT_RISK_CAPITAL仅取普通量化风险最低资本（考虑特征系数后）；'
            'QUANT_RISK_CAPITAL_BEFORE_FACTOR仅取未考虑特征系数前金额。'
            '两者是独立指标，禁止把1*复制给行1或把行1复制给1*。'
            'source_label须逐字保留括号内限定语，row_header_path须保留实际行次；'
            '找不到目标行就省略，不得拿另一行代替；不得用推算或跨表数值覆盖原始读数。'
        )
    elif card.table_id == "REGISTERED_CAPITAL":
        period_rule = (
            "3. 注册资本可能位于公司基本信息、公司基本情况或公司概况，并可能以句子"
            "而不是表格披露。只读取直接标注为‘注册资本’或‘注册资本金’的当前披露金额；"
            "不得使用实收资本、股本、实际资本、核心资本或其他资本数值替代。保留金额"
            "对应的原始单位；evidence_text必须逐字包含注册资本标签、金额和紧邻单位，"
            "unit必须从这段证据读取，严禁照抄expected_unit；页面未给期间表头时"
            "period_label可留空。"
        )
    else:
        period_rule = (
            "3. current/previous/forecast等多期间并存时，选择本季度末或本报告期实际值；"
            "period_label写原列表头。"
        )
    return f"""你是保险公司偿付能力报告的视觉数据抽取专家。

目标定义：
{json.dumps(payload, ensure_ascii=False)}
页面顺序及物理页码：{list(pages)}。每张输入图片与该列表按顺序对应。
{retry_note}

请把不同公司的任意源表结构直接映射为目标指标，不要重建整张源表。
要求：
{response_scope}
2. found、disclosed_zero或disclosed_na都必须保留原始值（包括“-”“—”“--”）、原始单位、原始行标签、完整行/列表头路径及证据原文。
{period_rule}
4. 实际资本表的“净资产”对应FINANCIAL_STATEMENT_NET_ASSETS，经营指标的“净资产”对应NET_ASSETS。重复名称结合资本层级、行次和表头区分。“19万人”保留单位“万人”。
   文本指标（如前五大产品的信息）在value_raw中保留有证据的原文，不拼造数值；只看到标题不等于披露内容。附加资本合计不能取其他附加资本明细。量化风险最低资本的调整前和调整后金额分别填写。
5. page必须是输入物理页之一。认可资产表只读取本季度末的“认可价值”，不得读取账面价值、非认可价值或期初列。每条证据仅写原始行标签、列口径、数值与单位，不重复整页内容。
6. confidence为0到1。只依据图像中可见内容。
7. unit必须为原始单位，expected_unit是输出标准单位，不可照抄。金额原值不得预先缩放，换算由系统完成。
   认可资产跨页时，当前页没有单位则查同表首页上下文；仍无单位证据时unit留空，禁止默认万元。
   同时返回表头单位table_unit、单位来源物理页unit_page和逐字单位原文unit_evidence；
   表头“单位：元”或表名“S03-认可资产表（元）”都属于有效原文，后者须连表名一起引用；无法确认则留空。

只返回JSON：
{{"metrics":[{{"metric_id":"...","status":"{status_values}","value_raw":"",
"unit":"","period_label":"","source_label":"","row_header_path":[""],
"column_header_path":[""],"page":1,
"evidence_text":"","confidence":0.95}}],"table_unit":"","unit_page":0,"unit_evidence":""}}
""".strip()


def _parse_number(value: object, source_unit: object = "") -> float | None:
    chinese_yuan = chinese_money_yuan(value)
    if chinese_yuan is not None:
        return float(chinese_yuan) / _AMOUNT_UNIT_FACTORS.get(_clean(source_unit), 1)
    text = _clean(value)
    if not text or text.lower() in {"n/a", "na", "null", "none", "未披露", "不适用"}:
        return None
    # Do not fall through to extracting an unrelated Arabic substring from an
    # unsupported/malformed Chinese amount (e.g. 伍拾亿元至60亿元).
    if re.search(r'[零〇一二三四五六七八九两兩壹贰貳叁參肆伍陆陸柒捌玖]', text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[,，\s￥¥元万亿元%％]", "", text)
    cleaned = cleaned.replace("−", "-").replace("—", "-")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    number = float(match.group())
    return -abs(number) if negative else number


def _normalize_number_and_unit(
    raw_value: object,
    source_unit: object,
    target_unit: object,
    metric_id: str = "",
) -> tuple[float | None, str]:
    number = _parse_number(raw_value)
    expected = re.sub(r"\s+", "", _clean(target_unit))
    observed = canonical_unit(source_unit)
    raw_text = re.sub(r"\s+", "", _clean(raw_value))
    chinese_yuan = chinese_money_yuan(raw_value)
    if chinese_yuan is not None:
        # The literal already contains its scale (e.g. 伍拾亿元整). Do not
        # multiply again by the unit returned separately by the model.
        unit = expected or observed or '元'
        if unit not in _AMOUNT_UNIT_FACTORS:
            return None, unit
        return float(chinese_yuan) / _AMOUNT_UNIT_FACTORS[unit], unit
    literal_unit = literal_money_unit(raw_value)
    if literal_unit:
        observed = literal_unit
    if not observed:
        known_units = {**_AMOUNT_UNIT_FACTORS, **_COUNT_UNIT_FACTORS}
        observed = next(
            (unit for unit in sorted(known_units, key=len, reverse=True) if unit in raw_text),
            "%" if "%" in raw_text or "％" in raw_text else "",
        )
    if is_disclosed_zero(raw_value):
        return 0.0, expected or observed
    if number is None:
        return None, expected
    if expected in _AMOUNT_UNIT_FACTORS and observed not in _AMOUNT_UNIT_FACTORS:
        # A numeric token without a verified monetary scale is not a standard
        # amount. Never silently attach the target unit to an unscaled value.
        return None, expected
    if observed in _AMOUNT_UNIT_FACTORS and expected in _AMOUNT_UNIT_FACTORS:
        number = number * _AMOUNT_UNIT_FACTORS[observed] / _AMOUNT_UNIT_FACTORS[expected]
    elif observed in _COUNT_UNIT_FACTORS and expected in _COUNT_UNIT_FACTORS:
        number = number * _COUNT_UNIT_FACTORS[observed] / _COUNT_UNIT_FACTORS[expected]
    return number, expected or observed


def _path_text(value: object) -> str:
    if isinstance(value, list):
        return " > ".join(_clean(item) for item in value if _clean(item))
    return _clean(value)


def _normalized_model_status(item: Mapping[str, object], metric_id: str) -> str:
    status = _clean(item.get("status")).lower()
    raw_value = _clean(item.get("value_raw"))
    if status in {"found", "已找到", "找到"}:
        normalized = "found"
    elif status in {"disclosed_zero", "披露为零", "横杠为零"}:
        normalized = "disclosed_zero"
    elif status in {"disclosed_na", "披露不适用", "不适用"}:
        normalized = "disclosed_na"
    else:
        normalized = "not_disclosed"
    if normalized != 'not_disclosed' and metric_id not in TEXT_FILING_CODES and is_disclosed_zero(raw_value):
        return 'disclosed_zero'
    return normalized


def _candidate_rank(
    item: Mapping[str, object],
    metric_id: str,
    table_id: str = "",
) -> tuple[int, int, int, float, int]:
    status_rank = {
        "found": 3,
        "disclosed_zero": 3,
        "disclosed_na": 2,
        "not_disclosed": 0,
    }[_normalized_model_status(item, metric_id)]
    try:
        confidence = float(item.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    evidence_length = len(_clean(item.get("evidence_text")))
    period_rank = 1
    if table_id == "OPERATING_METRICS":
        period_label = _clean(item.get("period_label")) or _path_text(
            item.get("column_header_path")
        )
        if _period_is_invalid(period_label, table_id):
            period_rank = 0
        elif _operating_period_kind(period_label) in {"current", "cumulative"}:
            period_rank = 2
    semantic_rank = int(not quant_risk_source_error(
        metric_id, item.get('source_label'), item.get('row_header_path'), item.get('evidence_text'),
    ))
    if table_id == 'RECOGNIZED_ASSETS' and recognized_column_error(item.get('column_header_path')):
        semantic_rank = 0
    return semantic_rank, period_rank, status_rank, confidence, evidence_length


class _PartialPageFailure(RuntimeError):
    """Carry successful page results without hiding unresolved page coverage."""

    def __init__(self, pages, records, call_count, errors):
        self.timed_out = all(isinstance(error, requests.Timeout) or getattr(error, 'timed_out', False) for error in errors)
        state = '超时' if self.timed_out else '失败'
        super().__init__(f"物理页{pages}请求{state}，其他页面结果已保留；" + '；'.join(str(e)[:150] for e in errors))
        self.pages = pages
        self.records = records
        self.call_count = call_count


class _PartialMetricFailure(RuntimeError):
    """Keep completed metric chunks; retry only failed target codes."""

    def __init__(self, metric_codes, records, call_count, errors, table_name):
        details = '；'.join(dict.fromkeys(str(error)[:200] for error in errors))
        super().__init__(f'{table_name}{len(metric_codes)}个目标的请求批次失败；成功批次已保留。{details}')
        self.metric_codes = metric_codes
        self.records = records
        self.call_count = call_count
        self.timed_out = all(isinstance(error, requests.Timeout) or getattr(error, 'timed_out', False) for error in errors)


def _merge_page_records(records):
    selected = {}
    priority = {'found': 3, 'disclosed_zero': 3, 'disclosed_na': 2, 'not_disclosed': 0}
    for record in records:
        period_key = ""
        if record['目标表ID'] == 'OPERATING_METRICS':
            period_text = _record_period_text(record)
            period_key = _operating_period_kind(period_text)
            if period_key == 'other':
                period_key = period_text or 'unspecified'
        key = (record['目标表ID'], record['指标语义键'], period_key)
        previous = selected.get(key)
        def rank(row):
            return (int(not quant_risk_source_error(row['指标编码'], row.get('原始标签'),
                        row.get('行表头路径'), row.get('证据原文'))
                        and not (row['目标表ID'] == 'RECOGNIZED_ASSETS'
                                 and recognized_column_error(row.get('列表头路径')))), priority.get(row['状态'], 0))
        if previous is None or rank(record) > rank(previous):
            selected[key] = record
    return list(selected.values())


def _extract_card_records(
    pdf_bytes: bytes,
    match: PageMatch,
    card: VLMV2TargetCard,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int,
    request_max_attempts: int,
    post_func: Callable | None,
    metric_codes: set[str] | None = None,
    retry_reason: str = "",
    retry: bool = False,
    page_image_cache: Mapping[int, str] | None = None,
    max_pages_per_request: int = 4,
    split_metrics: bool = True,
    unit_context_pages: Sequence[int] | None = None,
    recognized_view_cache: Mapping | None = None,
    visible_policy_codes: frozenset[str] | None = None,
) -> tuple[list[dict[str, object]], int]:
    selected_metrics = [
        metric
        for metric in card.metrics
        if metric_codes is None or _clean(metric.get("metric_id")) in metric_codes
    ]
    pages = sorted(set(match.pages))
    if card.table_id == 'ACTUAL_CAPITAL' and visible_policy_codes is None:
        visible = visible_policy_surplus_codes(pdf_bytes, pages)
        visible_policy_codes = (
            frozenset(POLICY_SURPLUS_SOURCE_LABELS) if visible is None else visible
        )
    if page_image_cache is None:
        page_image_cache = dict(_render_page_images(
            pdf_bytes,
            pages,
            zoom=2.0,
            jpeg_quality=84,
        ))
    images = [
        (page, page_image_cache[page])
        for page in pages
        if page in page_image_cache
    ]
    context_pages = sorted(set(unit_context_pages or pages))
    asset_units = recognized_unit_contexts(pdf_bytes, context_pages) if card.table_id == 'RECOGNIZED_ASSETS' else {}
    header_page = context_pages[0] if context_pages else None
    if card.table_id == 'RECOGNIZED_ASSETS' and recognized_view_cache is None:
        recognized_view_cache = build_recognized_views(pdf_bytes, context_pages, page_image_cache)
    split_on_first_pass = card.table_id in {
        'ACTUAL_CAPITAL', 'MINIMUM_CAPITAL', 'OPERATING_METRICS',
    }
    # S03 normally stays at one request per physical page. Repeating the same
    # 49-target request is ineffective when a dense continuation page fails,
    # so only that failed page is recovered with bounded target batches.
    split_failed_recognized_page = card.table_id == 'RECOGNIZED_ASSETS' and retry
    if split_metrics and (split_on_first_pass or split_failed_recognized_page):
        # Limit output, not the table context: the continuation page may omit
        # its unit. Keep confirmed pages together and reuse their cached images.
        completed, failed_codes, errors = [], set(), []
        calls = 0
        chunk_size = {
            'OPERATING_METRICS': 8,
            'ACTUAL_CAPITAL': 10,
            'RECOGNIZED_ASSETS': 10,
        }.get(card.table_id, 12)
        numeric_metrics = [metric for metric in selected_metrics if metric['metric_id'] not in TEXT_FILING_CODES]
        chunks = []
        if card.table_id in {'ACTUAL_CAPITAL', 'RECOGNIZED_ASSETS'}:
            core_codes = _REQUIRED_CODES['ACTUAL_CAPITAL']
            if card.table_id == 'RECOGNIZED_ASSETS':
                core_codes = _REQUIRED_CODES['RECOGNIZED_ASSETS']
            core_metrics = [metric for metric in numeric_metrics if metric['metric_id'] in core_codes]
            if core_metrics:
                chunks.append(core_metrics)
            numeric_metrics = [metric for metric in numeric_metrics if metric['metric_id'] not in core_codes]
        chunks.extend(numeric_metrics[start:start + chunk_size] for start in range(0, len(numeric_metrics), chunk_size))
        # A long product description must not delay insurance revenue and the
        # other numeric operating metrics in the same model response.
        chunks.extend([metric] for metric in selected_metrics if metric['metric_id'] in TEXT_FILING_CODES)
        for chunk in chunks:
            codes = {_clean(metric['metric_id']) for metric in chunk}
            try:
                rows, count = _extract_card_records(
                    pdf_bytes, match, card, api_key=api_key, base_url=base_url,
                    model=model, timeout=timeout, request_max_attempts=request_max_attempts,
                    post_func=post_func, metric_codes=codes, retry_reason=retry_reason,
                    retry=retry, page_image_cache=page_image_cache,
                    max_pages_per_request=max_pages_per_request, split_metrics=False,
                    unit_context_pages=unit_context_pages,
                    recognized_view_cache=recognized_view_cache,
                    visible_policy_codes=visible_policy_codes,
                )
                completed.extend(rows)
                calls += count
            except (requests.RequestException, RuntimeError) as exc:
                calls += getattr(exc, 'call_count', 1)
                if isinstance(exc, _PartialPageFailure):
                    completed.extend(row for row in exc.records if row['状态'] != 'not_disclosed')
                failed_codes.update(codes)
                errors.append(exc)
        if failed_codes:
            raise _PartialMetricFailure(failed_codes, completed, calls, errors, card.table_name)
        return completed, calls
    batch_size = 1 if card.table_id == 'RECOGNIZED_ASSETS' else max(1, int(max_pages_per_request))
    failed_pages, page_errors = [], []
    raw_by_id: dict[tuple[str, str], Mapping[str, object]] = {}
    model_calls = 0
    selected_ids = {_clean(metric.get("metric_id")) for metric in selected_metrics}
    for start in range(0, len(images), batch_size):
        image_batch = images[start:start + batch_size]
        data_pages = [page for page, _ in image_batch]
        context_page = None
        focused = (recognized_view_cache or {}).get(data_pages[0]) if card.table_id == 'RECOGNIZED_ASSETS' else None
        if (card.table_id == 'RECOGNIZED_ASSETS' and int(max_pages_per_request) >= 2
                and not focused and header_page not in data_pages and header_page in page_image_cache):
            context_page = asset_units.get(data_pages[0], ('', header_page, ''))[1]
            if context_page not in page_image_cache or context_page in data_pages:
                context_page = header_page
        prompt = _metric_prompt(card, data_pages, metric_codes=metric_codes, retry_reason=retry_reason)
        request_images = [image_url for _, image_url in image_batch]
        if focused:
            request_images = [focused[0]]
            prompt += '\n' + focused[1]
        if context_page is not None:
            request_images.append(page_image_cache[context_page])
            prompt += (f'\n最后一张图片为同表物理页{context_page}的表头/单位上下文，不是本次提取数据页。'
                       f'只返回物理页{data_pages}的指标，禁止重复提取上下文页金额；可引用其单位与表头。')
        model_calls += 1
        try:
            response = _call_vlm_json(
                prompt=prompt,
                image_urls=request_images,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
                max_attempts=request_max_attempts,
                post_func=post_func,
            )
            items = response.get('metrics')
            if card.table_id in {'ACTUAL_CAPITAL', 'RECOGNIZED_ASSETS', 'MINIMUM_CAPITAL', 'OPERATING_METRICS'} and (
                not isinstance(items, list) or any(not isinstance(item, Mapping) for item in items)
            ):
                raise RuntimeError(f'{card.table_name}提取响应缺少有效的metrics列表，不能判为未披露。')
        except (requests.RequestException, RuntimeError) as exc:
            if card.table_id not in {'RECOGNIZED_ASSETS', 'ACTUAL_CAPITAL'}:
                raise
            failed_pages.extend(data_pages)
            page_errors.append(exc)
            continue
        for item in response.get("metrics", []):
            if not isinstance(item, Mapping):
                continue
            metric_id = _clean(item.get("metric_id"))
            if metric_id not in selected_ids:
                continue
            if card.table_id == 'ACTUAL_CAPITAL' and metric_id in POLICY_SURPLUS_SOURCE_LABELS:
                evidence = _clean(item.get('evidence_text'))
                if (
                    (visible_policy_codes is not None and metric_id not in visible_policy_codes)
                    or any(term in evidence for term in ('未单独列示', '未列示', '未披露'))
                ):
                    continue
            if card.table_id == 'OPERATING_METRICS':
                period_text = _clean(item.get("period_label")) or _path_text(
                    item.get("column_header_path")
                )
                if _period_is_invalid(period_text, card.table_id):
                    # Historical columns are never valid substitutes for the
                    # requested current/cumulative operating periods.
                    continue
            if card.table_id == 'RECOGNIZED_ASSETS':
                # Context pages may not contribute duplicate source values.
                try:
                    item_page = int(item.get('page', 0))
                except (TypeError, ValueError):
                    continue
                if item_page not in data_pages:
                    continue
                # Wrong or unproven S03 columns never enter the extracted
                # dataset; validation retry can request the target again.
                if recognized_column_error(item.get('column_header_path')):
                    continue
                context = asset_units.get(item_page) or model_unit_context(response, data_pages + ([context_page] if context_page else []))
                # A model-supplied row unit without literal amount, row label,
                # or table-header evidence cannot be trusted as a source scale.
                # Clearing it makes NUMBER validation block Step 3 and trigger
                # a focused retry instead of silently accepting a 10,000x error.
                item = apply_unit_context(item, context, require_explicit=True)
                if focused:
                    item['evidence_text'] = str(item.get('evidence_text') or '') + '；原表网格认可价值专列视图核对'
            elif card.table_id == 'REGISTERED_CAPITAL':
                # Registration capital is a single amount whose scale is often
                # written inline. Require that literal evidence to prevent the
                # target standard unit (万元) from being echoed as the source.
                item = apply_item_unit_evidence(item, require_explicit=True)
            storage_key = _candidate_storage_key(card.table_id, metric_id, item)
            previous = raw_by_id.get(storage_key)
            if previous is None or _candidate_rank(
                item, metric_id, card.table_id,
            ) > _candidate_rank(previous, metric_id, card.table_id):
                raw_by_id[storage_key] = item

    if card.table_id == 'OPERATING_METRICS' and 'INSURANCE_REVENUE' in selected_ids:
        revenue_key = ('INSURANCE_REVENUE', 'current')
        revenue = raw_by_id.get(revenue_key, {})
        if _normalized_model_status(revenue, 'INSURANCE_REVENUE') == 'not_disclosed':
            # A continued operating table can start with the required revenue
            # row while its section title and unit remain on the prior page.
            # Recover only an exact source row/current-quarter cell and an
            # explicit source unit; never use the annual cumulative column.
            recovered = recover_insurance_revenue(pdf_bytes, pages)
            if recovered is not None:
                raw_by_id[revenue_key] = recovered

    if card.table_id == 'OPERATING_METRICS' and 'SURRENDER_RATE' in selected_ids:
        surrender_key = ('SURRENDER_RATE', 'current')
        surrender = raw_by_id.get(surrender_key, {})
        if _normalized_model_status(surrender, 'SURRENDER_RATE') == 'not_disclosed':
            recovered = recover_surrender_rate(pdf_bytes, pages)
            if recovered is not None:
                raw_by_id[surrender_key] = recovered

    records: list[dict[str, object]] = []
    for metric in selected_metrics:
        metric_id = _clean(metric.get("metric_id"))
        metric_items = [
            (period_key, item)
            for (stored_metric_id, period_key), item in raw_by_id.items()
            if stored_metric_id == metric_id
        ]
        if not metric_items:
            metric_items = [("", {})]
        metric_items.sort(key=lambda pair: {
            "current": 0, "cumulative": 1,
        }.get(pair[0], 2))
        for _, item in metric_items:
            status = _normalized_model_status(item, metric_id)
            raw_value = _clean(item.get("value_raw"))
            if status == "not_disclosed":
                raw_value = ""
            source_unit = _clean(item.get("unit"))
            standard_number, standard_unit = _normalize_number_and_unit(
                raw_value,
                source_unit,
                metric.get("expected_unit"),
                metric_id,
            )
            if metric_id in TEXT_FILING_CODES:
                standard_number = None
            try:
                page = int(item.get("page", 0) or 0)
            except (TypeError, ValueError):
                page = 0
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            records.append({
                "目标表ID": card.table_id,
                "目标表名称": card.table_name,
                "指标编码": metric_id,
                "指标名称": _clean(metric.get("name")),
                "指标语义键": _clean(metric.get("semantic_key")) or metric_id,
                "状态": status,
                "原始值": raw_value,
                "数值": 0.0 if status == 'disclosed_zero' else _parse_number(raw_value, source_unit),
                "单位": source_unit,
                "标准数值": standard_number,
                "标准单位": standard_unit,
                "期间口径": _clean(item.get("period_label")),
                "原始标签": _clean(item.get("source_label")),
                "行表头路径": _path_text(item.get("row_header_path")),
                "列表头路径": _path_text(item.get("column_header_path")),
                "物理页码": page,
                "证据原文": _clean(item.get("evidence_text")),
                "置信度": max(0.0, min(confidence, 1.0)),
                "提取轮次": _clean(item.get("recovery_mode")) or (RETRY_MODE if retry else EXTRACTION_MODE),
            })
    if failed_pages:
        raise _PartialPageFailure(failed_pages, records, model_calls, page_errors)
    return records, model_calls


def _validation_row(
    check_id: str,
    table_id: str,
    metric_id: str,
    status: str,
    rule: str,
    actual: object,
    expected: object,
    message: str,
) -> dict[str, object]:
    return dict(zip(VALIDATION_COLUMNS, (
        check_id, table_id, metric_id, status, rule, actual, expected, message,
    )))


def _cross_table_check(semantic_key, group):
    """Use a strict baseline or verified source rounding; never relative error."""
    values = sorted({float(value) for value in group['标准数值'].dropna()})
    units = {_clean(value) for value in group['标准单位']}
    unit = next(iter(units)) if len(units) == 1 else ''
    # Currency differences up to RMB 1; counts must agree exactly. Other
    # dimensionless values get only a 1e-6 absolute precision allowance.
    tolerance = (1.0 / _AMOUNT_UNIT_FACTORS[unit] if unit in _AMOUNT_UNIT_FACTORS
                 else 0.0 if unit in _COUNT_UNIT_FACTORS else 1e-6)
    finite = bool(values) and all(math.isfinite(value) for value in values)
    difference = values[-1] - values[0] if finite else float('inf')
    float_noise = min(2 * max((math.ulp(value) for value in values), default=0.0),
                      tolerance * 1e-6) if finite else 0.0
    passed = finite and len(units) == 1 and difference <= tolerance + float_noise
    rounded = False
    if not passed and finite and len(units) == 1 and unit in _AMOUNT_UNIT_FACTORS:
        rows = group.to_dict('records')
        passed = all(
            abs(float(left['标准数值']) - float(right['标准数值'])) <= tolerance + float_noise
            or source_rounding_agrees(left, right, unit)
            for i, left in enumerate(rows) for right in rows[i + 1:]
        )
        rounded = passed
    detail = (f'最大差额{difference:.12g}{unit}，容差{tolerance:.12g}{unit}；'
              + ('精度差异可忽略，保留各来源原值。' if passed else '超过容差或标准单位不一致，请核验。'))
    expected = f'差额≤{tolerance:.12g}{unit}'
    if rounded:
        expected = '按原文金额单位及小数位四舍五入后逐对一致'
        detail = (f'最大差额{difference:.12g}{unit}；按原文披露精度对齐后逐对一致，'
                  '精度差异可忽略，保留各来源原值及标准数值。')
    return _validation_row(
        f'CROSS:{semantic_key}:CONFLICT', '跨表', str(semantic_key),
        '通过' if passed else '失败', '跨表一致性', ', '.join(map(str, values)),
        expected, detail,
    )


def _validation_semantic_key(row: Mapping[str, object], semantic_column: str) -> str:
    semantic_key = _clean(row.get(semantic_column))
    if _clean(row.get('目标表ID')) != 'OPERATING_METRICS':
        return semantic_key
    period_kind = _operating_period_kind(_record_period_text(row))
    if period_kind in {'current', 'cumulative'}:
        return f'{semantic_key}::{period_kind.upper()}'
    return semantic_key


def refresh_vlm_v2_cross_table_checks(extraction_run):
    """Repair verified Chinese amounts and saved rounding checks only."""
    if extraction_run is None or extraction_run.records.empty:
        return extraction_run
    records = extraction_run.records.copy()
    validations = extraction_run.validations.copy()
    if validations.empty:
        validations = pd.DataFrame(columns=VALIDATION_COLUMNS)
    repaired = set()
    for index, row in records.iterrows():
        if (row['状态'] != 'found' or pd.notna(row['标准数值'])
                or row['指标编码'] in TEXT_FILING_CODES
                or chinese_money_yuan(row['原始值']) is None
                or _clean(row['标准单位']) not in _AMOUNT_UNIT_FACTORS):
            continue
        number, unit = _normalize_number_and_unit(row['原始值'], row['单位'], row['标准单位'])
        records.loc[index, '标准数值'] = number
        records.loc[index, '数值'] = _parse_number(row['原始值'], row['单位'])
        repaired.add((row['目标表ID'], row['指标编码']))
    for table_id, metric_id in repaired:
        # Do not clear unrelated request, evidence, period or business failures.
        matching = records[records['目标表ID'].eq(table_id) & records['指标编码'].eq(metric_id)
                           & records['状态'].isin({'found', 'disclosed_zero'})]
        if matching['标准数值'].isna().any():
            continue
        mask = (validations['校验ID'].eq(f'{table_id}:{metric_id}:NUMBER')
                & validations['规则'].eq('数值可解析'))
        validations.loc[mask, '状态'] = '通过'
        validations.loc[mask, '说明'] = '已按原文中文金额解析并换算为标准单位，保留原值。'
    semantic_column = '指标语义键' if '指标语义键' in records else '指标编码'
    validation_keys = records.apply(
        lambda row: _validation_semantic_key(row, semantic_column), axis=1,
    )
    # A newly parseable saved amount can introduce a conflict with another
    # source. Do not merely clear NUMBER without checking that pair as well.
    for table_id, metric_id in repaired:
        repaired_mask = records['目标表ID'].eq(table_id) & records['指标编码'].eq(metric_id)
        keys = validation_keys.loc[repaired_mask]
        for semantic_key in keys.unique():
            group = records[validation_keys.eq(semantic_key)
                            & records['状态'].isin({'found', 'disclosed_zero'})
                            & records['标准数值'].notna()]
            check_id = f'CROSS:{semantic_key}:CONFLICT'
            if len(group) >= 2 and not validations['校验ID'].eq(check_id).any():
                validations = pd.concat([validations, pd.DataFrame([
                    _cross_table_check(semantic_key, group)
                ], columns=VALIDATION_COLUMNS)], ignore_index=True)
    for index, check in validations.iterrows():
        check_id = str(check.get('校验ID', ''))
        if check.get('规则') != '跨表一致性' or not (check_id.startswith('CROSS:') and check_id.endswith(':CONFLICT')):
            continue
        semantic_key = check_id[len('CROSS:'):-len(':CONFLICT')]
        group = records[
            validation_keys.eq(semantic_key)
            & records['状态'].isin({'found', 'disclosed_zero'})
            & records['标准数值'].notna()
            & ~records['指标编码'].isin(TEXT_FILING_CODES)
        ]
        if len(group) >= 2:
            validations.loc[index, VALIDATION_COLUMNS] = _cross_table_check(semantic_key, group)
    validations = _refresh_source_semantic_checks(records, validations)
    if validations.equals(extraction_run.validations) and records.equals(extraction_run.records):
        return extraction_run
    return replace(extraction_run, records=records, validations=validations)


def _source_semantic_checks(records):
    checks = []
    for _, row in records.iterrows():
        code = _clean(row['指标编码'])
        if row['目标表ID'] == 'RECOGNIZED_ASSETS' and row['状态'] in {'found', 'disclosed_zero', 'disclosed_na'}:
            reason = recognized_column_error(row.get('列表头路径'))
            checks.append(_validation_row(
                f"{row['目标表ID']}:{code}:SOURCE_SEMANTICS", row['目标表ID'], code,
                '失败' if reason else '通过', '来源指标语义', row.get('列表头路径', ''),
                '本期/期末认可价值（非账面、非认可或期初列）', reason,
            ))
        if code not in QUANT_RISK_RULES or row['状态'] not in {'found', 'disclosed_zero', 'disclosed_na'}:
            continue
        reason = quant_risk_source_error(code, row.get('原始标签'), row.get('行表头路径'), row.get('证据原文'))
        checks.append(_validation_row(
            f"{row['目标表ID']}:{code}:SOURCE_SEMANTICS", row['目标表ID'], code,
            '失败' if reason else '通过', '来源指标语义', row.get('原始标签', ''),
            QUANT_RISK_RULES[code], reason,
        ))
    return checks


def _refresh_source_semantic_checks(records, validations):
    """Apply the same guard to saved runs/STEP2 uploads without changing amounts."""
    checks = _source_semantic_checks(records)
    if not checks:
        return validations
    ids = {check['校验ID'] for check in checks}
    keep = validations[~(validations['规则'].eq('来源指标语义') & validations['校验ID'].isin(ids))]
    return pd.concat([keep, pd.DataFrame(checks, columns=VALIDATION_COLUMNS)], ignore_index=True)


def validate_vlm_v2_metrics(
    records: pd.DataFrame,
    cards: Sequence[VLMV2TargetCard],
) -> pd.DataFrame:
    """Apply deterministic evidence, typing, coverage, and solvency equation checks."""
    checks: list[dict[str, object]] = []
    if records.empty:
        return pd.DataFrame([
            _validation_row(
                "NO_RECORDS", "", "", "失败", "结果非空", 0, ">0", "未生成任何指标记录。",
            )
        ], columns=VALIDATION_COLUMNS)

    metric_meta = {
        (card.table_id, _clean(metric.get("metric_id"))): metric
        for card in cards
        for metric in card.metrics
    }
    disclosed = records[records["状态"].isin({"found", "disclosed_zero", "disclosed_na"})].copy()
    found = disclosed[
        records.loc[disclosed.index, "状态"].isin({"found", "disclosed_zero"})
        & pd.notna(disclosed["标准数值"])
        & ~disclosed['指标编码'].isin(TEXT_FILING_CODES)
    ].copy()
    for _, row in disclosed.iterrows():
        table_id, metric_id = _clean(row["目标表ID"]), _clean(row["指标编码"])
        meta = metric_meta.get((table_id, metric_id), {})
        page_ok = int(row["物理页码"] or 0) > 0
        evidence_ok = bool(_clean(row["证据原文"]) and _clean(row["原始标签"]))
        numeric_expected = _clean(meta.get("数据类型") or meta.get("data_type")) in {
            "金额", "比例", "数量", "数值",
        }
        numeric_ok = (
            True if row["状态"] == "disclosed_na"
            else pd.notna(row["标准数值"]) if numeric_expected
            else True
        )
        for suffix, ok, rule, message in (
            ("PAGE", page_ok, "来源页码", "缺少有效来源页码。"),
            ("EVIDENCE", evidence_ok, "证据完整性", "缺少原始标签或证据原文。"),
            ("NUMBER", numeric_ok, "数值可解析", "原始值无法解析，或来源单位缺失/不受支持，无法可靠换算为标准数值。"),
        ):
            checks.append(_validation_row(
                f"{table_id}:{metric_id}:{suffix}", table_id, metric_id,
                "通过" if ok else "失败", rule, row["原始值"], "可核验", "" if ok else message,
            ))
        period_label = _record_period_text(row)
        if _period_is_invalid(period_label, table_id):
            expected_period = (
                "本季度数/本季度（末）数/本年度累计数/本年累计数"
                if table_id == "OPERATING_METRICS"
                else "本季度数/本季度（末）数/本报告期实际列"
            )
            checks.append(_validation_row(
                f"{table_id}:{metric_id}:PERIOD",
                table_id,
                metric_id,
                "失败",
                "本季度期间口径",
                period_label,
                expected_period,
                f"选中了不允许的期间列“{period_label}”，请改取本报告期允许的列。",
            ))

    for card in cards:
        card_rows = records[records["目标表ID"] == card.table_id]
        for metric_id in sorted(_REQUIRED_CODES.get(card.table_id, set())):
            if metric_id not in {str(metric["metric_id"]) for metric in card.metrics}:
                continue
            rows = card_rows[card_rows["指标编码"] == metric_id]
            covered_statuses = {"found", "disclosed_zero", "disclosed_na"}
            ok = not rows.empty and bool(rows["状态"].isin(covered_statuses).any())
            checks.append(_validation_row(
                f"{card.table_id}:{metric_id}:REQUIRED",
                card.table_id,
                metric_id,
                "通过" if ok else "失败",
                "目标覆盖",
                "found/disclosed_zero/disclosed_na" if ok else "not_disclosed",
                "明确披露（含不适用）",
                "" if ok else "必需目标指标未找到。",
            ))

    best: dict[str, pd.Series] = {}
    semantic_column = "指标语义键" if "指标语义键" in found.columns else "指标编码"
    found["_校验语义键"] = found.apply(
        lambda row: _validation_semantic_key(row, semantic_column), axis=1,
    )
    for semantic_key, group in found.groupby("_校验语义键"):
        ordered = group.sort_values("置信度", ascending=False)
        metric_id = _clean(ordered.iloc[0]["指标编码"])
        if not str(semantic_key).endswith(":THREE_YEAR_AVERAGE"):
            best[metric_id] = ordered.iloc[0]
        values = {
            float(value)
            for value in group["标准数值"].dropna().tolist()
        }
        if len(values) > 1:
            checks.append(_cross_table_check(semantic_key, group))

    def value(code: str) -> float | None:
        row = best.get(code)
        if row is None or pd.isna(row["标准数值"]):
            return None
        return float(row["标准数值"])

    financial_assets = value("FINANCIAL_STATEMENT_ASSETS")
    financial_liabilities = value("FINANCIAL_STATEMENT_LIABILITIES")
    financial_net_assets = value("FINANCIAL_STATEMENT_NET_ASSETS")
    if financial_assets is not None and financial_liabilities is not None:
        net_assets_found = financial_net_assets is not None
        checks.append(_validation_row(
            "ACTUAL_CAPITAL:FINANCIAL_STATEMENT_NET_ASSETS:CONDITIONAL",
            "ACTUAL_CAPITAL",
            "FINANCIAL_STATEMENT_NET_ASSETS",
            "通过" if net_assets_found else "失败",
            "条件目标覆盖",
            financial_net_assets if net_assets_found else "not_disclosed",
            "当财务报表资产和负债均披露时必须提取净资产",
            "" if net_assets_found else "已找到财务报表资产和负债，但漏提净资产行。",
        ))
        if net_assets_found:
            expected_net_assets = financial_assets - financial_liabilities
            tolerance = max(0.01, abs(expected_net_assets) * 0.005)
            net_assets_ok = abs(financial_net_assets - expected_net_assets) <= tolerance
            checks.append(_validation_row(
                "CROSS:FINANCIAL_NET_ASSETS",
                "跨表",
                "FINANCIAL_STATEMENT_NET_ASSETS|FINANCIAL_STATEMENT_ASSETS|FINANCIAL_STATEMENT_LIABILITIES",
                "通过" if net_assets_ok else "失败",
                "财务报表净资产=资产-负债",
                financial_net_assets,
                round(expected_net_assets, 6),
                "" if net_assets_ok else f"差异超过容差{tolerance:.4g}。",
            ))

    formulas = [
        (
            "MINIMUM_COMPONENTS",
            ("MINIMUM_CAPITAL", "QUANT_RISK_CAPITAL", "CONTROL_RISK_CAPITAL", "ADDITIONAL_CAPITAL"),
            lambda v: v[1] + v[2] + v[3],
            "最低资本=量化风险最低资本+控制风险最低资本+附加资本",
        ),
        (
            "ACTUAL_COMPONENTS",
            ("ACTUAL_CAPITAL", "CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "ANC_T1_CAPITAL", "ANC_T2_CAPITAL"),
            lambda v: v[1] + v[2] + v[3] + v[4],
            "实际资本=核心一级+核心二级+附属一级+附属二级资本",
        ),
        (
            "ASSET_LIABILITY",
            ("ACTUAL_CAPITAL", "RECOGNIZED_ASSETS", "RECOGNIZED_LIABILITIES"),
            lambda v: v[1] - v[2],
            "实际资本=认可资产-认可负债",
        ),
        (
            "COMBINED_RATIO",
            ("COMBINED_SOLVENCY_RATIO", "ACTUAL_CAPITAL", "MINIMUM_CAPITAL"),
            lambda v: v[1] / v[2] * 100 if v[2] else math.nan,
            "综合偿付能力充足率=实际资本/最低资本",
        ),
        (
            "CORE_RATIO",
            ("CORE_SOLVENCY_RATIO", "CORE_T1_CAPITAL", "CORE_T2_CAPITAL", "MINIMUM_CAPITAL"),
            lambda v: (v[1] + v[2]) / v[3] * 100 if v[3] else math.nan,
            "核心偿付能力充足率=(核心一级+核心二级资本)/最低资本",
        ),
    ]
    for check_id, codes, calculate, label in formulas:
        values = [value(code) for code in codes]
        if any(item is None for item in values):
            continue
        expected = float(calculate(values))
        actual = float(values[0])
        tolerance = max(0.5, abs(expected) * 0.005)
        ok = math.isfinite(expected) and abs(actual - expected) <= tolerance
        checks.append(_validation_row(
            f"CROSS:{check_id}",
            "跨表",
            "|".join(codes),
            "通过" if ok else "失败",
            label,
            actual,
            round(expected, 6),
            "" if ok else f"差异超过容差{tolerance:.4g}。",
        ))
    checks.extend(_source_semantic_checks(records))
    return pd.DataFrame(checks, columns=VALIDATION_COLUMNS)


def _retry_plan(
    validations: pd.DataFrame,
    records: pd.DataFrame,
) -> dict[str, set[str]]:
    failures = validations[validations["状态"] == "失败"]
    semantic_failures = failures[failures['规则'].eq('来源指标语义')]
    semantic_codes = set(semantic_failures['指标编码'])
    plan: dict[str, set[str]] = {}
    for _, failure in failures.iterrows():
        table_id = _clean(failure["目标表ID"])
        codes = [item for item in _clean(failure["指标编码"]).split("|") if item]
        if table_id and table_id not in {"跨表"}:
            plan.setdefault(table_id, set()).update(codes)
            continue
        if set(codes) & semantic_codes:
            # Fix the directly identified wrong row first, rather than reread
            # otherwise valid summary/formula inputs to agree with that row.
            continue
        for code in codes:
            source_rows = records[records["指标编码"] == code]
            for source_table in source_rows["目标表ID"].dropna().astype(str).unique():
                plan.setdefault(source_table, set()).add(code)
    return plan


def _notify_progress(
    callback: Callable[[Mapping[str, object]], None] | None,
    started_at: float,
    phase: str,
    **details: object,
) -> None:
    if callback is None:
        return
    event = {
        "phase": phase,
        "elapsed_seconds": max(0.0, perf_counter() - started_at),
        **details,
    }
    try:
        callback(event)
    except Exception:
        # Progress reporting must never interrupt extraction or validation.
        return


def extract_metrics_vlm_v2(
    pdf_bytes: bytes,
    matches: Sequence[PageMatch],
    table_configs: Sequence[Mapping[str, object]],
    taxonomy: pd.DataFrame,
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout: int = 90,
    request_max_attempts: int = 2,
    post_func: Callable | None = None,
    auto_retry: bool = True,
    max_workers: int = 3,
    retry_max_workers: int = 2,
    max_pages_per_request: int = 4,
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
) -> VLMV2ExtractionRun:
    """Extract metrics while isolating table failures and retrying bounded work."""
    started_at = perf_counter()
    if not all((_clean(api_key), _clean(base_url), _clean(model))):
        raise ValueError("请完整填写模型接口地址、模型名称和API Key。")
    cards = build_vlm_v2_target_cards(table_configs, taxonomy)
    card_by_id = {card.table_id: card for card in cards}
    active = [
        match for match in matches
        if match.pages and match.table_id in card_by_id
    ]
    records: list[dict[str, object]] = []
    logs: list[str] = []
    unique_pages = sorted({page for match in active for page in match.pages})
    _notify_progress(
        progress_callback,
        started_at,
        "rendering_started",
        page_count=len(unique_pages),
        total_tables=len(active),
    )
    page_image_cache = dict(_render_page_images(
        pdf_bytes,
        unique_pages,
        zoom=2.0,
        jpeg_quality=84,
    )) if unique_pages else {}
    asset_pages = sorted({page for match in active if match.table_id == 'RECOGNIZED_ASSETS' for page in match.pages})
    recognized_view_cache = build_recognized_views(pdf_bytes, asset_pages, page_image_cache) if asset_pages else {}
    _notify_progress(
        progress_callback,
        started_at,
        "rendering_completed",
        page_count=len(page_image_cache),
        total_tables=len(active),
    )

    request_attempts = min(2, max(1, int(request_max_attempts)))
    page_batch_size = max(1, int(max_pages_per_request))
    worker_count = min(max(1, int(max_workers)), max(1, len(active)))
    initial_calls = 0
    retry_calls = 0
    unavailable_table_ids: set[str] = set()
    request_errors: dict[str, str] = {}
    _notify_progress(
        progress_callback,
        started_at,
        "initial_started",
        completed_tables=0,
        total_tables=len(active),
    )
    def run_table_job(
        match: PageMatch,
        *,
        metric_codes: set[str] | None = None,
        retry_reason: str = "",
        retry: bool = False,
    ) -> dict[str, object]:
        try:
            result, call_count = _extract_card_records(
                pdf_bytes,
                match,
                card_by_id[match.table_id],
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout=timeout,
                # Retries are scheduled after the whole round so a slow table
                # never holds up or discards independent successful tables.
                request_max_attempts=1,
                post_func=post_func,
                metric_codes=metric_codes,
                retry_reason=retry_reason,
                retry=retry,
                page_image_cache=page_image_cache,
                max_pages_per_request=page_batch_size,
                unit_context_pages=next((item.pages for item in active if item.table_id == match.table_id), match.pages),
                recognized_view_cache=recognized_view_cache,
            )
            return {
                "match": match,
                "records": result,
                "call_count": call_count,
                "error": None,
                "timed_out": False,
            }
        except Exception as exc:
            return {
                "match": replace(match, pages=exc.pages) if isinstance(exc, _PartialPageFailure) else match,
                "records": exc.records if isinstance(exc, (_PartialPageFailure, _PartialMetricFailure)) else [],
                "retry_codes": exc.metric_codes if isinstance(exc, _PartialMetricFailure) else None,
                "retryable": isinstance(exc, (requests.Timeout, _PartialPageFailure, _PartialMetricFailure)),
                # A failed job made at least one model request. The exact count
                # can be lower than the number of page batches when it fails early.
                "call_count": exc.call_count if isinstance(exc, (_PartialPageFailure, _PartialMetricFailure)) else 1,
                "error": exc,
                "timed_out": isinstance(exc, requests.Timeout) or getattr(exc, 'timed_out', False),
            }

    timed_out_jobs: list[dict[str, object]] = []
    if active:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    run_table_job,
                    match,
                ): match
                for match in active
            }
            for completed_tables, future in enumerate(as_completed(futures), start=1):
                match = futures[future]
                job = future.result()
                initial_calls += int(job["call_count"])
                error = job["error"]
                if error is not None:
                    records.extend(job['records'])
                    unavailable_table_ids.add(match.table_id)
                    request_errors[match.table_id] = _clean(error)[:500]
                    if bool(job.get('retryable')):
                        timed_out_jobs.append(job)
                    status = "超时" if job["timed_out"] else "失败"
                    logs.append(f"{match.table_name}：首轮{status}，{_clean(error)[:300]}")
                    _notify_progress(
                        progress_callback,
                        started_at,
                        "initial_failed",
                        completed_tables=completed_tables,
                        total_tables=len(active),
                        table_id=match.table_id,
                        table_name=match.table_name,
                        status=status,
                        error=_clean(error)[:500],
                        model_calls=initial_calls,
                    )
                    continue
                result = list(job["records"])
                records.extend(result)
                found_count = sum(
                    item["状态"] in {"found", "disclosed_zero"}
                    for item in result
                )
                logs.append(
                    f"{match.table_name}：直提{found_count}/{len(result)}个目标指标。"
                )
                _notify_progress(
                    progress_callback,
                    started_at,
                    "initial_completed",
                    completed_tables=completed_tables,
                    total_tables=len(active),
                    table_id=match.table_id,
                    table_name=match.table_name,
                    model_calls=initial_calls,
                )

    if request_attempts > 1 and timed_out_jobs:
        retry_names = [str(job["match"].table_name) for job in timed_out_jobs]
        _notify_progress(
            progress_callback,
            started_at,
            "timeout_retry_started",
            retry_total=len(timed_out_jobs),
            retry_table_names=retry_names,
        )
        timeout_workers = min(
            max(1, int(retry_max_workers)),
            len(timed_out_jobs),
        )
        with ThreadPoolExecutor(max_workers=timeout_workers) as executor:
            futures = {
                executor.submit(run_table_job, job["match"], metric_codes=job.get('retry_codes'), retry=True): job["match"]
                for job in timed_out_jobs
            }
            for completed_retries, future in enumerate(as_completed(futures), start=1):
                match = futures[future]
                job = future.result()
                retry_calls += int(job["call_count"])
                error = job["error"]
                if error is not None:
                    records.extend(job['records'])
                    request_errors[match.table_id] = _clean(error)[:500]
                    status = "超时" if job["timed_out"] else "失败"
                    logs.append(
                        f"{match.table_name}：失败请求批次重试仍{status}，"
                        f"已保留其他成功表；{_clean(error)[:300]}"
                    )
                    _notify_progress(
                        progress_callback,
                        started_at,
                        "timeout_retry_failed",
                        completed_retries=completed_retries,
                        retry_total=len(timed_out_jobs),
                        table_id=match.table_id,
                        table_name=match.table_name,
                        status=status,
                        error=_clean(error)[:500],
                        model_calls=initial_calls + retry_calls,
                    )
                    continue
                unavailable_table_ids.discard(match.table_id)
                result = list(job["records"])
                records.extend(result)
                found_count = sum(
                    item["状态"] in {"found", "disclosed_zero"}
                    for item in result
                )
                logs.append(
                    f"{match.table_name}：失败请求批次重试成功，"
                    f"直提{found_count}/{len(result)}个目标指标。"
                )
                _notify_progress(
                    progress_callback,
                    started_at,
                    "timeout_retry_completed",
                    completed_retries=completed_retries,
                    retry_total=len(timed_out_jobs),
                    table_id=match.table_id,
                    table_name=match.table_name,
                    model_calls=initial_calls + retry_calls,
                )
    frame = pd.DataFrame(_merge_page_records(records), columns=METRIC_COLUMNS)
    validations = validate_vlm_v2_metrics(frame, cards)

    if auto_retry and not frame.empty:
        plan = _retry_plan(validations, frame)
        match_by_id = {match.table_id: match for match in active}
        retry_jobs: list[tuple[str, PageMatch, VLMV2TargetCard, set[str], str]] = []
        for table_id, metric_codes in plan.items():
            if table_id in unavailable_table_ids:
                continue
            match = match_by_id.get(table_id)
            card = card_by_id.get(table_id)
            valid_codes = {
                str(metric["metric_id"])
                for metric in card.metrics
            } if card else set()
            metric_codes = set(metric_codes) & valid_codes
            if not match or not card or not metric_codes:
                continue
            reasons = validations[
                (validations["状态"] == "失败")
                & validations["指标编码"].astype(str).map(
                    lambda value: bool(set(value.split("|")) & metric_codes)
                )
            ]["说明"].dropna().astype(str).tolist()
            retry_jobs.append((
                table_id,
                match,
                card,
                metric_codes,
                "；".join(dict.fromkeys(reasons)) or "确定性校验未通过",
            ))

        _notify_progress(
            progress_callback,
            started_at,
            "validation_completed",
            retry_total=len(retry_jobs),
            retry_table_names=[job[2].table_name for job in retry_jobs],
        )
        if retry_jobs:
            _notify_progress(
                progress_callback,
                started_at,
                "retry_started",
                completed_retries=0,
                retry_total=len(retry_jobs),
                retry_table_names=[job[2].table_name for job in retry_jobs],
            )
            retry_workers = min(
                max(1, int(retry_max_workers)),
                len(retry_jobs),
            )
            with ThreadPoolExecutor(max_workers=retry_workers) as executor:
                futures = {
                    executor.submit(
                        _extract_card_records,
                        pdf_bytes,
                        match,
                        card,
                        api_key=api_key,
                        base_url=base_url,
                        model=model,
                        timeout=timeout,
                        request_max_attempts=1,
                        post_func=post_func,
                        metric_codes=metric_codes,
                        retry_reason=reason,
                        retry=True,
                        page_image_cache=page_image_cache,
                        recognized_view_cache=recognized_view_cache,
                        max_pages_per_request=page_batch_size,
                    ): (table_id, match, card, metric_codes)
                    for table_id, match, card, metric_codes, reason in retry_jobs
                }
                for completed_retries, future in enumerate(as_completed(futures), start=1):
                    table_id, match, card, metric_codes = futures[future]
                    try:
                        retried, call_count = future.result()
                    except Exception as exc:
                        retry_calls += exc.call_count if isinstance(exc, (_PartialMetricFailure, _PartialPageFailure)) else 1
                        if isinstance(exc, _PartialPageFailure):
                            # Failed pages remain a coverage blocker. Only valid
                            # disclosed corrections from completed pages replace old rows.
                            unavailable_table_ids.add(table_id)
                            request_errors[table_id] = _clean(exc)[:500]
                            corrections = [row for row in exc.records if row['状态'] != 'not_disclosed'
                                           and not recognized_column_error(row.get('列表头路径'))]
                            successful_codes = {row['指标编码'] for row in corrections}
                            frame = frame[~((frame['目标表ID'] == table_id) & frame['指标编码'].isin(successful_codes))]
                            frame = pd.DataFrame(frame.to_dict('records') + corrections, columns=METRIC_COLUMNS)
                        if isinstance(exc, _PartialMetricFailure) and exc.records:
                            # Successful corrections replace only their own codes.
                            successful_codes = {row['指标编码'] for row in exc.records}
                            frame = frame[~((frame['目标表ID'] == table_id) & frame['指标编码'].isin(successful_codes))]
                            frame = pd.DataFrame(frame.to_dict('records') + exc.records, columns=METRIC_COLUMNS)
                        status = "超时" if isinstance(exc, requests.Timeout) or getattr(exc, 'timed_out', False) else "失败"
                        logs.append(
                            f"{card.table_name}：失败指标定向重试{status}，"
                            f"已保留首轮结果；{_clean(exc)[:300]}"
                        )
                        _notify_progress(
                            progress_callback,
                            started_at,
                            "retry_failed",
                            completed_retries=completed_retries,
                            retry_total=len(retry_jobs),
                            table_id=table_id,
                            table_name=card.table_name,
                            status=status,
                            error=_clean(exc)[:500],
                            model_calls=initial_calls + retry_calls,
                        )
                        continue
                    retry_calls += call_count
                    frame = frame[
                        ~(
                            (frame["目标表ID"] == table_id)
                            & frame["指标编码"].isin(metric_codes)
                        )
                    ]
                    retry_frame = pd.DataFrame(retried, columns=METRIC_COLUMNS)
                    frame = (
                        retry_frame.reset_index(drop=True)
                        if frame.empty
                        else pd.DataFrame(
                            frame.to_dict("records") + retry_frame.to_dict("records"),
                            columns=METRIC_COLUMNS,
                        )
                    )
                    logs.append(
                        f"{card.table_name}：定向重提失败指标{len(metric_codes)}个。"
                    )
                    _notify_progress(
                        progress_callback,
                        started_at,
                        "retry_completed",
                        completed_retries=completed_retries,
                        retry_total=len(retry_jobs),
                        table_id=table_id,
                        table_name=card.table_name,
                        model_calls=initial_calls + retry_calls,
                    )
        validations = validate_vlm_v2_metrics(frame, cards)
    else:
        _notify_progress(
            progress_callback,
            started_at,
            "validation_completed",
            retry_total=0,
            retry_table_names=[],
        )

    # Even if the required totals are on successful pages, an unread page must
    # not turn its optional metrics into apparently genuine non-disclosures.
    for table_id in sorted(unavailable_table_ids):
        validations = pd.concat([validations, pd.DataFrame([_validation_row(
            f'{table_id}:REQUEST_COVERAGE', table_id, '', '失败',
            '请求覆盖完整性', '页面请求未完成', '全部确认页成功读取',
            '存在超时或失败请求批次；已保留成功结果，不能据此确认其他指标未披露。'
            + request_errors.get(table_id, ''),
        )], columns=VALIDATION_COLUMNS)], ignore_index=True)
    if not active:
        logs.append("没有可用于VLM v2抽取的命中页。")
    frame = frame.sort_values(
        ["目标表ID", "指标编码"],
        kind="stable",
    ).reset_index(drop=True) if not frame.empty else frame
    _notify_progress(
        progress_callback,
        started_at,
        "completed",
        total_tables=len(active),
        model_calls=initial_calls + retry_calls,
        retry_calls=retry_calls,
    )
    return VLMV2ExtractionRun(
        records=frame,
        validations=validations,
        logs=tuple(logs),
        model_calls=initial_calls + retry_calls,
        retry_calls=retry_calls,
    )


def evaluate_vlm_v2_step3_gate(
    extraction_run: VLMV2ExtractionRun | None,
) -> VLMV2QualityGate:
    """Return a deterministic, non-overridable admission decision for STEP3."""
    columns = ["门槛项", "状态", "阻断", "说明"]
    if extraction_run is None or extraction_run.records.empty:
        checks = pd.DataFrame([
            {"门槛项": "VLM v2结果", "状态": "失败", "阻断": "是", "说明": "尚未生成VLM v2指标结果。"}
        ], columns=columns)
        return VLMV2QualityGate(False, checks, "未通过：尚无可用结果")

    extraction_run = refresh_vlm_v2_cross_table_checks(extraction_run)
    records = extraction_run.records
    validations = extraction_run.validations
    failures = validations[validations["状态"] == "失败"] if not validations.empty else validations
    disclosed = records[records["状态"].isin({"found", "disclosed_zero", "disclosed_na"})]
    value_bearing = disclosed[
        disclosed["状态"].isin({"found", "disclosed_zero"})
        & ~disclosed['指标编码'].isin(TEXT_FILING_CODES)
    ]
    text_disclosed = disclosed[disclosed['状态'].eq('found') & disclosed['指标编码'].isin(TEXT_FILING_CODES)]
    missing_text = text_disclosed['原始值'].fillna('').astype(str).str.strip().eq('').any()
    numeric_disclosed = value_bearing[pd.to_numeric(value_bearing["标准数值"], errors="coerce").notna()]
    bad_period = disclosed[
        disclosed.apply(
            lambda row: _period_is_invalid(
                _record_period_text(row), row.get("目标表ID"),
            ),
            axis=1,
        )
    ]
    missing_evidence = disclosed[
        disclosed["物理页码"].fillna(0).astype(float).le(0)
        | disclosed["证据原文"].astype(str).str.strip().eq("")
        | disclosed["原始标签"].astype(str).str.strip().eq("")
    ]
    checks = pd.DataFrame([
        {
            "门槛项": "确定性业务校验",
            "状态": "通过" if failures.empty else "失败",
            "阻断": "否" if failures.empty else "是",
            "说明": "全部校验通过" if failures.empty else f"仍有{len(failures)}项失败",
        },
        {
            "门槛项": "本季度期间口径",
            "状态": "通过" if bad_period.empty else "失败",
            "阻断": "否" if bad_period.empty else "是",
            "说明": "未选中上季度/期初/预测列" if bad_period.empty else f"有{len(bad_period)}项期间口径错误",
        },
        {
            "门槛项": "来源证据完整",
            "状态": "通过" if missing_evidence.empty and not disclosed.empty else "失败",
            "阻断": "否" if missing_evidence.empty and not disclosed.empty else "是",
            "说明": f"{len(disclosed)}项披露均有页码、标签和证据" if missing_evidence.empty and not disclosed.empty else f"有{len(missing_evidence)}项缺少来源证据",
        },
        {
            "门槛项": "标准数值可用",
            "状态": "通过" if len(numeric_disclosed) == len(value_bearing) and not missing_text else "失败",
            "阻断": "否" if len(numeric_disclosed) == len(value_bearing) and not missing_text else "是",
            "说明": '文本披露内容为空' if missing_text else f"{len(numeric_disclosed)}项数值披露均可进入标准化；{int((disclosed['状态'] == 'disclosed_na').sum())}项明确不适用将保留状态且数值留空" if len(numeric_disclosed) == len(value_bearing) else f"有{len(value_bearing) - len(numeric_disclosed)}项数值披露无法标准化",
        },
    ], columns=columns)
    passed = not bool((checks["阻断"] == "是").any())
    summary = "通过：可进入STEP3" if passed else "未通过：请修复失败项后再进入STEP3"
    return VLMV2QualityGate(passed, checks, summary)


def _apply_disclosed_zero_policy(records: pd.DataFrame) -> pd.DataFrame:
    """Upgrade older STEP2 dash-as-NA records while preserving source evidence."""
    records = records.copy()
    if records.empty:
        return records
    mask = (records['状态'].isin({'found', 'disclosed_zero', 'disclosed_na'})
            & ~records['指标编码'].isin(TEXT_FILING_CODES)
            & records['原始值'].map(is_disclosed_zero))
    records.loc[mask, '状态'] = 'disclosed_zero'
    records.loc[mask, ['数值', '标准数值']] = 0.0
    return records


def vlm_v2_to_extracted_tables(
    extraction_run: VLMV2ExtractionRun,
) -> list[ExtractedTable]:
    """Adapt gate-approved canonical VLM records to the existing STEP3 normalizer."""
    extraction_run = replace(extraction_run, records=_apply_disclosed_zero_policy(extraction_run.records))
    extraction_run = refresh_vlm_v2_cross_table_checks(extraction_run)
    gate = evaluate_vlm_v2_step3_gate(extraction_run)
    if not gate.passed:
        raise ValueError(gate.summary)
    records = extraction_run.records.copy()
    tables: list[ExtractedTable] = []
    for table_id, group in records.groupby("目标表ID", sort=False):
        first = group.iloc[0]
        if table_id == "OPERATING_METRICS":
            rows = [["指标名称", "本季度数", "本年度累计数"]]
            operating_rows: dict[str, list[str]] = {}
            for _, row in group.iterrows():
                if row['状态'] not in {'found', 'disclosed_zero'}:
                    continue
                unit = _clean(row["标准单位"])
                label = _clean(row["指标名称"])
                if unit:
                    label = f"{label}（{unit}）"
                values = operating_rows.setdefault(
                    _clean(row["指标编码"]), [label, "", ""],
                )
                value = row['原始值'] if row['指标编码'] in TEXT_FILING_CODES else row["标准数值"]
                period_kind = _operating_period_kind(_record_period_text(row))
                column = 2 if period_kind == "cumulative" else 1
                values[column] = "" if pd.isna(value) else str(value)
            rows.extend(operating_rows.values())
        else:
            rows = [["指标名称", "本季度末数"]]
            for _, row in group.iterrows():
                if row['状态'] not in {'found', 'disclosed_zero'}:
                    continue
                unit = _clean(row["标准单位"])
                label = _clean(row["指标名称"])
                if unit:
                    label = f"{label}（{unit}）"
                value = row['原始值'] if row['指标编码'] in TEXT_FILING_CODES else row["标准数值"]
                rows.append([label, "" if pd.isna(value) else str(value)])
        pages = sorted({int(page) for page in pd.to_numeric(group['物理页码'], errors='coerce').dropna() if page > 0})
        canonical_group = group.copy()
        for column in ('数值', '标准数值', '物理页码', '置信度'):
            canonical_group[column] = pd.to_numeric(canonical_group[column], errors='coerce')
        canonical_group = canonical_group.astype(object).where(pd.notna(canonical_group), None)
        tables.append(ExtractedTable(
            table_id=str(table_id),
            table_name=_clean(first["目标表名称"]),
            page=pages[0] if pages else 0,
            table_index=0,
            rows=rows,
            strategy=EXTRACTION_MODE,
            quality_score=100.0,
            evidence="通过VLM v2 STEP3前置门槛",
            source_pages=pages,
            profile_strategy_id="VLM_V2_CANONICAL",
            metric_records=canonical_group.to_dict('records'),
        ))
    return tables


def read_vlm_v2_extracted_tables(excel: pd.ExcelFile) -> list[ExtractedTable]:
    """Restore a STEP2 export through the same quality gate as live results."""
    if "确定性校验" not in excel.sheet_names:
        raise ValueError("VLM v2 工作簿缺少“确定性校验”工作表，请重新下载完整的 STEP2 提取文件。")
    records = pd.read_excel(
        excel, sheet_name="VLM_v2指标结果", keep_default_na=False,
        dtype={column: str for column in METRIC_COLUMNS if column not in {'数值', '标准数值', '物理页码', '置信度'}},
    )
    validations = pd.read_excel(excel, sheet_name="确定性校验", keep_default_na=False)
    for label, frame, columns in (
        ("VLM_v2指标结果", records, METRIC_COLUMNS),
        ("确定性校验", validations, VALIDATION_COLUMNS),
    ):
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{label}缺少字段：{'、'.join(missing)}。请上传完整的 STEP2 提取文件。")
    for column in ("物理页码", "标准数值", "数值", "置信度"):
        records[column] = pd.to_numeric(records[column], errors="coerce")
    if not records["状态"].isin({"found", "disclosed_zero", "disclosed_na", "not_disclosed"}).all():
        raise ValueError("VLM v2 指标结果包含无法识别的披露状态，请重新下载 STEP2 提取文件。")
    run = VLMV2ExtractionRun(_apply_disclosed_zero_policy(records), validations, (), 0, 0)
    gate = evaluate_vlm_v2_step3_gate(run)
    if not gate.passed:
        blocked = gate.checks[gate.checks["阻断"] == "是"]
        details = "；".join(f"{row['门槛项']}：{row['说明']}" for _, row in blocked.iterrows())
        raise ValueError(f"上传文件未通过 STEP3 前置门槛：{details}。请在 STEP2 修复后重新导出。")
    return vlm_v2_to_extracted_tables(run)


def vlm_v2_workbook_bytes(
    locator_run: VLMV2LocatorRun | None,
    extraction_run: VLMV2ExtractionRun,
    *,
    report_metadata: Mapping[str, object] | None = None,
    source_filename: str = "",
) -> bytes:
    extraction_run = refresh_vlm_v2_cross_table_checks(extraction_run)
    metadata = dict(report_metadata or {})
    report_year = _clean(metadata.get("报告年度", ""))
    report_quarter = _clean(metadata.get("报告季度", "")).upper()
    report_period = _clean(metadata.get("报告期", ""))
    if not report_period and report_year and re.fullmatch(r"Q[1-4]", report_quarter):
        report_period = f"{report_year}{report_quarter}"
    metadata_frame = pd.DataFrame({
        "字段": ["公司", "报告年度", "报告季度", "报告期", "披露日期", "来源文件"],
        "值": [
            _clean(metadata.get("公司", "")),
            report_year,
            report_quarter,
            report_period,
            _clean(metadata.get("披露日期", "")),
            _clean(source_filename or metadata.get("来源文件", "")),
        ],
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        metadata_frame.to_excel(writer, sheet_name="报告元信息", index=False)
        extraction_run.records.to_excel(writer, sheet_name="VLM_v2指标结果", index=False)
        extraction_run.validations.to_excel(writer, sheet_name="确定性校验", index=False)
        evaluate_vlm_v2_step3_gate(extraction_run).checks.to_excel(
            writer, sheet_name="STEP3前置门槛", index=False,
        )
        pd.DataFrame({"运行日志": list(extraction_run.logs)}).to_excel(
            writer, sheet_name="运行日志", index=False,
        )
        if locator_run is not None:
            pd.DataFrame([
                {
                    "目标表ID": match.table_id,
                    "目标表名称": match.table_name,
                    "物理页码": ",".join(map(str, match.pages)),
                    "定位评分": match.score,
                    "证据": match.evidence,
                    "需复核": "是" if match.review_required else "否",
                }
                for match in locator_run.matches
            ]).to_excel(writer, sheet_name="VLM_v2页码定位", index=False)
            locator_run.diagnostics.to_excel(writer, sheet_name="定位调用明细", index=False)
    return output.getvalue()


def vlm_v2_gold_evaluation(
    gold_case: Mapping[str, object],
    locator_run: VLMV2LocatorRun | None,
    extraction_run: VLMV2ExtractionRun | None,
) -> pd.DataFrame:
    """Evaluate a blind case without injecting its answers into VLM prompts."""
    rows: list[dict[str, object]] = []
    predicted_pages = {
        match.table_id: sorted(set(match.pages))
        for match in (locator_run.matches if locator_run else ())
    }
    records = extraction_run.records if extraction_run else pd.DataFrame(columns=METRIC_COLUMNS)

    def compact(value: object) -> str:
        return re.sub(r"[\s：:（）()、，,。·—\/_\-％%]", "", _clean(value))

    for table_id, expected in dict(gold_case.get("tables", {})).items():
        expected_pages = sorted(set(int(page) for page in expected.get("pages", [])))
        actual_pages = predicted_pages.get(str(table_id), [])
        overlap = set(expected_pages) & set(actual_pages)
        rows.append({
            "阶段": "页码定位",
            "目标表ID": table_id,
            "检查项": "物理页码",
            "期望": ",".join(map(str, expected_pages)),
            "实际": ",".join(map(str, actual_pages)),
            "通过": actual_pages == expected_pages,
            "召回率": len(overlap) / len(expected_pages) if expected_pages else 1.0,
        })
        table_records = records[
            (records["目标表ID"] == table_id) & (records["状态"] == "found")
        ] if not records.empty else records
        searchable = "\n".join(
            "|".join(map(_clean, row))
            for row in table_records[["指标名称", "原始标签", "原始值", "证据原文"]].values.tolist()
        ) if not table_records.empty else ""
        searchable_key = compact(searchable)
        for item in expected.get("required_items", []):
            hit = compact(item) in searchable_key
            rows.append({
                "阶段": "指标抽取",
                "目标表ID": table_id,
                "检查项": item,
                "期望": "应披露",
                "实际": "已找到" if hit else "未找到",
                "通过": hit,
                "召回率": None,
            })
        for item in expected.get("expected_values", []):
            label = _clean(item.get("item"))
            value = _clean(item.get("value"))
            matching = table_records[
                table_records.apply(
                    lambda row: compact(label) in compact(
                        f"{row['指标名称']}|{row['原始标签']}|{row['证据原文']}"
                    ),
                    axis=1,
                )
            ] if not table_records.empty else table_records
            hit = any(compact(value) == compact(raw) for raw in matching["原始值"].tolist())
            rows.append({
                "阶段": "关键数值",
                "目标表ID": table_id,
                "检查项": label,
                "期望": value,
                "实际": "；".join(matching["原始值"].astype(str).tolist()),
                "通过": hit,
                "召回率": None,
            })
    return pd.DataFrame(rows)
