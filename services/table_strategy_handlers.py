from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class PromptRequest:
    mode: str
    table_id: str
    table_name: str
    signatures: tuple[str, ...] = ()
    headers: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    completeness_terms: tuple[str, ...] = ()
    boundary_text: str = ""
    pages: tuple[int, ...] = ()
    page_number: int = 0
    retry_reason: str = ""
    table_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundaryRequest:
    mode: str
    table_id: str
    engine: Callable[..., Any]
    rows: list[list[str]] = field(default_factory=list)
    grid_text: str = ""
    source_grids: Any = None
    require_complete: bool = True
    table_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PostprocessRequest:
    table_id: str
    rows: list[list[str]]
    source_grids: Any = None
    normalize_three_year: Callable[..., tuple[list[list[str]], str]] | None = None
    recover_minimum_capital: Callable[..., tuple[list[list[str]], str]] | None = None
    trim_adjacent: Callable[..., tuple[list[list[str]], str]] | None = None
    table_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletenessRequest:
    mode: str
    table_id: str
    rows: list[list[str]]
    validator: Callable[..., Any]
    ragged_ratio: float = 0.0
    signatures: tuple[str, ...] = ()
    validator_kwargs: Mapping[str, Any] = field(default_factory=dict)
    table_config: Mapping[str, Any] = field(default_factory=dict)


PromptHandler = Callable[[PromptRequest], str]
BoundaryHandler = Callable[[BoundaryRequest], Any]
PostprocessHandler = Callable[
    [PostprocessRequest],
    tuple[list[list[str]], tuple[str, ...]],
]
CompletenessHandler = Callable[[CompletenessRequest], Any]


_PAGE_UNIT_NOTE = (
    "若原页表格上方或表头标有统一单位，必须把单位附加到相应数值列表头中，"
    "例如“期末数（元）”；项目名自带单位按原文保留，不得把单位说明作为数据行。"
)


def _config_text(
    config: Mapping[str, Any],
    key: str,
    default: str = "",
) -> str:
    value = config.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _config_bool(
    config: Mapping[str, Any],
    key: str,
    default: bool,
) -> bool:
    if key not in config or config.get(key) in (None, ""):
        return default
    value = config.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "是", "启用",
    }


def _config_float(
    config: Mapping[str, Any],
    key: str,
    default: float | None,
) -> float | None:
    if key not in config or config.get(key) in (None, ""):
        return default
    return float(config[key])


def _config_terms(
    config: Mapping[str, Any],
    key: str,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    if key not in config:
        return default
    value = config.get(key)
    if isinstance(value, str):
        return tuple(
            item.strip()
            for item in re.split(r"[|\n\r]+", value)
            if item.strip()
        )
    return tuple(str(item).strip() for item in value or () if str(item).strip())


def _config_actions(
    config: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    return _config_terms(config, key, default)


def make_prompt_handler(
    *,
    full_table_note: str = "",
    single_page_note: str = "",
) -> PromptHandler:
    def build_prompt(request: PromptRequest) -> str:
        config = request.table_config
        configured_full_note = _config_text(
            config,
            "prompt_full_table_note",
            full_table_note,
        )
        configured_page_note = _config_text(
            config,
            "prompt_single_page_note",
            single_page_note,
        )
        prompt_role = _config_text(
            config,
            "prompt_role",
            "四大会计师事务所的偿付能力报告数字化审阅专家",
        )
        profile_instructions = _config_terms(
            config,
            "profile_instructions",
        )
        profile_note = (
            "\nProfile补充要求：\n- " + "\n- ".join(profile_instructions)
            if profile_instructions
            else ""
        )
        signatures = "、".join(request.signatures) or (
            "无固定关键词" if request.mode == "full_table" else "以原页面为准"
        )
        headers = "、".join(request.headers) or "以原报告为准"
        exclusions = "、".join(request.exclusions) or "无"
        if request.mode == "full_table":
            completeness = (
                "、".join(request.completeness_terms)
                or "以原页面完整行数为准"
            )
            return f"""你是{prompt_role}。
目标表ID：{request.table_id}
目标表名称：{request.table_name}
物理页码：{list(request.pages)}
核心内容关键词：{signatures}
常见表头：{headers}
不得混入的相邻表内容：{exclusions}
完整性重点字段：{completeness}（原页面存在时必须全部保留）
组合范围补充：{configured_full_note}
项目边界约束：{request.boundary_text}
{profile_note}

请对输入的多页PDF进行网格化对齐重构：
1. 只提取目标表，不得混入同页其他表或正文。
2. 同一张表跨页时按页码顺序自动拼接，删除续表重复表名和重复表头。
3. 修复合并单元格造成的字段拆分，但不得推测或编造数值。
4. 每行必须和表头列严格对齐；缺失值用空字符串。
5. 保留金额、百分比、负号、括号和单位，不做换算。
   单位还必须独立写入unit_records；即使rows中的项目名被标准化，也不得丢失原页单位。
   unit_records的target必须使用columns或rows里实际输出的名称，source_text保留原页含单位文字。
   表格标题括号中的单位属于表级单位；例如“其他经营指标（元，%）”必须分别记录元和%两条，
   后续由金额/率类项目选择各自适用单位，不得遗漏或合并为“元，%”。
   只记录图片或文字网格中明确出现的单位，不得根据数值大小或常见口径猜测。
6. 一旦遇到不得混入的内容或下一张表标题，立即停止；边界行及其后内容不得写入rows。
7. 只输出JSON对象，不输出Markdown或解释文字。

JSON格式：
{{"table_id":"{request.table_id}","table_name":"{request.table_name}",
 "columns":["列1","列2"],"rows":[["数据1","数据2"]],
 "unit_records":[
   {{"scope":"行级","target":"项目名","raw_unit":"元",
     "source_page":1,"source_text":"项目名（元）","confidence":"高"}}
 ],
 "notes":"可选的简短重构说明"}}
""".strip()

        retry_note = (
            f"上一轮失败原因：{request.retry_reason}\n请逐行重新核对。"
            if request.retry_reason
            else ""
        )
        return f"""你是{prompt_role}。
当前任务：只提取PDF物理第{request.page_number}页中属于【{request.table_name}】的表格内容。

核心项目：{signatures}
常见表头：{headers}
不得混入：{exclusions}
项目边界：{request.boundary_text}
{configured_page_note}
{retry_note}
{profile_note}

{_PAGE_UNIT_NOTE}
【必须遵守】
1. 这是逐页任务。即使本页是续页、没有表名、只有最后一条数据，也必须提取该条数据。
2. 如果本页下方开始下一张表，只提取边界之前属于目标表的内容。
3. 每个单元格必须使用英文竖线“|”分隔；空单元格也必须用“|”占位。
4. 保留全部科目、行次、数字、百分号、负号、逗号和括号，不得概括、删除或编造。
5. 多级表头按原顺序保留；续页重复表头可以保留，系统会在拼接时去重。
6. 只输出竖线分隔的纯文本，不输出Markdown代码块、说明或JSON。
""".strip()

    return build_prompt


def make_boundary_handler(
    *,
    operating_metrics_grid: bool = False,
    preserve_adjustment_after_footnote: bool = False,
    respect_variant_activation: bool = False,
    require_value_for_boundary_items: bool = False,
) -> BoundaryHandler:
    def handle_boundary(request: BoundaryRequest) -> Any:
        config = request.table_config
        use_operating_metrics_grid = _config_bool(
            config,
            "boundary_operating_metrics_grid",
            operating_metrics_grid,
        )
        use_variant_activation = _config_bool(
            config,
            "boundary_respect_variant_activation",
            respect_variant_activation,
        )
        require_boundary_value = _config_bool(
            config,
            "boundary_require_value_for_items",
            require_value_for_boundary_items,
        )
        preserve_adjustment = _config_bool(
            config,
            "boundary_preserve_adjustment_after_footnote",
            preserve_adjustment_after_footnote,
        )
        if request.mode == "grid_slice":
            grid_options = {
                "operating_metrics": use_operating_metrics_grid,
            }
            if config:
                grid_options["table_config"] = config
            if use_variant_activation:
                grid_options["respect_variant_activation"] = True
            if require_boundary_value:
                grid_options["require_value_for_boundary_items"] = True
            return request.engine(
                request.table_id,
                request.grid_text,
                **grid_options,
            )
        if request.mode == "source_lines":
            source_options = {
                "preserve_adjustment_after_footnote": preserve_adjustment,
            }
            if config:
                source_options["table_config"] = config
            return request.engine(
                request.table_id,
                request.source_grids,
                **source_options,
            )
        output_options = {
            "require_complete": request.require_complete,
        }
        if config:
            output_options["table_config"] = config
        return request.engine(
            request.table_id,
            request.rows,
            **output_options,
        )

    return handle_boundary


def _grid_body(line: str) -> str:
    head, separator, body = str(line or "").partition("|")
    return body if separator else head


def _compact_grid_text(value: str) -> str:
    return re.sub(r"[\s：:（）()、，,。·—\-_/]", "", str(value or ""))


def _grid_line_has_any(line: str, terms: tuple[str, ...]) -> bool:
    compact = _compact_grid_text(_grid_body(line))
    return any(_compact_grid_text(term) in compact for term in terms)


def _grid_line_has_percent_value(lines: list[str], index: int) -> bool:
    body = _grid_body(lines[index])
    if re.search(r"[-+]?\d[\d,，.]*\s*[％%]", body):
        return True
    if index + 1 >= len(lines):
        return False
    # Only join a genuinely standalone value on the following line.  A
    # section title such as “近三年（综合）投资收益率” may be followed by the
    # ordinary-return row; treating any percentage on that row as the title's
    # value creates a duplicate business row and inflates completeness counts.
    return bool(re.fullmatch(
        r"\s*[-+]?\d[\d,，.]*\s*[％%]\s*[。.；;]?\s*",
        _grid_body(lines[index + 1]),
    ))


def _grid_line_has_two_values(line: str) -> bool:
    values = re.findall(
        r"(?<!\S)(?:[(（]?-?\d[\d,，.％%]*[)）]?|"
        r"--|—|-|N\s*/?\s*A|不适用)(?!\S)",
        _grid_body(line),
        flags=re.I,
    )
    return len(values) >= 2


def _slice_three_year_return_grid(
    grid_text: str,
    table_config: Mapping[str, Any] | None = None,
) -> str | None:
    """Keep explicit near-three-year rows before quarterly metric lookalikes."""
    lines = str(grid_text or "").splitlines()
    if not lines:
        return None

    config = table_config or {}
    variants = tuple(config.get("boundary_variants") or ())
    configured_starts = tuple(dict.fromkeys(
        str(item)
        for variant in variants
        for item in variant.get("start_items", ())
        if str(item).strip()
    ))
    configured_ends = tuple(dict.fromkeys(
        str(item)
        for variant in variants
        for item in (
            *variant.get("end_items", ()),
            *(
                term
                for group in variant.get("end_item_groups", ())
                for term in group
            ),
        )
        if str(item).strip()
    ))
    ordinary_terms = configured_starts or (
        "近三年平均投资收益率",
        "近三年投资收益率",
    )
    comprehensive_terms = configured_ends or (
        "近三年平均综合投资收益率",
        "近三年综合投资收益率",
    )
    start_hits = [
        index
        for index, line in enumerate(lines)
        if (
            _grid_line_has_any(line, ordinary_terms)
            and not _grid_line_has_any(line, comprehensive_terms)
            and _grid_line_has_percent_value(lines, index)
        )
    ]
    end_hits = [
        index
        for index, line in enumerate(lines)
        if (
            _grid_line_has_any(line, comprehensive_terms)
            and _grid_line_has_percent_value(lines, index)
        )
    ]
    item_hits = sorted(set(start_hits + end_hits))
    if not item_hits:
        return None

    selected: list[str] = []
    for index in item_hits:
        line = lines[index]
        body = _grid_body(line)
        if not re.search(r"[-+]?\d[\d,，.]*\s*[％%]", body):
            next_body = (
                _grid_body(lines[index + 1]).strip()
                if index + 1 < len(lines)
                else ""
            )
            if re.search(r"[-+]?\d[\d,，.]*\s*[％%]", next_body):
                # Some reports put the average value on a standalone line under
                # its section title. Join the logical row so downstream source
                # profiling and the model see the label and value together.
                line = f"{line} {next_body}"
        if line not in selected:
            selected.append(line)
    return "\n".join(selected)


_SOLVENCY_FORECAST_MARKERS = (
    "基本情景下的下季度预测数",
    "下季度预测数",
    "下季度末预测数",
)
_SOLVENCY_ADJACENT_TABLE_MARKERS = (
    "流动性风险监管指标",
    "流动性风险监测指标",
    "流动性覆盖率",
    "LCR1",
)


def _extend_solvency_main_forecast_grid(
    grid_text: str,
    sliced_text: str,
    table_config: Mapping[str, Any] | None = None,
) -> str:
    """Keep a separately disclosed next-quarter forecast subtable."""
    lines = str(grid_text or "").splitlines()
    if not lines:
        return sliced_text
    config = table_config or {}
    forecast_markers = _config_terms(
        config,
        "forecast_markers",
        _SOLVENCY_FORECAST_MARKERS,
    )
    adjacent_markers = _config_terms(
        config,
        "adjacent_table_markers",
        _SOLVENCY_ADJACENT_TABLE_MARKERS,
    )
    forecast_start = next(
        (
            index
            for index, line in enumerate(lines)
            if _grid_line_has_any(line, forecast_markers)
        ),
        None,
    )
    if forecast_start is None:
        return sliced_text

    sliced_lines = str(sliced_text or "").splitlines()
    start = forecast_start
    if sliced_lines:
        sliced_start = next(
            (
                index
                for index, line in enumerate(lines)
                if line == sliced_lines[0]
            ),
            forecast_start,
        )
        start = min(sliced_start, forecast_start)

    stop = next(
        (
            index
            for index in range(forecast_start + 1, len(lines))
            if _grid_line_has_any(
                lines[index],
                adjacent_markers,
            )
        ),
        len(lines),
    )
    return "\n".join(lines[start:stop])


def _slice_operating_metrics_tail(grid_text: str) -> str | None:
    """Keep a continuation tail that precedes a repeated product-detail table."""
    lines = str(grid_text or "").splitlines()
    if not lines:
        return None

    terminal_terms = (
        "营销员脱落率",
        "个人营销员脱落率",
        "代理人脱落率",
        "营销员流失率",
    )
    terminal_hits = [
        index
        for index, line in enumerate(lines)
        if _grid_line_has_any(line, terminal_terms)
    ]
    if not terminal_hits:
        return None
    terminal = max(terminal_hits)

    product_detail_terms = (
        "前五大产品的信息",
        "第一大产品的信息",
        "第二大产品的信息",
        "第三大产品的信息",
        "第四大产品的信息",
        "第五大产品的信息",
        "产品名称",
        "产品类型",
    )
    if not any(
        index > terminal and _grid_line_has_any(line, product_detail_terms)
        for index, line in enumerate(lines)
    ):
        return None

    disclosure_start_terms = ("保险业务收入", "保险业务收入合计")
    if any(
        index <= terminal and _grid_line_has_any(line, disclosure_start_terms)
        for index, line in enumerate(lines)
    ):
        return None

    continuation_titles = ("效益类指标", "规模类指标", "品质类指标")
    section_hits = [
        index
        for index, line in enumerate(lines[:terminal + 1])
        if _grid_line_has_any(line, continuation_titles)
    ]
    continuation_prefix_terms = (
        "净资产",
        "保险合同负债",
        "未到期责任准备金",
        "未决赔款准备金",
        "寿险责任准备金",
        "长期健康险责任准备金",
        "基本每股收益",
        "净资产收益率",
        "总资产收益率",
        "投资收益率",
        "综合投资收益率",
        "剩余边际",
        "新业务利润率",
        "新业务价值",
    )
    first_section = min(section_hits) if section_hits else terminal + 1
    prefix_hits = [
        index
        for index, line in enumerate(lines[:first_section])
        if (
            _grid_line_has_any(line, continuation_prefix_terms)
            and _grid_line_has_two_values(line)
        )
    ]
    start = min(prefix_hits) if prefix_hits else (
        min(section_hits) if section_hits else 0
    )
    return "\n".join(lines[start:terminal + 1])


def _remove_operating_metrics_interruption(sliced_text: str) -> str:
    """Reconnect operating sections after notes, product details, or other tables."""
    lines = str(sliced_text or "").splitlines()
    if not lines:
        return sliced_text

    continuation_titles = ("效益类指标", "规模类指标", "品质类指标")
    first_section = next(
        (
            index
            for index, line in enumerate(lines)
            if _grid_line_has_any(line, continuation_titles)
        ),
        None,
    )
    if first_section is None:
        return sliced_text

    interruption_titles = (
        "前五大产品的信息",
        "报告期内签单保费占前五位的产品",
        "签单保费占前五位的产品",
        "近三年（综合）投资收益率",
        "近三年平均投资收益率",
        "近三年平均综合投资收益率",
    )
    interruption = next(
        (
            index
            for index, line in enumerate(lines[:first_section])
            if (
                re.match(
                    r"^(?:\d+\s*)?(?:注释|注|(?:填表)?说明)\s*[：:]",
                    _grid_body(line).strip(),
                )
                or _grid_line_has_any(line, interruption_titles)
            )
        ),
        None,
    )
    if interruption is None:
        return sliced_text

    return "\n".join([
        *lines[:interruption],
        *lines[first_section:],
    ])


_OPERATING_TRAILING_FOOTNOTE_RE = re.compile(
    r"^(?:(?:\d+\s*)?(?:注释|注|(?:填表)?说明)\s*[：:]|"
    r"\d+\s+(?:表中|本表|上述))"
)


def _trim_operating_metrics_trailing_footnote(sliced_text: str) -> str:
    """Drop a page-bottom footnote while retaining the next page's table rows."""
    lines = str(sliced_text or "").splitlines()
    if not lines:
        return sliced_text
    footnote = next(
        (
            index
            for index, line in enumerate(lines)
            if _OPERATING_TRAILING_FOOTNOTE_RE.match(
                _grid_body(line).strip()
            )
        ),
        None,
    )
    if footnote is None:
        return sliced_text
    # Require visible tabular values before trimming so a narrative page whose
    # first line happens to look like a numbered note is left untouched.
    if not any(_grid_line_has_two_values(line) for line in lines[:footnote]):
        return sliced_text
    return "\n".join(lines[:footnote])


def _extend_operating_metrics_after_fallback(
    grid_text: str,
    sliced_text: str,
) -> str:
    """Retain ungrouped operating rows disclosed after the fallback endpoint."""
    lines = str(grid_text or "").splitlines()
    sliced_lines = str(sliced_text or "").splitlines()
    if not lines or not sliced_lines:
        return sliced_text

    fallback_terms = ("综合投资收益率",)
    terminal_terms = (
        "营销员脱落率",
        "个人营销员脱落率",
        "代理人脱落率",
        "营销员流失率",
    )
    if any(_grid_line_has_any(line, terminal_terms) for line in sliced_lines):
        return sliced_text

    sliced_end = next(
        (
            index
            for index in range(len(lines) - 1, -1, -1)
            if lines[index] == sliced_lines[-1]
        ),
        None,
    )
    if (
        sliced_end is None
        or not _grid_line_has_any(lines[sliced_end], fallback_terms)
    ):
        return sliced_text

    adjacent_terms = (
        "前五大产品的信息",
        "第一大产品的信息",
        "第二大产品的信息",
        "第三大产品的信息",
        "第四大产品的信息",
        "第五大产品的信息",
        "近三年（综合）投资收益率",
        "近三年综合投资收益率",
        "流动性风险监管指标",
        "流动性风险监测指标",
    )
    stop = len(lines)
    for index in range(sliced_end + 1, len(lines)):
        body = _grid_body(lines[index]).strip()
        if (
            re.match(r"^(?:\d+\s*)?(?:注释|注|(?:填表)?说明)\s*[：:]", body)
            or _grid_line_has_any(lines[index], adjacent_terms)
        ):
            stop = index
            break

    continuation_terms = (
        "效益类指标",
        "剩余边际",
        "新业务利润率",
        "新业务价值",
        "规模类指标",
        "签单保费",
        "新单首年期交签单保费",
        "十年期及以上新单首年期交签单保费",
        "续期签单保费",
        "分渠道",
        "期末个人营销员数量",
        "品质类指标",
        *terminal_terms,
    )
    if not any(
        _grid_line_has_any(line, continuation_terms)
        for line in lines[sliced_end + 1:stop]
    ):
        return sliced_text

    start = next(
        (
            index
            for index, line in enumerate(lines)
            if line == sliced_lines[0]
        ),
        0,
    )
    while stop > sliced_end + 1 and re.fullmatch(
        r"\s*\d+\s*",
        _grid_body(lines[stop - 1]),
    ):
        stop -= 1
    return "\n".join(lines[start:stop])


def _normalize_actual_capital_total_alias(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    """Normalize the standard actual-capital tail “5 合计” to its canonical name."""
    normalized = [list(row) for row in rows]
    changed = False
    last_data_index = max(
        (index for index, row in enumerate(normalized) if any(row)),
        default=-1,
    )
    for index, cells in enumerate(normalized):
        if (
            len(cells) >= 2
            and re.fullmatch(r"5(?:\.0)?", str(cells[0]).strip())
            and re.sub(r"[\s（）()]", "", str(cells[1])).startswith("合计")
        ):
            cells[1] = re.sub("合计", "实际资本合计", cells[1], count=1)
            changed = True
        elif (
            cells
            and re.match(r"^5\s*合计", str(cells[0]).strip())
        ):
            cells[0] = re.sub(
                r"^(5\s*)合计",
                r"\1实际资本合计",
                cells[0],
                count=1,
            )
            changed = True
        elif (
            index == last_data_index
            and len(cells) >= 3
            and re.sub(r"[\s（）()]", "", str(cells[0])).startswith("合计")
        ):
            cells[0] = re.sub(
                "合计",
                "实际资本合计",
                cells[0],
                count=1,
            )
            changed = True
    return normalized, changed


def _normalize_recognized_assets_total_alias(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    """Standardize the recognized-assets tail “合计” to its canonical name."""
    normalized = [list(row) for row in rows]
    changed = False
    for cells in normalized:
        if not cells:
            continue
        has_row_number = bool(
            re.fullmatch(r"\d+(?:\.\d+)*\*?", str(cells[0]).strip())
        )
        label_index = 1 if has_row_number and len(cells) >= 2 else 0
        if label_index >= len(cells):
            continue
        label = str(cells[label_index]).strip()
        if re.sub(r"[\s（）()]", "", label) == "合计":
            cells[label_index] = "认可资产合计"
            changed = True
    return normalized, changed


def _normalize_recognized_liabilities_total_alias(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    """Standardize the recognized-liabilities tail total label."""
    normalized = [list(row) for row in rows]
    changed = False
    for cells in normalized:
        if not cells:
            continue
        has_row_number = bool(
            re.fullmatch(r"\d+(?:\.\d+)*\*?", str(cells[0]).strip())
        )
        label_index = 1 if has_row_number and len(cells) >= 2 else 0
        if label_index >= len(cells):
            continue
        label = str(cells[label_index]).strip()
        if re.sub(r"[\s（）()]", "", label) == "合计":
            cells[label_index] = "认可负债合计"
            changed = True
    return normalized, changed


def _simplify_recognized_assets_columns(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    """将认可资产表两级表头化简为单级表头，仅保留“认可价值”列。

    输入（两级表头 期末数/期初数 × 账面价值/非认可价值/认可价值）：
        ["行次", "项目", "期末数", "", "", "期初数", "", ""],
        ["", "", "账面价值", "非认可价值", "认可价值", "账面价值", "非认可价值", "认可价值"],
        ["1", "现金及流动性管理工具", "100", "20", "80", "90", "18", "72"],
    输出（行次/项目/期末数/期初数，期末数在前）：
        ["行次", "项目", "期末数", "期初数"],
        ["1", "现金及流动性管理工具", "80", "72"],
    """
    if len(rows) < 3:
        return [list(row) for row in rows], False
    second = rows[1]
    if not any(
        term in "".join(str(cell) for cell in second)
        for term in ("账面价值", "非认可", "认可价值")
    ):
        return [list(row) for row in rows], False

    normalized = [list(row) for row in rows]

    def _subcolumn(cell: str) -> str:
        compact = re.sub(r"\s+", "", str(cell or ""))
        if "非认可" in compact:
            return "非认可价值"
        if "认可价值" in compact:
            return "认可价值"
        if "账面价值" in compact:
            return "账面价值"
        return ""

    def _period(cell: str) -> str:
        compact = re.sub(r"\s+", "", str(cell or ""))
        if "期末" in compact:
            return "期末数"
        if "期初" in compact:
            return "期初数"
        return ""

    # 前向填充一级表头，得到每一列对应的期间（期末数/期初数）。
    filled_first: list[str] = []
    last = ""
    for value in normalized[0]:
        last = value if value else last
        filled_first.append(last)

    width = max(len(row) for row in normalized)
    label_columns: list[int] = []
    recognized_columns: list[tuple[int, str]] = []  # (列序号, 期间)
    for col in range(width):
        sub = _subcolumn(second[col] if col < len(second) else "")
        if sub == "认可价值":
            recognized_columns.append(
                (col, _period(filled_first[col] if col < len(filled_first) else ""))
            )
        elif sub in ("账面价值", "非认可价值"):
            continue
        else:
            label_columns.append(col)

    # 认可价值列按 期末数/期初数 排序（期末数在前）。
    recognized_columns.sort(
        key=lambda item: {"期末数": 0, "期初数": 1}.get(item[1], 99)
    )

    label_defaults = ["行次", "项目"]
    label_headers: list[str] = []
    for index, col in enumerate(label_columns):
        value = (filled_first[col] if col < len(filled_first) else "") or ""
        if value:
            label_headers.append(value)
        elif index == 0 and len(label_columns) > 1:
            label_headers.append("行次")
        else:
            label_headers.append("项目")
    new_header = label_headers + [period for _, period in recognized_columns]

    simplified = [new_header]
    for row in normalized[2:]:
        padded = row + [""] * (width - len(row))
        new_row = [padded[col] for col in label_columns]
        new_row += [padded[col] for col, _ in recognized_columns]
        simplified.append(new_row)

    return simplified, True


def _simplify_recognized_liabilities_columns(
    rows: list[list[str]],
) -> tuple[list[list[str]], bool]:
    """Keep only recognized-value period columns in the S04 liability table."""
    return _simplify_recognized_assets_columns(rows)


def make_postprocess_handler(
    *,
    normalize_three_year: bool = False,
    recover_minimum_capital: bool = False,
    normalize_actual_capital_total: bool = False,
    normalize_recognized_assets_total: bool = False,
    simplify_recognized_assets_columns: bool = False,
    normalize_recognized_liabilities_total: bool = False,
    simplify_recognized_liabilities_columns: bool = False,
) -> PostprocessHandler:
    def postprocess(
        request: PostprocessRequest,
    ) -> tuple[list[list[str]], tuple[str, ...]]:
        default_actions = tuple(
            action
            for action, enabled in (
                ("normalize_three_year_return", normalize_three_year),
                ("recover_minimum_capital_terminal", recover_minimum_capital),
                ("normalize_actual_capital_total", normalize_actual_capital_total),
                ("simplify_recognized_assets_columns", simplify_recognized_assets_columns),
                ("normalize_recognized_assets_total", normalize_recognized_assets_total),
                ("simplify_recognized_liabilities_columns", simplify_recognized_liabilities_columns),
                ("normalize_recognized_liabilities_total", normalize_recognized_liabilities_total),
                ("trim_adjacent_rows", request.trim_adjacent is not None),
            )
            if enabled
        )
        actions = set(_config_actions(
            request.table_config,
            "postprocess_actions",
            default_actions,
        ))
        rows = request.rows
        notes: list[str] = []
        if "normalize_three_year_return" in actions and request.normalize_three_year:
            rows, note = request.normalize_three_year(request.table_id, rows)
            if note:
                notes.append(note)
        if (
            "recover_minimum_capital_terminal" in actions
            and request.recover_minimum_capital
        ):
            rows, note = request.recover_minimum_capital(
                request.table_id,
                rows,
                request.source_grids,
            )
            if note:
                notes.append(note)
        if "normalize_actual_capital_total" in actions:
            rows, changed = _normalize_actual_capital_total_alias(rows)
            if changed:
                notes.append(
                    "已将实际资本表末行‘5 合计’标准化为‘5 实际资本合计’"
                )
        if "simplify_recognized_assets_columns" in actions:
            rows, changed = _simplify_recognized_assets_columns(rows)
            if changed:
                notes.append(
                    "已将认可资产表两级表头化简为单级表头（行次/项目/期末数/期初数），仅保留认可价值列"
                )
        if "normalize_recognized_assets_total" in actions:
            rows, changed = _normalize_recognized_assets_total_alias(rows)
            if changed:
                notes.append(
                    "已将认可资产表末行‘合计’标准化为‘认可资产合计’"
                )
        if "simplify_recognized_liabilities_columns" in actions:
            rows, changed = _simplify_recognized_liabilities_columns(rows)
            if changed:
                notes.append(
                    "已将认可负债表两级表头化简为单级表头（行次/项目/期末数/期初数），仅保留认可价值列"
                )
        if "normalize_recognized_liabilities_total" in actions:
            rows, changed = _normalize_recognized_liabilities_total_alias(rows)
            if changed:
                notes.append(
                    "已将认可负债表末行‘合计’标准化为‘认可负债合计’"
                )
        if "trim_adjacent_rows" in actions and request.trim_adjacent:
            rows, note = request.trim_adjacent(request.table_id, rows)
            if note:
                notes.append(note)
        return rows, tuple(notes)

    return postprocess


def make_completeness_handler(
    *,
    require_all_signatures: bool = False,
    allow_single_value_boundary_rows: bool = False,
    track_operating_sections: bool = False,
    source_item_recall_ratio: float | None = None,
) -> CompletenessHandler:
    def validate(request: CompletenessRequest) -> Any:
        config = request.table_config
        require_every_signature = _config_bool(
            config,
            "completeness_require_all_signatures",
            require_all_signatures,
        )
        allow_single_value = _config_bool(
            config,
            "completeness_allow_single_value_boundary_rows",
            allow_single_value_boundary_rows,
        )
        track_sections = _config_bool(
            config,
            "completeness_track_sections",
            track_operating_sections,
        )
        configured_recall_ratio = _config_float(
            config,
            "completeness_source_item_recall_ratio",
            source_item_recall_ratio,
        )
        kwargs = dict(request.validator_kwargs)
        if config:
            kwargs["table_config"] = config
        if (
            configured_recall_ratio is not None
            and request.mode in {"full_table", "single_page"}
        ):
            kwargs["source_item_recall_ratio"] = configured_recall_ratio
        if request.mode == "full_table":
            kwargs["required_signature_hits"] = (
                len(request.signatures)
                if require_every_signature
                else min(1, len(request.signatures))
            )
            return request.validator(
                request.table_id,
                request.rows,
                request.ragged_ratio,
                **kwargs,
            )
        if request.mode == "source_profile":
            kwargs["allow_single_value_boundary_rows"] = (
                allow_single_value
            )
            return request.validator(request.table_id, **kwargs)
        if request.mode == "source_sections":
            kwargs["track_operating_sections"] = track_sections
            return request.validator(request.table_id, **kwargs)
        return request.validator(
            request.table_id,
            request.rows,
            **kwargs,
        )

    return validate


GENERIC_PROMPT_HANDLER = make_prompt_handler()
GENERIC_BOUNDARY_HANDLER = make_boundary_handler()
GENERIC_POSTPROCESS_HANDLER = make_postprocess_handler()
GENERIC_COMPLETENESS_HANDLER = make_completeness_handler()


def _solvency_main_boundary_handler(request: BoundaryRequest) -> Any:
    sliced = GENERIC_BOUNDARY_HANDLER(request)
    actions = _config_actions(
        request.table_config,
        "boundary_actions",
        ("extend_solvency_forecast",),
    )
    if request.mode == "grid_slice" and "extend_solvency_forecast" in actions:
        return _extend_solvency_main_forecast_grid(
            request.grid_text,
            sliced,
            request.table_config,
        )
    return sliced


def _three_year_return_boundary_handler(request: BoundaryRequest) -> Any:
    actions = _config_actions(
        request.table_config,
        "boundary_actions",
        ("select_three_year_percent_rows",),
    )
    if request.mode == "grid_slice" and "select_three_year_percent_rows" in actions:
        sliced = _slice_three_year_return_grid(
            request.grid_text,
            request.table_config,
        )
        if sliced is not None:
            return sliced
    return GENERIC_BOUNDARY_HANDLER(request)


THREE_YEAR_RETURN_BOUNDARY_HANDLER = _three_year_return_boundary_handler

SOLVENCY_MAIN_BOUNDARY_HANDLER = _solvency_main_boundary_handler
SOLVENCY_MAIN_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "本目标包含偿付能力充足率期末/期初表，以及原报告实际披露时紧随其后的"
        "“基本情景下的下季度预测数”子表。统一输出四列："
        "“指标名称、本季度末数、上季度末数、下季度末预测数”。"
        "原文“期末数、期初数、下季度预测数”分别映射到上述后三列。"
        "原文如有“行次/序号”列，仅用于定位，不得作为输出列；每条数据必须从指标名称开始。"
        "两个子表出现同名项目时必须合并到同一行；仅在预测表出现的“核心资本”等项目"
        "单独保留一行，前两期数值留空。不得在第一张表的“综合偿付能力充足率”处提前停止，"
        "也不得混入随后开始的流动性风险监管指标。"
    ),
    single_page_note=(
        "偿付能力充足率主表与“基本情景下的下季度预测数”均属于本目标。"
        "如本页出现预测子表，必须一并提取，并统一为"
        "“指标名称|本季度末数|上季度末数|下季度末预测数”四列；"
        "原文“行次/序号”不得输出，每条数据必须从指标名称开始；"
        "同名项目合并，预测表独有项目单列，未披露列留空。"
        "遇到流动性风险监管指标时立即停止。"
    ),
)
_SOLVENCY_MAIN_BASE_COMPLETENESS_HANDLER = make_completeness_handler(
    require_all_signatures=True,
)


def _solvency_main_completeness_handler(
    request: CompletenessRequest,
) -> Any:
    config = request.table_config
    recall_ratio = _config_float(
        config,
        "completeness_source_item_recall_ratio",
        1.0,
    )
    if (
        request.mode == "full_table"
        and "source_item_labels" in request.validator_kwargs
        and recall_ratio is not None
    ):
        request = replace(
            request,
            validator_kwargs={
                **request.validator_kwargs,
                "source_item_recall_ratio": recall_ratio,
            },
        )
    result = _SOLVENCY_MAIN_BASE_COMPLETENESS_HANDLER(request)
    if request.mode != "source_profile":
        return result
    expected_rows, required_terms = result
    grids = request.validator_kwargs.get("grids")
    require_forecast = _config_bool(
        config,
        "completeness_require_forecast_if_present",
        True,
    )
    if not require_forecast:
        return result
    forecast_markers = _config_terms(
        config,
        "forecast_markers",
        _SOLVENCY_FORECAST_MARKERS,
    )
    has_forecast = any(
        _grid_line_has_any(line, forecast_markers)
        for grid in grids or ()
        for line in str(getattr(grid, "grid_text", "") or "").splitlines()
    )
    if not has_forecast:
        return result
    return expected_rows, tuple(dict.fromkeys([
        *required_terms,
        _config_text(config, "forecast_required_term", "预测数"),
    ]))


SOLVENCY_MAIN_COMPLETENESS_HANDLER = (
    _solvency_main_completeness_handler
)

OPERATING_METRICS_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "本目标是组合经营指标，按原报告实际披露的项目提取并按原顺序合并。"
        "报告可能只披露主要经营指标至‘综合投资收益率’，若后续效益类、规模类、品质类指标未出现，"
        "应以‘综合投资收益率’作为实际终点并完整保留已披露内容。"
        "如果原文继续披露后三类指标，则必须继续提取至最后的‘营销员脱落率’（或其同义名称）。"
        "不得补造未披露的指标，也不得因存在备用终点而提前截断。"
        "不得输出‘前五大产品的信息’、‘签单保费占前五位的产品’或其产品明细。"
        "跨页合并单元格必须续接：若“保险合同负债（元）”在前页开始，下一页从"
        "“未决赔款准备金、寿险责任准备金、长期健康险责任准备金”等子项继续，"
        "这些页首子项以及紧随其后的“基本每股收益”均属于目标表，不得因续页没有"
        "重复表头或类别名称而删除；保留父级口径和单位，不得把子项误并成产品明细。"
        "“效益类指标、规模类指标、品质类指标”三个类别标题必须按原顺序分别保留为"
        "独立行，标题行的数值列留空；不得只保留类别下的指标而删除类别标题。"
    ),
    single_page_note=(
        "主要经营指标按原报告实际披露范围提取并保持原顺序。若报告只披露至‘综合投资收益率’，"
        "且后续效益类、规模类、品质类指标未出现，则以该项目作为实际终点；"
        "若原文继续披露后三类指标，则必须继续提取至最后的‘营销员脱落率’（或同义名称）。"
        "不得补造未披露指标，也不得因备用终点而提前截断。"
        "不得输出‘前五大产品的信息’、‘签单保费占前五位的产品’或其产品明细。"
        "若本页是“保险合同负债（元）”跨页续表，必须保留页首准备金子项和"
        "“基本每股收益”，即使本页没有重复表头或父级类别名称。"
        "本页出现“效益类指标、规模类指标、品质类指标”时，类别标题必须按原顺序"
        "分别保留为独立行，标题行的数值列留空。"
    ),
)
_OPERATING_METRICS_BASE_BOUNDARY_HANDLER = make_boundary_handler(
    operating_metrics_grid=True,
)


def _operating_metrics_boundary_handler(request: BoundaryRequest) -> Any:
    actions = _config_actions(
        request.table_config,
        "boundary_actions",
        ("recover_operating_metrics_sections",),
    )
    if (
        request.mode == "grid_slice"
        and "recover_operating_metrics_sections" in actions
    ):
        sliced = _slice_operating_metrics_tail(request.grid_text)
        if sliced is None:
            sliced = _OPERATING_METRICS_BASE_BOUNDARY_HANDLER(request)
            sliced = _extend_operating_metrics_after_fallback(
                request.grid_text,
                sliced,
            )
        sliced = _remove_operating_metrics_interruption(sliced)
        return _trim_operating_metrics_trailing_footnote(sliced)
    return _OPERATING_METRICS_BASE_BOUNDARY_HANDLER(request)


OPERATING_METRICS_BOUNDARY_HANDLER = _operating_metrics_boundary_handler
OPERATING_METRICS_COMPLETENESS_HANDLER = make_completeness_handler(
    track_operating_sections=True,
)

ACTUAL_CAPITAL_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "实际资本支持四种原文版式：财务报表口径汇总在前、认可资产口径汇总在前、"
        "表头后直接从“核心一级资本”开始的核心资本明细标准版，或“实际资本表”后"
        "继续披露“核心一级资本调整表”的版式。"
        "只有原文实际出现汇总表时，才完整保留财务报表资产总额或认可资产起始的汇总行，"
        "再继续提取各级资本明细；若原文直接从“核心一级资本”开始，则该页本身就是完整目标，"
        "必须从“核心一级资本”连续提取到“实际资本合计”，不得补造“认可资产”等未披露汇总行。"
        "若核心资本明细标准版末行写作“5 合计”，该行就是实际资本合计，必须保留并将项目名"
        "标准化为“实际资本合计”。"
        "若原文为“实际资本表”后接“核心一级资本调整表”，首张子表可能从“认可资产、"
        "认可负债”开始，也可能直接从“实际资本”开始；必须从首张子表实际披露的第一项"
        "连续提取至调整表最后的“银保监会规定的其他调整项目”或同义项目；随后开始的"
        "“认可资产表”和“认可负债表”明细不属于本目标，绝对不得混入。"
        "同页两张子表可能分别使用“本季度数/上季度可比数”和“期末数/期初数”；"
        "两组列按本期、上期一一对应，输出时统一为“行次、项目、期末数、期初数”，"
        "不得把第二张子表的表头当作数据行，也不得因此删除任一子表的数据。"
        "有汇总表时不得在汇总结束处停止；无汇总表时不得因缺少汇总行而判定提取不完整。"
    ),
    single_page_note=(
        "实际资本支持汇总表在前、直接从‘核心一级资本’开始，以及‘实际资本表’后接"
        "‘核心一级资本调整表’等结构。"
        "若原文存在汇总表，其首行可能是‘财务报表资产总额’或‘认可资产’，"
        "必须完整保留汇总及随后资本明细直到‘实际资本合计’；"
        "若原文表头后直接出现‘核心一级资本’，应从该行完整提取到‘实际资本合计’，"
        "不得要求或补造原文没有的‘认可资产’等汇总行；"
        "核心资本明细末行写作‘5 合计’时同样属于目标终点，必须保留并标准化为‘实际资本合计’；"
        "若原文继续披露‘核心一级资本调整表’，首张子表可能从‘认可资产、认可负债’开始，"
        "也可能直接从‘实际资本’开始；必须从首张子表实际披露的第一项完整提取两张子表，"
        "到‘银保监会规定的其他调整项目’或同义项目结束；随后开始的‘认可资产表’和"
        "‘认可负债表’明细不属于目标，不得混入；"
        "同页两张子表的‘本季度数/上季度可比数’与‘期末数/期初数’按本期、上期一一对应，"
        "统一输出为‘行次|项目|期末数|期初数’，不得把第二张子表表头作为数据行；"
        "存在汇总表时不得在第一张汇总表结束处提前停止。"
    ),
)
ACTUAL_CAPITAL_BOUNDARY_HANDLER = make_boundary_handler(
    preserve_adjustment_after_footnote=True,
    respect_variant_activation=True,
)
ACTUAL_CAPITAL_POSTPROCESS_HANDLER = make_postprocess_handler(
    normalize_actual_capital_total=True,
)
ACTUAL_CAPITAL_COMPLETENESS_HANDLER = make_completeness_handler(
    source_item_recall_ratio=0.80,
)

THREE_YEAR_RETURN_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "原报告可能使用两行表格、双列版式，或用一句/两句独立句子披露两个数值。"
        "无论版式如何，输出必须固定为两行："
        "‘近三年平均投资收益率’和‘近三年平均综合投资收益率’，每行只保留对应数值。"
        "两句话分别披露时也必须合并处理，不得漏掉第二句；"
        "不得把普通投资收益率与综合投资收益率的数值互换。"
    ),
    single_page_note=(
        "该目标表的数据区固定为两类各一行：投资收益率、综合投资收益率。"
        "项目名称可能带有或省略“近三年”“平均”等修饰语，均须按原文保留；"
        "原报告可能以一句或两句分别披露两个数值而不是表格；遇到句式披露时，必须将其拆分为"
        "‘近三年平均投资收益率|数值’和‘近三年平均综合投资收益率|数值’两行；"
        "不得混入备注、主要经营指标或前五大产品明细。"
    ),
)
THREE_YEAR_RETURN_POSTPROCESS_HANDLER = make_postprocess_handler(
    normalize_three_year=True,
)
THREE_YEAR_RETURN_COMPLETENESS_HANDLER = make_completeness_handler(
    allow_single_value_boundary_rows=True,
)

MINIMUM_CAPITAL_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "最低资本存在两类版式：一类从量化风险最低资本开始、以最低资本结束；"
        "另一类把最低资本放在主表首行，随后继续披露保险风险最低资本汇总、"
        "市场风险最低资本汇总和信用风险最低资本汇总等明细。"
        "遇到后一类版式时，所有风险汇总子表都属于本目标，必须按原顺序跨页完整保留，"
        "不得在首行“最低资本”或主表“附加资本”处停止。"
        "量化风险在前的标准版式中，“3 附加资本”后可能继续出现3.1至3.4子项，"
        "必须提取到最后的“4 最低资本”及其本期、上期两个数值。"
        "若原文末行写作“4 合计”，该行就是最低资本合计，必须提取并将项目名标准化为“最低资本”。"
    ),
    single_page_note=(
        "最低资本可能采用“最低资本在首行、风险汇总明细在后”的版式。"
        "若本页或后续定位页出现保险风险最低资本汇总、市场风险最低资本汇总、"
        "信用风险最低资本汇总等子表，均属于本目标，必须逐页完整提取；"
        "不得在首行‘最低资本’或主表‘附加资本’处提前结束。"
        "对于量化风险在前的标准版式，‘3 附加资本’后仍可能有3.1至3.4子项，"
        "最终必须保留末行‘4 最低资本’及其本期、上期两个数值；"
        "原文写作‘4 合计’时同样必须保留，并将项目名标准化为‘最低资本’。"
    ),
)
MINIMUM_CAPITAL_POSTPROCESS_HANDLER = make_postprocess_handler(
    recover_minimum_capital=True,
)
MINIMUM_CAPITAL_BOUNDARY_HANDLER = make_boundary_handler(
    require_value_for_boundary_items=True,
)

RECOGNIZED_ASSETS_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "“S03-认可资产表”列结构为两级表头：第一级为‘期末数/期初数’，"
        "第二级为‘账面价值/非认可价值/认可价值’。"
        "必须从‘现金及流动性管理工具’、‘投资资产’、‘再保险资产’等资产明细开始，"
        "连续提取到末行‘认可资产合计’（原文可能仅写作‘合计’，须保留）。"
        "随后开始的‘S04-认可负债表’（认可负债、准备金负债、金融负债等）不得混入。"
        "两级表头必须按原顺序保留，不得把第二级表头当作数据行，也不得合并或删除列。"
    ),
    single_page_note=(
        "本页属于‘S03-认可资产表’（或续页）。"
        "列结构为‘期末数/期初数’下的‘账面价值/非认可价值/认可价值’两级表头。"
        "从资产明细行到‘认可资产合计’（或‘合计’）逐行保留；续页不再重复表头时也不得删除数据。"
        "遇到‘认可负债’等下一张表内容立即停止。"
    ),
)
RECOGNIZED_ASSETS_BOUNDARY_HANDLER = make_boundary_handler()
RECOGNIZED_ASSETS_POSTPROCESS_HANDLER = make_postprocess_handler(
    normalize_recognized_assets_total=True,
)
RECOGNIZED_ASSETS_COMPLETENESS_HANDLER = make_completeness_handler(
    source_item_recall_ratio=1.0,
)

RECOGNIZED_ASSETS_V2_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "“S03-认可资产表”原为两级表头（期末数/期初数 × 账面价值/非认可价值/认可价值）。"
        "提取时只保留‘认可价值’口径：输出统一为单级表头‘行次/项目/期末数/期初数’"
        "（期末数在前、期初数在后），其中期末数、期初数分别取对应列的‘认可价值’数值，"
        "不得输出‘账面价值’‘非认可价值’列。"
        "按原报告行顺序提取‘现金及流动性管理工具’、‘投资资产’、"
        "‘在子公司合营企业和联营企业中的权益’、‘再保险资产’、‘应收及预付款项’、"
        "‘固定资产’、‘土地使用权’、‘独立账户资产’、‘其他认可资产’等资产明细至末行‘合计’。"
        "末行‘合计’必须保留并将项目名标准化为‘认可资产合计’。"
        "随后开始的‘S04-认可负债表’（认可负债、准备金负债、金融负债等）不得混入。"
    ),
    single_page_note=(
        "本页属于‘S03-认可资产表’（或续页）。"
        "原两级表头（期末数/期初数 × 账面价值/非认可价值/认可价值）仅保留‘认可价值’列，"
        "输出为单级表头‘行次/项目/期末数/期初数’（期末数在前）。"
        "从资产明细行到‘认可资产合计’（或‘合计’）逐行保留，只取认可价值数值；"
        "续页不再重复表头时也不得删除数据；遇到‘认可负债’等下一张表内容立即停止。"
    ),
)
RECOGNIZED_ASSETS_V2_BOUNDARY_HANDLER = make_boundary_handler()
RECOGNIZED_ASSETS_V2_POSTPROCESS_HANDLER = make_postprocess_handler(
    simplify_recognized_assets_columns=True,
    normalize_recognized_assets_total=True,
)
RECOGNIZED_ASSETS_V2_COMPLETENESS_HANDLER = make_completeness_handler(
    source_item_recall_ratio=1.0,
)

RECOGNIZED_LIABILITIES_PROMPT_HANDLER = make_prompt_handler(
    full_table_note=(
        "“S04-认可负债表”原为两级表头（期末数/期初数 × 账面价值/非认可价值/认可价值）。"
        "提取时只保留‘认可价值’口径，输出单级表头‘行次/项目/期末数/期初数’，"
        "其中期末数、期初数分别取对应期间的认可价值，不得输出账面价值或非认可价值列。"
        "按原报告行顺序从准备金负债、金融负债、应付及预收款项等明细连续提取到"
        "末行‘认可负债合计’；原文仅写‘合计’时也必须保留并标准化。"
        "随后开始的‘S05-最低资本表’不得混入。"
    ),
    single_page_note=(
        "本页属于‘S04-认可负债表’或其续页。原两级表头仅保留期末数/期初数下的"
        "‘认可价值’列，输出单级表头‘行次/项目/期末数/期初数’。"
        "逐行保留负债明细至‘认可负债合计’（或‘合计’），续页未重复表头时也不得删行；"
        "遇到最低资本表立即停止。"
    ),
)
RECOGNIZED_LIABILITIES_BOUNDARY_HANDLER = make_boundary_handler()
RECOGNIZED_LIABILITIES_POSTPROCESS_HANDLER = make_postprocess_handler(
    simplify_recognized_liabilities_columns=True,
    normalize_recognized_liabilities_total=True,
)
RECOGNIZED_LIABILITIES_COMPLETENESS_HANDLER = make_completeness_handler(
    source_item_recall_ratio=1.0,
)
