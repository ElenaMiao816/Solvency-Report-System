from __future__ import annotations

import re
from typing import Iterable, Sequence


TABLE_ITEM_BOUNDARIES = {
    "SOLVENCY_MAIN": {
        "start_items": ("认可资产", "认可资产合计"),
        "end_items": ("综合偿付能力充足率", "综合偿付充足率"),
    },
    "OPERATING_METRICS": {
        "title_terms": ("主要经营指标", "主要经营情况"),
        "start_items": ("保险业务收入", "保险业务收入合计"),
        "end_items": (
            "营销员脱落率",
            "个人营销员脱落率",
            "代理人脱落率",
            "营销员流失率",
        ),
        "end_item_groups": (
            (
                "营销员脱落率",
                "个人营销员脱落率",
                "代理人脱落率",
                "营销员流失率",
            ),
            ("综合投资收益率",),
        ),
        "exclude_items": ("前五大产品的信息",),
    },
    "ACTUAL_CAPITAL": {
        "scope_name": "实际资本",
        "variant_note": (
            "若实际资本汇总后继续披露核心资本明细，两张子表属于同一目标，"
            "必须从汇总首行连续提取到核心资本明细末行；若实际资本表后继续披露"
            "核心一级资本调整表，两张子表同样属于目标，认可资产、认可负债明细不属于目标。"
        ),
        "variants": (
            {
                "name": "财务报表口径汇总在前并展开实际资本明细",
                "start_items": ("财务报表资产总额", "财务报表资产合计"),
                "end_items": ("实际资本合计", "实际资本总额"),
                "required_items": (
                    "认可资产",
                    "财务报表负债总额",
                    "认可负债",
                    "财务报表净资产总额",
                    "实际资本",
                    "核心一级资本",
                    "核心二级资本",
                    "附属一级资本",
                    "附属二级资本",
                ),
            },
            {
                "name": "实际资本汇总在前并展开核心资本明细",
                "start_items": ("认可资产", "认可资产合计"),
                "end_items": ("实际资本合计", "实际资本总额"),
                "required_items": (
                    "认可负债",
                    "实际资本",
                    "核心一级资本",
                    "核心二级资本",
                    "附属一级资本",
                    "附属二级资本",
                ),
            },
            {
                "name": "认可资产汇总后接核心一级资本调整表",
                "activation_terms": (
                    "核心一级资本调整表",
                    "核心资本调整表",
                ),
                "start_items": ("认可资产", "认可资产合计"),
                "end_items": (
                    "银保监会规定的其他调整项目",
                    "金融监管总局规定的其他调整项目",
                    "监管机构规定的其他调整项目",
                ),
                "required_items": (
                    "认可负债",
                    "实际资本",
                    "核心一级资本",
                    "核心二级资本",
                    "附属一级资本",
                    "附属二级资本",
                    "净资产",
                    "对净资产的调整额",
                ),
            },
            {
                "name": "实际资本表后接核心一级资本调整表",
                "activation_terms": (
                    "核心一级资本调整表",
                    "核心资本调整表",
                ),
                "start_items": ("实际资本",),
                "end_items": (
                    "银保监会规定的其他调整项目",
                    "金融监管总局规定的其他调整项目",
                    "监管机构规定的其他调整项目",
                ),
                "required_items": (
                    "核心一级资本",
                    "核心二级资本",
                    "附属一级资本",
                    "附属二级资本",
                    "净资产",
                    "对净资产的调整额",
                ),
            },
            {
                "name": "核心资本明细标准版",
                "start_items": ("核心一级资本", "核心一级资本合计"),
                "end_items": (
                    "实际资本合计",
                    "实际资本总额",
                    "实际资本",
                    "合计",
                ),
            },
        ),
    },
    "THREE_YEAR_INVESTMENT_RETURN": {
        "title_terms": (
            "近三年（综合）投资收益率",
            "近三年综合投资收益率",
            "近三年投资收益率",
        ),
        "variants": (
            {
                "name": "明确近三年双项披露",
                "priority": 100,
                "start_items": (
                    "近三年平均投资收益率",
                    "近三年投资收益率",
                ),
                "end_items": (
                    "近三年平均综合投资收益率",
                    "近三年综合投资收益率",
                ),
            },
            {
                "name": "近三年标题下简称披露",
                "priority": 10,
                "start_items": ("投资收益率",),
                "end_items": ("综合投资收益率",),
            },
        ),
        "required_item_groups": (
            (
                "近三年平均投资收益率",
                "近三年投资收益率",
                "投资收益率",
            ),
            (
                "近三年平均综合投资收益率",
                "近三年综合投资收益率",
                "综合投资收益率",
            ),
        ),
        "exact_items_only": True,
    },
    "MINIMUM_CAPITAL": {
        "scope_name": "最低资本",
        "variant_note": (
            "若主表后继续披露保险、市场或信用风险最低资本汇总，"
            "这些明细属于同一目标表，必须连续提取。"
        ),
        "variants": (
            {
                "name": "可资本化风险在前的完整汇总",
                "priority": 100,
                "start_items": ("可资本化风险最低资本",),
                "end_items": (
                    "最低资本",
                    "最低资本合计",
                    "最低资本总额",
                    "合计",
                ),
            },
            {
                "name": "最低资本在前并展开风险汇总",
                "start_items": ("最低资本", "最低资本合计", "最低资本总额"),
                "end_item_groups": (
                    (
                        "信用风险间的相关性效应",
                        "信用风险分散效应",
                    ),
                    (
                        "市场风险间的相关性效应",
                        "市场风险分散效应",
                    ),
                    (
                        "非寿险业务保险风险间的相关性效应",
                        "寿险业务保险风险间的相关性效应",
                    ),
                    ("附加资本",),
                ),
            },
            {
                "name": "量化风险在前的标准汇总",
                "start_items": ("量化风险最低资本", "量化风险最低资本合计"),
                "end_items": (
                    "最低资本",
                    "最低资本合计",
                    "最低资本总额",
                    "合计",
                ),
            },
        ),
    },
}
TABLE_CANONICAL_HEADERS = {
    "SOLVENCY_MAIN": (
        "项目",
        "本季度数",
        "上季度可比数",
        "基本情景下的下季度预测数",
    ),
    "OPERATING_METRICS": ("指标名称", "本季度数", "本年度累计数"),
    "ACTUAL_CAPITAL": ("行次", "项目", "期末数", "期初数"),
    "THREE_YEAR_INVESTMENT_RETURN": ("项目", "数值"),
    "MINIMUM_CAPITAL": ("行次", "项目", "期末数", "期初数"),
}


class TableBoundaryError(RuntimeError):
    pass


def compact_item(value: str) -> str:
    return re.sub(r"[\s：:（）()、，,。．.·—\-_/％%]", "", str(value or ""))


def _strip_leading_index(value: str) -> str:
    text = compact_item(value)
    patterns = (
        r"^[（(]?[一二三四五六七八九十]+[）)]?",
        r"^\d+(?:\.\d+)*\*?",
    )
    changed = True
    while changed:
        changed = False
        for pattern in patterns:
            updated = re.sub(pattern, "", text, count=1)
            if updated != text:
                text = updated
                changed = True
                break
    return text


def _item_prefix_match(value: str, item: str) -> bool:
    candidate = _strip_leading_index(value)
    target = compact_item(item)
    if not candidate.startswith(target):
        return False
    remainder = candidate[len(target):]
    if not remainder:
        return True
    return bool(re.match(
        r"^(?:合计|总额|总计|亿元|万元|千元|元|人民币元|百分比|为|是|等于|=|"
        r"[<＜]不适用[>＞]|不适用|N/?A|\d|\.|\*|--|—|-)",
        remainder,
        flags=re.I,
    ))


def row_has_item(row: Sequence[str], item: str) -> bool:
    return any(_item_prefix_match(cell, item) for cell in row if str(cell or "").strip())


def line_has_item(line: str, item: str, *, require_value: bool = False) -> bool:
    content = str(line or "").partition("|")[2] if "|" in str(line or "") else str(line or "")
    if not _item_prefix_match(content, item):
        return False
    if not require_value:
        return True
    candidate = _strip_leading_index(content)
    remainder = candidate[len(compact_item(item)):]
    return bool(re.search(
        r"\d|--|—|-|[<＜]不适用[>＞]|不适用|N/?A",
        remainder,
        flags=re.I,
    ))


def text_has_item(text: str, item: str, *, require_value: bool = False) -> bool:
    return any(
        line_has_item(line, item, require_value=require_value)
        for line in str(text or "").splitlines()
    )


def item_hits_in_text(text: str, items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in items if text_has_item(text, item))


def item_hits_in_rows(rows: Sequence[Sequence[str]], items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in items if any(row_has_item(row, item) for row in rows))


def boundary_variants(table_id: str) -> tuple[dict, ...]:
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id, {})
    configured = tuple(boundary.get("variants", ()))
    if not configured:
        return (boundary,) if boundary else ()
    shared = {
        key: value
        for key, value in boundary.items()
        if key != "variants"
    }
    return tuple({**shared, **variant} for variant in configured)


def end_item_groups(
    table_id: str,
    boundary: dict | None = None,
) -> tuple[tuple[str, ...], ...]:
    boundary = boundary or TABLE_ITEM_BOUNDARIES.get(table_id, {})
    configured = boundary.get("end_item_groups")
    if configured:
        return tuple(tuple(group) for group in configured)
    end_items = tuple(boundary.get("end_items", ()))
    return (end_items,) if end_items else ()


def boundary_items(table_id: str) -> tuple[str, ...]:
    items: list[str] = []
    for boundary in boundary_variants(table_id):
        grouped = tuple(
            item
            for group in boundary.get("required_item_groups", ())
            for item in group
        )
        end_grouped = tuple(
            item
            for group in end_item_groups(table_id, boundary)
            for item in group
        )
        items.extend((
            *boundary.get("start_items", ()),
            *boundary.get("end_items", ()),
            *end_grouped,
            *boundary.get("required_items", ()),
            *grouped,
        ))
    return tuple(dict.fromkeys(items))


def boundary_instruction(table_id: str) -> str:
    variants = boundary_variants(table_id)
    if not variants:
        return ""
    if len(variants) > 1:
        configured = TABLE_ITEM_BOUNDARIES.get(table_id, {})
        scope_name = str(configured.get("scope_name", table_id))
        variant_note = str(configured.get("variant_note", "")).strip()
        descriptions: list[str] = []
        for index, boundary in enumerate(variants, start=1):
            start_items = tuple(boundary.get("start_items", ()))
            end_groups = end_item_groups(table_id, boundary)
            if not start_items or not end_groups or not end_groups[0]:
                continue
            name = str(boundary.get("name", f"版式{index}"))
            fallbacks = "、".join(
                group[0] for group in end_groups[1:] if group
            )
            description = (
                f"{index}. {name}：从“{start_items[0]}”类项目开始，"
                f"优先到“{end_groups[0][0]}”类项目结束"
            )
            if fallbacks:
                description += f"；未披露后续风险类别时依次允许以“{fallbacks}”结束"
            descriptions.append(description)
        return (
            f"{scope_name}披露支持多种原文顺序，必须按页面中实际出现且首尾顺序闭合的版式提取："
            + "；".join(descriptions)
            + "。首尾项目均须保留；"
            + variant_note
            + "不得在前一张子表结束处提前截断。"
        )

    boundary = variants[0]
    start_items = tuple(boundary.get("start_items", ()))
    end_groups = end_item_groups(table_id, boundary)
    end_items = end_groups[0] if end_groups else ()
    exclusions = "、".join(boundary.get("exclude_items", ()))
    instruction = (
        f"数据范围从“{start_items[0]}”类项目开始，到“{end_items[0]}”类项目结束；"
        "允许同义名称或省略“近三年/平均/合计”等修饰语，但首尾项目均须保留。"
    )
    if len(end_groups) > 1:
        fallback_labels = "、".join(group[0] for group in end_groups[1:] if group)
        instruction += (
            f"若原报告未披露主终止项目之后的分类，可按顺序以“{fallback_labels}”作为实际终止项目；"
            "只要后续终止项目在原页面出现，就必须完整提取到该项目，不得提前截断。"
        )
    if boundary.get("exact_items_only"):
        instruction += "最终数据区必须且只能包含投资收益率与综合投资收益率两类项目各一行。"
    if exclusions:
        instruction += f"不得输出项目：{exclusions}。"
    return instruction

def _find_row(rows: Sequence[Sequence[str]], items: Sequence[str], start: int = 0) -> int | None:
    for index in range(max(0, start), len(rows)):
        if any(row_has_item(rows[index], item) for item in items):
            return index
    return None


def _is_pagination_row(row: Sequence[str]) -> bool:
    nonempty = [
        re.sub(r"\s+", "", str(cell or ""))
        for cell in row
        if str(cell or "").strip()
    ]
    if len(nonempty) != 1:
        return False
    value = nonempty[0]
    return bool(
        re.fullmatch(r"\d{1,4}", value)
        or re.fullmatch(r"第\d{1,4}页", value)
        or re.fullmatch(r"[-—]\d{1,4}[-—]", value)
    )


def _canonical_header(table_id: str, width: int) -> list[str]:
    configured = list(TABLE_CANONICAL_HEADERS.get(table_id, ()))
    return [
        configured[index] if index < len(configured) else f"列{index + 1}"
        for index in range(width)
    ]


def _is_generic_header(row: Sequence[str]) -> bool:
    cells = [compact_item(cell).lower() for cell in row if compact_item(cell)]
    if not cells:
        return True
    return all(
        re.fullmatch(r"(?:(?:column|col)|列|字段)?\d+", cell, flags=re.I)
        for cell in cells
    )


def enforce_output_boundaries(
    table_id: str,
    rows: list[list[str]],
    *,
    require_complete: bool = True,
) -> tuple[list[list[str]], str]:
    """Trim reconstructed rows to the declared first and last business items."""
    variants = boundary_variants(table_id)
    if not variants or not rows:
        return rows, ""

    selected_boundary: dict | None = None
    selected_start_items: tuple[str, ...] = ()
    start_index: int | None = None
    end_index: int | None = None
    matched_end_items: tuple[str, ...] = ()
    for boundary in variants:
        candidate_start_items = tuple(boundary.get("start_items", ()))
        candidate_start = _find_row(rows, candidate_start_items)
        if candidate_start is None:
            continue
        for candidate_items in end_item_groups(table_id, boundary):
            candidate_end = _find_row(
                rows,
                candidate_items,
                start=candidate_start + 1,
            )
            if candidate_end is None and any(
                row_has_item(rows[candidate_start], item)
                for item in candidate_items
            ):
                candidate_end = candidate_start
            if candidate_end is None:
                continue
            selected_boundary = boundary
            selected_start_items = candidate_start_items
            start_index = candidate_start
            end_index = candidate_end
            matched_end_items = candidate_items
            break
        if selected_boundary is not None:
            break

    if selected_boundary is None or start_index is None or end_index is None:
        if require_complete:
            expected = "；".join(
                (
                    f"{' / '.join(boundary.get('start_items', ()))} -> "
                    + "；".join(
                        " / ".join(group)
                        for group in end_item_groups(table_id, boundary)
                    )
                )
                for boundary in variants
            )
            raise TableBoundaryError(f"未找到顺序闭合的首尾项目：{expected}。")
        return rows, ""

    header = rows[0] if start_index > 0 else None
    bounded = rows[start_index:end_index + 1]
    bounded = [row for row in bounded if not _is_pagination_row(row)]
    excluded = tuple(selected_boundary.get("exclude_items", ()))
    if excluded:
        bounded = [
            row for row in bounded
            if not any(row_has_item(row, item) for item in excluded)
        ]

    if selected_boundary.get("exact_items_only"):
        groups = tuple(selected_boundary.get("required_item_groups", ()))
        if not groups:
            groups = tuple(
                (item,)
                for item in selected_boundary.get("required_items", ())
            )
        selected: list[list[str]] = []
        for aliases in groups:
            row = next(
                (
                    candidate
                    for candidate in bounded
                    if any(row_has_item(candidate, item) for item in aliases)
                ),
                None,
            )
            if row is None:
                raise TableBoundaryError(
                    f"缺少指定项目类别：{' / '.join(aliases)}。"
                )
            selected.append(row)
        bounded = selected

    if header is not None and _is_generic_header(header):
        width = max(len(header), max((len(row) for row in bounded), default=0))
        header = _canonical_header(table_id, width)
    if header is not None and header not in bounded:
        bounded = [header, *bounded]
    return bounded, (
        "已按首尾项目边界截取，补全标准表头并过滤独立页码行；"
        f"采用版式：{selected_boundary.get('name', '标准版式')}；"
        f"实际起始项目：{selected_start_items[0] if selected_start_items else '未配置'}；"
        f"实际终止项目：{matched_end_items[0] if matched_end_items else '未配置'}"
    )
