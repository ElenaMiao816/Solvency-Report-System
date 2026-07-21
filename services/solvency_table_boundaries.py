from __future__ import annotations

import re
from typing import Iterable, Sequence


TABLE_ITEM_BOUNDARIES = {
    "SOLVENCY_MAIN": {
        "start_items": ("认可资产", "认可资产合计"),
        "end_items": ("综合偿付能力充足率", "综合偿付充足率"),
    },
    "OPERATING_METRICS": {
        "start_items": ("保险业务收入", "保险业务收入合计"),
        "end_items": (
            "营销员脱落率",
            "个人营销员脱落率",
            "代理人脱落率",
            "营销员流失率",
        ),
        "exclude_items": ("前五大产品的信息",),
    },
    "ACTUAL_CAPITAL": {
        "start_items": ("核心一级资本", "核心一级资本合计"),
        "end_items": ("实际资本合计", "实际资本总额", "实际资本"),
    },
    "THREE_YEAR_INVESTMENT_RETURN": {
        "title_terms": (
            "近三年（综合）投资收益率",
            "近三年综合投资收益率",
            "近三年投资收益率",
        ),
        "start_items": (
            "近三年平均投资收益率",
            "近三年投资收益率",
            "投资收益率",
        ),
        "end_items": (
            "近三年平均综合投资收益率",
            "近三年综合投资收益率",
            "综合投资收益率",
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
        "start_items": ("量化风险最低资本", "量化风险最低资本合计"),
        "end_items": ("最低资本", "最低资本合计", "最低资本总额"),
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
    return bool(re.match(r"^(?:万元|元|人民币元|百分比|\d|\.|\*|--|—|-)", remainder))


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
    return bool(re.search(r"\d|--|—|-", remainder))


def text_has_item(text: str, item: str, *, require_value: bool = False) -> bool:
    return any(
        line_has_item(line, item, require_value=require_value)
        for line in str(text or "").splitlines()
    )


def item_hits_in_text(text: str, items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in items if text_has_item(text, item))


def item_hits_in_rows(rows: Sequence[Sequence[str]], items: Iterable[str]) -> tuple[str, ...]:
    return tuple(item for item in items if any(row_has_item(row, item) for row in rows))


def boundary_items(table_id: str) -> tuple[str, ...]:
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id, {})
    grouped = tuple(
        item
        for group in boundary.get("required_item_groups", ())
        for item in group
    )
    return tuple(dict.fromkeys((
        *boundary.get("start_items", ()),
        *boundary.get("end_items", ()),
        *boundary.get("required_items", ()),
        *grouped,
    )))


def boundary_instruction(table_id: str) -> str:
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id)
    if not boundary:
        return ""
    start_items = tuple(boundary.get("start_items", ()))
    end_items = tuple(boundary.get("end_items", ()))
    exclusions = "、".join(boundary.get("exclude_items", ()))
    instruction = (
        f"数据范围从“{start_items[0]}”类项目开始，到“{end_items[0]}”类项目结束；"
        "允许同义名称或省略“近三年/平均/合计”等修饰语，但首尾项目均须保留。"
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
    boundary = TABLE_ITEM_BOUNDARIES.get(table_id)
    if not boundary or not rows:
        return rows, ""

    start_items = tuple(boundary.get("start_items", ()))
    end_items = tuple(boundary.get("end_items", ()))
    start_index = _find_row(rows, start_items)
    if start_index is None:
        if require_complete:
            raise TableBoundaryError(f"缺少起始项目：{'、'.join(start_items)}。")
        return rows, ""
    end_index = _find_row(rows, end_items, start=start_index + 1)
    if end_index is None and any(row_has_item(rows[start_index], item) for item in end_items):
        end_index = start_index
    if end_index is None:
        if require_complete:
            raise TableBoundaryError(f"缺少终止项目：{'、'.join(end_items)}。")
        return rows, ""

    header = rows[0] if start_index > 0 else None
    bounded = rows[start_index:end_index + 1]
    bounded = [row for row in bounded if not _is_pagination_row(row)]
    excluded = tuple(boundary.get("exclude_items", ()))
    if excluded:
        bounded = [
            row for row in bounded
            if not any(row_has_item(row, item) for item in excluded)
        ]

    if boundary.get("exact_items_only"):
        groups = tuple(boundary.get("required_item_groups", ()))
        if not groups:
            groups = tuple((item,) for item in boundary.get("required_items", ()))
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
    return bounded, "已按首尾项目边界截取，补全标准表头并过滤独立页码行"
